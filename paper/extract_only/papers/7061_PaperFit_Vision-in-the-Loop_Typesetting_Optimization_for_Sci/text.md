# arXiv:2605.10341v1[cs.AI]11May2026

[Figure 1]

### PaperFit: Vision-in-the-Loop Typesetting Optimization for Scientific Documents

Bihui Yu1∗, Xinglong Xu1∗, Junjie Jiang1∗, Jiabei Cheng3∗, Caijun Jia1, Siyuan Li2, Conghui He2, Jingxuan Wei1, Cheng Tan2

1University of Chinese Academy of Sciences, 2Shanghai Artificial Intelligence Laboratory, 3School of Automation and Intelligent Sensing, Shanghai Jiao Tong University

A LaTeX manuscript that compiles without error is not necessarily publication-ready. The resulting PDFs frequently suffer from misplaced floats, overflowing equations, inconsistent table scaling, widow and orphan lines, and poor page balance, forcing authors into repetitive compile-inspect-edit cycles. Rule-based tools are blind to rendered visuals, operating only on source code and log files. Text-only LLMs perform open-loop text editing, unable to predict or verify the two-dimensional layout consequences of their changes. Reliable typesetting optimization therefore requires a visual closed loop with verification after every edit. We formalize this problem as Visual Typesetting Optimization (VTO), the task of transforming a compilable LaTeX paper into a visually polished, page-budget-compliant PDF through iterative visual verification and source-level revision, and introduce a five-category taxonomy of typesetting defects to guide diagnosis. We present PaperFit, a vision-in-the-loop agent that iteratively renders pages, diagnoses defects, and applies constrained repairs. To benchmark VTO, we construct PaperFit-Bench with 200 papers across 10 venue templates and 13 defect types at different difficulty. Extensive experiments show that PaperFit outperforms all baselines by a large margin, establishing that bridging the gap from compilable source to publicationready PDF requires vision-in-the-loop optimization and that VTO constitutes a critical missing stage in the document automation pipeline.

Date: May 12, 2026 Correspondence: Cheng Tan, tancheng@pjlab.org.cn

Code Dataset

[Figure 2]

### 1 Introduction

The past decade has witnessed remarkable progress in document automation. Format conversion tools such as Pandoc [139] enable structural transformation from Word and Markdown to LATEX. Document understanding models [20, 190, 49] can reconstruct LATEX source code from PDF files. Recent large language models (LLMs) can generate complete LATEX document frameworks directly from natural descriptions [163, 218]. We refer to this stage collectively as structural formatting, whose primary objective is to produce compilable .tex files. However, compilation success does not guarantee visual quality. A syntactically valid LATEX project may still produce PDFs with misplaced floats, overflowing equations, inconsistent table scaling, widow and orphan lines, and poor page balance [144, 103]. The final page may contain excessive white space that makes the content appear incomplete, or spill into an extra half page that violates strict conference page limits. Currently, resolving these issues relies entirely on manual effort: researchers repeatedly compile the source, inspect the rendered PDF, identify visual defects, adjust the .tex file, and recompile. This compile–inspect–edit cycle, particularly intense in the final hours before submission deadlines, depends almost exclusively on visual judgment that no existing tool fully automates [94].

*Equal contribution.

Existing approaches fail to automate this process due to three fundamental limitations (Figure 1): (i) incomplete observability. Rule-based tools and compilation logs provide only one-dimensional, code-level signals (Figure 1a). They can detect overfull hbox warnings but cannot judge whether a minor overflow is visually significant, how figure placement affects reading flow, or how white space is distributed across a page. Typesetting quality is inherently a two-dimensional, spatial judgment that source code and logs alone cannot support. (ii) unconstrained repair space. When a model identifies a problem, it faces an enormous action space in which most options are pseudo-fixes: commands such as \vspace, \resizebox, and \newpage produce compilable output but violate implicit typesetting norms by distorting typography, masking issues, or shifting defects elsewhere. Template files define formatting rules for fonts, margins, and headings, yet encode none of the repair preferences that distinguish a legitimate fix from a cosmetic workaround. (iii) unverified cascading effects. LATEX edits are highly non-local: a small change in figure width can trigger page-break rearrangements across the entire document. Text-only LLMs operate in an open loop (Figure 1b), modifying source without rendering or inspecting the result, and thus cannot confirm whether an edit improves or degrades global layout. These challenges characterize typesetting as a closed-loop control problem requiring visual sensing, constrained action, and global verification after every edit.

The advancement of vision-language models (VLMs) [88, 185, 221] has made it feasible to automate this closed loop: a model that can both interpret rendered pages and generate LATEX modifications can replicate the human compile–inspect–edit workflow. Naively providing page images to a VLM across multiple rounds is insufficient; without structured diagnosis, constrained repair, and gated validation, the model tends to introduce new defects or ignore page-budget constraints [140, 166]. Based on this insight, we formalize Visual Typesetting Optimization (VTO) as the task of transforming a compilable LATEX paper into a visually polished, page-budget-compliant PDF through iterative visual verification and source-level revision, and introduce a five-category defect taxonomy covering space utilization, float placement, typographic consistency, overflow, and cross-template migration. We position VTO as a critical missing stage between structural formatting and final publication.

We present PaperFit, a vision-in-the-loop agent that closes the sense–act–verify loop for typesetting optimization (Figure 1c). It addresses the three challenges above through three design components: multi-source evidence integration fuses source, log, PDF, and page-image signals into structured defect records, resolving incomplete observability; a constrained repair policy explicitly defines permitted operations, forbidden pseudo-fixes, and protected content, taming the unconstrained repair space; and checklist-gated multi-round validation recompiles, re-renders, and re-inspects the full document after every edit, catching cascading effects before they propagate.

To benchmark VTO, we construct PaperFit-Bench with 10 venue templates, 200 papers, and 13 defect types at three difficulty levels, and design six baselines that incrementally add capabilities from rule-only to multi-round visual repair. PaperFit achieves perfect compilation and rendering success, the highest visual quality and page-budget compliance, and substantially outperforms all baselines. The most informative comparison is against a naive multi-round visual agent sharing the same page images but lacking structured diagnosis, constrained repair, and gated validation: PaperFit surpasses it by a large margin in both visual quality and page-budget satisfaction, confirming that visual feedback is necessary but not sufficient. These results establish VTO as a critical missing stage in the document automation pipeline and highlight the decisive role of structured visual closed-loop control in producing publication-ready documents.

### 2 Related Work

##### 2.1 Document Layout Analysis and Automated Formatting

Recent research in document automation primarily emphasizes structural formatting. Early foundational work in sequence modeling [76] and automatic evaluation [154] established the building

###### From Code-Level Formatting to Visual Closed-Loop Typesetting Optimization Rule-based tools

###### Text-only LLM

###### PaperFit

Inspect visual layout

semastic

polish PDF preview

PDF preview

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

LaTeX source

PDF preview

[Figure 7]

[Figure 8]

[Figure 9]

visual detector

[Figure 10]

[Figure 11]

\LaTeX{ \malfetx{vcont} beginz{consitive

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

resolution{

Recompile and verify

Render PDF vision-language model

pdflatex or

mase-lonfigure{ \scompile{

[Figure 21]

[Figure 22]

[Figure 23]

overflowing

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

\\sails{consmesresolution{

[Figure 28]

.LaTeX source

edited .tex

[Figure 29]

[Figure 30]

mansiorizations{ endi} fondelle

Repair LaTeX

Diagnose defects

[Figure 31]

[Figure 32]

[Figure 33]

\LaTeX}

no visual feedback semanticpolish

quality gate

repair engine

[Figure 34]

Visual Blindness

Structural Formatting Compiled-success PDF Publication-ready PDF

(a) Blind to Rendered Visuals (b) Open-Loop Code-level Formatting (c) Vision-in-the-Loop Optimization

- Figure 1: Comparison of typesetting optimization approaches: (a) Rule-based tools are blind to visuals; (b) Text-only LLMs operate in an open loop and cannot predict rendering outcomes; (c) Our PaperFit system establishes a visual closed-loop agent that mimics the iterative human workflow.

blocks for later document understanding systems. VTLayout [110] represents a significant milestone by improving content block recognition through the integration of deep and shallow visual features with textual information. This integrated approach is further demonstrated by the LayoutLM series [214, 215], DocFormer [11], and the OCR-free DONUT [100]. More recent efforts have extended document layout analysis to handle complex perturbations [37], generate diverse large-scale layouts [4, 97], and enable global-to-local adaptive perception [2]. These models excel at extracting structure from document images, but their output is a recognized layout or reconstructed markup rather than a visually optimized source file. A parallel line of work focuses on generating compilable LATEX documents from scratch. LLM-driven generators such as Rxiv-Maker [164] produce complete paper frameworks from natural descriptions, cross-lingual formatting systems [218] preserve layout across languages, and agentic writing tools [135, 205] can draft entire manuscripts including LATEX source. Recent systems such as FlexDoc [3] further address document adaptation and compilation efficiency. However, all of these systems treat successful compilation as the terminal goal.

##### 2.2 Vision-Language Models for Visual Code Editing

VLMs have significantly improved the mapping of visual signals to code, particularly in extracting structured representations from documents. Nougat [20] demonstrates this advancement by using a Swin Transformer to convert academic PDFs into markup language, thereby bridging the gap between human- and machine-readable formats. The process of converting images to LaTeX is further supported by benchmarks such as Im2Latex-100K [96] and advanced visual reasoning models like

- A2R2 [1]. Additional tools, including Math2LaTeX [150] and Vision-RWKV [55], have expanded the capabilities for mathematical and structural recognition. Nevertheless, a key limitation persists: most models treat LaTeX as a static translation target. LATTE [94] introduced an iterative refinement framework for tables and formulae using visual feedback. Other studies have explored high-fidelity conversion through reinforcement learning for complex table images [126, 90].

##### 2.3 Iterative Self-Refinement and Agentic Frameworks

The development of multi-agent systems has enabled autonomous document optimization through collaborative pipelines. For example, PaperTalker [5] employs a coordinated suite of agents for content parsing, slide generation, and virtual avatar rendering to convert papers into presentation videos. Similar agentic frameworks include Paper2Poster [153], which automates academic poster synthesis, and AutoFigure-Edit [125], which generates editable scientific illustrations. LaTeXAgent

[56] provides stateful editing capabilities. Recent studies also examine structured translation via multi-agent coordination [253] and domain-specific review feedback [136]. A persistent challenge is establishing a reliable evaluation-optimization loop. Seeing is Improving (VFLM) [71, 70] uses visual rewards to guide iterative text layout refinement, directly addressing readability issues that are invisible at the code level. ReLook [117] applies vision-grounded reinforcement learning to web code generation, and SimpleDoc [89] integrates visual verification into multi-modal document understanding. DocReward [130] proposes learned reward models that score rendered document quality, providing an automated proxy for human visual judgment.

### 3 The PaperFit-Benchmark

##### 3.1 Overview

We introduce PaperFit-Bench, a benchmark for evaluating automated LaTeX layout repair. Unlike existing benchmarks that assess compilation success or content correctness, PaperFit-Bench operationalizes evaluation as visual layout restoration from systematically perturbed sources. Each instance pairs a perturbed LaTeX source with its original compilable version as ground truth, enabling deterministic evaluation across five defect categories (Class A–E) and three difficulty tiers. The benchmark comprises 200 instances spanning 10 venues and both single- and double-column formats.

##### 3.2 Dataset Construction

Data Collection. LaTeX source code of published papers was retrieved from arXiv, covering multiple subfields of artificial intelligence including nature language processing, computer vision, and reinforcement learning. This diversity mitigates evaluation bias toward any single typesetting style. As shown in Table 1, the resulting corpus spans 10 venue templates covering both single-column formats and double-column formats, with page limits ranging from 7 to 14. Each sample contains an average of 6.3 figures and 5.3 tables, providing substantial floating-element density that exercises the full range of layout repair capabilities. This venue diversity ensures that evaluation is not biased toward any single layout style or page constraint.

Table 1: Benchmark papers statistics by conference.

Conference # Papers Columns Page Limit Avg. Figures Avg. Tables

AAAI 20 2 7 5.60 5.30 ACM MM 20 2 8 7.25 6.40 CVPR/ICCV 20 1 8 5.25 4.70 ECCV 20 1 14 5.65 4.35 ICLR 20 1 9 6.30 4.85 ICML 20 2 8 5.55 5.05 IEEE Trans 20 2 12 4.60 3.85 IJCAI 20 2 7 5.80 4.70 IJCV 20 2 10+ 10.75 9.20 NeurIPS 20 1 9 5.95 4.40

Preprocessing. A standardized compilation test is applied in a controlled build environment; samples that fail compilation or depend on private macro packages are excluded. Appendix sections are uniformly removed. A dual quality-control mechanism combining manual verification ensures that each sample contains at least three figures and at least two tables.

Perturbation Design and Difficulty Tiers. We adopt thirteen perturbation strategies organized into five categories aligned with our VTO defect taxonomy (Figure 2): space utilization (Class A), float placement (Class B), table width (Class C), overflow (Class D), and cross-template migration (Class E).

[Figure 35]

- Figure 2: Perturbation distribution and category composition. The inner ring shows proportions of five perturbation categories (Class A–E); the outer ring shows specific perturbation.

- A key design principle of PaperFit-Bench is that it prioritizes realism over simplicity. PaperFit-Bench is a mixed-disturbance benchmark rather than a collection of one-defect toy examples. Each case is generated from an academic paper project and is associated with a case metadata record and a disturbance manifest. The benchmark contains three difficulty buckets: easy, medium, and hard. These buckets should be interpreted as empirical difficulty groups, not as deterministic recipes. A hard case, for example, may combine template-transfer pressure, table overflow, and page-budget drift, while an easy case may still contain a nontrivial local table or float issue.

These five active disturbance families cover the main visual typesetting optimization failure modes considered in this work. Space-utilization disturbances create widows, orphans, trailing whitespace, column imbalance, or intra-column voids. Float disturbances move figures or tables away from their natural reading position, shrink graphics, or enlarge graphics beyond the available width. Table disturbances create underutilized or overwide tables. Overflow disturbances introduce long unbreakable tokens or single-line equations that exceed the line width. Template-transfer disturbances create width mismatches or page-budget shifts after changing the surrounding template constraints. A complete listing of perturbation strategies, including their implementation details, validation status, and adoption frequencies, is provided in Table 2.

Beyond defining the perturbation types themselves, our benchmark construction methodology includes an important additional layer of documentation. The benchmark construction records both the intended perturbation and its concrete source-level realization. This is important because the same high-level defect can appear in different LaTeX forms. For example, an overwide figure may arise from an explicit width larger than \linewidth, while a page-budget shift may arise from template transfer together with a text-height change. The evaluation therefore treats the manifest as the source of disturbance intent, and the compile/render outputs as evidence of the actual realized failure.

Each instance is assigned a difficulty tier by the number of co-occurring perturbations: Easy (1–2), Medium (3–4), and Hard (5–8), distributed in a 3:4:3 ratio (Table 3). Cross-template perturbations (E1,

Table 2: Summary of perturbation strategies. “Validated” column: ✓ = post-compilation semantic verification confirms the defect manifests (e.g., overfull log, page shift, or layout inspection), with results recorded in disturbance manifest.json; × = standard compilation check only, without defectlevel verification.

Perturbation ID Category Defect Implementation Validated Frequency

- A1 widow orphan A A1 Force widow/orphan lines

via truncated short paragraphs

✓ 59

- A2 trailing whitespace A A2 Inject trailing whitespace be-

× 72

fore bibliography

- A4 column imbalance A A4 Inject vertical gaps on

double-column final pages

✓ 16

- A5 column void A A5 Insert vertical voids within

✓ 13

body text columns

- B1 float to page B B1 Push selected floats to dedi-

cated float-only pages

× 50 B2 oversize graphics B B2 Enlarge graphics beyond

available column width

× 48

- B2 shrink graphics B B2 Shrink graphics to noticeably

undersized width

× 51 C1 table resizebox C C1 Wrap tables in \resizebox to

undersize

× 73

- C2 table oversize C C2 Widen tables beyond avail-

able column width

× 79 D1 long token overflow D D1 Append unbreakable tokens

to trigger line overflow

× 68

- D2 long formula D D2 Insert ultra-wide formulas to

trigger display overflow

✓ 80 E1 template mismatch E E1 Cross-template migration

with unreasonable image widths

✓ 76

- E2 template page shift E E2 Cross-template migration

✓ 76

with reduced \textheight

E2) become increasingly prominent in harder instances.

Assembly and Finalization. Perturbed sources are assembled into complete problem instances and undergo final quality verification to ensure compilation succeeds and visual perturbations are realized. The final benchmark contains 200 instances.

Having completed the description of our benchmark construction pipeline, we now compare PaperFitBench against representative existing document processing benchmarks to highlight its unique characteristics.

- As summarized in Table 4, PaperFit-Bench fills an important gap in the literature. It is the only benchmark that simultaneously supports systematic perturbation injection, visual evaluation based on rendered page outputs, multi-modal evidence integration, and iterative full-document repair workflows—all essential features for evaluating modern AI-powered LaTeX layout optimization agents.

- Table 3: Distribution of perturbation difficulty levels and most frequently used perturbation types.

Difficulty Samples Common Perturbation Types

Easy 60

A2 (trailing whitespace), C1 (table resizebox), C2 (table oversize), D1 (long token overflow), D2 (long formula), E1 (template mismatch), E2 (template page shift)

Medium 80

A2 (trailing whitespace), C1 (table resizebox), C2 (table oversize), D2 (long formula), E2 (template page shift)

Hard 60

A2 (trailing whitespace), C2 (table oversize), D1 (long token overflow), D2 (long formula), E1 (template mismatch), E2 (template page shift)

- Table 4: Comparison with representative benchmarks. PaperFit-Bench is the only benchmark that combines systematic perturbation injection, visual evaluation from rendered pages, multi-modal evidence chains, and full-document iterative repair.

Benchmark Task Perturbation Visual Eval Multi-Modal Iterative

Im2Latex-100K Formula reconstruction – ✗ ✗ ✗ TeXpert LaTeX code generation – ✗ ✗ ✗ RoDLA Layout robustness Limited Partial ✓ ✗ DocReward Quality assessment – ✗ ✗ ✗ LATTE Element-level refinement – ✗ ✓ ✓

PaperFit-Bench Visual typesetting repair 13 strategies ✓ ✓ ✓

### 4 Method

##### 4.1 Preliminaries

Let x denote a compilable LATEX project, τ the target template, and b an optional page budget. Executing the compile-render pipeline produces log evidence ℓ, a PDF P (upon successful compilation), rendered page images I, and a page count p. Visual Typesetting Optimization (VTO) seeks a revised source x∗ that minimizes residual visual defects under hard constraints:

x∗ = argmin

+ λe ∆(x, x′) (1)

∑d∈D(x′) wc(d) s(d) visual defect score

x′

s.t. COMPILE(x′, τ) = success, (2) RENDER(x′, τ) = success, (3) CONTENT(x′) ⊇ CONTENT(x), (4) |PAGES(x′, τ)| = b (when b is specified), (5)

where D(x′) is the set of visual defects detected in the rendered pages of x′ under template τ, each characterized by its category c(d) and severity s(d); wc(d) weights defect categories according to the VTO taxonomy; ∆(x, x′) measures source-level edit distance to encourage minimal, auditable changes; and λe balances edit conservatism against visual improvement.

The hard constraints enforce that x′ compiles and renders under template τ (Eqs. 2–3), preserves all scientific content including figures, tables, captions, labels, citations, and bibliography entries (Eq. 4), and meets the page budget when specified (Eq. 5). Constraints are prioritized in strict order: content preservation > compilation/rendering > page budget > visual quality > edit minimality. Because the

objective is observable only after compiling and rendering and because even minor source edits can trigger non-local layout cascades, VTO cannot be solved by single-pass generation. We formulate it as an iterative, evidence-driven search with visual verification after every edit.

##### 4.2 Sense: Multi-Source Evidence Integration

No single evidence source reliably captures all typesetting defects. A table may compile without warnings, use a standard tabular environment, and land on the correct page—yet overflow the column boundary. Only the page-image layer reveals this defect; only the source layer can localize the repair target. PaperFit therefore fuses four complementary evidence layers:

Source-layer signals (.tex). The source layer provides document structure, template configuration, macro definitions, float environments, table structure, and counts of protected objects such as figures, tables, captions, labels, citations, and bibliography commands. This layer identifies editable regions, safeguards key scientific objects, and reveals structural mismatches resulting from template migration.

Log-layer signals (.log). Compilation logs offer deterministic execution evidence, including compile failures, undefined control sequences, unresolved references, missing citations, overfull or underfull warnings, and template-compatibility errors. When the input fails to compile or render, this layer serves as the primary evidence for restoring an executable state.

PDF-layer signals (.pdf). The compiled PDF provides document-level outcomes, including final page count, page order, and float landing behavior. This layer helps determine whether the page budget is met and whether floats have drifted far from their first citation.

Page-image-layer signals. Rendered pages reveal two-dimensional visual defects that source code or logs cannot reliably detect, such as sparse final pages, double-column column-void artifacts, float stacking, oversized tables, local whitespace, cross-page imbalance, and visual inconsistency.

The diagnosis stage converts the collected evidence into structured defect records.

d = (c, o,r, e), (6)

where c ∈ {A,B,C,D,E} is the defect category, o is the location (page and spatial region), r ∈ {blocking,degrading,cosmetic} is the severity, and e is the supporting evidence. These records form the interface between diagnosis and repair: every subsequent edit is traceable to explicit multi-source evidence, and the severity field determines repair priority in the next stage.

##### 4.3 Act: Constrained Repair Policy

Given the defect set Dt, PaperFit must select repair actions from an enormous space, most of which are pseudo-fixes: technically compilable but typographically harmful. We control this space through a repair preference profile π that encodes action tiers, defect-category-specific strategies, forbidden operations, and protected content.

- 4.3.1 Repair Action Tiers We categorize all LATEX repair actions into three tiers based on their side-effect risk:

- • Layout-native (preferred): float re-anchoring ([htbp] parameter adjustment), equation splitting into multiline forms (align, multline), table restructuring with width-aware environments (tabularx, table*), and figure width normalization to template-safe values. These operations address the root cause of the defect without side effects.
- • Spacing-manipulative (restricted): local \vspace adjustment, \setlength modification, and columnbreak hints are permitted only with explicit local justification and must pass re-verification.

[Figure 36]

- Figure 3: Overview of the PaperFit pipeline. PaperFit diagnoses layout defects from source, log, PDF, and page-image evidence, applies repairs under a repair preference profile, and validates outputs through a checklist-gated multi-round loop.

• Pseudo-fix (forbidden as primary repair): \resizebox on tables, \newpage/\pagebreak for budget control, \scalebox for graphics, and content deletion. These commands may temporarily mask a defect but distort typography, violate template norms, or shift defects to other pages.

- 4.3.2 Defect-Aware Repair Selection The repair profile π specifies a priority ordering across defect categories and preferred strategies:

- • Compile errors (highest): restore compilation via log-guided source repair.
- • Overflow (D): split long equations; break unbreakable tokens.
- • Float placement (B): re-anchor floats near first citation; normalize figure widths.
- • Table consistency (C): replace \resizebox with tabularx; restructure overwide tables.
- • Space utilization (A): adjust float positions and parameters to eliminate widows/orphans and whitespace.
- • Cross-template (E): reconcile width/height mismatches from template migration.

At each round, the system selects the highest-priority unresolved defect from Dt and applies the top-ranked layout-native strategy for that category. If layout-native options are exhausted, spacingmanipulative actions may be attempted under the restricted policy.

- 4.3.3 Content Preservation and Semantic Polish Fallback

Before applying any repair, the system snapshots the count and location of all protected objects (figures, tables, captions, labels, citations, and bibliography entries). After repair, it verifies that no protected object has been deleted, displaced across section boundaries, or had its caption altered. Violations trigger automatic rollback to the pre-repair state.

When layout-native repairs have been exhausted but minor page-budget gaps, widows/orphans, or sparse final pages persist, PaperFit permits bounded semantic polishing: minimal wording adjustments (e.g., tightening a verbose sentence, replacing a long phrase with a concise equivalent) that do not alter claims, results, numbers, citations, or factual meaning. This fallback is invoked only after layout-native

options fail and remains subject to the content preservation guards. It serves as a last-resort mechanism rather than a primary repair strategy.

- 4.4 Verify: Checklist Quality Control

A single post-repair compilation check cannot ensure global layout because LATEX edits are highly non-local: a small change in float width can cascade into page-break rearrangements across the entire document. PaperFit recompiles, re-renders, and re-inspects the complete document after every edit:

St = (xt, ℓt, Pt, It, Dt, Ht, at), (7)

where xt is the current source, ℓt is compile-log evidence, Pt is the PDF, It is the rendered page set, Dt is the structured defect report, Ht is hard-constraint signals, and at stores next actions.

Each round follows six steps: (1) compile and collect logs; (2) parse deterministic signals (errors, references, overfull boxes); (3) render all pages; (4) build structured defect records from multi-source evidence; (5) apply constrained repairs per defect category and repair preference profile; (6) recompile/rerender and let the gatekeeper decide.

The gatekeeper outputs one of three decisions: DONE (all constraints pass, no blocking residual defects), CONTINUE (safe but issues remain), or BLOCKED (repair is unsafe or infeasible). The DONE checklist requires successful compilation, rendering, page-level visual inspection, absence of blocking defects, page-budget satisfaction, and preservation of protected content.

- 5 Experiment

- 5.1 Experimental Setting

We evaluate on PaperFit-Bench (Section 3.2). Each method receives the same LaTeX project and target page budget; outputs are compiled, rendered, and scored with both programmatic checks and VLM-based visual evaluation.

###### 5.1.1 Baselines.

We compare six baselines with PaperFit, spanning three feedback paradigms. Rule-based: Perturbed (unmodified input) and RuleLog (deterministic rule/log repair). Text-only: TextST (single-turn source edit) and TextMR (multi-round source-plus-log edit). Visual: VisualST (single-turn source-plus-image edit) and VisualMR (multi-round visual agent with fixed rounds).

These baselines are systematically constructed to isolate the incremental value of each core capability in the visual typesetting optimization pipeline. They differ only in the evidence sources they can access, the number of repair iterations allowed, and whether they incorporate PaperFit’s structured repair machinery:

Perturbed: perturbed input. Perturbed is the unmodified disturbed project. It measures the raw difficulty of the benchmark after perturbation and provides the visual reference for VLM pairwise comparison whenever the perturbed input can be rendered.

RuleLog: deterministic rule/log repair. RuleLog applies deterministic repair rules driven by source and compile-log signals. It does not use model-based visual feedback. Its role is to test how far simple execution and log repair can go without page-level visual evidence.

TextST: single-turn text-only model repair. TextST sends the LaTeX source to a model in a single repair turn. It does not inspect rendered page images. This baseline tests whether source-only reasoning can repair layout defects without observing the final pages.

TextMR: multi-round text/log repair. TextMR extends TextST with multiple text/log feedback rounds. It can react to compilation errors and logs across rounds, but it still does not use page images as visual evidence.

VisualST: single-turn visual repair. VisualST receives the LaTeX source and rendered page images, then performs one visual repair turn. If compilation or rendering fails, the failure is accounted for by the same evaluation protocol rather than being removed from the denominator. VisualST isolates the value and limitation of adding page images without closed-loop visual iteration.

VisualMR: naive multi-round visual agent baseline. VisualMR is a fixed-round visual agent baseline. It can inspect source, logs, and page images over a small fixed number of rounds and can directly repair compile, render, and layout issues. It does not use PaperFit’s defect taxonomy, structured diagnosis records, constrained repair policy, repair-plan artifacts, rollback-aware gatekeeper, or PaperFit runtime. This makes VisualMR the closest baseline for testing whether multi-round visual feedback alone is sufficient.

PaperFit. PaperFit uses the same basic input project and target page budget, but adds structured multi-source diagnosis, a repair preference profile, and checklist-gated validation. The distinction between VisualMR and PaperFit is therefore not whether a model sees page images, but whether the multi-round process is organized around explicit defects, constrained repairs, and acceptance gates.

###### 5.1.2 Evaluation protocol.

We evaluate all methods using a dual-metric framework that combines programmatic correctness checks and human-aligned visual assessment, ensuring outputs are both technically valid and publication-ready. We report four primary binary metrics: compile success, render success, Page hit (exact page-budget match), and Win rate (fraction of cases judged visually better than the Perturbed baseline).

For quantitative composite evaluation, we use two complementary scores: - Program: A 0–5 composite of non-visual execution reliability and content fidelity - VLM: A gated 0–5 visual quality score based on rendered page assessment

We report both scores because Visual Typesetting Optimization (VTO) requires outputs to simultaneously satisfy hard technical constraints and subjective visual quality standards. A method that produces visually appealing pages but fails to preserve content or meet page budgets is not acceptable for publication, just as a technically correct but visually defective document fails to meet the core goal of typesetting optimization.

Program Score. The Program score summarizes non-visual execution and fidelity signals on a 0–5 scale, computed as the average of five equally weighted dimensions, each normalized to [0,1]:

Program = 5 ·

5

1 5

#### ∑

sk.

k=1

The five dimensions are:

- • compile reliability: Whether the candidate compiles and renders into usable pages

- • content integrity: Whether protected scientific objects (figures, tables, captions, labels, citations, bibliography entries) are fully preserved

- • reference quality: Whether all references resolve correctly and no severe log errors remain

- • page precision: Whether the output satisfies the target page budget, with a penalty for excessive source rewriting

- • content embedding similarity: Semantic similarity between the original and final LaTeX sources

Program is intentionally not a visual score. A method can receive a high Program score by producing a compilable, faithful, page-controlled document even if its final layout still contains visible whitespace or float-quality issues. Conversely, an output that looks acceptable in a rendered screenshot will be penalized by Program if it loses protected content, violates page budget, or leaves unresolved references.

VLM Visual Score. The visual evaluation uses rendered page images and produces a gated 0–5 score. It operates in two modes: 1. Pairwise comparison mode: When the Perturbed baseline renders successfully, the evaluator compares the perturbed input and candidate output side-by-side 2. Renderrescue mode: When the Perturbed baseline cannot be rendered, a renderable candidate receives credit for recovering from a non-renderable state

The raw VLM score combines three weighted components:

VLMraw = 0.35 Sabs + 0.40 Srepair + 0.25 Sfinal.

Here: - Sabs measures absolute repair-oriented quality, including defect resolution, constraint alignment, visual quality, new-defect avoidance, and publication readiness - Srepair measures pairwise repair quality relative to Perturbed (when renderable) or render-rescue quality (when Perturbed is not renderable) - Sfinal measures final-paper aesthetics, including professionalism, space utilization, float placement, typographic consistency, and visual balance

The final reported VLM score applies strict constraint gates to the raw score: - Non-renderable candidates are capped at the minimum score - Compile-dirty but renderable outputs are penalized and capped - Page-budget failure, unresolved references, or major newly introduced visual defects also trigger score capping

The Win rate reported in the main results is the fraction of cases in which a method is judged better than the Perturbed baseline under this full visual evaluation protocol.

##### 5.2 Main Quantitative Results

- Table 5: Main results. VLM is the visual evaluation score; Program is the 0–5 composite programmatic score. All other quantities are rates in [0,1].

Method Compile ↑ Render ↑ VLM ↑ Win ↑ Program ↑ Page hit ↑ Perturbed 0.5800 0.8200 1.8275 0.0000 3.6344 0.3750

RuleLog 0.5200 0.7600 2.1838 0.3800 3.3401 0.4444 TextST 0.5850 0.5850 1.8522 0.2800 2.5738 0.4530 TextMR 0.6100 0.6100 2.1601 0.4250 2.7433 0.6230 VisualST 0.6250 0.6250 1.8741 0.2950 2.7681 0.4560 VisualMR 0.9750 0.9750 2.8006 0.6500 4.5789 0.5487 PaperFit 1.0000 1.0000 3.3907 0.8950 4.5790 0.8050

Neither text/log feedback nor single-turn visual feedback is sufficient. RuleLog, TextST, TextMR, and VisualST represent progressively richer feedback signals, from compile logs to multi-round text to rendered page images. Yet none exceeds a VLM score of 2.19 or a Win rate of 0.43 (Table 5). Text/log

methods cannot judge two-dimensional layout failures such as excessive white space or float cascades, while single-turn visual editing often fails to handle non-local cascades.

Naive multi-round visual repair improves usability but remains weak on page control. VisualMR reaches 0.975 on both compile and render success, confirming that multi-round visual and log feedback can remove most execution failures.

However, its Page hit is only 0.549 and its Win rate is 0.650. Without explicit planning, constrained repair, and gatekeeper validation, multi-round visual editing still struggles to satisfy page budgets and avoid newly introduced visual defects.

PaperFit gives the best trade-off between visual quality and constraint satisfaction. PaperFit achieves perfect compile and render success (1.000), the best VLM score (3.391), Win rate (0.895), and Page hit (0.805), with a Program score of 4.579 essentially tied with VisualMR.

All methods maintain high content embedding similarity (>0.97), confirming that these gains come from layout-structure repair rather than semantic drift.

##### 5.3 Capability Boundary Comparison

Rather than running additional external systems as direct experimental baselines—which would require substantial engineering adaptation to our specific task—we use recent papers and widely adopted open-source projects as external capability anchors. This choice avoids conflating method capability with engineering adaptation: existing external systems target related but different problems. Some systems specialize in parsing PDFs or document structure, some reconstruct local LaTeX objects from images, and some edit code repositories through command-line feedback. None of these system families directly targets full-paper visual typesetting repair for an existing LaTeX project.

- Table 6: External capability boundary matrix. System families: DocParser denotes PDF/document parsers, including MinerU, Marker, and Nougat [190, 49, 20]; LocalRecon denotes local LaTeX reconstruction systems, including LATTE, Table2LaTeX-RL, and LaTeX-OCR [94, 126, 19]; CodeAgent denotes general coding agents, including OpenHands, Aider, and SWE-agent [195, 66, 222]; B5-VisualAgent is our general-purpose visual coding-agent baseline; PaperFit is our full system. Capability abbreviations: MSI = multi-source input; Edit = LaTeX/code generation or editing; Loop = execution feedback loop; PVD = full-paper visual diagnosis from rendered page images; Layout = float/table/page-level layout repair; Gate = page-budget, template, and checklist-gated validation. ✓indicates full coverage, ✗indicates no coverage, and △ indicates partial coverage.

System family MSI Edit Loop PVD Layout Gate

DocParser [190, 49, 20] ✓ ✗ ✗ ✗ ✗ ✗ LocalRecon [94, 126, 19] ✓ ✓ △ ✗ ✗ ✗ CodeAgent [195, 66, 222] ✓ ✓ ✓ ✗ ✗ ✗ B5-VisualAgent ✓ ✓ ✓ △ △ ✗ PaperFit ✓ ✓ ✓ ✓ ✓ ✓

In Table 6, multi-source input means that a system can process at least one task-relevant input modality, such as PDFs, page images, local object images, code repositories, or textual instructions. DP systems extract text, equations, tables, and layout structure from PDF or document inputs, but their objective is document understanding or PDF-to-markup conversion rather than source-level repair. LR systems recover local LaTeX objects from formula or table images, but object-level image-to-LaTeX reconstruction is not equivalent to full-paper layout repair. CA systems can edit code repositories and iterate with command-line feedback, making them the closest external capability class to PaperFit; however, their feedback loop is usually organized around software task success or test passing, not visual diagnosis over page images rendered from compiled PDFs.

VisualMR instantiates the general-purpose visual coding-agent capability class in the controlled setting. It can inspect files, run commands, compile and render the project, view page images, and edit LaTeX over fixed rounds. However, VisualMR is denied PaperFit’s VTO taxonomy, structured repair plans, constrained repair policy, runtime state management, and checklist-gated validation. VisualMR is therefore marked as partial for full-paper visual diagnosis and layout repair, and as absent for page-budget, template, and gatekeeper constraints.

The capability matrix shows that external systems cover different local segments of the PaperFit capability chain, but no external family simultaneously covers multi-source input, LaTeX/code editing, execution feedback, page-image-based full-paper diagnosis, float/table/page-level repair, and pagebudget/template/gatekeeper constraints. PaperFit’s contribution is not any single input parser, code editor, or local LaTeX recognizer; it is the integration of these capabilities into a full-paper visual typesetting optimization loop for existing LaTeX projects.

##### 5.4 Model Backend Comparison

To isolate the role of the language-model backend, we run the same PaperFit workflow with four diverse LLMs on 20 representative cases. Table 7 and Figures 4–5 show three consistent patterns.

Table 7: Model comparison on 20 representative cases (6 easy, 8 medium, 6 hard).

LLM Backend Compile ↑ Render ↑ VLM ↑ Win ↑ Page hit ↑

GPT-5.4 [149] 100.00% 100.00% 3.656 95.00% 95.00% Claude Opus 4.6 [10] 100.00% 100.00% 3.548 90.00% 100.00% DeepSeek-V4 Pro [50] 95.00% 100.00% 3.521 95.00% 100.00% MiMo-v2.5-pro [210] 100.00% 100.00% 3.652 100.00% 95.00%

Aggregate performance is stable across backends. All backends obtain high VLM scores (3.52–3.66), strong win rates (90–100%), and near-perfect compile/render reliability. The overall VLM spread is only 0.14 points, far smaller than the 0.59-point gap between PaperFit and VisualMR in Table 5, suggesting that the main improvement is from PaperFit rather than a particular model.

Backend differences reflect repair style rather than a single dominant model. Figure 4(a) shows that MiMo-v2.5 has the strongest repair-oriented profile, leading in defect resolution (3.90), visual quality (3.85), and publication readiness (3.80). GPT-5.4 instead leads in new-defect avoidance (4.30) and remains competitive on constraint alignment, which explains why its gated visual score slightly exceeds MiMo despite a lower raw visual score.

The residual bottleneck is visual balance, not execution reliability. Figure 4(b) shows that DeepSeekV4 leads in space utilization (3.50), float placement (3.90), and visual balance (3.20), while MiMo-v2.5 has the highest overall professionalism (3.85). However, across all backends, typographic consistency is consistently high, whereas space utilization and visual balance remain the weakest dimensions. Venue-level results in Figure 5 further show that no backend dominates uniformly across templates.

Difficulty-split results. Table 8 reports the difficulty-split VLM scores for the four LLM backends. The VLM spread remains ≤0.14 within each difficulty tier, and no single backend dominates across all three levels—GPT-5.4 leads on easy and medium cases while DeepSeek-V4 Pro achieves the highest score on hard cases. This cross-over pattern confirms that the ranking reflects stochastic variation rather than a systematic backend advantage.

##### 5.5 Human–VLM Evaluation Correlation

To assess alignment between human judgments and automated scores, the Spearman correlation coefficient (r) is computed between VLM scores and average human ratings across all methods. As shown in

[Figure 37]

Table 8: Difficulty-split VLM scores for the LLM comparison (20 cases). All four backends remain effective across difficulty levels, with score spread ≤ 0.14 on each split.

LLM Backend Easy (n=6) Medium (n=8) Hard (n=6)

GPT-5.4 [149] 3.821 3.792 3.310 Claude Opus 4.6 [10] 3.598 3.442 3.638 DeepSeek-V4 Pro [50] 3.163 3.509 3.893 MiMo-v2.5-pro [210] 3.648 3.778 3.486

[Figure 38]

[Figure 39]

(a) Repair and Constraint Dimensions (b) Final Aesthetic Dimensions

- Figure 4: Fine-grained VLM scores for the LLM backend comparison. Panel (a) reports repair and constraint dimensions, and panel (b) reports final aesthetic dimensions.

- Figure 6, the overall correlation is exceptionally high (r = 0.8571), confirming that the automated metric closely reflects human-perceived quality and faithfully captures model performance trends in the typesetting repair domain.

- 5.6 Qualitative Case Study

- Figure 7– 10 presents the qualitative cases spanning distinct VTO modes.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Case Study: Realigning Tables and Figures with In-Text Citations. As shown in Figure 7 On the CVPR/ICCV case (target 10 pages), the disturbed input displaces tables and figures away from their semantic anchors. Both Perturbed and VisualMR render the ablation-study page but show a large region that mentions Table 3, Table 4, and Figure 3 without the corresponding visual objects nearby. PaperFit restores Tables 3–4 and Figure 3 near their references and satisfies the 10-page budget, whereas VisualMR produces 13 pages.

Figure 6: Human/VLM evaluation correlation.

Case Study: Fixing Page Budget Shift and Underfilled Pages. As shown in Figure 8, On the IJCAI case (target 8 pages), template migration creates excessive blank spaces and a page-count mismatch. VisualMR compiles and renders but leaves large blank areas on the final references page, stopping at 10 pages. PaperFit adopts compact typesetting to condense the layout and meets the 8-page limit while preserving the reference section.

Case Study: Aesthetic Detail Refinement. As shown in Figure 9, On the IEEE case (target 16 pages), the disturbed input exhibits poor footer aesthetics with misaligned reference layout at the document tail. VisualMR restores compilation but introduces severe typesetting defects and expands the document to

[Figure 40]

4.5

2.69 3.67 4.00 3.52 3.56 4.37 4.00 4.11 3.64 3.00

GPT-5.4

[Figure 41]

4.0

VLMScore

- 4.11 3.69 3.84 2.74 3.44 3.00 3.52 4.09 4.06 3.00
- 4.11 4.05 2.75 3.39 3.50 3.00 3.56 4.07 3.78 3.00

Claude Opus

3.5

DeepSeek-V4

3.0

4.17 3.27 3.56 3.56 3.56 3.00 4.03 4.04 4.32 3.00 2.5

MiMo-v2.5

AAAIACMMMCVPRICCV ECCV ICLR ICML IEEE IJCAI IJCVNeurIPS

Figure 5: Venue-level VLM score distribution for the LLM backend comparison.

###### Case Study: Realigning Tables and Figures with In-Text Citations

Perturbed disturbed input , p.6

VisualMR output , p.6 PaperFit output , p.7

|[Figure 42]<br><br>Missing table/figure<br><br>near reference|
|---|

|[Figure 43]<br><br>Table and figure restored near reference|
|---|

|[Figure 44]<br><br>Missing table/figure<br><br>near reference|
|---|

Mentions Table 3 , Table 4 , and Figure 3 but they do not appear near the references.

Tables 3, 4 and Figure 3 are now

Mentions Table 3 , Table 4 , and Figure 3

near their references.

but they do not appear near the references.

| |
|---|

| |
|---|

Missing (table/figure near reference)

Restored near reference

Figure 7: Case Study: Realigning Tables and Figures with In-Text Citations.

20 pages under a 16-page target. PaperFit fixes the footer misalignment, restores a compact reference layout, and returns to 16 pages.

Case Study: Template Migration. As shown in Figure 10, on two mainstream academic layout conversion cases (AAAI → ICLR for double-to-single column, and ICLR → CVPR for single-to-double column), direct template migration causes severe layout mismatches including figure width overflow and disordered float placement, failing to meet the target venue’s formatting requirements. PaperFit automatically adapts figure dimensions to fit the target layout constraints, validates and optimizes float placement, and precisely matches the target template specifications. All core validation checks (compilation, rendering, template matching, column alignment, and content integrity) are fully passed, achieving end-to-end compliant template migration without manual intervention.

Across all case studies, VisualMR produces renderable output but fails to resolve the underlying layout defects and misses the page constraint. It lacks persistent defect records and acceptance gates, so it stops at the first renderable result. PaperFit instead diagnoses each failure through its structured

###### Case Study: Fixing Page Budget Shift and Underfilled Pages

###### Perturbed disturbed input , p.10

VisualMR output , p.10 PaperFit output , p.8

|[Figure 45]<br><br>Large blank areas remain|
|---|

|[Figure 46]<br><br>condensed layout|
|---|

|[Figure 47]<br><br>Large blank areas remain|
|---|

Excessive blank spaces and page count inconsistent with journal requirements.

Excessive blank spaces and page count

Adopt compact typesetting to

meet the journal’s page limit.

inconsistent with journal requirements.

| |
|---|

| |
|---|

Large blank areas remain

condensed layout

Figure 8: Case Study: Fixing Page Budget Shift and Underfilled Pages.

taxonomy, applies constrained repairs, and validates outputs through a checklist gate. This qualitative evidence supports the quantitative trend in Section 5: visual feedback alone is useful, but reliable full-paper VTO requires an organized closed loop with explicit defect records, repair constraints, and acceptance gates.

##### 5.7 Error Analysis

To understand the remaining failure modes of the PaperFit, we examine four representative error cases grouped into two figures. Each figure contains two failure examples, comparing the perturbed input with the OURS candidate output.

###### 5.7.1 Global Page-Budget Violations

- Figure 11 shows two cases in which the agent violates the global page-budget constraint.

- Case A: Page-budget Gate Failed. An ACM Multimedia paper with a target of 10 pages produces a 16-page output. The agent’s iterative repairs create several sparse trailing pages, indicating effective local edits but insufficient global page-budget control.
- Case B: One Extra Float-Heavy Page Breaks the Page Budget. An ECCV paper with a target of

- 19 pages produces a 20-page output. The final page contains only a single large figure with substantial whitespace, showing that even a one-page deviation constitutes a hard failure when the added page is largely empty.

5.7.2 Residual Visual Defects and Invalid Output

Figure 12 shows two cases in which compilation and page-count metadata pass, but the visual output remains defective.

- Case C: Visual Defects Remain Unrepaired. An ACM Multimedia paper meets both compilation and page-budget targets (10/10), yet the oversized and cropped figure remains essentially unrepaired.

###### Case Study: Aesthetic detail refinement

###### Perturbed disturbed input , p.16

###### VisualMR output , p.20 PaperFit output , p.16

|[Figure 48]<br><br>Poor footer<br><br>aesthetics|
|---|

|[Figure 49]<br><br>Fix footer misalignment|
|---|

|[Figure 50]<br><br>Severe typesetting<br><br>defects|
|---|

Misaligned reference layout at the document tail.

Severe typesetting errors were

Misaligned reference layout at the

introduced during the revision process.

document tail.

| |
|---|

| |
|---|

| |
|---|

Poor footer aesthetics

Fix footer misalignment

Severe typesetting defects

Figure 9: Case Study: Aesthetic Detail Refinement. Satisfying hard constraints alone does not guarantee that the intended visual repair has been achieved.

- Case D: Successful Compilation & Rendering with Abnormal Output. An ICLR paper compiles successfully with the correct page count (13/13), but the rendered pages are grayed and visually invalid. LaTeX-level compilation success is insufficient as a sole quality indicator.

### 6 Conclusion

This paper identifies Visual Typesetting Optimization as a missing stage in document automation and introduces PaperFit, a vision-in-the-loop agent that bridges the gap between compilable and publication-ready LaTeX through multi-source evidence integration, constrained repair policies, and checklist-gated validation. On PaperFit-Bench (200 papers, 10 templates, 13 defects), PaperFit achieves perfect compile success, the highest VLM score, and an 80.5% page-budget hit rate.

###### Case Study: Template Migration

Source PaperFit migrated

[Figure 51]

|[Figure 52]|
|---|

| |[Figure 53]<br><br>| |
|---|---|---|
|fig<br><br>floa|ure width adapt and<br><br>t placement valid|ed<br><br>ated|
| | | |

Compile Render

[Figure 54]

AAAI → ICLR

[Figure 55]

Template

(D2S)

[Figure 56]

Column

[Figure 57]

Content

[Figure 58]

- page 2

Checks

| | | |
|---|---|---|
|targ|t template match|ed|
| |[Figure 59]<br><br>| |

page 2

|[Figure 60]|
|---|

- page 3

page 3

###### Gate DONE

[Figure 61]

Compile Render

[Figure 62]

figure

ICLR →CVPR

[Figure 63]

Template

width

(S2D)

[Figure 64]

adapted

Column

[Figure 65]

Content

[Figure 66]

Gate DONE

Figure 10: Case Study: Template Migration.

[Figure 67]

[Figure 68]

- Figure 11: Error analysis: page-budget violations. Case A: Page-budget gate failed; target 10 pages, OURS produces 16 sparse pages. Case B: One extra float-heavy page; target 19 pages, OURS produces

- 20 with an underfilled final page.

[Figure 69]

[Figure 70]

###### Figure 12: Error analysis: visual defects and invalid output. Case C: Visual defects remain unrepaired; compiles and meets page budget but the target figure defect persists. Case D: Successful compilation and rendering with abnormal output; correct page count but produces visually invalid grayed pages.

### References

- [1] $Aˆ2Rˆ2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement | OpenReview, . URL https://openreview.net/forum?id=UcrNxBtXWM.
- [2] DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-toLocal Adaptive Perception, . URL https://arxiv.org/html/2410.12628v1.
- [3] FlexDoc: Flexible Document Adaptation through Optimizing both Content and Layout, . URL https: //arxiv.org/html/2410.15504v1.
- [4] OmniLayout: Enabling Coarse-to-Fine Learning with LLMs for Universal Document Layout Generation, . URL https://arxiv.org/html/2510.26213v1.
- [5] PaperTalker Multi-Agent Framework, . URL https://www.emergentmind.com/topics/ papertalker-multi-agent-framework.
- [6] Daechul Ahn, Yura Choi, San Kim, Youngjae Yu, Dongyeop Kang, and Jonghyun Choi. Isr-dpo: Aligning large multimodal models for videos by iterative self-retrospective dpo. In AAAI, volume 39, pages 1728–1736, 2025.
- [7] Seokho Ahn, Hyungjin Kim, Sungbok Shin, and Young-Duk Seo. Real-time calibration model for low-cost sensor in fine-grained time series. In AAAI, volume 39, pages 3–11, 2025.
- [8] Jie An, Zhengyuan Yang, Jianfeng Wang, Linjie Li, Zicheng Liu, Lijuan Wang, and Jiebo Luo. Bring metric functions into diffusion models. arxiv, 2024.
- [9] Zhongde An, Jinhong You, Jiyanglin Li, Yiming Tang, Wen Li, Heming Du, and Shouguo Du. Fredn: Spectral disentanglement for time series forecasting via learnable frequency decomposition. In AAAI, volume 40, pages 19623–19631, 2026.
- [10] Anthropic. Claude Opus 4.6. https://www.anthropic.com/news/claude-opus-4-6, 2026. Accessed: 202605-03.
- [11] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. DocFormer: End-toEnd Transformer for Document Understanding, September 2021. URL http://arxiv.org/abs/2106.11539. arXiv:2106.11539 [cs].
- [12] Adrian Atienza, Jakob Bardram, and Sadasivan Puthusserypady. Contrastive learning is not optimal for quasiperiodic time series. arxiv, 2024.
- [13] Sithu Aung, Min-Cheol Sagong, and Junghyun Cho. Multi-view pedestrian occupancy prediction with a novel synthetic dataset. In AAAI, volume 39, pages 1782–1790, 2025.
- [14] Varun Babbar, Hayden McTavish, Cynthia Rudin, and Margo Seltzer. Near optimal decision trees in a split second. arxiv, 2025.
- [15] Lennart Bastian, Mohammad Rashed, Nassir Navab, and Tolga Birdal. Forecasting continuous nonconservative dynamical systems in so (3). In ICCV, pages 14845–14855, 2025.
- [16] Usha Bhalla, Alex Oesterling, Claudio Mayrink Verdun, Himabindu Lakkaraju, and Flavio P Calmon. Temporal sparse autoencoders: Leveraging the sequential nature of language for interpretability. arxiv, 2025.
- [17] Hanbo Bi, Yingchao Feng, Yongqiang Mao, Jianning Pei, Wenhui Diao, Hongqi Wang, and Xian Sun. Agmtr: Agent mining transformer for few-shot segmentation in remote sensing. IJCV, 133:1780–1807, 2025.
- [18] Matteo Bianchi, Antonio De Santis, Andrea Tocchetti, and Marco Brambilla. Interpretable network visualizations: A human-in-the-loop approach for post-hoc explainability of cnn-based image classification. arxiv, 2024.
- [19] Lukas Blecher. pix2tex: Using a ViT to convert images of equations into LaTeX code. https://github.com/ lukas-blecher/LaTeX-OCR, 2022. Accessed: 2026-04-28.

- [20] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural Optical Understanding for Academic Documents, August 2023. URL https://arxiv.org/abs/2308.13418v1.
- [21] Tom Burgert, Oliver Stoll, Paolo Rota, and Begum¨ Demir. Imagenet-trained cnns are not biased towards texture: Revisiting feature reliance through controlled suppression. arxiv, 2025.
- [22] Zhongnan Cai, Yingying Wang, Hui Zheng, Panwang Pan, ZiXu Lin, Ge Meng, Chenxin Li, Chunming He, Jiaxin Xie, Yunlong Lin, et al. Pan-lut: Efficient pan-sharpening via learnable look-up tables. arxiv, 2025.
- [23] Buqing Cao, Qian Peng, Xiang Xie, Liang Chen, Min Shi, and Jianxun Liu. Spiking heterogeneous graph attention networks. In AAAI, volume 40, pages 19853–19861, 2026.
- [24] Feiqi Cao, Caren Han, and Hyunsuk Chung. Peach: Pretrained-embedding explanation across contextual and hierarchical structure. arxiv, 2024.
- [25] Fuyuan Cao, Jiaxuan Zhang, and Xiaoli Li. Pite: Multi-prototype alignment for individual treatment effect estimation. In AAAI, volume 40, pages 19871–19879, 2026.
- [26] Hanqun Cao, Cheng Tan, Zhangyang Gao, Yilun Xu, Guangyong Chen, Pheng-Ann Heng, and Stan Z Li. A survey on generative diffusion models. IEEE TKDE, 36:2814–2830, 2024.
- [27] Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, et al. Web-shepherd: Advancing prms for reinforcing web agents. arxiv, 2025.
- [28] Jeffrey A Chan-Santiago, Praveen Tirupattur, Gaurav Kumar Nayak, Gaowen Liu, and Mubarak Shah. Mgd3: Mode-guided dataset distillation using diffusion models. arxiv, 2025.
- [29] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arxiv, 2025.
- [30] Haipeng Chen, Sifan Wu, Zhigang Wang, Yifang Yin, Yingying Jiao, Yingda Lyu, and Zhenguang Liu. Causal-inspired multitask learning for video-based human pose estimation. In AAAI, volume 39, pages 2052–2060, 2025.
- [31] Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, and Shouling Ji. Enhancing adversarial transferability with adversarial weight tuning. In AAAI, volume 39, pages 2061–2069, 2025.
- [32] Weirong Chen, Ganlin Zhang, Felix Wimbauer, Rui Wang, Nikita Araslanov, Andrea Vedaldi, and Daniel Cremers. Back on track: Bundle adjustment for dynamic scene reconstruction. In ICCV, pages 4951–4960, 2025.
- [33] Xiang Chen, Duanzheng Song, Honghao Gui, Chenxi Wang, Ningyu Zhang, Yong Jiang, Fei Huang, Chengfei Lv, Dan Zhang, and Huajun Chen. Factchd: Benchmarking fact-conflicting hallucination detection. arxiv, 2023.
- [34] Xiaocan Chen, Qilin Yin, Jiarui Liu, Wei Lu, Xiangyang Luo, and Jiantao Zhou. Glcf: A global-local multimodal coherence analysis framework for talking face generation detection. In AAAI, volume 39, pages 75–83, 2025.
- [35] Yanxi Chen, Chunxiao Li, Xinyang Dai, Jinhuan Li, Weiyu Sun, Yiming Wang, Renyuan Zhang, Tinghe Zhang, and Bo Wang. Boosting single positive multi-label classification with generalized robust loss. arxiv, 2024.
- [36] Yirui Chen, Xudong Huang, Quan Zhang, Wei Li, Mingjian Zhu, Qiangyu Yan, Simiao Li, Hanting Chen, Hailin Hu, Jie Yang, et al. Gim: A million-scale benchmark for generative image manipulation detection and localization. In AAAI, volume 39, pages 2311–2319, 2025.
- [37] Yufan Chen, Jiaming Zhang, Kunyu Peng, Junwei Zheng, Ruiping Liu, Philip Torr, and Rainer Stiefelhagen. RoDLA: Benchmarking the Robustness of Document Layout Analysis Models, 2024. URL https://arxiv. org/abs/2403.14442. Version Number: 1.

- [38] Yutao Chen, Xingning Dong, Tian Gan, Chunluan Zhou, Ming Yang, and Qingpei Guo. Eve: Efficient zero-shot text-based video editing with depth map guidance and temporal consistency constraints. arxiv, 2023.
- [39] Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, Haokun Wen, and Weili Guan. Hud: Hierarchical uncertainty-aware disambiguation network for composed video retrieval. In ACM MM, pages 6143–6152, 2025.
- [40] Zhuang Chen, Guanqun Bi, Wen Zhang, Jiawei Hu, Aoyun Wang, Xiyao Xiao, Kun Feng, and Minlie Huang. Unveiling the landscape of clinical depression assessment: From behavioral signatures to psychiatric reasoning. In AAAI, volume 40, pages 1748–1756, 2026.
- [41] Nuojin Cheng, Leonard Papenmeier, Stephen Becker, and Luigi Nardi. A unified framework for entropy search and expected improvement in bayesian optimization. arxiv, 2025.
- [42] Daewon Choi, Jongheon Jeong, Huiwon Jang, and Jinwoo Shin. Adversarial robustification via text-to-image diffusion models. In ECCV, pages 158–177. Springer, 2024.
- [43] Minkyu Choi, Harsh Goel, Mohammad Omama, Yunhao Yang, Sahil Shah, and Sandeep Chinchali. Towards neuro-symbolic video understanding. In ECCV, pages 220–236. Springer, 2024.
- [44] Ka-Ho Chow, Wenqi Wei, and Lei Yu. Imperio: Language-guided backdoor attacks for arbitrary model control. arxiv, 2024.
- [45] Davide Cozzolino, Giovanni Poggi, Matthias Nießner, and Luisa Verdoliva. Zero-shot detection of aigenerated images. In ECCV, pages 54–72. Springer, 2024.
- [46] Tianxiang Cui, Huibing Wang, Jinjia Peng, Ruoxi Deng, Xianping Fu, and Yang Wang. Fast one-stage unsupervised domain adaptive person search. arxiv, 2024.
- [47] Wei Dai, Peilin Chen, Chanakya Ekbote, and Paul Pu Liang. Qoq-med: Building multimodal clinical foundation models with domain-aware grpo training. arxiv, 2025.
- [48] Giannis Daras, Adrian Rodriguez-Munoz, Adam Klivans, Antonio Torralba, and Constantinos Daskalakis. Ambient diffusion omni: Training good models with bad data. arxiv, 2025.
- [49] Datalab. Marker: Convert documents to markdown, JSON, chunks, and HTML. https://github.com/ datalab-to/marker, 2024. Accessed: 2026-04-28.
- [50] DeepSeek-AI. DeepSeek-V4-Pro Technical Report. https://huggingface.co/deepseek-ai/ DeepSeek-V4-Pro/blob/main/DeepSeek V4.pdf, 2026. Accessed: 2026-05-03.

- [51] Zihao Deng, Yijia Li, Renrui Zhang, and Peijun Ye. Nl2ca: Auto-formalizing cognitive decision-making from natural language using an unsupervised criticnl2ltl framework. In AAAI, volume 40, pages 1766–1773, 2026.
- [52] Kun Ding, Qiang Yu, Haojian Zhang, Gaofeng Meng, and Shiming Xiang. Calibrated cache model for few-shot vision-language model adaptation. arxiv, 2024.
- [53] Shiyin Dong, Mingrui Zhu, Kun Cheng, Nannan Wang, and Xinbo Gao. Bridging generative and discriminative models for unified visual perception with diffusion priors. arxiv, 2024.
- [54] Wei Duan, Jie Lu, and Junyu Xuan. Group-aware coordination graph for multi-agent reinforcement learning. arxiv, 2024.
- [55] Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, and Wenhai Wang. Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures, March 2024. URL https://arxiv.org/abs/2403.02308v3.
- [56] EatingChew. Eric0801/LaTeXAgent, February 2026. URL https://github.com/Eric0801/LaTeXAgent. original-date: 2026-01-25T03:50:39Z.
- [57] Tianyu Fan, Lirong Wu, Yufei Huang, Haitao Lin, Cheng Tan, Zhangyang Gao, and Stan Z Li. Decoupling weighing and selecting for integrating multiple graph pre-training tasks. arxiv, 2024.

- [58] Xinyi Gao, Hongzhi Yin, Tong Chen, Guanhua Ye, Wentao Zhang, and Bin Cui. Robgc: Towards robust graph condensation. IEEE TKDE, 2025.
- [59] Zhangyang Gao, Cheng Tan, Pablo Chac´on, and Stan Z Li. Pifold: Toward effective and efficient protein inverse folding. arxiv, 2022.
- [60] Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z Li. Simvp: Simpler yet better video prediction. In CVPR, pages 3170–3180, 2022.
- [61] Zhangyang Gao, Cheng Tan, and Stan Z Li. Knowledge-design: Pushing the limit of protein design via knowledge refinement. arxiv, 2023.
- [62] Zhangyang Gao, Daize Dong, Cheng Tan, Jun Xia, Bozhen Hu, and Stan Z Li. A graph is worth k words: Euclideanizing graph using pure transformer. arxiv, 2024.
- [63] Zhangyang Gao, Jue Wang, Cheng Tan, Lirong Wu, Yufei Huang, Siyuan Li, Zhirui Ye, and Stan Z Li. Uniif: Unified molecule inverse folding. NeurIPS, 37:135843–135860, 2024.
- [64] Zhangyang Gao, Cheng Tan, Jue Wang, Yufei Huang, Lirong Wu, and Stan Z Li. Foldtoken: Learning protein language via vector quantization and beyond. In AAAI, volume 39, pages 219–227, 2025.
- [65] Andoni I Garmendia, Quentin Cappart, Josu Ceberio, and Alexander Mendiburu. Marco: a memoryaugmented reinforcement framework for combinatorial optimization. arxiv, 2024.
- [66] Paul Gauthier. Aider: AI pair programming in your terminal. https://github.com/Aider-AI/aider, 2023. Accessed: 2026-04-28.
- [67] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arxiv, 2025.
- [68] Federico Girella, Davide Talon, Ziyue Liu, Zanxi Ruan, Yiming Wang, and Marco Cristani. Lots of fashion! multi-conditioning for image generation via sketch-text pairing. In ICCV, pages 19711–19720, 2025.
- [69] Aditya Gopalan, Sayak Ray Chowdhury, and Debangshu Banerjee. Why dpo is a misspecified estimator and how to fix it. arxiv, 2025.
- [70] Junrong Guo, Shancheng Fang, Yadong Qu, Xiaorui Wang, and Hongtao Xie. Visual Feedback for SelfImproving Text Layout with MLLM via Reinforcement Learning. October 2025. URL https://openreview. net/forum?id=wUYRMxrULV.
- [71] Junrong Guo, Shancheng Fang, Yadong Qu, and Hongtao Xie. Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement, March 2026. URL https://arxiv.org/abs/2603.22187v1.
- [72] Mingzhe Guo, Zhipeng Zhang, Liping Jing, Yuan He, Ke Wang, and Heng Fan. Cyclic refiner: Object-aware temporal representation learning for multi-view 3d detection and tracking. IJCV, 132:6184–6206, 2024.
- [73] Yiwei Guo, Zhihan Li, Hankun Wang, Bohan Li, Chongtian Shao, Hanglei Zhang, Chenpeng Du, Xie Chen, Shujie Liu, and Kai Yu. Recent advances in discrete speech tokens: A review. IEEE TPAMI, 2025.
- [74] Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology 3d shape modeling. In ICCV, pages 14822–14833, 2025.
- [75] Alec Helbling, Tuna Han Salih Meral, Ben Hoover, Pinar Yanardag, and Duen Horng Chau. Conceptattention: Diffusion transformers learn highly interpretable features. arxiv, 2025.
- [76] Sepp Hochreiter and Jurgen¨ Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8):1735–1780, November 1997. ISSN 0899-7667, 1530-888X. doi: 10.1162/neco.1997.9.8.1735. URL https://direct.mit. edu/neco/article/9/8/1735-1780/6109.
- [77] Julia Hornauer, Amir El-Ghoussani, and Vasileios Belagiannis. Revisiting gradient-based uncertainty for monocular depth estimation. IEEE TPAMI, 2025.
- [78] Bozhen Hu, Cheng Tan, Siyuan Li, Jiangbin Zheng, Sizhe Qiu, Jun Xia, and Stan Z Li. Multimodal regression for enzyme turnover rates prediction. arxiv, 2025.

- [79] Fan Hu, Gaofeng Lu, Jun Chen, Channan Guo, Yuekui Yang, and Xirong Li. Aefs: Adaptive early feature selection for deep recommender systems. IEEE TKDE, 2025.
- [80] Fuxiang Huang, Xiaowei Fu, Shiyu Ye, Lina Ma, Wen Li, Xinbo Gao, David Zhang, and Lei Zhang. Unsupervised robust domain adaptation: Paradigm, theory and algorithm: F. huang etal. IJCV, 134:5, 2026.
- [81] Jiancheng Huang, Mingfu Yan, Songyan Chen, Yi Huang, and Shifeng Chen. Magicfight: Personalized martial arts combat video generation. In ACM MM, pages 10833–10842, 2024.
- [82] Kuan-Chih Huang, Yi-Hsuan Tsai, and Ming-Hsuan Yang. Weakly supervised 3d object detection via multi-level visual guidance. In ECCV, pages 175–191. Springer, 2024.
- [83] Yufei Huang, Odin Zhang, Lirong Wu, Cheng Tan, Haitao Lin, Zhangyang Gao, Siyuan Li, Stan Li, et al. Re-dock: towards flexible and realistic molecular docking with diffusion bridge. arxiv, 2024.
- [84] Zhehao Huang, Yuhang Liu, Baijiong Lin, Yixin Lou, Zhengbao He, Hanling Tian, Tao Li, and Xiaolin Huang. Rain-merging: A gradient-free method to enhance instruction following in large reasoning models with preserved thinking format. arxiv, 2026.
- [85] Zhijian Huang, Tao Tang, Shaoxiang Chen, Sihao Lin, Zequn Jie, Lin Ma, Guangrun Wang, and Xiaodan Liang. Making large language models better planners with reasoning-decision alignment. In ECCV, pages 73–90. Springer, 2024.
- [86] Zhongzhan Huang, Shanshan Zhong, Pan Zhou, Shanghua Gao, Marinka Zitnik, and Liang Lin. A causality-aware paradigm for evaluating creativity of multimodal large language models. IEEE TPAMI, 2025.
- [87] Zikai Huang, Xuemiao Xu, Cheng Xu, Huaidong Zhang, Chenxi Zheng, Jing Qin, and Shengfeng He. Beat-it: Beat-synchronized multi-condition 3d dance generation. In ECCV, pages 273–290. Springer, 2024.
- [88] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [89] Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, S hengyu Dai, Zhenwen Shao, Qingyun Wu, and Huazheng Wang. SimpleDoc: Multi-Modal Document Understanding with Dual-Cue Page Retrieval and Iterative Refinement, 2025. URL https://arxiv.org/abs/2506.14035. Version Number: 1.
- [90] Jayanth Jayanth, Jayaprakash Sundararaj, and Pushpak Bhattacharyya. Monotone Submodularity in Opinion Summaries. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 169–178, Lisbon, Portugal, 2015. Association for Computational Linguistics. doi: 10.18653/v1/ D15-1017. URL http://aclweb.org/anthology/D15-1017.
- [91] Saurabh Jha, Rohan Arora, Yuji Watanabe, Takumi Yanagawa, Yinfang Chen, Jackson Clark, Bhavya Bhavya, Mudit Verma, Harshit Kumar, Hirokuni Kitahara, et al. Itbench: Evaluating ai agents across diverse real-world it automation tasks. arxiv, 2025.
- [92] Feng Jiang, Mangal Prakash, Hehuan Ma, Jianyuan Deng, Yuzhi Guo, Amina Mollaysa, Tommaso Mansi, Rui Liao, and Junzhou Huang. Trident: Tri-modal molecular representation learning with taxonomic annotations and local correspondence. arxiv, 2025.
- [93] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. Rayzer: A self-supervised large view synthesis model. In ICCV, pages 4918–4929, 2025.
- [94] Nan Jiang, Shanchao Liang, Chengxiao Wang, Jiannan Wang, and Lin Tan. LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinement, February 2025. URL http://arxiv.org/ abs/2409.14201. arXiv:2409.14201 [cs].
- [95] Xiangjian Jiang, Nikola Simidjievski, and Mateja Jamnik. Tabstruct: Measuring structural fidelity of tabular data. arxiv, 2025.
- [96] Anssi Kanervisto. Im2Latex-100K , Arxiv:1609.04938, June 2016. URL https://zenodo.org/record/56198.
- [97] Hengrui Kang, Zhuangcheng Gu, Zhiyuan Zhao, Zichen Wen, Bin Wang, Weijia Li, and Conghui He. OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning, 2025. URL https://arxiv.org/abs/2510.26213. Version Number: 2.

- [98] O˘guzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, and Federico Tombari. Brave: Broadening the visual encoding of vision-language models. In ECCV, pages 113–132. Springer, 2024.
- [99] Taekyung Ki, Dongchan Min, and Gyeongsu Chae. Learning to generate conditional tri-plane for 3d-aware expression controllable portrait animation. In ECCV, pages 476–493. Springer, 2024.
- [100] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022.
- [101] Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arxiv, 2025.
- [102] Taeho Kim, Yanming Wang, Vatshank Chaturvedi, Lokesh Gupta, Seyeon Kim, Yongin Kwon, and Sangtae Ha. Llmem: Estimating gpu memory usage for fine-tuning pre-trained llms. arxiv, 2024.
- [103] Donald Ervin Knuth and Duane Bibby. The texbook, volume 15. Addison-Wesley Reading, 1984.
- [104] Zhaonian Kuang, Rui Ding, Meng Yang, Xinhu Zheng, and Gang Hua. Object-scene-camera decomposition and recomposition for data efficient monocular 3d object detection. IJCV, 134:155, 2026.
- [105] Guozheng Li, Peng Wang, Jiajun Liu, Yikai Guo, Ke Ji, Ziyu Shang, and Zijie Xu. Meta in-context learning makes large language models better zero and few-shot relation extractors. arxiv, 2024.
- [106] Hongyi Li, Jiawei Ye, Jie Wu, Tianjie Yan, Chu Wang, and Zhixin Li. Jailpo: A novel black-box jailbreak framework via preference optimization against aligned llms. In AAAI, volume 39, pages 27419–27427, 2025.
- [107] Jiangmeng Li, Zehua Zang, Qirui Ji, Chuxiong Sun, Wenwen Qiang, Junge Zhang, Changwen Zheng, Fuchun Sun, and Hui Xiong. Rethinking generalizability and discriminability of self-supervised learning from evolutionary game theory perspective. IJCV, 133:3542–3567, 2025.
- [108] Jie Li, Shengwei Tian, Long Yu, and Xin Ning. Ppc-mt: Parallel point cloud completion with mambatransformer hybrid architecture. arxiv, 2026.
- [109] Kailin Li, Jingbo Wang, Lixin Yang, Cewu Lu, and Bo Dai. Semgrasp: Semantic grasp generation via language aligned discretization. In ECCV, pages 109–127. Springer, 2024.
- [110] Shoubin Li, Xuyan Ma, Shuaiqun Pan, Jun Rect Hu, Lin Shi, and Qing Wang. VTLayout: Fusion of Visual and Text Features for Document Layout Analysis, August 2021. URL http://arxiv.org/abs/2108.13297. arXiv:2108.13297 [cs].
- [111] Siyuan Li, Zedong Wang, Zicheng Liu, Cheng Tan, Haitao Lin, Di Wu, Zhiyuan Chen, Jiangbin Zheng, and Stan Z Li. Moganet: Multi-order gated aggregation network. arxiv, 2022.
- [112] Siyuan Li, Weiyang Jin, Zedong Wang, Fang Wu, Zicheng Liu, Cheng Tan, and Stan Z Li. Semireward: A general reward model for semi-supervised learning. arxiv, 2023.
- [113] Siyuan Li, Zedong Wang, Zicheng Liu, Di Wu, Cheng Tan, Jiangbin Zheng, Yufei Huang, and Stan Z Li. Vqdna: Unleashing the power of vector quantization for multi-species genomic sequence modeling. arxiv, 2024.
- [114] Siyuan Li, Luyuan Zhang, Zedong Wang, Juanxi Tian, Cheng Tan, Zicheng Liu, Chang Yu, Qingsong Xie, Haonan Lu, Haoqian Wang, et al. Mergevq: A unified framework for visual generation and representation with disentangled token merging and quantization. In CVPR, pages 19713–19723, 2025.
- [115] Songtao Li and Hao Tang. Multimodal alignment and fusion: A survey. arxiv, 2024.
- [116] Yi Li, Hualiang Wang, Xinpeng Ding, Haonan Wang, and Xiaomeng Li. Token activation map to visually explain multimodal llms. In ICCV, pages 48–58, 2025.
- [117] Yuhang Li, Chenchen Zhang, Ruilin Lv, Ao Liu, Ken Deng, Yuanxing Zhang, Jiaheng Liu, Wiggin Zhou, and Bo Zhou. ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding, October

2025. URL http://arxiv.org/abs/2510.11498. arXiv:2510.11498 [cs].

- [118] Zhe Li, Zhangyang Gao, Cheng Tan, Stan Z Li, and Laurence T Yang. General point model with autoencoding and autoregressive. arxiv, 2023.
- [119] Zhe Li, Laurence T Yang, Bocheng Ren, Xin Nie, Zhangyang Gao, Cheng Tan, and Stan Z Li. Mlip: Enhancing medical visual representation with divergence encoder and knowledge-guided contrastive learning. In CVPR, pages 11704–11714, 2024.
- [120] Ziqiang Li, Yi Wu, Chaoyue Wang, Xue Rui, and Bin Li. One-shot generative domain adaptation in 3d gans. IJCV, 133:2371–2391, 2025.
- [121] Renzhao Liang, Sizhe Xu, Chenggang Xie, Jingru Chen, Feiyang Ren, Shu Yang, and Takahiro Yabe. Abstain mask retain core: Time series prediction by adaptive masking loss with representation consistency. arxiv, 2025.
- [122] Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, and Huan Sun. Redteamcua: Realistic adversarial testing of computer-use agents in hybrid web-os environments. arxiv, 2025.
- [123] Haitao Lin, Guojiang Zhao, Odin Zhang, Yufei Huang, Lirong Wu, Zicheng Liu, Siyuan Li, Cheng Tan, Zhifeng Gao, and Stan Z Li. Cbgbench: Fill in the blank of protein-molecule complex binding graph. arxiv, 2024.
- [124] Junan Lin, Daizong Liu, Xianke Chen, Xiaoye Qu, Xun Yang, Jixiang Zhu, Sanyuan Zhang, and Jianfeng Dong. Audio does matter: Importance-aware multi-granularity fusion for video moment retrieval. In ACM MM, pages 6027–6036, 2025.
- [125] Zhen Lin, Qiujie Xie, Minjun Zhu, Shichen Li, Qiyao Sun, Enhao Gu, Yiran Ding, Ke Sun, Fang Guo, Panzhong Lu, Zhiyuan Ning, Yixuan Weng, and Yue Zhang. Autofigure-edit: Generating editable scientific illustration, 2026. URL https://arxiv.org/abs/2603.06674.
- [126] Jun Ling, Yao Qi, Tao Huang, Shibo Zhou, Yanqin Huang, Jiang Yang, Ziqi Song, Ying Zhou, Yang Yang, Heng Tao Shen, and Peng Wang. Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models, September 2025. URL https://arxiv.org/abs/2509. 17589v1.
- [127] Lin Ling, Fazle Rabbi, Song Wang, and Jinqiu Yang. Bias unveiled: Investigating social bias in llm-generated code. In AAAI, volume 39, pages 27491–27499, 2025.
- [128] Fei Liu, Xi Lin, Weiduo Liao, Zhenkun Wang, Qingfu Zhang, Xialiang Tong, and Mingxuan Yuan. Prompt learning for generalized vehicle routing. arxiv, 2024.
- [129] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. arxiv, 2025.
- [130] Junpeng Liu, Yuzhong Zhao, Bowen Cao, Jiayu Ding, Yilin Jia, Tengchao Lv, Yupan Huang, Shaohan Huang, Nan Yang, Li Dong, Lei Cui, Tao Ge, Xun Wang, Huitian Jiao, Sun Mao, FNU Kartik, Si-Qing Chen, Wai Lam, and Furu Wei. DocReward: A Document Reward Model for Structuring and Stylizing, 2025. URL https://arxiv.org/abs/2510.11391. Version Number: 2.
- [131] Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. arxiv, 2025.
- [132] Yuyuan Liu, Yuanhong Chen, Hu Wang, Vasileios Belagiannis, Ian Reid, and Gustavo Carneiro. Ittakestwo: Leveraging peer representations for semi-supervised lidar semantic segmentation. In ECCV, pages 81–99. Springer, 2024.
- [133] Zicheng Liu, Siyuan Li, Ge Wang, Lirong Wu, Cheng Tan, and Stan Z Li. Harnessing hard mixed samples with decoupled regularizer. NeurIPS, 36:52884–52906, 2023.
- [134] Zichi Liu, Yinggui Wang, Tao Wei, and Chao Ma. Anchorsync: Global consistency optimization for long video editing. In ACM MM, pages 4494–4503, 2025.
- [135] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

- [136] Kai Lu, Shixiong Xu, Jinqiu Li, Kun Ding, and Gaofeng Meng. Agent Reviewers: Domain-specific Multimodal Agents with Shared Memory for Paper Review. June 2025. URL https://openreview.net/forum? id=s7HUJamWqX.
- [137] Lorenzo Lucchese, Mikko S Pakkanen, and Almut ED Veraart. Learning with expected signatures: Theory and applications. arxiv, 2025.
- [138] Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and Guoqi Li. Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection. In ECCV, pages 253–272. Springer, 2024.
- [139] John MacFarlane. Pandoc: a universal document converter, 2025. URL https://pandoc.org/. Version 3.x.
- [140] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.
- [141] Daniele Malitesta, Emanuele Rossi, Claudio Pomo, Tommaso Di Noia, and Fragkiskos D Malliaros. Trainingfree graph-based imputation of missing modalities in multimodal recommendation. IEEE TKDE, 2026.
- [142] Zihao Mei, Jianhao Li, Bolin Zhang, Chong Wang, Lijun Guo, Guoqi Li, and Jiangbo Qian. Temporal-aware spiking transformer hashing based on 3d-dwt. arxiv, 2025.
- [143] Utkarsh A Mishra, David He, Yongxin Chen, and Danfei Xu. Compositional diffusion with guided search for long-horizon planning. arxiv, 2025.
- [144] Frank Mittelbach, Michel Goossens, Johannes Braams, David Carlisle, and Chris Rowley. The LATEX companion. Addison-Wesley Professional, 2004.
- [145] Wentao Mo, Qingchao Chen, Yuxin Peng, Siyuan Huang, and Yang Liu. Advancing 3d scene understanding with mv-scanqa multi-view reasoning evaluation and tripalign pre-training dataset. In ACM MM, pages 12973–12980, 2025.
- [146] Jun-Yeong Moon, Jung Uk Kim, and Gyeong-Moon Park. Towards model-agnostic dataset condensation by heterogeneous models. In ECCV, pages 234–250. Springer, 2024.
- [147] Grigory Neustroev, Mirco Giacobbe, and Anna Lukina. Neural continuous-time supermartingale certificates. In AAAI, volume 39, pages 27538–27546, 2025.
- [148] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arxiv, 2025.
- [149] OpenAI. GPT-5.4 Thinking System Card. https://openai.com/index/gpt-5-4-thinking-system-card,

2026. Accessed: 2026-05-03.

- [150] Sultanul Ovi et al. Math2latex: Equation ocr to latex with qwen2-vl. https://github.com/sultanul-ovi/ Math2LaTeX-Equation-OCR-to-LaTeX-with-Qwen2-VL, 2025.
- [151] Zhiyu Pan, Yizheng Wu, Jiashen Hua, Junyi Feng, Shaotian Yan, Bing Deng, Zhiguo Cao, and Jieping Ye. Through the lens of contrast: Self-improving visual reasoning in vlms. arxiv, 2026.
- [152] Chaoxu Pang, Yixuan Cao, Ganbin Zhou, Hongwei Li, and Ping Luo. Document-level tabular numerical cross-checking: A coarse-to-fine approach. arxiv, 2025.
- [153] Wei Pang, Kevin Qinghong Lin, Xiangru Jian, Xi He, and Philip Torr. Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers, October 2025. URL http://arxiv.org/abs/2505.21497. arXiv:2505.21497 [cs].
- [154] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics ACL ’02, page 311, Philadelphia, Pennsylvania, 2001. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL http://portal.acm.org/citation.cfm?doid=1073083.1073135.
- [155] Wenjie Peng, Hongxiang Huang, Tianshui Chen, Quhui Ke, Gang Dai, and Shuangping Huang. Globally correlation-aware hard negative generation. IJCV, 133:2441–2462, 2025.

- [156] Guillaume Pourcel and Maxence Ernoult. Learning long range dependencies through time reversal symmetry breaking. arxiv, 2025.
- [157] Yurui Qian, Qi Cai, Yingwei Pan, Ting Yao, and Tao Mei. Creatively upscaling images with global-regional priors. IJCV, 133:5197–5215, 2025.
- [158] Feiwei Qin, Shichao Lu, Junhao Hou, Changmiao Wang, Meie Fang, and Ligang Liu. Drawing2cad: Sequence-to-sequence learning for cad generation from vector drawings. In ACM MM, pages 10573–10582, 2025.
- [159] Hongyu Qu, Xiangbo Shu, Rui Yan, Hailiang Gao, Wenguan Wang, and Jinhui Tang. Spatio-temporal decoupled knowledge compensator for few-shot action recognition. IEEE TPAMI, 2026.
- [160] Tao Ren, Zishi Zhang, Jingyang Jiang, Zehao Li, Shentao Qin, Yi Zheng, Guanghao Li, Qianyou Sun, Yan Li, Jiafeng Liang, et al. Half-order fine-tuning for diffusion model: A recursive likelihood ratio optimizer. arxiv, 2025.
- [161] Karsten Roth, Zeynep Akata, Dima Damen, Ivana Balazevic, and Olivier J H´enaff. Context-aware multimodal pretraining. In CVPR, pages 4267–4279, 2025.
- [162] Shouwei Ruan, Yinpeng Dong, Hanqing Liu, Yao Huang, Hang Su, and Xingxing Wei. Omniview-tuning: Boosting viewpoint invariance of vision-language pre-training models. In ECCV, pages 309–327. Springer, 2024.
- [163] Bruno M Saraiva, Ant´onio D Brito, Guillaume Jaquemet, and Ricardo Henriques. Rxiv-maker: an automated template engine for streamlined scientific publications. arXiv preprint arXiv:2508.00836, 2025.
- [164] Bruno M. Saraiva, Ant´onio D. Brito, Guillaume Jaquemet, and Ricardo Henriques. Rxiv-Maker: an automated template engine for streamlined scientific publications, 2025. URL https://arxiv.org/abs/ 2508.00836. Version Number: 5.
- [165] Hanrong Shi, Lin Li, Jun Xiao, Yueting Zhuang, and Long Chen. From easy to hard: Learning curricular shape-aware features for robust panoptic scene graph generation. IJCV, 133:489–508, 2025.
- [166] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36: 8634–8652, 2023.
- [167] Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff Schneider, and Ruslan Salakhutdinov. Training a generally curious agent. arxiv, 2025.
- [168] Cheng Tan, Jun Xia, Lirong Wu, and Stan Z Li. Co-learning: Learning from noisy labels with self-supervision. In ACM MM, pages 1405–1413, 2021.
- [169] Cheng Tan, Zhangyang Gao, Hanqun Cao, Xingran Chen, Ge Wang, Lirong Wu, Jun Xia, Jiangbin Zheng, and Stan Z Li. Deciphering rna secondary structure prediction: A probabilistic k-rook matching perspective. arxiv, 2022.
- [170] Cheng Tan, Zhangyang Gao, Lirong Wu, Siyuan Li, and Stan Z Li. Hyperspherical consistency regularization. In CVPR, pages 7244–7255, 2022.
- [171] Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, and Stan Z Li. Temporal attention unit: Towards efficient spatiotemporal predictive learning. In CVPR, pages 18770–18782, 2023.
- [172] Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, and Stan Z Li. Openstl: A comprehensive benchmark of spatio-temporal predictive learning. NeurIPS, 36:69819–69831, 2023.
- [173] Cheng Tan, Jue Wang, Zhangyang Gao, Siyuan Li, Lirong Wu, Jun Xia, and Stan Z Li. Revisiting the temporal modeling in spatio-temporal predictive learning under a unified view. arxiv, 2023.
- [174] Cheng Tan, Yijie Zhang, Zhangyang Gao, Bozhen Hu, Siyuan Li, Zicheng Liu, and Stan Z Li. Rdesign: Hierarchical data-efficient representation learning for tertiary structure-based rna design. arxiv, 2023.

- [175] Cheng Tan, Zhenxiao Cao, Zhangyang Gao, Lirong Wu, Siyuan Li, Yufei Huang, Jun Xia, Bozhen Hu, and Stan Z Li. Metoken: Uniform micro-environment token boosts post-translational modification prediction. arxiv, 2024.
- [176] Cheng Tan, Zhangyang Gao, Lirong Wu, Jun Xia, Jiangbin Zheng, Xihong Yang, Yue Liu, Bozhen Hu, and Stan Z Li. Cross-gate mlp with protein complex invariant embedding is a one-shot antibody designer. In AAAI, volume 38, pages 15222–15230, 2024.
- [177] Cheng Tan, Jingxuan Wei, Zhangyang Gao, Linzhuang Sun, Siyuan Li, Ruifeng Guo, Bihui Yu, and Stan Z Li. Boosting the power of small multimodal reasoning models to match larger models with self-consistency training. In ECCV, pages 305–322. Springer, 2024.
- [178] Cheng Tan, Qi Chen, Jingxuan Wei, Gaowei Wu, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, and Stan Z Li. Sketchagent: Generating structured diagrams from hand-drawn sketches. arxiv, 2025.
- [179] Cheng Tan, Yijie Zhang, Zhangyang Gao, Yufei Huang, Haitao Lin, Lirong Wu, Fandi Wu, Mathieu Blanchette, and Stan Z Li. dyab: Flow matching for flexible antibody design with alphafold-driven pre-binding antigen. In AAAI, volume 39, pages 782–790, 2025.
- [180] Hao Tang, Ling Shao, Nicu Sebe, and Luc Van Gool. Enhanced multi-scale cross-attention for person image generation. IEEE TPAMI, 47:3377–3393, 2025.
- [181] Jiabin Tang, Lianghao Xia, Zhonghang Li, and Chao Huang. Ai-researcher: Autonomous scientific innovation. arxiv, 2025.
- [182] Mingkai Tang, Yuanhang Li, Hongji Liu, Yingbing Chen, Ming Liu, and Lujia Wang. Mgcbs: an optimal and efficient algorithm for solving multi-goal multi-agent path finding problem. arxiv, 2024.
- [183] Tao Tang, Enhui Ma, Xia Zhou, Letian Wang, Tianyi Yan, Xueyang Zhang, Kun Zhan, Peng Jia, Xianpeng Lang, Jia-Wang Bian, et al. Omnigen: Unified multimodal sensor generation for autonomous driving. In ACM MM, pages 9365–9374, 2025.
- [184] Xiangjun Tang, Biao Zhang, and Peter Wonka. Generative human geometry distribution. arxiv, 2025.
- [185] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [186] Rakshit Trivedi, Kartik Sharma, and David C Parkes. Inner speech as behavior guides: Steerable imitation of diverse behaviors for human-ai coordination. arxiv, 2026.
- [187] Yuanpeng Tu, Hao Luo, Xi Chen, Xiang Bai, Fan Wang, and Hengshuang Zhao. Playerone: Egocentric world simulator. arxiv, 2025.
- [188] Vijay Viswanathan, Yanchao Sun, Shuang Ma, Xiang Kong, Meng Cao, Graham Neubig, and Tongshuang Wu. Checklists are better than reward models for aligning language models. arxiv, 2025.
- [189] Christian Walder and Deep Karkhanis. Pass@ k policy optimization: Solving harder reinforcement learning problems. arxiv, 2025.
- [190] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. MinerU: An open-source solution for precise document content extraction. arxiv, 2024. URL https: //arxiv.org/abs/2409.18839.
- [191] Cong Wang, Jinshan Pan, Liyan Wang, Wei Wang, and Yang Yang. Neural discrimination-prompted transformers for efficient uhd image restoration and enhancement: C. wang et al. IJCV, 134:84, 2026.
- [192] Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, and Liwei Wang. Git: Towards generalist vision transformer through universal language interface. In ECCV, pages 55–73. Springer, 2024.
- [193] Lei Wang and Piotr Koniusz. Feature hallucination for self-supervised action recognition: L. wang, p. koniusz. IJCV, 133:7612–7646, 2025.

- [194] Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang, Li Shen, Yong Luo, Bo Du, and Dacheng Tao. Retrieval-augmented perception: High-resolution image perception meets visual rag. arxiv, 2025.
- [195] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. OpenHands: An open platform for AI software developers as generalist agents. arxiv,

2024. URL https://arxiv.org/abs/2407.16741.

- [196] Xuesong Wang, He Zhao, and Edwin V Bonilla. R´enyi neural processes. arxiv, 2024.
- [197] Yi Wang, Zhitong Xiong, Chenying Liu, Adam J Stewart, Thomas Dujardin, Nikolaos Ioannis Bountos, Angelos Zavras, Franziska Gerken, Ioannis Papoutsis, Laura Leal-Taix´e, et al. Towards a unified copernicus foundation model for earth vision. In ICCV, pages 9888–9899, 2025.
- [198] Ziang Wang, Xiaoqin Wang, Dingyi Wang, Qiang Li, and Shushan Qiao. Dime-net: A dual-illumination adaptive enhancement network based on retinex and mixture-of-experts. In ACM MM, pages 8184–8193, 2025.
- [199] Ziyi Wang, Yongming Rao, Shuofeng Sun, Xinrun Liu, Yi Wei, Xumin Yu, Zuyan Liu, Yanbo Wang, Hongmin Liu, Jie Zhou, et al. Vision generalist model: A survey: Z. wang et al. IJCV, 133:6639–6667, 2025.
- [200] Jingxuan Wei, Linzhuang Sun, Yichong Leng, Xu Tan, Bihui Yu, and Ruifeng Guo. Sentence-level or token-level? a comprehensive study on knowledge distillation. arxiv, 2024.
- [201] Jingxuan Wei, Caijun Jia, Qi Chen, Honghao He, Linzhuang Sun, Conghui He, Lijun Wu, Bihui Yu, and Cheng Tan. Geoint-r1: Formalizing multimodal geometric reasoning with dynamic auxiliary constructions. arxiv, 2025.
- [202] Jingxuan Wei, Cheng Tan, Qi Chen, Gaowei Wu, Siyuan Li, Zhangyang Gao, Linzhuang Sun, Bihui Yu, and Ruifeng Guo. From words to structured visuals: A benchmark and framework for text-to-diagram generation and editing. In CVPR, pages 13315–13325, 2025.
- [203] Tonglong Wei, Youfang Lin, Yan Lin, Shengnan Guo, Lan Zhang, and Huaiyu Wan. Micro-macro spatialtemporal graph-based encoder-decoder for map-constrained trajectory recovery. IEEE TKDE, 36:6574–6587,

- 2024.

[204] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, et al. Videorope: What makes for good video rotary position embedding? arxiv,

- 2025.

- [205] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. Cycleresearcher: Improving automated research via automated review. arXiv preprint arXiv:2411.00816, 2024.
- [206] Daiqing Wu, Dongbao Yang, Yu Zhou, and Can Ma. Robust multimodal sentiment analysis of image-text pairs by distribution-based feature recovery and fusion. In ACM MM, pages 5780–5789, 2024.
- [207] Lirong Wu, Haitao Lin, Cheng Tan, Zhangyang Gao, and Stan Z Li. Self-supervised learning on graphs: Contrastive, generative, or predictive. IEEE TKDE, 35:4216–4235, 2021.
- [208] Lirong Wu, Yufei Huang, Cheng Tan, Zhangyang Gao, Bozhen Hu, Haitao Lin, Zicheng Liu, and Stan Z Li. Psc-cpi: Multi-scale protein sequence-structure contrasting for efficient and generalizable compound-protein interaction prediction. In AAAI, volume 38, pages 310–319, 2024.
- [209] Lirong Wu, Haitao Lin, Yufei Huang, Zhangyang Gao, Cheng Tan, Yunfan Liu, Tailin Wu, and Stan Z Li. Relation-aware equivariant graph networks for epitope-unknown antibody design and specificity optimization. In AAAI, volume 39, pages 895–904, 2025.
- [210] XiaomiMiMo. MiMo-V2.5-Pro Model Card. https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro, 2026. Accessed: 2026-05-03.
- [211] Mengling Xu, Ming Tao, and Bing-Kun Bao. Chain-of-cooking: Cooking process visualization via bidirectional chain-of-thought guidance. In ACM MM, pages 9287–9295, 2025.

- [212] Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Zifeng Wang, Xiangru Tang, Hang Wu, May D Wang, Peifeng Ruan, Donghan Yang, et al. Medagentgym: A scalable agentic training environment for code-centric reasoning in biomedical data science. arxiv, 2025.
- [213] Xiang Xu, Lingdong Kong, Hui Shuai, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, and Qingshan Liu. 4d contrastive superflows are dense 3d representation learners. In ECCV, pages 58–80. Springer, 2024.
- [214] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. 2019. doi: 10.48550/ARXIV.1912.13318. URL https://arxiv.org/abs/1912.13318. Publisher: arXiv Version Number: 5.
- [215] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1192–1200, August 2020. doi: 10.1145/3394486.

3403172. URL http://arxiv.org/abs/1912.13318. arXiv:1912.13318 [cs].

- [216] Feng Xue, Yicong Chang, Tianxi Wang, Yu Zhou, and Anlong Ming. Indoor obstacle discovery on reflective ground via monocular camera. IJCV, 132:987–1007, 2024.
- [217] Junxiao Xue, Xiaozhen Liu, Xuecheng Wu, Xinyi Yin, Danlei Huang, and Fei Yu. Ad-avsr: Asymmetric dual-stream enhancement for robust audio-visual speech recognition. In ACM MM, pages 3–11, 2025.
- [218] Vivek Yadav and Chandrashekar Ramanathan. Automated layout preservation in cross language translation of document: an integrated approach and implementation. In Proceedings of the 7th ACM India Computing Conference, pages 1–8, Nagpur India, October 2014. ACM. ISBN 978-1-60558-814-8. doi: 10.1145/2675744.

2675750. URL https://dl.acm.org/doi/10.1145/2675744.2675750.

- [219] Song Yan, Hui Wei, Jinlong Fei, Guoliang Yang, Zhengyu Zhao, and Zheng Wang. Universally unfiltered and unseen: Input-agnostic multimodal jailbreaks against text-to-image model safeguards. In ACM MM, pages 11279–11287, 2025.
- [220] Tianyu Yan, Zifu Wan, Xinhao Deng, Pingping Zhang, Yang Liu, and Huchuan Lu. Mas-sam: Segment any marine animal with aggregated features. arxiv, 2024.
- [221] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [222] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. NeurIPS, 37, 2024. URL https://arxiv.org/abs/2405.15793.
- [223] Qianyun Yang, Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, and Liqiang Nie. Stable: Efficient hybrid nearest neighbor search via magnitude-uniformity and cardinality-robustness. arxiv, 2026.
- [224] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multimodal large language models for vision-driven embodied agents. arxiv, 2025.
- [225] Xihong Yang, Cheng Tan, Yue Liu, Ke Liang, Siwei Wang, Sihang Zhou, Jun Xia, Stan Z Li, Xinwang Liu, and En Zhu. Convert: Contrastive graph clustering with reliable augmentation. In ACM MM, pages 319–327, 2023.
- [226] Zhao Yang, Yi Duan, Jiwei Zhu, Ying Ba, Chuan Cao, and Bing Su. Extending sequence length is not all you need: Effective integration of multimodal signals for gene expression prediction. arxiv, 2026.
- [227] Hangting Ye, Peng Wang, Wei Fan, Xiaozhuang Song, He Zhao, Dandan Guo, and Yi Chang. Deep tabular representation corrector. IEEE TPAMI, 2025.
- [228] Zijin Yin, Bing Li, Kongming Liang, Hao Sun, Zhongjiang He, Zhanyu Ma, and Jun Guo. Benchmarking semantic segmentation models via appearance and geometry attribute editing. IEEE TPAMI, 2026.
- [229] Hanzhe Yu, Yun Ye, Jintao Rong, Qi Xuan, and Chen Ma. Realhd: A high-quality dataset for robust detection of state-of-the-art ai-generated images. In ACM MM, pages 11394–11403, 2025.

- [230] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, pages 100–111, 2025.
- [231] Xiaohang Zhan and Dingming Liu. Larender: Training-free occlusion control in image generation via latent rendering. In ICCV, pages 19679–19688, 2025.
- [232] Chunkai Zhang, Jiarui Deng, Maohua Lyu, Wensheng Gan, and Philip S Yu. High-utility sequential rule mining utilizing segmentation guided by confidence. arxiv, 2026.
- [233] Jingmao Zhang, Zhiting Zhao, Yunqi Lin, Jianghong Ma, Tianjun Wei, Haijun Zhang, and Xiaofeng Zhang. Dual-phase playtime-guided recommendation: Interest intensity exploration and multimodal random walks. In ACM MM, pages 6232–6241, 2025.
- [234] Peirong Zhang, Kai Ding, and Lianwen Jin. Capturing more: Learning multi-domain representations for robust online handwriting verification. In ACM MM, pages 1471–1479, 2025.
- [235] Qingwen Zhang, Xiaomeng Zhu, Yushan Zhang, Yixi Cai, Olov Andersson, and Patric Jensfelt. Deltaflow: An efficient multi-frame scene flow estimation method. arxiv, 2025.
- [236] Tong Zhang, Yifan Zhao, Liangyu Wang, and Jia Li. Free lunch to meet the gap: Intermediate domain reconstruction for cross-domain few-shot learning. IJCV, 133:5118–5137, 2025.
- [237] Xiang Zhang, Yulun Zhang, and Fisher Yu. Hit-sr: Hierarchical transformer for efficient image superresolution. In ECCV, pages 483–500. Springer, 2024.
- [238] Xin Zhang, Ling Chen, Xing Tang, and Hongyu Shi. Dsgnn: A dual-view supergrid-aware graph neural network for regional air quality estimation. arxiv, 2024.
- [239] Yan Zhang, Gangyan Zeng, Daiqing Wu, Huawen Shen, Binbin Li, Yu Zhou, Can Ma, and Xiaojun Bi. Gather and trace: Rethinking video textvqa from an instance-oriented perspective. In ACM MM, pages 876–885, 2025.
- [240] Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, and Jun Zhu. Stair: Improving safety alignment with introspective reasoning. arxiv, 2025.
- [241] Yifan Zhang and Junhui Hou. Is contrastive distillation enough for learning comprehensive 3d representations? arxiv, 2024.
- [242] Yuanhe Zhang, Fanghui Liu, and Yudong Chen. Lora-one: One-step full gradient could suffice for finetuning large language models, provably and efficiently. arxiv, 2025.
- [243] Zhiyuan Zhang, Licheng Yang, and Zhiyu Xiang. Risurconv: Rotation invariant surface attention-augmented convolutions for 3d point cloud classification and segmentation. In ECCV, pages 93–109. Springer, 2024.
- [244] Zhongqun Zhang, Hengfei Wang, Ziwei Yu, Yihua Cheng, Angela Yao, and Hyung Jin Chang. Nl2contact: Natural language guided 3d hand-object contact modeling with diffusion model. In ECCV, pages 284–300. Springer, 2024.
- [245] Chen Zhao, Xuan Wang, Tong Zhang, Saqib Javed, and Mathieu Salzmann. Self-ensembling gaussian splatting for few-shot novel view synthesis. In ICCV, pages 4940–4950, 2025.
- [246] Xiaoqi Zhao, Youwei Pang, Shijie Chang, Yuan Zhao, Lihe Zhang, Chenyang Yu, Hanqi Liu, Jiaming Zuo, Jinsong Ouyang, Weisi Lin, et al. Inspiring the next generation of segment anything models: Comprehensively evaluate sam and sam 2 with diverse prompts towards context-dependent concepts under different scenes. arxiv, 2024.
- [247] Zijing Zhao, Zhu Xu, Qingchao Chen, Yuxin Peng, and Yang Liu. Investigating domain gaps for indoor 3d object detection. In ACM MM, pages 13198–13205, 2025.
- [248] Jiangbin Zheng, Yile Wang, Cheng Tan, Siyuan Li, Ge Wang, Jun Xia, Yidong Chen, and Stan Z Li. Cvt-slr: Contrastive visual-textual transformation for sign language recognition with variational alignment. In CVPR, pages 23141–23150, 2023.
- [249] Jintu Zheng, Yi Ding, Qizhe Liu, Yuehui Chen, Yi Cao, Ying Hu, and Zenan Wang. Sparsessp: 3d subcellular structure prediction from sparse-view transmitted light images. In ECCV, pages 267–283. Springer, 2024.

- [250] Chong Zhou, Xiangtai Li, Chen Change Loy, and Bo Dai. Edgesam: Prompt-in-the-loop distillation for sam. IJCV, 133:8452–8468, 2025.
- [251] Sifan Zhou, Jiahao Nie, Ziyu Zhao, Yichao Cao, and Xiaobo Lu. Focustrack: One-stage focus-and-suppress framework for 3d point cloud object tracking. In ACM MM, pages 7366–7375, 2025.
- [252] Zhe Zhu, Honghua Chen, Xing He, and Mingqiang Wei. Pointsea: point cloud completion via self-structure augmentation. IJCV, 133:4770–4794, 2025.
- [253] Ziming Zhu, Chenglong Wang, Shunjie Xing, Yifu Huo, Fengning Tian, Quan Du, Di Yang, Chunliang Zhang, Tong Xiao, and Jingbo Zhu. Latextrans: Structured latex translation with multi-agent coordination. arxiv, 2025.
- [254] Wei Zhuo and Siqiang Luo. Modality-free graph in-context alignment. arxiv, 2026.

## Appendix

### A Benchmark Papers

All papers comprising the PaperFit-Bench are cataloged in Table 9–16, where complete details pertaining to the benchmark corpus are available.

Table 9: Benchmark papers organized by conference.

###### Paper Conference

AAAI 2024

Cross-gate mlp with protein complex invariant embedding is a one-shot antibody designer [176]

Psc-cpi: Multi-scale protein sequence-structure contrasting for efficient and generalizable compound-protein interaction prediction [208]

AAAI 2024

Foldtoken: Learning protein language via vector quantization and beyond [64]

AAAI 2025

Isr-dpo: Aligning large multimodal models for videos by iterative selfretrospective dpo [6]

AAAI 2025

Gim: A million-scale benchmark for generative image manipulation detection and localization [36]

AAAI 2025

Enhancing adversarial transferability with adversarial weight tuning [31]

AAAI 2025

Bias unveiled: Investigating social bias in LLM-generated code [127] AAAI 2025 Multi-view pedestrian occupancy prediction with a novel synthetic dataset [13]

AAAI 2025

GLCF: A Global-Local Multimodal Coherence Analysis Framework for Talking Face Generation Detection [34]

AAAI 2025

JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs [106]

AAAI 2025

Neural continuous-time supermartingale certificates [147] AAAI 2025 Real-time calibration model for low-cost sensor in fine-grained time series [7]

AAAI 2025

Relation-aware equivariant graph networks for epitope-unknown antibody design and specificity optimization [209]

AAAI 2025

Causal-inspired multitask learning for video-based human pose estimation [30]

AAAI 2025

dyab: Flow matching for flexible antibody design with alphafold-driven pre-binding antigen [179]

- AAAI 2025

Unveiling the Landscape of Clinical Depression Assessment: From Behavioral Signatures to Psychiatric Reasoning [40]

- AAAI 2026

AAAI 2026

PITE: Multi-Prototype Alignment for Individual Treatment Effect Estimation [25]

AAAI 2026 Spiking Heterogeneous Graph Attention Networks [23] AAAI 2026

FreDN: Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition [9]

Table 10: Benchmark papers organized by conference (cont.).

###### Paper Conference

AAAI 2026

NL2CA: Auto-formalizing Cognitive Decision-Making from Natural Language Using an Unsupervised CriticNL2LTL Framework [51]

Co-learning: Learning from noisy labels with self-supervision [168] ACM MM 2021 Convert: Contrastive graph clustering with reliable augmentation [225] ACM MM 2023 Chain-of-Cooking: Cooking Process Visualization via Bidirectional Chain-of-Thought Guidance [211]

ACM MM 2025

Capturing more: Learning multi-domain representations for robust online handwriting verification [234]

ACM MM 2025

ACM MM 2025

Gather and trace: Rethinking video textvqa from an instance-oriented perspective [239]

Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval [124]

ACM MM 2025

Universally Unfiltered and Unseen: Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards [219]

ACM MM 2025

AD-AVSR: Asymmetric Dual-stream Enhancement for Robust AudioVisual Speech Recognition [217]

ACM MM 2025

ACM MM 2025

Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset [145]

DIME-Net: A Dual-Illumination Adaptive Enhancement Network Based on Retinex and Mixture-of-Experts [198]

ACM MM 2025

AnchorSync: Global Consistency Optimization for Long Video Editing [134]

ACM MM 2025

ACM MM 2025

Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks [233]

Investigating Domain Gaps for Indoor 3D Object Detection [247] ACM MM 2025 Drawing2CAD: Sequence-to-Sequence Learning for CAD Generation from Vector Drawings [158]

ACM MM 2025

Robust multimodal sentiment analysis of image-text pairs by distribution-based feature recovery and fusion [206]

ACM MM 2024

Hud: Hierarchical uncertainty-aware disambiguation network for composed video retrieval [39]

ACM MM 2025

Omnigen: Unified multimodal sensor generation for autonomous driving [183]

ACM MM 2025

Magicfight: Personalized martial arts combat video generation [81] ACM MM 2024 RealHD: A High-Quality Dataset for Robust Detection of State-of-theArt AI-Generated Images [229]

ACM MM 2025

ACM MM 2025

Focustrack: One-stage focus-and-suppress framework for 3d point cloud object tracking [251]

Hyperspherical consistency regularization [170] CVPR 2022 Simvp: Simpler yet better video prediction [60] CVPR 2022 Temporal attention unit: Towards efficient spatiotemporal predictive

CVPR 2023

- learning [171]

CVPR 2023 General point model with autoencoding and autoregressive [118] CVPR 2023

Cvt-slr: Contrastive visual-textual transformation for sign language recognition with variational alignment [248]

Table 11: Benchmark papers organized by conference (cont.).

###### Paper Conference

CVPR 2024

Mlip: Enhancing medical visual representation with divergence encoder and knowledge-guided contrastive learning [119]

ICCV 2025

Self-ensembling gaussian splatting for few-shot novel view synthesis [245]

From words to structured visuals: A benchmark and framework for text-to-diagram generation and editing [202]

CVPR 2025

Context-aware multimodal pretraining [161] CVPR 2025 Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models [230]

ICCV 2025

Towards a unified copernicus foundation model for earth vision [197] ICCV 2025 Sparseflex: High-resolution and arbitrary-topology 3d shape modeling [74]

ICCV 2025

Mergevq: A unified framework for visual generation and representation with disentangled token merging and quantization [114]

CVPR 2025

Back on track: Bundle adjustment for dynamic scene reconstruction [32] ICCV 2025 Rayzer: A self-supervised large view synthesis model [93] ICCV 2025 Token activation map to visually explain multimodal llms [116] ICCV 2025 LOTS of Fashion! multi-conditioning for image generation via sketchtext pairing [68]

ICCV 2025

Geoint-r1: Formalizing multimodal geometric reasoning with dynamic auxiliary constructions [201]

CVPR 2026

LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering [231]

ICCV 2025

Forecasting continuous non-conservative dynamical systems in so (3) [15]

ICCV 2025

Boosting the power of small multimodal reasoning models to match larger models with self-consistency training [177]

ECCV 2024

Weakly supervised 3d object detection via multi-level visual guidance [82]

ECCV 2024

Git: Towards generalist vision transformer through universal language interface [192]

ECCV 2024

Towards neuro-symbolic video understanding [43] ECCV 2024 Learning to generate conditional tri-plane for 3d-aware expression controllable portrait animation [99]

ECCV 2024

Semgrasp: Semantic grasp generation via language aligned discretization [109]

ECCV 2024

Brave: Broadening the visual encoding of vision-language models [98] ECCV 2024 Omniview-tuning: Boosting viewpoint invariance of vision-language pre-training models [162]

ECCV 2024

SparseSSP: 3D subcellular structure prediction from sparse-view transmitted light images [249]

ECCV 2024

ECCV 2024 4D contrastive superflows are dense 3D representation learners [213] ECCV 2024

HiT-SR: Hierarchical transformer for efficient image super-resolution [237]

Table 12: Benchmark papers organized by conference (cont.).

###### Paper Conference

ECCV 2024

Ittakestwo: Leveraging peer representations for semi-supervised lidar semantic segmentation [132]

Beat-it: Beat-synchronized multi-condition 3d dance generation [87] ECCV 2024 Nl2contact: Natural language guided 3d hand-object contact modeling with diffusion model [244]

ECCV 2024

Adversarial robustification via text-to-image diffusion models [42] ECCV 2024 Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection [138]

ECCV 2024

Risurconv: Rotation invariant surface attention-augmented convolutions for 3d point cloud classification and segmentation [243]

ECCV 2024

ECCV 2024

Making large language models better planners with reasoning-decision alignment [85]

ECCV 2024 Zero-shot detection of ai-generated images [45] ECCV 2024

Towards model-agnostic dataset condensation by heterogeneous models [146]

Pifold: Toward effective and efficient protein inverse folding [59] ICLR 2022 Moganet: Multi-order gated aggregation network [111] ICLR 2022 RDesign: Hierarchical data-efficient representation learning for tertiary structure-based RNA design [174]

ICLR 2023

Knowledge-design: Pushing the limit of protein design via knowledge refinement [61]

- ICLR 2023

Semireward: A general reward model for semi-supervised learning [112] ICLR 2023 Decoupling weighing and selecting for integrating multiple graph pretraining tasks [57]

- ICLR 2024

CBGBench: Fill in the blank of protein-molecule complex binding graph [123]

- ICLR 2024

Metoken: Uniform micro-environment token boosts post-translational modification prediction [175]

- ICLR 2025

Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer [160]

ICLR 2025

Generative human geometry distribution [184] ICLR 2025 Redteamcua: Realistic adversarial testing of computer-use agents in hybrid web-os environments [122]

ICLR 2025

MedAgentGym: A Scalable Agentic Training Environment for CodeCentric Reasoning in Biomedical Data Science [212]

ICLR 2025

TabStruct: Measuring Structural Fidelity of Tabular Data [95] ICLR 2026 Why DPO is a Misspecified Estimator and How to Fix It [69] ICLR 2025 Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability [16]

ICLR 2025

Compositional Diffusion with Guided search for Long-Horizon Planning [143]

- ICLR 2025

Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction [226]

- ICLR 2026

RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format [84]

ICLR 2026

Table 13: Benchmark papers organized by conference (cont.).

###### Paper Conference

Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs [151] ICLR 2026 Modality-free Graph In-context Alignment [254] ICLR 2026

Deciphering RNA secondary structure prediction: A probabilistic k-rook matching perspective [169]

ICML 2024

ICML 2024

A graph is worth k words: Euclideanizing graph using pure transformer [62]

Re-Dock: towards flexible and realistic molecular docking with diffusion bridge [83]

ICML 2024

Vqdna: Unleashing the power of vector quantization for multi-species genomic sequence modeling [113]

- ICML 2024

R´enyi Neural Processes [196] ICML 2024 A unified framework for entropy search and expected improvement in Bayesian optimization [41]

- ICML 2025

Sundial: A family of highly capable time series foundation models [131] ICML 2025 Lora-one: One-step full gradient could suffice for fine-tuning large language models, provably and efficiently [242]

ICML 2025

Stair: Improving safety alignment with introspective reasoning [240] ICML 2025 Videojam: Joint appearance-motion representations for enhanced motion generation in video models [29]

ICML 2025

Conceptattention: Diffusion transformers learn highly interpretable features [75]

ICML 2025

Videorope: What makes for good video rotary position embedding? [204]

ICML 2025

ICML 2025

Itbench: Evaluating ai agents across diverse real-world it automation tasks [91]

Train for the worst, plan for the best: Understanding token ordering in masked diffusions [101]

ICML 2025

Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents [224]

ICML 2025

Near optimal decision trees in a SPLIT second [14] ICML 2025 Training a generally curious agent [167] ICML 2025 Retrieval-augmented perception: High-resolution image perception meets visual rag [194]

ICML 2025

MGD3: Mode-Guided Dataset Distillation using Diffusion Models [28] ICML 2025 Learning with expected signatures: Theory and applications [137] ICML 2025

IEEE TPAMI 2021

Self-supervised learning on graphs: Contrastive, generative, or predictive [207]

A survey on generative diffusion models [26] IEEE TKDE 2024 Revisiting the temporal modeling in spatio-temporal predictive learning under a unified view [173]

IEEE TPAMI 2023

IEEE TKDE 2024

DSGNN: A dual-view supergrid-aware graph neural network for regional air quality estimation [238]

Micro-macro spatial-temporal graph-based encoder-decoder for mapconstrained trajectory recovery [203]

IEEE TKDE 2024

Table 14: Benchmark papers organized by conference (cont.).

###### Paper Conference

Robgc: Towards robust graph condensation [58] IEEE TKDE 2025 Temporal-Aware Spiking Transformer Hashing Based on 3D-DWT [142] IEEE TKDE 2025 Enhanced multi-scale cross-attention for person image generation [180] IEEE TPAMI 2025

- A causality-aware paradigm for evaluating creativity of multimodal large language models [86]

IEEE TPAMI 2025

Revisiting gradient-based uncertainty for monocular depth estimation [77]

IEEE TPAMI 2025

Recent advances in discrete speech tokens: A review [73] IEEE TPAMI 2025 Document-level tabular numerical cross-checking: A coarse-to-fine approach [152]

IEEE TPAMI 2025

IEEE TPAMI 2025

AEFS: Adaptive Early Feature Selection for Deep Recommender Systems [79]

IEEE TKDE 2026

High-utility Sequential Rule Mining Utilizing Segmentation Guided by Confidence [232]

Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation [141]

IEEE TPAMI 2026

Spatio-temporal Decoupled Knowledge Compensator for Few-Shot Action Recognition [159]

IEEE TPAMI 2026

PPC-MT: Parallel Point Cloud Completion with Mamba-Transformer Hybrid Architecture [108]

IEEE TPAMI 2026

Benchmarking Semantic Segmentation Models via Appearance and Geometry Attribute Editing [228]

IEEE TPAMI 2026

Deep Tabular Representation Corrector [227] IEEE TPAMI 2025 STABLE: Efficient Hybrid Nearest Neighbor Search via MagnitudeUniformity and Cardinality-Robustness [223]

IEEE TKDE 2026

Eve: Efficient zero-shot text-based video editing with depth map guidance and temporal consistency constraints [38]

IJCAI 2024

Factchd: Benchmarking fact-conflicting hallucination detection [33] IJCAI 2024 Imperio: Language-guided backdoor attacks for arbitrary model control [44]

IJCAI 2024

Bring metric functions into diffusion models [8] IJCAI 2024 Bridging generative and discriminative models for unified visual perception with diffusion priors [53]

IJCAI 2024

Llmem: Estimating gpu memory usage for fine-tuning pre-trained llms [102]

IJCAI 2024

Group-aware coordination graph for multi-agent reinforcement learning [54]

IJCAI 2024

PEACH: Pretrained-embedding Explanation Across Contextual and Hierarchical Structure [24]

IJCAI 2024

Sentence-level or token-level? a comprehensive study on knowledge distillation [200]

IJCAI 2024

MAS-SAM: Segment any marine animal with aggregated features [220] IJCAI 2024 Meta in-context learning makes large language models better zero and few-shot relation extractors [105]

IJCAI 2024

Table 15: Benchmark papers organized by conference (cont.).

###### Paper Conference

IJCAI 2024

MGCBS: an optimal and efficient algorithm for solving multi-goal multi-agent path finding problem [182]

Fast one-stage unsupervised domain adaptive person search [46] IJCAI 2024 Interpretable network visualizations: A human-in-the-loop approach for post-hoc explainability of cnn-based image classification [18]

IJCAI 2024

IJCAI 2024

Boosting single positive multi-label classification with generalized robust loss [35]

Prompt learning for generalized vehicle routing [128] IJCAI 2024 Contrastive learning is not optimal for quasiperiodic time series [12] IJCAI 2024 Marco: a memory-augmented reinforcement framework for combinatorial optimization [65]

- IJCAI 2024

Sketchagent: Generating structured diagrams from hand-drawn sketches [178]

- IJCAI 2025

Multimodal regression for enzyme turnover rates prediction [78] IJCAI 2025 EdgeSAM: Prompt-in-the-loop distillation for SAM [250] IJCV 2025 Indoor obstacle discovery on reflective ground via monocular camera [216]

IJCV 2024

Cyclic refiner: Object-aware temporal representation learning for multiview 3d detection and tracking [72]

IJCV 2024

IJCV 2025

From easy to hard: Learning curricular shape-aware features for robust panoptic scene graph generation [165]

AgMTR: Agent mining transformer for few-shot segmentation in remote sensing [17]

IJCV 2025

One-shot generative domain adaptation in 3d gans [120] IJCV 2025 Calibrated cache model for few-shot vision-language model adaptation [52]

- IJCV 2024

Globally correlation-aware hard negative generation [155] IJCV 2025 Multimodal alignment and fusion: A survey [115] IJCV 2024 Rethinking generalizability and discriminability of self-supervised learning from evolutionary game theory perspective [107]

- IJCV 2025

Inspiring the next generation of segment anything models: Comprehensively evaluate sam and sam 2 with diverse prompts towards contextdependent concepts under different scenes [246]

Is Contrastive Distillation Enough for Learning Comprehensive 3D Representations? [241]

IJCV 2024

- IJCV 2024

PointSea: point cloud completion via self-structure augmentation [252] IJCV 2025 Creatively Upscaling Images with Global-Regional Priors [157] IJCV 2025 Vision Generalist Model: A Survey: Z. Wang et al. [199] IJCV 2025 Feature Hallucination for Self-supervised Action Recognition: L. Wang, P. Koniusz [193]

- IJCV 2025

Unsupervised Robust Domain Adaptation: Paradigm, Theory and Algorithm: F. Huang etal [80]

- IJCV 2026

Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning [236]

IJCV 2025

Table 16: Benchmark papers organized by conference (cont.).

###### Paper Conference

IJCV 2026

Object-Scene-Camera Decomposition and Recomposition for Data Efficient Monocular 3D Object Detection [104]

Neural Discrimination-Prompted Transformers for Efficient UHD Image Restoration and Enhancement: C. Wang et al. [191]

IJCV 2026

Harnessing hard mixed samples with decoupled regularizer [133] NeurIPS 2023 Openstl: A comprehensive benchmark of spatio-temporal predictive

NeurIPS 2023

- learning [172]

Uniif: Unified molecule inverse folding [63] NeurIPS 2024 Large language diffusion models [148] NeurIPS 2025 Pan-lut: Efficient pan-sharpening via learnable look-up tables [22] NeurIPS 2025 Mean flows for one-step generative modeling [67] NeurIPS 2025 Pass@ k policy optimization: Solving harder reinforcement learning problems [189]

NeurIPS 2025

Web-shepherd: Advancing prms for reinforcing web agents [27] NeurIPS 2025 Ai-researcher: Autonomous scientific innovation [181] NeurIPS 2025 Qoq-med: Building multimodal clinical foundation models with domainaware grpo training [47]

NeurIPS 2025

Learning long range dependencies through time reversal symmetry breaking [156]

NeurIPS 2025

Playerone: Egocentric world simulator [187] NeurIPS 2025 Ambient diffusion omni: Training good models with bad data [48] NeurIPS 2025 Trident: Tri-modal molecular representation learning with taxonomic annotations and local correspondence [92]

NeurIPS 2025

Checklists are better than reward models for aligning language models [188]

NeurIPS 2025

DeltaFlow: An efficient multi-frame scene flow estimation method [235] NeurIPS 2025 ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression [21]

NeurIPS 2025

Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency [121]

NeurIPS 2025

Infinitystar: Unified spacetime autoregressive modeling for visual generation [129]

- NeurIPS 2025

Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI coordination [186]

- NeurIPS 2026

### B Prompt Records

For reproducibility, all model-facing baselines are associated with fixed prompt templates and saved run artifacts. RuleLog is not prompt-based: it is a deterministic rule/log baseline and therefore has no LLM prompt. TextST, TextMR, and VisualST use versioned text and vision prompt templates in the baseline adapter implementation. VisualMR writes the actual per-case agent prompt to reports/agent prompt.txt. PaperFit’s agent-backed runs write the actual per-case prompt to reports/claude prompt.txt, together with raw responses and usage reports.

The prompt templates also define the input boundary for each baseline. TextST is restricted to sourceonly repair, TextMR adds compile-log feedback, VisualST adds rendered page images but only one

Table 17: Prompt and artifact record for prompt-based methods.

Method Prompt role Template source Saved artifacts

RuleLog Rule/log only None Rule reports TextST Source-only edit Adapter prompts Response, usage, boundary TextMR Source + log edit Adapter prompts Responses, usage, boundaries VisualST Source + image edit Adapter prompts Response, usage, boundary VisualMR Fixed-round visual agent Agent prompt builder Prompt, response, usage PaperFit Structured repair agent PaperFit agent prompt Prompt, response, usage

visual edit turn, and VisualMR uses a fixed-round agent instruction that explicitly forbids PaperFit runtime, PaperFit skills, structured repair plans, taxonomy documents, and gatekeeper artifacts. PaperFit’s prompt, in contrast, includes the VTO taxonomy, forbidden operations, repair priority, and quality-gate workflow used by the proposed system. These prompt records make the baseline boundaries auditable and prevent hidden prompt differences from being treated as implementation details.

Prompt templates. The following boxes show the core prompt templates used by the prompt-based methods. Case-specific fields such as the main TeX filename, target page count, maximum rounds, source window, compile-log excerpt, and rendered page images are filled at runtime.

###### TextST Source-Only Repair Prompt

Role. You are a LaTeX editing model for academic paper layout repair. You only revise the provided TeX source directly.

Input evidence. Main .tex source only. No compile log and no rendered page images. Hard constraints.

- • Preserve academic meaning, figures, tables, captions, labels, citations, and bibliography entries.
- • Do not rewrite the whole paper; prefer one small local edit.
- • Keep unchanged lines byte-identical whenever possible.
- • Prefer local edits around floats, widths, line breaks, and spacing-safe LaTeX parameters.
- • If a safe local fix is not possible, return NO CHANGE.

Output form. Return either NO CHANGE or a revised TeX block, followed by a boundary-report JSON block describing the edit target and input scope.

- Figure 13: Prompt template used for TextST source-only LaTeX repair. The method receives only the main TeX source, applies a small local source edit when safe, and records a boundary report for traceability.

###### TextMR Source-and-Log Repair Prompt

Role. You are a text/log LaTeX repair model. You revise the TeX source directly using source-level evidence and compile-log feedback.

Input evidence. Main .tex source plus compile-log excerpt when available. No rendered page images. Repair loop.

- • Inspect compile errors, undefined controls, overfull warnings, template mismatch hints, and local source signals.
- • Propose a small source edit that addresses the most plausible local cause.

- • Keep the edit within the selected source window when a local target is provided.
- • Stop with NO CHANGE if the requested repair requires visual judgment or broad rewriting.

Output form. Return a revised TeX block or local replacement window, together with a boundary-report JSON block using input scope source plus log.

- Figure 14: Prompt template used for TextMR source-and-log LaTeX repair. The method augments source-only editing with compile-log feedback while still excluding rendered page images.

VisualST Single-Turn Visual Repair Prompt

Role. You are a vision-language model for academic paper layout repair. You may read TeX source, compile logs, and rendered page images, but you may not use structured planning artifacts, repair plans, gatekeeper logic, or a custom execution pipeline.

Input evidence. Source pack and rendered page images from the pre-repair output. One visual repair turn only.

Hard constraints.

- • Preserve academic meaning and all protected paper objects.
- • Do not add explanatory prose outside the required output form.
- • Do not rewrite the whole paper.
- • Prefer at most one local edit in one file, targeting visible layout defects such as overwide figures, underused table width, float placement, or local spacing.
- • Return NO CHANGE if a safe local fix cannot be identified from the images.

Output form. Return either NO CHANGE, a revised main TeX source, or one revised included TeX file, followed by the boundary-report JSON block.

- Figure 15: Prompt template used for VisualST single-turn visual repair. The method receives rendered page images and source context, then performs one constrained visual edit without a structured repair workflow.

###### VisualMR Fixed-Round Visual Agent Prompt

Role. You are an autonomous LaTeX paper layout repair agent running in unattended benchmark mode. Scope. Work only inside the current case directory. You may inspect and edit .tex, .bib, .cls, .sty, image metadata, and LaTeX build artifacts. You may run ordinary LaTeX and local PDF/image tools.

Forbidden PaperFit resources. Do not use PaperFit runtime, PaperFit CLI commands, PaperFit skills, PaperFit scripts, PaperFit repair plans, PaperFit gatekeeper outputs, PaperFit taxonomy documents, or any PaperFit structured artifacts.

###### Fixed-round loop.

- 1. Compile with latexmk or an equivalent local LaTeX command.
- 2. If compilation fails, inspect the log and repair the root LaTeX error.
- 3. If a PDF exists, inspect or render pages using ordinary local tools.
- 4. Edit source files minimally, then recompile and reinspect.
- 5. Stop when the paper compiles/renders cleanly and no obvious layout issue remains, or after the maximum number of repair rounds.

Output form. Return compact JSON with success, rounds used, compile/render status, summary, and remaining issues.

- Figure 16: Prompt template used for VisualMR fixed-round visual agent repair. The method can iterate over source, logs, and rendered pages for a fixed round budget while explicitly excluding PaperFit structured artifacts.

PaperFit (OURS) Structured Repair Agent Prompt

Role. You are running in unattended benchmark mode for academic paper layout repair using the PaperFit closed-loop workflow.

Execution rules.

- • Work only inside the current case directory.
- • Preserve all academic meaning and content.
- • Compile and render after every edit before proceeding.
- • Use the target page budget when provided.

Injected VTO knowledge. Diagnose and repair defects using the five-category VTO taxonomy: space utilization, float placement, typographic consistency, overflow, and cross-template migration.

Repair policy. Fix compile errors first, then overflow, float placement, space utilization, typographic consistency, and cross-template issues. Avoid pseudo-layout fixes such as brute-force spacing, forced page breaks, table scaling, object deletion, or content rewriting.

Quality gate. After each repair cycle, verify compilation, rendering, page-level visual inspection, residual defect status, page-budget satisfaction, and protected content preservation. Continue until all gate conditions pass or the maximum round budget is reached.

Output form. Return JSON with success, rounds used, summary, remaining defects, and notes.

- Figure 17: Prompt template used for PaperFit structured repair. The method injects the VTO taxonomy, repair priority, forbidden operations, and checklist quality gate used by the proposed closed-loop system.

### C Reproducibility Notes

For each method and case, the evaluation records the generated source, compile logs, rendered pages when available, programmatic metric outputs, and VLM reports. Aggregated tables are computed from case-level reports rather than from hand-entered summary values. Missing, non-compilable, or non-renderable outputs are not silently dropped; they are handled by the same failure accounting rules used for all methods. All LaTeX outputs are compiled with the local TeX toolchain and rendered into page images before visual evaluation. Baseline outputs and aggregate metrics are stored under the baseline result directory, while the paper tables are generated from the completed method-level summaries. The release package will include the benchmark metadata, disturbance manifests, baseline implementations, evaluation scripts, and aggregation scripts needed to reproduce the programmatic metrics and VLM-based visual scoring.

### D Limitations

PaperFit’s visual inspection depends on a VLM evaluator; subtle or ambiguous layout issues—such as microtypographic defects or font-level kerning errors—may still be missed by current vision models. On hard cases with 5–8 co-occurring perturbations, the page-budget hit rate drops to approximately 70%, indicating that highly complex multi-defect scenarios remain challenging.

The system is currently limited to LaTeX projects and has been evaluated only on English-language academic papers; coverage of other document languages is left to future work. Finally, multi-round recompilation and re-rendering incurs higher computational cost than single-pass methods, and reducing this overhead while preserving repair quality is an important practical direction.

