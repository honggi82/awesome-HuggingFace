# arXiv:2512.13168v5[cs.AI]15Apr2026

[Figure 1]

## FINCH: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows

###### FINCH members∗

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Structuring/

Calculation Visualization

Data Entry/Import

Formatting

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### Multimodal Enterprise Workspace:

Text, Tables, Formulas, Pivots,

Search/Retrieval Validation Financial Modeling Reporting

Charts, Code, Images…

Figure 1: Real-world F&A work is messy, spanning heterogeneous and large-scale artifacts such as spreadsheets and PDFs. It’s also long-horizon and knowledge-intensive: workflows interleave multiple tasks and span diverse domains such as budgeting, trading, asset management, and operations.

#### Abstract

We introduce FinWorkBench (a.k.a. FINCH) for evaluating agents on real-world, enterprise-grade finance and accounting workflows—interleaving data entry, structuring, formatting, web search, cross-file retrieval, calculation, modeling, validation, translation, visualization, and reporting. FINCH is sourced from authentic enterprise workspaces from Enron (15,000 files and 500,000 emails) and across various financial institutions, covering the period 2000–2025, preserving the in-the-wild messiness of multimodal artifacts such as tables and charts, across diverse domains including budgeting, trading, and asset management.

We propose a workflow construction process that combines LLM-assisted mining of workflows from authentic enterprise environments with expert annotation: (1) LLMassisted, expert-verified derivation of workflows from real-world email threads and version histories of spreadsheet files, and (2) meticulous annotation for workflows, requiring over 700 hours of expert effort. This yields 172 composite workflows with 384 tasks, involving 1,710 spreadsheets with 27 million cells, along with PDFs and other artifacts, capturing the intrinsically messy, long-horizon, knowledgeintensive, and collaborative nature of enterprise work.

We conduct both human and automated evaluations of frontier AI systems, including GPT 5.1, Claude Sonnet/Opus 4.5, Gemini 3 Pro, Grok 4, and Qwen 3 Max. GPT 5.1 Pro spends an average of 16.8 minutes per workflow yet passes only 38.4% of workflows. Comprehensive case studies further surface the challenges that real-world enterprise workflows pose for AI agents.

[Figure 12]

Dataset: https://huggingface.co/FinWorkBench

∗Correspondence to donghaoyu82@gmail.com; Full author list: Appendix A.

Preprint.

#### 1 Introduction

Frontier AI systems are increasingly transforming professional workspaces. AI-assisted tools like Claude [1], OpenClaw [42], ChatGPT [41], Gemini [21], and Copilot [39] are now embedded in daily enterprise workflows—helping professionals draft documents, explore data, manipulate spreadsheets, and generate reports. These tools are particularly impactful in finance and accounting (F&A), a high-stakes, knowledge- and labor-intensive domain critical to every organization.

However, real-world F&A work is inherently messy, with substantial contextual complexity: artifacts are interconnected across heterogeneous spreadsheets, PDFs, and other artifacts, evolving through multiple versions with collaborative edits [28]; spreadsheets contain large, complex structures [17] with cross-sheet references, intricate layouts, inconsistent formatting, cryptic terms, erroneous formulas, and multimodal artifacts such as charts, images, and code. It is also long-horizon [43]: workflows demand multi-step reasoning spanning data entry, editing, retrieval, calculation, modeling, validation, reporting, and more.

This raises a key question: Can today’s frontier AI agents actually handle the messy, long-horizon, and knowledge-intensive workflows that professionals face daily?

To answer this, we introduce FINCH, an enterprise-grade F&A benchmark sourced from authentic enterprise environments. FINCH captures the intrinsic complexity of professional work through:

- • In-the-wild enterprise sourcing: FINCH is built around authentic enterprise spreadsheets, emails, and PDFs from real-world enterprise workspaces—primarily Enron [19] (about 15,000 spreadsheet files and 500,000 emails from 150 executives and employees) and EUSES [20] (about 450 financial spreadsheet files from various sources), along with financial institutions, global organizations, and government agencies, including the International Debt Report 2025, World Bank 2024 reports, Adidas 2024 financial statement, Public Accounts of Canada 2024 [4, 11, 24], etc. Documents are large, cross-referenced, and messy—containing rich multimodal artifacts such as tables, formulas, charts, pivots, and images.
- • Rigorous construction process: We propose a novel workflow construction pipeline grounded in real collaborative context of emails and versioned artifacts. We induce workflows from enterprise email threads and attachments, where collaborators naturally describe, discuss, and track workflows as part of their daily work. We also propose an LLM-assisted, expert-verified method to derive workflows by analyzing changes across versioned spreadsheets, surfacing the underlying goals that drive professionals’ work. Annotators must reason over large multi-sheet workbooks, and subtle version deltas to infer what the original analyst was trying to achieve, making the annotation process substantially more difficult than curating QA pairs over isolated tables.

We compile 172 meticulously annotated, enterprise-grade workflows built on 1,710 spreadsheets, along with PDFs and other artifacts, collectively capturing the compositional, messy, knowledgeintensive, and collaborative nature of real work. Each workflow spans one or more interdependent tasks—data entry, editing, retrieval, calculation, modeling, validation, translation, visualization, and reporting—mirroring how professionals actually work on artifact manipulation and creation. Beyond their compositional task structure, FINCH workflows demand substantial multi-step agent interaction: a case study on 20 representative workflows reveals that each requires a median of 16 tool calls (range: 6–107).

[Figure 13]

- Figure 2: Model pass-rate comparison on FINCH workflows. Bars show overall workflow success rates for product-side agents and API-based models. Detailed settings can be found in Section 3.

Evaluating such workflows poses nontrivial challenges, because FINCH tasks usually involve complex and large spreadsheets and require assessing both structural correctness and semantic fidelity beyond

exact matching. To address this, we provide both expert human evaluation and a scalable LLM-asjudge pipeline. Human evaluation serves as the gold standard, judging whether a workflow has been satisfactorily completed end-to-end. In parallel, we introduce an efficient multimodal LLM-as-judge evaluation method that compares inputs, model outputs, and reference artifacts using structured diffs, compact snapshots, and multimodal renderings, enabling reliable assessment at scale.

We evaluate a spectrum of frontier AI systems—including Claude Sonnet/Opus 4.5, GPT 5.1, Gemini 3, Grok 4, and Qwen 3—using both expert evaluation and our automated pipeline. Even the strongest frontier agents pass fewer than 50% of FINCH workflows: GPT 5.1 Pro spends an average of 16.8 minutes per workflow yet achieves only 38.4% under human evaluation.

Case analyses further reveal how each dimension of real-world complexity contributes to failure. Long-horizon composition is a key bottleneck: GPT 5.1 Pro drops from 44.3% on workflows with ≤2 tasks to 23.5% on those with >2 tasks, as errors accumulate across steps. The messy, irregular structure of enterprise spreadsheets leads to frequent data retrieval errors, and agents often struggle to reconstruct latent knowledge-intensive business logic encoded in spreadsheet formulas. Multimodal artifacts further amplify difficulty: on the 20 workflows involving PDFs, images, or Word documents, GPT 5.1 Pro’s pass rate drops to 35.0%, below its overall rate. Taken together, these results suggest that it is the combination of messy inputs, long-horizon dependencies, domain knowledge, and heterogeneous formats—rather than any single factor—that drives the sharp performance degradation on real enterprise workflows.

Method 1: Go Long Index 1: When Moving Average is either rising or flat, AND the Deviation Rate falls within the specified open interval. Go Long Index 2 (i.e., -1): When Price <= Moving Average, AND Deviation Rate falls within the specified open interval. Otherwise: Maintain the most recent.

[Figure 14]

| |A|B|C|
|---|---|---|---|
|13|Date|000922.CSI|399370.SZ|
|14|2004-12-31|1000.0000|878.1710|
|15|2005-01-04|981.5550|861.9958|
|16|2005-01-05|986.5350|867.8566|
|17|2005-01-06|975.5260|858.8290|
|18|2005-01-07|980.7170|860.6270|
|19|2005-01-10|989.5290|869.8089|
|20|2005-01-11|990.2840|872.3875|
|21|2005-01-12|989.8200|872.7026|

Cross-sheet/

Collect Cleaning

file Retrieval

Online Sources

Validation Consistency

| |A|B| |
|---|---|---|---|
| |Model Parameters| | |
|9|MA| |60|
|10|Deviation Limit| |10%|
|11|Timing Model| |Method 1|

Data Vendors

| | | |
|---|---|---|
|Search Parameters| | |
|Dividend| |000922.CSI|
|Growth| |399370.SZ|
|Start Date| |2004/12/31|
|End Date| |2025/11/5|

Web Search/ Data Entry Import

Python Packages

Finance Modeling & Calculation

[Figure 15]

###### =IF(AND(E14>=E15,F15<$B$10),-

=IF($B$11=“Method 2”, U15,T15)

=AVERAGE(OFFSET(D15,0,0,MAX(13-ROW(B15),-$B$9),1)) 1,IF(AND(E14<=E15,F15<$B$10),1,0))

Visualization

=ABS(D15/E15-1)

| |T|
|---|---|
|13|Method 1|
|14|1|
|15|-1|
|16|-1|
|17|-1|
|18|1|

| |I|J|
|---|---|---|
|13|Signal_ori|Signal|
|14|1|1|
|15|-1|=IF(I15=0,J14, I15)|
|16|-1|-1|
|17|-1|-1|
|18|1|1|

| |D|E|
|---|---|---|
|13|Relative Change|MA|
|14|1.139|1.139|
|15|=B15/C15|1.139|
|16|1.137|1.138|
|17|1.136|1.138|
|18|1.140|1.138|

| |F|
|---|---|
|13|Bias Ratio|
|14|0.000|
|15|0.000|
|16|0.001|
|17|0.001|
|18|0.001|

NAV chart: Rotation vs Benchmark

|Strategy Return (Aggregated)| | |
|---|---|---|
|Return|Sharpe Ratio|Maximum Drawdown (MDD)|
|9.91%|0.376|75.59%|
|Benchmark Return (Aggregated)| | |
|Return|Sharpe Ratio|Maximum Drawdown (MDD)|
|9.16%|0.362|72.03%|
|Average Annual Trades|Comparison with Benchmark| |
| |Return Improvement|Drawdown Improvement|
|12.35|0.74%|-1.56%|

Formulas: =M14*(L15*B15/B14+(1-L15)*C15/C14)

Finance Modeling & Calculation

|𝐸𝑥𝑐𝑒𝑠𝑠𝑡<br><br>= 𝑅𝑒𝑡𝑡𝑡𝑖𝑚𝑖𝑛𝑔 − 𝑅𝑒𝑡𝑡𝑏𝑒𝑛𝑐ℎ|
|---|

Report / Formatting

|Signal<br><br>Row|Position|Strategy<br><br>Net Value|Benchmark|Timing<br><br>Return|Benchmark<br><br>Return|Excess<br><br>Return|Cumulative Excess<br><br>Return|Number of<br><br>Signals|
|---|---|---|---|---|---|---|---|---|
|14|1|1.00|1.00|0.00|0.00|0.00|1.00|1.00|
|15|1|0.98|0.98|-0.02|-0.02|0.00|1.00|2.00|
|15|0|0.99|0.99|0.01|0.01|0.00|1.00|0.00|
|15|0|0.98|0.98|-0.01|-0.01|0.00|1.00|0.00|
|18|0|0.98|0.98|0.00|0.00|0.00|1.00|2.00|

[Figure 16]

𝐶𝑢𝑚𝐸𝑥𝑐𝑒𝑠𝑠𝑡

= 𝐶𝑢𝑚𝐸𝑥𝑐𝑒𝑠𝑠𝑡−1 × (1

+ 𝐸𝑥𝑐𝑒𝑠𝑠𝑡)

𝑃𝑒𝑎𝑘≤𝜇 − 𝑁𝑉𝜇 𝑃𝑒𝑎𝑘≤𝜇

𝑀𝐷𝐷 = max

(

)

𝜇

|𝑅𝑒𝑡𝑢𝑟𝑛 =<br><br>𝑁𝑉𝑒𝑛𝑑 𝑁𝑉𝑠𝑡𝑎𝑟𝑡<br><br>− 1<br><br>…|
|---|

Timing model’s performance

- Figure 3: Illustration of an end-to-end predictive modeling workflow typically performed by financial analysts. The workflow involves multiple steps, including web search, data import, cross-sheet and cross-file retrieval, calculation and financial modeling, visualization, and report generation. More illustrative examples for data characteristics in FINCH are presented in Appendix F.

#### 2 FINCH: A Real-world Finance & Accounting Workflow Benchmark

###### 2.1 Dataset Construction

Echoing the existentialist dictum that existence precedes essence [44], we argue that professional work should be observed in real-world enterprise environment before it is formally defined. Motivated by this perspective, we propose a novel workflow construction pipeline grounded in the authentic collaborative context of emails, versioned spreadsheets, and final deliverable spreadsheets and reports, as illustrated in Figure 4 (a–c). First, we induce workflows from enterprise email threads and versioned documents, where collaborators naturally describe, discuss, and track work as part of their daily routines. Second, we derive workflows by analyzing changes across versioned spreadsheets, surfacing the actual data transformations and analysis steps that professionals performed. Third, we

leverage high-quality spreadsheets and reports: we design workflows, author task instructions, and revise these spreadsheets and reports so that they serve as the input files and reference solutions.

[Figure 17]

[Figure 18]

[Figure 19]

Email + partially grounded

Versioned spreadsheets

Email + strongly grounded attachments discovered by

Artifacts sourced from investment and securities companies,

attachments discovered by LLMs (Enron Email Corpus)

with LLM-discovered workflows (Enron, EUSES)

LLMs (Enron Email Corpus)

organizations, governments, etc)

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Human Annotation Human Annotation Human Annotation Human Annotation

Normalize emails into task instructions;

Normalize emails into task instructions; complete the input or reference solution

Refine LLM-induced workflows; edit LLMdrafted instructions, input/reference outputs

Design workflows; author instructions, input/reference outputs

verify completeness and no extra edits

(a) (b) (c) (d)

- Figure 4: Illustration of our workflow construction pipeline from real-world enterprise emails, versioned spreadsheets, and high-quality artifacts.

All annotated workflows from these different channels are consolidated into a unified schema with consistent fields (NL instruction, input files, reference outputs), and each workflow is tagged with task types (e.g., data entry/import, structuring, validation) and business types (e.g., planning and budgeting, pricing and valuation, operations, asset management). Note that the reference outputs may include both file-based reference answers (for most generation/editing cases) and textual reference answers (for a small number of QA and summary/visualization cases).

###### 2.1.1 Workflow from Enterprise Email Threads

We first mine real-world enterprise email threads to surface workflows. Starting from the Enron Email Corpus, we prompt GPT-5 to identify collaborative messages that (i) explicitly state a business goal (e.g., “update the RAC rankings” or “revise the 2002 allocations”) and (ii) reference one or more attached spreadsheets. For each selected thread, the model summarizes the communicative intent and articulates a workflow description.

In the strongly grounded case illustrated in Figure 4 (a), both the input and the final reference artifacts for the workflow are already present as attachments in the thread (e.g., an initial ranking file and a corrected version). Strongly grounded cases were a primary motivation of this benchmark, but they are relatively rare. In the partially grounded case illustrated in Figure 4 (b), the email specifies a clear goal, but only some of the required artifacts are attached (e.g., just the updated schedule, or an intermediate report). Across both cases, human experts normalize conversational email text and LLM-drafted descriptions into workflow instructions and abstract away idiosyncratic details while preserving the business intent. For strongly grounded threads, annotators primarily verify that the attached files exactly implement the requested change without extra edits. For partially grounded threads, they either identify the missing artifacts from attached spreadsheets and revise them to align with the described workflow—carefully avoiding the introduction of new changes—or create the missing artifacts themselves, which typically requires much more effort.

###### 2.1.2 Workflow Derivation from Versioned Spreadsheets

Beyond explicit messages in email threads, we propose to discover workflows that are implicitly captured in spreadsheet version histories, as illustrated in Figure 4(c). We collect families of versioned workbooks from the Enron and EUSES repositories and apply an LLM-based differencing procedure that recognizes consecutive versions and infers the underlying workflow.

For each recognized pair (or chain) of versions, we prompt GPT-5 to propose (i) one or more workflow types (e.g., “date-stamped versioning, assumption updates, and error correction”, “data

entry, structuring, and visualization”) and (ii) a detailed NL description of all changes. Human experts then validate and refine these LLM-induced workflow candidates. They first determine whether the proposed diffs constitute a coherent and meaningful workflow rather than incidental churn. For accepted cases, they (i) rewrite the draft description into a precise task instruction that describes the transformation, and (ii) edit the corresponding workbook versions so that the designated input and reference files cleanly realize the described workflow without introducing out-of-scope changes beyond the instruction. The corresponding input and reference files are thus grounded in the actual versions used in the diff, yielding workflow instances that do not rely on email context but are anchored in real enterprise spreadsheet evolution.

###### 2.1.3 Workflow Sourced from Final Deliverable Spreadsheets and Reports

Finally, we curate workflows from high-quality spreadsheets and reports drawn from the Enron and EUSES corpora, various investment and securities companies, international organizations, and national governments (e.g., the World Bank and the Canadian and British governments). Domain experts author realistic workflow instructions and construct input and reference files based on final deliverable artifacts. For example, a valuation model from an investment firm can be turned into a financial modeling task; a World Bank report can be used to define a data-driven summarization and visualization task; and bilingual reports from the Canadian government can be used to construct translation and consistency-checking tasks. We additionally leverage labeled samples from existing datasets: we adapt 10 financial cases from WideSearch [49] into web-search-centric workflows and extend them into multi-step calculation and visualization pipelines, and we leverage 3 examples from DABStep [18] to construct multi-source question answering workflows. We rely on WideSearch’s annotations without further revision, as they are supported by extensive web-crawled information available for validation.

###### 2.1.4 Quality Control

Given the high complexity of each workflow, we adopt a rigorous multi-stage quality control process. All workflows are annotated and reviewed by a team of five experts: two annotators with interdisciplinary backgrounds in finance and computer science, and three annotators with computer science backgrounds. Among them, three annotators (including one female) have over nine years of industry experience, and two are outstanding Ph.D. or master’s students.

Annotators are instructed to skip workflows that are similar to existing ones to maximize diversity. 20% of our annotators are full-time reviewers who do not create any original annotations, and 80% act as full-time workflow creators (producing the initial annotations) and also serve as part-time reviewers. Each workflow is proposed by one person and typically reviewed by two others, including a dedicated full-time reviewer who reviews all queries and also inspects LLM-generated review comments to surface potential annotation issues. Importantly, Quality Control (QC) is iterative rather than one-pass: approximately 40% of workflows undergo at least one round of revision; 20+ workflows went through three or more QC rounds, and we held regular group discussions to resolve challenging cases and align standards. In addition, ChatGPT 5.1 Pro and Claude Sonnet 4.5 are used as secondary checkers to flag potential issues, which are always verified by human experts. Together, these procedures required over 700 hours of expert annotation.

###### 2.2 Dataset Characteristics

FINCH comprises 172 meticulously annotated, enterprise-grade workflows that collectively capture the compositional, messy, multimodal, and collaborative nature of real finance and accounting work. Across these workflows, the corpus contains 1,710 spreadsheets (956 distinct ones in 301 Excel files) together with 17 PDFs, 12 images, 3 Word documents, as well as JSON, CSV, and Markdown. This mixture reflects how real analysts coordinate over heterogeneous artifacts rather than clean, single-table inputs.

[Figure 25]

[Figure 26]

Figure 5: Distribution of number of tasks per workflow and task types across business types.

- Figure 5 summarizes the coverage of task and business types. On the task side, categories are:

- • Calculation (119 workflows): filling in formulas or computing figures (e.g., net value).
- • Structuring / Formatting (86): reorganizing tables (e.g., adjusting hierarchies), formatting content (e.g., font size and cell fill), and inserting/deleting rows or columns.
- • Data Entry / Import (44): transcribing or importing data from spreadsheets, PDFs, images, or external sources into spreadsheets.
- • Validation / Review (37): checking consistency and reconciling calculations within a sheet or across sheets/files.
- • Cross-sheet/file Retrieval (36): pulling values from multiple sheets or files into a target workbook.
- • Summary / Visualization (33): producing summaries or charts that surface key financial insights.
- • Financial Modeling (15): extending or calibrating valuation and timing models, often via scenario and sensitivity analysis.
- • Web Search (11): collecting financial data from the web and integrating it into spreadsheets.
- • Translation (3): translating spreadsheets or reports while preserving structure, formatting, and layout.

On the business side, workflows span reporting (48 workflows), trading and risk management (35), predictive modeling (33), operational management (36), planning and budgeting (26), pricing and valuation (15), accounts payable/receivable (10), as well as procurement and sales (7) and asset management (3); some workflows are tagged with multiple business types. Overall, the distribution indicates that FINCH targets core finance and accounting verticals rather than curated toy tasks.

###### 2.2.1 Task Compositionality

FINCH is explicitly designed around composite workflows rather than isolated tasks. As shown in Figure 5, only 37 workflows (21.5%) are single-task; the remaining 135 (78.5%) involve multiple tasks. These tasks are not independent subtasks, but are interleaved around shared spreadsheets and files. A typical workflow may begin with structuring or importing raw data, proceed to cross-sheet or cross-file retrieval, and then culminate in calculations, modeling, or reporting. The distribution in Figure 5 shows that most workflows weave multiple tasks.

Importantly, each “task” itself typically requires substantial multi-turn reasoning: for example, web search often entails many rounds of LLM calls to discover, filter, and verify evidence; cross-sheet retrieval requires iterative calls to read and locate key information across multiple sheets; and calculation usually spans many formulas distributed over different rows and columns.

To further characterize the practical complexity of FINCH workflows, we conduct a case study on tool-call overhead using Claude Coworker (Opus 4.6) on 20 representative tasks (Appendix D). Excluding two extremely tool-intensive web-search workflows that require 71 and 107 tool calls respectively, the remaining tasks range from 6 to 25 tool calls (median 14). Notably, tool-call overhead does not scale linearly with the number of tasks; intrinsic task complexity—such as the depth of business-logic reasoning and cross-sheet dependencies—is a stronger driver. These findings suggest that even single- or two-task workflows can demand substantial multi-step agent interaction, reinforcing that FINCH captures meaningful operational complexity beyond what task counts alone convey.

###### 2.2.2 Messiness

The source files in FINCH are intentionally large, multi-sheet, and structurally complex. At the file level, 86.6% of workflows involve more than one file when counting both input and reference artifacts, and a workflow touches up to 14 distinct files. At the spreadsheet level, 92.4% of workflows involve multiple input and reference sheets, with an average of 8 sheets and a long tail reaching 91 sheets. As a result, systems must navigate cross-sheet dependencies, hidden logic, and scattered intermediate calculations rather than operating on a single “analysis” sheet. Moreover, most spreadsheets exhibit complex layouts that interleave text, numerical values, formulas, and charts, as well as intricate single- or multi- table structures with nested headers, hierarchical data, merged cells, blank rows and columns, and other irregularities.

Cell-level statistics further highlight the scale of the data. The median workflow covers 15K cells (157K on average for all workflows), with the largest one scaling to 3.7 million cells. Formula density is similarly skewed: while workflows contain an average of 21.5K formulas (the median is 212), reflecting deeply nested calculations and long dependency chains. Taken together, these properties create a challenging regime in which models must reason over large, noisy, and highly irregular spreadsheet layouts, rather than clean, rectangular tables.

[Figure 27]

[Figure 28]

Figure 6: Distribution of the number of sheets and cells per workflow.

###### 2.2.3 Multimodality

Although FINCH is spreadsheet-centric, the workflows are inherently multimodal. Around 10.5% of workflows link spreadsheets with additional non-spreadsheet artifacts such as PDFs, Word documents, and images, and 7.6% explicitly require reasoning over PDFs or images. Within the spreadsheets themselves, 20.3% of workflows include charts and 2.3% feature pivot tables, so models must understand not only raw cell values but also derived visual summaries and explicit aggregation structures (and most workflows involve implicit aggregation structures). This multimodal, crossartifact structure stands in contrast to prior benchmarks that operate purely on isolated tables, and better reflects the environments in which enterprise finance and accounting tasks actually occur.

- 2.3 Evaluation Method

- 2.3.1 Human Evaluation

We conduct human evaluation on all workflows to directly assess model performance. For each workflow, annotators read the NL instruction, inspect the input, reference, and model output files side by side (typically by aligning spreadsheets or documents in adjacent tabs), and determine whether the model has faithfully completed the requested job. A workflow is marked as successful only if the model generates or revises the content and structure in accordance with the instruction and no critical errors, omissions, or unintended changes are introduced; otherwise, it is labeled as a failure. Importantly, evaluation is based on whether the instruction has been satisfactorily fulfilled rather than on a purely mechanical comparison between model and reference outputs, since there may be multiple acceptable solutions for summarization, visualization, formatting, formulas, and related aspects. To reduce subjectivity and ambiguity, annotators ultimately assign a binary pass/fail label for each workflow. These human judgments serve as the gold standard for measuring model performance and for validating the reliability of our automatic evaluation method.

###### Eval Input (File Modification)

###### WorkflowWorkflow

Eval Rubrics

Completeness

Instruction & Input file snapshots

Instruction Input files Reference outputs

Correctness

Input→Reference outputs (diff_ref)

Over-edit avoidance

Input→Model outputs (diff_model)

Model outputs Screenshots Readability

[Figure 29]

[Figure 30]

Eval Input (File Generation & QA)

Eval Output

Instruction

[Figure 31]

Pass

[Figure 32]

Reference outputs & Model outputs Screenshots

### …

Fail

- Figure 7: Illustration of our automated evaluation pipeline. Here, diff_ref denotes the diff between the input file and the reference output, and diff_model denotes the diff between the input and the model output. We categorize all workflows into file modification, file generation, and file QA. This categorization is independent of the task types in Section 2.2; for example, a calculation task may generate a new file, modify an existing one, or simply return a textual answer.

###### 2.3.2 LLM-as-Judge Evaluation

To scale evaluation, we employ an LLM-as-judge framework that supports the three high-level task types in FINCH: modify (editing input artifacts), generate (creating new files such as workbooks and documents), and QA (answering questions based on one or more artifacts). The framework accepts heterogeneous inputs—including .xlsx, .txt, .docx, .md, .pdf, and images—and normalizes

- them into a sequence of textual inputs and screenshot images for the judge model.

For modification tasks, especially on spreadsheets, the framework computes structured diffs between the input and the reference output (diff_ref) and between the input and the model output (diff_model), and then builds a compact “input snapshot” (snapshot) that, for each modified sheet, retains only the first and last ten rows and the first five columns (which typically capture table headers and layout) together with rows and columns that contain edited cells. This preserves the crucial context for diff_ref and diff_model while dramatically reducing token length. In parallel, the framework renders screenshots (screenshot) of sheets containing changes from the input, reference, and model output, so that the judge perceives merged cells, conditional formatting, charts, and other layout-sensitive properties that are difficult to encode as text alone.

For generation tasks involving spreadsheets, the framework extracts all cell values and formulas from both the reference and the model output, and captures screenshots of every sheet, since the entire generated artifact must be verified rather than just localized edits. For QA tasks, it feeds the reference answer and the model’s response, optionally augmented with relevant input artifacts when the question requires grounding in input artifacts.

We design three task-specific judge prompts for modify, generate, and QA, respectively, but they share a common evaluation rubric. In all cases, the judge is instructed to focus on (i) completeness with respect to the NL instruction, (ii) numerical and logical correctness of derived values and formulas, (iii) the over-edit avoidance, penalizing unnecessary and unexpected changes of the workbook beyond instruction, and (iv) readability of the formatting and structure. Exact cell-by-cell equality with the reference is not required when multiple solutions are acceptable (e.g., alternative layouts, equivalent formulas, or different but semantically equivalent summaries); instead, the judge decides whether the model has satisfactorily fulfilled the instruction. To reduce subjectivity, the judge outputs a binary score (pass/fail) along with a short NL rationale. In some web search tasks, the rubric permits small tolerance bands when comparing values, allowing for discrepancies in data from different sources.

This LLM-as-judge framework not only automates large-scale evaluation but also surfaces subtle spreadsheet errors (such as formulas silently replaced with static values) that are difficult to catch with GUI-based human inspection alone. In Section 3.2, we report the consistency between human and automated evaluations and show that the LLM-as-judge scores closely track human judgments.

Looking forward, agentic evaluation is quite promising for future work for flexible workflow evaluation.

#### 3 Experiments

- 3.1 Agents and Models

- 3.1.1 Product-side Agents

We evaluate two frontier product-side agents: (i) ChatGPT using the GPT 5.1 model in Pro mode, and (ii) Claude using the Sonnet/Opus 4.5 models in thinking mode. We focus on these two systems rather than alternatives such as Gemini or Grok because they natively support returning downloadable files (e.g., spreadsheets) as outputs, rather than emitting code or markdown-formatted tables that are not intuitive for human evaluation. For both agents, we enable their external web browsing, but disable using historical chats so that each workflow is evaluated independently and without cross-run leakage. Since model updates are frequent and manual evaluation is very time-consuming, we used the latest model from our final round of experiments and did not consider subsequent updates.

- 3.1.2 API-based Models

We evaluate five frontier LLMs via API interfaces (Table 1). We adopt SpreadsheetBench [37] as the baseline framework because it provides a principled code-generation paradigm for spreadsheetcentric tasks, treating executable code as the model’s action space and enabling direct manipulation of spreadsheets through standard libraries. This design naturally aligns with FINCH, where workflows require complex spreadsheet operations, formula reasoning, and cross-sheet dependencies that cannot be reliably handled by text-only outputs.

While SpreadsheetBench was originally designed for relatively small and clean spreadsheets, we extend it with richer spreadsheet encodings, multimodal input support, and stricter execution and evaluation protocols, allowing it to scale to the large, messy, and long-horizon enterprise workflows in FINCH.

Model Provider Context Max Output Vision Native PDF

GPT 5.1 [40] OpenAI 400K 128K ✓ ✓ Claude Sonnet 4.5 [3] Anthropic 1M† 64K ✓ ✓ Grok 4 [51] xAI 256K 256K ✓ Qwen 3 Max [55] Alibaba 256K 32.8K — Gemini 3 Pro Preview [22] Google 1.05M 65.5K ✓ ✓

Table 1: API-based model configurations. Context and output limits are measured in tokens. Vision indicates native image input support, while Native PDF refers to direct PDF file ingestion via the provider’s API without explicit text extraction. †Available via long-context beta API mode.

Spreadsheet Encoding. SpreadsheetBench produces text tables without preserving cell addresses, data types, or formulas. However, these details are essential for tasks in FINCH. We extend SpreadsheetLLM [17] encoding and introduce a semantic-rich tuple encoding that preserves full structural and semantic fidelity. Each sheet begins with its name and the corresponding data bounding box (e.g. ## Sheet: [name] (A1:Z100)). We then serialize the bounded region using a Markdown-based format. Each cell is encoded as a tuple (Address, Value, Type, Formula), where Address denotes the cell reference (e.g. A3), Type indicates the data type (T = Text, I = Integer, F = Float, D = Date, B = Boolean), and Formula records the cell formula (e.g., =SUM(A1:A10)->100).

Multimodal Input Handling. We extend the framework to support multimodal inputs involving images and PDFs. For vision-capable models (GPT 5.1, Claude Sonnet 4.5, Grok 4, and Gemini 3 Pro), we use each provider’s official multimodal API to transmit visual inputs alongside text prompts. For PDF documents, we adopt a tiered strategy. Models with native PDF support—GPT 5.1, Claude Sonnet 4.5, and Gemini 3 Pro Preview—directly ingest PDF files via their file upload interfaces, enabling analysis of both textual and visual elements without pre-extraction. For Grok 4, which lacks native PDF support, we extract text using PyMuPDF and include it in the pdf_content field. For

Qwen 3 Max, which lacks multimodal support entirely, both image and PDF content are converted to textual descriptions. While this fallback retains semantic cues, it loses layout and visual context.

Context Management. To handle large spreadsheets that may exceed model context limits, we implement automatic truncation. We reserve 32K tokens for model output—sufficient for comprehensive code generation and analysis while remaining within the output limits of all evaluated models. Truncation is triggered when input exceeds the remaining capacity, removing content from the end of spreadsheet data with an explicit notice appended to inform the model of data loss.

###### 3.2 Experimental Results

Product-side agents (ChatGPT 5.1 Pro vs. Claude Opus/Sonnet 4.5). As shown in Figure 2 and Table 5, ChatGPT 5.1 Pro and Claude Opus 4.5 achieve the strongest overall pass rates on FINCH. Their advantage largely comes from rich interactive affordances: they can iteratively inspect spreadsheets, revise intermediate states, and recover from partial errors over many tool calls. However, they solve fewer than 50% of the FINCH workflows, suggesting that real-world finance and accounting work remains far from “solved” even for frontier agents. The detailed analysis in Figure 8 further highlights that long-horizon composition is a key bottleneck: when a workflow contains more than two tasks, the pass rate drops sharply—GPT 5.1 Pro decreases from 44.3% (workflows with ≤ 2 tasks) to 23.5% (workflows with > 2 tasks), and Claude Sonnet 4.5 decreases from 30.3% to 11.8%. This indicates that error accumulation across steps and missing intermediate affordances disproportionately hurt multi-step execution.

Pass rate also varies substantially by task type (Figure 8). Data Entry / Import and Structuring / Formatting are consistently among the most challenging categories, which aligns with FINCH spreadsheets exhibiting messy layouts, irregular tables, and nontrivial structural constraints. Moreover, Data Entry / Import workflows are frequently entangled with web search or PDF parsing, introducing multimodal dependencies that amplify failure modes. Notably, Translation—a task where modern LLMs typically excel in standard NLP settings—performs surprisingly poorly in FINCH. In financeheavy tables, translation can easily distort or drop critical structure and layout cues (e.g., header hierarchies, row/column alignment), and large grids make omissions more likely, leading to systematic failures. Detailed error analysis can be found in Section 3.3.

# Tasks per workflow # Workflows Pass Rate (%) Avg. time (min)

- 1 37 48.6 13.1
- 2 84 42.4 17.4
- 3 33 33.3 18.7

≥4 18 5.6 17.4

- Table 2: Average GPT 5.1 Pro completion time across workflows with different numbers of tasks.

We further analyze how GPT 5.1 web completion time scales with the number of tasks in a workflow (Table 2). The longest individual workflow run takes roughly 60 minutes to complete—but fails, highlighting how challenging workflows can be for current agents. On average, single-task workflows take 13.1 minutes, while workflows with two and three tasks require 17.4 and 18.7 minutes, respectively, reflecting the increased compositional complexity of multi-task settings. Interestingly, workflows with 4 or more tasks show a lower average time (17.4 minutes) than workflows with 3 tasks, most of which involve web search and almost all of which result in failure.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- Figure 8: Pass rate comparison for GPT 5.1 Pro and Claude Sonnet 4.5 across different task combinations and task types. The left chart visualizes the aggregated pass rate based on task combinations, revealing the models’ capabilities in handling multi-step workflows commonly seen in professional finance and accounting tasks. The k in the right chart represents the number of tasks included in a workflow. For example, a "k=3" workflow involves three distinct tasks, and its pass rate is calculated based on the collective performance of those tasks. The right chart shows the pass rates for individual tasks performed by both models in the FINCH benchmark. For workflows that contain multiple tasks, a task is counted as correct only if the entire workflow is completed successfully. If a workflow fails, all tasks within that workflow are counted as incorrect.

In addition, we examine performance on a subset of 20 workflows that involve non-spreadsheet artifacts (e.g., PDFs, Word documents, and images). As shown in Table 3, GPT 5.1 Pro achieves a 35.0% pass rate on these workflows (7/20), lower than its overall rate of 38.4%, while Claude Sonnet 4.5 achieves 25.0% (5/20), matching its overall rate. This suggests that handling heterogeneous document types introduces additional challenges.

Pass Rate Overall Other file types

GPT-5.1 Pro (Product) 38.4% 35.0% Claude Sonnet 4.5 (Product) 25.0% 25.0%

- Table 3: Pass rates on workflows involving non-spreadsheet artifacts (e.g., PDFs, Word documents, and images) under human evaluation.

Beyond binary pass/fail, our LLM-as-judge framework provides rubric-level scores along four dimensions: completeness, correctness, over-edit avoidance, and readability. Table 4 reports these scores for the two Claude product-side agents. Correctness is the primary bottleneck: Sonnet 4.5 achieves 55.8% completeness but only 36.0% correctness. These scores are produced by GPT-5mini using structured output, where a single call assigns all four scores jointly; this may introduce inter-rubric confusion. More robust agentic evaluation methods are a worthwhile direction for future work.

Rubric (%) Sonnet 4.5 Opus 4.5

Completeness 55.8 69.8 Correctness 36.0 45.3 Over-edit Avoidance 55.8 64.0 Readability 65.1 74.4

Table 4: Rubric-level scores under automated evaluation.

API-based. Figure 2 further shows that the adapted API-based baselines are generally weaker than official product-side agents. Under automated evaluation, for example, GPT 5.1 Pro achieves a pass rate of 41.9%, whereas GPT 5.1 with our API-based agent design reaches 32.0%. A key limitation of the API-based baselines is that they rely on a single LLM call, which precludes iterative interaction, execution feedback, and self-correction—an important direction for future work in designing F&A

enterprise-grade agent frameworks. Despite this constraint, our agent design narrows the performance gap by employing more efficient spreadsheet encodings and task-appropriate tool outputs within the single-call budget.

###### 3.2.1 Consistency Between Human and Automated Evaluation

We adopt a lightweight multimodal model, GPT-5-mini, as the judge for automated evaluation framework. As shown in Table 5, the automated evaluation largely aligns with human judgments: for GPT 5.1 Pro and Claude Sonnet 4.5, the judge agrees with human labels on 82.1% and 90.2% of workflows, respectively. The judge also achieves high recall (83.3% and 88.4%), meaning it recovers most human-labeled passes, and reasonably strong precision (73.6% and 76.0%), indicating that the majority of automatically predicted passes are also accepted by human evaluators. Overall, this suggests that automated evaluation may overestimate accuracy by several percentage points.

- Table 5: Comparison of human and automated evaluation on GPT and Claude product-side agents. “Automated Eval” shows pass counts/rates under the LLM-as-judge framework. “Agreement w/ Human Eval” reports how well the automated judgments match human labels: accuracy (Acc), recall (human-pass recall), and precision (human-pass precision).

Automated Eval Human Eval Agreement w/ Human Eval Pass Pass Rate (%) Pass Pass Rate (%) Acc (%) Recall (%) Precision (%)

Model (Product)

GPT 5.1 Pro 72/172 41.9 66/172 38.4 82.1 83.3 73.6 Claude Sonnet 4.5 50/172 29.1 43/172 25.0 90.2 88.4 76.0

On the model side, the LLM judge can occasionally miss nuances in the rubric—either failing to catch subtle visual or numerical errors in large spreadsheets or, conversely, being overly literal about certain instructions (e.g., penalizing benign formula-to-constant conversions). However, we also observe cases where the LLM-based judge is correct but human raters are wrong—for example, when formulas are silently replaced with static values, which are difficult to detect through GUI-based inspection alone. On the system side, limitations of our spreadsheet tooling and data pipeline (e.g., incomplete support for corrupted but human-readable workbooks or uncommon file formats) can cause valid outputs to be marked as failures. Taken together, these factors mean that our automated scores should be interpreted as approximate rather than exact, and that human review remains important for borderline or high-impact workflows.

###### 3.3 Error Analysis

To understand the sources of failure on FINCH, we conducted a qualitative error analysis of GPT 5.1 Pro and Claude Sonnet 4.5 in both their product-agent and API configurations. For all failed workflows in our evaluation, we manually inspected the trajectories and annotated the primary cause of failure.

From a workflow-centric perspective, we identify five dominant categories of error.2 Take Claude Sonnet 4.5 product-agent as an example. Across all examined failures, 10% stem from task misunderstanding: enterprise tasks often rely on implicit context in enterprise artifacts (e.g., spreadsheets), which models frequently overlook, leading them to misinterpret what is being asked and the required deliverable. 25% are data retrieval errors, including selecting the wrong cross-sheet, cross-table, or intra-table row/column ranges. 35% arise from formula reasoning errors, such as failing to reconstruct the latent business logic encoded in formulas or deriving incorrect new formulas. 25% are due to code generation errors, where generated scripts (e.g., Python with spreadsheet APIs) are syntactically invalid or misaligned with the spreadsheet layout. The remaining 5% correspond to data rendering errors, including incorrect formatting, misconfigured charts, or flawed final reports that deviate from the requested layout or narrative—for example, creating a brand-new spreadsheet instead of modifying the original one as requested. We also compare error patterns between web-based agents and API-based setups, with details provided in Appendix C.

Notably, all of these error types correspond to generic capabilities that modern LLMs already appear to master on many existing benchmarks. The question, then, is why these ostensibly strong

2For this study, once we identify the first clear error in a failed workflow, we stop further analysis for that workflow.

base abilities degrade so sharply on FINCH. Our analysis points to five intertwined properties of real-world enterprise Finance & Accounting workflows that make failures more likely and more catastrophic. First, FINCH workflows routinely involve large, fragmented spreadsheet ecosystems: dozens of interlinked workbooks and thousands of rows distributed across many sheets. Executing these workflows accurately requires long-range cross-sheet navigation and precise referencing, which substantially increases the likelihood of small retrieval errors. Second, the content is dense and semantically homogeneous: many cells contain domain-specific financial concepts that are subtly different yet lexically similar (e.g., variants of revenue/expense items, adjusted vs. unadjusted metrics), making entity disambiguation and cell grounding unusually difficult. Third, the table layouts and structures are complex and often irregular, including multi-level headers, merged cells, nested subtotals, and bespoke layouts that force the model to infer structure from noisy contents and ad hoc formatting. For example, at the code level, even tiny misinterpretations of these layouts (e.g., off-by-one errors when specifying ranges) can then propagate into globally incorrect outputs, especially when logic is applied in batch across many such sheets. Fourth, formulas encode latent structure and logic. In the FINCH dataset, each sheet contains a large number of formulas that encode latent business logic, temporal assumptions, and fine-grained dependencies that are not visible from displayed values alone; yet models typically prioritize cell values and under-use formulas, leading to systematic misinterpretations. For example, in a pricing sheet with the column header IF NGPL MidContinent index (@ Baker), the apparent semantics from the header alone suggest a daily exposure metric. However, inspecting the associated formula (25 * V21 + C41 * C22) reveals that the column in fact encodes a 55-day payment timing. Models that ignore or under-utilize formulas systematically misinterpret such columns’ roles in downstream calculations, and this misinterpretation

- then propagates through subsequent steps. Finally, many workflows involve multimodal artifacts and chat-centric tasks such as combining spreadsheets with PDFs, charts, and screenshots requiring the agent to jointly reason over heterogeneous formats. For example, tables embedded in PDFs are often only partially referenced, with key entries missing or truncated.

Many of these factors have been examined in prior work (e.g., multi-spreadsheet settings, complex table structures, formula reasoning, and multi-step workflows), and state-of-the-art models can perform reasonably well on benchmarks that emphasize a limited subset of these factors. In FINCH, however, these factors co-occur within the same workflow in real-world enterprise data, and our results suggest this composition is what drives the sharp performance drop. FINCH does not demand fundamentally new abilities; rather, it probes these abilities under an enterprise “extreme” regime of high complexity, noise, and long-horizon dependencies—closely mirroring real Finance & Accounting work. Progress on compositional capability, therefore, requires training and evaluation on long-running, computation- and reasoning-intensive workflows over large, messy multimodal enterprise artifacts.

#### 4 Related Work

The integration of LLMs into enterprise productivity tools has accelerated dramatically recently. The recently launched ChatGPT Agent [41] extends these capabilities to autonomous task completion, enabling multi-step workflows across web browsing, code execution, and spreadsheet manipulation. Microsoft Copilot [39] embeds AI capabilities across the Microsoft 365 suite, enabling users to draft documents, analyze spreadsheets, and automate workflows through natural language interaction. Similarly, Google has integrated Gemini [21] into Google Workspace, providing AI-assisted features in Docs, Sheets, and Gmail. Anthropic’s Claude Excel has also entered the enterprise space with spreadsheet automation capabilities [2].

The emergence of agentic AI systems marks a significant shift from understanding and QA to autonomous task completion [30, 7, 37]. However, there are long-standing challenges such as messy inputs and multimodal processing. SpreadsheetLLM [17] introduces novel encoding and compression methods to help LLMs understand large and messy spreadsheet structures, and further addresses this challenge through spreadsheet post-training. Beyond structural understanding, multimodal processing remains challenging for spreadsheet AI systems. On the formatting front, early work explored neural approaches for table formatting [15]. Recent advances in formula generation have progressed from pretraining with numerical reasoning [9] to natural language-driven formula synthesis [57], contrastive learning-based recommendation [6], and interactive formula prediction through hierarchical expansion [23].

Recent years have seen significant progress in benchmarks for financial reasoning [8, 27, 5, 53, 35, 10, 46, 29, 36, 43, 25, 56, 54], spreadsheet reasoning [30, 7, 37, 26, 33, 13, 59, 17, 50, 34, 16, 12, 31, 57, 45, 47], and multimodal document [38], table [32, 58], chart [48], table-chart [14], and spreadsheet [52, 13] reasoning, driving advances in LLM-based agents for enterprise tasks. However, FINCH targets messy artifacts and long-horizon workflows in real-world finance and accounting settings.

#### 5 Conclusion

We introduced FINCH, a new benchmark for real-world F&A enterprise workflows. FINCH combines workflows induced from enterprise email threads, version histories of spreadsheets, and high-quality financial artifacts with rigorous expert annotation and a calibrated LLM-as-judge framework, enabling systematic evaluation of agents on diverse workflows that operate over large, messy, and multimodal enterprise artifacts and require long-horizon, spreadsheet-centric reasoning. Our experiments show that even the strongest frontier systems pass fewer than 50% of workflows after spending 16.8 minutes per workflow, revealing a substantial gap between current AI capabilities and the demands of real enterprise practice. We hope FINCH will serve as a foundation for developing agents to tackle real, messy and long-horizon professional work.

##### 6 Acknowledgments We thank Chi Zhang and Jieqiong Li for reviewing the dataset and providing valuable advice.

#### References

- [1] Anthropic. Claude cowork. https://www.anthropic.com/product/claude-cowork, 2025.
- [2] Anthropic. Claude for excel. https://claude.com/claude-for-excel, 2025.
- [3] Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/claude-s onnet-4-5, 2025. Accessed: 2025-12-14.
- [4] World Bank. International Debt Report 2024. World Bank, Washington, DC, 2024. World Bank’s annual publication on external debt statistics.
- [5] Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance agent benchmark: Benchmarking llms on real-world financial research tasks. arXiv preprint arXiv:2508.00828, 2025.
- [6] Sibei Chen, Yeye He, Weiwei Cui, Ju Fan, Song Ge, Haidong Zhang, Dongmei Zhang, and Surajit Chaudhuri. Auto-formula: Recommend formulas in spreadsheets using contrastive learning for table representations. Proceedings of the ACM on Management of Data, 2(3):1–27, 2024.
- [7] Yibin Chen, Yifu Yuan, Zeyu Zhang, Yan Zheng, Jinyi Liu, Fei Ni, Jianye Hao, Hangyu Mao, and Fuzheng Zhang. Sheetagent: towards a generalist agent for spreadsheet reasoning and manipulation via large language models. In Proceedings of the ACM on Web Conference 2025, pages 158–177, 2025.
- [8] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. Finqa: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711, 2021.
- [9] Zhoujun Cheng, Haoyu Dong, Ran Jia, Pengfei Wu, Shi Han, Fan Cheng, and Dongmei Zhang. Fortap: Using formulas for numerical-reasoning-aware table pretraining. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1150–1166, 2022.
- [10] Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, et al. Finagentbench: A benchmark dataset for agentic retrieval in financial question answering. In Proceedings of the 6th ACM International Conference on AI in Finance, pages 632–637, 2025.
- [11] Department of Finance Canada. Fiscal reference tables, november 2025. Technical report, Government of Canada, Ottawa, Canada, 2025. Provides annual data on the financial position of the federal, provincial-territorial and local governments.
- [12] Haoyu Dong, Yue Hu, and Yanan Cao. Reasoning and retrieval for complex semi-structured tables via reinforced relational data transformation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1382– 1391, 2025.
- [13] Haoyu Dong, Shijie Liu, Shi Han, Zhouyu Fu, and Dongmei Zhang. Tablesense: Spreadsheet table detection with convolutional neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 69–76, 2019.
- [14] Haoyu Dong, Haochen Wang, Anda Zhou, and Yue Hu. Ttc-quali: A text-table-chart dataset for multimodal quantity alignment. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 181–189, 2024.
- [15] Haoyu Dong, Jinyu Wang, Zhouyu Fu, Shi Han, and Dongmei Zhang. Neural formatting for spreadsheet tables. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 305–314, 2020.

- [16] Haoyu Dong, Pengkun Zhang, Mingzhe Lu, Yanzhen Shen, and Guolin Ke. Machinelearninglm: Scaling many-shot in-context learning via continued pretraining. arXiv preprint arXiv:2509.06806, 2025.
- [17] Haoyu Dong, Jianbo Zhao, Yuzhang Tian, Junyu Xiong, Shiyu Xia, Mengyu Zhou, Yun Lin, José Cambronero, Yeye He, Shi Han, et al. Spreadsheetllm: encoding spreadsheets for large language models. arXiv preprint arXiv:2407.09025, 2024.
- [18] Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, and Thomas Wolf. Dabstep: Data agent benchmark for multi-step reasoning. arXiv preprint arXiv:2506.23719, 2025.
- [19] EnronData.org. Edo enron email pst dataset. https://enrondata.readthedocs.io/e n/latest/data/edo-enron-email-pst-dataset/. Creative Commons Attribution 3.0 United States License. To provide attribution, please cite to “EnronData.org.”.
- [20] Marc Fisher and Gregg Rothermel. The euses spreadsheet corpus: a shared resource for supporting experimentation with spreadsheet dependability mechanisms. In Proceedings of the first workshop on End-user software engineering, pages 1–5, 2005.
- [21] Google. Gemini for google workspace. https://workspace.google.com/solutions/ai/,

- 2024.

[22] Google DeepMind. Gemini 3 pro. https://deepmind.google/models/gemini/pro/,

- 2025. Accessed: 2025-12-14.

- [23] Wanrong He, Haoyu Dong, Yihuai Gao, Zhichao Fan, Xingzhuo Guo, Zhitao Hou, Xiao Lv, Ran Jia, Shi Han, and Dongmei Zhang. Hermes: Interactive spreadsheet formula prediction via hierarchical formulet expansion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8356–8372, 2023.
- [24] HM Treasury. Public expenditure statistical analyses 2023. Technical report, HM Treasury, London, United Kingdom, 2023. UK public expenditure statistical release (PESA).
- [25] Liang Hu, Jianpeng Jiao, Jiashuo Liu, Yanle Ren, Zhoufutu Wen, Kaiyuan Zhang, Xuanliang Zhang, Xiang Gao, Tianci He, Fei Hu, et al. Finsearchcomp: Towards a realistic, expert-level evaluation of financial search and reasoning. arXiv preprint arXiv:2509.13160, 2025.
- [26] Amila Indika and Igor Molybog. Sodbench: A large language model approach to documenting spreadsheet operations. arXiv preprint arXiv:2510.19864, 2025.
- [27] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944, 2023.
- [28] Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In European conference on machine learning, pages 217–226. Springer, 2004.
- [29] Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Kp Subbalakshmi, Jimin Huang, et al. Investorbench: A benchmark for financial decision-making tasks with llm-based agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2509–2525, 2025.
- [30] Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, and Zhao-Xiang Zhang. Sheetcopilot: Bringing software productivity to the next level through large language models. Advances in Neural Information Processing Systems, 36:4952–4984, 2023.
- [31] Jinyang Li, Nan Huo, Yan Gao, Jiayi Shi, Yingxiu Zhao, Ge Qu, Yurong Wu, Chenhao Ma, Jian-Guang Lou, and Reynold Cheng. Tapilot-crossing: Benchmarking and evolving llms towards interactive data analysis agents. arXiv preprint arXiv:2403.05307, 2024.
- [32] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li. Tablebank: Table benchmark for image-based table detection and recognition. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1918–1925, 2020.

- [33] Peng Li, Yeye He, Cong Yan, Yue Wang, and Surajit Chaudhuri. Auto-tables: Synthesizing multi-step transformations to relationalize tables without using examples. Proceedings of the VLDB Endowment, 16(11):3391–3403, 2023.
- [34] Zheng Li, Yang Du, Mao Zheng, and Mingyang Song. Mimotable: A multi-scale spreadsheet benchmark with meta operations for table reasoning. In Proceedings of the 31st International Conference on Computational Linguistics, pages 2548–2560, 2025.
- [35] Shu Liu, Shangqing Zhao, Chenghao Jia, Xinlin Zhuang, Zhaoguang Long, Jie Zhou, Aimin Zhou, Man Lan, and Yang Chong. Findabench: Benchmarking financial data analysis ability of large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 710–725, 2025.
- [36] Zhaowei Liu, Xin Guo, Haotian Xia, Lingfeng Zeng, Fangqi Lou, Jinyi Niu, Mengping Li, Qi Qi, Jiahuan Li, Wei Zhang, et al. Visfineval: A scenario-driven chinese multimodal benchmark for holistic financial understanding. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24099–24157, 2025.
- [37] Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang. Spreadsheetbench: Towards challenging real world spreadsheet manipulation. Advances in Neural Information Processing Systems, 37:94871–94908, 2024.
- [38] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [39] Microsoft. Microsoft 365 copilot. https://www.microsoft.com/en-us/microsoft-365 /copilot, 2024.
- [40] OpenAI. Gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. Accessed: 2025-12-14.
- [41] OpenAI. Introducing chatgpt agent. https://openai.com/index/introducing-chatgpt

-agent/, 2025.

- [42] OpenClaw. Openclaw: Personal ai assistant. https://github.com/openclaw/openclaw, 2025.
- [43] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. arXiv preprint arXiv:2510.04374, 2025.
- [44] Jean-Paul Sartre. Existentialism Is a Humanism. 1946.
- [45] Weixuan Wang, Dongge Han, Daniel Madrigal Diaz, Jin Xu, Victor Rühle, and Saravan Rajmohan. Odysseybench: Evaluating llm agents on long-horizon complex office application workflows. arXiv preprint arXiv:2508.09124, 2025.
- [46] Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jeff Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Jimin Huang, Guojun Xiong, et al. Finauditing: A financial taxonomy-structured multi-document benchmark for evaluating llms. arXiv preprint arXiv:2510.08886, 2025.
- [47] Zilong Wang, Yuedong Cui, Li Zhong, Zimin Zhang, Da Yin, Bill Yuchen Lin, and Jingbo Shang. Officebench: Benchmarking language agents across multiple applications for office automation. arXiv preprint arXiv:2407.19056, 2024.
- [48] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024.
- [49] Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, et al. Widesearch: Benchmarking agentic broad info-seeking. arXiv preprint arXiv:2508.07999, 2025.

- [50] Pengzuo Wu, Yuhang Yang, Guangcheng Zhu, Chao Ye, Hong Gu, Xu Lu, Ruixuan Xiao, Bowen Bao, Yijing He, Liangyu Zha, et al. Realhitbench: A comprehensive realistic hierarchical table benchmark for evaluating llm-based table analysis. arXiv preprint arXiv:2506.13405, 2025.
- [51] xAI. Grok 4. https://x.ai/news/grok-4, 2025.
- [52] Shiyu Xia, Junyu Xiong, Haoyu Dong, Jianbo Zhao, Yuzhang Tian, Mengyu Zhou, Yeye He, Shi Han, and Dongmei Zhang. Vision language models for spreadsheet understanding: Challenges and opportunities. In Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR), pages 116–128, 2024.
- [53] Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, et al. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37:95716–95743, 2024.
- [54] Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, et al. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37:95716–95743, 2024.
- [55] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [56] Zhihan Zhang, Yixin Cao, and Lizi Liao. Xfinbench: Benchmarking llms in complex financial problem solving and reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8715–8758, 2025.
- [57] Wei Zhao, Zhitao Hou, Siyuan Wu, Yan Gao, Haoyu Dong, Yao Wan, Hongyu Zhang, Yulei Sui, and Haidong Zhang. Nl2formula: Generating spreadsheet formulas from natural language queries. In Findings of the Association for Computational Linguistics: EACL 2024, pages 2377–2388, 2024.
- [58] Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, and Weiping Wang. Multimodal table understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9102–9124, 2024.
- [59] Ruiyan Zhu, Xi Cheng, Ke Liu, Brian Zhu, Daniel Jin, Neeraj Parihar, Zhoutian Xu, and Oliver Gao. Sheetmind: An end-to-end llm-powered multi-agent framework for spreadsheet automation. arXiv preprint arXiv:2506.12339, 2025.

#### Contents

- 1 Introduction 2
- 2 FINCH: A Real-world Finance & Accounting Workflow Benchmark 3

- 2.1 Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 2.1.1 Workflow from Enterprise Email Threads . . . . . . . . . . . . . . . . . . 4
- 2.1.2 Workflow Derivation from Versioned Spreadsheets . . . . . . . . . . . . . 4
- 2.1.3 Workflow Sourced from Final Deliverable Spreadsheets and Reports . . . . 5
- 2.1.4 Quality Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 2.2 Dataset Characteristics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 2.2.1 Task Compositionality . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2.2 Messiness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2.3 Multimodality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.3 Evaluation Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.3.1 Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3.2 LLM-as-Judge Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3 Experiments 9

- 3.1 Agents and Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3.1.1 Product-side Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.1.2 API-based Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3.2 Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 3.2.1 Consistency Between Human and Automated Evaluation . . . . . . . . . . 12
- 3.3 Error Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 4 Related Work 13
- 5 Conclusion 14
- 6 Acknowledgments 14

- A Author List 21
- B Experiment Details 21 B.1 API-based Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C Detailed Analysis 22
- D Tool-Call Overhead Case Study 22

- D.1 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.2 Results and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.3 Representative Tool-Call Logs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D.3.1 Task 26: Chart Generation across Scenario Sheets . . . . . . . . . . . . . . 23
- D.3.2 Task 139: XNPV Sensitivity Analysis with Corrected Trajectory . . . . . . 24

- E Ethics Statement 26
- F Examples 26

- F.1 Example 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- F.2 Example 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.3 Example 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.4 Example 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- F.5 Example 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- F.6 Example 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- F.7 Example 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- F.8 Example 8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- F.9 Example 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- F.10 Example 10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- F.11 Example 11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

#### A Author List

Author Affiliation Email

Haoyu Dong* UCAS; Microsoft donghaoyu82@gmail.com Pengkun Zhang South China University of Technology sezhangpengkun@mail.scut.edu.cn Yan Gao Zhongguancun Academy gaoyan@zgci.ac.cn Xuanyu Dong Harvest Fund qianmuxuanyu@126.com Yilin Cheng Fudan University; Zhongguancun Academy ylcheng23@m.fudan.edu.cn Mingzhe Lu UCAS lumingzhe23@mails.ucas.edu.cn Zikun Zhu Duke University zz295@duke.edu Adina Yakefu Hugging Face adina@huggingface.com Shuxin Zheng Zhongguancun Academy sz@bjzgca.edu.cn

*Corresponding author. UCAS stands for the University of Chinese Academy of Sciences

#### B Experiment Details B.1 API-based Models

Execution Paradigm. We frame the API evaluation as a code generation task. Models are instructed to solve spreadsheet manipulation and generation workflows by generating executable Python scripts, which are then executed in a sandboxed environment to produce output artifacts. This paradigm aligns with SpreadsheetBench’s philosophy of treating model-written code as the primary action space, but is adapted here to accommodate long-horizon, multimodal enterprise tasks.

- • Action Space: Models generate Python code using standard libraries including openpyxl (for Excel manipulation), pandas (for data processing), matplotlib (for visualization), and scikit-learn (for statistical analysis).
- • Output Format: Models must produce complete, self-contained Python scripts wrapped in markdown code blocks (“‘python ... “‘).
- • Sandboxed Execution: Generated code is extracted via regex parsing and executed in isolated Docker containers running Jupyter Kernel Gateway. Each container mounts the dataset volume at /mnt/data/ with a 10-minute session timeout.
- • Single-shot Protocol: We employ a strict one-shot generation protocol without iterative refinement—each model produces exactly one solution per workflow. If the generated code fails to execute (e.g., due to syntax errors or runtime exceptions), the workflow is marked as failed without retry. This strict setting is designed to evaluate the model’s raw code generation capability under realistic deployment constraints.

This unified code-as-action setting ensures that the measured performance reflects the model’s inherent competence on complex workflows rather than benefits derived from interactive debugging.

Prompting Strategy. We employ a zero-shot setting with a structured system prompt comprising:

- 1. A role definition: “You are an expert who can manipulate spreadsheets through Python code.”
- 2. A detailed description of the compact spreadsheet encoding format with illustrative examples.
- 3. The task instruction and explicit input/output file paths.
- 4. Library-specific best practices (e.g., openpyxl chart creation patterns) to mitigate common code errors.
- 5. An explicit directive to generate Python code as the final output.

This structured design explicitly guides models toward generating valid, context-aligned Python code, minimizing ambiguity in task interpretation. However, for models that support reasoning traces (GPT 5.1, Gemini 3 Pro), we request explicit reasoning via the include_reasoning API

parameter, enabling us to capture the model’s internal deliberation process for subsequent qualitative error analysis. Temperature is set to 0.7 across all models.

#### C Detailed Analysis

Web agents (ChatGPT 5.1 Pro vs. Claude Sonnet 4.5). ChatGPT 5.1 Pro tends to decompose workflows into more, smaller steps, with explicit reasoning, tool calls, execution, and self-checking at each step. This leads to longer traces and noticeably higher latency, but also more opportunities for intermediate validation (e.g., sanity-checking partial results). However, the code it generates is often hidden behind tool abstractions, so our error attribution is limited to observed behavior and natural language reasoning rather than the exact implementation details. Claude Sonnet 4.5 typically uses fewer steps and produces more direct solutions. In visualization-heavy workflows, its generated charts are often both more accurate and more aesthetically polished than those produced by ChatGPT 5.1 Pro, leading to relatively fewer failures in the data visualization sub-tasks.

ChatGPT 5.1 Pro and Claude Sonnet 4.5 agents can explore Excel files through many API calls within a single workflow, but their encoding methods are not well-suited to spreadsheets with complex layouts and structures. Thanks to efficient encoding and appropriate tool use, the following single-call API-based method achieves a pass rate that is much closer to that of product-side agents.

API-based Our API-based runs are single-call: they leverage the models’ underlying reasoning capabilities but lack two crucial affordances that web agents exploit. (i) interleaved code execution with feedback, and (ii) explicit reflection based on intermediate tool outputs. As a result, the API agents must generate the entire plan, code, and outputs within a single LLM call. When their initial structural assumptions about a spreadsheet are slightly off, they have no mechanism to detect or correct the mistake, leading to a significantly higher error rate, particularly in categories related to schema understanding and table manipulation. It’s desirable for future work to explore agentic methods with multiple rounds of API calls.

#### D Tool-Call Overhead Case Study

To better understand the practical complexity of completing FINCH tasks, we conduct a small-scale case study on tool-call overhead. From the full set of 172 workflows, we select 20 representative examples covering different task types and varying levels of complexity, and analyze how many tool calls are required for an LLM-based agent to complete them.

###### D.1 Methodology

For each selected task, we use the up-to-date (March 2026) Claude Coworker (Opus 4.6) in an isolated session and ask it to solve the task according to the provided instruction. The agent is also required to write its tool-use trajectory into a log file, recording each tool call step by step. Since the number of tool calls in a failed attempt may differ from that in a successful completion, we use a second session to compare the produced output against the reference answer file and determine whether the result is correct.

When the output is incorrect, we observe two distinct cases from the perspective of tool usage. In the first case, the overall tool-use procedure is already complete, but the execution within one or more steps is wrong; for example, the agent invokes a Python script as needed, but the script contains an incorrect formula. In the second case, the trajectory is missing one or more critical steps that are necessary for producing the correct output. This second case is relatively uncommon, appearing only twice among the 20 examples we analyze. For such cases, we ask the model to identify the missing indispensable steps and merge them into the original trajectory. Therefore, our final count is intended to approximate the tool calls required to correctly complete the task, rather than merely reporting the raw tool calls from a failed run.

###### D.2 Results and Analysis

- Table 6 summarizes the 20 selected tasks and their corresponding numbers of tool calls.

Workflow ID Task Type(s) #Tasks #Tool Calls

- 46 Calculation 1 7 31 Summary / Visualization 1 8 26 Summary / Visualization 1 10
- 47 Calculation 1 13 17 Cross-sheet/file Retrieval 1 14 137 Structuring / Formatting 1 16 91 Calculation, Financial Modeling 2 6 106 Structuring / Formatting, Calculation 2 6 49 Validation / Review, Calculation 2 14 16 Calculation, Summary / Visualization 2 17 28 Validation / Review, Structuring / Formatting 2 17

- 158 Validation / Review, Calculation 2 18 15 Structuring, Translation, Summary / Vis. 3 9 150 Calculation, Data Entry, Cross-sheet Retrieval 3 9 66 Calculation, Structuring, Financial Modeling 3 17 139 Financial Modeling, Calculation, Summary / Vis. 3 25
- 159 Structuring, Calc., Fin. Modeling, Validation 4 14 81 Data Entry, Structuring, Validation, Calc. 4 17 19 Web Search, Data Entry, Structuring, Calc. 4 107 88 Web Search, Data Entry, Structuring, Calc., Vis. 5 71

Table 6: Case study of tool-call overhead on 20 representative tasks.

Web-grounded tasks incur substantially higher tool-call overhead. Among the 20 cases, the two tasks involving web search (ID 19 and ID 88) require dramatically more tool calls than the others. In Task 19, websearch and webfetch together account for 60 out of 107 tool calls (56%). In Task 88, these two tools account for 34 out of 71 tool calls (48%). During inspection, we find that web-grounded tasks often require repeated retrieval from multiple external sources, followed by cross-source comparison, integration, and verification. As a result, the dominant cost in such tasks comes not from spreadsheet manipulation itself, but from external information acquisition and validation.

Distribution of tool calls for non-web tasks. After excluding the two web-search tasks, the number of tool calls ranges from 6 to 25, with a mean of 13.2 and a median of 14. The first and third quartiles are 9 and 17, respectively, giving an interquartile range of 8.

Task-type count alone does not explain tool-call overhead. After removing the two web-search outliers, we do not observe a clear positive relationship between the number of task-type labels and the number of tool calls. For example, both Task 46 and Task 47 are labeled as Calculation, but Task 47 requires nearly twice as many tool calls due to additional business-logic reasoning (sign conventions, Purchase vs. Pooling Method tax-base logic, and cross-checking of business consistency).

Limitations. This analysis should be viewed as a case study rather than a full-scale statistical evaluation. The 20 tasks are selected as representative examples rather than a random sample, and the measured tool-call counts depend on the behavior of a specific agent configuration (Claude Coworker, Opus 4.6). In addition, our corrected trajectories for failed runs involve a post hoc reconstruction of missing indispensable steps, which introduces some degree of judgment into the counting process.

###### D.3 Representative Tool-Call Logs

We present two representative tool-call logs. Task 26 illustrates a standard successful trajectory; Task 139 illustrates a case where the original trajectory omits critical steps (marked with +Bash ) that must be added for correct completion.

###### D.3.1 Task 26: Chart Generation across Scenario Sheets

Task. Add two charts to each of the three worksheets (Scenario1–3): (1) a Hedge Coverage stacked column chart (first year; X = months; stacked series = contract MW; Short = reference line;

Y = MW), and (2) a Portfolio Cost line chart (all years; X = months; series = Portfolio Weighted Avg and Monthly Price; Y = $/MWh).

###### Tool-call log (10 calls)

- 1. Bash ls mnt/26/—confirm source file exists.

- 2. Read metadata.json—parse task instructions and output requirements.

- 3. Read SKILL.md—review best practices for Excel operations.

- 4. Bash Python: inspect workbook structure via openpyxl (sheet names, first 20 rows, column mapping).

- 5. Bash Read full output (previous step was truncated).

- 6. Bash Python: inspect first 20 rows of each Scenario sheet in detail—column semantics, data types.

- 7. Bash Python: read date column (Z, rows 4–63) to identify first-year range (rows 4–15) and full range (Jan 2001–Dec 2005).

- 8. Bash cp 26_src_0.xlsx 26.xlsx—preserve original, work on copy.

- 9. Bash Run add_charts.py: create all 6 charts (2 per Scenario sheet) via openpyxl.

- 10. Bash Python: verify chart count, titles, types, and series per sheet.

###### D.3.2 Task 139: XNPV Sensitivity Analysis with Corrected Trajectory

Note. This is a corrected tool-call log. Steps marked +Bash were missing from the original trajectory and added post hoc; they mainly involve constructing monthly cash-flow data for XNPV Variables, building multi-year analysis data for Cedar Brakes, and writing native Excel XNPV formulas.

Tool-call log (25 calls)

Phase 1 — Understand the workbook

- 1. Bash ls mnt/139/—confirm source file.

- 2. Read metadata.json—task instructions, source file list, output requirements.

- 3. Read SKILL.md—Excel operation best practices.

- 4. Bash Python (data_only=True): inspect all sheet names, dimensions, and data values.

- 5. Bash Python: read Damage Calculations sheet—annual capacity payments (Q1–Q4), XNPV values, sensitivity-table region.

- 6. Bash Python: read Summary sheet—layout of Q1–Q4 sections and Delta region.

- 7. Bash Python (data_only=False): read formulas in Damage Calculations—calculation logic, DataTableFormula objects.

- 8. Read Read full output of step 7 (truncated).

- 9. Bash Python: read formulas in Summary sheet—formula reference relationships.

- 10. Bash Python: read formulas and variables in Tenaska’s i Adjustment sheet.

- 11. Bash Python: read variable values in Tenaska’s i Adjustment.

- 12. Bash Python: read formulas and variables in Brazos’ i Adjustment—compare with Tenaska.

- 13. Bash Python: validate XNPV calculation method—date conventions, discounting logic.

- 14. Bash Python: inspect DataTableFormula parameters—confirm input variable is F8 (plant capacity).

Phase 2 — Build missing intermediate data (added steps)

- 15. +Bash Python: read Cedar Brakes sheet—annual data layout (Year, Volume, Old PPA,

Market, SI, New PPA, Savings), column relationships, year range (2000–2016+), and rows/columns to fill.

- 16. +Bash Python: compute annual capacity payments for Q1–Q4 under four capacity assump-

tions (240/245/258/263 MW), applying the rate structure and interest-rate adjustments from Tenaska and Brazos.

- 17. +Bash Python: convert annual payments into monthly cash-flow sequences (date sequence from Jan 1997 to contract end; monthly = annual/12); verify totals.

Phase 3 — Execute main script

- 18. TodoWrite Create task list.

- 19. Bash Main Python script—copy source to 139.xlsx, then:

+XNPV Variables Write title, headers (Date, Q1–Q4), and monthly cash-flow rows starting at row 88 of Damage Calculations; scale payments proportionally for 240/245/258/263 MW.

+Cedar Brakes Fill multi-year financial data (rows 24+): Volume, Old/New PPA, Market, SI, Savings for all years 2000–end.

+XNPV formulas Write =XNPV(rate, flows, dates) formulas into the sensitivitytable region, referencing the cash-flow data above.

Original Fill Q1–Q4 capacity values and Delta section (differences vs. 263 MW baseline) in Summary; save.

- 20. TodoWrite Update progress.

- 21. Glob Locate recalc.py.

- 22. Bash python recalc.py—recalculate all Excel formulas (including XNPV).

Phase 4 — Verify outputs

- 23. +Bash Python (data_only): verify XNPV Variables integrity (row count, date continuity, cash-flow values), Cedar Brakes data, and XNPV formula results.

- 24. Bash Python: verify Summary (PV5, debt, SWAP, implied equity for Q1–Q4) and Damage Calculations sensitivity table (all XNPV values, Delta vs. 263 MW baseline).

- 25. TodoWrite Final progress update.

#### E Ethics Statement

The FINCH benchmark is constructed entirely from existing, publicly available data sources. Concretely, our workflows are derived from (1) the Enron email corpora, including the parsed Enron email dataset on Kaggle (released under the CC0 Public Domain dedication) and the Enron Email Dataset from EnronData.org (licensed under CC BY 3.0 US); (2) the EUSES spreadsheet corpus and its modified variants (CC BY 4.0); and (3) a diverse collection of enterprise-like artifacts, including documents from investment and securities companies, the World Bank (CC BY 3.0), Canadian and British government websites (Open Government License), and public corpora such as WideSearch (MIT license) and DABStep (CC BY 4.0). We respect the original licenses of all upstream resources and only redistribute content within the terms they allow.

On top of these sources, we apply additional filtering, normalization, and expert annotation to organize spreadsheets and related documents into coherent workflows with task instructions, input files, and reference outputs. We do not introduce any new personally identifiable information. During curation, we remove obviously sensitive fields when they are not necessary for the task (e.g., personal contact information or signatures) and avoid annotating workflows whose successful completion would depend on sensitive personal attributes rather than business logic. The resulting FINCH dataset is released under the Creative Commons Attribution 3.0 United States license (CC BY 3.0 US), which permits broad reuse while requiring appropriate attribution.

The language in FINCH is primarily English, reflecting the dominant language of the underlying Enron and EUSES corpora and many of the public institutional sources. Because some artifacts originate from funds and securities institutions and from Canadian government materials, a small fraction of workflows include Chinese or French content.

#### F Examples

This section presents representative examples from FINCH to illustrate the diversity and complexity of spreadsheet-centric enterprise workflows covered by the benchmark. The examples span a wide range of realistic professional tasks, including cross-sheet verification, formula auditing, documentgrounded extraction, schema transformation, financial modeling, and reporting.

Note that the figures shown in this section are illustrative excerpts rather than complete workflow inputs or full task specifications. Many FINCH workflows involve large, multi-sheet or multi-file inputs and extensive supporting documents, which cannot be fully visualized in static screenshots. Accordingly, the figures are intended to highlight representative data characteristics, structural complexity, and reasoning challenges that arise in the underlying workflows, rather than to fully specify the tasks themselves.

Across these workflows, AI agents are required to jointly reason over heterogeneous artifacts, while preserving structural consistency and semantic correctness. Many tasks demand multi-step reasoning, precise interpretation of formulas, and careful cross-referencing across sheets or external documents. Collectively, these examples highlight the core challenges targeted by FINCH: messy and multimodal document processing, long-horizon reasoning, robust code generation under schema variation, and faithful execution of complex financial and accounting logic.

###### F.1 Example 1

[Figure 39]

[Figure 40]

Validate

Cross-sheet retrieval

[Figure 41]

Example Sheets

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Sheet: Legal Sheet: Gas Trading

Sheet: Portland Trading & Origination Sheet: Canda

[Figure 46]

[Figure 47]

[Figure 48]

Sheet: Regulatory

Sheet: Market Risk & Research Sheet: Portland Fundies & Structuring

- Figure 9: For this task, the model must verify the department headcount summary by cross-checking each of the 39 departments against its detailed roster sheet. It should correct discrepancies such as miscounts and missing or duplicate entries. The summary must be updated by fixing incorrect totals, removing departments that no longer exist, and adding any omitted departments. Furthermore, the underlying schema varies slightly across departments, which challenges reliable code generation.

- F.2 Example 2
- Figure 10: An example of extracting data from tables and charts in PDFs and saving it to a spreadsheet. AI agents must understand the layout, parse hierarchical structures, interpret how values map to specific cells, and reconstruct formulas from aggregated values.

F.3 Example 3

[Figure 49]

- Figure 11: Cross-sheet reference validation. This example is relatively easy for frontier AI agents.

[Figure 50]

[Figure 51]

- F.4 Example 4
- Figure 12: This task requires deriving the XNPV5 of the contract under different combinations of assumptions. The analysis uses contract capacity rates, plant capacity, and the specified discount rate provided in the table. While key adjustment components—namely the Interest Rate Adjustment, Apache Savings, and O&M Adjustment—must be retrieved from supporting documents and applied according to each scenario (included, excluded). For each assumption set, the analyst must then construct the annual capacity payment cash flows by deriving the adjusted capacity rate, converting it into monthly and annual capacity payments, and assembling the full month-by-month cash-flow schedule. Only after these intermediate steps are completed can the cash flows be discounted to the valuation date (e.g., December 31, 2000) to compute XNPV5.

F.5 Example 5

[Figure 52]

- Figure 13: The sum of A&B and the equity roll-forward test require cross-sheet retrieval and calculation.

[Figure 53]

###### F.6 Example 6

[Figure 54]

[Figure 55]

Translate

- Figure 14: A workflow that translates a French report into English while preserving its format and structure. The report contains many tables to translate, along with text, notes, and even charts.

F.7 Example 7

Transform

| |
|---|

[Figure 56]

[Figure 57]

- Figure 15: Transforming a table from one structure to another requires reorganizing data and retrieving information across sheets (e.g., area_info and summary). This example poses additional challenges: (1) distinguishing value-driven operators from volume-driven operators, and (2) performing aggregation and validation over the reorganized data.

- F.8 Example 8
- Figure 16: The apparent semantics from the headers suggest a monthly/daily exposure metric. However, inspecting the underlying formula (e.g., C5=$A4*Volumes!B6*Curves!G7+25*Volumes!B7*Curves!G8) reveals that it actually encodes a 55-day payment timing schedule. Models that ignore or underutilize formula information, therefore systematically misattribute the column’s role in downstream computations, and this misinterpretation then propagates through subsequent steps.

F.9 Example 9

[Figure 58]

- Figure 17: This workflow requires creating a new spreadsheet with all values converted to USD. It also requires correct in-sheet and cross-sheet formula references while preserving the original spreadsheet layout.

[Figure 59]

- F.10 Example 10
- Figure 18: Generating reports from tabular data requires financial knowledge of data analysis, financial events, and visualization. For example, one may plot two series with different units on a single chart (e.g., using a secondary y-axis) to reveal their correlation.

F.11 Example 11

[Figure 60]

- Figure 19: This Excel sheet shows an assumption-update workflow, where a mix of forward contracts is used to cover monthly peak-load short positions. It lists the contract allocations and MW volumes, along with monthly peak loads and the resulting short MW. A table on the right computes blended prices and portfolio costs, and the stacked chart visualizes coverage by contract type over the year.

[Figure 61]

[Figure 62]

[Figure 63]

