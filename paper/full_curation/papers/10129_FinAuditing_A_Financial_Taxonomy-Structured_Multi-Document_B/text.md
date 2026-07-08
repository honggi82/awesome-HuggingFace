# arXiv:2510.08886v3[cs.CL]17May2026

## FINAUDITING: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs

Yan Wang

The Fin AI USA wy2266336@gmail.com

Keyi Wang

Columbia University USA

Shanshan Yang

Stevens Institute of Technology USA

Jaisal Patel

Rensselaer Polytechnic Institute USA

Jeff Zhao

UT Austin USA

Fengran Mo

University of Montreal Canada

Xueqing Peng* Lingfei Qian*

The Fin AI USA

Yankai Chen*

McGill University MBZUAI Canada yankaichan3@gmail.com

Víctor GutiérrezBasulto*

Jimin Huang

The University of Manchester Manchester, United Kingdom

Cardiff University UK

GutierrezBasultoV@cardiff.ac.uk

The Fin AI USA

Guojun Xiong

Harvard University USA

Xiao-Yang Liu

Columbia University USA

### Abstract

Going beyond simple text processing, financial auditing requires detecting semantic, structural, and numerical inconsistencies across large-scale disclosures. As financial reports are filed in XBRL, a structured XML format governed by accounting standards, auditing becomes a structured information extraction and reasoning problem involving concept alignment, taxonomy-defined relations, and cross-document consistency. Although large language models (LLMs) show promise on isolated financial tasks, their capability in professional-grade auditing remains unclear. We introduce FINAUDITING, a taxonomy-aligned, structure-aware benchmark built from real XBRL filings. It contains 1,102 annotated instances averaging over 33k tokens and defines three tasks: Financial Semantic Matching (FinSM), Financial Relationship Extraction (FinRE), and Financial Mathematical Reasoning (FinMR). Evaluations of 13 state-of-the-art LLMs reveal substantial gaps in concept retrieval, taxonomy-aware relation modeling, and consistent cross-document reasoning. These findings highlight the need for realistic, structureaware benchmarks. We release the evaluation code1 and dataset2 publicly, and the task currently serves as the official benchmark of an ongoing public evaluation contest3.

*Corresponding authors.

- 1https://github.com/The-FinAI/FinAuditing
- 2https://huggingface.co/collections/TheFinAI/finauditing
- 3https://open-finance-lab.github.io/SecureFinAI_Contest_2026/

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference’17, Washington, DC, USA © 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

Xue (Steve) Liu

Jian-Yun Nie

McGill University MBZUAI Canada

University of Montreal Canada

### CCS Concepts

• Applied computing → Extensible Markup Language (XML); • Information systems → Test collections; Information extraction; Question answering.

### Keywords

XBRL auditing, Benchmark, Large language model, Information retrieval, Information extraction, Question answering

ACM Reference Format:

Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jeff Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Yankai Chen, Víctor Gutiérrez-Basulto, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, Xue (Steve) Liu, and Jian-Yun Nie. 2026. FINAUDITING: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs. In . ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

[Figure 1]

Figure 1: An illustrative example of cross-document financial inconsistency.

### 1 Introduction

Consider a scenario where a global corporation reports “Total Cash” as $100M in its high-level summary, while detailed supporting disclosures across sub-documents sum to only $95M. Detecting such

Table 1: Comparison of existing financial NLP benchmarks and FINAUDITING. We compare support for structured retrieval (Struct.Retrieval), hierarchical information extraction (Hiera.IE), multi-step numerical reasoning (MultiStep.Reasoning), taxonomy-driven supervision (Need.Taxonomy), and crossdocument reasoning (At.Cross-Doc).

|Benchmark|Struct.Retr.<br><br>|Hiera.IE<br><br>|MultiStep.Reas.|Need.Taxon.|At.Cross-Doc<br><br>|
|---|---|---|---|---|---|
|FiNER [18] FNXL [23] FinTagging [28] FinQA [1] ConvFinQA [2] TAT-QA [33] MultiHiertt [31] DOCMATH-EVAL [32] FinAuditing (ours)<br><br>| | | | | |

inconsistencies is a non-trivial information verification problem. In modern financial reporting, this information spans thousands of pages of structured filings, making manual verification nearly impossible. While ensuring data accuracy is vital for market transparency, identifying these “hidden” inconsistencies remains challenging even for advanced AI systems, due to the massive scale and structural complexity involved.

Modern financial disclosures are governed by Generally Accepted Accounting Principles (GAAP) and operationalized via the eXtensible Business Reporting Language (XBRL) [4]. Unlike plain text, these filings are heterogeneous, hierarchical, and highly interdependent [4, 10]. Auditing these documents requires more than keyword search, as it demands sophisticated structured information retrieval (SIR) and reasoning [3, 15]. Models must semantically align concepts across disparate tables, interpret nested hierarchical metadata, and maintain logical consistency across multiple interdependent documents. When these capabilities fall short, the consequences are tangible: firms issue restatements, i.e., costly corrections of previous reports, which have risen persistently as disclosure complexity outpaces current analytical tools4.

Recent advances in large language models (LLMs) have sparked interest in applying them to financial tasks, such as information extraction [18, 23, 28], question answering [1, 2, 31–33], and numerical reasoning [25] and trading [13, 21]. However, it remains unclear whether these models can support professional-grade auditing, which requires consistent reasoning over hierarchical structures, taxonomy-defined relations, and cross-document dependencies. This raises a critical question: how should we evaluate models on structured, regulation-driven financial information tasks? As summarized in Table 1, existing financial NLP benchmarks exhibit three fundamental limitations: (1) Contextual Isolation: Most benchmarks operate on isolated text snippets or tables, failing to capture the cross-document dependencies that are central to real financial filings [31]. (2) Surface-Level Evaluation: Prior work predominantly evaluates local semantic or numerical correctness, without assessing multi-step structural reasoning over reporting hierarchies [32]. (3) Taxonomy Underutilization: Although official GAAP taxonomies define valid financial concepts and relations, few works evaluate whether models can retrieve and reason over these constraints [28].

- 4https://www.ft.com/content/716c4ad5-e8fa-4a34-afba-9fb2d1db019d

To address these gaps, we introduce FINAUDITING, a taxonomyaligned, structure-aware, multi-document benchmark derived from real-world US-GAAP-compliant XBRL filings. FINAUDITING reframes financial auditing as a structured information reasoning problem, and defines three complementary subtasks: Financial Semantic Matching (FinSM) for concept-level semantic alignment, Financial Relationship Extraction (FinRE) for structural and relational consistency, and Financial Mathematical Reasoning (FinMR) for taxonomy-grounded numerical verification. Together, these tasks shift evaluation from isolated text understanding to holistic reasoning over structured financial information, providing a high-fidelity testbed for financial capabilities under formal constraints. Our main contributions are threefold:

- • A Taxonomy-Grounded Financial Resource: We release the first benchmark built from real-world XBRL filings that explicitly integrates multi-document structures and official GAAP taxonomies for evaluation.
- • Structure-Aware Task Design: We propose three complementary subtasks (FinSM, FinRE, FinMR) that capture semantic, relational, and numerical reasoning challenges inherent to structured financial information.
- • Systematic Evaluation of Intelligent Systems: We provide an initial evaluation of 13 SOTA LLMs, while emphasizing that FINAUDITING is model-agnostic and designed to support the evaluation of complex architectures, including retrieval-augmented generation (RAG) pipelines and multi-agent auditing systems, for structured financial information retrieval and reasoning.

### 2 Related work

Prior financial NLP research has advanced information extraction [18, 23, 28], semantic understanding [20, 24, 29], and numerical reasoning [1, 2, 31–33]. However, most benchmarks focus on unstructured or semi-structured data from individual documents, such as financial reports or news articles, and overlook the structured, hierarchical, and interconnected nature of XBRL filings. Early XBRL-focused studies primarily address localized subtasks, including numerical entity extraction [18], sentence-level numeral labeling [23], tag normalization [28], and retrieval- or agent-based evaluation of LLMs on XBRL data [8, 12]. While informative, these efforts evaluate narrow components in isolation and do not capture the taxonomy-level dependencies that characterize real-world XBRL structures. Related benchmarks on tabular and textual reasoning, such as FinQA [1], ConvFinQA [2], TAT-QA [33], MultiHiertt [31], and DocMathEval [32], similarly lack taxonomy-driven supervision. In contrast, FINAUDITING is designed to evaluate LLMs on structured, hierarchical, and cross-document reasoning grounded in realistic XBRLs.

### 3 FinAuditing

Detecting errors in structured financial documents requires semantic, structural, and numerical reasoning. As shown in Figure 2, we introduce FINAUDITING, a benchmark for evaluating LLMs in identifying inconsistencies across multi-document XBRL filings.

### 3.1 Task Formulation

Given a target XBRL filing consisting of six interconnected, XML documents D = {𝑑1,𝑑2,𝑑3,𝑑4,𝑑5,𝑑6} (described in Table 3) and a

[Figure 2]

#### Figure 2: Overview of the FINAUDITING construction pipeline.

reference US-GAAP taxonomy T, the task is to identify errors E within the filing. These errors include semantic-matching inconsistencies, relationship-extraction failures, and mathematicalreasoning violations. Models must analyze the filing’s internal structure and semantics while leveraging domain knowledge from the reference taxonomy. We formulate this as follows:

𝑓 : (D, T) ↦→ E (1) where E is the specific description of the errors found in the D.

To operationalize error detection across XBRL filings, we identify three fundamental capabilities: semantic consistency, structural relationship understanding, and numerical reasoning. Accordingly, FINAUDITING defines three tasks: (1) Financial Semantic Matching (FinSM) identifies mismatches between textual mentions in filings and standardized taxonomy labels; (2) Financial Relationship Extraction (FinRE) detects errors in hierarchical and compositional relationships among financial items; and (3) Financial Mathematical Reasoning (FinMR) verifies numerical correctness based on accounting logic and contextual inference.

- 3.1.1 FinSM. The semantic matching task evaluates whether models can identify financial tags in XBRL filings that misalign with standardized US-GAAP taxonomy concepts. Given a filing with multiple candidate tag assignments, the objective is to detect semantically inconsistent or inappropriate concept assignments within a set of complex, structured documents.

FinSM Example

You are an auditor for XBRL filings. Given the question and the provided filing (schema, presentation, calculation, definition , label, instance, and US-GAAP taxonomy), identify erroneous US-GAAP concepts.

You must reference the US-GAAP taxonomy to retrieve and check elements across the entire filing. Output only a JSON array of strings, each string is one erroneous

concept. Output Example: ["us-gaap:Revenues", "us-gaap:OperatingIncomeLoss"] No explanations. No extra text. Only the JSON array.

This task follows an information retrieval paradigm. Given (1) a query 𝑄 describing a financial term related to currency or credit risk concentration, (2) an XBRL filing D, and (3) the US-GAAP taxonomy T. The goal is to retrieve the set of mismatched USGAAP tags within the filing, denoted as E. For the FinSM task, we evaluate retrieval performance using metrics: Hit Rate (𝐻𝑅@𝑘) [9], Recall (𝑅@𝑘), and Macro-F1 (𝑀𝐹1@𝑘), with 𝑘 ∈ {1, 5, 10, 20}.

- 3.1.2 FinRE. The relationship extraction task evaluates whether models can identify structural errors among financial elements by interpreting hierarchical and compositional dependencies between the filing and the external taxonomy. This task assesses the LLMs’ capacity for semantic understanding and structured data interpretation within the financial reporting context.

FinRE Example

You are an auditor for XBRL filings. Given two concepts and the provided filing (schema, presentation, calculation, definition , label, instance, and US-GAAP taxonomy), determine which one of the following erroneous relationship types best describes the relationship between the two concepts:

- (1) Reversal Definition: <definition>.
- (2) Inappropriateness Definition: <definition>.
- (3) CombinationErr Definition: <definition>. You must reference the US-GAAP taxonomy to retrieve and check the

two concepts across the entire filing.

Output only one of the following labels exactly, chosen strictly from:["Reversal", "Inappropriateness", "CombinationErr"]. No explanations. No extra text. Only the label.

Formally, given a relation error type space 𝐿 with predefined semantic definitions, an XBRL filing D, and the US-GAAP taxonomy T, the task aims to classify the specific relation error type ℓ ∈ 𝐿 between two specific elements 𝑒1 and 𝑒2, which are derived from the current XBRL filing D, based on the information provided by T.

We define three relation error types {ℓ1, ℓ2, ℓ3} in 𝐿: (1) Reversal, which occurs when the hierarchical relationship between parent and child elements is mistakenly reversed; (2) Inappropriateness, which indicates that a child element is incorrectly associated with an inappropriate parent element; and (3) CombinationErr, which refers to an invalid combination of axis and member elements that violates the structural constraints defined in the taxonomy. We use the metrics, including Accuracy (Acc), Macro Precision (P), Macro Recall (R), and Macro F1, to evaluate the performance of this task.

- 3.1.3 FinMR. The mathematical reasoning task focuses on inferring correct numerical values among financial elements based on structured representations, calculation hierarchies, and constraints defined in the XBRL filing. The objective is to assess whether reported values are internally consistent and compliant with domainspecific standards.

FinMR Example

You are an auditor for XBRL filings. Given the question and the provided filing (schema, presentation, calculation, definition , label, instance, and US-GAAP taxonomy), identify the reported value of a financial element and calculate the actual

value that should be reported based on calculation

relationships. Answer strictly in the following JSON format: {"extracted_value": "<numeric value reported in the instance

document, or 0 if not found, keep the same format as in the XBRL filing>",

"calculated_value": "<numeric value computed from calculation relationships, or 0 if not computable, use the same number formatting style as extracted_value>"}

No explanations. No extra text. Only the JSON object.

Specifically, the task involves an XBRL filing D, the US-GAAP taxonomy T, and a pair of questions (𝑞1, 𝑞2). Here, 𝑞1 requests the extraction of a reported value and 𝑞2 pertains to the computation of the corresponding true value. The model is required to: (1) extract the reported numeric value 𝑣 for a given instance from D, and (2) compute the corresponding real value 𝜇 based on the structural and numerical information in D and T. The extracted value 𝑣 and the computed value 𝜇 are then compared to determine whether the reported value is correct. The output of this task is a structured answer containing both 𝑣 and 𝜇, represented in a JSON format. For this task, we evaluate the overall Accuracy using an LLM-as-ajudge framework [7, 14], together with three complementary error indicators. The structural error rate (SER) measures cases where the model fails to produce a valid output structure in the required JSON format. The extraction error rate (EER) captures errors in identifying the correct numerical value to be extracted, even when the output structure is valid. The calculation error rate (CER) reflects mistakes in numerical computation, where the final calculated value is incorrect despite the correct structure and extraction.

### 3.2 Data Collection and Construction

FINAUDITING is established as a high-fidelity testbed to catalyze research on taxonomy-grounded reasoning and structured retrieval within complex, multi-document financial environments. As illustrated in Figure 2, our resource bridges the gap between raw XML data and the semantic constraints of the US-GAAP taxonomy.

- 3.2.1 Authoritative Error Signals from DQC Messages. To identify reliable auditing errors, we ground our benchmark construction in authoritative error signals released by the Data Quality Committee (DQC)5. Our construction follows a two-step pipeline:

(1) Pre-screening: We collected 4,545 official DQC error messages for 372 companies between 2020 and 2024 via the XBRL US portal. Each message links to a deterministic DQC US rule, providing unambiguous labels for semantic, relational, or numerical inconsistencies. (2) Filtering: We selected the 9 most frequent DQC error types, covering 60.33% of all cases (Table 2), and retrieved 218 corresponding XBRL filings from the SEC EDGAR database, the official repository for public company disclosures.

- 3.2.2 XBRL Filing Segmentation. Using “SEC-url” identifiers from DQC messages (Section 3.2.1), we retrieve the corresponding XBRL filings as the primary data representation for structured financial reporting. An XBRL filing comprises multiple interconnected

- 5https://xbrl.us/data-quality/filing-results/

#### Table 2: Distribution of common error types in XBRL filings.

###### Broad Category Error Type # Cases Proportion (%) DQC US ID

Semantic-based FS with no associated calculation 386 8.49 DQC_0099 Concentration risk 217 4.77 DQC_0109 Location axis with a single member 208 4.58 DQC_0137

Relation-based Sibling and child relationships 570 12.54 DQC_0081 Axis with inappropriate members 313 6.89 DQC_0001 Inappropriate cash flow presentation 219 4.82 DQC_0145

Calculation-based FS calculation check with no dimensional data 260 5.72 DQC_0126 Negative values 368 8.10 DQC_0015 FS tables dimensional cross check 202 4.44 DQC_0117

Total 2,743 60.33 -

#### Table 3: Main Components of XBRL Filings

|Component<br><br>|Description|
|---|---|
|Instance Document|Actual financial data structured according to a predefined taxonomy.<br><br>|
|Schema Document<br><br>|Defines the structure, classification, and data types of financial elements.|
|Calculation Linkbase|Specifies arithmetic relationships between elements to ensure logical consistency.|
|Presentation Linkbase<br><br>|Organizes financial data into a human-readable structure for reporting.|
|Definition Linkbase<br><br>|Captures semantic relationships between elements, such as component relations.|
|Label Linkbase|Provides human-readable names and descriptions, supporting multiple languages.|

XML documents that jointly encode both financial facts and their structural constraints. As summarized in Table 3, each filing contains: an instance document with reported values, a schema document defining element types, and several linkbases (calculation, presentation, definition, and label) that specify hierarchical organization, arithmetic relationships, and semantic metadata [4, 10].

Due to this rich but highly interdependent structure, complete XBRL filings are often extremely long and difficult to process directly, posing challenges for both manual inspection and longcontext LLM evaluation. To balance document length with structural fidelity, we perform statement-level filing segmentation. Specifically, we extract roleType identifiers from the schema document to identify distinct financial statements (e.g., balance sheets or income statements). We then leverage the presentation linkbase to map the hierarchical layout of each statement to its corresponding facts in the instance document. This segmentation preserves the original XML dependencies across the schema, linkbases, and instance documents, while producing manageable, structure-aware sub-filings suitable for downstream auditing tasks and long-context LLM processing.

3.2.3 US-GAAP Taxonomy Digitalization. To facilitate structured access to the US-GAAP Taxonomy, we digitalize it into conceptcentric chunks. We extract each concept and its associated semantic information from the official taxonomy files (e.g., GAAP Taxonomy 2024.xlsx), parsing all relevant sheets: Concepts, Presentation, Calculation, Definition, Reference, and Enumeration. We organized the extracted information into two major chunk types: core chunks contain each concept’s intrinsic attributes, such as label, type, balance, period type, and documentation. relation chunks encode structural and semantic links to other concepts, including hierarchical presentation paths, parent–child calculation relationships, definitional arcs, and external references. This digitalization process preserves the taxonomy’s hierarchical and relational structure while enabling fine-grained concept-level indexing, retrieval, and reasoning.

Table 4: Resource statistics across tasks (tokens via cl100k_base).

|Task<br><br>|Total Tokens Max Tokens Avg. Tokens Std. Tokens Instances|
|---|---|
|FinSM FinRE FinMR|11,170,024 62,777 33,848.56 11,402.49 330 14,819,627 65,030 33,680.97 10,847.23 440 11,845,255 59,641 35,678.48 9,852.73 332<br><br>|
|Total<br><br>|37,834,906 65,030 34,338.64 10,700.82 1,102|

### 3.3 Data Annotation and Quality Control

FINAUDITING provides 1,102 annotated instances, each averaging over 33k tokens (Table 4), capturing the long-context characteristics of practical financial auditing scenarios.

- 3.3.1 Annotation via LLM-Assisted Parsing. Ground-truth supervision is deterministically derived from DQC error messages. We employ GPT-4o-mini solely as a high-efficiency parser, rather than a decision-maker, to convert these messages into structured annotations. The model extracts (i) DQC-defined ground-truth labels, (ii) task-specific elements for formulating queries, relations, or questions, and (iii) evidence pointers linking errors to relevant XBRL sub-filings and US-GAAP taxonomy chunks. These pointers enable the retrieval of the corresponding XBRL segments and taxonomy chunks, grounding each instance in both the reported filing and the governing rules. Task-specific instances are then constructed as follows: (1) FinSM: Semantically incorrect US-GAAP concepts identified in DQC messages serve as ground-truth answers. The corresponding statement-level XBRL sub-filings are retrieved as input. (2) FinRE: DQC rule IDs are deterministically mapped to relation labels (DQC_0081: Reversal, DQC_0001: CombinationErr, DQC_0145: Inappropriateness). Participating elements form (head, relation, tail) triples, where the relation is the ground-truth label and the associated XBRL sub-filings provide input context. (3) FinMR: DQC messages contain both reported values and correct values implied by calculation linkbases. We use the target concept and reporting period to construct the input questions, with statementlevel XBRL sub-filings serving as the input for numerical reasoning.

3.3.2 Quality Control. To ensure the reliability and consistency of the FINAUDITING benchmark, we conducted a multi-stage quality control process: (1) Structural Verification: We manually inspected 50 randomly sampled filings to verify that extracted statement segments correctly matched their roleType identifiers and preserved the original reporting hierarchy. (2) Annotation Validation: 50% of all instances underwent double-validation, where GPT-4o-mini outputs were compared against independent human review, with all discrepancies corrected to ensure ground-truth accuracy. (3) Referential Integrity: We verified that all relation chunks reference valid core concepts in the taxonomy, ensuring the integrity of taxonomygrounded reasoning paths.

### 3.4 Evaluated LLMs

Our goal is to evaluate the foundational capabilities of state-of-theart LLMs on the FINAUDITING benchmark, to assess their capabilities and limitations in financial auditing. To this end, we select a diverse set of models, including one closed-source general-purpose LLM (GPT-4o [11]), ten open-source general-purpose LLMs of varying sizes (DeepSeek-V3 [16], Qwen3-235B-A22B-Instruct-2507 [27], Llama-4-Scout-17B-16E-Instruct [19], Qwen2.5-72B-Instruct [30], Llama-3.3-70B-Instruct [6], Qwen3-32B [27], Gemma-3-27B-IT [26],

#### Table 5: Zero-shot performance (%) on the FinSM task.

|LLMs|Hit Rate @5 @10 @20<br><br>|Recall @5 @10 @20<br><br>|Macro-F1 @5 @10 @20|
|---|---|---|---|
|GPT-4o DeepSeek-V3 Qwen3-235B-A22B-Instruct-2507 Llama-4-Scout-17B-16E-Instruct Qwen2.5-72B-Instruct Llama-3.3-70B-Instruct Qwen3-32B gemma-3-27b-it gemma-3-12b-it<br><br>Llama-3.1-8B-Instruct<br><br>Llama-3.2-3B-Instruct Fin-o1-14B Fin-R1<br><br><br>|9.09 9.09 9.09 11.82 11.82 12.42<br><br>10.00 10.00 10.00<br><br><br>2.42 2.73 3.03<br><br>8.18 8.48 8.48<br><br>5.15 5.15 5.15 0.00 0.00 0.00<br><br>10.30 10.30 10.30 9.70 9.70 10.30<br><br>6.06 6.06 6.06<br><br><br><br><br>3.94 4.24 4.55 0.00 0.00 0.00 2.12 2.42 2.73<br><br><br>|6.82 6.98 7.01 8.82 9.33 10.11<br><br>7.77 8.17 8.33<br><br><br>1.58 1.91 2.21 4.81 5.18 5.18<br><br>3.78 3.85 3.85 0.00 0.00 0.00 7.81 8.20 8.26 7.06 7.61 8.49<br><br>4.32 4.73 4.82 3.23 3.68 3.98 0.00 0.00 0.00<br><br><br>2.02 2.32 2.63<br><br><br>|6.91 7.03 7.02 9.54 9.98 10.17<br>7.83 7.96 7.97<br><br>1.55 1.76 1.80 5.29 5.42 5.42 4.07 4.14 4.14 0.00 0.00 0.00<br><br>8.38 8.67 8.71 7.62 8.04 8.27 4.47 4.77 4.80<br><br>2.88 2.95 2.94<br><br><br><br><br>0.00 0.00 0.00<br>1.92 1.97 2.00<br>|

Gemma-3-12B-IT [26], Llama-3.1-8B-Instruct [6], and Llama-3.2-

- 3B-Instruct [6]), as well as two open-source financial LLMs (Fino1 [22] and FinR1 [17]). All evaluations use the LM Evaluation Harness [5]. Proprietary models are accessed via APIs, while open-source models are run locally on a 4×H200 GPU cluster. For all tasks, we standardize the maximum input length to 81,920 tokens and the generation limit to 512 tokens.
- 4 Benchmarking Results and Empirical Analysis

### 4.1 FinSM

From Table 5, we observe that retrieval performance on FinSM remains uniformly low across models, with no consistent advantage from larger parameter scales or domain-specific financial pretraining. Even state-of-the-art general-purpose LLMs achieve Hit Rate@20 below 13%, while several models fail. This suggests that FinSM performance is not primarily driven by model size or surface-level domain familiarity, but instead depends on the ability to identify semantically inappropriate accounting concepts under strict taxonomydefined constraints.

Notably, increasing model capacity yields marginal improvements, indicating implicit representation learning is insufficient when hierarchical, taxonomy-defined semantics are not explicitly modeled. Financial models show similarly limited gains, highlighting that exposure to financial text does not guarantee accurate detection of mis-tagged concepts within regulated taxonomies.

The qualitative analysis reveals that retrieval errors arise from confusion over which reported concepts are semantically inappropriate under the given reporting context. For example, a filing incorrectly reports CashAndCashEquivalentsAtCarryingValue in a context where only Cash is permitted. In such cases, models often fail to flag the mismatch, instead retrieving tags or omitting the erroneous concept entirely. Similarly, models may overlook the misuse of aggregate tags such as AssetsCurrent when more granular components are required by the disclosure context. These errors indicate that models tend to rely on surface-level semantic similarity, rather than reasoning about contextual validity and taxonomy-defined reporting granularity.

### 4.2 FinRE

Table 6 indicates that FinRE is substantially more challenging than standard relation classification, as it requires precise interpretation of hierarchical, compositional, and cross-document constraints rather than surface-level semantic cues.

Among all evaluated models, GPT-4o exhibits the strongest and most stable performance, achieving high accuracy and balanced

#### Table 6: The overall performance (%) under the zero-shot settings on the FinRE task.

|LLMs<br><br>|Acc Macro P Macro R Macro F1|
|---|---|
|GPT-4o DeepSeek-V3 Qwen3-235B-A22B-Instruct-2507 Llama-4-Scout-17B-16E-Instruct Qwen2.5-72B-Instruct Llama-3.3-70B-Instruct Qwen3-32B gemma-3-27b-it gemma-3-12b-it<br><br>Llama-3.1-8B-Instruct<br><br>Llama-3.2-3B-Instruct Fin-o1-14B Fin-R1<br><br><br>|91.82 90.15 90.03 90.09 82.73 87.81 80.19 80.68 62.73 69.22 67.18 63.56<br><br>27.50 17.09 35.82 22.71 67.50 72.14 71.50 68.27<br>28.86 23.06 38.26 21.54 0.00 0.00 0.00 0.00<br><br><br>45.91 35.97 53.00 40.85 27.95 31.78 25.81 27.37 30.45 15.25 26.58 19.34 17.73 22.31 12.74 15.70 0.00 0.00 0.00 0.00 32.73 34.05 21.00 25.97|

macro-level precision, recall, and F1. This suggests that frontier models are capable of jointly reasoning over multiple relationship types when sufficient representational and reasoning capacity is available. DeepSeek-V3 follows as the strongest open-source model, but with a noticeable performance gap, indicating reduced robustness across relation categories.

In contrast, most open-source models struggle to generalize to this setting. Model scale alone does not lead to consistent improvements: several large models underperform smaller counterparts, and the Llama family exhibits uniformly weak results across scales. Notably, domain-specific financial LLMs do not demonstrate a clear advantage, with one model failing entirely and another achieving only limited accuracy. These observations suggest that relationship error classification in FinRE cannot be addressed by domain pretraining or parameter scaling alone, but instead requires explicit alignment with taxonomy-defined structural semantics.

[Figure 3]

#### Figure 3: The F1-score (%) for individual relation type under the zero-shot settings on the FinRE task.

Finer-grained analysis by relation type (Figure 3) reveals substantial difficulty variation. While Reversal and Inappropriateness relations are handled reasonably well by top models, CombinationErr emerges as the most challenging category. Most models collapse to near-zero F1 on this relation type, indicating difficulty in validating axis–member combinations that require reasoning across multiple interrelated linkbases. Even competitive models exhibit sharp performance drops, highlighting the complexity of enforcing structural consistency beyond pairwise relationships.

[Figure 4]

(a) Overall accuracy under zero-shot settings.

[Figure 5]

(b) Error-rate composition, including SER, ERR, and CER.

Figure 4: Zero-shot performance on the FinMR task.

### 4.3 FinMR

Figure 4 summarizes the zero-shot performance of representative large language models on the FinMR task. Overall accuracy remains consistently low, with the best-performing model, Fin-o114B, achieving only 13.86% accuracy. This underscores the intrinsic difficulty of FinMR, which requires jointly reasoning over hierarchical calculation structures, cross-fact dependencies, and arithmetic constraints defined in XBRL filings, rather than answering isolated numerical questions.

The error composition analysis in Figure 4b provides a more finegrained diagnostic view of model failures. Calculation errors (CER) dominate across most models, typically accounting for 70–83% of failures. This indicates that while models may identify relevant financial facts, they frequently fail to execute or validate multi-step arithmetic relations specified by taxonomy-grounded calculation rules. Extraction errors (EER) are comparatively moderate (approximately 10–27%), suggesting that locating candidate values is often feasible, but transforming them into correct numerical inferences remains a major bottleneck.

Structural errors (SER) further expose limitations in producing structure-compliant outputs. Models such as Qwen3-32B and FinR1 exhibit extremely high SER values, indicating frequent failures to generate outputs that conform to the required structured format. In contrast, Fin-o1-14B shows a markedly different error profile: despite achieving the highest overall accuracy and maintaining a low CER (9%), it still suffers from a high SER (71%). This pattern suggests that even when numerical reasoning is reliable, failures in adhering to the expected output format can prevent correct answers from being properly expressed and evaluated.

### 5 Conclusion

In this work, we present FINAUDITING, the first benchmark designed to evaluate LLMs on structured, hierarchical, and taxonomydriven financial reasoning across multi-document filings. Built from real-world XBRL data, FINAUDITING defines three complementary tasks, FinSM, FinRE, and FinMR, which jointly assess structured semantic consistency, hierarchical relationship understanding, and multi-step mathematical reasoning. Our extensive zero-shot evaluation of state-of-the-art LLMs shows that even leading models, such as GPT-4o, DeepSeek-V3, and Fin-o1-14B, still struggle with cross-document reasoning, structured interpretation, and taxonomy alignment. These findings highlight fundamental limitations in current LLMs for financial auditing and emphasize the importance of developing models with stronger structural grounding and explicit reasoning awareness. We release all datasets and evaluation code to support future research on trustworthy financial intelligence.

### References

- [1] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al.

2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122 (2021).

- [2] Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849

(2022).

- [3] Paulo Caetano da Silva. 2018. xAudit: Auditing Representation in XBRL Based Documents. In International Conference on Internet and Web Applications and Services.
- [4] Roger Debreceny, Stephanie Farewell, Maciej Piechocki, Carsten Felden, and André Gräning. 2010. Does it add up? Early evidence on the data quality of XBRL filings to the SEC. Journal of Accounting and Public Policy 29, 3 (2010), 296–306.
- [5] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. A framework for few-shot language model evaluation. doi:10.5281/zenodo.12608602
- [6] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [7] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. 2024. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594 (2024).
- [8] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, and Steve Y Yang. 2024. Xbrl agent: Leveraging large language models for financial report analysis. In Proceedings of the 5th ACM International Conference on AI in Finance. 856–864.
- [9] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web. 173–182.
- [10] Rani Hoitash and Udi Hoitash. 2018. Measuring accounting reporting complexity with XBRL. The Accounting Review 93, 1 (2018), 259–287.
- [11] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [12] Steven Katz, Yu Gu, and Lanxin Jiang. 2024. Information extraction from ESG reports using NLP: a ChatGPT comparison. Available at SSRN 4836432 (2024).
- [13] Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Kp Subbalakshmi, Jimin Huang, et al.

2025. Investorbench: A benchmark for financial decision-making tasks with llmbased agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2509–2525.

- [14] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. 2024. Llms-as-judges: a comprehensive survey on llm-based evaluation methods. arXiv preprint arXiv:2412.05579 (2024).
- [15] Hongqin Li and Jun Zhai. 2015. Literature review of XBRL semantic research. In 2015 International Conference on Computer Science and Intelligent Communication. Atlantis Press, 316–320.

- [16] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- [17] Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, and Liwen Zhang. 2025. Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning. arXiv:2503.16252 [cs.CL] https://arxiv.org/abs/2503.16252
- [18] Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malakasiotis, Ion Androutsopoulos, and Georgios Paliouras. 2022. FiNER: Financial numeric entity recognition for XBRL tagging. arXiv preprint arXiv:2203.06482 (2022).
- [19] AI Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodal-intelligence/, checked on 4, 7 (2025), 2025.
- [20] Rajdeep Mukherjee, Abhinav Bohra, Akash Banerjee, Soumya Sharma, Manjunath Hegde, Afreen Shaikh, Shivani Shrivastava, Koustuv Dasgupta, Niloy Ganguly, Saptarshi Ghosh, et al. 2022. Ectsum: A new benchmark dataset for bullet point summarization of long earnings call transcripts. arXiv preprint arXiv:2210.12467

(2022).

- [21] Lingfei Qian, Xueqing Peng, Yan Wang, Vincent Jim Zhang, Huan He, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, et al. 2025. When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents. arXiv preprint arXiv:2510.11695 (2025).
- [22] Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Jimin Huang, and Qianqian Xie. 2025. Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance. arXiv preprint arXiv:2502.08127 (2025).
- [23] Soumya Sharma, Subhendu Khatuya, Manjunath Hegde, Afreen Shaikh, Koustuv Dasgupta, Pawan Goyal, and Niloy Ganguly. 2023. Financial numeric extreme labelling: A dataset and benchmarking. In Findings of the Association for Computational Linguistics: ACL 2023. 3550–3561.
- [24] Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Future of Information and Communication Conference. Springer, 589–601.
- [25] Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate stock movement prediction with self-supervised learning from sparse noisy tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.
- [26] Gemma Team. 2024. Gemma. (2024). doi:10.34740/KAGGLE/M/3301
- [27] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https: //arxiv.org/abs/2505.09388
- [28] Yan Wang, Yang Ren, Lingfei Qian, Xueqing Peng, Keyi Wang, Yi Han, Dongji Feng, Xiao-Yang Liu, Jimin Huang, and Qianqian Xie. 2025. FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information. arXiv preprint arXiv:2505.20650 (2025).
- [29] Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443

(2023).

- [30] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671 (2024).
- [31] Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. arXiv preprint arXiv:2206.01347 (2022).
- [32] Yilun Zhao, Yitao Long, Hongjun Liu, Ryo Kamoi, Linyong Nan, Lyuhao Chen, Yixin Liu, Xiangru Tang, Rui Zhang, and Arman Cohan. 2023. DocMath-eval: Evaluating math reasoning capabilities of LLMs in understanding long and specialized documents. arXiv preprint arXiv:2311.09805 (2023).
- [33] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. arXiv preprint arXiv:2105.07624 (2021).

### A LLM Judgment for Data Annotation A.1 Prompt for the FinSM task

##### DQC_0099 Example

You are given a DQC validation message. Your task is to extract two specific pieces of information:

- 1. **Statement Name**: Extract the name of the financial statement mentioned in the message. If it appears in a phrase like "0000007 - Statement Consolidated Statements of Operations", ignore any prefixes such as "Statement -" or ID numbers. Only return the clean name of the statement (e. g., "Consolidated Statements of Operations").
- 2. **Erroneous us-gaap Tag(s)**: From all `us-gaap:` prefixed tags mentioned in the message, extract only those that are clearly described as having an

error or violation. A tag should be included **only if** it satisfies both of the following:

- - It is explicitly stated to be **not included in any calculation relationship**.
- - The message states that it **will produce an error**, **violates a rule

**, or **should have been reported differently**. Do not include `us-gaap:` tags that are merely suggested, referenced as valid abstract parents, or shown as examples for correction.

- 3. **Reasoning**: For each extracted erroneous tag, provide a short explanation summarizing why it was classified as erroneous, based on the message.

Output your result as a structured JSON in the following format: ```json {

"statement_name": "...", "error_tags": [

{

"tag": "...", "reason": "..."

} ]

} ```

DQC_0109 Example

You are given a DQC validation message. Your task is to extract all Dimensions information from the message.

- 1. Extract all key-value pairs under Dimensions, where each dimension is formatted as us-gaap:XXX=YYY.
- 2. For each extracted dimension, provide a short explanation in the "reason" field based on the message context. If no explicit issue is mentioned about

a specific dimension, you may leave the "reason" as an empty string. Output your result as a structured JSON in the following format: ```json {

"error_tags": [ {

"dimension": "...", "reason": "..."

} ]

} ```

DQC_0137 Example

You are given a DQC validation message. Your task is to extract the following information:

- 1. Target Tag: Extract the primary us-gaap tag being referenced or discussed as erroneous in the message. Only include the tag explicitly stated as problematic or requiring correction.
- 2. Dimensions: Extract all dimension key-value pairs listed under "Dimensions" in the message. Each dimension should be in the format us-gaap:XXX=YYY.
- 3. For each extracted dimension, provide a short explanation in the "reason" field based on the message context. If the message does not explicitly mention an issue with the dimension, you may leave the "reason" field empty

or give a general summary. Output your result as a structured JSON in the following format: ```json {

"target_tag": "...", "error_tags": [

{

"dimension": "...", "reason": "..."

} ]

} ```

### A.2 Prompt for the FinRE task

##### DQC_0081 Example

You are given a DQC validation message. Your task is to extract two specific pieces of information:

- 1. **Statement Name**: Extract the name of the financial statement mentioned in the message. If it appears in a phrase like "00000002 - Statement BALANCE SHEET", ignore any prefixes such as ID numbers or "- Statement -". Only return the clean name of the statement (e.g., "BALANCE SHEET").
- 2. **Erroneous us-gaap Tag Relationships**: From all `us-gaap:` tags mentioned in the message, identify pairs of tags where the relationship between them is described as incorrect or inconsistent. A tag pair should be extracted

**only if** it satisfies both of the following conditions:

- - The two tags are stated to be in a specific structural relationship in the filer's taxonomy (e.g., sibling, parent-child).
- - The message clearly indicates that this relationship contradicts the definition in the official US-GAAP taxonomy, or requires reviewer attention for potential misclassification.

- 3. **Reasoning**: For each extracted tag pair, provide a short explanation summarizing the inconsistency or incorrect relationship between them as described in the message.

Output your result as a structured JSON in the following format: ```json {

"statement_name": "...", "error_tags": [

{

- "tag1": "...",
- "tag2": "...", "reason": "..."

} ]

} ```

DQC_0001 Example

You are an expert in XBRL taxonomies, tasked with parsing DQC validation messages for relationship errors (Rule DQC.US.0001), specifically an axis with an inappropriate member.

- 1. **Main Concept**: Extract the primary concept that is being dimensionally qualified. This is usually the first concept mentioned in the message.
- 2. **Dimension Pair**: From the 'Dimensions' field in the message, extract the full axis and member combination string **exactly as it appears**,

including all prefixes.

- 3. **Reasoning**: Provide a short explanation that the member is unallowable for the specified axis, as described in the message.

Output your result as a single, structured JSON object in the following format

: ```json {

"main_concept": "UnrealizedGainLossOnCashFlowHedgingInstruments", "dimension_pair": "us-gaap:FairValueByFairValueHierarchyLevelAxis=us-gaap:

FairValueMeasurementsRecurringMember", "reason": "The message indicates the member is unallowable for the specified axis."

} ```

DQC_0145 Example

You are an expert in XBRL taxonomies, tasked with parsing DQC validation messages for relationship errors (Rule DQC.US.0145), specifically ' Inappropriate Cash Flow Presentation'.

- 1. **Head Concept**: Extract the primary concept that is inappropriately presented. This is usually the first concept mentioned in the message.
- 2. **Tail Concept**: Extract the presentation concept that the head concept is incorrectly a descendant of.
- 3. **Reasoning**: Provide a short explanation that the head concept should not be presented as a component of the tail concept.

Output your result as a single, structured JSON object in the following format

: ```json {

"head_concept": "element 1", "tail_concept": "element 2", "reason": "The message indicates the head concept is inappropriately

presented as a descendant of the tail concept and should be outside this group."

} ```

### A.3 prompt for the FinMR task

##### DQC_0015 Example

You are given a DQC validation message. Your task is to extract the following key information:

1. **Erroneous Tag Information:**:

Identify the primary us-gaap tag in the message that is reported with an incorrect value due to sign (positive/negative) error. For this tag, extract:

- - `tag`: The us-gaap element tag with the error.
- - `period`: The reporting period for this tag, as shown in the "Period" field.
- - `reported_value`: The actual value reported in the filing, as stated in the message or "Value" field.
- - `correct_value`: The correct value, which should be the absolute value of the reported value (i.e., reported_value without the negative sign), as implied or explicitly required by the message.
- - `reason`: Briefly explain why the reported value is considered incorrect, based on the message content (for example, "This element should not

have a negative value. The amount should be input as a positive value and, if necessary, a negated label should be provided.").

Return your result as structured JSON in the following format: ```json {

"error_tags": {

"tag": "...", "period": "...", "reported_value": "...", "correct_value": "...", "reason": "..."

}

} ```

DQC_0117 Example

You are an expert financial analyst parsing DQC validation messages for mathematical reasoning errors (Rule DQC.US.0117). Your task is to extract all required components from the message.

- 1. **Statement Name**: Extract the clean name of the financial statement (e.g

., "Condensed Consolidated Income Statements").

- 2. **Input Data**: From the properties list at the end of the message, extract the following:

- * `target_concept`: The full name of the primary `us-gaap:` concept being evaluated.
- * `period`: The reporting period for the fact (e.g., "2020-10-01 to 2020-12-31").

- 3. **Output Data**: From the main body of the message, extract the following values:

- * `extracted_value`: The value of the concept as reported in the filing (e

.g., "255,500,000").

- * `calculated_value`: The correct value based on the dimensional breakdown (e.g., "142,400,000").
- * `is_correct`: This should always be "No" for these error messages.

Output your result as a single, structured JSON object in the following format

: ```json {

"statement_name": "...", "input": {

"target_concept": "us-gaap:

RevenueFromContractWithCustomerIncludingAssessedTax", "period": "..."

}, "output": {

"extracted_value": "...", "calculated_value": "...", "is_correct": "No"

}

} ```

DQC_0126 Example

You are given a DQC validation message. Your task is to extract the following key information:

- 1. **Statement Name**: Extract the name of the financial statement mentioned in the message. If it appears in a phrase like "000004 - Statement CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)", ignore any numeric prefixes and the phrase "- Statement -". Only return the clean name of the statement, such as "CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)".
- 2. **Erroneous Total Element Tag Information**: Extract detailed information about the total element (`us-gaap:` tag) that

is reported incorrectly. Specifically, you should extract:

- - `tag`: The total element tag, as indicated by the **"Total Element"** field in the message.
- - `period`: The reporting period associated with this tag, as indicated by the **"Total period"** field.
- - `reported_value`: The value that was reported in the filing, taken from the **"Total Value"** field.
- - `correct_value`: The expected value, i.e., the **sum of the child components** as defined in the calculation linkbase.
- - This is usually explicitly stated in a sentence like: *"The sum of these child components is..."*
- - You may cross-validate this by summing the listed child component values in the message if provided.
- - `reason`: Provide a concise explanation of the error - why the reported value is considered incorrect - based directly on the message content. Only include this tag if all of the following are true:
- - The message identifies a total element with incorrect calculation
- - Both reported and correct values are stated or inferable
- - The error occurs in a specified reporting period Return your result as structured JSON in the following format: ```json {

"statement_name": "...", "error_tags":

{

"tag": "...", "period": "...", "reported_value": "...", "correct_value": "...", "reason": "..."

}

} ```

### B US-GAAP Taxonomy Chunking

To enable structured retrieval and reasoning, we convert the official US-GAAP taxonomy spreadsheets into concept-centric chunks. Each GAAP concept is represented by a core chunk capturing its intrinsic attributes and a set of relation chunks describing its semantic and structural links to other concepts. The conversion operates directly on the official Excel workbook through systematic normalization and extraction.

Sheet Normalization. All sheets are loaded from the taxonomy workbook, column headers are standardized, and relevant sheets (Concept, Presentation, Calculation, Definition, Reference, Enumeration) are identified by keyword matching. Unnamed or irrelevant reference sheets are excluded, and each extracted element is annotated with provenance information such as file name, sheet name, and row number.

Core Chunk Extraction. From the Concepts sheet, each concept’s identifier, label, type, balance, period type, abstract flag, documentation, and deprecation status are extracted and rendered into a canonical text form (concept::core) suitable for retrieval or embedding.

Relation Chunk Extraction. For each concept, all relation-bearing sheets are parsed to construct chunks representing presentation hierarchies, calculation formulas, definition arcs, reference citations, and enumerations. The parser handles heterogeneous layouts, missing columns, and varying role conventions. Each chunk is recorded

with its role, arcrole, and provenance metadata under the identifier concept::relations:pres|calc|def|ref|enum, with integrity checks ensuring that all relation targets correspond to valid core concepts.

Outputs. All chunks are written into chunks_core.jsonl and chunks_relations.jsonl, and summary statistics are stored in meta.json.

This representation preserves the hierarchical and relational structure of the US-GAAP taxonomy while making each concept independently retrievable.

To enable structured retrieval and reasoning, we convert the official US-GAAP taxonomy spreadsheets into concept-centric chunks. Each GAAP concept is represented by two complementary components: a core chunk that captures intrinsic attributes and a collection of relation chunks that encode its semantic and structural links to other concepts. The conversion process operates directly on the official Excel workbook, performing systematic normalization and extraction as follows.

Sheet Normalization and Identification. We first load all sheets within the taxonomy workbook, standardize their column headers by lowercasing and trimming whitespace, and identify relevant sheets by keyword matching (Concept, Presentation, Calculation, Definition, Reference, and Enumeration). Suspicious “all-unnamed” reference sheets are ignored. The taxonomy version number is automatically inferred from the filename. Each extracted element is annotated with its provenance, including the file name, sheet name, and row number.

Core Chunk Extraction. From the Concepts sheet, we extract for each concept its identifier (prefix:name), label, type, balance, period type, abstract flag, documentation, and deprecation status. These attributes are rendered into a canonical, human-readable chunk_text and stored with a stable identifier in the form concept::core. The core chunk provides a self-contained textual summary suitable for direct retrieval or embedding.

Relation Chunk Extraction. For every concept, we parse all relationbearing sheets to construct one chunk per relation family:

- • Presentation: hierarchical parent–child paths with associated roles and preferred labels.
- • Calculation: quantitative relationships including parent/child direction, weights, and roles.
- • Definition: arcs between dimensions, domains, and hypercubes, including arcrole, source, and target.
- • Reference: external standard citations consisting of source, section, and note, filtered to the relevant concept.
- • Enumeration: extensible lists aggregated by domain and linkrole, with de-duplicated members.

The extraction is designed to handle heterogeneous sheet layouts and missing columns: when explicit from*/to* fields are absent, the parser falls back to parent/child-style columns and merges multiple role or linkrole conventions. Each relation chunk records the roles and arcroles encountered, as well as complete provenance metadata, and uses the identifier concept::relations:{pres|calc|def|ref|enum} to store. Integrity validation ensures that all relation targets reference existing core concepts, and malformed rows are automatically skipped.

Outputs. All chunks are written into chunks_core.jsonl and chunks_relations.jsonl, while global statistics are summarized in meta.json. This chunked representation preserves the hierarchical and relational organization of the US-GAAP taxonomy while making each concept addressable as an atomic, retrievable unit, as shown in the following.

Example of a Core Chunk

[Concept Core] ID: us-gaap:Revenues Label: Revenues Type: monetaryItemType | Balance: credit | PeriodType: duration | Abstract:

False Status: active | DeprecatedLabel: None | DeprecatedDate: None Documentation: Amount of revenue recognized from goods sold or services rendered during the

period. Provenance: file=GAAP_Taxonomy_2024.xlsx | sheet=Concepts | row=248

##### Example of a Relation Chunk

[Concept Relations] ID: us-gaap:Revenues Presentation:

- Role: Statement - Income Statement

Path: Revenues > CostOfRevenue > GrossProfit Calculation:

- - As Parent: child=us-gaap:CostOfRevenue | weight=-1.0 | role=Calculation Linkbase
- - As Parent: child=us-gaap:GrossProfit | weight=+1.0 | role=Calculation Linkbase

Definition (Dimensions/Domain/Hypercube):

- domain-member: from=RevenueRecognitionPolicyTextBlock -> to=

RevenueCategoryAxis References:

- FASB ASC 606 | Section 25 | Revenue Recognition Provenance: sheets include Presentation-like and Calculation-like sheets

### C Evaluation Metrics C.1 The metrics for FinSM

For the FinSM task, we evaluated retrieval performance using Hit Rate (𝐻𝑅@𝑘), Recall (𝑅@𝑘), and Macro-F1 (𝑀𝐹1@𝑘), where 𝑘 = {1, 5, 10, 20}. Hit Rate evaluates whether the LLM retrieves at least one relevant element within the top-𝑘 results. It reflects the proportion of queries for which the model can successfully hit a relevant item, without considering how many are retrieved. Recall measures the fraction of all relevant elements that are successfully retrieved within the top-𝑘 predictions. It captures how completely the model covers the relevant set under a cutoff of 𝑘. Macro-F1 balances precision and recall at the query level by computing the F1 score for each query using its top-𝑘 predictions and then averaging across all queries. It reflects both the accuracy (precision) and completeness (recall) of retrieval, giving equal weight to each query.

∑︁𝑁

1 𝑁

1 𝐸𝑖(𝑘) ∩ 𝐺𝑖 ≠ ∅ (2)

HR@𝑘 =

𝑖=1

∑︁𝑁

|𝐸𝑖(𝑘) ∩ 𝐺𝑖 | |𝐺𝑖 |

1 𝑁

R@𝑘 =

(3)

𝑖=1

∑︁𝑁′

2 × 𝑃𝑖(𝑘) × 𝑅𝑖(𝑘) 𝑃𝑖(𝑘) + 𝑅𝑖(𝑘)

1 𝑁′

Macro-F1@𝑘 =

(4)

𝑖=1

where

 

 

|𝐸𝑖(𝑘) ∩ 𝐺𝑖 | |𝐸𝑖(𝑘) |

|𝐸𝑖(𝑘) ∩ 𝐺𝑖 | |𝐺𝑖 |

, |𝐸𝑖(𝑘) | > 0 0, otherwise

, |𝐺𝑖 | > 0 0, otherwise

, 𝑅𝑖(𝑘) =

𝑃𝑖(𝑘) =

 

 

and 𝑁 is the total number of queries, and 𝑁′ is the number of

queries with non-empty ground truth sets. 𝐸𝑖(𝑘) denotes the top-𝑘 predicted elements for the 𝑖-th query. 𝐺𝑖 denotes the set of ground

truth elements for the𝑖-th query. |𝐸𝑖(𝑘)∩𝐺𝑖| is the number of correctly predicted elements among the top-𝑘 results. 𝑘 is the cutoff rank

indicating the number of top predictions considered. 1(·) is the indicator function, returning 1 if the condition holds and 0 otherwise.

### C.2 The metrics for FinRE

For the FinRE task, we assessed the performance of various LLMs using Accuracy (Acc), Precision (Macro P), Recall (Macro R), and F1-score (Macro F1). Accuracy measures the overall proportion of correctly classified relations among all predictions. Precision evaluates the proportion of correctly predicted relations within each class, averaged across the three relation categories (Macro P). Recall quantifies the proportion of true relations that were successfully identified for each class, also averaged across classes (Macro R). F1-score provides a balanced measure of Precision and Recall by computing their harmonic mean per class and then averaging across the three classes (Macro F1).

∑︁𝑁

1 𝑁

1(𝑦ˆ𝑖 = 𝑦𝑖 ) (5)

Acc =

𝑖=1

∑︁𝐶

1 𝐶

𝑇𝑃𝑐 𝑇𝑃𝑐 + 𝐹𝑃𝑐

Macro-P =

(6)

𝑐=1

∑︁𝐶

1 𝐶

𝑇𝑃𝑐 𝑇𝑃𝑐 + 𝐹𝑁𝑐

Macro-R =

(7)

𝑐=1

∑︁𝐶

1 𝐶

2 × 𝑃𝑐 × 𝑅𝑐 𝑃𝑐 + 𝑅𝑐

Macro-F1 =

(8)

𝑐=1

where 𝑁 is the total number of instances. 𝐶 is the number of relation categories (here 𝐶 = 3). 𝑦𝑖 and 𝑦ˆ𝑖 denote the ground truth and predicted relation labels for the 𝑖-th instance. 𝑇𝑃𝑐, 𝐹𝑃𝑐, and 𝐹𝑁𝑐 denote the number of true positives, false positives, and false negatives for class 𝑐, respectively. 𝑃𝑐 and 𝑅𝑐 denote the precision and recall for class 𝑐. 1(·) is the indicator function that equals 1 when the prediction is correct and 0 otherwise.

### C.3 The metrics for FinMR

For the FinMR task, we evaluated the mathematical reasoning capabilities of various LLMs using an LLM-as-a-judge framework to assess the overall Accuracy of generated answers. This metric reflects whether each model produces the correct numerical result under the specified auditing rule. In addition to overall correctness, we introduced three fine-grained error indicators, namely the Structural Error Rate (SER), Extraction Error Rate (EER), and Calculation Error Rate (CER), each corresponding to a specific stage of reasoning validation.

SER measures the proportion of outputs with invalid structure, where the model fails to produce the required JSON format containing both the extracted and calculated values. EER quantifies errors in identifying or reproducing the correct extracted value, meaning that the predicted numerical entity does not match the true extracted value in mathematical meaning. CER captures computational mistakes, where the predicted calculated value deviates from the ground truth even when the structure and extracted value are both correct.

Together, these indicators provide a hierarchical view of reasoning quality, reflecting whether an error arises from structural formatting, factual extraction, or numerical computation, thereby offering fine-grained insights into model performance on complex financial mathematical reasoning tasks. The specific prompt for judgment is shown below.

LLM-as-a-judge prompt for FinMR Task

Instruction: You are an evaluator. Your task is to judge whether the model's

output pred_answer is correct compared to the given true_answer. Follow the rules strictly:

- Step 1 (Structure Check): Verify whether pred_answer has the same structure as true_answer. The

required structure is a JSON object with exactly two keys: {{"extracted_value": <value>, "calculated_value": <value>}} Minor formatting differences (e.g., line breaks, indentation, whitespace)

are acceptable. If the structure is invalid, output the label: S If valid, continue to Step 2

- Step 2 (Extracted Value Check): Compare true_answer["extracted_value"] and pred_answer["extracted_value"]

by their mathematical meaning, not their string form. For example, "-1,284" and "-1284" are considered equal.

If they are not equal in numeric meaning, output the label: E If equal, continue to Step 3

- Step 3 (Calculated Value Check): Compare true_answer["calculated_value"] and pred_answer["calculated_value

"] strictly in numeric meaning. They must be exactly equal (zero tolerance).

If they are not equal, output the label: C If equal, then everything is correct

Final Decision: If all three checks pass, output the label: A Output only one label: S, E, C, or A. Do not explain your reasoning.

Given a total of 𝑁 evaluated instances, let 𝑁𝐴, 𝑁𝑆, 𝑁𝐸, and 𝑁𝐶

denote the numbers of instances labeled as Accurate (A), Structural Error (S), Extraction Error (E), and Calculation Error (C), respectively. Let 𝑁valid represent the number of successfully parsed instances that received a valid label. We define the following metrics:

𝑁𝐴 𝑁valid

Accuracy (ACC) =

𝑁𝑆 𝑁valid

Structural Error Rate (SER) =

(9)

(10)

𝑁𝐸 𝑁valid

Extraction Error Rate (EER) =

(11)

𝑁𝐶 𝑁valid

Calculation Error Rate (CER) =

(12)

where each instance is judged by the LLM-as-a-judge framework to output exactly one label from {𝐴,𝑆,𝐸,𝐶}. ACC, SER, EER, and CER, respectively, quantify the proportions of correct, structural, extraction, and calculation outcomes among validly parsed cases.

### D Evaluation Models

Table 7 summarizes all models evaluated in this study, categorized by openness, domain specialization, and architectural foundation. The evaluation covers a wide range of both closed- and open-source large language models (LLMs) as well as domain-specific and pretrained baselines.

• Closed-source LLMs: We include GPT-4o [11], accessed via OpenAI’s API, as a representative of state-of-the-art proprietary models. Although the architecture and model size remain undisclosed, GPT-4o serves as a strong upper-bound reference in our benchmark.

#### Table 7: Model categories and corresponding repositories.

Model Size Repository Closed-source LLMs GPT-4o – gpt-4o-2024-08-06 Open-source LLMs DeepSeek-V3 685B deepseek-chat Llama-4-Scout-17B-16E-Instruct 109B meta-llama/Llama-4-Scout-17B-16E-Instruct Qwen3-235B-A22B-Instruct-2507 235B Qwen/Qwen3-235B-A22B-Instruct-2507 Llama-3.3-70B-Instruct 70B meta-llama/Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct 72B Qwen/Qwen2.5-72B-Instruct Qwen3-32B 32B Qwen/Qwen3-32B gemma-3-27b-it 27B google/gemma-3-27b-it gemma-3-12b-it 12B google/gemma-3-12b-it

- Llama-3.1-8B-Instruct 8B meta-llama/Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct 3B meta-llama/Llama-3.2-3B-Instruct

Financial-specific LLMs Fin-o1-14B 14B TheFinAI/Fin-o1-14B Fin-R1 7B SUFE-AIFLM-Lab/Fin-R1

- • Open-source LLMs: This category encompasses a diverse set of publicly available, instruction-tuned models, including DeepSeek-V3 [16], Qwen3-235B-A22B-Instruct-2507 [27], Qwen2.5-72B-Instruct [30], and Qwen3-32B [27], offering a wide scale range. We also include several Llama models from Meta: Llama-4-Scout-17B-16E-Instruct [19], Llama3.3-70B-Instruct [6], Llama-3.1-8B-Instruct [6], and Llama3.2-3B-Instruct [6], as well as Google’s Gemma models [26] (gemma-3-27b-it and gemma-3-12b-it). These models collectively enable cross-architecture and scaling comparisons across modern open-source systems.
- • Financial-specific LLMs: We include Fin-o1-14B [22] and Fin-R1 [17], both trained on domain-specific financial corpora to capture specialized terminology and reasoning patterns in financial reporting. These models allow us to assess the impact of domain adaptation on structured auditing and numerical reasoning tasks.

Together, these models form a comprehensive evaluation spectrum, spanning closed- and open-source systems, general-purpose and domain-specialized models, and encoder- versus decoder-based architectures, enabling an in-depth analysis of performance across all proposed benchmark tasks.

### E Practical Deployment and Scalability Considerations

This section provides additional discussion on how FINAUDITING may be integrated into real-world auditing workflows and scaled under practical constraints. The goal is not to prescribe a specific deployment system, but to illustrate how the benchmark design aligns with existing auditing practices and supports modular instantiation.

Integration into Existing Auditing Pipelines. In practice, financial auditing commonly relies on rule-based validation systems, such as XBRL consistency checks and Data Quality Committee (DQC) rules, followed by targeted human review. FINAUDITING is naturally positioned to operate downstream of such validators. Rather than processing entire filings, models evaluated under FINAUDITING are applied only to a subset of suspicious items already identified by deterministic checks. Within this setting, the three benchmark tasks correspond to distinct stages of post-validation

analysis. FinSM supports taxonomy disambiguation by assisting auditors in identifying appropriate concepts when potential mis-tagging is detected. FinRE provides structured categorization of detected issues into interpretable error types, which can guide remediation and prioritization. FinMR focuses on targeted numerical verification by tracing calculation structures and comparing reported values with recomputed ones, and is intended for precision-critical cases. This workflow reflects common auditing practices in which automated components emphasize recall, while high-stakes decisions remain subject to expert review.

Scalability across Regulatory Environments. Although instantiated using the US-GAAP taxonomy, the task formulation of FINAUDITING is not tied to any specific accounting standard. The benchmark abstracts auditing as taxonomy-grounded reasoning over instance documents, schema definitions, and linkbase relations. Extending FINAUDITING to other regulatory environments, such as IFRS or jurisdiction-specific reporting standards, primarily requires substituting the underlying taxonomy and rule sources while preserving the same task interfaces. This decoupling between task definitions and accounting standards enables scalable adaptation without redesigning the benchmark structure.

Lightweight Variants and Semi-automatic Deployment. FINAUDITING also supports tiered or hybrid deployment strategies that balance performance with computational cost. Full-capacity large language models are not required at all auditing stages. For example, FinSM can be instantiated using embedding-based retrieval or compact instruction-tuned models to achieve high recall at low cost, effectively narrowing the candidate space for auditors. FinRE involves a small and fixed label set, making it suitable for fine-tuning lightweight classifiers. FinMR, which requires precise numerical reasoning and evidence tracing, can be selectively applied to a limited number of high-risk cases and handled by larger models or human-in-the-loop verification. Such semi-automatic configurations enable practical deployment by combining lightweight automation with selective use of more advanced reasoning components.

Overall, by aligning with existing auditing workflows and supporting model stratification, FINAUDITING provides a modular evaluation framework that reflects how machine-assisted financial auditing systems may be integrated and scaled under realistic operational constraints.

