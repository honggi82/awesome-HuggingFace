## arXiv:2603.22327v2[cs.IR]4Jun2026

### Evaluating AI-based Scientific Knowledge Synthesis with Epidemiological Systematic Reviews

Shreyansh Padarha1,∗ Ryan Othniel Kearns1,∗ Tristan Naidoo2 Lingyi Yang3 Łukasz Borchmann4 Piotr Błaszczyk5 Christian Morgenstern2 Ruth McCabe2 Sangeeta Bhatia2 Philip H. Torr1 Jakob Foerster1 Scott A. Hale1 Thomas Rawson1 Anne Cori2 Elizaveta Semenova2 Adam Mahdi1

1University of Oxford 2Imperial College London 3University of Nottingham 4Snowflake AI Research 5Independent

##### Abstract

Systematic literature reviews (SLRs) are a demanding and high-stakes form of scientific knowledge synthesis that remains underspecified as an evaluation setting for large language models (LLMs). We introduce AgentSLR, a large-scale evaluation harness comprising an SLR automation workflow and an expert annotated dataset covering 16,248 articles, designed to test LLM capabilities across the stages of SLRs in epidemiology. Reference annotations were derived from peer-reviewed studies on WHO priority pathogens and produced by domain experts. The harness evaluates each review stage as a separate unit with dedicated metrics enabling targeted failure analysis. We evaluated five frontier reasoning models and found that no single model dominated across all tasks, showing sub-task specialisation often hidden by aggregate benchmarks. Structured data extraction is a major bottleneck, with no model exceeding an average field-level F1 of 0.67. Estimated costs vary substantially, by up to 96× across evaluated models. Documented failure modes suggest that the evaluated models are not yet reliable enough for unsupervised deployment in epidemiology, where findings can inform public policy.

[Figure 1]

[Figure 2]

[Figure 3]

oxrml.com/agent-slr Dataset Harness Code

##### 1 Introduction

Large language models (LLMs) now answer expert-level scientific questions [1], reason over extended research tasks [2] and interpret scientific figures [3]. These capabilities motivate applications beyond isolated question answering, towards scientific workflows where evidence must be retrieved, judged, organised and interpreted under domain constraints [4–6]. Systematic literature reviews (SLRs) are among the most important examples of such workflows. They require screening, extraction and synthesis of thousands of scientific articles.

LLM-based evidence synthesis workflows show benefits, but their reliability remains limited. Models can reduce the time required for title and abstract screening [7], while still producing false inclusions and false exclusions that require expert review. Beyond screening, LLM summaries can omit details that limit the scope of study findings, making conclusions appear more general than the source article support [8]. These problems get worse when models are chained together in longer workflows, where errors compound and result in failures [9].

Infectious disease epidemiology provides a particularly rigorous evaluation setting for LLM knowledge synthesis. Epidemiological SLRs require careful extraction of key parameters such as the basic reproduction number, serial interval and case-fatality ratio. The same numbers may be reported

∗Equal contribution.

Correspondence to: shreyansh.padarha@oii.ox.ac.uk, adam.mahdi@oii.ox.ac.uk.

b Title and Abstract Screening c PDF-to-Markdown Conversion

a Article Search and Retrieval

Title

TXT

Abstract

PDF JPEG OCR M

###### Title

PDF

Abstract

Exclusion Criteria

Inclusion Criteria

…

###### …

d Full-text Screening f Report Generation

###### e Data Extraction

|Title<br><br>Abstract<br><br>| | |
|---|---|---|
| | | |

Title

M

M

Abstract

Report

Exclusion Criteria

Inclusion Criteria

…

…

Parameters Outbreaks Models

- Figure 1: AgentSLR evaluation harness (LLM-assisted workflow). (a) Article Search and Retrieval queries bibliographic databases with domain-specific Boolean searches and obtains PDFs from openaccess sources. (b) Title and Abstract Screening uses LLMs to filter articles using expert-designed inclusion and exclusion criteria. (c) PDF-to-Markdown Conversion uses OCR to convert PDFs to markdown files. (d) Full-text Screening applies the same approach as in Title and Abstract Screening but with stricter criteria. (e) Data Extraction employs multi-stage tool-calling with schema validation to extract structured epidemiological data (parameters, models, outbreaks). (f) Report Generation synthesises extracted data through artefact generation followed by LLM refinement.

differently across studies, split by age group, geography or clinical severity [10, 11]. This variation in reporting is not just a difficulty. It is the core test. A model that can only retrieve numbers but cannot match them to the right context will fail at the task.

In this work, we ask whether LLMs can carry out the core evidence handling tasks of epidemiological SLRs. We conduct this study in collaboration with the Pathogen Epidemiology Review Group (PERG), whose verified expert annotations from WHO-designated priority pathogen reviews serve as the human reference standard for evaluation.

Our contributions are as follows:

- • We release an expert-annotated evaluation dataset for scientific evidence synthesis in infectious disease epidemiology, constructed from SLRs on WHO-designated priority pathogens. The dataset contains 16,248 article records matched to expert screening labels, 3,808 parameter extractions, 687 transmission model extractions and 189 outbreak extractions.
- • We introduce AgentSLR, an evaluation harness for LLM-assisted epidemiological SLR workflows covering retrieval, screening and structured extraction. AgentSLR scores extraction at the record level, including cases where the same evidence can be recorded in more than one correct way.
- • We evaluate five frontier reasoning LLMs and find that no single model consistently dominates across stages of the workflow. Failure modes such as refusal behaviour in closed systems show that open, locally reproducible infrastructure is critical for scientific evidence synthesis.

##### 2 Related Work

Evaluating AI systems on systematic literature reviews requires operationalising the full review workflow, spanning retrieval, screening and evidence structuring [12, 13]. LLMs can transfer screening logic across title, abstract and full-text stages without task-specific fine-tuning, achieving high sensitivity and specificity across multiple systematic reviews [14, 15]. For data extraction, LLMs perform well on constrained schemas but degrade on complex fields, with human-incorporated workflows generally outperforming LLM-only approaches [16–18]. Building on stage-specific

advances, recent work has shifted towards aggregate workflow-level pipelines [19–22]. Yet these pipelines tend to be tailored to individual domains. For example, Parkinson et al. [21] target bee ecotoxicology and Cao et al. [20] focus on Cochrane-style clinical reviews. Epidemiology poses a distinct challenge requiring its own dedicated pipeline or evaluation harness.

Recent evaluation work examines which review and research-synthesis capabilities LLMs reliably support. Wang et al. [23] evaluate clinical evidence synthesis using pre-defined stable review questions and test whether extracted values match reference annotation values from published systematic reviews. Madeyski et al. [24] recommend recall-focused screening metrics, full confusion matrices and cost-sensitive analysis under class imbalance. Wang et al. [25] evaluate within paper sentence retrieval for biomedical hypotheses using expert written evidence summaries. Polzak et al. [26] test whether models match Cochrane review conclusions from source studies.

These studies motivate direct evaluation of review workflows. AgentSLR differs in its evaluation construct and introduces additional degrees of complexity. A model must identify the evidence family present in a paper (parameters, transmission models or outbreaks reported), use the appropriate schema and recover values with uncertainty bounds and population context. The same paper may report multiple estimates across age groups, locations or time periods, with the values implicit in disaggregated tables. Records of the same evidence family often appear in one paper with no canonical alignment to human annotations, requiring design of metrics beyond exact match. To our knowledge, AgentSLR is the first evaluation of LLMs on epidemiological SLRs against expert annotations from peer-reviewed WHO priority pathogen reviews. See Appendix A for an extended related work comparison.

##### 3 Evaluation Harness

Here, we describe AgentSLR, the proposed open-source evaluation harness. The harness has two parts. The first is a workflow built to replicate the main evidence-handling stages of a human-conducted epidemiological SLR, as shown in Figure 1. The second is an evaluation protocol that maps workflow outputs to expert labels through stage-isolated metrics.

###### 3.1 Building SLR Workflows

Article Search and Retrieval AgentSLR queries three bibliographic databases (OpenAlex, PubMed, and Europe PMC) using domain-specific Boolean search strategies covering seven core epidemiological domains (Figure 1a, Appendix G). Retrieved records are first deduplicated using identifier and bibliographic metadata-based matching, then full texts are automatically retrieved from open-access sources. The download pipeline incorporates caching, streaming and file validation, parallel execution and checkpointing.

Title and Abstract Screening Initial screening is conducted using titles and abstracts based on predefined inclusion and exclusion criteria (Figure 1b, Appendix H). Following the ScreenPrompt methodology [14], we structure screening with five components: study objectives, inclusion/exclusion criteria, chain-of-thought reasoning instructions, article abstract and structured output format.

PDF-to-Markdown Conversion Each downloaded PDF is rendered page-wise into high-resolution images, then processed with an OCR model to recover text while preserving document hierarchy, equations (LaTeX) and tables (Figure 1c). The process produces one Markdown file per article.

Full-text Screening Converted articles undergo full-text screening using the tested model with a prompt structure analogous to abstract screening but with stricter criteria, requiring extractable quantitative epidemiological parameters (e.g. transmission rates, incubation periods and severity outcomes) while excluding literature reviews, meta-analyses and case studies describing fewer than 10 infected individuals (Figure 1d, Appendix H).

Data Extraction We extract data for three categories (epidemiological parameters, transmission models and concluded outbreaks) using a multi-stage, schema-constrained framework (Figure 1e, Appendix I). Extraction assumes the use of agentic models and provides tools that enforce schematised and structured outputs, mimicking human annotators extracting relevant data and filling out survey

forms. For each data category, the pipeline first conducts presence flagging to identify articles containing relevant data, followed by targeted extraction using parameter-specific, model-specific, or outbreak-specific tool calls for validated outputs. For epidemiological parameters, extraction also involves population tagging (e.g. age groups, geographic locations and clinical severity), which enables subsequent aggregation of parameter estimates into summary statistics across population contexts.

Report Generation Extracted data are converted into a report using LLM self-refinement on top of standardised code producing SLR-like artefacts, including figures, tables and aggregate statistics (Figure 1f, Appendix J). Report generation sits outside the evaluation construct studied. Our metrics target evidence handling, whereas evaluating generated reports would require exact data-matched reference reports and a separate construct for narrative synthesis. LLM interpretation of aggregate public-health statistics also introduces safety risks beyond the scope of this study.

###### 3.2 Evaluation Protocol

Screening Metrics We evaluate screening as a binary article decision y ∈ {include,exclude} against PERG (human) labels, reporting precision, recall and macro F1. Consistent with PRISMA style study selection [12], AgentSLR supports the usual abstract to full-text funnel and two ablations: (a) AI abstract→AI full-text, (b) human abstract→AI full-text and (c) AI direct full-text.

Extraction metrics Structured extraction cannot be evaluated by a single exact match. An article may contain zero, one or many records of the same data type, and the records have no unique identifiers that tell us which model output corresponds to which expert annotation. A generated record can also be partly useful, it may identify the correct outbreak but miss a date, or recover the correct parameter value but attach incomplete population context. We therefore separate three constructs: whether relevant data are detected, whether the correct number of records is produced and whether the matched records contain the correct field values.

Flagging measures detection. For each ⟨article,data_type⟩ pair, we compare a PERG-derived presence label (at least one extraction) to the LLM’s flag and compute precision and recall over these binary labels. Count measures volume. If the reference contains n records of a data type and the LLM generates nˆ records, we assign partial credit as

TP = min(n,nˆ), FP = max(0,nˆ − n), FN = max(0,n − nˆ).

Thus, if the reference has two records and the LLM generates five, the count score gives two true positives and three false positives. It does not decide whether the two TP are field correct. Extraction measures field-level content. We first establish correspondences between reference and generated records within the same article by computing a pairwise similarity score. For a reference extraction E, generated extraction Eˆ and key fields we use

wkdk(E[k],Eˆ[k]), dk(v,vˆ) = |v ∩ vˆ| |v ∪ vˆ|

s(E,Eˆ) =

.

k∈F

where v = E[k] and vˆ = Eˆ[k] are the sets of values for field k. Here wk is a normalised field weight and dk is Jaccard similarity, with single-value fields treated as singleton sets. We use the modified Jonker-Volgenant assignment algorithm [27] on the cost matrix 1 − s to find the one-to-one matching that maximises total similarity. Field-level precision and recall are computed after this matching step. This design separates errors from over-extraction, missed records and inaccurate fields, which better reflects how expert reviewers inspect structured records. See Appendix C for details.

##### 4 Experiment Settings

###### 4.1 Dataset

The SLR workflow part of the harness (Section 3.1) allows for automatic and repeated retrieval of articles in real time. However, to evaluate LLM extractions we use human-reference labels gathered from SLRs conducted by epidemiologists at the Pathogen Epidemiology Review Group (PERG). PERG has undertaken an initiative to conduct SLRs for nine priority pathogens identified by the WHO

- as having high epidemic or pandemic potential [28]. The group has published five peer-reviewed SLRs [29–33] with two more in the data extraction phase (Table 1). We evaluated each workflow stage with all pathogen data available, so seven pathogens are evaluated for screening and four (Ebola, Lassa, SARS, and Zika) are evaluated for data extraction. Within data extraction, parameter and transmission model extraction are evaluated on all four pathogens, while outbreak extraction is evaluated on Lassa and Zika only, as PERG’s published SLRs for Ebola and SARS did not include outbreak extraction. After correspondence with PERG, we exclude Marburg due to inconsistencies in data format, and MERS and Nipah because PERG’s extraction phase is still in progress.

Table 1: Released dataset (benchmark) record counts. AgentSLR Matched denotes the downloaded article subset that matched PERG labelled records and was released on HuggingFace. The coloured symbols represent the progress of PERG’s SLRs per priority pathogen: published •, conducting data extraction •, and yet to begin screening •, as of March 2026.

Pathogen PERG* AgentSLR Matched

Marburg virus 2,593 801 (30.9%) Ebola virus 11,605 4,119 (35.5%) Lassa fever 2,131 667 (31.3%) SARS-CoV-1 12,280 2,047 (16.7%) Zika virus 10,510 2,164 (20.6%) MERS-CoV 19,656 5,714 (29.1%) Nipah virus 1,458 736 (50.5%)

•RVFCCHFvirusvirus –– ––

Total 60,233 16,248 (27.0%)

*Articles post deduplication and empty abstract removal.

The data labelled and extracted by humans for the peer-reviewed SLRs have been partially available as open-source resources through the epireview and priority-pathogen R packages [34, 35]. Through AgentSLR we release an extended version of this dataset on HuggingFace, with associated code on GitHub. Our released dataset contains 16,248 downloaded article records matched to PERG linked labels, based on article Covidence ID. It also contains 3,808 parameter extractions, 687 transmission model extractions and 189 outbreak extractions.2 For reproducibility, we evaluate the downloaded open-access subset retrieved through the bibliographic databases we queried (Appendix G). The harness can retrieve a larger article corpus, but only articles with accessible full text can be downloaded, processed and matched to PERG labels. This yields 16,248 matched records, corresponding to approximately 27.0% of the PERG corpus. To assess whether the open-access subset is representative of the broader PERG corpus, we manually retrieve 1,004 closed-access articles from the broader PERG corpus and run the AgentSLR harness on a matched comparison sample. We find broadly comparable performance across stages (see Appendix B).

4.2 Models

AgentSLR is compatible with both open and closed weight models, with tool calls and requests schematised through OpenAI’s Responses and Chat Completions APIs. Due to complex calculations required, especially during data extraction, we evaluate LLMs that can reason and scale

- at inference-time. We conduct comparative evaluations using OpenAI’s GPT-5.2 (closed-source) and gpt-oss-120b (open-source), Moonshot AI’s Kimi K2.5, Z.AI’s GLM-4.7 and DeepSeek’s DeepSeek-V3.2. Attempts to evaluate Claude Opus 4.5 and Sonnet 4.5 resulted in streaming refusals.3 All models have reasoning set to high where possible, with a maximum generation limit of 64K tokens per pass. Open-source models are hosted with vllm [36] on an NVIDIA H200 cluster node. For PDF-to-Markdown conversion, we use the mistral-ocr-2512 API endpoint [37], a state-of-the-art OCR model suited to scanned documents with complex mathematical and tabular content. For reproducibility, AgentSLR is also configured to run with open-weight OCR models. We find no statistically significant change in performance when running OCR (10% sample) with DeepSeek-OCR [38] or PaddleOCR [39].

##### 5 Results

###### 5.1 Scientific Synthesis Is Not a Single Capability

Evaluating five LLMs using the AgentSLR harness across pathogens to produce epidemiological SLRs, we find that no single model dominates the full synthesis workflow (Figure 2). At the article

- 2The data released includes metadata, URLs and structured annotations. The supplementary harness code can be used to download articles and conduct OCR, with instructions and single command scripts provided in the README.
- 3We experienced this refusal problem with all Claude models above version 4.0 (See Documentation from Anthropic). Potential causes and implications are discussed in Section 6.

GPT-OSS-120B GLM-4.7 DeepSeek-V3.2 Kimi-K2.5 GPT-5.2

- 0.5
- 1

|0.87|
|---|

|0.85|
|---|

| ||0.74|
|---|
<br><br>|0.72|
|---|
<br><br>|0.62|
|---|
<br><br>|0.77|
|---|
<br><br>|0.65|
|---|
|
|---|---|
| | |

|0.83|
|---|

|0.81|
|---|

|0.82|
|---|

|0.76|
|---|

|0.81|
|---|

|0.81|
|---|

|0.70|
|---|

|0.72|
|---|

|0.77|
|---|

|0.74|
|---|

|0.75|
|---|

|0.73|
|---|

|0.77|
|---|

|0.63|
|---|

|0.63|
|---|

|0.59|
|---|

|0.58|
|---|

|0.56|
|---|

FScore1

0

|Title & Abstract Screening|
|---|

|Full-text Screening|
|---|

|Parameter Extraction|
|---|

|Model Extraction|
|---|

|Outbreak Extraction|
|---|

- Figure 2: Stage-wise model performance on AgentSLR harness. Each model is run on individual pathogens with AgentSLR harness. Averages are computed over the pathogens evaluated at each stage, following reference data availability described in Section 4.1. Error bars indicate one standard deviation across pathogens. No single model dominates across all stages: Kimi-K2.5 and gpt-oss-120b lead screening, while extraction leaders vary by data type. Full metrics are provided in Appendix E.

screening stages, Kimi-K2.5 and gpt-oss-120b perform the best, with the former leading title and abstract screening (F1 = 0.77) and the latter full-text screening (F1 = 0.87). All models struggle with parameter extraction, where Kimi-K2.5 again achieves the highest performance (F1 = 0.63). GLM-4.7 performs best on model extraction, while GPT-5.2 leads on outbreak extraction. DeepSeek-V3.2 exhibits the most variable performance, ranking last in article screening but becoming competitive in extraction, where function calling is enabled. Across all extraction stages, the gap between the best and worst model is considerably narrower than at screening. These disparities suggest scientific synthesis comprises separable sub-tasks, each posing distinct challenges.

###### 5.2 Structured Evidence Extraction Is the Bottleneck Across Models

Table 2 isolates the type and sub-task of the structured data extracted. Flagging the presence of a data type is the most reliable construct on average, with mean F1 ranging from 0.72 to 0.80 across models, while field-level extraction of evidence remains lower and shows less variance, ranging from 0.61 to 0.67. This bottleneck is particularly evident for epidemiological parameter values, where no model reaches F1 = 0.60 for parameter extraction, despite stronger parameter Flagging for Kimi-K2.5 and GLM-4.7 (0.72). Transmission model flagging is consistently high (0.87 to 0.93), yet field-level model extraction remains bounded at 0.65 to 0.68, indicating that recognising a relevant article is substantially easier than producing the correct structured record. Outbreak extraction is the main exception, with field-level scores reaching 0.84 for GPT-5.2, but this result is evaluated only on Lassa and Zika and shows high variance in Count performance. With no LLM exceeding an average extraction(•)F1 of0.67,thelimitingfactorextendsbeyondmodelchoicetoafundamentalcapability gap in grounding heterogeneous scientific reporting conventions onto a single validated schema.

- Table 2: Structured evidence extraction performance. F1 scores are averaged across pathogens for

five frontier models and three extraction tasks: • Flagging, • Counts, and • field-level Extraction.

Standard deviations (±) are reported across pathogens.

Parameters Models Outbreaks Average

#### Model • • • • • • • • • • • •

0.66

0.59

0.54

0.91

0.68

0.67

0.61

0.69

0.79

0.75

0.65

0.64

gpt-oss-120b

±0.07

±0.08

±0.03

±0.05

±0.05

±0.03

±0.13

±0.31

±0.02

±0.15

±0.13

±0.10

0.66

0.50

0.59

0.90

0.72

0.67

0.66

0.80

0.84

0.76

0.65

0.67

GPT-5.2

±0.07

±0.07

±0.03

±0.05

±0.04

±0.03

±0.12

±0.05

±0.04

±0.14

±0.14

±0.10

0.60

0.56

0.50

0.87

0.92

0.65

0.65

0.78

0.75

0.72

0.75

0.61

DeepSeek-V3.2

±0.07

±0.04

±0.03

±0.04

±0.08

±0.09

±0.04

±0.19

±0.02

±0.14

±0.19

±0.11

0.72

0.62

0.56

0.92

0.81

0.68

0.64

0.87

0.78

0.78

0.75

0.65

Kimi-K2.5

±0.09

±0.05

±0.02

±0.04

±0.04

±0.04

±0.07

±0.11

±0.07

±0.14

±0.12

±0.10

0.72

0.61

0.54

0.93

0.93

0.68

0.68

0.72

0.77

0.80

0.76

0.64

GLM-4.7

±0.09

±0.05

±0.02

±0.07

±0.06

±0.05

±0.03

±0.29

±0.02

±0.13

±0.18

±0.10

AI Screen (Abstract) → AI Screen (Full-text) AI Screen (Direct Full-text) Human Screen (Abstract) → AI Screen (Full-text)

|0.97|
|---|

|0.96|
|---|

|0.95|
|---|

|0.95|
|---|

|0.94|
|---|

|0.93|
|---|

|0.92|
|---|

|0.91|
|---|

|0.91|
|---|

|0.91|
|---|

|0.90|
|---|

|0.89|
|---|

- 0.5
- 1

|0.88|
|---|

|0.85|
|---|

|0.85|
|---|

|0.84|
|---|

|0.84|
|---|

| ||0.76|
|---|
<br><br>|0.82|
|---|
<br><br>|0.83|
|---|
<br><br>|0.78|
|---|
<br><br>|0.79|
|---|
<br><br>|0.83|
|---|
<br><br>|0.81|
|---|
|
|---|---|
| | |

Recall

0

Marburg Ebola Lassa SARS Zika MERS Nipah Overall

- Figure 3: Recall of article screening strategies across pathogens. Two ablation strategies (direct full-text and human-conditioned screening) with AgentSLR (GPT-OSS-120B) offer better recall (or ‘fetch rate’) than standard AI-based two-stage screening, with confidence intervals overlapping across most pathogens. Full article screening metrics results are reported in Appendix E.3.

###### 5.3 Screening Strategy Controls What Evidence Survives

Using gpt-oss-120b, we evaluate three full-text screening strategies to quantify how review (and workflow) design affects evidence survival (Figure 3). The fully automated two-stage strategy (mimicking humans), where abstract screening filters articles before full-text screening, achieves recall 0.81 and F1 = 0.77 against PERG decisions. Conditioning full-text screening on human abstract decisions raises recall to 0.92 and gives the strongest overall classification performance (F1 = 0.87). This shows how early human triage can protect relevant studies from upstream model errors. Direct AI full-text screening without abstract gating also improves recall to 0.89, but lowers F1 to 0.73. This configuration avoids abstract information loss without human input, but it increases screening runtime by 2.3× (9.55 versus 4.16 hours) and raises OCR costs from USD 36.6 to USD 303.2. Screening strategy therefore controls the false negative risk and operating cost of the evaluation beyond the aggregate score.

###### 5.4 Expert Review Separates Utility from Autonomy

We validated gpt-oss-120b extraction quality across all three data types using a survey completed by six expert epidemiologists (Table 3). Each expert reviewed a random subset of system outputs alongside the corresponding articles (details in Appendix F). This validation audits output quality rather than re-annotating the PERG reference set. Expert-rated flagging precision is highest for parameters (0.66) and outbreaks (0.61), but substantially lower for models (0.40), indicating that a majority of model extractions are flagged by the system, but deemed irrelevant by experts. Field-level extraction accuracy is more consistent across types (0.77–0.83), suggesting that when an extraction is correctly flagged, populated fields are largely accurate. The competence ratings make the deployment boundary clear: parameters score 4.23 and outbreaks 3.90 on the 1–7 scale, close to the survey threshold for a tool usable under moderate supervision, while models score only 2.80. Qualitative feedback from epidemiologists is consistent with this split. Recurring failure modes include limited use of document structure, failure to distinguish newly reported findings from cited prior work, and insufficient contextual reasoning for fields that require holistic inference. The system can reduce human workload by providing a correctable starting point, and the false positives are typically easy to identify, but this warrants human review rather than full autonomous use.

- Table 3: Expert-rated holistic evaluation of data extraction quality. We report expert-rated flagging precision, field-level extraction accuracy, and perceived competence (AgentSLR with gpt-oss-120b) for parameter, model, and outbreak extractions, aggregated across six epidemiologists. Values show mean ± standard error. Competence is rated on a 1–7 Likert scale (1: Incompetent, 7: Autonomous).

Data Type Flagging Precision Extraction Accuracy Competence (1–7)

Parameters 0.66 ± 0.06 0.77 ± 0.05 4.23 ± 0.20 Models 0.40 ± 0.07 0.83 ± 0.05 2.80 ± 0.32 Outbreaks 0.61 ± 0.09 0.80 ± 0.07 3.90 ± 0.45

- 5.5 Performance Does Not Scale Reliably with Cost

$73.6 $1348.2

$277.2 $810.8

$10 $20 $50 $100 $200 $500 $1k

0.6

0.7

0.8

Total Cost per Pathogen run (log10 USD)

AveragePerformance(FScore)1

$13.9

0.69 ±0.09

0.70 ±0.07

0.73

±0.09 0.67

±0.11

0.74 ±0.07

GPT-OSS-120B DeepSeek-V3.2 Kimi-K2.5 GLM-4.7 GPT-5.2

Figure 4: Cost vs. average performance per AgentSLR pathogen run. Each point shows a model’s mean macro F1 across all pipeline stages against estimated total cost per run (USD, log10 scale), with error bars denoting one standard deviation across stages. Full cost and token breakdowns are provided in Appendix D.2.

Running the SLR workflow at scale requires models to process a large and narrowing evidence funnel. Across pathogens on average 9,132 articles reach title and abstract screening, 1,102 reach full-text screening, and 395 reach data extraction. At this scale, model choice affects both evidence quality and operating cost, but higher cost does not reliably predict higher performance (Figure 4). gpt-oss-120b reaches competitive average performance (F1 = 0.70) at the lowest estimated cost per pathogen run (USD 13.9), while GPT-5.2 costs over 96× more (USD 1,348.2) and has slightly lower performance (F1 = 0.69). The best-performing model overall, Kimi-K2.5 (F1 = 0.74), falls in the mid-cost range ($277), while GLM-4.7 incurs the secondhighest cost ($811) for a comparable F1 of 0.73. DeepSeek-V3.2 is cheaper than Kimi-K2.5 at USD 73.6, but has lower average performance (F1 = 0.67). The cost differences are driven by per article token use during parameter extraction, where GPT-5.2 produces 91.1K output tokens per article4 compared with 3.0K for DeepSeek-V3.2. Cost and quality therefore need to be evaluated at the subtask level before selecting a model for SLR deployment.

- 6 Discussion

- 6.1 Key Findings

Evaluating five frontier LLMs using AgentSLR reveals that epidemiological SLRs expose scientific synthesis as a set of separable and unsaturated capabilities. No evaluated model dominates across the workflow. Across all five LLMs, no model exceeds an average field-level extraction F1 of 0.67. Critically, this performance ceiling is not attributable to model choice alone. The performance spread between models narrows markedly at extraction relative to screening. This convergence under tool-calling highlights the difficulty of schema-grounded extraction from heterogeneous scientific reporting, where the same quantities are presented through different conventions and must be normalised across study designs rather than simply identified.

- 6.2 Deployment Risk

Deployment of LLMs in public health research workflows faces two distinct risk categories, namely infrastructure risk and error structure risk. Infrastructure risk encompasses model availability, content restrictions, version instability and operating cost. This risk applies primarily to closed-source models. Attempts to evaluate Claude Opus 4.5 and Sonnet 4.5 consistently produced streaming refusals, which we attribute to content filters on epidemiological terminology as these may indicate dangerous intent.5 When applied too broadly, such restrictions render entire model families unavailable for legitimate public-health research and reinforce the case for open-weight alternatives. We find that gpt-oss-120b reaches competitive average performance at over 96× lower cost than GPT-5.2. Such open-source models allow for version pinning and local hosting, both scientific requirements for long-running living reviews where reproducibility matters. At the error structure level, our stage-isolated evaluation shows why average scores, or replications of papers (like PaperBench [40]), are insufficient in high-stakes SLRs. False negatives in screening remove evidence permanently, provenance and context failures can attach correct values to the wrong claim. Similarly, pathogenspecific performance gaps, e.g. when LLMs struggle on Zika extraction, signal systematic bias that can propagate through any synthesis drawing on AI-assisted extraction.

4While maximum generation length is capped at 64K, a model can have multiple extractions per article 5See the Anthropic documentation on Sonnet 4.5 API safety filters.

###### 6.3 Assisted Evidence Synthesis

Title & Abstract Screening

Full-text Screening

AgentSLR substantially reduces the cost of running review stages compared with humans (Figure 5), but these numbers should be interpreted as conditional efficiency rather than evidence of expert replacement. With gpt-oss-120b, the per-pathogen workflow processes an average funnel of 9,132 articles at title and abstract screening, 1,102 at full-text screening and 395 at data extraction in 20 hours. The corresponding human estimate is 385 labour hours, giving a 19.3× efficiency gain. As the system runs continuously, this is also a reduction from 48.1 eight-hour workdays to 0.83 calendar days, or 58× fewer calendar days. Full-text screening is the clearest example, taking under two seconds per article compared with a four minute human estimate, or 118× faster. These runtime figures are reported separately from quality metrics because speed is only meaningful once a stage meets an acceptable quality threshold. Therefore, LLMs are not at the capability threshold to replace expert judgment, but they can substantially reduce the time required for evidence triage.

PDF-to-MD Conversion*

Data Extraction

| |AgentSLR Magnified<br><br>10 8 6 4 2 0<br><br>| | |
|---|---|
| | |
| | |
| | |
| | |
<br><br>| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

350

300

250

Time(hours)

200

150

100

LR ied

50

0

Humans AgentSLR

Figure 5: SLR completion time (Human vs. AgentSLR). Using gpt-oss-120b across the full SLR creation pipeline takes 20 hours versus 385 hours for manual review, a 19.3× wall-clock speed-up and 58× reduction in calendar days. Full runtime statistics are reported in Appendix D.

The clearest short-term role for LLMs is to accelerate human-led review at the points where missed evidence is most costly. This is especially true for full-text screening after human abstract triage, where manual review limits SLR scalability, and for direct full-text screening when abstracts do not fully report relevance. In these settings, false inclusions can be removed during validation, while false exclusions are much harder to recover. Expert validation extends this picture to data extraction, where parameter and outbreak competence sit near the threshold for use under moderate supervision and experts consistently report that outputs reduce workload by providing a correctable starting point. Applied to living systematic reviews, where the cost of each update cycle is the primary barrier to continuous monitoring, this combination can make routine updates feasible in priority pathogen surveillance [41].

###### 6.4 Limitations and Future Work

We identify key limitations of our evaluation study. Our evaluation is restricted to articles available through open access routes in the queried sources, covering approximately 27.0% of the PERG corpus. However, a manually retrieved closed-access subset yields consistent results. The workflow follows English-only screening criteria, and misses multilingual evidence. Our metrics prioritise evidence survival, which is appropriate for high-stakes screening but this can shift some burden to downstream filtering when precision is low. Finally, we do not evaluate meta-analysis or final review writing, where statistical modelling and narrative claims would require separate expert-grounded tests. Future work should measure human uplift directly, develop the annotation interface described in Appendix M into a production tool, extend the evaluation to other scientific domains and study heterogeneous model routing across tasks where different LLMs show complementary strengths.

##### 7 Conclusion

We introduce AgentSLR as an evaluation harness to study LLM-assisted systematic literature review in infectious disease epidemiology. Epidemiological SLRs provide an expert-grounded, high-stakes setting for evaluating AI scientific knowledge synthesis. The released dataset and harness decompose review production into retrieval, screening and schema-grounded extraction, enabling stage-level diagnosis against human reference annotations rather than aggregate claims of automation. Across current LLMs, we find clear assistive value, especially for screening and correctable preliminary extraction, but also persistent limitations in structured evidence extraction, error structure, cost and model availability. These results argue for human-supervised deployment in epidemiology, and for treating SLRs as a demanding evaluation setting for measuring whether future AI systems can synthesise scientific evidence reliably.

##### Acknowledgement

S.P. and A.M. are supported in part by the Engineering and Physical Sciences Research Council (EPSRC) under Grant EP/X028909/1 and Oxford Internet Institute’s Research Programme funded by the Dieter Schwarz Foundation. R.O.K. is supported by the Clarendon Scholarship and the Jesus College Old Members’ Scholarship. E.S. acknowledges support in part from the AI2050 program at Schmidt Sciences (Grant No. G-22-64476). E.S. and A.C. acknowledge that this study is funded by the National Institute for Health Research (NIHR) Health Protection Research Unit in Health Analytics & Modelling (NIHR207404), a partnership between UK Health Security Agency (UKHSA), London School of Hygiene & Tropical Medicine, and Imperial College of Science, Technology, & Medicine. The views expressed are those of the author(s) and not necessarily those of the NIHR, UKHSA, or the Department of Health and Social Care. The authors thank Saverio Trioni for helpful conversations. The authors would like to thank the Pathogen Epidemiology Review Group (PERG), School of Public Health, Imperial College London, for their support and eagerness to contribute to this project.

##### References

- [1] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A graduate-level Google-proof Q&A benchmark. In First Conference on Language Modeling, 2024.
- [2] Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, and Lawrence Chan. Measuring AI ability to complete long tasks. CoRR, abs/2503.14499, 2025.
- [3] Jonathan Roberts, Kai Han, Neil Houlsby, and Samuel Albanie. SciFIBench: Benchmarking large multimodal models for scientific figure interpretation. Advances in Neural Information Processing Systems, 37:18695–18728, 2024.
- [4] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, Anima Anandkumar, Karianne Bergen, Carla P. Gomes, Shirley Ho, Pushmeet Kohli, Joan Lasenby, Jure Leskovec, Tie-Yan Liu, Arjun Manrai, Debora Marks, Bharath Ramsundar, Le Song, Jimeng Sun, Jian Tang, Petar Veliˇckovi´c, Max Welling, Linfeng Zhang, Connor W. Coley, Yoshua Bengio, and Marinka Zitnik. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.
- [5] Yanbo Zhang, Sumeer A Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, et al. Exploring the role of large language models in the scientific method: from hypothesis to discovery. npj Artificial Intelligence, 1(1):14, 2025.
- [6] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.
- [7] Takehiko Oami, Yohei Okada, and Taka-aki Nakada. Performance of a large language model in screening citations. JAMA Network Open, 7(7):e2420496–e2420496, 2024.
- [8] Uwe Peters and Benjamin Chin-Yee. Generalization bias in large language model summarization of scientific research. Royal Society Open Science, 12(4):241776, 2025.
- [9] Melissa Z Pan, Mert Cemri, Lakshya A Agrawal, Shuyi Yang, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Kannan Ramchandran, Dan Klein, et al. Why do multiagent systems fail? In ICLR 2025 Workshop on Building Trust in Language Models and Applications, 2025.
- [10] Wenqing He, Grace Y Yi, and Yayuan Zhu. Estimation of the basic reproduction number, average incubation time, asymptomatic infection rate, and case fatality rate for COVID-19: Meta-analysis and sensitivity analysis. Journal of Medical Virology, 92(11):2543–2550, 2020.

- [11] Jack Ward, Oswaldo Gressani, Sol Kim, Niel Hens, and W. John Edmunds. The epidemiology of pathogens with pandemic potential: A review of key parameters and clustering analysis. Epidemics, 54:100882, 2026.
- [12] Matthew J Page, Joanne E McKenzie, Patrick M Bossuyt, Isabelle Boutron, Tammy C Hoffmann, Cynthia D Mulrow, Larissa Shamseer, Jennifer M Tetzlaff, Elie A Akl, Sue E Brennan, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ, 372, 2021.
- [13] Iain J Marshall and Byron C Wallace. Toward systematic review automation: a practical guide to using machine learning tools in research synthesis. Systematic Reviews, 8(1):163, 2019.
- [14] Christian Cao, Jason Sang, Rohit Arora, David Chen, Robert Kloosterman, Matthew Cecere, Jaswanth Gorla, Richard Saleh, Ian Drennan, Bijan Teja, et al. Development of prompt templates for large language model–driven screening in systematic reviews. Annals of Internal Medicine, 178(3):389–401, 2025.
- [15] Ava Homiar, James Thomas, Edoardo G Ostinelli, Jaycee Kennett, Claire Friedrich, Pim Cuijpers, Mathias Harrer, Stefan Leucht, Clara Miguel, Alessandro Rodolico, et al. Development and evaluation of prompts for a large language model to screen titles and abstracts in a living systematic review. BMJ Mental Health, 28(1), 2025.
- [16] Gerald Gartlehner, Leila Kahwati, Rainer Hilscher, Ian Thomas, Shannon Kugley, Karen Crotty, Meera Viswanathan, Barbara Nussbaumer-Streit, Graham Booth, Nathaniel Erskine, Amanda Konet, and Robert Chew. Data extraction for evidence synthesis using a large language model: A proof-of-concept study. Research Synthesis Methods, 15(4):576–589, 2024.
- [17] Hesam Mahmoudi, Doris Chang, Hannah Lee, Navid Ghaffarzadegan, and Mohammad S Jalali. Critical assessment of large language models’(ChatGPT) performance in data extraction for systematic reviews: Exploratory study. JMIR AI, 4(1):e68097, 2025.
- [18] Honghao Lai, Jiayi Liu, Chunyang Bai, Hui Liu, Bei Pan, Xufei Luo, Liangying Hou, Weilong Zhao, Danni Xia, Jinhui Tian, et al. Language models for data extraction and risk of bias assessment in complementary medicine. npj Digital Medicine, 8(1):74, 2025.
- [19] Dmitry Scherbakov, Nina Hubig, Vinita Jansari, Alexander Bakumenko, and Leslie A Lenert. The emergence of large language models as tools in literature reviews: a large language modelassisted systematic review. Journal of the American Medical Informatics Association, 32(6): 1071–1086, 2025.
- [20] Christian Cao, Rohit Arora, Paul Cento, Katherine Manta, Elina Farahani, Matthew Cecere, Anabel Selemon, Jason Sang, Ling Xi Gong, Robert Kloosterman, et al. Automation of systematic reviews with large language models. medRxiv, pages 2025–06, 2025.
- [21] Rachel H Parkinson, Henry Cerbone, Mikael Mieskolainen, Shuxiang Cao, Alasdair D Wilson, Sergio Albacete, Emily B Armstrong, Chris Bass, Cristina Botías, Andrew Brown, et al. Metabeeai: an AI pipeline for full-text systematic reviews in biology. bioRxiv, pages 2025–11, 2025.
- [22] Kyeryoung Lee, Hunki Paek, Nneka Ofoegbu, Steven Rube, Mitchell K. Higashi, Dalia Dawoud, Hua Xu, Lizheng Shi, and Xiaoyan Wang. A4SLR: An agentic artificial intelligence-assisted systematic literature review framework to augment evidence synthesis for health economics and outcomes research and health technology assessment. Value in Health, 28(11):1655–1664, 2025.
- [23] Zifeng Wang, Lang Cao, Benjamin Danek, Qiao Jin, Zhiyong Lu, and Jimeng Sun. Accelerating clinical evidence synthesis with large language models. npj Digital Medicine, 8(1):509, 2025.
- [24] Lech Madeyski, Barbara Kitchenham, and Martin Shepperd. LLM4SCREENLIT: Recommendations on assessing the performance of large language models for screening literature in systematic reviews, 2026.

- [25] Jianyou Wang, Weili Cao, Kaicheng Wang, Xiaoyue Wang, Ashish Dalvi, Gino Prasad, Qishan Liang, Hsuan lin Her, Ming Wang, Qin Yang, Gene W. Yeo, David E. Neal, Maxim Khan, Christopher D. Rosin, Ramamohan Paturi, and Leon Bergen. EvidenceBench: A benchmark for extracting evidence from biomedical papers, 2025.
- [26] Christopher Polzak, Alejandro Lozano, Min Woo Sun, James Burgess, Yuhui Zhang, Kevin Wu, and Serena Yeung-Levy. Can large language models match the conclusions of systematic reviews?, 2025.
- [27] Roy Jonker and Anton Volgenant. A shortest augmenting path algorithm for dense and sparse linear assignment problems. Computing, 38(4):325–340, 1987.
- [28] World Health Organization. Pathogens prioritization: A scientific framework for epidemic and pandemic research preparedness. Technical report, World Health Organization, 2024.
- [29] Gina Cuomo-Dannenburg, Kelly McCain, Ruth McCabe, H Juliette T Unwin, Patrick Doohan, Rebecca K Nash, Joseph T Hicks, Kelly Charniga, Cyril Geismar, Ben Lambert, Dariya Nikitin, Janetta Skarp, Jack Wardle, Mara Kont, Sangeeta Bhatia, Natsuko Imai, Sabine van Elsland, Anne Cori, Christian Morgenstern, Aaron Morris, Alpha Forna, Amy Dighe, Anne Cori, Arran Hamlet, Ben Lambert, Charlie Whittaker, Christian Morgenstern, Cyril Geismar, Dariya Nikitin, David Jorgensen, Ed Knock, Ettie Unwin, Gina Cuomo-Dannenburg, Hayley Thompson, Isobel Routledge, Janetta Skarp, Joseph Hicks, Keith Fraser, Kelly Charniga, Kelly McCain, Lily Geidelberg, Lorenzo Cattarino, Mara Kont, Marc Baguelin, Natsuko Imai, Nima Moghaddas, Patrick Doohan, Rebecca Nash, Ruth McCabe, Sabine van Elsland, Sangeeta Bhatia, Sreejith Radhakrishnan, Zulma Cucunuba Perez, and Jack Wardle. Marburg virus disease outbreaks, mathematical models, and disease parameters: A systematic review. The Lancet Infectious Diseases, 24(5):e307–e317, 2024.
- [30] Patrick Doohan, David Jorgensen, Tristan M Naidoo, Kelly McCain, Joseph T Hicks, Ruth McCabe, Sangeeta Bhatia, Kelly Charniga, Gina Cuomo-Dannenburg, Arran Hamlet, Rebecca K Nash, Dariya Nikitin, Thomas Rawson, Richard J Sheppard, H Juliette T Unwin, Sabine van Elsland, Anne Cori, Christian Morgenstern, Natsuko Imai-Eaton, Aaron Morris, Alpha Forna, Amy Dighe, Anna Vicco, Anna-Maria Hartner, Anne Cori, Arran Hamlet, Ben Lambert, Bethan Cracknell Daniels, Charlie Whittaker, Christian Morgenstern, Cosmo Santoni, Cyril Geismar, Dariya Nikitin, David Jorgensen, Dominic Dee, Ed Knock, Ettie Unwin, Gina CuomoDannenburg, Hayley Thompson, Ilaria Dorigatti, Isobel Routledge, Jack Wardle, Janetta Skarp, Joseph Hicks, Kanchan Parchani, Keith Fraser, Kelly Charniga, Kelly McCain, Kieran Drake, Lily Geidelberg, Lorenzo Cattarino, Mantra Kusumgar, Mara Kont, Marc Baguelin, Natsuko Imai-Eaton, Pablo Perez Guzman, Patrick Doohan, Paul Lietar, Paula Christen, Rebecca Nash, Rich Fitzjohn, Richard Sheppard, Rob Johnson, Ruth McCabe, Sabine van Elsland, Sangeeta Bhatia, Sequoia Leuba, Shazia Ruybal-Pesantez, Sreejith Radhakrishnan, Thomas Rawson, Tristan Naidoo, and Zulma Cucunuba Perez. Lassa fever outbreaks, mathematical models, and disease parameters: A systematic review and meta-analysis. The Lancet Global Health, 12(12): e1962–e1972, 2024.
- [31] Rebecca K Nash, Sangeeta Bhatia, Christian Morgenstern, Patrick Doohan, David Jorgensen, Kelly McCain, Ruth McCabe, Dariya Nikitin, Alpha Forna, Gina Cuomo-Dannenburg, Joseph T Hicks, Richard J Sheppard, Tristan Naidoo, Sabine van Elsland, Cyril Geismar, Thomas Rawson, Sequoia Iris Leuba, Jack Wardle, Isobel Routledge, Keith Fraser, Natsuko Imai-Eaton, Anne Cori, and H Juliette T Unwin. Ebola virus disease mathematical models and epidemiological parameters: A systematic review. The Lancet Infectious Diseases, 24(12):e762–e773, 2024.
- [32] Christian Morgenstern, Thomas Rawson, Isobel Routledge, Mara Kont, Natsuko Imai-Eaton, Janetta Skarp, Patrick Doohan, Kelly McCain, Rob Johnson, H Juliette T Unwin, Tristan Naidoo, Dominic P Dee, Kanchan Parchani, Bethan N Cracknell Daniels, Anna Vicco, Kieran O Drake, Paula Christen, Richard J Sheppard, Sequoia I Leuba, Joseph T Hicks, Ruth McCabe, Rebecca K Nash, Cosmo N Santoni, Gina Cuomo-Dannenburg, Sabine van Elsland, Sangeeta Bhatia, Anne Cori, Aaron Morris, Alpha Forna, Amy Dighe, Anna Vicco, Anna-Maria Hartner, Anne Cori, Arran Hamlet, Ben Lambert, Bethan Cracknell Daniels, Charlie Whittaker, Christian Morgenstern, Cosmo Santoni, Cyril Geismar, Dariya Nikitin, David Jorgensen, Dominic Dee, Ed Knock, Ettie Unwin, Gina Cuomo-Dannenburg, Hayley Thompson, Ilaria Dorigatti, Isobel

- Routledge, Jack Wardle, Janetta Skarp, Joseph Hicks, Kanchan Parchani, Keith Fraser, Kelly Charniga, Kelly McCain, Kieran Drake, Lily Geidelberg, Lorenzo Cattarino, Mantra Kusumgar, Mara Kont, Marc Baguelin, Natsuko Imai-Eaton, Pablo Perez Guzman, Patrick Doohan, Paul Lietar, Paula Christen, Rebecca Nash, Rich Fitzjohn, Richard Sheppard, Rob Johnson, Ruth McCabe, Sabine van Elsland, Sangeeta Bhatia, Sequoia Leuba, Shazia Ruybal-Pesantez, Sreejith Radhakrishnan, Thomas Rawson, Tristan Naidoo, and Zulma Cucunuba Perez. Severe acute respiratory syndrome (SARS) mathematical models and disease parameters: A systematic review. The Lancet Microbe, 6(9), 2025.
- [33] Kelly McCain, Anna Vicco, Christian Morgenstern, Thomas Rawson, Tristan M Naidoo, Sangeeta Bhatia, Dominic P Dee, Patrick Doohan, Keith Fraser, Anna-Maria Hartner, et al. A systematic review and meta-analysis of Zika virus epidemiology. Nature Health, pages 1–13, 2026.
- [34] Tristan Naidoo, Rebecca Nash, Christian Morgenstern, Patrick Doohan, Ruth McCabe, Joshua Lambert, Richard Sheppard, Cosmo Santoni, Thomas Rawson, Shazia Ruybal-Pesántez, Juliette H Unwin, Gina Cuomo-Dannenburg, Kelly McCain, Joseph Hicks, Anne Cori, and Sangeeta Bhatia. Epireview: Tools to Update and Summarise the Latest Pathogen Data from the Pathogen Epidemiology Review Group (PERG), 2025.
- [35] Rebecca Nash, Christian Morgenstern, Sangeeta Bhatia, Richard Sheppard, Joseph Hicks, Gina Cuomo-Dannenburg, Ruth McCabe, Kelly McCain, Anna Vicco, Patrick Doohan, and Tristan Naidoo. Priority-Pathogens, 2026.
- [36] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [37] Mistral AI. Mistral OCR 3, 2025.
- [38] Haoran Wei, Yaofeng Sun, and Yukun Li. DeepSeek-OCR: Contexts optical compression, 2025.
- [39] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025.
- [40] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, et al. PaperBench: Evaluating AI’s ability to replicate AI research. arXiv preprint arXiv:2504.01848, 2025.
- [41] Carl T Bergstrom and Kevin Gross. Screening, sorting, and the feedback cycles that imperil peer review. PLoS biology, 24(2):e3003650, 2026.
- [42] Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Zaharia, and Carlos Guestrin. Deepscholar-bench: A live benchmark and automated evaluation for generative research synthesis, 2026.
- [43] Abdullah Mushtaq, Muhammad Rafay Naeem, Ibrahim Ghaznavi, Alaa Abd-alrazaq, Aliya Tabassum, and Junaid Qadir. Can agents judge systematic reviews like humans? evaluating slrs with llm-based multi-agent system, 2025.
- [44] Miguel Zabaleta and Baihan Lin. Scilitbench: Benchmark and design principles for llm-powered systematic literature reviews.
- [45] Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan, Chris Schmitz, Karolina Korgul, Hunar Batra, Oishi Deb, Emma Beharry, Cornelius Emde, Thomas Foster, Anna Gausen, María Grandury, Simeng Han, Valentin Hofmann, Lujain Ibrahim, Hazel Kim, Hannah Rose Kirk, Fangru Lin, Gabrielle Kaili-May Liu, Lennart Luettgau, Jabez Magomere, Jonathan Rystrøm, Anna Sotnikova, Yushi Yang, Yilun Zhao, Adel Bibi, Antoine Bosselut, Ronald Clark, Arman Cohan, Jakob Foerster, Yarin Gal, Scott A. Hale, Inioluwa Deborah Raji, Christopher Summerfield, Philip H. S. Torr,

- Cozmin Ududec, Luc Rocher, and Adam Mahdi. Measuring what matters: Construct validity in large language model benchmarks. The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.
- [46] Bushra Alhetelah and Irfan Ahmad. Measuring LLMs’ sensitivity to paraphrased opinion prompts. Association for Computational Linguistics.

# Appendix

- A Extended Related Work 16

- A.1 SLR Automation and Workflows . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 Comparison with Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- B Data Representativeness & Ecological Validity 18
- C Evaluation Constructs 19

- C.1 Article Screening . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Data Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 Human Expert Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- D Pipeline Statistics: Data Processed & Time 25

- D.1 Runtime Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.2 Token Usage and Operational Cost of AgentSLR harness . . . . . . . . . . . . . . 26

- E Extended Evaluation Results 28

- E.1 Results Across Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.2 Field-Level Results Possible with AgentSLR . . . . . . . . . . . . . . . . . . . . 31
- E.3 Article Screening Strategy Ablations . . . . . . . . . . . . . . . . . . . . . . . . . 35

- F Extended Expert Validation Results 36
- G Article Search and Retrieval 38

- G.1 Base Search Query (PubMed and Europe PMC) . . . . . . . . . . . . . . . . . . . 38
- G.2 OpenAlex Adapted Queries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- G.3 Pathogen-Specific Query Modifications . . . . . . . . . . . . . . . . . . . . . . . 39
- G.4 Metadata Extraction and Deduplication . . . . . . . . . . . . . . . . . . . . . . . 40
- G.5 PDF Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- G.6 Final Quality Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- H Article Screening Criteria and Prompts 42
- I Data Extraction Process 45

- I.1 Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- I.2 Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
- I.3 Outbreaks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60

- J Report Generation: Building Systematic Living Reviews 66

- J.1 Deterministic Report Assembly . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
- J.2 Evidence grounded narrative refinement . . . . . . . . . . . . . . . . . . . . . . . 67
- J.3 Report Generation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67

- K Living Systematic Reviews with AgentSLR 72
- L The PERG Review Pipeline (Human Reference Workflow) 74
- M AgentSLR Annotation Tool (Beta) 76

- M.1 Human-in-the-Loop Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
- M.2 Current Status and Field Testing . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
- M.3 Transparency and Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . 77

##### A Extended Related Work

We structure the related work for our evaluation harness into two parts. First is the workflow, and the attempt at automating SLRs for a particular domain, the need and challenge in doing that. Second, is benchmark and evaluation studies within scientific evidence synthesis and SLRs in particular.

###### A.1 SLR Automation and Workflows

Prior LLM-based SLR work has addressed individual pipeline stages in isolation [14, 16], and more recent systems have demonstrated feasibility in clinical and biological domains [20–22]. Each existing system is bespoke to its target domain, uses incompatible reference annotation schemas, and cannot be transferred to epidemiological parameter extraction. Direct quantitative comparison across systems is therefore methodologically ill-posed. Table 4 provides a structured qualitative comparison across domain, methodology, and evaluation rigour.

Table 4: Comparison against related work. LLM-based SLR pipelines differ substantially in domain, extraction methodology, and evaluation rigour, making direct quantitative comparison methodologically ill-posed. ✓ and × denote presence and absence of each property. Open code: source code is publicly released. Open weight: compatible with open-weight models. Human reference: evaluation uses expert-curated annotations as reference labels (proxy). Independent evaluation: performance is not assessed using LLM-as-a-judge. Stage evaluation: evaluation is decomposed into stage-specific metrics enabling failure attribution. AgentSLR workflow is the only framework satisfying all five properties.

otto-SR [20] A4SLR [22] MetaBeeAI [21] AgentSLR (ours) Domain Clinical evidence Health economics Biology Epidemiology Methodological focus

Cochrane-style intervention reviews

HEOR and HTA use cases

WHO-priority pathogen epidemiology

Bee ecotoxicology

LLM extraction with post-hoc human correction

Prompt-based with domain templates

Multi-pass chunk retrieval

Tool-calling with JSON schema constraints

Extraction method

Open code × × ✓ ✓ Open weight × × × ✓ Human reference ✓ ✓ ✓ ✓ Independent evaluation × ✓ ✓ ✓ Stage evaluation × ✓ × ✓

AgentSLR workflow is distinguished on three axes within LLM-assisted automation of SLRs that no existing pipeline jointly satisfies. It is the first system to target epidemiological SLRs on WHOpriority pathogens, the first pipeline compatible with open-weight models with disclosed source code, and the first to report stage-isolated evaluation against expert-curated annotations without LLM-as-ajudge. Each property has direct operational consequences. Open-weight compatibility enables version pinning and local deployment, which are requirements for living reviews where reproducibility is a scientific constraint. As we observe in the main text, refusal to process prompts can render entire closed-source model families unavailable for legitimate public-health research. Finally, stage-isolated evaluation enables failure attribution at the component level, supporting targeted improvement without confounding from cross-stage error propagation.

###### A.2 Comparison with Benchmarks

Recent benchmarks evaluate parts of scientific evidence synthesis, but they target different objects. Some test clinical review workflows, others test evidence sentence retrieval, conclusion matching, or long form research reports. Table 5 summarises these differences before we discuss the closest comparators.

Clinical workflow evaluation The closest workflow level comparator is TrialReviewBench [23]. It evaluates search, screening, extraction and synthesis for clinical oncology reviews. Its strength is breadth across PRISMA aligned stages and human AI collaboration. Its construct differs from ours

- Table 5: Evaluation studies adjacent to AgentSLR. We compare each study by the capability evaluated, the reference signal used, and its relation to epidemiological SLR evidence handling.

Study Evaluated capability Reference signal Relation to AgentSLR

TrialReview Bench [23]

Clinical SR search, screening, extraction and synthesis

Included studies and study characteristics from 100 cancer treatment reviews

Broad clinical workflow evaluation, focused on intervention evidence rather than epidemiological parameters, transmission models and outbreaks.

LLM4 SCREENLIT [24]

Screening evaluation methodology

Reanalysis of confusion matrix based screening studies

Directly motivates recall focused and cost sensitive screening metrics, but does not provide a biomedical extraction dataset.

Evidence Bench [25]

Evidence sentence retrieval within biomedical papers

Expert written evidence summaries and sentence annotations

Strong evidence grounding benchmark, but it tests locating supporting sentences rather than study selection or schema grounded records.

Med Evidence [26]

Matching Cochrane review conclusions from source studies

284 human curated conclusion questions from 100 Cochrane reviews

Tests cross study conclusion reasoning, while abstracting away article screening and structured extraction.

DeepScholar Bench [42]

Generative related work synthesis from live scholarly search

Human written related work sections and automated synthesis, retrieval and citation metrics

Evaluates long form academic synthesis, but not clinical or epidemiological review labels.

Mushtaq et al. [43] Judging completed SLR manuscripts

PRISMA aligned scores on five SLRs

Evaluates manuscript quality after a review exists, not evidence retrieval, screening or extraction.

because the extraction targets are intervention centred study characteristics and outcomes. AgentSLR instead evaluates epidemiological parameters, transmission models and outbreaks across WHO priority pathogen reviews.

Evidence reasoning MedEvidence [26] and EvidenceBench [25] isolate complementary evidence reasoning skills. MedEvidence asks whether models can match Cochrane conclusions from the same source studies, and shows failures around uncertainty and scientific scepticism. EvidenceBench evaluates whether models can retrieve the sentences that support biomedical hypotheses. Both are important controlled evaluations, but neither measures evidence survival through screening nor the conversion of heterogeneous paper level evidence into structured epidemiological records.

Screening metrics LLM4SCREENLIT [24] is primarily methodological. Its recommendations on recall, lost evidence, full confusion matrices and cost sensitive screening are directly aligned with high stakes SLR evaluation. We follow the same risk aware motivation for screening, then extend the evaluation construct to extraction through flagging, count and field level matching metrics.

SciLitBench SciLitBench [44] is a review automation dataset on AI assisted literature reviews, but we omit detailed comparison because the public draft contains unresolved placeholders, including “TODOXXX”, and undisclosed evaluation labels and design.

AgentSLR positioning Taken together, these studies occupy three neighbouring regimes: clinical SR workflow automation, biomedical evidence retrieval and conclusion matching, and open ended deep research report evaluation. AgentSLR is situated between the first two. It is not a general benchmark for deep research agents, and it does not evaluate final meta analysis or narrative report writing. Its contribution is a high stakes evaluation dataset and harness for epidemiological SLR evidence handling, using PERG expert annotations from peer reviewed WHO priority pathogen reviews, stage isolated screening metrics, and record matching metrics for parameters, transmission models and outbreaks. This fills a gap left by settings that stop at binary screening, sentence retrieval or conclusion classification, because public health reviews require models to preserve evidence through screening and then normalise heterogeneous estimates into schema validated records with uncertainty and population context.

##### B Data Representativeness & Ecological Validity

We evaluate LLMs against systematic literature reviews (SLRs) produced by the Pathogen Epidemiology Review Group (PERG), whose curated article sets and extracted data are made available through the epireview and priority-pathogen R packages [34, 35]. The evaluation spans up to seven WHO priority pathogens [28], with screening evaluated across all seven and structured extraction evaluated across four (Ebola, Lassa, SARS-CoV-1, and Zika) for which PERG’s published SLRs and extraction artefacts were both complete and consistently formatted. This section details the article corpus underlying these evaluations and provides evidence that the open-access subset used for benchmarking is representative of the broader PERG population corpus.

A potential concern with the data used is whether the 27.0% open-access overlap (mentioned in Table 1) constitutes a biased sample of the broader PERG corpus, which would affect the validity of reported performance estimates. To assess this directly, we manually retrieved (through institutional access) and processed 1,004 closed-access articles across four pathogens (Ebola, Lassa, SARS-CoV1, and Zika) that appear in the broader PERG population corpus but were not accessible through OpenAlex at the time of retrieval. We then constructed a matched stratified random sample from the AgentSLR open-access subset and ran the full gpt-oss-120b pipeline on both groups under identical conditions.

- Table 6: Representativeness of the open-access evaluation subset across pipeline stages. Macro F1 with 95% bootstrap confidence intervals for the AgentSLR open-access evaluation sample and a matched sample of 1,004 closed-access articles from the broader PERG population corpus, retrieved via institutional access across Ebola, Lassa, SARS-CoV-1, and Zika. Both groups are processed using gpt-oss-120b under identical conditions. ∆ F1 denotes the signed difference (open-access minus population). Differences are small across stages (range: −5.8 to +3.1 pp). The full-text screening interval excludes zero, indicating lower performance for the open-access subset at this stage. The abstract screening and parameter extraction intervals include zero. Overall, these results suggest broad comparability between the open-access subset and the broader PERG population corpus, with a modest full-text screening gap.

Stage AgentSLR (Open-Access) PERG Population ∆ F1

Abstract Screening 0.877 [0.853, 0.899] 0.906 [0.878, 0.930] −0.028 [−0.062, 0.010] Full-text Screening 0.860 [0.830, 0.886] 0.918 [0.890, 0.942] −0.058 [−0.093, −0.017] Parameter Extraction 0.728 [0.654, 0.799] 0.697 [0.607, 0.781] +0.031 [−0.079, 0.142]

- Table 6 reports stage-level macro F1 with 95% bootstrap confidence intervals for the AgentSLR open-access sample and the PERG population sample (closed-access). Performance is broadly comparable across evaluated stages. The largest difference occurs in full-text screening (−5.8 percentage points), where the confidence interval excludes zero, indicating slightly reduced sensitivity on the open-access subset at the stage. However, abstract screening and parameter extraction show overlapping confidence intervals and similar effect sizes, indicating that the main workflow trends remain qualitatively consistent across accessible and inaccessible subsets. Absolute performance estimates should therefore be interpreted relative to the lawfully retrievable evaluation corpus.6

6OpenAlex notes that cached PDFs retain their original copyright and grant no additional reuse rights (https: //developers.openalex.org/download/full-text-pdfs). We therefore release identifiers, metadata and annotations, but not OCR text or PDF-derived Markdown unless redistribution is explicitly permitted by the source licence.

##### C Evaluation Constructs

Construct validity asks whether the metrics measure the phenomenon a benchmark is designed to test, with limited contamination from confounding sources of variance [45]. We name the construct of AgentSLR as evidence handling fidelity: the ability of an LLM-driven pipeline to identify, retain and structurally represent claims from primary scientific literature in a manner consistent with expert review practice. This construct is narrower than scientific knowledge synthesis. Synthesis additionally requires aggregation across studies, weighting of heterogeneous evidence, and the production of meta-analytic conclusions, none of which we evaluate. Stating the construct explicitly clarifies what AgentSLR can and cannot test, and bounds the population of claims we are entitled to make from its results.

Each metric operationalises one facet of this construct. Screening precision, recall and macro F1 measure inclusion fidelity at the article level. Macro averaging is chosen because the construct treats correct inclusions and correct exclusions as both meaningful, and the class imbalance in PERG screening would otherwise allow the majority class to dominate the score. We report recall alongside F1 because the operationally costly screening error is the missed relevant article, which propagates irrecoverably to all downstream stages. The Flagging, Counts and Extraction decomposition maps onto three separable failure modes that an expert reviewer distinguishes when inspecting machinegenerated records: whether the system saw this kind of evidence, whether it produced the right number of records, and whether the field values were correct. The bipartite matching step encodes the construct decision that reference and predicted records have no canonical alignment within an article, and that a record which is partially correct (correct outbreak, wrong date) should contribute partial credit rather than be discarded. Field weights and the choice of Jaccard similarity for multi-value fields are explicit modelling choices and not neutral measurement, and we therefore treat cross-model rankings as more robust than absolute scores when comparing systems.

Three threats to construct validity remain in the present harness. Construct under-representation arises because we do not evaluate meta-analytic synthesis or report writing, and our results consequently bound only the evidence handling component of an SLR. Construct-irrelevant variance arises from OCR error, prompt formulation and the choice of field weights in the matching step, each of which introduces variation that is not part of the construct. We mitigate this by reporting standard deviations across pathogens and by relying on cross-model rankings rather than absolute scores when drawing comparative conclusions. Reference-label noise arises because PERG annotations, although produced by trained reviewers under a documented protocol, are not free of disagreement. The expert validation in Section C.3 provides a partial check on this by recovering field-level accuracy estimates from a fresh pool of reviewers, and we read the resulting numbers as a ceiling for the automated metrics rather than as a pure measure of system error.

###### C.1 Article Screening

We evaluate screening as a binary article-level decision y ∈ {✓,×} (✓ = include; × = exclude) against the PERG reference label. Let TP be articles correctly labelled as ✓, FP be those incorrectly labelled as ✓and FN be those incorrectly labelled as ×; then

TP TP + FP

TP TP + FN

2PR P + R

Precision =

, Recall =

, F1 =

,

where Precision measures the reliability of ✓ decisions and Recall measures how well we avoid assigning × to PERG-✓ papers. We report macro-F1 to weight ✓ and × performance equally, rather than letting the majority class dominate.

By default, article screening happens in two subsequent stages: first on the abstract and then on the full text. Full-text screening is therefore evaluated with different ablations so we can quantify both the stage-specific and holistic performance. We use three code-defined evaluation configurations for the ablations: (i) AI abstract→AI full-text, where any abstract × forces final ×; (ii) Human abstract→AI full-text (PERG-conditioned), where any PERG abstract × forces final ×; and (iii) AI direct full-text, which evaluates the AI full-text decision without filtering by abstract screening decisions.

###### C.2 Data Extraction

Schema validation and data quality Prior to evaluation, we validated and filtered our human reference extractions to ensure that only properly formatted annotations were compared. For fields typed as Enums in the schemas outlined in Section I, we defined acceptable values based on the PERG REDCap survey schema, which standardises entries through dropdown lists. Other fields provide a multi-select option — these we handled as List[Enum] types in the tool call schemas. We filter any reference extractions where Enum-typed values do not agree with the schema, in order to avoid penalising LLMs for extractions it is not allowed to produce. Because of schema verification applied in the tool-calling stage, LLMs produce no such invalid extractions.

After validation, we aligned articles using shared identifiers, retaining only articles labelled ✓ by both PERG and LLMs (through AgentSLR workflow). This intersection matches the reference-labelled data to our article pool (Table 1), and thus avoids counting errors due to paper availability from the article screening stages.

Evaluation Framework We evaluated extraction performance according to three measures: Flagging, Count, and Extraction. All three measures are operationalised with standard classification metrics, specifically, we define and collect precision and recall for each.

Flagging measures whether LLMs correctly identify the relevant data types to extract from each article. This measure considers all ⟨article,data_type⟩ pairs, assigning labels with the functions

1 There is a human extraction of data_type from article

y(⟨article,data_type⟩) =

- 0 otherwise

yˆ(⟨article,data_type⟩) =

- 1 LLM identifies data_type as relevant in article 0 otherwise

and calculating precision and recall on these labels as in a standard binary classification task.

Count measures whether the overall volume of LLM extractions agrees with those in the human reference-labelled data, irrespective of any agreement between the extraction contents. We operationalise this measure using a partial credit scheme: if an article had n models in the reference and our extractor identified nˆ models, we counted true positives as correctly matching counts

TP = min(n,nˆ), false positives as excess extractions

FP = max(0,nˆ − n), and false negatives as missed extractions

FN = max(0,n − nˆ).

For example, if the reference contained 2 data points but we extracted 5, we would receive credit for the 2 correct extractions (TP = 2), be penalised for 3 spurious models (FP = 3), and would receive no penalty for missed models (FN = 0). We sum all counts across all common articles and calculate precision and recall as standard.

Extractions faced a more complex matching challenge: while extractions can be trivially compared by raw count, they consist of many metadata fields, and lack unique identifiers to establish canonical correspondence. Matching every field value exactly is an unreasonably challenging task, and it provides no measure beyond absolute correspondence. To assess the field-level quality of our extractions, we first established optimal one-to-one correspondences between human reference annotations and LLM extractions within each article by computing pairwise similarity.

For each extraction pair, we defined a subset of key fields F from the fields defined in Section I and compared these using normalised weights. The similarity between a true extraction E and an LLM extraction Eˆ was computed as

s(E,Eˆ) =

wk · dk(E[k],Eˆ[k]),

k∈F

where wk is the normalised weight for field k (with k∈F wk = 1) and dk is the Jaccard similarity between fields in the extractions

dk(v,vˆ) = J(v,vˆ) = |v ∩ vˆ| |v ∪ vˆ|

.

We then applied the modified Jonker–Volgenant algorithm [27] using SciPy’s scipy.optimize.linear_sum_assignment()7 function to the cost matrix (cost = 1 − s), finding the matching that maximised total similarity.

- Table 7 illustrates this optimal bipartite matching on an example. Suppose a single article has two reference models extracted by expert epidemiologists (PERG), while the LLM produces three extractions. Because the sets differ in size, no perfect bijection exists, and the algorithm must leave at least one LLM extraction unmatched. For explanation purposes we restrict to two fields: model_type, a single-value field scored by exact match (δtype ∈ {0,1}), and interventions, a multi-value field scored by Jaccard similarity (Jint = |v ∩ vˆ|/|v ∪ vˆ|). With equal weights, each pairwise cell reduces to sij = 0.5δtype + 0.5Jint.

The algorithm correctly recovers both reference correspondences – achieving total similarity 2.00 out of a maximum possible 2.00 – while the spurious LLM-extraction M2 is left unmatched and counted as a false positive under the Count metric. Crucially, this unmatched extraction incurs a Count penalty only; it does not contaminate field-level Extraction scores, ensuring over-extraction and extraction inaccuracy are penalised independently.

Once optimal correspondences are established, we evaluated each field within each matched pair to compute field-level precision and recall. For single-value fields, we counted true positives as sets of equal values, false positives as all LLM-extracted values with no or an unequal match, and false negatives as all human reference values with no or an unequal match.

For multi-value fields, we defined

TP = |v ∩ vˆ| FP = |vˆ \ v| FN = |v \ vˆ|

where v ∈ E and vˆ ∈ Eˆ are sets of values. Aggregating across all matched pairs and articles, we computed precision and recall as standard.

###### Data Extraction: Parameters

Parameter extraction is more varied than model and outbreak extraction. While there is only one data_type for each of model and outbreak extraction, parameters are broken down into nine distinct parameter classes (listed in Section I.1) each with different fields to extract. Therefore, we resolve nine parameter data_types at the level of parameter classes and calculate Flagging and Count metrics for each of these separately.

We defined our key parameter fields as

F = {parameter_class,parameter_type,value,unit,method,value_type, statistical_approach,paired_uncertainty,single_type_uncertainty, population_sex,population_group,population_sample_type},

ensuring that each sub-stage of value extraction, uncertainty extraction, and population context extraction are represented by multiple fields common across parameter classes. We normalise weights wk so as to make each sub-stage equally important in determining similarity. This reflects the multistage structure of extraction (Appendix I), where each stage is gated on the previous and therefore warrants equal evaluative weight. Moreover perturbing within-group field weights by ±10% leaves cross-model rankings unchanged. Fields are grouped as follows:

- • Categorical fields (2 fields): parameter class; parameter type;
- • Value fields (3 fields): value; unit; method;
- • Uncertainty fields (4 fields): value type; statistical approach; single type uncertainty; paired uncertainty;
- • Population fields (3 fields): population sex; population group; population sample type.

7https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment. html

###### Table 7: Optimal bipartite matching example: 2 PERG reference models, 3 LLM-extracted models. (a) Input field values (two fields shown for illustration). (b) Pairwise similarity matrix S.

(c) Optimal matching; LLM M2 is unmatched (FP). (d) Per-cell similarity calculations for entries of S.

(a) Model Field Values

Model Type Interventions PERG Reference

PERG M1 SIR Vaccination PERG M2 SEIR Quarantine; Vaccination

AI-Extracted

LLM M1 SIR Vaccination LLM M2 SIR Treatment LLM M3 SEIR Quarantine; Vaccination

(b) Pairwise Similarity Matrix S

LLM M1 LLM M2 LLM M3

1.00 0.50 0.25 0.25 0.00 1.00

← PERG M1 ← PERG M2

S =

(c) Optimal Matching

PERG M1 ↔ LLM M1 s = 1.00 PERG M2 ↔ LLM M3 s = 1.00 LLM M2 : unmatched (FP) —

Total similarity 2.00

(d) Similarity Calculations sij = 0.5 δtype

+ 0.5 Jint

Jaccard on interventions

exact match on type

Type match δtype Jaccard Jint sij

- S1,1 SIR = SIR: 1.0 J({V }, {V }) = 1/1 = 1.0 1.00
- S1,2 SIR = SIR: 1.0 J({V }, {T}) = 0/2 = 0.0 0.50
- S1,3 SIR ̸= SEIR: 0.0 J({V }, {Q, V }) = 1/2 = 0.5 0.25

- S2,1 SEIR ̸= SIR: 0.0 J({Q, V }, {V }) = 1/2 = 0.5 0.25
- S2,2 SEIR ̸= SIR: 0.0 J({Q, V }, {T}) = 0/3 = 0.0 0.00
- S2,3 SEIR = SEIR: 1.0 J({Q, V }, {Q, V }) = 2/2 = 1.0 1.00

V=Vaccination, Q=Quarantine, T=Treatment; J(A, B) = |A ∩ B| / |A ∪ B|.

Data Extraction: Transmission Models

###### Table 8 shows the filtering statistics across pathogens: human reference datasets had between 3.85% (Lassa) and 23.14% (Zika) invalid entries removed. For data extraction for models, we defined our key fields as

F = {model_type,compartmental_type,stoch_deter,theoretical_model, assumptions,interventions_type,transmission_route}.

Data Extraction: Outbreaks

###### Table 9 shows the filtering statistics across pathogens: PERG datasets had between 0% (Lassa) and 9.43% (Zika) invalid entries removed.

- Table 8: Validation statistics for PERG reference data and AI-extracted transmission model annotations across four pathogens. PERG entries contained invalid field values due to manual data entry inconsistencies, while AI-extracted values showed no invalid entries due to structured schema enforcement during extraction. The AI-extracted numbers below are with gpt-oss-120b, but we find across all LLMs the number of invalid entries are 0.

Pathogen PERG Total PERG Invalid Invalid (%) AI-extracted Total

Lassa 52 2 3.85 19 Ebola 294 46 15.7 239 SARS 112 8 7.14 85 Zika 229 53 23.1 132

- Table 9: Validation statistics for PERG reference data and AI-extracted outbreak annotations across two pathogens. PERG entries contained invalid field values due to manual data entry inconsistencies, while AI-extracted values showed 0% invalid entries due to structured schema enforcement during extraction.

Pathogen PERG Total PERG Invalid Invalid (%) AI-Extracted Total

Lassa 30 0 0.00 62 Zika 159 15 9.43 240

For data extraction for outbreaks, we defined our key fields as

F = {outbreak_start_day,outbreak_start_month,outbreak_start_year, outbreak_end_day,outbreak_end_month,outbreak_end_year, cases_confirmed,deaths,outbreak_country,outbreak_location, detection_mode,pre_outbreak_status}.

Weights wk were determined by the discriminative power of each field k for identifying unique outbreak events. outbreak_country, outbreak_start_year, cases_confirmed, and deaths received weights of 1.0, while supporting temporal fields (outbreak_start_month, outbreak_end_year) received weights of 0.6–0.8, and contextual fields (outbreak_location, mode_of_detection) received weights of 0.5–0.7. Given the limited number of outbreak records (n = 169), we verified that setting all weights to 1.0 does not produce a statistically significant change in extraction F1 at the 95% level. The reported weights reflect a cascading importance ordering.

To provide interpretable summaries of extraction performance, we grouped the 17 outbreak fields into four categories based on their epidemiological function:

- • Temporal Features (7 fields): outbreak start/end dates (year, month, day) and duration;
- • Geographic and Spatial Features (2 fields): outbreak country and specific location;
- • Case Burden (5 fields): confirmed, suspected, asymptomatic, and severe case counts, plus deaths;
- • Epidemiological Context and Metadata (3 fields): mode of detection, pre-outbreak status, and asymptomatic transmission description.

###### C.3 Human Expert Validation

For human expert validation, we recruited six epidemiologists to complete a series of form submissions to grade AgentSLR workflow generated data extractions (with gpt-oss-120b). Each epidemiologist was on-boarded with the expectation to spend up to 10 hours on the validation process over 1 to 2 weeks as their availability permitted. We did not assign experts randomly across parameters, models, and outbreaks – instead, we considered expertise with specific pathogens and familiarity with specific SLR workflows when making assignments. Our assignments resulted in three epidemiologists completing validation solely for parameters, two solely for models, and the final sixth epidemiologist completing validation across all three data modalities.

For each data type in parameters, models, and outbreaks, and for each pathogen in Lassa, Ebola, SARS, and Zika, we sample screened articles randomly without replacement until we generate subsamples guaranteed to exceed the time commitment from each expert. The experts are then instructed to proceed through their assigned extractions in order. Despite normalising for counts

across different pathogens, different articles may have varying numbers of extractions, and these extractions may take varying amounts of time to grade. Thus, our experimental setup does not guarantee parity across pathogens or across data types.

Experts are onboarded with a private GitHub repository that contains the Markdown extractions from our OCR model (Section 3.1), along with Markdown documents rendering the structured data extractions in a readable format. Submissions are collected through Google Forms. Each form proceeds through groups of questions in the same order. Each question contains an optional free-text field for providing context, which we use to collect and synthesise qualitative impressions of the pipeline as well as specific error patterns.

The groups of questions in each form cover the following:

- 1. The expert records the article identifier, pathogen identifier, and the pathogen.
- 2. The expert assesses whether the Markdown document has any significant issues that would affect data extraction.
- 3. Before looking at the AgentSLR-extracted data, the expert determines whether there is any relevant data in the article to extract.
- 4. The expert rates their particular extraction for overall relevance.
- 5. The expert answers a series of yes-or-no questions to validate the accuracy of each extracted field.
- 6. The expert grades the overall pipeline competence using a Likert scale rating between 1 and

7. We provide these particular descriptions to calibrate the Likert scale:

- • “1" means “the system gets nothing right; I couldn’t use it to speed up my process at all."
- • “4" means “the system identifies some things but struggles with edge cases; I could use it with moderate supervision / secondary screening."
- • “7" means “the system is perfectly capable of doing all parameter extraction for me."

- 7. The expert provides a self-reported estimate of the time they took to complete the survey.

##### D Pipeline Statistics: Data Processed & Time

- D.1 Runtime Statistics Article counts across SLR stages

To contextualise runtime estimates for both human and automated pipelines, we first summarise the approximate number of articles processed at each stage of the systematic literature review (SLR). These counts are intended to reflect annotator workload rather than final inclusion totals, and correspond to successive filtering stages commonly used in SLR workflows. In particular, counts decrease substantially between title and abstract screening, full-text screening, and data extraction as relevance criteria are progressively applied.

- Table 10: Estimated number of articles reviewed by human annotators at PERG across successive stages of each systematic literature review (SLR). Counts for Title and Abstract Screening correspond to records remaining after deduplication and exclusion of entries with missing or empty abstract metadata. Full-text Screening includes articles flagged as potentially relevant during abstract screening and advanced for full-text review. Data Extraction represents articles deemed suitable for extracting structured, task-relevant evidence. All values are estimates intended to reflect annotator workload at each phase rather than finalised inclusion totals.

Pathogen Title & Abstract Screening Full-text Screening Data Extraction Ebola 11,605 1,674 522 Lassa 2,131 512 193 SARS 12,280 878 289 Zika 10,510 1,343 574 Average 9,132 1,102 395

Runtime estimation methodology

Using the average article counts from Table 10, we estimate total processing time for both the PERG human SLR workflow and the AgentSLR automated pipeline. Per-article time estimates for PERG were obtained through consultation with a Research Associate at PERG who routinely contributes to SLR projects. AgentSLR workflow runtimes (with gpt-oss-120b) were measured directly from pipeline execution logs. All per-article times are converted to hours and multiplied by the average number of articles processed at each stage. We observe differing run-times based on latency and throughput of models, gpt-oss-120b represents the upper-bound across the spectrum.

- Table 11: Comparison of average human time investment (PERG) versus automated processing time (AgentSLR workflow) across systematic literature review stages. The table reports average articles processed across the four pathogens (Ebola, Lassa, SARS, Zika), average per-article time (in seconds), and total processing time (in hours), highlighting efficiency gains from automation. AgentSLR workflow timings are computed using gpt-oss-120b as the underlying model. PDF-toMarkdown conversion is applied to all 9,132 retrieved articles to preserve the option of direct full-text screening across the complete corpus.

Articles (Avg.)

AgentSLR (s/article)

PERG (s/article)

AgentSLR (Hours)

PERG (Hours)

Stage

Article Retrieval 9,132 0.63 0 1.6 0.00 Title & Abstract Screening 9,132 0.63 45 1.6 114.2 PDF-to-Markdown Conversion 9,132 1.1 0 2.8 0.00 Full-text Screening 1,102 2.0 240 0.62 73.5 Data Extraction 395 122.1 1,800 13.4 197.5

###### Total – – – 20.0 385.1

###### PERG runtime calculations

Title and abstract screening at PERG is estimated at 30 to 60 seconds per article. Assuming an average of 45 seconds (0.0125 hours) per article and 9,132 articles screened on average, the estimated time is 9,132 × 0.0125 = 114.15 hours.

Full-text screening is estimated at 2 to 6 minutes per article. Assuming an average of 4 minutes (0.0666 hours) per article and 1,102 articles screened on average, the estimated time is 1,102 × 0.0666 = 73.47 hours.

Data extraction is estimated at a median of 30 minutes (0.5 hours) per article. With 395 articles processed on average, the estimated time is 395 × 0.5 = 197.50 hours.

AgentSLR workflow (gpt-oss-120b) runtime calculations Article retrieval in AgentSLR requires 0.63 seconds per article. With 9,132 articles retrieved on average, the estimated time is 1.6 hours. Title and abstract screening requires 0.63 seconds per article. With 9,132 articles screened on average, the estimated time is 1.6 hours.

PDF-to-Markdown conversion requires 1.1 seconds per article. PDF-to-Markdown conversion is applied to all 9,132 retrieved articles to preserve the option of direct full-text screening across the complete corpus, giving an estimated time of 2.8 hours. This estimate reflects parallel execution with 14 concurrent requests, yielding an average processing time of 0.05 seconds per page and 1.1 seconds per document. Under sequential execution, the measured processing time increases substantially to an average of 0.95 seconds per page and 16.47 seconds per document.

Full-text screening requires 2.0 seconds per article. With 1,102 articles screened on average, the estimated time is 0.62 hours.

Data extraction requires 122.1 seconds per article, comprising outbreak identification, model extraction, and parameter extraction. With 395 articles processed on average, the estimated time is 13.4 hours.

###### D.2 Token Usage and Operational Cost of AgentSLR harness

Title & Abstract Full-text Screening Param. Extraction Model Extraction Outbreak Extraction

$435

9,132

| |1,102<br><br>394 394 394<br><br>|
|---|---|
| | |
| | |
| | |
| | |

| |20.28M 15.30M<br><br>445.26M<br><br>12.91M 12.87M<br><br>|
|---|---|
| | |
| | |
| | |
| | |

| |11.08M<br><br>2.03M<br><br>14.70M<br><br>871.05K 862.79K<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| |$46<br><br>$17 $11 $13<br><br>|
|---|---|
| | |
| | |
| | |
| | |

14M

400

400M

8000

12M

300

10M

300M

6000

8M

200

200M

4000

6M

4M

100

100M

2000

2M

0

0

0

0

|Articles|
|---|

|Input Tokens|
|---|

|Output Tokens|
|---|

|Cost (USD)|
|---|

Figure 6: Articles processed, scaled token and costs by pipeline stage across models. We report the average number of articles reaching each stage, and the corresponding total input tokens, output tokens, and USD cost per stage averaged across models. Token totals are computed by multiplying per article token usage (Table 12) by the average article counts per stage (Table 10). Parameter extraction dominates overall compute, with substantially higher input and output token totals than other stages. Title and abstract screening processes the largest volume of articles but contributes comparatively less to total cost.

Using the average article counts reported in Table 10, we estimate total token usage and USD cost across pipeline stages by combining per-article token statistics with model-specific pricing. Figure 6 summarises the resulting distribution of articles processed, aggregate input tokens, output tokens, and total cost by stage, averaged across models. Per-article input and output token usage by stage and model is reported in Table 12. Total stage costs are computed by multiplying mean per-article token usage by the average number of articles reaching each stage, and then applying published

per-million-token pricing for both input and output tokens. All prices used in these calculations are retrieved directly from the primary API pricing documentation of each model provider at the time of evaluation. Under this pricing regime, parameter extraction dominates overall compute and cost due to substantially higher input and output token volumes, while title and abstract screening processes the largest number of articles but contributes comparatively little to total cost. All reported costs reflect managed API usage; alternative cost estimates could be derived for deployments hosted on dedicated GPU nodes, where pricing would depend on hardware configuration, utilisation, and amortisation assumptions rather than per-token billing.

- Table 12: Per article token usage and estimated cost by stage and model. We report mean input tokens, output tokens, and USD cost for processing a single article at each stage. Green marks the minimum and Red marks the maximum within each row and subcolumn.

Article Screening (AI Cond.)

Title & Abstract Screening

Parameter Extraction

Model Extraction

Outbreak Extraction Overall

Model

Input Tok. 2.3K 16.9K 510.2K 35.9K 49.9K 615.3K Output Tok. 1.2K 1.1K 19.8K 1.9K 2.6K 26.7K Cost (USD) <0.01 <0.01 0.02 <0.01 <0.01 0.02

GPT-OSS-120B (High)

Input Tok. 2.2K 13.1K 961.1K 31.5K 32.4K 1040.4K Output Tok. 0.6K 1.4K 91.1K 2.1K 3.5K 98.7K Cost (USD) 0.01 0.04 2.95 0.08 0.10 3.20

GPT-5.2 (High)

Input Tok. 2.2K 12.9K 523.2K 37.2K 26.1K 601.6K Output Tok. 0.6K 0.8K 3.0K 3.1K 0.2K 7.8K Cost (USD) <0.01 <0.01 0.14 0.01 <0.01 0.17

DeepSeek-V3.2

Input Tok. 2.3K 13.1K 605.5K 29.3K 29.5K 679.8K Output Tok. 2.0K 2.7K 40.0K 2.2K 2.7K 49.6K

Kimi-K2.5

- Cost (USD) <0.01 0.01 0.48 0.02 0.02 0.55

GLM-4.7

Input Tok. 2.2K 13.4K 3050.4K 29.7K 25.4K 3121.1K Output Tok. 1.6K 3.1K 32.7K 1.7K 2.0K 41.1K

- Cost (USD) <0.01 0.01 1.90 0.02 0.01 1.96

7GPT-OSS-120B: https://openrouter.ai/openai/gpt-oss-120b. GPT-5.2: https://developers.openai.com/api/docs/pricing/. DeepSeek-V3.2: https://api-docs.deepseek.com/quick_start/pricing. Kimi-K2.5: https://platform.moonshot.ai/docs/pricing/chat. GLM-4.7: https://docs.z.ai/guides/overview/pricing.

##### E Extended Evaluation Results

In this section we report the disaggregated and extended version of results presented in the main text. The section is structured into three parts. First, we report model-level results from evaluating five LLMs with the AgentSLR harness workflow. Second, we report field-level results for gpt-oss-120b, showing how the harness supports detailed failure analysis beyond aggregate model comparisons. Third, we report expanded article screening strategy (ablation) results for the gpt-oss-120b run.

###### E.1 Results Across Models

Tables 13–17 report pathogen-level metrics for the five models summarised in Figure 2. Results are grouped by article screening, parameter extraction, transmission model extraction and outbreak extraction.

###### Title & Abstract Screening

At title and abstract screening (Table 13), the spread in overall F1 from 0.62 (DeepSeek-V3.2) to 0.77 (Kimi-K2.5) is driven almost entirely by recall rather than precision. DeepSeek-V3.2 and

GPT-5.2 are the two most precise models (0.83 and 0.82 respectively) yet rank last and fourth on F1, with recalls of 0.59 and 0.61 against Kimi-K2.5’s 0.75. Kimi-K2.5 is the best-performing model for all seven pathogens. Nipah is the worst-performing pathogen for three models and sits at or below 0.72 for all five. This is consistent with it being one of the smallest and most heterogeneous corpora in the PERG dataset. The same pathogen has the narrowest precision-recall gap across models, suggesting that the difficulty is intrinsic to the articles rather than a model-specific calibration issue.

###### Table 13: Title and abstract screening metrics across LLMs. Green and Red denote the best-

and worst-performing pathogens for each model (in terms of F1 score). Bold indicates the bestperforming model for each pathogen, and Underline indicates the second-best. P = precision;

R = recall; F1 = F1-Score.

Pathogen gpt-oss-120b GPT-5.2 DeepSeek-V3.2 Kimi-K2.5 GLM-4.7

P R F1 P R F1 P R F1 P R F1 P R F1

Marburg 0.80 0.64 0.69 0.97 0.58 0.62 0.97 0.55 0.58 0.79 0.65 0.69 0.88 0.61 0.66 Ebola 0.74 0.75 0.75 0.76 0.64 0.68 0.80 0.61 0.64 0.79 0.79 0.79 0.88 0.72 0.77 Lassa 0.82 0.72 0.75 0.78 0.63 0.66 0.84 0.60 0.63 0.84 0.77 0.80 0.88 0.68 0.73 SARS 0.78 0.76 0.77 0.77 0.62 0.65 0.82 0.62 0.66 0.80 0.78 0.79 0.89 0.73 0.78 Zika 0.73 0.77 0.75 0.70 0.62 0.64 0.73 0.63 0.66 0.78 0.79 0.79 0.76 0.69 0.72 MERS 0.83 0.74 0.78 0.87 0.62 0.67 0.86 0.60 0.65 0.86 0.78 0.81 0.89 0.67 0.73 Nipah 0.84 0.66 0.70 0.92 0.58 0.59 0.81 0.54 0.53 0.85 0.68 0.72 0.90 0.61 0.65

Overall 0.79 0.72 0.74 0.82 0.61 0.65 0.83 0.59 0.62 0.82 0.75 0.77 0.87 0.67 0.72

###### Full-text Screening

The ranking reorders substantially at full-text screening (Table 14), where gpt-oss-120b leads (F1 0.77) and the two highest-precision abstract-stage models fall furthest. DeepSeek-V3.2’s precision drops from 0.83 to 0.64 and its recall from 0.59 to 0.56. It is the only model that loses ground on both measures at once, with its Marburg result (F1 0.42, precision 0.37) the single weakest pathogen-level score in either screening table. gpt-oss-120b’s advantage at this stage comes from recall. It achieves 0.81 overall against the next-best Kimi-K2.5 at 0.73, and is one of only two models, alongside GLM-4.7, for which recall increases from abstract to full-text screening. The Nipah-to-Zika contrast also reverses. Nipah is the best-performing pathogen for gpt-oss-120b and Kimi-K2.5 at full text (F1 0.85 and 0.82), whereas it was the worst for three models at the abstract stage. This suggests that the richer context of full texts resolves ambiguity that titles and abstracts leave open for this pathogen.

###### Parameter Extraction

Parameter extraction results are disaggregated by pathogen and extraction type in Table 15. Kimi-K2.5 achieves the highest overall average F1 (0.63), marginally ahead of GLM-4.7 (0.63), with

- Table 14: Full-text screening metrics across LLMs. Green and Red denote the best- and worst-

performing pathogens for each model (in terms of F1 score). Bold indicates the best-performing model for each pathogen, and Underline indicates the second-best. P = precision; R = recall;

- F1 = F1-Score.

Pathogen gpt-oss-120b GPT-5.2 DeepSeek-V3.2 Kimi-K2.5 GLM-4.7

P R F1 P R F1 P R F1 P R F1 P R F1

Marburg 0.75 0.76 0.75 0.76 0.59 0.59 0.37 0.49 0.42 0.66 0.66 0.66 0.86 0.72 0.76 Ebola 0.73 0.84 0.77 0.61 0.60 0.60 0.68 0.59 0.55 0.72 0.74 0.71 0.75 0.75 0.75 Lassa 0.79 0.78 0.78 0.66 0.63 0.63 0.63 0.54 0.47 0.74 0.75 0.74 0.77 0.73 0.73 SARS 0.71 0.85 0.76 0.60 0.58 0.58 0.73 0.61 0.59 0.67 0.69 0.66 0.73 0.72 0.72 Zika 0.66 0.79 0.69 0.50 0.50 0.50 0.61 0.55 0.52 0.63 0.64 0.61 0.60 0.59 0.59 MERS 0.76 0.83 0.79 0.66 0.61 0.61 0.74 0.60 0.58 0.76 0.79 0.77 0.76 0.68 0.69 Nipah 0.87 0.84 0.85 0.80 0.63 0.63 0.73 0.56 0.53 0.83 0.81 0.82 0.72 0.61 0.61

Overall 0.75 0.81 0.77 0.66 0.59 0.59 0.64 0.56 0.52 0.72 0.73 0.71 0.74 0.69 0.69

lower performance for gpt-oss-120b (0.59), GPT-5.2 (0.58) and DeepSeek-V3.2 (0.56). Performance is most variable in the Counts sub-task. gpt-oss-120b attains strong precision (0.83) but low recall (0.47), reproducing the asymmetric pattern observed in the detailed field-level results below. GPT-5.2 shows the inverse pattern, with recall 0.83 and precision 0.36. At the field-level Extraction sub-task, GPT-5.2 achieves the highest overall F1 (0.59), followed by Kimi-K2.5 (0.56). The five models are broadly comparable, consistent with the interpretation that cross-model differences in average F1 are driven by flagging and counting behaviour rather than the quality of individual field extractions. Zika is the weakest pathogen across most models and sub-tasks, while SARS is frequently the best-performing for gpt-oss-120b.

- Table 15: Parameter extraction metrics across LLMs. Average denotes means across sub-tasks; Overall denotes means across pathogens. Green and Red denote the best- and worst-performing pathogens for each model (in terms of F1 score). Bold indicates the best-performing model for each pathogen, and Underline indicates the second-best. P = precision; R = recall; F1 = F1-Score.

Pathogen Type gpt-oss-120b GPT-5.2 DeepSeek-V3.2 Kimi-K2.5 GLM-4.7

P R F1 P R F1 P R F1 P R F1 P R F1

Flagging 0.60 0.92 0.72 0.58 0.93 0.71 0.49 0.91 0.64 0.67 0.90 0.77 0.72 0.82 0.77 Counts 0.79 0.47 0.59 0.46 0.80 0.58 0.57 0.59 0.58 0.52 0.73 0.61 0.59 0.65 0.62 Extraction 0.48 0.54 0.50 0.58 0.57 0.57 0.54 0.47 0.49 0.55 0.57 0.55 0.51 0.56 0.52 Average 0.62 0.64 0.60 0.54 0.77 0.62 0.54 0.65 0.57 0.58 0.73 0.64 0.61 0.68 0.64

Ebola

Flagging 0.56 0.98 0.71 0.58 1.00 0.73 0.54 0.91 0.68 0.70 0.94 0.81 0.77 0.87 0.82 Counts 1.00 0.35 0.51 0.30 0.85 0.45 0.46 0.57 0.51 0.59 0.81 0.69 0.74 0.59 0.66 Extraction 0.58 0.54 0.55 0.66 0.63 0.63 0.55 0.46 0.47 0.58 0.60 0.57 0.58 0.56 0.56 Average 0.71 0.62 0.59 0.51 0.83 0.61 0.52 0.65 0.55 0.62 0.79 0.69 0.70 0.67 0.68

Lassa

Flagging 0.50 0.81 0.62 0.47 0.83 0.60 0.39 0.78 0.52 0.56 0.69 0.62 0.58 0.67 0.62 Counts 0.80 0.61 0.69 0.37 0.88 0.52 0.60 0.60 0.60 0.45 0.72 0.56 0.51 0.59 0.55 Extraction 0.51 0.63 0.56 0.56 0.63 0.58 0.58 0.54 0.55 0.53 0.65 0.57 0.51 0.62 0.55 Average 0.61 0.69 0.62 0.47 0.78 0.57 0.52 0.64 0.56 0.51 0.69 0.58 0.53 0.63 0.57

SARS

Flagging 0.40 0.96 0.57 0.43 0.95 0.59 0.41 0.94 0.57 0.56 0.86 0.68 0.62 0.75 0.68 Counts 0.72 0.47 0.57 0.31 0.80 0.45 0.50 0.60 0.55 0.55 0.73 0.63 0.59 0.67 0.63 Extraction 0.52 0.57 0.53 0.56 0.59 0.56 0.55 0.50 0.51 0.54 0.59 0.55 0.52 0.58 0.54 Average 0.55 0.67 0.56 0.43 0.78 0.53 0.49 0.68 0.54 0.55 0.73 0.62 0.58 0.66 0.61

Zika

Flagging 0.51 0.92 0.66 0.51 0.93 0.66 0.46 0.88 0.60 0.62 0.85 0.72 0.67 0.78 0.72 Counts 0.83 0.47 0.59 0.36 0.83 0.50 0.53 0.59 0.56 0.53 0.75 0.62 0.61 0.62 0.61 Extraction 0.52 0.57 0.54 0.59 0.61 0.59 0.56 0.49 0.50 0.55 0.60 0.56 0.53 0.58 0.54 Average 0.62 0.65 0.59 0.49 0.79 0.58 0.52 0.66 0.56 0.57 0.73 0.63 0.60 0.66 0.63

Overall

- Table 16: Model extraction metrics across LLMs. Green and Red denote the best- and worst-

performing pathogens for each model (in terms of F1 score). Average denotes means across sub-tasks; Overall denotes means across pathogens. Bold indicates the best-performing model for

each pathogen, and Underline indicates the second-best. P = precision; R = recall; F1 = F1-Score.

Pathogen Type gpt-oss-120b GPT-5.2 DeepSeek-V3.2 Kimi-K2.5 GLM-4.7

P R F1 P R F1 P R F1 P R F1 P R F1

Flagging 0.92 0.92 0.92 0.89 0.90 0.89 0.87 0.86 0.86 0.93 0.93 0.93 0.95 0.94 0.95 Counts 0.50 1.00 0.67 0.56 1.00 0.71 0.81 0.99 0.89 0.63 0.99 0.77 0.88 0.99 0.93 Extraction 0.59 0.72 0.64 0.62 0.74 0.66 0.57 0.65 0.61 0.60 0.71 0.64 0.62 0.71 0.66 Average 0.67 0.88 0.74 0.69 0.88 0.76 0.75 0.83 0.79 0.72 0.88 0.78 0.81 0.88 0.84

Ebola

Flagging 0.95 0.99 0.97 0.95 0.99 0.97 0.96 0.85 0.89 0.95 0.99 0.97 1.00 1.00 1.00 Counts 0.60 1.00 0.75 0.60 1.00 0.75 1.00 1.00 1.00 0.75 1.00 0.86 1.00 1.00 1.00 Extraction 0.68 0.73 0.70 0.68 0.79 0.71 0.79 0.78 0.78 0.70 0.78 0.73 0.73 0.77 0.74 Average 0.74 0.91 0.81 0.74 0.92 0.81 0.92 0.88 0.89 0.80 0.92 0.85 0.91 0.92 0.91

Lassa

Flagging 0.86 0.86 0.86 0.86 0.86 0.86 0.83 0.81 0.82 0.87 0.87 0.87 0.85 0.84 0.84 Counts 0.49 0.97 0.65 0.49 1.00 0.66 0.70 1.00 0.82 0.67 1.00 0.81 0.76 1.00 0.86 Extraction 0.60 0.71 0.64 0.61 0.73 0.64 0.55 0.64 0.59 0.63 0.74 0.66 0.59 0.68 0.62 Average 0.65 0.85 0.72 0.65 0.86 0.72 0.70 0.82 0.74 0.72 0.87 0.78 0.73 0.84 0.77

SARS

Flagging 0.87 0.89 0.88 0.89 0.91 0.90 0.90 0.89 0.90 0.90 0.92 0.91 0.93 0.93 0.93 Counts 0.48 0.98 0.65 0.61 1.00 0.76 0.97 0.97 0.97 0.72 0.97 0.83 0.88 0.97 0.93 Extraction 0.66 0.78 0.70 0.67 0.78 0.69 0.59 0.64 0.61 0.67 0.77 0.70 0.69 0.76 0.71

Zika

- Average 0.67 0.88 0.74 0.72 0.90 0.78 0.82 0.84 0.83 0.76 0.89 0.81 0.83 0.89 0.85

Overall

Flagging 0.90 0.91 0.91 0.90 0.91 0.90 0.89 0.85 0.87 0.91 0.93 0.92 0.93 0.93 0.93 Counts 0.52 0.99 0.68 0.56 1.00 0.72 0.87 0.99 0.92 0.69 0.99 0.81 0.88 0.99 0.93 Extraction 0.63 0.74 0.67 0.64 0.76 0.67 0.63 0.68 0.65 0.65 0.75 0.68 0.66 0.73 0.68

- Average 0.68 0.88 0.75 0.70 0.89 0.77 0.80 0.84 0.81 0.75 0.89 0.81 0.82 0.88 0.85

###### Transmission Model Extraction

Transmission model extraction results are presented in Table 16. GLM-4.7 achieves the highest overall average F1 (0.85), with strong performance across all three sub-tasks: Flagging (0.93), Counts (0.93), and Extraction (0.68). DeepSeek-V3.2 ranks second overall (0.81), driven by notably high Counts performance (0.92), whilst gpt-oss-120b ranks last (0.75), held back by comparatively low Counts precision (0.52 overall). Lassa is the best-performing pathogen for all five models, with GLM-4.7 achieving an overall average F1 of 0.91 for that pathogen alone, including perfect Flagging and Counts scores. SARS is consistently the most challenging pathogen. Flagging F1 ranges from 0.82 (DeepSeek-V3.2) to 0.87 (Kimi-K2.5), and Extraction F1 from 0.59 to 0.66. These patterns are consistent with the field-level difficulties in transmission route and assumption extraction reported below.

###### Outbreak Extraction

Outbreak extraction results, evaluated across Lassa and Zika, are shown in Table 17. GPT-5.2 achieves the highest overall average F1 (0.77), followed closely by Kimi-K2.5 (0.76), DeepSeek-V3.2 (0.73), GLM-4.7 (0.72), and gpt-oss-120b (0.70). Results diverge sharply between pathogens. Lassa Counts are strong across all models, ranging from F1 0.77 (GPT-5.2) to 0.95 (Kimi-K2.5), and field-level Extraction is uniformly high (0.76 to 0.83). Zika Counts performance falls substantially for gpt-oss-120b (F1 0.47) and GLM-4.7 (0.52), whilst GPT-5.2 remains comparatively strong (0.83). A notable divergence in Flagging is also observed. DeepSeek-V3.2 performs weakest on Lassa Flagging (F1 0.62) whilst achieving the second-best result on Zika (0.68). These per-pathogen contrasts are consistent with the field-level analysis of suspected cases and epidemiological context fields reported below.

- Table 17: Outbreak extraction metrics across LLMs. Average denotes means across sub-tasks; Overall denotes means across pathogens. Green and Red denote the best- and worst-performing pathogens for each model (in terms of F1 score). Bold indicates the best-performing model for each pathogen, and Underline indicates the second-best. P = precision; R = recall; F1 = F1-Score.

Pathogen Type gpt-oss-120b GPT-5.2 DeepSeek-V3.2 Kimi-K2.5 GLM-4.7

P R F1 P R F1 P R F1 P R F1 P R F1

Lassa

Flagging 0.69 0.82 0.70 0.72 0.84 0.74 0.61 0.65 0.62 0.67 0.77 0.69 0.65 0.68 0.66 Counts 0.83 1.00 0.91 0.62 1.00 0.77 0.83 1.00 0.91 0.90 1.00 0.95 1.00 0.86 0.92 Extraction 0.85 0.73 0.77 0.84 0.79 0.81 0.75 0.78 0.76 0.84 0.83 0.83 0.83 0.76 0.78 Average 0.79 0.85 0.80 0.73 0.88 0.78 0.73 0.81 0.77 0.80 0.87 0.82 0.83 0.76 0.79

Zika

Flagging 0.58 0.71 0.53 0.59 0.75 0.57 0.65 0.82 0.68 0.61 0.80 0.59 0.67 0.87 0.70 Counts 0.49 0.45 0.47 0.76 0.92 0.83 0.68 0.61 0.64 0.72 0.88 0.79 0.84 0.38 0.52 Extraction 0.84 0.78 0.80 0.88 0.87 0.87 0.69 0.81 0.73 0.71 0.78 0.73 0.74 0.78 0.76 Average 0.64 0.65 0.60 0.75 0.84 0.76 0.68 0.74 0.69 0.68 0.82 0.70 0.75 0.68 0.66

Overall

Flagging 0.63 0.76 0.61 0.66 0.79 0.66 0.63 0.73 0.65 0.64 0.78 0.64 0.66 0.77 0.68 Counts 0.66 0.72 0.69 0.69 0.96 0.80 0.76 0.80 0.78 0.81 0.94 0.87 0.92 0.62 0.72 Extraction 0.85 0.76 0.79 0.86 0.83 0.84 0.72 0.79 0.75 0.78 0.81 0.78 0.78 0.77 0.77 Average 0.71 0.75 0.70 0.74 0.86 0.77 0.70 0.78 0.73 0.74 0.84 0.76 0.79 0.72 0.72

- E.2 Field-Level Results Possible with AgentSLR

The main text reports the aggregated data extraction metrics for each model. However, the AgentSLR harness allows us to use field-level values for understanding localised LLM errors on the dataset. This section gives the disaggregated extraction results possible with the harness for gpt-oss-120b. Results are reported by pathogen, data type and field where applicable.

The human reference labelled datasets are provided open source by the Pathogen Epidemiology Review Group, available through the R epireview package or on GitHub at https://github.com/mrcide/epireview/tree/main/inst/extdata. As of March 2026, owing to PERG’s continual progress through SLRs on nine priority pathogens, human reference extraction data is available in a standardised format for four pathogens: Lassa, Ebola, SARS, and Zika. For each pathogen, we evaluate classification measures for each of the Flagging, Count, and Extraction metrics defined formally in Section C.2.

Parameters

- Table 18 presents the results for parameter extraction Flagging and Count metrics. These results are used to produce the aggregate data presented in the main body text (Table 2 in Section 5). For flagging relevant parameters, AgentSLR performs consistently across all pathogens with high recall (0.92 average), though precision is lower and more variable (0.51 average). The results suggest that while the model identifies nearly all relevant extractions, this coverage comes at the cost of many false positive flags that may propagate errors to later sub-stages. In terms of overall parameter extraction counts, the performance flips in favour of precision (0.83) now at the expense of lower recall (0.47). The discrepancy with parameter flagging performance is understandable, as the structured extraction step allows flagged parameters to be discarded through tool calls. When the model produces a final extraction, this extraction is likely correct. However, it often fails to produce all required extractions according to human reference data. An article may have multiple extractions of the same parameter class, and in these cases gpt-oss-120b can underestimate the number of extractions required.

###### Table 18: Flagging and Count classification metrics for parameter extraction with AgentSLR harness (gpt-oss-120b). P = precision; R = recall; F1 = F1-Score.

Metric Lassa Ebola SARS Zika

P R F1 P R F1 P R F1 P R F1

Flagging 0.56 0.98 0.71 0.60 0.92 0.72 0.50 0.81 0.62 0.40 0.96 0.57 Count 1.00 0.35 0.51 0.79 0.47 0.59 0.80 0.61 0.69 0.72 0.47 0.57

- Table 19: Field-level precision, recall, and F1 for Extraction on parameters with AgentSLR harness (gpt-oss-120b). Group corresponds to the sub-stage of parameter extraction where the field is collected. The final row shows averages across all fields. P = precision; R = recall;

F1 = F1-Score.

Group Field Lassa Ebola SARS Zika

P R F1 P R F1 P R F1 P R F1

Value

value 0.22 0.22 0.22 0.20 0.20 0.20 0.23 0.23 0.23 0.14 0.14 0.14 unit 0.50 0.43 0.46 0.62 0.35 0.44 0.69 0.61 0.65 0.65 0.43 0.52 method 1.00 0.89 0.94 0.48 0.78 0.59 0.76 0.83 0.79 0.86 0.80 0.83

Average 0.57 0.51 0.54 0.44 0.44 0.41 0.56 0.56 0.56 0.55 0.46 0.50

Uncertainty

value type 0.38 0.43 0.40 0.30 0.33 0.32 0.35 0.57 0.43 0.12 0.22 0.16 statistical approach – – – – – – – – – 0.44 0.66 0.53 single type uncertainty 1.00 1.00 1.00 0.98 0.94 0.96 0.81 0.95 0.88 0.97 0.99 0.98 paired uncertainty 0.25 0.40 0.31 0.59 0.72 0.65 0.39 0.88 0.54 0.46 0.90 0.61

Average 0.54 0.61 0.57 0.62 0.67 0.64 0.52 0.80 0.62 0.50 0.69 0.57

Population

population sex 0.86 0.67 0.75 0.62 0.79 0.69 0.59 0.75 0.66 0.59 0.70 0.64 population group 0.14 0.11 0.12 0.24 0.33 0.28 0.23 0.25 0.24 0.54 0.54 0.54 population sample type 0.86 0.67 0.75 0.32 0.41 0.36 0.58 0.63 0.60 0.37 0.36 0.37

Average 0.62 0.48 0.54 0.40 0.51 0.44 0.47 0.54 0.50 0.50 0.54 0.52 Overall 0.58 0.53 0.55 0.49 0.54 0.50 0.51 0.63 0.56 0.51 0.57 0.53

For Extraction, complete field-level results are presented in Table 19. The aggregate results in the main text show parameter extractions to have moderate quality across all pathogens, with little variation among them. Analysing the results at the field level reveals patterns in the model’s handling of different data modalities and different types of epidemiological context. The system performs worst on value fields, with population fields also showing relatively weak performance compared to other groups. We suspect the difficulty with population context arises from the large numbers of valid options for many fields, notably population group and population sample type. Many of these options have precise interpretations in epidemiological literature. Without fine-tuning, gpt-oss-120b may struggle to apply these interpretations in a complex tool-calling environment. Classification is near perfect for single type uncertainty and generally strong for method, with the exception of Ebola. Fields with unrestricted domains, like value, are much harder to classify correctly. Expert validation in Section F suggests that at least some of this difficulty may stem from our exact-match criteria being overly punitive of equivalent numbers in different formats.

Transmission Models

- Table 20 presents the complete results for Flagging, Count, and Extraction evaluations of transmission models across the four priority pathogens. Screening performance was strong across all pathogens for article flagging, with recall ranging from 0.86 to 0.99 and precision from 0.86 to 0.96, indicating reliable identification of modelling studies. Model count extraction achieved consistently high recall (0.97–1.00) but notably lower precision (0.48–0.60), suggesting a systematic tendency to overestimate the number of models reported per article rather than failing to identify them.

Field-level extraction showed a clear gradient in task difficulty. Core structural characteristics were extracted with high accuracy. Model type classification and the theoretical versus data-fitted distinction achieved balanced precision and recall between 0.62 and 0.89 across pathogens. Single-value fields such as stochastic versus deterministic modelling and code availability frequently exceeded 0.75 and reached perfect scores for some pathogens. In contrast, more complex or multi-value fields exhibited substantially lower performance. Transmission route extraction was particularly challenging for Ebola and SARS, while assumptions and interventions showed modest precision and recall across all pathogens. Overall, across screening and extraction tasks, precision ranged from 0.61 to 0.70 and recall from 0.75 to 0.81. This indicates that the model reliably captures core model characteristics, with remaining limitations concentrated in the extraction of nuanced descriptive details.

- Table 20: Precision, recall, and F1 metrics for transmission model screening and extraction across four pathogens with AgentSLR harness (gpt-oss-120b). Screening includes article flagging and model count accuracy. Extraction evaluates field-level accuracy for matched model pairs, covering core structural characteristics (model type, stochastic vs deterministic, theoretical vs data-fitted, code availability) and more complex multi-value fields (transmission routes, assumptions, interventions). Strong performance is observed for core model characteristics, while extraction of assumptions, interventions, and transmission routes remains more challenging. P = precision; R = recall; F1 = F1-Score.

Lassa Ebola SARS Zika

P R F1 P R F1 P R F1 P R F1

Flagging Article Flagging 0.95 0.99 0.97 0.92 0.92 0.92 0.86 0.86 0.86 0.87 0.89 0.88

Counts Model Count 0.60 1.00 0.75 0.50 1.00 0.67 0.49 0.97 0.65 0.48 0.98 0.65

Extraction

Model Type 0.89 0.89 0.89 0.89 0.89 0.89 0.77 0.77 0.77 0.88 0.88 0.88 Compartmental Type 0.00 0.00 0.00 — — — 0.80 0.80 0.80 0.83 0.83 0.83 Stochastic vs Deterministic 1.00 1.00 1.00 0.75 0.85 0.80 0.76 0.78 0.77 0.82 0.79 0.81 Theoretical vs Data-Fitted 0.78 0.78 0.78 0.88 0.88 0.88 0.62 0.62 0.62 0.81 0.81 0.81 Code Available 1.00 0.89 0.94 0.85 0.84 0.84 1.00 1.00 1.00 0.82 0.76 0.79 Transmission Routes 1.00 0.94 0.97 0.13 0.15 0.14 0.26 0.32 0.29 0.68 0.74 0.71 Assumptions 0.29 0.46 0.36 0.27 0.46 0.34 0.21 0.39 0.28 0.31 0.52 0.39 Interventions 0.54 0.64 0.58 0.48 0.69 0.56 0.46 0.79 0.58 0.32 0.69 0.43 Overall 0.70 0.78 0.73 0.62 0.77 0.67 0.61 0.75 0.66 0.66 0.81 0.71

Outbreaks

Applying the evaluation framework described in Appendix C, we analysed outbreak extraction performance across two priority pathogens, Lassa and Zika (Table 21). Screening performance differed between pathogens and screening subtasks. For Lassa, article flagging achieved moderate precision (0.69) and strong recall (0.82), indicating that most outbreak-containing papers were

- Table 21: Outbreak screening and extraction with AgentSLR harness (gpt-oss-120b) across feature categories. Screening measured article flagging (identifying papers containing outbreaks) and outbreak count accuracy (extracting the correct number of outbreaks per paper). Extraction evaluated field-level accuracy for matched outbreak pairs across four epidemiological categories: temporal features (start/end dates), geographic features (country, specific location), case burden (confirmed cases, deaths), and epidemiological context (detection mode, pre-outbreak status, ongoing status, asymptomatic transmission). Overall metrics represent the average across all extraction fields. P = precision; R = recall; F1 = F1-Score.

Lassa Zika

Metric Field

P R F1 P R F1 Flagging Article Flagging 0.69 0.82 0.75 0.58 0.71 0.64 Counts Outbreak Counts 0.83 1.00 0.91 0.49 0.45 0.47

Temporal Features 0.83 0.74 0.78 0.85 0.82 0.83 Geographic and Spatial Features 0.75 0.78 0.76 0.75 0.75 0.75 Case Burden 0.82 0.75 0.79 0.93 0.93 0.93 Epidemiological Context and Metadata 0.93 0.70 0.80 0.84 0.67 0.75

Extraction

Overall 0.85 0.73 0.79 0.84 0.78 0.81

- Table 22: Outbreak feature extraction by category for AgentSLR harness (gpt-oss-120b). Each row shows field-level performance within the four major epidemiological categories. Temporal and case burden features showed consistently high performance, while location-specific fields and epidemiological context features showed greater variability. P = precision; R = recall; F1 =

- F1-Score.

Lassa Zika

P R F1 P R F1 Temporal Features

Start Year 0.89 0.80 0.84 0.90 0.82 0.86 Start Month 0.78 0.78 0.78 0.80 0.80 0.80 Start Day 0.86 0.67 0.75 0.95 0.95 0.95 End Month 0.78 0.78 0.78 0.65 0.65 0.65 End Day 0.86 0.67 0.75 0.95 0.86 0.90

- Average 0.83 0.74 0.78 0.85 0.82 0.83

Geographic and Spatial Features

Outbreak Country 1.00 1.00 1.00 1.00 1.00 1.00 Location 0.50 0.56 0.53 0.50 0.50 0.50 Average 0.75 0.78 0.77 0.75 0.75 0.75

Case Burden

Confirmed Cases 0.75 0.60 0.67 0.86 0.86 0.86 Deaths 0.90 0.90 0.90 1.00 1.00 1.00

- Average 0.83 0.75 0.79 0.93 0.93 0.93

###### Epidemiological Context and Metadata

Mode of Detection 0.71 0.50 0.59 0.50 0.41 0.45 Pre-outbreak Status 1.00 0.30 0.46 1.00 0.41 0.58 Ongoing Status 1.00 1.00 1.00 0.86 0.86 0.86 Asymptomatic Transmission 1.00 1.00 1.00 1.00 1.00 1.00 Average 0.93 0.70 0.76 0.84 0.67 0.72

Overall 0.85 0.73 0.79 0.84 0.78 0.81

identified, although a non-trivial fraction of flagged papers were false positives. For Zika, article flagging was more balanced (precision 0.58, recall 0.71), suggesting improved sensitivity relative to precision, but still leaving missed outbreak descriptions and over-inclusion of non-outbreak papers. Outbreak counting remained strong for Lassa (precision 0.83, recall 1.00), while Zika outbreak counting was substantially lower (precision 0.49, recall 0.45), consistent with continued difficulty in reliably enumerating outbreak events in the Zika corpus.

Field-level extraction performance, grouped by epidemiological feature categories, revealed consistent strengths and persistent weaknesses (Table 22). Temporal features remained robust across both pathogens, with precision 0.83 and recall 0.74 for Lassa, and precision 0.85 and recall 0.82 for Zika. Case burden metrics also showed strong extraction, with precision 0.83 and recall 0.75 for Lassa, and precision 0.93 and recall 0.93 for Zika. Death extraction was especially accurate, with 0.90 precision and recall for Lassa and 1.00 precision and recall for Zika.

Geographic extraction continued to show a split between coarse and fine granularity. Outbreak country identification was perfect for both pathogens (1.00 precision and recall), while specific location extraction was notably weaker for Lassa (precision 0.50, recall 0.56) and Zika (precision 0.50, recall 0.50), consistent with variability in how places are described in scientific text. Epidemiological context fields showed the greatest variability. For Lassa, mode of detection was moderate (0.71 precision, 0.50 recall) and pre-outbreak status exhibited high precision but low recall (1.00/0.30), indicating frequent omission of this attribute. Ongoing status was extracted perfectly for Lassa (1.00/1.00) but showed moderate performance for Zika (0.86/0.86). For Zika, mode of detection (0.50/0.41) and pre-outbreak status (1.00/0.41) remained challenging, although asymptomatic transmission was extracted perfectly for both pathogens (1.00/1.00). The overall extraction performance averaged 0.85 precision and 0.73 recall for Lassa, and 0.84 precision and 0.78 recall for Zika, suggesting reliable field-level accuracy with remaining gaps concentrated in context-dependent and location-specific attributes.

###### E.3 Article Screening Strategy Ablations

This section gives the precision, recall and F1 values for the article screening ablation strategies summarised in Figure 3.

###### Title and Abstract Screening

- Table 23 summarises title-and-abstract screening performance across seven pathogens, showing

moderate overall recall (0.72) alongside high precision (0.79), for an overall F1 of 0.74. This pattern suggests the abstract-stage screening is tuned toward specificity, prioritising the rejection of irrelevant studies at the cost of more false negatives. Performance is broadly consistent across pathogens, with the strongest balance for MERS (F1 of 0.78) and the weakest for Marburg (F1 of 0.69).

- Table 23: Precision, recall, and F1 for title-and-abstract screening across seven pathogens with AgentSLR harness (gpt-oss-120b). Metrics summarise how well the abstract-stage classifier retained studies judged relevant under PERG screening criteria, reported for each pathogen and overall. P = precision; R = recall; F1 = F1-Score.

Pathogen P R F1

Marburg 0.80 0.64 0.69 Ebola 0.74 0.75 0.75 Lassa 0.82 0.72 0.75 SARS 0.78 0.76 0.77 Zika 0.73 0.77 0.75 MERS 0.83 0.74 0.78 Nipah 0.84 0.66 0.70

Overall 0.79 0.72 0.74

Full-text Screening

- Table 24 compares three full-text screening strategies and highlights a clear precision-recall trade-off. Human abstract → AI full-text achieves the strongest overall performance (precision 0.83, recall 0.92), while the two-stage strategy (AI abstract → AI full-text) shows lower recall (0.81), consistent with error propagation from abstract gating. Direct AI full-text screening improves recall (0.89) but reduces precision (0.68), reflecting a recall-maximising approach when abstracts are treated as an information bottleneck.

- Table 24: Full-text screening performance on AgentSLR harness (gpt-oss-120b) under three operational strategies for identifying relevant articles. Metrics compare a two-stage AI pipeline (AI abstract→AI full-text), a mixed workflow (human abstract→AI full-text), and direct AI full-text screening, reported for each pathogen and overall. R = recall; F1 = F1-Score.

|AI Screen (Abstract) → AI Screen (Full-text)|Human Screen (Abstract) → AI Screen (Full-text)|AI Screen (Direct Full-text)|
|---|---|---|
|P R F1|P R F1|P R F1|

Pathogen

Marburg 0.75 0.76 0.75 0.77 0.83 0.80 0.64 0.82 0.69 Ebola 0.73 0.84 0.77 0.86 0.97 0.91 0.67 0.93 0.72 Lassa 0.79 0.78 0.78 0.83 0.94 0.88 0.71 0.91 0.77 SARS 0.71 0.85 0.76 0.80 0.95 0.86 0.64 0.91 0.68 Zika 0.66 0.79 0.69 0.81 0.91 0.85 0.64 0.85 0.67 MERS 0.76 0.83 0.79 0.83 0.96 0.88 0.69 0.95 0.76 Nipah 0.87 0.84 0.85 0.89 0.90 0.90 0.74 0.88 0.79

Overall 0.75 0.81 0.77 0.83 0.92 0.87 0.68 0.89 0.73

##### F Extended Expert Validation Results

Six epidemiology researchers contributed to our validation survey. We collected 62 submissions for parameters, 50 for models, and 31 for outbreaks. Table 25 reports all metrics collected from the survey. The validation survey was designed as an expert audit of AgentSLR harness outputs, not as a duplicate annotation study of the PERG reference set. Each item contained the source article and one LLM extraction. Experts judged whether the extraction was relevant and whether populated fields were correct. The sampling was not designed to provide systematic overlap of the same PERG fields across multiple raters. We therefore do not report Cohen’s κ for PERG annotations. This limits our ability to separate model errors from residual reference annotation noise. To reduce schema-level noise, we filter reference entries with invalid enum values before evaluation and report the filtering rates in Appendix C.2. We interpret the PERG data as peer-reviewed reference annotations rather than as an errorless target.

- Table 25: Extended expert validation results. Results are reported as expert-rated flagging precision and expert-rated extraction accuracy. Within each section, rows are ordered from overall scores to subgroup scores and then field-level scores.

Item Score Parameters

Overall — Flagging precision 0.66 Overall — Extraction accuracy 0.77 Precision by class

Attack rate 0.25 Growth rate 1.00 Human delay 0.62 Reproduction number 1.00 Seroprevalence 0.50 Severity 0.57

Accuracy by group

Value 0.89 Uncertainty 0.76 Population 0.59 Aggregation 0.83

Value fields

Value 0.81 Unit 0.96 Type 0.88 Bounds 0.79 Value type 0.90 Statistical approach 0.97

Uncertainty fields

Single-type uncertainty 0.88 Paired uncertainty 0.84 Distribution type 0.57

Population fields

Sample type 0.74 Population group 0.49 Sample size 0.66 Sex 0.50 Age range 0.58 Countries 0.82 Locations 0.71 Method moment value 0.23

Aggregation fields

Aggregation 0.83 Models

Overall — Flagging precision 0.40 Overall — Extraction accuracy 0.83 Field accuracy

Model type 0.89 Continued on next page

Table 25 continued from previous page Item Score

Compartmental type 0.89 Stochastic or deterministic 0.70 Theoretical model 0.84

###### Outbreaks

Overall — Flagging precision 0.61 Overall — Extraction accuracy 0.80 Accuracy by group

Temporal 0.62 Geographical 0.87 Case burden 0.85 Epidemiological 0.85

Temporal fields

Start year 0.84 Start month 0.70 Start day 0.62 End year 0.50 End month 0.60 End day 0.56 Duration in months 0.50

Geographical fields

Country 0.95 Location 0.80

Case burden fields

Confirmed cases 0.88 Suspected cases 0.64 Asymptomatic cases 1.00 Severe cases 1.00 Deaths 0.71

Epidemiological fields

Mode of detection 0.82 Pre-outbreak status 0.82 Asymptomatic transmission described 0.89

For flagging (sub-task) precision, models and outbreaks are reported only at the overall level (as in the main text). For parameters, precision is averaged over flagging decisions made for each parameter class. The random subsample of articles assigned gave six relevant parameter classes: attack rate, growth rate, human delay, reproduction number, seroprevalence, and severity. The remaining two parameter classes, mutation rate and relative contribution, were absent from the sample. Since there is a flagging decision made for each parameter class on each article, each parameter class-level precision is calculated over the same sample size (N = 62).

For extraction accuracy, the aggregate statistics are normalised over groups of similar fields. For example, outbreaks have clusters of fields related to temporal features (start date, end date, and duration), geographical features (country and location), case burden (case counts and fatalities) and epidemiological factors (mode of detection, status pre-outbreak, and asymptomatic transmission). We normalise at the group level to treat each aspect of the extraction as equally important, in order to avoid overemphasising groups with larger numbers of metadata fields. We omit group-level normalisation for models owing to the smaller number of validated fields.

The disaggregated statistics reveal findings that are masked in the average statistics. For example, among parameter classes, AgentSLR workflow (with gpt-oss-120b) performs worst on flagging attack rate (experts reported several instances where the system confused attack rate with seroprevalence information). At the field level, the LLM struggles the most with understanding parameter population context and with the temporal outbreak features (group accuracy 0.62). Parameter population fields are multiple-choice selections with many options (see Table 33), and these designations often have specific interpretations in epidemiology. For example, “persons under investigation” is a population group of patients exhibiting clinical and epidemiological risk factors, a definition that an LLM may struggle to apply consistently in different article contexts.

##### G Article Search and Retrieval

This section details the search query construction, database-specific adaptations, and PDF retrieval strategy used for article acquisition across priority pathogens in the AgentSLR workflow. Following the Pathogen Epidemiology Review Group (PERG) methodology8, we developed a standardised base query structure that captures core epidemiological domains including transmission dynamics, disease severity, temporal parameters, transmission heterogeneity, and evolutionary characteristics.

Different bibliographic databases support different search capabilities, requiring tailored query implementations. We maintain two versions of each pathogen query: one for PubMed and Europe PMC (which support wildcard truncation operators using *), and another for OpenAlex (which requires fully expanded term variants).

###### G.1 Base Search Query (PubMed and Europe PMC)

The base query for PubMed and Europe PMC uses Boolean operators with truncation symbols to capture morphological term variations:

[PATHOGEN_IDENTIFIER] AND ( (transmissi* OR epidemiolog*) OR (model* NOT imag*) OR (severity OR "case fatality ratio*" OR CFR OR "case fatality rate*"

OR "mortality rate*" OR "attack rate*") OR

("infectious period*" OR "serial interval*" OR "incubation period*" OR "generation time*" OR "generation interval*" OR "latent period*" OR latency) OR

(heterogeneit* OR superspread* OR "super spread*" OR super-spread* OR overdispersion OR overdispersed OR over-dispersion OR over-dispersed OR "over dispersion" OR "over dispersed") OR

(infectivity OR infectiousness OR "growth rate*" OR "reproduction number*" OR "reproductive number*" OR R0 OR "reproduction ratio*" OR "reproductive rate*") OR

("pre-existing immunity" OR serological OR serology OR serosurvey*) OR (evolution* OR mutation* OR substitution*) OR (outbreak* OR cluster* OR epidemic*) OR ("risk factor*") [ADDITIONAL_TERMS]

) [EXCLUSION_CRITERIA]

###### G.2 OpenAlex Adapted Queries

Because the OpenAlex API does not support wildcard operators9 and strips these characters during query processing, we expanded all truncated terms into their common morphological variants:

[PATHOGEN_IDENTIFIER] AND (

(transmission OR transmissibility OR transmissible OR transmitted OR transmitting OR transmit OR epidemiology OR epidemiological OR epidemiologic) OR

(model OR models OR modeling OR modelling OR modeled OR modelled NOT (image OR images OR imaging)) OR

(severity OR "case fatality ratio" OR "case fatality ratios" OR CFR OR "case fatality rate" OR "case fatality rates" OR "mortality rate" OR "mortality rates" OR "attack rate" OR "attack rates") OR

("infectious period" OR "infectious periods" OR "serial interval" OR "serial intervals" OR "incubation period" OR "incubation periods" OR "generation time" OR "generation interval" OR "generation intervals" OR "latent period" OR "latent periods" OR latency) OR

(heterogeneity OR heterogeneous OR superspread OR superspreader OR superspreaders OR superspreading OR "super spread" OR "super spreader" OR "super spreaders" OR "super spreading" OR overdispersion OR overdispersed OR "over dispersion"

- 8https://github.com/mrc-ide/priority-pathogens/wiki/Search-terms
- 9https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities

OR "over dispersed") OR

(infectivity OR infectiousness OR "growth rate" OR "growth rates" OR "reproduction number" OR "reproduction numbers" OR "reproductive number" OR "reproductive numbers" OR R0 OR "reproduction ratio" OR "reproduction ratios" OR "reproductive rate" OR "reproductive rates" OR "basic reproduction number") OR

("pre-existing immunity" OR serological OR serology OR serosurvey OR serosurveys OR seroprevalence OR serosurveillance) OR

(evolution OR evolutionary OR evolving OR evolved OR mutation OR mutations OR mutant OR mutants OR mutate OR mutated OR substitution OR substitutions) OR

(outbreak OR outbreaks OR cluster OR clusters OR clustering

OR epidemic OR epidemics OR pandemic OR pandemics) OR ("risk factor" OR "risk factors") [ADDITIONAL_TERMS]

) [EXCLUSION_CRITERIA]

###### G.3 Pathogen-Specific Query Modifications

- Table 26 summarises the pathogen-specific modifications applied across all database implementations. Most pathogens require only customised identifiers to ensure relevant literature retrieval. However, the queries for SARS explicitly exclude COVID-19 literature to prevent cross-contamination with SARS-CoV-2 studies. Similarly, queries for Zika include vector-specific epidemiological parameters (extrinsic incubation period, vector competence) that are essential for capturing mosquito-borne transmission dynamics. For Rift Valley fever, Crimean-Congo hemorrhagic fever (CCHF) and MERS, we incorporated additional virus-specific identifiers and spelling variants to enhance retrieval comprehensiveness. Despite these modifications, all databases share consistent pathogen identifiers and exclusion criteria, differing only in their use of wildcard forms (PubMed/Europe PMC) versus expanded term variants (OpenAlex).

- Table 26: Pathogen-specific modifications to the standardised search query. All databases share consistent pathogen identifiers and exclusion criteria; PubMed/Europe PMC use wildcard forms while OpenAlex uses expanded variants.

Pathogen PATHOGEN_IDENTIFIER ADDITIONAL_TERMS EXCLUSION_CRITERIA

Marburg virus Marburg virus — Ebola virus Ebola — Lassa virus Lassa — SARS-CoV-1 SARS OR SARS-CoV-1 OR “Severe acute

— NOT (COVID-19 OR SARSCoV-2)

respiratory syndrome"

Zika virus zika OR (“extrinsic incubation period" OR “EIP" OR “vector competence" OR “vectorial capacity")†

—

Nipah virus Nipah — MERS-CoV MERS OR MERS-CoV OR “Middle East

— —

respiratory syndrome" OR “Middle East Respiratory Syndrome Coronavirus"‡

Rift Valley fever virus “Rift valley fever" OR RVF OR “Rift Valley Fever Virus" OR RVFV‡

— —

CCHF virus “Crimean Congo haemorrhagic fever" OR “Crimean-Congo hemorrhagic fever" OR CCHF OR “CCHF virus" OR CCHFV‡

— —

†Vector-specific terms capture mosquito transmission parameters unique to arboviral epidemiology. ‡Expanded identifiers include alternative spellings (American/British English), virus-specific nomenclature, and common abbreviations for comprehensive coverage.

###### G.4 Metadata Extraction and Deduplication

We extract bibliographic metadata from each database as summarised in Table 27. OpenAlex provides direct PDF URLs and internal work identifiers, PubMed supplies standardised medical literature identifiers (PMID: PubMed ID; PMCID: PubMed Central ID), and Europe PMC offers full-text availability metadata. The Digital Object Identifier (DOI) serves as a persistent identifier across databases.

We implement a hierarchical five-level deduplication strategy:

- 1. DOI-based: Normalised DOI strings (case-insensitive, URL prefixes stripped);
- 2. PMID-based: Numeric PMID extraction and normalisation;
- 3. PMCID-based: Normalised PMC identifiers (uppercase, “PMC” prefix standardised);
- 4. OpenAlex ID-based: Internal OpenAlex work identifiers;
- 5. Title-year combination: Normalised title strings (lowercase, alphanumeric only) paired with publication year.

When duplicate records are detected, identifier fields (DOI, PMID, PMCID, OpenAlex ID, URLs) preserve all non-null values while narrative fields (title, abstract, journal) retain the first non-null value. Source provenance is marked as “Both" when records appear in multiple databases.

- Table 27: Metadata fields extracted during article search. PMID: PubMed ID; PMCID: PubMed Central ID; DOI: Digital Object Identifier.

Field Description

article_id Generated unique identifier source Database origin pmid PubMed Identifier pmcid PubMed Central Identifier doi Digital Object Identifier title Article title authors Semicolon-delimited author list journal Publication venue year Publication year abstract Article abstract url Canonical article URL openalex_id OpenAlex work identifier openalex_pdf_url Direct PDF link from OpenAlex pathogen Target pathogen query Search query used harvested_at ISO 8601 timestamp

Table 28: Additional fields populated during PDF retrieval attempts.

Field Description

downloaded Boolean success flag downloaded_path Filesystem path to PDF download_source Source that provided PDF download_attempted_at ISO 8601 timestamp download_error Error messages from attempts

###### G.5 PDF Retrieval

We attempt PDF downloads through multiple open access sources using a cascading retrieval strategy. Before attempting downloads, available identifiers (PMID, PMCID, DOI) are cross-referenced using NCBI’s PMC ID Converter API10 to maximise source compatibility. The system then attempts downloads from up to four sources in priority order (Table 29), stopping at the first successful retrieval.

###### G.5.1 Implementation Details

Downloads employ HTTP streaming to temporary files with 64 KB chunks and validate each file through two stages: (1) magic byte verification (%PDF header), and (2) content inspection for HTML access denial pages. Files exceeding 500 MB or failing validation are immediately discarded. Threadpool parallelism with 16 workers processes downloads concurrently while respecting per-source

10https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/

- Table 29: PDF retrieval sources in cascading priority order. Sources are queried sequentially until success or exhaustion. Identifier cross-referencing via NCBI PMC ID Converter API precedes all download attempts (10 req/s, cached).

Priority Source & Endpoint Rate Limit Cached

- 1 OpenAlex Direct PDF URL 30 req/s No Metadata field openalex_pdf_url

- 2 Europe PMC Fulltext API 20 req/s Yes ebi.ac.uk/europepmc/webservices/rest/search

- 3 Unpaywall API 50 req/s Yes api.unpaywall.org/v2/{DOI}?email={EMAIL}

- 4 OpenAlex DOI Lookup 30 req/s Yes api.openalex.org/works/https://doi.org/{DOI}

rate limits. In-memory caches keyed by normalised identifiers store both successful PDF URLs and negative markers to eliminate redundant API calls. Progress is checkpointed every 50 records for crash recovery.

Successfully validated PDFs are saved with standardised filenames following identifier priority (PMID → PMCID → DOI hash → title hash). Metadata is augmented with download provenance including source, timestamp, and error diagnostics.

###### G.6 Final Quality Control

After retrieval, we applied deduplication and quality filtering that removes: records lacking abstracts, duplicate article IDs, duplicate DOIs (retaining first occurrence) and records with file validation failures.

##### H Article Screening Criteria and Prompts

Following article search and retrieval, the articles are screened for relevance to the study. The screening is conducted on abstracts, and then on full-text articles. We present the study objectives, inclusion and exclusion criteria, along with the detailed prompts used to screen for relevant priority pathogen articles. Through prompt sensitivity tests with the open-source models under evaluation, similar to Alhetelah and Ahmad [46] but replacing yes/no with include/exclude, we settle on using a modified version of the prompt from ScreenPrompt [14]. The prompts follow a structured format: basic instruction, study objectives, inclusion/exclusion criteria, article content, and chain-of-thought screening instructions with parsable output request.

Study Objectives This systematic review aims to collate transmission and modelling parameters for {pathogen_name}. The review seeks to:

- 1. Provide estimates of key infectious disease metrics (reproduction number, CFR, generation time, serial interval, incubation period, etc.)
- 2. Document historical outbreak characteristics (size, location, duration, deaths)
- 3. Identify mathematical/statistical models of transmission
- 4. Collate risk factors for infection, severe disease, and death
- 5. Summarize seroprevalence data
- 6. Support infectious disease modelling and outbreak response efforts

This information enables effective outbreak preparedness, resource targeting, and mathematical modelling for nowcasting and forecasting of {pathogen_name}.

###### Inclusion Criteria ALL must be met:

- 1. Pathogen: Must be about {pathogen_name}
- 2. Language: English only
- 3. Study type: Peer-reviewed, original research (note systematic reviews/meta-analyses for special consideration)
- 4. Population: Human subjects (animal studies acceptable if reporting EITHER: (a) transmission parame-

ters: R0, Rt, Re, r, growth rate, mutation rate, OR (b) vector parameters: extrinsic incubation period, vector reproduction numbers, vector competence, mosquito delays)

- 5. Content: Must contain AT LEAST ONE of:

- (a) Quantitative details of concluded/ongoing human outbreak (size, year, location, duration, spatial scale)
- (b) Mathematical or statistical model of disease transmission
- (c) Measures/estimates of transmission parameters: R, R0, Rt, r, Re, growth rate, doubling time
- (d) Measures/estimates of timing parameters: generation time, serial interval, incubation period, latent period, infectious period
- (e) Measures/estimates of severity: CFR, IFR, hospitalization rate, mortality rate, attack rate
- (f) Measures/estimates of genetic evolution: mutation rate, substitution rate, evolutionary rate
- (g) Measures of overdispersion or superspreading (k parameter, transmission heterogeneity)
- (h) Seroprevalence data or serological surveys
- (i) Risk factors for infection, severe disease, death, or hospitalization (with statistical measures)
- (j) Measures/estimates of vector parameters: extrinsic incubation period (EIP), mosquito reproduction numbers, vector competence, mosquito delays, or relative transmission contributions (human-tohuman vs vector-borne/zoonotic)

Full-text only

- 6. Data Extraction Requirement: Must contain extractable mathematical models, transmission models, or quantitative parameter estimates (with values or ranges) for disease modeling. This includes: reproduction numbers, transmission rates, incubation periods, case fatality ratios, model structures, intervention effects, or other modeling parameters. Articles without extractable quantitative parameters or models should be excluded.

###### Title & Abstract Screening Prompt

You are an expert epidemiologist screening full-text articles for a systematic review on the target pathogen.

Study Objectives [See Study Objectives above]

###### Screening Criteria

The following is an excerpt of 2 sets of criteria. A study is considered included if it meets ALL inclusion criteria. If a study meets ANY exclusion criteria, it should be excluded. Here are the 2 sets of criteria:

Inclusion Criteria [See Inclusion Criteria 1–5 above]

Exclusion Criteria Exclude if ANY apply:

- 1. Pathogen: Not about {pathogen_name} (excludes studies on other pathogens)
- 2. Language: Non-English
- 3. Publication type: Conference proceedings, abstract-only, posters, correspondence
- 4. Study design: In-vitro studies only (no human or animal component)
- 5. Study design: Solely animal studies AND animal studies that do not report transmission parameters (R0, Rt, Re, r, growth rate, mutation rate)
- 6. Outbreak type: Accidental laboratory outbreaks (not natural disease transmission)

Abstract (To Screen) Title: {{title}} Abstract: {{abstract}}

###### Screening Instructions

We now assess whether the paper should be included in the systematic review by evaluating it against each and every predefined inclusion and exclusion criterion. First, we will reflect on how we will decide whether a paper should be included or excluded. Then, we will think step by step for each criterion, giving reasons for why they are met or not met.

Studies that may not fully align with the primary focus of our inclusion criteria but provide data or insights potentially relevant to our review deserve thoughtful consideration. Given the nature of abstracts as concise summaries of comprehensive research, some degree of interpretation is necessary.

Our aim should be to inclusively screen abstracts, ensuring broad coverage of pertinent studies while filtering out those that are clearly irrelevant.

We will conclude by outputting (on the very last line) <decision>EXCLUDE</decision> if the paper warrants exclusion, or <decision>INCLUDE</decision> if inclusion is advised or uncertainty persists.

Finally, the articles that pass the abstract screening have their full text screened as follows.

###### Full-Text Screening Prompt

You are an expert epidemiologist screening abstracts for a systematic review on the target pathogen.

Study Objectives [See Study Objectives above]

###### Screening Criteria

The following is an excerpt of 2 sets of criteria. A study is considered included if it meets ALL inclusion criteria. If a study meets ANY exclusion criteria, it should be excluded. Here are the 2 sets of criteria:

Inclusion Criteria [See Inclusion Criteria 1–6 above, including full-text criterion]

Exclusion Criteria Exclude if ANY apply:

- 1. Not about {pathogen_name} (excludes other pathogens)
- 2. Non-English language
- 3. Conference proceedings, abstract-only, posters, correspondence, Literature reviews, metaanalyses
- 4. In-vitro studies only (no human or animal component)
- 5. Animal studies without transmission parameters (R0, Rt, Re, r, growth rate, mutation rate) or solely animal studies.
- 6. Case studies/reports with <10 human cases
- 7. Accidental laboratory outbreaks

Full-Text Article (To Screen) Title: {{title}} Full Text: {{fulltext}}

###### Screening Instructions

We now assess whether the paper should be included in the systematic review by evaluating it against each and every predefined inclusion and exclusion criterion. First, we will reflect on how we will decide whether a paper should be included or excluded. Then, we will think step by step for each criterion, giving reasons for why they are met or not met.

Critically evaluate: Does this paper contain extractable quantitative data, models, or parameters relevant to disease transmission and outbreak response? This is essential for inclusion.

We will conclude by outputting (on the very last line) <decision>EXCLUDE</decision> if the paper warrants exclusion, or <decision>INCLUDE</decision> if inclusion is advised or uncertainty persists.

##### I Data Extraction Process

After screening, the finalised pool of relevant articles underwent rigorous data extraction. This extraction stage evaluates the use of structured tool-calling to extract three categories of data: epidemiological parameters, transmission models and outbreak data from full-text articles. Each category followed a multi-stage workflow with validation on each tool output.

###### I.1 Parameters

Valid Epidemiological Parameters for Extraction Epidemiological parameters are quantitative summaries of how an infection behaves in a population, such as its rate of spread, the delays between key stages of infection, the infection and fatality rates, and risk factors across demographic groups. We used PERG’s data entry tool, a REDCap survey, as the reference list of epidemiological quantities that human reviewers would extract from the literature.11 This gave a fixed catalogue of 47 parameter types that cover mutation processes, transmission intensity, delay distributions in humans and mosquitoes, severity, seroprevalence, and risk factors. These higher-order groupings are labelled parameter classes, and AgentSLR defines data extraction criteria at the parameter class-level.

- Table 30 lists all parameter types targeted by our pipeline, together with brief definitions that match the guidance given to human experts.

###### Table 30: Valid parameters for extraction, according to PERG’s process.

Parameter type Parameter class Description Attack rate Attack rate Proportion of a population that becomes infected during

an outbreak. Secondary attack rate Attack rate Proportion of contacts of a primary case who become

infected. Doubling time Doubling time Time required for the number of infections to double. Growth rate Growth rate Exponential rate at which new infections increase over

time. Evolutionary rate Mutations Rate of genetic change in a population over time, typically substitutions per site per year. Mutation rate Mutations Frequency at which new genetic mutations arise per site per replication cycle. Substitution rate Mutations Speed at which mutations become fixed in a population’s genome. Generation time Human delay Average interval between infection in a case and infection in a secondary case. Serial interval Human delay Time between symptom onset in a primary and secondary

case. Latent period Human delay Time from infection to becoming infectious. Incubation period Human delay Time from infection to symptom onset. Infectious period Human delay Duration during which an infected person can transmit

the pathogen. Time in care Human delay Average duration of hospitalisation or clinical care. Symptom onset → admission to care Human delay Time from symptom onset to hospital or clinical admis-

sion. Symptom onset → discharge / recovery Human delay Time from symptom onset to recovery or discharge. Symptom onset → death Human delay Time from symptom onset to death. Admission → discharge / recovery Human delay Time from hospital admission to recovery or discharge. Admission → death Human delay Time from hospital admission to death. Other human delay Human delay Other reported delays related to human infection or re-

sponse. 11https://redcap.imperial.ac.uk/surveys/?s=CEX3YKW8W47NMFA4

Overdispersion Overdispersion Measure of variation in the distribution of individual infectiousness.

Human-to-human Relative contribution Proportion of total transmission attributable to human-tohuman spread.

Zoonotic-to-human Relative contribution Proportion of total transmission from animal or vector sources to humans.

Basic (R0) Reproduction number Average number of secondary cases from one case in a fully susceptible population.

Effective (Re) Reproduction number Average number of secondary cases in a population with

partial immunity or interventions. Case fatality rate (CFR) Severity Proportion of diagnosed cases that result in death. Infection fatality rate (IFR) Severity Proportion of all infections (symptomatic and asymp-

tomatic) that result in death. Proportion of symptomatic cases Severity Proportion of infections that develop symptoms. IgM Seroprevalence Proportion of individuals with detectable IgM antibodies,

indicating recent infection. IgG Seroprevalence Proportion of individuals with IgG antibodies, indicating past infection or immunity. PRNT Seroprevalence Proportion with neutralising antibodies detected by plaque reduction neutralization test. HAI/HI Seroprevalence Proportion with antibodies detected by hemagglutination inhibition assay. IFA Seroprevalence Proportion with antibodies detected by immunofluores-

cence assay. Unspecified Seroprevalence Seroprevalence reported without specifying assay type. Risk factors Risk factors Host, environmental, or behavioural characteristics asso-

ciated with infection risk.

Multi-Stage Parameter Extraction Pipeline Parameter extraction utilises a five-step workflow that mirrors how a careful human reader would process scientific articles. Starting from full-text contents, we identify relevant estimates in the text, extract them into a standardised format, and collect relevant metadata about population context and parameter uncertainty.

For our first step, we ask a reasoning language model with tool calling to “screen” each article for each parameter class. The reasoning model is provided with a tool to extract (potentially discontiguous) quotations from the source text that relate to the parameter class. We provide specific details for each parameter class as displayed in Table 31, which are copied quotations from the parameter extraction documentation from the priority-pathogens codebase [35], accessible at https://github.com/mrcide/priority-pathogens/wiki/Parameter-Data.

- Table 31: Screening details for each parameter class. This is inputted into the “Parameter Class Screening Details” section of the Parameter Screening Prompt below.

Parameter Class Screening Details Attack rate The attack rate is the proportion of an at-risk population contracting the disease during a

specified time interval. It is often reported as a percentage or rate, e.g. 52 people per 10,000 people.

Growth rate The epidemic growth rate is a key metric that reflects how quickly the number of infections is changing day by day in a population. It is a time-dependent measure, usually expressed as a percentage or a rate per unit of time (e.g. per day), and is crucial for monitoring the speed and trajectory of an outbreak.

Human delay These parameters all refer to time intervals in the natural history of infection of the host. Mutation rate Mutation rates, like substitution rate or evolutionary rate, describe the speed at which genetic

changes accumulate in a population.

Relative contribution This parameter is intended for pathogens (e.g. MERS) where there is both human to human (h2h) and animal to human (a2h) transmission, and aims to capture the relative magnitude of these two routes of infection in humans. We expect these to be proportions or percentages. E.g. a study might estimate 60% of infections in humans to be from h2h infection.

Reproduction number We are extracting either the basic reproduction number R0 or the effective reproduction number Re.

Risk factors We are extracting general information about risk factors in the included papers. We are extracting both univariate (naive) and multivariate (adjusted) risk factors, even if they are both available.

Seroprevalence These parameters refer to estimations of seroprevalence in the paper. This may also be referred to as antibody prevalence. These parameters will all be expressed in a proportion or percentage of the population.

Severity Severity refers to either the case fatality ratio or the infection fatality ratio. The case fatality ratio is the proportion of cases who end up dying of the disease. Note this depends on the case definition used, as the denominator is people identified as “cases”. The infection fatality ratio is the proportion of infections who end up dying of the disease.

The model is also provided with the study objectives from Section H and instructed to only extract parameters “estimated from or fitted to actual data”. If no relevant information is found, the model is told to refrain from calling the tool. The full prompt for this step is templatised as follows:

###### Parameter Screening Prompt

You are an expert epidemiologist extracting epidemiological parameters from scientific articles. You will be provided with the processed text of a scientific article. Your task is to extract information about epidemiological parameters according to the provided schema.

Study Objectives See study objectives in Section H.

###### Summary Extraction Task Definition

For your first task, you will be provided with the full text of a scientific article and a specific type of parameter. We are only extracting parameters that are estimated from or fitted to actual data. For transmission models, if it is only a theoretical model and they have just chosen parameters from other studies/randomly, then please don’t extract these.

Your task is to scan the provided text and determine whether this article estimates any parameters of the provided type. If it does, you must use the provided tool to extract relevant summaries from the text about this parameter. If the article makes no mention of the parameter, simply do not call the tool.

If there are multiple pieces of information about the same parameter, return them as separate list items. You will need to call the tool multiple times if there are multiple separate parameter estimates of the provided type.

In future steps, we will be using the provided summaries to extract structured information about the parameter, including:

- (a) The estimated value
- (b) Uncertainty intervals
- (c) Sample study population

Please make sure your summaries provide all of this information if it is provided. Please be thorough: err on the side of extracting more information rather than less.

###### Parameter Class Screening Details

See the details provided for each parameter class in Table 31.

Full Text Title: {{title}} Full Text: {{fulltext}}

Our next steps are executed for each value of the summaries array returned by the model’s tool call. We prompt the model in a new context, omitting the full text, to focus the model on the relevant text snippets from summaries and to save both inference time and API cost. If no relevant parameters are identified for a given article, summaries will be empty, and the extraction process will terminate.

Otherwise, we move to our second step, value extraction. At this step, the model utilises the value_info of the parameter to extract structured information about its value and uncertainty bounds. As before, we provide instructions for using the tool for each parameter class. These are listed below:

###### Value Extraction Details for Attack rate

If the attack rate is reported as a percentage, extract the percentage in the value field and set unit to percentage. If the attack rate is reported as a rate, extract the numerator in the value field and set rate_denominator to the denominator of the rate.

Please extract attack rates as written in the paper.

###### Value Extraction Details for Growth rate

Please extract growth rates from the paper. Populate the value field with a numerical value as it is specified in the paper. If the paper provides a percentage value like 33%, record this value as 0.33. Populate the unit field with one of the provided units according to the tool schema.

Value Extraction Details for Human delay Delay type The delay_type field records the specific type of time interval. It can take one of the following values:

- • generation_time: The generation time is the time interval between infector exposure to infection and infectee exposure to infection. It may be used in reproduction number estimation, but given the difficulties in its observation, it may be replaced by the serial interval (see below).
- • serial_interval: The serial interval is the time interval between infector symptom onset and infectee symptom onset. It is frequently used in reproduction number estimation, as a substitute for the generation time.
- • latent_period: The latent period is the time interval between exposure to infection and becoming infectious. It is sometimes used interchangeably with the incubation period (see below). It may also be referred to as the latency period or the pre-infectious period.
- • incubation_period: The incubation period is the time interval between exposure to infection and symptom onset. It often coincides with the latent period, but may be shorter (symptom onset before infectiousness, e.g. SARS) or longer (infectiousness before symptom onset, e.g. Covid-19). It may also be referred to as the intrinsic incubation period (in the context of vector-borne diseases) or a subclinical infection.
- • infectious_period: The infectious period is the time interval during which the host remains infectious. It directly follows the latent period (see above). It may also be referred to as the infective period, the contagious period, the transmission period or the communicability period.
- • time_in_care: The time in care is the time interval between admission to care and discharge from care or death. Unless there is a delay in receiving care, it directly follows the time from symptom to careseeking. It may vary according to health outcome and is typically highly skewed. It may also be referred to as the length of stay (LOS). Human delays other than the six listed above may also be reported, for example the time from symptom onset to recovery, symptom onset to death, time from seeking care to admission to care etc. We allow delay_type to take on one of these other time interval values:
- • admission__to__death
- • admission__to__discharge_or_recovery
- • symptom_onset__to__admission
- • symptom_onset__to__death
- • symptom_onset__to__discharge_or_recovery

In the case that none of the above values apply to a human delay parameter you have found, set delay_type

= ’other’ and record the type of delay in the delay_type_note field.

Value and unit Use the value and unit fields to record the parameter estimate (e.g. x hours, days, weeks, or other).

###### Value Extraction Details for Mutation rate

For this task, we extract parameters estimated from pathogen genetic sequences. If no parameters were derived from genetic sequences, then this section can be skipped even if sequencing was performed and reported.

substitution_rate, evolutionary_rate, and mutation_rate are different parameter_type values for describing the speed at which genetic changes accumulate in a population. When selecting the parameter_type, choose the value type and units based on the wording used by the authors in the article. If there are multiple terms used for the same measure (e.g. substitution rate is used in the text, evolutionary rate is used in the table), choose either the most frequently used term or default to substitution_rate (if the units are substitutions per site per year). These values are often in the supplemental material. So if genetic sequences or phylogenetic analyses are presented, check the supplement. We are not extracting parameters associated with selection pressure or synonymous/nonsynonymous mutations, unless based on data or methodological limitations they have only been able to calculate substitution rate from nonsynonymous mutations (in that case specify this in the ‘Gene’ field, similar to in vitro experiments - see next bullet point). If substitution rates are calculated for subgroups (e.g. ‘clades,’ ‘strains,’ ‘branches’, etc), report the global estimate and indicate disaggregated data is available in the Parameter Disaggregation section.

As always, the unit value is very important for these parameters. The most common unit is substitutions_per_site_per_year. If units are not clear or they do not match the available options in the drop-down menu, set to unspecified.

Fill the genome_site field with the portion of the pathogen’s genome used to estimate any extracted parameters (e.g. reproduction number, growth rate, substitution rate). This can be a gene, a gene segment, a codon position, or a more generic description (e.g. ‘whole genome’ or ‘intergenic positions’). If parameter values are independently estimated for different portions of the genome, please enter each on a separate parameter value form. If a mutation rate is estimated by in vitro experiments of recombinant variants (for example, measuring the rate of mutation in an inserted gene, such as green fluorescent protein [GFP]), enter the name of the inserted gene used, even though this gene might not be naturally occurring in the virus’s genome. In addition, they may measure different types of mutations (SNPs vs indels) during in vitro experiments. If this is the case, enter the type of mutation used to calculate the rate (ex. GFP-SNP, to signify that SNP mutations in the GFP gene were used to calculate the mutation rate).

###### Value Extraction Details for Severity

- • parameter_type – we extract case fatality ratios (CFR), infection fatality ratios (IFR), and the proportion of cases that are symptomatic and asymptomatic.

- – Case fatality ratio (CFR) – the proportion of cases who end up dying of the disease. Note this depends on the case definition used, as the denominator is people identified as “cases”. All CFRs should be extracted, even when a subset of the population is selected (e.g. severe cases); make sure to describe the population denominator in the context and notes.
- – Infection fatality ratio (IFR) – the proportion of infections who end up dying of the disease (harder to calculate but less context dependent).
- – Symptomatic proportion of infections – the proportion of total infections that are symptomatic.
- – Asymptomatic proportion of infections – the proportion of total infections that are asymptomatic.

- • Parameter value – we don’t do any calculation ourselves i.e. if a paper quotes number of deaths and number of cases, but not a CFR, we don’t calculate the CFR.
- • Ratio/prevalence values – please extract the numerator and denominator that generate the severity ratio. In line with the rule of 3, only extract the numerator and denominator of the central CFR value, even if disaggregated numerators and denominators are available. If there is no central value, do not extract any numerator or denominator. If the numerator and denominator are presented, but the percentage severity is not, extract the numerator, denominator and context, but leave the central value blank.
- • method – we extract information about the method used to calculate CFR (or IFR), mainly whether it is:

- – a “naive” method, i.e. percentage mortality which computes total deaths divided by total cases (or infections); this is wrong because there may be many cases or infections who do not have final status information, so the naive estimate is typically an underestimate of true CFR (or IFR).
- – an adjusted method, which somehow accounts for infections or cases with unknown final status (e.g. calculates deaths / (deaths + recoveries) or does something more fancy).
- – an unknown method.

- • value_type: mean, median, shape, etc. Please note that it may be the case that multiple measures of central tendency are provided, especially when the entire distribution of a parameter is presented. To avoid extracting multiple measures of centrality for the same parameter and to avoid bias, only one parameter value_type can be extracted. Central parameter types are prioritised based on the available uncertainty types in the following way:

- – When SD/variance/CIs are available: extract mean.
- – Else when only IQR/CrIs are available: extract median.
- – If mode is presented, this should be prioritised after the mean or median.
- – If Weibull distribution parameters are presented: prioritise extraction of the shape rather than mean/CIs or median/CrIs. We can get mean/CIs from shape/scale analytically but can only get shape/scale from mean/CIs numerically.

- • statistical_approach – if the central parameter estimates are summarised directly from empirical data, select observed_sample_statistic. If the central parameter is estimated using a transmission model, select estimated_model_parameter. Due to limited data sources, the Oropouche systematic review only was extended to include case_study data.

The full prompt for the value extraction step is templatised below, incorporating text from both the parameter class screening details and the value extraction details.

###### Value Extraction Prompt

You are an expert epidemiologist extracting epidemiological parameters from scientific articles. You will be provided with the processed text of a scientific article. Your task is to extract information about epidemiological parameters according to the provided schema.

Study Objectives See study objectives in Section H.

###### Value Extraction Task Definition Value extraction task

For your next task, you will be provided with excerpts from a scientific article and a specific type of parameter. We are only extracting parameters that are estimated from or fitted to actual data. For transmission models, if it is only a theoretical model and they have just chosen parameters from other studies/randomly, then please don’t extract these.

Scan the provided text and for the requested parameter and return all estimated parameter values using the provided tool. You will need to call the tool multiple times if there are multiple separate estimates.

Parameter Class Screening Details {{parameter_class}}: parameter value extraction

{{Screening details from Table 31}}

Value Extraction Details for {parameter_class} See the specific details of value extraction above.

Value Excerpts The following are excerpts from the scientific article about parameter value: {{value_info}}

The tool provided to the language model is distinct per parameter class. In Table 32, we specify the schemas utilised for these tool calls.

- Table 32: Schemas used for value extraction tool calls for each parameter class. Here “–” means that any values of the correct type are allowed.

Parameter class Variable Type Allowed values Description Attack rate value Float – The value of the attack rate.

unit Enum percentage; rate The unit of the provided attack rate. type Enum primary; secondary Whether primary or secondary

attack rate. rate denominator Integer;

– The denominator of the value, if the parameter is provided as a rate.

Null

Doubling time value Float – The value of the doubling time, in days.

Growth rate value Float – The value of the growth rate.

unit Enum per hour; per day; per week; per month; per year; other; unspecified

The unit of the provided growth rate.

Human delay value Float – The value of the human delay parameter.

delay type Enum admission to death; admission to discharge or recovery; generation time; incubation period; infectious period; serial interval; symptom onset to admission; symptom onset to death; symptom onset to discharge or recovery; time in care; other

The specific delay parameter reported.

Mutation rate value Float – The value of the mutation rate

parameter. type Enum evolutionary rate; mutation

The specific mutation rate parameter reported.

rate; substitution rate

unit Enum substitutions per site per year; mutations per genome per generation; percentage; other; unspecified

The unit of the mutation rate parameter value.

genome site String – The specific genome site or region associated with the mutation rate value.

Overdispersion value Float – The value of the overdispersion

parameter unit Enum no units; max number of

The unit of the overdispersion parameter

cases superspreading

Relative contribution

value Float – The value of the relative contri-

bution parameter. type Enum human-to-human; zoonotic-

The type of relative contribution reported.

to-human

Reproduction number

value Float – The value of the reproduction number parameter. type Enum basic R0; effective Re The type of reproduction number reported.

transmission Enum human; mosquito; unspecified; other

The type of transmission for this reproduction number estimate.

method Enum branching process; growth rate; compartmental model; next generation matrix; empirical; genomic; other

The method used to obtain the reproduction number estimate.

The name of the risk factor.

Risk factors name List[Enum] age; close contact; breastfeeding; comorbidity; contact with animal; environmental; funeral; hospitalisation; household contact; humidity; non-household contact; occupation; prior immunity to arboviruses; rainfall; sex; social gathering; temperature; other

outcome List[Enum] death in general population; Guillain Barre Syndrome; infection; low birthweight; microcephaly; miscarriage or stillbirth; other neurological symptoms in general population; premature birth; serology; severe disease in general population; spillover risk; recovery; Zika congenital syndrome or other birth defects; other

The outcome for which the risk factor was evaluated.

occupation List[Enum] abattoir services; correctional facilities; education; funeral and burial services; healthcare; laboratory; livestock and animal herders; public transport; quarantine facilities; veterinary; other; unspecified

If name is set to ‘occupation’, the occupation(s) that correspond(s) most closely to that described in the paper.

significant Enum significant; not significant; unspecified

Whether the risk factor is significant or not.

adjusted Enum adjusted; not adjusted; unspecified

Whether the estimates of the risk factors are adjusted or unadjusted.

Seroprevalence value Float – The seroprevalence value as a

proportion between 0.0 and 1.0. parameter type Enum IgG; IgM; PRNT; HAI; IFA;

The type of seroprevalence parameter.

unspecified

numerator Integer; Null

The numerator used to calculate the seroprevalence value. If not provided, set to Null.

denominator Integer; Null

The denominator used to calculate the seroprevalence value. If not provided, set to Null.

Severity value Float – The value of the severity parameter as a proportion between 0.0 and 1.0.

numerator Integer; Null

– The numerator of the CFR or

IFR parameter, if provided. denominator Integer;

– The denominator of the CFR or IFR parameter, if provided.

Null

parameter type Enum CFR; IFR; proportion of symptomatic cases; proportion of asymptomatic cases

The type of severity parameter reported.

method Enum; Null

naive; adjusted; unknown The method used to calculate the CFR or IFR.

Following value extraction, all parameters move to our third step: population context extraction. We extract population context with the same prompt and tool for all parameter classes (see below).

###### Population Extraction Prompt

You are an expert epidemiologist extracting epidemiological parameters from scientific articles. You will be provided with the processed text of a scientific article. Your task is to extract information about epidemiological parameters according to the provided schema.

Study Objectives See study objectives in Section H.

###### Population Extraction Task Definition

For your next task, you will be provided with excerpts from a scientific article and an estimated parameter that has been extracted from that article. Your task is to scan the provided text and extract relevant sample population information for the given parameter. You will use the provided tool, which sets the schema you should follow when returning population information.

Population Excerpts The following are excerpts from the scientific article about parameter population context: {{population_info}}

###### The population tool call is schematised as follows: Table 33: The schema for the population context extraction tool call.

###### Variable Type Allowed values Description

population sex Enum female; male; both; unspecified The sex composition of your study population. If you have 99 men and 1 woman you would still put both in this option.

population sample type Enum community based; hospital based; household based; housing estate based; population based; school based; travel based; trade or business based; contact based; mixed settings; other; unspecified

How was the study conducted?

population group Enum healthcare workers; farmers; outdoor workers; animal workers; butchers; abattoir workers; pregnant women; children; sex workers; people who inject drugs; household contacts of survivors; persons under investigation; general population; persons with symptoms; mixed settings; unspecified; other

Demographic i.e. who was sampled?

population sample size Integer; Null

– Number of participants/samples

tested etc. population age min Integer;

– These must be number fields. If your sample is people over 18 you would put age min = 18 and leave age max blank.

Null

population age max Integer; Null

– These must be number fields. If your sample is people over 18 you would put age min = 18 and leave age max blank.

population countries List[String] – Where was the study undertaken? population location String – Location reported i.e. Kerry Town

Ebola Treatment Centre.

method moment value Enum start outbreak; mid outbreak; end outbreak; post outbreak; endemic; unspecified

When in the outbreak was this study undertaken?

For our final step, if we have multiple extractions of the same class for an article, we ask the language model to aggregate parameters that should be reported as ranges over population disaggregations. Our aggregation logic follows PERG’s rule of three, which specifies certain pathogen-specific conditions for when aggregated reporting is appropriate. These are detailed to the language model in the instruction prompt below, which is provided identically for all parameter classes.

###### Aggregation Prompt

You are an expert epidemiologist extracting epidemiological parameters from scientific articles. You will be provided with the processed text of a scientific article. Your task is to extract information about epidemiological parameters according to the provided schema.

Study Objectives See study objectives in Section H.

Aggregation Task Definition Aggregation task For your next task, you will be provided with a list of parameters already extracted from an epidemiological study. Your task is to provide aggregations of these parameter values when suitable.

Aggregation context Some epidemiological papers have a huge level of parameter disaggregation (e.g. age, sex, location) and so we have established different rules to ease our meta-review process. For non-location-related disaggregations, please remember the rule of three. If there are three or more disaggregations for a parameter, e.g. Rt values for three or more age groups, extract these as a range and specify that disaggregated data is available and what the parameter is disaggregated by. Each pathogen has different rules on location, which we state here:

- • marburg; ebola; MERS: Location is included within the rule of three.
- • lassa; SARS; zika; nipah: Please do not aggregate values if the disaggregation is by location as much as possible and do not apply the rule of three for geographic regions down to admin level 2 (sub-regions) of a country. However, please respect the rule of three for estimates by neighborhood for example.

If the provided parameters do not contain adequate population information to perform an aggregation, then do not return any aggregated values.

If you decide that an aggregation is necessary, use the provided tool to submit aggregated values according to the tool’s schema. Provide the lower_bound and upper_bound of the parameter values, and list the types of population disaggregation (like “age”, “sex”, etc.) in the disaggregated_by field. Fill the aggregated_ids list with all of the ids from the parameters you aggregated.

###### Extracted parameters Extracted parameters: {{parameters}}

###### I.2 Models

Valid transmission models for extraction Epidemiological transmission models are mathematical frameworks that simulate how infectious diseases spread through populations by mechanistically describing the interactions between infected and susceptible individuals. We extract models that mechanistically represent disease transmission dynamics, excluding purely statistical analyses, regression-based forecasting without transmission mechanisms, and risk factor studies. Table 34 defines the categories of model characteristics extracted in this study, organised into structural properties, epidemiological features, assumptions, intervention categories, and reproducibility indicators.

Table 34: Model characteristic categories targeted by the extraction pipeline. Category Description Structural Properties Model type (compartmental, agent-based, branching process) and

compartmental architecture (SIS, SIR, SEIR, etc.). Whether the model is stochastic or deterministic.

Epidemiological Features

Primary transmission routes (airborne, direct contact, vectorborne, sexual). Spatial heterogeneity and spillover dynamics from animal reservoirs.

Behavioural Assumptions

Mixing patterns (homogeneous or heterogeneous), age-dependent susceptibility, cross-immunity between pathogens, and temporal variation in transmission rates.

Theoretical vs. Fitted Whether the model was fitted to actual data or uses parameters from literature or arbitrary values.

Intervention Categories Control measures evaluated including vaccination, quarantine, vector control, treatment, contact tracing, behaviour changes, and various vector management strategies.

Reproducibility Indicators

Code availability, programming language used, data sharing status, and presence of documentation (README).

Model extraction schema Table 35 defines the complete extraction schema with field specifications, data types, allowed values, and descriptions. The schema uses controlled vocabularies to ensure consistency and enable structured analysis of modelling approaches across the literature.

Table 35: Model extraction schema with field specifications, data types, allowed values, and descriptions.

Field Name Type Allowed Values Description model_type Enum Compartmental; Branching

Primary modeling framework used for transmission dynamics.

process; Agent/Individual based; Other; Unspecified

compartmental_type Enum SIS; SIR; SEIR; SEIR-SEI; SAIR-SEI; Not compartmental; Other compartmental

Specific compartmental model architecture if applicable. Use “Not compartmental” for non-compartmental models.

stoch_deter Enum; Null

Stochastic; Deterministic Whether the model is stochastic or deterministic. Only extract if explicitly stated. Null if not specified.

transmission_route List[Enum] Airborne or close contact; Human to human (direct contact); Human to human (direct non-sexual contact); Vector/Animal to human; Sexual; Unspecified

Primary pathway(s) through which pathogen transmission occurs. Multiple routes can be selected.

uncertainty_was_consideredBoolean; Null

True; False Whether uncertainty was considered through stochasticity, multiple models, or parameter variation (e.g. sensitivity analyses, Bayesian analysis). Null if not specified.

spatial_model Boolean; Null

True; False Whether the model explicitly incorporated spatial or geographic heterogeneity. Null if not specified.

spillover_included Boolean; Null

True; False Whether the model explicitly modelled spillover (e.g. animal reservoir component, contribution to force of infection from zoonosis). Null if not specified.

assumptions List[Enum] Homogeneous mixing; Latent period is same as incubation period; Heterogeneity in transmission rates (between human groups; between groups; between human and vector; over time); Age dependent susceptibility; Cross-immunity between Zika and dengue; Other; Unspecified

Key structural and behavioural assumptions. Should be explicitly described in the paper or clear from model equations. Multiple assumptions can be selected.

theoretical_model Boolean True; False Whether the model was NOT fitted to data (parameters from literature or arbitrary values). True if not fitted; False if fitted to actual data.

interventions_type List[Enum] Vaccination; Quarantine; Vector/Animal control; Treatment; Contact tracing; Hospitals; Treatment centres; Safe burials; Behaviour changes; Wolbachia replacement/suppression; Genetically modified mosquitoes; Mechanical removal of breeding sites; Pesticides/larvicides; Insecticide-treated nets; Indoor residual spraying; Other; Unspecified

Categories of control measures evaluated or incorporated in the model(s). Multiple interventions can be selected.

code_available Boolean True; False Whether model implementation

code was made publicly available. coding_language Enum;

R; Python; Matlab; Julia; C++; Other

Programming language(s) used for model implementation if code is available. Null if not specified.

Null

is_data_used_available Enum; Null

Yes (as an attachment; with a DOI; on Github; on another platform); Not available; Unspecified

Whether input data used for the model was shared and how it was shared. Null if not specified.

readme_included Boolean; Null

True; False Whether a README or detailed documentation accompanied the code repository. Null if not applicable.

notes String; Null

– Additional context or notes about the extracted model. Free text field.

Multi-stage model extraction pipeline Model extraction employs a two-stage agentic workflow operating on full-text article content. Unlike parameter extraction, which requires fine-grained text excerpting and value parsing, model extraction focuses on identifying the presence of dynamic transmission models and characterising their structural properties using controlled vocabularies.

In the first stage, a binary screening step identifies articles containing dynamic transmission models while excluding purely statistical analyses, regression-based forecasting, and risk factor studies without transmission dynamics (see the “Model Screening Prompt” below). The language model returns a simple “True” or “False” response indicating whether the article contains models suitable for extraction.

For articles passing this screen, the extraction stage deploys a structured tool-calling approach where the language model iteratively invokes an extract_model_data function once per distinct model identified in the article (see the “Model Extraction Prompt” below). Each tool call populates the standardised schema defined in Table 35.

The schema enforces controlled vocabularies for all fields through strict JSON validation. Multipleselect fields (transmission_route, assumptions, interventions_type) accept arrays of values from predefined enumerations, while single-select fields enforce unique values or null for optional characteristics. Validation logic rejects outputs violating vocabulary constraints or logical rules. For example, a non-compartmental model_type must have compartmental_type set to “Not compartmental”, this prompts the model to correct errors before proceeding.

The complete extraction workflow is coordinated by the ModelExtractionRunner class, which loads full-text data, applies screening decisions, manages iterative tool calls with validation feedback, and logs all outputs to structured CSV files.

###### Model Screening Prompt

You are an epidemiologist specializing in infectious disease modeling. Determine if this article contains dynamic transmission models for infectious disease.

Screening Task Definition Include (respond “True”):

- • Compartmental models (SIR, SEIR, etc.)
- • Agent-based or individual-based models
- • Branching process models
- • Network transmission models Exclude (respond “False”):
- • Pure statistical/regression analyses
- • Time series forecasting without mechanistic transmission
- • Risk factor analyses without transmission dynamics
- • Seroprevalence studies without modeling Respond with only “True” or “False”.

Full Text Title: {{title}} Full Text: {{fulltext}}

###### Model Extraction Prompt

You are an epidemiologist specializing in infectious disease modeling. Extract information about transmission models from scientific articles.

Study Objectives Study Objectives This systematic review collates transmission models, outbreaks and parameters for {{pathogen}}.

Extraction Task Definition Model extraction task Extract ALL dynamic transmission models described in the article that were actually used or implemented.

Do not extract:

- • Models only mentioned as possible alternatives without implementation
- • Regression-only analyses
- • Purely statistical forecasting Tool Calling:
- • Call extract_model_data once per model identified in the article
- • After extracting all model/s, stop calling the tool (no completion call needed) Schema Requirements:
- • transmission_route, assumptions, interventions_type are arrays (multiple-select)
- • All other fields are single values (single-select)
- • Use null for optional single-select fields when not stated
- • Use ["Unspecified"] for required arrays when not stated

Full Text Title: {{title}} Full Text: {{fulltext}}

The language model uses the extract_model_data() tool (provided to it) to populate the schema defined in Table 35. The tool enforces strict JSON validation with controlled vocabularies for all fields, rejecting invalid outputs and prompting corrections. The complete tool specification follows standard OpenAI function calling conventions with enum constraints for single-select fields and array validation for multiple-select fields.

###### I.3 Outbreaks

Valid outbreak data for extraction Outbreak data capture the epidemiological characteristics of concluded epidemic events, including temporal bounds, geographic scope, transmission sources, case counts stratified by confirmation status, and demographic breakdowns. We extracted outbreak information as stated in articles, without performing additional calculations or inferring missing values.

Following extraction guidelines suggested by PERG,12 outbreak inclusion criteria varied by pathogen based on reporting completeness and literature volume. For Marburg and Lassa, all reported outbreaks were captured regardless of size. For Zika, only outbreaks with at least 10 confirmed, probable, or suspected cases were extracted, reflecting the assumption that smaller events may not be systematically documented and contribute minimally to population-level immunity estimates. Table 36 defines the outbreak characteristics and their meanings in natural language.

###### Table 36: Outbreak field descriptions and meanings.

###### Outbreak Characteristic Description

Outbreak start day Day of outbreak onset (1–31). Extracted as stated in paper. Outbreak start month Month of outbreak onset. Extracted as stated in paper. Outbreak start year Year of outbreak onset. Extracted as stated in paper. Outbreak end day Day of outbreak conclusion (1–31). Extracted as stated in paper. Outbreak end month Month of outbreak conclusion. Extracted as stated in paper. Outbreak end year Year of outbreak conclusion. Extracted as stated in paper. Outbreak duration (months) Duration of outbreak in months. ONLY extracted if explicitly stated in paper; not

calculated. Outbreak is currently ongoing Whether outbreak was concluded or ongoing at time of publication. Outbreak country Country where outbreak occurred, using WHO standard country names. Refers to

reporting country rather than infection source for imported cases. Outbreak location Specific geographic location within country (city, district, province) as written in

paper. Multiple locations separated by semicolons. Outbreak location type Administrative or geographic unit type of outbreak location. Outbreak source Known or suspected source of outbreak introduction. Mode of detection Primary method(s) used to identify and confirm cases. Method of case definition Criteria used to classify cases. Extracted as described in paper. Pre-outbreak baseline Baseline disease status in affected area prior to outbreak. Rarely reported. Cases confirmed Number of laboratory-confirmed cases (e.g. via molecular testing). Cases probable Number of probable cases as defined in paper. Definition may vary across studies. Cases suspected Number of suspected cases as defined in paper. Definition may vary across studies. Cases unspecified Number of cases where confirmation status was not specified. Cases asymptomatic Number of asymptomatic infections as defined in paper. Cases severe Number of severe cases or hospitalizations as stated in paper. Deaths Number of deaths attributed to outbreak. Asymptomatic transmission described Whether article explicitly described or discussed asymptomatic transmission. Population size (geographical area) Total population of affected geographic area. Rarely reported. Type of cases (sex disaggregation) Case type for which sex disaggregation was reported. Male cases Number of cases in males for specified case type. Proportion male cases Proportion (0.0–1.0) or percentage (0–100) of cases in males. Female cases Number of cases in females for specified case type. Proportion female cases Proportion (0.0–1.0) or percentage (0–100) of cases in females. Notes Additional context or clarifications about outbreak characteristics.

Outbreak extraction schema Table 37 defines the complete extraction schema with field specifications, data types, allowed values, and descriptions. The schema uses controlled vocabularies to ensure consistency and enable structured analysis of outbreak characteristics across the literature.

12https://github.com/mrc-ide/priority-pathogens/wiki/Outbreak-data

###### Table 37: Outbreak extraction schema with field specifications, data types, allowed values, and descriptions.

Field Name Type Allowed Values Description outbreak_start_day Integer; Null 1-31 Day of outbreak onset. Null

if not provided. outbreak_start_month String

Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

Month of outbreak onset. Null if not provided.

(Enum); Null

outbreak_start_year Integer; Null Integer year Year of outbreak onset. Null if not provided.

outbreak_end_day Integer; Null 1-31 Day of outbreak conclusion.

Null if not provided. outbreak_end_month String

Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

Month of outbreak conclusion. Null if not provided.

(Enum); Null

outbreak_end_year Integer; Null Integer year Year of outbreak conclusion. Null if not provided.

outbreak_duration_months Float; Null Numeric value Duration in months. ONLY if explicitly stated; not calculated. Null if not stated.

outbreak_is_currently_ongoing Boolean True; False Whether outbreak was concluded (False) or ongoing (True) at publication.

outbreak_country String (Enum)

WHO standard country names (195 member states)

Country where outbreak occurred. MUST match WHO standard names.

outbreak_location String; Null Free text Specific location within country. Multiple locations separated by semicolons. Null if not provided.

outbreak_location_type String; Null Free text (e.g. district, province, county, state, hospital)

outbreak_source String (Enum); Null

mode_of_detection String (Enum); Null

Domestic animal; Wild animal; Date palm sap; Unknown; Other

Molecular (PCR etc); Symptoms; Confirmed + Suspected; Unspecified

Type of administrative or geographic unit. Null if not specified.

Known or suspected source of outbreak introduction. Null if not provided.

Primary method used to identify and confirm cases. Null if not provided.

method_of_case_definition String; Null Free text Criteria used to classify cases as described in paper. Null if not provided.

Continued on next page

pre_outbreak String (Enum); Null

Disease-free baseline; Endemic equilibrium; Unspecified; Probable

Baseline disease status prior to outbreak. Null if not provided.

cases_confirmed Float; Null Non-negative numeric Number of laboratory-confirmed cases. Null if not provided.

cases_probable Float; Null Non-negative numeric Number of probable cases. Null if not provided.

cases_suspected Float; Null Non-negative numeric Number of suspected cases. Null if not provided.

cases_unspecified Float; Null Non-negative numeric Number of cases with unspecified confirmation status. Null if not provided.

cases_asymptomatic Float; Null Non-negative numeric Number of asymptomatic infections. Null if not provided.

cases_severe Float; Null Non-negative numeric Number of severe cases or hospitalizations. Null if not provided.

deaths Float; Null Non-negative numeric Number of deaths attributed to outbreak. Null if not provided.

asymptomatic_transmission_describedBoolean True; False Whether article explicitly described or discussed asymptomatic transmission.

population_size_geographical_area Float; Null Non-negative numeric Total population of affected geographic area. Null if not provided.

type_cases_sex_disagg String (Enum); Null

Confirmed; Suspected; Other; Unspecified

Case type for which sex disaggregation was reported. Null if not provided.

male_cases Float; Null Non-negative numeric Number of male cases for specified case type. Null if not provided.

prop_male_cases Float; Null Numeric (0.0-1.0 or 0-100) Proportion or percentage of cases in males. Null if not provided.

female_cases Float; Null Non-negative numeric Number of female cases for specified case type. Null if not provided.

prop_female_cases Float; Null Numeric (0.0-1.0 or 0-100) Proportion or percentage of cases in females. Null if not provided.

Continued on next page

notes String; Null Free text Additional context or clarifications about outbreak characteristics. Null if not needed.

Multi-stage outbreak extraction pipeline Outbreak extraction employs a two-stage workflow operating on full-text article content. The first stage applies binary screening to identify articles reporting concluded, real-world outbreak events with defined case counts, excluding ongoing outbreaks, modelled scenarios, routine surveillance, and single case reports (see the “Outbreak Screening Prompt” below). The language model returns a simple “True” or “False” response indicating whether the article contains outbreaks suitable for extraction.

For articles passing this screen, the extraction stage deploys a structured tool-calling approach where the language model iteratively invokes an extract_outbreak_data function once per distinct outbreak identified in the article (see the “Outbreak Extraction Prompt” below). Outbreaks are considered distinct if they differ by location, time period, or both. Each tool call populates the standardised schema defined in Table 37.

The schema enforces controlled vocabularies for categorical fields through strict JSON validation. The required fields must be provided (outbreak_is_currently_ongoing, outbreak_country, asymptomatic_transmission_described), while all other fields accept null values when data are not stated in the article. The outbreak_country field enforces WHO standard country names, and the outbreak_location field prohibits commas, requiring semicolon separators for multiple locations to avoid parsing ambiguities. Validation logic rejects outputs violating vocabulary constraints or data type rules, prompting the model to correct errors before proceeding.

The complete extraction workflow is coordinated by the OutbreakExtractionRunner class, which loads full-text data, applies screening decisions, manages iterative tool calls with validation feedback, and logs all outputs to structured JSONL files for downstream analysis.

###### Outbreak Screening Prompt

You are an epidemiologist conducting systematic review of infectious disease outbreaks. Determine if this article reports concluded, real-world outbreak events with defined case counts.

Screening Task Definition Include (respond “True”):

- • Discrete outbreak events with ALL of: specific location, defined time period, and case counts
- • Outbreak investigations describing a bounded epidemic event
- • Case series (2 or more cases) from a specific outbreak Exclude (respond “False”):
- • Ongoing outbreaks at time of publication
- • Modeled, simulated, or forecasted outbreaks
- • Routine surveillance or annual disease burden (e.g., “X cases per year”)
- • Seroprevalence or risk factor studies without outbreak context
- • Single case reports

Key Question: Can you identify a specific outbreak event (not general disease occurrence) with a start/end period and case count? Respond with only “True” or “False”.

Full Text Title: {{title}} Full Text: {{fulltext}}

Extraction Task Definition Outbreak extraction task Extract concluded outbreaks with defined case counts from the article. Call extract_outbreak_data once for each distinct outbreak (different location, time, or both).

Important Notes: We are extracting everything as presented in the paper, even if you think it is an error by the author(s). Extract data EXACTLY as stated in the paper. Do NOT perform calculations, convert units, or infer missing values. DO NOT use commas in any field. If you need to separate items within a field, please use a semicolon.

###### Tool Calling Rules:

- • Call extract_outbreak_data once per distinct outbreak
- • Outbreaks are distinct if they differ by location, time, or both
- • After extracting all outbreaks, stop calling the tool (no completion call needed) Schema Requirements: Only three fields are required:
- • outbreak_is_currently_ongoing: true or false
- • outbreak_country: Must be valid WHO country name
- • asymptomatic_transmission_described: true or false All other fields: Use null when not stated in the paper. Extraction Rules:
- • Only select values that appear in the allowed lists for categorical fields
- • Extract dates as separate components (day, month, year)
- • Do NOT calculate outbreak_duration_months; only extract if explicitly stated
- • If you receive a validation error message, correct the tool call and try again

Field-Specific Guidance: Location:

- • outbreak_country: MUST match WHO standard names exactly (e.g., “United States of America” not “USA”, “Viet Nam” not “Vietnam”)
- • outbreak_location: Extract as written; use semicolons not commas (e.g., “Lagos; Abuja”) Case Counts: Extract all categories as reported
- • cases_confirmed: Laboratory-confirmed cases
- • cases_probable: Probable cases (clinical diagnosis)
- • cases_suspected: Suspected cases under investigation
- • cases_unspecified: Cases without clear classification
- • cases_asymptomatic: Asymptomatic cases identified
- • cases_severe: Severe cases OR hospitalizations (note if hospitalizations in notes)
- • deaths: Reported deaths Mode of Detection: Select ONE
- • “Molecular (PCR etc)” Laboratory confirmation (PCR, ELISA, culture, etc.)
- • “Symptoms”: Clinical/syndromic diagnosis only
- • “Confirmed + Suspected”: Both lab-confirmed and clinical cases
- • “Unspecified”: Not clearly stated Sex Disaggregation: When provided, extract:
- • male_cases / female_cases: Counts
- • prop_male_cases / prop_female_cases: Proportion/percentage as reported
- • type_cases_sex_disagg: Which case type is disaggregated (Confirmed/Suspected/Other/Unspecified) Pre-Outbreak Baseline:
- • “Disease-free baseline”: No previous cases
- • “Endemic equilibrium”: Disease was endemic
- • “Probable”: Suggested but not definitive
- • “Unspecified”: Not discussed

Dates: Provide as separate components (day, month, year). Partial dates are acceptable (e.g., only month and year). Duration: ONLY extract if paper explicitly states duration. Do NOT calculate from dates. Notes: Use this field for important context, data quality issues, or special circumstances.

Pathogen-Specific Rules:

- • Zika, RVF: Only extract outbreaks with 10 or more cases
- • Marburg, Lassa, Nipah: Extract all outbreaks
- • OROV: Include even single case reports

###### Outbreak Extraction Prompt

You are an epidemiologist conducting systematic review of infectious disease outbreaks. Extract structured data about concluded outbreak events from scientific articles.

Study Objectives This systematic review collates transmission models, outbreaks and parameters for {{pathogen}}.

Extraction Task Definition See Extraction Task Definition details above.

Full Text Title: {{title}} Full Text: {{fulltext}}

The language model uses the extract_outbreak_data() tool (provided to it) to populate the schema defined in Table 37. The tool enforces strict JSON validation with controlled vocabularies for categorical fields, rejecting invalid outputs and prompting corrections. The complete tool specification follows standard OpenAI function calling conventions with enum constraints for single-select fields and null acceptance for optional fields.

Provenance extraction Following successful extraction of parameters, models, and outbreaks, a provenance stage systematically mapped each extracted value to supporting textual excerpts from the article, ensuring complete traceability and grounding of all characteristics in source material. For each extracted record (parameter estimate, model descriptor, or outbreak summary), the provenance extraction invoked a dedicated tool (extract_parameter_provenance, extract_model_provenance, or extract_outbreak_provenance) that received the complete set of previously extracted characteristics and identified verbatim quotes, equation references, or table citations justifying each value selection. For multi-select fields (e.g. transmission routes, assumptions, interventions in models; multiple locations in outbreaks), each selected option required independent textual support. This additional stage enabled potential validation of extraction quality, provided transparency for subsequent data synthesis, and formed an audit trail linking structured outputs to primary literature, with all provenance traces logged to structured files for downstream analysis.

##### J Report Generation: Building Systematic Living Reviews

Report generation is not evaluated within AgentSLR. The dataset shared by PERG and the openaccess subset retrieved by AgentSLR overlap but are not identical, which precludes the construction of data-matched reference reports necessary for a fair evaluation. Consultation with epidemiologists at PERG further clarified that screening and data extraction constitute the primary time bottleneck in their workflow, and that research questions, output artefacts, and meta-analysis variables are typically determined once an evidence base has already been assembled. Additionally, the risks associated with incorrect public health artefacts are substantial: In our testing, LLM-generated summaries produced in unconstrained settings tend towards over-interpretation. Report generation along with meta-analysis are future work, beyond the scope of our study.

###### J.1 Deterministic Report Assembly

Given a pathogen p, we generate human-readable reports directly from the extracted, structured datasets for outbreaks. The report build aggregates extraction records into descriptive summary tables and figures, then compiles a Markdown draft, and finally renders a PDF. The report build is lightweight relative to retrieval, screening, and extraction and is omitted from our main runtime breakdown (typically < 5 minutes per pathogen).

###### Inputs and derived artefacts

Inputs Let DpO be the set of extracted outbreak records for pathogen p (one row per outbreak entity). Each record is schema-validated at extraction time (Appendix I), so report generation treats the datasets as structured inputs.

Content manifest. The manifest stores: pathogen identifier, timestamp, summary statistics (e.g. outbreak counts and geographic coverage), the list of narrative sections, and structured metadata for each figure and table (number, title, caption, path, and row or observation counts). This manifest is later used as part of the evidence packet in the LLM refinement stage (next subsection).

###### Table 38: Artefact inventory for outbreak report generation (per pathogen p). All artefacts are derived from the extracted outbreak dataset DpO.

Artefact Path (relative to repo root) Purpose Outbreak report (Markdown) writeup/p/outbreaks_writeup.md Human-readable draft with embed-

ded figures and tables. Outbreak report (PDF) writeup/p/outbreaks_writeup.pdf Portable rendering for sharing and archiving.

Figures directory writeup/p/figures/ Generated plots referenced by Markdown (e.g., temporal distribution, geographic spread, case counts).

Summary tables (embedded) (in outbreaks_writeup.md) Count and proportion tables computed from DpO. Content manifest writeup/p/content_manifest.json Machine-readable inventory of figures, tables, and dataset statistics.

Evidence packet construction Evidence packet For pathogen p and outbreak report type O, code constructs an evidence packet EpO = STATSpO, FIGSpO, TABLESpO, WpO,(0) ,

where STATSpO is a concise text summary of dataset counts and geographic breakdowns, FIGSpO is the required figure list (paths and captions), TABLESpO is the set of tables to be included (as Markdown blocks), and WpO,(0) is the programmatic Markdown draft. The model is instructed to rely only on EpO and not to introduce external facts.

###### J.2 Evidence grounded narrative refinement

Report writing proceeds by an LLM revision stage that refines WpO,(0) into a narrative synthesis, while enforcing evidence grounding and artefact presence.

Self-refinement loop and non-negotiable checks Grounding and asset checks (non-negotiable). Two constraints are enforced for every refined version:

- 1. Asset presence: every required figure path from the manifest must appear at least once as a Markdown image line.
- 2. Table preservation: every table provided in the evidence packet must be present, with values unchanged (reformatting is allowed).

If either constraint is violated, we deterministically append missing figures or tables verbatim at the end of the Markdown so the final PDF always renders with the full artefact set.

Minimal formalisation Let W(0) denote the initial (programmatic) Markdown draft. Each iteration applies:

critique(W(k−1)) → C(k), revise(W(k−1),C(k)) → W(k).

This is only a notation convenience: in practice the evidence packet always accompanies both steps, and the critique output is structured JSON used to drive the next revision.

Rubric and prompts Rubric We use an 8-dimension rubric, each scored from 1 (poor) to 5 (excellent). The dimensions are the same for both report types, except for the scope constraint. Shared dimensions

- 1. data_fidelity: descriptive claims match the evidence packet; no invented statistics or outbreak characteristics.
- 2. figure_table_presence: all required figures and tables appear.
- 3. traceability: outside interpretation blocks, claims cite their source as (Figure X), (Table Y), or (Dataset Statistics).
- 4. clarity: consistent terminology, clear writing, minimal ambiguity.
- 5. completeness: covers the major patterns visible in the available figures and tables.
- 6. interpretation_blocks: interpretation is confined to dedicated blocks and labelled as such.
- 7. formatting: valid Markdown and sensible figure layout hints.

Interpretation policy Interpretation is allowed only inside blockquotes beginning with > AI-Interpretation:. Outside those blocks, the narrative must remain descriptive and evidencelinked; no new numbers may be introduced.

###### J.3 Report Generation Prompts

We present the exact prompts used for outbreak report generation and self-refinement, formatted consistently with the model report prompts. All prompts are instantiated programmatically by filling placeholders (e.g. {EVIDENCE_PACKET}) at runtime.

###### Outbreak report prompts

###### Outbreak Report: Initial Synthesis Prompt

You are a senior epidemiologist editing a living outbreak surveillance review. You are revising a first draft prepared by a research assistant who summarized extracted outbreak records.

Method Basis Do not cite external sources; just follow these behaviors:

- • Iterative critique→refine loop (Self-Refine).
- • Rubric-based form-filling evaluation mindset (G-Eval).
- • Attribution-first revision: every descriptive claim must be attributable to the provided evidence packet (RARR-style editing for attribution).
- • Living review principles: explicitly describe what is present in the dataset snapshot and what is missing; avoid academic formatting.

###### Hard Scope Constraint

Focus on documented outbreak events and outbreak characteristics. Do not broaden into transmission modelling, pathogen biology, or clinical management beyond what is supported by the outbreak dataset.

###### Truthfulness Constraints

- • Do not invent outbreak characteristics, case counts, geographic locations, or external facts.
- • Outside of AI-Interpretation blocks, every numeric or categorical claim must be directly supported by the evidence packet and must cite its support as (Figure X), (Table Y), or (Dataset Statistics).
- • Interpretation is allowed ONLY inside blockquotes starting with: > AI-Interpretation:
- • Inside AI-Interpretation blocks, you may propose plausible implications for outbreak surveillance and preparedness, but you must label them as hypotheses and you must not introduce new numbers that are not in the evidence packet.

###### Figures and Tables Constraints

- • All figures must appear as markdown images using their existing paths (e.g., ![Alt](figures/fig1_...png)). Placement is free.
- • Tables must all be present. You may reformat tables, but values must remain identical.

Formatting Agency

- • You may include an OPTIONAL HTML comment immediately after any figure image line to suggest sizing for PDF rendering.
- • Format: <!– fig-layout: width_in=5.5 max_height_in=7.5 –>
- • If absent, defaults will be used.

Output Requirements

- • Produce a living outbreak surveillance review in Markdown.
- • Use descriptive, report-like sections rather than academic paper structure.
- • For each main section, include: (1) Evidence-based description, then (2) one AI-Interpretation blockquote.

Task Definition Task: Produce Version 1 of the living outbreak surveillance review. Use the evidence packet below. Maintain honesty and verifiability.

Required structure (you may adapt headings, but keep these concepts):

- 1) Snapshot (dataset size, temporal coverage, geographic scope, what this review represents)
- 2) Outbreak temporal distribution (outbreak frequency over time, identification of major epidemic periods)
- 3) Geographic distribution and spread patterns (countries affected, spatial clustering, cross-border transmission)
- 4) Outbreak size and severity (case counts, fatality rates, outbreak durations)
- 5) Detection and reporting patterns (modes of detection, case definitions used, reporting delays if mentioned)
- 6) Demographic patterns (sex disaggregation, age patterns if available)
- 7) Data quality and gaps (completeness of reporting, missing information, asymptomatic transmission documentation)
- 8) Evidence-based recommendations (only tied to observed gaps in outbreak surveillance)
- 9) Change log stub (for future updates)

Evidence Packet

{EVIDENCE_PACKET}

###### Outbreak Report: Critique Prompt

You are a meticulous scientific editor. Return only valid JSON.

Critique Task Definition You are a scientific editor evaluating a living outbreak surveillance review for faithfulness to the provided evidence packet. Return STRICT JSON only.

###### Evidence Packet Summary {DATASET_STATISTICS}

Required Figure Paths All of the following must appear at least once: {REQUIRED_FIGURE_PATHS}

Report to Critique {CURRENT_REPORT}

Evaluation Dimensions Evaluate dimensions (score 1-5). Provide issues and concrete suggestions.

###### Dimensions:

- 1) data_fidelity: descriptive claims supported by evidence packet; no invented outbreak characteristics, case counts, or geographic information.
- 2) outbreak_focus: stays centered on documented outbreak events and outbreak surveillance rather than transmission modelling or pathogen biology.
- 3) figure_table_presence: all required figures present; all tables present.
- 4) traceability: outside AI-Interpretation blocks, claims cite support as (Figure X)/(Table Y)/(Dataset Statistics).
- 5) clarity: readable, precise, minimal ambiguity, consistent terminology for outbreak characteristics and surveillance metrics.
- 6) completeness: covers major patterns in outbreak temporal distribution, geographic spread, and detection practices described by available figures/tables.
- 7) interpretation_blocks: each main section includes a blockquote starting with > AI-Interpretation: and interpretation stays inside it.
- 8) formatting: figure layout directives used sensibly where needed; no broken markdown.

JSON Response Format Return JSON of the form:

{

"dimensions": { "data_fidelity": {"score": 1-5, "issues": [...], "suggestions": [...]}, "outbreak_focus": {"score": 1-5, "issues": [...], "suggestions": [...]}, "figure_table_presence": {"score": 1-5, "issues": [...], "suggestions": [...]}, "traceability": {"score": 1-5, "issues": [...], "suggestions": [...]}, "clarity": {"score": 1-5, "issues": [...], "suggestions": [...]}, "completeness": {"score": 1-5, "issues": [...], "suggestions": [...]}, "interpretation_blocks": {"score": 1-5, "issues": [...], "suggestions": [...]}, "formatting": {"score": 1-5, "issues": [...], "suggestions": [...]} }, "priority_fixes": [...] }

###### Outbreak Report: Revision Prompt

You are a senior epidemiologist performing an evidence-grounded revision.

###### Revision Constraints

- • Follow an attribution-first editing approach: outside AI-Interpretation blocks, every claim must be supported by the evidence packet.
- • Keep the document outbreak-focused.
- • All figures must appear at least once as markdown images with their existing paths.
- • All tables must be present; you may reformat, but values must not change.
- • Interpretation is permitted only within blockquotes beginning with > AI-Interpretation:.
- • You may add optional figure sizing directives as HTML comments immediately after image lines: <!– fig-layout: width_in=5.5 max_height_in=7.5 –>

Quality Scores

{DIMENSION_SCORES}

Priority Fixes

{PRIORITY_FIXES}

Evidence Packet

{EVIDENCE_PACKET}

Current Report

{CURRENT_REPORT}

###### Revision Requirements

- • Fix all critique issues.
- • Ensure each main section has (1) evidence-based description with citations (Figure/Table/Dataset Statistics), then (2) > AI-Interpretation: block.
- • Remove or relabel any statement not supported by the evidence packet.
- • Ensure outbreak-only framing (documented outbreak events and surveillance patterns).
- • Keep document a living surveillance review (descriptive, update-ready), not an academic paper.

Return the complete revised Markdown.

##### K Living Systematic Reviews with AgentSLR

Utilising AgentSLR harness workflow on the data extracted corpus from previous stages (with gpt-oss-120b), we generated living reviews for nine WHO priority pathogens: Marburg virus, Ebola virus, Lassa virus, SARS-CoV-1, Zika virus, MERS-CoV, Nipah virus, Rift Valley fever (RVF) virus, and Crimean Congo haemorrhagic fever (CCHF) virus. Each review comprises two complementary documents (a transmission-modelling review synthesising extracted model characteristics, and an outbreak surveillance review aggregating historical outbreak data) alongside structured datasets and visualisations. While four of these pathogens (Ebola, Lassa, SARS, Zika) have been validated against PERG’s expert annotations, the remaining five represent preliminary syntheses for pathogens where PERG’s systematic review process has not yet commenced or is in early stages.

Figure 7 presents excerpts from the Ebola living reviews, illustrating the structure and content of AgentSLR workflow’s outputs for a validated pathogen. The transmission-modelling review (Figure 7a) provides a quantitative overview of the 513 extracted models, including distributions across model architectures, stochasticity classifications, and code availability. The outbreak surveillance

- review (Figure 7b) synthesises 1,104 outbreak records spanning nearly six decades, with temporal coverage, geographic distribution, and detection methodology patterns presented through evidence-based descriptions paired with interpretive commentary blocks.

[Figure 4]

- (a) Transmission-Modelling Review excerpt showing dataset scope, model architecture distribution, and reproducibility indicators for 513 Ebola models extracted from 232 articles.

[Figure 5]

- (b) Outbreak Surveillance Review excerpt presenting snapshot statistics for 1, 104 outbreak records from 490 publications, covering temporal span 1967–2025 and 48 countries.

- Figure 7: Ebola living reviews generated by AgentSLR. Both reviews follow a structured format: evidence-based descriptions citing supporting figures and tables, followed by interpretation blocks explicitly labelled as AI-generated synthesis.

For emerging or understudied pathogens, rapid synthesis of available evidence can inform outbreak preparedness even when comprehensive expert review remains infeasible. Figure 8 presents excerpts from RVF and CCHF reviews, two pathogens for which PERG has not yet initiated systematic screening. The RVF transmission-modelling review (Figure 8a) characterises 115 models extracted from the retrieved literature, revealing a predominance of compartmental architectures and vector-to-human transmission pathways consistent with RVF’s arboviral ecology. The CCHF outbreak surveillance

- review (Figure 8b) maps 59 outbreak records with quantitative case data, identifying geographic clusters and temporal patterns across affected regions. While these syntheses lack the validation rigour applied to Ebola, Lassa, SARS, and Zika, they demonstrate AgentSLR workflow’s capacity to

generate preliminary evidence summaries for resource allocation and hypothesis generation in under 48 hours of wall-clock time. Once the evaluation scores are above a threshold, this same workflow can be utilised for assist in generate agile living reviews.

[Figure 6]

(a) RVF transmission-modelling review showing distribution of 115 extracted models across architecture types, stochasticity, transmission routes, and code availability.

[Figure 7]

(b) CCHF outbreak surveillance review presenting geographic burden (choropleth maps) and temporal distribution of 59 outbreaks with case-count data.

- Figure 8: Preliminary living reviews for pathogens without completed PERG validation. RVF and CCHF represent pathogens for which PERG has not yet commenced systematic screening.

- Table 39 summarises the standardised artefact structure maintained across all pathogen reviews. Both transmission-modelling and outbreak surveillance reports follow consistent schemas: model reviews characterise architecture distributions, stochasticity classifications, transmission pathways, and reproducibility indicators, whilst outbreak reviews present temporal timelines, geographic burden maps, detection methodology breakdowns, and case-count summaries. This structural consistency enables direct cross-pathogen comparison and ensures that future updates (as literature accumulates or as PERG completes validation for additional pathogens) maintain compatibility with existing syntheses.

Table 39: Key artefacts in AgentSLR workflow generated living reviews. Each pathogen generates two review types with consistent visualisation and evidence table structures. The text-based LLM uses manifests, with summary statistics of the figures to write its interpretation.

Type Transmission-Modelling Review Outbreak Surveillance Review

Figures Model architecture distribution (compartmental, branching process, agent-based); Stochasticity classification; Transmission route breakdown; Code availability

Geographic burden (choropleth maps for cases and deaths); Outbreak timeline by country; Detection mode distribution

Tables Model type counts and proportions; Deterministic vs stochastic breakdown; Transmission routes with sample sizes; Modelling assumptions; Intervention categories; Spatial scale indicators; Code availability and language

Outbreak source categories; Detection methodology breakdown; Ongoing outbreaks at extraction; Case burden stratified by confirmation status; Sex disaggregation where reported

##### L The PERG Review Pipeline (Human Reference Workflow)

The Pathogen Epidemiology Review Group (PERG) is an expert-led effort (started in 2019) whose goal is to maintain a definitive, curated source of epidemiological parameters for pathogens prioritised for epidemic preparedness. In practice, PERG delivers this through systematic literature reviews and meta-analyses targeting the WHO priority pathogens, with the explicit aim of supporting outbreak response and modelling when time is short and parameter choices matter.

The scope is defined by the WHO priority pathogens framing: diseases that “pose the greatest public health risk due to their epidemic potential and/or whether there is no or insufficient countermeasures." Examples highlighted in PERG onboarding include CCHF virus, Ebola virus, Marburg virus, Lassa virus, Middle East respiratory syndrome coronavirus (MERS-CoV), Severe Acute Respiratory Syndrome coronavirus 1 (SARS-CoV-1), Nipah virus, Rift Valley fever, and Zika virus.

PERG’s workflow is end-to-end: it starts from a protocolised literature search, then moves through screening (title & abstract, then full text), structured extraction into REDCap (including qualityassessment fields guided by the PERG wiki), meta-analysis, and finally the write-up of a review that can be used by modellers and public health teams.

###### Step 1: Paper search (protocol-driven, pathogen-specific)

PERG begins from a registered systematic review protocol (PROSPERO ID: CRD42023393345), and uses a standardised query template that is then tailored to each pathogen. The core idea is to search broadly across the epidemiological concepts that tend to matter during outbreak response: transmission and epidemiology terms, transmission modelling (with explicit exclusion of imaging-related “model” matches), severity outcomes (e.g. CFR), key delays (e.g. incubation period, serial interval, generation time), transmission heterogeneity and superspreading/overdispersion, transmissibility measures (e.g. growth rate and reproduction numbers), serology/serosurveys, evolutionary signals (mutation/substitution/evolution), outbreak/cluster terminology and risk factors. The query is written with wildcards to capture term variants, and then adjusted where needed to avoid cross-contamination with neighbouring literatures (for example, excluding SARS-CoV-2 when the target is SARS-CoV-1).

###### Step 2: Title and abstract screening (broad triage against explicit criteria)

The first screening pass is based on titles and abstracts. The emphasis here is not on perfect specificity, but on ensuring the pool remains wide enough to avoid missing relevant evidence that is only clearly described later in the paper. PERG’s inclusion criteria are simple but concrete: studies must be English-language, peer-reviewed original research (systematic reviews and meta-analyses are flagged rather than treated as primary extraction targets), and must involve human data. A paper is kept if it contains any one of several types of useful information, including: quantitative descriptions of a human outbreak (size, year, location, duration, spatial scale), a mathematical or statistical model of transmission, estimates of key transmission or timing quantities (e.g., R, R0, Rt, growth rate, generation time, serial interval, incubation or latent period, other delays), severity metrics (CFR, attack rate), evolutionary rates, overdispersion/superspreading, risk factors (together with the measure), seroprevalence, relative contributions of human-to-human vs zoonotic transmission, and, where relevant, vector-related quantities such as mosquito delays or mosquito reproduction numbers.

PERG’s exclusion criteria are equally explicit: non-English items; posters, conference proceedings, correspondence, and abstract-only records; in-vitro-only studies; solely animal studies (unless the paper provides clearly relevant transmission quantities); and small case studies with fewer than 10 cases.

###### Step 3: Full-text review (confirm “extractability")

Articles passing abstract screening move to full-text review. PERG applies the same conceptual criteria, but with a different mindset: reviewers scan the entire paper to confirm that there is something extractable, i.e. not just that the topic is on-target. Importantly, PERG explicitly runs both title and abstract screening and this stage with two reviewers, reflecting the goal of consistency and defensible inclusion decisions when judgement calls are required.

###### Step 4: Parameter extraction (read, highlight, enter structured fields)

Once a paper is included, PERG’s extraction process is deliberately hands-on. Reviewers (i) check which papers they have been assigned, (ii) download and read the PDF, highlighting everything they may want to extract as they go, and then (iii) enter the extracted information into a REDCap web database (PERG maintains pathogen-specific REDCap projects).

PERG structures extraction into four broad blocks:

- • Article metadata. Basic bibliographic information such as title, DOI, journal, and related identifiers are recorded.
- • What the paper contains: outbreaks, models, parameters. PERG extracts (i) outbreak descriptions where present, (ii) mathematical models of transmission (these are not limited to SIR-type models; they can be theoretical and not necessarily fitted to data), and (iii) epidemiological parameter estimates. Parameter families include genomic/evolutionary quantities (mutation/evolution rates),

reproduction numbers (R0, Rt, and human-only or vector-related variants where relevant), human delays (serial interval, incubation period, time-to-death, etc.), severity (CFR/IFR), seroprevalence (e.g. IgG/IgM markers), risk factors (with attention to whether effects are statistically significant and adjusted), relative contributions (human-to-human vs animal-to-human), attack rates (including secondary attack rates), and overdispersion (e.g. the negative binomial k parameter).

- • Associated context for interpretation. PERG captures the contextual details that make parameter estimates comparable (or not): sex (male/female/both/unspecified), sample size, setting (general population vs hospital), subgroup (children, pregnant, etc.), age ranges, country and more specific location, study start/end dates, and whether the study was conducted before/mid/after an outbreak.
- • Structured outbreak fields (when applicable). In addition to “is there an outbreak?”, PERG-style extraction treats outbreaks as structured entities. In our draft’s PERG-aligned outbreak guidance, outbreak characteristics include temporal bounds (start/end day/month/year; whether ongoing), geographic scope (country plus sub-location), outbreak source, mode of detection, case definition method, case counts by confirmation status (confirmed/probable/suspected/unspecified), asymptomatic and severe cases when reported, deaths, and (when available) demographic breakdown such as sex-disaggregated counts. A key principle is that these values are extracted as stated in the paper, without calculating missing quantities or inferring unreported fields.

Across all extraction types, PERG points reviewers to the PERG wiki for “how to extract this specific thing” guidance, so that extraction decisions remain consistent across pathogens and across reviewers.

###### Step 5: Meta-analysis

After extraction and quality assessment, PERG moves into synthesis and reporting. PERG maintains shared tooling for priority pathogens, including codebases that step through cleaning the extracted database, transforming quantities into a common format where needed, performing meta-analysis, and producing plots and summary tables. These outputs feed directly into the final PERG systematic review and meta-analysis write-up.

###### Step 6: Write-up

The final stage is to turn the extracted REDCap database and the meta-analysis outputs into a PERG review that can be used in practice. In PERG, meta-analysis is implemented through shared, pathogen-focused tooling (the priority-pathogens and epireview codebases), which steps through cleaning and transforming the extracted data, running the statistical synthesis, and producing the figures and summary tables. These tables and plots then provide the backbone of the manuscript: the review documents what evidence was found for each parameter family (and in what contexts), presents the quantitative summaries produced by the meta-analysis, and translates them into a curated resource for outbreak modelling and public health decision-making. In PERG’s framing, this write-up is not just a paper draft: it is the mechanism by which extracted parameters become a stable, citable reference for the WHO priority pathogens, with the longer-term aim of supporting an evolving “live” resource as evidence accumulates.

##### M AgentSLR Annotation Tool (Beta)

This section documents the AgentSLR annotation and validation interface, a beta-stage prototype designed to facilitate systematic literature reviews (SLRs) through the integration of LLM-assisted information extraction and expert-led verification. This would also allow evaluation of web and computer-use agents for SLRs beyond epidemiology.

[Figure 8]

- Figure 9: The AgentSLR extraction and validation tool interface. The dual-pane view presents the source document (left) alongside structured extraction fields (right). AI-predicted entries are prefilled and accompanied by highlighted, AI-tagged evidence excerpts from the manuscript. Reviewers can accept, revise, or reject individual fields, with validation status indicators (e.g. “AI Match” or “Revised”) reflecting whether human intervention was required.

System Architecture and Core Functionality The AgentSLR annotation tool provides an interactive environment for technical validation of automated information extractions. The system utilises data gathered by LLMs equipped with structured tool-calling to identify and parse epidemiological parameters, transmission models, and outbreak characteristics. To ensure transparency and auditability, the tool implements a provenance layer that maps every extracted value to specific textual excerpts (AI-tagged evidence) within the source article.

User Interface Design The interface is optimised for high-throughput expert review via a dual-panel architecture:

- • Document Viewer (Left Panel): Provides the original article text or rendered PDF, ensuring reviewers can verify the context of any extracted data point without context switching.
- • Verification Interface (Right Panel): Displays a form-based view of pre-filled fields generated by the AgentSLR pipeline. The extraction schema is dynamic, adapting based on the identified content type, such as compartmental model variables or spatio-temporal outbreak data.

###### M.1 Human-in-the-Loop Validation

The framework enforces a human-in-the-loop (HITL) protocol where automated extractions are audited by subject matter experts before being finalised for evidence synthesis. Within the interface, reviewers perform the following actions:

- • Verify: Confirm the accuracy of the AI-extracted value and its linked evidence.
- • Modify: Correct extraction errors or refine data granularity; modified entries are flagged as “Revised” to facilitate system error analysis.
- • Reject: Entirely dismiss false positive extractions that do not meet inclusion criteria.

[Figure 9]

[Figure 10]

(a) Tool Management Dashboard (b) Verification and Submission

- Figure 10: Review management and submission tracking interfaces. (a) The study review list displays papers awaiting expert validation, along with associated model and outbreak counts and direct links to extracted excerpts. (b) The verification view presents finalised AI-assisted extractions, enabling reviewers to explicitly reject predictions or verify and save corrected entries, which are then recorded for downstream quality control and system evaluation.

###### M.2 Current Status and Field Testing

The tool is currently in a beta development phase, with pilot testing conducted by epidemiologists focusing on WHO priority pathogens. While the current pilot utilises epidemiology-specific schemas, the architecture is designed to be domain-agnostic and can be adapted through schema reconfiguration and expert consultation.

Planned field testing is aligned with the Pathogen Epidemiology Review Group (PERG) workflow for remaining priority pathogens. This testing will utilise standardised extraction schemas for parameter, model, and outbreak data, and specifically target systematic reviews for CCHF virus and Rift Valley fever virus.

###### M.3 Transparency and Reproducibility

By maintaining a persistent link between the structured database and the source text, AgentSLR tool ensures that synthesised reports can be fully disaggregated. This audit trail is critical for scientific reproducibility, allowing researchers to trace every reported parameter, model and outbreak back to its exact location in the primary literature.

