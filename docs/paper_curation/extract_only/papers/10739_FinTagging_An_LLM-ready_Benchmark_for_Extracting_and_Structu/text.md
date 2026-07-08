# arXiv:2505.20650v5[cs.CL]17May2026

## FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information

Lingfei Qian∗

Keyi Wang

Yi Han

Xueqing Peng Yang Ren

Yan Wang

Columbia University USA

Georgia Institute of Technology USA

The Fin AI USA lfqian94@gmail.com

The Fin AI USA wy2266336@gmail.com

The Fin AI USA

Kaiwen He Chenri Luo Jianxing Chen Junwei Wu

Qinchuan Zhang

Shengyuan Lin

Fengran Mo

Dongji Feng

Rensselaer Polytechnic Institute USA

Carnegie Mellon University USA

University of Montreal Canada

California State University USA

Columbia University USA

Chen Xu Ziyang Xu

Jimin Huang

Guojun Xiong

Xiao-Yang Liu

Qianqian Xie

The University of Manchester Manchester, United Kingdom

Harvard University USA

Columbia University USA

The Fin AI USA

The Fin AI USA

The Fin AI USA

Jian-Yun Nie

University of Montreal Canada

### Abstract

LLMs’ capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero-shot settings reveals that while models generalize well in extraction, they struggle significantly with fine-grained concept linking, highlighting critical limitations in domain-specific structure-aware reasoning. Code is available at GitHub 1 and datasets are available at Hugging Face 2.

Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over 10k US-GAAP concepts remains costly and error-prone. Existing benchmarks oversimplify this task as flat, single-step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. Consequently, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structureaware and full-scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts (text and tables); and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US-GAAP taxonomy. This two-stage formulation enables a fair assessment of

### CCS Concepts

• Computing methodologies → Natural language processing; Information extraction; • Information systems → Information extraction; Top-k retrieval in databases.

### Keywords

XBRL tagging, Benchmark, Large language model, Information extraction, Information retrieval, Reranking

ACM Reference Format:

Yan Wang, Lingfei Qian, Xueqing Peng, Yang Ren, Yi Han, Keyi Wang, Dongji Feng, Fengran Mo, Shengyuan Lin, Qinchuan Zhang, Kaiwen He, Chenri Luo, Jianxing Chen, Junwei Wu, Chen Xu, Ziyang Xu, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, Qianqian Xie, and Jian-Yun Nie. 2018. FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 17 pages. https://doi.org/XXXXXXX.XXXXXXX

∗Corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

- 1https://github.com/The-FinAI/FinTagging
- 2https://huggingface.co/collections/TheFinAI/fintagging

### 1 Introduction

Although LLMs excel at question answering [18], information extraction [2], and long-document summarization [32], their ability to perform financial tagging remains underexplored. Unlike open-ended generation, tagging requires committing each reported number to a precise semantic interpretation, grounding it in mixed table and text evidence, and aligning it to standardized concepts interpretable by downstream systems.

Each year, over 2 million companies publish financial reports disclosing earnings, expenses, and liabilities. However, inconsistent terminology complicates reliable interpretation. XBRL, introduced in 1999, defines the global standard for machine-readable financial reporting [19] and is now adopted in 65 countries3 . In practice, XBRL tagging requires grounding thousands of reported numbers per filing in mixed evidence and aligning them to over 17,000 standardized taxonomy concepts (Figure 1), a process that remains largely manual and error-prone4.

Nevertheless, existing XBRL tagging benchmarks emphasize annotation convenience over semantic difficulty. Many reduce tagging to classification over truncated taxonomies and remove tabular structure that carries critical evidence in real filings (Table 1). As a result, evaluation is limited to a small subset of common concepts, obscuring long-tailed generalization. Moreover, prior work typically formulates tagging as closed-set extreme classification [14, 22], which becomes brittle when the full taxonomy cannot be represented in-context and increases the risk of invalid or hallucinated tags. Our ablations confirm this limitation: even strong LLMs show near-zero performance under full-taxonomy settings. In addition, most benchmarks omit tables entirely, despite tables being the primary carrier of numerical disclosures, and thus fail to evaluate the joint reasoning over structured tables and narrative context required in realistic tagging [3, 33].

To address these issues, we introduce FinTagging, the first LLMoriented benchmark that evaluates end-to-end financial tagging over mixed table and text inputs with alignment to the full USGAAP taxonomy. Unlike prior work [14, 22], FinTagging reframes tagging as an extract-and-align process rather than flat classification, enabling evaluation across all 17k+ concepts. FinTagging adopts a two-stage formulation. Financial Numerical Identification (FinNI) focuses on extracting numerical facts with supporting evidence, while Financial Concept Linking (FinCL) maps each fact to the correct US-GAAP concept via semantic alignment with taxonomy definitions. We release two expert-verified evaluation sets, FinNI-eval and FinCL-eval, derived from real-world XBRL.

We conclude our main contributions as follows: (1) A New Formulation: We introduce FinTagging, the first benchmark that reframes XBRL tagging into a two-stage “Extract-and-Align” paradigm, enabling full-scope evaluation against 17k+ US-GAAP concepts, a scale previously intractable for zero-shot LLMs. (2) High-Fidelity Resources: We release two expert-verified datasets, FinNI-eval and FinCL-eval, which incorporate complex tabular structures to assess joint reasoning across structured and unstructured financial modalities. (3) Diagnostic Insights: By benchmarking 13 state-of-the-art

- 3https://www.xbrl.org/the-standard/what/what-is-xbrl/
- 4https://xbrl.us/wp-content/uploads/2023/03/DQC-SECMeetingNotes-20240314.pdf

LLMs, we identify a critical “knowledge-alignment gap”, demonstrating that while LLMs excel at numeric extraction, they require our alignment framework to prevent performance collapse during large-scale taxonomy mapping.

### 2 Related Work

Prior XBRL tagging research primarily formulates the task as extreme multi-class classification over truncated label sets, as in FiNER and FNXL [14, 22], with methods such as GalaXC, SECBERT, and AttentionXML [14, 20, 22] addressing label sparsity via hierarchical modeling. However, these closed-set formulations do not reflect the semantic reasoning required to navigate the full 17k+ US-GAAP taxonomy and are poorly aligned with LLM capabilities. In parallel, financial document understanding has advanced from entity and relation extraction [21, 23] to joint text–table numerical reasoning in QA benchmarks such as FinQA, TAT-QA, and DocVQA [3, 15, 33], yet these tasks do not require alignment to standardized accounting concepts. Recent LLM-based tools such as XBRL-Agent [9] explore filing analysis but stop short of end-to-end fact-to-taxonomy evaluation. In contrast, FinTagging reframes XBRL tagging as a two-stage extract-and-align task, explicitly targeting multi-modal fact extraction and fine-grained alignment to official US-GAAP definitions.

### 3 FinTagging 3.1 Task Formulation

Formally, given a financial document 𝐷 containing text and structured tables, a set of predefined XBRL value types 𝐿, and a taxonomy T of financial concepts, the XBRL tagging task is to identify all numerical values in 𝐷 and annotate each with a structured triplet {Fact, Type, Tag}. Here, Fact denotes the extracted numerical value as expressed in context, Type specifies its XBRL data type in 𝐿, and Tag refers to the linked concept in T (e.g., us-gaap:CashAndDueFromBanks). Each numerical fact is assigned a single, most specific taxonomy concept. The value type set 𝐿 includes monetaryItemType, percentItemType, sharesItemType, perShareItemType, and integerItemType, corresponding to monetary amounts, ratios, share counts, per-share values, and integer quantities, respectively. We formalize the task as a mapping:

𝑓 : (𝐷,𝐿, T) ↦→ {(Fact𝑖, Type𝑖, Tag𝑖)}𝑛𝑖=1 (1) where each triplet corresponds to a financial value mention extracted from 𝐷 and semantically grounded in T.

Inspired by information extraction and alignment works [17, 27, 29, 30] and after discussing with financial reporting specialists, we formulated FinTagging into two sub-tasks: Financial Numeric Identification (FinNI), a multi-modal numerical information extraction task, detects numerical value in 𝐷 and classifies each with its appropriate Type. Financial Concept Linking (FinCL), a numerical entity normalization task, then associates each identified value with its most appropriate Tag in T based on contextual and structural cues.

3.1.1 Financial Numeric Identification (FinNI). The first subtask of FinTagging focuses on identifying numerical values in a financial document and assigning each a coarse-grained value data type. This

[Figure 1]

Figure 1: An example of Financial Tagging in realistic scenarios, where the blue numbers mark facts to be tagged.

Table 1: Detailed comparison of financial NLP benchmarks across task types, sources, and structural capabilities. “Num.Reasoning” indicates the numerical reasoning. “Struct.IE” denotes the structured information extraction.

#Entity Label

#Taxonomy Label

Num. Reasoning?

Struct. IE?

Concept Linking?

Benchmark Scenario Data Source Task Modality

FinQA [3] Decision making SEC 10-K QA text/table 0 0 ✓ ✗ ✗ ConvFinQA [4] Decision making SEC 10-K QA text/table 0 0 ✓ ✗ ✗ TAT-QA [33] Financial analysis Chinese financial reports QA text/table 0 0 ✓ ✗ ✗ DocVQA [15] Enterprise automation Financial document images QA text/image 0 0 ✓ ✗ ✗ FiNER-ORD [21] Financial tagging Chinese financial disclosures Classification text 49 0 ✗ ✗ ✗ FinRED [23] Knowledge graph construction English financial news Classification text 38 0 ✗ ✗ ✗ FiNER [14] XBRL tagging SEC 10-K Extreme classification text 0 139 / 17,688 ✓ ✗ ✗ FNXL [22] XBRL tagging SEC 10-K Extreme classification text 0 2,800 / 17,688 ✓ ✗ ✗ FINTAGGING Financial & XBRL tagging SEC 10-K IE + Alignment text/table 5 17,688 / 17,688 ✓ ✓ ✓

corresponds to detecting the Fact and Type components of each triplet {Fact, Type, Tag} defined in the overall task. We formalize this subtask as a mapping:

𝑓FinNI : (𝐷 = (𝑆,𝑇), 𝐿) ↦→ {(𝑒𝑖,𝑙𝑖)}𝑘𝑖=1 (2) where 𝑆 and𝑇 represent the textual and tabular components of the document, respectively, and 𝐿 is the set of predefined value data types. Each 𝑒𝑖 is a numerical entity extracted from either 𝑆 or 𝑇, and 𝑙𝑖 ∈ 𝐿 denotes its assigned data type.

3.1.2 Financial Concept Linking (FinCL). Building on the output of the FinNI subtask, the goal of FinCL is to semantically ground each identified numerical entity 𝑒 by linking it to a concept 𝑐ˆ in a predefined financial taxonomy, which is the Tag component of each triplet {Fact, Type, Tag} defined in the overall task. Formally, we define the mapping as:

𝑓FinCL : (𝑒,𝑙,𝐶𝑒, T) ↦→ 𝑐ˆ (3) where 𝑒 is a numerical entity extracted from 𝐷 = (𝑆,𝑇), and 𝑙 ∈ 𝐿 is its associated data type provided by the FinNI stage. 𝐶𝑒 represents the structural context of 𝑒, which encompasses linguistic dependencies for narrative mentions and hierarchical row-column coordinates for tabular mentions. T = {𝑐1,𝑐2, . . .,𝑐𝑛} denotes the full-scope US-GAAP taxonomy containing 𝑛 uniquely defined and semantically grounded concepts. The objective of FinCL is to identify the optimal concept 𝑐ˆ ∈ T that maximizes the semantic alignment between the fact 𝑒 and its specific context 𝐶𝑒 within the high-dimensional taxonomy space.

### 3.2 Raw Data Collection

We compiled 142 annual 10-K filings from publicly listed U.S. companies submitted to the U.S. Securities and Exchange Commission (SEC)5 in 2023–2024 (Table 2). The collection covers all 11 major industry sectors, with companies primarily based in the United States and a small portion internationally. Parsing the filings with BeautifulSoup yields 319,893 narrative sentences and 21,576 financial tables. All filings follow the SEC Inline XBRL (iXBRL) format, which embeds machine-readable tags within HTML disclosures and links textual and tabular values to US-GAAP taxonomy concepts. However, these regulatory tags are designed for compliance rather than model evaluation, providing limited negative coverage and substantial structural noise. We therefore repurpose the raw iXBRL metadata to construct a standardized, high-fidelity evaluation suite for financial fact extraction and concept linking.

#### Table 2: Statistics of collected financial reports.

##### Item Value

Report type 10-K Period 2023-02-13 to 2025-02-13 #Company 142 #Covered sector 11 #Covered jurisdiction 31 States + outside US #Sentence 319,893 #Char 75,748,949 #Table 21,576

5https://www.sec.gov/

Algorithm 1: Automated Instance Annotation

Input: Raw iXBRL reports D; taxonomy T; valid types L Output: Positive instances P; Negative instances N Initialize P ←∅, N←∅, seen text set S←∅; foreach instance 𝑖 ∈ D (sentence/table) do

𝑡𝑒𝑥𝑡 ← ExtractText(𝑖); if |𝑡𝑒𝑥𝑡 | ≤ 20 or 𝑡𝑒𝑥𝑡 ∈ S then

continue 𝐸 ← {(𝑣,𝜏,𝑐) | (𝑣,𝑐,𝜏) ←ParseIXBRL(𝑖), 𝑐 ∈ T, 𝜏 ∈ L, 𝑣 ≠ ∅}; 𝐸 ← DedupBy(𝐸, (𝑣,𝑐)); if 𝐸 ≠ ∅ then

P ← P ∪ {(𝑡𝑒𝑥𝑡,𝐸)}; else

N ← N ∪ {(𝑡𝑒𝑥𝑡, ∅)}; S ← S ∪ {𝑡𝑒𝑥𝑡};

return P, N;

### 3.3 Large-Scale Automated Annotation

Our automated pipeline produces a large, high-fidelity corpus reflecting real financial disclosures. As summarized in Table 3, the dataset contains 15,986 narrative sentences and 12,801 tables, yielding 261,457 numerical entities linked to 3,953 unique US-GAAP concepts. Including both valid facts and naturally occurring noise, it supports a realistic evaluation of LLMs under regulatory-style conditions.

- Table 3: Statistical information for the original dataset in our benchmark. Tokens are calculated using the “cl100k_base” tokenizer (± standard deviation).

Item Sentence Table Positive instances 7,768 8,709 Negative instances 8,218 4,092 Avg. Tokens/S 80.91 ± 63.62 1212.42 ± 1421.76 Avg. Entities/S 1.24 ± 1.82 18.87 ± 37.16 Avg. Concepts/S 1.24 ± 1.82 18.87 ± 37.16 Total Entities 261,457 Entity Types 5 Unique Concepts 3,953

Automated Annotation Engine. We build a rule-based parser that reconstructs semantic links between human-readable disclosures and Inline XBRL (iXBRL) metadata. For each instance 𝑖, the parser extracts iXBRL fact elements (e.g., ix:nonfraction) and retrieves the value 𝑣, concept𝑐, and data type𝜏, implemented as ParseIXBRL in Algorithm 3. To avoid duplicates from nested tags, we restrict extraction to atomic leaf elements 𝜖.

Valid Entity Types. To define the label space, we profile all numeric XBRL item types. As shown in Figure 2, the distribution is long-tailed: although eleven types appear, five dominate. We therefore restrict L to monetaryItemType, percentItemType, sharesItemType, perShareItemType, and integerItemType. An instance is labeled POSITIVE (P) only if it contains at least one concept 𝑐 ∈ T from L; otherwise it is labeled NEGATIVE (N).

Structural Refinement and Filtering. For reliable evaluation, we keep only text segments longer than 20 characters and deduplicate content globally using 𝑡𝑒𝑥𝑡. For tables, we retain structural tags (<table>, <tr>, <td>, <th>) but discard tables without numerical facts. This filtering preserves semantic accuracy while maintaining realistic layout structure.

[Figure 2]

#### Figure 2: Statistics of numerical entity types.

Algorithm 2: Strategic Audit Set Sampling

Input: Positive instances P; observed concepts C𝑎𝑙𝑙; cap 𝐾 = 10 Output: Strategic subset P∗

P∗ ←∅; C𝑟𝑒𝑚 ← C𝑎𝑙𝑙; foreach company 𝑔 do

Add up to 𝐾 instances from 𝑔 to P∗; C𝑟𝑒𝑚 ← C𝑟𝑒𝑚 \ Concepts(P∗);

while C𝑟𝑒𝑚 ≠ ∅ and P \ P∗ ≠ ∅ do

𝑖∗ ← argmax𝑖∈P\P∗ |Concepts(𝑖) ∩ C𝑟𝑒𝑚 |; if ties then

select 𝑖∗ with the shortest text length;

P∗ ← P∗ ∪ {𝑖∗}; C𝑟𝑒𝑚 ← C𝑟𝑒𝑚 \ Concepts(𝑖∗); return P∗;

### 3.4 Data Quality Control

Validating all 261,457 entities is infeasible. We therefore adopt a strategic sampling scheme based on a greedy minimal-cover method (algorithm 4) to construct a compact audit set P∗. The procedure balances two objectives: Company Balance, by capping the number of sampled instances per company, and Taxonomy Completeness, by ensuring coverage of all observed US-GAAP concepts. Specifically, we first seed P∗ with up to 𝐾 instances per company, then iteratively add instances that cover the largest number of remaining uncovered concepts. This process yields a diversity-maximized subset consisting of 928 sentence instances and 1,021 table instances, which is reviewed by two junior analysts and one senior adjudicator. Review Protocol. The audit followed two stages. A pilot phase aligned interpretations of the Verification Guideline6, after which analysts independently verified whether extracted triplets (𝑣,𝜏,𝑐) were fully correct. The senior expert adjudicated disagreements to create the final gold set.

AgreementResults.On theindependentround,annotators achieved 96% raw agreement and Cohen’s 𝜅 = 0.81, indicating substantial agreement [12]. This high-verified accuracy on the most semantically diverse subset provides strong evidence for the reliability of our automated annotation pipeline.

### 3.5 Task-Specific Dataset Construction

Following validation, we integrated the corrected P∗ subset back into the full corpus to form a high-fidelity hybrid dataset. This refined pool ensures that every US-GAAP concept in the taxonomy has at least one expert-verified representation. From this foundation, we derive two subtask-specific datasets: FinNI-eval, targeting numerical entity identification, and FinCL-eval, targeting concept

6https://docs.google.com/document/d/1TgTxz-fozBImeRs8Y9kLViHnr8Bjs1qZbsB8u5NjfmA/ edit?usp=sharing

linking. Both are the first datasets in the financial domain to emphasize structured information extraction and concept-level alignment. Their detailed statistics are reported in Table 4.

- Table 4: Statistics of evaluation datasets for FinNI-eval and FinCL-eval. Tokens are calculated using the “cl100k_base” tokenizer (± standard deviation).

Item FinNI-eval FinCL-eval

#Instance 28,787 261,457 Avg. Input Tokens 1101.50 ± 992.00 60.43 ± 48.25 Max Input Tokens 18,831 750 Avg. Output Tokens 175.24 ± 375.90 12.82 ± 5.09 Max Output Tokens 7,813 41

[Figure 3]

Figure 3: A unified evaluation framework on FinTagging benchmark. Note: The Input content is drawn from the FinTagging dataset, and the "Shared LLMs" are used for identification in the FinNI task and reranking in the FinCL task.

- 3.5.1 FinNI-eval Dataset Construction. Following the annotation procedure in Section 3.3, we construct the FinNI-eval dataset with 28,787 instances derived from annotated sentences and tables. Each instance consists of an input block and an answer block. The input block includes a task instruction and the corresponding content, while the answer block is a JSON list of identified numerical values with their types (or an empty list).

FinNI-eval Dataset

Instruction: <FinNI Task Instruction> Input: <Sentences or Table> Answer: {"results": [{"Fact": <numeric entity>,

"Type": <entity type>}]} or {"results": []}

- 3.5.2 FinCL-eval Dataset Construction. Following the FinCL formulation in Section 3.1.2, we construct the FinCL-eval dataset with 261,457 query–answer pairs for numerical entity normalization, along with a US-GAAP taxonomy containing 17,688 concepts. Each query includes a numerical entity, its type, and surrounding context, while the answer is the corresponding US-GAAP concept.

FinCL-eval Dataset

Query: <entity> + <entity type> + <context> Answer: <US-GAAP Concept>

### 3.6 Evaluation

We propose a unified evaluation framework for FinTagging. As shown in Figure 3, each input is decomposed into two subtasks, which together produce structured triplets. This design jointly evaluates zero-shot numerical extraction and concept alignment. For comparison with token-classification baselines, we report macro and micro Precision, Recall, and F1 [24]. FinNI is evaluated using pair-level metrics, while FinCL is evaluated with accuracy.

FinNIEvaluation.FinNIusespair-levelmetrics toassess whether models correctly extract numerical values from text-table inputs.

FinCL Evaluation. FinCL is formulated as a retrieval–reranking task to avoid impractical extreme classification. We embed each taxonomy concept using text-embedding-3-small7 and retrieve the top-ℎ candidates from the US-GAAP taxonomy by semantic similarity. The LLM then reranks candidates using contextual reasoning to select the final tag.

7https://platform.openai.com/docs/models/text-embedding-3-small

3.6.1 Evaluation Models. Our goal is to assess the foundational capabilities of SOTA LLMs on FinTagging and understand their strengths and limitations in XBRL tagging. We evaluate models across three categories using our unified framework. (1) Closedsource general LLM: GPT-4o [10]. (2) Open-source general LLMs: DeepSeek-V3 [13], DeepSeek-R1-Distill-Qwen-32B [5], Qwen3-32B / 14B / 1.7B / 0.6B [26], Llama-4-Scout-17B-16E-Instruct [16], Llama-

- 3.3-70B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct [8], and gemma-2-27b-it [25]. (3) Domain-specific financial LLM: Fino18B [18]. We also compare against strong PLM baselines: BERTlarge [6], FinBERT [1], and SECBERT [14]. Models are evaluated with the LM Evaluation Harness [7] on a 4×H100 cluster and via APIs, using input limits of 24,576 (FinNI) and 8,192 (FinCL) tokens. Total cost is approximately 500 GPU-hours and $1,500 in API usage. 4 Experiment and Result
- 4.1 Overall Results

Table 5 presents the overall performance on the FinTagging benchmark. It clearly demonstrates that under our framework, LLMs can effectively handle both frequent and rare financial tags, indicating their ability to mitigate long-tail label challenges and underscoring the advantage of our information extraction and alignment formulation over traditional token-level classification approaches.

Table 5: Overall Performance. Bolded values indicate the best performance, underlined values represent the second-best, and italicized values denote the third-best performance.

Macro Micro P R F1 P R F1

Models

GPT-4o 0.0865 0.0626 0.0569 0.1052 0.0731 0.0863 DeepSeek-V3 0.0949 0.0778 0.0659 0.1074 0.1264 0.1162 Llama-4-Scout-17B-16E-Instruct 0.0683 0.0414 0.0400 0.1045 0.0526 0.0699 Llama-3.3-70B-Instruct 0.0544 0.0279 0.0288 0.0665 0.0382 0.0485 DeepSeek-R1-Distill-Qwen-32B 0.0532 0.0283 0.0285 0.0814 0.0214 0.0339 Qwen3-32B 0.0639 0.0314 0.0324 0.1127 0.0230 0.0382 gemma-2-27b-it 0.0471 0.0291 0.0276 0.0533 0.0390 0.0451 Qwen3-14B 0.0591 0.0274 0.0288 0.1069 0.0182 0.0311

- Llama-3.1-8B-Instruct 0.0345 0.0178 0.0169 0.0575 0.0166 0.0258
- Llama-3.2-3B-Instruct 0.0194 0.0118 0.0100 0.0178 0.0084 0.0114 Qwen3-1.7B 0.0207 0.0067 0.0080 0.1063 0.0031 0.0060 Qwen3-0.6B 0.0026 0.0008 0.0010 0.0562 0.0002 0.0005 Fino1-8B 0.0344 0.0143 0.0151 0.0419 0.0128 0.0197

BERT-large 0.0252 0.0266 0.0205 0.1518 0.1283 0.1391 FinBERT 0.0046 0.0064 0.0042 0.0872 0.0526 0.0656 SECBERT 0.0231 0.0295 0.0203 0.1870 0.1697 0.1779

From a macro perspective, which emphasizes balanced performance across frequent and rare tags, large general-purpose LLMs clearly dominate. DeepSeek-V3, GPT-4o, and Llama-4-Scout achieve the strongest macro-F1 scores, surpassing all fine-tuned PLMs and indicating better generalization to long-tail concepts. Qwen3-32B also performs competitively, suggesting that model architecture and pretraining can compensate for a smaller scale. From a micro perspective, which reflects performance on frequent labels, the trend is similar: top LLMs remain competitive even without finetuning, while fine-tuned PLMs such as SECBERT and BERT-large excel mainly because they benefit from label frequency rather than broader coverage.

### 4.2 Subtask Results

Table 6 summarizes performance on the FinNI and FinCL subtasks. Overall, large LLMs outperform smaller models across both tasks in a zero-shot setting. On FinNI, DeepSeek-V3 achieves the best F1 score, driven by substantially higher recall, while GPT-4o performs competitively. In contrast, several models (e.g., Qwen3 variants) exhibit high precision but extremely low recall, indicating that they extract only a small fraction of target facts, which leads to poor F1 performance. Mid-sized models such as gemma-2-27B show more balanced behavior. In contrast, smaller models and the financial LLM Fino1-8B lag, suggesting that domain pretraining alone does not yield robust gains for numerical extraction.

Table 6: Performance comparison of different models on the FinNI and FinCL subtasks. Bolded values indicate the best performance, underlined values represent the second-best, and italicized values denote the third-best performance.

FinNI FinCL

Model

Precision Recall F1 Accuracy GPT-4o 0.5893 0.4977 0.5397 0.1829 DeepSeek-V3 0.5886 0.8430 0.6932 0.1889 Llama-4-Scout-17B-16E-Instruct 0.4668 0.3164 0.3771 0.1649 Llama-3.3-70B-Instruct 0.4826 0.3301 0.3920 0.1318 DeepSeek-R1-Distill-Qwen-32B 0.5676 0.1942 0.2894 0.1141 Qwen3-32B 0.6991 0.1804 0.2868 0.1277 gemma-2-27b-it 0.5060 0.4526 0.4778 0.1099 Qwen3-14B 0.6912 0.1487 0.2448 0.1144

- Llama-3.1-8B-Instruct 0.3874 0.1761 0.2421 0.0913
- Llama-3.2-3B-Instruct 0.1856 0.1203 0.1460 0.0415 Qwen3-1.7B 0.7362 0.0281 0.0541 0.0735 Qwen3-0.6B 0.2803 0.0019 0.0038 0.0414 Fino1-8B 0.3431 0.1293 0.1878 0.0807

FinCL remains significantly more challenging than FinNI. All FinCL results are reported under a retrieval–reranking setup, where models select the final tag from the top-200 retrieved US-GAAP candidates. Although retrieval results are not separately tabulated due to space constraints, structure-aware retrieval achieves over 0.65 Acc@200 for sentence-based entities but remains below 0.30 Acc@200 for table-based entities, resulting in an overall Acc@200 of approximately 0.33. This indicates that retrieval coverage is already limited, especially for tabular contexts, which imposes a ceiling on downstream reranking. Even under this constrained candidate space, the best-performing models achieve accuracy below 0.19, with DeepSeek-V3 and GPT-4o forming a narrow top tier. These results highlight a clear extraction–alignment gap: while modern

LLMs can recover numerical facts with reasonable accuracy, finegrained US-GAAP disambiguation remains intrinsically difficult, particularly for table-origin entities.

### 4.3 Ablation analysis

We further compare our benchmark against extreme multi-class classification settings. We select the best-performing model from each category. The evaluation uses triplet-level ({Tag, Fact, Type}) Precision, Recall, and F1.

Table 7: Performance comparison between with/without our evaluation framework on the FinTagging benchmark dataset.

Evaluation Mode Model Precision Recall F1

GPT-4o 0.1166 0.0915 0.1026 DeepSeek-V3 0.1187 0.1581 0.1356 Fino1-8B 0.0336 0.0140 0.0197

FINTAGGING

GPT-4o 0 0 0 DeepSeek-V3 0 0 0 Fino1-8B 0 0 0

Extreme Classification

- Table 7 shows that modeling XBRL tagging as single-step ex-

treme classification fails in practice: models must select from thousands of taxonomy concepts without grounding in identified numerical entities, leading all models to collapse to zero. In contrast, our two-stage framework reflects real tagging workflows: FinNI first identifies and types numerical facts, and FinCL then selects from a retrieved candidate set using context. This formulation yields non-trivial performance and exposes meaningful differences among models that are entirely obscured under extreme classification.

4.4 Error propagation in FinTagging pipeline

- Table 8 compares accuracy under three settings: the full pipeline, FinCL with gold entities, and rerank-only with gold candidates. Results show clear error cascading: end-to-end performance is lowest due to accumulated extraction and retrieval errors. Providing gold entities substantially improves accuracy, identifying entity extraction as a key bottleneck. Rerank-only settings further increase performance, yet accuracy remains low, indicating that fine-grained US-GAAP disambiguation is inherently challenging even with perfect retrieval.

### 5 Conclusion

This paper introduces FinTagging, a benchmark for evaluating LLMs on XBRL tagging of real financial reports. The task is structured into two subtasks, Financial Numerical Identification and Financial Concept Linking, to separately assess extraction and concept alignment. Our results show that LLMs perform reasonably well in zero-shot settings and generalize to long-tail entities, but still struggle to align facts with precise US-GAAP concepts. These findings point to gaps in structure-aware and semantic reasoning. FinTagging provides a foundation for future work on reliable XBRL tagging and regulatory reporting.

### References

[1] Dogu Araci. 2019. Finbert: Financial sentiment analysis with pre-trained language models. arXiv preprint arXiv:1908.10063 (2019).

#### Table 8: Accuracy comparison across the full pipeline, FinCL subtask, and rerank-only settings.

Model Pipeline FinCL Rerank-only

GPT-4o 0.1052 0.1829 0.2023 DeepSeek-V3 0.1074 0.1889 0.2144 Llama-4-Scout-17B-16E-Instruct 0.1045 0.1649 0.1792 Llama-3.3-70B-Instruct 0.0665 0.1318 0.1482 DeepSeek-R1-Distill-Qwen-32B 0.0814 0.1141 0.1258 Qwen3-32B 0.1127 0.1277 0.1402 gemma-2-27b-it 0.0533 0.1099 0.1210 Qwen3-14B 0.1069 0.1144 0.1321

- Llama-3.1-8B-Instruct 0.0575 0.0913 0.1076
- Llama-3.2-3B-Instruct 0.0178 0.0415 0.0566 Qwen3-1.7B 0.0598 0.0735 0.0880 Qwen3-0.6B 0.0362 0.0414 0.0533 Fino1-8B 0.0419 0.0807 0.0908

- [2] Jeya Balaji Balasubramanian, Daniel Adams, Ioannis Roxanis, Amy Berrington de Gonzalez, Penny Coulson, Jonas S Almeida, and Montserrat García-Closas.

2025. Leveraging large language models for structured information extraction from pathology reports. arXiv preprint arXiv:2502.12183 (2025).

- [3] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al.

2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122 (2021).

- [4] Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849

(2022).

- [5] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948 [cs.CL] https://arxiv.org/abs/2501. 12948
- [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018). arXiv:1810.04805 http://arxiv.org/abs/1810.04805
- [7] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. A framework for few-shot language model evaluation. doi:10.5281/zenodo.12608602
- [8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [9] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, and Steve Yang. 2024. XBRLAgent: Leveraging Large Language Models for Financial Report Analysis. In Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF ’24).
- [10] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [11] Rik Koncel-Kedziorski, Michael Krumdick, Viet Lai, Varshini Reddy, Charles Lovering, and Chris Tanner. 2023. Bizbench: A quantitative reasoning benchmark for business and finance. arXiv preprint arXiv:2311.06602 (2023).
- [12] J Richard Landis and Gary G Koch. 1977. The measurement of observer agreement for categorical data. biometrics (1977), 159–174.
- [13] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- [14] Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malakasiotis, Ion Androutsopoulos, and Georgios Paliouras. 2022. FiNER: Financial numeric entity recognition for XBRL tagging. arXiv preprint arXiv:2203.06482 (2022).
- [15] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2200–2209.
- [16] AI Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodal-intelligence/, checked on 4, 7 (2025), 2025.
- [17] Rajdeep Mukherjee, Abhinav Bohra, Akash Banerjee, Soumya Sharma, Manjunath Hegde, Afreen Shaikh, Shivani Shrivastava, Koustuv Dasgupta, Niloy Ganguly, Saptarshi Ghosh, et al. 2022. Ectsum: A new benchmark dataset for bullet point summarization of long earnings call transcripts. arXiv preprint arXiv:2210.12467

(2022).

- [18] Lingfei Qian, Weipeng Zhou, Yan Wang, Xueqing Peng, Jimin Huang, and Qianqian Xie. 2025. Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance. arXiv preprint arXiv:2502.08127 (2025).
- [19] Jim Richards, Barry Smith, and Ali Saeedi. 2006. An introduction to XBRL. Available at SSRN 1007570 (2006).
- [20] Rachit Saini, Ankit Gupta, and Harshil Singh. 2021. GalaXC: Graph Neural Networks for Extreme Classification in Financial Text. arXiv preprint arXiv:2104.05709

(2021).

- [21] Agam Shah, Ruchit Vithani, Abhinav Gullapalli, and Sudheer Chava. 2023. Finer: Financial named entity recognition dataset and weak-supervision model. arXiv preprint arXiv:2302.11157 (2023).
- [22] Soumya Sharma, Subhendu Khatuya, Manjunath Hegde, Afreen Shaikh, Koustuv Dasgupta, Pawan Goyal, and Niloy Ganguly. 2023. Financial numeric extreme labelling: A dataset and benchmarking. In Findings of the Association for Computational Linguistics: ACL 2023. 3550–3561.
- [23] Soumya Sharma, Tapas Nayak, Arusarka Bose, Ajay Kumar Meena, Koustuv Dasgupta, Niloy Ganguly, and Pawan Goyal. 2022. FinRED: A dataset for relation extraction in financial domain. In Companion Proceedings of the Web Conference

2022. 595–597.

- [24] Marina Sokolova and Guy Lapalme. 2009. A systematic analysis of performance measures for classification tasks. Information processing & management 45, 4

(2009), 427–437.

- [25] Gemma Team. 2024. Gemma. (2024). doi:10.34740/KAGGLE/M/3301
- [26] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https: //arxiv.org/abs/2505.09388
- [27] David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi. 2019. Entity, relation, and event extraction with contextualized span representations. arXiv preprint arXiv:1909.03546 (2019).
- [28] Richard Zhe Wang. 2023. Standardizing XBRL financial reporting tags with natural language processing. Available at SSRN 4613085 (2023).
- [29] Yan Wang, Jian Wang, Huiyi Lu, Bing Xu, Yijia Zhang, Santosh Kumar Banbhrani, Hongfei Lin, et al. 2022. Conditional probability joint extraction of nested biomedical events: design of a unified extraction framework based on neural networks. JMIR Medical Informatics 10, 6 (2022), e37804.
- [30] Ledell Wu, Fabio Petroni, Martin Josifoski, SebastianRiedel,and Luke Zettlemoyer.

2019. Scalable zero-shot entity linking with dense entity retrieval. arXiv preprint arXiv:1911.03814 (2019).

- [31] Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443

(2023).

- [32] Haohan Yuan, Sukhwa Hong, and Haopeng Zhang. 2025. StrucSum: GraphStructured Reasoning for Long Document Extractive Summarization with LLMs. arXiv preprint arXiv:2505.22950 (2025).
- [33] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. arXiv preprint arXiv:2105.07624 (2021).

### A Significance analysis

To further examine the comparative performance across models, we conducted a pairwise significance analysis, with the results summarized in Figure 4. The upper-triangular matrix in SubFigure 4a presents the pairwise significance outcomes, where a value of 1 indicates that the row model’s performance is significantly different from the column model. In contrast, 0 indicates that the difference is not statistically significant. The diagonal elements are set to zero, since self-comparisons are not meaningful.

[Figure 4]

(a) Model-vs-Model significance matrix.

[Figure 5]

(b) Per-model count of significant pairwise differences.

Figure 4: Visualization of significance analysis across models.

- (a) shows the pairwise-bootstrap significance matrix, while
- (b) summarizes how many significant differences each model has relative to others.

As shown in SubFigure 4b, the number of significant pairwise differences varies across models, reflecting heterogeneous performancebehaviorson theevaluatedbenchmark. The models DeepSeekR1-Distill-Qwen-32B and Qwen3-32B exhibit the largest numbers

of significant differences, followed by GPT-4o, Llama-4, and Qwen314B, suggesting that their performance diverges more noticeably from other models. These results may indicate that models within the Qwen family display stronger variability across comparisons, potentially due to differences in training objectives or architectural configurations. In contrast, DeepSeek-V3, Llama-3.3-70B-Instruct, gemma-2-27b-it, and Fino1-8B demonstrate relatively fewer significant differences, implying that their performance remains more stable and comparable to other high-performing models. Overall, these findings suggest that larger or instruction-tuned models tend to produce more consistent outcomes, while models with smaller sizes or domain-oriented tuning show greater variation and clearer statistical separations under significance testing.

Overall, this analysis highlights not only which models outperform or underperform others but also which models behave similarly when subjected to statistical evaluation. The pairwise significance matrix, therefore, provides an interpretable view of model robustness and performance distinctiveness beyond average accuracy metrics.

### B Literature Review

The XBRL provides a comprehensive taxonomy for financial reporting, encompassing thousands of detailed tags corresponding to concepts within financial statements. Applying NER to assign XBRL tags is an emerging yet challenging area.

### B.1 XBRL Tagging benchmark

FiNER systematically benchmarked several neural architectures on the finer-139 dataset to address numeric-heavy XBRL tagging [14]. The initial experiments showed that standard BERT underperforms due to subword fragmentation, then the authors introduced pseudotoken strategies replacing numerals with [NUM] or [SHAPE] tokens to stabilize label assignment across fragmented numeric spans. These strategies, combined with domain-specific pretraining on SEC-BERT, significantly improved tagging performance, reaching 82.1 micro-F1 without the need for computationally expensive CRF layers. Their experiments demonstrated that subword-aware models with numeric-aware pseudo-tokens outperform both word-level BiLSTMs and vanilla BERT, particularly in numeric-heavy contexts, and avoid nonsensical label sequences. FNXL extended this benchmarking paradigm to a much larger label space of 2,794 US-GAAP tags, reframing the task as an extreme classification problem [22]. They compared the FiNER sequence labeling approach with a twostep pipeline that first identifies numeric spans and then assigns labels using AttentionXML. While FiNER achieved stronger microF1 (75.84), reflecting better performance on frequent tags, AttentionXML outperformed FiNER in macro-F1 (47.54), highlighting its strength in predicting infrequent, tail-end labels. FNXL further evaluated both models under a Hits@k setting, confirming that label recommendations from the AttentionXML pipeline could substantially reduce manual effort and maintain high inter-annotator agreement. Together, these benchmarks reveal the need for contextaware reasoning and label-ranking mechanisms in realistic XBRL tagging scenarios.

### B.2 XBRL Tagging Methods

The previous studies also explored approaches to address scalability, semantic ambiguity, and reasoning gaps in XBRL tagging to improve performance. Saini et al. [20] proposed GalaXC is a graph-based extreme classification framework that jointly learns over document-label graphs with per-label attention across multihop neighborhoods. By integrating label metadata and transitive label correlations, GalaXC outperformed leading deep classifiers by up to 18% in micro-F1 on standard benchmarks and achieved 25% gains in warm-start scenarios where partial labels are available. Moreover, Wang et al. [28] addressed the practical challenge of custom tag standardization through a semantic similarity pipeline that leverages TF-IDF, Word2Vec, and FinBERT embeddings. Although unsupervised, the method was tested across nearly 200,000 custom tags from SEC filings between 2009 and 2022, and showed strong alignment performance, with vector-based mappings identifying viable standard tag candidates for a substantial proportion of noncompliant elements, offering a low-cost and interpretable solution for downstream financial analysis. Shifting focus from classification to comprehension, XBRL-Agent evaluated the capabilities of large language models to reason over full XBRL reports [9]. The authors introduced two task types, domain taxonomy understanding and numeric reasoning, and found that base LLMs often hallucinated or misinterpreted financial content. To overcome these issues, XBRL-Agent incorporated retrieval-augmented generation (RAG) and symbolic calculators within an LLM-agent framework. The enhanced system achieved a 17% accuracy gain on domain query tasks and a 42% boost on numeric reasoning queries compared to base LLMs, validating the utility of modular tool augmentation. These improvements enabled reliable multi-step reasoning over complex disclosures such as debt instruments and derivative gains, which are difficult to capture using span-level classifiers. Collectively, these works broaden the methodological landscape of XBRL tagging from graph-based label propagation and embedding-based normalization to LLM-driven report analysis and point to a hybrid future where structured priors and reasoning tools jointly support accurate, scalable financial information extraction.

### B.3 Financial Evaluation Benchmarks

In parallel to XBRL-specific advances, the financial NLP community has developed comprehensive benchmarks to assess broader capabilities in information extraction, numerical reasoning, and document understanding. FiNER-ORD [21] introduced a high-quality, domain-specific NER dataset annotated over financial news, emphasizing general entity types like persons, organizations, and locations. While not numerically focused like FiNER-139, it highlights the lexical diversity of financial discourse and establishes a strong baseline for testing pretrained and zero-shot LLMs in real-world financial NER scenarios. FinQA [3] pushed toward explainable QA by pairing expert-written questions with annotated multi-step reasoning programs derived from earnings reports. ConvFinQA [4] extended this challenge to conversational contexts, simulating real-world question flows over sequential financial queries. TAT-QA [33] focused on hybrid tabular-text reasoning and required models to align cell values and document narratives, often involving aggregation, comparison, and unit-scale interpretation. Pixiu [31] introduced

[Figure 6]

#### Figure 5: Distribution of Tickers by Industry Sector.

a broader evaluation framework by releasing FinMA, a financial LLM instruction-tuned across five tasks, and assessing it on a new benchmark covering sentiment classification, QA, summarization, NER, and stock prediction. BizBench [11] framed financial QA as program synthesis over realistic, multi-modal contexts, integrating reasoning, code generation, and domain knowledge into a single evaluation pyramid. While these benchmarks highlight the growing ability of models to integrate structured and unstructured financial data, they overlook taxonomy-driven fact alignment and do not support the structured output formats required for XBRL tagging.

C Data and Ticker Information

- C.1 The Statistics of the Tickers

Figure 5, Figure 6, and Figure 7 illustrate the distribution of 142 tickers across industry sectors, market capitalization categories, and geographic regions, highlighting the diversity of our collected financial reports. Considering that practical XBRL tagging practices often vary by industry, company size, and legal jurisdiction, we curated a diverse set of companies covering all 11 major industry sectors. The sample maintains a balanced distribution across firm sizes, following market capitalization categories: micro-cap (<$300M), small-cap ($300M–$2B), mid-cap ($2B–$10B), large-cap ($10B–$200B), and mega-cap (>$200B). In addition, we incorporated firms from over 30 states and international jurisdictions to capture regional differences in reporting and tagging conventions. This diversity ensures that the benchmark reflects realistic heterogeneity observed in financial disclosures across sectors, scales, and regulatory environments.

- C.2 Data Granularity Justification

Each 10-K report follows the SEC’s XBRL filing standard, where all narrative and tabular disclosures are linked through machinereadable tags defined in the US-GAAP taxonomy. While the XBRL framework provides document-level structure via instance, schema, and linkbase files, the tagging of financial values and concepts occurs at a much finer granularity—typically within individual sentences or table rows that contain numeric facts and their contextual descriptions.

[Figure 7]

Figure 6: Distribution of Tickers by Market Cap Size.

[Figure 8]

Figure 7: Distribution of Tickers by Geography.

For this reason, our benchmark adopts the sentence and table sequence as the fundamental input unit rather than entire reports. This design choice is motivated by three considerations:

- • Locality of Semantic Scope: Financial facts and their USGAAP tags are usually expressed within a single sentence or a bounded table region. Modeling at this level aligns the input scope with the tagging scope, reducing noise from unrelated content within long filings.
- • Computational Efficiency: Full 10-K filings often exceed hundreds of pages and contain heterogeneous sections (e.g., risk factors, management discussion, and financial statements). Processing the entire report as one input sequence would exceed model context limits and obscure local numerical relations. Segment-level inputs enable fine-grained supervision and efficient evaluation.
- • Faithful Tag Alignment: Because each numerical entity is explicitly linked to its taxonomy tag in the XBRL instance document, using sentence or table segments ensures a oneto-one correspondence between input content and groundtruth annotations. This granularity preserves the original tagging fidelity of the filings.

Algorithm 3: Automated Instance Annotation

Input: Raw 10-K reports (iXBRL) D; US-GAAP taxonomy T; Set of valid

data types L Output: Positive instances P; Negative instances N Initialize: P ← ∅, N ← ∅, Global unique text set S𝑡𝑒𝑥𝑡 ← ∅; foreach instance (sentence or table) 𝑖 ∈ D do

𝑡𝑒𝑥𝑡 ← Extract plain text content from instance 𝑖; if length of 𝑡𝑒𝑥𝑡 ≤ 20 or 𝑡𝑒𝑥𝑡 ∈ S𝑡𝑒𝑥𝑡 then

skip to the next instance; 𝐸 ← ∅ // List of validated entities in 𝑖; 𝑆𝑓 𝑎𝑐𝑡 ← ∅ // Local fact deduplication set; foreach atomic iXBRL element 𝜖 within instance 𝑖 do

Retrieve value 𝑣, financial concept 𝑐, and data type 𝜏 from 𝜖; if 𝑐 ∈ T and 𝜏 ∈ L and 𝑣 is not empty then

if (𝑣,𝑐) ∉ 𝑆𝑓 𝑎𝑐𝑡 then

Add (𝑣,𝜏,𝑐) to the entity list 𝐸; Add (𝑣,𝑐) to the local deduplication set 𝑆𝑓 𝑎𝑐𝑡;

if 𝐸 is not empty then

Add (𝑡𝑒𝑥𝑡,𝐸) to the positive set P; else

Add (𝑡𝑒𝑥𝑡, ∅) to the negative set N; Add 𝑡𝑒𝑥𝑡 to the global set S𝑡𝑒𝑥𝑡;

return P, N;

Overall, this segmentation strategy allows our benchmark to remain consistent with how XBRL tagging operates in practice (on localized factual statements), while maintaining scalability and interpretability across large collections of filings.

### C.3 Automated Data Annotation

Algorithm 3 converts Inline XBRL (iXBRL) filings into structured instances for our benchmark. The procedure operates over both narrative sentences and tables, and assigns each instance a positive or negative label based on whether it contains at least one validated financial entity.

We begin by iterating through all disclosure instances in the corpus and extracting their plain-text content. Very short fragments (length ≤ 20) and duplicated passages are discarded using a global text set, ensuring that the benchmark contains unique and meaningful contexts.

For each remaining instance, the algorithm traverses all atomic iXBRL elements. From each element, we retrieve the numerical value 𝑣, its associated concept 𝑐 from the US-GAAP taxonomy, and the declared data type 𝜏. We retain only elements whose concept is in T and whose type belongs to the predefined valid numeric set L. Restricting extraction to leaf-level elements avoids double counting values that appear in nested XML structures. A local deduplication set further prevents repeated annotation of identical (𝑣,𝑐) pairs within the same instance.

If at least one valid entity is identified, the instance is recorded as a positive example together with its entity list; otherwise, it is stored as a negative example. Negative instances are intentionally preserved, as they reflect realistic disclosure text that models must learn to ignore. Overall, this process yields a corpus that is faithful to regulatory metadata while remaining structurally clean and suitable for downstream evaluation.

Algorithm 4: Strategic Audit Set Sampling

Input: Positive instances P from Algorithm 1; Observed taxonomy

concepts C𝑎𝑙𝑙; Company cap 𝐾 = 10 Output: Strategic audit subset P∗ Initialize: P∗ ← ∅; Initialize: C𝑟𝑒𝑚 ← C𝑎𝑙𝑙 // Set of concepts yet to be

covered;

- Step 1: Diversity-Preserving Seeding foreach unique company in P do

Select up to 𝐾 instances belonging to this company and add to

P∗;

Update C𝑟𝑒𝑚 by removing concepts covered by these seed instances;

- Step 2: Greedy Minimal Cover

while C𝑟𝑒𝑚 is not empty and candidates remain in P do foreach candidate instance 𝑖 ∈ P \ P∗ do

Calculate 𝑆𝑐𝑜𝑟𝑒(𝑖) = |Concepts in 𝑖 ∩ C𝑟𝑒𝑚|; Select instance 𝑖∗ with the highest 𝑆𝑐𝑜𝑟𝑒(𝑖); if multiple instances have the same 𝑆𝑐𝑜𝑟𝑒(𝑖∗) then

Break ties by selecting the instance with the shortest text length;

Add 𝑖∗ to P∗ and update C𝑟𝑒𝑚 by removing newly covered

concepts; return P∗;

### D Annotation Data Quality Control D.1 Validation Data Filtering

To establish a reliable upper bound on annotation quality, we construct a strategic audit subset rather than validating the full corpus. Algorithm 4 formalizes the sampling process as a greedy minimalcover problem over observed US-GAAP concepts.

The algorithm takes as input the set of positive instances P, together with the set of all taxonomy concepts observed in the corpus, C𝑎𝑙𝑙. The goal is to produce a compact subset P∗ that is both diverse and concept-complete.

Seeding for company diversity. We first iterate over companies and select up to 𝐾 instances per company. This prevents the audit set from being dominated by firms with longer filings or more frequent disclosures. After each selection, we update the set of uncovered concepts C𝑟𝑒𝑚 by removing concepts already represented among the seeds.

Greedy minimal cover. We then repeatedly add instances that contribute maximal marginal concept coverage. For each candidate instance, we compute the number of uncovered concepts it contains and select the instance with the largest score. When ties occur, we prefer shorter instances to reduce annotation effort while retaining conceptual value. After adding each instance, we update C𝑟𝑒𝑚 accordingly and continue until either all concepts are covered or no additional candidates remain.

Resulting audit subset. This procedure yields a compact subset that balances two competing objectives: coverage of the observed taxonomy and representation across companies. The resulting P∗ therefore serves as a stress-test sample that is both manageable for human review and representative of the broader benchmark space.

### D.2 Annotator Demography

To ensure reliable verification of financial entities and taxonomy mappings, all audits were conducted by annotators with prior exposure to XBRL-based reporting and financial statement analysis. The review team consisted of two junior analysts and one senior adjudicator.

The two junior analysts are graduate-level researchers with training in accounting and financial data analytics. Both have received formal instruction on US-GAAP concepts, XBRL tagging conventions, and common disclosure practices in SEC filings. They have prior experience working with financial datasets in academic or industry projects, including tasks involving concept normalization, document review, and regulatory text interpretation. Before participating in the audit, each analyst completed a guided calibration session using our Verification Guideline to ensure consistent treatment of numerical entities, metadata attributes, and taxonomy semantics.

The senior adjudicator has over a decade of experience in financial reporting analysis and structured disclosure systems. Their professional background includes work with XBRL instances, taxonomy structure, and validation procedures used in regulatory environments. In addition to overseeing the audit workflow, the senior adjudicator resolved disagreements, provided clarification on ambiguous tagging scenarios, and ensured alignment with USGAAP definitions and XBRL best practices.

Together, this composition provides both annotation robustness and domain oversight: junior analysts contribute detailed review capacity, while the senior adjudicator ensures conceptual correctness and methodological consistency across the audit.

All annotators were members of the research team and were invited to participate as part of their normal research training and collaborative project responsibilities. Participation was voluntary, and no separate monetary compensation was provided. Because the work involved publicly available financial disclosures and did not require annotators to process personal or sensitive data, the review posed minimal risk to participants.

### D.3 Validation Guideline

- D.3.1 Task Definition. Given a context and an extracted triplet consisting of:

- • an entity,
- • its entity type,
- • a US-GAAP concept,

Your task is to determine whether the triplet is correct. Make a binary decision:

- • 1 for correct,
- • 0 for incorrect.

- D.3.2 Validation Procedure. Follow the rules below, the definitions of entity type are shown in Table 9:

(1) Entity Check Determine whether the extracted entity is a numerical value.

- • If yes, proceed to Step 2.
- • If no, set is_correct = 0.

- (2) Entity Type Validation Verify whether the identified entity type is correct based on the context and the definitions in Table 9.

- • If yes, proceed to Step 3.
- • If no, set is_correct = 0.

- (3) US-GAAP Concept Validation Assess whether the assigned US-GAAP concept is appropriate based on the taxonomy tag definitions.

- • If yes, set is_correct = 1.
- • If no, set is_correct = 0.

#### Table 9: Entity type definitions used for validation.

###### Entity Type Definition

monetaryItemType Financial amounts expressed in currency, such as revenue, profit, or total assets. integerItemType Counts of discrete items, such as the number of employees or total transactions. perShareItemType Per-share values, such as earnings per share (EPS) or book value per share. sharesItemType Counts of shares, such as outstanding shares or ownership stakes. percentItemType Ratios or percentages, such as tax rates, growth rates, or discount rates, usually

expressed with a percentage symbol (%).

- D.3.3 Important Instructions.

- (1) Non-Arabic Formats Financial numerical entities may appear in word form (e.g., ten million) and must be correctly identified and converted into standard numerical format.
- (2) Magnitude Terms If a number is followed by a magnitude term (e.g., hundred, million, billion), do not expand it into the full numerical value:

- • Two hundred → extract only two, not 200.
- • 10.6 million → extract only 10.6, not 10,600,000.

- (3) Standardization of Format Remove formatting symbols while preserving the numerical value:

- • Remove currency symbols (e.g., USD).
- • Remove percentage signs (e.g., %).
- • Remove commas (e.g., 1,000 → 1000).

- D.4 Agreement Computation

We report both raw agreement and Cohen’s 𝜅 to quantify interannotator reliability.

Let 𝑁 denote the total number of instances, and let 𝑛𝑦𝑦 be the number of cases on which both annotators assign the same label 𝑦. The raw agreement is

∑︁

1 𝑁

𝑛𝑦𝑦. (4)

𝑃𝑜 =

𝑦

To adjust for agreement occurring by chance, we compute Cohen’s 𝜅. Let 𝑝1𝑦 and 𝑝2𝑦 denote the marginal probabilities that annotator 1 and annotator 2 assign label 𝑦, respectively. The expected chance agreement is

𝑃𝑒 = ∑︁

𝑝1𝑦 𝑝2𝑦, (5) and Cohen’s 𝜅 is

𝑦

𝑃𝑜 − 𝑃𝑒 1 − 𝑃𝑒

. (6)

𝜅 =

Following Landis and Koch [12], our 𝜅 = 0.81 indicates substantial agreement.

### E Evaluation Metrics

To provide a fair evaluation of overall benchmark performance, we adopt a set of metrics, focusing primarily on macro-level and micro-level evaluation strategies inspired by the previous work [22]. Macro-level evaluation computes precision, recall, and F1 scores independently for each BIO-concept label derived from the USGAAP taxonomy, and then averages them without weighting. This ensures that each concept, including rare or infrequent ones, contributes equally to the final score, making it especially suitable for domains with skewed label distributions. In contrast, micro-level evaluation aggregates token-level true positives, false positives, and false negatives across all labels before computing precision, recall, and F1. This approach emphasizes the model’s overall tagging accuracy by treating every token equally and thus better reflects performance on frequent concepts. Together, these two metrics provide a balanced view of both per-concept performance and overall tagging quality.

For the FinNI subtask, the objective is to extract correct (entity, type) pairs, that is (Fact, Type), from the financial document. Let G = (𝑒𝑖,𝑙𝑖) denote the set of ground-truth (entity, type) pairs, and P = (𝑒𝑖′,𝑙𝑖′) denote the set of predicted (entity, type) pairs. We evaluate the performance based on the following metrics:

Precision = |G ∩ P|

(7)

|P|

Recall = |G ∩ P|

(8)

|G|

2 × Precision × Recall Precision + Recall

(9)

F1 =

where (𝑒𝑖,𝑙𝑖) = (𝑒′𝑗,𝑙′𝑗) if and only if both the entity span 𝑒 and its assigned type 𝑙 exactly match.

For theFinCL subtask,Given aset of queries Q = {𝑞1,𝑞2, . . .,𝑞𝑁 },

where each 𝑞𝑖 is associated with a ground-truth concept 𝑐𝑖∗ and a predicted concept 𝑐ˆ𝑖, the accuracy is defined as:

∑︁𝑁

1 𝑁

𝛿(𝑐ˆ𝑖,𝑐𝑖∗) (10)

Accuracy =

𝑖=1

where 𝛿(𝑐ˆ𝑖,𝑐𝑖∗) = 1 if 𝑐ˆ𝑖 = 𝑐𝑖∗, and 0 otherwise.

### F Evaluation Models Details

Table 10 provides an overview of the models evaluated in this study, categorized by openness, domain specialization, and architectural foundation. The evaluation covers a diverse range of models:

- • Closed-source LLMs: We include GPT-4o [10], accessed via OpenAI’s API, as a representative of cutting-edge proprietary models with demonstrated performance across a variety of NLP tasks. Although model size details are undisclosed, GPT4o serves as an upper-bound reference in our benchmark.
- • Open-source LLMs: This group encompasses recent, highperforming open models such as DeepSeek-V3 (685B) [13], DeepSeek-R1-Distill-Qwen (32B) [5], and multiple variants of Qwen3 [26](ranging from 0.6B to 32B). We also include Llama-3.3, Llama-3.2, and 3.1 variants [8] (70B, 3B, and 8B), Llama-4-Scout-17B-16E-Instruct[16](109B),aswellasGoogle’s

- gemma-2-27B [25], to ensure architectural diversity and scalability comparison. These models are primarily instructiontuned and optimized for general-purpose NLP tasks.
- • Financial-specificLLMs: Weevaluate Fino1-8B [18], a domainspecialized model trained on financial corpora, designed to better capture the terminology and structure unique to financial disclosures. This category allows us to assess the benefits of domain adaptation in complex tagging and reasoning tasks.
- • Pretrained Language Models (PLMs): To establish strong baselines, we include non-generative encoder models: BERTlarge [6], FinBERT [1], and SECBERT [14]. These models have been widely used in prior financial NLP tasks and allow for a comparative analysis between generative LLMs and traditional pretrained models in terms of domain understanding and structured output capability.

Together, these models offer a comprehensive evaluation spectrum, from general-purpose to domain-specific, encoder-based to decoder-based, and open to closed source, facilitating an in-depth assessment of their performance across our proposed benchmark tasks.

#### Table 10: Model categories and corresponding repositories.

Model Size Source Closed-source Large Language Models GPT-4o - gpt-4o-2024-08-06 Open-source Large Language Models DeepSeek-V3 685B deepseek-ai/DeepSeek-V3 Llama-4-Scout-17B-16E-Instruct 109B meta-llama/Llama-4-Scout-17B-16E-Instruct Llama-3.3-70B-Instruct 70B meta-llama/Llama-3.3-70B-Instruct DeepSeek-R1-Distill-Qwen 32B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B Qwen3-32B 32B Qwen/Qwen3-32B gemma-2-27b-it 27B google/gemma-2-27b-it Qwen3-14B 14B Qwen/Qwen3-14B

- Llama-3.1-8B-Instruct 8B meta-llama/Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct 3B meta-llama/Llama-3.2-3B-Instruct Qwen3-1.7B 1.7B Qwen/Qwen3-1.7B Qwen3-0.6B 0.6B Qwen/Qwen3-0.6B

Financial-specific Large Language Models Fino1 8B TheFinAI/Fino1-8B

Pretrained Language Models BERT-large ∼340M google-bert/bert-large-uncased FinBERT ∼110M ProsusAI/finbert SECBERT ∼110M nlpaueb/sec-bert-base

### G The details for the fine-tuning PTMs G.1 Training data collection and processing

Similar to the collection process for the FinTagging benchmark data, we gathered an additional 10 annual 10-K financial reports filed with the SEC for the period from February 13, 2024, to February 13, 2025, as summarized in Table 11. These reports contain a total of 33,848 standard taxonomy-tagged facts. Using BeautifulSoup to parse these documents, we identified 22,847 narrative sentences (approximately 5.5 million characters) and 1,236 financial tables. The companies included in this dataset follow the XBRL standard, ensuring comprehensive coverage for training PTMs.

After collection, we employed the same procedure to filter texts and tables, subsequently annotating numerical entities, entity types, and US-GAAP tags (concepts). Finally, as detailed in Table 12, we generated a total of 1,116 sentences and 953 tables as the training set for PTMs. Specifically, the sentence-level data consists of 558

#### Table 11: Financial report statistics summary for raw training data

Item Information

Report type 10-K Period 2024-02-13 to 2025-02-13 #Company 10 #Sentence 22,847 #Table 1,236 #Characters 5,539,198 #Standard Tags 33,848

#### Table 12: Statistics of training data (tokens calculated with “cl100k_base” tokenizer, ± standard deviation).

Structure Pos/Neg #Instance Avg. Tokens/S Avg. Entities/S Avg. Concepts/S Total Entities Unique Concepts Sentence

Positive 558

84.24 ± 69.29 1.22 ± 1.78 1.22 ± 1.78

Negative 558 Table

25,199 1,435

Positive 594

1281.86 ± 6438.37 25.00 ± 213.77 25.00 ± 213.77 Negative 359

positive and 558 negative instances, averaging approximately 84.24 tokens (± 69.29), with 1.22 annotated entities and concepts per sentence. The table-level data comprises 594 positive and 359 negative instances, with a significantly higher average of 1,281.86 tokens (± 6,438.37), and approximately 25 entities and concepts annotated per table. Overall, the annotated dataset includes 25,199 entities, covering 1,435 unique US-GAAP concepts.

However, to align with the extreme classification format used in previous XBRL tagging benchmarks, we directly adopt the USGAAP tags as entity labels, annotating each token in sentences and tables using the BIO scheme. Specifically, B denotes the beginning of an entity phrase, I marks the continuation (inside) of an entity phrase, and O indicates tokens outside of any entity. As shown in Figure 8, "4.9" and "4.5" are single-token numerical entities labeled only withaBprefix(e.g., “B-us-gaap:AccountsReceivableNetNoncurrent”). To comprehensively cover all US-GAAP tags, we combine the entire set of 17,688 tags from the US-GAAP 2023 and 2024 taxonomy with the BIO labeling scheme to construct an extreme classification label space, resulting in 34,777 unique entity labels (2 × 17388 + 1).

After constructing the training set, we reconstruct the testing set from the original benchmark dataset. The training settings are detailed below.

### G.2 Training settings

We fine-tune three pretrained models, BERT-large [6], FinBERT [1], and SECBERT [14], on our training set using the HuggingFace Transformers library. All models are trained with a batch size of 4, a learning rate of 3e-5, and for 20 epochs. Optimization is performed using AdamW without gradient accumulation or early stopping. Token classification heads are randomly initialized and trained jointly with the base encoder. Input sequences are tokenized with a maximum length of 512, and labels are aligned at the sub-token level following the BIO tagging scheme. Loss is computed only on the first sub-token of each word to avoid misalignment bias.

Training is conducted on two NVIDIA A5000 GPUs (24GB each) using data parallelism for 24 hours. All other hyperparameters follow the default settings in the HuggingFace Trainer API. Models are evaluated using the checkpoint from the final epoch. All experiments are run under a fixed random seed to ensure reproducibility.

[Figure 9]

Figure 8: An example for training set annotation with BIO scheme.

Table 13: Acc@k retrieval performance on the FinCL task. FWC: fixed window-based; SAC: structure-aware.

Strategy Structure Acc@1 Acc@10 Acc@20 Acc@50 Acc@100 Acc@150 Acc@200

Sentence 0.0658 0.2614 0.3562 0.4650 0.5618 0.6114 0.6502

FWC

Table 0.0000 0.0029 0.0038 0.0042 0.0050 0.0065 0.0089 Overall 0.0055 0.0237 0.0316 0.0409 0.0492 0.0544 0.0608

Sentence 0.0696 0.2742 0.3631 0.4798 0.5701 0.6125 0.6535

SAC

Table 0.0159 0.0872 0.1245 0.1938 0.2452 0.2723 0.2959 Overall 0.0202 0.1152 0.1534 0.2188 0.2727 0.3012 0.3274

### H Retrieval Results in FinCL subtask

We investigate the impact of different context construction strategies for the queried entity at the retrieval stage. We consider two approaches: Fixed-Window Context (FWC) and Structure-Aware Context (SAC). In the FWC strategy, context is constructed by extracting a fixed window of 50 characters before and after the entity mention, regardless of whether the entity appears in a sentence or a table. In contrast, the SAC strategy builds context based on the structural location of the entity: if the entity appears in a sentence, the entire sentence is used; if the entity appears in a table, we linearize the entire row into a Markdown-style key-value format to serve as the context.

From Table 13, we observe that the SAC strategy consistently outperforms FWC, particularly in the table context, highlighting the importance of aligning the context window with the underlying structural unit.

For sentence-based entities, both strategies yield comparable results, with SAC achieving slightly higher Acc@1 (0.0696 vs. 0.0658) and showing marginal improvements across all cutoff values (e.g., Acc@100 of 0.5701 vs. 0.5618). This suggests that even for relatively unstructured text, preserving sentence boundaries provides minor benefits over fixed-length windows.

In contrast, for table-based entities, the performance gap is substantial. SAC achieves significantly higher retrieval accuracy (e.g., Acc@100 of 0.2452 vs. 0.0050), indicating that row-level context is far more informative than arbitrary character windows when dealing with tabular structures. FWC performs poorly in this setting, likely due to the fragmented and semantically sparse nature of partial table text.

When aggregating results across both structures, SAC outperforms FWC by a wide margin at all retrieval depths (e.g., Acc@200 of 0.3274 vs. 0.0608). These findings underscore the importance of structure-aware context construction, especially in scenarios where inputs span multiple formats such as sentences and tables.

#### Table 14: Error cases from DeepSeek-V3 rerank.

###### Case 1: $31m

Gold Concept us-gaap:RestrictedCashAndCashEquivalentsAtCarryingValue Gold Concept Rank 18 / 200 (not in Top-5) Model Prediction us-gaap:RestrictedCashAndCashEquivalentsCurrent

###### Case 2: $121m

Gold Concept us-gaap:RestrictedCashAndCashEquivalentsNoncurrent Gold Concept Rank 2 / 200 (in Top-5) Model Prediction us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents

### I Error cases

To better illustrate the semantic ambiguity challenge, we present a representativeerror casefrom ourbest-performing model, DeepSeekV3. Consider the following input paragraph:

“Cash equivalents include term deposits with banks, money market funds, and all highly liquid investments with original maturities of three months or less...... At December 28, 2024, we had restricted cash of $31 million recorded in other current assets and restricted cash of $121 million recorded in other non-current assets.”

This paragraph contains 2 “monetary entities”: 31 and 121. The gold concepts and predicted concepts that are assigned to both entities are shown in Table 14. For case 1, the model’s choice was likely influenced by the phrase “recorded in current assets”. Yet the paragraph describes the overall composition of restricted cash, and the correct concept reflects this aggregate view rather than a single classification. For case 2, although both concepts denote non-current restricted cash, the model selected a narrower variant, showing the difficulty of distinguishing between highly similar concepts with subtle differences in scope.

Prompt Template for FinNI Subtask

You are a financial information extraction expert specializing in identifying financial numerical entities in XBRL reports. Your task is to extract all such numerical entities from the provided text or serialized <table ></ table > data and classify them into one of five categories:

- - "integerItemType": Counts of discrete items , such as the number of employees or total transactions.
- - "monetaryItemType": Financial amounts expressed in currency , such as revenue , profit , or total assets.
- - "perShareItemType": Per -share values , such as earnings per share (EPS) or book value per share.
- - "sharesItemType": Counts of shares , such as outstanding shares or ownership stakes.
- - "percentItemType": Ratios or percentages , such as tax rates , growth rates , or discount rates , usually expressed with a percentage symbol ("%").

Important Instructions:

- (1) Financial numerical entities are not limited to Arabic numerals (e.g., 10,000). They may also appear in word form (e.g., "ten million"), which must be correctly identified

and converted into standard numerical format.

- (2) Not all numbers in the text should be extracted. Only those that belong to one of the five financial entity categories above should be included. Irrelevant numbers (such as

phone numbers , dates , or general IDs) must be ignored.

- (3) If a number is followed by a magnitude term (e.g., Hundred , Thousand , Million , Billion ), do not expand it into the full numerical value.

- * "Two hundred" -> Extract only "two", not "200".
- * "10.6 million" -> Extract only "10.6", not "10,600,000".

- (4) Standardize numerical formatting by removing currency symbols (e.g., "USD"), percentage signs ("%"), and commas (",") while preserving the numeric value. These elements must be removed to ensure consistency.
- (5) Output the extracted financial entities in JSON list format without explanations , structured as follows: {"result":[{"Fact": <Extracted Numerical Entity >, "Type": < Identified Entity Type >}]}

Input: {text/table} Output:

#### Figure 9: Prompt template used for the FinNI subtask.

Prompt Template for FinCL Subtask (Reranking)

You are a financial tagging assistant trained in US-GAAP taxonomy.

Given a query consisting of an entity , its type , its surrounding context , and the source format ( either text or table), your task is to select the single most appropriate US-GAAP tag from a list of 200 candidate tags.

Make your decision by carefully analyzing the meaning and context of the entity and matching it with the semantics of the tags.

Only output one tag , the best match. Do not explain or list multiple tags. The output is a JSON format: {"result": <the best matched tag >}.

Input Query: <entity > + <entity type > + <context > Candidate Tags: {Top 200 US-GAAP tags}

Answer:

#### Figure 10: Prompt template used for the Reranking stage in the FinCL subtask.

Prompt Template for Ablation

You are an XBRL tagging expert specializing in annotating financial numerical facts in XBRL reports.

Your task is to (1) extract all such numerical entities from the provided text or serialized < table ></table > data , (2) classify them into one of five categories , and (3) assign an appropriate US-GAAP tag to each entity.

Categories:

- - "integerItemType": Counts of discrete items , such as the number of employees or total transactions.
- - "monetaryItemType": Financial amounts expressed in currency , such as revenue , profit , or total assets.
- - "perShareItemType": Per -share values , such as earnings per share (EPS) or book value per share.
- - "sharesItemType": Counts of shares , such as outstanding shares or ownership stakes.
- - "percentItemType": Ratios or percentages , such as tax rates , growth rates , or discount rates , usually expressed with a percentage symbol ("%").

US-GAAP tags:

- - A US-GAAP tag is a standardized semantic label used in XBRL filings to identify specific financial concepts defined by the U.S. Generally Accepted Accounting Principles (GAAP).

Each tag represents a distinct accounting item and enables consistent , machine readable financial reporting.

- - Examples: "us-gaap:AssetsCurrentAbstract", "us-gaap:AccruedInsuranceNoncurrent".

Important Instructions:

- (1) Financial numerical entities are not limited to Arabic numerals (e.g., 10,000). They may also appear in word form (e.g., "ten million"), which must be correctly identified

and converted into standard numerical format.

- (2) Not all numbers in the text should be extracted. Only those that belong to one of the five financial entity categories above should be included. Irrelevant numbers (such as

phone numbers , dates , or general IDs) must be ignored.

- (3) If a number is followed by a magnitude term (e.g., Hundred , Thousand , Million , Billion ), do not expand it into the full numerical value.

- * "Two hundred" -> Extract only "two", not "200".
- * "10.6 million" -> Extract only "10.6", not "10,600,000".

- (4) Standardize numerical formatting by removing currency symbols (e.g., "USD"), percentage signs ("%"), and commas (",") while preserving the numeric value. These elements must be removed to ensure consistency.
- (5) You should assign the most appropriate US-GAAP tag to each identified entity based on your internal understanding of the 2024 US-GAAP taxonomy.
- (6) Output the extracted financial entities in JSON list format without explanations , structured as follows: {"result ":[{"Fact": <Extracted Numerical Entity >, "Type": < Identified Entity Type >, "Tag": <Assigned US-GAAP tag >}]}

Input: {text/table} Output:

#### Figure 11: Prompt template used for ablation study.

