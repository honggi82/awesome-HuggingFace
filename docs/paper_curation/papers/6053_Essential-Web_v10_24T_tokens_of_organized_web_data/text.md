arXiv:2506.14111v2[cs.CL]19Jun2025

# ESSENTIAL-WEB V1.0: 24T tokens of organized web data

Essential AI San Francisco, CA research@essential.ai

## Abstract

Data plays the most prominent role in how language models acquire skills and knowledge. The lack of massive, well-organized pre-training datasets results in costly and inaccessible data pipelines. We present ESSENTIAL-WEB V1.0, a 24-trillion-token dataset in which every document is annotated with a twelve-category taxonomy covering topic, format, content complexity, and quality. Taxonomy labels are produced by EAI-Distill-0.5b, a fine-tuned 0.5b-parameter model that achieves an annotator agreement within 3% of Qwen2.5-32B-Instruct. With nothing more than SQL-style filters, we obtain competitive web-curated datasets in math (-8.0% relative to SOTA), web code (+14.3%), STEM (+24.5%) and medical (+8.6%). ESSENTIALWEB V1.0 is available on HuggingFace: EssentialAI/essential-web-v1.0.

### Training High-Recall Classifier

curation timeline: weeks to months

### ESSENTIAL-WEB V1.0 Approach

curation timeline: hours to days

ESSENTIAL-WEB V1.0

100TB Deduplicated, Filtered CC

100TB

>1B distinct document labels

| | |
|---|---|
| | |

train base math classifier

#### subject == math arbitrary subject

100TB run inference

>100B high recall math high recall dataset

manually inspect output

curate new training data

retrain

>100B high recall math

Essential AI (2025)

## Contents

- 1 Introduction 3
- 2 Related Work 4

- 2.1 Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

3 Taxonomy 5

- 3.1 Formal Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 4 Downstream Results 6

- 4.1 Experimental Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.2 Math . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.3 Code . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.4 Medical . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.5 STEM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5 Developing Taxonomy 11

- 5.1 Measuring Orthogonality, Correctness, & Expressivity . . . . . . . . . . . . . . . . . . . . . . 11
- 5.2 Teacher Model Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 6 Running at Scale 15

- 6.1 Performance Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 6.2 Distillation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 6.3 EAI-Distill-0.5b Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 7 Conclusion 18

- 3.2 Desiderata . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.3 Development Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.4 Selected Categories . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 7.1 Broader impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 7.2 Possible Next Steps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

##### A Appendix 25

## 1 Introduction

Among innovations that can accelerate the path to more intelligent AI models, data innovations frequently outpace others, such as architectures or optimizers [Kaplan et al., 2020, Beck et al., 2024, Bahri et al., 2024, Sorscher et al., 2023, Shen et al., 2024a, Liu et al., 2025]. A careful curation of the bytes consumed by large language models (LLMs) enables greater control over the skills they acquire. With pre-training datasets scaling to trillions of tokens, a detailed examination of their content can be intimidating. Moreover, open-weight models rarely disclose the composition of their datasets, and the broader ecosystem is trending toward reduced transparency as pre-training datasets continue to grow. This introduces challenges in reproducibility and auditing models – challenges that will grow as models become more autonomous. An accessible and interpretable open-data ecosystem is therefore essential for training competitive models in the open.

Training Tokens (Trillions)

40

Llama 1 Qwen 2 DeepSeek 3

30

20

10

0

Q12023Q22023Q32023Q42023Q12024Q22024Q32024Q42024Q12025Q22025

Model Release Date

Figure 1: LLM pre-training dataset sizes over time.

Open-source pre-training datasets are divided into: (1) enormous, general-purpose datasets sorted by uninterpretable quality classifiers (2) smaller, domain-specific datasets curated with bespoke, complex pipelines. Both general and domain-specific datasets are unstructured, difficult to explore, and difficult to iteratively improve. Constructing these datasets requires significant computational resources given the scale and complexity of the data pipelines. Once the datasets are publicly released, their metadata rarely enables tuning beyond modifying classifier thresholds.

We release ESSENTIAL-WEB V1.0, a 24-trillion-token dataset with expressive and extensive metadata at a document-level. This metadata includes subject matter, web page type, content complexity, and document quality. Practitioners can now rapidly and inexpensively curate new datasets by writing SQL-like filters that utilize these metadata columns. Suppose a researcher wants to prepare a multi-billion-token chemistry corpus using publicly-available web data. Today, the researcher must first train a high-recall chemistry classifier, a task hindered by scarce labeled data. Then, the classifier is run across hundreds of millions of documents to recall sufficient data. With ESSENTIAL-WEB V1.0, a researcher can filter for chemistry, skip low-quality web pages (ads, product listings), and surface reasoning-dense documents — all with a query that takes under 15 minutes to write.

To construct ESSENTIAL-WEB V1.0, we take advantage of powerful open-weight LLMs to synthetically label web documents with a 12-category taxonomy. We utilize these labels to train a more efficient classifier, EAI-Distill-0.5b, and run inference on 23.6B documents. Inference at this scale requires ≈90k AMD MI300x GPU-hours.4 We expect this one-off inference cost to be amortized as the community iterates on datasets and methods utilizing ESSENTIAL-WEB V1.0.

3Llama dataset sizes sourced from Touvron et al. [2023a], Touvron et al. [2023b], Llama Team [2024], Meta AI [2025] 3Qwen dataset sizes sourced from Bai et al. [2023], Yang et al. [2024], Qwen [2025], Yang et al. [2025] 3DeepSeek dataset sizes sourced from DeepSeek-AI [2024b], DeepSeek-AI [2024c], and DeepSeek-AI [2025] 4The inference job ran on 512 AMD MI300x for about 1 week.

To demonstrate the utility of ESSENTIAL-WEB V1.0, we construct simple filters to curate high-performing datasets in math, web code, STEM, and medical domains. Our math dataset performs within 8.0% of SOTA and our web code, STEM, and medical datasets outperform SOTA by 14.3%, 24.5%, 8.6% respectively.

## 2 Related Work

Most web-scale datasets rely on Common Crawl, an 18-year crawl of more than 250 billion web pages [Common Crawl Foundation, 2025]. Research on Common Crawl data can be naturally grouped into (1) heuristic filtering and de-duplication (2) monolithic, model-based filtering (3) domain-focused, model-based filtering, and (4) taxonomies for data curation.

Heuristic filtering and de-duplication. C4 (160B tokens, 2019) pioneered heuristic filtering for Common Crawl. C4 proposes a processing pipeline to deduplicate and filter low quality text using document-level statistical heuristics [Raffel et al., 2023]. It was followed by an explosion of large-scale, open-source datasets and pipelines such as RefinedWeb (600B tokens), SlimPajama (600B tokens), Dolma-CC (3T tokens), and RedPajama (30T tokens) [Penedo et al., 2023, Weber et al., 2024a, Shen et al., 2024b, Soldaini et al., 2024].

Monolithic, model-based filtering. In 2024, we see the introduction of monolithic model-based classifiers applied after well-tuned heuristic filtering and de-duplication. FineWeb-Edu (1.3T tokens) filters documents with an education-density classifier whose labels are bootstrapped from Llama-3-70B-Instruct [Penedo et al., 2024]. Similarly, DCLM-baseline (3.6T tokens) filters documents using a fastText classifier trained to detect high instruction density [Li et al., 2025]. Nemotron-CC ensembles several quality classifiers into one score before sampling Common Crawl [Su et al., 2025]. While classifier-ranked datasets excel on broad language benchmarks, they lag on math and code evaluations. Their minimal metadata, often just the URL and a handful of quality scores, makes post-hoc domain filtering with existing metadata almost impossible. Researchers therefore build domain-specific processing pipelines to compensate.

Domain-focused, model-based filtering. Math, for instance, is rare (less than 0.5% of Common Crawl)5 yet beneficial for reasoning skills. OpenWebMath (12B tokens) builds a LaTeX detection pipeline, develops a custom HTML-to-text extractor to preserve mathematical formatting, and a math-detecting classifier [Paster et al., 2023]. DeepSeek Math (120B tokens, not publicly released) iteratively trains a classifier to maximize recall of math documents resulting in a significantly larger dataset [Shao et al., 2024]. MegaMath (264B web-based math tokens) builds multi-step classification, extraction, and processing pipelines for math and code [Zhou et al., 2025]. FineMath trains an initial classifier to recall math documents, re-extracts billions of pages with a math-specific extractor, and re-classifies these documents using a second, more powerful math classifier [Allal et al., 2025]. Math is not the only domain that has been targeted: OpenCoder iteratively trains a classifier to detect web code and TheBlueScrubs-v1 trains a classifier to detect medical documents [Huang et al., 2025, Felipe et al., 2025].

Taxonomies for curating training data. Wettig et al. [2025] introduce taxonomies to structure web data. The authors develop a two-category taxonomy that measures topic and format, train a classifier using synthetic labels generated by Llama-3.1-Instruct models, label 200B tokens of Common Crawl, and demonstrate the utility of a combining model-based quality filters and a taxonomy. The authors introduce normalized mutual information to measure label and category redundancy. Using normalized mutual information, they show that embedding-based clusters mostly overlap with topical labels, not format ones.

Unlike prior multi-trillion-token datasets, our work, ESSENTIAL-WEB V1.0 attaches a 12-field taxonomy to every document, enabling both general and domain-specific filtering through a single interface.

- 2.1 Contributions We make four contributions:

- 1. ESSENTIAL-WEB V1.0 a deduplicated, 23.6B-document (24T token) corpus drawn from Common Crawl and tagged with EAI-TAXONOMY, a 12-field taxonomy, covering topic, web page type, content complexity, and quality.
- 2. Downstream validation. Simple SQL filters over the taxonomy produce datasets competitive with the best open-source, web-based baselines on math (-8.0% relative to SOTA), code (+14.3%), STEM (+24.5%), and 5Calculated using our subject matter label, FDC, to filter for Mathematics.

- medical (+8.6%). Our results demonstrate that taxonomy-curated datasets are competitive with complex, domain-specific pipelines with no domain-specific training.
- 3. Taxonomy evaluation toolkit. We introduce a metric suite of normalized mutual information (NMI) to gauge category independence [Wettig et al., 2025], annotator agreement using a variant of Cohen’s κ to ensure clear decision boundaries between category labels, and domain-recall to measure how well we recall high-value domains.
- 4. Efficient annotator model. We release EAI-Distill-0.5b a 0.5B-parameter classifier produced by finetuning Qwen2.5-0.5b-Instruct using labels from Qwen-2.5-32B-Instruct [Qwen, 2025]. It labels the entire dataset in ≈90k MI300x GPU-hours while remaining within 3%, 14%, 1% of the teacher on annotator agreement, NMI, and domain-recall.

## 3 Taxonomy

##### 3.1 Formal Definition

Definition 1. A taxonomy is a finite set T = {C1,...,Ck} of categories. Each category Ci has a non-empty, finite label set Li. For a document d, the taxonomic annotation is:

T(d) = (λ1,µ1),...,(λk,µk) , λi ∈ Li, µi ∈ Li \ {λi} ∪ {⊥},

where λi is the primary label for category Ci, µi is an optional secondary label distinct from λi (useful when a document legitimately fits two labels), and ⊥ denotes abstention. All categories and label sets are fixed a priori, allowing us to train a single static classifier.

##### 3.2 Desiderata

A useful metaphor for approaching taxonomy development is to think of the categories as axes in a highdimensional grid space. Each document lives on a coordinate in that space, namely the category/label pairs assigned to it. We argue that a well-designed, web-scale taxonomy should satisfy five desiderata:

- 1. Orthogonality. Every category should contribute information that is largely independent of the others analogous to orthogonal axes in Euclidean space.
- 2. Expressivity. The coordinate system must be sufficiently fine-grained that unions of category/label pairs can assemble a wide variety of targeted subsets.
- 3. Correctness. Expressive, orthogonal axes are useless if labels are arbitrary. Annotators, human or LLM, must consistently assign the same labels to a document far more often than by chance.
- 4. Efficiency. Scaling to Common Crawl demands a fast and high-performing classifier.
- 5. Effectiveness. Ultimately, a taxonomy matters only if it delivers in downstream performance.

##### 3.3 Development Methodology

1 Taxonomy Development

2 Teacher Model Selection

3 Student Model Distillation

4 Large-scale Inference

5 Downstream Performance

- Metrics:
- - orthogonality
- - expressivity
- - correctness

###### Candidates:

- - Qwen2.5-32B-Inst.
- - Qwen2.5-72B-Inst.
- - DeepSeekV3

Performance: 3% relative drop in annotator κ Efficiency: max throughput

process 23.6B docs Curate datasets for:

- - math
- - web code
- - medical
- - STEM

Figure 2: Overview of our five-stage methodology for developing and deploying EAI-TAXONOMY. The development of EAI-TAXONOMY and EAI-Distill-0.5b (Figure 2) are discussed in the following sections:

- • Downstream Results (Section 4, p. 6) compares datasets for math, web code, medical, and STEM curated using EAI-TAXONOMY against top-performing, open-source datasets in each domain.
- • Measuring Orthogonality, Correctness, & Expressivity (Section 5.1, p. 11) defines the metrics used to evaluate orthogonality, correctness, and expressivity and introduces held-out evaluation sets.
- • Teacher Model Selection (Section 5.2, p. 13) evaluates top performing open-source LLMs and motivates the selection of Qwen2.5-32b-Instruct as a teacher model.
- • Running at Scale (Section 6, p. 15) discusses the inference optimizations, training recipe, and performance evaluation of EAI-Distill-0.5b, the student model.

##### 3.4 Selected Categories

Each web page receives labels for 12 categories across 5 logical groupings, which we refer to as EAI-TAXONOMY. In Table 1, we briefly introduce our categories and explain what each tries to capture, see Appendix A.3 for more details. Many of these categories are inspired by or directly taken from existing work in the open-source data curation community.6 ESSENTIAL-WEB V1.0 has 14.1M unique combinations of primary labels and 1.2B unique combinations of primary / secondary labels across all 23.6B docs.

##### Description Categories

FDC. Free Decimal Correspondence is a public-domain analogue of the Dewey Decimal System.7 Labels subject matter hierarchically.

- Level 1: broad topic (ex: 5 - Science)
- Level 2: fine topic (ex: 51 - Mathematics)
- Level 3: very fine topic (ex: 512 - Algebra)

Bloom. Educational-objective taxonomy. Cognitive Process: mental effort required

Knowledge Domain: content abstraction Document Type. Web page types. V1: broad types

V2: fine types

Content Quality. Measures rigor and target audience. Reasoning Depth: thinking steps Educational Level: reader background Technical Correctness: correctness

Extraction. Crawl artifacts and text extraction errors. Extraction Artifacts: HTML extraction errors

Missing Content: from scrape or extraction Table 1: Taxonomy categories and their descriptions.

## 4 Downstream Results

We show that taxonomy-based datasets with no domain-specific training are competitive with bespoke pipelines built to target math, web code, medical and STEM data.

##### 4.1 Experimental Protocol

In line with recent work such as DeepSeekMath, SmolLM2, and MegaMath, we anneal "pre-trained" models with domain-specific datasets to evaluate performance [Shao et al., 2024, Allal et al., 2025, Zhou et al., 2025]. We train two 2.3B parameters transformer models for 320B tokens, 8× the Chinchilla compute-optimal ratio [Hoffmann et al., 2022].8 To evaluate domain-specific datasets, we decay the learning rate of one of the base models to zero while training on 80B tokens of the new dataset. We also anneal on the original data mix to provide a baseline. The rationale behind starting with a base model is that it improves signal on difficult benchmarks such as MMLU

- 6The Free Decimal Correspondance was designed by Ockerbloom [2010]. The Bloom taxonomy was designed by Anderson and Krathwohl [2001]. Document Type V2 was designed by Wettig et al. [2025]. Reasoning Depth and Technical Correctness were designed by Yuan et al. [2025]. Education Level was designed by Penedo et al. [2024].
- 7"Dewey," "Dewey Decimal," "Dewey Decimal Classification," and "DDC" are trademarks of OCLC. 8When calculating Pareto, we ignore the size of the tied embedding/un-embedding matrix, which is 295M parameters.

Learning Rate

General Training Domain-Specific Annealing

- 3 × 10−5
- 3 × 10−6

2 320 400

Tokens Consumed (B tokens)

Figure 3: We train two base models: (1) General Base (100% web data) and (2) Code Base (50% web data, 50% code data) for 320B tokens. We then evaluate all domain-specific EAI-TAXONOMY and top-performing, web-curated datasets by annealing one of the two base models for 80B tokens.

and GSM8K, as opposed to training on each dataset from scratch for 80B tokens. After annealing, each model has seen 400B tokens, 10× Chinchilla Pareto.

We vary the data composition of the two base models, while keeping all other training hyper-parameters fixed. The model architecture, configuration, and training hyper-parameters can be found in Appendix A.6.

- • General-Base: DCLM-baseline [Li et al., 2025]
- • Code-Base: 50% DCLM-baseline + 50% Python from Stack v2 Dedup [Lozhkov et al., 2024]

DCLM-baseline. DCLM-baseline is a 3.6T token pre-training dataset based on Common Crawl. It is deduplicated, heuristically-filtered, and labeled using a model-based classifier. For classification, the authors train a fastText classifier on instruction-formatted data from OpenHermes 2.5 and r/ExplainLikeImFive subreddit [Teknium, 2023]. DCLM-baseline is curated by selecting the top 10% of documents after de-duplication and heuristic filters based on this classifier score [Li et al., 2025].9 The HuggingFace dataset card notes the dataset is not intended for domains such as code and math.

Stack v2. The Stack v2 is a dataset of 104.2M github repositories collected from Software Heritage [Software Heritage Foundation, 2025]. We use the Minhash LSH deduplicated version [Lozhkov et al., 2024].10 We restrict training to the Python subsect because HumanEval+ and MBPP+ are Python only [Chen et al., 2021, Austin et al., 2021, Liu et al., 2023].

Table 17 and Table 24 in the Appendix contain HuggingFace links for our datasets and public datasets used in this section, respectively.

Decontamination. All datasets were decontaminated with a 13-gram Bloom filter that normalizes text and punctuation before white-space tokenization. The following evals were decontaminated against: GSM8K, MATH, HumanEval, MBPP, MMLU, CareQA, MedMCQA, MedQA-USMLE, and PubMedQA.

Evaluation framework. We run evaluations using lm-eval-harness [Gao et al., 2024]. See Appendix A.4 for details.

9DCLM-baseline can be found on HuggingFace: mlfoundations/dclm-baseline-1.0. 10Stack v2 dedup can be found on HuggingFace: bigcode/the-stack-v2-dedup.

- 4.2 Math There has been extensive work in the open source community to develop high-quality math datasets [Paster et al.,

- 2023, Allal et al., 2025, Zhou et al., 2025, Shao et al., 2024]. We compare the performance of taxonomy-based math datasets with all top performing, public math datasets that are curated from Common Crawl. We only select math datasets that are filtered web data and do not report performance of synthetic math datasets or datasets where an LLM was used to rewrite documents.11 A list of datasets tested and the domain-specific steps used during preparation can be found in Table 2.12 We use General-Base for the annealing experiments. To evaluate performance, we run 8-shot GSM8K, 4-shot MATH, and 5-shot MMLU-Math, which consists of all MMLU subtasks related to math [Hendrycks et al., 2021].13

Dataset Size (B tok) Math extract Math classifier Domain–specific curation notes FineMath 3+ 32 ✓ ✓ 118M math classifer on 7.1B docs OpenWebMath 12 ✓ ✓ math fastText MegaMath Web (Top 10%)14 30 ✓ ✓ iteratively trained math fastText EAI-TAXONOMY Top Math 29 ✗ ✗ EAI-TAXONOMY filter EAI-TAXONOMY Math w/ FM 34 ✗ ✓ EAI-TAXONOMY filter;

FineMath classifier on 116M docs

- Table 2: Overview of top-performing, open-source math datasets and taxonomy-based math datasets along with brief summary of effort to curate. "Math extract" denotes a domain-specific HTML-to-plaintext extraction was used. "Math classifier" indicates a domain-specific classifier was used.

- 4.2.1 Taxonomy-Based Math Datasets

We prepare two taxonony-based math datasets. EAI-TAXONOMY Top Math (29B tokens) targets high-quality math documents that exhibit reasoning and are technically correct, the full filter can be found in Algorithm 2 in Appendix A.8. EAI-TAXONOMY Math w/ FM (34B tokens) filters for any document labeled as 51 - Mathematics. We then label all 116M recalled documents with the FineMath Classifier15 and filter for the top 34B tokens so that the size is comparable to FineMath 3+ (Algorithm 3).

- 4.2.2 Downstream Math Results

Dataset GSM8K MATH MMLU–Math

FineMath 3+ 26.4%±1.4 11.7%±0.4 32.3%±1.5 OpenWebMath 14.6%±1.1 9.3%±0.4 29.9%±1.5 MegaMath Web (Top 10%) 9.8%±0.9 7.9%±0.3 29.9%±1.5 DCLM-baseline 4.8%±0.7 4.4%±0.3 27.0%±1.4

EAI-TAXONOMY Top Math 21.3%±1.3 11.0%±0.4 30.5%±1.5 EAI-TAXONOMY Math w/ FM 22.4%±1.3 11.5%±0.4 30.9%±1.5

- Table 3: Model performance (mean ± standard error) on GSM8K, Hendrycks Math, and MMLU-Math benchmarks. All datasets reported are filtered web data. In Appendix A.7, we include MegaMath Web Pro, which filters MegaMath Web using the FineMath classifier and rewrites the top documents using an LLM.

FineMath 3+ achieves the highest score in GSM8K, with EAI-TAXONOMY Top Math and EAI-TAXONOMY Math w/ FM datasets trailing by -4.0pp (15.2%) and -5.1pp (19.3%) respective. On MATH and MMLU-Math,

- 11We only report filtered web data with no document-level LLM changes to the isolate performance of different data filtering

methods. These results focus on the quality of the natural web data selected by different filtering methods.

- 12We acknowledge that the total dataset preparation cost of the taxonomy is much higher than any of these individual datasets

because we run a 500M parameter transformer to classify 23.6B documents. However, we only focus on domain-specific curation costs in Table 2.

13MMLU-Math: MMLU Abstract Algebra, MMLU College Mathematics, MMLU Elementary Mathematics, MMLU High

School Mathematics, MMLU High School Statistics 13We prepared MegaMath Web (Top 10%) by selecting the top 10% of documents based on the math_score column. 15A 118M parameter transformer-based classifier trained on synthetic labels generated by LLama3-70B-Instruct. More

details can be found on HuggingFace: HuggingFaceTB/finemath-classifier.

EAI-TAXONOMY Math w/ FM performs within standard error of FineMath 3+. Other curated sets (MegaMath Web (Top 10%), OpenWebMath) lag both FineMath and our taxonomy splits on GSM8K and MATH. The full results to the experiments can be found in Table 3. FineMath, MegaMath Web (Top 10%), and OpenWebMath are domain-specific datasets with complex pipelines to maximize performance on mathematics benchmarks. The FineMath classifier targets “high school and early undergraduate levels” of mathematics, which directly caters to GSM8K and MATH. The performance of EAI-TAXONOMY Top Math comes from a simple semantic filter. EAI-TAXONOMY Math w/ FM takes advantage of the FineMath classifier by recalling a small, high-density subset of Common Crawl to reclassify.

To understand the composition of FineMath 3+, we annotate it using the taxonomy. We find that only 61.9% of FineMath 3+ documents are labeled with FDC 51 - Mathematics, as primary or secondary label. By inspecting the FDC categories of the remaining documents in FineMath 3+, we find other common subject matters in the dataset: 53 - Physics, 33 - Economics, and 621 - Applied Physics.

##### 4.3 Code

Recent code LMs mix execution-ready code with code tutorials, documentation, and API docs recalled from web data [Hui et al., 2024, DeepSeek-AI, 2024a, Huang et al., 2025]. We curate two web code datasets using the taxonomy and compare them to the only openly available, large-scale web code dataset: OpenCoder FineWeb Code Corpus (OpenCoder FW) [Huang et al., 2025]. OpenCoder FW is a 49B token dataset curated with a fastText classifier iteratively trained to maximize math and code recall.16 We anneal web code datasets (Table 4) from Code-Base to get signal on code knowledge and code generation evals. During annealing, we train on a 1:1 mixture of the dataset being evaluated and Python from the Stack-Edu [Allal et al., 2025]. To measure code generation performance, we run 0-shot HumanEval+ and 3-shot MBPP++.17 To gauge code-related knowledge, we run 5-shot MMLU-CS, which consists of all MMLU subtasks related to Computer Science.18

Dataset Size (B tok) Code classifier Domain–specific curation notes

OpenCoder FW 49 ✓ iteratively-trained math and web code fastText EAI-TAXONOMY Top Code 145 ✗ EAI-TAXONOMY filter EAI-TAXONOMY Code w/ DCLM 564 ✗ EAI-TAXONOMY filter; DCLM fastText

- Table 4: Overview of existing SOTA open-source web code datasets and taxonomy-based web code datasets along with brief summary of effort to curate. "Code classifier" indicates a domain-specific classifier was used to detect code.

##### 4.3.1 Taxonomy-Based Web Code Datasets

We prepare EAI-TAXONOMY Top Code (145B tokens) to target high-quality code documentation that is technically correct and exhibits intermediate to advanced reasoning (Algorithm 4). In addition, we construct EAI-TAXONOMY Code w/ DCLM (564B tokens), which combines taxonomy filters targeting code documentation and the DCLM classifier, at the same threshold as DCLM-baseline, to filter for instruction-dense documents (Algorithm 5). We add 51 - Mathematics to EAI-TAXONOMY Code w/ DCLM because OpenCoder FW targets both web code and mathematics.

##### 4.3.2 Downstream Code Results

Across the code-generation evaluations, all datasets perform within standard error of each other (Table 5). We see an absolute +15.0pp (46.8%) improvement in the MMLU-CS score from DCLM-baseline to EAI-TAXONOMY Code w/ DCLM. Both DCLM-baseline and taxonomy-based code datasets outperform OpenCoder FW on MMLUCS. However, there is a clear impact on general code knowledge when using a taxonomy-curated web code dataset.

- 16The classifier is trained with a strategy similar to DeepSeek Math [Shao et al., 2024]. The authors use 500,000 LLM-scored code / math documents as an initial training set [Huang et al., 2025].
- 17HumanEval+ and MBPP++ are variants of HumanEval and MBPP with more challenging test cases and corrected ground-truth solutions [Chen et al., 2021, Austin et al., 2021, Liu et al., 2023]. When running 0-shot MBPP+, the model struggles to properly output the termination condition ("[DONE]") in the default lm-eval-harness config given it isn’t lack of explanation in the prompt. Therefore, we run with 3-shot to provide in-context examples of the termination condition.
- 18MMLU-CS: MMLU College Compute Science, MMLU High School Computer Science, MMLU Computer Security

##### Web-Code Dataset HumanEval+ MBPP+ MMLU–CS

DCLM-baseline 28.0%±3.5 45.5%±2.6 32.0%±2.7 OpenCoder FW 26.2%±3.4 45.8%±2.6 27.7%±2.6

EAI-TAXONOMY Top Code 27.4%±3.5 46.6%±2.6 29.0%±2.6 EAI-TAXONOMY Code w/ DCLM 28.7%±3.5 45.0%±2.6 47.0%±2.9

- Table 5: Pass@1 accuracy (mean ± standard error) on 0-shot HumanEval+, 3-shot MBPP+, and accuracy on the MMLU computer-science subset.

- 4.4 Medical

TheBlueScrubs-v1 (24B tokens) is the only public, curated Common Crawl medical corpus. The authors train a logistic regression classifier to detect medical documents and use it to filter SlimPajama, a 627B web dataset [Felipe et al., 2025]. We compare TheBlueScrubs-v1 against DCLM-baseline and taxonomy-based medical datasets by annealing General-Base. We evaluate medical performance using 3-shot PubMedQA, 5-shot CareQAen, 5-shot MedMCQA, 5-shot MedQA-USMLE-4-options, and 5-shot MMLU-Med [Jin et al., 2019, Arias-Duart et al., 2025, Jin et al., 2020, Pal et al., 2022].19

Dataset Size (B tok) Medical classifier Domain–specific curation notes

TheBlueScrubs-v1 24 ✓ logistic regression to detect medical EAI-TAXONOMY Med 433 ✗ EAI-TAXONOMY filter EAI-TAXONOMY Med w/ DCLM 205 ✗ EAI-TAXONOMY filter; DCLM fastText

- Table 6: Overview of existing open-source medical datasets and taxonomy-based medical datasets along with brief summary of effort to curate. "Medical classifier" indicates a domain-specific classifier was used to detect medical.

##### 4.4.1 Taxonomy-Based Medical Dataset

We prepare the EAI-TAXONOMY Med (433B tokens) to target scientific medical documents that exhibit reasoning and are technically correct (Algorithm 6). We then apply the DCLM classifier, at the same threshold as DCLMbaseline, to the EAI-TAXONOMY Med to construct EAI-TAXONOMY Med w/ DCLM (205B tokens, Algorithm 7).

##### 4.4.2 Downstream Medical Results

###### Model CareQA-en MedMCQA MedQA-USMLE PubMedQA MMLU–Med

DCLM-baseline 26.9%±0.6 31.6%±0.7 25.9%±1.2 70.6%±2.0 31.0%±1.5 TheBlueScrubs-v1 25.1%±0.6 32.2%±0.7 25.3%±1.2 69.2%±2.1 25.7%±1.4

Taxonomy Medical 27.7%±0.6 32.5%±0.7 28.1%±1.3 67.0%±2.1 29.5%±1.5 Taxonomy Medical w/ DCLM 31.5%±0.6 32.7%±0.7 30.1%±1.3 68.6%±2.1 39.2%±1.6

Table 7: Accuracy (mean ± standard error) on four medical QA benchmarks and the MMLU medical subset.

Across medical evaluations, EAI-TAXONOMY Med w/ DCLM either achieves the best performance or performs within standard error of the best dataset (Table 7). Both taxonomy-based medical datasets are able to perform above random chance (≈25%) on MedQA-USMLE, where DCLM-baseline and TheBlueScrubs-v1 are unable to do so. TheBlueScrubs-v1 is also unable to perform above chance on CareQA-en. Across all evals, EAI-TAXONOMY Med w/ DCLM beats DCLM-baseline by +3.2pp (8.6%) and TheBlueScrubs-v1 by +4.9pp (13.8%).

19MMLU-Med: MMLU Anatomy, MMLU Clinical Knowledge, MMLU College Biology, MMLU College Medicine, MMLU Medical Genetics, MMLU Professional Medicine. These categories were grouped for medical evaluation by Singhal et al. [2022]

##### 4.5 STEM

In addition to evaluating EAI-TAXONOMY in domains with existing, multi-billion-token baseline datasets curated from Common Crawl, we also select a domain where we wish there was a large-scale dataset available. Given the importance of STEM domains for reasoning and benchmarking performance of LLMs, we curate a large, STEM-specific dataset. We select two high-performing, general datasets as baselines: DCLM-baseline and FineWeb-Edu [Penedo et al., 2024]. We benchmark the performance of each dataset (Table 8) by annealing General-Base.

##### Dataset Size (T tok.) Domain–specific curation notes

DCLM-baseline 3.65 instruction density classifier (DCLM fastText) FineWeb-Edu 1.27 educational quality classifier EAI-TAXONOMY STEM 1.74 EAI-TAXONOMY filter EAI-TAXONOMY STEM w/ DCLM 0.91 EAI-TAXONOMY filter, DCLM fastText

Table 8: Overview of high-perfoming general web datasets and taxonomy-based stem dataset.

##### 4.5.1 Taxonomy-Based STEM Dataset

We prepare the EAI-TAXONOMY STEM (1742B tokens) targeting science, engineering, medical, and computer science documents. We select high quality document types per sub-topic and filter for documents that exhibit reasoning (Algorithm 8). We then apply the DCLM classifier, at the same threshold as DCLM-baseline, to the EAI-TAXONOMY STEM dataset to construct EAI-TAXONOMY w/ DCLM (912B tokens, Algorithm 9).

- 4.5.2 Downstream STEM Results

Model MMLU–STEM

DCLM-baseline 27.7%±0.8 FineWeb-Edu 26.7%±0.8

Taxonomy STEM 29.1%±0.8 Taxonomy STEM w/ DCLM 34.5%±0.8

Table 9: Accuracy (% ± standard error) on the MMLU–STEM subset.

EAI-TAXONOMY STEM is able to outperform DCLM-baseline and FineWeb-Edu beyond standard error on MMLU-STEM. EAI-TAXONOMY w/ DCLM outperforms DCLM-baseline by +6.8pp (24.5%) and FineWeb-Edu by +7.8pp (29.2%).

## 5 Developing Taxonomy

- 5.1 Measuring Orthogonality, Correctness, & Expressivity In this section, we introduce methods to measure: (1) orthogonality, (2) correctness, and (3) expressivity.

##### 5.1.1 Orthogonality: Category Independence (NMI)

For two categories Ci,Cj, let the random variables X,Y denote their empirical primary label codes. Redundancy is measured by the normalized mutual information

2I(X;Y ) H(X) + H(Y )

NMI(X,Y ) =

, I(X;Y ) =

pxy pxpy

.

pxy log

x,y

where H is Shannon entropy. NMI = 0 indicates statistical independence, NMI = 1 perfect duplication. This metric was proposed by Wettig et al. [2025] to evaluate taxonomies.

##### 5.1.2 Correctness: Annotator Agreement (annotator κ)

To gauge label clarity we compare a candidate model M with two gold annotators (GPT-4o [OpenAI, 2024] and Claude Sonnet-3.5 [Anthropic, 2024]) via a variant of Cohen’s κ [Cohen, 1960].

Annotation format. Each annotator outputs an ordered set

S ∈ {∅,{ℓ},{ℓ1,ℓ2}}.

Under the taxonomy (Definition 1) a primary label is mandatory; the empty set ∅ arises only when a model’s raw output is malformed and cannot be parsed. Such issues are negligible for high-capacity LLMs but more frequent for small ones. Two annotations agree iff Sa ∩ Sb ̸= ∅ or Sa ∪ Sb = ∅.

Observed and expected agreement. Let Po be the empirical agreement rate over held-out documents. The expected agreement Pe is computed by assuming label sets are constructed independently from each annotator’s empirical fertility distribution (probability of emitting 0/1/2 labels, where the 0-case captures parse failures) and their label-choice distribution (Appendix A.9.4). Then κ is computed as usual:

Po − Pe 1 − Pe

, −1 ≤ κ ≤ 1.

κ =

We report the mean of the two gold-vs-M scores per category (annotator κ); a high κ implies unambiguous label definitions.

##### 5.1.3 Expressivity: Domain–Recall

Some rare domains (math, code) drive downstream performance. To measure how well a classifier surfaces such niche material we introduce domain–recall score inspired by recent works that iteratively train a classifier to maximize recall of a specific domain [Shao et al., 2024]:

- 1. Select a small set U = {u1,...,um} of human-vetted base URLs that are human-judged ≥ 90% in-domain. All documents whose URL begins with any u ∈ U form the domain positives D+ ⊂ D.
- 2. Apply a classifier (taxonomy filter, fastText classifier, etc.) that returns a subset of documents D ⊆ D.

The recall of the classifier on this topic is

Recall = | D ∩ D+ | |D+|

.

A higher recall value means the strategy retrieves more of the trusted in-domain pages. We report recall alongside the data kept fraction | D|/|D| to indicate the overall data volume returned by the classifier.

##### 5.1.4 Held-out Evaluation Sets

To evaluate the categories in EAI-TAXONOMY (Section 3.4), we report the normalized mutual information and annotator κ on the following evaluation sets. Both evaluations are sampled from the deduplicated and heuristicallyfiltered Common Crawl used to prepare ESSENTIAL-WEB V1.0 (Appendix A.2). Neither set was seen during fine-tuning.

- 1. Random Set: 2,017 randomly sampled documents.
- 2. STEM Set: 871 STEM documents 20

In addition, we report domain-recall for web code and math domains. We label a set of 104.6M documents with EAI-TAXONOMY using Qwen2.5-32b-Instruct [Qwen, 2025]. Details about the annotation can be found in Appendix A.13. Domain-recall is then calculated on documents from two sets of human-vetted "gold" URL sets (Appendix A.10.1):

- 1. Web Code: 30 base-URLs; |D+| = 330,934 documents
- 2. Math: 42 base-URLs; |D+| = 16,199 documents

20See Appendix A.10.2 for details about the STEM evaluation set.

##### 5.2 Teacher Model Selection

When selecting a teacher model, we evaluate Qwen2.5-32b-Instruct, Qwen2.5-72b-Instruct, and DeepSeek-V3 using NMI (orthogonality), annotator κ (correctness), and domain-recall (expressivity). We seek to maximize performance and inference efficiency of the teacher model.

##### 5.2.1 Annotator κ Results

We report annotator κ between powerful open source LLMs (DeepSeek-V3 [DeepSeek-AI, 2025], Qwen2.5-72b-Instruct, and Qwen2.5-32b-Instruct) and our two gold annotators (GPT-4o, Claude Sonnet 3.5) for all 12 categories of EAI-TAXONOMY on the random and STEM evaluation sets in Table 10. A discussion justifying the use of LLMs as gold annotators instead of humans can be found in Appendix A.9.3.

DeepSeek-V3 Qwen 2.5-72B-Inst. Qwen 2.5-32B-Inst. Random STEM Random STEM Random STEM

Category

Bloom Knowledge Domain 0.69±0.02 0.64±0.03 0.46±0.02 0.39±0.03 0.62±0.02 0.68±0.02 Bloom Cognitive Process 0.76±0.01 0.76±0.02 0.67±0.02 0.70±0.02 0.73±0.01 0.79±0.02 Document Type V1 0.90±0.01 0.91±0.01 0.88±0.01 0.91±0.01 0.86±0.01 0.89±0.01

- Free Decimal Corr. (level 1) 0.92±0.01 0.95±0.01 0.90±0.01 0.92±0.01 0.88±0.01 0.92±0.01
- Free Decimal Corr. (level 2) 0.86±0.01 0.87±0.01 0.83±0.01 0.81±0.01 0.81±0.01 0.81±0.01
- Free Decimal Corr. (level 3) 0.71±0.01 0.70±0.01 0.67±0.01 0.63±0.01 0.64±0.01 0.60±0.01 Extraction Artifacts 0.81±0.02 0.86±0.03 0.57±0.02 0.53±0.03 0.74±0.01 0.65±0.03 Missing Content 0.83±0.01 0.85±0.02 0.63±0.01 0.65±0.02 0.66±0.02 0.65±0.02 Document Type V2 0.89±0.01 0.89±0.01 0.80±0.01 0.80±0.01 0.85±0.01 0.83±0.01 Education Level 0.89±0.01 0.86±0.02 0.82±0.01 0.81±0.02 0.88±0.01 0.85±0.02 Reasoning Depth 0.75±0.01 0.72±0.02 0.70±0.01 0.67±0.02 0.67±0.02 0.67±0.02 Technical Correctness 0.60±0.02 0.61±0.02 0.52±0.01 0.60±0.02 0.52±0.01 0.51±0.02 Overall mean 0.80 0.80 0.70 0.70 0.74 0.74

- Table 10: Annotator κ (± standard error) between each candidate model and two gold annotators (GPT-4o, Claude Sonnet-3.5) on the random (n=2,017) and STEM (n=871) evaluation sets.

Observations. DeepSeek-V3 achieves the highest average pairwise Cohen’s κ for both random and STEM of 0.80. However, the 671B-parameter Mixture-of-Experts is expensive to serve for large-scale inference.21 Qwen2.5-32b-Instruct provides faster inference and is within 0.06 (7.5%) of DeepSeek-V3’s performance, achieving 0.74 for the random set and STEM set, outperforming its larger sibling Qwen2.5-72b-Instruct. We also measure the performance of Qwen2.5-14B-Instruct and find that random and STEM annotator κ drop to 0.53 and 0.52 (see Appendix A.11.1 for full table). We select Qwen2.5-32b-Instruct to label the training set of EAI-Distill-0.5b given its balance of fast inference speed and high annotator κ (see Table 12 for an analysis of inference performance).

##### 5.2.2 Inter-Category NMI Results

In Figure 4, we report the inter-category NMI values as a heatmap. The NMI heatmaps for DeepSeek-V3 and Qwen2.5-72b-Instruct are in the Appendix A.11.2 and average inter-category NMIs can be found in Table 14.

Observations. The average inter-category NMI for Qwen2.5-32b-Instruct is 0.079 and 0.083 for random and STEM.22 This value indicates very weak dependence between different categories of EAI-TAXONOMY. In addition, this value is in line with the reported inter-category NMI of 0.10 in the 2-category taxonomy defined by Wettig et al. [2025]. The average NMI for Qwen2.5-32b-Instruct, Qwen2.5-72b-Instruct, and DeepSeek-V3 are all under 0.10 for both random and STEM. Qwen2.5-32b-Instruct has the lowest average NMI out of the three tested open-source models (Table 14).

21At the time of annotation, the SGLang image for DeepSeek-V3 on AMD MI300x was much slower. See Table 12 and

- Appendix A.13.1 [Zheng et al., 2024]. 22When calculating the average inter-category NMI we omit Document Type V2, given its similar purpose to Document

Type V1, and Free Decimal Correspondence Level 1 and Level 2 to avoid double-counting FDC hierarchical splits.

[Figure 1]

[Figure 2]

(a) random NMI (b) STEM NMI

Figure 4: Comparison of NMI heatmaps on random and STEM eval sets for Qwen2.5-32B-Instruct

##### 5.2.3 Domain–Recall Results

- Table 11 shows how effectively different filtering strategies retrieve documents from two human-vetted "gold" URL sets: web code (30 base-URLs; 330,934 documents) and math (42 base-URLs; 16,199 documents) drawn from a 104M-document Common Crawl sample. To provide a baseline we train two fastText classifiers using high quality math and code documents as target sets, with random samples of Common Crawl as negative sets [Bojanowski et al., 2017]. The fastText classifiers should be considered simple baselines given we do not apply techniques such as iterative recall bootstrapping or BPE tokenization to improve recall performance [Shao et al.,

- 2024, Huang et al., 2025]. Details on the training of the fastText classifiers can be found in Appendix A.10.3.

##### Domain Filter Classifier Recall (%) Data kept (%)

FDC ∈ 004,005∗ & Doc Type V1† EAI-TAXONOMY filter 94.9 4.80 FDC ∈ 004,005∗ EAI-TAXONOMY filter 96.5 7.50 score > 0.01 fastText web code 25.2 6.40

Web Code

FDC ∈ 51 Mathematics EAI-TAXONOMY filter 98.0 0.50 score > 0.5 fastText math 74.6 3.80 score > 0.1 fastText math 98.5 47.9

Math

- Table 11: Domain–recall for web code and math of Qwen2.5-32b-Instruct-annotated EAI-TAXONOMY in comparison to basic fastText classifiers. ∗FDC codes 004 and 005 denote Computer Science and Software Engineering. †Document Type V1 filters for "Code/Software", "Reference/Encyclopedic/Educational", and "Social/Forum".

Code. The Free Decimal Correspondence (FDC) alone recalls 96.5% of vetted code pages while discarding 92.5% of the web. Adding Document Type V1 constraint trims 35.9% of volume of data kept with only 1.6pp loss in recall of code pages. The baseline fastText model performs significantly worse recalling only 25.2% of target web code documents.

Math. Tagging with the single FDC level 2 code 51 - Mathematics surfaces 98.0% of vetted math pages while keeping just 0.5% of the corpus—an efficient high-recall, low-volume filter. Lowering thresholds of the fastText can match recall (98.5%) but at the cost of retaining nearly half the crawl resulting in a much lower density of math documents in the returned set of documents.

The hierarchical Free Decimal Correspondence, combined with the Document Type V1 category tags, yields very high recall for specialized domains at modest data volumes, outperforming simple, topic-specific fastText classifiers with no need for domain-specific training.

##### 5.2.4 Teacher Model Selection Summary

- 1. Model choice. Given its annotator κ–speed trade-off, we adopt Qwen-2.5-32B-Instruct as the teacher for EAI-Taxonomy-0.5B.
- 2. Low redundancy. Inter-category NMI of <0.10 on both random and STEM evaluation sets indicates that EAI-TAXONOMY captures largely orthogonal signals.
- 3. High label clarity. Qwen2.5-32b-Instruct reaches an annotator κ of 0.74 vs. two powerful gold LLM annotators.
- 4. Efficient domain targeting. Simple EAI-TAXONOMY filters recall >96% of vetted math and code pages while retaining <5% of Common Crawl, far denser than fastText baselines.

## 6 Running at Scale

Using a sample of the 104.6M Common Crawl documents labeled with Qwen2.5-32b-Instruct (Appendix A.13), we fine-tune Qwen2.5-0.5b-Instruct to perform the taxonomy classification task. The fine-tuned classifier, EAI-Distill-0.5b, achieves 50 times faster inference speed relative to prompting Qwen2.5-32b-Instuct while maintaining performance across NMI, annotator κ, and domain-recall for math and web code.

##### 6.1 Performance Considerations

To annotate at a billion-document scale, we optimize three levers to maximize throughput: (1) reducing model size, (2) shorter generations, and (3) context distillation. Table 12 summarizes speedups.

##### Model Prompt Tokens Generation Tokens RPS/GPU Speed ∆

DeepSeek-V3 (old image) Original Original 0.14 0.10 DeepSeek-V3 Original Original 0.54 0.39 Qwen2.5-72B Original Original 0.62 0.44

Original Original 1.40 1.00 None Condensed 6.30 4.50

Qwen2.5-32B

Original Original 5.07 3.62

None Original 5.46 3.90 Original Condensed 50.91 36.36 None Condensed 70.28 50.20 None Embedding 189.23 135.16

Qwen2.5-0.5B

- Table 12: Inference performance comparison. Requests per second per GPU are measured with prefix caching enabled. Speed ∆ is calculated relative to Qwen2.5-32B with the original prompt which is provided in

- Appendix A.14.3. The italicized row indicates the configuration used by EAI-Distill-0.5b. When we were annotating the sample of 104M document, the SGLang image for DeepSeek-V3 was much slower as shown in the table. Embedding signifies using a classification-head instead of token generation.

Measuring performance. To compare performance of different model sizes and prefill/generation strategies, we report requests per second per GPU (RPS/GPU). Given the large batch workload, we care exclusively about maximizing throughput (total number of documents processed per second). To calculate the effects of design decisions on performance, we calculate the number of tokens in the shared prefix, average document, and average generation.23 We use vLLM for inference of the Qwen2.5 models and SGLang for DeepSeek-V3 [Zheng et al., 2024, Kwon et al., 2023].

23These calculation were done solely using the prompt from Appendix A.14.3 given we annotated the 104.6M Common Crawl documents in two passes to add additional categories before fine-tuning (Appendix A.13). Therefore, these are lower bounds on the true gains given the Original prompt and Original generation only encapsulate 8 categories, whereas the Condensed generation contains all 12 categories.

Condensing model size. We select the smallest model in the Qwen2.5 model family. We decide to use a pre-trained small LM to take advantage of strong priors developed during training.24. When maintaining the original prompt and generation format with prefix caching enabled, we only see a 3.6× increase in RPS/GPU.

Condensing generation tokens. We are able to extract and condense the output generated by Qwen2.5-32b-Instruct programatically before fine-tuning. By reducing the average generation tokens from 791 to 51, the RPS/GPU of Qwen2.5-0.5b increase by 10×.

Context distillation. We remove the prompt during finetuning, which increases the RPS/GPU of Qwen2.5-0.5b by 1.4× [Snell et al., 2022]. This speed increase is in spite of the fact that we have prefix caching enabled during inference.

Why not use a classification head? As a first iteration, we opted to condense the number of generated tokens instead of relying on a classification-head. However, there are many potential benefits to using a classification head such as persisting a document-embedding that can be re-used and faster inference (2.7× increase in RPS/GPU over the configuration used for EAI-Distill-0.5b). We hope to look into this in the future.

##### 6.2 Distillation

To train the Qwen2.5-0.5b-Instruct model, we fine-tune the model on 82B tokens of documents synthetically labeled by Qwen2.5-32b-Instruct. We perform context distillation and condense the number of generated tokens. See Appendix A.14.2 for the output format of EAI-Distill-0.5b. The loss is computed only on the Qwen2.5-32b-Instruct’s completion tokens; the input document, chat template, and system prompt are masked out during loss calculation. The hyper-parameters used for fine-tuning the model can be found in Table 33,

- Appendix A.12.2. An ablation of prompt/generation formatting, learning rate, and token budget can be found in
- Appendix A.12.3. The ablations find that there is little-to-no degradation from context distillation and generation condensation. We also find that we achieve similar annotator κ across categories with as little as 12B training tokens. EAI-Distill-0.5b is available on HuggingFace: EssentialAI/eai-distill-0.5b.

###### 6.3 EAI-Distill-0.5b Performance

We evaluate the student model, EAI-Distill-0.5b, against its teacher, Qwen2.5-32b-Instruct, along three metrics: NMI (orthogonality), annotator κ (correctness), and domain recall (expressivity).

##### 6.3.1 Annotation Consistency (κ)

Observations. EAI-Distill-0.5b’s annotator κ decreases by 0.03 (4.1%) and 0.01 (1.4%) relative to Qwen2.5-32b-Instruct (Table 13). We observe consistent performance across the two models in both Bloom categories, Document Type V1/V2, and the Free Decimal Correspondence. EAI-Distill-0.5b exhibits stronger performance than its teacher model on Reasoning Depth and Technical Correctness and weaker performance on Extraction Artifacts, Missing Content, and Education Level. We provide in depth analysis of these shifts in Appendix A.9.2.

- 6.3.2 Inter-Category NMI Results The average inter-category NMI of EAI-Distill-0.5b is 0.092 and 0.093 for random and STEM, increasing by

- 0.013 (16.5%) and 0.01 (12.0%) relative to Qwen2.5-32b-Instruct. However, EAI-Distill-0.5b remains comparable to other powerful open-source LLMs and stays below 0.10, indicating low redundancy. See Table 14 for the average inter-category NMI of all models tested and Figure 5 for heatmaps with the relative change in inter-category NMI from Qwen2.5-32b-Instruct to EAI-Distill-0.5b.

- 6.3.3 Domain–Recall on Unseen 100M Document Slice

When calculating domain-recall for EAI-Distill-0.5b, we sample a fresh set 102.6M unseen documents from Common Crawl because a subset of the data used to calculate domain-recall in Section 5.2.3 was used to train EAI-Distill-0.5b. The size of D+ for math and web code for the new set of 102.6M documents are are:

24Recently, even smaller pre-trained LMs have been released such as SmolLM2-135M-Instruct, which could potentially further improve inference performance [Allal et al., 2025]

Qwen2.5-32B-Inst. EAI-Distill-0.5b Random STEM Random STEM

Category

Bloom Knowledge Domain 0.62±0.02 0.68±0.02 0.63±0.02 0.65±0.03 Bloom Cognitive Process 0.73±0.01 0.79±0.02 0.70±0.02 0.76±0.02 Document Type V1 0.86±0.01 0.89±0.01 0.83±0.01 0.86±0.01

- Free Decimal Corr. (level 1) 0.88±0.01 0.92±0.01 0.88±0.01 0.93±0.01
- Free Decimal Corr. (level 2) 0.81±0.01 0.81±0.01 0.81±0.01 0.80±0.01
- Free Decimal Corr. (level 3) 0.64±0.01 0.60±0.01 0.63±0.01 0.61±0.01 Extraction Artifacts 0.74±0.01 0.65±0.03 0.27±0.02 0.37±0.03 Missing Content 0.66±0.02 0.65±0.02 0.48±0.01 0.57±0.02 Document Type V2 0.85±0.01 0.83±0.01 0.88±0.01 0.86±0.01 Education Level 0.88±0.01 0.85±0.02 0.79±0.02 0.86±0.02 Reasoning Depth 0.67±0.02 0.67±0.02 0.87±0.01 0.76±0.02 Technical Correctness 0.52±0.01 0.51±0.02 0.72±0.01 0.75±0.02 Overall mean 0.74 0.74 0.71 0.73

- Table 13: Annotator κ (± standard error) against gold annotators on the random (n = 2,017) and STEM (n = 871) evaluation sets, after fine-tuning. We did not report Qwen2.5-0.5b-Instruct because it achieves an annotator κ of 0.00 for both random and STEM evaluation sets.

Model Average NMI Random STEM

Claude Sonnet-3.5 0.083 0.085 GPT-4o 0.063 0.084 DeepSeek-V3 0.093 0.099 Qwen2.5-72b-Instruct 0.092 0.103 Qwen2.5-32b-Instruct 0.079 0.083 EAI-Distill-0.5b 0.092 0.093

- Table 14: Average inter-category NMI results for random and STEM evaluation sets. We omit Document Type V1, given its similar purpose to Document Type V2, and Free Decimal Correspondence Level 1 and Level 2 to avoid double-counting FDC hierarchical splits.

[Figure 3]

[Figure 4]

(a) Random (b) STEM

Figure 5: Change in inter-category NMI per category from Qwen2.5-32b-Instruct to EAI-Distill-0.5b. A higher value indicates an increase in inter-category co-occurrence.

- • Web Code: |D+| = 266,051 documents
- • Math: |D+| = 12,835 documents

##### Domain Filter Model Recall (%) Data kept (%)

FDC ∈ 004,005 & Doc Type V1 EAI-Distill-0.5b 95.8 4.91 FDC ∈ 004,005 & Doc Type V1 Qwen2.5-32b-Instruct 94.9 4.79 FDC ∈ 004,005 EAI-Distill-0.5b 97.1 7.61 FDC ∈ 004,005 Qwen2.5-32b-Instruct 96.5 7.47

Web Code

FDC ∈ 51 Mathematics EAI-Distill-0.5b 97.6 0.52 FDC ∈ 51 Mathematics Qwen2.5-32b-Instruct 98.0 0.50

Math

- Table 15: Domain–recall for web code and math domains. FDC codes 004 and 005 correspond to Computer Science and Software Engineering. The Document Type V1 filters for "Code/Software", "Reference/Encyclopedic/Educational", and "Social/Forum". Qwen2.5-32b-Instruct measured on the original 104.6M-document corpus; numbers shown for reference.

Observations. EAI-Distill-0.5b model matches Qwen2.5-32b-Instruct within ±1pp recall while preserving the same thin data footprint (<1% of crawl for math, ≈5% for code with tight filters) as seen in Table 15.25

##### 6.3.4 EAI-Distill-0.5b Performance Summary

- 1. 50× faster annotation. Shrinking to a 0.5B-parameter model, removing the prompt, and condensing generations boost throughput from 1.4 to 70 requests per second per GPU.
- 2. Minimal quality loss. Average annotator κ across random and STEM sets of EAI-Distill-0.5b falls by <3% (0.74 → 0.72) and inter-category NMI stays <0.10, preserving label quality and low redundancy.
- 3. Domain coverage intact. On an unseen 102M-document sample of Common Crawl, math and code recall remain within 1pp of Qwen2.5-32b-Instruct while keeping the same compact filtered sets (<5% of data).

## 7 Conclusion

ESSENTIAL-WEB V1.0 delivers 24 trillion tokens of web data with document-level annotations spanning subject, page type, content complexity, and quality. Applying simple filters on these labels produce competitive corpora for mathematics, code, medicine, and STEM in minutes, not months — competing with or surpassing specialist datasets while avoiding domain-specific training.

Building such a corpus required three technical ingredients: (1) A principled metric suite of NMI for redundancy, annotator κ for label clarity, and domain-recall for domain-specific filtering to guide taxonomy design. (2) A fast annotator, EAI-Distill-0.5b, that retains the teacher’s quality (<3% drop in annotator κ, <1pp drop in recall of math and code, average inter-category NMI below 0.10) yet runs 50× faster, making billion-page labeling practical. (3) A transparent release with all data and EAI-Distill-0.5b available on HuggingFace, enabling reproducibility and iteration.

##### 7.1 Broader impact

Structured web data transforms corpus curation from an complex, expensive processing pipeline into a search problem that anyone can solve. We hope ESSENTIAL-WEB V1.0 becomes a community commons: a foundation others can refine, audit, or curate in new ways, accelerating open research on LLM training data, arguably the most valuable, yet least shared, asset contributing to modern LLM capabilities.

##### 7.2 Possible Next Steps

Taxonomies serve as an excellent bridge to build an interpretable coordinate system on data, despite their biases. In addition to using our ESSENTIAL-WEB V1.0 for organic and synthetic data curation, we see new opportunities to

25Because Qwen2.5-32b-Instruct was evaluated on a different sample of Common Crawl, small drift is expected; nevertheless, parity holds. We did not re-evaluate Qwen2.5-32b-Instruct on the new sample given the cost to relabel 102.6M documents.

automatically curate data to train large neural networks at scale. Whether taxonomies will remain a core element of state-of-the-art data pipelines or merely serve as a stepping-stone en route to fully unsupervised methods is still an open question.

## References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíˇcek, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025. URL https://arxiv.org/abs/2502.02737.

L.W. Anderson and D.R. Krathwohl. A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom’s Taxonomy of Educational Objectives. Longman, 2001.

Anthropic. Claude 3.5 Sonnet Model Card Addendum. Model card, Anthropic, June 2024. URL https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_ Claude_3_Addendum.pdf. Accessed 23 May 2025.

Anna Arias-Duart, Pablo Agustin Martin-Torres, Daniel Hinjos, Pablo Bernabeu-Perez, Lucia Urcelay Ganzabal, Marta Gonzalez Mallo, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Sergio Alvarez-Napagao, and Dario Garcia-Gasulla. Automatic evaluation of healthcare llms beyond question-answering, 2025. URL https://arxiv.org/abs/2502.06666.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. Proceedings of the National Academy of Sciences, 121(27), June 2024. ISSN 1091-6490. doi: 10.1073/pnas.

2311878121. URL http://dx.doi.org/10.1073/pnas.2311878121.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023. URL https://arxiv.org/abs/2309.16609.

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long short-term memory,

2024. URL https://arxiv.org/abs/2405.04517.

Janek Bevendorff, Benno Stein, Matthias Hagen, and Martin Potthast. Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl. In Leif Azzopardi, Allan Hanbury, Gabriella Pasi, and Benjamin Piwowarski, editors, Advances in Information Retrieval. 40th European Conference on IR Research (ECIR 2018), Lecture Notes in Computer Science, Berlin Heidelberg New York, March 2018. Springer.

Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information, 2017. URL https://arxiv.org/abs/1607.04606.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.

Jacob Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20:37

###### – 46, 1960. URL https://api.semanticscholar.org/CorpusID:15926286. Yann Collet. xxhash, 2025. URL https://xxhash.com. Common Crawl Foundation. Common Crawl Corpus. https://commoncrawl.org, 2025. all shards uo to

CC-MAIN-2024-38. DeepSeek-AI. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence, 2024a. URL https://arxiv.org/abs/2406.11931. DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism, 2024b. URL https:

###### //arxiv.org/abs/2401.02954.

DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024c. URL

https://arxiv.org/abs/2405.04434. DeepSeek-AI. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437. Alvan R Feinstein and Domenic V Cicchetti. High agreement but low kappa: I. the problems of two paradoxes.

Journal of clinical epidemiology, 43(6):543–549, 1990.

Luis Felipe, Carlos Garcia, Issam El Naqa, Monique Shotande, Aakash Tripathi, Vivek Rudrapatna, Ghulam Rasool, Danielle Bitterman, and Gilmer Valdes. Thebluescrubs-v1, a comprehensive curated medical dataset derived from the internet, 2025. URL https://arxiv.org/abs/2504.02874.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

- Gemma 2 Team. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv. org/abs/2408.00118.
- Gemma 3 Team. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https://arxiv.org/abs/2009.03300.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of computeoptimal large language model training. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 30016–30030. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf.

Siming Huang, Tianhao Cheng, J. K. Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, Jiaheng Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, and Wei Chu. Opencoder: The open cookbook for top-tier code large language models, 2025. URL https://arxiv.org/abs/2411.04905.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-coder technical report, 2024. URL https://arxiv.org/abs/2409.12186.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams, 2020. URL https://arxiv.org/abs/2009.13081.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL https://aclanthology.org/D19-1259/.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https: //arxiv.org/abs/2001.08361.

AS Kolesnyk and NF Khairova. Justification for the use of cohen’s kappa statistic in experimental studies of nlp and text mining. Cybernetics and Systems Analysis, 58(2):280–288, 2022.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention,

###### 2023. URL https://arxiv.org/abs/2309.06180.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2025. URL https://arxiv.org/abs/2406.11794.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation, 2023. URL https://arxiv.org/ abs/2305.01210.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025. URL https://arxiv.org/abs/2502.16982.

Llama Team. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder 2 and the stack v2: The next generation, 2024. URL https://arxiv.org/abs/2402.19173.

Meta AI. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, April 2025. URL

###### https://ai.meta.com/blog/llama-4-multimodal-intelligence/. Accessed: 2025-06-15.

John Mark Ockerbloom. Free decimal correspondence. https://everybodyslibraries.com/ free-decimal-correspondence/, August 2010. Released 19 August 2010; dedicated to the public domain (CC0).

###### OpenAI. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Gerardo Flores, George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann, editors, Proceedings of the Conference on Health, Inference, and Learning, volume 174 of Proceedings of Machine Learning Research, pages 248–260. PMLR, 07–08 Apr 2022. URL https://proceedings.mlr.press/v174/pal22a.html.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023. URL https://arxiv.org/abs/2310.06786.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only, 2023. URL https://arxiv.org/abs/2306.01116.

Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale,

###### 2024. URL https://arxiv.org/abs/2406.17557. Qwen. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher, 2022. URL https://arxiv.org/abs/2112.11446.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. URL https://arxiv.org/abs/1910.10683.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Xuyang Shen, Dong Li, Ruitao Leng, Zhen Qin, Weigao Sun, and Yiran Zhong. Scaling laws for linear complexity language models, 2024a. URL https://arxiv.org/abs/2406.16690.

Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing. Slimpajama-dc: Understanding data combinations for llm training, 2024b. URL https://arxiv.org/abs/2309.10818.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Nathaneal Scharli, Aakanksha Chowdhery, Philip Mansfield, Blaise Aguera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge, 2022. URL https://arxiv.org/abs/2212.13138.

Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context, 2022. URL https://arxiv.org/

###### abs/2209.15189.

Software Heritage Foundation. Software Heritage Archive. https://archive.softwareheritage.org, 2025. Snapshot accessed 10 June 2025.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research, 2024. URL https://arxiv.org/abs/2402.00159.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: beating power law scaling via data pruning, 2023. URL https://arxiv.org/abs/2206.14486.

Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset, 2025. URL https://arxiv.org/abs/2412.02595.

Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. URL https:

###### //huggingface.co/datasets/teknium/OpenHermes-2.5.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023a. URL https: //arxiv.org/abs/2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b. URL https://arxiv.org/abs/2307.09288.

Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang. Redpajama: an open dataset for training large language models, 2024a. URL https://arxiv.org/abs/2411.12372.

Maurice Weber, Daniel Y. Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang. Redpajama: an open dataset for training large language models. NeurIPS Datasets and Benchmarks Track, 2024b.

Alexander Wettig, Kyle Lo, Sewon Min, Hannaneh Hajishirzi, Danqi Chen, and Luca Soldaini. Organize the web: Constructing domains enhances pre-training data curation, 2025. URL https://arxiv.org/abs/2502. 10341.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li,

Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, and Xian Li. Naturalreasoning: Reasoning in the wild with 2.8m challenging questions,

###### 2025. URL https://arxiv.org/abs/2502.13124.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs, 2024. URL https://arxiv.org/abs/2312.07104.

Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P. Xing. Megamath: Pushing the limits of open math corpora, 2025. URL https://arxiv.org/abs/2504. 02807.

## A Appendix

##### A.1 Contributions and Acknowledgments Core Contributors

Andrew Hojel∗ Michael Pust∗ Tim Romanski Yash Vanjani Ritvik Kapila Mohit Parmar Ashish Vaswani

Contributors

Adarsh Chaluvaraju Alok Tripathy Anil Thomas Ashish Tanwer Darsh J Shah Ishaan Shah Karl Stratos Khoi Nguyen Kurt Smith Michael Callahan Peter Rushton Philip Monk Platon Mazarakis Saad Jamal Saurabh Srivastava Somanshu Singla

##### A.2 ESSENTIAL-WEB V1.0: Dataset Overview

We begin with DCLM Pool (89 resiliparse-extracted Common Crawl WARC snapshots from CC-MAIN-2013-20 to CC-MAIN-2022-49 [Common Crawl Foundation, 2025, Li et al., 2025]. We then extract 12 additional snapshots from CC-MAIN-2023-06 to CC-MAIN-2024-38 using resiliparse [Bevendorff et al., 2018].

Post-extraction, we process the data using the following steps:

- 1. Generate document ids using xxhash.xxh3_64_intdigest(document) [Collet, 2025]
- 2. Globally deduplicate all 101 snapshots of Common Crawl using the hash-based document identifier.
- 3. Minhash LSH de-duplication at a snapshot-level. We run Minhash LSH with a target Jaccard threshold of 0.7, using 14 bands and 9 rows per band. Running at a snapshot-level was inspired by Penedo et al. [2024] and DeepSeek-AI [2024b].
- 4. Annotate every document with statistical and model-based quality signals using a variant of the RedPajama-Data-V2 processing pipeline [Weber et al., 2024b]. The model-based signals include the DCLM-baseline fastText classifier [Li et al., 2025].
- 5. Filter out low quality documents and only keep English, while maximally allowing math and code, using manually tuned quality signal filters (see Algorithm 1). The filters and starting values for tuning were inspired by Penedo et al. [2024], Weber et al. [2024b], and Rae et al. [2022].26
- 6. Label every document with EAI-TAXONOMY using EAI-Distill-0.5b (Section 6).

- See Table 16 for the stage-by-stage removal rates.

*Equal contribution.

26In the future, we hope to remove this step or automatically tune the quality signal thresholds to minimize the human-inthe-loop.

Algorithm 1: Quality Signal Filter Rules

|# Rule 1: Initial Quality Filters (applied to ALL documents)<br><br>RULE_1_CONDITIONS = [ word_count < 50,<br><br>frac_chars_top_2gram > 0.20,<br><br>frac_chars_top_3gram > 0.18, frac_chars_dupe_10grams > 0.50, frac_chars_dupe_9grams > 0.52, frac_chars_dupe_8grams > 0.54, frac_chars_dupe_7grams > 0.56, frac_chars_dupe_6grams > 0.58, frac_chars_dupe_5grams > 0.60<br><br><br>] # Rule 2: Bypass Conditions (if ANY is true, skip Rule 3)<br><br>RULE_2_CONDITIONS = [ ml_math_score > 0.3, ml_web_code_score > 0.3<br><br>] # Rule 3: Additional Filters (only if Rule 2 bypass fails)<br><br>RULE_3_CONDITIONS = [ frac_unique_words > 0.95, frac_no_alph_words > 0.6, ldnoobw_words > 10, ml_english_score < 0.6<br><br><br><br><br>] # Quality Signals Filter Pipeline for each document d in corpus D:<br><br># Step 1: Apply initial quality filters<br><br>if ANY condition in RULE_1_CONDITIONS is true: REJECT document d<br><br># Step 2: Check for bypass conditions<br><br>if ANY condition in RULE_2_CONDITIONS is true: ACCEPT document d # Skip Rule 3<br><br># Step 3: Apply additional filters<br><br>if ANY condition in RULE_3_CONDITIONS is true: REJECT document d<br><br><br><br><br>else: ACCEPT document d<br><br>|
|---|

##### Processing step % removed Cumulative % removed Docs (B)

101 raw CC snapshots – – 248.4 Snapshot exact-dedup 32.1% 32.1% 168.7 Global exact-dedup 45.6% 63.1% 91.7 Snapshot LSH dedup 23.1% 71.6% 70.5 Language filter 50.8% 86.0% 34.8 Document quality filters 32.1% 90.5% 23.6

- Table 16: Common-Crawl processing pipeline. "% removed” is the drop at that stage; "Cumulative % removed” is with respect to the original 248.4 B documents. Some steps are run at the same time or in parallel, so we estimate the step-level removal rate on a subset of documents.

- See Table 17 for the datasets used and released in this work.

##### Dataset Type Hugging Face Repository

ESSENTIAL-WEB V1.0 EssentialAI/essential-web-v1.0 ESSENTIAL-WEB V1.0 1T FDC L2 EssentialAI/essential-web-1t-sample-fdc-partitioned EAI-TAXONOMY w/ FM EssentialAI/eai-taxonomy-math-w-fm EAI-TAXONOMY w/ DCLM EssentialAI/eai-taxonomy-code-w-dclm EAI-TAXONOMY Med w/ DCLM EssentialAI/eai-taxonomy-med-w-dclm EAI-TAXONOMY STEM w/ DCLM EssentialAI/eai-taxonomy-stem-w-dclm

Table 17: Datasets we release in this work.

##### A.3 Descriptions of Taxonomy Categories

In Section 3.4 we briefly introduce the categories in our taxonomy. What follows is an in-depth description of those categories and their labels.

##### A.3.1 FDC

The Free Decimal Correspondence [Ockerbloom, 2010] is a set of decimal numbers, each corresponding to a group of subjects or disciplines. It is intended to be compatible with the Dewey Decimal System 27, popularly used to catalog libraries. We use the FDC to provide three nested categories – Level 1,2,3 – each successive level being a refinement of its parent. Enumerating all codes and their corresponding labels would be impractical, but Table 18 lists the codes and labels for Level 1.

Code Label

- Level 1

- 0 General works, books and libraries, information sciences
- 1 Philosophy and psychology
- 2 Religion
- 3 Social Sciences
- 4 Philology and Laguage and languages
- 5 Science and Natural history
- 6 Industrial arts and Technology and Engineering
- 7 Arts and recreation
- 8 Literature
- 9 History and Geography

- Level 2 00 - 99 Sub-divisions of Level 1 categories

- Level 3 000 - 999 Sub-divisions of Level 2 categories

Table 18: FDC category codes and labels. Level 2 and Level 3 labels omitted for brevity. A full set of label/code correspondences can be found at https://github.com/JohnMarkOckerbloom/fdc/blob/master/fdc.txt. A web viewer can be found at https://www.librarything.com/mds.

##### A.3.2 Bloom

Bloom’s Taxonomy of Educational Objectives has had many updates since its introduction in 1948. We use two categories and their labels from Anderson and Krathwohl’s 2001 revision of the taxonomy Anderson and Krathwohl [2001]. Table 19 describes the categories and labels used, but note that we did not provide our teacher LLM model with the label descriptions, instead relying on its internal knowledge of Bloom’s taxonomy.

27"Dewey," "Dewey Decimal," "Dewey Decimal Classification", and "DDC" are trademarks of OCLC.

Label Description Knowledge Domain Author demonstrates use of...

Factual basic elements to learn or solve problems in the discipline Conceptual interrelationships between basic elements within a larger context Procedural methods in the discipline Metacognitive awareness of how learning works in relation to one’s self

Cognitive Processing Level Author demonstrates ability to...

Remember retrieve relevant knowledge from memory Understand determine the meaning of instructional messages Apply use a procedure in a given situation Analyze break materials into components and determine how they work together Evaluate make judgments based on criteria and standards Create create a new or original work

Table 19: Bloom categories and label descriptions.

##### A.3.3 Document Type

Our taxonomy includes two collections of labels that categorize by common web document types. The two collections have a good degree of overlap. Version 1 was created in-house. Version 2 is a replication of the document type category in WebOrganizer [Wettig et al., 2025]. The labels for both versions are found in Table 20. In addition to the labels described, each category also has a label option Unclassified, for documents that resist classification.

##### A.3.4 Content Quality

The categories in this group are designed to assess the sophistication of material discussed in a document. They are inspired by two other efforts to categorize web data by information quality: NaturalReasoning [Yuan et al., 2025], and FineWeb [Penedo et al., 2024]. Table 21 describes the categories and labels we developed based on their work. In addition to the labels described, each category also has a label option Indeterminate, if a document does not have enough context to make a judgement for that category.

##### A.3.5 Extraction

We provide two sets of labels to flag issues that make a document difficult to read, likely from errors converting structured text formats such as HTML to plain text: Extraction Artifacts and Missing Content. Their labels are found in Table 22. Documents could potentially be flagged with multiple errors, but only the most notable errors are allowed to be flagged for primary / secondary labels. Either category may label a document Indeterminate if the document does not have enough content to pass judgment on it.

Label Examples Label Examples V1 News / Editorial CNN articles, opinion

Academic / Research ArXiv papers, articles

columns

Reference / Encyclopedic / Educational

FAQs, Wikipedia Code / Software Github repos, code

examples Social / Forum conversation threads, Q&A

Promotional / Advertisement

product pages, calls to action

boards

Search / Directory / Bibliography

link pages, search results Adult / Pornographic Justice Potter Stewart

quotes Personal / Misc blogs, user profiles Machine-Generated "lorem ipsum", garbled text Legal / Regulatory contracts, terms of service Government / Political legislation proposals, press

releases Literary / Creative poems, short stories Reviews / Criticism film critiques, product

reviews E-Commerce / Marketplace eBay listings, Amazon

Images / Videos / Audio YouTube video, Imgur page

product pages

###### V2

About (Org.) org. self-description About (Personal) personal profile or intro Academic Writing research paper, abstract Audio Transcript interview or court transcript,

captions Comment Section Reddit, comment sections Content Listing site maps, product catalogs Creative Writing song lyrics, Novel exerpts Documentation API docs, README files FAQ question / answer lists Knowledge Article Wikipedia, Britannica Legal Notices privacy policy, license

Listicle Buzzfeed-style articles News (Org.) government blog posts News Article newspaper digital article,

agreement

CNN article Nonfiction Writing editorials, obituaries,

memoirs Personal Blog individual’s daily journal Product Page product promotion, course

description Q&A Forum Quora, Stack Exchange Spam / Ads spam content, SEO

keyword stuffing Structured Data datasheets, glossaries,

Customer Support troubleshooting guides Truncated pay-walled site, Image

JSON files

Tutorial cooking recipies, WikiHow

page UserReview Yelp or TripAdvisor review

galleries

Table 20: Document Type categories and labels / examples

Label Description Reasoning Depth Document contains...

No Reasoning facts, but no evidence of reasoning about facts Basic Reasoning basic analysis, with minimal explanation and summarization Intermediate Reasoning some logical steps that connect ideas, and structured thinking, but is missing deep examples of

either Advanced Reasoning multi-step reasoning, and thorough analysis with well-developed explanations Exceptional Reasoning novel abstractions, theoretical frameworks, long chain-of-thought, original insights, or proofs

Technical Correctness Document demonstrates...

Technically Flawed significant errors, inaccuracies, undermining validity of contents Partially Correct some technical correctness, but contains flaws, omissions, errors in calculation or terminology.

generally understandable Mostly Correct technical correctness, with minor flaws, omissions, incomplete explanations, or other small issues Highly Correct high technical correctness, precise definitions, accurate and clear explanations, and strong command of technical material with at most minimal errors Exceptionally Correct exceptional technical correctness, formal proofs, flawless technical content, precise calculations,

and mastery of material Education Level Content...

General Audience is accessible to anyone with basic literacy. Uses simple terms most can understand High School Level requires high school level education to fully comprehend. Contains specialized terminology

explained for non-experts. Assumes basic background knowledge Undergraduate Level requires college-level education. Uses specialized terminology/concepts, and assumes significant subject-area background knowledge Graduate/Expert Level requires graduate-level education or domain expertise. Assumes deep background knowledge and specialized training to comprehend

- Table 21: Content Quality categories and label descriptions.

Label Description Extraction Artifacts Document has...

No Artifacts no leftover HTML or irrelevant elements. Text is clean Leftover HTML HTML/code artifacts remaining after extraction Text Extraction Errors broken math expressions, encoding errors, improperly parsed tables Irrelevant Content headers, footers, nav menus, side-bars, or non-core sections extracted by mistake

Missing Content Document exhibits...

No Missing Content no signs of missing content. text seems complete/coherent Truncated Snippets obvious "...", incomplete paragraphs, or cut-off text Click Here References "Download here", "Click here", or references to content not present in text Incoherent Flow signs of unreadable or illogical flow due to missing key context Missing Images or Figures placeholders or references to images/figures not included in text Missing Referenced Data signs that data/tables/datasets are not included. e.g. "See Table 3", but it’s absent

- Table 22: Extraction categories and label descriptions.

##### A.4 Downstream Eval Details

Eval # shots Task Name Config

MMLU 5 mmlu mmlu/default/_mmlu.yaml GSM8K 8 gsm8k gsm8k/gsm8k.yaml MATH 4 hendrycks_math hendrycks_math/hendrycks_math.yaml MBPP+ 3 mbpp_plus mbpp/mbpp_plus.yaml HumanEval+ 0 humaneval_plus humaneval/humaneval_plus.yaml CareQA_en 5 careqa_en careqa/careqa_en.yaml MedMCQA 5 medmcqa medmcqa/medmcqa.yaml MedQA–USMLE 5 medqa_4options medqa/medqa.yaml PubMedQA 3 pubmedqa pubmedqa/pubmedqa.yaml

Table 23: Number of shots and configuration files used for each evaluation. Files refer to commit d09e03d of the EleutherAI/lm-evaluation-harness repository; prepend lm_eval/tasks/ to each path for the full location.

##### A.5 Evaluated Open-Source Datasets

Dataset Type Hugging Face Repository

Math HuggingFaceTB/finemath Math open-web-math/open-web-math Math LLM360/MegaMath General mlfoundations/dclm-baseline-1.0 General HuggingFaceFW/fineweb-edu Code OpenCoder-LLM/opc-fineweb-code-corpus Code bigcode/the-stack-v2-dedup Code HuggingFaceTB/stack-edu Medical TheBlueScrubs/TheBlueScrubs-v1

Table 24: Overview of all the open-source datasets used in Section 4.

##### A.6 Model Config and Hyperparameters

- A.6.1 Architecture

We use a modern decoder-only transformer model based on Gemma 3, which replaces the soft-capping method in Gemma 2 with QK-norm for attention stability [Gemma 3 Team, 2025, Gemma 2 Team, 2024]. We use a local attention sliding window size of 4096.

- A.6.2 2.3B Model Config

For the 2.3B parameter model, we use Llama3’s 128k tokenizer, an embedding dimension of 2304, an MLP dimension of 9216, and 26 layers [Llama Team, 2024]. Given that the embedding and un-embedding matrices are tied, the total embedding parameters is 294.9M parameters.

- A.6.3 2.3B Training Configurations

##### A.7 Additional Math Results

In Table 26, we report the math performance on an extended set of datasets. It include MegaMath Web Pro, which rewrites top math documents using Llama3.3-70b-Instruct [Llama Team, 2024]. Its strong performance shows the promise of LLM-assisted data cleaning [Zhou et al., 2025].

Hyperparameter Value Optimizer AdamW β1, β2 0.9, 0.95 Weight decay 0.1 Global batch size 2M tokens Peak LR 3.5 × 10−5 LR warm-up 2B tokens Cosine decay 3.5 × 10−5→3.5 × 10−6 Total training tokens 320B Sequence length 8,192

(a) Base-model training config.

##### Hyperparameter Value

Peak LR 3.5 × 10−6 Linear decay 3.5 × 10−6→ 0 Total annealing tokens 80B

(b) Annealing training config. Load the parameters and optimizer state from base model checkpoint at 320B tokens.

Table 25: Training and annealing configs. Values not listed in Table 25b are inherited from Table 25a.

##### Dataset GSM8K MATH MMLU–Math

MegaMath Web Pro 27.3%±1.2 12.2%±0.4 41.4%±0.9 FineMath 3+ 26.4%±1.4 11.7%±0.4 32.3%±1.5 OpenWebMath 14.6%±1.1 9.3%±0.4 29.9%±1.5 MegaMath Web (Top 10%) 9.8%±0.9 7.9%±0.3 29.9%±1.5 DCLM-baseline 4.8%±0.7 4.4%±0.3 27.0%±1.4

EAI-TAXONOMY Top Math 21.3%±1.3 11.0%±0.4 30.5%±1.5 EAI-TAXONOMY Math w/ FM 22.4%±1.3 11.5%±0.4 30.9%±1.5

- Table 26: Model performance (mean ± standard error) on GSM8K, Hendrycks Math, and MMLU-Math on expanded set of math datasets that include LLM-assisted rewriting of documents.

##### A.8 Taxonomy-Based Dataset Filters

Algorithm 2: Semantic filter used to curate Taxonomy Top Math dataset

DOC_TYPE_V1 = [ "Reference/Encyclopedic/Educational", "Code/Software", "Social/Forum", "Personal/Misc"

] DOC_TYPE_V2 = [

"Comment Section", "Documentation", "FAQ", "Knowledge Article", "Nonfiction Writing", "Personal Blog", "Q&A Forum", "Structured Data", "Tutorial"

] REASONING_DEPTH = [

"Basic Reasoning", "Intermediate Reasoning", "Advanced Reasoning", "Exceptional Reasoning"

] TECH_CORRECTNESS = ["Highly Correct", "Exceptionally Correct"] # Free Decimal Correspondence: 51 = Mathematics FDC_KEEP = ["51"]

# D: Essential Common Crawl (23.6B documents) # R: Taxonomy Top Math (19.8M documents) R = {

d for d in D if prefix(d.fdc.primary) in FDC_KEEP and d.doc_type_v1.primary in DOC_TYPE_V1 and d.doc_type_v2.primary in DOC_TYPE_V2 and d.reasoning_depth.primary in REASONING_DEPTH and d.technical_correctness.primary in TECH_CORRECTNESS

}

Algorithm 3: Semantic filter used to curate Taxonomy Math w/ FM dataset

|# Free Decimal Correspondence: 51 = Mathematics FDC_KEEP = ["51"]<br><br># D: Essential Common Crawl (23.6B documents) # R: Taxonomy Math w/ FM (21.6M documents) R = {<br><br>d for d in D if (prefix(d.fdc.primary) in FDC_KEEP or prefix(d.fdc.secondary) in FDC_KEEP) and d.finemath_score >= 3.25<br><br>}<br><br>|
|---|

Algorithm 4: Filter used to curate Taxonomy Code dataset

|DOC_TYPE_V1 = [ "Reference/Encyclopedic/Educational", "Social/Forum"<br><br>] DOC_TYPE_V2 = [<br><br>"Comment Section", "Documentation", "Knowledge Article", "Tutorial", "Personal Blog", "Q&A Forum"<br><br>] REASONING_DEPTH = [<br><br>"Intermediate Reasoning", "Advanced Reasoning", "Exceptional Reasoning"<br><br>] TECH_CORRECTNESS = ["Highly Correct"] # Free Decimal Correspondence: 005.1 = Programming, 005.3 = Systems Programming FDC_KEEP = ["005.1", "005.3"]<br><br># D: Essential Common Crawl (23.6B documents) # R: Taxonomy Code (42.7M documents) R = {<br><br>d for d in D if prefix(d.fdc.primary) in FDC_KEEP and d.doc_type_v1.primary in DOC_TYPE_V1 and d.doc_type_v2.primary in DOC_TYPE_V2 and d.reasoning_depth.primary in REASONING_DEPTH and d.technical_correctness.primary in TECH_CORRECTNESS<br><br>}<br><br>|
|---|

###### Algorithm 5: Filter used to curate Taxonomy Code w/ DCLM dataset

|DOC_TYPE_V2 = [ "Personal Blog", "Knowledge Article", "Comment Section", "Documentation", "Tutorial", "Q&A Forum"<br><br>] REASONING_DEPTH = [<br><br>"Basic Reasoning", "Intermediate Reasoning", "Advanced Reasoning", "Exceptional Reasoning"<br><br>] # threshold used to filter DCLM Pool -> DCLM-baseline DCLM_baseline_thresh = 0.01811<br><br># Free Decimal Correspondence: 004 = Computer Science, 005 = Software Development, # 51 = Mathematics FDC_KEEP = ["004", "005", "51"]<br><br># D: Essential Common Crawl (23.6B documents) # R: Taxonomy Code w/ DCLM (274M documents) R = {<br><br>d for d in D if prefix(d.fdc.primary) in FDC_KEEP and d.doc_type_v2.primary in DOC_TYPE_V2 and d.reasoning_depth.primary in REASONING_DEPTH and d.quality_signals.rps_doc_ml_eli5_score > DCLM_baseline_thresh<br><br>}<br><br>|
|---|

Algorithm 6: Filter filter used to curate Taxonomy Medical

|DOC_TYPE_V1 = [ "Academic/Research", "Reference/Encyclopedic/Educational"<br><br>] DOC_TYPE_V2 = [<br><br>"Academic Writing", "Documentation", "Knowledge Article", "Q&A Forum"<br><br>] REASONING_DEPTH = [<br><br>"Basic Reasoning", "Intermediate Reasoning", "Advanced Reasoning", "Exceptional Reasoning"<br><br>] TECH_CORRECTNESS = ["Highly Correct", "Exceptionally Correct"] # Hard-science Free Decimal Correspondence Prefixes SCIENCE_CODES = ["50", "51", "54", "57", "58", "59", "61"] # Free Decimal Correspondence: 61 = Medicine FDC_KEEP = ["61"] # Document-type blacklists DOC_TYPE_V1_BLACKLIST = [<br><br>"News/Editorial", "Code/Software", "Social/Forum", "Promotional/Advertisement", "Adult/Pornographic", "Personal/Misc", "Machine-Generated", "E-Commerce/Marketplace", "Images/Videos/Audio"<br><br>] DOC_TYPE_V2_BLACKLIST = [<br><br>"About (Org.)", "About (Personal)", "Audio Transcript", "Comment Section", "Content Listing", "Creative Writing", "Legal Notices", "Listicle", "News (Org.)", "News Article", "Personal Blog", "Product Page", "Spam / Ads", "Structured Data", "Truncated", "Tutorial", "User Review"<br><br>]<br><br># D: Essential Common Crawl (23.6B documents) # S: Taxonomy Medical (235M documents) S = {<br><br>d for d in D # FDC pairing: Medicine (61) w/ another science code if (<br><br>(prefix(d.dds.primary) in FDC_KEEP and prefix(d.dds.secondary) in SCIENCE_CODES) or (prefix(d.dds.secondary) in FDC_KEEP and prefix(d.dds.primary) in SCIENCE_CODES)<br><br>) # Document-type whitelist & blacklist checks and (<br><br>d.doc_type_v1.primary in DOC_TYPE_V1 or d.doc_type_v2.primary in DOC_TYPE_V2<br><br>) and d.doc_type_v1.primary not in DOC_TYPE_V1_BLACKLIST and d.doc_type_v2.primary not in DOC_TYPE_V2_BLACKLIST # Quality and d.reasoning_depth.primary in REASONING_DEPTH and d.technical_correctness.primary in TECH_CORRECTNESS<br><br>}<br><br>|
|---|

Algorithm 7: Filter to curate the Taxonomy Medical w/ DCLM. Refer to Algorithm 6 for Taxonomy Medical.

|# threshold used to filter DCLM Pool -> DCLM-baseline DCLM_baseline_thresh = 0.01811<br><br># D: Taxonomy Medical (235M documents) # R: Taxonomy Medical w/ DCLM (81.2M documents) R = {<br><br>d for d in D if d.quality_signals.rps_doc_ml_eli5_score > DCLM_baseline_thresh<br><br>}<br><br>|
|---|

Algorithm 8: Filter used to curate the Taxonomy STEM corpus

|# Document-types CODE_DOC_TYPE_V1 = ["Academic/Research", "Reference/Encyclopedic/Educational", "Code/Software", "Social/Forum"] CODE_DOC_TYPE_V2 = ["Academic Writing", "Comment Section", "Documentation", "Knowledge Article", "Personal Blog",<br><br>"Q&A Forum", "Tutorial"]<br><br>MED_DOC_TYPE_V1 = ["Academic/Research", "Reference/Encyclopedic/Educational", "Code/Software", "Legal/Regulatory"] MED_DOC_TYPE_V2 = ["Academic Writing", "Documentation", "FAQ", "Knowledge Article", "News Article", "Tutorial"]<br><br>ENG_DOC_TYPE_V1 = ["Academic/Research", "Reference/Encyclopedic/Educational", "Personal/Misc", "Legal/Regulatory"] ENG_DOC_TYPE_V2 = ["Academic Writing", "Audio Transcript", "Documentation", "FAQ", "Knowledge Article", "News Article",<br><br>"Tutorial"]<br><br>DEF_DOC_TYPE_V1 = ["Academic/Research", "Reference/Encyclopedic/Educational"] DEF_DOC_TYPE_V2 = ["Academic Writing", "Knowledge Article", "News Article"]<br><br># Reasoning depth REASONING_DEPTH_BAD = ["Abstain", "Indeterminate"]<br><br># Free Decimal Correspondence prefixes SCIENCE_FDC = ["50","51","52","53","54","55","56","57","58","59"] TECH_FDC = ["60","61","62","66","00"] # 00 = Computer science VALID_FDC = SCIENCE_FDC + TECH_FDC PROG_FDC = ["005.1","005.4"] # programming & systems prog.<br><br># D: Essential Common Crawl (23.6 B docs) # R: Taxonomy-STEM (1B docs) R = {<br><br>d for d in D if prefix(d.fdc.primary) in VALID_FDC<br><br>and prefix(d.fdc.secondary) in VALID_FDC and d.reasoning_depth.primary not in REASONING_DEPTH_BAD and d.reasoning_depth.primary is not None and (<br><br># Code ((prefix(d.fdc.primary,5) in PROG_FDC or prefix(d.fdc.secondary,5) in PROG_FDC)<br><br>and d.doc_type_v1.primary in CODE_DOC_TYPE_V1 and d.doc_type_v2.primary in CODE_DOC_TYPE_V2)<br><br>or # Medical<br><br>((prefix(d.fdc.primary)=="61" or prefix(d.fdc.secondary)=="61") and d.doc_type_v1.primary in MED_DOC_TYPE_V1 and d.doc_type_v2.primary in MED_DOC_TYPE_V2)<br><br>or # Engineering<br><br>((prefix(d.fdc.primary)=="62" or prefix(d.fdc.secondary)=="62") and d.doc_type_v1.primary in ENG_DOC_TYPE_V1 and d.doc_type_v2.primary in ENG_DOC_TYPE_V2)<br><br>or # Default science/tech (d.doc_type_v1.primary in DEF_DOC_TYPE_V1<br><br>and d.doc_type_v2.primary in DEF_DOC_TYPE_V2) )<br><br>}<br><br>|
|---|

Algorithm 9: Filter to curate the Taxonomy STEM w/ DCLM. Refer to Algorithm 8 for Taxonomy STEM.

|# threshold used to filter DCLM Pool -> DCLM-baseline DCLM_baseline_thresh = 0.01811<br><br># D: Taxonomy STEM (1B documents) # R: Taxonomy STEM w/ DCLM (342M documents) R = {<br><br>d for d in D if d.quality_signals.rps_doc_ml_eli5_score > DCLM_baseline_thresh<br><br>}<br><br>|
|---|

- A.9 Agreement Metric Discussion

- A.9.1 Motivating Our Variant of Cohen’s Kappa

Cohen’s κ is a popular choice for measuring inter-annotator agreement across a number of fields. Its robustness against annotators guessing with an imbalanced label distribution has been both criticized [Feinstein and Cicchetti, 1990] and lauded [Kolesnyk and Khairova, 2022]. For our part, we are attempting to measure agreement across multiple categories, and there are several points in our analysis where it is helpful to aggregate scores. While not fool-proof, we believe we are on firmer ground doing so using a kappa metric, than raw accuracy, since a kappa’s normalization term can make comparing categories with different numbers of labels easier.

Standard Cohen’s κ (and discrete agreement metrics in general) is most appropriate when the decision boundaries between class labels are clear. In cases where there is potential for fuzzy agreement (for example "assign a quality score from 1 to 10”), other agreement metrics like Kendall’s W are preferred. For our tasks, however, we believe there is fuzzyness in some of our class labels that cannot be solved by sorting the label space. For example, a document can match multiple document types (eg. a document can be both a "Tutorial" and "FAQ"), or its subject-matter can cross multiple FDC labels.

To this end we developed a variant of Cohen’s κ that works with overlapping agreement. We allow our annotators to output up to two labels for each category. Overlaps between annotation label sets x and y can either be scored as 0 or 1

s(x,y) = 1(x∩y̸=∅)∨(x∪y=∅) or weighted by proportion overlap

s(x,y) =

1 if x ∩ y = ∅ |x∩y| |x∪y| otherwise

with appropriate Pe for each.

- A.9.2 On wild swings of κ. The κ function is normally described as

Po − Pe 1 − Pe

0 ≤ Po ≤ 1, 0 ≤ Pe < 1, −1 ≤ κ ≤ 1

κ =

but its not necessary for Po and Pe to take on values over their whole range for κ to do so. In fact, for any ϵ > 0

1 − ϵ ≤ Po ≤ 1, 1 − ϵ ≤ Pe < 1 =⇒ −1 ≤ κ ≤ 1

This implies that as Po,Pe → 1, κ must become increasingly sensitive to small changes in its inputs.

In Table 13, there are big swings of κ in a few categories, the largest occurring to the Extraction Artifacts category. If we include Pe and Po when looking at κ values (Table 27), we can see the effect of small changes to κ’s inputs. While the Extraction Artifacts category’s κ falls almost 50pp, the Po’s are within a margin of 15pp, and the Pe’s are within 4pp. In the Education Level category on the random evaluation set, Po and κ disagree on whether Qwen2.5-32b-Instruct or EAI-Distill-0.5b performs better.

Qwen2.5-32B-Inst. EAI-Distill-0.5b Po Pe κ Po Pe κ

Category

Extraction Artifacts (Random) 0.93 0.75 0.74 0.79 0.71 0.27 Reasoning Depth (Random) 0.90 0.69 0.67 0.98 0.89 0.87 Technical Correctness (STEM) 0.77 0.55 0.51 0.94 0.78 0.75 Education Level (Random) 0.96 0.72 0.88 0.98 0.91 0.79

- Table 27: Po and Pe components help to explain whether a rise or drop in κ is real, or an artifact of the metric. In the first row, κ exaggerates Po. In the second two, Po and κ are roughly in line with relative improvement. The last row illustrates an ordering disagreement between Po and κ

There are at least two sources for simultaneously large Po and Pe:

Annotators A B C D E F Claude Sonnet 3.5 DeepSeek-V3

- A 0.41 0.50 0.55 0.59

- B 0.41 0.54 0.68 0.71

- C 0.50 0.54 0.72 0.74

- D 0.38 0.42 0.48 0.56

- E 0.38 0.51 0.65 0.64

- F 0.42 0.51 0.66 0.70

Claude Sonnet 3.5 0.55 0.68 0.72 0.48 0.65 0.66 0.81 DeepSeek-V3 0.59 0.71 0.74 0.56 0.64 0.70 0.81

- Table 28: Cohen κ between human (A, B, C, D, E, and F) and LLM annotators, averaged across four taxonomy categories. All human annotators have higher agreement with LLM annotators than they do with other human annotators

- • Unbalanced distribution of labels, as discussed in Feinstein and Cicchetti [1990].
- • An agreement score that is too relaxed. To that end, we created a version of our κ that weights agreement via degree of overlap (Appendix A.9.4). This version reduced ordering discrepancies and big swings in κ, but since the unweighted version of our κ did not affect any decisions after averaging across categories, we do not report this version of our κ scores.

##### A.9.3 Motivating the Use of LLMs as Reference Annotators

While assessing the quality of the different versions of our labeler, we’ve compared agreement with labeling carried out by a set of reference LLMs. A fair question would be whether we are missing something by not evaluating against humans. To that end, we enlisted six people to annotate 1150 documents with four of our taxonomy categories: FDC, Document Type V2, Reasoning Depth, and Education Level. Three annotators (A, B, and C in Table 28) annotated 575 documents, and the other three annotators (D, E, and F) labeled the remaining 575 documents. We then compared the human annotators against each other, and against two strong LLMs, using average Cohen’s κ across the labeled categories. For every human annotator, we found that the annotator had higher average κ with the LLMs than they did with other human annotators.

##### A.9.4 Deriving Pe Estimator for Annotator κ

For annotators An,n = 1,2 let fn,k,k = 0,1,2 be the probability An generates k labels for a document. Assuming k > 0, let wn,x be the probability of An first labeling the document with label x. Assume the second label is picked with the same wn, without replacement, so that if k = 2, then 1−wwn,y

is the probability An picks

n,x

label y given that x was picked first. Denote pj,k the probability of an agreement between A1 and A2’s annotations, given A1’s annotation is length j and A2’s annotation is length k, as discussed in 5.1.2. Then

2

Pe = f1,0f2,0 +

j=1

So we need to calculate pj,k for j = 1,2, k = 1,2

2

f1,jf2,kpj,k

k=1

- j = 1, k = 1 This is just the same calculation for Cohen’s κ

p1,1 =

x y

1x=yw1,xw2,ys([x],[y])

= s1,1

x

w1,xw2,x

s([x],[y]) is the weighted or unweighted matching score, and in both the weighted and unweighted case, s1,1 = 1 (in subsequent calculations, si,j is always 1 in the unweighted case)

- j = 1, k = 2 Begin calculating all possible matches of [x] and [y,z]

w2,z 1 − w2,y

p1,2 =

1y̸=z∧(x=y∨x=z)w1,xw2,y

s([x],[y,z])

x y z

we can break the calculation into two conditions: x = y and x = z. The conditions are disjoint, since both being true would imply y = z

w2,z 1 − w2,x

p1,2 = s1,2

1x̸=zw1,xw2,x

x z

w2,x 1 − w2,y p1,2 = s1,2

+ s1,2

1y̸=xw1,xw2,y

x y

w2,z 1 − w2,x

w1,xw2,x

1x̸=z

x

z

w2,y 1 − w2,y

+ s1,2

w1,xw2,x

1y̸=x

x

y

In the weighted case, s1,2 = 1/2 . If we denote

wn,x 1 − wn,x

(1) And note the identity

###### rn,y =

1x̸=y

x

wn,x 1 − wn,y

= 1 (2) then

1y̸=x

x

###### w1,xw2,x(1 + r2,x) (3)

p1,2 = s1,2

x

- j = 2, k = 1 The same calculation leading to equation 3 implies

p2,1 = s2,1

x

w1,xw2,x(1 + r1,x) where s2,1 = s1,2

- j = 2, k = 2 Begin by calculating all possible matches of [x,y] and [z,v].

w2,zw2,v 1 − w2,z

w1,xw1,y 1 − w1,x

p2,2 =

s([x,y],[z,v])

1x̸=y∧z̸=v∧(x=z∨x=v∨y=z∨y=v)

x y z v

We can calculate the four scenarios x = z, x = v, y = z, and y = v separately, but unlike case j = 1,k = 2, there are intersections we will end up double-counting, which will require subtractions (or weight adjustment in the weighted case). Thinking about the implications of intersections of scenarios in x ̸= y ∧ z ̸= v leads to determining that the only non-empty intersections are x = z ∧ y = w and x = w ∧ y = z. So, p2,2 can be rewritten:

w1,xw1,y 1 − w1,x

w2,xw2,v 1 − w2,x

p2,2 = s2,2

1x̸=y∧x̸=v

(x = z)

x y v

w2,zw2,x 1 − w2,z

w1,xw1,y 1 − w1,x

(x = v)

+ s2,2

1x̸=y∧x̸=z

x y z

w1,xw1,y 1 − w1,x

w2,yw2,v 1 − w2,y

+ s2,2

1x̸=y∧y̸=v

(y = z)

x y v

w1,xw1,y 1 − w1,x

w2,zw2,y 1 − w2,z

+ s2,2

1x̸=y∧z̸=y

(y = v)

x y z

w1,xw1,y 1 − w1,x

w2,xw2,y 1 − w2,x

(x = z ∧ y = v)

+ a

1x̸=y

x y

w1,xw1,y 1 − w1,x

w2,yw2,x 1 − w2,y

(x = v ∧ y = z)

+ a

1x̸=y

x y

Where s2,2 = 1/3 in the weighted case, and a is an over-count adjustment. a = −1 in the unweighted case, since we have counted matches of the form [x,y],[x,y] and [x,y],[y,x] twice. a = +1/3 in the weighted case, because the true weighted score for such matches is 1, not 1/3, and they were counted twice. After a little rearrangement on the four scenarios, and then substituting in equations 1 and 2

p2,2 = s2,2

+ s2,2

x

+ s2,2

y

+ s2,2

y

+ a

x y

+ a

x y

p2,2 = s2,2

+ a

x y

+ a

x y

x

x

w2,v 1 − w2,x

w1,y 1 − w1,x v

1x̸=v

w1,xw2,x

1x̸=y

y

w2,z 1 − w2,z

w1,y 1 − w1,x z

1x̸=z

w1,xw2,x

1x̸=y

y

w2,v 1 − w2,y

w1,x 1 − w1,x v

1y̸=v

w1,yw2,y

1x̸=y

x

w2,z 1 − w2,z

w1,x 1 − w1,x z

w1,yw2,y

1y̸=z

1x̸=y

x

w1,xw1,y 1 − w1,x

w2,xw2,y 1 − w2,x

1x̸=y

w1,xw1,y 1 − w1,x

w2,yw2,x 1 − w2,y

1x̸=y

w1,xw2,x(1 + r2,x + r1,x + r1,xr2,x)

w2,xw2,y 1 − w2,x

w1,xw1,y 1 − w1,x

1x̸=y

w2,yw2,x 1 − w2,y

w1,xw1,y 1 − w1,x

1x̸=y

- A.10 Evaluation Sets

- A.10.1 Domain-recall Gold URL Sets

For the human-vetted "gold" URL sets, we manually inspect and collect a set of high quality base URLs for the math and web code domains. Our goal is to ensure that these vetted URLs have a very high density of documents from their respective domain. For example, when visiting https://dlmf.nist.gov/ it is clear that the vast majority of the website should be designated as a mathematics. The math URLs can be found at Algorithm 10 and the code can be found at Algorithm 11.

- A.10.2 Sampling STEM Evaluation Set

The STEM evaluation set was created by up-sampling STEM categories as labeled by Qwen2.5-32b-Instruct. A breakdown of the Free Decimal Classification counts of the STEM evaluation set can be found in Table 29.

- Algorithm 10: Vetted Math Base URLS

|vetted_math_urls = ["https://dlmf.nist.gov/", "https://tutorial.math.lamar.edu/", "https://math24.net", "https:// whitman.edu/mathematics/calculus_online", "https://sfu.ca/math-coursenotes", "https://www2.clarku.edu/faculty/ djoyce", "https://clarku.edu/faculty/djoyce/trig", "https://math.libretexts.org", "https://sophisticatedprimate.com ", "https://mathisfun.com", "https://chilimath.com", "https://encyclopediaofmath.org/wiki", "https:// betterexplained.com/articles", "https://settheory.net/", "https://math24.net", "https://www.opentextbookstore.com/ buscalc/buscalc/", "https://oeis.org/", "https://yutsumura.com", "https://math.mit.edu/~djk/calculus_beginners", " https://bookdown.org/slcodia/Stat_136", "https://bookdown.org/huckley/Physical_Processes_In_Ecosystems/", "https:// stacks.math.columbia.edu/tag", "https://golem.ph.utexas.edu/", "https://johncarlosbaez.wordpress.com/", "https:// qchu.wordpress.com/", "https://cornellmath.wordpress.com/", "https://ncatlab.org", "https://proofwiki.org/wiki/", " https://mathoverflow.net/questions", "https://math.stackexchange.com/questions", "https://projecteuler.net", "https ://aperiodical.com/", "https://unapologetic.wordpress.com/", "https://www.jeremykun.com/", "https://terrytao. wordpress.com/", "https://11011110.github.io/blog/", "https://planetmath.org/", "https://alexsisto.wordpress.com/",<br><br>"https://cstheory.stackexchange.com/questions", "https://web.ma.utexas.edu/mediawiki", "http://theoremoftheweek. wordpress.com/", "http://tqft.net/mlp"]<br><br>|
|---|

- Algorithm 11: Vetted Web Code Base URLS

|vetted = ["github.com", "www.geeksforgeeks.org", "stackoverflow.com/questions", "www.experts-exchange.com/questions", " mail.python.org/pipermail", "www.linuxquestions.org/questions", "www.tomshardware.com/forum", "www.coderanch.com/t" , "superuser.com/questions", "community.spiceworks.com/topic", "sourceforge.net/p", "sourceforge.net/directory", " docs.microsoft.com/en-us", "serverfault.com/questions", "askubuntu.com/questions", "www.bleepingcomputer.com/forums ", "msdn.microsoft.com/en-us", "coderanch.com/t", "learn.microsoft.com/en-us", "unix.stackexchange.com/questions", "forums.unrealengine.com/t", "community.filemaker.com/thread", "www.geekstogo.com/forum", "forum.arduino.cc/t", " www.daniweb.com/programming", "sharepoint.stackexchange.com/questions", "www.construct.net/en/forum", "apple. stackexchange.com/questions", "forums.developer.nvidia.com/t", "techcommunity.microsoft.com/t5"]<br><br>|
|---|

##### Class FDC top-level category Documents

- 0 Information sciences 54
- 1 Philosophy & psychology 156

- 3 Social sciences 130
- 4 Languages & linguistics 46
- 5 Science & natural history 237
- 6 Technology & applied sciences 232
- 7 Arts & recreation 9
- 8 Literature 4
- 9 History & geography 3

##### Total 871

- Table 29: Document counts by Free Decimal Classification level 1 of the STEM evaluation set as labeled by Qwen2.5-32b-Instruct.

##### A.10.3 fastText Classifier Details

The fastText classifier for math and web code were both trained on 500k positive examples and 500k negative examples. The negative samples were documents randomly sampled from Common Crawl [Shao et al., 2024]. The hyper-parameters used to train the classifiers can be found in Table 30. The positive sets used to train the classifiers we use in the domain-recall experiments are:

- • Math: sample from OpenWebMath [Paster et al., 2023]
- • Web Code: documentation from package manager platforms (such as npm, PyPI, etc.)

##### Hyperparameter Value

Tokenizer whitespace Word n-grams 4 Embedding dimension 128 Hash bucket size 1,000,000 Min count 10 Epochs 2

Table 30: Hyperparameters use to train math and web code fastText classifiers.

The fastText classifiers did not utilize BPE tokenization or iterative training, which have both been shown to improve performance on domains such as math and code [Shao et al., 2024, Huang et al., 2025].

##### A.11 Detailed Taxonomy Evaluation Results

- A.11.1 Annotator κ including Qwen2.5-14b-Instruct See Table 31 for annotator κ of all open-source models tested.

DeepSeek-V3 Qwen 2.5-72B-Inst. Qwen 2.5-32B-Inst. Qwen 2.5-14B-Inst. Random STEM Random STEM Random STEM Random STEM

Category

Knowledge Domain 0.69±0.02 0.64±0.03 0.46±0.02 0.39±0.03 0.62±0.02 0.68±0.02 0.20±0.01 0.24±0.02 Cognitive Process 0.76±0.01 0.76±0.02 0.67±0.02 0.70±0.02 0.73±0.01 0.79±0.02 0.30±0.01 0.40±0.02 Document Type V1 0.90±0.01 0.91±0.01 0.88±0.01 0.91±0.01 0.86±0.01 0.89±0.01 0.60±0.01 0.62±0.02 FDC (level 1) 0.92±0.01 0.95±0.01 0.90±0.01 0.92±0.01 0.88±0.01 0.92±0.01 0.88±0.01 0.91±0.01 FDC (level 2) 0.86±0.01 0.87±0.01 0.83±0.01 0.81±0.01 0.81±0.01 0.81±0.01 0.78±0.01 0.76±0.01 FDC (level 3) 0.71±0.01 0.70±0.01 0.67±0.01 0.63±0.01 0.64±0.01 0.60±0.01 0.57±0.01 0.55±0.01 Extraction Artifacts 0.81±0.02 0.86±0.03 0.57±0.02 0.53±0.03 0.74±0.01 0.65±0.03 0.27±0.02 0.30±0.02 Missing Content 0.83±0.01 0.85±0.02 0.63±0.01 0.65±0.02 0.66±0.02 0.65±0.02 0.43±0.01 0.45±0.02 Document Type V2 0.89±0.01 0.89±0.01 0.80±0.01 0.80±0.01 0.85±0.01 0.83±0.01 0.71±0.01 0.69±0.02 Education Level 0.89±0.01 0.86±0.02 0.82±0.01 0.81±0.02 0.88±0.01 0.85±0.02 0.69±0.02 0.71±0.02 Reasoning Depth 0.75±0.01 0.72±0.02 0.70±0.01 0.67±0.02 0.67±0.02 0.67±0.02 0.47±0.01 0.50±0.02 Technical Corr. 0.60±0.02 0.61±0.02 0.52±0.01 0.60±0.02 0.52±0.01 0.51±0.02 0.30±0.01 0.25±0.02

Overall mean 0.80 0.80 0.70 0.70 0.74 0.74 0.52 0.53

- Table 31: Annotator κ (± s.e.) between each candidate model and the two gold annotators on the random (n = 2,017) and STEM (n = 871) evaluation sets.

- A.11.2 Inter-Category NMI Additional NMI Heatmaps Additional inter-category NMI heatmaps can be found in Figure 6.

[Figure 5]

[Figure 6]

(a) DeepSeek-V3 random NMI (b) DeepSeek-V3 STEM NMI

[Figure 7]

[Figure 8]

(c) Qwen2.5-72b-Instructrandom NMI (d) Qwen2.5-72b-Instruct STEM NMI

Figure 6: NMI heatmaps on random and STEM evaluation sets for DeepSeek-V3 and Qwen2.5-72b-Instruct

- A.12 EAI-Distill-0.5b Training Details & Ablations

- A.12.1 Comparison of prefill and generation tokens of Qwen2.5-32b-Instruct vs EAI-Distill-0.5b

Given we annotated the 104.6M sample of Common Crawl in two passes as explained in Apprendix A.13, we decide to use the Prompt 1 (Appendix A.14.3) when calculating performance deltas from Qwen2.5-32b-Instruct to EAI-Distill-0.5b. This is a lower bound on performance impact given Prompt 1 just annotates a document for 8 categories (not all 12). However, EAI-Distill-0.5b annotates a document with all 12 categories.

Average Input & Output Tokens. We sample 1,024,83 documents annotated by both Qwen2.5-32b-Instruct and EAI-Distill-0.5b. We then calculate and report the average number of tokens in the prompt and generated output. We ignore the cached prefix of EAI-Distill-0.5b given it is only 11 tokens before the document is provided (these tokens come from applying the chat template and the short system prompt). The average prompt and generated tokens can be found in Table 32.

Prompt Output Shared prefix Avg. prompt tok. Avg. gen. tok.

Prompt 1 Prompt 1 Output 2104 3037 791 No Prompt Final Output 0 934 51

Table 32: Average token statistics across 1,024,836 documents. Prompt 1 Output is produced by Qwen2.5-32B-Instruct; Final Output by EAI-Distill-0.5b.

- A.12.2 Training Hyperparameters The final hyperparameters used to fine-tuned Qwen2.5-0.5b-Instruct can be found in Table 33.

Hyperparameter Value Notes Optimizer type AdamW β1,β2 0.9, 0.95 Weight decay 0.1 Global batch size 2M tokens Peak learning rate 1 × 10−4 LR warm-up 2B tokens linear to peak LR Cosine decay phase 1 × 10−4→1 × 10−5 follows warm-up Linear anneal phase 1 × 10−5 → 0 final 2B tokens Total fine-tuning tokens 82 B synthetic labels Sequence Length 16,384

Table 33: Hyper-parameters used to fine-tune Qwen2.5-0.5b-Instruct.

- A.12.3 EAI-Distill-0.5b Fine-Tuning Ablations

Fine-tuning prompt / generation format ablation. We ablate the affects on annotator κ of context distillation and shortening generation tokens when fine-tuning Qwen2.5-0.5b-Instruct. We compare training on Prompt

- 1 (Appendix A.14.3) with its standard generation format against no prompt and a highly condensed output format (Algorithm 13).28 Both models are trained for 200B tokens, have a learning rate of 2 × 10−6, and a max sequence length of 8192. Other than that all hyper-parameters match Table 33. Unfortunately, one run crashed at around 80B tokens seen, so we take the latest shared checkpoint of both models for comparison.29 We see that annotator κ of the two variants are very similar (Table 34).

Fine-tuning learning rate ablation. Having selected no prompt and the condensed generation format (Algorithm 13), we run ablations to determine the best learning rate to use during training. For these experiments, we fine-tune Qwen2.5-0.5b-Instruct for 12B tokens and test the following learning rates:

28We used an abridged version of the condensed output format that only outputs the 8 categories in Prompt 1. 29The model with no prompt, condensed generation format saw more examples because more examples can be packed into a

given sequence.

Prompt 1 Input & Format No Prompt, Condensed Generation Random STEM Random STEM

Category

Knowledge Domain 0.47±0.03 0.52±0.03 0.54±0.02 0.58±0.03 Cognitive Process 0.56±0.02 0.62±0.03 0.68±0.02 0.74±0.02 Document Type V1 0.82±0.01 0.84±0.01 0.82±0.01 0.83±0.01 Free Decimal Corr. (level 1) 0.83±0.01 0.86±0.01 0.86±0.01 0.91±0.01 Free Decimal Corr. (level 2) 0.74±0.01 0.65±0.01 0.79±0.01 0.78±0.01 Free Decimal Corr. (level 3) 0.54±0.01 0.48±0.02 0.62±0.01 0.58±0.01 Extraction Artifacts 0.31±0.03 0.48±0.04 0.20±0.03 0.21±0.04 Missing Content 0.64±0.02 0.74±0.02 0.50±0.02 0.55±0.03

Overall mean 0.62 0.65 0.63 0.65

- Table 34: Annotator κ (± standard error) against gold annotators on the random (n = 2,017) and STEM (n = 871) evaluation sets for prompt / generation format ablation.

2 × 10−6, 1 × 10−5, 1 × 10−4, and 1 × 10−3. All other hyper-parameters match Table 33. We can see that performance drops off at a learning rate of 1 × 10−3. We select 1 × 10−4 given its the highest learning rate before the drop. The results for this ablation can be found in Table 35.

Category 2 × 10−6 1 × 10−5 1 × 10−4 1 × 10−3

Rand. STEM Rand. STEM Rand. STEM Rand. STEM

Knowledge Domain 0.63±0.02 0.61±0.03 0.63±0.02 0.65±0.03 0.64±0.02 0.64±0.03 0.58±0.02 0.61±0.03 Cognitive Process 0.68±0.02 0.76±0.02 0.69±0.02 0.76±0.02 0.69±0.02 0.75±0.02 0.67±0.02 0.73±0.02 Document Type V1 0.82±0.01 0.84±0.01 0.84±0.01 0.86±0.01 0.85±0.01 0.86±0.01 0.83±0.01 0.85±0.01

- Free Decimal Corr. (level 1) 0.87±0.01 0.89±0.01 0.88±0.01 0.92±0.01 0.88±0.01 0.92±0.01 0.86±0.01 0.87±0.01
- Free Decimal Corr. (level 2) 0.79±0.01 0.70±0.01 0.80±0.01 0.78±0.01 0.81±0.01 0.80±0.01 0.78±0.01 0.73±0.01
- Free Decimal Corr. (level 3) 0.61±0.01 0.52±0.02 0.64±0.01 0.58±0.01 0.64±0.01 0.62±0.01 0.62±0.01 0.54±0.01 Extraction Artifacts 0.28±0.02 0.39±0.04 0.26±0.02 0.43±0.03 0.28±0.02 0.42±0.04 0.23±0.03 0.38±0.03 Missing Content 0.52±0.02 0.60±0.02 0.50±0.02 0.60±0.02 0.50±0.01 0.60±0.02 0.50±0.02 0.59±0.03 Document Type V2 0.86±0.01 0.86±0.01 0.88±0.01 0.87±0.01 0.88±0.01 0.87±0.01 0.87±0.01 0.83±0.01 Educational Level 0.75±0.03 0.87±0.02 0.77±0.02 0.86±0.02 0.78±0.03 0.85±0.02 0.74±0.03 0.83±0.02 Reasoning Depth 0.86±0.02 0.75±0.02 0.86±0.02 0.75±0.02 0.86±0.02 0.75±0.03 0.86±0.02 0.73±0.03 Technical Correctness 0.70±0.02 0.74±0.03 0.71±0.01 0.74±0.03 0.72±0.02 0.73±0.02 0.71±0.01 0.70±0.03 Overall mean 0.70 0.71 0.71 0.73 0.71 0.74 0.69 0.70

- Table 35: Annotator κ (mean ± standard error) of Qwen2.5-0.5b-Instruct finetuned for 12B tokens with four learning rates. Each pair of columns shows Random (n=2,017) and STEM (n=871) subsets; headers indicate the learning rate.

Fine-tuning token budget ablation. Having selected a learning rate of 1 × 10−4, we experiment with the token budget used for fine-tuning Qwen2.5-0.5b-Instruct. All hyper-parameters are set to the values in Table 33 and we vary the token budget as follows: 12B, 22B, 42B, 82B. Annotator κ does not change as we fine-tune for longer token budgets. Given we trained all 4 models, we select the model trained on the most tokens for EAI-Distill-0.5b. However, it is clear that similar performance can be achieved with a smaller token budget. This has important implications for determining the number of synthetic annotations needed from a powerful LLM to adequately fine-tune a much smaller LM for classification / data labeling tasks like the taxonomic annotation. Results of this ablation can be found in Table 36.

##### A.13 Qwen2.5-32b-Instruct Large Scale Annotation

For the purposes of fine-tuning EAI-Distill-0.5b and getting a large sample for experimentation, we label 104.6M documents randomly sampled from our processed Common Crawl (Appendix A.2) with Qwen2.5-32b-Instruct. This annotation was done in two phases given we added additional categories to the taxonomy before fine-tuning. The two prompts used for annotation can be found in Appendix A.14.3 and Appendix A.14.3. Both phases used the system prompt found in Appendix A.14.3.

12 B tokens 22 B tokens 42 B tokens 82 B tokens Rand. STEM Rand. STEM Rand. STEM Rand. STEM

Category

Knowledge Domain 0.63±0.02 0.64±0.03 0.62±0.02 0.65±0.03 0.61±0.02 0.66±0.02 0.63±0.02 0.65±0.03 Cognitive Process 0.69±0.01 0.75±0.02 0.70±0.01 0.76±0.02 0.69±0.01 0.76±0.02 0.70±0.02 0.76±0.02 Document Type V1 0.85±0.01 0.87±0.01 0.84±0.01 0.86±0.01 0.85±0.01 0.87±0.01 0.83±0.01 0.86±0.01 Free Decimal Corr. (level 1) 0.88±0.01 0.92±0.01 0.88±0.01 0.93±0.01 0.88±0.01 0.93±0.01 0.88±0.01 0.93±0.01 Free Decimal Corr. (level 2) 0.81±0.01 0.80±0.01 0.81±0.01 0.80±0.01 0.81±0.01 0.80±0.01 0.81±0.01 0.80±0.01 Free Decimal Corr. (level 3) 0.64±0.01 0.62±0.01 0.64±0.01 0.61±0.01 0.65±0.01 0.61±0.01 0.63±0.01 0.61±0.01 Extraction Artifacts 0.28±0.02 0.42±0.03 0.28±0.02 0.39±0.03 0.28±0.02 0.39±0.03 0.27±0.02 0.37±0.03 Missing Content 0.50±0.01 0.60±0.02 0.50±0.01 0.59±0.02 0.50±0.01 0.58±0.02 0.48±0.01 0.57±0.02 Document Type V2 0.88±0.01 0.87±0.01 0.88±0.01 0.87±0.01 0.88±0.01 0.86±0.01 0.88±0.01 0.86±0.01 Education Level 0.78±0.02 0.85±0.02 0.78±0.02 0.85±0.02 0.80±0.03 0.86±0.02 0.79±0.02 0.86±0.02 Reasoning Depth 0.86±0.02 0.74±0.03 0.87±0.01 0.76±0.03 0.86±0.01 0.76±0.02 0.87±0.01 0.76±0.02 Technical Correctness 0.72±0.01 0.73±0.03 0.72±0.01 0.73±0.02 0.72±0.02 0.75±0.03 0.72±0.01 0.75±0.02

Overall mean 0.71 0.74 0.71 0.73 0.71 0.74 0.71 0.73

- Table 36: Annotator κ (mean ± s.e.) for Qwen2.5-0.5b-Instruct fine-tuned on 12, 22, 42, and 82 billion training tokens. Each pair of columns shows Random (n=2,017) and STEM (n=871) subsets.

Algorithm 12: Function to subsample documents with more than 30,000 characters

|def chunk_text(text, max_char_per_doc): if len(text) <= max_char_per_doc: return text<br><br>chunk_size = max_char_per_doc // 3 start = text[:chunk_size]<br><br># Calculate valid range for middle section middle_start = chunk_size middle_end = len(text) - chunk_size<br><br># Randomly select middle point within valid range mid_point = random.randint(middle_start + chunk_size//2, middle_end - chunk_size//2) middle = text[mid_point - chunk_size//2:mid_point + chunk_size//2] end = text[-chunk_size:] return f"[beginning]\n{start}\n[middle]\n{middle}\n[end]\n{end}"<br><br>|
|---|

##### A.13.1 DeepSeek-V3 SGLang Image

At the time we ran this large scale annotation, the SGLang image for AMD MI300x was much slower. The old image mentioned in Table 12 refers to v0.4.1.post3 and the newer image benchmarked refers to v0.4.3.post4.

A.14 Prompts, Generation Formats, & Examples

##### A.14.1 Document Sampling

When running inference with Qwen2.5-32b-Instruct and EAI-Distill-0.5b, we subsample documents with more than 30,000 characters. We do this to improve inference speed. We use the function in Algorithm 12 to take the beginning, random sample of the middle, and end of any document with over 30,000 characters. Assuming there are 4 bytes per token, the maximum document should be about 7500 tokens. We decide not to tokenize when sub-sampling to avoid the CPU-cost of tokenizing all document before inference. The training data for EAI-Distill-0.5b is prepared using subsampling to ensure that the training and inference distributions match.

Algorithm 13: Template for EAI-Distill-0.5b Condensed Model Output Format

|{FDC primary classification},{FDC secondary classification or skip} {Bloom cognitive process primary (1-6)},{Bloom cognitive process secondary (1-6) or skip} {Bloom knowledge domain primary (1-4)},{Bloom knowledge domain secondary (1-4) or skip} {Document type v1 primary (1-17)},{Document type v1 secondary (1-17) or skip} {Extraction artifacts primary (0-4)},{Extraction artifacts secondary (0-4) or skip} {Missing content primary (0-6)},{Missing content secondary (0-6) or skip} {Document type v2 primary (1-25)},{Document type v2 secondary (1-25) or skip} {Reasoning depth primary (1-6)},{Reasoning depth secondary (1-6) or skip} {Technical correctness primary (1-6)},{Technical correctness secondary (1-6) or skip} {Educational level primary (1-5)},{Educational level secondary (1-5) or skip}<br><br>|
|---|

##### A.14.2 EAI-Distill-0.5b Generation Templates

During fine-tuning we update the generation to a highly-condensed format. We do so by programatically extracting the response codes from Generations 1 and 2. The generation template for EAI-Distill-0.5b can be found in Algorithm 13.

###### A.14.3 Prompts EAI-Distill-0.5b System Prompt GitHub: EAI-Distill-0.5b System Prompt Qwen2.5-32b-Instruct System Prompt GitHub: System Prompt

- Qwen2.5-32b-Instruct Prompt 1 GitHub: Prompt 1
- Qwen2.5-32b-Instruct Prompt 2 GitHub: Prompt 2

