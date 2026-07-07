# arXiv:2512.14051v1[cs.AI]16Dec2025

[Figure 1]

[Figure 2]

### OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value

OpenDataArena Team1 1Shanghai Artificial Intelligence Laboratory, OpenDataLab

The rapid evolution of Large Language Models (LLMs) is predicated on the quality and diversity of post-training datasets. However, a critical dichotomy persists: while models are rigorously benchmarked, the data fueling them remains a “black box”—characterized by opaque composition, uncertain provenance, and a lack of systematic evaluation. This opacity hinders reproducibility and obscures the causal link between data characteristics and model behaviors. To bridge this gap, we introduce OpenDataArena (ODA), a holistic and open platform designed to benchmark the intrinsic value of post-training data. ODA establishes a comprehensive ecosystem comprising four key pillars: (i) a unified training–evaluation pipeline that ensures fair, open comparisons across diverse models (e.g., Llama, Qwen) and domains; (ii) a multi-dimensional scoring framework that profiles data quality along tens of distinct axes; (iii) an interactive data lineage explorer to visualize dataset genealogy and dissect component sources; and (iv) a fully open-source toolkit for training, evaluation, and scoring to foster data research. Extensive experiments on ODA—covering over 120 training datasets across multiple domains on 22 benchmarks, validated by more than 600 training runs and 40 million processed data points—reveal non-trivial insights. Our analysis uncovers the inherent trade-offs between data complexity and task performance, identifies redundancy in popular benchmarks through lineage tracing, and maps the “genealogical” relationships across datasets. We release all results, tools, and configurations to democratize access to high-quality data evaluation. Rather than merely expanding a leaderboard, ODA envisions a shift from trial-and-error data curation to a principled science of Data-Centric AI, paving the way for rigorous studies on data mixing laws and the strategic composition of foundation models.

Date: December 17, 2025 Correspondence: Lijun Wu, wulijun@pjlab.org.cn Project Page: https://opendataarena.github.io/ Toolkit: https://github.com/OpenDataArena/OpenDataArena-Tool HuggingFace: https://huggingface.co/OpenDataArena/datasets

### 1 Introduction

The rapid evolution of Large Language Models (LLMs), such as the GPT series [6, 2, 24], Qwen series [4, 60, 59] and Llama series [53, 54, 19], has marked a paradigm shift in Artificial Intelligence (AI), demonstrating remarkable capabilities in understanding, generation, and reasoning. While much of the community’s focus has been on architectural innovations [36] and scaling laws [26], a critical determinant of these models’ ultimate performance and alignment lies in the post-training phase. This stage, encompassing Supervised Fine-Tuning (SFT) and alignment processes [42], relies heavily on curated datasets to sculpt a base model’s behavior, imbuing it with the ability to follow instructions, engage in dialog, and adhere to human values. The quality, diversity, and composition of this post-training data are therefore not just influential but are arguably the key ingredients that

[Figure 3]

- Figure 1: Overview of the OpenDataArena framework. We provide four integral components: a Data Value Leaderboard for standardized benchmarking, a Multi-dimension Data Scorer for granular quality assessment, a Data Analysis Platform for lineage and composition tracing, and an Open-source Evaluation Toolkit to ensure reproducibility.

transform a powerful predictive engine into a helpful and reliable AI assistant [49, 18, 50, 52, 8].

Despite its pivotal role, the landscape of post-training datasets is fraught with opacity and lacks a standardized evaluation protocol. The creation and selection of datasets is often an ad-hoc process, leading to a proliferation of resources with varying quality, such as those generated through distillation from proprietary models like Alpaca [50] or crowd-sourcing efforts like Dolly [13]. While some studies have argued for the power of small, high-quality datasets [67], and others have begun to analyze the factors that make data effective for alignment [37], the community still lacks a systematic and fair methodology to evaluate dataset quality and its downstream impact. This opacity hinders scientific progress by making it difficult to reproduce results, understand the source of performance gains, and efficiently allocate resources for data curation. The fundamental question of “what constitutes a ‘good’ dataset?” remains largely unanswered in a quantifiable and generalizable way.

To bridge this gap, we present OpenDataArena (ODA), a fair, open, and transparent platform designed to systematically benchmark the value of post-training datasets. Our primary contributions through ODA are fourfold (as shown in Figure 1).

- • We establish a unified data evaluation leaderboard built upon a standardized training and evaluation pipeline, enabling fair “apples-to-apples” comparisons of datasets across various models and downstream benchmarks.
- • We leverage a multi-dimensional scoring system that moves beyond single performance metrics to generate a holistic quality profile for each dataset, assessing it along numerous axes, including instruction complexity, response quality, and data diversity.
- • We introduce an interactive data lineage explorer, a web-based analytic tool that visualizes dataset genealogy, allowing users to transparently trace data provenance and dissect the constituent sources of aggregated datasets.
- • We release a fully open-source platform, providing the community with all the necessary tools,

configurations, and results to ensure reproducibility and facilitate further research.

Leveraging ODA, we bridge the gap between dataset transparency and performance evaluation. We begin by mapping the “genealogy” of the ecosystem, identifying clusters of related datasets and their influence flows. We then proceed to a large-scale evaluation involving 120 training datasets (continuously expanding), 600+ training runs, 22 benchmarks, 10000+ evaluation runs, and 40 million processed samples. This extensive benchmarking uncovers domain-specific performance variations and model-specific preferences (e.g., Llama3.1 [19], Qwen2.5 [60], Qwen3 [59]). Moreover, we conduct a correlation analysis between our fine-grained quality metrics and downstream results to determine the true drivers of data value. The study concludes with an efficiency analysis, highlighting high-yield datasets to inform future data curation.

The establishment of ODA aims to provide tangible benefits to the academic and research community. (1) For model trainers and data researchers, it streamlines the evaluation and selection process, helping them quickly identify high-quality datasets and mitigate costly blind trial-and-error. (2) For researchers in data synthesis, our multi-dimensional scores, data lineage analysis platform, and open-source tools offer critical guidance, helping them identify high-value “seed data” and benchmark the quality of their generated datasets. (3) Ultimately, ODA empowers academic researchers to explore the intrinsic link between data characteristics and model performance, providing solid empirical grounding and objective evaluation criteria for cutting-edge research in data selection and generation.

In summary, ODA serves as a foundational infrastructure to inject rigor and transparency into the critical yet opaque domain of post-training data. We view this initiative as a pivotal step toward a principled, data-centric evaluation paradigm. We offer this work as a catalyst, galvanizing the community to move beyond empirical intuition toward a shared mission of rigorous assessment. As we look ahead, the challenge of decoding the fundamental laws of data utility and composition will demand a collective, open effort to transform data selection from an art into a science.

### 2 The OpenDataArena Platform

The core of our contribution is the OpenDataArena (ODA) platform, a comprehensive ecosystem designed for the systematic and transparent evaluation of post-training datasets. This section details the architecture, design principles, and core components of ODA. We begin with an overview of the platform’s goals and design philosophy, followed by a detailed breakdown of our standardized benchmarking pipeline, from data selection to training and evaluation. Finally, we introduce our multidimensional data scoring framework, which provides deeper diagnostic insights into the intrinsic properties of each dataset.

#### 2.1 Design Principle

The primary motivation behind ODA is to address the critical gap in the data-centric AI landscape: the lack of fair, reproducible, and systematic methods for evaluating the value of post-training data. In recent years, the community has produced a deluge of post-training datasets, but this explosion in quantity has not necessarily translated to a consistent increase in quality. Furthermore, the fundamental question of how to define and measure what makes a dataset “good” remains largely unresolved. Our work is grounded in a fundamental tenet:

“The data value is demonstrated solely by its ability to concretely improve a model’s capabilities.” To provide a clear, empirical framework for measuring this, the ODA platform was architected with the following core design principles.

Fairness and Unbiased Comparison. Our foremost principle is to ensure that all datasets are evaluated under identical conditions. By fixing the base model, training hyperparameters, and evaluation

[Figure 4]

- Figure 2: An overview process of OpenDataArena, which includes a four-stage data evaluation/benchmarking pipeline we designed (data input layer, data evaluation layer, data analysis layer, and data visualization layer), the user interaction module, and open-source tools.

protocols, we isolate the dataset as the sole variable. This allows any observed performance differences to be directly attributed to the data itself, enabling a true “apples-to-apples” comparison.

Reproducibility and Openness. To build trust and foster community collaboration, ODA is built on a foundation of openness. We provide open access to all our code, training and evaluation configurations, raw results, and analysis scripts. This transparency allows any researcher to verify our findings, replicate our experiments, and build upon our work.

Comprehensiveness and Extensibility. A meaningful data evaluation must cover a wide range of capabilities. Therefore, ODA incorporates benchmarks across diverse domains, including general chat, scientific knowledge, mathematical reasoning, and coding. The platform is also designed to be modular, allowing for the seamless integration of new datasets, models, and evaluation benchmarks as the field evolves.

Data-Centricity. ODA is fundamentally a data-centric evaluation platform. Its entire design is geared towards answering the question: “What is the intrinsic value of this dataset?” The leaderboard rankings and multi-dimensional scores serve as quantitative and qualitative answers to this question, shifting the research focus from model-centric tuning to a deeper understanding of data.

#### 2.2 Platform Overview

To implement these principles, the ODA platform is designed as a systematic, end-to-end workflow, as illustrated in Figure 2. This workflow involves the following key components and stages.

User Interaction and Tool Support. The entire ecosystem is designed for Users—such as researchers and developers—who can leverage the platform for data evaluation, insights, and contributions. We support users to upload their interested datasets and evaluation results. The whole process is supported by a unified, open-source set of Tools that ensures reproducibility, transparency, and allows the community to extend the platform’s capabilities.

[Figure 5]

- Figure 3: A summarized visualization of data leaderboards. More detailed leaderboards on different domains can be found in the project page: https://opendataarena.github.io/leaderboard.html.

Four-Stage Evaluation Pipeline. The core component is the data benchmarking/evaluation pipeline, where Data flows through sequential four stages to move from ingestion to insight.

- 1. The Data Input Layer serves as the entry of the whole pipeline where datasets are collected, normalized into a unified format, and systematically classified by domain.
- 2. The Data Evaluation Layer acts as the core engine, executing the standardized train-evaluate benchmarks on models like Qwen [60, 59] and Llama [19] (Section 2.3), while also calculating the multi-dimensional data quality scores (Section 2.4).
- 3. The Data Analysis Layer synthesizes the raw outputs to perform in-depth analyses, including cross-performance comparisons, domain efficacy assessments, data scoring evaluations, and explorations of data family relationships (Section 3 and 4).
- 4. The Data Visualization Layer renders these processed insights into the interactive leaderboards, charts, and comparison views for the end-user. An example of the visualized data leaderboard is shown in Figure 3.

This structured architecture forms the foundation for the four core deliverables ODA provides to the community. (1) First, a visualized and interactive data leaderboard offers an intuitive gateway to our findings, allowing users to easily compare dataset performance. (2) Second, we provide a multidimensional scoring framework with its open-source data, detailing over 15 intrinsic properties of each dataset to help researchers understand why certain data is effective. (3) Third, an interactive data lineage platform maps the genealogy of the ecosystem, enabling users to trace the provenance of datasets and dissect the composition of aggregated sources. (4) Finally, a complete and reproducible open-source toolkit is released to ensure full transparency and empower the community to benchmark their own datasets using our standardized methodology.

#### 2.3 Benchmarking Pipeline

The heart of the OpenDataArena platform is a rigorously standardized train-evaluate benchmarking pipeline. The overall principle is straightforward yet powerful: for each post-training dataset being evaluated, we take a common, pre-trained base model and fine-tune it exclusively on that dataset. The resulting model is then subjected to a comprehensive evaluation across a suite of downstream benchmarks. The aggregated performance of the fine-tuned model serves as a direct proxy for the value and utility of the dataset it was trained on. This entire process is automated to ensure consistency and scalability.

##### 2.3.1 Dataset and Benchmark Selection

The selection of both the training datasets to be evaluated and the benchmark datasets for evaluation is critical to the utility of the platform. Details can be found in: https://opendataarena.github.io/ configurations.html.

Training Dataset Selection. Our goal is to provide broad coverage of the publicly available datasets used by the research community. To achieve this, we established a set of clear selection rules. Our process prioritizes datasets with demonstrated community impact (e.g., a minimum number of likes or downloads on Hugging Face), recency (after 2023), and direct suitability for SFT. To maintain computational feasibility, we also impose a practical size limitation on each dataset. Furthermore, all selected datasets undergo a quality assurance process that includes content review for safety and standardization of data formats. We ensured that our collection spans key domains such as general dialog, math, and coding. Recognizing that many valuable datasets are not confined to a single category, we also explicitly include numerous mixed-domain datasets to reflect the complexity of real-world data. In the current version of ODA, we have incorporated over 120 prominent SFT training datasets, including well-known examples like OpenThoughts3 [21], LIMO [61], and Tulu3-SFT [28]. The scale of these datasets ranges from a few thousand to over one hundred thousand samples, culminating in a collection of over 40 million data points for analysis.

Benchmark Selection. To ensure a holistic assessment of model capabilities, we curated a suite of over 22 benchmarks spanning multiple critical domains. Our selection provides a multi-faceted view of a dataset’s impact.

- • General Capabilities: To assess broad language understanding, instruction following, and factual knowledge, we use a diverse set of benchmarks including DROP [15], IFEval [68], AGIEval [66], and MMLU-PRO [56].
- • Math: To specifically test mathematical and logical reasoning abilities, we employ OmniMATH [17], OlympiadBenchMath [22], GSM8K [12], MATH-500 [23], AIME 2024, AIME 2025 [1], HMMT Feb 2025 [5], BRUMO 2025 [5], CMIMC 2025 [5].

- • Code: Competency in programming and code generation is evaluated using a suite of standard benchmarks: HumanEval, HumanEval+ [9], MBPP, and LiveCodeBench(v5) [25].
- • Reasoning: To measure complex and multi-step reasoning skills, we include ARC c [10], BBH [48], KOR-Bench [40], CaLM, and GPQA diamond [47].

##### 2.3.2 Training and Evaluation Setup

To ensure the fairness and credibility of our data value assessment, we design and implement a rigorously standardized training and evaluation pipeline. Every aspect of this process, from software frameworks to model configurations, is held constant across all experiments, ensuring that the dataset itself is the sole variable under investigation.

Training Setting. To ensure fair training and minimize the impact of training configurations on evaluation results, we carefully reference a wide range of existing literature to standardize the hyperparameter settings across different models. This approach helps eliminate performance bias caused by inconsistent training setups. We utilize LLaMA-Factory [65], a popular and highly efficient open-source fine-tuning framework. Our current release focuses on two state-of-the-art base models: Llama3.18B [19], Qwen2.5-7B [60] and Qwen3-8B [59]. A stringent and consistent set of hyperparameters, including learning rate, optimizer, and LoRA configurations, is applied to every training run. This strict consistency is paramount for attributing any performance variations solely to the data. The detailed training configurations are available for reference in Appendix A.

Evaluation Setting. Following training, each fine-tuned model is assessed using OpenCompass [14], a comprehensive and standardized evaluation framework. Our evaluation employs methods tailored to each benchmark, utilizing both zero-shot and few-shot prompting strategies depending on the task’s nature. The evaluation metrics are also specific to each benchmark, ranging from straightforward accuracy and pass@1 for definite-answer tasks to the average score across multiple sub-tasks. Importantly, evaluation strictly follows official protocols or widely adopted tools in the community, such as math-eval-harness1 and lm-evaluation-harness2, to ensure consistency and comparability with existing benchmarks. A critical aspect is the rigorous answer verification of model outputs. Therefore, for code-related benchmarks, we adopt the default evaluation logic provided by the original tools. While for non-code benchmarks, we use powerful large models (e.g., xVerify3, Omni-Judge4) to extract and evaluate answers, enhancing the robustness and credibility of our final results. This end-to-end, standardized pipeline enables large-scale, reliable analysis. To date, we have executed over 600 independent training runs and more than 10,000 evaluation runs, forming the empirical basis of the OpenDataArena leaderboard.

#### 2.4 Data Scoring

While the leaderboard provides a clear ranking based on downstream performance, it does not explain why certain datasets are more effective than others. To address this, ODA incorporates a multi-dimensional data scoring system designed to provide a rich, diagnostic “fingerprint” for each dataset. This system moves beyond a single performance score to evaluate each dataset along tens of distinct axes (continuously expanding). A key aspect of our approach is the separate evaluation of the instruction itself (Question, or Q) and the complete instruction-response pair (Question-Answer, or Q&A). The quality of an instruction alone—for instance, the inherent difficulty of a mathematical problem—is a valuable signal of a dataset’s potential. Evaluating the QA pair, on the other hand, allows us to assess the quality, correctness, and helpfulness of the provided response. To achieve this comprehensive analysis, our scoring metrics are organized into three distinct methodological categories as follows.

- • Model-based Evaluation. This approach utilizes specialized, pre-trained models to automatically assess specific, complex attributes of the data. These models can, for example, predict the difficulty of an instruction or the probability that a question requires multi-step reasoning (thinking probability).
- • LLM-as-Judge. We leverage powerful LLMs (e.g., GPT-4) to provide human-like qualitative assessments. This method is particularly effective for evaluating subjective axes that are difficult to capture with traditional models, such as the clarity, coherence, and overall meaningfulness of a Q&A pair.

- 1https://github.com/ZubinGou/math-evaluation-harness
- 2https://github.com/EleutherAI/lm-evaluation-harness
- 3https://github.com/IAAR-Shanghai/xVerify
- 4https://huggingface.co/KbsdJames/Omni-Judge

• Heuristic Rules. This category includes straightforward, objective calculations based on welldefined rules. These metrics measure fundamental properties of the text, such as the token count or character length of the responses, providing a baseline for quantitative analysis.

From the perspective of measurable attributes, ODA constructs a holistic quality profile through a stratified suite of metrics. Several metrics focus on assessing difficulty and complexity from various angles (e.g., Deita Complexity [38], Instruction Following Difficulty [30], LLM-judged Complexity). Another group evaluates correctness and quality, using reward models or judge models to assess the factual accuracy and utility of the responses (Deita Quality, Fail Rate, Correctness). Further dimensions scrutinize the linguistic and structural properties of the text, such as its Clarity, Coherence, and Relevance. This multi-faceted approach ensures a comprehensive understanding of each dataset’s strengths and weaknesses. The detailed definitions for each specific metric are provided in Appendix C.

### 3 Dataset Landscape and Analysis

Having detailed the architecture and methodology of the OpenDataArena platform, we now turn focus to the subject of our investigation: the post-training datasets themselves. This section presents a deep, intrinsic analysis of the data landscape, independent of downstream model performance. Our analysis is twofold. First, we conduct a macro-level, descriptive study of the ecosystem’s characteristics, examining its evolution, composition, and statistical properties. Second, we introduce the novel concept of “Data Lineage” to perform a micro-level analysis, mapping the intricate relationships, derivations, and redundancies between individual datasets. This section provides a factual baseline of the data’s nature, paving the way for the performance-based analysis in Section 4.

#### 3.1 Macro-Analysis of the Data Landscape.

We characterize the overall landscape of the 120 datasets that we collected within OpenDataArena, aiming to uncover broad trends and patterns that define the current state of the post-training data ecosystem. We begin by examining the temporal evolution of dataset creation before moving to the distribution of data domains and other statistical properties.

Temporal Evolution of the Data Ecosystem. Figure 4 illustrates the exponential evolution of the field (Q1 2023 – Q3 2025). We observe a clear two-phase acceleration: initial exploration shifted to mass production around early 2024, stabilizing at a peak release rate of 18–19 datasets per quarter. Correspondingly, the total available data volume skyrocketed from near-zero to over 40 million samples, with the growth curve steepening significantly after 2024Q3, underscoring the massive scale-up in data-centric investments.

Domain Distribution. As illustrated in Figure 5a, the data landscape is heavily skewed towards specialized domains. Math (34.3%) and Code (30.6%) are the dominant categories, together comprising nearly two-thirds of the ecosystem. This distribution highlights a concentrated research effort on enhancing quantitative and logical reasoning, surpassing General conversational datasets (20.8%) and the established Science domain (14.4%). In essence, the field has matured from general instruction tuning to a focus on technical proficiency.

Evolutionary Trends. Figure 5b reveals the dynamic shift in these priorities over time. The landscape has evolved from an initially balanced distribution in early 2023 to distinct phases of specialization: a surge in Code data in late 2023, followed by a pivot to Math in 2024 (peaking at 46.2% in Q2). Most recently, the Science domain has seen steady growth, rising from 6.2% to 21.6% by mid-2025. This progression underscores a trajectory towards increasingly complex, knowledge-intensive data creation.

###### Evolution of Post-Training Data Ecosystem: Scale & Volume

Newly Released Datasets (Count) 19 19

Newly Released Datasets (Count)

18 18

Cumulative Number of Samples

CumulativeNumberofSamples(Millions)

17.5

40.0M

###### NumberofNewDatasetsperQuarter

15.0

13

30.0M

12.5

10.0

20.0M

8

7.5

7

7

5

5

5.0

10.0M

2.5

1

0.0M

0.0

2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3 2024Q4 2025Q1 2025Q2 2025Q3

Quarter

- Figure 4: Evolution of the post-training dataset landscape (Q1 2023 – Q3 2025), the chart tracks the quarterly rate of newly released datasets alongside the cumulative volume of training samples.

General

20.8%

Code

30.6%

Math

34.3%

Science

14.4%

120 Datasets

Distribution of Post-Training Datasets by Domain

(a) Distribution by Domain.

| |31.2%<br><br>16.7%<br><br>10.0%<br><br>19.0%<br><br>15.4% 14.3%<br><br>18.5%<br><br>23.7%<br><br>20.0% 21.6%<br><br>31.2%<br><br>16.7%<br><br>20.0%<br><br>33.3%<br><br>46.2%<br><br>42.9% 37.0%<br><br>39.5%<br><br>32.5%<br><br>32.4%<br><br>31.2%<br><br>66.7%<br><br>60.0%<br><br>42.9% 23.1%<br><br>42.9%<br><br>33.3% 18.4%<br><br>27.5% 24.3%<br><br>6.2%<br><br>10.0%<br><br>15.4%<br><br>11.1%<br><br>18.4% 20.0% 21.6%| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3 2024Q4 2025Q1 2025Q2 2025Q3

Quarter

0

20

40

60

80

100

PercentageofTagsperQuarter(%)

Evolution of Data Focus: Domain Distribution Over Time

Domain

General

Math Code Science

(b) Temporal Evolution.

- Figure 5: Statistical overview of the ODA dataset repository. (a) The breakdown of datasets across diverse domains (e.g., Math, Code, Science) and (b) the temporal trends of dataset releases, illustrating the evolution of the data landscape.

#### 3.2 Data Lineage: Uncovering Relationships and Redundancy

The proliferation of post-training datasets has driven rapid progress in general instruction following, code generation, and mathematical reasoning. However, existing analysis and evaluation practices typically treat these datasets as independent objects [46, 32, 18], obscuring their ancestral dependencies and extensive content overlap. In practice, most new open-source datasets are constructed by recursively repurposing existing resources through semantic reformulation [63, 41], compositesample synthesis via multi-sample fusion [45, 43], rejection sampling [52, 34], or knowledge distillation from stronger teacher models [27, 20, 44, 35]. As a result, truly novel data generated entirely from scratch remains scarce [11, 33], and contemporary post-training corpora instead form a dense, highly interdependent web of data lineage.

To address this complexity, we introduce data lineage as a unifying perspective on the compositional evolution of post-training datasets. By explicitly mapping derivation pathways that connect each dataset to its upstream sources via well-defined transformation operations, we obtain a structured representation of how data is constructed and reused. This representation provides a principled basis for downstream tasks such as redundancy detection, provenance tracking, and benchmark contamination analysis, and offers concrete guidance for designing future datasets in a more systematic and reliable way.

##### 3.2.1 Automated Data Lineage Framework Design

We present a systematic framework for automating the tracing of dataset lineage. Formally, we model data lineage as a directed graph G = (V, E), where nodes V correspond to individual post-training datasets and directed edges E encode inheritance relations between them. An edge (vi, vj) ∈ E indicates that vj is (partially) derived from vi through some transformation process.

Challenges in Lineage Tracing. Tracing lineage at scale is non-trivial due to the informal and heterogeneous nature of dataset documentation. Provenance information is typically scattered across academic papers, Hugging Face repositories, and technical blogs, and is rarely expressed in a standardized form. Moreover, the dependency structure can be extensive and deeply nested: a single dataset may cite dozens of upstream sources, and recursively expanding these references quickly leads to a combinatorial explosion in the search space. To address these challenges, we design a Multi-Agent Collaborative Framework that centers on multi-source evidence fusion and semantic reasoning over noisy, incomplete documentation.

Multi-Agent Collaborative Framework. Our Multi-Agent Collaborative Framework applies a depthfirst search (DFS) multi-agent pipeline to a queue of pending datasets, recursively tracing their upstream dependencies and incrementally constructing the lineage graph. For each pending dataset, the pipeline consists of four steps:

- 1. Candidate validation. We initialize the framework by enqueuing all candidate datasets into a centralized processing queue. For each candidate dataset, we query the Hugging Face API to verify its existence and retrieve its release timestamp. To focus on data shaped by the scale and paradigm shifts of modern LLMs, particularly following the release of GPT-3 [7], we retain only datasets hosted on Hugging Face with release timestamps ≥ 2020. Only these validated candidate datasets proceed to subsequent fine-grained lineage analysis.
- 2. Multi-source Information Retrieval. For each validated candidate dataset, we issue a request to retrieve its README. A Retrieval Agent parses the README to discover external resources, including GitHub repositories, technical blogs, and papers, and dispatches specialized Resource Agents to fetch the associated content. Specifically, for GitHub repositories and blogs, the agents issue requests to retrieve the corresponding web content; for papers, the agents query the arXiv

- API using the paper URL or title. To ensure context quality, we apply an intelligent pruning mechanism to eliminate noise, such as code blocks in READMEs, HTML tags in blogs, and irrelevant sections in papers. Finally, the curated materials are consolidated into a unified resource context to support lineage analysis.
- 3. Semantic Source Inference and Extraction. Building on this resource context, we deploy a pool of Source-Tracing Agents to identify the source data used in constructing the candidate dataset. The agents are explicitly instructed to distinguish actual source data from incidental mentions, for instance, by excluding evaluation benchmarks, comparison baselines, and non-integrated references. The extraction results are formalized as structured JSON records ⟨Source,Relationship,Confidence,Evidence⟩, with Source identifying the constituent source data; Relationship categorizing how the source data contributes (e.g., fusion or distillation); Confidence reflecting the identification reliability, computed from both the strength of textual support and the credibility of the information source; and Evidence providing the textual span supporting the claim. These records are aggregated to instantiate directed edges (Source,Target,meta info) in the lineage graph, where meta info encapsulates the extracted attributes.

- 4. Aggregation, disambiguation, and verification. The raw edges extracted from diverse sources are first pooled for consolidation. An Aggregation Agent processes these records to perform deduplication and resolve naming ambiguities—specifically by canonicalizing informal dataset names to a unique Hugging Face ID. This agent enforces cross-source consistency to filter out potential hallucinations, effectively discarding relations that are low-confidence, contradictory, or unsupported by verifiable evidence. This final verification step ensures both the factual accuracy and structural coherence of the constructed lineage graph.

Graph Construction and Human-in-the-Loop Verification. Lineage construction proceeds as a bottom-up, DFS-style recursive traversal: starting from modern post-training datasets, the system incrementally traces their upstream ancestors, completing the lineage tree for one target before moving to the next candidate in the queue. This strategy enables global modeling of derivation paths while controlling search depth and branching. To handle semantic ambiguities or conflicting multi-source evidence, we incorporate a human-in-the-loop mechanism: edges whose confidence falls below a predefined threshold are automatically flagged for expert review and cross-checking. This automationfirst, human-backed protocol preserves scalability while improving the logical consistency and factual reliability of the final lineage graph.

##### 3.2.2 Landscape-Level Redundancy Analysis

With the automated lineage framework in place, we quantitatively characterize the post-training data ecosystem. We apply our framework to top-tier datasets from ODA leaderboard in four domains: Math (Top-30), Code (Top-30), General (Top-25), and Science (Top-10). Focusing on these communityrecognized “high-quality” datasets, we analyze per-dataset and global lineage graphs and obtain four key observations: (i) the ecosystem is topologically dense and homogenized, (ii) domains follow distinct evolutionary patterns, (iii) a few cross-domain super-aggregators act as central hubs, and (iv) benchmark contamination propagates along lineage edges.

Topological Connectivity and Systemic Homogenization. Our framework produces five lineage graphs (four domain-specific and one global) with distinct densities. In Math, 30 root datasets expand to 378 unique nodes and 890 edges; in Code, 30 roots expand to 371 nodes and 882 edges, yielding the highest edge-to-node ratio. The General domain (25 roots) connects to 352 nodes and 818 edges, while Science (10 roots) traces back to 331 nodes and 781 edges. Across all domains, the 70 seed datasets expand into a global lineage graph with 411 unique nodes and 941 edges, i.e., ≈ 2.29 edges per node. To highlight core connectivity, we selected the 8 datasets with the highest degree (most

[Figure 6]

 D a t a   L i n e a g e   -  ™–˜u Figure 6: High-level overview of data lineage relationships, where node size[üQúeö•ô : reflects 2 0 2 5 / 1 2 / 0 7 data download 1 8 : 5 0 : 0 9 count, different colors represent distinct data sub-networks, and darker colors indicate higher-degree nodes with greater importance.

connections) in the network for visualization, as shown in Figure 6—notably, these 8 datasets alone form an extremely extensive relational structure. This high connectivity reveals that a small set of core datasets is repeatedly reused, fused, and reformulated, leading to strong systemic homogenization.

Domain-Specific Evolutionary Patterns. Domains exhibit different structural and evolutionary behaviors:

- • Mathematics (Deep Iteration). The Mathematics domain exhibits the deepest lineage structure, with an average depth of 5.18 and a maximum depth of 11 (achieved by alibaba-pai/OmniThought

-0528, part of whose data lineage relationships are illustrated in Figure 7). This extensive depth reflects a trend where modern math datasets are constructed through iterative distillation and Chain-of-Thought enhancement rather than raw collection. Notably, EleutherAI/hendrycks math is reused 16 times across other datasets, while openai/gsm8k and AI-MO/NuminaMath-CoT [29] each appear 13 times, highlighting their central roles in the mathematical data ecosystem.

- • Code (Algorithm Contest-Derived Aggregation).The Code domain (average depth 4.15) features aggregation centered on algorithm contest resources. A representative SOTA dataset in this domain is microsoft/rStar-Coder [39], whose sources are predominantly from algorithmic competition ecosystems—e.g., deepmind/cod contests, open-r1/codeforces, USACO, IOI, BAAI/TACO and codeparrot/apps. These contest-derived resources (problems, solutions, test cases) are systematically integrated, revealing the domain’s core essence: its fundamental data largely originates from human algorithmic competitions.

- • General (Reasoning-Driven). The General domain (average depth 4.83) reveals a marked dependency on mathematical reasoning datasets. openai/gsm8k and EleutherAI/hendrycks math are central to its construction, participating even more frequently than general-domain resources such as Wikimedia. This pattern underscores a growing trend where mathematical logic serves as a universal scaffolding for enhancing the reasoning alignment of general-purpose

[Figure 7]

###### Figure 7: Data lineage relationships of the first four layers for alibaba-pai/OmniThought-0528 (43 data nodes in total), where blue nodes represent target data, green nodes represent traceable nodes in source data, and orange nodes represent basic datasets.

[Figure 8]

###### Figure 8: Data lineage relationship composition of SynthLabsAI/Big-Math-RL-Verified and agentica-org/DeepCoder-Preview-Dataset, where the former incorporates the benchmark Omni-MATH in its data makeup and the latter incorporates the benchmark LiveCodeBench-v5.

models, prioritizing cognitive process over static knowledge accumulation.

• Science (Scarcity and Cross-Domain Reliance). The Science domain exhibits the shallowest average depth (3.71), highlighting a critical scarcity of specialized, high-quality native base datasets. Consequently, its construction relies heavily on cross-domain sourcing, frequently tracing back to traditional mathematics datasets such as EleutherAI/hendrycks math. This strong dependency not only reflects the intrinsic coupling between scientific and mathematical reasoning but also underscores the current lack of foundational corpora dedicated specifically to scientific domains.

Emergence of Cross-Domain Super-Aggregators. We also observe cross-domain “super-aggregators” that dominate multiple domains. A canonical example is a-m-team/AM-Thinking-v1-Distilled [51]. Although primarily focused on Math and Code tasks, it effectively acts as a broad repository, directly citing 19 primary sources while incorporating 435 source data nodes via recursive tracing. This extreme aggregation correlates with strong downstream performance. On Qwen2.5-7B, it ranks first in Math (avg. 77.4, +37.7 gain) and third in Code (avg. 68.7, +18.5 gain); on Llama3.1-8B, it ranks first in Math (avg. 74.5, +58.5 gain) and second in Code (avg. 64.0, +34.9 gain). These results highlight the pivotal role of compositional breadth and diversity, suggesting a synergistic transfer where the structured logic inherent in code reinforces mathematical reasoning. Crucially, this underscores the necessity of deep lineage tracing to uncover the hidden data complexity driving SOTA performance.

Cascading Propagation of Benchmark Contamination. Finally, our lineage analysis reveals pervasive leakage between training corpora and evaluation benchmarks. The tool identifies concrete cases where standard test benchmarks are directly included in the training data. For example, as illustrated in Figure 8, the lineage of SynthLabsAI/Big-Math-RL-Verified [3] connects directly to the KbsdJames/Omni-MATH [17] benchmark, and agentica-org/DeepCoder-Preview-Dataset explicitly includes PrimeIntellect/LiveCodeBench-v5 [25]. Such leakage threatens evaluation integrity, as improvements may reflect memorization rather than generalization. Even worse, these contaminated datasets often serve as “high-quality” seeds for further derivations, causing leakage to propagate recursively through the lineage graph and leading downstream users to unknowingly inherit contaminated data, thereby undermining leaderboard-style evaluations.

### 4 Findings from Benchmarking

Our extensive benchmarking, encompassing over 120 datasets, fine-tuned on leading models and evaluated across 22 benchmarks, yields a multi-faceted view of the post-training data landscape. The results reveal critical patterns in how different models respond to data, the intricate relationship between dataset characteristics and performance, and crucial trade-offs between data scale and efficiency. This section synthesizes these findings, providing empirically-grounded insights for dataset selection, curation, and future research. To ensure these insights are immediately accessible, we distill the most impactful conclusions from our analysis into the following key takeaways. For the detailed analysis, please find in the following contents.

##### Key Takeaways: Benchmarking & Analysis

- • Stronger Base, Higher Ceiling: Advanced base models (e.g., Qwen3) generally provide a higher performance floor and are more robust to data noise. In contrast, weaker models are prone to performance regression in specialized domains like Math when trained on suboptimal data.
- • The Critical Mass of Efficiency: While extreme efficiency (e.g., tiny datasets) is possible, it often hits a performance ceiling. The optimal strategy is High-Density Volume—curated datasets that are large enough to ensure robustness but clean enough to avoid redundancy.
- • The Renaissance of Reasoning Data: Temporal analysis reveals a massive surge in Math data quality starting in 2024, driven by synthetic Chain-of-Thought techniques. Conversely, the Code domain remains volatile, and General capabilities show signs of saturation.
- • Response Quality Dictates Value: Correlation analysis confirms that prompt complexity alone is a poor predictor of value. Response Length (reflecting detailed reasoning) is the strongest positive signal for data quality, particularly in Math and Science.
- • The Uniqueness of Code: The Code domain exhibits distinct correlation patterns (e.g., a preference for conciseness over verbosity), suggesting that coding data requires specialized evaluation criteria different from general reasoning tasks.

#### 4.1 Performance Analysis

To understand the efficacy of post-training datasets, we conduct a multi-faceted analysis covering absolute performance distributions, relative performance gains over base models, temporal quality trends, and the consistency of dataset rankings across models. For a meaningful analysis, we restrict the evaluation to the top-performing datasets (union of top-50 ranked datasets) within each domain.

##### 4.1.1 Absolute Performance

We first examine the absolute performance distribution of datasets when fine-tuned on three distinct base models: Llama3.1-8B, Qwen2.5-7B, and Qwen3-8B. Figure 9a illustrates the score distributions across five domains.

The results reveal a clear hierarchy in base model capability that persists after fine-tuning. Qwen3 consistently achieves the highest median scores across all domains, followed by Qwen2.5, with Llama3.1 generally trailing. This suggests that a stronger base model provides a higher “performance floor” for post-training data. Notably, the Math and Code domains exhibit significant variance (large interquartile ranges and long whiskers), particularly for Llama3.1. This indicates that these specialized domains are highly sensitive to data quality; while high-quality data can yield competitive results, low-quality data can lead to severe performance degradation. In contrast, the General domain shows a more compact distribution, suggesting that performance in general instruction following is less volatile across different datasets.

##### 4.1.2 Relative Performance Delta

While absolute scores reflect the final model capability, the relative performance delta (∆)—defined as the score difference between the fine-tuned model and its base version—provides a lens into the “growth space” that different datasets can unlock. It isolates the value added by the dataset itself. Figure 9b presents these deltas across different domains.

A key observation from the results is the significant variance in learnability across different domains, similar to the absolute values, particularly in Math and Code. These specialized domains exhibit a

Model Llama3.1

| |
|---|

- Qwen2.5

| |
|---|

- Qwen3

70

60

50

40

Score

30

20

10

0

General Math Code Science Global Domain

(a) Absolute Performance.

50

Model Llama3.1

| |
|---|

- Qwen2.5

| |
|---|

- Qwen3

| |
|---|

40

30

PerformanceDelta(vsBase)

20

10

0

10

20

30

General Math Code Science Global Domain

(b) Performance Delta (∆).

- Figure 9: Comparative analysis of model performance across domains. (a) Absolute Performance: Distribution of final scores for the same subset of top-tier datasets. (b) Performance Delta: Distribution of net gains relative to the base model, calculated using the union of top-50 performing datasets for each domain.

much wider range of performance deltas compared to the General domain, indicating that they are highly sensitive to data quality. For instance, in the Math domain, the box plots span a broad range of values, suggesting that while high-quality datasets can trigger substantial capability jumps, there is also a risk that less suitable datasets fail to activate the model’s reasoning potential. This high variance underscores that dataset selection is far more critical for specialized reasoning tasks than for general conversation, where the performance lift is relatively uniform.

Analyzing the relationship between base model strength and relative gains offers further insight, particularly when comparing Llama3.1 and the Qwen series. In domains like Math, Llama3.1 exhibits a notable median performance delta that is often higher than that of Qwen2.5. However, referring back to the absolute scores, Qwen2.5 consistently maintains a higher final performance. This suggests that while weaker base models like Llama3.1 have a larger “growth space” and can show dramatic relative improvements from SFT, they still may not surpass a stronger base model. Conversely, Qwen3 demonstrates a unique characteristic: it achieves both high absolute scores and high positive deltas. This indicates that a sufficiently advanced base model does not merely rely on its pre-training; it also possesses a superior capacity to absorb and utilize post-training data to further extend its capabilities.

##### 4.1.3 Temporal Evolution of Dataset Quality

To evaluate the community’s progress in data construction, we analyzed the performance of datasets released between 2023Q2 and 2025Q3. By fixing Qwen2.5 as the base model for all experiments, we isolate the intrinsic quality of the data from model advancements. The aggregate results (Figure 10) reveal distinct evolutionary trajectories across different domains.

The most striking trend is observed in the Math domain (orange line). Initially, the lowest-performing category with scores hovering around 35 in 2023, starting from 2024Q2, dataset quality surged dramatically, with scores climbing steeply to reach approximately 56 by 2025Q3. This trajectory mirrors

60

55

50

Score

45

40

Global

General

Math Code Science

35

2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3 2024Q4 2025Q1 2025Q2 2025Q3

Quarter

- Figure 10: Temporal evolution of post-training dataset quality across domains, evaluated using Qwen2.5-7B as the base model. The chart shows the performance of datasets released in each quarter from 2023Q2 to 2025Q3.

the research community’s intense recent focus on mathematical reasoning and synthetic data generation, transforming what was once a weakness into a leading capability. In stark contrast, the Code domain (green line) exhibits significant volatility. It remained stagnant throughout the development; this instability suggests that while the volume of coding data has likely increased, quality control is inconsistent; the introduction of noisy or low-quality coding datasets during certain periods severely impacted the fine-tuning potential, even for a strong base model like Qwen2.5. Meanwhile, the General domain (blue line) maintains a consistently high baseline. This indicates that the methodologies for general instruction following matured earlier, leaving less headroom for explosive growth compared to specialized reasoning tasks. Overall, the Global trend (black dashed line) shows a gradual upward movement, driven largely by the breakthroughs in specialized domains like Math rather than uniform improvements across the board.

##### 4.1.4 Consistency Across Model Generations

Finally, we investigate whether a “good” dataset for one model remains “good” for another one by analyzing the rankings of the same datasets between two models. Table 1 shows the Spearman Rank Correlation of dataset rankings between Qwen2.5 and Qwen3.

The results reveal a stark contrast between general and specialized domains. The General domain exhibits a negative correlation (0.323), suggesting a saturation effect. As Qwen3 is a significantly more capable model, it has likely already absorbed the common instruction-following patterns found in these datasets during its pretraining or alignment stages, rendering standard SFT less effective or even noisy compared to its predecessor. Conversely, Code and Science show weak positive correlations (0.28 - 0.35). This indicates that while the base model’s capabilities are steadily improving, it has not yet fully mastered the specialized knowledge within these domains. Consequently, these datasets retain their value for finetuning, although the specific datasets that yield the highest marginal gains shift as the base model’s competence evolves. Notably, Math (0.902) remains highly consistent, implying that high-quality mathematical reasoning data is objectively

Table 1: Spearman rank correlation between Qwen2.5 and Qwen3 across domains.

##### Domain Correlation

Math 0.902 Science 0.354 Code 0.281 General -0.323 Overall 0.440

beneficial regardless of the model version.

#### 4.2 Data Efficiency Analysis

In this section, we evaluate the cost-effectiveness of post-training datasets. We introduce a metric termed Data Efficiency, defined as the ratio of the model’s performance gain (relative to the base model) to the size of the dataset used for fine-tuning:

SiSFT,M − SBaseM |Di|

, (1)

DEi,M =

where |Di| represents the data size of dataset Di, SiSFT,M and SBaseM are the performances of the base model M and the model after SFT on dataset Di over the base model M. This metric quantifies the “value density” of a dataset—specifically, how much performance improvement is gained per unit of data. By plotting Data Efficiency against the performance score in each domain (Figure 11), we identify distinct clusters of datasets that offer different trade-offs between computational costs and model quality.

##### 4.2.1 Efficiency Landscape Across Domains

The plotted results for each domain, which visualize the relationship between efficiency and final capability, are shown in Figure 11a to Figure 11d. In the General, Math, and Code domains, the efficiency landscape exhibits a consistent hierarchy. The dataset distributions for the Qwen series (Qwen2.5 and Qwen3) are generally situated higher on the performance axis compared to Llama3.1, maintaining a relatively stable correlation. This suggests that while the base models have different starting capabilities, their response to high-quality data in these established domains is predictable: better data yields better results across model families.

However, the Science domain presents a starkly different and more chaotic landscape. The distribution of data points here is highly scattered, with no clear stratification between model families. For instance, datasets that are highly efficient for Qwen often fail to produce similar gains for Llama, and vice versa. This uneven distribution indicates that base model capabilities in scientific reasoning are highly unbalanced and less mature; unlike math or code, the underlying “science literacy” of these models varies so drastically that the effectiveness of a dataset becomes unpredictable and highly model-dependent.

##### 4.2.2 The Impact of Data Scale on Performance

Building on the efficiency analysis, we further investigate the trade-off between extreme efficiency and absolute peak performance. Our results challenge the notion that “less is more”, revealing that the scale of data is often requisite for stability. We observe that high efficiency does not always equate to the best final outcome. Taking the Math and Code domains as examples, the AM-Thinking [51] dataset series (e.g., AM-Thinking-v1-Distilled) consistently achieves top-tier absolute performance. While its efficiency score is moderate (situated in the middle of the x-axis), its robust data volume ensures reliable, high-level capability transfer.

Conversely, datasets designed for extreme efficiency, such as LIMA [67] and LIMO [61], occupy the far right of the efficiency scale but often lag in final performance. While efficient, they fail to reach the performance ceiling set by larger, high-quality datasets. A critical finding is observed with LIMO in the Math domain: despite its high efficiency, it leads to performance degradation (a score drop) when fine-tuned on the weaker Llama3.1 model. This suggests that while strong base models might generalize from a handful of examples, weaker models require a certain volume of data redundancy to stabilize their learning. Thus, for practical applications, “moderate efficiency with sufficient volume” appears to be a safer strategy than chasing “extreme efficiency” at the cost of robustness.

Llama3.1

Raiden-DeepSeek-R1

- Qwen2.5

- Qwen3

| |
|---|

65

| |
|---|

| |
|---|

Magpie-Pro-10K-GPT4o-mini

| | |
|---|---|

Raiden-DeepSeek-R1

| |
|---|

60

| |
|---|

SYNTHETIC-2-SFT-verified

| |
|---|

o1-journey

| |
|---|

GeneralScore

| |
|---|

| |
|---|

| |
|---|

| |
|---|

LIMA

| |
|---|

55

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

SYNTHETIC-2-SFT-verified

Raiden-DeepSeek-R1

50

45

Magpie-Pro-10K-GPT4o-mini

40

LIMA

0.001 0.0050.0100.020 0.0500.100.20 0.50 1.0 2.0 5.010.0

Data Efficiency (Log Scale)

(a) General Efficiency

Llama3.1

rStar-Coder

| |
|---|

- Qwen2.5

- Qwen3

70

AM-Thinking-v1-Distilled-code

| |
|---|

| |
|---|

AM-Thinking-v1-Distilled-code

| |
|---|

AM-Thinking-v1-Distilled-code

Magpie-Pro-10K-GPT4o-mini

60

rStar-Coder

| |
|---|

| |
|---|

| | | | |
|---|---|---|---|
| | | | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

python_code_instructions_18k_alpaca

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

CodeScore

LIMA

| |
|---|

| |
|---|

| |
|---|

| |
|---|

OmniThought-0528

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

Magpie-Pro-10K-GPT4o-mini

40

OmniThought-0528

LIMA

30

python_code_instructions_18k_alpaca

0.001 0.0050.0100.020 0.050 0.10 0.20 0.50 1.0 2.0 5.0

Data Efficiency (Log Scale)

(c) Code Efficiency

80

AM-Thinking-v1-Distilled-math

Llama3.1

AM-Thinking-v1-Distilled-math

| |
|---|

- Qwen2.5

- Qwen3

AM-Thinking-v1-Distilled-math

| |
|---|

| |
|---|

| |
|---|

70

SYNTHETIC-2-SFT-verified

SYNTHETIC-2-SFT-verified

| |
|---|

| |
|---|

SYNTHETIC-2-SFT-verified

60

| |
|---|

| |
|---|

MegaScience

| |
|---|

MathScore

MegaScience

50

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|
| | | |

LIMO

| |
|---|

| |
|---|

o1-journey

| |
|---|

| |
|---|

40

| |
|---|

MegaScience

30

o1-journey

20

0.001 0.0050.0100.020 0.0500.100.20 0.50 1.0 2.0 5.010.020.0

Data Efficiency (Log Scale)

(b) Math Efficiency

Llama3.1

| |
|---|

- Qwen2.5

- Qwen3

58

SYNTHETIC-2-SFT-verified

SYNTHETIC-2-SFT-verified

tulu3-sft-mixture

| |
|---|

56

OmniThought-0528

OpenThoughts-114k

MegaScience

| |
|---|

54

OpenThoughts3-1.2M

| |
|---|

ScienceScore

| |
|---|

Light-R1-SFTData

52

SCP-116K

Light-R1-SFTData

MegaScience

50

48

Dolphin-flan1m-alpaca-uncensored

hercules-v1.0

| |
|---|

Magpie-Pro-10K-GPT4o-mini

46

Magpie-Pro-10K-GPT4o-mini

44

WebInstruct-CFT

| |
|---|

0.001 0.005 0.010 0.020 0.050 0.10 0.20

Data Efficiency (Log Scale)

(d) Science Efficiency

- Figure 11: Efficiency point analysis across domains. (a)-(d) show the efficiency trends in General, Math, Code, Science domains respectively. For clear visualization, we only plot the datasets with positive data efficiency scores and the x-axis is on log scale.

#### 4.3 Correlation Analysis: Data Scorers vs. Model Performance

We now analyze the relationship between our multi-dimensional data scorers and the final downstream performance of the models. By calculating the Spearman correlation between various data quality metrics (Appendix C) and domain-specific benchmarks (as shown in Figure 13), we aim to identify which features are the most reliable predictors of data value. This analysis provides critical guidance for data selection and synthetic data generation strategies.

##### 4.3.1 The Primacy of Response Length and Quality

The most prominent positive correlation observed across all settings is associated with Response Length (QA A Length). As illustrated in the correlation heatmap, this feature exhibits a notably high correlation coefficient of 0.81 within the Math domain and 0.40 at the global level. This relationship is further substantiated in Figure 12a, which reveals a clear linear trend: datasets with longer average responses consistently correspond to higher model performance. For instance, the OpenThought3 dataset has an average response length exceeding 14k tokens—substantially surpassing other datasets—and accordingly demonstrates superior performance. These observations provide empirical support for the Chain-of-Thought hypothesis in post-training: detailed, step-by-step reasoning sequences are substantially more informative for model learning than concise answers. This effect is particularly pronounced in reasoning-intensive domains such as Math and Science, where verbose explanations that articulate intermediate reasoning steps enable the model to internalize problem-solving procedures rather than merely replicate final answers. Consequently, in data synthesis, promoting verbosity and encouraging explicit derivations emerge as an effective strategy for enhancing training signal quality.

##### 4.3.2 The “Instruction-Response” Dependency (Q vs. QA)

Interestingly, a counterintuitive yet critical pattern emerges when contrasting metrics that assess the instruction alone (Q) with those that evaluate the instruction–response pair (QA). The correlation heatmap indicates that instruction-only metrics frequently exhibit negative or negligible correlations with downstream performance. For example, both Q Clarity and Q Coherence display negative correlations in the General and Math domains (e.g., –0.51 for Q Clarity in Math), suggesting that a “clear” or “well-written” prompt is insufficient on its own. In many cases, simpler prompts may fail to provide the level of complexity necessary to elicit informative reasoning traces from the model. This insight is further supported by the Instruction Complexity visualization in Figure 12b, where the trend line remains nearly flat and exhibits substantial variance. Elevated instruction complexity (Q -

Complexity) does not reliably translate into improved performance when the corresponding response is of low quality. In contrast, metrics that jointly evaluate the instruction and response (QA , e.g., QA Correctness, QA Completeness) consistently show positive correlations, reinforcing the dominant role of response quality.

Taken together, these findings point to a central implication for data curation: the value of a training instance is determined primarily by the response rather than the prompt alone. A complex prompt paired with a weak response can be detrimental, whereas a high-quality response can compensate for only moderately complex instructions.

##### 4.3.3 Domain Divergence: The Unique Case of Code

From Figure 13, we further observe notable cross-domain variation. Although the Math, General, and Science domains exhibit broadly consistent correlation structures—sharing positive signals for metrics such as response length and complexity—the Code domain stands out as a clear outlier. In the heatmap, the Code column frequently displays correlation signs that are reversed relative to the other domains. For instance:

- • QA A Length: While this metric is strongly positive in Math (0.81), it becomes negative in Code (-0.29), implying that verbosity, which benefits reasoning-oriented tasks, is detrimental in programming contexts where concise and efficient solutions are preferable.

- • Q Thinking Prob: This feature is strongly positive in Code (0.54) but strongly negative in Math (-0.69). This contrast suggests that coding tasks benefit from prompts that explicitly encourage multi-step reasoning, whereas the formulation of this metric appears misaligned with the nature of mathematical reasoning.

These divergences indicate that Code data requires a dedicated evaluation framework. Generalpurpose heuristics—such as the assumption that “longer responses yield better training signals”—do not transfer reliably to programming tasks. As a result, domain-specific criteria must be adopted when assessing or selecting data for code-focused post-training.

OpenThoughts3-1_2M

60

55

AM-Thinking-v1-Distilled-math

GlobalScore

50

AM-Thinking-v1-Distilled-code

tulu-3-sft-mixture

Light-R1-SFTData

45

RabotniKuma_Fast-Math-R1-SFT_train

40

R1-Distill-SFT-math

35

0 5000 10000 15000 20000

Avg Response Length

(a) Impact of Response Length.

OpenThoughts3-1_2M

60

55

AM-Thinking-v1-Distilled-math

GlobalScore

50

AM-Thinking-v1-Distilled-code

tulu-3-sft-mixture

Magpie-Reasoning-V1-150K-code

Light-R1-SFTData

45

DeepMath-309K

40

alpaca_gpt4 code_instructions_122k

EpiCoder-func-380k

R1-Distill-SFT-math

35

4 5 6 7 8

Instruction Complexity

(b) Impact of Instruction Complexity.

- Figure 12: Metric correlation analysis based on Qwen2.5 model. (a) A strong positive correlation is observed between average response length and global performance, supporting the density hypothesis. (b) Instruction complexity shows weak correlation with downstream performance, illustrating that problem difficulty alone is insufficient for effective alignment.

### 5 Looking Forward

The release of OpenDataArena marks a significant step toward transparent data evaluation, but the landscape of generative AI is evolving rapidly. To maintain relevance and drive further innovation, we have outlined a comprehensive roadmap for the future, focusing on the following key directions.

- • Embracing multimodality. We are extending the ODA pipeline beyond text to support multimodal datasets, including image-text, catering to the growing demand for MLLM evaluation.
- • Benchmarking alignment data. Recognizing the central importance of safety and preference tuning, we plan to introduce dedicated evaluation tracks for RLHF and DPO-style preference datasets. These tracks will enable systematic assessment of how effectively such data aligns model behavior with human values and normative expectations.
- • Developing efficient evaluation. To democratize access and mitigate computational overhead, we will investigate training-free or training-light data valuation techniques—such as influencefunction-based estimators and core-set selection methods—that approximate data utility without requiring full-scale fine-tuning.

- • Enhancing data scorers. We are extending our multi-dimensional scoring framework to capture more fine-grained characteristics, including long-context dependency, cultural and linguistic diversity, and resilience to adversarial or intentionally perturbed prompts.
- • Expanding to vertical domains. We intend to broaden benchmark coverage to high-stakes, domain-specific areas such as finance, law, and medicine, where data scarcity, domain expertise requirements, and rigorous quality control pose distinctive challenges.
- • Building a shared standard. We seek collaborative efforts from the community and relevant committees to guide the evolution of standardized evaluation metrics. We also warmly invite the global research community to contribute new datasets, benchmarks, and methodological advances to foster an open, shared ecosystem for data-centric model evaluation.

We envision ODA not just as a tool, but as a collaborative ecosystem that transforms data curation and evaluation from an art into a rigorous science.

### 6 Conclusion

This report introduces OpenDataArena (ODA) as a step toward rectifying the imbalance between innovative algorithms and datasets, grounded in the conviction that data must be evaluated with the same rigor, transparency, and scientific discipline that we apply to models. Through the development of ODA, we have addressed a pressing need for a systematic, standardized benchmarking ecosystem. By integrating a unified training–evaluation pipeline with a novel multi-dimensional scoring framework, we have enabled fair and open comparisons across over 120 heterogeneous datasets. Our largescale experiments across multiple domains have transformed abstract notions of “data quality” into measurable, empirical quantities. The findings reveal several insights that challenge prevailing assumptions in the field. We show that data efficiency is not a simple function of scale: carefully curated, information-dense datasets can outperform substantially larger but less structured collections. Our analyses further underscore the pivotal role of response quality, demonstrating that rich reasoning traces are far more predictive of downstream performance than prompt complexity alone. Moreover, by charting the emergent “data lineage” that links diverse datasets, we document the community’s transition from broad instruction tuning to specialized domains such as mathematics and code.

Looking ahead, we envision OpenDataArena as more than a leaderboard—rather, it serves as foundational infrastructure for the next era of data-centric AI. As we extend the platform to multimodal corpora, RLHF preference datasets, and high-stakes vertical domains, we remain committed to openness, reproducibility, and community-driven progress. By equipping researchers with robust tools to analyze, compare, and understand data, we aim to catalyze the evolution of dataset creation from an artisanal practice into a systematic, shared scientific discipline.

### Contribution List

- • Leaderboard Construction: Mengzhang Cai, Xin Gao, Honglin Lin, Zheng Liu, Zhuoshi Pan, Qizhi Pei, Xiaoyang Wang, Zhanping Zhong, Yun Zhu
- • Toolkit Development: Mengzhang Cai, Yu Li, Zhanping Zhong
- • Data Lineage Analysis: Xin Gao, Yu Li, Honglin Lin, Xiaoran Shang
- • Data Scoring System: Xin Gao, Zhuoshi Pan, Qizhi Pei, Mengyuan Sun, Zinan Tang, Xiaoyang Wang, Yun Zhu
- • Project Lead/Correspondence: Lijun Wu, wulijun@pjlab.org.cn
- • Advisor: Conghui He, Dahua Lin

### References

- [1] Aime problems and solutions, 2025. URL https://artofproblemsolving.com/wiki/index.php/ AIMEProblemsandSolutions. Accessed on 2025-12-08.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, et al. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models. arXiv preprint arXiv:2502.17387, 2025.
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [5] Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovic, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, february 2025. URL https://matharena. ai, 8, 2025.
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165.
- [8] Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, and Xiangzhong Fang. Reasoning with omnithought: A large cot dataset with verbosity and cognitive difficulty annotations. arXiv preprint arXiv:2505.10937, 2025.
- [9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. Evaluating large language models trained on code, 2021.
- [10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [13] Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world’s first truly open instructiontuned llm. 2023.
- [14] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.
- [15] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. arXiv preprint arXiv:1903.00161, 2019.
- [16] Run-Ze Fan, Zengzhi Wang, and Pengfei Liu. Megascience: Pushing the frontiers of post-training datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025. URL https://arxiv.org/abs/2507.16812.

- [17] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985, 2024.
- [18] Xin Gao, Qizhi Pei, Zinan Tang, Yu Li, Honglin Lin, Jiang Wu, Lijun Wu, and Conghui He. A strategic coordination framework of small llms matches large llms in data synthesis, 2025. URL https://arxiv.org/ abs/2504.12322.
- [19] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [20] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.
- [21] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.
- [22] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.
- [23] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [24] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [25] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [26] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [27] Bespoke Labs. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation. https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoningdistillation, 2025. Accessed: 2025-01-22.
- [28] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [29] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.
- [30] Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7602–7635, 2024.

- [31] Xingxuan Li, Yao Xiao, Dianwen Ng, Hai Ye, Yue Deng, Xiang Lin, Bin Wang, Zhanfeng Mo, Chong Zhang, Yueyi Zhang, et al. Miromind-m1: An open-source advancement in mathematical reasoning via context-aware multi-stage policy optimization. arXiv preprint arXiv:2507.14683, 2025.
- [32] Yu Li, Zhuoshi Pan, Honglin Lin, Mengyuan Sun, Conghui He, and Lijun Wu. Can one domain help others? a data-centric study on multi-domain reasoning via reinforcement learning, 2025. URL https: //arxiv.org/abs/2507.17512.
- [33] Yu Li, Qizhi Pei, Mengyuan Sun, Honglin Lin, Chenlin Ming, Xin Gao, Jiang Wu, Conghui He, and Lijun Wu. Cipherbank: Exploring the boundary of llm reasoning capabilities through cryptography challenges. arXiv preprint arXiv:2504.19093, 2025.
- [34] Honglin Lin, Zhuoshi Pan, Yu Li, Qizhi Pei, Xin Gao, Mengzhang Cai, Conghui He, and Lijun Wu. Metaladder: Ascending mathematical solution quality via analogical-problem reasoning transfer, 2025. URL https: //arxiv.org/abs/2503.14891.
- [35] Honglin Lin, Qizhi Pei, Xin Gao, Zhuoshi Pan, Yu Li, Juntao Li, Conghui He, and Lijun Wu. Scaling codeassisted chain-of-thoughts and instructions for model reasoning, 2025. URL https://arxiv.org/abs/2510. 04081.
- [36] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [37] Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. arXiv preprint arXiv:2312.15685, 2023.
- [38] Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=BTKAeLqLMw.
- [39] Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, and Mao Yang. rstarcoder: Scaling competitive code reasoning with a large-scale verified dataset. arXiv preprint arXiv:2505.21297, 2025.
- [40] Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, et al. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks. arXiv preprint arXiv:2410.06526, 2024.
- [41] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math, 2024.
- [42] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [43] Zhuoshi Pan, Qizhi Pei, Yu Li, Qiyao Sun, Zinan Tang, H. Vicky Zhao, Conghui He, and Lijun Wu. Rest: Stress testing large reasoning models by asking multiple problems at once, 2025. URL https://arxiv.org/ abs/2507.10541.
- [44] Qizhi Pei, Zhuoshi Pan, Honglin Lin, Xin Gao, Yu Li, Zinan Tang, Conghui He, Rui Yan, and Lijun Wu. Scalediff: Scaling difficult problems for advanced mathematical reasoning. arXiv preprint arXiv:2509.21070, 2025.
- [45] Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, and Rui Yan. Mathfusion: Enhancing mathematical problem-solving of llm through instruction fusion. arXiv preprint arXiv:2503.16212, 2025.
- [46] Alberto S´anchez P´erez, Alaa Boukhary, Paolo Papotti, Luis Castej´on Lozano, and Adam Elwood. An llm-based approach for insight generation in data analysis, 2025. URL https://arxiv.org/abs/2503.11664.
- [47] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

- [48] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.
- [49] Zinan Tang, Xin Gao, Qizhi Pei, Zhuoshi Pan, Mengzhang Cai, Jiang Wu, Conghui He, and Lijun Wu. Middo: Model-informed dynamic data optimization for enhanced llm fine-tuning via closed-loop learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6882–6902, 2025.
- [50] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023.
- [51] Xiaoyu Tian, Yunjie Ji, Haotian Wang, Shuaiting Chen, Sitong Zhao, Yiping Peng, Han Zhao, and Xiangang Li. Not all correct answers are equal: Why your distillation source matters. arXiv preprint arXiv:2505.14464, 2025.
- [52] Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. Advances in Neural Information Processing Systems, 37:7821–7846, 2024.
- [53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [54] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [55] Yaoxiang Wang, Haoling Li, Xin Zhang, Jie Wu, Xiao Liu, Wenxiang Hu, Zhongxin Guo, Yangyu Huang, Ying Xin, Yujiu Yang, et al. Epicoder: Encompassing diversity and complexity in code generation. arXiv preprint arXiv:2501.04694, 2025.
- [56] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.
- [57] Shijie Xia, Yiwei Qin, Xuefeng Li, Yan Ma, Run-Ze Fan, Steffi Chern, Haoyang Zou, Fan Zhou, Xiangkun Hu, Jiahe Jin, et al. Generative ai act ii: Test time scaling drives cognition engineering. arXiv preprint arXiv:2504.13828, 2025.
- [58] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. arXiv preprint arXiv:2406.08464, 2024.
- [59] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [60] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [61] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.
- [62] Hiroshi Yoshihara, Taiki Yamaguchi, and Yuichi Inoue. A practical two-stage recipe for mathematical llms: Maximizing accuracy with sft and efficiency with reinforcement learning. arXiv preprint arXiv:2507.08267, 2025.

- [63] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.
- [64] Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. arXiv preprint arXiv:2402.14658, 2024.
- [65] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand,

2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

- [66] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.
- [67] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36: 55006–55021, 2023.
- [68] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

## Appendix

### A Detailed Benchmarking Settings

This appendix provides the full training and evaluation configurations used in OpenDataArena (ODA). Our goal is to ensure strict reproducibility, fairness across datasets, and consistent model behavior independent of data source.

#### A.1 Training Configurations

All supervised fine-tuning (SFT) experiments are conducted using the LLaMA-Factory framework5 (version 0.9.2). We adopt a fully standardized training pipeline so that differences in downstream performance arise from dataset quality rather than variations in optimization or hyperparameters. Table 2 summarizes the detailed hyperparameters for training.

Table 2: Detailed training hyperparameters for different models and scenarios.

Llama Llama for long CoT Qwen2.5 Qwen2.5 for long CoT Qwen3 for long CoT

GPU 8*A100 8*A100 8*A100 8*A100 8*A100 Base model Meta-Llama3.1-8B Meta-Llama3.1-8B Qwen2.5-7B Qwen2.5-7B Qwen3-8B-Base deepspeed ds z3 config ds z3 config ds z3 config ds z3 config ds z3 config template default default default default default cutoff len 4096 32768 4096 32768 32768 preprocessing num workers 16 16 16 16 16 packing false true false true true per device train batch size 4 2 4 2 2 gradient accumulation steps 4 2 4 2 2 learning rate 2.0e-5 3.0e-5 5.0e-6 5.0e-5 5.0e-5 use liger kernel true true true true true num train epochs 3.0 3.0 3.0 3.0 3.0 lr scheduler type cosine cosine cosine cosine cosine warmup ratio 0.03 0.03 0.1 0.1 0.1

#### A.2 Evaluation Configurations

All evaluations are conducted using OpenCompass6 with version v0.4.2 and vLLM7 inference engine, ensuring high-throughput and deterministic decoding. The tables (Table 3 and Table 4) below detail the specific evaluation settings, including shot samples and metrics, used for each benchmark during the evaluation phase.

Table 3: Inference settings for Llama3.1, Qwen2.5, and Qwen3 models.

##### Llama3.1 Qwen2.5 Qwen3

max-out-len 32768 32768 32768 hf-type chat chat chat inference setting vllm llama3 1 8b instruct vllm qwen2 5 7b instruct vllm qwen3 8b instruct accelerator vllm + cutoff vllm + cutoff vllm + cutoff temperature 0 0 0.6 top-p - - 0.95 top-k - - 20

- 5https://github.com/hiyouga/LLaMA-Factory
- 6https://github.com/open-compass/opencompass
- 7https://github.com/vllm-project/vllm

Table 4: Evaluation benchmark configurations.

Domain Benchmarks Evaluator Shot Metric

DROP xVerify-9B-C 3 shot accuracy IFEval IFEvaluator 0 shot Average accuracy on all IFEval benchmarks AGIEval xVerify-9B-C 5 shot accuracy MMLU-PRO xVerify-9B-C 5 shot Average accuracy on all mmlu-pro benchmarks

General

Omni-MATH Omni-Judge 0 shot accuracy OlympiadBenchMath xVerify-9B-C 0 shot accuracy GSM8K xVerify-9B-C 0 shot accuracy MATH-500 xVerify-9B-C 0 shot accuracy

Math

- AIME 2024 xVerify-9B-C 0 shot Average accuracy of 8 run

- AIME 2025 CompassVerifier-7B 0 shot Average accuracy of 8 run HMMT Feb 2025 CompassVerifier-7B 0 shot Average accuracy of 8 run CMIMC 2025 CompassVerifier-7B 0 shot Average accuracy of 8 run BRUMO 2025 CompassVerifier-7B 0 shot Average accuracy of 8 run

HumanEval HumanEvalEvaluator 0 shot pass@1 HumanEval+ HumanEvalPlusEvaluator 0 shot pass@1 MBPP MBPPEvaluator 3 shot pass@1 LiveCodeBench(v5) LCBCGgenerationEvaluator 0 shot pass@1

Code

ARC c xVerify-9B-C 0 shot accuracy BBH xVerify-9B-C 0 shot accuracy KOR-Bench xVerify-9B-C 0 shot Average accuracy on all kor-bench benchmarks CaLM CaLMEvaluator 0 shot Average accuracy on all calm benchmarks GPQA xVerify-9B-C 0 shot accuracy

Reasoning

### B Comparative Performance Alignment Rankings of Datasets on Qwen2.5 vs. Qwen3 Models

To understand how dataset value generalizes across different models, we fine-tune both Qwen2.5-7B and Qwen3-7B on the same datasets under identical settings, then compare domain-wise rankings in Table 5.

Table 5: Rankings correlation on Qwen2.5 and Qwen3 models

General Math Code Science Global Qwen2.5 Qwen3 Qwen2.5 Qwen3 Qwen2.5 Qwen3 Qwen2.5 Qwen3 Qwen2.5 Qwen3

Dataset Affiliation Year Size

AM-Thinking-Math [51] a-m-team 2025 558k 13 2 1 1 19 9 12 10 3 1 MiroMind-M1 [31] miromind-ai 2025 719k 19 5 3 3 22 11 8 7 17 2 OmniThought0528 [8] alibaba-pai 2025 365k 23 15 2 2 15 6 2 3 1 3 AM-Thinking-Code [51] a-m-team 2025 324k 22 4 9 7 1 1 13 18 7 4 Light-R1-SFT Qiyuan Tech 2025 79k 17 3 5 5 20 12 7 11 12 5 OpenThoughts3 [21] Stanford 2025 114k 9 16 6 6 12 3 18 8 5 6 SYNTHETIC-2-SFT-verified PrimeIntellect 2025 105k 20 9 4 4 18 21 1 2 2 7 MegaScience [16] MegaScience 2025 1.25M 2 11 10 11 6 2 19 5 9 8 QwQ-LongCoT-130K Guijin Son 2025 130k 5 6 8 12 14 10 15 16 13 9 Fast-Math-R1-SFT [62] University of Tokyo 2025 7.9k 10 8 7 9 17 19 21 17 19 10 Tulu3-SFT-mixture [28] Allen AI 2024 939k 8 13 18 18 9 5 3 1 10 11 LIMO-v2 [61] GAIR 2025 800 21 7 19 15 21 13 22 20 23 12 OpenO1-SFT [57] GAIR 2025 77.7k 15 14 15 14 13 15 4 13 8 13 LIMO [61] GAIR 2025 817 7 12 17 16 11 7 5 19 4 14 Magpie-Reasoning-v2 [58] Allen AI 2025 250k 18 10 14 13 16 20 6 21 16 15 Code-Feedback [64] M-A-P 2024 66.4k 16 18 23 22 10 4 20 4 21 16 Raiden-DeepSeek-R1 sequelbox 2025 62.9k 6 1 11 10 23 23 9 6 22 17 hercules-v1 Sebastian Gabarain 2024 463k 14 17 22 19 8 14 16 9 20 18 math-gpt-4o PKRD 2024 200k 1 20 12 17 5 16 11 14 6 19 CodeFeedback-Filtered [64] M-A-P 2024 157k 12 21 20 21 3 18 17 12 18 20 Magpie-Reasoning-v1 [58] Allen AI 2024 150k 4 19 21 23 4 17 10 15 15 21 EpiCoder-func [55] Microsoft 2025 380k 11 23 16 8 2 8 23 23 14 22 tulu-3-sft-personas [28] Allen AI 2024 20k 3 22 13 20 7 22 14 22 11 23

[Figure 9]

###### Figure 13: Quality vs. performance correlation matrix. The heatmap displays Spearman correlations between automated quality metrics and domain performance, highlighting the superior predictive power of Response-side metrics (QA) compared to Instruction-side metrics (Q).

### C Data Scoring Metric Definitions

This appendix details the multi-dimensional metrics used to score each dataset in the OpenDataArena. Each metric is applied either to the instruction only (Q) or the full instruction-response pair (QA).

- C.1 Model-based Evaluation Metrics calculated using specialized scoring models.

- • Deita Complexity (Q): A model-based score that estimates the inherent complexity of an instruction based on its lexical and structural features. Higher scores indicate a more challenging prompt.
- • Thinking Probability (Q): A score from a specialized model that predicts the likelihood that an instruction requires deep, multi-step reasoning (Chain-of-Thought) to be answered correctly.
- • Deita Quality (QA): A reward model score that assesses the overall quality, helpfulness, and correctness of a response in the context of its corresponding instruction.
- • Instruction Following Difficulty (IFD) (QA): A model-based estimation of how difficult it is to generate a response that correctly adheres to all constraints and requirements present in the instruction.
- • Fail Rate (QA): The proportion of responses within a dataset that a verification model flags as incorrect, irrelevant, or failing to answer the instruction.

- C.2 LLM-as-Judge Metrics evaluated by a powerful Large Language Model (e.g., GPT-4).

- • Difficulty (Q): An LLM’s rating (e.g., on a scale of 1-5) of the perceived difficulty of an instruction for a typical advanced LLM.
- • Relevance (QA): An LLM’s assessment of how well the response directly addresses the instruction without including extraneous or irrelevant information.
- • Clarity (Q & QA): An evaluation of the linguistic clarity and lack of ambiguity in the instruction, and separately, in the response.
- • Coherence (Q & QA): An LLM’s judgment of the logical flow and consistency within the instruction and, separately, within the response.
- • Completeness (Q & QA): An assessment of whether the instruction contains all necessary information for a good response, and whether the response fully answers all parts of the instruction.
- • Complexity (Q & QA): An LLM’s holistic judgment on the complexity of the concepts or tasks described in the instruction and addressed in the response.
- • Correctness (Q & QA): An LLM’s assessment of the factual accuracy and logical soundness of the response, given the instruction.
- • Meaningfulness (Q & QA): A score for how purposeful and non-generic the instruction and its corresponding response are.

- C.3 Heuristic Metrics calculated using direct, rule-based methods.

• Length (QA): A simple heuristic that measures the number of tokens or characters in the response portion of the data pair.

### D Visualized Results

We provide key visualizations from the ODA website, illustrating dataset relationships by Data Lineage Analysis (Figure 14) and dataset scoring comparisons (Figure 15).

[Figure 10]

###### Figure 14: Data lineage visualization. An interactive graph showing dataset families, derivation relationships and so on.

[Figure 11]

###### Figure 15: Dataset comparison interface. A visualization comparing multiple datasets across model performance and multi-dimensional quality metrics.

