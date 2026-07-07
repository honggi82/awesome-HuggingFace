# arXiv:2603.12572v3[cs.CL]7May2026

## LMEB: Long-horizon Memory Embedding Benchmark

Xinping Zhao1,2, Xinshuo Hu, Jiaxin Xu1, Danyu Tang1, Xin Zhang1, Mengjia Zhou1, Yan Zhong3, Yao Zhou, Zifei Shan, Meishan Zhang1, Baotian Hu1,2∗, Min Zhang1,2 1Harbin Institute of Technology (Shenzhen); 2Shenzhen Loop Area Institute (SLAI); 3Peking University zhaoxinping@stu.hit.edu.cn, mason.zms@gmail.com hubaotian@hit.edu.cn, zhangmin2021@hit.edu.cn

### Abstract

Memory embeddings are crucial for memory-augmented systems, such as OpenClaw, but their evaluation is underexplored in current text embedding benchmarks, which narrowly focus on traditional passage retrieval and fail to assess models’ ability to handle long-horizon memory retrieval tasks involving fragmented, contextdependent, and temporally distant information. To address this gap, we introduce the Long-horizon Memory Embedding Benchmark (LMEB), a comprehensive framework for evaluating embedding models on complex, long-horizon memory retrieval. LMEB comprises 22 datasets and 193 zero-shot retrieval tasks spanning four memory types: episodic, dialogue, semantic, and procedural. These memory types differ in terms of level of abstraction and temporal dependency, capturing distinct aspects of memory retrieval that reflect the diverse challenges of the real world. We evaluate 15 widely used embedding models, ranging from hundreds of millions to ten billion parameters. The results reveal that (1) LMEB provides a reasonable level of difficulty; (2) Larger models do not always perform better; (3) LMEB and MTEB measure orthogonal capabilities. This suggests that the field has yet to converge on a universal model capable of excelling across all memory retrieval tasks, and that strong performance on traditional passage retrieval does not necessarily transfer to long-horizon memory retrieval. LMEB provides a standardized and reproducible framework that fills a key gap in memory embedding evaluation and supports future advances in long-term, context-dependent retrieval. LMEB is available at https://kalm-embedding.github.io/LMEB.github.io/.

### 1 Introduction

Memory embeddings are foundational to a wide range of advanced applications, including agentic systems [OpenClaw Contributors, 2026, Zheng et al., 2025, Fang et al., 2025, Song et al., 2024] and evolving environments [Cao et al., 2025, Ouyang et al., 2025, Chen et al., 2025b]. These memory-augmented systems [Du et al., 2025] require sophisticated mechanisms to store, retrieve, update, and reason over vast amounts of memories, with retrieval being central to their effectiveness [Roediger III and Abel, 2022]. However, despite their growing importance, the evaluation of memory embeddings remains underexplored, especially for long-horizon, context-rich retrieval tasks. Current text embedding benchmarks mainly focus on traditional passage retrieval [Thakur et al., 2021, Muennighoff et al., 2023, Xiao et al., 2024, Enevoldsen et al., 2025], and therefore do not adequately capture the challenges of memory retrieval. Unlike passage retrieval over well-organized documents, long-horizon memory retrieval often requires recalling fragmented, context-dependent, and temporally distant information [Wu et al., 2025, Huet et al., 2025, Kohar and Krishnan, 2025]. As a result, the lack of a comprehensive evaluation protocol for long-horizon memory retrieval leaves a significant gap in understanding how embedding models perform in memory-intensive scenarios.

∗Corresponding Author

To bridge this gap, we introduce the Long-horizon Memory Embedding Benchmark (LMEB), a unified framework for evaluating embedding models on complex, long-horizon memory retrieval tasks. Building on the evaluation standards established for text embeddings, such as MTEB [Muennighoff et al., 2023], LMEB extends this evaluation protocol to memory retrieval tasks. LMEB comprises four memory types: (i) Episodic, (ii) Dialogue, (iii) Semantic, and (iv) Procedural Memory [Du et al., 2025]. Each of these memory types captures distinct aspects of memory retrieval, reflecting the varying demands of real-world scenarios. Specifically, we organize LMEB around these memory types to ensure a comprehensive evaluation of memory retrieval for embedding models:

- • Episodic Memory involves the retrieval of past events linked to temporal cues, entities, contents, and spatial contexts [Fountas et al., 2024, Pink et al., 2025]. The ability to effectively retrieve and utilize episodic memories is critical for enhancing adaptability, decision-making, and temporal reasoning in complex, real-world tasks [Miao et al., 2024].
- • Dialogue Memory focuses on maintaining context across multi-turn interactions, enabling systems to recall previous dialogue turns and user preferences [Wu et al., 2025, Maharana et al., 2024]. This facilitates coherent conversations and improves the system’s ability to adapt and provide personalized responses over time [Li et al., 2025, Du et al., 2024].
- • Semantic Memory involves recalling general knowledge and facts about the world, independent of time or specific context [Tulving et al., 1972]. Unlike episodic memory, semantic memory is stable, generalizable, and not tied to specific events. It forms the foundation for memory-augmented reasoning and adaptive knowledge utilization [Zhou et al., 2025].
- • Procedural Memory focuses on the retrieval of learned skills and action sequences, which are essential for tasks that require problem-solving and multi-step reasoning [Fang et al., 2025, Ouyang et al., 2025]. It is critical for automating and generalizing task-oriented experiences, especially in agentic systems and reinforcement learning systems [Lumer et al., 2025, Yu et al., 2025].

LMEB is designed to clarify how embedding models perform across a spectrum of long-horizon memory retrieval demands and to support the development of more capable memory embeddings. We evaluate 15 widely used embedding models on LMEB, ranging from several hundred million to 10 billion parameters. To this end, LMEB consolidates a diverse set of memory retrieval datasets and evaluation settings into a single, standardized protocol, comprising 22 datasets across 4 memory types and 193 retrieval tasks. We provide an open-source evaluation framework that enables the evaluation of new embedding models and the integration of new datasets with minimal effort, together with a public leaderboard to facilitate reproducible comparisons and future progress. We evaluate 15 widely used embedding models on LMEB, ranging from several hundred million to 10 billion parameters. The results reveal several findings: (1) LMEB provides a reasonable level of difficulty. The top model achieved a Mean (Dataset) score of 61.41 on N@10, indicating that LMEB offers a meaningful challenge for evaluating memory retrieval. (2) Larger models do not always perform better. In some cases, larger models underperform smaller models, highlighting the significance of model architecture and task adaptability. (3) LMEB and MTEB are orthogonal. With Pearson and Spearman coefficients close to 0, LMEB focuses on long-horizon memory retrieval, while MTEB evaluates passage retrieval, indicating that they assess complementary capabilities. Overall, LMEB serves as both a standardized benchmark for long-horizon memory retrieval and a diagnostic tool for developing more reliable memory embeddings in practical, memory-augmented systems.

### 2 The LMEB Benchmark

#### 2.1 LMEB Overview and Taxonomy

The Long-horizon Memory Embedding Benchmark (LMEB) is a comprehensive evaluation benchmark for embedding models on long-horizon memory retrieval tasks Figure 1 and Table 1 summDBLP:conf/iclr/HuetB025ries, memory specificities, and dataset statistics in LMEB. Unlike existing text embedding benchmarks [Thakur et al., 2021, Muennighoff et al., 2023, Xiao et al., 2024, Enevoldsen et al., 2025] mainly focusing on passage retrieval, LMEB is tailored to assess scenarios involving recalling fragmented, context-dependent, and temporally distant memory information, thereby bridging a significant flaw in current embedding benchmarks.

Design Principles. In line with MTEB Muennighoff et al. [2023], LMEB is guided by four key principles to ensure effective evaluation of embedding models on long-term memory retrieval: (1)

[Figure 1]

Figure 1: Overview of LMEB. Tables 5, 6, 7, and 8 present examples of query–relevant document pairs for each dataset. Tables 10, 11, 12, and 13 summarize task types and example abilities assessed.

Generalization: LMEB emphasizes zero-shot evaluation, assessing models based on previously learned embedding capabilities without task-specific fine-tuning. (2) Usability: LMEB supports seamless extension, requiring minimal code changes to integrate new models and simple configuration files to add new datasets. (3) Diversity: LMEB covers a broad range of memory types and tasks, including episodic, dialogue, semantic, and procedural memory, using both AI-generated and humanannotated datasets. (4) Difficulty: LMEB includes tasks with varying levels of difficulty, considering factors such as granularity, corpus size, relevant documents per query, and query/document lengths.

Dataset Scope and Diversity. In total, LMEB includes 22 English zero-shot evaluation datasets spanning 4 memory types and 193 retrieval tasks. Table 1 summarizes their detailed statistics. LMEB covers four memory types: (i) Episodic, (ii) Dialogue, (iii) Semantic, and (iv) Procedural Memory [Du et al., 2025]. The datasets vary substantially in task diversity and difficulty, including differences in Granularity, Querysrc, Corpussrc, #Corpus, Avg. D / Q, and Avg. Word Lengths. For instance, granularity ranges from event-level retrieval in episodic memory to turn-, round-, or session-level retrieval in dialogue memory. Semantic memory typically involves sentence- or paragraph-level retrieval, whereas procedural memory ranges from item-level to trajectory-level retrieval. This diversity ensures that LMEB evaluates models across various aspects of memory retrieval. The sources of queries and corpora also vary substantially across tasks. Many episodic, dialogue, and procedural tasks combine AI-generated and human-annotated data, ensuring that models are tested on both scenarios. In contrast, semantic tasks rely primarily on human-annotated content, ensuring genuine knowledge retrieval. Dataset descriptions and links are provided in §A; task types and example abilities assessed in §B; task instructions in §C; and dataset licenses in §E.

Memory Taxonomy. As shown in Figure 2, LMEB organizes memory into four types characterized along two key dimensions: (1) Level of Abstraction, which ranges from concrete, event-specific memories (e.g., episodic memory) to more abstract representations (e.g., dialogue and procedural memory); and (2) Temporal Dependency, which captures the extent to which retrieval depends on temporal context. Under this taxonomy, (i) Episodic Memory is low in abstraction and high in temporal dependency, focusing on specific events and their order; (ii) Dialogue Memory is relatively more abstract but remains highly time-dependent, requiring the recall of fragmented conversations; (iii) Semantic Memory is low in both abstraction and temporal dependency, centering on stable, general

[Figure 2]

Figure 2: Memory taxonomy of LMEB.

Setting (→) Dataset Information Statistics of dataset Avg. Word Lengths Memory Type (↓) Dataset (↓) Granularity (↓) # Tasks Querysrc Corpussrc #Query #Corpus #Qrels Avg. D / Q Query Document

|Episodic Memory<br><br>|EPBench [Huet et al., 2025] KnowMeBench [Wu et al., 2026] Total|Event Event Event<br><br>|54 15 69<br><br>|AI AI AI<br><br>|AI Human Hybrid<br><br>|3,644 2,838 11,458 3.144 2,162 27,062 4,840 2.239 5,806 29,900 16,298 2.807<br><br>|24.07 410.69<br><br>31.80 58.68 - -<br><br>|
|---|---|---|---|---|---|---|---|
|Dialogue Memory<br><br>|LoCoMo [Maharana et al., 2024] LongMemEval [Wu et al., 2025] REALTALK [Lee et al., 2025b] TMD [Alonso et al., 2024] MemBench [Tan et al., 2025] ConvoMem [Pakhomov et al., 2025] Total|Turn Session Turn Turn Round Turn Multi-granularity<br><br>|5<br><br>6<br><br><br>3 12 10 6 42<br><br>|AI AI & Human Human AI AI AI Hybrid<br><br>|AI & Human AI Human AI AI AI Hybrid<br><br>|1,976 5,882 2,801 1.418<br><br>500 237,655 948 1.896 679 8,944 1,553 2.287<br><br>2,134 7,463 76,287 35.748<br><br><br>10,000 929,115 29,592 2.959 5,867 500,221 13,779 2.349 21,156 1,689,280 124,960 5.907<br><br>|10.36 38.73 15.89 242.97 8.990 34.20 15.66 45.85 10.41 42.92 23.19 27.33 - -<br><br>|
|Semantic Memory<br><br>|QASPER [Dasigi et al., 2021] NovelQA [Wang et al., 2025] PeerQA [Baumgärtner et al., 2025] Covid-QA [Möller et al., 2020] ESG-Reports [Macé et al., 2025] MLDR [Chen et al., 2024] LooGLE [Li et al., 2024] SciFact [Wadden et al., 2020] Total|Paragraph Paragraph Sentence Paragraph Paragraph Paragraph Paragraph Sentence Multi-granularity<br><br>|1 7 1 1 1<br><br>1<br><br>2<br><br><br>1 15<br><br>|Human Human Human Human Human AI AI & Human Human Hybrid<br><br>|Human Human Human Human Human Human Human Human Human<br><br>|1,335 65,300 2,319 1.737 1,541 79,286 2,506 1.626<br><br>136 18,593 389 2.860<br><br>1,111 3,351 1,111 1.0<br><br>36 2,407 75 2.083<br><br>100 1,536 100 1.0 3,052 28,190 5,176 1.696<br><br>188 1,748 366 1.947 7,499 200,411 12,042 1.606<br><br>|7.934 77.45 19.79 139.51 15.65 24.67 9.541 110.91 9.389 129.35<br><br>11.34 112.94 13.72 164.82<br>12.89 39.93<br><br><br>- -<br><br>|
|Procedural Memory<br><br>|Gorilla [Patil et al., 2024] ToolBench [Qin et al., 2024] ReMe [Cao et al., 2025] Proced_mem_bench 2025 MemGovern [Wang et al., 2026] DeepPlanning [Zhang et al., 2026] Total|Tool Tool Experience Trajactory Experience Item Multi-granularity<br><br>|3 1 9 3 48 3 67<br><br>|AI AI AI Human Human AI Hybrid<br><br>|Human Human AI AI AI Human Hybrid<br><br>|598 1,005 598 1.0 1,100 13,862 2,629 2.39<br><br>1,217 914 1,217 1.0<br><br>40 336 529 13.225<br><br>121,475 121,475 121,475 1.0<br><br>120 19,839 515 4.292 124,550 157,431 126,963 1.019<br><br>|22.41 146.83<br><br>46.06 87.93 13.51 47.36<br><br>8.175 362.94 18.59 104.01<br><br>161.55 127.43 - -<br><br>|

- Table 1: Statistics of datasets in the LMEB benchmark. Querysrc and Corpussrc denote the source of the queries and corpus, respectively, where “AI” indicates AI-generated content, and “Human” indicates content created through crowdsourcing. Qrels denotes Query Relevance Judgments. Avg. D / Q denotes the average number of relevant documents per query.

knowledge; and (iv) Procedural Memory combines higher abstraction with lower temporal dependency, focusing on generalized skills, action sequences, and experiences.

#### 2.2 Dataset and Diversity Analysis

- Table 1 summarizes the LMEB datasets, which span diverse memory types and retrieval granularities, including event-, turn-, round-, session-, sentence-, paragraph-, tool-, and experience-level retrieval settings. Following BEIR [Thakur et al., 2021], we quantify inter-dataset diversity using pairwise weighted Jaccard similarity (JS) [Ioffe, 2010] over unigram word distributions in each dataset corpus. The theoretical formulation is provided in Appendix F. We further visualize dataset relationships using a 2D force-directed layout implemented in NetworkX [Hagberg et al., 2007], where nodes denote datasets and edge weights are proportional to their JS scores. The 2D visualization includes only episodic, dialogue, and semantic datasets. Procedural datasets are omitted because their corpora consistently yield low JS scores and therefore appear as weakly connected outliers.

From Figure 3, we draw three main observations: (1) Dialogue datasets show relatively high similarity because they share conversational topics, whereas procedural datasets exhibit low similarity due to their focus on domain-specific tasks such as coding, planning, and tool use. (2) TMD and LoCoMo are highly similar because TMD incorporates the LoCoMo corpus. Likewise, PeerQA and QASPER are closely related because both are built from academic papers in natural language processing (NLP) and machine learning (ML). (3) Figure 3(b) further illustrates these relationships, with datasets from the same memory type tending to cluster together. Overall, this analysis highlights the diversity of LMEB across memory types and datasets, reinforcing its value as a comprehensive benchmark for long-horizon memory retrieval.

#### 2.3 Evaluation Protocol and Extensibility

Evaluation Framework. LMEB builds on MTEB2. It provides a unified evaluation pipeline and plug-and-play model wrappers for reproducible benchmarking across widely used model backends, including Transformers [Wolf et al., 2020], Sentence-Transformers [Reimers and Gurevych, 2019], and vLLM [Kwon et al., 2023]. LMEB will be integrated more deeply into the MTEB ecosystem, further improving usability and making memory retrieval evaluation more accessible to the broader embedding community.

Unified Data Format. To support scalable evaluation across heterogeneous memory sources, LMEB adopts a standardized IR-style format with four components: queries, corpus, qrels, and an

2https://github.com/embeddings-benchmark/mteb

[Figure 3]

[Figure 4]

(a) Heatmap of weighted JS scores. (b) 2D force-directed visualization.

- Figure 3: Inter-dataset diversity in LMEB. The left side illustrates pairwise weighted Jaccard Similarity (JS) scores between unigram word distributions of each dataset corpus, while the right side shows dataset relationships in a force-directed 2D layout.

optional candidates file. All collected datasets are converted into this unified format, ensuring consistent preprocessing, indexing, and evaluation across tasks. The optional candidates file restricts retrieval to a bounded candidate pool when required. For example, in dialogue settings, each query is evaluated only against its corresponding conversation history rather than the full corpus, yielding a more realistic retrieval setting.

Evaluation Metrics. LMEB reports standard IR metrics, including Precision, Recall, MAP (Mean Average Precision), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain) at arbitrary top-k values. Following MTEB [Muennighoff et al., 2023] and BEIR [Thakur et al., 2021], we use NDCG@k as the primary metric and report NDCG@10 by default, as it captures both ranking quality and graded relevance when multiple memories may be relevant.

LMEB Extensibility. LMEB is designed to be easily extensible along two axes. First, new embedding models can be integrated with minimal code changes through pre-built model wrappers, supporting both local inference and server-based deployment. Second, new datasets can be added by providing files in the standardized corpus/queries/qrels format, with candidates included when needed. This design makes LMEB suitable for both academic research and industrial applications, where memory sources and evaluation needs continue to evolve.

#### 2.4 Construction Details

LMEB converts all collected resources into a unified IR-style schema with four components: queries, corpus, qrels, and an optional candidates file. The resulting files are organized as follows:

- (1) queries.jsonl: Stores query identifiers and query text in the id and text fields. For queries containing relative time expressions, we append an explicit query-time anchor to disambiguate temporal references (e.g., “What did we discuss 2 days ago? [Current time: 11:17 AM on Sunday 22 October, 2023]”). (2) corpus.jsonl: Stores memory items with id, text, and title. For episodic and dialogue memories, timestamps are preserved in title and/or text to support time-sensitive queries. For dialogue memories with a session-, round-, or turn-level hierarchy, the corresponding metadata is encoded in title so that queries targeting a specific span are evaluated within the intended memory scope (e.g., “What did we discuss over sessions 1 through 3?”). (3) qrels.tsv: Stores relevance annotations between queries and memory items, where each relevant query–memory pair is assigned a relevance label of 1. (4) candidates.jsonl (optional): Defines a bounded candidate pool for each query or scenario. For tasks with a restricted memory scope, each scene_id (either a query id or a shared scenario id) is mapped to a list of admissible candidate_doc_ids, thereby constraining retrieval to the permitted memory space.

# queries.jsonl {"id": "query_id", "text": "query text"}

-----------------------------------------------------------------------# corpus.jsonl {"id": "corpus_id", "text": "corpus text", "title": "corpus title"}

-----------------------------------------------------------------------# qrels.tsv query_id corpus_id 1

-----------------------------------------------------------------------# candidates.jsonl (optional) {"scene_id": "scene_id", "candidate_doc_ids": ["corpus_id", ......]}

### 3 Experimental Setup

Experimental settings, including the benchmarked models and implementation details, are provided in Appendix D. Task instructions are provided in Appendix C.

### 4 Main Results

- Table 2 and Table 3 report the performance of embedding models of various scales on LMEB. From these results, we draw the following observations:

LMEB Offers a Reasonable Level of Difficulty. LMEB presents a reasonable level of difficulty for current embedding models. The best-performing model (i.e., bge-multilingual-gemma2) achieves a Mean (Dataset) score of 61.41 on N@10 under the w/ inst. setting, indicating that the benchmark is challenging without being excessively difficult. The performance of the top models suggests that LMEB strikes a reasonable balance in task complexity, providing a meaningful evaluation of embedding models across episodic, dialogue, semantic, and procedural memory retrieval.

Larger Embedding Models Do Not Always Perform Better. The results show that larger embedding models, such as KaLM-Embedding-Gemma3 and bge-multilingual-gemma2, do not consistently outperform smaller models. For example, under the w/o inst. setting, both KaLM-Embedding-Gemma3 and bge-multilingual-gemma2 underperform smaller models such as EmbeddingGemma-300M and bge-m3 (Dense). These results suggest that model size alone does not determine performance; model architecture, training data, and task adaptation also play important roles.

The Impact of Task Instructions Varies Across Models. As shown in Figure 4, task instructions affect model performance differently across embedding models. Some models benefit from task instructions, some show little sensitivity to them, and others perform better without them. Models that tend to benefit from task instructions include KaLM-Embedding-Gemma3, bge-multilingual-gemma2, Qwen3-Embedding-8B, e5-mistral-7b-instruct, Qwen3-Embedding-4B, Qwen3-Embedding-0.6B, multilingual-e5-large-instruct, KaLM-Embedding-V2.5, KaLM-Embedding-V1, and jina-v5-textnano. By contrast, NV-Embed-v2 and jina-v5-text-small show little sensitivity to task instructions, while bge-m3 (Dense), bge-large-en-v1.5, and EmbeddingGemma-300M perform better without them. These differences are likely influenced by the training data and training objectives used for each embedding model.

### 5 Correlation Analysis

Figures 5–9 present the correlation between LMEB, LMEB-Episodic, LMEB-Dialogue, LMEBSemantic, and LMEB-Procedural with the traditional passage retrieval benchmark, MTEB (eng, v2) (retrieval subset) Enevoldsen et al. [2025]. We compute Pearson and Spearman rank correlation coefficients to assess these relationships. From these results, we draw the following key observations:

LMEB and MTEB Exhibit Orthogonality in Evaluation Capabilities. As shown in Figure 5, LMEB and MTEB (eng, v2) (retrieval subset) exhibit low Pearson and Spearman correlation coefficients of -0.115 and -0.130, respectively, indicating that the two benchmarks assess substantially different capabilities. MTEB primarily focuses on traditional passage retrieval, whereas LMEB is

LMEB-Episodic LMEB-Dialogue LMEB-Semantic LMEB-Procedural Mean (Dataset) Mean (Type) N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10

Model Size Dim

Embedding Models > 1B parameters KaLM-Embedding-Gemma3 12B 3840 67.01 74.56 50.89 61.86 47.81 65.82 60.70 71.08 53.91 66.97 56.60 68.33 bge-multilingual-gemma2 9B 3584 54.77 62.46 40.68 51.62 40.67 59.84 52.20 62.86 45.10 58.66 47.08 59.19 Qwen3-Embedding-8B 8B 4096 61.03 68.73 48.99 59.61 55.47 72.00 57.00 68.58 54.63 67.39 55.62 67.23 NV-Embed-v2 7B 4096 70.44 77.40 56.47 67.35 59.12 73.70 60.40 70.70 59.78 71.49 61.61 72.29 e5-mistral-7b-instruct 7B 4096 57.21 66.64 48.50 59.97 45.99 63.67 53.03 65.23 49.61 63.36 51.18 63.88 Qwen3-Embedding-4B 4B 2560 58.26 67.44 41.20 51.07 53.74 70.37 56.32 67.78 51.44 64.13 52.38 64.16 Embedding Models < 1B parameters jina-v5-text-small 596M 1024 61.03 69.06 51.06 61.43 51.80 67.24 56.79 66.50 53.80 65.62 55.17 66.05 Qwen3-Embedding-0.6B 596M 1024 57.66 65.73 50.41 61.13 52.01 67.84 57.14 67.12 53.49 65.62 54.30 65.46 multilingual-e5-large-instruct 560M 1024 57.49 66.68 48.13 60.10 48.36 65.38 51.02 62.73 49.85 63.34 51.25 63.72 bge-m3 (Dense) 560M 1024 67.00 73.60 55.61 66.10 56.29 71.26 55.37 64.68 56.83 68.27 58.57 68.91 KaLM-Embedding-V2.5 494M 896 61.86 69.63 48.95 59.94 52.00 68.15 52.97 64.23 52.33 64.98 53.94 65.49 KaLM-Embedding-V1 494M 896 56.60 66.33 41.32 51.64 47.49 64.16 54.85 65.39 48.64 61.28 50.07 61.88 bge-large-en-v1.5 335M 1024 55.85 65.49 53.12 63.85 52.55 67.94 52.62 62.89 53.02 65.22 53.54 65.04 EmbeddingGemma-300M 307M 768 68.19 75.29 53.94 65.00 53.58 69.33 57.32 67.42 56.03 68.17 58.26 69.26 jina-v5-text-nano 239M 768 56.73 64.58 43.22 53.18 49.49 65.01 50.48 59.64 48.71 60.28 49.98 60.60

- Table 2: Experiment results in the w/o inst. setting, where models use only the query for retrieval. The best results are boldfaced, and the second-best are underlined. We report the Mean (Dataset) score for LMEB-Episodic, LMEB-Dialogue, LMEB-Semantic, and LMEB-Procedural. Detailed performance for each dataset is provided in Table 16, Table 17, Table 18, and Table 19.

Model Size Dim

LMEB-Episodic LMEB-Dialogue LMEB-Semantic LMEB-Procedural Mean (Dataset) Mean (Type) N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10 N@10 R@10

Embedding Models > 1B parameters KaLM-Embedding-Gemma3 12B 3840 70.89 77.40 56.59 67.27 57.53 73.83 63.43 73.04 60.10 72.15 62.11 72.89 bge-multilingual-gemma2 9B 3584 70.88 77.67 59.60 69.86 60.41 75.91 61.40 71.71 61.41 73.27 63.07 73.79 Qwen3-Embedding-8B 8B 4096 60.85 68.80 51.69 61.99 55.51 71.39 59.12 69.53 55.94 68.08 56.79 67.93 NV-Embed-v2 7B 4096 68.45 76.21 56.42 66.77 62.18 76.74 58.77 68.07 60.25 71.61 61.46 71.95 e5-mistral-7b-instruct 7B 4096 60.64 69.58 55.03 65.25 53.16 69.66 56.30 67.34 55.21 67.82 56.28 67.95 Qwen3-Embedding-4B 4B 2560 57.24 66.33 43.58 53.86 54.52 69.86 59.81 69.94 53.23 65.20 53.79 65.00 Embedding Models < 1B parameters jina-v5-text-small 596M 1024 61.76 69.60 51.12 61.39 52.57 68.09 56.87 67.01 54.18 66.10 55.58 66.52 Qwen3-Embedding-0.6B 596M 1024 59.54 66.93 50.97 61.30 52.67 68.02 59.56 68.94 54.71 66.34 55.69 66.30 multilingual-e5-large-instruct 560M 1024 59.77 68.79 54.62 65.52 53.86 68.92 55.54 65.65 55.06 67.09 55.95 67.22 bge-m3 (Dense) 560M 1024 62.63 69.92 48.20 59.40 50.07 66.24 50.74 60.58 50.88 63.17 52.91 64.04 KaLM-Embedding-V2.5 494M 896 62.70 70.15 50.78 61.28 54.72 70.12 56.72 66.73 54.92 66.79 56.23 67.07 KaLM-Embedding-V1 494M 896 60.21 68.23 51.60 61.73 54.81 70.32 57.70 67.23 55.21 66.95 56.08 66.88 bge-large-en-v1.5 335M 1024 46.07 56.31 36.74 48.20 40.83 57.99 48.10 58.25 42.17 55.24 42.94 55.19 EmbeddingGemma-300M 307M 768 63.73 71.50 42.58 54.03 45.21 65.54 53.24 63.40 48.37 62.36 51.19 63.62 jina-v5-text-nano 239M 768 58.00 66.05 50.50 60.66 54.99 70.94 54.27 64.60 53.84 65.97 54.44 65.56

- Table 3: Experiment results in the w/ inst. setting, where models use both the query and task instruction for retrieval. The best results are boldfaced and the second-best are underlined. Detailed results for each dataset are provided in Table 20, Table 21, Table 22, and Table 23.

designed to assess long-horizon memory retrieval, which requires models to handle fragmented, context-dependent, and temporally distant information. This orthogonality highlights the unique value of LMEB for assessing long-term memory retrieval, especially in complex, real-world, memoryintensive scenarios such as OpenClaw [OpenClaw Contributors, 2026].

MTEB Performance Does Not Generalize Well to LMEB-Episodic/Dialogue. Models that perform well on MTEB do not generalize well to LMEB-Episodic/Dialogue, especially LMEB-Dialogue. As shown in Figure 7, LMEB-Dialogue and MTEB (eng, v2) (retrieval subset) exhibit Pearson and Spearman correlation coefficients of -0.496 and -0.364, respectively. These results suggest that strong performance on traditional passage retrieval does not readily transfer to dialogue memory retrieval involving fragmented, redundant, and context-dependent information. This further highlights the unique value of LMEB for evaluating embedding models in real-world scenarios where episodic and dialogue memory retrieval is essential.

MTEB Shows Partial Generalization to LMEB-Semantic/Procedural. MTEB exhibits limited generalization to LMEB-Semantic and LMEB-Procedural, as shown in Figures 8 and 9. LMEBSemantic and MTEB (eng, v2) (retrieval subset) exhibit Pearson and Spearman correlation coefficients of 0.103 and 0.061, respectively, while the corresponding values for LMEB-Procedural are 0.291 and 0.429. Although LMEB-Semantic and MTEB both evaluate semantic passage retrieval, their

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

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- Figure 4: Performance comparison between the w/o inst. and w/ inst. settings. The x-axis shows the two settings, and the y-axis shows N@10 performance.

weak correlations suggest substantial differences in task characteristics. One possible explanation is that LMEB-Semantic focuses on retrieval within contextually coherent and bounded scenarios, whereas MTEB emphasizes open-domain retrieval. By contrast, LMEB-Procedural shows a somewhat stronger correlation with MTEB, possibly because some embedding models are also trained on toolor code-retrieval tasks that overlap with procedural memory retrieval. Overall, performance on MTEB transfers to LMEB only to a limited extent, reflecting meaningful differences in domain focus and task characteristics between the two benchmarks.

### 6 Related Work

Embedding Benchmarks. A long line of benchmarks has been proposed to evaluate text embeddings across downstream tasks, including retrieval, semantic textual similarity (STS), and clustering. Early evaluations centered on a small set of sentence-level similarity tasks, e.g., SentEval [Conneau and Kiela, 2018] and SemEval [Agirre et al., 2012, 2013, 2014, 2015, Afzal et al., 2016]. To broaden coverage, retrieval-focused benchmarks such as BEIR [Thakur et al., 2021] consolidate heterogeneous IR datasets for cross-domain evaluation, while multilingual benchmarks such as MIRACL [Zhang

- et al., 2023] and AIR-Bench [Chen et al., 2025a] focus on multilingual retrieval. More recently, MTEB/C-MTEB/MMTEB [Muennighoff et al., 2023, Xiao et al., 2024, Enevoldsen et al., 2025] provide unified evaluation protocols and leaderboards for text embeddings across diverse tasks (e.g., retrieval, classification, clustering, reranking, and STS), lowering the barrier to standardized comparison. Meanwhile, multimodal embedding benchmarks evaluate image-text embeddings for retrieval and alignment. For example, MIEB [Xiao et al., 2025] and VLM2Vec-V1 [Jiang et al., 2025] provide unified evaluation protocols across diverse image-text tasks. Overall, existing benchmarks have substantially advanced standardized evaluation for embedding quality. However, they rarely evaluate long-horizon memory retrieval involving fragmented, context-dependent, and temporally distant evidence, leaving it unclear how well embedding models support memory retrieval in practice.

Embedding Models. Text embedding models map text into continuous vectors for efficient similarity search and downstream tasks [Zhao et al., 2025, Zhang et al., 2025a, Zhao et al., 2024]. Early approaches learned static word embeddings, such as GloVe [Pennington et al., 2014], and composed sentence representations via pooling, but lacked contextual awareness. Transformer architectures [Devlin et al., 2019] introduced contextualization through self-attention and have become the dominant backbone for embeddings. High-quality sentence embeddings typically require task-specific fine-

[Figure 21]

- Figure 5: Correlation between the evaluation scores on LMEB and MTEB (eng, v2) (retrieval subset) Enevoldsen et al. [2025]. For LMEB, we use Mean (Dataset) scores based on N@10, consistent with the metric used in MTEB (eng, v2) (retrieval subset). Since bge-m3 (Dense), bgelarge-en-v1.5, and EmbeddingGemma-300M perform better without task instructions, their results are taken from the w/o inst. setting; all others use w/ inst.. Point size is proportional to model size.

tuning. Sentence-BERT [Reimers and Gurevych, 2019], for example, showed that fine-tuning on sentence-pair objectives yields embeddings directly comparable with cosine similarity. Recently, adapting large language models (LLMs) into embedding models has become mainstream, such as GTE [Li et al., 2023, Zhang et al., 2024a], Qwen3-Embedding [Zhang et al., 2025b], BGE [Chen

- et al., 2024], Jina [Günther et al., 2025, Akram et al., 2026], NV-Embed [Lee et al., 2025a], and KaLM [Hu et al., 2025, Zhao et al., 2026]. Common approaches include multi-task contrastive training [Mohr et al., 2024] to unify heterogeneous supervision and instruction tuning [Asai et al.,

- 2023, Su et al., 2023] to condition embeddings on natural-language task descriptions. In parallel, multimodal embedding models [Zhang et al., 2024b, Jiang et al., 2025, Meng et al., 2025] adopt similar contrastive training recipes to learn shared image-text representations. While LLM-based embeddings often generalize well, they introduce trade-offs in efficiency, latency, and deployment cost, and gains on standard semantic benchmarks may not transfer to long-horizon memory retrieval.

### 7 Conclusion

In this work, we present the Long-horizon Memory Embedding Benchmark (LMEB), a benchmark for evaluating embedding models on long-horizon memory retrieval tasks. LMEB consists of 22 datasets spanning 4 memory types and 193 retrieval tasks, filling a critical gap left by existing benchmarks such as MTEB, which do not adequately evaluate the retrieval of long-term, fragmented, contextdependent, and temporally distant memory information. We evaluate state-of-the-art embedding models on LMEB and uncover several key findings: (1) LMEB offers a reasonable level of difficulty;

- (2) even models that perform well on traditional passage retrieval benchmarks struggle with memory retrieval tasks; and (3) LMEB and MTEB are largely orthogonal in the capabilities they assess. By open-sourcing LMEB with a standardized data format compatible with MTEB and easy-to-adapt code, we provide the research community with a unified and reproducible framework for evaluating embedding models on real-world memory retrieval tasks. LMEB not only advances current evaluation practices but also provides a foundation for developing embedding models that can better handle complex, real-world memory retrieval challenges. We hope LMEB will foster further progress in text embeddings and contribute to the next generation of memory-augmented systems.

### References

Naveed Afzal, Yanshan Wang, and Hongfang Liu. Mayonlp at semeval-2016 task 1: Semantic textual similarity based on lexical semantic net and deep learning semantic model. In SemEval@NAACLHLT, pages 674–679. The Association for Computer Linguistics, 2016.

Eneko Agirre, Daniel M. Cer, Mona T. Diab, and Aitor Gonzalez-Agirre. Semeval-2012 task 6: A pilot on semantic textual similarity. In SemEval@NAACL-HLT, pages 385–393. The Association for Computer Linguistics, 2012.

Eneko Agirre, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. *sem 2013 shared task: Semantic textual similarity. In *SEM@NAACL-HLT, pages 32–43. Association for Computational Linguistics, 2013.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, and Janyce Wiebe. Semeval-2014 task 10: Multilingual semantic textual similarity. In SemEval@COLING, pages 81–91. The Association for Computer Linguistics, 2014.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Iñigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, and Janyce Wiebe. Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability. In SemEval@NAACL-HLT, pages 252–263. The Association for Computer Linguistics, 2015.

Mohammad Kalim Akram, Saba Sturua, Nastia Havriushenko, Quentin Herreros, Michael Günther, Maximilian Werk, and Han Xiao. jina-embeddings-v5-text: Task-targeted embedding distillation. CoRR, abs/2602.15547, 2026.

Nick Alonso, Tomas Figliolia, Anthony Ndirango, and Beren Millidge. Toward conversational agents with context and time sensitive long-term memory. CoRR, abs/2406.00057, 2024.

Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. Task-aware retrieval with instructions. In ACL (Findings), pages 3650–3675. Association for Computational Linguistics, 2023.

Tim Baumgärtner, Ted Briscoe, and Iryna Gurevych. Peerqa: A scientific question answering dataset from peer reviews. In NAACL (Long Papers), pages 508–544. Association for Computational Linguistics, 2025.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. arXiv preprint arXiv:2512.10696, 2025.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. CoRR, abs/2402.03216, 2024.

Jianlyu Chen, Nan Wang, Chaofan Li, Bo Wang, Shitao Xiao, Han Xiao, Hao Liao, Defu Lian, and Zheng Liu. Air-bench: Automated heterogeneous information retrieval benchmark. In ACL (1), pages 19991–20022. Association for Computational Linguistics, 2025a.

Silin Chen, Shaoxin Lin, Xiaodong Gu, Yuling Shi, Heng Lian, Longfei Yun, Dong Chen, Weiguo Sun, Lin Cao, and Qianxiang Wang. Swe-exp: Experience-driven software issue resolution. CoRR, abs/2507.23361, 2025b.

Alexis Conneau and Douwe Kiela. Senteval: An evaluation toolkit for universal sentence representations. In LREC. European Language Resources Association (ELRA), 2018.

Max Conti, Manuel Faysse, Gautier Viaud, Antoine Bosselut, Céline Hudelot, and Pierre Colombo. Context is gold to find the gold passage: Evaluating and training contextual document embeddings. In EMNLP, pages 22594–22608. Association for Computational Linguistics, 2025.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In NAACL-HLT, pages 4599–4610. Association for Computational Linguistics, 2021.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT (1), pages 4171–4186. Association for Computational Linguistics, 2019.

Yiming Du, Hongru Wang, Zhengyi Zhao, Bin Liang, Baojun Wang, Wanjun Zhong, Zezhong Wang, and Kam-Fai Wong. Perltqa: A personal long-term memory dataset for memory classification, retrieval, and synthesis in question answering. CoRR, abs/2402.16288, 2024.

Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, KamFai Wong, and Jeff Z Pan. Rethinking memory in llm based agents: Representations, operations, and emerging topics. arXiv preprint arXiv:2505.00675, 2025.

Kenneth C. Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzeminski, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Veysel Çagatan, Akash Kundu, and et al. MMTEB: massive multilingual text embedding benchmark. In ICLR. OpenReview.net, 2025.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. CoRR, abs/2508.06433, 2025.

Zafeirios Fountas, Martin Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou-Ammar, and Jun Wang. Human-like episodic memory for infinite context llms. CoRR, abs/2407.09450, 2024.

Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang, and Han Xiao. jina-embeddings-v4: Universal embeddings for multimodal multilingual retrieval. CoRR, abs/2506.18902, 2025.

Aric Hagberg, Pieter J Swart, and Daniel A Schult. Exploring network structure, dynamics, and function using networkx. Technical report, Los Alamos National Laboratory (LANL), 2007.

Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Haofen Wang, Jun Yu, and Min Zhang. Kalm-embedding: Superior training data brings A stronger embedding model. CoRR, abs/2501.01028, 2025.

Alexis Huet, Zied Ben-Houidi, and Dario Rossi. Episodic memories generation and evaluation benchmark for large language models. In ICLR. OpenReview.net, 2025.

Sergey Ioffe. Improved consistent sampling, weighted minhash and L1 sketching. In ICDM, pages 246–255. IEEE Computer Society, 2010.

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. In ICLR. OpenReview.net, 2025.

Ishant Kohar and Aswanth Krishnan. A benchmark for procedural memory retrieval in language agents. CoRR, abs/2511.21730, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models. In ICLR. OpenReview.net, 2025a.

Dong-Ho Lee, Adyasha Maharana, Jay Pujara, Xiang Ren, and Francesco Barbieri. REALTALK: A 21-day real-world dataset for long-term conversation. CoRR, abs/2502.13270, 2025b.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts? In ACL (1), pages 16304–16333. Association for Computational Linguistics, 2024.

Xintong Li, Jalend Bantupalli, Ria Dharmani, Yuwei Zhang, and Jingbo Shang. Toward multi-session personalized conversation: A large-scale dataset and hierarchical tree framework for implicit reasoning. CoRR, abs/2503.07018, 2025.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. CoRR, abs/2308.03281, 2023.

Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, and James A. Burke. Memtool: Optimizing short-term memory management for dynamic tool calling in LLM agent multi-turn conversations. CoRR, abs/2507.21428, 2025.

Quentin Macé, António Loison, and Manuel Faysse. Vidore benchmark V2: raising the bar for visual retrieval. CoRR, abs/2505.17166, 2025.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In ACL (1), pages 13851–13870. Association for Computational Linguistics, 2024.

Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, and Semih Yavuz. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. CoRR, abs/2507.04590, 2025.

Xin Miao, Yongqi Li, Shen Zhou, and Tieyun Qian. Episodic memory retrieval from llms: A neuromorphic mechanism to generate commonsense counterfactuals for relation extraction. In ACL (Findings), pages 2489–2511. Association for Computational Linguistics, 2024.

Isabelle Mohr, Markus Krimmel, Saba Sturua, Mohammad Kalim Akram, Andreas Koukounas, Michael Günther, Georgios Mastrapas, Vinit Ravishankar, Joan Fontanals Martínez, Feng Wang, Qi Liu, Ziniu Yu, Jie Fu, Saahil Ognawala, Susana Guzman, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. Multi-task contrastive learning for 8192-token bilingual text embeddings. CoRR, abs/2402.17016, 2024.

Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. COVID-QA: A question answering dataset for COVID-19. In Karin Verspoor, Kevin Bretonnel Cohen, Mark Dredze, Emilio Ferrara, Jonathan May, Robert Munro, Cecile Paris, and Byron Wallace, editors, Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020, Online, July 2020. Association for Computational Linguistics. URL https://aclanthology.org/2020.nlpcovid19-acl.18/.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: massive text embedding benchmark. In EACL, pages 2006–2029. Association for Computational Linguistics, 2023.

OpenClaw Contributors. Your own personal ai assistant. any os. any platform. the lobster way. https://github.com/openclaw/openclaw, 2026. GitHub repository, accessed: 2026-05-06.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory. CoRR, abs/2509.25140, 2025.

Egor Pakhomov, Erik Nijkamp, and Caiming Xiong. Convomem benchmark: Why your first 150 conversations don’t need RAG. CoRR, abs/2511.10523, 2025.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. In NeurIPS, 2024.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): from tool use to agentic evaluation of large language models. In ICML, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In EMNLP, pages 1532–1543. ACL, 2014.

Mathis Pink, Qinyuan Wu, Vy Ai Vo, Javier Turek, Jianing Mu, Alexander Huth, and Mariya Toneva. Position: Episodic memory is the missing piece for long-term LLM agents. CoRR, abs/2502.06975, 2025.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis. In ICLR. OpenReview.net, 2024.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In EMNLP/IJCNLP (1), pages 3980–3990. Association for Computational Linguistics, 2019.

Henry L Roediger III and Magdalena Abel. The double-edged sword of memory retrieval. Nature Reviews Psychology, 1(12):708–720, 2022.

Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, and Zhaochun Ren. Retrieval models aren’t tool-savvy: Benchmarking tool retrieval for large language models. In ACL (Findings), volume ACL 2025 of Findings of ACL, pages 24497–24524. Association for Computational Linguistics, 2025.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In ICLR. OpenReview.net, 2021.

Yifan Song, Weimin Xiong, Xiutian Zhao, Dawei Zhu, Wenhao Wu, Ke Wang, Cheng Li, Wei Peng, and Sujian Li. Agentbank: Towards generalized LLM agents via fine-tuning on 50000+ interaction trajectories. In EMNLP (Findings), pages 2124–2141. Association for Computational Linguistics,

- 2024.

Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. In ACL (Findings), pages 1102–1121. Association for Computational Linguistics, 2023.

Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. Membench: Towards more comprehensive evaluation on the memory of llm-based agents. In ACL (Findings), volume ACL 2025 of Findings of ACL, pages 19336–19352. Association for Computational Linguistics, 2025.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogenous benchmark for zero-shot evaluation of information retrieval models. CoRR, abs/2104.08663, 2021.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. In ACL (1), pages 16022–16076. Association for Computational Linguistics, 2024.

Endel Tulving et al. Episodic and semantic memory. Organization of memory, 1(381-403):1, 1972. Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram

Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Ju-yeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divyashree Sreepathihalli, Aashi Jain,

Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Qin Yin, Yun-Hsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, and Mojtaba Seyedhosseini. Embeddinggemma: Powerful and lightweight text representations. CoRR, abs/2509.20354, 2025.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. In EMNLP (1), pages 7534–7550. Association for Computational Linguistics, 2020.

Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng Bao, Xiangkun Hu, Zheng Zhang, Qian Wang, and Yue Zhang. Novelqa: Benchmarking question answering on documents exceeding 200k tokens. In ICLR. OpenReview.net, 2025.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. CoRR, abs/2212.03533, 2022. doi: 10.48550/ARXIV.2212.03533. URL https://doi.org/10.48550/ arXiv.2212.03533.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multilingual E5 text embeddings: A technical report. CoRR, abs/2402.05672, 2024. doi: 10.48550/ARXIV.2402.05672. URL https://doi.org/10.48550/arXiv.2402.05672.

Qihao Wang, Ziming Cheng, Shuo Zhang, Fan Liu, Rui Xu, Heng Lian, Kunyi Wang, Xiaoming Yu, Jianghao Yin, Sen Hu, Yue Hu, Shaolei Zhang, Yanbing Liu, Ronghao Chen, and Huacan Wang. Memgovern: Enhancing code agents through learning from governed human experiences. CoRR, abs/2601.06789, 2026.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In EMNLP (Demos), pages 38–45. Association for Computational Linguistics, 2020.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In ICLR. OpenReview.net, 2025.

Tingyu Wu, Zhisheng Chen, Ziyan Weng, Shuhe Wang, Chenglong Li, Shuo Zhang, Sen Hu, Silin Wu, Qizhen Lan, Huacan Wang, et al. Knowme-bench: Benchmarking person understanding for lifelong digital companions. arXiv preprint arXiv:2601.04745, 2026.

Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth C. Enevoldsen, and Niklas Muennighoff. MIEB: massive image embedding benchmark. CoRR, abs/2504.10471, 2025.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In SIGIR, pages 641–649. ACM, 2024.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context LLM with multi-conv rl-based memory agent. CoRR, abs/2507.02259, 2025.

Meishan Zhang, Xin Zhang, Xinping Zhao, Shouzheng Huang, Baotian Hu, and Min Zhang. On the role of pretrained language models in general-purpose text embeddings: A survey. CoRR, abs/2507.20783, 2025a.

Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval. In EMNLP (Industry Track), pages 1393–1412. Association for Computational Linguistics, 2024a.

Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. GME: improving universal multimodal retrieval by multimodal llms. CoRR, abs/2412.16855, 2024b.

Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. MIRACL: A multilingual retrieval dataset covering 18 diverse languages. Trans. Assoc. Comput. Linguistics, 11:1114–1131, 2023.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. CoRR, abs/2506.05176, 2025b.

Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, and Junyang Lin. Deepplanning: Benchmarking long-horizon agentic planning with verifiable constraints. CoRR, abs/2601.18137, 2026.

Xinping Zhao, Dongfang Li, Yan Zhong, Boren Hu, Yibin Chen, Baotian Hu, and Min Zhang. SEER: self-aligned evidence extraction for retrieval-augmented generation. In EMNLP, pages 3027–3041. Association for Computational Linguistics, 2024.

Xinping Zhao, Yan Zhong, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Dongfang Li, Baotian Hu, and Min Zhang. Funnelrag: A coarse-to-fine progressive retrieval paradigm for RAG. In NAACL (Findings), volume NAACL 2025 of Findings of ACL, pages 3029–3046. Association for Computational Linguistics, 2025.

Xinping Zhao, Xinshuo Hu, Zifei Shan, Shouzheng Huang, Yao Zhou, Xin Zhang, Zetian Sun, zhenyu liu, Dongfang Li, Xinyuan Wei, Youcheng Pan, Yang Xiang, Meishan Zhang, Haofen Wang, Jun Yu, Baotian Hu, and Min Zhang. KaLM-embedding-v2: Superior training techniques and data inspire a versatile embedding model. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=Y7qzhvWhcz.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills. CoRR, abs/2504.07079, 2025.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. MEM1: learning to synergize memory and reasoning for efficient long-horizon agents. CoRR, abs/2506.15841, 2025.

###### Dataset Website (Link)

Episodic Memory EPBench [Huet et al., 2025] https://doi.org/10.6084/m9.figshare.28244480 KnowMeBench [Wu et al., 2026] https://github.com/QuantaAlpha/KnowMeBench/tree/main/KnowmeBench

Dialogue Memory LoCoMo [Maharana et al., 2024] https://github.com/snap-research/locomo/tree/main/data LongMemEval [Wu et al., 2025] https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned REALTALK [Lee et al., 2025b] https://github.com/danny911kr/REALTALK/tree/main/data TMD [Alonso et al., 2024] https://github.com/Zyphra/TemporalMemoryDataset MemBench [Tan et al., 2025] https://github.com/import-myself/Membench/tree/main/MemData ConvoMem [Pakhomov et al., 2025] https://huggingface.co/datasets/Salesforce/ConvoMem

Semantic Memory QASPER [Dasigi et al., 2021] https://huggingface.co/datasets/allenai/qasper NovelQA [Wang et al., 2025] https://huggingface.co/datasets/NovelQA/NovelQA PeerQA [Baumgärtner et al., 2025] https://huggingface.co/datasets/UKPLab/PeerQA Covid-QA [Möller et al., 2020] https://huggingface.co/datasets/illuin-conteb/covid-qa ESG-Reports [Macé et al., 2025] https://huggingface.co/datasets/illuin-conteb/esg-reports MLDR [Chen et al., 2024] https://huggingface.co/datasets/illuin-conteb/mldr-conteb-eval LooGLE [Li et al., 2024] https://huggingface.co/datasets/bigai-nlco/LooGLE SciFact [Wadden et al., 2020] https://huggingface.co/datasets/allenai/scifact

###### Procedural Memory

Gorilla [Patil et al., 2024] https://huggingface.co/datasets/mangopy/ToolRet-Queries

https://huggingface.co/datasets/mangopy/ToolRet-Tools

ToolBench [Qin et al., 2024] https://huggingface.co/datasets/mangopy/ToolRet-Queries

https://huggingface.co/datasets/mangopy/ToolRet-Tools ReMe [Cao et al., 2025] https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task Proced_mem_bench 2025 https://github.com/qpiai/Proced_mem_bench/tree/main/procedural_memory_benchmark MemGovern [Wang et al., 2026] https://github.com/QuantaAlpha/MemGovern/blob/main/data DeepPlanning [Zhang et al., 2026] https://huggingface.co/datasets/Qwen/DeepPlanning

Table 4: Original dataset website (link) for all datasets present in LMEB.

### A Datasets

Below, we list the 22 evaluation datasets in LMEB, spanning four memory types. Each dataset includes a corpus and a set of test queries. We provide dataset website links in Table 4. We describe each dataset below:

#### A.1 Episodic Memory Retrieval

Episodic memory retrieval aims to recall past events grounded in temporal cues, entities, contents, and spatial contexts [Fountas et al., 2024, Pink et al., 2025]. In LMEB, we treat an episodic query as input and retrieve the corresponding event memories as output. Examples are presented in Table 5.

EPBench [Huet et al., 2025] is a synthetic episodic memory benchmark for evaluating event recall and episodic reasoning in LLMs. It represents episodic events with structured fields, including temporal and spatial context, involved entities, and detailed descriptions. We use its event set as the corpus C and the provided task queries as queries Q.

KnowMeBench [Wu et al., 2026] is a benchmark for long-horizon person understanding built from long-form autobiographical narratives. It converts each narrative into a time-anchored, flashbackaware memory stream and evaluates evidence-linked queries spanning factual recall, subjective state attribution, and principle-level reasoning. We use the reconstructed narrative stream as the corpus C and the benchmark questions as queries Q.

#### A.2 Dialogue Memory Retrieval

Dialogue memory retrieval aims to maintain context across multi-turn interactions by recalling relevant dialogue history and user preferences [Wu et al., 2025, Maharana et al., 2024]. Such recall supports coherent conversations and enables personalization over time [Du et al., 2024, Li et al.,

- 2025]. In LMEB, we treat a dialogue query as input and retrieve the corresponding dialogues as output. Examples are presented in Table 6.

LoCoMo [Maharana et al., 2024] is a dataset for very long-term, persona-grounded dialogues with explicit temporal event structure. It contains conversations of up to 35 sessions (300 turns; ∼9k

Dataset Query Relevant-Document Granularity

|EPBench<br><br>|Think about Aurora Chavez’s experiences. Describe all the key events they’ve been involved in, focusing on what happened rather than when or where it occurred.|......Aurora implemented blockchain solutions with a determination that bordered on obsession. ......It wasn’t until Samara Bayes tapped her on the shoulder that Aurora Chavez realized how much time had passed. She blinked, momentarily disoriented as she emerged from her coding trance. Samara Bayes gestured towards the refreshment table, a knowing smile on their face. ......<br><br>|Event|
|---|---|---|---|
|KnowMeBench|On August 15, 1969, which shipping company’s bus did the narrator’s family take?<br><br>|{<br><br>"id": 1, "timestamp": "1969-08-15 14:00:30", "location": "Rocky path", "action": "I watched the vehicle cross a bridge,<br><br>driving down the narrow fjord with its right turn signal blinking before finally stopping.",<br><br>→ → →<br><br>"dialogue": null, "environment": "The bus belonged to Aucity<br><br>Shipping Company, painted in two shades of brown.",<br><br>→ →<br><br>"background": null, "inner_thought": null<br><br>}<br><br>|Event|

Table 5: Examples of query-relevant document pairs for episodic memory retrieval datasets.

tokens on average) and evaluates long-range dialogue memory via tasks such as question answering. We use the dialogue history as the corpus C and the benchmark questions as queries Q.

LongMemEval [Wu et al., 2025] is used for evaluating long-term memory in chat assistants under sustained, multi-session interactions. It probes five core abilities: information extraction, multisession reasoning, temporal reasoning, knowledge updates, and abstention. We use the chat history as the corpus C and the curated questions as queries Q.

REALTALK [Lee et al., 2025b] consists of a 21-day corpus of authentic dialogues, capturing long-term, open-domain interactions with richer emotional expressions and more varied persona consistency than synthetic data. It includes tasks like persona simulation and memory probing, where models must answer questions based on long-term dialogue. LMEB focuses on the memory-probing task, using the dialogue history as the corpus C and the memory-probing questions as queries Q.

TMD [Alonso et al., 2024] introduces a dataset designed to evaluate retrieval-augmented generation (RAG) models in the context of long-term conversational memory. It focuses on two key challenges: time/event-based queries, which require retrieving information from previous conversations based on temporal cues or event order, and ambiguous queries, which necessitate surrounding conversational context to understand. In LMEB, we use this dataset’s conversation history as the corpus C and the time- or context-dependent queries as queries Q.

MemBench [Tan et al., 2025] introduces a comprehensive benchmark for evaluating the memory capabilities of LLM-based agents. It incorporates multiple memory levels, such as factual and reflective memory, and evaluates agents in a variety of interactive scenarios, including participation and observation. We use the dialogues from this dataset as the corpus C and the associated queries as queries Q.

ConvoMem [Pakhomov et al., 2025] introduces a benchmark for conversational memory evaluation, containing 75,336 question-answer pairs across categories such as user facts, assistant recall, preferences, and temporal changes. LMEB samples 5,867 queries that focus on dialogue-based retrieval, highlighting the evolving nature of conversational memory, where memory systems grow progressively with each conversation. We use the dialogue history as the corpus C and the sampled queries as queries Q.

#### A.3 Semantic Memory Retrieval

Semantic memory retrieval focuses on recalling general knowledge and concepts that are largely independent of time or specific context [Tulving et al., 1972]. Semantic memory is more stable and generalizable, and supports knowledge-augmented reasoning and use [Zhou et al., 2025]. In LMEB, we treat the question as input and retrieve the passage containing the answer as output. Examples are presented in Table 7. For long-document datasets (i.e., LooGLE and NovelQA), we segment texts using semchunk3 with a chunk size of 256.

3https://github.com/isaacus-dev/semchunk

QASPER [Dasigi et al., 2021] is a question-answering (QA) dataset grounded in full research papers, designed to reflect information-seeking queries that require reasoning across multiple document sections. It contains 5,049 questions over 1,585 NLP papers, where questions are written from the title and abstract and answered using evidence from the full text. In LMEB, we process the dataset and retain 1,335 valid queries. We use the paper content (segmented into passages) as the corpus C and the QASPER questions as queries Q.

NovelQA [Wang et al., 2025] evaluates long-context comprehension over complex narratives drawn from English novels. It features manually annotated questions targeting nuanced understanding (e.g., detail-oriented and multi-hop queries) under very long inputs, with average context lengths exceeding 200K tokens. We use the novel text (segmented into passages) as the corpus C and the benchmark questions as queries Q.

PeerQA [Baumgärtner et al., 2025] is a document-level scientific QA dataset built from questions raised in peer reviews and answers provided by the original authors. It contains 579 QA pairs over 208 academic papers, with a majority from ML and NLP, and supports evidence retrieval. In LMEB, we filter out questions without the corresponding paper text and retain 136 questions. We use the paper text (segmented into sentences) as the corpus C and the PeerQA questions as queries Q.

Covid-QA [Möller et al., 2020] is a biomedical QA dataset annotated from COVID-19 scientific articles in CORD-19. It contains expert-written question-answer pairs, where annotators mark text as answer and formulate corresponding questions. In LMEB, we use the processed version released by [Conti et al., 2025]. We use the selected CORD-19 articles (segmented into passages) as the corpus C and the annotated questions as queries Q.

ESG-Reports [Macé et al., 2025] consists of lengthy ESG reports from the fast-food industry. It is derived from ViDoRe Benchmark v2 [Macé et al., 2025], which originally annotates query-page relevance for visual document retrieval. In LMEB, we use the processed version released by Conti et al. [2025], which converts documents to text, chunks them into passages (i.e., the corpus C), and manually re-annotates query (i.e., queries Q)-passage pairs after filtering queries that rely primarily on visual evidence (e.g., tables or graphs).

MLDR [Chen et al., 2024] is a multilingual long-document retrieval dataset constructed from Wikipedia, Wudao, and mC4 across 13 languages, where questions are generated from sampled paragraphs and paired with their source documents. In LMEB, we use only the English subset. We adopt the processed version Conti et al. [2025], where documents are chunked into passages (i.e., the corpus C), and GPT-4o is used to annotate the gold chunk within the corresponding document for each query (i.e., queries Q).

LooGLE [Li et al., 2024] is a long-context understanding benchmark built from post-2022 documents, with long inputs (often >24K tokens) and newly generated questions spanning diverse domains. It includes a curated high-quality QA pairs that target long-dependency and short-dependenct reasoning. We use the chunked documents as the corpus C and the generated questions as queries Q.

SciFact [Wadden et al., 2020] is a scientific claim verification dataset that requires retrieving evidence abstracts from the research literature to support or refute a given claim. It contains ∼1.4K expertwritten claims paired with annotated evidence abstracts (with labels and rationales). In LMEB, we use the dev split (300 claims) since the test split is unlabeled, and after filtering claims with empty evidence, we retain 188 valid queries. We use the abstract collection as the corpus C and the retained claims as queries Q.

#### A.4 Procedural Memory Retrieval

Procedural memory retrieval focuses on recalling learned skills, action patterns, and structured procedures that guide task execution and multi-step reasoning [Fang et al., 2025, Ouyang et al.,

- 2025]. Such retrieval is essential for automating complex behaviors, supporting tool use, and enabling adaptive decision-making in reinforcement learning and tool-augmented systems [Lumer et al., 2025, Yu et al., 2025]. In LMEB, we treat a task instruction or problem description as input and retrieve the corresponding procedure, action sequence, or solution trajectory as output. Examples are presented in Table 8.

Gorilla [Patil et al., 2024] introduces APIBench, a benchmark covering APIs from HuggingFace, TorchHub, and TensorHub, designed to evaluate tool-use through API call generation. A key

component of Gorilla is the integration of a document retriever, which enables models to fetch up-to-date API documentation and adapt to version changes at test time. In LMEB, we treat the API documentation as the corpus C and natural language tool-use instructions as queries Q. Note that we use the processed version released by [Shi et al., 2025].

ToolBench [Qin et al., 2024] is a large-scale tool-use benchmark constructed from 16,464 real-world RESTful APIs collected from RapidAPI Hub. It provides automatically generated instructions covering both single-tool and multi-tool scenarios, along with annotated solution paths. In LMEB, we treat the API documentation as the corpus C and tool-use instructions as queries Q, evaluating procedural retrieval in both single-tool and multi-tool scenarios. Note that we use the processed version released by [Shi et al., 2025].

ReMe [Cao et al., 2025] extracts fine-grained procedural experiences through multi-faceted distillation, supports context-adaptive retrieval via scenario-aware indexing, and maintains a compact memory pool through utility-based refinement. Beyond static storage, ReMe highlights retrieval as a core mechanism for adapting past solution trajectories to new tasks. In LMEB, we use the ready-to-use memories released for BFCL [Patil et al., 2025] and AppWorld [Trivedi et al., 2024], treating the distilled experiences as the corpus C and task and generalized queries as queries Q.

Proced_mem_bench [Kohar and Krishnan, 2025] is a benchmark that explicitly isolates procedural memory retrieval from task execution, evaluating whether agents can retrieve functionally equivalent procedures under novel object instantiations. Built on ALFWorld [Shridhar et al., 2021], it provides dual corpora of expert and LLM-generated trajectories, along with systematically stratified queries. In LMEB, we treat the trajectory corpus as C and procedural task queries as queries Q, where we consider trajectories with a relevance score of 7.0 or higher as relevant memories.

MemGovern [Wang et al., 2026] introduces a procedural memory framework that transforms largescale, unstructured GitHub issue data into agent-usable experiential memory. Through experience governance, it converts historical bug-fixing trajectories into structured experience cards and employs an agentic search strategy for logic-driven retrieval of relevant past solutions. In LMEB, we use the governed experience cards as the corpus C and bug-fixing task descriptions as queries Q.

DeepPlanning [Zhang et al., 2026] is a long-horizon planning benchmark that emphasizes procedural reasoning under local and global constraints. It includes multi-day travel and multi-product shopping tasks, requiring models to retrieve, integrate, and execute action sequences that satisfy time, budget, and resource constraints. For LMEB, we focus on the shopping subset, treating shopping task instructions as queries Q and the corresponding items as the corpus C.

### B Tasks

In this section, we provide a detailed overview of the tasks associated with each dataset under four memory types. As illustrated in Table 10, Table 11, Table 12, and Table 13, each task is characterized by its Task Type, the example abilities assessed. The total number of tasks (#Tasks) for a dataset is generally determined by multiplying the number of subsets (#Subsets) by the number of task types, unless otherwise specified. Here, “Task Type” refers to the specific category of a task, reflecting the type of retrieval the model is expected to perform. “Subset” represents a partition of a dataset, typically organized by topic, scenario, or source, allowing multiple tasks to be instantiated under different contexts while maintaining consistent evaluation criteria

### C Instructions

To ensure embeddings follow instructions during downstream tasks, we prepend the specific task instructions to the queries. The instructed query is formulated as follows:

##### qinst = Instruct: {task instruction} \n Query: q. (1)

where q denotes the original query; qinst is the instructed query. Task instructions for different datasets as well as task types are summarized in Table 14 and Table 15.

### D Experimental Setup

- D.1 Benchmarked Models

We evaluate a diverse set of embedding models, ranging in size from several hundred million (M) to 10 billion (B) parameters. These models include: jina-v5-text-nano/jina-v5-text-small4 [Akram et al., 2026], Qwen3-Embedding-0.6B/Qwen3-Embedding-4B/Qwen3-Embedding-8B [Zhang et al.,

- 2025b], EmbeddingGemma-300M [Vera et al., 2025], KaLM-Embedding-V1/KaLM-EmbeddingV2.5/KaLM-Embedding-Gemma3-12B-2511 (KaLM-Embedding-Gemma3) [Hu et al., 2025, Zhao et al., 2026], bge-m3/bge-multilingual-gemma2/bge-large-en-v1.5 [Xiao et al., 2024, Chen et al.,

2024], NV-Embed-v2 [Lee et al., 2025a], multilingual-e5-large-instruct/e5-mistral-7b-instruct [Wang et al., 2024, 2022]. Table 9 provides links to the publicly available models used in the LMEB evaluation.

- D.2 Implementation Details

Following [Thakur et al., 2021, Muennighoff et al., 2023], we evaluate retrieval performance using two widely adopted IR metrics: (i) NDCG@k (N@k) (the main metric), which measures ranking quality with logarithmic discounting over the top-k results; and (ii) Recall@k (R@k), which measures the proportion of gold memories successfully retrieved within the top-k results. Note that Recall@k refers to capped Recall@k. Standard Recall@k can sometimes yield unintuitive results, especially when the number of relevant documents for a query exceeds k. To address this issue, we cap the denominator at k, meaning that when the number of relevant documents for a query exceeds k, we use the minimum of k and the total number of relevant documents to compute the recall score. We report results under two query settings: (i) w/o inst., where the model encodes the query alone; and (ii) w/ inst., where the model encodes the concatenation of the instruction and the query, allowing us to quantify the impact of instructions on retrieval performance. We use a maximum input length of 1024 tokens for both queries and corpus passages. However, multilingual-e5-large-instruct and bge-large-en-v1.5 use a maximum input length of 512 tokens, as this is the maximum length they support. All experiments are conducted on 4 NVIDIA H100 GPUs.

### E Dataset Licenses

The authors of 2 datasets in the LMEB benchmark (REALTALK and TMD) do not specify the dataset license in the paper or repository. We summarize the licenses for the remaining datasets below.

- • EPBench, LongMemEval, MemBench, MLDR, MemGovern: Provided under the MIT License.
- • KnowMeBench, Covid-QA, Gorilla, ToolBench, ReMe, Proced_mem_bench, DeepPlanning: Provided under Apache License 2.0 license.
- • LoCoMo, ConvoMem, QASPER: Provided under the CC BY-NC 4.0 license.
- • SciFact: Provided under the CC BY-NC 2.0 license.
- • PeerQA: Provided under the CC BY-NC-SA 4.0 license.
- • NovelQA, ESG-Reports: Data collection archives are under Copyright.
- • LooGLE: Provided under the CC BY-SA 4.0 license.

### F Weighted Jaccard Similarity

The weighted Jaccard similarity J(S,T) [Ioffe, 2010] is used to quantify lexical overlap between a source dataset S and a target dataset T. Let Sk and Tk denote the normalized frequency of the unique word k in S and T, respectively, where the normalized frequency is computed as the count of k divided by the sum of frequencies of all words in the dataset. The weighted Jaccard similarity is then defined as:

min(Sk,Tk) k max(Sk,Tk)

, (2)

J(S,T) = k

4We use the retrieval-optimized versions, i.e., jina-embeddings-v5-text-small-retrieval and jina-embeddings-v5-text-nanoretrieval.

where the sum is over all unique words k appearing in S or T.

### G Limitations and Broader Impacts

#### G.1 Limitations.

Although LMEB is, to the best of our knowledge, the first benchmark for long-horizon memory retrieval with embedding models and is well aligned with the needs of emerging memory-augmented systems, this work focuses primarily on the text-only setting. In particular, LMEB does not yet cover multimodal long-horizon memory retrieval, where relevant memories may be distributed across text, images, audio, video, and other modalities. Extending the benchmark to multimodal settings is an important direction for future work.

#### G.2 Broader Impact.

LMEB advances the evaluation of embedding models by introducing a benchmark for long-horizon memory retrieval, a setting that is increasingly important for personalized assistants, long-context agents, and memory-augmented systems but remains underexplored in prior work. By providing a comprehensive evaluation of memory retrieval across episodic, dialogue, semantic, and procedural settings, LMEB can help identify the strengths and weaknesses of existing models and support the development of more effective memory embedding models. More broadly, LMEB can serve as a practical reference for memory-augmented systems such as OpenClaw [OpenClaw Contributors,

- 2026], where choosing an embedding model with stronger long-horizon memory retrieval performance can improve the accuracy of recalling relevant memories and thereby support more personalized and contextually appropriate behaviors. There are no negative societal impacts of the work performed.

Dataset Query Relevant-Document Granularity

|LoCoMo<br><br>|When is Melanie planning on going camping?|{<br><br>"speaker": "Melanie", "dia_id": "D2:1", "date_time": "1:14 pm on 25 May, 2023" "text": "Hey Caroline, since we last chatted, I've had a lot of<br><br>things happening to me. I ran a charity race for mental health last Saturday \u2013 it was really rewarding. Really made me think about taking care of our minds."<br><br>→ → →<br><br>}<br><br>|Turn|
|---|---|---|---|
|LongMemEval|I received a piece of jewelry last Saturday from whom?<br><br>|[<br><br>...... {<br><br>"role": "user", "content": "I'm looking for some advice on cleaning and<br><br>maintaining my antique furniture. I recently acquired a beautiful vintage armchair from the 1950s and I want to make sure I'm taking good care of it. By the way, I also got a stunning crystal chandelier from my aunt today, which used to belong to my great-grandmother - it's got intricate etchings and sparkling droplets that catch the light. Do you have any general tips on how to dust and polish these types of pieces?",<br><br>→ → → → → → → →<br><br>"has_answer": true }<br><br>...... ]<br><br>|Session|
|REALTALK<br><br>|When is Elise’s first yoga class?|{<br><br>"clean_text": "Hey Kate! Yes, I did end up signing up for a Yoga class. My first class is today at 5:00pm! I’m super excited. I went and got myself a new yoga set and a yoga mat to be prepared. Any other tips for my first day?",<br><br>→ → →<br><br>"speaker": "elise", "date_time": "17.01.2024, 19:57:41", "dia_id": "D13:2"<br><br>} {<br><br>"clean_text": "I just had my first yoga class and it went amazing!", "speaker": "elise", "date_time": "18.01.2024, 01:18:57", "dia_id": "D13:4"<br><br>}<br><br>|Turn|
|TMD<br><br>|What did we chat about on May 8th?|{<br><br>"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you! How have you been?", "date_time": "01:56:04 AM on Monday 08 May, 2023", "response_number": "0"<br><br>}<br><br>...... {<br><br>"speaker": "Melanie", "dia_id": "D1:18", "text": "Yep, Caroline. Taking care of ourselves is vital. I'm off to<br><br>→ go swimming with the kids. Talk to you soon!", "date_time": "01:58:09 AM on Monday 08 May, 2023", "response_number": "17"<br><br>}|Turn<br><br>|
|MemBench|Who is older, Nolan Hayes or Sophie Turner?<br><br>|{<br><br>"sid": 49, "user_message": "My boss, Nolan Hayes, is 39 years old.", "assistant_message": "Noted, Nolan Hayes is 39 years old.", "time": "'2024-10-01 08:49' Tuesday", "place": "Boston, MA", "rel": "boss", "attr": "age", "value": "39 years old"<br><br>} {<br><br>"sid": 76, "user_message": "Sophie Turner, my aunt, is 27 years old.", "assistant_message": "That's great to hear! It's always nice to have<br><br>family connections, especially with someone as talented as Sophie.",<br><br>→ →<br><br>"time": "'2024-10-01 09:16' Tuesday", "place": "Boston, MA", "rel": "Aunt", "attr": "age", "value": "27 years old"<br><br>}|Round|
|ConvoMeme<br><br>|Who is my favorite sports team?|{<br><br>"speaker": "User", "text": "I'm trying to schedule a trip back to Portland to see my<br><br>family soon. I should check the Trail Blazers schedule; taking a client to a game would be a great touch. They've been my favorite team since I was a kid."<br><br>→ → →<br><br>} {<br><br>"speaker": "User", "text": "I know I used to be all about basketball, but after living in Seattle for so long and getting season tickets, I'm completely hooked on football. The Seattle Seahawks are hands down my favorite team now, the atmosphere at the games is just incredible."<br><br>→ → →<br><br>}<br><br>|Turn|

Table 6: Examples of query-relevant document pairs for dialogue memory retrieval datasets.

Dataset Query Relevant-Document Granularity

|QASPER|How big is the ANTISCAM dataset?<br><br>|<Title>Datasets ::: AntiScam Dataset</Title> <Text> To enrich available noncollaborative task datasets, we created a corpus of human-human anti-scam dialogs in order to learn human elicitation strategies. We chose a popular Amazon customer service scam scenario to collect dialogs between users and attackers who aim to collect users information. We posted a role-playing task on the Amazon Mechanical Turk platform and collected a typing conversation dataset named AntiScam. We collected 220 human-human dialogs. The average conversation length is 12.45 turns and the average utterance length is 11.13 words. Only 172 out of 220 users successfully identified their partner as an attacker, suggesting that the attackers are well trained and not too easily identifiable. We recruited two expert annotators who have linguistic training to annotate 3,044 sentences in 100 dialogs, achieving a 0.874 averaged weighted kappa value.</Text><br><br>|Paragraph|
|---|---|---|---|
|NovelQA<br><br>|Down which river do Huck and Jim travel?|<Text> ‘All right — that’s mighty good; they won’t find me, and they’ll think I’ve been killed, and floated down the river — there’s something up there that ‘ll help them think so — so don’t you lose no time, Jim, but just shove off for the big water as fast as ever you can.’ I never felt easy till the raft was two mile below there and out in the middle of the Mississippi. Then we hung up our signal lantern, and judged that we was free and safe once more. I hadn’t had a bite to eat since yesterday, so Jim he got out some corn-dodgers and buttermilk, and pork and cabbage and greens — there ain’t nothing in the world so good when it’s cooked right — and whilst I eat my supper we talked and had a good time. I was powerful glad to get away from the feuds, and so was Jim to get away from the swamp. We said there warn’t no home like raft, after all. Other places do seem so cramped up and smothery, but a raft don’t. You feel mighty free and easy and comfortable on a raft. </Text>|Paragraph|
|PeerQA<br><br>|How many words are in the corpus in total?|<Title> Descriptive Statistics </Title> <Text> The corpus includes 180 essays: 120 (Draft1 and Draft2) with an average of about 400 words and 60 (Draft3) with an average of around 500 words. </Text><br><br>|Sentence|
|Covid-QA|What does the NTCP protein mediate?<br><br>|<Text> NTCP is a transmembrane protein, usually located in the lateral surface (canalicular) of hepatocytes, which mediates bile acid transport [7]. Geographically, it is a good candidate for an HBV receptor. Moreover, the authors could convert the cell lines previously non-permissible to HBV infection to permissible by overexpression of NTCP, again supporting its possible role in the HBV infection process. This can be a critical and long-awaited discovery toward understanding HBV receptors and establishing an experimental HBV infection system. </Text>|Paragraph|
|ESG-Reports<br><br>|What are McDonald’s Franchisees?|<Text> ¢ Franchisees: Collective group of independent individuals and entities owning and operating McDonald’s restaurants under one of the following structures — conventional franchise, developmental license or affiliate. For more information on the Company’s structure and scope see our latest Annual Report. ¢ McDonald’s restaurants/Restaurants: Includes restaurants owned by the Company and its Franchisees. Introduction Our Planet Forward-Looking Statements This report contains certain forward-looking statements or projections that reflect our expectations regarding future events, performance and goals and involve several risks and uncertainties. Generally speaking, any statement in this report not based upon historical fact is a forward-looking statement. Forward-looking statements can also be identified by the use of forward-looking or conditional words, such mas “could,” “should,” “can,” “enable,” “continue,” estimate,” “forecast,” “intend,” “look,” “may,” “will,”</Text><br><br>|Paragraph|
|MLDR|<Text> What is Steven Whyte’s occupation?<br><br>|<Text> Steven Whyte (born 17 March 1969) is a sculptor classically trained in the traditional methodology of figurative bronze and portrait sculpture living in Carmel, California. He has produced many public memorials and installations in both England and throughout the United States with subjects ranging from miners, to soldiers and fire fighters. He is credited with over fifty life size and larger bronze public figures and major monuments including The Silverdale Mining Memorial, The Lance Sergeant Jack Baskeyfield VC Tribute, The Spirit of 1948, and The Dr. John Roberts Monument. Whyte’s multimillion-dollar, sixteen-figure monument in San Diego, California entitled National Salute to Bob Hope and the Military is one of his most notable works. </Text>|Paragraph|
|LooGLE<br><br>|What were some of the subjects that Picardo illustrated in his poststudent days?"<br><br>|<Title>José Luis Picardo</Title> <Text> While he was still a student, architects commissioned murals from him for the interiors of their buildings, and employed him within their practices for the graphic representations and perspectives of their plans. Picardo executed his first professional mural painting at the age of 20 in 1939 in the Cine Fígaro (Figaro Cinema) in Madrid, commissioned by his architecture mentor Luis Moya Blanco. The painting of murals was the main source of income for Picardo during his youth and early career.As a student Picardo also began to illustrate many articles and later several covers for the Spanish architectural magazines Revista Nacional de Arquitectura and the Boletín de la Dirección General de Arquitectura. His drawings in these publications have been described as<br><br>showing ïncreasing sophisticationänd being of complexity and extraordinary quality¨ ¨. Particularly noted in his post-student days were illustrations portraying Madrid in the 1950s and 1960s, the Spanish protectorate in Morocco, and sketches of the Canarias (Canary Islands) in 1953. He also showed a growing interest in historic architecture, in particular its preservation and restoration. Picardo completed his training by making increasingly numerous travels to study buildings around Spain and abroad. </Text>|Paragraph|
|SciFact<br><br>|All hematopoietic stem cells segregate their chromosomes randomly.|<Text> Sequential administration of 5-chloro-2-deoxyuridine and 5-iodo-2deoxyuridine indicated that all HSCs segregate their chromosomes randomly.", "title": "Haematopoietic stem cells do not asymmetrically segregate chromosomes or retain BrdU </Text><br><br>|Sentence|

Table 7: Examples of query-relevant document pairs for semantic memory retrieval datasets. The corpus title is enclosed within <Title> and </Title> tags, while the corpus text is enclosed within <Text> and </Text> tags.

Dataset Query Relevant-Document Granularity

|Gorilla|You are building a virtual global tour guide that can identify languages from the audio of people speaking. Use a model to identify which language is being spoken.|{<br><br>"domain": "Audio Classification", "framework": "Hugging Face Transformers", "functionality": "Language Identification", "api_call": "AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi<br><br>→ →<br><br>/whisper-medium-fleurs-lang-id')", "api_arguments": ["model =<br><br>→ AutoModelForSpeechClassification.from_pretrained( 'sanchit-gandhi/whisper-medium-fleurs-lang-id')", "processor =<br><br>→ Wav2Vec2Processor.from_pretrained(<br><br>'sanchit-gandhi/whisper-medium-fleurs-lang-id')"], "python_environment_requirements": ["transformers==4.27.0.dev0", "pytorch==1.13.1", "datasets==2.9.0", "tokenizers==0.13.2"], "example_code": "", "performance": {"dataset": "google/xtreme_s", "accuracy": 0.8805}, "description": "This model is a fine-tuned version of openai/whisper-medium on the FLEURS subset of the google/xtreme_s dataset. It is used for language identification in audio classification tasks.", "name": "sanchit-gandhi/whisper-medium-fleurs-lang-id"<br><br>→ → → → →<br><br>}|Tool<br><br>|
|---|---|---|---|
|ToolBench|My family is considering moving to a new city and we want to compare the sales tax rates of different cities. Can you help us by providing the sales tax breakdowns by ZIP code for the cities we’re interested in? We would like to know the state, county, city, and any additional special sales taxes.<br><br>|{<br><br>"category_name": "Finance", "required_parameters": [], "optional_parameters": [{"name": "state", "type": "STRING", "description": "State name.",<br><br>"default": ""}, {"name": "city", "type": "STRING", "description": "City name.", "default": ""}, {"name": "zip_code", "type": "STRING", "description": "Valid US ZIP code.", "default": "90210"}],<br><br>→ → →<br><br>"method": "GET", "template_response": {"zip_code": "str", "total_rate": "str", "state_rate": "str",<br><br>→ "city_rate": "str", "county_rate": "str", "additional_rate": "str"}, "name": "Sales_Tax_by_API_Ninjas_v1_salestax", "description": "API Ninjas Sales Tax API endpoint. Returns one or more sales tax breakdowns<br><br>by ZIP code according to the specified parameters. Each breakdown includes the state sales tax (if any), county sales tax (if any), city sales tax (if any), and any additional special sales taxes. All tax values are presented in decimals (e.g. 0.1 means 10% tax). \n\n Exactly one of the following must be set: zip_code or (city + state)"<br><br>→ → → →<br><br>}|Tool|
|ReMe<br><br>|For your investment portfolio, could you inform me of the current price of ’Quasar Ltd.’?<br><br>|{<br><br>"when_to_use": "When a user requests stock information and subsequent watchlist<br><br>→ management\n",<br><br>"content": "1. Use get_symbol_by_name to map company name to stock symbol\n2. Fetch stock details via get_stock_info for price data\n3. Add symbol to watchlist using add_to_watchlist\n4. Verify watchlist state with get_watchlist\n5. Use send_message with proper receiver_id and formatted content for communication\n"<br><br>→ → →<br><br>}<br><br>|Experience|
|Proced_mem_bench<br><br>|Examine a phone with the desk lamp while holding it|{<br><br>"task_description": "examine the alarmclock with the desklamp.", "state_action_pairs": [<br><br>{<br><br>"step_id": 1, "state": "You are in the middle of a room. Looking quickly around you, you see a bed 1, a<br><br>desk 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, and a garbagecan 1.",<br><br>→ →<br><br>"action": "go to desk 1"<br><br>}, {<br><br>"step_id": 2, "state": "On the desk 1, you see a alarmclock 4, a alarmclock 3, a alarmclock 2, a<br><br>alarmclock 1, a book 1, a cd 2, a cd 1, a creditcard 3, a creditcard 2, a desklamp 1, a keychain 3, a pen 1, and a pencil 3.",<br><br>→ →<br><br>"action": "use desklamp 1"<br><br>}, {<br><br>"step_id": 3, "state": "You turn on the desklamp 1.", "action": "examine alarmclock 1"<br><br>}, {<br><br>"step_id": 4, "state": "On the desk 1, you see a alarmclock 4, a alarmclock 3, a alarmclock 2, a<br><br>alarmclock 1, a book 1, a cd 2, a cd 1, a creditcard 3, a creditcard 2, a desklamp 1, a keychain 3, a pen 1, and a pencil 3.",<br><br>→ →<br><br>"action": "examine desk 1"<br><br>}, {<br><br>"step_id": 5, "state": "On the desk 1, you see a alarmclock 4, a alarmclock 3, a alarmclock 2, a<br><br>alarmclock 1, a book 1, a cd 2, a cd 1, a creditcard 3, a creditcard 2, a desklamp 1, a keychain 3, a pen 1, and a pencil 3.",<br><br>→ →<br><br>"action": "use desklamp 1 on alarmclock 1"<br><br>}, {<br><br>"step_id": 6, "state": "You turn on the desklamp 1.", "action": "examine alarmclock 1 with desklamp 1"<br><br>} ],<br><br>}<br><br>|Trajactory|
|MemGovern<br><br>|Statistical distribution moments incorrectly defined as infinite instead of undefined (NaN) for distributions with undefined moments, causing inconsistent test behavior.|Strategic Approach: The fix addresses the fundamental mathematical issue that certain statistical distributions have undefined moments rather than infinite moments. The strategy involves correctly representing undefined moments as NaN (Not a Number) instead of infinity to properly reflect the mathematical reality and ensure consistent test behavior. Step-by-Step Implementation: 1. Update the distribution’s moment calculation method to return NaN for all moments instead of infinite values 2. Modify test validation logic to skip moment checks when the expected moment value is not finite (NaN or infinite) 3. Replace infinite moment checks with finite moment checks in test assertions 4. Ensure sample mean and variance tests only run when the theoretical moments are finite<br><br>|Experience|
|DeepPlanning<br><br>|I’m putting together a complete outdoor adventure outfit and need to find three specific items online. First, I’m looking for a Women’s Rho Hybrid Zip Neck Top that’s proven popular with customers - it needs to have more than 25 four-star reviews and total sales volume exceeding 4000 to ensure it’s a reliable choice. Next, I need an all-seasons item that has solid customer feedback, specifically one with exactly 50 three-star ratings and more than 3000 total reviews, which shows it’s been thoroughly tested by many buyers. Finally, I’m searching for Women’s Omni-Shield Adventure Trek Pants targeted specifically for women, and I need the exact item that has 18 fourstar ratings and 689 total sales volume - these specific numbers help me identify the right product variant I’ve been researching.|{<br><br>"product_id": "8e4ca530", "name": "Women's Rho Hybrid Zip Neck Top",<br><br>..... "sales_volume": {"monthly": 350, "total": 4200}, "rating": {"average_score": 4.8, "total_reviews": 215, "distribution": {"5_star": 180, "4_star":<br><br>→ 28, "3_star": 5, "2_star": 1, "1_star": 1}},<br><br>......<br><br>} {<br><br>"product_id": "a393e541", "name": "Women's AirSense Pleated Wide Pants",<br><br>...... "suitable_season": "All Seasons",<br><br>...... "rating": {"average_score": 4.8, "total_reviews": 3150, "distribution": { "5_star": 2700,<br><br>→ "4_star": 350, "3_star": 50, "2_star": 30, "1_star": 20}},<br><br>......<br><br>} {<br><br>"product_id": "581cf5c8", "name": "Women's Omni-Shield Adventure Trek Pants",<br><br>...... "target_demographic": "Women",<br><br>...... "sales_volume": { "monthly": 74, "total": 689 }, "rating": { "average_score": 4.8, "total_reviews": 135, "distribution": { "5_star": 112,<br><br>→ "4_star": 18, "3_star": 3, "2_star": 1, "1_star": 1}},<br><br>...... }<br><br>|Item|

Table 8: Examples of query-relevant document pairs for procedural memory retrieval datasets.

###### Model Public Model Checkpoints (Link)

jina-v5-text-nano [Akram et al., 2026] https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval jina-v5-text-small [Akram et al., 2026] https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval Qwen3-Embedding-0.6B [Zhang et al., 2025b] https://huggingface.co/Qwen/Qwen3-Embedding-0.6B Qwen3-Embedding-4B [Zhang et al., 2025b] https://huggingface.co/Qwen/Qwen3-Embedding-4B Qwen3-Embedding-8B [Zhang et al., 2025b] https://huggingface.co/Qwen/Qwen3-Embedding-8B EmbeddingGemma-300M [Vera et al., 2025] https://huggingface.co/google/embeddinggemma-300m KaLM-Embedding-V1 [Hu et al., 2025] https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5 KaLM-Embedding-V2.5 [Zhao et al., 2026] https://huggingface.co/KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5 KaLM-Embedding-Gemma3-12B-2511 2026 https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511 bge-m3 [Chen et al., 2024] https://huggingface.co/BAAI/bge-m3 bge-multilingual-gemma2 2024, 2024 https://huggingface.co/BAAI/bge-multilingual-gemma2 bge-large-en-v1.5 [Xiao et al., 2024] https://huggingface.co/BAAI/bge-large-en-v1.5 NV-Embed-v2 [Lee et al., 2025a] https://huggingface.co/nvidia/NV-Embed-v2 multilingual-e5-large-instruct [Wang et al., 2022] https://huggingface.co/intfloat/multilingual-e5-large-instruct e5-mistral-7b-instruct [Wang et al., 2024] https://huggingface.co/intfloat/e5-mistral-7b-instruct

Table 9: Publicly available model links used for evaluation.

Dataset #Subsets Task Type Example abilities assessed # Tasks

|EPBench<br><br>|9<br><br>|Entities<br><br>|Retrieve entities associated with specific events, times, or locations.<br><br>|54|
|---|---|---|---|---|
| | |Event_contents<br><br>|Retrieve events or activities involving specific entities.| |
| | |Full_event_details|Retrieve events with specific time, location, entities, and content| |
| | |Other_entities|Retrieve all other entities involved in a given event, excluding the specified ones.| |
| | |Spaces|Retrieve the most recent location where a given entity was observed in the chronological timeline.| |
| | |Times<br><br>|Retrieve all dates when the specific events occurred.| |
|KnowMeBench<br><br>|3<br><br>|adversarial_abstention<br><br>|Fail to retrieve supporting events for invalid queries, triggering abstention.<br><br>|15|
| | |information_extraction|Retrieve all relevant entities under strict temporal and spatial constraints.| |
| | |mind-body_interaction<br><br>|Retrieve action–state pairs to support reasoning about ironic or contradictory behaviors.| |
| | |mnestic_trigger_analysis|Retrieve memory events triggered by specific sensory or associative cues.| |
| | |temporal_reasoning|Retrieve temporally constrained memories for timeline-based reasoning.| |

Table 10: Overview of episodic retrieval tasks with task types and example abilities.

Dataset #Subsets Task Type Example abilities assessed # Tasks

|LoCoMo|1<br><br>|adversarial|Retrieve evidence to support detecting unanswerable or misleading questions.<br><br>|5|
|---|---|---|---|---|
| | |multi_hop|Retrieve and integrate relevant information across multiple sessions to support multi-step reasoning.| |
| | |open_domain<br><br>|Retrieve both user-provided information and external knowledge such as commonsense or world facts.| |
| | |single_hop<br><br>|Retrieve relevant information from a single session.| |
| | |temporal_reasoning<br><br>|Retrieve time-stamped or chronologically organized dialogue to support temporal reasoning.| |
|LongMemEval<br><br>|1<br><br>|knowledge_update<br><br>|Retrieve the most recent user information to detect changes in the user’s states or preferences.<br><br>|6|
| | |multi_session<br><br>|Retrieve and aggregate relevant user information across two or more sessions.| |
| | |single_session_assistant<br><br>|Retrieve the assistant’s own previously stated content within a single session.| |
| | |single_session_preference|Retrieve the user’s stated preferences and constraints within a single session and apply them to generate personalized responses.| |
| | |single_session_user|Retrieve user-mentioned facts or attributes within a single session to answer follow-up questions that depend on that information accurately and consistently.| |
| | |temporal_reasoning<br><br>|Retrieve timestamped metadata and explicit time references from dialogue to align events chronologically and support reasoning about order, recency, duration, and time-dependent constraints.| |
|REALTALK|1<br><br>|commonsense<br><br>|Retrieve conversational context and combine it with external commonsense/world knowledge to answer questions that cannot be resolved from dialogue alone.<br><br>|3|
| | |multi_hop|Retrieve evidence across multiple sessions to connect dispersed details and support multi-step reasoning over integrated context.| |
| | |temporal_reasoning|Retrieve chronologically relevant dialogue and time cues to reason about event sequences and time-based dependencies.| |
|TMD<br><br>|1|content_time_qs<br><br>|Retrieve relevant content by conditioning on speaker identity, timestamp/session metadata, etc.|12<br><br>|
| | |date_span_time_qs|Retrieve dialogue content within a specified absolute date range (DATE1DATE2).| |
| | |dates_time_qs<br><br>|Retrieve dialogue associated with a specific absolute date.| |
| | |day_span_time_qs<br><br>|Retrieve discussions over a rolling recent window (e.g., last three days/last week).| |
| | |earlier_today_time_qs|Retrieve topics discussed earlier the same day by using intra-day time cues.| |
| | |last_named_day_time_qs<br><br>|Retrieve dialogue from the most recent occurrence of a specified weekday (e.g., “last Sunday”).| |
| | |month_time_qs|Retrieve dialogue occurring within a specified month.| |
| | |rel_day_time_qs|Retrieve dialogue from N days ago by resolving relative day offsets and mapping them to the correct date(s).| |
| | |rel_month_time_qs<br><br>|Retrieve dialogue from N months ago by resolving relative month offsets and month boundaries.| |
| | |rel_session_time_qs|Retrieve content from N sessions ago by identifying session order and selecting the correct prior session(s).| |
| | |session_span_time_qs|Retrieve dialogue between two referenced sessions (S1-S2).| |
| | |session_time_qs<br><br>|Retrieve dialogue from a specified session number.| |
|MemBench†|10<br><br>|aggregative<br><br>|Retrieve evidence about more than two entities on a common attribute.|10<br><br>|
| | |comparative<br><br>|Retrieve evidence about two entities from multiple messages and compare them on a shared attribute.| |
| | |emotion<br><br>|Retrieve consecutive messages within a time window to infer the user’s emotional state.| |
| | |knowledge_updating<br><br>|Retrieve the most recent mention among conflicting updates.| |
| | |multi_hop<br><br>|Retrieve multiple relevant messages.| |
| | |multi_session_assistant|Retrieve multiple relevant assistant messages.| |
| | |post_processing<br><br>|Retrieve multiple messages that collectively support the required additional reasoning steps.| |
| | |preference<br><br>|Retrieve multiple preference signals across multiple messages and apply them to generate a personalized response.| |
| | |single_hop<br><br>|Retrieve the single supporting message.| |
| | |single_session_assistant<br><br>|Retrieve a single supporting assistant message.| |
|ConvoMem‡|6<br><br>|abstention_evidence<br><br>|Retrieve evidence indicating the information is missing/undecided and support a correct abstention or “not specified yet” response.|6<br><br>|
| | |assistant_facts_evidence<br><br>|Retrieve multiple supporting assistant-provided statements.| |
| | |changing_evidence|Retrieve the full chain of updates and select the latest valid fact, suppressing superseded evidence.| |
| | |implicit_connection_evidence<br><br>|Retrieve multiple context cues and connect them to infer the implied answer, grounding the inference in retrieved evidence.| |
| | |preference_evidence<br><br>|Retrieve distributed preference constraints and use them as retrieval criteria to generate personalized response.| |
| | |user_evidence<br><br>|Retrieve and link multiple user-stated facts across turns to answer a query that requires jointly grounded evidence.| |

- Table 11: Overview of dialogue retrieval tasks with task types and example abilities. † MemBench has 10 subsets, each corresponding to a single task type; therefore, MemBench has 10 tasks. ‡ ConvoMem has 6 subsets, each corresponding to a single task type; therefore, ConvoMem has 6 tasks.

Dataset #Subsets Task Type Example abilities assessed # Tasks

|QASPER|1|QASPER|Retrieve evidence passages from full NLP research papers to answer information-seeking questions.<br><br>|1|
|---|---|---|---|---|
|NovelQA|1<br><br>|Character|Retrieve plot evidence and localize the key episode marking the event/character change.<br><br>|7|
| | |Meaning<br><br>|Retrieve symbol/metaphor contexts or semantically matched passages to support an implication or chapter-level match.| |
| | |Plot<br><br>|Retrieve narrative cues about plot developments.| |
| | |Relation<br><br>|Retrieve distributed factual evidence and connect it to resolve entity relations.| |
| | |Setting<br><br>|Retrieve distributed factual evidence and connect it to resolve settings.| |
| | |Span|Retrieve distributed factual evidence and connect it to resolve spans.| |
| | |Times<br><br>|Retrieve time-related evidence (timestamps, durations, ordering cues) and align it to support time-based reasoning.| |
|PeerQA|1<br><br>|PeerQA|Retrieve sentence-level evidence from full scientific papers to answer peer-review–style questions grounded in the paper text.<br><br>|1|
|Covid-QA<br><br>|1|Covid-QA<br><br>|Retrieve answer-bearing passages from COVID-19 biomedical articles (CORD-19) to support expert-annotated scientific question answering.<br><br>|1|
|ESG-Reports<br><br>|1<br><br>|ESG-Reports<br><br>|Retrieve relevant passages from long, text-converted ESG reports.|1<br><br>|
|MLDR|1|MLDR|Retrieve the gold passage from long English documents drawn from Wikipedia, Wudao, and mC4.<br><br>|1|
|LooGLE†|2|LongDepQA<br><br>|Retrieve multiple supporting passages distributed across the document.|2|
| | |ShortDepQA|Retrieve the single supporting sentence or paragraph.| |
|SciFact<br><br>|1|SciFact<br><br>|Retrieve evidence abstracts from the scientific literature to support or refute a claim.|1|

- Table 12: Overview of semantic retrieval tasks with task types and example abilities. † LooGLE has 2 subsets, each corresponding to a single task type; therefore, LooGLE has 2 tasks.

Dataset #Subsets Task Type Example abilities assessed # Tasks

|Gorilla†|3<br><br>|gorilla_huggingface|Retrieve the appropriate HuggingFace model for the specified task query.<br><br>|3|
|---|---|---|---|---|
| | |gorilla_pytorch|Retrieve the most task-aligned TorchHub API entry by matching the query intent.| |
| | |gorilla_tensor<br><br>|Retrieve the most task-aligned TensorHub API entry by matching the query intent.| |
|ToolBench<br><br>|1|ToolBench|Retrieve the relevant API documentation to support tool-use instructions, including both single-tool and multi-tool scenarios.<br><br>|1|
|ReMe‡|6<br><br>|generalized_query<br><br>|Retrieve experience that remain applicable under a generalized query.|9|
| | |task_query<br><br>|Retrieve the most relevant experience that match a specific task query.| |
|Proced_mem_bench|1<br><br>|easy|Retrieve the correct single-object placement procedure for a specified object-location query.<br><br>|3|
| | |medium<br><br>|Retrieve multi-step procedures that include required state changes (e.g., heat/clean) before placement.| |
| | |hard<br><br>|Retrieve composite procedures requiring multi-object coordination (e.g., handling counts or multiple instances) and ensure coverage of all required objects.| |
|MemGovern<br><br>|48|MemGovern|Retrieve governed experience cards from historical GitHub bug-fixing trajectories to identify relevant prior solution patterns and steps for a given bug-fixing task.<br><br>|48|
|DeepPlanningδ<br><br>|3<br><br>|shopping_level1<br><br>|Retrieve items that satisfy step-level/local constraints.|3|
| | |shopping_level2<br><br>|Retrieve a set of items that jointly satisfy global constraints.| |
| | |shopping_level3|Retrieve items that match the specified conditions.| |

- Table 13: Overview of procedural retrieval tasks with task types and example abilities. † Gorilla has 3 subsets, each corresponding to a single task type; therefore, Gorilla has 3 tasks. ‡ ReMe has 6 subsets: appworld_qwen3_8b, appworld_qwen3_14b, appworld_qwen3_32b, bfcl_qwen3_8b, bfcl_qwen3_14b, and bfcl_qwen3_32b. The first three subsets contain only the generalized_query task type, while the latter three subsets include both generalized_query and task_query. Therefore, ReMe comprises 9 tasks in total. δ DeepPlanning has 3 subsets, each corresponding to a single task type; therefore, DeepPlanning has 3 tasks.

###### Dataset Task Type Instruction

Episodic Memory EPBench Entities Given a specific event with time or location constraints, retrieve documents identifying the protagonists

involved Event_contents Given a query, retrieve documents that answer the query Full_event_details Given time, location, entities, and event content, retrieve relevant documents about the event(s), including

what occurred and any other pertinent information Other_entities Given an event description with time/location and a known entities, retrieve passages describing the event and naming any additional involved entities Spaces Given a query with constraints, retrieve passages that explicitly mention associated locations that satisfy those constraints Times Given a specific event description, retrieve passages that contain when the event occurred

KnowMeBench adversarial_abstention Given a query, retrieve documents that answer the query information_extraction Given a query, retrieve documents that answer the query mind-body_interaction Given a mind-body interaction query, retrieve passages linking physical actions with internal states to answer

the query mnestic_trigger_analysis Given a mnestic trigger query, retrieve passages where sensory or associative cues trigger memories, highlighting the cue, recalled event, and context to answer the query temporal_reasoning Given a temporally constrained query, retrieve passages that answer the question

###### Dialogue Memory

LoCoMo adversarial Given a query, retrieve documents that answer the query multi_hop Given a multi-hop question, retrieve documents from multiple sessions to answer the question open_domain Given a question, retrieve user-provided context or external world-knowledge passages that answer the

question single_hop Given a query, retrieve documents that answer the query temporal_reasoning Given a temporally grounded query, retrieve relevant passages that answer the question

LongMemEval knowledge_update Given a question, retrieve the latest information to answer the question multi_session Given a query, retrieve documents that answer the query single_session_assistant Given a query, retrieve documents that answer the query single_session_preference Given a query, retrieve the user’s stated preferences that can help answer the query single_session_user Given a query, retrieve documents that answer the query temporal_reasoning Given a temporal query, retrieve relevant passages that answer the query

REALTALK commonsense Given a query, retrieve documents that answer the query multi_hop Given a multi-hop question, retrieve relevant passages that answer the query temporal_reasoning Given a temporal query, retrieve relevant passages that answer the query

TMD content_time_qs Given a temporal query, retrieve relevant passages that answer the query date_span_time_qs Given an absolute date span (DATE1–DATE2), retrieve dialogue occurring within this period dates_time_qs Given a specific absolute date, retrieve dialogue from that date day_span_time_qs Given a rolling recent window and the current date, retrieve dialogue that occurred within that period earlier_today_time_qs Given an intra-day time cue and the current date, retrieve dialogue that occurred earlier today last_named_day_time_qs Given a query for a specific weekday with the current date, retrieve dialogue from the most recent matching

weekday month_time_qs Given a specific month and the current date, retrieve dialogue occurring within that month rel_day_time_qs Given a relative-day query and the current date, retrieve dialogue from the resolved date corresponding to

that number of days ago rel_month_time_qs Given a relative-month query with the current date, retrieve dialogue from the month corresponding to that number of months ago rel_session_time_qs Given a relative-session query with current session index, retrieve dialogue from the session corresponding

to that number of sessions ago session_span_time_qs Given a session-span query (S1–S2), retrieve dialogue from sessions that fall within the specified range session_time_qs Given a query for a specific session, retrieve dialogue from that session

MemBench aggregative Given a query, retrieve documents that answer the query comparative Given a query, retrieve documents that answer the query emotion Given a query with a timestamp, retrieve documents that capture emotional cues to infer most likely user

sentiment knowledge_updating Given a question, retrieve the latest information to answer the question multi_hop Given a multi-hop query, retrieve documents that answer the query multi_session_assistant Given a query, retrieve assistant messages containing past recommendations that help answer the query post_processing Given a query, retrieve documents that answer the query preference Given a query, retrieve the user’s stated preferences that can help answer the query single_hop Given a query, retrieve documents that answer the query single_session_assistant Given a query, retrieve assistant messages containing past recommendations that help answer the query

ConvoMem abstention_evidence Given a query, retrieve documents that answer the query assistant_facts_evidence Given a query, retrieve assistant messages that answer the query changing_evidence Given a question, retrieve the latest information to answer the question implicit_connection_evidence Given a query, retrieve documents that can be used to implied answer preference_evidence Given a query, retrieve the user’s stated preferences that can help answer the query user_evidence Given a query, retrieve documents that answer the query

###### Table 14: Task instructions (datasets 1–8 of 22) for LMEB evaluation.

Dataset Task Type Instruction

Semantic Memory

QASPER QASPER Given a query, retrieve documents that answer the query NovelQA Character Given a query, retrieve documents that answer the query

Meaning Given a query, retrieve documents that answer the query Plot Given a query, retrieve documents that answer the query Relation Given a query, retrieve documents that answer the query Setting Given a query, retrieve documents that answer the query Span Given a query, retrieve documents that answer the query Times Given a query, retrieve all passages mentioning the event to answer the query

PeerQA PeerQA Given a query, retrieve documents that answer the query Covid-QA Covid-QA Given a query on COVID-19, retrieve documents that answer the query ESG-Reports ESG-Reports Given a query, retrieve documents that answer the query MLDR MLDR Given a query, retrieve documents that answer the query LooGLE LongDepQA Given a long-dependency question, retrieve documents that answer the query

ShortDepQA Given a query, retrieve documents that answer the query

SciFact SciFact Given a scientific claim, retrieve documents that support or refute the claim Procedural Memory

Gorilla gorilla_huggingface Given a task query, retrieve HuggingFace models to complete it gorilla_pytorch Given a task query, retrieve TorchHub APIs to fulfill the query’s needs gorilla_tensor Given a task query, retrieve TensorHub APIs to meet the query’s requirements

ToolBench ToolBench Given a tool-use query, retrieve API documentations aligned with the needs of the query ReMe generalized_query Given a generalized query, retrieve experience that is suitable for this query

task_query Given a query, retrieve the most relevant experience that aligns with the specified task requirements

Proced_mem_bench easy Given a query, retrieve useful procedures applicable to this query medium Given a query, retrieve useful procedures applicable to this query hard Given a query, retrieve useful procedures applicable to this query

MemGovern MemGovern Given a query, retrieve experiences or solutions that help to solve the query DeepPlanning shopping_level1 Given a query, retrieve items that satisfy the requirements shopping_level2 Given a query, retrieve items that satisfy the requirements shopping_level3 Given a query, retrieve items that satisfy the requirements

###### Table 15: Task instructions (datasets 9–22 of 22) for LMEB evaluation.

- 30

Model EPBench KnowMeBench LoCoMo LongMemEval REALTALK TMD MemBench ConvoMem QASPER NovelQA PeerQA Embedding Models > 1B parameters

KaLM-Embedding-Gemma3 86.31 47.72 37.57 75.72 38.08 21.14 72.77 60.03 32.66 23.64 24.17 bge-multilingual-gemma2 68.94 40.59 40.8 46.69 32.51 18.69 57.87 47.55 32.28 22.59 11.95 Qwen3-Embedding-8B 76.57 45.48 41.26 70.38 41.53 20.15 69.3 51.32 43.12 28.72 31.18 NV-Embed-v2 84.56 56.32 49.65 80.31 46.18 20.78 72.16 69.73 51.06 35.29 30.88 e5-mistral-7b-instruct 68.53 45.88 39.83 69.56 35.61 18.97 67.11 59.91 40.62 25.69 24.79

- Qwen3-Embedding-4B 73.87 42.66 35.52 34.2 36.33 21.06 66.35 53.74 44.95 26.63 28.24 Embedding Models < 1B parameters

jina-v5-text-small 81.24 40.82 39.13 74.31 36.55 23.61 72.22 60.54 49.84 22.39 30.79 Qwen3-Embedding-0.6B 75.22 40.1 38.51 67.6 38.9 25.98 70.46 61.0 40.65 24.92 26.93 multilingual-e5-large-instruct 71.15 43.83 44.53 64.43 38.34 25.33 62.31 53.82 42.6 22.48 28.02 bge-m3 (Dense) 87.46 46.55 46.58 78.37 44.97 28.09 71.23 64.42 51.51 32.22 29.73 KaLM-Embedding-V2.5 82.87 40.85 40.81 68.68 38.29 18.04 69.64 58.26 39.76 25.13 25.64 KaLM-Embedding-V1 75.4 37.8 32.49 42.37 33.59 21.6 66.44 51.41 36.66 19.4 28.97 bge-large-en-v1.5 70.75 40.96 43.76 74.24 47.21 24.01 71.1 58.38 45.46 24.8 27.57 EmbeddingGemma-300M 89.23 47.14 42.15 77.03 41.52 33.94 70.54 58.43 45.16 26.23 29.83 jina-v5-text-nano 78.56 34.9 32.06 58.46 29.35 16.03 61.72 61.68 39.63 22.24 27.9

Table 16: Experiment results (datasets 1–11 of 22) in w/o inst. setting, with metric N@10.

Model Covid-QA ESG-Reports MLDR LooGLE SciFact Gorilla ToolBench ReMe Proced_mem_bench MemGovern DeepPlanning Embedding Models > 1B parameters

KaLM-Embedding-Gemma3 62.13 44.21 72.34 49.46 73.88 46.38 63.3 63.27 51.24 87.24 52.78 bge-multilingual-gemma2 39.7 54.96 61.23 40.72 61.92 31.85 56.17 50.29 52.4 69.01 53.47 Qwen3-Embedding-8B 75.45 52.09 79.14 59.46 74.64 46.88 48.52 63.65 49.17 86.14 47.65 NV-Embed-v2 76.91 49.52 82.06 64.15 83.1 44.01 60.26 65.15 49.96 91.2 51.82 e5-mistral-7b-instruct 50.77 39.37 66.47 40.13 80.06 37.36 55.81 58.35 49.6 80.83 36.23

- Qwen3-Embedding-4B 74.17 44.93 79.78 59.91 71.33 44.73 51.42 63.05 50.32 84.39 44.0 Embedding Models < 1B parameters

jina-v5-text-small 73.29 31.65 76.92 51.67 77.86 36.87 57.95 64.09 47.69 85.32 48.79 Qwen3-Embedding-0.6B 69.81 41.39 78.15 56.12 78.15 41.58 54.47 60.49 51.08 82.27 52.93 multilingual-e5-large-instruct 62.05 31.99 77.83 48.61 73.27 34.92 46.57 60.61 48.82 78.56 36.66 bge-m3 (Dense) 79.42 40.74 81.33 58.93 76.39 34.3 46.31 61.4 51.4 85.99 52.81 KaLM-Embedding-V2.5 69.97 43.12 78.57 56.05 77.75 34.24 53.96 61.11 47.8 72.14 48.55 KaLM-Embedding-V1 58.65 39.34 73.09 50.25 73.59 36.97 51.95 58.73 45.59 84.08 51.82 bge-large-en-v1.5 72.68 41.38 80.57 54.06 73.89 33.68 40.14 62.82 49.78 84.01 45.28 EmbeddingGemma-300M 70.96 42.71 76.75 58.96 78.06 37.01 56.43 63.23 51.66 86.46 49.12 jina-v5-text-nano 69.89 38.04 76.06 52.54 69.64 36.83 52.57 47.91 52.6 67.22 45.76

Table 17: Experiment results (datasets 12–22 of 22) in w/o inst. setting, with metric N@10.

- 31

Model EPBench KnowMeBench LoCoMo LongMemEval REALTALK TMD MemBench ConvoMem QASPER NovelQA PeerQA Embedding Models > 1B parameters KaLM-Embedding-Gemma3 92.98 56.13 53.16 87.66 47.79 21.98 82.36 78.22 53.88 32.89 33.01 bge-multilingual-gemma2 77.23 47.69 54.45 55.62 40.52 20.1 73.45 65.56 56.41 30.66 17.6 Qwen3-Embedding-8B 85.0 52.45 54.8 81.42 50.65 20.18 82.02 68.57 65.85 40.04 42.87

- NV-Embed-v2 91.02 63.78 66.2 90.67 57.34 22.0 83.23 84.63 75.71 46.34 42.64 e5-mistral-7b-instruct 79.37 53.91 54.4 82.66 45.73 20.57 78.19 78.28 63.63 36.34 35.19 Qwen3-Embedding-4B 84.02 50.85 48.34 42.71 43.74 21.28 79.14 71.19 68.81 36.89 39.03

Embedding Models < 1B parameters jina-v5-text-small 89.48 48.63 52.71 85.91 45.75 23.77 82.29 78.13 72.18 31.49 40.41 Qwen3-Embedding-0.6B 83.91 47.56 52.77 78.59 48.92 26.97 81.48 78.08 64.43 34.49 36.82 multilingual-e5-large-instruct 81.48 51.88 60.55 75.12 48.17 26.22 77.34 73.22 64.53 32.72 38.4 bge-m3 (Dense) 93.5 53.69 60.85 90.06 54.0 29.33 82.85 79.54 74.2 42.63 40.62 KaLM-Embedding-V2.5 90.71 48.54 55.6 82.54 47.66 19.14 79.96 74.76 61.4 34.87 37.46 KaLM-Embedding-V1 86.7 45.96 46.15 51.0 42.54 22.28 78.0 69.84 57.68 27.66 41.89 bge-large-en-v1.5 82.54 48.44 58.02 86.54 58.36 24.82 80.51 74.82 67.74 35.07 36.05 EmbeddingGemma-300M 94.86 55.72 57.68 89.03 51.47 35.7 80.56 75.57 68.77 36.57 39.3 jina-v5-text-nano 86.68 42.48 43.03 70.01 39.14 16.86 72.85 77.22 61.43 31.22 37.88

Table 18: Experiment results (datasets 1–11 of 22) in w/o inst. setting, with metric R@10.

Model Covid-QA ESG-Reports MLDR LooGLE SciFact Gorilla ToolBench ReMe Proced_mem_bench MemGovern DeepPlanning Embedding Models > 1B parameters

KaLM-Embedding-Gemma3 89.2 57.41 99.0 66.85 94.29 69.58 71.58 85.53 49.23 95.86 54.68 bge-multilingual-gemma2 68.41 69.54 96.0 59.52 80.61 51.9 64.41 72.9 52.74 80.95 54.27 Qwen3-Embedding-8B 93.34 68.47 99.0 72.39 94.05 70.97 58.99 85.86 49.25 95.16 51.23

- NV-Embed-v2 92.62 60.51 99.0 76.24 96.55 65.01 68.27 86.94 50.57 97.51 55.89 e5-mistral-7b-instruct 76.51 49.91 95.0 57.16 95.66 61.56 64.42 80.67 50.25 92.39 42.08 Qwen3-Embedding-4B 92.17 62.45 98.0 72.58 93.03 68.45 61.06 84.53 50.56 94.33 47.73

Embedding Models < 1B parameters jina-v5-text-small 91.09 47.78 96.0 64.72 94.22 55.48 65.61 85.83 47.03 94.81 50.24 Qwen3-Embedding-0.6B 90.37 56.53 96.0 69.59 94.47 62.24 62.91 83.13 47.84 93.28 53.29 multilingual-e5-large-instruct 84.7 49.31 97.0 63.64 92.73 58.18 52.89 83.71 48.74 90.56 42.29 bge-m3 (Dense) 93.79 57.96 97.0 71.78 92.1 54.37 52.95 82.65 50.27 94.75 53.07 KaLM-Embedding-V2.5 90.19 60.05 97.0 69.76 94.5 55.17 61.58 83.71 48.46 86.15 50.28 KaLM-Embedding-V1 84.43 50.42 94.0 66.21 90.97 59.5 59.8 81.85 44.43 94.16 52.6 bge-large-en-v1.5 90.19 56.57 97.0 68.25 92.69 52.14 48.74 84.98 49.07 93.84 48.55 EmbeddingGemma-300M 90.91 52.78 99.0 72.49 94.86 57.33 64.16 85.22 50.19 95.44 52.18 jina-v5-text-nano 88.21 52.87 97.0 65.76 85.69 58.1 60.72 65.43 51.03 75.69 46.84

Table 19: Experiment results (datasets 12–22 of 22) in w/o inst. setting, with metric R@10.

- 32

Model EPBench KnowMeBench LoCoMo LongMemEval REALTALK TMD MemBench ConvoMem QASPER NovelQA PeerQA Embedding Models > 1B parameters

KaLM-Embedding-Gemma3 89.53 52.26 42.77 81.44 44.38 33.1 73.79 64.08 51.54 33.14 32.53 bge-multilingual-gemma2 88.27 53.5 56.11 81.69 48.64 31.1 72.95 67.11 52.79 37.7 32.79 Qwen3-Embedding-8B 79.32 42.38 36.35 77.59 41.22 28.22 71.31 55.43 47.39 28.06 30.14 NV-Embed-v2 80.43 56.48 51.06 77.34 48.01 19.61 71.45 71.04 57.45 40.67 34.92 e5-mistral-7b-instruct 74.08 47.2 46.22 77.47 42.77 25.63 69.48 68.64 51.04 26.75 32.01 Qwen3-Embedding-4B 73.91 40.57 29.87 47.06 34.48 29.21 66.28 54.59 50.19 25.8 27.15

Embedding Models < 1B parameters jina-v5-text-small 82.79 40.73 39.41 74.63 36.93 22.57 72.67 60.51 50.7 23.33 30.3 Qwen3-Embedding-0.6B 82.11 36.97 34.2 72.77 37.19 27.99 70.52 63.17 47.29 22.64 27.34 multilingual-e5-large-instruct 75.58 43.95 50.07 69.86 44.92 27.07 67.37 68.42 51.25 28.42 32.4 bge-m3 (Dense) 83.19 42.08 41.84 65.58 42.01 24.21 59.38 56.19 44.2 27.19 28.18 KaLM-Embedding-V2.5 85.17 40.23 42.2 74.43 38.96 17.06 69.8 62.24 46.11 27.42 30.46 KaLM-Embedding-V1 79.81 40.62 40.88 75.77 39.82 19.81 67.52 65.78 51.7 27.43 31.95 bge-large-en-v1.5 65.93 26.21 25.44 57.33 34.25 14.88 54.3 34.27 37.29 10.93 18.23 EmbeddingGemma-300M 83.72 43.74 27.61 61.89 29.79 23.31 63.01 49.87 38.24 18.77 23.71 jina-v5-text-nano 78.53 37.48 41.21 69.87 35.51 25.51 68.2 62.68 48.34 28.59 32.14

Table 20: Experiment results (datasets 1–11 of 22) in w/ inst. setting, with metric N@10.

Model Covid-QA ESG-Reports MLDR LooGLE SciFact Gorilla ToolBench ReMe Proced_mem_bench MemGovern DeepPlanning Embedding Models > 1B parameters KaLM-Embedding-Gemma3 78.05 49.49 76.37 58.6 80.53 50.23 64.72 66.39 52.36 91.27 55.63 bge-multilingual-gemma2 79.81 46.2 85.27 64.53 84.14 45.9 62.4 64.97 52.63 88.94 53.56 Qwen3-Embedding-8B 79.38 43.16 77.54 59.53 78.84 45.21 48.6 68.09 51.11 89.35 52.35 NV-Embed-v2 78.58 52.39 85.39 66.05 82.01 43.1 54.38 61.57 52.08 92.5 49.0 e5-mistral-7b-instruct 61.56 49.18 70.73 51.99 81.99 42.84 55.93 61.42 50.71 86.45 40.42 Qwen3-Embedding-4B 76.69 40.77 77.78 58.8 79.0 42.41 51.85 67.0 54.34 89.53 53.75

Embedding Models < 1B parameters jina-v5-text-small 72.96 36.82 76.47 51.74 78.27 33.94 58.26 64.84 49.13 86.6 48.43 Qwen3-Embedding-0.6B 74.2 37.41 76.61 55.55 80.31 42.93 53.71 64.34 53.8 87.82 54.78 multilingual-e5-large-instruct 69.28 35.19 80.93 53.61 79.81 36.1 49.22 62.92 51.77 84.6 48.66 bge-m3 (Dense) 65.12 35.92 76.99 52.29 70.65 28.79 41.46 54.48 44.79 81.75 53.18 KaLM-Embedding-V2.5 73.38 42.98 79.07 57.75 80.56 33.59 54.17 64.15 52.83 84.66 50.96 KaLM-Embedding-V1 74.86 44.03 75.41 57.1 76.02 34.65 57.81 63.27 47.91 86.9 55.66 bge-large-en-v1.5 51.58 32.0 68.83 38.04 69.73 28.75 36.8 56.65 50.17 72.47 43.77 EmbeddingGemma-300M 57.86 40.96 66.64 50.19 65.33 29.0 51.15 62.0 49.68 82.19 45.4 jina-v5-text-nano 76.67 43.38 77.12 58.17 75.49 32.93 54.32 56.07 46.47 87.26 48.55

Table 21: Experiment results (datasets 12–22 of 22) in w/ inst. setting, with metric N@10.

- 33

Model EPBench KnowMeBench LoCoMo LongMemEval REALTALK TMD MemBench ConvoMem QASPER NovelQA PeerQA Embedding Models > 1B parameters KaLM-Embedding-Gemma3 94.82 59.99 58.43 93.02 54.92 33.29 82.72 81.27 73.79 45.12 44.7 bge-multilingual-gemma2 94.38 60.95 70.69 92.68 59.13 32.16 82.61 81.89 75.93 48.22 46.79 Qwen3-Embedding-8B 87.63 49.96 49.6 87.94 50.4 28.61 82.65 72.74 70.52 38.85 41.19 NV-Embed-v2 88.47 63.95 66.61 88.64 57.77 20.69 81.9 85.0 80.1 51.78 44.23 e5-mistral-7b-instruct 83.96 55.19 60.73 88.98 52.17 26.67 79.82 83.11 73.74 37.67 43.19 Qwen3-Embedding-4B 84.6 48.05 41.28 58.74 42.69 30.0 79.03 71.42 72.89 35.79 35.95

Embedding Models < 1B parameters jina-v5-text-small 90.44 48.76 53.13 85.84 45.91 22.91 82.47 78.07 73.12 32.65 40.9 Qwen3-Embedding-0.6B 89.41 44.46 46.82 84.66 46.39 28.44 81.7 79.78 69.72 32.4 37.14 multilingual-e5-large-instruct 85.56 52.02 65.61 80.53 54.47 28.43 81.25 82.84 72.6 37.61 40.58 bge-m3 (Dense) 90.4 49.44 55.73 77.79 52.43 25.64 74.0 70.81 66.44 36.37 37.31 KaLM-Embedding-V2.5 92.02 48.28 56.36 86.64 48.88 18.02 80.08 77.7 68.92 37.3 42.46 KaLM-Embedding-V1 88.2 48.25 54.03 85.86 49.3 21.14 79.45 80.62 73.32 36.62 42.15 bge-large-en-v1.5 79.3 33.31 37.57 71.99 43.75 14.9 68.26 52.72 59.5 17.73 24.34 EmbeddingGemma-300M 90.85 52.15 41.05 75.32 40.21 24.1 74.93 68.54 63.66 28.49 33.4 jina-v5-text-nano 87.31 44.79 54.48 82.03 44.88 26.27 78.38 77.94 72.48 38.39 44.48

Table 22: Experiment results (datasets 1–11 of 22) in w/ inst. setting, with metric R@10.

Model Covid-QA ESG-Reports MLDR LooGLE SciFact Gorilla ToolBench ReMe Proced_mem_bench MemGovern DeepPlanning Embedding Models > 1B parameters KaLM-Embedding-Gemma3 92.98 63.61 100.0 72.84 97.62 72.16 72.91 87.19 51.17 97.55 57.24 bge-multilingual-gemma2 94.78 69.77 98.0 76.12 97.66 67.6 70.36 86.44 55.03 96.26 54.6 Qwen3-Embedding-8B 93.88 63.61 95.0 72.36 95.73 69.55 59.17 88.24 48.78 96.53 54.92 NV-Embed-v2 93.16 70.97 99.0 77.62 97.03 60.31 63.52 82.87 52.99 97.99 50.76 e5-mistral-7b-instruct 83.53 57.87 99.0 66.41 95.89 62.52 64.47 83.61 52.59 95.35 45.48 Qwen3-Embedding-4B 93.25 57.92 96.0 71.61 95.45 63.3 62.21 87.33 53.9 96.81 56.12

Embedding Models < 1B parameters jina-v5-text-small 90.28 53.33 96.0 64.84 93.58 54.04 66.24 86.58 49.28 95.39 50.53 Qwen3-Embedding-0.6B 91.63 52.82 95.0 69.3 96.19 65.31 62.56 85.57 50.31 95.88 54.02 multilingual-e5-large-instruct 89.47 51.39 97.0 67.34 95.38 54.71 56.59 85.44 51.21 94.18 51.78 bge-m3 (Dense) 87.22 50.97 97.0 66.58 88.07 47.42 48.42 76.5 46.05 92.01 53.1 KaLM-Embedding-V2.5 90.55 57.73 97.0 71.57 95.43 53.07 62.76 85.79 53.49 94.18 51.06 KaLM-Embedding-V1 92.53 58.84 96.0 70.75 92.35 52.64 65.42 85.38 48.05 95.73 56.16 bge-large-en-v1.5 78.13 43.89 98.0 53.86 88.46 48.61 44.29 79.49 46.69 85.52 44.91 EmbeddingGemma-300M 86.32 59.63 100.0 66.12 86.68 45.38 59.14 83.77 49.48 93.13 49.53 jina-v5-text-nano 87.31 44.79 54.48 82.03 44.88 26.27 78.38 77.94 72.48 38.39 44.48

Table 23: Experiment results (datasets 12–22 of 22) in w/ inst. setting, with metric R@10.

[Figure 22]

###### Figure 6: Correlation between the scores on LMEB-Episodic and MTEB (eng, v2) (retrieval subset).

[Figure 23]

###### Figure 7: Correlation between the scores on LMEB-Dialogue and MTEB (eng, v2) (retrieval subset).

[Figure 24]

###### Figure 8: Correlation between the scores on LMEB-Semantic and MTEB (eng, v2) (retrieval subset).

[Figure 25]

###### Figure 9: Correlation between the scores on LMEB-Procedural and MTEB (eng, v2) (retrieval subset).

