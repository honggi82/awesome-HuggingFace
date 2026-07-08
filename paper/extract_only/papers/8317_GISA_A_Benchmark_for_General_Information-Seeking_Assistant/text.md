# arXiv:2602.08543v2[cs.CL]13Feb2026

## GISA: A Benchmark for General Information Seeking Assistant

Yutao Zhu1,†, , Xingshuo Zhang1,†, Maosen Zhang1,†, Jiajie Jin1, Liancheng Zhang1, Xiaoshuai Song1, Kangzhi Zhao2, Wencong Zeng2, Ruiming Tang2, Han Li2, Ji-Rong Wen1, Zhicheng Dou1,

†Equal Contribution Affiliation: 1Renmin University of China; 2Kuaishou Technology

The advancement of large language models (LLMs) has significantly accelerated the development of search agents capable of autonomously gathering information through multi-turn web interactions. Various benchmarks have been proposed to evaluate such agents. However, existing benchmarks often construct queries backward from answers, producing unnatural tasks misaligned with real-world needs. Moreover, these benchmarks tend to focus on either locating specific information or aggregating information from multiple sources, while relying on static answer sets prone to data contamination. To bridge these gaps, we introduce GISA, a benchmark for General Information-Seeking Assistants comprising 373 human-crafted queries that reflect authentic information-seeking scenarios. GISA features four structured answer formats (item, set, list, and table), enabling deterministic evaluation. It integrates both deep reasoning and broad information aggregation within unified tasks, and includes a live subset with periodically updated answers to resist memorization. Notably, GISA provides complete human search trajectories for every query, offering gold-standard references for process-level supervision and imitation learning. Experiments on mainstream LLMs and commercial search products reveal that even the best-performing model achieves only 19.30% exact match score, with performance notably degrading on tasks requiring complex planning and comprehensive information gathering. These findings highlight substantial room for future improvement.

Contact: yutaozhu94@gmail.com, dou@ruc.edu.cn Code: https://github.com/RUC-NLPIR/GISA Leaderboard: https://huggingface.co/spaces/RUC-NLPIR/GISA-LeaderBoard

[Figure 1]

[Figure 2]

1 Introduction

With the development of large language models (LLMs), LLM-driven agents have been widely applied in many tasks (Park et al., 2023; Yu et al., 2024; Qian et al., 2024; Li et al., 2025d; Gao et al., 2024). Among these applications, information retrieval stands out as a fundamental need in daily life that has been notably transformed by agent-based systems. Traditional search engines require users to manually formulate queries, navigate through multiple web pages, and synthesize information themselves. This process is often time-consuming and requires significant effort. In contrast, search agents leverage the strong reasoning capabilities of modern LLMs to understand complex user information needs. These agents can autonomously conduct multiple rounds of searching and web browsing to collect relevant information and fulfill user requirements. By automating the process of querying and information gathering, search agents significantly reduce user burden and improve information access efficiency. As a critical tool for information acquisition, search agents have attracted increasing attention from both academia and industry (Li et al., 2025c,b,d), becoming essential components in many research-oriented products and knowledge discovery platforms (such as OpenAI Deep Research and Gemini Deep Research).1

A high-quality benchmark is essential to systematically evaluate the capabilities of search agents and identify directions for further improvement. During the past few years, the research community has proposed various benchmarks to evaluate agents’ abilities to retrieve and synthesize information from the web (Wei et al., 2025; Xi et al., 2025b; Chen et al., 2025; Mialon et al., 2024; Wong et al., 2025). However, upon closer examination, we identify several critical limitations in existing benchmarks that hinder their effectiveness in comprehensively evaluating general-purpose

1OpenAI Deep Research: https://openai.com/index/introducing-deep-research/, Gemini Deep Research: https://gemini.google/overview/ deep-research/.

Table 1 Comparison between GISA with other agentic search benchmarks.

Benchmark Deep Wide Question-driven Human Trajectory Live Answer Type Evaluation

BrowseComp-EN/ZH ✓ × × × × Item LLM InfoDeepSeek ✓ × × × × Item LLM ScholarSearch ✓ × ✓ × × Item LLM GAIA ✓ × ✓ × × Item, List Metric-based Xbench-DeepSearch ✓ × Part of ✓ ✓ Item LLM WideSearch × ✓ ✓ × × Table LLM DeepWideSearch ✓ ✓ Part of × × Table LLM

GISA (Ours) ✓ ✓ ✓ ✓ ✓ Item, Set, List, Table Metric-based

information-seeking agents. First, to increase task complexity or assess agents’ planning capabilities over challenging problems, many existing benchmarks adopt a reverse-engineering approach that constructs queries backward from pre-selected answers (e.g., BrowseComp (Wei et al., 2025)). While this approach can produce difficult tasks, the resulting queries often deviate significantly from authentic human information needs. Some constructed problems may even be unsolvable through a natural forward search process. Consequently, optimizing agent performance on such benchmarks may not translate into improved real-world user experience. Second, existing benchmarks mainly focus on evaluating agents’ ability to perform deep search within the web, i.e., navigating through multiple links and pages to locate a specific piece of information (Wei et al., 2025; Wu et al., 2025; Xi et al., 2025b). However, real-world information needs often require not only depth but also breadth, where agents must gather and aggregate information from diverse sources. Although recent efforts such as WideSearch (Wong et al., 2025) have begun to explore the evaluation of broad information collection capabilities, achieving a balanced and unified evaluation of both deep and wide search remains an open challenge. Third, search serves as a primary means for humans to acquire information, particularly timely and up-to-date knowledge. Ideally, benchmarks should evolve alongside the information landscape to remain relevant. Nevertheless, most existing benchmarks, for the sake of evaluation convenience and reproducibility, rely on questions with long-term stable answers (Xi et al., 2025b). As LLMs are continuously trained on increasingly recent data, such static benchmarks may become out-of-date, as models may have already memorized the answers during pre-training, thereby failing to genuinely test their search capabilities.

To address these limitations, we propose GISA, a benchmark for General Information-Seeking Assistants containing carefully crafted queries along with their answers and human-annotated search trajectories. GISA distinguishes itself through the following key features:

- • Diverse answer formats with deterministic evaluation: GISA formulates answers into four structured types: items, sets, lists, and tables. This design enables deterministic and reproducible evaluation using strict matching metrics, avoiding the subjectivity and instability of LLM-based judgment while preserving task complexity and diversity.
- • Unification of deep and wide search capabilities: Real-world information seeking often requires both exploring deep reasoning paths and aggregating information across diverse sources. GISA integrates both dimensions, evaluating an agent’s ability to perform vertical investigation and horizontal summarization within complex, long-horizon tasks.
- • Dynamic and anti-static evaluation: To prevent data contamination from pre-training memorization, GISA categorizes queries into stable and live subsets. The live subset requires accessing real-time information and is periodically updated, ensuring the benchmark remains challenging and resistant to memorization over time.
- • Process-level supervision via human trajectories: Beyond question-answer pairs, GISA provides complete human search trajectories for every query. These trajectories serve as gold standards for process reward modeling and imitation learning, while also verifying that all tasks are solvable through realistic search behaviors.

We construct GISA through a rigorous multi-stage annotation process involving question design, answer labeling, and human trajectory collection, with each stage undergoing multiple rounds of quality verification. On average, each query requires over one hour of human effort from initial design to final validation. The final benchmark comprises 373 high-quality queries spanning diverse domains and complexity levels. To facilitate future research, we release comprehensive documentation, evaluation scripts, and a public leaderboard open for submission.

We evaluate a diverse set of state-of-the-art LLMs and commercial search products on GISA. Results show that even the best-performing model achieves only 19.30% overall exact match, with commercial deep research systems struggling to outperform LLM-based agents. These findings underscore the challenging nature of GISA and highlight significant opportunities for advancing general-purpose information-seeking agents.

### 2 Related Work

Search Agent. LLM-driven search agents have evolved beyond early retrieval-augmented generation into complex systems capable of autonomous planning and dynamic decision-making (Zhu et al., 2023; Xi et al., 2025a). To meet diverse information needs, current research employs structures ranging from breadth-oriented parallel search to depth-oriented serial reasoning (Li et al., 2025c; Ko et al., 2025; Li et al., 2025a), while further integrating complex architectures like knowledge graph, Monte Carlo Tree Search (MCTS), and multi-agent collaboration (Lu et al., 2025; Ren et al., 2025; Huang et al., 2025). In terms of optimization, combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) significantly enhances agent planning and tool usage in dynamic environments, and drives the emergence of applications centered on web search and research (Lin et al., 2025; Li et al., 2025d; Jin et al., 2025; Li et al., 2025b). Despite these advances, current evaluation systems remain limited by unrealistic task construction, a lack of unified assessment for deep reasoning and breadth aggregation, over-reliance on unstable LLM-based scoring, and data contamination in static benchmarks. Consequently, accurately measuring comprehensive agent performance in real-world, dynamic web environments remains a challenge.

Benchmark for Agentic Search. The rapid advancement of search agents has given rise to a variety of benchmarks, as shown in Table 1. Benchmarks including BrowseComp (Wei et al., 2025), InfoDeepSeek (Xi et al., 2025b), and xbenchDeepSearch (Chen et al., 2025) focus on deep search, evaluating multi-hop reasoning and cross-page information integration (Mialon et al., 2024; Xu et al., 2025; Pham et al., 2025; Chen et al., 2025; Zhou et al., 2025c,a,b). However, tasks in these datasets are often reverse-engineered from answers; while challenging, they may deviate from real user queries. Conversely, benchmarks like WideSearch (Wong et al., 2025) emphasize reliability in large-scale atomic information collection and structured organization. While DeepWideSearch (Lan et al., 2025) attempts to evaluate both depth and breadth by combining BrowseComp and WideSearch, it still lacks task-driven construction and the support of real user behavioral trajectories. Moreover, most existing benchmarks rely on static snapshots, making it difficult to discern whether an agent acts based on web information or its own internal knowledge. In comparison, GISA provides a realistic, dynamic, and process-interpretable benchmark. To ensure tasks are meaningful and aligned with actual information needs, GISA centers on human-designed tasks covering both deep and wide information seeking. To prevent data contamination, GISA maintains queries involving live and evolving information. Finally, GISA provides complete human search trajectories, enabling fine-grained process analysis and serving as references for training agents to emulate human search strategies.

### 3 Benchmark In this section, we detail the design and construction process of GISA, followed by the evaluation protocol.

- 3.1 Construction

The construction of GISA is a human-centered process designed to ensure that each query is challenging, realistic, and strictly aligned with the goal of evaluating real-world information-seeking capabilities. The entire workflow, ranging from initial question design to final inclusion, is governed by rigorous criteria and multi-stage verification. As illustrated in Figure 1, the construction pipeline comprises four distinct stages: (1) brainstorming, (2) query refinement, (3) human annotation, and (4) quality checking.

- 3.1.1 Brainstorming

To ensure the benchmark covers a diverse range of topics that truly reflect human interests, we adopt the taxonomy from BrowseComp (Wei et al., 2025). This taxonomy categorizes information needs into ten distinct domains: TV Shows & Movies, Science & Technology, Art, History, Sports, Music, Video Games, Geography, Politics, and Other. The distribution of topics in GISA is illustrated in Figure 2.

##### (1) Brainstorming

##### (2) Query refinement

##### (3) Human annotation (4) Quality checking

Browsing & Answer checking

Q1: Provide a table listing all presidents in South Korean history, featuring the following columns: 'President Name', 'Start Date' (format: MM-DD-YYYY), 'End Date' (format: MM-DD-YYYY), and 'Martial Law' (Yes/No, indicating whether martial law was declared during their tenure). The table should be sorted in ascending chronological order by the 'Start Date'.

Trajectory: [Query] list of presidents in South Korean [1] List of presidents of South Korea Wikipedia Answer (Table):

[Figure 3]

Topic-related web pages

[Figure 4]

[Figure 5]

en.wikipedia.org/wiki/List_of_presidents_of_...

naijadetails.com/presidents-of-south-korea/

nytimes.com/2024/12/03/world/asia/ma...

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

bbc.com/news/articles/c86py30qezvo

[Figure 10]

[Figure 11]

[Figure 12]

###### Name Start Date End Date Martial Law

[Figure 13]

##### Trajectory:

[Figure 14]

[Figure 15]

Political

Rhee Syng-man 07-24-1948 04-27-1960 Yes Ho Chong 04-27-1960 06-15-1960 No Kwak Sang-hoon 06-16-1960 06-23-1962 No

[Figure 16]

[Figure 17]

<query> list of presidents in South Korean <results> [1]…[2]…[3]… <clicks> [from]…[to]…[from]…[to]…

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

|Seed question 1: How many presidents have there been in South Korean history, and which of them declared martial law during their tenure?<br><br>[Figure 23]|
|---|

… … …

[Figure 24]

##### Polish queries, and clarify Fix and adjust format and sorting

[Figure 25]

[Figure 26]

##### Answer (Table):

Trajectory: [Query] martial law in Korea history

[Figure 27]

Name Start Date End Date Martial Law Rhee Syng-man 07-24-1948 04-27-1960 Yes Ho Chong 04-27-1960 06-15-1960 No Kwak Sang-hoon 06-16-1960 06-23-1962 No … … … Yoon Suk Yeol 05-10-2022 12-14-2024 No Choi Sang-mok 12-27-2024 03-24-2025 No Han Duck-soo 03-24-2025 05-01-2025 No Lee Ju-ho 05-02-2025 06-04-2025 No Lee Jae Myung 06-05-2025 - No

Q2: Provide a comprehensive table detailing the declarations of Martial Law or States of Emergency in South Korean history, listing the following columns: 'Date Declared' (format: MM-DD-YYYY) and 'Initiator/Authority' (e.g., President, Military). The table should be sorted in ascending chronological order by the 'Date Declared'.

- [1] Korea's History of Martial Law
- [2] South Korea's long history of martial law [Query] martial law Syngman Rhee

[Figure 28]

Wrong!

|Seed question 2: What is martial law? What is the history of martial law in South Korea?<br><br>[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

Wrong!

- [1] April Revolution - Wikipedia

- [2] History of Korea's martial law

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Missing rows!

[Figure 39]

[Figure 40]

[Figure 41]

Cannot find the detailed information about the martial law during Syngman Rhee!!

[Figure 42]

Seed question collecting

Return

##### Answer type: Set

Answer type: Item Query Please provide a list of all the confirmed bands and solo artists who performed at the "Woodstock Music and Art Fair 2023" rock festival hosted in South Korea in 2023. The list must only contain the name of the Band or Solo Artist.

Query Since 2008, which scholar has won two or more best paper (full paper) awards at the ACL conference?

Live Stable

Answer type: List Query Identify the artist who won the Archibald Prize the most times in the 20th century (1901–2000), and present a table listing their winning works with the following three columns: 'Artist Name', 'Year' (as YYYY), and 'Title'. The table should be sorted in ascending order by the year of the win.

Answer type: Table

Query Please provide an alphabetically sorted list by last name of all the faculty members in CMU's Department of Statistics & Data Science.

Live Stable

Example queries with various types of answers

Figure 1 Illustration of benchmark construction process and some example queries with various types of answers.

Based on these categories, we employ an open-ended browsing strategy to stimulate query generation. Annotators are instructed to visit domain-specific websites, such as news aggregators for Politics or encyclopedic archives for History, and freely explore content. During this process, they are encouraged to document any questions that arise spontaneously from the information they consume. This approach mimics the natural cognitive process of “encountering information and seeking further knowledge” (Kidd and Hayden, 2015). For instance, as illustrated in Figure 1, an annotator browsing political news might encounter a report regarding President Yoon Suk Yeol and the declaration of martial law. This stimulus could trigger specific inquiries, such as What is the definition of martial law? or Are there historical precedents for martial law in South Korea? By systematizing this browsing-inspired workflow across all ten domains, we successfully obtain a large collection of seed questions.

- 3.1.2 Query Refinement

After obtaining a diverse pool of seed questions, the next step is to transform these raw questions into formal and structured queries. This stage is critical for realizing two core design goals of GISA: enabling deterministic evaluation through structured answer formats, and unifying deep and wide search capabilities within realistic tasks.

In this stage, annotators formulate the final query while determining the optimal format of the target answer. We categorize the target answer structures into four types: item, set, list, and table (example queries are shown in the bottom side of Figure 1). This structured design enables deterministic and reproducible evaluation using strict matching metrics, avoiding the subjectivity of LLM-based judgment. If a query is compatible with multiple formats, annotators are allowed to select the most appropriate format. Besides, to ensure the uniqueness of search trajectories in the dataset, we adopt a strict one-to-one rule: each seed question is used to construct exactly one formal query.

Crucially, this formulation process involves a preliminary feasibility check that ensures queries require genuine search effort spanning both depth and breadth. Annotators are required to perform a quick search to verify that the target answer is not readily available in a pre-aggregated form on existing web pages. For example, rather than simply

12

[60, +∞)

Video Games (n=41)

Timeforhumantosolve(mins)

11.0%

11.3%

Others (n=39) Politics (n=29) Sports (n=42)

9

[50, 60)

10

- 7.0%
- 8.6%
- 9.1%

[40, 50)

- 10.5%

7.8%

- 11.3%

20

[30, 40)

Science & Technology (n=44)

41

[20, 30)

Geogrphy (n=44)

History (n=26)

100

[10, 20)

Art (n=32)

181

[0, 10)

11.8%

Music (n=34)

11.8%

0 50 100 150 200

TV Shows & Movies (n=42)

Number of questions

Figure 2 Distribution of topics in GISA.

Figure 3 Statistics of human annotation time on GISA.

asking for the “total count of South Korean presidents”, a fact often directly displayed in search snippets, we instead request a structured table listing all presidents along with whether each declared martial law during their tenure (as illustrated in Figure 1). This task exemplifies the integration of deep and wide search: the agent must first gather basic information about all presidents (broad aggregation), then investigate each president’s historical record to determine martial law declarations (deep reasoning for individual verification). Such queries cannot be answered by retrieving a single pre-existing source, but require systematic collection and synthesis across multiple pages. This filtering step eliminates trivial queries and ensures that GISA evaluates agents’ ability to perform both vertical investigation and horizontal summarization within unified tasks.

Furthermore, we address the temporal nature of information needs. Annotators must classify each query as either stable (immutable for at least three years) or live (subject to change). For the latter, we commit to a monthly maintenance schedule, updating the ground-truth regularly to ensure the benchmark remains accurate over time.

Beyond general formulation, we impose strict constraints for structured output formats to ensure the evaluation is deterministic: (1) For list-type queries, the sorting criterion must be explicitly defined in the prompt (e.g., “sorted alphabetically by name”) to avoid ambiguity in the output order. (2) For table-type queries, the schema must be fully specified. Annotators must define the exact column names to be retrieved and designate a primary column for sorting. To handle duplicate values in the primary column, multi-level sorting criteria (i.e., secondary and tertiary keys) must be provided as tie-breaking rules, applied sequentially until a unique row order is guaranteed.

- 3.1.3 Human Annotation

Prior to the official annotation, a pilot phase is conducted using five trial questions. This phase serves to train annotators on tool usage and align their outputs with our strict formatting specifications (e.g., ensuring column headers match the table schema, and verifying sorting logic). Feedback is provided iteratively during this phase. To maintain data integrity, these pilot samples and their ground-truths are excluded from the final dataset.

To capture the fine-grained decision-making processes of human searchers, we develope a custom browser extension. This tool operates in the background to log interaction data, including: (1) search queries issued to Google; (2) content of search engine results pages; (3) click-through behaviors; and (4) precise timestamps for duration analysis.

The annotation workflow follows a standardized protocol:

- (1) Initialization: The annotator activates the extension to begin a session for a specific query.
- (2) Search and restrictions: Annotators are restricted to using Google Search as their retrieval tool.2 While they may navigate freely through resulting web links, the use of other search engines or LLMs is prohibited.
- (3) Exception handling: In cases where the target information is inaccessible, or where sources exhibit irreconcilable conflicts, annotators are instructed to document the specific issue and skip the query. These flagged queries are collected and returned to the query refinement stage for revision.

2Given that Google Search may occasionally trigger automatic AI-generated summaries, annotators are explicitly instructed to ignore these modules. They are strictly prohibited from viewing or utilizing any information provided by these built-in LLMs. This constraint ensures that the recorded trajectories reflect a complete and organic human information-seeking process rather than the passive consumption of AI-synthesized answers.

- Table 2 Statistics of GISA. The input length is measured by the average number of tokens, while the output length is measured by the average number of items.

Answer Type # Stable # Live Input Len Output Len

Item 12 10 42.96 1.00 Set 33 17 43.40 16.90 List 30 18 54.23 13.67

Table 148 105 98.45 45.98

- (4) Answer construction: Annotators use spreadsheet software to organize intermediate findings and format the final answer.
- (5) Submission: Upon completion, the final answer is saved as a CSV file, and the extension exports the search trajectory as a JSON log. Finally, we employ a post-processing script to parse these logs into the standardized trajectory format (as shown in Appendix A). We present the statistics of human annotation time on GISA in Figure 3.

- 3.1.4 Quality Checking

In the final phase, the triplet of query, search trajectory, and answer is distributed to a dedicated team of verifiers. This stage involves three rigorous validation steps:

- (1) Verifiers first inspect the consistency of the recorded search logs. They screen for two common issues: (a) missing initial queries, where the log fails to capture the start of the session because the annotator performs the first search before activating the tool; and (b) noise injection, where irrelevant queries or browsing behaviors are recorded because the annotator fails to terminate the session immediately after the task. Trajectories exhibiting these issues are flagged for re-annotation.
- (2) Upon clearing the trajectory check, verifiers evaluate the correctness of the final answer. They are granted the autonomy to conduct independent web searches, referencing the recorded trajectory at their discretion, to validate the facts. Beyond factual accuracy, verifiers strictly enforce compliance with the pre-defined format (e.g., specific table schema) and sorting constraints. If errors are identified (e.g., citation failures, missing entries, or hallucinations), the answer undergoes revision. A critical constraint is applied during this correction: the revised answer must be fully deducible from the information present in the original search trajectory. If the original trajectory is insufficient to support the corrected answer, the instance is deemed invalid and discarded, triggering a re-annotation of the query.

Finally, to ensure that the benchmark evaluates search capabilities rather than the parametric memory of LLMs, we conduct a memorization check using the DeepSeek-V3.2 model (DeepSeek-AI, 2025). We feed the queries into the model with its reasoning and web search capabilities disabled. If the model can answer a query perfectly using only its internal knowledge, the query is considered “solved” by current pre-training data and is consequently excluded from the dataset. This ensures that GISA focuses exclusively on tasks requiring external information retrieval. After filtering, we obtain a final set of 373 queries, with detailed statistics reported in Table 2.

Throughout the annotation process, we recruit 15 graduate students specializing in information retrieval as expert annotators. Their domain knowledge ensures proficiency in formulating effective search strategies and synthesizing information from multiple sources. The entire pipeline, from initial question design to final answer validation, requires over one hour per query on average, reflecting the complexity and rigor of our annotation process. Figure 3 provides detailed statistics of the time spent on the human annotation stage (i.e., trajectory collection and answer labeling).

- 3.2 Evaluation

- 3.2.1 Data Preprocessing

To facilitate automated evaluation, we incorporate specific output format constraints within the agent’s instructions, requiring the final results to be encapsulated in TSV format within Markdown code blocks. We employ regular expressions to target and extract content from these blocks in the agent’s final response. To ensure parsing robustness,

the extraction pipeline automatically filters out redundant empty lines and handles cases where the code block may be missing by treating the entire response as raw content for fallback processing.

Both the extracted predictions and the ground-truth answers undergo a uniform normalization procedure to eliminate the impact of formatting discrepancies. This process begins with standardizing column headers through lowercase conversion and whitespace removal. For cell values, we implement a multi-tiered cleaning logic: numeric entries are stripped of currency symbols, commas, and percentage signs, with the latter being converted to decimal equivalents. These values are then represented as strings, where integers are formatted directly and floating-point numbers are rounded to a maximum of six decimal places with trailing zeros removed. Additionally, string-based entries are normalized to lowercase and stripped of interior spaces, newlines, and Markdown artifacts such as asterisks. Finally, all null or missing values are represented as empty strings, ensuring a consistent and objective comparison at the cell level.

- 3.2.2 Metrics

Since our benchmark contains various types of answers (i.e., item, set, list, and table), we employ specific metrics tailored to each type to comprehensively evaluate the performance.

Universal Metric: Exact Match First, we apply exact match (EM) as the strictest evaluation metric across all answer types. This metric assigns a score of 1 only if the generated result is completely identical to the ground truth; otherwise, the score is 0.

Metrics for Specific Answer Types Depending on the answer type, we use additional metrics to capture different aspects of the model’s performance:

- (1) Set-type: For answers where the output is a collection of items but the order does not matter, we use the F1 score to evaluate the overlap between the predicted set and the ground-truth set, allowing us to assess both precision and recall.
- (2) List-type: For ordered lists, both the content accuracy and the sequence of items are crucial. We initially consider using ranking-biased overlap (RBO) (Webber et al., 2010) to evaluate ranking quality. However, RBO cannot properly handle cases where the generated results contain duplicate items, which occasionally occurs in our task. Therefore, we adopt a two-metric approach that separately evaluates content accuracy and order accuracy. For content accuracy, we still use the standard F1 score, which measures the precision and recall of items in the predicted list compared to the ground truth, regardless of their order. For order accuracy, we employ the Sequence Matcher algorithm, which computes an order-aware similarity score.3 Given a ground-truth list and a predicted list, the order score is calculated as: Order Score = 2M/T, where M represents the number of matching elements (considering both content and position), and T is the total number of elements in both lists. The matching process identifies the longest contiguous matching subsequences and recursively applies to remaining portions. The order score ranges from 0 to 1, where 1 indicates perfect match in both content and order.
- (3) Table-type: Following the protocols of previous wide search benchmarks (Wong et al., 2025), we evaluate tabular data at two granularities. We use row-level F1 to measure the accuracy of entire rows and item-level F1 to assess the correctness of individual cell values.
- 4 Experiments

- 4.1 Baseline Methods & Settings

We evaluate two paradigms of search agents on GISA: ReAct-based LLM Agents and Commercial Agent Systems. To ensure fair comparison, all agents receive prompts containing identical task descriptions, formatting constraints, current date information, and target queries. ReAct-based agents are additionally provided with tool-use specifications following the standard ReAct (Yao et al., 2023) workflow.

- (1) ReAct-based LLM Agents. We equip these agents with two core tools: Search and Browse, powered by the Google Serper API and Jina API, respectively.4 Following previous studies (Li et al., 2025b,d), content retrieved via Browse is

3difflib.SequenceMatcher, https://docs.python.org/3/library/difflib.html 4Serper: https://serper.dev/, Jina: https://jina.ai/reader

- Table 3 Experimental results of all methods on GISA. The overall score is calculated by the weighted average of all EM scores. Tool calling number with ∗ is provided by the API service provider.

Item Set List Table Overall Avg. # Tool Calls EM EM F1 EM F1 Order EM Row-F1 Item-F1 EM Search Browse

Model / System

LLM-based ReAct Agents Qwen3-235B-A22B (thinking) 40.91 18.00 52.37 14.58 36.48 35.96 4.35 28.32 43.93 9.65 2.16 4.03 Claude 4.5 Sonnet (non-thinking) 59.09 26.00 60.87 22.92 58.76 57.78 9.49 47.85 63.71 16.36 10.11 5.67 Claude 4.5 Sonnet (thinking) 63.64 28.00 64.86 22.92 59.24 56.42 13.04 49.92 65.17 19.30 7.57 4.63 Gemini 3 Pro (low) 45.45 28.00 63.82 27.08 57.55 56.37 7.11 45.93 64.93 14.74 9.13 6.69 Gemini 3 Pro (high) 50.00 22.00 62.66 27.08 60.87 60.12 8.70 47.01 66.02 15.28 11.87 5.79 GPT-5.2 (thinking) 63.64 26.00 62.70 16.67 54.11 53.17 9.49 43.04 60.20 15.82 8.14 13.78 DeepSeek-V3.2 (non-thinking) 22.73 20.00 52.00 22.92 56.02 55.45 6.72 44.14 62.24 11.53 12.62 10.18 DeepSeek-V3.2 (thinking) 63.64 28.00 60.79 20.83 62.25 60.41 6.32 43.44 62.42 14.47 12.14 12.35 GLM-4.7 (thinking) 50.00 22.00 59.44 20.83 51.99 50.97 8.30 43.97 61.28 14.21 15.75 12.26 Seed-1.8 (thinking) 45.45 32.00 56.77 16.67 56.11 53.54 6.32 38.49 57.13 13.40 7.94 4.14 Qwen3-Max (thinking) 59.09 30.00 63.45 25.00 66.51 64.08 10.67 48.48 66.86 17.96 11.50 8.94 Kimi K2.5 (thinking) 68.18 28.00 61.71 18.75 50.52 48.81 7.91 45.19 61.23 15.55 13.02 7.89

Commercial Agent Systems

GPT-4o Search Preview 13.64 4.00 38.70 8.33 36.65 36.00 4.74 29.59 45.61 5.63 1.00∗ OpenAI o4 Mini Deep Research 18.18 14.00 63.03 18.75 53.72 52.59 3.56 36.78 56.47 7.78 69.88∗ Perplexity Sonar Pro Search 22.73 20.00 47.04 6.25 34.74 33.16 3.95 34.76 49.05 7.51 - Google Search AI Mode 31.82 20.00 46.34 8.33 40.64 39.36 5.53 31.15 50.79 9.38 - -

summarized before being returned to the model to minimize noise. For this summarization task, we use the same underlying LLM as the agent, operating in non-thinking mode (when applicable). To leverage the inherent capabilities of each model, we utilize their native function-calling interfaces to construct the agents. For models supporting thinking mode, we employ an interleaved reasoning approach, allowing the model to perform multiple tool calls within a continuous thinking path. We set a maximum of 30 tool invocations per task, 8,192 output tokens per step, and disable parallel function calling for fair comparison. A retry strategy is implemented to ensure execution stability.

Our evaluation covers a diverse suite of state-of-the-art models: Qwen3-235B-A22B (Yang et al., 2025), Claude 4.5 Sonnet, Gemini 3 Pro, GPT-5.2, DeepSeek-V3.2 (DeepSeek-AI, 2025), GLM-4.7 (Zeng et al., 2025), Seed-1.8, Qwen3-Max (Yang et al., 2025), and Kimi K2.5. Unless otherwise specified, the native thinking mode is enabled by default with the maximum thinking budget allocated. We utilize official APIs for DeepSeek-V3.2, Qwen-Max, and Kimi K2.5, while all other models are accessed via the OpenRouter platform.5

- (2) Commercial Agent Systems. In addition to custom implementations, we evaluate closed-source commercial systems to establish a baseline for industrial-grade performance. These include DeepSearch systems (GPT-4o Search Preview, Perplexity Sonar Pro Search, and Google Search AI Mode) and DeepResearch systems (OpenAI o4-mini DeepResearch). Google Search AI Mode results are manually collected and converted to CSV format; all other systems are accessed via OpenRouter APIs. More implementation details can be found at Appendix B.

- 4.2 Main Results

- Table 3 presents the performance of various LLM-based agents and commercial search systems on GISA. We highlight several key findings from our evaluation.

- (1)Significantroomforimprovementremains. Even the best-performing model, Claude 4.5 Sonnet (thinking), achieves only 19.30% overall EM score, indicating that current search agents are far from solving complex information-seeking tasks reliably. Through manual inspection of search trajectories, we identify several recurring failure modes: (a) limited problem decomposition capabilities, where agents fail to devise effective search plans for complex queries; (b) insufficient self-correction abilities, where agents struggle to adjust strategies based on intermediate search results;

5OpenRouter: https://openrouter.ai/

and (c) inadequate web traversal skills, where agents fail to exploit hyperlinks within pages, leading to incomplete information gathering.

- (2) Task complexity scales with information breadth. Performance degrades substantially as the required information scope increases. Agents perform reasonably well on item-type questions but struggle significantly on table-type tasks, which demand collecting and organizing information across multiple dimensions. Notably, in GISA, the amount of information to be gathered does not necessarily correlate with question difficulty, i.e., an item-type question may still require extensive search before arriving at the correct answer. The pronounced performance drop on structured answer formats (lists and tables) suggests that current models face challenges not only in information collection but also in answer organization and formatting.
- (3) Efficient tool usage matters. Interestingly, the best-performing Claude 4.5 Sonnet model uses tools quite efficiently, with moderate numbers of search and browse calls compared to other models. In contrast, models like DeepSeek-V3.2 and GLM-4.7 invoke substantially more tools yet achieve lower scores. This suggests that excessive tool usage does not necessarily improve performance; rather, noise from irrelevant retrieved content and increased context length may negatively impact reasoning quality.
- (4) Reasoning mode provides consistent gains. Comparing thinking and non-thinking variants of the same model family reveals clear benefits from extended inference-time computation. Claude 4.5 Sonnet improves from 16.36% to 19.30% overall EM, and DeepSeek-V3.2 improves from 11.53% to 14.47% when thinking mode is enabled. However, these gains come at the cost of significantly higher token consumption, presenting a trade-off between performance and efficiency.
- (5) Commercial search systems underperform. Surprisingly, commercial deep research and search products do not outperform LLM-based ReAct agents. GPT-4o Search Preview, which performs only a single retrieval per query, is clearly insufficient for the complex tasks in GISA. OpenAI o4 Mini Deep Research and Perplexity Sonar Pro Search, despite their sophisticated pipelines, suffer from poor instruction-following, resulting in numerous formatting errors in their final answers. Google Search AI Mode demonstrates relatively better performance among commercial systems; while it lags behind LLM-based agents in accuracy, it offers substantial advantages in response speed, positioning itself closer to traditional search engine experiences.

- 4.3 Further Analysis We conduct further analyses to better understand model behavior and identify potential directions for improvement.

Comparison with Human Search Behavior. To better understand the gap between model and human search strategies, we compare the behavioral patterns of Claude 4.5 Sonnet (thinking) with human annotations, analyzing the average number of search queries and browsed pages, query refinement patterns, and the correlation between human-model similarity and task performance (detailed metrics for all models are provided in Appendix C). We observe that humans and models exhibit different strategies: humans issue fewer queries (3.53 on average) but browse substantially more pages (19.03), whereas models search more frequently (7.57 queries) but browse far fewer pages (4.63). This suggests that humans favor thorough exploration within search results, while models rely on repeated querying rather than deep examination of retrieved content. Additionally, humans show higher adjacent query overlap (0.31 vs. 0.22), indicating more targeted query refinement, whereas models tend to construct less related queries between consecutive searches. We also find a positive correlation between human-model behavioral similarity and task performance: the high-similarity group achieves an average F1 of 0.76, compared to 0.56 for the low-similarity group, and successfully solved cases exhibit higher URL overlap rates (0.31 vs. 0.15). These findings suggest that aligning model behavior more closely with human search strategies, particularly in terms of deeper content exploration and more coherent query refinement, may be a promising direction for improving agent performance.

Performance Difference across Question Types. We analyze model performance across stable and live subsets to explore the potential impact of data contamination. Intriguingly, Kimi K2.5, the most recently released (Jan. 28, 2026) model in our evaluation, achieves notably lower performance on the live subset compared to the stable subset (11.33% vs. 18.39% overall EM), while other models (Claude 4.5 Sonnet) show no significant difference. We hypothesize that this difference arises because Kimi K2.5’s training data, being the most recent, is more likely to contain answers to stable questions. In contrast, older models may not have memorized either subset, making their performance uniformly dependent on search capabilities. This finding validates the design of our live subset: as models are trained on increasingly current data, static benchmarks risk measuring memorization rather than search ability. The live subset,

Best@k Majority@k

Kimi K2.5 Stable Kimi K2.5 Live Claude Stable Claude Live

25

83.33

22.22

90

| | | |20.|00| | | |
|---|---|---|---|---|---|---|---|
| |17.|78| | | | | |
|8.90<br><br>13.|30<br><br>13.|33|15.|60| | | |
|8.90<br><br>11.|10| | | | | | |
| | | | | | | | |

75.00

80

20

70

OverallEMScore

60

50.00

50.00

17.50

15

50

EM

40

30.30

30.30

10

23.53

23.53

23.33

22.22

30

20.00

16.67

14.29

12.16

20

10.14

5

4.76

10

0

0

0 2 4 6 8 10 12 14 16

Item Set List Table

Answer Type

k

Figure 4 Performance of Kimi K2.5 (thinking) and Claude 4.5 (thinking) on different subsets of GISA.

Figure 5 Inference-time scaling performance of Qwen3-Max over 40 random samples.

with its periodically updated answers, provides a more robust evaluation that remains resistant to such contamination over time.

Inference-time Scaling. Recent research has shown that allocating more computational resources at inference time can effectively improve model performance (DeepSeek-AI, 2025; Wong et al., 2025; Wei et al., 2025). To investigate whether search agents benefit from such scaling, we conduct experiments on 40 randomly sampled queries using Qwen3-Max, generating k independent runs per query (k ∈ {1,2,4,8,16}). We report Best@k (whether any attempt succeeds) and Majority@k (confidence-weighted voting). As shown in Figure 5, both metrics improve consistently as k increases. Best@k rises from 8.90% to 22.22% at k=16, a 2.5× improvement, suggesting that models have potential capabilities not reliably activated in single attempts. Majority@k also improves but consistently lags behind Best@k, indicating that selecting the correct answer from multiple candidates remains challenging. These results demonstrate that inference-time scaling offers a promising direction for improving search agents, while also highlighting the need for better answer verification mechanisms.

Error Analysis. To gain deeper insight into the failure modes of current search agents, we manually analyze 50 error cases from the best-performing model (Claude 4.5 Sonnet with thinking). As shown in Figure 6, we categorize the identified errors into three levels based on where they occur in the search pipeline (noting that a single sample may exhibit multiple error types). (1) At the comprehension level, query misunderstanding accounts for only 3.2% of errors, indicating that current LLMs possess strong semantic understanding capabilities.

|3%<br><br>14%<br><br>18%<br><br>17%<br><br>16%<br><br>32%<br><br>3%<br><br>49%<br><br>48%<br><br>Query understanding<br><br>Query formulation<br><br>Link utilization<br><br>Conflict resolution<br><br>Information extraction<br><br>Instruction following<br><br>Comprehension level<br><br>Output level<br><br>Search level<br><br>|
|---|

- (2) The majority of failures occur at the search level, comprising 49.2% of all errors. These include inability to formulate effective queries (14.3%), failure to exploit hyperlinks within pages for deeper exploration (17.5%), and inability to initiate verification queries when encountering conflicting information across sources (17.5%). These errors suggests that current agents lack effective strategies for thorough web exploration and information verification.
- (3) At the output level, we observe failures in accurate information extraction from retrieved pages (15.9%) and instruction-following errors (31.7%), such as generating incorrect table headers. Notably, instruction-following errors constitute the largest single category, which is consistent with our main finding that commercial systems often struggle with output formatting.

Figure 6 Illustration of different error types and their ratios.

Cost Analysis. Table 4 presents the average token consumption and cost per query for each model evaluated on GISA. We report the number of input and output tokens in the main reasoning chain, tokens consumed by the browsing tool, official API prices, and the resulting average cost per query. Based on the results, we have several observations. First, Claude 4.5 Sonnet, despite achieving the best performance, requires relatively high costs ($1.37–$1.62 per query) due to its premium pricing ($3.00/$15.00 per million input/output tokens). However, Claude demonstrates

- Table 4 Average cost per question for each model on GISA. The input and output token counts reflect only the main reasoning chain, while tokens consumed by the browsing tool are reported separately. Prices are reported in US dollars per million tokens, with RMB converted at a rate of 7:1.

Model # Input # Output # Tool Input # Tool Output Input Price Output Price Average Cost

Qwen3-235B-A22B (thinking) 162888.63 46399.86 134491.03 32838.69 0.29 2.86 0.31 Claude 4.5 Sonnet (non-thinking) 321680.56 7185.19 156620.54 5173.38 3.00 15.00 1.62 Claude 4.5 Sonnet (thinking) 267248.29 7619.70 133077.07 4308.98 3.00 15.00 1.37 Gemini 3 Pro (low) 242960.98 27886.18 134920.32 6702.21 2.00 12.00 1.17 Gemini 3 Pro (high) 292792.10 35536.54 138258.97 7145.17 2.00 12.00 1.37 GPT-5.2 (thinking) 471289.43 20408.00 234744.65 14354.61 1.75 14.00 1.72 DeepSeek-V3.2 (non-thinking) 499634.20 11204.50 222052.20 7938.16 0.29 0.43 0.22 DeepSeek-V3.2 (thinking) 594672.30 16671.11 256626.42 10004.80 0.29 0.43 0.26 GLM-4.7 (thinking) 567243.06 45994.95 278159.63 41877.75 0.29 1.14 0.34 Seed-1.8 (thinking) 358530.71 15270.81 216757.47 10905.33 0.11 1.14 0.10 Qwen3-Max (thinking) 444898.78 15354.54 199051.86 10042.04 0.36 1.43 0.27 Kimi K2.5 (thinking) 416011.44 14367.39 183547.81 10318.11 0.57 3.00 0.42

remarkable efficiency in token usage, consuming significantly fewer tokens than most other models. Interestingly, the thinking mode variant is both more effective and cheaper than its non-thinking counterpart, as it uses tools more efficiently and generates fewer total tokens. Second, Chinese models (DeepSeek, GLM, Seed, Qwen, and Kimi) offer substantially lower costs, ranging from $0.10 to $0.42 per query. Among these, Seed-1.8 achieves the lowest cost at $0.10 per query, while DeepSeek-V3.2 provides an attractive balance between cost ($0.26) and performance. Third, we observe that token consumption does not necessarily correlate with performance. Models like DeepSeek-V3.2 and GLM-4.7 consume the most tokens (over 500K input tokens per query) yet do not achieve top performance, suggesting that effective tool usage and reasoning quality matter more than sheer volume of computation. Finally, GPT-5.2 incurs the highest average cost ($1.72) due to both high token consumption and relatively expensive pricing, without delivering proportionally better results. These findings highlight the importance of considering cost-effectiveness when deploying search agents in practice, as the most expensive models are not always the best performing.

- 5 Limitations

While GISA provides a comprehensive benchmark for evaluating information-seeking agents, we acknowledge several limitations. First, due to the complexity of the annotation process, which requires substantial human involvement at every stage, the scale of our benchmark remains relatively limited (373 queries). Although sufficient for evaluation purposes, this size may not support large-scale model training such as supervised fine-tuning. Second, GISA currently focuses exclusively on text-based information seeking and does not incorporate multimodal content such as images or videos. As search agents evolve to integrate visual capabilities from GUI agents, future benchmarks should consider evaluating agents’ ability to process and extract information from multimodal web content, which would better simulate real-world human information gathering. Third, due to cost constraints, we set a maximum of 30 tool invocations per query in our experiments. While this limit is sufficient for most tasks, we observed that a small number of cases failed to collect adequate information due to this constraint. Future work could explore more flexible resource allocation strategies.

- 6 Ethical Considerations

We strictly adhere to ethical standards throughout the construction of GISA. For annotation tool development, our browser extension is engineered to record only task-relevant interactions within specific search sessions, with no personally identifiable information, cookies, or extra browsing history collected. All annotators participated voluntarily with informed consent. To mitigate potential biases, we intentionally diversified query domains, though we acknowledge that the benchmark may still underrepresent certain languages or specialized domains. Regarding potential misuse, GISA focuses exclusively on publicly available information and does not include queries related to sensitive personal data or harmful content.

- 7 Conclusion

This paper introduces GISA, a benchmark for evaluating general information-seeking agents, comprising 373 humancrafted queries with four structured answer formats (item, set, list, and table). GISA addresses key limitations of existing benchmarks by unifying the evaluation of deep reasoning and broad information aggregation, enabling deterministic evaluation through structured answers, incorporating a live subset with periodically updated answers to resist data contamination, and providing complete human search trajectories for process-level supervision. Our evaluation of mainstream LLMs and commercial search products reveals that current systems still fall substantially short of human performance, with even the best model achieving only 18.23% overall exact match. These results highlight significant room for improvement in problem decomposition, adaptive search planning, and efficient tool utilization, and we hope GISA will serve as a valuable resource for developing more capable information-seeking agents.

- 8 Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 62402497) and the China Postdoctoral Science Foundation under Grant Number 2025T180440. The authors would like to thank Jinghan Yang, Zhao Wang, Binyu Xie, Shangze Li, Yifei Chen, Zhixin Lin, and Yuyao Zhang for data annotations.

References

Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, Chen Sun, Han Hou, Hui Yang, James Pan, Jianan Lou, Jiayi Mao, Jizheng Liu, Jinpeng Li, Kangyi Liu, Kenkun Liu, Rui Wang, Run Li, Tong Niu, Wenlong Zhang, Wenqi Yan, Xuanzheng Wang, Yuchen Zhang, Yi-Hsin Hung, Yuan Jiang, Zexuan Liu, Zihan Yin, Zijian Ma, and Zhiwen Mo. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations. CoRR, abs/2506.13651, 2025. doi: 10.48550/ARXIV.2506.13651. https://doi.org/10.48550/arXiv.2506.13651.

DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models. CoRR, abs/2512.02556, 2025. doi: 10.48550/ ARXIV.2512.02556. https://doi.org/10.48550/arXiv.2512.02556.

Shanghua Gao, Ada Fang, Yepeng Huang, Valentina Giunchiglia, Ayush Noori, Jonathan Richard Schwarz, Yasha Ektefaie, Jovana Kondic, and Marinka Zitnik. Empowering biomedical discovery with AI agents. CoRR, abs/2404.02831, 2024. doi: 10.48550/ ARXIV.2404.02831. https://doi.org/10.48550/arXiv.2404.02831.

Lisheng Huang, Yichen Liu, Jinhao Jiang, Rongxiang Zhang, Jiahao Yan, Junyi Li, and Wayne Xin Zhao. Manusearch: Democratizing deep search in large language models with a transparent and open multi-agent framework. CoRR, abs/2505.18105, 2025. doi: 10.48550/ARXIV.2505.18105. https://doi.org/10.48550/arXiv.2505.18105.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. CoRR, abs/2503.09516, 2025. doi: 10.48550/ARXIV.2503.09516. https: //doi.org/10.48550/arXiv.2503.09516.

Celeste Kidd and Benjamin Y. Hayden. The psychology and neuroscience of curiosity. Neuron, 88(3):449–460, November 2015. ISSN 0896-6273. doi: 10.1016/j.neuron.2015.09.010. http://dx.doi.org/10.1016/j.neuron.2015.09.010.

Dayoon Ko, Jihyuk Kim, Haeju Park, Sohyeon Kim, Dahyun Lee, Yongrae Jo, Gunhee Kim, Moontae Lee, and Kyungjae Lee. Hybrid deep searcher: Integrating parallel and sequential search reasoning. CoRR, abs/2508.19113, 2025. doi: 10.48550/ARXIV.2508.19113. https://doi.org/10.48550/arXiv.2508.19113.

Tian Lan, Bin Zhu, Qianghuai Jia, Junyang Ren, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, and Kaifu Zhang. Deepwidesearch: Benchmarking depth and width in agentic information seeking. CoRR, abs/2510.20168, 2025. doi: 10.48550/ARXIV.2510.20168. https://doi.org/10.48550/arXiv.2510.20168.

Baixuan Li, Dingchu Zhang, Jialong Wu, Wenbiao Yin, Zhengwei Tao, Yida Zhao, Liwen Zhang, Haiyang Shen, Runnan Fang, Pengjun Xie, Jingren Zhou, and Yong Jiang. Parallelmuse: Agentic parallel thinking for deep information seeking. CoRR, abs/2510.24698, 2025a. doi: 10.48550/ARXIV.2510.24698. https://doi.org/10.48550/arXiv.2510.24698.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou.

Websailor: Navigating super-human reasoning for web agent. CoRR, abs/2507.02592, 2025b. doi: 10.48550/ARXIV.2507.02592. https://doi.org/10.48550/arXiv.2507.02592.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. CoRR, abs/2501.05366, 2025c. doi: 10.48550/ARXIV.2501.05366. https: //doi.org/10.48550/arXiv.2501.05366.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. CoRR, abs/2504.21776, 2025d. doi: 10.48550/ARXIV.2504.21776. https://doi.org/10.48550/arXiv.2504.21776.

Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu C. Aggarwal, Hui Liu, Xiang Zhang, and Suhang Wang. A comprehensive survey on reinforcement learning-based agentic search: Foundations, roles, optimizations, evaluations, and applications. CoRR, abs/2510.16724, 2025. doi: 10.48550/ARXIV.2510.16724. https://doi.org/10.48550/arXiv.2510.16724.

Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, and Yuxiao Dong. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn RL. CoRR, abs/2509.10446, 2025. doi: 10.48550/ARXIV.2509.10446. https://doi.org/10.48550/arXiv.2509.10446.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: a benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. https://openreview.net/forum?id=fibxvahvs3.

Joon Sung Park, Joseph C. O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Sean Follmer, Jeff Han, Jürgen Steimle, and Nathalie Henry Riche, editors, Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST 2023, San Francisco, CA, USA, 29 October 2023- 1 November 2023, pages 2:1–2:22. ACM, 2023. doi: 10.1145/3586183.3606763. https://doi.org/10.1145/3586183.3606763.

Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, and Tu Vu. Sealqa: Raising the bar for reasoning in search-augmented language models. CoRR, abs/2506.01062, 2025. doi: 10.48550/ARXIV.2506.01062. https://doi.org/10.48550/ arXiv.2506.01062.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Chatdev: Communicative agents for software development. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 15174–15186. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.810. https://doi.org/10.18653/v1/2024.acl-long.810.

Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang, Wayne Xin Zhao, Wenjie Wang, and Tat-Seng Chua. Holistically guided monte carlo tree search for intricate information seeking. CoRR, abs/2502.04751, 2025. doi: 10.48550/ARXIV.2502.04751. https://doi.org/10.48550/arXiv.2502.04751.

William Webber, Alistair Moffat, and Justin Zobel. A similarity measure for indefinite rankings. ACM Trans. Inf. Syst., 28(4): 20:1–20:38, 2010. doi: 10.1145/1852102.1852106. https://doi.org/10.1145/1852102.1852106.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. CoRR, abs/2504.12516, 2025. doi: 10.48550/ARXIV.2504.12516. https://doi.org/10.48550/arXiv.2504.12516.

Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, Wenhao Huang, Yang Wang, and Ke Wang. Widesearch: Benchmarking agentic broad info-seeking. CoRR, abs/2508.07999, 2025. doi: 10.48550/ARXIV.2508.07999. https://doi.org/10.48550/arXiv.2508.07999.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. Webwalker: Benchmarking llms in web traversal. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 10290–10305. Association for Computational Linguistics, 2025. https://aclanthology.org/2025.acl-long.508/.

Yunjia Xi, Jianghao Lin, Yongzhao Xiao, Zheli Zhou, Rong Shan, Te Gao, Jiachen Zhu, Weiwen Liu, Yong Yu, and Weinan Zhang. A survey of llm-based deep search agents: Paradigm, optimization, evaluation, and challenges. CoRR, abs/2508.05668, 2025a. doi: 10.48550/ARXIV.2508.05668. https://doi.org/10.48550/arXiv.2508.05668.

Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao, Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Weiwen Liu, Yasheng Wang, Ruiming Tang, Weinan Zhang, and Yong Yu. Infodeepseek: Benchmarking agentic information seeking for retrieval-augmented generation. CoRR, abs/2505.15872, 2025b. doi: 10.48550/ARXIV.2505.15872. https://doi.org/10.48550/arXiv.2505.15872.

Yilong Xu, Xiang Long, Zhi Zheng, and Jinhua Gao. Ravine: Reality-aligned evaluation for agentic search. CoRR, abs/2507.16725,

2025. doi: 10.48550/ARXIV.2507.16725. https://doi.org/10.48550/arXiv.2507.16725.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025. doi: 10.48550/ARXIV.2505.09388. https://doi.org/10.48550/arXiv.2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. https://openreview.net/forum?id=WE_vluYUL-X.

Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan W. Suchow, Zhenyu Cui, Rong Liu, Zhaozhuo Xu, Denghui Zhang, Koduvayur Subbalakshmi, Guojun Xiong, Yueru He, Jimin Huang, Dong Li, and Qianqian Xie. Fincon: A synthesized LLM multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. http://papers.nips.cc/paper_files/paper/2024/ hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, and Mingshu Zhai. GLM-4.5: agentic, reasoning, and coding (ARC) foundation models. CoRR, abs/2508.06471, 2025. doi: 10.48550/ARXIV.2508.06471. https://doi.org/10.48550/arXiv.2508.06471.

Heng Zhou, Ao Yu, Yuchen Fan, Jianing Shi, Li Kang, Hejia Geng, Yongting Zhang, Yutao Fan, Yuhao Wu, Tiancheng He, Yiran Qin, Lei Bai, and Zhenfei Yin. Livesearchbench: An automatically constructed benchmark for retrieval and reasoning over dynamic knowledge. CoRR, abs/2511.01409, 2025a. doi: 10.48550/ARXIV.2511.01409. https://doi.org/10.48550/arXiv.2511.01409.

Junting Zhou, Wang Li, Yiyan Liao, Nengyuan Zhang, Tingjia Miao, Zhihui Qi, Yuhan Wu, and Tong Yang. Scholarsearch: Benchmarking scholar searching ability of llms. CoRR, abs/2506.13784, 2025b. doi: 10.48550/ARXIV.2506.13784. https://doi.org/ 10.48550/arXiv.2506.13784.

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, and Yining Hua. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. CoRR, abs/2504.19314, 2025c. doi: 10.48550/ARXIV.2504.19314. https://doi.org/10.48550/arXiv. 2504.19314.

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. Large language models for information retrieval: A survey. CoRR, abs/2308.07107, 2023. doi: 10.48550/ARXIV.2308.07107. https: //doi.org/10.48550/arXiv.2308.07107.

- A Data Format GISA is organized into three components: questions, answers, and human trajectories.

- (1) Questions: Each question is stored as a JSON object containing the unique identifier, encrypted question text, answer type, question type (stable or live), and topic category. The encryption is applied to prevent data contamination from model pre-training.
- (2) Answers: Each answer is stored as a separate CSV file corresponding to its question identifier. The CSV format accommodates all four answer types: item answers are placed in the first cell, set and list answers contain a single column, and table answers contain multiple columns with a predefined schema.

search: [<query_1>, <query_2>, ...] # A list of search queries result: { # A Python dictionary of search engine results page

"query_1": [ # Results of the first query {

"page": 1, # The number of search pages "results": [

{

"snippet": # The short snippet of the web page "title": # The title of the web page "url": # The URL link of the web page

}, {...}

]

}, {...}

]

} click: [ # A list of click actions

{

"from": # The source URL of the click action "to": # The target URL of the click action

}, {...}

]

Figure 7 The JSON template of annotated trajectories.

Table 5 Comparison of search behaviors between different models and human annotators.

Model Search Similarity Search Diversity Browsing Similarity

Qwen3-235B-A22B (thinking) 30.59 43.05 13.08 Claude 4.5 Sonnet (non-thinking) 32.21 24.03 17.83 Claude 4.5 Sonnet (thinking) 30.02 21.73 17.63 Gemini 3 Pro (low) 35.98 38.36 14.14 Gemini 3 Pro (high) 36.30 36.54 14.12 GPT-5.2 (thinking) 24.82 23.00 8.81 DeepSeek-V3.2 (non-thinking) 35.54 26.74 13.42 DeepSeek-V3.2 (thinking) 38.97 28.87 13.82 GLM-4.7 (thinking) 34.36 29.00 14.92 Seed-1.8 (thinking) 31.42 39.43 17.42 Qwen3-Max (thinking) 36.68 31.81 15.69 Kimi K2.5 (thinking) 32.69 25.92 16.82

- (3) Human Trajectories: Each human trajectory is stored as a JSON file containing three fields: the search queries issued by the annotator, the search engine results for each query, and the click actions recording navigations between pages. An example is illustrated in Figure 7.

- B Implementation Details

Following recent studies (Li et al., 2025b,d; Xi et al., 2025a), we adopt the ReAct (Yao et al., 2023) architecture for our agent implementation. The system prompt and tool definitions are provided in Figure 8 and Figure 10, respectively. Each model is equipped with two tools: Search and Browse, with a maximum limit of 30 tool invocations per session.

- • Search Tool: This tool is implemented using the Google Search engine (through Serper API). It accepts a query as input and retrieves the top-10 search results.
- • Browse Tool: This tool is built upon the Jina API to extract content from a given URL. To reduce noise, the retrieved content is summarized by an LLM. For experimental consistency, the summarization model is identical to the agent’s backbone model, using the non-thinking mode to reduce cost.

The prompt for commercial systems follows a similar structure and is shown in Figure 9.

System Prompt for ReAct-based Agent

You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources. You have 30 chances to call tools, use them wisely.

# Final Answer Format When you have gathered sufficient information, you must output the final answer within <answer></answer> tags. Inside these tags, you must strictly follow the TSV (Tab-Separated Values) format enclosed in a code block ```tsv. Determine the nature of the answer (Item, List, or Table) and format it as follows:

- 1. Single Item (Fact/Value): Use a single column with the header Value.
- 2. List: Use a single column with the header Item.
- 3. Table (Structured Data): Use standard TSV with appropriate headers for each column. CRITICAL:

- • The content inside ```tsv must be valid TSV format.
- • Always include a header row.
- • Do not add markdown notes or explanations inside the code block. Put any summary text outside the code block but still inside the <answer> tags.

Current date: {current_date} User Question: {question}

Figure 8 The system prompt used in ReAct-based Agent.

- C Human-Model Search Behavior Analysis

We provide detailed metrics quantifying the similarity between model and human search behaviors. We compute three metrics for each model, which are defined as:

- • Search Similarity measures the overlap between model-generated queries and human queries. For each human query, we compute the Jaccard similarity with all model queries and take the maximum value. The Jaccard similarity is defined as the ratio of the intersection to the union of term sets: Sim(Qh,Qm) = |Th∩Tm|/|Th∪Tm|, where Th and Tm are the term sets of human and model queries after lowercasing and tokenization. The questionlevel similarity is the average of all human query scores, and the final score is averaged across all questions.
- • Search Diversity measures the diversity of consecutive queries, computed as the average Jaccard similarity between adjacent queries. Lower values indicate more diverse query reformulations.
- • Browsing Similarity measures the overlap between URLs visited by the model and those visited by human annotators, computed as the Jaccard similarity of the two URL sets.

From the results shown in Table 5, we can observe: First, search similarity scores range from 24.82% (GPT-5.2) to 38.97% (DeepSeek-V3.2 thinking), indicating moderate alignment between model and human query formulations. Second, Claude 4.5 Sonnet exhibits the lowest search diversity (21.73%), suggesting more focused and incremental query refinement similar to human behavior, while Qwen3-235B-A22B shows the highest diversity (43.05%), indicating more exploratory search patterns. Third, browsing similarity remains relatively low across all models (8.81%–17.83%), confirming that models and humans often follow different navigation paths even when seeking the same information. Notably, Claude 4.5 Sonnet achieves the highest browsing similarity (17.83%), which aligns with its superior task performance.

Prompt for Commercial Systems

You are a helpful assistant. Given an user’s question, your task is to thinking step by step and output the final answer in the format of TSV.

# Final Answer Format You must output the final answer within <answer></answer> tags. Inside these tags, you must strictly follow the TSV (Tab-Separated Values) format enclosed in a code block ```tsv. Determine the nature of the answer (Item, List, or Table) and format it as follows:

- 1. Single Item (Fact/Value): Use a single column with the header Value.
- 2. List: Use a single column with the header Item.
- 3. Table (Structured Data): Use standard TSV with appropriate headers for each column. CRITICAL:

- • The content inside ```tsv must be valid TSV.
- • Always include a header row.
- • Do not add markdown notes or explanations inside the code block. Put any summary text outside the code block but still inside the <answer> tags.

Current date: {current_date} User Question: {question}

Figure 9 The system prompt used in commercial systems.

Tools Definition

- 1. Search

- • Description: Perform a Google web search and return the top search results.
- • Parameters:

– query (string, required): The search query string to be issued to Google.

- 2. Browse

- • Description: Visit one or more web pages and return a summarized version of their content based on a specific goal.
- • Parameters:

- – url (array of strings, required): One or more URLs of the web pages to visit.
- – goal (string, required): The specific information objective to focus on when summarizing the web pages.

Figure 10 The functional schema of the tools.

