# arXiv:2406.09170v1[cs.CL]13Jun2024

## Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning

Bahare Fatemi1∗, Mehran Kazemi2∗, Anton Tsitsulin1, Karishma Malkan2, Jinyeong Yim3, John Palowitch2, Sungyong Seo3, Jonathan Halcrow1, and Bryan Perozzi1 1Google Research, 2Google DeepMind, 3Google

### Abstract

Large language models (LLMs) have showcased remarkable reasoning capabilities, yet they remain susceptible to errors, particularly in temporal reasoning tasks involving complex temporal logic. Existing research has explored LLM performance on temporal reasoning using diverse datasets and benchmarks. However, these studies often rely on real-world data that LLMs may have encountered during pre-training or employ anonymization techniques that can inadvertently introduce factual inconsistencies. In this work, we address these limitations by introducing novel synthetic datasets specifically designed to assess LLM temporal reasoning abilities in various scenarios. The diversity of question types across these datasets enables systematic investigation into the impact of the problem structure, size, question type, fact order, and other factors on LLM performance. Our findings provide valuable insights into the strengths and weaknesses of current LLMs in temporal reasoning tasks. To foster further research in this area, we are open-sourcing the datasets and evaluation framework used in our experiments: https://huggingface.co/datasets/baharef/ToT.

### 1 Introduction

Recent breakthroughs in large language model (LLM) research and applications have been significant [59, 16, 46, 8, 56, 1, 55, 47]. These models, capable of generating new content, have fascinated the AI community, leading to the release of numerous LLMs trained on diverse tasks and data types [68]. All of these advancements have led to a growing consensus that LLMs are a pivotal advancement on the path to artificial general intelligence (AGI) [9]. Benchmarking reasoning capabilities in LLMs is therefore a problem of pressing interest to the field [29].

In this work we focus on temporal reasoning, an essential task for intelligent systems across many domains. Temporal reasoning is focused on understanding reasoning between events in time. Despite this area’s importance, existing temporal reasoning benchmarks do not effectively measure the full scope of temporal reasoning relationships. Instead they typically rely on question-answering tasks based on Knowledge Graph (KG)-style temporal facts about well-known entities.

This overemphasis on KG-style temporal facts limits the scope of research and creates several issues. First, it neglects the diverse temporal structure and reasoning tasks found in real-world applications beyond KGs. Second, the results on such data often reflect a model’s ability to exploit prior knowledge rather than genuine temporal reasoning, making findings less relevant to domains where models lack this knowledge (see Figure 1 as an example.). In addition, previous research has shown that shortcuts or heuristics can often answer questions on these datasets without explicit temporal reasoning [13, 54]. Finally, the simple temporal structure of knowledge graphs overlooks the extensive time arithmetic skills required in real-world temporal questions.

∗Equal contribution. Corresponding authors: baharef@google.com or mehrankazemi@google.com.

Preprint. Under review.

Prompt: Below are the list of head coaches for Chelsea FC. Who was the coach before Pochettino? Pochettino: July 2023 to May 2024 Potter: September 2022 to April 2023 Tuchel: January 2021 to September 2022 Lampard: July 2019 to January 2021 and April 2023 to June 2023 Sarri: July 2018 to June 2019

Model Response: The coach before Pochettino was Frank Lampard during his second stint with the club from April 2023 to June 2023.

Grounded prompt (answered correctly)

Prompt: Below are the list of head coaches for a club. Who was the coach before E5? E5: July 2023 to May 2024 E4: September 2022 to April 2023 E3: January 2021 to September 2022 E2: July 2019 to January 2021 and April 2023 to June 2023 E1: July 2018 to June 2019

Model Response: E4 was the coach before E5.

Abstract prompt (answered incorrectly)

Figure 1: Comparison of the same temporal query using real (left) and anonymized (right) entity names. Gemini Advanced correctly answered the query with real names but failed with anonymized names, suggesting that LLMs might rely on their parametric knowledge to solve temporal tasks.

Our Contributions: To address these limitations, we develop tasks specifically designed to assess temporal reasoning in a more comprehensive and controlled manner. Our benchmark, Test of Time, ToT, is centered around the observation that temporal reasoning often involves two primary skills: 1) understanding the semantics and logic of time, and 2) the ability to carry out temporal arithmetic. ToT has two tasks to cover each essential skill of temporal reasoning, which enable measuring and improving model performances along these two axes independently. ToT-Semantic, a synthetic task, focuses on temporal semantics and logic; it allows for a flexible exploration of diverse graph structures and reasoning task complexity, isolating and evaluating reasoning abilities independent of prior knowledge. ToT-Arithmetic, a crowd-sourced task, assesses the ability to perform calculations involving time points and durations. Our experimental results with ToT provide valuable insights into the strengths and weaknesses of current LLMs in temporal reasoning tasks.

### 2 Related work

Reasoning. The ability to draw valid conclusions from explicitly provided knowledge has been a fundamental goal for AI since its early days [39, 26]. In the past few years, several LLM-based techniques have been developed which have advanced the general automated reasoning capabilities of the state-of-the-art models [62, 65], or their capabilities in specific directions including mathematical reasoning [36, 2], logical reasoning [34, 14], multi-modal reasoning [61], commonsense reasoning [67], and more. Advancing reasoning may explicitly or implicitly translate to improvements in several downstream NLP applications.

Temporal reasoning. Temporal reasoning has recently gained substantial attention [e.g., 63, 7, 32, 42, 28, 24, 38, 58]. Much research focuses on enhancing LLMs’ understanding of temporal concepts, primarily through pre-training and fine-tuning strategies to improve their temporal reasoning capabilities [e.g., 63, 32, 64, 69, 41]. Benchmark creation is another active area, with many benchmarks centered on knowledge graphs [e.g., 40, 31, 30]. While TempTabQA [23] offers crowd-sourced questions based on Wikipedia infoboxes, the process is resource-intensive and prone to issues like LLM overuse by workers. Xiong et al. [63] recently proposed TGQA, a data set derived from the YAGO11k knowledge graph [15]. To prevent data leakage, TGQA changes each entity name to a name generated by GPT3.5 that is guaranteed to (i) align with the entity’s type and (ii) not be otherwise present in YAGO11k. This strategy has several weaknesses. First, it can introduce spurious entity name correlations (LLMs could even potentially guess the original entities due to their adjacent relations). Second, it can generate factually incorrect or anti-commonsensical claims, for instance if an entity’s generated replacement name is a real name that is nonetheless not in YAGO11k. Finally, relying on GPT for copying facts introduces the potential for hallucinations to contaminate the dataset.

Synthetic datasets. A new trend in probing various LLMs capabilities, especially in the case of reasoning, is through synthetic data that allows for a more systematic evaluation. Previous work has developed synthetic datasets for probing and improving various kinds of reasoning including logical reasoning [53, 35, 51]and mathematical reasoning [52, 33]. Most similar to our work, Fatemi et al. [21] develop a synthetic probe for measuring the graph-based reasoning abilities of LLMs [49, 45]. Our work extends this concept to the case of temporal reasoning with graph-like facts.

Present work. In this work, we introduce ToT, a novel benchmark for temporal reasoning generated synthetically. Unlike many existing benchmarks that rely on knowledge graphs, ToT aims to encompass a wider variety of graph structures. Our synthetic generation approach offers precise control over the type of data produced. Importantly, when evaluating LLMs against ToT, they cannot exploit their latent knowledge for shortcuts; instead, they must genuinely reason with the presented facts. This design promotes a more rigorous assessment of temporal reasoning capabilities in LLMs.

### 3 ToT: A benchmark for evaluating LLMs on temporal reasoning

We propose that effective temporal reasoning hinges on two distinct skills: understanding the semantics and logic of time, and performing accurate temporal arithmetic. To measure and improve model performance along these independent axes, we create a dedicated task for each skill. By decoupling the evaluation of temporal semantics from arithmetic, we aim to provide a more nuanced analysis of LLM capabilities, pinpointing strengths and weaknesses in each aspect. Experiments on these tasks enable us to independently benchmark LLM performance on both dimensions.

#### 3.1 ToT-Semantic: A Synthetic Dataset

The first task we introduce, ToT-Semantic, consists of synthetic problems designed to highlight temporal semantics and logic in reasoning. This task is unique because it allows us to freely experiment with a wide range of temporal dependencies and manipulate the difficulty of the reasoning problem. This allows us to isolate and analyze the core reasoning capabilities of a LLM, separating

- them from any reliance on pre-existing parametric knowledge. To create the ToT-Semantic task, we took the following steps (summarized in Figure 2):

- Step 1: Random structure generation. We begin by generating random structures that we will

then use to create temporal questions. To ensure we generate a diverse set of random structures for this purpose, we turn to the literature on graph structure generation. From it, we employ several existing algorithms for generating graphs of varying properties. This includes Erd˝os-Rényi (ER) graphs [18], scale-free networks (SFN) [6], graphs following the Barabási–Albert (BA) model [3] and stochastic block model (SBM) [27], as well as star and complete graphs. Each of these graph generation algorithms exhibit different properties and correspond to graphs that appear in different applications. For instance, Erd˝os-Rényi graphs are typically sparse with low average degree, while Barabási-Albert graphs are dense and exhibit power-law degree distributions. We leverage the NetworkX library for generating our random graphs. Additionally, we extracted anonymized EgoNets from WikiData [60] by replacing the entity and relation names with generic names. We refer to this structure as Anonymized Wikidata Extract (AWE) in our experiments. We generate graphs with the number of nodes selected uniformly at random from the [5-30] interval. More details on the random graph generators used (with visualizations) are available in the Appendix B.

Once we have an initial graph structure, we assign relations to the edges. For each graph, we first decide a number of relation types to be assigned to the edges, and assign each of these relation types to one of one-to-one, one-to-many, many-to-one and many-to-many. Then, for each edge in the graph, we randomly assign between 1 to p (=3 in our experiments) relations types without violating the relation type arity. Then, for each edge (u,v) labeled with a relation r, we assign a valid time interval that respects the relation types.

- Step 2: Question Generation. Having generated the random graphs in step 1, we then create questions about those graphs. We consider eight types of questions that are frequently used in day to day life and are common in various benchmarks. EventAtTimeT: asking which entity had some relation R with entity E at some T; EventAtWhatTime: asking at what time a relation R between two entities E1 and E2 started/ended; NumberOfEventsInTimeInterval: asking how many entities had relation R with entity E between T1 to T2; BeforeAfter: asking which entity had relation R with

###### 1. Generate a graph

A B

E

C

D

###### 2. Assign entity and relation names 3. Generate temporal facts

E11 was the R21 of E23 from 1983 to 1985. E23 was the R21 of E32 from 2007 to 2013. E51 was the R17 of E23 from 2004 to 2009. E32 was the R30 of E4 from 2010 to 2012.

R21

E11 E23

R21

R17

E32

E51

4. Generate a question Which entity was the R17 of E23 at the time when E32 started being the R21 of E23?

R30

E4

Figure 2: Steps for creating the ToT-Semantic dataset.

E1 right before/after E2; EventAtTimeOfAnotherEvent: asking when E1 had relation R1 with E2, which entity had relation R2 with E3; FirstLast: asking which entity was the first to have relation R with E; RelationDuration: Asking the k-th time relation R happened between E1 and E2, how long did it last; and timeline: Asking to sort the entities that had relation R with E chronologically.

To create any of the above questions, we keep sampling graphs and fact(s) from the graph until the a proper question of the desired type can be created for that graph and for that fact. For example, to create a BeforeAfter question, we keep sampling a graph G and fact F = (S,R,O,T1,T2) until we have a case where there was a unique entity E that had relation R with O right before [T1,T2].

Following the above two steps, we generated 10 questions per graph generation and per question type. We sorted the facts in five different ways as will be discussed later. This gives as a benchmark with a total of 7 ∗ 8 ∗ 5 ∗ 10 = 2800 questions, where 7 is the number of graph generation algorithms, 8 is the number of question types, 5 is the number of sorting algorithms, and 10 is the number of samples we generated. Example questions of each category type are shown in Table 1.

#### 3.2 ToT-Arithmetic: A Temporal Arithmetic Dataset

Our second task, ToT-Arithmetic, shifts from synthetic data to a real-world focus. This task moves beyond understanding the logic and semantics of time and delves into the practical application of mathematical operations within a temporal context. Through it, we are able to measure a LLM’s proficiency in temporal arithmetic and its practical utility in handling time-related computations.

Table 1: Example for each question type in the ToT-Semantic dataset.

Question Type Example EventAtTimeT Find the entity that evidently was the R17 of E69 in year 1932. EventAtWhatTime At what time did E69 start being the R90 of E22? NumberOfEventsInTimeInterval Find the number of unique entities that were the R82 of E27 between

1952 to 1957. Relations that ended in 1952 or started in 1957 must be counted as well.

BeforeAfter Immediately before E59, which entity was the R20 of E6? EventAtTimeOfAnotherEvent E94 was the R82 of which entity at the time when E83 started being

the R20 of E59? FirstLast Which entity was the first that was the R35 of E91? RelationDuration When E24 was the R53 of E11 for the 2nd time, for how many years

did the relation last? The duration can be computed by subtracting the start time from the end time.

Timeline Which entities were the R17 of E69?

Categorize We grouped the questions based on the required time arithmetic operations.

|Expand| |
|---|---|
|The annotators expanded the seed set into a large set of questions.| |
| | |

|Filter| |
|---|---|
|We filtered knowledge heavy and corner case questions.| |
| | |

|Seed Set| |
|---|---|
|We selected a seed set of questions.| |
| | |

| |Functionalize|
|---|---|
| |We implemented a functional version of each question, where the input arguments are sampled and final answers are calculated using python libraries.<br><br># EXAMPLE: Add days function def add_days(start_time, end_time):<br><br>date = random_date() n = random.randint(10,100) question = f"If today is {date}, what is the day {n} days from now?" answer = current_day + datetime.timedelta(days = n) return question, answer|
| | |

###### Sample

We generated a dataset by sampling questions and answers from the codes.

Figure 3: Steps for creating the ToT-Arithmetic dataset. The green and blue colors represent the operations done by the authors and the annotators respectively.

#### 3.2.1 Task Creation

To create a large time-arithmetic dataset that covers a wide variety of problems, we took the steps summarized in Figure 3. We explain each step in more detail below.

- • Seed Set: By examining the existing benchmarks and the kind of temporal arithmetic questions that arise in them and through searching the web, we gathered a small set of initial questions.
- • Expand: We presented our seed set to 15 annotators who were tasked to propose either new time arithmetic questions that were not in our seed set, or to provide questions corresponding to other scenarios or question templates where one requires to do similar time arithmetic operations to one of the questions in our seed set. We gathered a large list of questions through this process.
- • Filter: We manually went through all the questions and filtered the ones that were focusing on corner cases, or that required extensive knowledge (e.g., requiring to memorize the entire calendar).
- • Categorize: We then grouped the remaining problems into seven categories, shown with examples in Table 2. Categories are formed based on the time arithmetic operations required, as follows: AddSubtract: adding or subtracting a number (corresponding to days, weeks, minutes, hours, etc.) from a date or time; Compare: comparing a number of dates/times provided in different formats chronologically; Duration: computing the difference between two dates/times; Schedule: finding mutual free spots within multiple blocked times; Timezone: involving dealing with different timezones; Trick: involving questions with slight twists; and MultiOp: involving questions where multiple of the above operations are needed.
- • Funcionalizing: Following [52], we implemented a functional version of each problem to enable sampling different values for each question and solving based on those values. A functional version of one of our simple problems is provided in Figure 3.
- • Sampling: We then sampled questions and answers from our functionalized problems. We made the number of samples proportional to the number of different problems that fell under each category. Specifically, we sampled 350 for AddSubtract, 350 for Compare, 200 for Duration, 250 for Schedule, 100 for Timezone, 250 for Trick, and 350 for MultiOp. This resulted in a dataset with 1850 questions in total.

#### 3.3 Quality Check

For both tasks, we did multiple rounds of quality checks where we verified 1) whether the generated labels are correct, and 2) whether the question is clear and the provided instructions are sufficient to know in what format the output should be produced. This procedure was done until no more issues could be found in the dataset.

Table 2: Examples for each question type in the ToT-Arithmetic dataset.

Category Example AddSubtract Your driver’s license expires on 18 May, 2017. You receive a renewal notice

saying it can be renewed 117 days in advance. What’s the earliest date you can renew your license?

Compare E42 was discovered in 14 April, 52 BC and E11 was discovered in 05 October, 530 BC. Which was discovered earlier?

Duration Stella and William were born on 1999-Dec-16 and 2000-Oct-03 respectively. When William was 400 days old, how old was Stella in days?

Schedule Lucas is available from 11 to noon and also from 3:30 to 5. Asher is available from 11 to 12:30 and also from 4 to 5. They want to have a 30 minute meeting. The meeting has to start on the hour or half hour. How many possibilities are there for the meeting time?

Timezone Flight departs location A at 11:08 (24hr) UTC(+0000). It reaches location B at 07:23:20 PM IST(+0530). What is the total time duration taken to fly?

Trick If the date for the day before tomorrow in yyyy-mm-dd format is 2016-01-20, what is the date 27 days from now in the same format?

MultiOp Alex solves 2 puzzles in 4 hours, 50 minutes, and 22 seconds. What is the time taken by them to solve 6 puzzles, at the same pace.

### 4 Experiments and Results

In this study, we evaluate the performance of three frontier large language models (LLMs) on our benchmark. The models evaluated include Claude-3-Sonnet [4], GPT-4 [1], and Gemini 1.5 Pro [47]. As the ToT-Semantic task requires a large context size, for this benchmark we used the GPT4-Turbo because of its higher context limit; For ToT-Arithmetic we used GPT4 due to its higher quality.

In our experiments, we aim to answer the following questions:

- • RQ1: What is the effect of the temporal structure on the LLM performance?
- • RQ2: What kind of temporal questions are easier/harder for LLMs to answer?
- • RQ3: How important is the order of the facts in the model prompt and what is the best way of ordering the facts?
- • RQ4: How well do the frontier models perform on two aspects of temporal reasoning: semantics and arithmetic?

#### 4.1 Investigating the impact of temporal structure on LLM temporal reasoning

Table 3: LLM accuracy on temporal reasoning tasks by graph structure.

In different applications where temporal reasoning arises, the structure of the facts can be different. Some tasks may provide all the information about an entity (corresponding to a star graph) and then asks questions about it, whereas in some applications such as social networks the structure of the facts may follow a power-law distribution. It is natural to question whether the inherent temporal structure of a problem might influence an LLM’s ability to reason over its data. Drawing inspiration from recent work analyzing graph neural networks [43, 57, 66, 21], this section aims to quantify how different temporal dependencies affects an LLM’s temporal reasoning capabilities using graph generators to create many different kinds of temporal structure. Our findings demonstrate that graph structure can significantly impact an LLM’s reasoning performance.

Graph Claude-3-Sonnet GPT-4 Gemini 1.5 Pro

BA 48.50 63.25 62.75 Complete 34.00 40.25 52.50 ER 42.25 48.75 60.50 SBM 42.00 50.75 57.75 SFN 58.75 75.25 75.75 Star 59.50 80.25 74.25 AWE 68.75 92.00 87.50

Average Rank 3.00 1.57 1.43

The graph structure of the temporal relationships significantly affects LLM performance, as demonstrated in Table 3. Notably, GPT-4 accuracy more than doubled between complete graphs (40.25%)

Accuracy,%

Gemini 1.5 Pro GPT-4 Claude Sonnet

100

100

80

80

60

60

40

40

20

20

0

0

<250 200–500 500–750 750–1000 >1000

<10 10–20 >20

Number of edges

Number of nodes

Figure 4: Accuracy of models for different number of edges and nodes.

and AWE graphs (92.00%). GPT-4 accuracy varied drastically across graph types, from 20.33% for complete graphs to 90.83% for AWE graphs. This highlights a critical gap in temporal reasoning research, which has largely overlooked the diverse graph structures and reasoning tasks found in real-world applications, instead focusing primarily on specific knowledge graphs (like YAGO11k). This may explain the superior performance of LLMs on AWE graphs in our experiments.

#### 4.1.1 Influence of graph size on LLM performance

A key question is whether the size of a graph, measured by the number of edges (facts) and nodes (entities) respectively, impact LLM performance. As illustrated in Figure 4, increasing either the number of edges or nodes in the ToT-Semantic dataset leads to a decrease in LLM performance.

Table 4: Average number of nodes and edges by graph structure.

Graph #nodes #edges BA 17.41 144.07 Complete 17.25 619.86 ER 16.18 316.4 SBM 17.51 368.15 SFN 17.52 53.46 Star 16.16 34.12 AWE 18.99 25.41 Average 17.29 223.07

This raises the question of whether the graph structure’s impact observed in Section 4.1 is merely a consequence of varying graph sizes. To address this, we present the average number of nodes and edges for each graph structure in Table 4. While the average number of nodes does not appear to consistently influence LLM performance across structures, the number of edges does show some correlation. However, there are exceptions. For instance, SFN graphs, despite having far fewer edges on average than Star graphs, exhibit slightly better average performance in Table 3. This indicates that both the number of edges and the specific structure of the graph play a significant role in determining LLM performance.

- 4.2 Effects of temporal question type on LLM temporal reasoning Table 5: LLM accuracy on temporal reasoning by question category.

In this experiment, we systematically investigate the impact of different temporal tasks on the reasoning ability of LLMs. We quantify this impact by evaluating model performance across a variety of tasks, as summarized in Table 5.

Temporal Question Type Claude-3-Sonnet GPT-4 Gemini 1.5 Pro

EventAtTimeT 47.14 65.43 72.29 EventAtWhatTime 90.29 89.43 93.14

NumberOfEventsInTimeInterval 29.71 61.43 59.14 BeforeAfter 53.14 55.43 52.86 EventAtTimeOfAnotherEvent 50.00 67.14 71.43 FirstLast 68.57 67.71 68.57 RelationDuration 41.43 80.00 84.57 Timeline 24.00 28.29 36.29

Task type and reasoning requirements. A key question in our investigation is whether the type of temporal task and the associated reasoning requirements influence LLM performance. The ToT-Semantic dataset includes questions of varying difficulty levels, which can be categorized based on the nature of the reasoning involved: Single-fact solutions: Questions EventAtTimeT and EventAtWhatTime require retrieving one single fact and answering the question directly based on the retrieved fact. Multi-fact solutions: The remaining questions require retrieving

Average Rank 2.56 2.00 1.44

multiple facts and performing operations (e.g., counting, sorting) to extract the relevant information and formulate an answer.

LLMs consistently demonstrate superior performance on tasks requiring the retrieval of a single fact compared to those necessitating the integration of multiple facts. This performance gap can be attributed to the increased cognitive demands associated with multi-fact tasks. While singlefact questions primarily rely on the identification and extraction of relevant information, multi-fact questions demand a deeper comprehension and synthesis of retrieved information.

Performance variations within question types. Even within zero-order reasoning tasks, LLMs demonstrate varying levels of proficiency. For example, EventAtTimeT and EventAtWhatTime are structurally similar, yet LLMs tend to excel at the latter. We hypothesize that this performance difference may be attributed to the fact that EventAtTimeT requires a simple time arithmetic operation to recognize that a timestamp T falls within a time interval [T1,T2], whereas EventAtWhatTime does not require any time arithmetic operation.

Analysis on Timeline questions. An analysis of timeline questions in Table 5 reveals that they pose the greatest challenge across all tasks. The answer to these questions, typically structured as “Sort the entities that were the R17 of E69 chronologically?”, is often a list of entities. In the ToT-Semantic dataset, every timeline question has more than one entity in its label. To further evaluate performance on these multi-answer questions, we calculated the average precision and recall for each model in Table 6. We report the results once averaged over all graph structures and once only for complete graphs (the most challenging graph structure). Gemini 1.5 Pro demonstrates superior precision and recall, aligning with its higher accuracy observed in Table 5. Notably, GPT-4, despite having higher accuracy than Claude-3-Sonnet on timeline questions, exhibits the lowest precision and recall. This suggests that GPT-4 frequently outputs fewer entities than are present in the true answers (50% of the times), leading to missed correct entities (lower recall) and a higher proportion of false positives among its predictions (lower precision).

Table 6: LLM precision and recall on timeline questions.

All Complete Graph structure Precision Recall Precision Recall

Claude-3-Sonnet 0.73 0.75 0.56 0.54 GPT-4 0.60 0.56 0.36 0.23 Gemini 1.5 Pro 0.81 0.83 0.82 0.65

Since complete graphs pose the greatest difficulty among all graph structures (Table 3), we provide a separate analysis of average precision and recall for these graphs in the final two columns of Table 6. Notably, Claude-3-Sonnet and GPT-4 experienced declines in both precision and recall on complete graphs, whereas Gemini was primarily impacted in terms of recall.

Analysis on NumberOfEventsInTimeInterval questions. When evaluating models on NumberOfEventsInTimeInterval questions, a clear pattern emerged. Claude-3-Sonnet frequently overestimated the correct count. In contrast, GPT-4 and Gemini 1.5 Pro demonstrated higher accuracy, although minor errors were observed: Gemini 1.5 Pro tended towards slight overestimation, and GPT-4 towards slight underestimation.

#### 4.3 Impact of temporal fact order on LLM performance

Table 7: LLM accuracy on temporal reasoning tasks as a function of the order of the facts.

A noteworthy question arises regarding the potential influence of fact order on LLM performance in temporal reasoning tasks. For investigating this, we conducted experiments on ToT-Semantic dataset. We sorted the facts using different methods: Shuffle: randomizing the order of facts; RelationAndStartTime: prioritizing facts based on their relation name, then by start time; StartTimeAndRelation: prioritizing facts based on start time, then by relation name; StartTimeAndTarget: prioritizing facts based on start time,

Order of facts Claude-3-Sonnet GPT-4 Gemini 1.5 Pro

Shuffle 45.71 60.71 63.04 RelationAndStartTime 54.29 65.36 68.57 StartTimeAndRelation 47.68 60.54 64.64 StartTimeAndTarget 49.11 61.61 65.18 TargetAndStartTime 73.57 62.60 75.00

Average Rank 2.80 2.20 1.00

then by the target entity; TargetAndStartTime: Prioritizing facts based on the target entity, then by start time.

Ideally, LLMs should exhibit robustness to the order in which facts are presented, given the independent nature of each fact. However, as shown in Table 7, our observations reveal a significant impact of fact order on LLM performance. Notably, performance is consistently lowest when facts are presented in a shuffled order and consistently highest when facts are sorted based on the target entity and start time (TargetAndStartTime). This finding offers valuable practical insights into how facts should be structured when temporal reasoning is a key component of the LLM task. By organizing facts in a manner that aligns with the temporal flow of the narrative or task, we can potentially enhance LLM performance and ensure more accurate and reliable results. While previous work has shown that ordering premises in the correct order of chain-of-thought solution improves LLM’s logical reasoning [12, 50], our results extend that to general-purpose temporal orderings (independent of the chain-of-thought).

#### 4.4 Temporal semantics vs temporal arithmetic

This study examined the performance of temporal arithmetic capabilities in LLMs using the ToT-Arithmetic dataset. Results, as shown in Table 8, indicate that the models consistently excelled in Timezone questions, while struggling the most with Duration questions. This superior performance in Timezone questions could be attributed to the abundance of information about various timezones available online, compared to other question types. Scheduling and Trick questions also proved challenging for LLMs, likely due to their creative nature and requirement for deeper reasoning. In contrast, AddSubtract results were relatively strong, potentially reflecting LLMs’ optimization for mathematical reasoning and their ability to apply that knowledge to temporal reasoning tasks.

Table 8: LLM accuracy on the ToT-Arithmetic dataset by question type.

Category Claude-3-Sonnet GPT-4 Gemini 1.5 Pro

AddSubtract 58.57 76.28 71.14 Compare 39.14 63.14 55.43 Duration 15.00 16.00 13.50 Schedule 29.60 43.60 40.00 Timezone 74.00 88.00 90.00 Trick 40.40 45.60 41.20 MultiOp 26.57 46.86 62.57

Average Rank 2.86 1.29 1.86

Compared with the semantic questions results from Table 5, we see that there is a substantial difference in average rank results, with GPT-4 doing better on arithmetic and Gemini 1.5 Pro performing better in the reasoning category.

Analysis on Duration questions. Analysis of Duration questions in the ToT-Arithmetic dataset revealed them to be the most challenging for the evaluated models. Notably, the most common error among incorrect answers was a deviation of precisely one day from the ground truth label. Specifically, when GPT-4 or Gemini 1.5 Pro erred on Duration questions, approximately 21% and 25% of its responses were within one day of the ground truth, respectively. This suggests that LLMs can approximate the correct calculation but often stumble in the final steps, highlighting a gap in their ability to execute complex arithmetic with precision.

Common failure: direction. One frequent error in ToT-Arithmetic occurs when determining the number of months between two dates. For example, from February 11th, 2002, to October 11th, 2002, the correct duration is eight months, but the model sometimes incorrectly calculates it as four months. This issue is particularly noticeable in questions that involve going back in time, such as: “Sam’s birthdate is October 11th, 1996. Today is February 25th, 2002. Calculate Sam’s age in days.”

Common failure: leap year calculation. Another frequent error in ToT-Arithmetic arises when determining the number of days between two dates that span multiple years. Incorrectly accounting for leap years, which have an extra day (February 29th), often leads to inaccurate results.

### 5 Conclusion

In conclusion, we introdwuce Test of Time (ToT), a novel benchmark designed to assess LLMs’ temporal reasoning abilities in a more comprehensive and controlled manner than existing work. Our two-pronged approach, encompassing both semantic and arithmetic tasks, enables a nuanced evaluation of temporal reasoning. Through extensive experiments with ToT, we have gained valuable

insights into the strengths and weaknesses of current LLMs in these critical aspects of temporal reasoning. By open-sourcing our datasets and evaluation framework, we hope to stimulate further research and development in this field, ultimately contributing to the advancement of LLM capabilities in complex reasoning tasks.

### 6 Acknowledgement

We express our sincere gratitude to Don Metzler, Radu Soricut, Anastasios Kementsietsidis, Silvio Lattanzi, and Vahab Mirrokni for their great feedback. We also extend our thanks to all participants who generously provided data for the time arithmetic dataset.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157, 2024.
- [3] Réka Albert and Albert-László Barabási. Statistical mechanics of complex networks. Reviews of modern physics, 74(1):47, 2002.
- [4] Anthropic. Introducing the next generation of claude. https://www.anthropic.com/ news/claude-3-family, 2024. Available at: https://www.anthropic.com/news/ claude-3-family.
- [5] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 2024.
- [6] Albert-László Barabási and Réka Albert. Emergence of scaling in random networks. science, 286(5439):509–512, 1999.
- [7] Himanshu Beniwal, Mayank Singh, et al. Remember this event that year? assessing temporal information and reasoning in large language models. arXiv preprint arXiv:2402.11997, 2024.
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020.
- [9] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.
- [10] Ines Chami, Sami Abu-El-Haija, Bryan Perozzi, Christopher Ré, and Kevin Murphy. Machine learning on graphs: A model and comprehensive taxonomy. Journal of Machine Learning Research, 23(89):1–64, 2022.
- [11] Wenhu Chen, Xinyi Wang, and William Yang Wang. A dataset for answering time-sensitive questions. arXiv preprint arXiv:2108.06314, 2021.
- [12] Xinyun Chen, Ryan A Chi, Xuezhi Wang, and Denny Zhou. Premise order matters in reasoning with large language models. arXiv preprint arXiv:2402.08939, 2024.
- [13] Ziyang Chen, Xiang Zhao, Jinzhi Liao, Xinyi Li, and Evangelos Kanoulas. Temporal knowledge graph question answering via subgraph reasoning. Knowledge-Based Systems, 251:109134, 2022.
- [14] Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. arXiv preprint arXiv:2205.09712, 2022.

- [15] Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. Hyte: Hyperplane-based temporally aware knowledge graph embedding. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2001–2011, 2018.
- [16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology. org/N19-1423.
- [17] Bhuwan Dhingra, Jeremy R Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W Cohen. Time-aware language models as temporal knowledge bases. Transactions of the Association for Computational Linguistics, 10:257–273, 2022.
- [18] Paul Erd˝os and Alfred Rényi. On random graphs. Publicationes Mathematicae Debrecen, 6: 290–297, 1959.
- [19] Bahare Fatemi, Layla El Asri, and Seyed Mehran Kazemi. Slaps: Self-supervision improves structure learning for graph neural networks. Advances in Neural Information Processing Systems, 34:22667–22681, 2021.
- [20] Bahare Fatemi, Sami Abu-El-Haija, Anton Tsitsulin, Mehran Kazemi, Dustin Zelle, Neslihan Bulut, Jonathan Halcrow, and Bryan Perozzi. Ugsl: A unified framework for benchmarking graph structure learning. arXiv preprint arXiv:2308.10737, 2023.
- [21] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. In ICLR, 2024.
- [22] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12):86–92, 2021.
- [23] Vivek Gupta, Pranshu Kandoi, Mahek Bhavesh Vora, Shuo Zhang, Yujie He, Ridho Reinanda, and Vivek Srikumar. Temptabqa: Temporal question answering for semi-structured tables. arXiv preprint arXiv:2311.08002, 2023.
- [24] Wes Gurnee and Max Tegmark. Language models represent space and time. arXiv preprint arXiv:2310.02207, 2023.
- [25] Jonathan Halcrow, Alexandru Mosoi, Sam Ruth, and Bryan Perozzi. Grale: Designing networks for graph learning. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2523–2532, 2020.
- [26] Carl Hewitt. Planner: A language for proving theorems in robots. In Proceedings of the 1st International Joint Conference on Artificial Intelligence, IJCAI’69, page 295–301, San Francisco, CA, USA, 1969. Morgan Kaufmann Publishers Inc.
- [27] Paul W Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. Stochastic blockmodels: First steps. Social networks, 5(2):109–137, 1983.
- [28] Xuming Hu, Junzhe Chen, Xiaochuan Li, Yufei Guo, Lijie Wen, Philip S Yu, and Zhijiang Guo. Do large language models know about facts? arXiv preprint arXiv:2310.05177, 2023.
- [29] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey, 2023.
- [30] Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard Weikum. Tempquestions: A benchmark for temporal question answering. In Companion Proceedings of the The Web Conference 2018, pages 1057–1062, 2018.
- [31] Zhen Jia, Soumajit Pramanik, Rishiraj Saha Roy, and Gerhard Weikum. Complex temporal question answering on knowledge graphs. In Proceedings of the 30th ACM international conference on information & knowledge management, pages 792–802, 2021.

- [32] Zhen Jia, Philipp Christmann, and Gerhard Weikum. Tiq: A benchmark for temporal question answering with implicit time constraints. In Companion Proceedings of the ACM on Web Conference 2024, pages 1394–1399, 2024.
- [33] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.
- [34] Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. LAMBADA: Backward chaining for automated reasoning in natural language. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6547–6568, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.361. URL https://aclanthology.org/2023.acl-long.361.
- [35] Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, and Deepak Ramachandran. Boardgameqa: A dataset for natural language reasoning with contradictory information. Advances in Neural Information Processing Systems, 36, 2023.
- [36] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.
- [37] Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tayfun Terzi, Eren Sezener, Devang Agrawal, D’Autume Cyprien De Masson, Tim Scholtes, Manzil Zaheer, Susannah Young, et al. Streamingqa: A benchmark for adaptation to new knowledge over time in question answering models. In International Conference on Machine Learning, pages 13604–13622. PMLR, 2022.
- [38] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, and Ankit Shah. Grounding complex natural language commands for temporal tasks in unseen environments. In Conference on Robot Learning, pages 1084–1110. PMLR, 2023.
- [39] John McCarthy. Programs with common sense. In Proceedings of the Teddington Conference on the Mechanization of Thought Processes, pages 75–91, London, 1959. Her Majesty’s Stationary Office. URL http://www-formal.stanford.edu/jmc/mcc59.html.
- [40] Sumit Neelam, Udit Sharma, Hima Karanam, Shajith Ikbal, Pavan Kapanipathi, Ibrahim Abdelaziz, Nandana Mihindukulasooriya, Young-Suk Lee, Santosh Srivastava, Cezar Pendus, et al. Sygma: System for generalizable modular question answering overknowledge bases. arXiv preprint arXiv:2109.13430, 2021.
- [41] Qiang Ning, Zhili Feng, Hao Wu, and Dan Roth. Joint reasoning for temporal and causal relations. arXiv preprint arXiv:1906.04941, 2019.
- [42] Kai Nylund, Suchin Gururangan, and Noah A Smith. Time is encoded in the weights of finetuned language models. arXiv preprint arXiv:2312.13401, 2023.
- [43] John Palowitch, Anton Tsitsulin, Brandon Mayer, and Bryan Perozzi. Graphworld: Fake graphs bring real insights for gnns. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3691–3701, 2022.
- [44] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 701–710, 2014.
- [45] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for llms, 2024.
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

- [47] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [48] Benedek Rozemberczki, Peter Englert, Amol Kapoor, Martin Blais, and Bryan Perozzi. Pathfinder discovery networks for neural message passing. In Proceedings of the Web Conference 2021, pages 2547–2558, 2021.
- [49] Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. arXiv preprint arXiv:2405.18512, 2024.
- [50] Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. arXiv preprint arXiv:2210.01240, 2022.
- [51] Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, and He He. Testing the general deductive reasoning capacity of large language models using ood examples. Advances in Neural Information Processing Systems, 36, 2023.
- [52] Saurabh Srivastava, Anto PV, Shashank Menon, Ajay Sukumar, Alan Philipose, Stevin Prince, Sooraj Thomas, et al. Functional benchmarks for robust evaluation of reasoning performance, and the reasoning gap. arXiv preprint arXiv:2402.19450, 2024.
- [53] Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.317. URL https://aclanthology.org/2021.findings-acl.317.
- [54] Qingyu Tan, Hwee Tou Ng, and Lidong Bing. Towards benchmarking and improving the temporal reasoning capability of large language models. arXiv preprint arXiv:2306.08952, 2023.
- [55] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [57] Anton Tsitsulin, Benedek Rozemberczki, John Palowitch, and Bryan Perozzi. Synthetic graph generation to benchmark graph learning. arXiv preprint arXiv:2204.01376, 2022.
- [58] Siddharth Vashishtha, Adam Poliak, Yash Kumar Lal, Benjamin Van Durme, and Aaron Steven White. Temporal reasoning in natural language inference. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4070–4078, 2020.
- [59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017.
- [60] Denny Vrandeˇci´c and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Communications of the ACM, 57(10):78–85, 2014.
- [61] Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning. arXiv preprint arXiv:2401.06805, 2024.
- [62] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [63] Siheng Xiong, Ali Payani, Ramana Kompella, and Faramarz Fekri. Large language models can learn temporal reasoning. arXiv preprint arXiv:2401.06853, 2024.
- [64] Sen Yang, Xin Li, Lidong Bing, and Wai Lam. Once upon a time in graph: Relative-time pretraining for complex temporal reasoning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11879–11895, 2023.
- [65] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2023.
- [66] Mustafa Yasir, John Palowitch, Anton Tsitsulin, Long Tran-Thanh, and Bryan Perozzi. Examining the effects of degree distribution and homophily in graph learning models, 2023.
- [67] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019.
- [68] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.
- [69] Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal, and Dan Roth. Temporal reasoning on implicit events from distant supervision. arXiv preprint arXiv:2010.12753, 2020.

### A Detailed comparison of ToT with existing benchmarks.

Table 9: Comparison of ToT against related benchmarks

Benchmark Semantics Arithmetic Real-World Synthetic Hermetic Implicit StreamingQA [37] ✓ ✗ ✓ ✗ ✗ ✗

TimeSensitiveQA [11] ✓ ✗ ✓ ✗ ✗ ✗ TempLama [17] ✓ ✗ ✓ ✗ ✗ ✗

TEMPTABQA [23] ✓ ✗ ✓ ✗ ✗ ✓ TEMPREASON [54] ✓ ✓ ✓ ✗ ✗ ✓

.

TIQ [32] ✓ ✗ ✓ ✗ ✗ ✓ TempUN [7] ✓ ✗ ✓ ✗ ✗ ✗

TGQA [63] ✓ ✗ ✓ ✗ ✗ ✗ ToT (ours) ✓ ✓ ✓ ✓ ✓ ✓

Table 9 provides a comprehensive comparison of ToT with existing benchmarks across six key dimensions:

- 1. Semantics: whether the benchmark has semantic-type questions.
- 2. Arithmetic: whether the benchmark has arithmetic-type questions.
- 3. Real-world: whether the benchmark has questions generated from real-world data.
- 4. Synthetic: whether the benchmark has questions generated from synthetic data.
- 5. Hermetic: whether the benchmark is sealed off from potential LLM training data.
- 6. Implicit: whether the benchmark includes implicit questions.

Our analysis reveals that ToT is unique in incorporating all these question types while effectively mitigating training data leakage. Notably, TEMPREASON [54] only covers one category of the arithmetic operations as defined in Section 3.2.

### B Description of graph generators.

Here we detail each graph generator used to create the examples in ToT. First, we cover the six random graph generators used to create the synthetic examples. All random graph generators are probabalistic models which take hyperparameters that control the expected macro-properties of each graph [43]:

- • Erd˝os-Rényi (ER) [18]: This model takes an edge probability parameter p and generates each edge with probability p, i.i.d. over all possible edges.
- • Scale-Free Networks (SFN) [6]: a graph is grown by a sequence of steps, each step either

(1) adding a new node and connecting it to an existing node, or (2) adding an edge between two existing nodes. Input parameters control the probability of these events. This process generates a “scale-free” power law of node degrees, in sharp contrast to the ER model.

- • Barabási–Albert (BA) model [3]: a graph is grown by a sequence of steps, each step adding a new node to the graph, and connecting the node to m existing nodes with probability proportional to their current degree. Similar to SFN, this process also generates a “scale-free” graph with a particular distribution known as the Barabási–Albert model.
- • Stochastic Block Model (SBM) [27]: This graph model can be thought of as clustered ER. It divides n nodes into k clusters, and then connects two nodes with probability p if they are in the same cluster, else with probability q if they are in different clusters. k, p, and q are all hyperparameters.
- • A star-graph generator creates a “star” graph on n nodes: node 0 is the center of the star, and all other nodes connect to it (and only it).
- • A complete-graph generate creates a “complete” graph on n nodes, in which all nodes are connect to each other node.

[Figure 1]

Figure 5: A visualization of a representative graph from each graph generator: Erd˝os-Rényi (ER), Scale-Free Networks (SFN), Barabási–Albert (BA), Stochastic Block Model (SBM), star-graph, and complete-graph.

An example from each of the above graph generators is shown in Figure 5. In the figure, edges are annotated with temporal relationships in the format relation_id: [interval_1, ...,

interval_k]. Note that each edge can have multiple relationships, and each relationship can have multiple intervals. The visualization shows the diversity of temporal knowledge graphs that our framework is able to generate. We note that while our study was limited to parametric graph generators in this work, the field of graph machine learning [10] offers many options for both modeling [44] and learning [25, 19, 48, 20] link structure.

Second, we describe our Anonymized Wikidata Extract (AWE) strategy for creating anonymized questions from real-world data. We first identify the 78 most common relations in WikiData that specify time-bound entity relationships. Each relation encodes a temporal edge between two entities. To match the schema of our synthetic graphs, we convert each time specification on each edge to an interval. Then, for each entity in the graph, we extract the ego-graph of the entity by (1) collecting the entity and all its neighbors and (2) collecting all edges (along with temporal information) between nodes collected in (1). This process produces a temporal graph with a schema identical to those produced from random graph generators. Before generating questions from the graphs, we anonymize them by (a) mapping each entity name to a unique identifier such as E679; and then (b) mapping each relation name to A unique identifier such as R3. We then generate questions from the graph as described in 3.1.

### C Quality of instructions

A critical question is whether the instructions in the prompts of ToT-Arithmetic regarding the format of the output are of high quality and easily understood by large language models (LLMs). Notably, Gemini 1.5 Pro, GPT-4, and Claude-3-Sonnet missed the instructions only 0.05%, 0.65%, and 4.76% of the time, respectively. This low error rate suggests that the instructions are of high quality. Consequently, we were able to establish a benchmark that allows us to focus purely on the temporal reasoning abilities of the models.

### D Large-Scale ToT-Semantic Experiments

To facilitate a more comprehensive analysis and enable deeper insights, we expanded our synthetic dataset significantly. This enlarged dataset now encompasses approximately 50,000 examples, a substantial increase from the previous set of around 3,000 examples. We anticipate that this expanded resource will prove valuable for future research endeavors that necessitate a larger and more diverse synthetic dataset. Due to the computational demands of evaluating all LLMs on this large dataset, results are reported solely for Gemini 1.5 Pro.

Impact of Graph Structure on LLM Accuracy. Our initial experiment with this expanded dataset involved replicating the graph structure analysis. As illustrated in Table 10, graph structure continues to exert a significant influence on the final accuracy of the LLM, even within this larger dataset.

Table 10: LLM temporal reasoning by graph structure on the larger set of ToT-Semantic.

Graph Structure Accuracy (%)

BA 70.96 Complete 51.07 ER 61.85 SBM 60.32 SFN 79.13 Star 73.77 AWE 88.72

Average 69.40

Impact of graph structure and temporal task on LLM performance. Our second experiment examined the accuracy of the model across various question types and graph generators. The expanded dataset provided sufficient examples per category, enabling more robust results. The results are reported in Table 11. Consistent with our earlier findings, single-fact questions generally outperformed multi-fact questions. Notably, the highest accuracy was observed for EventAtWhatTime in single-fact questions and RelationDuration in multi-fact questions. This alignment with the results from the smaller dataset reinforces their significance and suggests that the smaller dataset serves as a reliable proxy for the larger one.

Impact of Graph Structure and order of facts on LLM Performance. In this experiment, we evaluated LLM performance across various combinations of graph structure and fact order. The results, presented in Table Table 12, reveal that the target_and_start_time ordering consistently yields the best performance across the expanded dataset, regardless of graph structure. Conversely, the shuffle ordering consistently underperforms across most graph structures.

### E Instructions to participants

For the crowd-sourcing section in creating the ToT-Arithmetic dataset (Expand step), we gave the following instructions to the annotators.

Table 11: Impact of graph structure and question type on a larger set of ToT-Semantic.

Temporal task BA Complete ER SBM SFN Star AWE Average Rank EventAtTimeT 74.46 54.22 65.54 68.07 80.84 76.75 91.93 3.57 EventAtWhatTime 98.19 81.69 90.72 90.48 98.31 98.43 97.95 1.00 BeforeAfter 53.49 34.46 48.07 45.66 68.55 58.80 73.98 7.00 EventAtTimeOfAnotherEvent 76.99 52.89 62.53 65.18 84.82 85.78 90.48 3.79 FirstLast 70.84 49.04 61.69 55.66 87.23 68.80 92.53 4.43 NumberOfEventsInTimeInterval 57.71 40.84 54.22 49.64 64.22 70.84 83.73 6.14 RelationDuration 88.55 80.60 83.49 82.77 87.47 88.80 90.48 2.36 Timeline 47.47 14.82 28.55 25.06 61.57 41.93 88.67 7.71

Time Arithmetic Benchmark Compilation

Thank you for participating in our eval hour to help us expand our dataset to cover all the categories of time arithmetic that we can think of.

#### Terminology:

- • Time arithmetic: Calculations with time values, often involving years, months, days, hours, minutes, seconds.
- • Category: A high-level category of time arithmetic operations, such as addition/subtraction, time conversion, etc.
- • Examples: Real-life sentences that fall into a category. For instance, "Today is 27 July 2020 and I was told that my furniture will be delivered to me in exactly 60 days from now. On what date will the furniture be delivered?" is an example of addition.

Goal: Our goal is to cover as many real-life categories and subcategories related to time arithmetic as possible. We also want each subcategory to have multiple different real-life examples.

#### Levels of Importance of Contributions:

- 1. Discovering/adding a new category.
- 2. Adding new real-life examples within a subcategory (please contribute more in less densely populated areas).

Corner cases are useful, but please don’t focus all your time on them. Discovering broader categories would be the most useful!

Please try to add new examples which are as different from existing ones as possible. Thanks!

- F Limitation and future work The current work has several limitations that provide avenues for future research:

Single-Sentence Time Anchoring . This benchmark focuses on scenarios where the start and end times of a fact are both mentioned within a single sentence. However, in real-world scenarios,

Table 12: Impact of graph structure and sorting type on a larger set of ToT-Semantic.

Order of facts BA Complete ER SBM SFN Star AWE Rank Average relation_and_start_time 73.42 52.03 64.98 61.45 81.93 74.32 90.36 2.00 shuffle 66.72 44.65 54.74 54.14 74.17 72.74 85.02 4.71 start_time_and_relation 67.55 46.31 57.76 55.72 77.86 72.14 88.48 4.00 start_time_and_target 68.60 46.61 58.96 55.95 78.31 70.78 88.63 3.29 target_and_start_time 78.54 65.74 72.82 74.32 83.36 78.84 91.11 1.00

temporal information can be spread across multiple sentences or even documents. It is worth noting that this setup is easily convertible to the more general case where temporal information can be spread across multiple sentences. While we chose to focus on the single-sentence setup for this initial work, future research could readily adapt the benchmark to the multi-sentence scenario and explore the challenges and opportunities it presents.

Exclusive Focus on Explicit Temporal Facts (By Design). This benchmark intentionally focuses solely on explicit temporal facts (those with clear time anchors), excluding static facts (those without time anchors). This deliberate choice was made to ensure the benchmark specifically targets and assesses models’ capabilities in temporal reasoning. However, future work could expand the scope to include static facts, offering a more comprehensive evaluation of both temporal and general factual reasoning.

### G Potential negative societal impacts

While our research aims to enhance the temporal reasoning capabilities of LLMs, it is important to acknowledge potential negative societal impacts. The advancements made in temporal reasoning could potentially be exploited to generate misleading or manipulative content, such as fake news articles with fabricated timelines. Additionally, if these models are primarily accessible to certain groups or organizations, it could lead to an uneven distribution of information power, potentially exacerbating existing societal inequalities. Furthermore, as LLMs become more adept at understanding and generating temporal information, there could be an increased risk of privacy violations, as these models might inadvertently reveal sensitive personal details or historical events that individuals would prefer to keep private. It is crucial to consider these potential negative consequences alongside the technological advancements to ensure responsible and ethical development of LLM technologies.

### H Dataset documentation: datasheet

In this section, we follow the recommendations in Gebru et al. [22] to provide comprehensive documentation for our dataset.

#### H.1 Motivation

- • For what purpose was the dataset created? The dataset was created to address the limitations of existing temporal reasoning benchmarks and to provide a more comprehensive and controlled assessment of LLMs temporal reasoning abilities. The dataset aims to provide a more rigorous and informative evaluation of LLM temporal reasoning abilities, contributing to the development of more advanced and reliable AI systems.
- • Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? The dataset was created by a team of ML researchers at Google all listed as authors of the paper. The authors represent various divisions within Google, including Google Research, Google DeepMind, and Google Cloud teams.
- • Who funded the creation of the dataset? The dataset was created by Google employees as part of their work at the company, so Google funded its development.

#### H.2 Composition

- • What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Each instance within the dataset represents a text-only data point as sentences describing the reasoning problem to be used as a prompt for the LLM.
- • How many instances are there in total (of each type, if appropriate)? Our ToT benchmark contains 46,480 data points in the ToT-Semantic dataset and 2,800 data points in the ToT-Arithmetic dataset. A sample of size 2800 from ToT-Semantic is also created for smaller scale experiments.
- • Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? We included both larger set and smaller sample set.

- • What data does each instance consist of? Each data point within the dataset represents a text-only data point as sentences describing the reasoning problem to be used as a prompt for the LLM.
- • Is there a label or target associated with each instance? Each data point has a label as ground truth associated to that.
- • Is any information missing from individual instances? No. All instances have both their text and label available.
- • Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? There is no relationship between different instances.
- • Are there recommended data splits (e.g., training, development/validation, testing)? This dataset serves as a benchmark for evaluating LLMs on temporal reasoning and therefore only a test split has been provided to evaluate models on it.
- • Are there any errors, sources of noise, or redundancies in the dataset? The dataset creation process is automated, but manual sampling indicates no errors were found in the examined subset. While rare errors may still exist, they haven’t been detected in the samples reviewed.
- • Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? The dataset is self-contained and does not link to external resources.
- • Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? The dataset does not contain any confidential or sensitive information.
- • Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? The dataset does not contain any data that could be considered offensive, insulting, threatening, or anxiety-inducing.
- • Does the dataset identify any subpopulations (e.g., by age, gender)? The dataset does not identify or contain any information that would allow for the identification of subpopulations based on attributes.
- • Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? The dataset does not contain any information that could be used to directly or indirectly identify individuals, either on its own or in combination with other data.
- • Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? The dataset does not contain any sensitive data that could reveal attributes.

#### H.3 Collection

- • How was the data associated with each instance acquired? Each data point is a temporal reasoning question and is generated automatically using codes. The details can be found in the main paper.
- • What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? The data collection process primarily involved a combination of automated procedures and manual human input. The majority of the data collection was performed using software programs and scripts that were developed and executed by the authors. These programs included algorithms and techniques designed to generate and curate the specific types of data required for the dataset (details in the paper). The authors played a direct role in curating the seed questions used to initiate or guide certain data collection processes. This human input involved selecting relevant topics or prompts to ensure the collected data met the desired criteria. This combination of automated and manual methods allows for both efficiency

- and a degree of human oversight, which can help ensure the quality and relevance of the collected data.
- • If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? While the primary dataset isn’t derived from another, a smaller subset was created for specific experimental needs. This subset was generated through random sampling, utilizing a fixed seed to ensure reproducibility and allow for consistent comparisons across different runs.
- • Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? The data collection process was conducted solely by the authors of the paper, who are full-time Google employees. As such, they were compensated through their regular salaries and did not receive any additional payment specifically for this task.
- • Over what timeframe was the data collected? The data collection process took place over a four-month period, spanning from March to June, leading up to the project deadline.
- • Were any ethical review processes conducted (e.g., by an institutional review board)? No.
- • Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? While the majority of the dataset was not collected, a small portion of the data collection process did involve direct interaction and collection.
- • Were the individuals in question notified about the data collection? Yes. The description is provided in the appendix
- • Did the individuals in question consent to the collection and use of their data? Yes.
- • If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? No. The full description is provided in the main appendix of the paper.
- • Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? No.

- H.4 Uses

- • Has the dataset been used for any tasks already? The dataset is used for temporal reasoning in this paper. The scope is temporal reasoning but different models can be evaluated by the same setup.
- • Is there a repository that links to any or all papers or systems that use the dataset? The dataset is generated and used in this paper only.
- • What (other) tasks could the dataset be used for? These datasets can be used for any tasks related to temporal reasoning.
- • Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? No.
- • Are there tasks for which the dataset should not be used? No.

- H.5 Distribution

- • Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? Yes, the dataset is available publicly in Huggingface.
- • How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? The dataset is distributed through Huggingface (https://huggingface.co), and the code used for generating the dataset will be available in GitHub.
- • When will the dataset be distributed? The dataset is available for reviewers at the time of submission and will be open to public soon.

- • Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? In the spirit of open science and collaboration, we have released the datasets under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. For comprehensive details about the terms of the CC BY 4.0 license, please visit the Creative Commons website: https://creativecommons. org/licenses/by/4.0/.
- • Have any third parties imposed IP-based or other restrictions on the data associated with the instances? Please refer to the copyright.
- • Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? Please refer to the copyright.

#### H.6 Maintenance

- • Who will be supporting/hosting/maintaining the dataset? Google (the authors listed here in particular) will support, host, and maintain the dataset.
- • How can the owner/curator/manager of the dataset be contacted (e.g., email address)? The owner (Bahare Fatemi) can be contacted through baharef@google.com.
- • Is there an erratum? No. If errors are found in the future, we will release errata using the same link.
- • Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? Yes, the datasets will be updated as needed to ensure accuracy.
- • If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? N/A.
- • Will older versions of the dataset continue to be supported/hosted/maintained? Yes, older versions of the dataset will continue to be maintained and hosted.
- • If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? Yes, others are welcome to extend, augment, build on, or contribute to the dataset. They are encouraged to download the dataset, create their own modified versions, and publish their work on their preferred platform.

### I Accessibility

The datasets created in this research are available for download at https://huggingface.co/ datasets/baharef/ToT. In the spirit of open science and collaboration, we have released the datasets under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. For comprehensive details about the terms of the CC BY 4.0 license, please visit the Creative Commons website: https://creativecommons.org/licenses/by/4.0/.

### J Reproducibility

In our pursuit of reproducibility, we conducted experiments with three prominent model families: Gemini [47], Claude [5], and GPT-4 [1]. Specifically, we investigated the capabilities of Gemini 1.5 Pro, Claude’s Sonnet model, and GPT-4 Turbo. All of these variants are publicly accessible through various platforms, ensuring transparency and enabling further research. We utilized the Google Cloud Platform as our computational infrastructure to execute these experiments. This choice was motivated by its comprehensive suite of machine learning tools it offers.

