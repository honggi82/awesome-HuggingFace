## HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches

### Jiejun Tan12*, Zhicheng Dou1†, Yan Yu2, Jiehan Cheng12, Qiang Ju2, Jian Xie2, Ji-Rong Wen1

1Gaoling School of Artificial Intelligence, Renmin University of China 2Baichuan Intelligent Technology {zstanjj, dou, jrwen}@ruc.edu.cn

# arXiv:2508.08088v1[cs.IR]11Aug2025

##### Abstract

Recently, large reasoning models have demonstrated strong mathematical and coding abilities, and deep search leverages their reasoning capabilities in challenging information retrieval tasks. Existing deep search works are generally limited to a single knowledge source, either local or the Web. However, enterprises often require private deep search systems that can leverage search tools over both local and the Web corpus. Simply training an agent equipped with multiple search tools using flat reinforcement learning (RL) is a straightforward idea, but it has problems such as low training data efficiency and poor mastery of complex tools. To address the above issue, we propose a hierarchical agentic deep search framework, HierSearch, trained with hierarchical RL. At the low level, a local deep search agent and a Web deep search agent are trained to retrieve evidence from their corresponding domains. At the high level, a planner agent coordinates low-level agents and provides the final answer. Moreover, to prevent direct answer copying and error propagation, we design a knowledge refiner that filters out hallucinations and irrelevant evidence returned by low-level agents. Experiments show that HierSearch achieves better performance compared to flat RL, and outperforms various deep search and multisource retrieval-augmented generation baselines in six benchmarks across general, finance, and medical domains. 1

### Introduction

Recently, large reasoning models (LRMs) such as DeepSeek-R1 (DeepSeek-AI et al. 2025) and OpenAI’s O-series (Openai 2025b) models have shown impressive capabilities in mathematics and coding. However, LRMs are troubled by higher hallucination rates (Chowdhury et al. 2025; Vectara 2025; Sun et al. 2025b) and restricted by limited internal knowledge in knowledge-intensive tasks. Thus, studies have combined LRMs with retrievalaugmented generation (RAG) to enable models to obtain external knowledge assistance, which is referred to as deep search (Li et al. 2025a,b).

Existing deep search works often equip LRMs with a local corpus search tool (Chen et al. 2025; DeepSeek-AI et al.

*This work was done when Jiejun Tan was doing an internship

at Baichuan. †Corresponding author. 1Code and datasets are available at https://github.com/plageon/

HierSearch

2025; Song et al. 2025) or a Web search tool (Li et al. 2025a,b; Zheng et al. 2025). However, a common scenario for most enterprises is that their private deep search system interacts with both local knowledge sources and Web knowledge sources (Yu et al. 2025). To be specific, enterprises often possess private domain-specific documents. Existing methods for building private RAG systems usually involve processing them into a text chunk corpus and constructing knowledge graphs (Edge et al. 2024; Guo et al. 2024; Zhao et al. 2025). Web knowledge sources generally include search engines and web pages. Generally speaking, local knowledge sources are more professional and targeted. Meanwhile, Web knowledge sources are more comprehensive and timely (Zhao et al. 2024b; Wang et al. 2024a). This deep search scenario with multiple knowledge sources poses challenges to existing methods: Deep search agents need to selectively use different knowledge sources based on user questions and the characteristics of knowledge sources, and cross-supplement missing knowledge.

A straightforward solution for the above challenge is equipping the deep search agent with all search tools for all knowledge sources and conducting flat reinforcement learning (RL). However, the flat RL solution is not suitable for the following reasons: (1) Numerous search tools result in a large action space during RL, leading to low training efficiency and instability. (2) Search tools within the same knowledge source have stronger synergy (e.g., browsing a Web page via a URL retrieved by a search engine or retrieving text chunks mentioning an entity from the knowledge graph), while that between tools across different knowledge sources is weaker. However, flat RL fails to effectively utilize this characteristic. (3) Moreover, preliminary experiments show that during flat RL, rewards encourage the agent to search more frequently in easily retrievable knowledge sources, while less frequently in hard ones (Web search is more difficult in our setting due to a wider search scope and more noise). Thus, the training efficiency of flat RL for the difficult knowledge source is poor due to limited exploration of the corresponding tools.

To address the above issues, we propose a hierarchical agentic deep search paradigm, HierSearch, which comprises a local deep search agent, a Web deep search agent, and a planner agent. Two deep search agents interact directly with search tools within their knowledge sources and re-

trieve evidence for the planner agent. Specifically, the local deep search agent has access to the local text chunk corpus and the local knowledge graph. The Web deep search agent has access to the Web search engine and online web pages. Meanwhile, the planner agent drafts search plans, coordinates search agents, analyzes evidence provided by search agents, and provides the final answer.

Accordingly, we leverage a hierarchical reinforcement learning (HRL) (Pateria et al. 2022) algorithm to train this hierarchical agentic framework. Also, we use Group Relative Policy Optimization (GRPO) (Shao et al. 2024) and rule-based rewards. HRL overcomes the challenges above, mainly manifested in: (1) In the first stage, we train lowlevel agents, the local deep search agent and the Web deep search agent separately. They master search tools within the same domain well, because the number of tools is limited and the tools are closely related. (2) In the second stage, we train the high-level planner agent, equipped with both deep search agents. Well-trained deep search agents mask the complex interaction process with search tools, and greatly lower the difficulty of knowledge acquisition. The planner agent can learn search planning and knowledge integration across multiple knowledge sources faster and better.

In the planner agent’s training stage, we find that directly providing the complete trajectories of deep search agents would introduce irrelevant search results and the agents’ hallucinatory reasoning contents. To address this, we design a reasoning-aware knowledge refiner. This refiner first selects the evidence that contributes to each round of reasoning by the deep search agent. Second, it selects the evidence helpful to the agent’s conclusion from an overall perspective.

We conduct extensive experiments on six benchmarks from the general domain, the medical domain, and the financial domain. The results show that HierSearch outperforms baselines and the flat RL solution across all benchmarks.

In summary, our contributions are threefold: (1) We explore the deep search framework in multi-knowledge-source scenarios and propose a hierarchical agentic paradigm and train with HRL; (2) We notice drawbacks of the naive information transmission among deep search agents and developed a knowledge refiner suitable for multi-knowledgesource scenarios; (3) Our proposed approach for reliable and effective deep search across multiple knowledge sources outperforms existing baselines the flat-RL solution in various domains.

### Related Works

Deep Search Traditional RAG combines large language models (LLMs) with information retrieval to provide external knowledge and mitigate hallucination (Zhou et al. 2024a; Fan et al. 2024; Jin et al. 2025b). Traditional RAG methods generally follow a fixed retrieve-then-generate pipeline (Dong et al. 2024; Tan et al. 2025; Jin et al. 2025b), while several works explore flexible agenitic pipelines (Asai et al. 2024; Yao et al. 2023). Compared to traditional RAG, deep search combines LRM with search tools (Li et al. 2025c; Gao et al. 2025). Equipped with stronger reasoning

abilities, deep search pushes iterative RAG further, and analyzes deeper for users’ questions (Li et al. 2025b), which can “search, read and reason until best answer found” (JinaAI 2025). Several organizations have developed their representative products, such as Google (Google 2025), OpenAI (Openai 2025a), and Jina (JinaAI 2025). Meanwhile, several researchers builds deep search on open-source LRMs (DeepSeek-AI et al. 2025; Team 2025; Yang et al. 2025a), like RAG-Star (Jiang et al. 2025), Search-o1 (Li et al. 2025a), and WebThinker (Li et al. 2025b). These works often have issues of excessive reasoning and inaccurate searching in search tasks, but they have the advantage of greater flexibility in choosing models and search tools (Lee et al. 2025; Wu, Zhu, and Liu 2025; Huang et al. 2025). To make reasoning models perform better in deep search tasks, another branch of works train LLMs to conduct deep search tasks following the RL paradigm introduced by DeepSeek-R1 (DeepSeek-AI et al. 2025), like DeepResearcher (Zheng et al. 2025), R1-Searcher (Song et al. 2025), and ReCall (Chen et al. 2025). The aforementioned deep search works are all limited to a single knowledge source, and at most two search tools (Jin et al. 2025a; Sun et al. 2025a). However, enterprise private deep search often needs to access local and Web knowledge sources as well as multiple search tools. Existing methods cannot supplement knowledge and handle knowledge conflicts across multiple knowledge sources. Moreover, they produce a lot of unnecessary search tool calls, especially expensive Web search tool calls. In contrast, HierSearch uses multiple deep search agents to tackle different knowledge sources, and a planner agent that selectively calls agents of different knowledge sources as needed and integrates knowledge from these sources.

Multi-Knowledge Source RAG In traditional RAG research, some works have identified the challenges RAG faces in multi-knowledge-source scenarios and proposed solutions. PruningRAG (Yu et al. 2025) uses multi-granularity pruning strategies to integrate information from documents of different sources and mitigate the impact of misleading information. PrefRAG (Zhao et al. 2024b) introduces preference-driven adaptive retrieval to handle multi-retrieval source data, and calls web retrieval as a supplement when local retrieval does not satisfy knowledge requirements. HMRAG (Liu et al. 2025) applies multi-source agents to conduct retrieval in parallel, and uses consistency voting to integrate multi-source answers. These works are still static RAG paradigms that need to follow a predefined pipeline. They use prompting or DPO methods to enable agents to learn multi-source RAG tasks. In contrast, we apply the GRPO RL algorithm to develop the agent’s deep thinking and searching capabilities. Agents with deep thinking capabilities are not limited to a specific search path; instead, they can independently plan when to call search tools, when to interact with other agents, and when to terminate.

Hierarchical RL HRL decomposes complex tasks into simpler subtasks and uses a high-level policy to select subtasks and a low-level policy to perform specific actions (Vezhnevets et al. 2017; Dayan and Hinton 1992; Diet-

Planner Agent

Web Deep Search & Refine

Local Deep Search & Refine

Dual-Source Deep Search & Refine

Web Evidences

Local Evidences Query ... T

[Figure 1]

Local & Web Evidences

Query ...

Query ...

System User T T T FinalAnswer

[Figure 2]

[Figure 3]

[Figure 4]

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

Local Agent Web Agent Refiner Local Agent Refiner Local Agent Refiner

Local Deep Search Agent

Web Deep Search Agent

[Figure 18]

[Figure 19]

###### ...

...

...

...

System T Sub Query Snippets

T

Sub Query Triplets

Query

System Query Chunks

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Graph Search

Web Search

Chunk Search

T Sub Query Web Page ... T

Temp Answer

Sub Query Chunks ... T

Temp Answer

T

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Browse Web Page

Get Adjacent Passages

[Figure 33]

[Figure 34]

- Figure 1: Illustration of the hierarchical agentic framework for HierSearch. We show exemplary trajectories of all low-level and high-level agents.

Preliminary: Tool-Augmented Reasoning We follow a commonly used synergized tool-augmented reasoning paradigm of current deep search methods (Li et al. 2025c). Our deep search agents and the planner agent roll out similarly. We use the following wrapping tags to distinguish different part in the trajectory: (1) The thinking processes are wrapped in <think>...</think>; (2) Tool calls are wrapped in <tool name>...</tool name> (The tool name varies). (3) Returned contents tool functions are wrapped in <result>...</result>. (4) The answer is wrapped in <answer>...</answer>. All tools accessible are demonstrated in the system prompt. The generation process pauses when the ending tags of tool calls are generated, and restarts until the tool call result is appended to the end of the sequence. The whole generation process ends when </answer> is generated or the number of tool call rounds reaches an upper limit.

terich 2000). HRL is effective and data-efficient when used in tasks with multiple turns, long horizons, and delayed rewards (Pateria et al. 2022; Hutsebaut-Buysse, Mets, and Latr´e 2022). HRL has performed well in robot control and game AI (Nachum et al. 2018; Kulkarni et al. 2016; Zhang, Yu, and Xu 2021). Recent works have also applied HRL to agents built on LLMs (Zhou et al. 2024b; Zhao et al. 2024a). To the best of our knowledge, this work is the first to use HRL in the deep search field. Multi-knowledge-source RAG task is broken down into two levels: in-knowledge-source deep search and cross-knowledge-source planning.

### Methodology

We present HierSearch, a hierarchical agentic framework designed for enterprise-wide deep search across multiple knowledge sources. The framework comprises two levels: 1) low-level agents, including local and Web deep search agents, and 2) a high-level planner agent.

Local Deep Search Agent The local deep search agent has access to two local knowledge sources: the text chunk corpus and the knowledge graph. The local agent accesses the text chunk corpus through <chunk search> to retrieve chunks related to the input query. The local agent accesses knowledge graph by two tools: (1) <graph search> retrieves triples (consisting of a subject, a predicate, and an object) related to the input query by calculating the similarity of semantic embeddings; (2) <get adjacent passages> returns relevant text chunks mentioning the input entity in the knowledge graph. The linking edges between graph entities and relevant chunks are identified and saved during the knowledge graph construction process.

#### Problem Formulation

In a deep search task, the agent takes a user’s question x, iteratively performs thinking processes or search tool calls, and finally outputs an answer yˆ. The optimization goal is to make the final answer as correct and helpful as possible. In the enterprise scenario, a deep search needs to access multiple knowledge sources before providing an answer. Given knowledge sources including a local text chunk corpus C, a local knowledge graph G, a Web search engine E, and accessible Web pages on the Internet P, the deep search framework is meant to maximize the probability of the golden answer y.

Web Deep Search Agent The Web deep search agent accesses Web knowledge through two tools: (1) <web search> calls a search engine API to retrieve web links and corresponding titles and snippets related to the input query; (2) <browse url> takes both a web link and a query as inputs. We chunk the original HTML pages and only return query-relevant pieces, because the original HTML pages are generally lengthy and hard to read.

#### Hierarchical Agentic Deep Search

A straightforward idea for the multi-knowledge-source deep search task is equipping an agent with all search tools and conducting flat RL. However, our preliminary experiment shows that the flat RL displays drawbacks such as poor mastery of difficult Web search tools and low training data efficiency. Thus, we propose a hierarchical agentic deep search framework, HierSearch. As shown in Figure 1, HierSearch consists of a local deep search agent, a Web deep search agent, and a planner agent. We will discuss all three agents in the following sections in detail, including their accessible tools and their roles.

Multi-Knowledge Source planner agent Both the local deep search agent and the Web deep search agent are lowlevel agents that are manipulated by a high-level planner agent. The planner agent drafts search plans, integrates re-

Collect directly contributing evidences

[Figure 35]

Question: Who is the father of the Labo M performer? <think> … To answer this question, I need to search for the performer associated with Labo M. </think> <chunk_search> performer associated with Labo M </chunk_search> <result> Labo M (2003) is the third studio album by French singer-songwriter Matthieu Chedid… Mimi Goese Mimi Goese (last name rhymes with "hazy") is a professional musician… LaKiesha Berri LaKiesha Berri (born 1974, Cincinnati, Ohio) is an African-American R&B…</result> <think> The chunk search result shows that the performer associated with Labo M is Matthieu Chedid. Now, I will find the father of Matthieu Chedid using the graph search tool. </think> <graph_search> Matthieu Chedid father </graph_search>

Directly contributing evidences: Labo M (2003) is the third studio album by French singer-songwriter Matthieu Chedid…

[Figure 36]

Filter by thinking process

[Figure 37]

###### LocalAgentRollout

Other evidences: Mimi Goese Mimi Goese (last name rhymes with "hazy") is a professional musician… LaKiesha Berri LaKiesha Berri (born 1974, Cincinnati, Ohio) is an African-American R&B…

Collect globally contributing evidences

Filter by temporary answer & crossdomain agent’s answer

Globally contributing evidences: Matthieu Chedid Matthieu Chedid was born in … the son of French singer Louis Chedid… [Subject]: labo m [Predicate]: is an instrumental work by [Object]: matthieu chedid…

Directly contributing evidences: [Subject]: matthieu chedid [Predicate]: has sister [Object]: milie chedid…

| | |
|---|---|
|Filter by| |

<result> [Subject]: matthieu chedid [Predicate]: has sister [Object]: milie chedid [Subject]: matthieu chedid [Predicate]: is son of [Object]: louis chedid… </result>

[Figure 38]

Other evidences: [Subject]: matthieu chedid [Predicate]: has sister [Object]: milie chedid [Subject]: matthieu chedid [Predicate]: is [Object]: french singer songwriter…

[Figure 39]

<think> The graph search result shows that the father of Matthieu Chedid is Louis Chedid. Therefore, the father of the Labo M performer is Louis Chedid. </think> …

thinking process

[Figure 40]

[Figure 41]

Web Agent Answer: Louis Chedid

… <answer> Louis Chedid </answer>

- Figure 2: Illustration of the knowledge refining process from the local agent trajectory. The first step filters directly contributing evidence according to the subsequent thinking process of each round. The second step filters globally contributing evidence according to the local agent’s answer and the web agent’s answer (if available).

turned evidence from low-level agents, and provides the final answer. Low-level agents are packaged as tools for high-level agents to call, which includes the following: (1) <local search agent> calls the local deep search agent; (2) <web search agent> calls the Web deep search agent; (3) <all search agent> calls both lowlevel agents simultaneously.

from low-level agents induce the planner agent directly copy them instead of thinking by itself; (2) Irrelevant evidence makes the contents low-level agents’ returned lengthy and hard to read and confuses the planner agent; (3) The hallucinations generated by low-level agents lead to an error propagation to the planner agent.

Therefore, we design a knowledge refiner that filters key evidence contributing to the low-level agents’ thinking processes and conclusions, as shown in Figure 2. The refiner filters evidence helpful for the thinking process in two steps. In the first refining step, we select evidence directly contributing to the next thinking process. Given a trajectory sequence S, which contains an input question x, and K rounds where thinking and tool calls alternate, and ends with a last thinking process tK+1 followed by a conclusion cˆ. The round k contains a thinking process tk, a query qk, and N returned evidence {eN(k−1)+1 ···eNk}. The trajectory sequence is like:

Hierarchical RL for Multi-Source Deep Search Considering the hierarchical framework and taking inspiration from HRL works, we employ HRL for HierSearch. That is, we first train two low-level search agents, and then the highlevel planner agent. To be specific, we randomly sample the training set from MuSiQue (Trivedi et al. 2022), OmniEval (Wang et al. 2024b), and BioASQ (Nentidis et al. 2024). We mix these samples as the training data for agents.

We follow the GRPO algorithm introduced by DeepSeekR1 (DeepSeek-AI et al. 2025), and we use rule-based rewards, which are designed as follows. Agent trajectories with incorrect formats are punished with a zero reward. If the format is correct, we calculate the F1 score between the predicted answer yˆ and the golden answer y. If the F1 score is larger than zero, we take the F1 score as the reward. If the rollout has a correct format but a zero F1 score, we encourage the agent to explore more tools. We calculate the proportion of the types of tools used during the rollout to the total types of tools accessible to the agent, and multiply it by a coefficient of 0.1 to serve as the reward. To sum up, the reward function can be formulated as:

S = {x,t1,··· ,tk,qk,eN(k−1)+1 ···eNk,··· ,tK+1,cˆ}

(2) The contribution score for each evidence in round k is given by its contribution to the next thinking process:

Score(ei) = P(ei|tk+1), N(k − 1) + 1 ≤ i ≤ Nk (3)

The contribution score is calculated by the embedding similarity score P between the evidence and the subsequent thinking process. In the first step, in each think & search round, top α% evidence is selected.

 

In the second refining step, we distinguish evidence not selected in the first step but contributing globally to the agent’s conclusion. As preparation, unselected evidence after the first step is gathered as candidates. If the planner agent calls only one low-level agent, we consider only that low-level agent’s conclusion cˆ. If the planner agent calls both low-level agents, we concatenate cˆ with the other agent’s conclusion cˆ

0, if the format is incorrect, 0.1 × t/T, if F1 = 0 and format is correct, F1(ˆy,y), if F1 > 0 and format is correct.

(1)

R =



, where t is the number of tools used in the trajectory and T is the number of all tools accessible.

′

′

}, and consider them as a whole. The global contribution score for the conclusion is given by:

as {c,ˆ cˆ

#### Reasoning-Aware Knowledge Refiner

This hierarchical framework requires information exchange between low-level deep search agents and the high-level planner agent. A straightforward idea is that low-level agents return the whole trajectory containing collected evidence (search results from search tools), thinking processes, and conclusions (temporary answers in answer tags). However, analytical experiments show that inputting all those information indiscriminately will be harmful for the planner, which mainly shows in: (1) Thinking processes and conclusions

′

′

exists,

P(ei|{c,ˆ cˆ

}), cˆ

Score(ei) =

(4)

P(ei|cˆ), otherwise.

In the second step, the top β% of the remaining evidence is selected. Both α and β are hyperparameters of the refiner. Finally, evidence selected from the two steps is merged and tagged with its knowledge source. The planner receives a list of refined evidence collected by agents (e.g.,

MuSiQue OmniEval BioASQ NQ HotpotQA PubmedQA # Searches EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 Local Web Local Search

Method

DeepSeek-R1 26.00 36.45 0.80 29.50 6.18 24.10 28.50 44.88 29.75 45.36 41.00 51.35 2.00 HippoRAG 30.25 43.36 0.00 29.27 9.71 36.87 43.25 59.71 35.25 52.23 68.50 70.95 2.00 R1-Searcher 44.50 55.86 2.93 9.85 34.12 50.87 44.50 56.92 48.25 63.88 64.00 64.12 1.68 -

- ReCall 42.75 53.82 8.53 23.01 24.71 43.30 47.50 61.09 49.50 63.99 28.00 34.64 2.55 -

Web Search

DeepSeek-R1 22.50 32.60 0.53 24.23 5.29 20.25 26.75 39.89 26.50 40.31 15.25 30.22 - 1.00 DeepResearcher 30.00 39.44 2.40 17.95 28.82 46.80 41.50 54.99 39.50 52.95 56.25 56.79 - 2.84 Search-o1 28.50 39.03 3.20 15.37 30.59 47.24 36.00 48.79 42.00 53.80 64.00 67.19 - 1.72 WebThinker 30.75 42.15 1.33 15.90 33.24 49.82 36.75 50.52 43.50 58.68 65.00 66.07 - 2.55

Parallel Search

DeepSeek-R1 26.50 37.47 1.07 28.31 4.41 22.34 23.75 39.51 28.50 44.37 40.25 50.13 2.00 1.00 HippoRAG 33.25 46.39 0.00 29.69 10.29 37.29 43.00 59.88 39.75 57.70 70.25 70.93 2.00 1.00 HM-RAG 26.25 37.59 7.73 35.93 13.53 39.01 43.75 59.76 44.00 59.50 71.25 71.29 5.27 2.64 R1-Searcher 46.50 57.19 2.67 9.22 33.82 50.54 44.75 56.97 47.75 62.93 66.25 66.52 3.36 1.68

- ReCall 43.00 52.69 9.33 22.02 26.18 42.45 48.25 61.13 47.00 62.12 31.75 39.34 4.62 2.31 DeepResearcher 33.75 44.94 6.40 24.96 32.94 52.44 46.25 59.76 45.75 60.23 64.75 65.50 4.20 2.10 Search-o1 36.25 47.53 5.60 18.82 32.06 50.18 39.25 53.36 44.00 59.13 65.50 68.93 3.10 1.55 WebThinker 33.00 44.53 5.60 19.87 33.82 50.54 40.25 53.39 46.75 61.04 67.75 69.04 4.38 2.19

###### Selective Search

CRAG 26.50 36.89 1.07 28.50 5.88 23.76 25.00 42.25 30.00 45.43 41.50 51.76 2.00 0.61 PrefRAG 33.75 47.47 9.60 40.19 11.18 38.47 40.00 57.01 43.50 61.56 60.25 65.29 2.18 0.04 HierSearchw/oHRL 46.00 56.34 7.73 39.49 39.41 62.42 47.75 59.65 42.00 57.99 67.50 69.31 4.82 1.02 HierSearch 53.00 62.83 10.67 46.37 49.94 66.99 57.00 68.00 53.25 67.40 71.75 72.81 5.74 1.06

Table 1: Main Results of HierSearch. The best and second best of each model are in bold and underlined.

“<result> Local Knowledge Graph: [Subject] matthieu chedid ... Search Engine: Labo M (2003) is the third studio album ... </result>”).

### Experiments

#### Benchmarks

We select three general-domain benchmarks, including: (1) MuSiQue (Trivedi et al. 2022): A synthetic multi-hop QA dataset; (2) Natural Questions (NQ) (Kwiatkowski et al. 2019): Real search engine questions collected by Google; (3) HotpotQA (Yang et al. 2018): A multi-hop QA dataset based on Wikipedia. We select one financial-domain benchmark, OmniEval (Wang et al. 2024b), a Chinese large-scale RAG benchmark targeting the financial domain with human annotations. We select two medical-domain benchmarks: (1) BioASQ (Nentidis et al. 2024): An annually updated biomedicine challenge with QA tasks; (2) PubMedQA (Jin et al. 2019): A human-annotated QA dataset based on research papers on PubMed. All benchmarks in finance and biomedicine include numerous queries that can only be answered using local knowledge. We randomly sample 373 samples for OmniEval, 340 samples for BioASQ, and 400 samples for other benchmarks from their corresponding test set (if available). We calculate Exact Match (EM) and F1 score as evaluation metrics for all benchmarks. Also, we count the average local search and Web search times (Web page browsing not included) required to process a query for each method.

#### Baselines

To demonstrate the effectiveness of our method, we select the following baselines:

- • Local Search. (1) HippoRAG(Gutierrez et al. 2024): The graph RAG backbone method, with GPT-4o-mini (Openai 2024) as the base model. (2) DeepSeek-R1 (DeepSeekAI et al. 2025): A powerful reasoning model augmented by single-time chunk search and graph search; (3) R1Searcher (Song et al. 2025) and (4) Recall (Chen et al. 2025): Both are deep search agents trained from scratch on QA datasets in local retrieval environments.
- • Web Search. (1) A powerful reasoning model augmented by single-time Web search; (2) Search-o1 (Li et al. 2025a): A deep search method that incorporates Web search into reasoning in a single inference chain; (3) WebThinker (Li et al. 2025b): A deep search method that involves a deep web explorer in a main reasoning chain; (4) DeepResearcher (Zheng et al. 2025): A deep search agent trained from scratch in real-world web environments.
- • Parallel Search. To align the knowledge sources and make a fair comparison, we reproduce the above baselines in a parallel search setting, where the same query is sent to both local and Web search tools in parallel. Also, we reproduce HM-RAG (Liu et al. 2025), which conducts parallel RAG based on text search, graph search, and Web search, and merges three answers with a majority vote.
- • Selective Search. The agent autonomously decide which knowledge source to use or both, including: (1) Pre-

###### Method MuSiQue OmniEval BioASQ NQ HotpotQA PubmedQA

HierSearch 53.00 10.67 49.94 57.00 53.25 71.75 w/o Local Agent 29.75 (23.25%↓) 3.20 (7.47%↓) 35.00 (14.94%↓) 36.00 (21.00%↓) 33.25 (20.00%↓) 65.00 (6.75%↓) w/o Web Agent 47.50 (5.50%↓) 9.87 (0.80%↓) 46.18 (3.76%↓) 55.50 (1.50%↓) 51.50 (1.75%↓) 69.50 (2.25%↓) w/o Refiner 50.75 (2.25%↓) 9.60 (1.07%↓) 48.82 (1.12%↓) 56.25 (0.75%↓) 48.50 (4.75%↓) 68.50 (3.25%↓)

Table 2: Ablation Study.

fRAG (Zhao et al. 2024b): A multi-turn RAG pipeline that decides wether to involve Web search basing on local retrieval results; (2) CRAG (Yan et al. 2024): A plug-in discriminator that decides using Web search, local search or both basing on local retrieval results; (3) HierSearchw/o HRL: A deep search agent equipped with all search tools and trained by flat RL.

#### Implementation Details

For local search, we prepare local knowledge bases separately for general, medical, and financial domains. For the general domain, we sample passages from the Wikipedia dump, and for the medical domain, we sample passages from the PubMed dump. The sampling passages consist of directly related passages for questions and hard negatives retrieved by BM25 (Robertson and Zaragoza 2009). This corpus sampling process for Wikipedia and PubMed is necessary because their original sizes are too large for constructing a graph upon them. For the financial domain, we use the original retrieval corpus of OmniEval. The knowledge graph is constructed upon the text chunk corpus. We follow HippoRAG (Gutierrez et al. 2024) and employ GPT-4o-mini (Openai 2024) and BGE-M3 (Chen et al. 2024) in graph construction. BGE-M3 is also the embedding model for all local search tools. As for the Web search, <web search> uses Bing Search API for English queries and the Quark Search API for Chinese queries. <browse url> accesses real-time Web pages and extract relevant evidence. For training settings, we collect training samples from Musique, OmniEval, and BioASQ. We train the local deep search agent, Web deep search agent, and the planner agent for 300 steps with a batch size of 64 and Qwen2.5-7B-Instruct (Yang et al. 2025b) as the backbone. More implementation details are in the appendix.

#### Main Results

Main experimental results are shown in Table 1. “# Searches” is the average local or Web search tools (Web page browsing excluded) called to search for a user’s question. Experiments demonstrate that HierSearch outperforms baselines without many additional search tool calls. Additionally, we make the following observations: (1) Baseline methods generally perform better if they have access to more knowledge sources. Local search has a larger augmentation than Web search because they are more professional and targeted. (2) Compared to methods using parallel search to access multiple knowledge sources, our method exhibits stronger deep search capabilities in multi-knowledge-source environments. Also, parallel search methods generate more

Web search tool calls, which are slow and expensive. (3) Compared to multi-knowledge-source RAG baselines using selective search, our method is not constrained by a fixed workflow in knowledge source selection and integration, and makes deeper search and thinking. As for the comparison to flat RL (HierSearchw/o HRL), we make a detailed analysis in the section below “Effectiveness of Hierarchical RL”. (4) NQ, HotPotQA, and PubmedQA are not included in our training data, so the performances on them demonstrate our method’s generalization ability in out-of-scope scenarios.

#### Further Analysis

Ablation Study We conduct an ablation study on key modules of our method, as shown in Table 2. We make the following observations: (1) We ablate the local deep search agent. In practice, we return an empty result when the planner calls the local deep search agent. Due to the lack of local information, the ultimate performance decreases. (2) Similar to the local deep search agent, when the Web deep search agent is ablated, the ultimate performance decreases due to a lack of Web knowledge. (3) We ablate the knowledge refiner. In practice, we directly provide the planner with the complete content of agent trajectories. Since the trajectories contain irrelevant search results and hallucinations from low-level agents, the overall performance is affected.

Effectiveness of Hierarchical RL To demonstrate that HRL has an edge over flat RL under the multi-knowledgesources environment, we conduct a training comparison experiment. We start from an identical backbone model, Qwen2.5-7B-Instruct, and train with identical training samples. The training batch size is set to 64, and the total number of training steps is 300. We evaluate the checkpoint every 10 steps during training with a validation set sampled from MuSiQue and OmniEval. The results for the first 200 steps are shown in Figure 3 (due to space limitations). The green curve represents HierSearch using HRL, while the orange curve represents the flat-RL-trained agent. Comparing the reward curves, we can see: (1) At the initial stage, both methods’ performances grow rapidly due to learning the tool-calling format, and the performance of HierSearch grows faster than flat RL. (2) Both methods’ performance enters a plateau on MuSiQue after 20 steps, which is the same on OmniEval after 10 steps. During this period (steps 20 to 300), both methods are improving their deep search abilities slowly with fluctuations, and HierSearch consistently performs better than flat RL.

Additionally, through case analysis (more details in the Appendix), we find that: (1) The strong performance of HierSearch benefits from the low-level agents’ stronger deep

0.6

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>HierSearch<br><br>Flat RL| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

MuSiQue

0.4

0.2

0 0 40 80 120 160 200

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>HierSearch<br><br>Flat RL| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.4

OmniEval

0.3

0.2

0.1

0

0 40 80 120 160 200

Training Steps

Figure 3: Rewards on Validation Sets during Training.

search ability compared to original search tools, as well as the refiner’s ability to refine key evidence. (2) Flat RL faces multiple search tools and a larger action space, resulting in low sample utilization efficiency. Further analysis shows that at step 290, the Web search tools only account for 18.5% of the total search tool calls, leading to low optimization efficiency.

Analysis of Multi-Knowledge Source Searching To further reveal the detailed reason that our method performs better in multi-knowledge-source environments, we analyze search success rates and reasoning success rates, and identify both of them according to different knowledge sources. To be specific, if the gold answer is contained in returned local search results, it is regarded as a successful local search. This also applies to Web search, and “both” means both local and Web search are successful. The search success rate is calculated by dividing the number of search successful samples by the total number of samples. Meanwhile, under the premise of a successful local search, if the planner agent gives a correct final answer (EM = 1), it is regarded as successful reasoning. The reasoning success rate is calculated by dividing the number of reasoning successful samples by the number of search successful samples, which are the same for the Web and both.

Results are shown in Table 3, and we make the following observations: (1) Local search is easier than Web search, while Web search supplements some knowledge. (2) Specialized deep search agents have a higher search and reasoning success rate than agents built on general reasoning models. (3) Compared to deep search baselines, our method is better at searching as well as reasoning. (4) The flat RL solution (HarmoSw/o HRL) outperforms all baselines in local search success rate and is close to our hierarchical method. However, its performance in web search success rate is unsatisfactory. This confirms our observation in preliminary experiments: the flat RL solution insufficiently explores and optimizes web search tools.

Efficiency Analysis Since we employ a hierarchical framework consisting of three agents, which may raise efficiency concerns, we make a comprehensive computational efficiency analysis, as shown in Table 4. We count the num-

Search Success (%) Reasoning Success (%) Local Web Both Local Web Both

Method

R1-Searcher 84.75 51.00 47.75 49.85 58.82 59.16 ReCall 87.50 55.75 50.75 51.71 60.09 60.10 DeepResearcher 77.25 55.00 51.00 53.72 60.45 61.27 Search-o1 70.25 39.00 35.50 44.84 40.38 40.14 WebThinker 71.50 52.00 48.75 47.20 57.69 56.41 HierSw/oHRL 89.75 23.50 22.25 51.81 62.77 61.80 HierSearch 94.25 81.25 77.75 59.15 63.38 64.63

Table 3: Multi-Knowledge-Source Utility Analysis on NQ.

Method # LS # WS # WB # Tokens Latency (s) Parallel Search

R1-Searcher 4.26 2.13 - 297.62 8.84 ReCall 4.70 2.35 - 165.15 7.70 DeepResearcher 4.42 2.21 0.01 192.57 7.72 Search-o1 3.46 1.73 16.56 1,503.71 75.43 WebThinker 5.72 2.86 25.36 4,276.77 140.83

###### Selective Search

CRAG 0.99 0.72 - 1,820.88 24.59 PrefRAG 2.08 0.05 - 1,077.02 13.63 HierSw/oHRL 5.16 1.03 1.02 334.98 10.04 HierSearch 3.54 1.06 2.23 408.68 14.79

Table 4: Efficiency Analysis on MuSiQue.

ber of local search tool calls (# LS), Web search tool calls (# WS), Web page browsing tool calls (# WB), reasoning tokens (# Tokens), and the overall latency. For parallel search baselines, we call graph search, chunk search, and Web search tools in parallel whenever the agent provides a query. The first two are local search tools, so their local search count is exactly twice that of Web search. For latency calculation, we estimate it with 43.99ms for a local search, 2.30s for a Web search, 3.16s for a Web page browsing, and 12.57ms for a reasoning token. In addition, we made the following observations: (1) Compared with the parallel search baselines, our method does not significantly increase search and reasoning cost. (2) Using the parallel search to integrate knowledge from different sources leads to unnecessary Web search tool calls, which are a lot more expensive and slower than local search tool calls. (3) Prompting reasoning models to build deep search agents significantly consumes more reasoning tokens, such as WebThinker and Search-o1. Such token consumption is of limited help for deep search tasks.

### Conclusions

In this work, we propose a hierarchical agentic paradigm that integrates local and Web searches for enterprise deep search. Our method consists of a low-level local deep search agent and a Web deep search agent that conduct deep search in their corresponding knowledge sources, and a planner agent that coordinates low-level agents and provides the final answer. Furthermore, we devise a knowledge refiner that extracts helpful evidence from low-level agents’ trajectories. Extensive experiments demonstrate that our method is effec-

tive and efficient across various domains, with better performance in searching and reasoning. This work explores the field of multi-knowledge-source deep search. We anticipate future research questions and research works in this field.

### References

Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.

- 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and Liu, Z. 2024. BGE M3-Embedding: Multi-Lingual, MultiFunctionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. CoRR, abs/2402.03216.

Chen, M.; Li, T.; Sun, H.; Zhou, Y.; Zhu, C.; Wang, H.; Pan, J. Z.; Zhang, W.; Chen, H.; Yang, F.; Zhou, Z.; and Chen, W.

- 2025. ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning.

Chowdhury, N.; Johnson, D.; Huang, V.; Steinhardt, J.; and Schwettmann, S. 2025. Investigating truthfulness in a prerelease o3 model. Technical report, Transluce.

Dayan, P.; and Hinton, G. E. 1992. Feudal Reinforcement Learning. In Hanson, S. J.; Cowan, J. D.; and Giles, C. L., eds., Advances in Neural Information Processing Systems 5, [NIPS Conference, Denver, Colorado, USA, November 30 December 3, 1992], 271–278. Morgan Kaufmann.

DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang, X.; Yu, X.; Wu, Y.; Wu, Z. F.; Gou, Z.; Shao, Z.; Li, Z.; Gao, Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li,

- G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin,
- H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J. L.; Ni, J.; Liang, J.; Chen, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang, M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen,

- Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen,
- R. J.; Jin, R. L.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye,
- S.; Wang, S.; Yu, S.; Zhou, S.; Pan, S.; and Li, S. S. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. CoRR, abs/2501.12948.

Dietterich, T. G. 2000. Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition. J. Artif. Intell. Res., 13: 227–303.

Dong, G.; Zhu, Y.; Zhang, C.; Wang, Z.; Dou, Z.; and Wen, J. 2024. Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation. CoRR, abs/2406.18676.

Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody, A.; Truitt, S.; and Larson, J. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. CoRR, abs/2404.16130.

Fan, W.; Ding, Y.; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua, T.; and Li, Q. 2024. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. In Baeza-Yates, R.; and Bonchi, F., eds., Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2024, Barcelona, Spain, August 2529, 2024, 6491–6501. ACM.

Gao, Y.; Xiong, Y.; Zhong, Y.; Bi, Y.; Xue, M.; and Wang, H. 2025. Synergizing RAG and Reasoning: A Systematic Review. CoRR, abs/2504.15909.

Google. 2025. Gemini Deep Research.

Guo, Z.; Xia, L.; Yu, Y.; Ao, T.; and Huang, C. 2024. LightRAG: Simple and Fast Retrieval-Augmented Generation. CoRR, abs/2410.05779.

Gutierrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Globersons, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J. M.; and Zhang, C., eds., Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Huang, L.; Liu, Y.; Jiang, J.; Zhang, R.; Yan, J.; Li, J.; and Zhao, W. X. 2025. ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework. CoRR, abs/2505.18105.

Hutsebaut-Buysse, M.; Mets, K.; and Latr´e, S. 2022. Hierarchical Reinforcement Learning: A Survey and Open Research Challenges. Mach. Learn. Knowl. Extr., 4(1): 172– 221.

Jiang, J.; Chen, J.; Li, J.; Ren, R.; Wang, S.; Zhao, X.; Song, Y.; and Zhang, T. 2025. RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement. In Chiruzzo, L.; Ritter, A.; and Wang, L., eds., Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, 7064–7074. Association for Computational Linguistics.

Jin, B.; Zeng, H.; Yue, Z.; Wang, D.; Zamani, H.; and Han, J. 2025a. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. CoRR, abs/2503.09516.

Jin, J.; Zhu, Y.; Dou, Z.; Dong, G.; Yang, X.; Zhang, C.; Zhao, T.; Yang, Z.; and Wen, J. 2025b. FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research. In Long, G.; Blumestein, M.; Chang, Y.; LewinEytan, L.; Huang, Z. H.; and Yom-Tov, E., eds., Companion Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025 - 2 May 2025, 737–740. ACM.

Jin, Q.; Dhingra, B.; Liu, Z.; Cohen, W. W.; and Lu, X. 2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In Inui, K.; Jiang, J.; Ng, V.; and Wan, X., eds., Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th Interna-

tional Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, 2567–2577. Association for Computational Linguistics.

JinaAI. 2025. DeepSearch: Search, read and reason until best answer found.

Kulkarni, T. D.; Narasimhan, K.; Saeedi, A.; and Tenenbaum, J. 2016. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation. In Lee, D. D.; Sugiyama, M.; von Luxburg, U.; Guyon,

- I.; and Garnett, R., eds., Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, 3675–3683.

Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A. P.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,

- J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.; Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019. Natural Questions: a Benchmark for Question Answering Research. Trans. Assoc. Comput. Linguistics, 7: 452–466.

Lee, Z.; Cao, S.; Liu, J.; Zhang, J.; Liu, W.; Che, X.; Hou, L.; and Li, J. 2025. ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation. CoRR,

- abs/2503.21729.

Li, X.; Dong, G.; Jin, J.; Zhang, Y.; Zhou, Y.; Zhu, Y.; Zhang, P.; and Dou, Z. 2025a. Search-o1: Agentic Search-Enhanced Large Reasoning Models. CoRR, abs/2501.05366.

- Li, X.; Jin, J.; Dong, G.; Qian, H.; Zhu, Y.; Wu, Y.; Wen,

- J.; and Dou, Z. 2025b. WebThinker: Empowering Large Reasoning Models with Deep Research Capability. CoRR,

abs/2504.21776.

- Li, Y.; Zhang, W.; Yang, Y.; Huang, W.-C.; Wu, Y.; Luo, J.; Bei, Y.; Zou, H. P.; Luo, X.; Zhao, Y.; Chan, C.; Chen, Y.; Deng, Z.; Li, Y.; Zheng, H.-T.; Li, D.; Jiang, R.; Zhang, M.; Song, Y.; and Yu, P. S. 2025c. Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs.

- Liu, P.; Liu, X.; Yao, R.; Liu, J.; Meng, S.; Wang, D.; and Ma, J. 2025. HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation. CoRR, abs/2504.12330.

Nachum, O.; Gu, S.; Lee, H.; and Levine, S. 2018. DataEfficient Hierarchical Reinforcement Learning. In Bengio, S.; Wallach, H. M.; Larochelle, H.; Grauman, K.; CesaBianchi, N.; and Garnett, R., eds., Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada, 3307–3317.

Nentidis, A.; Katsimpras, G.; Krithara, A.; Lima-L´opez, S.; Farr´e-Maduell, E.; Krallinger, M.; Loukachevitch, N. V.; Davydova, V.; Tutubalina, E.; and Paliouras, G. 2024. Overview of BioASQ 2024: The Twelfth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering. In Goeuriot, L.; Mulhem, P.; Qu´enot, G.; Schwab, D.; Nunzio, G. M. D.; Soulier, L.; Galusc´akov´a, P.; de Herrera, A. G. S.; Faggioli, G.; and Ferro, N., eds.,

Experimental IR Meets Multilinguality, Multimodality, and Interaction - 15th International Conference of the CLEF Association, CLEF 2024, Grenoble, France, September 9-12, 2024, Proceedings, Part II, volume 14959 of Lecture Notes in Computer Science, 3–27. Springer.

- Openai. 2024. GPT-4o mini: advancing cost-efficient intelligence. Technical report, Openai.
- Openai. 2025a. Introducing deep research. Technical report, Openai.

Openai. 2025b. OpenAI o3 and o4-mini System Card. Technical report, Openai.

Pateria, S.; Subagdja, B.; Tan, A.; and Quek, C. 2022. Hierarchical Reinforcement Learning: A Comprehensive Survey. ACM Comput. Surv., 54(5): 109:1–109:35.

Robertson, S. E.; and Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr., 3(4): 333–389.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. CoRR, abs/2402.03300.

Song, H.; Jiang, J.; Min, Y.; Chen, J.; Chen, Z.; Zhao, W. X.; Fang, L.; and Wen, J. 2025. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning. CoRR, abs/2503.05592.

Sun, H.; Qiao, Z.; Guo, J.; Fan, X.; Hou, Y.; Jiang, Y.; Xie, P.; Zhang, Y.; Huang, F.; and Zhou, J. 2025a. ZeroSearch: Incentivize the Search Capability of LLMs without Searching. CoRR, abs/2505.04588.

Sun, Z.; Wang, Q.; Wang, H.; Zhang, X.; and Xu, J. 2025b. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective. arXiv preprint arXiv:2505.12886.

Tan, J.; Dou, Z.; Wang, W.; Wang, M.; Chen, W.; and Wen, J. 2025. HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems. In Long, G.; Blumestein, M.; Chang, Y.; Lewin-Eytan, L.; Huang, Z. H.; and Yom-Tov, E., eds., Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025, 1733–1746. ACM.

Team, Q. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning.

Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Trans. Assoc. Comput. Linguistics, 10: 539–554.

Vectara. 2025. DeepSeek-R1 hallucinates more than DeepSeek-V3. Technical report, Vectara.

Vezhnevets, A. S.; Osindero, S.; Schaul, T.; Heess, N.; Jaderberg, M.; Silver, D.; and Kavukcuoglu, K. 2017. FeUdal Networks for Hierarchical Reinforcement Learning. In Precup, D.; and Teh, Y. W., eds., Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, 3540–3549. PMLR.

Wang, F.; Wan, X.; Sun, R.; Chen, J.; and Arik, S. O.¨ 2024a. Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models. CoRR, abs/2410.07176.

Wang, S.; Tan, J.; Dou, Z.; and Wen, J. 2024b. OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain. CoRR, abs/2412.13018.

Wu, J.; Zhu, J.; and Liu, Y. 2025. Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research. CoRR, abs/2502.04644.

- Yan, S.; Gu, J.; Zhu, Y.; and Ling, Z. 2024. Corrective Retrieval Augmented Generation. CoRR, abs/2401.15884.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou, F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,

- J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men, R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,

- X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang,
- Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang,
- Z.; Zhou, Z.; and Qiu, Z. 2025a. Qwen3 Technical Report. Technical report, Qwen Team.

Yang, A.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Huang, H.; Jiang, J.; Tu, J.; Zhang, J.; Zhou, J.; Lin, J.; Dang, K.; Yang,

- K.; Yu, L.; Li, M.; Sun, M.; Zhu, Q.; Men, R.; He, T.; Xu, W.; Yin, W.; Yu, W.; Qiu, X.; Ren, X.; Yang, X.; Li, Y.; Xu,

Z.; and Zhang, Z. 2025b. Qwen2.5-1M Technical Report. CoRR, abs/2501.15383.

Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Riloff, E.; Chiang, D.; Hockenmaier, J.; and Tsujii, J., eds., Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, 2369–2380. Association for Computational Linguistics.

- Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,

- K. R.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. Yu, S.; Cheng, M.; Yang, J.; Ouyang, J.; Luo, Y.; Lei, C.;

- Liu, Q.; and Chen, E. 2025. Multi-Source Knowledge Pruning for Retrieval-Augmented Generation: A Benchmark and Empirical Study.

Zhang, J.; Yu, H.; and Xu, W. 2021. Hierarchical Reinforcement Learning by Discovering Intrinsic Options. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Zhao, Q.; Fu, H.; Sun, C.; and Konidaris, G. 2024a. EPO: Hierarchical LLM Agents with Environment Preference Optimization. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y., eds., Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024,

Miami, FL, USA, November 12-16, 2024, 6401–6415. Association for Computational Linguistics.

Zhao, Q.; Wang, R.; Wang, X.; Zha, D.; and Mu, N. 2024b. Towards Multi-Source Retrieval-Augmented Generation via Synergizing Reasoning and Preference-Driven Retrieval. CoRR, abs/2411.00689.

Zhao, X.; Liu, S.; Yang, S.; and Miao, C. 2025. MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot. In Long, G.; Blumestein, M.; Chang, Y.; Lewin-Eytan, L.; Huang, Z. H.; and Yom-Tov, E., eds., Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025, 4442–4457. ACM.

Zheng, Y.; Fu, D.; Hu, X.; Cai, X.; Ye, L.; Lu, P.; and Liu, P. 2025. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments. CoRR, abs/2504.03160.

Zhou, Y.; Liu, Y.; Li, X.; Jin, J.; Qian, H.; Liu, Z.; Li, C.; Dou, Z.; Ho, T.; and Yu, P. S. 2024a. Trustworthiness in Retrieval-Augmented Generation Systems: A Survey. CoRR, abs/2409.10102.

Zhou, Y.; Zanette, A.; Pan, J.; Levine, S.; and Kumar, A. 2024b. ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

### appendix

#### Prompts

We provide the prompts used in HierSearch’s training and inference processes. These include prompts for the flat-RLtrained deep search agent (Figure 4), local deep search agent (Figure 5), Web deep search agent (Figure 6), and planner agent (Figure 7). Some agents’ prompts refer to the prompt design of ReCall.2

#### Benchmarks

In Table 5, we provide the number of training samples (# Train Set), test samples (# Test Set), text chunks in the local retrieval library (# Text Chunks), as well as the number of entities (# Entities) and triples (# Triplets) in the local knowledge graph for the 6 datasets we used.

For local search, we prepare local knowledge bases separately for general, medical, and financial domains. For the general domain, we sample passages from the Wikipedia dump, and for the medical domain, we sample passages from the PubMed dump. 3

#### Implementation Details

We collect 19,938 training samples from Musique, 4,769 samples from OmniEval, and 35,601 samples from BioASQ for agent training. We train the local deep search agent, Web deep search agent, and the planner agent for 300 steps with a batch size of 64. We use the GRPO algorithm implemented in the VERL framework 4. We set the learning rate to 1e-6, batch size to 64, and train on an 8-GPU H20 machine. We choose Qwen2.5-Instruct as the backbone model for both search agents and the planner agent, which is the backbone model for RL-related baselines (R1-Searcher, ReCall, and DeepResearcher).

#### Case Study

We further analyze deep search performances of our method and the agent trained by flat RL in multi-knowledge-source environments through case studies. Figure 8, Figure 9, and Figure 10 show the thinking trajectories of agents used in HierSearch. Figure 11 shows the trajectory of the flatRL-trained agent. The question-answer pair we selected is from MuSiQue. Local search tools can only access part of the knowledge, while web search tools can supplement the knowledge. The flat-RL-trained agent is not proficient in using web search tools, leading to incorrect answers. In our method, the local agent found some helpful evidence, and the web agent successfully supplemented the missing knowledge. Finally, the planner reaches the correct answer.

2https://github.com/Agent-RL/ReCall 3Wikipedia dump: https://huggingface.co/datasets/RUC-

NLPIR/FlashRAG datasets/tree/main/retrieval-corpus. PubMed dump: https://huggingface.co/datasets/MedRAG/pubmed.

4VERL: https://github.com/volcengine/verl.

QA Dataset Language Domain # Train Set # Test Set # Text Chunks # Entities # Triplets Musique English General 19.938 400 135,808 968,274 1,516,001 Omnieval Chinese Financial 4,769 375 165,661 1,375,359 1,858,853 BioASQ English Medical 35,601 340 69,292 236,697 418,836 NQ English General - 400 161,022 1,121,290 1,791,408 HotPotQA English General - 400 158,105 1,101,900 1,774,500 PubmedQA English Medical - 400 72,285 256,281 444,928

Table 5: Statistics of all benchmarks.

You are a helpful assistant that can solve the given question step by step with the help of search tools. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke search tools to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. You have access to the following tools:

- 1. <chunk search>: A dense passage search tool that can be used to search for key passages about specific topics from local corpus.

- 2. <graph search>: A graph search tool that can be used to search for fact triplets about specific topics.

- 3. <get adjacent passages>: A tool that can be used to get adjacent passages from an entity in the graph.

- 4. <web search>: A web search tool that can be used to search for fact information about specific topics from the Internet. ’/. 5. <browse url>: A tool that can be used to browse a single webpage. You should provide a URL and a question to the tool, and separate them with ”—”. The search query for each tool is enclosed within <chunk search> </chunk search>, <graph search> </graph search>, <get adjacent passages> </get adjacent passages>, <web search> </web search>, and <browse url> </browse url> tags respectively. You should invoke local search tools (<chunk search>, <graph search>, <get adjacent passages>) first, and then invoke web search tools (<web search>, <browse url>) if the local search results are not sufficient. For example, <think> This is the reasoning process. </think> <graph search> search query here </graph search> <result> search result here </result> <think> This is the reasoning process. </think> <chunk search> search query here </chunk search> <result> search result here </result> <think> This is the reasoning process. </think> <get adjacent passages> search query here </get adjacent passages> <result> search result here </result> <think> This is the reasoning process. </think> <web search> search query here </web search> <result> search result here </result> <think> This is the reasoning process. </think> <browse url> URL — question here </browse url> <result> search result here </result> <think> This is the reasoning process. </think> <answer> The final answer </answer>. Please ensure that all reasoning processes and final answers are enclosed within the correct tags.

- Figure 4: The prompt for the flat deep search agent.

You are a helpful assistant that can solve the given question step by step with the help of search tools. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke search tools to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. You have access to the following tools:

- 1. <chunk search>: A dense passage search tool that can be used to search for key passages about specific topics.

- 2. <graph search>: A graph search tool that can be used to search for fact triplets about specific topics.

- 3. <get adjacent passages>: A tool that can be used to get adjacent passages from an entity in the graph. The search query for each tool is enclosed within <chunk search> </chunk search>, <graph search> </graph search>, and <get adjacent passages> </get adjacent passages> tags respectively. For example, <think> This is the reasoning process. </think> <all search agent> original question here </all search agent> <result> local search result and web search result here </result> <think> This is the reasoning process after revising the question. </think> <local search agent> revised question here </local search agent> <result> local search result here </result> <think> This is the reasoning process after revising the question. </think> <web search agent> revised question here </web search agent> <result> web search result here </result> <think> This is the reasoning process after revising the question. </think> <answer> answer here </answer>. Make sure to provide a clear and concise final answer based on the most reliable and relevant information.

- Figure 5: The prompt for the local deep search agent.

You are a helpful assistant that can solve the given question step by step with the help of web search tools. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke web search tools to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively. You have access to the following tools:

- 1. <web search>: A web search tool that can be used to search for fact information about specific topics from the Internet.

- 2. <browse url>: A tool that can be used to browse a single webpage. You should provide a URL and a question to the tool, and separate them with ”—”. The search query for each tool is enclosed within <web search> </web search> and <browse url> </browse url> tags respectively. For example, <think> This is the reasoning process. </think> <web search> search query here </web search> <result> search result here </result>. <think> This is the reasoning process. </think> <browse url> URL — question here </browse url> <result> search result here </result> <think> This is the reasoning process. </think> <answer> The final answer </answer>. Please ensure that all reasoning processes and final answers are enclosed within the correct tags.

- Figure 6: The prompt for the Web deep search agent.

You are a helpful assistant that can solve the given question step by step with the help of search tools. You have access to the following search agents:

- 1. <local search agent>: A local search agent that can search for fact information from local corpus and answer questions.

- 2. <web search agent>: A web search agent that can search for fact information from the Internet and answer questions.

- 3. <all search agent>: An agent that invokes both local search agent and web search agent. Both agents will return evidences, hypotheses, and conclusions. Keep in mind that evidences are reliable but sometimes irrelevant, while hypotheses and conclusions are not always correct and may contain hallucinations. Given a question, you are first provided with the evidences, hypotheses, and conclusions from both agents. The information returned are enclosed within <result> </result> tags. In case of conflicting information from the two agents, the local search agent’s information is considered more reliable. You need to carefully scrutinize the evidences, hypotheses, and conclusions from both agents. If you find that the information from neither agent is sufficient to answer the question, you can revise the question and invoke the search agents again. You can invoke the agents using <all search agent> question </all search agent> for both agents, <local search agent> question </local search agent> for local search agent, <web search agent> question </web search agent> for web search agent, and then provide a final answer based on the most reliable and relevant information. After you have gathered enough information, you need to provide a final answer. The final answer is enclosed within <answer> </answer> tags, and enclose your reasoning process within <think> </think> tags. For example, <think> This is the reasoning process. </think> <all search agent> original question here </all search agent> <result> local search result and web search result here </result> <think> This is the reasoning process after revising the question. </think> <local search agent> revised question here </local search agent> <result> local search result here </result> <think> This is the reasoning process after revising the question. </think> <web search agent> revised question here </web search agent> <result> web search result here </result> <think> This is the reasoning process after revising the question. </think> <answer> answer here </answer>. Make sure to provide a clear and concise final answer based on the most reliable and relevant information.

Figure 7: The prompt for the planner agent.

Question: Who is the sibling of the author of Kapalkundala? Golden Answer: Sanjib Chandra Chattopadhyay <all search agent> Who is the sibling of the author of Kapalkundala </all search agent> <result> Local Chunk Corpus: Kapalkundala Kapalkundala (), also known as Mrinmoyee, is a Bengali romance novel by Indian writer Bankim Chandra Chattopadhyay. Published in 1866, it is a story of a forest-dwelling girl named Kapalkundala ... Web Page: Kapalkundala - Wikipedia — https://en.wikipedia.org/wiki/Kapalkundala ... Works of Bankim Chandra Chattopadhyay Kapalkundala (1869) ... </result> <think> The author of Kapalkundala is Bankim Chandra Chattopadhyay. His sibling is the answer. </think> <local search agent> Who is the sibling of Bankim Chandra Chattopadhyay? </local search agent> <result> Local Chunk Corpus: Sanjib Chandra Chattopadhyay Sanjib Chandra was born to an orthodox Brahmin family at Kanthalpara, North 24 Parganas ... </result> <think> The author of Kapalkundala is Bankim Chandra Chattopadhyay. His sibling is the answer. </think> <web search agent> Who is the sibling of Bankim Chandra Chattopadhyay? </web search agent> <result> Web Page: One of his brothers, was also a novelist and is known for his book ”Palamau”. Sanjib Chandra Chattopadhyay Bankim Chandra and his elder brother both went to (then Governmental Zilla School), where he wrote his first poem ... </result> <think> The sibling of Bankim Chandra Chattopadhyay is Sanjib Chandra Chattopadhyay. </think> <answer> Sanjib Chandra Chattopadhyay. </answer>

Figure 8: Example trajectory for the planner agent.

Question: Who is the sibling of the author of Kapalkundala? Golden Answer: Sanjib Chandra Chattopadhyay <think> First, I need to search for the author of Kapalkundala. Then, I will find the sibling of the author.</think> <chunk search> Kapalkundala author </chunk search> <result> ... </result> <think> The author of Kapalkundala is Bankim Chandra Chattopadhyay. Now, I will search for the sibling of Bankim Chandra Chattopadhyay using the graph search tool.</think> <graph search> Bankim Chandra Chattopadhyay sibling </graph search> <result> ... </result> <think> The graph search tool does not provide information about the sibling of Bankim Chandra Chattopadhyay. I will try using the chunk search tool to find more information about his siblings.</think> <chunk search> sibling of Bankim Chandra Chattopadhyay </chunk search> <result> ... </result> <think> The chunk search tool provides information about the sibling of Bankim Chandra Chattopadhyay. From the search result, it is clear that the sibling of Bankim Chandra Chattopadhyay is Latchmiudayi.</think> <answer> Latchmiudayi </answer>

- Figure 9: Example trajectory for the local deep search agent.

Question: Who is the sibling of the author of Kapalkundala? Golden Answer: Sanjib Chandra Chattopadhyay <think> To find the answer to this question, I will need to find the author of the book Kapalkundala and then find information about their siblings. </think> <web search> the author of the book Kapalkundala and their siblings</web search> <result> ... </result> <think> The author of the book Kapalkundala is Bankim Chandra Chattopadhyay. Now I need to find information about his siblings. </think> <browse url> https://en.wikipedia.org/wiki/Bankim Chandra Chattopadhyay | siblings of Bankim Chandra Chattopadhyay </browse url> <result> ... </result> <think> The author of the book Kapalkundala is not found in the search results. Let me try to find a more specific answer to the question. </think> <browse url> https://en.wikipedia.org/wiki/Kapalkundala | the author of the book Kapalkundala </browse url> <result> ... </result> <think> The author of the book Kapalkundala is Bankim Chandra Chattopadhyay. Now I need to find information about his siblings to answer the question. </think> <browse url> https://en.wikipedia.org/wiki/Bankim Chandra Chattopadhyay | siblings of Bankim Chandra Chattopadhyay </browse url> <result> ... </result> <think> The author of the book Kapalkundala is Bankim Chandra Chattopadhyay. To find the answer to the question, I need to find specific information about his siblings. Let me look for specific information about Bankim Chandra Chattopadhyay’s siblings. </think> <browse url> https://en.wikipedia.org/wiki/Bankim Chandra Chattopadhyay | specific information about the siblings of Bankim Chandra Chattopadhyay </browse url> <result> ... </result> <think> The author of the book Kapalkundala is Bankim Chandra Chattopadhyay. His siblings are Sanjib Chandra Chattopadhyay and Sarat Chandra Chattopadhyay. Therefore, the answer is Sanjib Chandra Chattopadhyay. </think> <answer> Sanjib Chandra Chattopadhyay </answer>

- Figure 10: Example trajectory for the Web deep search agent.

Question: Who is the sibling of the author of Kapalkundala? Golden Answer: Sanjib Chandra Chattopadhyay <think> To answer this question, I need to find the author of Kapalkundala and their sibling. I will use the graph search and chunk search to find the author of Kapalkundala. Then, I will use the chunk search and get adjacent passages to find the sibling of the author. </think> <graph search> author of Kapalkundala </graph search> <result> ... </result> <think> The author of Kapalkundala is Bankim Chandra Chattopadhyay. Now, I will use the chunk search to find the sibling of Bankim Chandra Chattopadhyay. </think> <chunk search> sibling of Bankim Chandra Chattopadhyay </chunk search> <result> ... </result> <think> The chunk search did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the get adjacent passages to find the sibling of Bankim Chandra Chattopadhyay. </think> <get adjacent passages> Bankim Chandra Chattopadhyay </get adjacent passages> <result>

</result> <think> The get adjacent passages did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the chunk search again to find the sibling of Bankim Chandra Chattopadhyay. </think> <chunk search> sibling of Bankim Chandra Chattopadhyay </chunk search>

<result> ... </result> <think> The chunk search did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the web search to find the sibling of Bankim Chandra Chattopadhyay. </think> <web search> sibling of Bankim Chandra Chattopadhyay </web search> <result> ... </result> <think> The web search did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the browse url to find the sibling of Bankim Chandra Chattopadhyay. </think> <browse url> https://www.britannica.com | BankimChandra-Chattopadhyay - Encyclopædia Britannica </browse url> <result> ... </result> <think> The browse url did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the graph search again to find the sibling of Bankim Chandra Chattopadhyay. </think> <graph search> sibling of Bankim Chandra Chattopadhyay </graph search> <result> ... </result> <think> The graph search did not provide any information about the sibling of Bankim Chandra Chattopadhyay. I will use the chunk search again to find the sibling of Bankim Chandra Chattopadhyay. </think> <chunk search> sibling of Bankim Chandra Chattopadhyay </chunk search> <result> ... </result> <think> The chunk search provided information about the sibling of Bankim Chandra Chattopadhyay. The sibling of Bankim Chandra Chattopadhyay is Kanaka Durga. </think> <get adjacent passages> Kanaka Durga </get adjacent passages> <result> </result> <think> The get adjacent passages confirmed that the sibling of Bankim Chandra Chattopadhyay is Kanaka Durga. </think> <answer> Kanaka Durga </answer>

Figure 11: Example trajectory for the flat-RL-trained deep search agent.

