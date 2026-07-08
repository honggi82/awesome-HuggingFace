# arXiv:2604.14518v2[cs.AI]17Apr2026

Li Auto

[Figure 1]

## Mind DeepResearch Technical Report

##### MindDR Team, Li Auto Inc

We present Mind DeepResearch (MindDR), an efficient multi-agent deep research framework that achieves leading performance with only ~30B-parameter models through a meticulously designed data synthesis and multi-stage training pipeline. The core innovation of MindDR lies in a collaborative three-agent architecture (Planning Agent, DeepSearch Agent, and Report Agent) and a four-stage agent-specialized training pipeline comprising SFT cold-start, Search-RL, Report-RL and preference alignment. With this regime, MindDR demonstrates competitive performance even with ~30B-scale models. Specifically, MindDR achieves 45.7% on BrowseComp-ZH, 42.8% on BrowseComp, 46.5% on WideSearch, 75.0% on xbench-DS, and 52.5 on DeepResearch Bench, outperforming comparable-scale open-source agent systems and rivaling larger-scale models. MindDR has been deployed as an online product in Li Auto. Furthermore, we introduce MindDR Bench, a curated benchmark of 500 real-world Chinese queries from our internal product user interactions, evaluated through a comprehensive multidimensional rubric system rather than relying on a single RACE metric. On MindDR Bench, MindDR achieves a state-of-the-art score of 51.8.

Last Update Date: April 14, 2026 Correspondence: yangsheng3@lixiang.com

### 1 Introduction

The rapid advancement of large language models (LLMs) has fundamentally transformed human workflows and daily life, substantially boosting productivity across a wide spectrum of tasks [5, 22]. The field has undergone a clear paradigm shift: LLMs have evolved from simple conversational chatbots to reasoners capable of complex multi-step logical inference [37], and further to autonomous agents that can plan, reason, and interact with external tools [48, 25, 23]. Among the emerging applications of LLM agents, deep research agents have become one of the most representative and capable product paradigms [10, 22].

Compared to traditional single-turn QA or retrieval-augmented generation (RAG) pipelines relying on one-shot retrieval, deep research agents support a substantially richer capability set, encompassing open-domain information retrieval, multi-source evidence verification and synthesis, long-horizon reasoning [47], external tool use and structured report generation. The paradigm gained widespread attention when Google and OpenAI introduced their powerful deep research agents [10, 22], exhibiting human-level performance in domains such as scientific research and financial analysis, catalyzing rapid adoption of the deep research paradigm across academia and industry. Following these pioneering closed-source systems, significant open-source efforts have emerged to democratize deep research capabilities, including Tongyi DeepResearch [34, 16, 40, 30], MiroThinker [33] and Step-DeepResearch [13].

While the open-source ecosystem has made notable progress, deep research agents continue to face a fundamental bottleneck: the prohibitive cost of training and inference, which severely hinders

Preprint.

###### DeepResearch-Bench

###### MindDR-Bench

###### MindDR-Bench-Cite

84

54

52.5

51.8

82.1 81.9

52.2

52

82

52

80.2

80

49.7

50

49.6

50

77.2

78

76.4

48

48.3

46.5

76

RACE

75.0

score

Cite

48

46

44.6

74

44.3

43.8

46.2

44

72

46

45.2 45.1

42

70

40.5

44.3

68.4

68

40

44

MindDR1.5Gemini3.1ProGemini2.5Pro OpenAI Kimi Doubao MindDR1.0 Tongyi

MindDR1.5 Gemini3.1Pro Gemini2.5Pro Doubao Kimi Tongyi MindDR1.0

MindDR1.0 Gemini2.5Pro MindDR1.5 Gemini3.1Pro Tongyi Kimi Doubao

###### BrowseComp-ZH

###### BrowseComp

###### xbench-DS

WideSearch

48

|46.5<br><br>44.3<br><br>44.0<br><br>43.1<br><br>41.7|41.3<br><br>40.3|
|---|---|
|MindDR-v1.5-30B-A3BDeepSeekR1 Tongyi-DR-30B-A3BGLM-4.6MindDR-v1.W<br><br>researc denotes th<br><br>the mode Bench<br><br>massive extensiv|0-32BebShaper-32BMiroThinker-v1.5-30B-A3BWebSailor-32BDeepMiner-32B<br><br>39.7<br><br>37.9<br><br>35.2<br><br>h products e previousl presented<br><br>leaderboard:<br><br>foundation e continual|

80

45.7

49.5

75.0

45.1

73.0

50

45

69.0

46

43.2

70

42.8

64.0

40.7

44

60

40

53.0

40

50.0

33.5

46.0

Average@4

50

31.8

42

30.4

35.6

Pass@1

Pass@1

score

34.6

30

35

40

40

31.9

30

18.6

38

20

30

28.4 28.0

14.8 14.1

20

13.3

11.0

25.6 25.3

36

10.5

10

5.0

10

25

34

0

MindDR-v1.5-30B-A3BTongyi-DR-30B-A3BGLM-4.6MindDR-v1.5-32BMiroThinker-v1.5-30B-A3BDeepSeekR1MindDR-v1.0-32BDeepMiner-32BWebShaper-32BWebSailor-32B

MindDR-v1.5-30B-A3BGLM-4.6Tongyi-DR-30B-A3BDeepMiner-32BMindDR-v1.5-32BMiroThinker-v1.5-30B-A3BMindDR-v1.0-32BWebShaper-32BDeepSeekR1WebSailor-32B

MindDR-v1.5-30B-A3BTongyi-DR-30B-A3BGLM-4.6MindDR-v1.5-32BDeepMiner-32BDeepSeekR1WebShaper-32BMindDR-v1.0-32BMiroThinker-v1.5-30B-A3BWebSailor-32B

MindDR-v1.5-32B

- Figure 1: Benchmark Performance of MindDR, comparing with mainstream deep ts and state-of-the-art models at comparable parameter scale. MindDR 1.0 de sgeneration model trained with large-scale RFT, while MindDR 1.5 denotes t d in this paper. MindDR 1.5 has been listed on the official DeepResearch d: https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard

practical user experience. State-of-the-art deep research systems often rely on n models (e.g., >100B parameters) and expensive training paradigms, such as l pre-training (mid-training), to inject domain knowledge and reasoning capabilities. Furthermore, at inference time, complex research tasks require long-horizon reasoning and multi-step tool use [47]. Without explicit optimization for search efficiency, models tend to exhaust substantial computational budgets on marginally relevant exploration with limited information gain [43]. This inefficiency drastically increases token consumption and system latency, and also risks diluting key findings through excessive context accumulation [19], ultimately degrading the user experience.

Therefore, the core challenge in deep research is: how to achieve leading performance and excellent user experience using a small-sized model through low-cost training and inference?

To address this challenge, we present MindDR, a cost-effective framework that achieves state-ofthe-art deep research capabilities using only ~30B-parameter models. MindDR tackles the costperformance trade-off by decomposing the complex research problem into specialized subtasks at the inference stage and applying a highly targeted, multi-stage training pipeline at the training stage.

Inference-stage decomposition. At inference time, MindDR employs a collaborative three-agent architecture—a Planning Agent, a DeepSearch Agent, and a Report Agent. This decomposition allows parallel execution of search tasks and context isolation, inherently improving inference efficiency and alleviating the burden of ultra-long context processing on a single model. The DeepSearch Agent efficiently navigates multi-step retrieval scenarios, while the Report Agent focuses on resolving information conflicts [18] and generating human-aligned content.

Training-stage targeted optimization. At the training stage, rather than relying on computationally expensive mid-training or monolithic end-to-end reinforcement learning (RL), we design a four-phase training pipeline: (i) supervised fine-tuning (SFT) for behavioral cold-start, establishing foundational instruction-following and tool-use capabilities while maintaining low data scale; (ii) Search-RL, which explicitly optimizes the DeepSearch Agent’s long-horizon reasoning and search efficiency through step-level credit assignment, significantly reducing redundant token consumption during inference;

- (iii) Report-RL, which employs RACE Rubrics and format-based reward shaping to specialize the Report Agent in resolving information conflicts and generating high-quality long-form content; and
- (iv) preference alignment, which further calibrates output reports to human expectations via human feedback signals.

We evaluate the performance of MindDR on deep search and deep research benchmarks such as BrowseComp(-ZH), GAIA, xbench-DS and DeepResearch Bench to verify its effectiveness.

Furthermore, we also establish MindDR Bench, a curated benchmark comprising 500 deep research queries extracted from real-world user interactions with our AI assistant. Our main contributions are summarized as follows:

- 1. Task-driven multi-stage training pipeline for heterogeneous agents. We design a fourstage training pipeline tailored to distinct agent requirements: SFT for behavioral cold-start and Search-RL for long-horizon reasoning and search efficiency in DeepSearch Agent, Report-RL for information conflict resolution and report quality, and preference alignment for human experience optimization in Report Agent. The Search-RL stage achieves prominent accuracy improvement on BrowseComp-ZH while reducing context and tool-call consumption compared to SFT baseline model and the subsequent Report-RL stage also improves report RACE scores.
- 2. Introduction of MindDR Bench and a Comprehensive Evaluation System. We introduce MindDR Bench, a rigorously curated benchmark comprising 500 Chinese deep research queries extracted from real-world user interactions with Livis (Li Auto’s intelligent assistant). Rather than relying on a single, abstract RACE metric for content evaluation, we propose a comprehensive evaluation system tracking the search process, assessing both the content quality and the presentation format of generated reports. MindDR Bench explicitly aligns the evaluation metrics with practical user experience, providing diverse insights for deep research community.

The remainder of this paper is organized as follows. Section 2 reviews related works on deep research agents, search reinforcement learning, and report generation reinforcement learning. Section 3 presents the MindDR multi-agent framework. Section 4 describes the data synthesis pipeline. Section 5 elaborates on the multi-stage training pipeline. Section 6 shows main experimental results across benchmarks with comparisons with representative open-source and closed-source systems.

- Section 7 concludes the paper with a discussion of current limitations and future research directions.
- Section 8 appends the prompts and examples.

### 2 Related Works

##### 2.1 Deep Research Agents

Knowledge-intensive tasks have evolved from single-turn Retrieval-Augmented Generation (RAG) [28] toward autonomous agents capable of iterative tool use and long-horizon reasoning [48, 47]. Proprietary systems like Gemini Deep Research [10] and OpenAI Deep Research [22] demonstrated near-human performance on complex investigative tasks spanning scientific research and financial analysis, yet their closed-source nature limits reproducibility and systematic analysis [17]. This opacity has spurred open-source alternatives. Tongyi DeepResearch [34] proposed an end-to-end optimization architecture for agentic training. MiroThinker [33, 32] advanced “interactive scaling” by training models to handle hundreds of tool calls via reinforcement learning. WebSailor [16, 15] focused on uncertainty-reducing web navigation, while WebWeaver [18] explored dual-agent architectures for open-ended report generation. Nanbeige4.1-3B [45] further showed that small models can be competitive through specialized training. Existing systems predominantly optimize for retrieval accuracy under a monolithic end-to-end RL objective, which is difficult to train due to the high complexity of deep research tasks.

##### 2.2 Search Reinforcement Learning

Search-RL optimizes retrieval decision-making and multi-step reasoning within deep research agents, directly targeting the challenge of long-horizon reasoning in complex information-seeking scenarios. Research in this area advances along two principal directions.

Data Synthesis. Bootstrapping effective policies requires high-quality retrieval-reasoning trajectories. Knowledge-graph-driven methods (e.g., WebSailor [16], DeepDive [20]) produce logically consistent data but are constrained by graph coverage and struggle in dynamic open-domain scenarios. Agent-simulation-based approaches (e.g., MiroThinker [33], Cognitive Kernel-Pro [9]) improve task alignment and realism, yet incur prohibitive computational costs and lack fine-grained modeling of

per-step retrieval contributions, making it difficult to provide differentiated training signals for critical versus peripheral search steps.

Credit Assignment. Beyond data quality, training efficiency hinges on precise credit assignment. Trajectory-level RL methods [36, 14, 6] broadcast a uniform reward across all steps, providing no targeted signal for critical retrievals and thus failing to explicitly optimize search efficiency. Steplevel approaches offer stronger supervision: PPO-based methods [53] introduce per-step rewards but depend on costly critic models; branch-sampling methods (e.g., ARPO [6], TreeRL [12]) reduce overhead yet remain infeasible for full-coverage, long-horizon tasks. MindDR addresses this gap with a lightweight step-level credit assignment mechanism that achieves fine-grained advantage estimation without critic overhead or exponential sampling complexity, enabling explicit optimization for both retrieval accuracy and search efficiency.

##### 2.3 Report Reinforcement Learning

Report-RL trains models to generate well-structured, factually consistent long-form reports from retrieved evidence, directly tackling the challenges of information conflict resolution and humanaligned report generation. Progress has been driven by two complementary efforts.

Alignment and Generation Frameworks. Industrial systems apply multi-stage pipelines with structured rewards: Step-DeepResearch [13] employs checklist-based rubrics, and Tongyi DeepResearch [34] develops end-to-end RL for long-text outputs. Extensive research augment this direction with reflection-guided writing (SuperWriter [41]), recursive revision (Re3 [46]), and backward-inference trajectory construction (REER [35]), collectively strengthening global planning and iterative self-correction. However, these approaches generally lack explicit mechanisms for adjudicating conflicting evidence across heterogeneous sources—a critical requirement in realistic research scenarios.

Evaluation and Reward Design. Classic evaluation metrics for deep research agents are the RACE rubric score comprising Comprehensiveness, Insight, Instruction Following and Readability as shown in DeepResearch Bench [7]. Several other multi-dimensional rubric frameworks are also proposed including WritingBench [42], ResearchRubrics [27], and DEER [11], that enable reliable LLM-asa-Judge evaluation along axes such as factual accuracy, structural coherence, and theme alignment. These rubrics in turn supply reward signals for RL algorithms including GRPO [26], GSPO [52], and DAPO [49]. Nevertheless, as indicated by FINDER [51], current models still exhibit critical gaps in global logical structure and factual fidelity under ultra-long contexts, and few works incorporate human preference feedback to align outputs with real-world user expectations. MindDR targets these limitations through RACE Rubrics-based reward shaping combined with a dedicated preference alignment stage, directly optimizing report generation for both information quality and user reading experience.

### 3 MindDR Framework

In this section, we present the overall architecture of MindDR. As illustrated in Fig. 2, MindDR consists of two tightly coupled components: an inference pipeline that orchestrates multi-agent collaboration for deep research report generation (Section 3.1), and a four-phase training pipeline that progressively builds the underlying model capabilities required by each agent (Section 3.2). In the inference end, given a natural-language research query, the inference pipeline decomposes the problem into manageable subtasks, retrieves and synthesizes evidence from heterogeneous sources, and assembles a polished, structured report. In the training end, a four-stage pipeline equips the agents with the necessary capabilities through supervised fine-tuning, search-oriented reinforcement learning, and report-oriented reinforcement learning.

##### 3.1 Inference Pipeline

The inference pipeline comprises three functional agents—the Planning Agent, the DeepSearch Agent, and the Report Agent—coordinated through a shared memory substrate and a novel reasoning mechanism termed the Extended Chain-of-Thought (XoT). We describe each component below.

[Figure 2]

[Figure 3]

[Figure 4]

###### Query 基于汽⻋⾏业内卷，如何让企业⾼质竞争的开展价格战？

[Figure 5]

[Figure 6]

[Figure 7]

Planning Agent

[Figure 8]

Task #1 Task #k

...

[Figure 9]

Search Agent Swarm

[Figure 10]

[Figure 11]

[Figure 12]

DeepSearch Agent #1

DeepSearch Agent #k

...

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Synthesis

[Figure 17]

SubReport #1 SubReport #k

Report Agent

- Figure 2: Overview of the MindDR multi-agent framework. A user query is first processed by the Planning Agent, which performs intent analysis and task decomposition to produce a structured subtask specification. Each subtask is dispatched to an independent DeepSearch Agent instance that executes a ReAct-style loop while maintaining an Extended Chain-of-Thought (XoT) reasoning trace. The resulting sub-reports are aggregated by the Report Agent, which synthesizes a coherent, citation-grounded research report.

Planning Agent When MindDR receives a user query, the Planning Agent initiates the research pipeline by analyzing user intent and decomposing the query into a set of subtasks. Then, these subtasks will be imported into the following DeepSearch Agent for parallel deep searching.

DeepSearch Agent Each subtask produced by the Planning Agent is dispatched to an independent DeepSearch Agent instance, enabling parallel execution across all subtasks. Each DeepSearch Agent implements a ReAct-style agent loop [48], iteratively invoking search tools to perform multi-source retrieval, evidence integration, and intermediate reasoning until the agent determines that sufficient information has been gathered to address the assigned sub-problem.

Report Agent The Report Agent serves as the final synthesis stage of the inference pipeline. It receives the complete task specification and all sub-reports from the DeepSearch Agents. Based on these inputs, the Report Agent first generates a hierarchical outline, then performs global information aggregation and structural organization to produce a coherent, comprehensive, and well-structured research report in Markdown format. The Report Agent is designed to excel in several dimensions aligned with the RACE evaluation framework [7] and realistic user experience.

Memory To enable effective coordination across the multi-agent pipeline, we introduce the memory mechanism including Extended Chain-of-Thought (XoT) memory and tool memory. Unlike standard chain-of-thought prompting [37], which operates within a single model invocation, XoT memory extends the reasoning trace across multiple agent interactions and tool calls. DeepSearch and Report Agent maintain and append to a shared reasoning context that captures not only the current agent’s thought process but also the inter-connection information between agents. Furthermore, a too-call memory module is also constructed to maintain the interactions with external environment. This shared memory enables downstream agents (e.g., the Report Agent) to access the full provenance of retrieved information, facilitating more faithful and well-grounded report generation.

##### 3.2 Training Pipeline Overview

The multi-agent MindDR system requires diverse capabilities spanning tool use, multi-step reasoning, long-form generation, and subjective quality alignment—objectives that differ substantially in their reward structure, optimization landscape, and data requirements. Rather than optimizing all objectives end-to-end, we decompose the training into a four-phase curriculum (Fig. 3), where each phase targets a well-defined capability frontier with tailored optimization algorithms and reward signals. This decomposition is guided by three principles:

###### Evaluation

###### Data Dynamic Synthesis

Format correctness

DR Benchmarks

UX-Eval

DS Benchmarks

[Figure 18]

MindDR Module Evaluation system

BrowseComp …… WideSearch

DR Bench MindDR-Bench

Special-token

Reasoning

Complex QA Synthesis

Training Stages

User Query Mixing

###### Pre. Alignment

Report-RL

Search-RL

###### SFT Cold-start

Long-Horizon Reasoning & Action Decision-Making Opt.

Teacher Model Sampling

Tool-Call & Reasoning Format Opt.

Report Quality Opt.

UX Issue Opt.

PRM + ORM & Tool + Format Reward

RACE Rubrics & Auxiliary Rule Reward

LLM-Judge & Rule Scoring

Format Correctness & Entropy Monitoring

Quality Filter

Sample Difficulty Distribution Adjustment

On-Policy SelfSampled Data

Adjust distribution based on evaluation

Frontier LLM Reference Generation

Seq-Len Distribution Scheduling

GSPO with Dynamic Coefficient Scheduling

SFT/RL Data Synthesis

DAPO/GSPO Loss

DPO & Self-SFT

Dynamic Early Stopping

AI-Infra

Training and Inference Framework Tool Environment

Li-veRL SGLang vLLM Code Interpreter Sandbox Web Browser Scholar Search Internal Database

Unified Interface Search APIs

Figure 3: Four-stage training pipeline of MindDR.

- • Reward tractability. End-to-end optimization over the full DR pipeline would require a single reward to capture tool correctness, reasoning quality, report coherence, and subjective preferences simultaneously. Such a composite reward is inevitably sparse and noisy, making credit assignment across dozens of reasoning steps intractable. Staged training decomposes this into dense, welldefined signals at each phase.
- • Capability dependency. Later capabilities critically depend on earlier ones: RL exploration requires stable format adherence from SFT; report generation quality is bottlenecked by retrieval completeness from Search-RL; and preference alignment presupposes functionally correct outputs from prior phases. The ordering reflects this dependency chain.
- • Data efficiency. Each phase operates on data specifically curated for its target capability, avoiding the need for expensive end-to-end trajectory annotation and enabling independent iteration on data quality per phase.

- Phase 1: Supervised Fine-Tuning. SFT provides a behavioral cold-start, establishing foundational capabilities in tool invocation, ReAct-format adherence, and multi-turn reasoning patterns through behavior cloning on expert trajectories. The training extent is carefully calibrated: sufficient to ensure stable format correctness under long contexts, yet restrained to preserve policy entropy for subsequent RL exploration (Section 5.1).
- Phase 2: Search-RL. This phase optimizes the DeepSearch Agent’s long-horizon reasoning and action decision-making ability via online RL with real tool execution. A unified GRPO/GSPO framework with dynamically scheduled rewards—progressing from tool-call correctness to processlevel entity coverage to outcome-level answer accuracy—enables progressive capability acquisition without hard stage boundaries (Section 5.2).
- Phase 3: Report-RL. Report-RL targets the Report Agent’s long-form generation quality. Using RACE Rubrics [7] evaluated by LLM-as-Judge, the model is optimized along comprehensiveness, readability, insight, and instruction-following dimensions, supplemented by rule-based citation and format rewards for efficiently addressable quality issues (Section 5.3).
- Phase 4: Preference Alignment. In the generated long-form report, there exists user experience (UX) issues such as temporal correctness, table formate errors. In order to improve the report quality

LLM-powered Baidu Baike Q&A product

LLM

English Wikipedia corpus

Webpage Tree

Chinese corpus

knowledge graph construction

quality filtering pipeline

###### Sampled KG subgraph

###### generation pipeline

multi-hop QA generation

direct-answer filtering

High-quality DeepResearch QA pairs

Encyclopedia Knowledge Graph

reasoning plausibility filtering

condition obfuscation

answer entity

intermediate entity entity node

condition plausibility evaluation

QA semantic coherence filtering

sampled path

- Figure 4: Overview of the knowledge-graph-grounded query synthesis pipeline, consisting of four stages: graph construction and subgraph sampling, initial QA generation, text obfuscation and complexity enhancement, and reasoning validity filtering.

stably without catastrophic forgetting, MindDR adopts on-policy self-improved framework with DPO [24] and Self-SFT [1] to align the final report quality with human expectations.(Section 5.4).

### 4 Data Synthesis

##### 4.1 Query Synthesis

We propose an end-to-end framework for synthesizing multi-hop reasoning questions from structured knowledge graphs. The overall pipeline, illustrated in Fig. 4, comprises four stages: graph construction and subgraph sampling, initial QA generation, condition obfuscation and complexity enhancement, and reasoning validity filtering.

Graph Construction and Subgraph Sampling. We construct unified knowledge graphs from Baidu Baike and English Wikipedia corpora, organizing information through entity-attribute relationships and sample nodes and paths over web-page tree structures. Beyond connectivity, subgraph sampling enforces three key constraints: each hop in the reasoning path must correspond to retrievable or verifiable conditions in a real search environment (reasoning reachability); direct associations between intermediate nodes and answers are constrained to eliminate shortcuts, ensuring every hop contributes necessary information to the reasoning chain (path necessity); and reasoning paths in multi-branch subgraphs maintain semantic and structural independence, preventing implicit crossbranch associations from reducing combinatorial complexity (structural independence). Together, these constraints guaranty that subgraphs are inferable, non-simplifiable, and verifiable.

Multi-hop QA Generation. Given a subgraph structure, we prompt a state-of-the-art LLM to transform explicit structured relations into implicit natural-language QA pairs. Generation constraints ensure questions comprehensively cover key subgraph information while avoiding direct exposure of intermediate reasoning nodes, tightly binding the solution process to the graph structure and yielding high-quality multi-hop QA data for deep-search applications.

Condition Obfuscation and Plausibility Evaluation. To increase question difficulty while preserving solvability, we apply controlled obfuscation to the initial QA pairs: low-value attributes are combined with other conditions or replaced with more distinctive descriptions to ensure condition retrievability; overly explicit clues are weakened or rephrased to force models to synthesize multiple conditions, preventing any single strong constraint from dominating the solution; obfuscated expressions must remain semantically consistent with the original, preferring natural, human-cognitionaligned formulations; and condition combinations are continuously monitored for their effect on the solution space, with constraints supplemented or restructured as needed to maintain answer uniqueness.

Quality Filtering. Generated questions are passed through a multi-stage data quality filtering pipeline: a direct-answer test first removes questions solvable without multi-step reasoning; condition plausibility evaluation and reasoning plausibility filtering then eliminate samples with logical contradictions

or insufficient real-retrieval support; finally, QA semantic coherence filtering performs a holistic quality assessment combining automated scoring and human auditing, ensuring only high-quality samples enter the final training set.

##### 4.2 Query Source Diversification and Mixing

The query synthesis pipeline (Section 4.1) produces structured queries equipped with complete multi-hop reasoning chains, ground-truth answers, and intermediate reasoning steps. Grounded in high-quality encyclopedic corpora with well-defined inference paths, these queries are well-suited for constructing structured deep reports and serve as the primary training signal across all pipeline stages.

However, the sampling space of knowledge-graph queries is inherently constrained by corpus coverage: queries involving rare entities exhibit distributional shift relative to high-frequency user demands, and capabilities acquired from such data may not transfer reliably to real-world usage patterns. To mitigate this gap, we supplement synthesized queries with real user queries mined from online interaction logs, yielding a complementary set that faithfully reflects genuine user intent. Across all training stages, knowledge-graph-synthesized queries and real user queries are mixed at carefully calibrated proportions to balance controllability with ecological validity.

Beyond training, a representative subset of these real user queries is further curated into a dedicated evaluation benchmark—MindDR Bench—which we describe next.

##### 4.3 MindDR Bench

While the diversified query mixture described above strengthens training, a parallel challenge arises on the evaluation side. During the course of benchmark investigation, we identify two primary limitations in existing deep-research benchmarks that restrict their utility for guiding practical system development. First, there remains a notable scarcity of authentic Chinese queries; the majority of existing test cases are synthetic or derived from academic datasets, introducing distributional bias relative to actual user experience. Second, prevailing evaluation frameworks rely predominantly on macro-level metrics (e.g., a single aggregate RACE score). Given the inherently long execution horizons of deep research tasks, such coarse-grained metrics fail to provide actionable, fine-grained feedback to intermediate pipeline modules, thereby hindering targeted iteration and continuous improvement.

To bridge this gap, we introduce MindDR Bench, a rigorous benchmark explicitly designed to reflect genuine user intent and support comprehensive, actionable evaluation.

Query Curation. We construct MindDR Bench by mining 500 deep research queries directly from the online interaction logs of real users with Li Auto’s intelligent assistant. Quality and complexity are ensured through a two-stage hybrid filtering pipeline: a state-of-the-art LLM first prescreens candidates for the required reasoning depth, followed by expert annotation and review. The resulting queries span 16 distinct domains—including automotive, travel, technology, and finance—faithfully capturing authentic, high-complexity research demands grounded in automotive industry scenarios.

Comprehensive Evaluation System. To overcome the limitations of macro-level scoring, we propose a fine-grained, multi-dimensional MindDR Module Evaluation system built upon the foundational DeepResearch Bench framework [7]. Rather than relying solely on a holistic RACE score, our system systematically decomposes evaluation across four critical stages of the deep research pipeline, as detailed in Table 1.

By combining these modular process indicators with content-focused RACE metrics, our evaluation system comprehensively captures both the intermediate behaviors and the final presentation of the system. This granular feedback mechanism explicitly aligns model performance with practical user experience metrics and allows for targeted iteration of individual pipeline modules, significantly accelerating the iterative development of the MindDR pipeline.

##### 4.4 SFT Data Synthesis

SFT data endows the model with cold-start capabilities spanning tool invocation, structured formatting, and multi-turn reasoning. We build an end-to-end synthesis system covering multi-source sampling, multi-tier filtering, standardized post-processing, and automated configuration. All data follow the

Table 1: MindDR Module Evaluation system across four critical pipeline stages.

Evaluation Module Evaluation Metrics Evaluation Items

- 1. Reflection turns count
- 2. Search query repetition rate

Reasoning Trajectory Thinking Efficiency

- 1. Proportion of usage for each tool
- 2. Tool call failure rate

Tool Call Correctness of tool use

- 1. Outline title miss rate
- 2. Incorrect directory hierarchy count

Outline Generation Correctness of outline logic

- 1. Tense error rate
- 2. Valid format tables rate

Report Generation Correctness of content logic

ReAct paradigm, decomposing complete reasoning trajectories into independent steps organized as multi-turn dialogs; each step contains a thought Tt, an action At, and an observation Ot, enabling unified decision-process learning. The resulting trajectory corpus serves a dual purpose: one partition is used directly for supervised fine-tuning, while the other is repurposed as seed data for Report-RL training (Section 4.6).

Multi-Source Sampling and Scale. We construct approximately 12K high-quality trajectories from three complementary sources: (i) knowledge-graph trajectories (60%), derived from the multi-hop reasoning data described in Section 4.1 and spanning 1–5 hops at varying complexity; (ii) real-world scenario trajectories (35%), covering automotive, technology, transportation, and industrial domains to ensure ecological validity; and (iii) human-annotated high-difficulty edge cases (5%) for robustness. All trajectories are generated via multi-model parallel sampling in a simulation environment closely aligned with the online inference stack, ensuring sufficient data diversity.

Trajectory Diversity Design. Data distribution is controlled along three dimensions: task complexity (easy: 1–2 hops, 40%; medium: 3 hops, 35%; hard: 4–5 hops, 25%), trajectory length (5–30 steps, avoiding fixed-length decision biases), and tool invocation patterns (sequential retrieval, parallel verification, and hierarchical deepening).

Long-Context Data Augmentation. We adopt a progressive length-generalization strategy proceeding through a base phase (average 8K tokens), an extension phase (32K/64K, 30% of the mixture), and an extreme-length phase (128K, 15%). Positional encoding is fine-tuned by resampling trajectories that contain redundant observations such as history reviews and intermediate summaries. This strategy raises 128K-context format correctness from 72% to 94%.

Multi-Tier Quality Filtering and Configuration. We establish a three-tier filtering system combining rule engines and LLM-based evaluation. The pre-admission stage filters out simple queries, retaining only complex open-ended questions that require multi-hop reasoning. The process-validation stage performs real-time verification of reasoning logic, tool invocation accuracy, and content relevance. The final-approval stage conducts multi-dimensional quality assessment—covering format compliance, factuality, instruction following, and logical coherence—supplemented by human auditing. After uniform format validation, metadata completion, and multi-dimensional tagging via automated post-processing, a dynamic configuration strategy prioritizes complex multi-turn reasoning samples (60–70%) while maintaining balanced coverage of tool invocation and report generation capabilities.

##### 4.5 Search-RL Data Synthesis

Search-RL training data is built on the multi-hop reasoning queries generated in Section 4.1. The key design principle is to preserve the complete reasoning chain from question to answer during synthesis, providing the supervision signals required for step-level process reward modeling (PRM) and outcome reward modeling (ORM). We use approximately 35K synthesized queries as the training data foundation.

Entity Annotation and Reward Data Construction. During query generation, we extract and retain the intermediate entity nodes, relational transition paths, and reasoning dependency structure from the knowledge graph, forming the key entity set E = {e1,...,eM} for each query. This entity set is used directly for string-matching-based PRM verification during training (see Section 5.2), requiring no additional LLM inference and substantially reducing reward computation cost. The ground-truth answers required for ORM are recorded at query generation time, supporting final-output matching verification.

Difficulty Annotation and Distribution Control. Each query is annotated with a three-level difficulty label based on reasoning hop count (1–5 hops), intermediate entity category complexity, and retrieval difficulty (entity ambiguity, information sparsity). These labels support dynamic sampling during training, where the actual sampling proportions are adjusted adaptively according to ORM accuracy on the validation set (see Section 5.2); the annotation layer is responsible only for providing standardized difficulty metadata, without prescribing fixed curriculum learning ratios.

Data Format. Each training instance contains a query, the intermediate entity set E, the ground-truth answer, and a difficulty label. The format is consistent with the SFT data described in Section 4.4, using the ReAct multi-turn dialogue structure, and can be ingested directly by the RL training pipeline.

##### 4.6 Report-RL Data Synthesis

Report-RL training requires query–report pairs augmented with fine-grained scoring rubrics. Rather than constructing an independent data pipeline, we reuse the high-quality trajectory corpus produced during SFT data synthesis (Section 4.4) as the foundation, and derive two complementary data formats—long-form and short-form—to balance training fidelity with data efficiency.

Long-form Data Synthesis. Each long-form training instance comprises six components: a query, a system prompt, upstream deep-search retrieval data, an outline, RACE Rubrics, and a reference report. The query, system prompt, retrieval data, outline, and reference report are jointly drawn from the SFT trajectory corpus, preserving the original retrieval context and reasoning chain. The RACE Rubrics are synthesized separately: queries from the SFT set, together with the RACE Rubrics generation template from DeepResearch Bench [7], are fed into a strong LLM to produce query-specific scoring criteria tailored to each individual query. These rubrics subsequently serve as part of the reward model’s input prompt during training, enabling differentiated evaluation of reports generated for different queries. An illustrative example of the RACE Rubrics is provided in Appendix 8.1.

Short-form Data Synthesis. Long-form instances that carry full upstream retrieval results are expensive to sample and inevitably introduce retrieval noise. We therefore introduce a complementary short-form synthesis strategy that reduces data collection cost while substantially expanding the volume of usable training data. Each short-form instance consists of a query, a system prompt, RACE Rubrics, and a reference report—deliberately omitting the upstream retrieval content. Queries are again sourced from the SFT trajectory corpus, and the RACE Rubrics are directly reused from the long-form pipeline to maintain evaluation consistency. The reference reports, however, are newly synthesized: a strong LLM is prompted with both the query and the dimension-specific evaluation criteria from the Rubrics, guiding it to produce a high-quality report that explicitly addresses each RACE dimension. By decoupling report generation from the retrieval stage, this strategy yields cleaner supervision signals while preserving alignment with the rubric-based reward framework. The synthesis prompts are provided in Appendix 8.2.

The overall data synthesis process for both formats is illustrated in Fig. 6.

### 5 Training Pipeline

The DR system involves diverse task types and multi-level capability demands. We design a fourstage training pipeline: SFT cold-start training establishes instruction-following and basic tool invocation; Search-RL builds long-horizon reasoning and complex search capabilities through online reinforcement learning; Report-RL specializes the model in producing high-quality long-form reports; and human preference alignment closes the gap between RL-optimized behavior and nuanced user expectations.

##### 5.1 Supervised Fine-Tuning

Training Objective and Data Representation. SFT provides cold-start capabilities for tool invocation, format adherence, and multi-turn reasoning patterns, establishing a policy foundation for subsequent RL. As described in Section 4.4), training data follows the ReAct paradigm with trajectories decomposed into independent steps. At each step t, the model predicts thought content Tt and tool invocation action At based on history H<t. Tools are not executed during training; observations Ot are pre-recorded as contextual inputs.

Given dataset (x,H) ∼ DSFT, the training objective is the standard autoregressive language modeling loss:

LSFT(θ) = −E(x,H)

TH

log πθ(Tt,At | x,H<t) (1)

t=1

This objective performs behavior cloning on expert trajectories, theoretically optimizing a lower bound of the RL policy gradient objective, thereby providing a reasonable initial policy distribution and avoiding the prohibitive sample complexity of training from a random policy.

Training Strategy and Long-Context Enhancement. We curate approximately 15K trajectories through sampling and filtering (composition detailed in Section 4.4), with a systematic adjustment to the data distribution: earlier training is dominated by shorter, simpler samples with an average context length of around 8K tokens, establishing stable format adherence and basic tool invocation; as training progresses, longer and harder samples are gradually introduced, with 32K–64K context data comprising approximately 30% of the mixture and 128K data approximately 15%, alongside lengthadaptive positional encoding fine-tuning. This curriculum data arrangement improves 128K-context format correctness from 72% to 94% while maintaining short-context performance.

The Delicate Balance Between SFT and RL. SFT training extent has a non-intuitive trade-off with subsequent RL effectiveness, directly determining the reasonable boundaries of the data volume and distribution described above. Under-trained models (20–40K samples) retain good exploration capacity but exhibit insufficient format adherence under long contexts, yielding fewer than 30% valid trajectories in early RL. Over-trained models (beyond 200K samples) achieve near-perfect format correctness but exhibit rigidity during RL sampling—over-fitting to the surface patterns of expert trajectories causes sampled rollouts to be highly similar, entropy to collapse rapidly, and gradient signals to vanish. Theoretically, SFT optimizes a lower bound of the RL policy gradient objective: excessive training compresses the policy’s support set, causing πθ(τ) to assign negligibly low probability to non-expert trajectories even when they may yield higher rewards; simultaneously, an over-fitted reference policy πref causes KL divergence DKL(πθ∥πref) to increase rapidly in early RL, triggering KL penalties that suppress exploration.

Training Termination Criteria. Based on the above analysis, we adopt long-context format correctness as the core early stopping metric to quantitatively determine the appropriate SFT termination point. In RL training, each query requires G = 8 sampled trajectories to compute group relative advantages (GRPO), with at least 2 valid trajectories required for stable advantage estimation. By the binomial distribution, format error rate p must satisfy the probability of at least 2 successes ≥ 0.95, yielding p ≤ 0.0253 (i.e., error rate below 2.53%). Accordingly, every 10K steps we measure format error rates at 64K and 128K context lengths on a validation set; early stopping is triggered when both fall below 2.5% and training loss plateaus. We additionally require that policy entropy at this point be no lower than 90% of its mid-training value (at 60K steps), ensuring sufficient exploration capacity is retained. In practice, these conditions are typically satisfied at 100K–150K high-quality samples, achieving an optimal balance between competence and plasticity.

##### 5.2 Search Reinforcement Learning (Search-RL)

The Search-RL stage targets three core challenges of deep search agents: accurate tool invocation, analytical reflection over intermediate reasoning, and information consistency with reasoning correctness in long-context complex tasks. We adopt a unified GRPO-based RL framework that progressively builds capabilities from basic tool invocation to complex long-horizon reasoning within

(a) Reward Component Scores

1.0

Tool = 0.95

0.8

Format = 0.70

0.6

Score

PRM = 0.50

0.4

0.2

ORM (answer accuracy) PRM (entity coverage) Tool invocation Format

0.0

(b) Reward Coefficient Scheduling

tool PRM format PRM ORM

0.6

Coefficient

0.4

tool format PRM ORM

0.2

0.0

(c) Total Reward & Validation Score

0.8

0.8

Fixed-coeff. baseline Dynamic scheduling Dynamic sched. valid.

0.6

0.7

Fixed coeff. valid.

0.4

0.6

TotalReward(train)

ValidationScore

0.2

0.5

Fixed coeff. reward saturates

0.0

0.4

0.2

0.3

0.4

0.2

0.6

0.1

0.8

0.0

0 20 40 60 80 100 120 140 160 180

Training Step

- Figure 5: Training dynamics of Search-RL over 180 steps. (a) Reward component scores: ORM (answer accuracy), PRM (average entity coverage), tool invocation success, and format compliance; dashed horizontal lines mark the scheduling thresholds and dotted vertical lines indicate the three

transition events. (b) Reward coefficient scheduling: λtool, λformat, λPRM, and λORM evolve through three threshold-triggered phases as each capability saturates. (c) Total reward curve, annotated with the three key coefficient adjustment events.

a single continuous training process, through dynamic scheduling of reward weights and training data difficulty. This avoids the hyperparameter sensitivity and capability degradation risks associated with hard-staged decomposition.

Environment. We build a large-scale training and inference environment based on a targeted optimization of the veRL framework. Li-veRL extends the native veRL framework along two dimensions: expert routing optimization and expert load balancing for improved MoE training stability; and efficiency enhancements including asynchronous trajectory generation, inter-trajectory asynchronous execution, and asynchronous tool calling, achieving a 2.9× speedup over native veRL. Training and inference stages employ fully consistent sampling and execution flows, fundamentally eliminating distribution shift. Drawing on the engineering experience of Tongyi DeepResearch [34], we prioritize stability at the tool layer. All tool calls are routed through a unified entry layer responsible for traffic control, exception retry, and result caching, ensuring consistent tool behavior, reducing overall tool error rate to below 0.1%, and providing clear error feedback to the model to prevent repetitive failure patterns.

Tool Capabilities. The model has access to three categories of search capabilities: internal knowledge search, external web search, and academic literature search. Internal knowledge search uses a proprietary search engine combined with a tens-of-billions-scale high-quality internal knowledge base, providing superior coverage in company-relevant domains (automotive, technology, finance) while substantially reducing tool invocation cost. External search tools automatically route queries to the most suitable search providers (including Sogou, Bing, and Quark) based on query content and category tags, ensuring optimal search coverage and quality. The system additionally provides web

crawling and document processing tools for full-text retrieval to support deep reading and information extraction.

For each sampled query x, the model generates a trajectory H = {(Tt,At,Ot)}Tt=1 within the above environment, where Tt is the reasoning content at step t, At the model action (thought or tool call), and Ot the environment-returned observation. Steps are appended sequentially to the context until a final answer is produced.

Sampling and Optimization Objective. The base optimization framework is Group Relative Policy Optimization (GRPO): for each input x, we sample G trajectories {H1,...,HG} and compute group-relative advantages:

G

1 G

Aˆ(x,Hi) = R(x,Hi) −

R(x,Hj) (2)

j=1

The policy optimization objective maximizes expected advantages while constraining policy drift via a KL penalty:

A ˆ(x,H) · log πθ(H|x) − β DKL(πθ(·|x)∥πref(·|x)) (3) where πref is the reference policy (initialized from SFT) and β controls KL constraint strength.

LGRPO(θ) = Ex∼D, H∼π

θ

GRPO yields stable and effective training on dense models. However, the sparse activation property of MoE models exacerbates instability: after one or more gradient updates, the expert networks activated by the same response may shift substantially, causing the activation paths of trajectories sampled under the old policy to be inconsistent with those of the current policy, thereby violating the importance sampling assumption and introducing large gradient estimation bias. We attempted a “Routing Replay” technique—forcing the target policy to activate the same experts as the old policy during updates—to mitigate activation path drift, but experiments showed that training remained unstable. Consequently, for MoE Search-RL training, we adopt GSPO [52] (Group Sequence Policy Optimization).

The key innovation of GSPO is elevating the importance ratio from the token level to the sequence level with length normalization, unifying the numerical range across responses of varying lengths and reducing variance. Let {yi}Gi=1 be the response group sampled from the old policy πθ

for query x, and Aˆi the group-relative advantages. The GSPO objective is:

old

G

1 G

min si(θ)Aˆi, clip(si(θ),1−ε,1+ε)Aˆi (4)

JGSPO(θ) = Ex∼D,{y

i}∼πθold

i=1

where the sequence-level importance ratio si(θ) is defined as the exponentiated mean of token-level log-probability ratios:

  1

  (5)

|yi|

1 |yi|

πθ(yi,t | x,yi,<t) πθ

πθ(yi | x) πθ

log

si(θ) =

= exp

(yi | x)

|yi|

(yi,t | x,yi,<t)

old

old

t=1

Compared to GRPO’s direct cumulative product of token-level probability ratios, the sequence-level si(θ) naturally averages over local expert routing changes, effectively suppressing ratio fluctuations induced by MoE sparse activation, enabling the clip constraint to function stably and preventing search capability degradation.

Dynamic Reward. The reward function comprises four signal types, all evaluated via LLM-asJudge: each assessment involves 3 independent models producing binary (0/1) judgments with rationale, aggregated via majority vote. Both ORM and PRM employ this mechanism to improve evaluation consistency and robustness.

Step-level Reward Definitions. Tool invocation reward and format reward are computed independently at each trajectory step t. Let ct ∈ {0,1} denote the success indicator for the tool call at step t; the tool invocation reward is:

 

+0.1 ct = 1 −0.2 ct = 0 and ct−1 = 0 (consecutive failure) −0.1 ct = 0 and ct−1 = 1 (isolated failure)

(6)

rtool(t) =



The format reward checks the structural validity of each step’s output:

+0.1 output format correct −0.2 output format incorrect

(7)

rformat(t) =

Empirically, applying stronger negative penalties (rather than positive incentives) for error-prone behaviors such as format and tool invocation failures leads to faster error avoidance.

PRM. The process reward is evaluated against the set of key intermediate entities E = {e1,e2,...,eM} associated with the query. Entities are verified via string matching within the reasoning content Tt and tool call observations Ot at each step. If entity ej is detected at any step, that step receives the corresponding entity score contribution. The final PRM reward is the ratio of cumulatively observed entities to the total entity count:

1 M

RPRM(H) =

M

eˆj, eˆj = ⊮[∃t ∈ [1,T] : ej ∈ Tt ∪ Ot] (8)

j=1

This design requires no LLM inference for verification, is low-cost and scalable, and provides stable intermediate process supervision signals.

ORM. ORM evaluates the overall correctness of the final answer via the same LLM-as-Judge majority-vote mechanism, yielding RORM(H) ∈ {0,1}. Trajectory-level Composite Reward. The total reward for trajectory H is defined as:

1 T

R(x,H) = λORM RORM(H) + λPRM RPRM(H) +

T

[λtool rtool(t) + λformat rformat(t)] (9)

t=1

All coefficients satisfy the normalization constraint λORM + λPRM + λtool + λformat = 1, where λORM = 1 − λtool − λformat − λPRM is an implicit variable that adjusts automatically as the other three are tuned, constrained to [0, 0.5].

The four coefficients follow a threshold-triggered scheduling strategy, initialized at (λtool,λformat,λPRM,λORM) = (0.6,0.3,0.1,0.0) and adjusted across three phases: as tool invocation saturates, λtool is reduced and the released weight transferred to λPRM; once format compliance stabilizes, λformat is similarly reduced; and when PRM coverage matures, dominance shifts to λORM. This adaptive coupling ensures each capability receives focused supervision at its appropriate training stage, mirroring a “grokking” progression from basic skills to deep reasoning (Fig. 5).

Dynamic Data. To complement the reward scheduling, we dynamically manage training data difficulty. At fixed training step intervals, we sample and evaluate the current policy on a validation set, compute ORM accuracy across difficulty bins, and adjust the sampling proportions of different difficulty levels in the next training batch accordingly—targeting an ORM accuracy of 10%–50% per sampling round. An accuracy below 10% indicates tasks are too hard, yielding overly sparse rewards and ineffective gradients; above 50% indicates tasks are too easy, with policy saturation and diminishing returns. By maintaining the model within this “effective learning zone”, dynamic data scheduling works in concert with dynamic reward scheduling to sustain sufficient and effective learning signals throughout training.

##### 5.3 Report Reinforcement Learning

Report-RL is a reinforcement learning framework specifically designed for long-form report generation. The core idea is to leverage LLM-based evaluators scoring model outputs against structured RACE Rubrics to form reward signals for RL training. This stage focuses on improving the model’s ability to produce comprehensive, readable, insightful, and instruction-following reports.

Framework and Environment. The overall Report-RL pipeline is illustrated in Fig. 6. The training framework adopts the same Li-veRL unified infrastructure as Search-RL, supporting asynchronous inference generation and asynchronous reward computation to improve training throughput for longform generation. Unlike Search-RL, Report-RL involves neither complex tool invocations nor multiturn environment interactions, requiring no additional tool layer or external service dependencies;

|Rubrics<br><br>Reference Report|
|---|

|RACE Rubrics Scorer<br><br>Compreh. Readability<br><br>Insight Instr. Follow<br><br>| |
|---|---|
| | |

###### Inputs

Reference Report & Rubrics Generation Template

LLM

Query

System Prompt

Policy Model Dense (Qwen3-32B) MoE (Qwen3-30B-A3B)

Retrieval Data

Generated Report

⊕

|Citation Reward<br><br>Citation Amount<br><br>Format Reward<br><br>Citation Format<br><br>Tag Match<br><br>Markdown Format|
|---|

Outline

###### Long-form Input

Query

System Prompt

RL OPtimization GRPO/DAPO/GSPO

Short-form Input

- Figure 6: Overview of the Report-RL framework. Given a long-form input, the policy model and a frontier LLM such as Gemini 3.1 Pro generate sample report and reference report, respectively. The frontier LLM is also used to generate RACE Rubrics on which the sample report is evaluated. MindDR surpasses the performance of the distilled frontier LLM on DeepResearch Bench and MindDR Bench as shown in Fig. 1 .

environment feedback is provided entirely by the scoring model, keeping the setup simple and deployment cost low.

Given an input consisting of a query, system prompt, upstream deep search retrieval data, and an outline, the base model generates a report. The generated report, together with pre-computed RACE Rubrics, is evaluated by a scoring model along four dimensions—comprehensiveness, readability, insight, and instruction following—combined into a weighted composite reward signal. The Rubrics design follows the RACE evaluation framework from DeepResearch Bench [7]; the scoring prompt is detailed in Appendix 8.3.

Reward Design. In RL training practice, auxiliary reward signals are commonly introduced alongside a core reward to address specific quality issues. During our training we observe systematic errors such as tense inconsistency and malformed table output. Our guiding principle is to proactively categorize problems exposed during training: those detectable by rules without LLM inference are incorporated as auxiliary RL rewards; those requiring holistic LLM judgment are deferred to the preference alignment stage. This separation prevents sparse sampling noise from distorting training dynamics and avoids reward ambiguity from additional LLM-based evaluators. We add two auxiliary rewards beyond the RACE Rubrics signal:

Citation Reward Rcite. Let ngen denote the number of citations in the generated report, nref the number in the reference report, and nvalid the number of valid citations—a citation is considered valid if the textual relevance between its description and the cited passage exceeds threshold τ. We define:

 

+0.1 ngen ≥ 0.7nref and nvalid ≥ 0.7nref −0.1 ngen ≥ 0.7nref and nvalid < 0.7nref −1 ngen < 0.7nref

(10)

Rcite =



Sufficient citations with adequate validity receive a positive reward; sufficient citations with poor validity incur a mild penalty; insufficient citations incur a heavy penalty.

Format Reward Rformat. We define binary violation indicators for three structural issues in the generated report:

- • vtag = ⊮[final answer not properly enclosed in <final_answer> tags]
- • vmd = ⊮[Markdown formatting errors (list numbering, table rendering, etc.)]
- • vref = ⊮[citation formatting errors]

Each violation is penalized independently:

Rformat = − vtag + vmd + vref ∈ [−3, 0] (11)

The overall reward is:

RReport = RRACE + λcRcite + λfRformat (12) where RRACE is the weighted RACE score, and λc, λf are tunable balancing coefficients. Different base models exhibit different rates of citation deficiency and format violations during training; calibrating these coefficients to observed error rates enables fine-grained control over auxiliary reward influence on the primary optimization direction, preventing convergence disruption from over- or under-penalizing individual issues.

Optimization Objective. We adopt GRPO as the baseline optimization algorithm. While GRPO achieves stable results on dense models, its limitations are amplified in long-form generation: sequence-level policy gradients assign equal weight to responses of varying length, diluting gradient contributions from longer sequences; symmetric clipping restricts upward exploration of importance ratios, risking entropy collapse; and the KL divergence constraint further suppresses exploration. For Report-RL on dense models, we therefore adopt DAPO [49].

Relative to GRPO, DAPO introduces four key improvements: (1) token-level policy gradients, normalizing loss by total token count to eliminate gradient imbalance across response lengths; (2) asymmetric clipping (clip-higher), using εlow < εhigh to provide greater upward exploration room and suppress entropy collapse; (3) removal of the KL constraint, enhancing free exploration in reward space; and (4) dynamic sampling filter, discarding groups where all samples are correct or all incorrect to ensure non-trivial advantage estimates. Letting ri,t(θ) = πθ(yi,t | x,yi,<t)/πθ

(yi,t | x,yi,<t), the DAPO objective is:

old

1 i |yi| i t

min ri,t(θ)Aˆi, clip(ri,t(θ),1−εlow,1+εhigh)Aˆi

JDAPO(θ) = Ex∼D′,{yi}∼πθold

(13) where D′ is the dynamically filtered training set and Aˆi are group-relative advantages.

Given the high inference cost of Qwen3-32B, we also explore Qwen3-30B-A3B (MoE) as a unified backbone. Token-level importance ratios are highly sensitive to expert routing changes under MoE sparse activation, causing the same clipping instability for DAPO as for GRPO. We therefore apply GSPO [52] consistently with the Search-RL stage (Section 5.2): its sequence-level importance ratio si(θ) averages over local routing changes, stabilizing the clip constraint and preventing report writing capability from degrading.

##### 5.4 Preference Alignment

After SFT and RL training, the model has acquired strong deep search and report writing capabilities. However, RL reward design is primarily driven by quantifiable objective metrics and is inherently limited in addressing experience-quality issues that require holistic LLM judgment to identify and cannot be defined by simple rules—such as tense inconsistency, unnatural tone, abrupt paragraph transitions, and conclusions inconsistent with retrieved content. The preference alignment stage aims to close this gap by systematically shifting the model’s output distribution from low-quality regions toward high-quality regions aligned with human expectations.

Data Construction. To maintain consistency between training data and the model’s own generation distribution, we adopt a self-sampling strategy: for each input x, we sample multiple outputs {y1,...,yK} from the current policy and score each through a quality pipeline combining two signal types: (1) LLM-as-Judge, where multiple independent evaluators assess fine-grained dimensions including comprehensiveness, logical consistency, language quality, and tense accuracy; and (2) rule-based detection, applying hard penalties for detectable structural issues (missing citations, format violations, etc.). The two signals are weighted and combined into a quality score s(x,y), which is used to partition outputs into a high-quality set D+ and a low-quality set D−.

Training Methods. The alignment stage employs two complementary methods—DPO [24] and Self-SFT [1]—driving distribution shift from the perspectives of preference contrast and behavior cloning, respectively.

DPO. We construct a preference dataset Dpref = {(x,y+,y−)} from high- and low-scoring output pairs for the same input, where y+ ∈ D+ and y− ∈ D−. DPO directly optimizes the log-probability

- Table 2: Performance on five DS benchmarks. Best results in our evaluation environment are shown in bold, and second-best results are underlined.

Wide Search Large-Scale Foundation Models

Browse Comp-ZH

Browse Comp

xbench -DS

GAIA -DS

Model

GLM-4.6 [50] 45.1 49.5 73.0 52.6 43.1 Kimi K2 [31] 28.8 14.1 50.0 57.7 54.4 DeepSeek R1 [5] 34.6 14.1 50.0 57.7 44.3 Qwen3-235B [44] 31.1 21.7 57.0 63.1 46.4

Comparable-Scale Agent Models

WebDancer-32B [40] 25.3 10.5 11.0 63.1 39.7 WebSailor-32B [16] 25.6 14.8 46.0 50.5 40.3 WebShaper-32B [29] 28.0 33.5 53.0 54.4 35.2 MiroThinker-v1.5-30B-A3B [33] 31.9 30.4 5.0 23.3 37.9 OpenSeeker-30B-A3B[8] 26.4 12.9 48.5 46.7 36.4 Tongyi-DR-30B-A3B [34] 43.2 40.7 69.0 68.9 41.7

Our Agent Models

MindDR-v1.0-32B 28.4 18.6 13.3 50.3 41.3 MindDR-v1.5-32B 35.6 31.8 64.0 67.1 46.5 MindDR-v1.5-30B-A3B 45.7 42.8 75.0 70.9 44.0

margin of the policy relative to a reference policy, without requiring an explicit reward model:

πθ(y− | x) πref(y− | x)

πθ(y+ | x) πref(y+ | x) − β log

(14)

LDPO(θ) = −E(x, y+, y−)∼Dpref log σ β log

where πref is the reference policy at the start of the alignment stage and β controls preference strength. Self-SFT. We directly fine-tune on samples in D+ to reinforce the model’s ability to reproduce high-quality output patterns:

LSelf-SFT(θ) = −E(x, y+)∼D+ log πθ(y+ | x) (15)

Self-SFT can be understood as reinforcement on static data: analogous to RL’s mechanism of driving policy updates through online sampling, Self-SFT uses high-quality self-sampled data as supervision signals to improve output quality through static iteration while preserving on-policy characteristics. Compared to online RL, it incurs lower computational cost and exhibits more stable convergence, making it a suitable complement to the alignment stage.

Both methods build training data exclusively from the model’s own samples, constraining distribution adjustment to the model’s existing output space. This prevents the policy from being pulled toward unexplored regions, preserving the search, reasoning, and writing capabilities accumulated during SFT and RL, while continuously shifting outputs toward quality levels aligned with human expectations.

### 6 Main Results

In this section, we evaluate the performance of MindDR across various deep search and deep research tasks. We first introduce the experimental setup, detailing the compared baselines and the evaluation benchmarks. We then present the main results in three parts: DeepSearch benchmark performance, DeepResearch benchmark performance, and the final gains from preference alignment.

##### 6.1 Experimental Setup

Evaluated Models. We evaluate our models against a diverse set of strong baselines, including both large-scale foundation models and comparable-scale agent systems:

• Large-Scale Foundation Models: We compare against leading proprietary and open-source models, including Gemini 3.1 [4], Gemini 2.5 Pro [3], GLM-4.6 [50], Kimi K2 [31], DeepSeek R1 [5], Doubao, and Qwen3-235B [44].

MindDR-30B-A3B MindDR-32B

###### BrowseComp

###### BrowseComp-ZH

###### xbench-DS

80

-2.0

70

+24.0

-1.0

60

+17.0

Performance

50

+11.0

-0.4

40

+13.0

-0.1

-1.7

+29.3

30

+10.8

- +11.1

+19.8

-1.4

- +12.2

+21.9

20

+5.2

10

+4.7

0

Base After SFT After Search-RL Final model

Base After SFT After Search-RL Final model

Base After SFT After Search-RL Final model

- Figure 7: Stage-wise DS benchmark performance from the base model to SFT, SEARCH-RL, and the final model. SEARCH-RL consistently delivers the largest gains across all three benchmarks and both model sizes, while the final stage introduces only minor regressions, indicating a small trade-off in search performance.

- • Comparable-Scale Agent Systems: We evaluate against leading open-source agent models around the 30B parameter scale, including WebDancer-32B [40], WebSailor-32B [16], WebShaper-32B [29], MiroThinker-v1.5-30B-A3B [33], OpenSeeker-30B-A3B [8], TongyiDR-30B-A3B [34].
- • MindDR Variants: We also evaluate two versions of our system: MindDR-v1.0, which is trained using only the Reinforcement Fine-Tuning (RFT) stage, and MindDR-v1.5, the fully upgraded version that incorporates the entire RL pipeline (Search-RL, Report-RL, and Preference Alignment). In addition, we have MindDR-v1.5-32B to represent MindDR-v1.5 with Qwen3-32B as the base model and MindDR-v1.5-30B-A3B with Qwen3-30B-A3B as the base model.

Benchmarks. We comprehensively evaluate the systems across two distinct categories of benchmarks:

- • DeepSearch (DS) Benchmarks: We measure multi-step information retrieval and reasoning capabilities using BrowseComp-ZH [54], BrowseComp [38], xbench-DS [2], GAIA-DS [21], and WideSearch [39]. These benchmarks require agents to autonomously navigate web environments and extract accurate answers for complex queries.
- • DeepResearch (DR) Benchmarks: We assess the ability to generate comprehensive, structured, and human-aligned long-form reports on DeepResearch Bench [7] and our proposed MindDR Bench. Evaluation is conducted using RACE rubrics covering comprehensiveness, insight, instruction following, and readability alongside user experience metrics such as citation accuracy, table format and temporal error.

- Table 3: Performance on MindDR Bench on RACE score and Citation Accuracy. Best results are shown in bold, and second-best results are underlined. Comp.: Comprehensiveness; Inst.: Instruction Following; Read.: Readability; C.Acc.: Citation Accuracy.

Model RACE Comp. Insight Inst. Read. C.Acc. Gemini 3.1 49.65 49.83 50.75 49.32 46.89 77.20% Gemini 2.5 Pro 48.34 47.56 46.88 49.47 50.23 81.86% Doubao 46.25 48.21 40.88 48.71 48.18 68.43% Kimi 45.20 46.08 40.60 47.94 47.53 75.01% Qwen 45.07 45.55 39.94 48.34 48.01 76.42% MindDR-v1.0 44.33 44.72 39.49 47.58 47.66 82.14% MindDR-v1.5 51.77 52.17 51.77 50.55 52.18 80.25%

Gemini 2.5 Pro Kimi

Qwen MindDR-v1.0 MindDR-v1.5

| |
|---|

| |
|---|

| |
|---|

55.30

Doubao

55

52.54

51.54

51.26

50.45

50.12 50.00

49.71 49.51 49.45

50

47.95

47.14

47.00

46.22

45.63

45.59

44.96

44.84

44.69

44.64

45

44.34

Score

44.27 43.83

42.62

41.97

41.60

40.56

40.46

40

39.46

35

34.44

30

Overall Comprehensiveness Insight Instruction Following Readability

- Figure 8: Comparison with mainstream DR systems on the public DeepResearch-Benchmark leaderboard. Each group of bars represents a evaluation dimension, and the scores are annotated above the corresponding bars. MindDR-v1.5 (green) achieves the highest scores across all five metrics.

##### 6.2 Overall Performance

- Table 2 summarizes the overall DeepSearch performance. MindDR-v1.5-30B-A3B establishes a strong DS frontier among open-source agent-style systems in our evaluation environment, achieving the best results on BrowseComp-ZH, BrowseComp, xbench-DS, and GAIA-DS. MindDR-v1.5-32B achieves the best WideSearch result, indicating that the gains generalize across backbones rather than being specific to a single checkpoint. Overall, the final MindDR models close or surpass the gap to stronger foundation-style baselines while clearly outperforming comparable-scale open agent systems. We next evaluate DR quality using RACE and its subdimensions on MindDR Bench. Fig. 8 and
- Table 3 show that MindDR-v1.5 moves into the top tier of DR systems. The gains are broad rather than metric-specific: MindDR-v1.5 leads on overall RACE and on the main report-quality dimensions, including comprehensiveness, insight, instruction following, and readability. At the product level, the main remaining exception is citation accuracy, where MindDR-v1.0 remains slightly stronger, suggesting that report-quality optimization and citation-faithfulness optimization are related but not identical objectives.

The DS and DR results together indicate that MindDR achieves a favorable balance between strong search performance and high-quality report generation. The system is therefore competitive not only

- as a search agent, but also as a complete deep research system.

- Table 4: Report-RL ablation results evaluated on MindDR Bench. Higher RACE, tag format, citation format, and BrowseComp-ZH are better; lower table error is better.

Model RACE Tag Format Citation Format Table Error BrowseComp-ZH

Qwen3-30B-A3B + Search-RL 39.54 69% 78% 0.41% 47.1% + Report-RL (GRPO) 41.68 91% 98% 0.85% 29.1% + Report-RL (DAPO) 43.27 95% 99% 0.65% 43.3% + Report-RL (GSPO) 44.05 99% 99% 0.50% 45.7%

Qwen3-32B +SFT + Search-RL 43.32 96% 97% 0.41% 35.7% + Report-RL (GRPO) 48.06 97% 97% 2.60% 28.6% + Report-RL (DAPO) 48.82 99% 99% 2.70% 35.6%

##### 6.3 Detailed Analysis

We now turn to a more detailed analysis of the training stages, with particular attention to the trade-off between DS capability, DR quality, and final product-level refinement.

- Table 5: Ablation results of long-form and mixed-form inputs for Report-RL evaluated on MindDR Bench.

Model RACE Comp. Insight Inst. Read. Qwen3-32B + SFT + Search-RL 43.32 44.30 37.93 46.96 45.89

+ Report-RL (DAPO), Long Only 48.82 49.10 47.68 49.42 49.56 + Report-RL (DAPO), Long + Short 50.60 51.08 50.96 49.67 49.66

Table 4 shows that the main challenge in DR optimization is not whether Report-RL improves report quality—it does—but how much prior DS capability is preserved while doing so. Sequence-level methods such as GSPO and DAPO provide the best balance in this regard: they improve report quality and formatting substantially while keeping search regression small, whereas GRPO causes noticeably larger DS degradation.

Effects of Long-form and Short-form Queries. Starting from an SFT checkpoint, we apply DAPO [49] with RACE Rubrics as the reward. Table 5 compares Report-RL trained on long-formonly data against a mixture of long-form and short-form data. Adding short-form data yields a consistent gain across all metrics: RACE improves from 48.82 to 50.60, with notable increases in Comprehensiveness (49.10→51.08) and Insight (47.68→50.96). This suggests that mixing compact supervision into long-form RL training provides a stronger optimization signal and helps the model generalize to both response lengths.

- Table 6: Effect of phase 4 preference alignment on DeepResearch Bench score and report-quality error metrics. DPO targets table and temporal correctness, while Self-SFT further improves overall writing consistency.

DeepResearch Bench

Table Error

Post-proc. Table Error

Temporal Error

Expression / Logic Issue Rate

Model

MindDR-v1.5-ReportRL 50.06 2.70% 1.35% 6.2% 1.8% + DPO 50.07 1.22% 0.16% 2.0% 1.8% + Self-SFT 51.77 1.22% 0.16% 2.0% 0.3%

Quality Refinement by preference alignment. Even after Report-RL, rubric-based evaluation reveals residual defects that are only weakly captured by scalar DR rewards, including table-information misalignment, temporal-expression errors, paragraph-level logical discontinuities, and language inconsistency. We therefore apply a final refinement stage composed of DPO and Self-SFT. DPO targets the most structured and objectively judgeable issues—table correctness and temporal correctnessusing a curated 1.8K temporal dataset and a 2.8K table-repair dataset, while Self-SFT on 4.3K high-quality self-sampled reports improves coherence and stylistic consistency.

- Table 7: Examples of temporally correct and temporally incorrect statements in generated reports. Type Example Explanation

Correct 1. “According to XX organization in 2024, the global VR/AR training market is expected to reach tens of billions by 2025.”

- 1. Prediction about a past time point with explicit attribution.
- 2. Prediction about a future time point, consistent with temporal logic.

2. “The global robotics market is expected to reach hundreds of billions by 2027.”

Incorrect “By 2025, the global VR/AR training market will reach tens of billions.”

Prediction about a past time point without attribution.

Table 6 shows a clear division of labor between the two refinement steps. DPO sharply reduces structured factual and formatting errors, especially table and temporal errors, while leaving the aggregate DR benchmark essentially unchanged. Self-SFT then improves higher-level expression and coherence, reducing the expression/logic issue rate further and yielding an additional gain on the benchmark itself. In combination, the two stages improve report correctness and presentation without sacrificing the performance established by Report-RL.

We define a temporal error as a predictive statement about a past time point that lacks explicit attribution to a source.

- Table 8: Temporal error rates on MindDR Bench. Lower is better.

#### Model Temporal Error Rate

MindDR-v1.5 2.0% Kimi 3.2% Qwen3 8.2% Gemini 10.2% MindDR-v1.0 11.4% Doubao 14.0%

Table 7 illustrates this distinction, and the temporal error comparison in Table 8 shows that MindDR-v1.5 achieves the lowest temporal error rate among all compared systems. This indicates that the final refinement stage corrects a report defect that remains common even in strong industrial systems and helps turn a strong model into a product-quality system. MindDR-v1.5 achieves the lowest temporal error rate among all compared systems at 2.0%, substantially improving over MindDR-v1.0 and outperforming all external baselines. This highlights the effectiveness of the final refinement stage in correcting a subtle but important reportquality defect.

Training and Test-time Efficiency. Fig. 9 examines inference efficiency on BrowseComp-ZH from four complementary perspectives: accuracy versus average tool calls (upper-left) and average context tokens (upper-right), and performance under varying context limits (lower-left) and tool-call limits (lower-right). The upper scatter plots adopt a quadrant view where the top-left region denotes the ideal Accurate & Efficient zone.

The five compared systems exhibit distinct efficiency–accuracy trade-offs. OpenSeeker-30B-A3B is the most resource-frugal yet its accuracy lags noticeably. Miro-30B-A3B sits at the opposite extreme, incurring the highest context and tool-call consumption while delivering limited accuracy gains. GLM-4.6, a leading proprietary model, achieves competitive scores at the cost of substantially greater resource usage. In contrast, MindDR-v1.5-30B-A3B falls squarely within the Accurate & Efficient quadrant, attaining the highest BrowseComp-ZH score (45.7) while requiring the fewest average context tokens and tool calls among top-performing systems.

###### Score vs. Avg Tool Calls

###### Score vs. Avg Context Tokens

* Accurate & Efficient Accurate but Resource-heavy

* Accurate & Efficient Accurate but Resource-heavy

50

50

MindDR-v1.5-30B-A3B

MindDR-v1.5-30B-A3B

GLM-4.6

45

45

GLM-4.6

Accuracy(%)

Accuracy(%)

Tongyi-30B-A3B

Tongyi-30B-A3B

40

40

MiroThinker-30B-A3B

MiroThinker-30B-A3B

35

35

OpenSeeker-30B-A3B

OpenSeeker-30B-A3B

30

30

Efficient but Inaccurate Costly & Inaccurate

Efficient but Inaccurate Costly & Inaccurate

5 10 15 20 25 30 35

0 10 20 30 40 50 60 70

Avg. Tool Calls per Query

Avg. Context Tokens per Query (k)

###### Context Limit (Browsecomp-zh)

###### Tool Call Limit (Browsecomp-zh)

50

50

45.7

45.7

45.1

45.1

43.2

43.2

43.2

43.2

41.3

40.5

40.1

39.8

40

40

39.2

38.8

Accuracy(%)

Accuracy(%)

37.4

37.1

36.7

36.7

35.9

35.7 33.9

35.7

34.3

34.2

34.2

33.9

33.6

33.2

33.0

32.5

32.5

31.5

31.4

30.1

30

30

28.7

27.8

27.3

25.9 26.3

25.1

20

20

16k 32k 64k 128k

8 16 32 64

Context Limit

Tool Call Limit

GLM-4.6 Tongyi-30B-A3B OpenSeeker-30B-A3B MiroThinker-30B-A3B MindDR-v1.5-30B-A3B

- Figure 9: Efficiency and scalability analysis on BrowseComp-ZH. Top row: quadrant plots of accuracy versus average tool calls (left) and context tokens (right) per query. Bottom row: accuracy under varying context-length limits (left) and tool-call limits (right). MindDR-v1.5-30B-A3B consistently occupies the Accurate & Efficient region and maintains leading performance across all budget settings among comparable 30B systems.

The lower bar charts confirm that this advantage is robust across operational constraints. Under context limits from 16k to 128k and tool-call limits from 8 to 64, MindDR-v1.5-30B-A3B consistently matches or surpasses all compared models. Although it starts modestly under the most restrictive tool-call budget, it scales rapidly and achieves the best accuracy at standard and permissive settings, indicating that its deep-search strength reflects genuinely higher per-step retrieval efficiency rather than simply larger contexts or more aggressive tool use.

More broadly, these results support the efficiency claim of the overall pipeline. A key difference from many industry deep-research systems is that MindDR removes the midtraining stage that is often used to inject tool use, strategy selection, and reflection behaviors into the base model, sometimes

- at a scale exceeding 150B tokens. This design also yields a clear training-efficiency advantage relative to our previous-generation system. A single MindDR 1.0 training run used 280K high-quality RFT samples, corresponding to approximately 3.6B tokens, together with about 15K GPU cardhours. By contrast, MindDR 1.5 uses only about 0.18B SFT tokens and 0.85B RL sampling tokens, for a total of roughly 1.03B training-related tokens, and requires only about 6K GPU card-hours. Instead of relying on a large intermediate training phase followed by end-to-end RL, MindDR 1.5 adopts a staged optimization design: the system first establishes cold-start behavior with SFT, then separately optimizes search capability with Search-RL, report generation with Report-RL, and final product quality with preference alignment. This decomposition substantially reduces training resource requirements while yielding stronger downstream performance. The results therefore suggest that MindDR’s gains arise less from brute-force scaling and more from replacing monolithic midtraining-plus-end-to-end optimization with a more efficient staged training and reward design.

### 7 Discussion and Conclusion

##### 7.1 Limitations

Context Management. While our progressive length generalization strategy achieves 94% format correctness at 128K context, scaling to even longer contexts for extremely complex research tasks remains an open challenge. Developing more effective context management strategies—such as hierarchical memory, selective context compression, or adaptive attention mechanisms—that allow agents to maintain focus on the most relevant evidence while operating over very long horizons is an important direction for future work.

Evaluation coverage. Current evaluation primarily relies on the RACE framework and factual accuracy metrics. However, deeper aspects of research quality—such as methodological soundness, argument novelty, and appropriate hedging of uncertain claims—are difficult to capture with automated rubrics and would benefit from more nuanced human evaluation protocols.

##### 7.2 Conclusion

We presented MindDR, a cost-effective, open multi-agent framework designed to address the fundamental bottleneck of deep research agents: achieving top-tier performance and excellent user experience without relying on prohibitive training and inference costs. By avoiding computationally expensive continual pre-training and adopting a highly targeted optimization strategy, MindDR demonstrates that ~30B-parameter models can match or surpass the deep research capabilities of much larger foundation models.

MindDR tackles the cost-performance trade-off through both inference-stage decomposition and training-stage targeted optimization. At inference, the three-agent architecture (Planning, DeepSearch, and Report Agents) coordinates via an Extended Chain-of-Thought mechanism to parallelize search and isolate contexts, naturally alleviating long-context burdens. At training, the staged pipeline progresses from SFT to Search-RL (explicitly optimizing search efficiency to reduce redundant token consumption), then to Report-RL (resolving information conflicts for long-form generation), and finally to preference alignment (correcting residual formatting and temporal defects).

Empirically, MindDR achieves strong results on both DeepSearch (DS) and DeepResearch (DR) evaluations. On DS benchmarks, MindDR-v1.5-30B-A3B attains the best results among opensource agent-style systems on BrowseComp-ZH, BrowseComp, xbench-DS, and GAIA-DS, while maintaining its advantage under varying context-length and tool-call constraints. Furthermore, we

introduced MindDR Bench alongside a comprehensive multi-dimensional evaluation system, moving beyond single-metric assessments to systematically evaluate both content quality and presentation format. On this benchmark, MindDR reaches a state-of-the-art RACE score of 51.8, leading across comprehensiveness, insight, instruction following, and readability.

### 8 Appendix

- 8.1 RACE Rubrics Example

Below is an illustrative example of the RACE Rubrics generated for a specific query. Each rubric contains four evaluation dimensions—comprehensiveness, insight, instruction following, and readabilitywith multiple fine-grained criteria, explanations, and importance weights. The scoring model uses these rubrics to produce per-dimension scores that serve as reward signals during Report-RL training. For brevity, we show a representative subset of criteria per dimension.

RACE Rubrics Example (Abridged)

{

"Query": "How has the release of the domestic LLM DeepSeek triggered educational reform?", "dimension_weight": {

"comprehensiveness": 0.29, "instruction_following": 0.10, "insight": 0.44, "readability": 0.17

}, "criterions": {

"comprehensiveness": [

{ "criterion": "Completeness of DeepSeek model and release event information", "explanation": "Whether the article comprehensively introduces DeepSeek’s core information, including R&D institution, release timeline, key technical parameters, core capabilities, and open-source/API access strategy.", "weight": 0.6 },

{ "criterion": "Coverage of specific educational application scenarios", "explanation": "Whether the article enumerates concrete applications such as intelligent tutoring, AI teaching assistants, content generation, and assessment.", "weight": 0.8 },

{ "criterion": "Identification of challenges, risks, and ethical issues", "explanation": "Whether the article identifies issues including academic integrity, data privacy, educational equity, and over-reliance on AI.", "weight": 0.7 }

], "insight": [

{ "criterion": "Depth of reform mechanism analysis", "explanation": "Whether the article deeply analyzes the core transmission mechanism rather than merely listing application scenarios.", "weight": 1.0 },

{ "criterion": "Critical thinking and deep risk foresight", "explanation": "Whether the article demonstrates balanced and critical perspectives.", "weight": 0.6 }

], "instruction_following": [

{ "criterion": "Subject focus on DeepSeek", "weight": 1.2 }, { "criterion": "Scope strictly limited to education", "weight": 1.2 }

], "readability": [

{ "criterion": "Macro structure and logical framework", "weight": 0.8 }, { "criterion": "Language accuracy and fluency", "weight": 0.6 }

] }

}

- 8.2 Short-form Data Synthesis Prompts

###### Inference Prompt for Report Generation

System: You are a professional report generation agent, skilled at writing well-organized and insightful professional reports.

## Report Quality Requirements

- - The report should be complete and comprehensive, without omitting key information.
- - The report should have sectional conclusions and a final summary for easy comprehension.
- - Sections listing facts or statistics should use tables for readability.

User: Based on the following user QUERY, write a report. User Query: {query}. Generate the report directly.

Reference Report Synthesis Prompt System: [Same system prompt as above] User: Based on the following user QUERY, considering the following dimensions

{criterions}, write a report. User Query: {query}. Generate the report directly.

##### 8.3 Scoring Model Prompt

The scoring model evaluates two articles (the generated report and a reference report) using the RACE Rubrics. The prompt template instructs the model to act as a rigorous evaluation expert, performing criterion-by-criterion comparative analysis with scores on a 0–10 scale.

Scoring Model Prompt Template

System: You are a rigorous, meticulous, and objective research article evaluation expert. You excel at comparing two articles written for the same task based on specific evaluation criteria, providing precise scores and clear justifications.

User:

**Task Background** A deep research task requires evaluating two articles across four dimensions: comprehensiveness, insight, instruction following, and readability. <task>{task_prompt}</task>

**Articles to Evaluate**

- <article_1>{article_1}</article_1>
- <article_2>{article_2}</article_2>

**Evaluation Criteria** <criteria_list>{criteria_list}</criteria_list>

**Scoring Rules** 0–2: Very poor. Almost completely fails to meet the criterion. 2–4: Poor. Minimally meets the criterion with obvious deficiencies. 4–6: Average. Basically meets the criterion. 6–8: Good. Largely meets the criterion with notable strengths. 8–10: Excellent. Fully or exceeds expectations.

**Output Format** { "comprehensiveness": [ { "criterion": "[text]", "analysis": "[analysis]",

"article_1_score": [0-10], "article_2_score": [0-10] }, ... ], "insight": [...], "instruction_following": [...], "readability": [...] }

###### 8.4 Temporal Tense Error Detection The temporal tense error detection pipeline consists of two stages:

- Stage 1: Regex-based extraction. We use a rule-based extractor to identify sentences containing predictive temporal expressions. The extractor matches date patterns (e.g., “by 2025”, “in 2024Q3”) co-occurring with future-tense keywords (e.g., “is expected to”, “will”, “is projected to”) within a bounded text window. Sentences where the predicted date falls before the current date are flagged as candidates for tense errors.
- Stage 2: LLM-based institution verification. Each flagged sentence, together with its surrounding context (1–2 sentences before and after), is fed to an LLM with the following verification prompt:

###### Temporal Verification Prompt

System: You are a rigorous fact-checking assistant. Based on the context, objectively determine whether an explicit forecasting entity is mentioned. Output only the specified JSON format.

User: Context: {context} Please determine: within 1–2 sentences of "{matched_text}", is there an explicit mention of the forecasting institution that made this prediction? "Forecasting institution" includes:

- - Named institutions (e.g., IMF, Goldman Sachs)
- - Generic but explicit third parties (e.g., "a research institute")
- - The company itself describing its own plans (e.g., "the company expects") Output: {"result": "yes/no", "detail": "brief explanation"}

A sentence is classified as a tense error if and only if (i) it contains a prediction about a past time point and (ii) no forecasting institution is identified in the surrounding context.

### Contributions

Project Lead Sheng Yang

Core Contributors Biao Wang, Haozhi Xie, Heyang Xu, Liping Wang, Shirui Zhang, Shuai Wang, Sirui Miao, Tiankuo Xu, Xuefeng Hao, Ying Liu, Yingjie Feng, Yuchen Liu, Yuhang Wu, Zhengxin Yu, Zhuo Liu

Contributors Bin Huang, Dong Wu, Handong Cui, He Cao, Jiabang He, Jiajun Yang, Jialu Chen, Jiqing Zhan, Li Gong, Lian Wen, Qingfeng Cai, Xiaobo Liu, Yuan Xue, Yun Zhu

Sponsors Xiaofei Gou, Wei Chen

### References

- [1] Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. Retaining by doing: The role of on-policy data in mitigating forgetting. arXiv preprint arXiv:2510.18874, 2025. doi: 10.48550/arXiv.2510.18874. URL https://arxiv.org/abs/2510.18874.
- [2] Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, et al. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations. arXiv preprint arXiv:2506.13651, 2025.
- [3] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, NanJiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell, Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Alvin Abdagic, Lior Belenki, James Allingham, Anima Singh, Theo Guidroz, Srivatsan Srinivasan, Herman Schmit, Kristen Chiafullo, Andre Elisseeff, Nilpa Jha, Prateek Kolhar, Leonard Berrada, Frank Ding, Xiance Si, Shrestha Basu Mallick, Franz Och, Sofia Erell, Eric Ni, Tejasi Latkar, Sherry Yang, Petar Sirkovic, Ziqiang Feng, Robert Leland, Rachel Hornung, Gang Wu, Charles Blundell, Hamidreza Alvari, Po-Sen Huang, Cathy Yip, Sanja Deur, Li Liu, Gabriela Surita, Pablo Duque, Dima Damen, Johnson Jia, Arthur Guez, Markus Mircea, Animesh Sinha, Alberto Magni, Paweł Stradomski, Tal Marian, Vlado Gali´c, Wenhu Chen, Hisham Husain, Achintya Singhal, Dominik Grewe, François-Xavier Aubet, Shuang Song, Lorenzo Blanco, Leland Rechis, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. URL https://arxiv.org/abs/2507.06261.
- [4] Google DeepMind. Gemini 3.1 pro. Google DeepMind Product Page, February 2026. URL https://deepmind.google/models/gemini/pro/.
- [5] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501. 12948.
- [6] Guanting Dong, Hangyu Mao, Jiajie Zhang, Xiaoxi Li, Kangzhi Zhao, Zhongyuan Wang, Guanting Dong, Licheng Bao, Fuzheng Zhang, and Ji-Rong Wen. Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849, 2025.
- [7] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A comprehensive benchmark for deep research agents. arXiv preprint arXiv:2506.11763, 2025.
- [8] Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, and Siheng Chen. Openseeker: Democratizing frontier search agents by fully open-sourcing training data, 2026. URL https: //arxiv.org/abs/2603.15594.
- [9] Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, et al. Cognitive kernel-pro: A framework for deep research agents and agent foundation models training. arXiv preprint arXiv:2508.00414, 2025.
- [10] Google. Google gemini deep research: Your personal ai research assistant, 2024. URL https://blog.google/products/gemini/google-gemini-deep-research/.
- [11] Janghoon Han, Minseok Kim, Jihyung Yoon, Hyeonju Jo, Kyumin Lee, et al. Deer: A benchmark for evaluating deep research agents on expert report generation. arXiv preprint arXiv:2512.17776, 2025.
- [12] Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, and Yuxiao Dong. Treerl: Llm reinforcement learning with on-policy tree search. arXiv preprint arXiv:2506.11902, 2025.

- [13] Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, Wei Yuan, Wenjin Deng, Xiaojian Yuan, Xiaoyun Zhang, Xiangyu Liu, Xikai Liu, Yanming Xu, Yicheng Cao, Yifei Zhang, Yongyao Wang, Yubo Shu, Yurong Zhang, Yuxiang Zhang, Zheng Gong, Zhichao Chang, Binyan Li, Dan Ma, Furong Jia, Hongyuan Wang, Jiayu Liu, Jing Bai, Junlan Liu, Manjiao Liu, Na Wang, Qiuping Wu, Qinxin Du, Shiwei Li, Wen Sun, Yifeng Gong, Yonglin Chen, Yuling Zhao, Yuxuan Lin, Ziqi Ren, Zixuan Wang, Aihu Zhang, Brian Li, Buyun Ma, Kang An, Li Xie, Mingliang Li, Pan Li, Xi Chen, Xiaojia Liu, Yuchu Luo, Yuan Song, Yuanhao Ding, Yuanwei Liang, Zexi Li, Zhaoning Zhang, Zixin Zhang, Binxing Jiao, Daxin Jiang, Jiansheng Chen, Jing Li, Xiangyu Zhang, Yibo Zhu, et al. Step-deepresearch technical report. arXiv preprint arXiv:2512.20491, 2025.
- [14] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [15] Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, et al. Websailor-v2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning. arXiv preprint arXiv:2509.13305, 2025.
- [16] Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025. URL https://arxiv.org/abs/2507.02592.
- [17] Ming Li et al. Evaluating deep research agents via academic survey generation. OpenReview,

2026. URL https://openreview.net/forum?id=zvL42fmtbG.

- [18] Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun Xie, Fei Huang, Jun Zhang, and Jingren Zhou. Webweaver: Structuring web-scale evidence with dynamic outlines for open-ended deep research. arXiv preprint arXiv:2509.13312, 2025. URL https://arxiv.org/abs/2509.13312.
- [19] Jiaqi Liang et al. A survey on reasoning agentic retrieval-augmented generation. ACL Findings,

2025. URL https://aclanthology.org/2025.findings-ijcnlp.122.pdf.

- [20] Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, and Yuxiao Dong. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn rl. arXiv preprint arXiv:2509.10446, 2025.
- [21] Grégoire Mialon, Roberto Dessì, Maria Lomeli, et al. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2311.12983.
- [22] OpenAI. Introducing deep research. OpenAI Blog, 2025. URL https://openai.com/index/ introducing-deep-research/.
- [23] Yujia Qin, Shengding Liang, Yining Ye, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023. URL https: //arxiv.org/abs/2307.16789.
- [24] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 53428–53451, 2023. doi: 10.48550/arXiv.2305.18290. URL https://arxiv.org/abs/ 2305.18290.
- [25] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.
- [26] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [27] Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. arXiv preprint arXiv:2511.07685, 2025.
- [28] A Singh et al. Agentic retrieval-augmented generation: A survey on agentic rag. arXiv preprint arXiv:2501.09136, 2025. URL https://arxiv.org/abs/2501.09136.
- [29] Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webshaper: Agentically data synthesizing via information-seeking formalization, 2025. URL https://arxiv.org/abs/2507.15061.
- [30] Alibaba NLP Team. Webweaver: Dual-agent framework for open-ended deep research. arXiv preprint arXiv:2509.13312, 2025. URL https://arxiv.org/abs/2509.13312.
- [31] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
- [32] MiroMind Team. Mirothinker-1.7 and h1: Towards heavy-duty reasoning with open-source research agents. arXiv preprint arXiv:2603.15726, 2026. URL https://arxiv.org/abs/ 2603.15726.
- [33] MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, et al. Mirothinker: Pushing the performance boundaries of open-source research agents via model, context, and interactive scaling. arXiv preprint arXiv:2511.11793, 2025.
- [34] Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.
- [35] Haozhe Wang, Haoran Que, Qixin Xu, Minghao Liu, Wangchunshu Zhou, Jiazhan Zhang, Jian Lou, and Roy Ka-Wei Lee. Reverse-engineered reasoning for open-ended generation. arXiv preprint arXiv:2509.06160, 2025.
- [36] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding selfevolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.
- [37] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 24824–24837, 2022.
- [38] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.
- [39] Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, Wenhao Huang, Yang Wang, and Ke Wang. Widesearch: Benchmarking agentic broad info-seeking. arXiv preprint arXiv:2508.07999, 2025.
- [40] Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, et al. Webdancer: Towards autonomous information seeking agency. arXiv preprint arXiv:2505.22648, 2025.
- [41] Yuhao Wu, Yushi Bai, Zhiqiang Hu, Juanzi Li, and Roy Ka-Wei Lee. Superwriter: Reflectiondriven long-form generation with large language models. arXiv preprint arXiv:2506.04180, 2025.

- [42] Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, and Fei Huang. Writingbench: A comprehensive benchmark for generative writing. arXiv preprint arXiv:2503.05244, 2025.
- [43] Tian Xue et al. Online-mind2web: A benchmark for evaluating web agents in online environments. arXiv preprint arXiv:2504.01382, 2025. URL https://arxiv.org/abs/2504. 01382.
- [44] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [45] Chen Yang et al. Nanbeige4.1-3b: A small general model that reasons, aligns, and acts. arXiv preprint arXiv:2602.13367, 2026. URL https://arxiv.org/abs/2602.13367.
- [46] Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein. Re3: Generating longer stories with recursive reprompting and revision. arXiv preprint arXiv:2210.06774, 2022.
- [47] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.
- [48] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id= WE_vluYUL-X.
- [49] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [50] Z.ai. Glm-4.6: Advanced agentic, reasoning and coding capabilities, 2025. URL https: //z.ai/blog/glm-4.6.
- [51] Dingling Zhang, He Zhu, Jincheng Ren, Kangqi Song, Xinran Zhou, Boyu Feng, Shudong Liu, Jiabin Luo, Weihao Xie, Zhaohui Wang, Tianrui Qin, King Zhu, Yuqing Wang, Qianben Chen, Yuchen Eleanor Jiang, Wei Wang, Jiaheng Liu, and Wangchunshu Zhou. How far are we from genuinely useful deep research agents? arXiv preprint arXiv:2512.01948, 2025.
- [52] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [53] Xuhui Zheng, Kang An, Ziliang Wang, Yuhang Wang, and Yichao Wu. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21816–21841, 2025.
- [54] Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, et al. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. arXiv preprint arXiv:2504.19314, 2025.

