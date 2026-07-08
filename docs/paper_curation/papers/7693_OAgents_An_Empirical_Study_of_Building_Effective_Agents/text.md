arXiv:2506.15741v2[cs.AI]23Jun2025

# OAgents: An Empirical Study of Building Effective Agents

OPPO AI Agent Team

## Abstract

Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI.

Date: June 24, 2025 Correspondence: Wangchunshu Zhou at zhouwangchunshu@oppo.com, Jiaheng Liu at liujiaheng@nju.edu.cn Code: https://github.com/OPPO-PersonalAI/OAgents

## 1 Introduction

In recent years, language agents [11, 19, 22, 29, 31, 38–40] have received significant attention due to their potential in resolving general, complex tasks that traditionally required human intervention. However, despite the surge in the number of research works and open-sourced agent frameworks, current practices in Agentic AI research are far from being rigorous and scientific. Specifically, the current landscape of agent research suffers from a lack of standardized designs and implementation details. Critical components such as planning [12, 21, 35], memory [20, 32, 36, 38], and tool use [17, 27] vary widely across different papers and frameworks, making it difficult to attribute performance improvements to specific innovations. Compounding this issue, reported results are often hard to reproduce due to inconsistent evaluation settings or undisclosed framework configurations [2, 9]. This fragmentation undermines the scientific rigor of the field, as findings cannot be reliably compared or built upon.

Take the widely researched GAIA Benchmark [14] as an example. Despite the organizers provide a public leaderboard with evaluation code and a number of papers and projects being open-sourced, it is still very hard, if not impossible, for other researchers to reproduce their results because of a number of inconspicuous factors are not standardized, including the implementation details of tools and prompts, as well as details

Planning

#### Ensembled Tools

Subtask Decompose

Parallelize

Multi-Source Retrieval

|Complex task|
|---|

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[(𝑆𝑇1,𝑆𝑇3),(𝑆𝑇2,𝑆𝑇3)]

[Figure 7]

Dynamic Plan Revise

Multimodal Toolkit

|𝑆𝑇1|
|---|

[Figure 8]

Dependency

[Figure 9]

Search Agent

[Figure 10]

|𝑆𝑇3|
|---|

[Figure 11]

Query Optimization

Analysis

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|𝑆𝑇2|
|---|

Adaptive Browsing

[Figure 18]

Text

|𝑆𝑇1|
|---|

|𝑆𝑇2|
|---|

|𝑆𝑇3|
|---|

Document Parsing

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Voice

[Figure 23]

###### Plan Tips Design

[Figure 24]

[Figure 25]

|𝑆𝑇2|
|---|

𝑆𝑇1

[Figure 26]

[Figure 27]

[Figure 28]

Image

[Figure 29]

|[Figure 30]|
|---|

Previous

[Figure 31]

Video

[Figure 32]

[Figure 33]

Steps

Summarize Previous Log

[Figure 34]

|[Figure 35]|
|---|

Any-to-Text

[Figure 36]

Step Trigger

[Figure 37]

Memory

[Figure 38]

[Figure 39]

Memory Summarization

[Figure 40]

Multi-Nodes

[Figure 41]

###### [summary]

[Figure 42]

highlight 1: ... ... highlight n: ...

[Figure 43]

[Figure 44]

Execution log of the current step

[Figure 45]

Diversity Enhancement

Search Algorithms

### OAgents

Memory Module

[Figure 46]

[Figure 47]

[Figure 48]

Most Similar

Memory Retrieval

List

[Figure 49]

Verifier

Temperature

[Figure 50]

[Figure 51]

Score Voting

Current

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|

| |
|---|

[Figure 52]

Step

Mix of Agents

[Figure 53]

Historical Steps

[Figure 54]

Long-Term

Long-Term Memory

[Figure 55]

Reflection

Current Memory

[Figure 56]

[Figure 57]

| |
|---|

prompt

Memory

Execution log of

Progress Reword Model

[Figure 58]

Test-time Scaling

[Figure 59]

Current Step Historical Steps

| |
|---|

| |
|---|

| |
|---|

the current step

Figure 1 The key components of the OAgents framework, including planning, memory, tools, and test-time scaling.

in the evaluation protocol such as how many runs are performed, how errors and failures are handled, and how different results are ensembles or aggregated. These factors often lead to a large impact on the overall performance, sometimes the impact is even larger than some new architecture innovations in new research papers. However, they are generally not mentioned in the technical reports of different agent frameworks and are not even included in their open-sourced codebases. For example, some previous work conducted multiple runs and merged the results, but reported the results as “pass@1”. Moreover, the engineering design and details in different agent research papers and codebases are so large that it makes it impossible to conduct apples-to-apples comparisons on specific technical designs. This makes it very hard for the research community on agentic AI to properly conduct scientific research instead of digging into tricks on engineering details and evaluation protocols. As a result, despite a lot of agent research papers being released and the numbers on public benchmarks keeping increasing, the best practices on building effective agents are still very obscure. In this work, to promote truly scientific research on agentic AI and provide researchers with a clear understanding of key factors in building effective agents, we conduct a systematic empirical study on GAIA benchmark to sort out the core design choices in current agent research, analyze their impact on performance, and report practical tips for improving experimental stability and reproducibility. Specifically, we: (1) carefully implement and compare different designs on key agent components including planning, tool use, memory, test-time scaling strategies, etc., (2) systematically investigate the impacts of different LLM backbones and their combinations; (3) thoroughly analyze different practices for evaluation and provide a more robust evaluation protocol.

Based on the empirical study, we implement and release OAgents, a language agent framework that achieves state-of-the-art performance among open-sourced agent frameworks on GAIA benchmark and BrowseComp. More importantly, OAgents supports modularized integration of almost all critical designs and features in critical components of language agents, including: (1) different agentic planning mechanisms including static and dynamic workflow designs; (2) a complete tool box including web search with different search sources, browsing tools and web crawlers, parsing tools compatible with more document types. (3) different design of the agentic memory module; (4) test-time scaling strategies for agents including different search algorithms and reflection/self-refine mechanisms. Hopefully, OAgents will facilitate scientific research on language agents by promoting apple-to-apple comparisons and standardizing evaluation protocols.

In summary, our main contributions are as follows.

- 1. We present a comprehensive agent framework - OAgents. OAgents encompass periodically revised plan generation, fine-grained task decomposition & simultaneous execution, optimization of multi-source web browsing, enhanced document parsing, and adaptive memory mechanisms that collectively enhance

- performance across various tasks, ranking 1st among open-source agent frameworks on GAIA benchmark.
- 2. We conduct a systematic empirical study and performance analyzes based on the OAgents framework, offering principles to decompose, analyze and optimize agent designs, uncovering optimal architectural choices and key factors influencing experimental stability.
- 3. We introduce practical techniques for reducing experimental variance, including optimization of inference parameters and majority voting strategies, enabling a more reliable and consistent evaluation of agent performance.

## 2 Related Work

Agent Pipeline. Agents typically undergo iterative planning and execution to accomplish complex tasks, motivating a large amount of research focusing on pipeline design of agents. Yao et al. [35] integrated reasoning and action to simulate the human task execution process, mitigating the issues of hallucination and error propagation, thereby yielding more reliable and interpretable results. Significant-Gravitas [22] is a versatile agent framework supporting task decomposition and tool invocation. Moreover, an integrated memory management mechanism endows the Large Language Model (LLM) with long-term and short-term memory capabilities. Shinn et al. [21] proposed a training-free approach to learn from failure and trial in through linguistic feedback. By recording the observations generated from environment executions, the reflective texts are integrated into the memory, which further guides the planning and execution of subsequent steps. Liu et al. [12] proposed a dual-component memory design, which employs a “Scratchpad” mechanism as short-term memory and maintain a long-term memory repository, significantly enhances the retention and continuity of conversational context. Zhou et al. [37] incorporated the Monte Carlo Tree Search (MCTS) method from reinforcement learning to achieve more flexible and adaptive problem-solving, by conducting searches within the combinatorial space of possible reasoning and action steps.

Multi-Agent Systems. A single agent may encounter performance bottlenecks. Better performance could be achieved by introducing multiple specialized agents. In a multi-agent system, agents typically work as a team or society. Guo et al. [6] propose Criticize-Reflect, a hierarchically-organized team within a leader agent and several worker agents. Their study demonstrates that agent teams with a leader are superior than those without a leader. DyLAN [13] creates dynamic agent teams that adapt based on past performance, with top contributors reserved for future tasks. The authors demonstrate that re-evaluating and ranking agent contributions would benefit arithmetic and reasoning tasks. AgentVerse [4] is a multi-agent architecture that enhances reasoning and problem-solving through four key stages: recruitment, collaborative decision making, independent action execution, and evaluation. These stages are repeated iteratively, helping agents reason, collaborate, and act more effectively towards achieving the overall goal. MetaGPT [8] tackles the problem of unproductive chatter in multi-agent architectures by having agents generate structured outputs rather than unstructured chat messages. This approach improves collaboration and focuses the agents’ efforts on achieving the team goal.

Agents for GAIA Benchmark.. GAIA [14] presents real-word questions that necessitate fundamental skills including reasoning, handling multiple modalities, web searching, and using various tools. These skills are similar to the requirements of AGI, thus attracting a large number of studies to challenge GAIA. We separate the agent frameworks for GAIA into closed-sourced [1, 7, 15, 16, 26] and open-sourced ones [3, 5, 9, 10, 19, 24, 30]. Smolagents [19] combines the ReAct [35] and Code Act [27] architectures to build a multi-functional agents hierarchy to perform multiple rounds of interactions and actions in code to accomplish complex tasks. Magentic-One [5] achieves efficient processing of vision-language tasks by decoupling perception [33, 34], planning [23, 25], and execution modules [17, 27]. Trase-Agent [26] proposes task reallocation strategies based on real-time feedback, while TapeAgents [3] employs an asynchronous communication framework to enhance system resilience. AutoAgent [24] enables intelligent task execution and personalized agent creation without coding through the core components such as natural language-driven multi-agent coordination, customizable workflows, and self-managing file systems. Hybrid architecture exploration is exemplified by h2oGPTe-Agent [7], which transfers single agent optimization techniques to multi-agent scenarios. Owl [9] proposes two variants: One is a horizonal architecture named Roleplaying where a user agent asks the

questions and assistant agent gives the solutions. The other is a decentralized framework consists of a planner, a coordinator, and specialized workers. Alita [18] is a concurrent work that achieves excellent results by implementing a self-evolving MCP Box, however the authors did not disclose the algorithm and implementation details of the MCP Box, which is inconsistent with the motivation of our paper.

## 3 Building Effective Agents

Table 1 Performance of various agent frameworks on the GAIA benchmark.

Framework Model Family Avg. Level 1 Level 2 Level 3

Agentic Model

Search-o1-32B - 39.8 53.8 34.6 16.7 WebThinker-32B-RL - 48.5 56.4 50.0 16.7

Closed-source Agent Frameworks

Langfun Agent Claude-3-7 etc. 71.52 83.02 68.60 57.69 TraseAgent Claude etc. 70.30 83.02 69.77 46.15 Deep Research Unknown 67.36 74.29 69.06 47.60 h2oGPTe Claude-3.5 63.64 67.92 67.44 42.31 Desearch GPT-4o 56.97 71.70 58.14 23.08

Open-source Agent Frameworks

OWL–Workforce Claude-3-7 etc. 69.09 84.91 67.44 42.31 OWL–Roleplaying 4o & o3-mini etc. 58.18 81.14 54.65 23.08 TapeAgents Claude-3-7 etc. 55.76 71.70 53.49 30.77 AutoAgent Claude-3-5 etc. 55.15 71.70 53.40 26.92 Open Deep Research OpenAI o1 55.15 67.92 53.49 34.62 Smolagents Openai o1 etc. 49.70 54.72 53.49 26.92 Magnetic-1 OpenAI o1 etc. 46.06 56.60 46.51 23.08 FRIDAY GPT-4 turbo 34.55 45.28 34.88 11.54

OAgents Claude-3-7 etc. 66.67 77.36 66.28 46.15 OAgents-Pass@3 Claude-3-7 etc. 73.93 83.02 74.42 53.85

We present a dual-axis analytical paradigm for architecting cognitive agents in open-world environments, focusing on two orthogonal evaluation dimensions: factual acquisition capacity (FAC) and logical reasoning fidelity (LRF). The FAC axis quantifies an agent’s proficiency in assimilating and updating domain-specific knowledge from dynamic information streams, while the LRF axis measures its capability to maintain rigorous causal relationships and deduction chains during complex problem-solving. Through systematic examination of these complementary dimensions, we establish methodological guidelines for 1) Enhancing environmental perception through adaptive knowledge integration and 2) Ensuring decision-making robustness via verifiable inference processes. This bifocal approach addresses the fundamental challenges of balancing empirical learning with formal reasoning in autonomous artificial systems operating under partial observability.

Factual Acquisition Capacity. FAC quantifies an agent’s ability to retrieve, validate, and integrate external knowledge, fundamentally governed by the tools component, which include:

- • Tool Heterogeneity: Diversity of integrated resources (e.g., search APIs, vision and audio modules) defining accessible knowledge domains.
- • Orchestration Scalability: Architectural capacity to manage concurrent tool utilization and cross-modal data fusion.

Empirical boundaries emerge directly from toolset limitations, establishing hard constraints on factual knowledge acquisition.

Logical Reasoning Fidelity. The LRF framework establishes formal foundations for stable and coherent decision-making through synergistic integration of three constitutive elements: Plan, Memory, and Test-Time Scaling. This triadic architecture manifests distinct operational principles per component:

- • Plan: Maintains cognitive consistency through temporal synchronization between algorithmic planning strategies and memory-encoded experiential patterns.
- • Memory: Ensures behavioral coherence through persistent state representations that anchor planning operations across decision episodes.
- • Test-Time Scaling: Facilitates adaptive resilience by leveraging real-time performance diagnostics to dynamically recalibrate operational parameters.

##### 3.1 Factual Acquisition Capacity (FAC)

Factual acquisition competence enables agents to systematically gather, verify, and integrate external knowledge via diverse tools. This capacity is fundamentally bounded by two critical operational vectors: multimodal tool interoperability and search tool efficacy, which jointly define the epistemic frontiers of agent-environment interactions.

We focus on quantifying current capability ceilings through two investigative lenses:

- • Multimodal tool constraints: Characterizing temporal alignment errors and modality fusion bottlenecks in cross-domain information synthesis.
- • Search tool limitations: Evaluating knowledge coverage gaps imposed by Search API constraints, index freshness thresholds, and semantic disambiguation failures in web-scale data retrieval.

###### 3.1.1 Multimodal Toolkit

To address the limitations in contextual understanding faced by current agent systems, a multimodal toolkit is employed that integrates capabilities for processing text, speech, images, and video. Unlike traditional frameworks that rely solely on unimodal conversion to transform non-textual content into textual descriptions, this approach enables synchronized and cross-modal semantic parsing:

Response =A(xtext,Timage(I),Tvideo(V )) (1)

where A is the agent function, xtextis the textual input, and Timage,Tvideo are tool functions that extract features from images I and videos V , respectively. This capability enhances the agent’s ability to acquire and interpret factual information in complex, real-world scenarios through direct interaction with multimodal inputs.

###### 3.1.2 Search Agent Framework

Web search enables LLM-agents to address real-time information needs and expand epistemic boundaries. We optimize three subsystems: (i) Multi-source retrieval, (ii) Query refinement, and (iii) Minimalist browsing architecture via the Search Agent framework.

Multi-Source Search. To mitigate single-source bias, we integrate commercial APIs (Google, Bing) and archival systems (Wayback Machine CDX API). Source selection is state-aware, driven by query temporal constraints (historical/real-time) and domain requirements (academic/commercial). Historical retrieval uses structured 〈url,date〉 queries to Internet Archive’s temporal index.

Query Optimization Pipeline. Closed-loop refinement combines semantic calibration (Reflect) with morphological expansion (Expand):

Qopt = Reflect(Qinit,Mtask)→ Expand(Qopt,Lterm) (2)

where Reflect(⋅) resolves semantic ambiguities by calibrating specificity through prompt-based constraints and logical simplification guided by predefined rewrite rules, while Expand(⋅) generates morphological

and semantic variants via stemming or lemmatization transformations, as well as domain-specific synonym expansion (e.g., COVID-19 → SARS-CoV-2).

Minimalist Browsing. Conventional frameworks suffer from tool overload. We reduce complexity to three atomic functions: Search (query): Find relevant web pages to the query from search engines. Visit (url): Navigate to the webpage corresponding to url and Read (url, mode): Extract contens in a page and present observations.

##### 3.2 Logical Reasoning Fidelity (LRF)

In this section, we investigate three key strategies to improve logical reasoning in agents: dynamic plan generation and task decomposition, memory-augmented knowledge system, and test-time scaling for exploration optimization. These approaches address challenges in logical consistency, environmental adaptability, and efficiency-accuracy trade-offs.

###### 3.2.1 Dynamic Plan Generation

Strategic Plan Review. To enhance agents’ complex task management, planning modules generate high-level plans P=(s1,s2,...,sn) that decompose tasks into executable steps, improving reasoning efficiency. Execution follows the ReAct framework, alternating reasoning rt and actions at. For adaptability in dynamic environments, plans are revised every N steps using recent observations {ot−N+1,...,ot}): P

= revise(P,{ot−N+1,...,ot}) This iterative planning-execution loop sustains goal-directed behavior and strengthens long-term decision-making. Subtask Decomposition. To enhance systematic reasoning in planning modules, we propose hierarchical

′

taskand constructsdecomposition:a dependencyThe agentgraphbreaksD=(Sdown,εthe), wheremain goaledgesG eintoij ∈ interdependentε encode precedencesubtasksconstraints.S=(s1,s2At,...,seachn) reasoning step t, dynamic scheduling selects executable subsets St ⊆S satisfying all dependencies in D. Intermediate outputs from completed subtasks are formalized as structured knowledge representations i ∈K, which are cross-validated against global constraints C(G). A validity function ensures alignment with the overarching goal:

truefalse,, ifotherwiseκi ⊧ C(G)

valid(κi)={

(3)

This mechanism enables error detection through consistency checks, strengthens long-horizon reasoning, and improves decision-making resilience in complex environments.

Plan Tips. To augment planning capabilities, we propose integrating experiential knowledge from historical execution trajectories τ{(st,at,rt)}Tt . Analysis of past attempts reveals common bottlenecks and failure patterns, which are distilled into heuristic guidelines H={h1,h2,...,hm} as soft constraints for the planner. These domain-specific heuristics influence action selection during planning through an augmented policy:

πθ (at ∣ st,H)= softmax(Q(st,at)+ β ⋅ fH (st,at)) (4)

where fH(⋅) encodes the influence of heuristics and β controls their weight. This integration enables preemptive avoidance of known pitfalls, enhances robustness in plan generation, and improves adaptability to dynamic environments by embedding empirical knowledge into decision-making.

###### 3.2.2 Memory-augmented Knowledge System

The hierarchical memory module enhances agent cognition through four components: Current Memory, Memory Summarization, Vectorized Retrieval, and Long-Term Memory, each addressing distinct aspects of perception and decision-making.

Current Memory. Serves as a short-term buffer storing temporally ordered task-specific information Mc = {(st,at)tt−τ}, for real-time processing and on-the-fly decisions.

Memory Summarization. This component transforms raw experience sequences into structured semantic units using topic modeling and sequence-to-sequence generation:

zi = Summarize({(st,at,rt)t

t }) (5)

i+1

where zi denotes a memory summarization. By extracting high-salience knowledge, it facilitates efficient downstream processing.

Vectorized Memory Retrieval. This component retrieves beneficial historical memories via vector similarity. Specifically, the execution log of each step is embedded into a shared latent space E: E(x)= Encode(x). Contextually relevant memories are then retrieved based on vector similarity:

Mretrieved = arg max m∈M

sim(E(q),E(m)) (6)

Long-Term Memory. Addresses challenges in lengthy reasoning chains and contextual redundancy during task execution by integrating historical insights. Updates occur through fusion of current memory with existing long-term knowledge, enabling continuous optimization recommendations for task execution.

These components form a structured framework that organizes, stores, and retrieves knowledge at multiple abstraction levels, helping the agent perform effectively in complex environments.

- 3.2.3 Test-Time Scaling

The Test-Time Scaling (TTS) module enhances agent capabilities through three mechanisms: diversity enhancement, optimization, and reward modeling.

Diversity Enhancement. A mixture-of-agents sampling strategy combines multiple LLM policies πθ

i

with weights αi:

at ∼

K

∑

i=1

αi ⋅ πθ

i

(⋅∣ st) (7) This exploits inter-model diversity to generate broader solution spaces and improve outcome quality.

Optimization.that assess taskTheprogression,TTS moduleerrorguideshandling,agentandreasoningefficiencythroughat eachprocess-basedstep. Rewardsrewardare functionstemporallyrtaggregated= R(st,at) as:

Rtotal =

T

∑

t=1

γtrt (8) providing continuous feedback to refine reasoning trajectories and improve solution accuracy. Reward Modeling. The TTS module enables real-time reflection for adaptive problem-solving through::

ct = Reflect({(sτ,aτ)}tτ=1) (9)

where ct captures corrective insights from past steps, improving error detection and on-the-fly adjustments to enhance overall performance..

- 4 Empirical Study

##### 4.1 Experimental Setup

Dataset.. GAIA[14] presents real-world challenges that demand essential skills like reasoning, handling multi-modal inputs, web browsing, and overall proficiency in tool-calling. True answers are provided for each question, and the correctness of the model response is evaluated with exact match. Due to the instability and randomness of networked experiments, we allow the model to re-answer a question when the answer given by the model is empty or contains “Unable to determine” specified in the prompt. However, recalling incorrect answers is illegal.

Evaluation Protocol.. We follow the evaluation protocol of the GAIAbenchmark [14], which is based on exact match accuracy. The primary metric used is Pass@N, which measures the probability that at least one correct solution is found among N independent model attempts. This metric is widely adopted in tasks such as code generation, where the key evaluation criterion is whether the model can produce a valid solution at least once. In our experiments, unless otherwise stated, we report the average Pass@1 score, reflecting the model’s performance in generating a correct answer across all questions within a single evaluation run.

Implementation Details.. In both the FAC Evaluations and LRF Evaluations, the baselines are implemented with the integrated multi-modal toolkit in OAgents. For FAC evaluations, we adopt Claude-3-7-Sonnet to activate the code agent and we add plan tips into the system prompts. The web pages are fetched from all 5 types of search engines. While we only set Google as the only search source and disable plan tips for LRF evaluations. Unless otherwise specified, all models employed in the agent are based on GPT-4.1 to ensure consistency in model architecture and capabilities across experiments.

Baseline Methods.. The selected baselines are categorized into two main groups. Open-source systems include FRIDAY [30] , Magnetic-1 [5], TapeAgent [3], AutoAgent [24], Open Deep Research [10], and OWL [9]. Closed-source frameworks comprise Langfun Agent [16], Trase Agent [26], Deep Research [15], h2oGPTe [7], and Desearch [1]. These baselines collectively capture a broad spectrum of current advancements in both open and proprietary multi-agent systems, offering a solid foundation for benchmarking the effectiveness and performance of our proposed OAgents framework.

##### 4.2 Main Results

The results in Table 1 reveal several key insights into the performance landscape across various agent frameworks on the GAIA benchmark. Notably, our method (OAgents-Pass@3) achieves the highest overall average score of 73.93%, outperforming all other frameworks, including both closed-source and open-source systems. This highlights the robustness and effectiveness of our agent design.

In terms of Level 1 task performance, our method reaches 83.02%, tying with the best-performing frameworks and establishing a new standard for esay task handling. This superior performance reflects the reliability and consistency of our low-level agents and the underlying System Utilities.When compared with leading closed-source agents like Langfun Agent (71.52%) and TraseAgent (70.30%), our method shows a clear edge in both average and Level 2 accuracy. Finally, in the open-source domain, OAgents-Pass@3 demonstrates a significant margin over the best alternative, OWL-Roleplaying (58.18%), reaffirming our method’s leading position among publicly available systems. Overall, these results validate our approach as a state-of-the-art solution for generalist agent tasks.

We replicate Open Deep Research [10] and note the results as “Smolagents”, and the performance of the replication shows a significant degradation. This indicates that the reproducibility of the current agencies framework is poor.

###### 4.2.1 FAC Evaluations

Multimodal Toolkit. We have refined text extraction tools with format-specific strategies tailored for various document types (pdf, xlsx, and etc.). For audio inputs, we employ the whisper-1 speech-to-text model to generate accurate transcriptions. For video content, we implement a pipeline combining keyframe extraction with vision-language models for temporal and contextual analysis. Importantly, we incorporate a multi-source image understanding module, which leverages multiple vision language models source to understand visual features. Evaluated on the GAIA dataset (Table 2), our toolkit achieves a cross-modal task accuracy of 74.07%, outperforming the baseline system’s 48.15%. Notably, in audio question-answering subtasks, temporal reasoning accuracy improves from 0% to 100% (3/3). These results demonstrate that a deeply optimized multimodal architecture can effectively bridge modality gaps in intelligent agent systems.

Search Agent. Our empirical analysis quantitatively evaluates how search infrastructure design affects the performance of GAIA. As shown in Table 3, Jina reader outperforms raw HTML parsing by 9.3% in Level 2

Table 2 Performance (%) of OAgents before and after integrating multimodal toolkit.

GAIA multimodal tasks Sum Audio Image Tubular Task number 27 3 10 14

Method

OAgents 48.15 0.00 40.00 64.29 OAgents + Toolkit 74.07 100.00 60.00 78.57

tasks. Its structured text extraction benefits mid-complexity factual acquisition, highlighting preprocessing’s role in enhancing retrieval quality.

- Table 3 Performance comparison of browser methods on GAIA benchmark. All results are obtained using information retrieved from Google Search.

Browser Method

GAIA Average Level 1 Level 2 Level 3

Text web browser 44.20 54.71 43.02 26.92 Raw reader 49.70 64.51 46.51 30.76 Crawler crawl4ai 50.90 67.92 51.16 15.38 Jina reader 51.52 67.92 48.83 26.92

From Table 4, integrating complementary search engines (DuckDuckGo, Baidu, Bing) consistently improves retrieval accuracy, with the largest gain in Level 3 tasks (+7.69%). This indicates that diversifying information sources mitigates individual engine limitations, particularly in complex retrieval scenarios.

- Table 4 OAgents performance of different search source configurations on GAIA. Note that “single-source” refers to Google only. “Multi-source (k = 3)” includes Google, Wikipedia, and DuckDuckGo. “multi-source (k = 5)” further adds Bing and Baidu as additional search sources.

GAIA Average Level 1 Level 2 Level 3

Search Method

Single-source 51.52 67.92 48.83 26.92 Multi-source (k 3) 52.12 67.92 50.00 26.92 Multi-source (k = 5) 55.15 67.92 53.49 34.61

The proposed query optimization strategy, combining reflection and expansion mechanisms, significantly enhances system performance (Table 5). It yields a 7.55% improvement in Level 1 and 2.31% in Level 2, underscoring the effectiveness of refined query formulation in improving search outcomes. Finally, the

Table 5 OAgents performance comparison of query-optimization configurations on GAIA.

GAIA Average Level 1 Level 2 Level 3

Query Optimization

Raw data 55.15 67.92 53.49 34.61 Reflection-Expansion 58.18 75.47 55.80 30.76

minimalist system architecture demonstrates competitive performance, supporting the hypothesis that reduced interface complexity can improve robustness without sacrificing functionality.

OAgents. By integrating an optimized search infrastructure with a multimodal toolkit, and employing the Jina reader with multi-source (k = 5) strategies, our OAgents achieves strong improvement s on the GAIA benchmark across diverse base models. With GPT-4o, OAgents improves the overall score by 8.09%, including a 7.69% gain in Level 3 tasks. Gemini-2.5 shows a 9.09% average improvement, with Level 3 jumping 19.24%, confirming the effectiveness of the multimodal toolkit and refined search agent. Notably, Claude-3-7 gains 20.61%, the highest observed boost, demonstrating the framework’s adaptability to models with varying baseline performance. The integrated design enhances FAC through advanced search and multimodal capabilities, establishing a solid foundation for knowledge-intensive agent systems. These results confirm that FAC improvements significantly elevate intelligent agent performance across architectures.

Table 6 OAgents performance of various base models on GAIA.

GAIA Score Average Level 1 Level 2 Level 3

Model Type

Baseline 36.97 54.72 34.88 7.69

GPT-4o

Advance 45.06 62.26 45.35 15.38 Gap ↑8.09 ↑7.54 ↑10.47 ↑7.69

Baseline 44.20 54.71 43.02 26.92 Advance 55.15 67.92 53.49 34.62

GPT-4.1

Gap ↑10.95 ↑13.21 ↑10.47 ↑7.70

Baseline 49.70 54.72 53.49 26.92 Advance 53.94 67.92 52.33 30.77

OpenAI-o1

Gap ↑4.24 ↑13.20 ↓1.16 ↑3.85

Baseline 38.18 56.60 36.05 7.69

Claude-3-7

Advance 58.79 64.15 61.63 38.46 Gap ↑20.61 ↑7.55 ↑25.58 ↑30.77

Baseline 33.90 45.28 33.72 11.54 Advance 49.70 62.26 50.00 23.08

DeepSeek-R1

Gap ↑15.80 ↑16.98 ↑16.28 ↑11.54

Baseline 49.09 69.81 46.51 15.38 Advance 58.18 73.58 55.81 34.62

Gemini-2.5

Gap ↑9.09 ↑3.77 ↑9.30 ↑19.24

###### 4.2.2 LRF Evaluations

Dynamic Plan Generation. The results in Table 7 show that our planning and workflow design significantly enhance GPT-4.1’s ability to solve complex tasks. Strategic plan review (baseline) improves overall accuracy by 3.64% over the static workflow, confirming that dynamic plan revision supports better adaptability and long-term reasoning. Subtask Decomposition achieves a 2.42% improvement over baseline, demonstrating that breaking down tasks into structured subtasks enhances systematic reasoning, particularly for tasks of moderate complexity. The Plan tips are summarized from analysis of historical error logs and incorporate heuristic knowledge gained from past failures. They contribute to a 14.54% performance improvement, proving that leveraging prior experience helps prevent errors and build more robust plans. This is especially important for high complexity tasks. Together, these components significantly enhance the system’s planning capabilities for complex reasoning.

Memory. The experimental evaluation on the GAIA benchmark highlights the effectiveness of the memory components. From Figure 2, adding memory summarization slightly improved average accuracy from 51.52% to 52.12%. With memory retrieval, performance increased further to 53.33%. The most significant gain came

Table 7 OAgents performance evaluation of plan studies on GAIA. Note that Static workflow refers to a scenario in which all tasks follow the same manually designed workflow.

GAIA Average Level 1 Level 2 Level 3

Model combination

OAgents 51.52 67.92 48.83 26.92 r.p. Static workflow 47.88 62.26 47.67 19.23 + Subtask 53.94 71.70 51.16 26.92 + Plan tips 66.06 79.25 66.28 38.46

from long-term memory, raising the average to 55.76%, while also achieving the competitive results across all difficulty levels.

OAgents

+ Summary

+ Memory Retrieval

+ Long Term Memory

Figure 2 OAgents performance evaluation of various memory methods on GAIA.

Results confirm the benefits of memory components in enhancing agent cognition, future work will explore dynamic component allocation based on task complexity metrics.

Test-Time Scaling. As shown in Table 3, we conduct an ablation study to examine how test-time scaling (TTS) strategies influence the performance of OAgents across different task complexities. Reflection leads to a moderate overall improvement (3.03%), yet its effects vary across task levels. While it enhances performance on Level 1 and Level 2 tasks through iterative reasoning, it unexpectedly degrades results on Level 3 tasks by 6.62%, suggesting potential instability or error accumulation in complex reasoning chains. Best-of-N sampling demonstrates more consistent gains, with performance improving as the sample size increases. BO2 yields modest improvements (1.82%), while BO4 achieves the best overall performance (5.19%), particularly benefiting simpler tasks (Level 1: 9.44%, Level 2: 10.46%). This indicates that answer diversification helps in navigating simpler solution spaces more effectively.

Nonetheless, neither strategy substantially improves performance on Level 3 tasks, underscoring the persistent difficulty in achieving robust multi-step reasoning at scale. These findings reveal that TTS strategies exhibit differential effectiveness depending on task complexity—offering clear benefits for straightforward tasks but requiring further innovation to address advanced reasoning challenges.

##### 4.3 Evaluations on BrowseComp

To validate the search agent’s capabilities, we evaluate OAgents on a more challenging benchmark named BrowseComp [28], where single language models rarely answered correctly or scored. As shown in Table 8, OAgents significantly improved the model’s abilities in web browsing.

Average Level 1 Level 2 Level 3

Figure 3 OAgents performance evaluation of TTS methods on GAIA. Table 8 OAgents performance of search agent on BrowserComp-Subset.

Model BrowserComp-Subset Claude-3-7 4.76% GPT-4.1 7.94% OpenAI-o1 14.29% OAgents - GPT-4.1 22.22% OAgents - Claude-3-7 22.22%

## 5 GAIA Benchmark

The GAIA benchmark has emerged as a prominent evaluation framework for assessing the performance of autonomous agents in real-world scenarios. As the leaderboard for this benchmark continues to grow, it becomes increasingly evident that reported results often vary in terms of evaluation metrics—particularly in the use of different Pass@K criteria. While some methods report Pass@1, others adopt more lenient metrics such as Pass@3 or even Pass@5. This inconsistency complicates fair comparisons across different agent frameworks and limits the transparency of their actual capabilities.

Table 9 Comparison of performance on the GAIA benchmark under different Pass@K metrics. Note that "OWL" stands for the open-source role-playing version.

GAIA Average Level 1 Level 2 Level 3 OAgents Claude-3-7

Method Model Metric

66.67 77.36 66.28 46.15

OWL 4o & o3-mini 53.33 71.70 50.00 26.92 AWorld Claude-3-7 61.81 - - -

Pass@1

OAgents Claude-3-7

73.93 83.02 74.42 53.85

Pass@3

OWL 4o & o3-mini 58.18 81.14 54.65 23.08 AWorld Claude-3-7 Unknown 77.58 88.68 77.91 53.85

To address this issue and ensure alignment with the leaderboard standards, we reimplemented the state-ofthe-art OWL framework to obtain its Pass@1 performance for comparison. Additionally, we evaluated our proposed open-source framework, OAgents, under the Pass@3 setting, as summarized in Table 9. Built upon integrated multi-modal toolkit, multi-source information retrieval, and test-time scaling (TTS) strategies, OAgents demonstrates competitive performance among existing open-source frameworks under the Pass@3

metric. These results highlight the framework’s effectiveness in handling complex reasoning tasks and its strong potential for deployment in real-world applications requiring robust and scalable reasoning capabilities.

## 6 Conclusion

In this work, we conduct a systematic study on GAIA and BrowseComp. We identify key components for effective agents, such as planning, memory, and tool use, and propose a robust evaluation protocol. We release OAgents, an open-source modular agent framework achieves state-of-the-art performance on GAIA (73.93), providing a foundation for future research on agentic agent area.

## 7 Contributions

Core Contributors

- • He Zhu
- • Tianrui Qin

Contributors

- • King Zhu
- • Heyuan Huang
- • Yeyi Guan
- • Jinxiang Xia
- • Yi Yao
- • Hanhao Li
- • Ningning Wang
- • Pai Liu
- • Tianhao Peng
- • Xin Gui
- • Xiaowan Li
- • Yuhui Liu

Organizers

- • Yuchen Eleanor Jiang
- • Jun Wang
- • Changwang Zhang
- • Xiangru Tang
- • Ge Zhang
- • Jian Yang
- • Minghao Liu
- • Xitong Gao

Corresponding Authors

- • Wangchunshu Zhou
- • Jiaheng Liu

## References

- [1] Desearch AI. Desearch, 2024. URL https://desearch.ai/.
- [2] Agent Team at Ant Group. Aworld: A unified agent playground for computer and phone use tasks, 2025. URL https://github.com/inclusionAI/AWorld.
- [3] Dzmitry Bahdanau, Nicolas Gontier, Gabriel Huang, Ehsan Kamalloo, Rafael Pardinas, Alex Piché, Torsten Scholak, Oleh Shliazhko, Jordan Prince Tremblay, Karam Ghanem, Soham Parikh, Mitul Tiwari, and Quaizar Vohra. Tapeagents: a holistic framework for agent development and optimization, 2024. URL https://arxiv. org/abs/2412.08445.
- [4] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv preprint arXiv:2308.10848, 2(4):6, 2023.

- [5] Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, et al. Magentic-one: A generalist multi-agent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

- [6] Xudong Guo, Kaixuan Huang, Jiale Liu, Wenhui Fan, Natalia Vélez, Qingyun Wu, Huazheng Wang, Thomas L. Griffiths, and Mengdi Wang. Embodied LLM agents learn to cooperate in organized teams. In Language Gamification - NeurIPS 2024 Workshop, 2024. URL https://openreview.net/forum?id=VKlrzygQlT.

- [7] H2O.ai. Autonomous agentic ai: execute multi-step workflows autonomously. [Online], 2024. https://h2o.ai/ platform/enterprise-h2ogpte/#AgenticAI.
- [8] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.

- [9] Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Ping Luo, and Guohao Li. Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation, 2025. URL https://github.com/camel-ai/owl.
- [10] LangChain. Open deep research. [Online], 2024. https://github.com/langchain-ai/open_deep_research.
- [11] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023.

- [12] Na Liu, Liangyu Chen, Xiaoyu Tian, Wei Zou, Kaijiang Chen, and Ming Cui. From llm to conversational agent: A memory enhanced architecture with fine-tuning of large language models, 2024. URL https://arxiv.org/abs/ 2401.02777.
- [13] Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization. arXiv preprint arXiv:2310.02170, 2023.

- [14] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.

- [15] OpenAI. deepresearch, 2024. URL https://openai.com/index/introducing-deep-research/.
- [16] Daiyi Peng. Langfun, September 2023. URL https://github.com/google/langfun.
- [17] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, et al. Tool learning with foundation models. ACM Computing Surveys, 57(4):1–40, 2024.

- [18] Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.

- [19] Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/huggingface/smolagents, 2025.

- [20] Dingfeng Shi, Jingyi Cao, Qianben Chen, Weichen Sun, Weizhen Li, Hongxuan Lu, Fangchen Dong, Tianrui Qin, King Zhu, Minghao Yang, Jian Yang, Ge Zhang, Jiaheng Liu, Changwang Zhang, Jun Wang, Yuchen Eleanor Jiang, and Wangchunshu Zhou. Taskcraft: Automated generation of agentic tasks, 2025. URL https://arxiv. org/abs/2506.10055.
- [21] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.
- [22] Significant-Gravitas. Autogpt. [Online], 2023. https://github.com/Significant-Gravitas/AutoGPT.
- [23] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2998–3009, 2023.

- [24] Jiabin Tang, Tianyu Fan, and Chao Huang. Autoagent: A fully-automated and zero-code framework for llm agents. arXiv e-prints, pages arXiv–2502, 2025.

- [25] Jesus Tordesillas and Jonathan P How. Mader: Trajectory planner in multiagent and dynamic environments. IEEE Transactions on Robotics, 38(1):463–476, 2021.

- [26] Trase. Meet trase systems. [Online], 2024. https://www.trasesystems.com/.
- [27] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. In Forty-first International Conference on Machine Learning, 2024.

- [28] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- [29] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

- [30] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456, 2024.

- [31] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, et al. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.

- [32] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

- [33] Dingkang Yang, Kun Yang, Yuzheng Wang, Jing Liu, Zhi Xu, Rongbin Yin, Peng Zhai, and Lihua Zhang. How2comm: Communication-efficient and collaboration-pragmatic multi-agent perception. Advances in Neural Information Processing Systems, 36:25151–25164, 2023.

- [34] Kun Yang, Dingkang Yang, Jingyu Zhang, Hanqi Wang, Peng Sun, and Liang Song. What2comm: Towards communication-efficient collaborative perception via feature decoupling. In Proceedings of the 31st ACM international conference on multimedia, pages 7686–7695, 2023.

- [35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

- [36] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.

- [37] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models, 2024. URL https://arxiv.org/abs/2310.04406.
- [38] Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, and Mrinmaya Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text, 2023. URL https: //arxiv.org/abs/2305.13304.

###### [39] Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Xiangru Tang, Ningyu Zhang, Huajun Chen, Peng Cui, and Mrinmaya Sachan. Agents: An open-source framework for autonomous language agents.

2023. URL https://arxiv.org/abs/2309.07870.

###### [40] Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, Huajun Chen, and Yuchen Eleanor Jiang. Symbolic learning enables self-evolving agents. 2024. URL https://arxiv.org/abs/2406.18532.

# Appendix

## A Details of OAgents

##### A.1 Search Agent

Web search constitutes a foundational capability for LLM-agents to address real-time information needs and extend their epistemic boundaries. We focus on optimizing three critical subsystems: (i) Multi-source retrieval, (ii) Query refinement, and (iii) Adaptive browsing – implemented through the SearchAgent framework.

Multi-Source Search. Contemporary search engines exhibit non-overlapping ranking mechanisms and temporal coverage limitations. To mitigate single-source bias, our implementation integrates:

- • Commercial APIs: Google Custom Search JSON API and Bing Web Search API.
- • Archival Systems: Wayback Machine CDX Server API for historical snapshots.

In a state-aware routing mechanism, source selection is autonomously driven by:

- • Query temporal constraints (historical vs. real-time).
- • Domain-specific coverage requirements (academic vs. commercial).

The historical retrieval tool accepts structured inputs as <url,date> tuples, querying the Internet Archive’s temporal index through:

Listing 1 Example of construct a CDX query to retrieval archive information. def fetch_historical_page(url: str , timestamp: str) -> str:

cdx_query = f"http://web.archive.org/cdx/search/cdx?url={url}&output=json&from={ timestamp}"

Query Optimization Pipeline. The closed-loop query refinement follows: Qopt = Reflect(Qinit,Mtask)→ Expand(Qopt,Lterm) (10)

where Reflect(⋅) resolves semantic ambiguities by calibrating specificity through prompt-based constraints and logical simplification guided by predefined rewrite rules, while Expand(⋅) generates morphological and semantic variants via stemming or lemmatization transformations, as well as domain-specific synonym expansion (e.g., COVID-19 → SARS-CoV-2).

Minimalist browsing architecture. Conventional browser emulation frameworks impose cognitive overhead through excessive tool options. Our streamlined implementation reduces interaction complexity by:

- • Eliminate non-essential operations (e.g., click, scroll, find).
- • Consolidate functionality into three atomic tools: Search, Visit, and Read.

###### A.2 Strategic Plan Review In order to improve an agent’s capability to manage complex tasks, the incorporation of a planning module is

ofbeforecriticalexecution,importance.breakingPlanningdown modulecomplexenablestasks intothe manageableagent to generatesteps anda high-levelimprovingplanreasoningP=(s1,sefficiency.2,...,sn) Execution typically follows the ReAct framework, interleaving reasoning and actions: at each step t, the agent performs either an action at or a reasoning step rt. To ensure adaptability in dynamic environments, the plan P is periodically revised—every N steps—based on new observations ot, updating the sequence of subtasks as P

###### = revise(P,{ot−N+1,...,ot}) This iterative approach supports sustained, goal-directed behavior and enhances the agent’s long-term reasoning and decision-making capabilities.

′

##### A.3 Subtask Decomposition.

Given the role of the planning module in managing complex tasks, we can further consider a hierarchical task decomposition mechanism to enhance systematic reasoning. During planning, the agent decomposes the main goal G into a set of interdependent subtasks S=(s1,s2,...,sn), and constructs a dependency graph D=(S,ε), where edges eij ∈ ε represent precedence constraints between subtasks. This structure enables dynamic scheduling of non-conflicting subtasks at each reasoning step t, formalized as selecting an executable subset St ⊆S such that all dependencies in D are satisfied. A key component is the iterative synthesis of intermediate outputs: results from completed subtasks are formalized as structured knowledge representations i ∈K, and refined through cross-validation against global constraints C(G). This process ensures alignment with the overarching goal and supports error detection and correction via consistency checks:

truefalse,, ifotherwiseκi ⊧ C(G)

valid(κi)={

(11)

Collectively, these mechanisms strengthen the planning module’s capacity for long-horizon reasoning, enabling more effective and resilient decision-making in complex environments.

##### A.4 Plan Tips.

Beyond designing diverse planning strategies, another promising direction lies in enriching the planning process with additional prior knowledge. By analyzing the execution trajectories τ{(st,at,rt)}Tt of past attempts, w, we can identify common bottlenecks, failure points, and suboptimal behaviors encountered by the agent during task realization. These insights can then be distilled into actionable tips or heuristic guidelines H={h1,h2,...,hm}, which are subsequently injected into the planning module as soft constraints or preferences.

Such domain-specific knowledge serves as supplementary guidance during plan generation, influencing the selection of actions and subgoals:

πθ (at ∣ st,H)= softmax(Q(st,at)+ β ⋅ fH (st,at)) (12)

where fH(⋅) encodes the influence of heuristics and β controls their weight. As a result, the planner is better equipped to anticipate potential issues, avoid known pitfalls, and construct more robust strategies for complex problem-solving. This integration of experiential knowledge enhances not only the effectiveness of individual planning steps but also the overall resilience of the agent in dynamic and uncertain environments.

##### A.5 Memory-augmented Knowledge System

The hierarchical memory module is designed to enhance the cognitive capabilities of intelligent agents through four complementary components: Current memory, Memory summarization, Memory retrieval, and Long-term memory. Each component contributes uniquely to different aspects of perception, reasoning, and decision-making.

###### A.5.1 Current Memory.

As a fundamental default component of the agent, current memory acts as a short-term buffer to capture fine-grained, task-specific information in real time. This buffer maintains recent observations and actions in a temporal sequence Mc ={(st,at)tt−τ}, enabling the agent to process dynamic environmental inputs with high fidelity and support on-the-fly decision-making.

###### A.5.2 Memory Summarization.

This component transforms raw experience sequences into structured semantic units using topic modeling and sequence-to-sequence generation:

t }) (13)

zi = Summarize({(st,at,rt)t

i+1

where zi denotes a memory summarization. By extracting high-salience knowledge, it facilitates efficient downstream processing.

###### A.5.3 Vectorized Memory Retrieval.

This component retrieves beneficial historical memories via vector similarity. Specifically, the execution log of each step is embedded into a shared latent space E: E(x)= Encode(x). Contextually relevant memories are then retrieved based on vector similarity:

Mretrieved = arg max m∈M

sim(E(q),E(m)) (14)

###### A.5.4 Long-Term Memory.

This component is designed to address the challenges of lengthy reasoning chains and redundant contextual information when agents perform tasks by integrating key insights from historical reasoning processes and generating subsequent optimization recommendations. Specifically, the long-term memory component achieves updates by fusing current memory with existing long-term memory, continuously guiding agents in task execution.

##### A.6 Test-Time Scaling

Agent capabilities can be significantly enhanced through the integration of test-time scaling mechanisms, which dynamically refine decision-making, improve adaptability, and promote more robust exploration. TestTime-Scaling (TTS) module contributes to this enhancement by addressing three core aspects: diversity, optimization, and reward modeling.

###### A.6.1 Diversity Enhancement.

Enhancing the diversity of reasoning paths is crucial for improving agent performance in complex tasks. By leveraging a mixture-of-agents sampling strategy:

K

at ∼

αi ⋅ πθ

(⋅∣ st) (15)

∑

i

i=1

where αi denotes the weight of each agent policy πθ

, the TTS module exploits differences in capability profiles across multiple LLMs, generating a broader range of potential solutions and increasing the likelihood of identifying high-quality outcomes.

i

###### A.6.2 Optimization.

To guide agents toward more effective reasoning trajectories, the TTS module introduces process based reward functions rt = R(st,at), which evaluate each step along the generation path. These multi dimensional assessments cover key aspects such as task progression, error handling, and efficiency. The rewards are aggregated over time:

T

Rtotal =

∑

γtrt (16)

t=1

providing fine-grained feedback that enables iterative refinement and convergence toward more accurate final responses.

###### A.6.3 Reward Modeling.

Real-time reflection and self-correction are essential for adaptive problem-solving. The TTS module incorporates a reflection mechanism that evaluates intermediate steps during exploration:

ct = Reflect({(sτ,aτ)}tτ=1) (17)

where ct represents corrective insights fed back into subsequent reasoning stages. This iterative refinement enhances the agent’s ability to detect and rectify errors on-the-fly, leading to improved overall performance.

## B Prompts

In this section, we post the prompts of essential modules in OAgents including planning §(B.1), search agent §(B.2), memory §(B.3), and test-time scaling §(B.4).

##### B.1 Planning Prompts Plan Tips Prompt

You are a world expert at making efficient plans to solve any task using a set of carefully crafted tools. Now for the given task, develop a step-by-step high-level plan taking into account the above inputs and list of facts. This plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer. Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS. After writing the final step of the plan, write the ’<end_plan>’ tag and stop there.

Here is your task: {task} You can leverage these tools: % - for tool in tools.values() %

- - {tool.name}: {tool.description} Takes inputs: {tool.inputs} Returns an output of type: {tool.output_type} %- endfor %

%- if managed_agents and managed_agents.values() | list % You can also give tasks to team members. Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is ’task’, a long string explaining your task. Given that this team member is a real human, you should be very verbose in your task. Here is a list of the team members that you can call: %- for agent in managed_agents.values() %

- - {agent.name}: {agent.description} %- endfor % %- else % %- endif %

List of facts that you know: {answer_facts} Please strictly follow the suggestions below: {plan_tips} Now begin! Write your plan below.

###### Subtasks Prompt

You are a world expert at making efficient plans to solve any task using a set of carefully crafted tools.

For the given task, you will analyze whether it can be effectively broken down into independent subtasks:

- • If the task is more cohesive and cannot be efficiently divided (or would create artificial, highly dependent subtasks), you will treat it as a single task with one comprehensive plan.
- • ELSE if the task can be logically divided into subtasks, you will break it down and provide a separate plan for each subtask.
- • If the task is more cohesive and cannot be efficiently divided (or would create artificial, highly dependent subtasks), you will treat it as a single task with one comprehensive plan.
- • ELSE if the task can be logically divided into subtasks, you will break it down and provide a separate plan for each subtask.
- • Only give the plan, without any explanations or other text.

IF given task cannot be effectively divided, you will:

- • Provide just one plan without the subtask structure.
- • You will format your response as follows:

- 1. [First step]
- 2. [Second step]

...

ELSE if given task can be broken down into subtasks, you will:

- • Identify all subtasks needed to complete the overall task.
- • Provide a single PARALLEL-LIST that ONLY contains subtasks index joined by comma that can start immediately with NO dependencies on other subtasks.
- • Provide a complete step-by-step plan for each subtask.
- • Clearly indicate dependencies between subtasks in the first step of any dependent subtask.
- • You will format your response as follows: PARALLEL-LIST [ONLY subtasks index joined by comma that can be started immediately with no dependencies]

- **ST1:[subtask description]

- 1. [First Step]
- 2. [Second Step]

...

- **ST2:[Subtask Description]

- 1. [First Step - If this subtask depends on another, explicitly state: "Wait for ST[X] to complete" as the first step]
- 2. [Second Step]

...

**STx:[xxx]

...

Here is your task: {Task} List of facts that you know: {Answer_Facts} Please strictly follow the suggestions below: {Experience} Now begin! Write your plan below.

- B.2 Search Agent Prompts Query-Reflection Prompt You are a highly skilled query evaluation and augmentation agent for a given search query

###### • Evaluation Guidline

- – First, identify whether the original query is good enough. Generally, a search query from user may have different problems, for this part, you can refer to Problem Identification.
- – Second, refer to Available Solution to try to solve the problems that the original query have.
- – Finally, based on your solution, provide a modified query with your problem analysis and proposed solution.

###### • Problem Identification

- – Information Ambiguity: The entire query is ambiguous, or a part of a long query has semantic ambiguity, which lead to lack of information.
- – Semantic Ambiguity: The query contains terms with multiple meanings, leading to potential misunderstandings.
- – Complex Requirements: The query involves multiple steps or conditions that need to be addressed separately.
- – Overly Specific: The query is too narrow or detailed, potentially excluding relevant results.

###### • Available Solution

- – Information Ambiguity: Solutions include query expansion or removing the ambiguous part.
- – Semantic Ambiguity: Return the search results for the current query and prompt the model to resolve the ambiguity.
- – Complex Requirements: Break down complex requirements into simpler ones, plan multiple steps, and return the search results for the first step along with the remaining steps to be completed.
- – Overly Specific: Use Less specific query to enhance search quality, remember to retain the core content of the original query.

###### • Objective

- – Based on the analysis of the query’s issues and the proposed solutions outlined above, you need to optimize and revise the original query
- – Provide your analysis of the problems, revise suggestions and the final optimized query.
- – In terms of revising query, please keep your query consice and control the length of new query.
- – If the original query is already effective with minimal issues, avoid over-revising it.
- – Attention! You are not allowed to fabricate non-existent information yourself, which is untolerate!

###### • Output Format You must provide your evaluation in valid JSON format with the following structure:

{ "Problems": "Consise Analysis of Issues in the Query", "Suggestions": "suggestions that help revising the original query" "Augmented Query": "your final query" }

###### Query-Rollout Prompt

You are an expert in search strategy and query optimization. Your task is to analyze the user’s original query and generate {roll_out} distinct, high-quality search queries. that: Please Maintain the core intent of the original query. Read the following guidance.

Expand the search scope by incorporating:

- • Synonyms or related terms (e.g., "AI" → "artificial intelligence").
- • Different phrasings (e.g., "causes of climate change" → "climate change drivers").
- • Domain-specific terminology (e.g., "quantum computing" → "qubit entanglement").
- • Cross-domain keywords (e.g., "medical AI" → "AI in healthcare diagnostics").
- • Avoid redundancy while ensuring each query targets a unique angle or sub-topic.

Query Generate Guidelines:

You are required to Generate {roll_out} queries. Each query should be:

- • Grammatically correct and concise.
- • Focused on a specific aspect of the topic.
- • Appropriately general without exceeding the original’s specificity.
- • Potentially a sub-query of the original.
- • Different from other queries.

Input:

Original query: {query}

Output:

Attention! You are strictly required to provide your final queries within a Python List! The final output {roll_out} queries should in the format as: [query1, query2, ..., queryN]

##### B.3 Memory Prompts Memory Prompt

You are an expert in agent memory management, specializing in leveraging the Memory Summarization, the Memory Retrieval, and the Long-term Memory to boost agent reasoning.

Memory Summarization:

- • Summarize the following text which is the execution content of the agent at the current step: {memory of current step}.
- • Highlight the key points to assist the agent in better reasoning during subsequent steps.
- • Additionally, you must provide optimization suggestions for the next step.

Memory Retrieval:

- • Summarize the following text, and highlight key points: {memory of current step}.
- • Note that you are only responsible for summarizing, not providing optimization suggestions for the next step.
- • Your summary of the current step’s memory will be vectorized, and then the most relevant historical step memories will be recalled from the vector database by calculating cosine similarity.

Long-term Memory:

- • Here is the agent’s execution content from the previous step: {memory of previous step}.
- • Here is the long-term memory formed by summarizing the agent’s historical execution content: {long term memory}.
- • Please combine the agent’s previous execution content and the existing long-term memory, summarize them while highlighting the key points, and form a new long-term memory to help the agent reason better in subsequent steps.

Input:

- • Agent’s execution content at current step: {memory of current step}.
- • Agent’s execution content at previous step: {memory of previous step}.
- • Agent’s historical execution content: {long term memory}.

Output:

- • Memory Summarization: A point-by-point summary of agent’s current execution step and optimization suggestions.
- • Memory Retrieval: The retrieval of the most relevant historical steps.
- • Long-term Memory: An ongoing updated memory for recording the agent’s long-term historical steps.

##### B.4 Test-Time Scaling Prompts PRM-score Evaluation Prompt Evaluation Guidelines:

###### • Objective:

- – You will evaluate a candidate ActionStep node, which includes the following fields: ∗ step_number: Depth of this step within the TTS search tree. ∗ observations: Observations recorded after executing this action. ∗ action_output: Direct output resulting from this action. ∗ model_output: Raw LLM output that led to this action. ∗ error: Any encountered errors (can be None). ∗ score: Previously assigned score (for reference only). ∗ previous_steps: The history of earlier steps, including TaskStep and PlanningStep, along with the

trajectory of ActionSteps leading to the current state.

- – Your goal is to judge how promising this ActionStep is for advancing toward the user’s task, using your independent judgment while considering the continuity and logical flow of the ActionStep sequence, including the historical context.

###### • Evaluation Criteria:

- – Progress Toward Goal: ∗ Assess whether the action_output clearly and tangibly advances the overall task. ∗ Reward meaningful progress or valuable new information. ∗ Penalize irrelevant actions or weak impact.
- – Error and Stability: ∗ Penalize based on the severity of errors:

· Fatal/blocking errors: 0-1 points. · Significant errors: 1-3 points. · Minor or recoverable errors: 3-5 points.

∗ Reduce the score if the model_output is ambiguous or unstable.

- – TTS Efficiency: ∗ Reward actions that contribute efficiently toward reaching the goal. ∗ Penalize redundant or repetitive actions without meaningful progress.
- – Reflection Usage: ∗ Reward active utilization of reflection to improve upon past mistakes. ∗ Penalize ignoring reflection insights.
- – Loop Detection: ∗ Detect loops or repetitions compared to previous steps. ∗ Identify true loops and penalize based on severity.
- – Contextual Awareness: ∗ Infer alignment with previous PlanningStep and TaskStep. ∗ Ensure consistency with the TTS strategy and penalize deviations.

###### • Scoring Criteria:

- – 9-10: Clearly advances the goal; highly efficient; strong reflection use; no loops.
- – 7-8: Good advancement; minor inefficiencies; clear reflection use; minimal loop risk.
- – 5-6: Moderate progress; limited efficiency; moderate reflection use; mild repetition risks.
- – 3-4: Poor advancement; inefficient; weak reflection use; noticeable loop risks.
- – 1-2: Minimal advancement; repetitive actions; true loops; significant errors.
- – 0: Severe issues: explicit loops, critical errors, or complete irrelevance to the task context.

###### • Final Evaluation Output: You must provide your evaluation in valid JSON format with the following structure: { "analysis": "Detailed analysis addressing progress, TTS efficiency, reflection

usage, loop detection, contextual alignment with PlanningStep/TaskStep, error severity, and overall action quality.", "score": [integer between 0-10] }

###### PRM-list Evaluation Prompt

Evaluation Guidelines:

- • Objective:

- – You will evaluate N candidate trajectories, each representing a series of nodes in a search tree. Each trajectory contains the following: ∗ step_number: Depth of the node in the trajectory. ∗ observations: Observations recorded at each step of the trajectory. ∗ action_output: Direct action output at each step. ∗ model_output: Raw model output (LLM). ∗ error: Any errors encountered (can be None). ∗ score: Previously calculated score (if available).

∗ previous_steps: The history of earlier steps, including TaskStep and PlanningStep, with the trajectory of ActionSteps leading to the

current state.

- – Your goal is to evaluate each trajectory holistically, considering how well it progresses toward solving the user’s task. Select the trajectory that most effectively achieves this goal.

- • Evaluation Criteria:

- – Progress Toward Goal:

∗ Assess how well each trajectory advances the task at hand, considering both the individual node’s progress and the overall progression

of the entire trajectory. ∗ Reward trajectories that demonstrate tangible and meaningful progress toward the goal. ∗ Penalize trajectories with weak actions or minimal/no advancement.

- – Trajectory Efficiency: ∗ Evaluate how efficiently each trajectory progresses toward the goal, considering the depth and complexity of the steps. ∗ Favor trajectories that achieve significant progress with fewer steps. ∗ Consider the overall value-to-depth ratio when comparing trajectories of different lengths. ∗ Reward efficient exploration of the search space.
- – Loop Detection: ∗ Detect loops or repetitions within each trajectory, especially those related to previous steps. ∗ Loop types:

· Real Loops: Identical nodes (observations, action output, and model output) that do not add value to the trajectory. · Benign Repetitions: Similar strategies with variations yielding additional progress.

∗ Heavily penalize trajectories with real loops. ∗ Slight penalties for benign repetitions if they lead to meaningful improvements.

- – Error and Stability: ∗ Evaluate the severity of errors encountered in each trajectory and penalize based on their impact on progression. ∗ Error Severity:

· Fatal/Blocking Errors: Major penalty. · Significant Errors: Moderate penalty. · Minor/Recoverable Issues: Minor penalty.

∗ Penalize unstable or unclear model outputs. ∗ Consider how errors affect the overall trajectory’s ability to move toward the goal.

- – Overall Trajectory Quality: ∗ Evaluate the coherence and overall quality of the trajectory. ∗ Consider the logical sequence of steps and the exploration-exploitation balance. ∗ Evaluate the final node’s closeness to achieving the goal. ∗ Reward trajectories that make consistent progress and demonstrate coherent planning.

- • Final Output Format: Provide your evaluation in the following JSON format. Select the best trajectory and provide a detailed analysis explaining why it is the most promising trajectory.

{ "index": [integer], # Index of the best trajectory "analysis": "Detailed analysis addressing progress, efficiency, reflection usage, loop detection, error severity, and overall trajectory quality." }

###### Single Node Reflection Prompt

Node Information:

- • step_number: The depth of the node within the BON/beam search tree.
- • observations: The data or observations recorded during this step.
- • action_output: The direct output resulting from an action taken at this step (e.g., API call, tool response).
- • model_output: The raw output generated by the model at this step.
- • error: Any errors encountered during this step (if applicable).

Goal:

- • Summarize:

- – Provide a brief overview of what occurred at this node.
- – Describe the action taken and the results or new information that emerged as a result of this action.

- • Reflect:

- – Assess whether the action taken in this node was successful, partially successful, or unsuccessful.
- – Identify any errors, issues, or incompleteness relevant to this step.
- – Compare the node’s outcome with its assigned score, providing an evaluation of whether the score is aligned with the actual result.

- • Confidence:

- – Evaluate your confidence in the action taken at this node (High/Medium/Low).
- – If confidence is high, explicitly suggest continuing along this exploration path.
- – If confidence is medium or low, recommend potential improvements or alternatives, while leaving room for exploration to remain open.

- • Suggest:

- – Provide specific and focused suggestions for refining the current step.
- – These should be based on the evaluation of the current node, with an emphasis on actionable changes that can be made in the next attempt of a similar step.
- – Focus exclusively on improvements that can be applied within this node. Avoid proposing changes that span multiple steps or introduce larger, long-term strategies.
- – Base your evaluation strictly on the provided fields—action_output, observations, error, etc. Do not infer additional context or hypothesize about alternative paths or unknown factors.
- – Only flag a step as unsuccessful or in need of improvement if there is clear, tangible evidence (e.g., explicit errors, missing or incorrect outputs).
- – Do not override factual results based on subjective judgment, even if the node’s score does not seem to match the outcome.

- • General Guidelines:

- – Your suggestions should be conservative, focusing only on changes where there is a clear issue or opportunity for improvement.
- – If no significant issues are identified, provide minimal or no suggestions for improvement.

Output Format:

- • experience_summary: A concise overview of the events at this node and the key outcomes.
- • confidence_assessment: High/Medium/Low with a recommendation for future exploration.
- • lessons_learned: Key takeaways or specific improvements based on the evaluation of the current node’s action.
- • comments: Optional minor remarks, clarifications, or additional observations.

