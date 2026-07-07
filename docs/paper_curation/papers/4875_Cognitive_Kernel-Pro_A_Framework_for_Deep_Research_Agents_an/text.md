# arXiv:2508.00414v3[cs.AI]22Apr2026

## Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training

###### Tianqing Fang∗, Zhisong Zhang∗, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, Yonglin Wang, Jingchen Ni, Tianshi Zheng, Chun Chen, Wenhao Yu, Zhenwen Liang, Hongming Zhang, Haitao Mi, Dong Yu

###### Tencent AI Lab https://github.com/Tencent/CognitiveKernel-Pro

Open-Source Agent & Free Tools

Proprietary Agent & Charged Tools

###### SFT RL

###### SFT RL

60

60

60

58

80 Pass@1 Voting Pass@3

Pass@1 Pass@3

|60.0| |
|---|---|
|52.752.7| |
|37.0<br><br>45.4| |
|35.2| |
| | |
| | |
| | |

75.2

53.4

52.8

73.9

69.1

XBench-DeepSearch

70

50

67.4

47.6 41.3

50

66.1 66.1 66.7

48

44.7

45

60

37.9

40

36.2

40

GAIA-text

GAIA-text

50

36

34.0

32

###### GAIA

31.0

30

30

40

24

20

30

20

20

20

12

10

10

10

N/A

0

0

0

0

Pass@1 Pass@3

WebDancer 7BWebSailor 7B

CK-Pro-8B

Pass@1 Pass@3

CK-Pro-8B Claude-3.7

Claude-3.7

OAgent DeepResearch OWLManus

Cognitive Kernel-Pro

Smolagents

(a) (b)

(c)

Figure 1: (a) Performance comparison on the full GAIA development set (n=165). The left panel presents results from our Cognitive Kernel-Pro framework, utilizing our Qwen3-8B SFT model and Claude-3.7-sonnet as foundation models with exclusively free tools. The right panel displays Pass@1 scores for proprietary agents and systems employing paid tools. (b) Performance on the text-only GAIA subset (n=103), demonstrating our 8B model’s superiority over 7B models in the WebDancer/WebSailor family (∼2% higher Pass@1, over 10% higher Pass@3). (3) Performance after reinforcement learning, over SFT baselines on GAIA-text and XBench-DeepSearch.

#### Abstract

General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present Cognitive Kernel-Pro, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-ofthe-art results among open-source and free agents. Notably, our 8B-parameter opensource model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents.

Note: The term Cognitive Kernel (Zhang et al., 2024) refers to the core computational framework of the agent, designed to emulate the cognitive processes of the human mind.

∗Equal Core Contributors

[Figure 3]

[Figure 4]

CogKernel (Zhang et al, 2024)

[Figure 5]

Agent Framework

[Figure 6]

[Figure 7]

[Figure 8]

Hierarchical Multi-Agent Structure Planner-Action Interaction (§2.1)

[Figure 9]

+

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

(Open)WebVoyager (He et al, 2024)

[Figure 17]

[Figure 18]

[Figure 19]

DocBench (Zou et al, 2024)

[Figure 20]

[Figure 21]

PersonHub (Ge et al, 2024)

CK-Pro-8B (Qwen-3)

[Figure 22]

[Figure 23]

Agent Data CogKernel-Pro

[Figure 24]

[Figure 25]

WebAggregator (§3.1, 4.1, Wang et al, 2025) AgentWebQA (§3.2) PersonaHub Augmentation (§3.3) Reasoning, Coding, etc (§3.4)

[Figure 26]

[Figure 27]

+

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

WebEvolver (Fang et al, 2025) WebCoT (Hu et al, 2025)

[Figure 36]

Agent Foundation Model Post-Training

[Figure 37]

+ Reflection&Planning(§4.3,Wanetal,2026)

- Figure 2: Technical roadmap showcasing prior innovations from Tencent AI Lab (Cognitive Kernel; Zhang et al., 2024, WebVoyager; He et al., 2024a, etc) and their integration to Cognitive KernelPro via three core components, agent framework development, agent data construction, and agent foundation model training. Yellow blocks highlight novel contributions in this work and the corresponding section numbers.

#### 1 Introduction

The rapid advancement of Deep Research Agents (Monica.Im, 2025; OpenAI, 2025) has transformed the landscape of automated knowledge discovery and problem-solving. These agents, powered by large language models (LLMs) and vision-language models (VLMs), excel in tasks such as coding, web navigation, file processing, and complex reasoning. However, efforts toward fully open-source agent frameworks (Roucher et al., 2025; Wu et al., 2025a; Li et al., 2025a) remain limited. Existing open-source implementations (Zhu et al., 2025; Hu et al., 2025) often rely on proprietary tools like Jina Reader, FireCrawl, or Chunkr to achieve competitive performance, creating barriers to accessibility and reproducibility, or lack of multimodal or general agentic abilities (Wu et al., 2025a; Li et al., 2025a). This dependency on paid tools underscores the need for a robust, fully opensource framework that maximizes the inherent capabilities of LLMs and VLMs without external dependencies.

To address this gap, we propose Cognitive Kernel-Pro, a multi-module, hierarchical agent framework designed to facilitate fully open-source agent development. Cognitive Kernel-Pro leverages Python code as its action space, harnessing the full reasoning and code-generation potential of modern LLMs. The framework adopts a modular architecture, featuring a main agent that orchestrates specialized sub-agents for web navigation, file handling, and tool invocation. Each module operates independently, ensuring modularity and extensibility while simplifying the collection of task-specific training data. By minimizing reliance on proprietary tools, Cognitive Kernel-Pro emphasizes the intrinsic capabilities of Agent Foundation Models.

In addition to the framework, we introduce a comprehensive training recipe tailored for Cognitive Kernel-Pro, covering diverse domains such as web navigation, file processing, code generation, and reasoning. Our approach includes the construction of verifiable agent query-answer pairs, ensuring high-quality training data. To enhance data collection, we incorporate intermediate process hints and employ hint-based rejection sampling, which significantly improves the quality and relevance of the collected data. This structured training methodology enables Cognitive Kernel-Pro to achieve robust performance across diverse tasks while maintaining full open-source compatibility.

###### Agent Class

###### CogKernel-Pro Agent Framework

[Figure 39]

[Figure 40]

completed_list todo_list

click

type

restart

[Figure 41]

Web

Task (str)

goback

goto

Planner

scroll_up scroll_down screenshot read_text read_multimodal

[Figure 42]

Agent

experience information

[Figure 43]

task

File Agent

search

Main

Env. Obs.

Agent

Trajectory

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Output

[Figure 48]

Thought: … my thought

Reflection Module

Tool

(str)

web

Search API ask_llm

code:

[Figure 49]

Calling

```python

Action

retry

task="my task" print(web_agent(task))

[Figure 50]

doc

[Figure 51]

[Figure 52]

return

```

Code

Python

code

Text-only Model or Multimodal Model

Multimodal Model Required

- Figure 3: Overview of the Cognitive Kernel-Pro Agent Framework. The left panel illustrates the functionality of agent class, where the main agent, web agent, and file agent inherit from the common base class. The planner maintains a state dictionary containing ‘completed_list’, ‘todo_list’, ‘experience’, and ‘information’ (§2). The action generator produces Python code as a code agent or invokes predefined functions of sub-agents, such as the web agent, as well as other built-in tools. The right panel illustrates the hierarchical structure of Cognitive Kernel-Pro, listing all functions defined by each agent. Additionally, a standalone reflection module is included to assess task completion; if the task is incomplete, the agent will retry (§4.3). The agent foundation model behind each module/sub-agent is the same.

Furthermore, we explore inference-time optimization techniques to address the inherent randomness in tasks like web browsing. To mitigate variability, we propose a pipeline that integrates retry mechanisms and ensemble-based multi-run strategies. This approach enhances the reliability and consistency of Cognitive Kernel-Pro’s performance, particularly in dynamic and unpredictable environments. By combining a modular framework, a robust training recipe, and optimized inference strategies, Cognitive Kernel-Pro sets a new standard for open-source agent development, paving the way for accessible and reproducible advancements in agent-based research.

#### 2 Cognitive Kernel-Pro Framework

We present an overview of the Cognitive Kernel-Pro framework in Figure 3.

We adopt a two-tier multi-module framework in our agent implementation. This framework consists of a main agent, responsible for task decomposition, sub-task delegation, and information aggregation, tool calling, code generation, as well as several sub-agents, whose objective is to solve the sub-tasks assigned by the main agent. Both the main agent and sub-agents inherit from the same base class, where the input is a task string, the output is a response string, and intermediate actions are executed as Python code.

Main-Agent. The main agent directly manages the problem-solving process towards achieving the overall goal. It decomposes the original complex tasks into manageable sub-tasks and assigns these to sub-agents as needed. Upon receiving responses from the sub-agents, the main agent aggregates the information and continues with the main procedure. Notably, the main agent does not possess specialized skills such as web browsing or file processing; only the sub-agents (e.g., the web agent and the file agent) are equipped with such capabilities. Nevertheless, the main agent is aware of the functionalities of the sub-agents and is in charge of delegating appropriate sub-tasks accordingly.

Sub-Agents. The sub-agents are equipped with specialized skills that are essential for a generalpurpose task-solving agent system. Each sub-agent follows a multi-step procedure similar to that of

Open Framework

Open-source Model

No Proprietary Tool (excl. Google)

Agent Ability Web File Code

Agent

| | | |
|---|---|---|
|Deep Research OWL OAgents WebDancer WebSailor<br><br>|✗ ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✓|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✗ ✗ ✓ ✗ ✗|
|Cognitive Kernel-Pro|✓ ✓ ✓|✓ ✓ ✓|
| | | |

Table 1: Feature Comparison of AI Agent Frameworks. Google Search API (which, can be easily switched to free APIs such as DuckDuckGo if needed) is excluded when comparing proprietary tools because it’s a must in search-related tasks. Note: WebDancer and WebSailor support PDF fetching but lack general file agent capabilities

the main agent but is enhanced with specialized actions that enable direct interaction with specific resources. In our system, we primarily include the following two sub-agents:

- • Web Agent. The web agent is equipped with a browser and can navigate live web pages to collect relevant and time-sensitive information. We implement an autonomous web browser using playwright, which provides both the accessibility tree and the screenshot of the current web page. The web agent makes decisions based on the current web page’s observations. We adopt typical web agent actions, including “click”, “type”, “scroll”, “wait”, “goback”, “restart”, “goto”, “save”, “stop”, “screenshot”. Here, “save” refers to explicitly saving a web file to a local path for the file agent to process, while “stop” denotes terminating the navigation process due to task completion or unrecoverable errors. “screenshot” is a special function to turn on screenshot mode to call a multimodal language model to process the image. If this function is not called, the default input to the agent foundation model is the text-only accessibility tree of the current webpage.
- • File Agent. The file agent is designed to process a variety of file types, such as PDF files (.pdf), spreadsheet files (.xlsx, .xls, .csv), and image files (’.png’, ’.jpg’, ’.gif’, etc.). Inspired by the web agent, we adopt a similar task-solving process. To manage potentially large files, we split each file into pages (using specialized tools for each file type) and allow the file agent to read a subset of pages at a time. Correspondingly, the action space includes “load_file”, “read_text”, “read_screenshot”, “search” and “stop”. Here, the file agent can decide whether to read the screenshot of certain pages or only the text, with the screenshot mode being essential for image-based tasks.

While we do not have a standalone Code Agent in the system, every sub-agent is a code agent since the output action of every agent is essentially python code. For example, every agent can generate Python code to perform calculation or other reasoning tasks that can be solved by code generation and execution.

In addition to these sub-agents, our framework is flexible and can be extended to support more sub-agents with specialized skills. The design of this two-tier multi-module framework enables the decoupling of the main agent’s task-solving procedure from the detailed sub-task execution of the sub-agents, providing a flexible and adaptable system capable of supporting a wide range of scenarios.

Tool Calling Our system utilizes minimum paid tools. We use Google Search API to return search results, wrapped as a function “simple_web_search”. Besides, we only use “ask_llm” as an additional function to directly let the base agent foundation model to answer a question. Despite Google Search API is not free but it is required by most information seeking agents. Other than that, we do not use any proprietary tools, as illustrated in Table 1.

The detailed implementations are presented in the Appendix A.

Type Data Name Data Type #Query #Steps Web OpenWebVoyager (He et al., 2024b) Web Browsering 1,259 9,098

| |Multihop URLQA (§3.2) Web Information Seeking 4,225 25,589 AgentWebQA (w/ hint) (§3.3) Web Information Seeking 2,721 32,231 PersonaHub-Aug (§3.4) (No Ground Answer) 1,000 2,088 WebWalkerQA (Wu et al., 2025b) Web Information Seeking 1,904 18,116<br><br>|
|---|---|
|File|DocBench (Zou et al., 2024) .pdf 300 1,566 TableBench (Wu et al., 2025c) .csv, .xlsx 1,000 9,482<br><br>|
|Reasoning<br><br>|NumiaMath (Beeching et al., 2024) Math Reasoning 616 524 BAAI/TACO (Li et al., 2023) Code/Puzzle 225 730 RiddleSense (Lin et al., 2021) Riddle/Puzzle 179 165 LogiCoT (Liu et al., 2023) Logical Reasoning 1,400 1,400<br><br>|
| | |

Table 2: Summary of the training recipe.

###### Information Aggregation Question Construction

[Figure 55]

[Figure 56]

Calculation: Divide Q: During 2014 to 2023, in which year did Moldova have the highest year-on-year growth percentage of GDP per capita, and what was the corresponding GDP per square kilometer in that year?

[Figure 57]

[Figure 58]

[Figure 59]

Calculation: Divide Set Operation: Sort

[Figure 60]

[Figure 61]

Annual GDP of Moldova

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Annual GDP per capita of Moldova = Annual GDP of Moldova ÷ Total population of Moldova Annual GDP of

Annual GDP growth rate of Moldova, and the year with the highest growth

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

A: In 2017, the GDP per square kilometer was approximately $281,134.

Total population

Moldova ÷ Area of Moldova

of Moldova of 2o14, 2015, …

[Figure 72]

[Figure 73]

Moldova land area.

Figure 4: Illustration of information aggregation in the creation of URLQA.

#### 3 Cognitive Kernel-Pro Agent Foundation Model Supervised Fine-Tuning

[Figure 74]

###### 3.1 Overall Data Recipe

The overall training data recipe is presented in Table 2. We divide the ability of deep research agent into three types, Web, File, and Reasoning. For each of the category, we either convert the existing benchmarks to the format that we need or construct new deep research queries (§3.2, §3.3, and §3.4).

###### 3.2 Multi-hop Web Search Data Construction

The data synthesis procedure here aims to create diverse and complex multi-hop information-seeking QA pairs grounded in web pages. We expect the constructed questions requiring information cannot be obtained without a retrieval process. To cover multiple domains, we first collect a seed URL set by searching for topic-diverse texts from several datasets using the commercial API of Google. Then, an agent traverses and browses web pages starting from these seed URLs with the designed prompt and examples, gathering information and composing questions accordingly.

Additionally, to simulate varied task intents, we add the principals and several examples in the prompt, constraining that the answer must be derived through information aggregation operations, as shown in Figure 4. The composition rules are specially designed for different forms of information sources, such as math calculation for numbers, sorting for candidate sets, data analysis for tables.

A comprehensive extension of this pipeline is presented as a separate paper in WebAggregator (Wang et al., 2025).

###### 3.3 Agent Exploration-based Data Construction

The data synthesizing process can be viewed as a specialized task, for which we can re-use our existing agent framework. In this context, all sub-agents remain unchanged, while the main agent is adapted specifically for data synthesis. The overall procedures and mechanisms are largely consistent with those of our general-purpose framework. We adopt several modifications to further tailor the framework for the data synthesizing process.

Prompt Adjustments. The data synthesizing process operates in a way that is essentially the reverse of the ordinary task-solving procedure. In ordinary task-solving, the agent is provided with a question and aims to search for the answer. In contrast, data synthesizing requires the agent to construct the query itself by integrating pieces of information gathered throughout the exploration process. To accommodate this use case, we retain the core mechanisms of the main agent but revise its prompts to address the unique requirements of data synthesis. In particular, we instruct the agent to construct complex queries by combining information from multiple verifiable sources.

Topic Sampling. To allow a diverse set of interesting queries that are synthesized, before the agent-based data synthesizing procedure, we first generate an overall topic for each query to be synthesized. Using a self-instruct based method, we use an LLM to generate broad and interesting topics with verifiable sources of truth. We provide several seed examplse and let the LLMs to generate more. After generation, we adopt a diversity-based sampling procedure to sample a diverse subset of topics for our actual query synthesizing process.

Hint-based Training Trajectory Sampling. The query synthesizing procedure yields not only the constructed queries but also all associated intermediate and final results. We observe that, during trajectory sampling for training, providing these intermediate results as hints to the task-solving agent significantly improves the success rate for training data collection. To enhance this, we augment the queries with additional textual hints. It is important to note that this augmentation is employed only during training data collection, where we can assume that answers are available. Once the trajectories are obtained, all such hints are removed from the model inputs and outputs prior to the actual training process. The hints are wrapped between <secret> and </secret> during sampling, and will be removed during training.

The prompt of Data Synthesizing Requirements, Topic Sampling, and Hint-based Training Trajectory Sampling are presented in Appendix B.

###### 3.4 Persona Hub-based Data Augmentation

Persona-triggered Query Synthesis. PersonaHub (Ge et al., 2024) provides an effective strategy to synthesize large-scale diverse queries for various LLM tasks like math, logic reasoning, instruction following, etc. In this work, we explore to utilize Persona Hub to synthesize training queries for the deep research agent task. Seeded with a manually crafted deep research question and its corresponding persona as the in-context example, we utilize an LLM to generate a synthetic deep research query given a synthetic persona from Persona Hub. Though with limited number of manually crafted deep research questions, we can easily scale up the synthesis of diverse deep research queries by using more personas from Persona Hub as triggers.

Trajectory Sampling and Validation. The main challenge of Persona Hub-based data augmentation for deep research agent task is the lack of ground truth answer for the synthetic query. To tackle this challenge, we conduct cross-validation of the trajectory outcomes from different agent systems and include 1k synthetic queries with their trajectories from Cognitive Kernel-Pro agent system into our training set, as detailed in Table 2. Our ablation study suggests a small number of training trajectories from Persona Hub-based data augmentation can effectively improve performance. On the other hand, manual validation and response annotation of these synthetic queries are considerable but not yet included in this work.

Framework Agent Model Paid Tools Avg. Level 1 Level 2 Level 3 Closed-source Agent Frameworks

TraseAgent Claude Unknown 70.30 83.02 69.77 46.15 Deep Research Unknown Unknown 67.36 74.29 69.06 47.60 h2oGPTe Claude-3.5 Unknown 63.64 67.92 67.44 42.31 Desearch GPT-4o Unknown 56.97 71.70 58.14 23.08 Manus Claude, etc. Unknown 73.3 86.5 70.1 57.7

Open-source Agent Frameworks w/ Paid Tools OWL–Roleplaying 4o & o3-mini Chunkr, FireCrawl,

58.18 81.14 54.65 23.08 OWL–Workforce Claude-3-7 69.09 84.91 67.44 42.31 OWL–Workforce* Claude-3-7 w/o whisper 60.61 73.58 62.79 26.92 OAgent

Whisper, o3-mini

Jina Reader, Whisper, Baidu & Bing API

66.67 77.36 66.28 46.15

Claude-3-7

-Pass@3 73.93 83.02 74.42 53.85 w/o Paid Tools

TapeAgents Claude-3-7 — 55.76 71.70 53.49 30.77 AutoAgent Claude-3-5 — 55.15 71.70 53.40 26.92 Magnetic-1 OpenAI o1 — 46.06 56.60 46.51 23.08 Smolagents Openai o1 — 49.70 54.72 53.49 26.92 Smolagents* Claude-3-7 — 52.10 60.38 54.65 26.92

- - Voting — 53.99 66.04 52.33 33.33
- - Pass@3 — 63.64 75.47 61.63 46.15

Cognitive Kernel-Pro — 57.58 77.36 54.65 26.92

- -Voting Claude-3-7 — 63.64 69.81 65.12 46.15
- -Pass@3 — 70.91 83.02 68.60 53.85 Cognitive Kernel-Pro — 32.73 43.40 32.56 11.54
- -Voting CK-Pro-8B — 34.54 47.17 33.72 11.54
- -Pass@3 — 38.18 50.94 38.37 11.54

- Table 3: Performance of various agent frameworks on GAIA dev set (n=165). * after agent names indicate our reproduced results. We boldface the best pass@3 performance and underline the best pass@1 performance of open-source agent frameworks without paid tools (except for Google Search).

###### 3.5 Reasoning Data Construction

We refine several existing reasoning datasets relevant to general agents, including NumiaMath (applied mathematical reasoning), LogiCoT (logical reasoning), TACO (code reasoning), and RiddleSense (puzzles and riddles). For TACO, we extract input/output pairs from the task descriptions and construct code agent queries by concatenating the task description with the input case, using the corresponding output as the expected gold answer. For the other datasets, we directly transform the question-answer pairs into the input-output format compatible with the “ask_llm” function.

###### 3.6 Trajectory Sampling

For all constructed query-answer pairs, we utilize gpt-4.1 as the foundational backbone model within our Cognitive Kernel-Pro framework to generate agent trajectories. Subsequently, we apply rejection sampling using similarity-based matching, facilitated by the ‘cot_qa‘ of LangChain, with gpt-4.1 as the backbone model. For hint-based sampling, we exclude all hints enclosed within <secret> and </secret> tags to prevent information leakage. Each query is sampled up to three times until successful completion.

Framework Model Size Avg. Level 1 Level 2 Level 3 WebThinker-Base 32B 44.7 53.8 44.2 16.7

|WebThinker-RL 32B Search-o1 32B WebDancer 32B WebDancer 32B WebSailor 32B WebSailor 72B WebShaper 32B WebShaper 72B|48.5 28.2 40.7 51.5 53.2 55.4 53.3 60.1|56.4 50.0 16.7 33.3 25.0 0.0 46.1 44.2 8.3 61.5 50.0 25.0<br><br>- - -<br>- - -<br><br><br>69.2 50.0 16.6 69.2 63.4 16.6|
|---|---|---|
| | | |
|Search-o1 7B R1-Searcher 7B WebDancer 7B<br><br>-Pass@ 3 7B WebSailor 7B<br><br>|17.5 20.4 31.0 34.0 37.9|23.1 17.3 0.0 28.2 19.2 8.3 41.0 30.7 0.0<br><br>- - -<br>- - -<br>|
|Cognitive Kernel-Pro 8B<br><br>-Voting 8B<br>-Pass@3 8B<br>|43.7 41.1 49.3<br><br>|56.4 42.3 8.33 53.8 34.6 16.7 61.5 44.2 16.7<br><br>|
| | | |

- Table 4: Performance of open-source agent frameworks on the text-only subset of GAIA (n=103). We boldface the best Pass@3 performance and underline the best pass@1 performance for models with size 7 or 8B.

###### 3.7 Experiment Setup

Baselines Based on the open-source code of OWL, we reproduced OWL’s performance using Claude-3.7-Sonnet in our own environment. All experimental settings followed the default configurations provided by OWL, including the use of corresponding LLM APIs for each agent and the integration of certain paid tools, such as Chunkr and FireCrawl. All agents adopted greedy decoding during their inference, and the maximum number of replanning tries was set to the default value of

- 2. It should be noted that we did not use the Whisper API, and our network environment was different from that of the original experiments. These factors may have contributed to the reproduced performance being lower than the original results reported by OWL. As for the implementation of the SmolAgents, our experiment utilizes most of the tools provided by the Open Deep-Research version of SmolAgents and follows its configuration, except that we enhance the web browsing tool with DOM tree parsing to display web structure, enable element clicking, and text input.

Cognitive Kernel-Pro We only use one paid tool, Google Search API, which is a must for almost all agent frameworks. Claude-3.7 is used as the backbone for supporting the agent framework. We also use our fine-tuned CK-Pro-8B (based on Qwen-3-8B) as the agent foundation model.

Datasets We use the GAIA dataset (Mialon et al., 2024) as the evaluation benchmark, a comprehensive suite designed to assess the general intelligence and multi-step reasoning capabilities of AI agents across diverse tasks, including web navigation, question answering, file manipulation, and multimodal processing, making it ideal for evaluating the performance of our Cognitive Kernel-Pro framework.

- 3.8 Results

Full dev set of GAIA Table 3 shows the performance of various agent frameworks on the complete GAIA dataset, differentiating between closed-source and open-source systems, with the latter grouped by their use of paid tools, and featuring our reproduced results marked with an asterisk (*). Cognitive Kernel-Pro, utilizing Claude-3.7, surpasses Smolagents by 5% in Pass@1 and 7% in

Inference-time Alg. Inference-time

Avg. Level 1 Level 2 Level 3

|Model| | |
|---|---|---|
|w/o Reflection Reflection CK-Pro-8B Reflection Qwen-3-32B Reflection GPT-4.1|27.0 28.5 31.5 32.7|35.8 27.9 7.7 37.9 29.4 7.7 41.5 32.5 7.7 43.4 32.6 11.5|
| | | |

Table 5: Ablations on different backbone LLM used for reflection and voting.

Base Agent Model MLLM Avg. Level 1 Level 2 Level 3

|CK-Pro-8B pass@1 Qwen-2.5-VL-72B CK-Pro-8B pass@1 GPT-4.1<br><br>|33.94 32.67<br><br>|43.40 34.88 11.54 43.40 32.56 11.54<br><br>|
|---|---|---|
|CK-Pro-8B pass@3 Qwen-2.5-VL-72B CK-Pro-8B pass@3 GPT-4.1|37.56 38.12|49.06 38.64 11.54<br>50.94 38.37 11.54<br>|
| | | |

- Table 6: Performance of using CK-Pro-8B as the base agent foundation model and variances of multimodal language model we use. We underline the best results on pass@1 and boldface the best performance on pass@3

Pass@3 under identical experimental conditions (e.g., LLM and Search APIs, Internet connectivity), demonstrating its efficacy. Its performance also rivals OWL, which relies on proprietary tools like Chunkr for file processing and FireCrawl for web browsing, underscoring its significant potential.

Additionally, we present results from fine-tuning a Qwen-3-8B model on the trajectories outlined in Section §3, supported by GPT-4.1 for multimodal functions, achieving a Pass@3 score of 38.18%—a 30% gap from the state-of-the-art Claude-3.7 model—suggesting considerable scope for future enhancements.

Text-only Subset of GAIA We present the performance comparisons on the text-only subset of GAIA in Table 4. The major baseline is the 7B version of WebDancer and WebSailor. In addition, we list the performance of 32B and 72B models as a reference in the upper half of the table. We also include the performance of Search-o1 (Li et al., 2025b), R1-Searcher (Song et al., 2025), and WebThinker (Li et al., 2025c) in the table. Cognitive Kernel-Pro under CK-Pro-8B model yield the best pass@1 and pass@3 performance across all levels of GAIA.

Ablation Study of Reflection We present an ablation study of the effect of the reflection module in Table 5. Using an open-source model Qwen-3-32B is already good enough, counterparting GPT-4.1. However, if we use our trained CK-Pro-8B model, without being finetuned with reflection ability, there is only marginal improvement. This indicates a future direction of involving the ability of reflection to the training of agent foundation models.

Ablation Study of the Multimodal Language Model Table 6 presents the impact of using different backbones for our multimodal language model. Our results show that replacing Qwen-2.5-VL-72B with GPT-4.1 yields only marginal performance improvements. This suggests that the observed performance gains are not solely because of the use of a more advanced multimodal model like GPT-4.1, as Qwen-2.5-VL-72B achieves comparable results. Future work will be developing a fully multimodal language model as the backbone, designed to seamlessly support both text and multimodal inputs.

#### 4 Cognitive Kernel-Pro Agent Reinforcement Learning

###### 4.1 Data Construction and Filtering

Unlike standard SFT data, RL training necessitates samples characterized by rigorous internal consistency and unambiguous grounding to ensure the derivation of high-fidelity reward signals. Using the methodology of WebAggregator (Wang et al., 2025), our data samples are structured as <Question, Answer, Reference Solution, Grounded URLs>.

Multi-stage Quality Filtering Starting with the initial dataset, we apply a multi-stage filtering protocol to ensure that every sample is deterministic and verifiable.

We first evaluate whether the provided solution can logically and correctly lead to the reference answer. We utilize an Agent to derive the answer from the reference reasoning steps. If the derived prediction from the reference solution is inconsistent with the reference answer, the sample is discarded. This step acts as a rigorous sanity check, filtering out raw noisy data. Then, a computationally intensive verification where an agent confirms the validity of external sources and the correctness of the reasoning steps.

Moreover, for each candidate sample, we perform four independent rollouts using the CK-Pro SFT checkpoint and retain 400 high-quality samples whose rollouts exhibit partial correctness. Based on this multi-rollout verification, we derive a curriculum signal that captures both verifiability and logical depth. RL training starts with high-consistency samples to provide a denser and more stable reward signal, and then progressively shifts to more complex, lower-consistency tasks.

Structured Subgoal Recovery A primary challenge in RL for deep research is the sparsity and delay of rewards caused by exceptionally long reasoning trajectories. To facilitate more granular credit assignment, we decompose these long-form solutions into intermediate milestones. We leverage the rich metadata from the synthesis stage (e.g., URLs, and intermediate states) to reconstruct the reasoning path into a structured subgoal dictionary. Each entry in the dictionary represents a discrete, verifiable milestone with clear success criteria.

###### 4.2 Subgoal-Conditioned GRPO

Due to the sparsity and temporal delay of reward signals in long-horizon trajectories, applying reinforcement learning (RL) to deep research agents for complex, multi-step information retrieval is inherently sample-inefficient. To address this challenge, we propose a novel subgoal-conditioned GRPO framework that explicitly separates subgoal-level and task-level supervision, thereby mitigating the “reward collapse” phenomenon commonly observed in multi-objective RL. In this section, we formally define the setting and present the proposed optimization method.

Atomic Main-Agent Step Tuple (ht). At each discrete time step t, the interaction of the main agent with the environment is represented by an atomic step tuple

ht = ⟨st, at, τtsub, kt,rt,ro⟩. Each component is defined as follows:

- 1. st ∈ S: the main-agent state, comprising the current observation Ot together with the accumulated memory or context mt.
- 2. at ∈ C: the main-agent action, consisting of the generated reasoning content, plan, and executable Python code.
- 3. τtsub: the sub-agent trajectory induced by at, i.e., the execution trace produced by the lowerlevel agent after the main agent issues action at. Concretely, τtsub consists of a sequence of sub-agent state–action pairs ⟨ssubℓ , asubℓ ⟩.

- 4. kt ∈ N: the subgoal index, indicating the currently active phase of the overall plan.
- 5. rt: the dense step reward, providing immediate feedback on the quality of at.
- 6. ro: the sparse outcome reward, reflecting the final task-level result Rfinal.

Subgoal-Conditioned GRPO for CK-Pro. We optimize the main agent using a subgoalconditioned variant of Group Relative Policy Optimization (GRPO), in which local subgoal-level feedback and global task-level feedback are normalized separately and then combined into a unified step-wise advantage.

###### 1. Trajectory sampling.

Given the current policy πθold, we sample a group of G trajectories, {T (1), . . . , T (G)}. Each trajectory i has length Ti and is represented as

T (i) = {h0(i), h1(i), . . . , h(Ti)

}.

i

###### 2. Decoupled normalization. To prevent high-variance final rewards from overwhelming informative local signals (i.e., reward collapse), we normalize the two objectives independently. Rsub(i,k) denotes the cumula-

tive subgoal reward for trajectory i in phase k, and Rfinal(i) denotes the final task reward for trajectory i.

- Stream A: Subgoal-level advantage (Asub). The subgoal reward is compared against the group of trajectories within the same subgoal

phase k. Let Gk ⊆ {1, . . . , G} denote the subset of trajectories that reach subgoal phase k. We compute the phase-wise statistics across trajectories as:

µsub,k =

1

|Gk| ∑

i∈Gk

Rsub(i,k), σsub,k =

1

|Gk| ∑

i∈Gk

Rsub(i,k) − µsub,k

2

.

For any step t assigned to phase kt, the corresponding normalized subgoal-level advantage is

Asub,(i) t =

Rsub(i,kt) − µsub,kt σsub,kt + ϵ

.

- Stream B: Final task-level advantage (Afinal). In contrast, the final task reward is normalized using global group statistics across all G trajectories:

G

G

1 G

1 G

2

Rfinal(i) − µfinal

Rfinal(i) , σfinal =

∑

∑

.

µfinal =

i=1

i=1

The resulting final task-level advantage is

Rfinal(i) − µfinal σfinal + ϵ

Afinal(i) =

.

##### 3. Advantage composition (At). We combine the local tactical advantage and the global strategic advantage at step t as

A(ti) = Asub,(i) t + λAfinal(i) .

Here, λ is a hyperparameter that controls the trade-off between subgoal execution and overall task alignment.

###### 4. Policy optimization.

We update the main agent policy πθ by maximizing the GRPO surrogate objective JGRPO(θ). For each token x in the action sequence a(ti), we define the importance ratio ρθ(x) = ππθ(x|st,<x)

θold(x|st,<x). The objective is formulated as:

JGRPO(θ) =

Ti

G

1 G

### ∑

### ∑

### ∑

t=0

i=1

x∈a(ti)

min ρθ(x)A(ti), clip (ρθ(x),1 − ϵc,1 + ϵc) A(ti)

− βDKL (πθ(· | st,<x) ∥ πref(· | st,<x)) ,

where ϵc is the clipping margin and β is the KL penalty coefficient. The KL penalty can be estimated per-token using an unbiased estimator, preventing the policy from deviating excessively from the reference model.

###### 4.3 Inference-time Scaling

We introduce two inference-time optimization procedures—reflection with verification and voting—that allow the agent to evaluate and refine its own trajectories, improving robustness and accuracy.

Reflection with Verification Reflection allows the agent to review its previous actions after each task attempt. The reflection module summarizes the full trajectory in an action–observation format (e.g., “Action 1: ..., Observation 1: ...”). To obtain structured evaluation signals, we use DeepVerifier (Wan et al., 2026), a rubric-based verifier built on an automatically constructed failure taxonomy. Specifically, DeepVerifier evaluates both the trajectory and the predicted answer along five dimensions: finding sources (use of specific and authoritative evidence), reasoning (logical consistency and faithfulness to evidence), problem understanding and decomposition (correct interpretation of the task and subgoals), action execution (correct use of tools, formats, and modalities), and trajectory efficiency (ability to reach a valid answer within the step budget). Each dimension is rated on a four-level scale: excellent, good, needs improvement, or poor.

If the agent identifies any violations of these criteria, it will attempt the task again with additional feedbacks: comprising per-rubric scores and targeted error localization that pinpoint which failure category was triggered and at which trajectory step. The agent repeats this process until a satisfactory answer is produced or a predefined retry limit is reached.

Voting The voting process enables the agent to aggregate multiple trajectories, enhancing its decision-making and increasing the likelihood of achieving optimal outcomes. In practice, the agent attempts the same task several times, summarizes all resulting trajectories, and then selects the trajectory that best adheres to the rubric-based guidelines established in the reflection process as the final output. DeepVerifier assigns a structured score to each trajectory based on actively retrieved web evidence, and the candidate with the highest verifier score is retained. Unlike reflection, which evaluates each attempt in isolation, the voting process allows the agent to compare and contrast information across multiple trajectories. This comparative approach helps the agent identify higherquality solutions by leveraging differences among the attempts. For example, when the agent is asked to find a singer’s earliest album, one attempt might return an album from the 2000s while another finds one from the 1990s. By comparing these results, the voting module can recognize that the album from the 1990s is the more accurate answer, as it is earlier.

GAIA-text

Deep Research Agent

xbench-DeepSearch Level 1 Level 2 Level 3 Avg.

| | | |
|---|---|---|
|Cognitive Kernel-Pro SFT<br><br>-pass@3|56.40 42.30 8.33 43.68 61.50 44.20 16.70 47.55|20.00 32.00<br><br>|
|Cognitive Kernel-Pro RL3<br><br>- GRPO + Verifier<br>- Subgoal-GRPO + Verifier<br>- GRPO pass@3<br>- Subgoal-GRPO pass@3<br>- Subgoal-GRPO pass@3 + Verifier<br>|50.85 30.45 13.89 36.24 58.97 34.61 12.50 41.25 60.47 33.65 12.50 41.34 58.97 48.07 25.00 49.51 63.72 46.92 41.67 52.67 68.46 47.21 41.67 54.61|20.00 38.00 45.00 56.00 56.00 58.00|
| | | |

- Table 7: Performance of CK-Pro RL on GAIA-text (n=103) and xbench-DeepSearch (n=100), reported using Pass@1 and Pass@3.

- 4.4 Experiments

Our RL implementation is developed using verl1 as the core RL library, with the agent runtime deployed on GPU clusters. Given the volatility of web environments, we specifically utilize Browserless2 for remote Playwright execution to maximize the stability of web navigation during training.

Table 7 presents the performance of CK-Pro RL on GAIA-text and xbench-DeepSearch. On GAIAtext, the SFT baseline achieves an average Pass@3 of 47.55%. Although the vanilla RL checkpoint was affected by the volatility of the web-agent environment during experiments, reinforcement learning demonstrates clear advantages when combined with inference-time scaling. Specifically, SubgoalGRPO achieves an average Pass@3 of 52.67%, which further improves to 54.61% when integrated with the verifier. The gains are particularly notable on Level 3 tasks, where Subgoal-GRPO Pass@3 + Verifier reaches 41.67%, significantly outperforming both the SFT Pass@1 (8.33%) and Pass@3 (16.70%) results. On xbench-DeepSearch, the benefits of RL are even more pronounced: SubgoalGRPO + Verifier improves the Pass@1 from 20.00% to 45.00% over the SFT baseline, while RL + Pass@3 + Verifier achieves a final score of 58.00%. These results demonstrate that RL substantially enhances the agent’s ability to solve challenging deep research tasks, with performance gains further amplified by inference-time scaling. The comparison between GRPO and our Subgoal-GRPO also demonstrates the effectiveness of finer-grained credit assignment, grounded by milestone achievement scores.

- 5 Related Work

- 5.1 Deep Research Agents

The field of deep research agents has rapidly evolved, driven by the need for autonomous systems capable of conducting complex, multi-step research tasks. These agents leverage large language models (LLMs) and vision-language models (VLMs) to perform tasks such as web navigation, data analysis, code generation, and report synthesis. Below we introduce both close-source and open-source deep research agents.

Proprietary Deep Research Agents Proprietary systems have set a high standard for deep research agents by demonstrating robust performance in autonomous task execution. OpenAI’s Deep Research (OpenAI, 2025) integrates most advanced OpenAI models to autonomously browse the

- 1https://github.com/verl-project/verl
- 2https://www.browserless.io 3During our experiments with CK-Pro RL, the web-agent environment deteriorated due to a firewall policy

update to Bing Search, causing the baseline performance to decline notably. Nevertheless, CK-Pro RL still substantially outperforms the SFT baseline on GAIA-text Pass@3.

web, analyze data, and generate comprehensive reports. Powered by a specialized version of the o3 model, it achieves strong performance on benchmarks like GAIA (67.36% average pass@1 accuracy, 72.57% cons@64 accuracy) and Humanity’s Last Exam (26.6% accuracy), significantly outperforming other models. Google’s Gemini Deep Research (Google DeepMind, 2025) was part of the Gemini 2.5 suite, it autonomously searches hundreds of websites, reasons iteratively, and produces detailed reports, emphasizing real-time adaptability and multimodal processing. Perplexity’s Deep Research (Perplexity AI, 2025) excels in domains like finance, marketing, and technology, achieving 21.1% accuracy on Humanity’s Last Exam and 93.9% on SimpleQA. It iteratively searches, reads documents, and refines research plans. More recent work, Kimi-Researcher (Moonshot AI, 2025a;b), an advanced feature of Moonshot AI’s Kimi platform, excels in delivering precise research outputs for complex queries across diverse domains.

Open-Source Deep Research Frameworks Open-source frameworks have made significant strides in democratizing deep research agents, with notable contributions including Hugging Face’s SmolAgents (Roucher et al., 2025), a lightweight Python library that supports various LLMs for web search and data processing but may lack optimization for complex, multi-step research tasks; Alibaba Tongyi’s WebAgent Framework, comprising WebDancer (Wu et al., 2025a), WebSailor (Li et al., 2025a), and WebShaper (Tao et al., 2025), which excels in super-human reasoning for web-based tasks like BrowseComp, GAIA, and WebWalkerQA (Wu et al., 2025b). OWL (Optimized Workforce Learning; Hu et al., 2025), a hierarchical multi-agent system that leads open-source frameworks with a 69.09% average score on the GAIA benchmark, supporting tools for online search, multimodal processing, browser automation, document parsing, and code execution. TapeAgent (Bahdanau et al.,

- 2024), from ServiceNow, uses a “tape” log to streamline LLM agent development, matching GPT-4o in tasks like form-filling with cost efficiency. AutoAgent (Tang et al., 2025) enables non-technical users to create LLM agents via natural language, achieving 55.15% GAIA accuracy and excelling in multi-agent tasks (Tang et al., 2025). OAgent (Zhu et al., 2025), an open-source platform, supports modular agent building for reasoning and automation but may rely on proprietary tools.

In all, open-source frameworks for deep research agents lag behind proprietary systems in performance and accessibility. Although some open-source agents demonstrate competitive results, they often depend on proprietary tools, limiting their reproducibility. Furthermore, research on open-source agent foundation models remains underexplored, as most efforts rely on prompting external APIs. In this work, we address these gaps by developing a fully open-source framework and model, leveraging (to the maximum extent) freely available tools to enhance accessibility and performance.

5.2 Process Credit Assignment and Multi-Reward Optimization in Agentic RL

Recent GRPO-style reinforcement learning for LLM agents has progressed along two largely complementary directions. One line of work addresses the credit-assignment problem in long-horizon agent trajectories, where supervision is often available only at the final outcome. The other studies how to optimize effectively once multiple intermediate rewards or objectives are already available. This distinction is useful for situating our method: unlike approaches that must infer process supervision indirectly from sparse outcomes, we assume explicit hierarchical subgoal structure and focus on how to preserve these structured signals during optimization.

Process Credit Assignment from Sparse Outcome Rewards. A major line of recent work seeks to transform sparse trajectory-level rewards into denser step-level supervision. TreeRPO (Yang et al.,

- 2025) does so through explicit tree sampling: for an intermediate reasoning step, its pseudo-reward is estimated from the expected final rewards of descendant branches, avoiding a separate step reward model while providing denser feedback. ARPO (Dong et al., 2025b) makes this idea more practical for agentic settings by allocating additional exploration to high-entropy states, especially after tool interactions, and by introducing advantage attribution estimation for stepwise tool-use decisions. AEPO (Dong et al., 2025a) further argues that entropy-guided branching can itself become unstable; it therefore balances rollout allocation across global and branch sampling, penalizes

consecutive high-entropy tool-call steps, and modifies the clipping objective with a stop-gradient design to preserve learning signals on high-uncertainty tokens. Orthogonal to tree-based expansion, GiGPO (Feng et al., 2025) remains critic-free and avoids auxiliary reward models by grouping repeated environment states across trajectories through anchor state grouping, enabling micro-level relative advantage estimation for actions taken from the same state. Finally, CSO (Li et al., 2026) adopts a selective verification-based strategy: instead of scoring every step, it identifies candidate critical steps in failed trajectories, proposes alternative actions with stronger models, and keeps only those alternatives that the current policy can successfully roll out to a correct outcome, thereby constructing high-quality local preference data. Collectively, these methods show several ways to induce process supervision when explicit subgoal rewards are unavailable.

Multi-Reward Optimization with Explicit Intermediate Signals. A complementary line of work assumes multiple reward components are already available and studies how to combine them effectively in GRPO-style optimization. MO-GRPO (Ichihara et al., 2025) introduces variance-based reweighting to prevent over-optimizing easier objectives at the expense of others. Related MultiReward GRPO variants (Deng & Ji, 2025; Zhong et al., 2025) combine weighted rule-based and model-based rewards in applications such as text-to-speech and debiasing. However, these methods largely follow a “weight-then-sum” or “sum-then-normalize” design, which can blur fine-grained supervision when high-variance outcome rewards dominate auxiliary signals. Most relevant to our work is GDPO (Liu et al., 2026), which identifies a reward-collapse phenomenon in multi-reward GRPO: distinct reward combinations can map to identical advantages when rewards are summed before normalization. By decoupling normalization across reward components before aggregation, GDPO preserves finer-grained signal differences and improves training stability. Our work is closest to this line: rather than inferring process rewards from sparse outcomes, we exploit explicit hierarchical subgoal rewards and optimize them without collapsing their distinct training signals.

#### 6 Conclusion

In this work, we introduce Cognitive Kernel-Pro, a fully open-source generalist agent framework that maximizes the use of free tools, achieving state-of-the-art performance on the GAIA benchmark among open-source, free-tool agents while remaining competitive with frameworks relying on proprietary tools. Additionally, we explore the training of an open-source agent foundation model within this framework, developing an 8B-based model that surpasses previous counterparts such as WebDancer and WebSailor. Furthermore, we improve agent performance through reinforcement learning, leveraging multi-reward optimization with subgoal-conditioned GRPO. Future efforts will concentrate on advancing more capable, multimodal agent foundation models to address increasingly complex tasks.

#### References

Dzmitry Bahdanau, Nicolas Gontier, Gabriel Huang, Ehsan Kamalloo, Rafael Pardinas, Alex Piché, Torsten Scholak, Oleh Shliazhko, Jordan Prince Tremblay, Karam Ghanem, Soham Parikh, Mitul Tiwari, and Quaizar Vohra. Tapeagents: a holistic framework for agent development and optimization. arXiv preprint arXiv:2412.08445, 2024. Published: 2024-12-11, Accessed: 2025-07-25.

Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 7b tir. https://huggingf ace.co/AI-MO/NuminaMath-7B-TIR, 2024.

Yixuan Deng and Xiaoqiang Ji. Multi-reward grpo fine-tuning for de-biasing large language models: A study based on chinese-context discrimination data. arXiv preprint arXiv:2511.06023, 2025.

Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng

Dou. Agentic entropy-balanced policy optimization, 2025a. URL https://arxiv.org/abs/2510

###### .14545.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization, 2025b. URL https://arxiv.org/abs/2507.19849.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training, 2025. URL https://arxiv.org/abs/2505.10978.

Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094, 2024.

Google DeepMind. Gemini deep research — your personal research assistant. https://gemini.goo

gle.com, 2025. Accessed: 2025-07-25.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 6864–6890. Association for Computational Linguistics, 2024a. doi: 10.18653/V1/2024.ACL-LONG.371. URL https://doi.org/10.18653/v1/2024.acl-long.371.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Hongming Zhang, Tianqing Fang, Zhenzhong Lan, and Dong Yu. Openwebvoyager: Building multimodal web agents via iterative real-world exploration, feedback and optimization. CoRR, abs/2410.19609, 2024b. doi: 10.48550/ARXIV.241 0.19609. URL https://doi.org/10.48550/arXiv.2410.19609.

Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, and Guohao Li. Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation, 2025. URL https://arxiv.org/abs/2505.23885.

Yuki Ichihara, Yuu Jinnai, Tetsuro Morimura, Mitsuki Sakamoto, Ryota Mitsuhashi, and Eiji Uchibe. Mo-grpo: Mitigating reward hacking of group relative policy optimization on multi-objective problems. arXiv preprint arXiv:2509.22047, 2025.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025a. URL https://arxiv.org/abs/2507.02592.

Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, Haitao Mi, and Dong Yu. Verified critical step optimization for llm agents, 2026. URL https://arxiv.org/abs/ 2602.03412.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. CoRR, abs/2501.05366, 2025b. doi: 10.48550/ARXIV.2501.05366. URL https://arxiv.org/abs/ 2501.05366. Accessed: 2025-07-26.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability, 2025c. URL https://arxiv.org/abs/2504.21776.

Bill Yuchen Lin, Ziyi Wu, Yichi Yang, Dong-Ho Lee, and Xiang Ren. Riddlesense: Reasoning about riddle questions featuring linguistic creativity and commonsense knowledge. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 1504–1515, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.131. URL https://aclanthology.org/2021.fi ndings-acl.131.

Hanmeng Liu, Zhiyang Teng, Leyang Cui, Chaoli Zhang, Qiji Zhou, and Yue Zhang. LogiCoT: Logical chain-of-thought instruction tuning. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 2908–2921, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emn lp.191. URL https://aclanthology.org/2023.findings-emnlp.191/.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization, 2026. URL https://arxiv.org/abs/2601.05242.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: a benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=fibxvahvs3.

Monica.Im. Manus ai. Technical report, Monica.Im, 2025. URL https://manus.im/. Moonshot AI. Kimi-k2. https://github.com/MoonshotAI/Kimi-K2, 2025a. Published: 2025-07-11,

Accessed: 2025-07-25. Moonshot AI. Kimi-researcher: End-to-end rl training for emerging agentic capabilities. https:

//moonshotai.github.io, 2025b. Published: 2025-06-20, Accessed: 2025-07-25.

OpenAI. Introducing deep research. Technical report, OpenAI, 2025. URL https://openai.com/i

###### ndex/introducing-deep-research/.

Perplexity AI. Introducing perplexity deep research. https://www.perplexity.ai/hub/blog/int

roducing-perplexity-deep-research, 2025. Published: 2025-02-14, Accessed: 2025-07-25.

Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/hugging face/smolagents, 2025.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning,

###### 2025. URL https://arxiv.org/abs/2503.05592.

Jiabin Tang, Tianyu Fan, and Chao Huang. Autoagent: A fully-automated and zero-code framework for llm agents. arXiv preprint arXiv:2502.05957, 2025. Published: 2025-02-18, Accessed: 2025-07-25.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webshaper: Agentically data synthesizing via information-seeking formalization. https://arxiv.org/abs/ 2507.15061, 2025. Published: 2025-07-20, Accessed: 2025-07-25.

Yuxuan Wan, Tianqing Fang, Zaitang Li, Yintong Huo, Wenxuan Wang, Haitao Mi, Dong Yu, and Michael R. Lyu. Inference-time scaling of verification: Self-evolving deep research agents via test-time rubric-guided verification, 2026. URL https://arxiv.org/abs/2601.15808.

Rui Wang, Ce Zhang, Jun-Yu Ma, Jianshu Zhang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu, and Kam-Fai Wong. Explore to evolve: Scaling evolved aggregation logic via proactive online exploration for deep research agents, 2025. URL https://arxiv.org/abs/2510.14438.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webdancer: Towards autonomous information seeking agency, 2025a. URL https://arxiv.org/abs/2505.22648.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. Webwalker: Benchmarking llms in web traversal. CoRR, abs/2501.07572, 2025b. doi: 10.48550/ARXIV.2501.07572. URL https: //doi.org/10.48550/arXiv.2501.07572.

Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xinrun Du, Di Liang, Daixin Shu, Xianfu Cheng, Tianzhen Sun, Guanglin Niu, Tongliang Li, and Zhoujun Li. Tablebench: A comprehensive and complex benchmark for table question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 25497–25506, 2025c. Preprint available at arXiv:2408.09174.

Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, and Jing Tang. Treerpo: Tree relative policy optimization, 2025. URL https://arxiv.org/abs/2506.05183.

Hongming Zhang, Xiaoman Pan, Hongwei Wang, Kaixin Ma, Wenhao Yu, and Dong Yu. Cognitive kernel: An open-source agent system towards generalist autopilots. CoRR, abs/2409.10277, 2024. doi: 10.48550/ARXIV.2409.10277. URL https://doi.org/10.48550/arXiv.2409.10277.

Yicheng Zhong, Peiji Yang, and Zhisheng Wang. Multi-reward grpo for stable and prosodic singlecodebook tts llms at scale, 2025. URL https://arxiv.org/abs/2511.21270.

He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Jiaheng Liu, and Wangchunshu Zhou. Oagents: An empirical study of building effective agents, 2025. URL https://arxiv.org/abs/2506.15741.

Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. Docbench: A benchmark for evaluating llm-based document reading systems, 2024. URL https://arxiv.org/abs/2407.10701.

#### A Technical Details of Cognitive Kernel-Pro Framework

Code-based Action and Tool-using. Both the main agent and the sub-agents employ a similar multi-step workflow for their problem-solving process. We utilize code-based actions: all actions, including sub-agent and tool invocations, are defined as Python functions. The agents generate code to call these functions, which are then executed directly to perform the corresponding actions (using Python’s built-in exec function). To capture the outputs from action execution, we use Python’s built-in print function; these outputs are subsequently fed into later steps to provide intermediate results. Here are the instructions with regard to the action output format:

ACTION OUTPUT FORMAT

## Output Please generate your response, your reply should strictly follow the format:

- • Thought: Provide an explanation for your action in one line. Begin with a concise review of the previous steps to provide context. Next, describe any new observations or relevant information obtained since the last step. Finally, clearly explain your reasoning and the rationale behind your current output or decision.
- • Code: Output your python code blob for the next action to execute. Remember to wrap the code with markdown python code marks and print your output.

State-enhanced Problem-solving Workflow. In addition to code-based actions, our workflow incorporates explicit planning and state management. Specifically, before each action decision, the agent adopts a planning step, formulating plans based on previous steps and the latest observations.

- A crucial mechanism in this process is the maintenance of a progress state, which records summaries of important information from previous steps, including intermediate results and lessons learned from earlier attempts. This progress state offers concise historical context and guides subsequent actions. The following instructions detail the structure of the progress state for the main agent:

PROGRESS STATE

## Progress State The progress state is crucial for tracking the task’s advancement and includes:

- • completed_list (List[str]): A list of completed steps and gathered information essential for achieving the final goal.
- • todo_list (List[str]): A list of planned future steps; aim to plan multiple steps ahead when possible.
- • experience (List[str]): Summaries of past experiences and notes, such as failed attempts or special tips, to inform future actions.
- • information (List[str]): A list of collected important information from previous steps. These records serve as the memory and are important for tasks such as counting (to avoid redundancy). Here is an example progress state for a task to locate and download a specific paper for analysis:

{

‘completed_list’: [‘Located and downloaded the paper (as paper.pdf) using the web agent.’, ‘Analyze the paper with the document agent.’],

‘todo_list’: [‘Perform web search with the key words identified from the

paper.’], ‘experience’: [], ‘information’: [‘The required key words from the paper are AI and NLP.’]

}

Unified Multi-module Communication. A key aspect of our multi-module system design is the specification of communication between the main agent and the sub-agents. To ensure simple and robust communication, we adopt a unified and minimal text-based interface for all sub-agent calling. Each sub-agent is implemented as a callable function following the protocol below:

- • Input: The sub-agent accepts an input argument of “task”, which is a plain string describing the sub-task assigned to it. Optionally, the sub-agent may accept additional arguments specific to its functionality (e.g., file paths for the file agent).
- • Output: The sub-agent returns a dictionary with two fields: “output”, a string containing the well-formatted answer that strictly adheres to any specified output format; and “log”, a string providing supplementary notes, such as steps taken, issues encountered, or relevant context.
- • Definition: To enable the main agent to understand the utilities and use cases of each sub-agent, all sub-agents provide a Python docstring-style definition, which is provided to the main agent. For example, the definition of the web agent is as follows:

WEB-AGENT DEFINITION

def web_agent(task: str) → dict:

""" Employs a web browser to navigate and interact with web pages to accomplish a specific task.

Args:

task (str): A detailed description of the task to perform. This may include: 1) The target website(s) to visit (include valid URLs); 2) Specific output formatting requirements; 3) Instructions to download files (specify desired output path if needed).

Returns:

dict: A dictionary with the following structure: ‘output’ (str): The well-formatted answer, strictly following any specified output format; ‘log’(str): Additional notes, such as steps taken, issues encountered, or relevant context.

Notes:

- - If the ‘task‘ specifies an output format, ensure the ‘output’ field matches it exactly.
- - The web agent can download files, but cannot process or analyze them. If file analysis is required, save the file to a local path and return control to an external planner or file agent for further processing.

Example:

>>> answer = web_agent(task=‘What is the current club of Messi? (Format your output directly as club_name.)’)

>>> print(answer) """

With these unified input/output definitions, our system can flexibly manage interactions and collaboration between the main agent and sub-agents, facilitating extension to a wide range of processing scenarios.

- B Details of Agent-Based Data Construction We present the key prompt templates for agent-based data synthesis.

DATA SYNTHESIZING REQUIREMENTS

- • Source-Based Queries: Each query must be based on verifiable sources of truth (e.g., Wikipedia, arXiv, Papers With Code, GitHub, or a specific downloadable file whose location is unambiguous). Clearly specify the sources within the query to avoid ambiguity.
- • Cross-Source Reasoning: Combine information from multiple sources to formulate a challenging and interesting query. The answer should require synthesis, not simple lookup.
- • Novelty Requirement: The answer must not exist verbatim on the internet. Construct queries that require combining facts or data in a way that produces a new, non-trivial answer.
- • Stable & Unambiguous Answers: The answer should be a number or at most a few

- words, concise and unambiguous. Avoid queries whose answers may change over time or due to data updates.
- • Self-Containment: The query must be fully self-contained, requiring no external context or references beyond what is provided in the query itself. All necessary details must be included to ensure only one correct answer.
- • Clarity & Precision: Ensure the query is clear and precise, specifying all necessary details to avoid multiple interpretations. Clearly state the expected answer format within the query.
- • Minimal Procedural Detail: Do not include step-by-step instructions or detailed procedures in the query. Focus on the information need, not the process.
- • Annotator Feasibility: The query should be answerable in a reasonable amount of time by a human annotator.
- • Interest & Utility: The query should be interesting and useful – answering it should provide value and demonstrate the assistant’s ability to synthesize and reason across sources.
- • Multi-Ability Requirement: Queries are encouraged to require the agent to use multiple abilities, such as Web Browsing, File Handling and Multi-Modal Processing.

SEED TOPICS

Notable open-source projects in natural language processing (GitHub, Papers With Code) The evolution of jazz music in the 20th century (Smithsonian Institution, Wikipedia) Key literary works of the 19th century (Project Gutenberg, Wikipedia) Advances in space exploration since 2000 (NASA, Wikipedia) The history and cultural significance of the Olympic Games (Olympic.org, Wikipedia) Overview of major world languages and their distribution (Ethnologue, Wikipedia)

HINT-BASED QUERY AUGMENTATION

{Original_Query} <secret> Below are some confidential hints for your reference:

{Hint} Important Instructions:

- • Do not disclose or imply in any way that you have access to these hints during your problem-solving or reasoning process.
- • A strict evaluator will review your entire solution. If your output suggests you relied on these hints, you will be disqualified from your role as a problem-solving agent.
- • For any sub-problems where you do not know the answer, continue to use appropriate tools and sub-agents as if you are unaware of the hints.
- • If there is a conflict between information obtained from your tools and the provided hints, always prioritize the information from your tools.
- • Do not attempt to plan everything in advance or act as if you have privileged foresight.
- • Remember, maintaining this role is crucial – do not risk your position by revealing or depending on the hints. Proceed with utmost caution and professionalism. </secret>

