## Benchmark Test-Time Scaling of General LLM Agents

Xiaochuan Li1 Ryan Ming1 Pranav Setlur1 Abhijay Paladugu1 Andy Tang1 Hao Kang1 Shuai Shao2 Rong Jin2 Chenyan Xiong1

# arXiv:2602.18998v1[cs.AI]22Feb2026

### Abstract

LLM agents are increasingly expected to function

- as general-purpose systems capable of resolving open-ended user requests. While existing benchmarks focus on domain-aware environments for developing specialized agents, evaluating generalpurpose agents requires more realistic settings that challenge them to operate across multiple skills and tools within a unified environment. We introduce General AgentBench, a benchmark that provides such a unified framework for evaluating general LLM agents across search, coding, reasoning, and tool-use domains. Using General AgentBench, we systematically study test-time scaling behaviors under sequential scaling (iterative interaction) and parallel scaling (sampling multiple trajectories). Evaluation of ten leading LLM agents reveals a substantial performance degradation when moving from domain-specific evaluations to this general-agent setting. Moreover, we find that neither scaling methodology yields effective performance improvements in practice, due to two fundamental limitations: context ceiling in sequential scaling and verification gap in parallel scaling. Code is publicly available
- at https://github.com/cxcscmu/Gen eral-AgentBench.

### 1. Introduction

Agents powered by large language models (LLMs) are at a turning point, transitioning from domain-specific (Liu et al., 2023b; Yang et al., 2024a; Yue et al., 2024) to generalpurpose (Xi et al., 2023; Luo et al., 2025). Real-world user requests are often open-ended and require LLM agents to operate end-to-end through planning (Wang et al., 2023; Erdogan et al., 2025), reasoning (Wei et al., 2022; Yao et al.,

1Language Technologies Institute, School of Computer Science, Carnegie Mellon University 2Meta. All experiments, data collection, and processing activities were conducted by Carnegie Mellon University. Meta was involved solely in an advisory role.. Correspondence to: Xiaochuan Li <xiaochu4@andrew.cmu.edu>.

Preprint. February 24, 2026.

2022b; Parmar et al., 2025), and tool use (Schick et al., 2023; Patil et al., 2024) under uncertain and evolving conditions. A capable general agent is expected to compose multiple skills and tools (e.g., search, coding, computation, and MCP APIs) to handle the diversity of realistic requests(Anthropic, 2026), while exhibiting effective test-time scaling abilities to address increasing task complexity and enhance response quality (Wang et al., 2022; Brown et al., 2024; Snell et al., 2024). This shift raises an important evaluation gap: beyond asking “can the model solve a task,” we must assess whether agents can infer user intent, select specific tools, and scale up their performance under a unified evaluation framework across diverse domains.

Existing agentic benchmarks typically evaluate LLM agents in domain-specific settings, where the environment and available toolsets are explicitly designed for a particular task category (e.g., software engineering with a Docker environment and terminal tools (Jimenez et al., 2023; Aleithan et al., 2024), or web navigation with a browsing interface (Zhou et al., 2023b; He et al., 2024; Wei et al., 2025)). Conversely, actual user interactions are rarely so constrained; they typically involve multi-turn, open-ended requests spanning disparate domains, requiring agents to remain ready across a broad toolset to handle unpredictable queries. As a result, while current benchmarks are informative for domainspecific agent development, they may not fully capture the demands of real-world usage and can overestimate robustness under realistic, multi-domain conditions.

In this paper, we address this gap by introducing General AgentBench, a benchmark designed to evaluate generalpurpose agents across diverse scenarios under a unified framework that more closely reflects real-world user interactions. We consolidate tools from all domains into a shared interface that is consistently exposed to evaluated agents across different tasks, while domain-specific environments and implementations remain hidden. We evaluate ten leading LLM-based agents, each of which must first interpret the user request, then choose suitable tools from a large and diverse tool pool, and iteratively interact with the environment until producing a final response.

For complex requests that exceed the capabilities of a short interaction horizon, agents can benefit from increased

Tau2-Bench

Expected

MathHay

MCP-Bench

past@K (Growing)

Context Ceiling

Actual

58%

6%

34%

SuccessRate

Accuracy

Webvoyager

SWEBench

16%

39%

Verifiability Gap

Fluctuation & Decline

Improve

20%

10% 22%

Self-Choice (Flat)

Mind2Web2

Terminal-

Bench BrowseComp

Number of Samples (K)

Interaction Turns / Computation

Standard Setting General Setting

(a) Performace comparsion.

(b) Sequential test-time scaling.

(c) Parallel test-time scaling.

Figure 1. Evaluating general LLM agents under a realistic user-interaction scenario. A: GPT-5’s performance drop under General AgentBench compared to static, domain-specified evaluation. B: Sequential test-time scaling via longer interaction histories can lead to unstable or degraded performance. C: While correct solutions increasingly appear in the generation space (past@K), agents often fail to select them, revealing a verification gap.

inference-time computation—a strategy known as test-time scaling and extensively studied in the context of non-agentic reasoning (Cobbe et al., 2021; Zelikman et al., 2024; Guo et al., 2025). We focus on two primary scaling strategies: (1) sequential scaling, which extends interaction histories to support continued reasoning, reflection, and exploration; and (2) parallel scaling, which independently samples K candidate trajectories and selects a single candidate to return. More concretely, effective parallel scaling requires agents not only to generate correct solutions, but also to reliably identify and choose the correct one, since real-world agents cannot present multiple responses simultaneously. Together, these settings enable a systematic study of test-time scaling behaviors in general LLM agents.

Our results lead to three key conclusions. (1) Across ten leading LLMs, we observe a substantial performance drop when moving from domain-specific configurations to the general-agent setting (Figure 1a), with pronounced differences in robustness across model families. Among them, Claude exhibits the strongest robustness, while most other models experience performance drops of approximately 30%. (2) Sequential scaling exhibits an effective context length ceiling (context ceiling): while performance improves within a modest range of additional interaction turns, it often fluctuates or degrades thereafter. This suggests that the accumulated history eventually overwhelms the agent’s reasoning capacity, leading to instability in long-horizon tasks. (Figure 1b). (3) Although parallel scaling increases the theoretical upper bound of performance (past@K), we consistently observe a gap between this upper bound and self-choice accuracy, revealing a substantial verification gap that ultimately limits achievable performance in realistic settings (Figure 1c). In summary, our contributions are:

- • General AgentBench for Realistic Evaluation. We introduce General AgentBench, a benchmark for evaluating whether agents can compose multiple skills and tools to solve open-ended requests from diverse domains under a unified framework, more closely reflecting real-world user interactions.
- • Sequential Test-time Scaling in General Agents. We study the sequential test-time scaling behavior of general agents, showing that performance improvements are bounded by an effective context ceiling, beyond which additional computation often leads to instability and performance degradation. This inherent point varies across models and domains.
- • Parallel Test-time Scaling. We analyze parallel testtime scaling and show that, despite increasing the theoretical performance upper bound (past@K), its practical gains are limited by a verification gap between generation and model self-choice accuracy.

### 2. General AgentBench

In this section, we introduce the construction of General AgentBench (Section 2.1) and the unified evaluation framework (Section 2.2 ). Detailed prompt templates, tool specifications, and the unified policy are deferred to the Appendix E.

#### 2.1. Domains and Sources

Our benchmark spans four task domains: Coding, Search, Tool-use, and Reason. These domains reflect common real-world applications such as software engineering, information seeking, service workflows, and analytical reasoning, positioning General AgentBench as an initial step toward

evaluating general-purpose agents in open-ended and unified settings. Table 1 summarizes the benchmark composition.

###### Host Server Side

|[Figure 1]<br><br>Database<br><br>Search Data<br><br>Code Data<br><br>Reason Data<br><br>Tool-Use Data|
|---|

|[Figure 2]<br><br>Search Client| |
|---|---|
| | |

[Figure 3]

###### Search Server

|[Figure 4]<br><br>Code Client|
|---|

|[Figure 5]<br><br>Code Server|
|---|

Table 1. Composition of General AgentBench

[Figure 6]

|[Figure 7]<br><br>Omni Agents| |
|---|---|
| | |

Domain Dataset Original Sampled Search

|[Figure 8]<br><br>Reason Client|
|---|

|[Figure 9]<br><br>Reason Server|
|---|

###### Router

BrowseComp 1266 124 WebVoyager 643 65

|[Figure 10]<br><br>Tool-Use Client|
|---|

|[Figure 11]<br><br>Tool-Use Server|
|---|

SWE-Bench Verified 500 50 Terminal-Bench 230 80

Coding

[Figure 12]

|[Figure 13]<br><br>Unified Toolbox|
|---|

Servers

Reason MathHay 602 75 Tool-Calling

Tau2-Bench 278 50 MCP-Bench 104 52

Figure 2. Illustration of how General AgentBench covers a wide range of task categories while providing a unified interface to simulate real-world user interactions. The green region indicates the specific task currently being handled by the agent (e.g., a search task). Orange boxes denote other clients and servers that remain active and responsive but are not directly involved in the current interaction. Red indicates that other domain-specific data are excluded.

Coding We include tasks from SWE-Bench Verified (OpenAI, 2024b) and Terminal Bench, which evaluate an agent’s ability to analyze production-level software issues, reason over long instructions, and iteratively interact with execution environments to reach a correct final state.

we adopt the Model Context Protocol (MCP) (Anthropic, 2024) as the backbone of our framework. Each benchmark environment is instantiated as an MCP server, while all servers are centrally managed by a unified Host. The Host maintains a global tool registry that records all available tools and their corresponding server mappings, presenting the agent with a single, unified tool space across all domains.

Search The search domain includes tasks from BrowseComp (Wei et al., 2025) and WebVoyager (He et al., 2024). These benchmarks assess an agent’s ability to identify missing information, decide when additional search steps are needed, and navigate long, evolving web contexts, going beyond static retrieval or single-turn question answering.

Centralized interaction abstraction. The Host serves as the sole interaction interface for the general agent, abstracting away individual benchmark implementations. When the agent invokes a tool, the Host resolves the call via the tool registry and routes the request to the appropriate server for execution.

Tool-use For tool-use, we adopt Tau2-Bench (Barres et al., 2025) and MCP-Bench (Wang et al., 2025b), both of which provide rich tool suites requiring models to select, invoke, and coordinate multiple tools. These tasks emphasize structured tool calling and multi-step planning in realistic service and workflow scenarios.

Evolving interaction context. Because all tools and benchmark environments are exposed simultaneously, the unified tool descriptions alone can span tens of thousands of tokens. When combined with user queries and accumulated multi-turn interaction histories, the resulting context naturally grows into the long-context regime. In this setting, agents must reason over heterogeneous information sources, including task instructions, tool documentation, execution feedback, and their own prior decisions. This distinguishes agentic interaction from many existing long-context benchmarks that focus on static, single-turn question answering or summarization with short outputs. We provide further long-context study in Appendix A.3

Reason For long-context reasoning, we use MathHay (Wang et al., 2024a), which constructs queries by embedding relevant mathematical documents into noisy longcontext haystacks. This benchmark isolates sustained reasoning over long inputs without relying on external tool execution, complementing the other domains.

#### 2.2. Unified Realistic Evaluation Framework

To support realistic evaluation of general LLM agents, we design a unified framework that exposes all tasks and tools through a shared interaction interface. These choices reflect three fundamental properties of real-world agent usage: cross-domain task diversity, comprehensive skill requirements, and dynamically evolving multi-turn interactions. An overview of the framework is illustrated in Figure 2.

Execution process. For each evaluation instance, the framework provides it to the agent together with the unified policy and toolset as the context. All benchmark servers (e.g., Docker-based environments in the coding domain) are instantiated simultaneously and remain idle while awaiting requests from the agent. When the agent issues a tool call,

Unified tool interface. In practical deployments, agents must select appropriate tools from a large pool without prior knowledge of task domains. To reflect this setting,

Table 2. Main results on General AgentBench. Benchmarks are grouped by domain. Avg. denotes the mean score across all available benchmarks for each model. Bold indicates the best score.

Search Code Reason Tool-use

Models

Avg. BrowseComp WebVoyager SWE-Bench Terminal-Bench MathHay Tau2-Bench MCP-Bench

Open-Source

GPT-OSS-120B 4.0 27.7 12.0 6.3 38.7 26.0 63.3 25.4 Qwen3-235B-A22B 8.9 30.8 20.4 23.8 32.0 38.3 66.1 31.5 Qwen3-Next 10.5 35.4 18.0 8.8 42.0 48.9 64.6 32.6 DeepSeek-V3.2 19.4 46.2 31.8 22.2 33.3 54.0 66.0 39.0 DeepSeek-R1 9.7 43.1 14.0 8.8 46.7 17.1 62.2 28.8 Proprietary

Gemini 2.5-Flash 6.5 32.3 14.0 20.0 36.0 38.3 66.6 30.5 Gemini 2.5-Pro 8.9 46.2 26.0 27.5 24.0 46.0 67.2 35.1 Claude Haiku 4.5 17.7 47.7 56.0 25.0 34.7 44.0 69.0 42.0 Claude Sonnet 4.5 23.1 56.9 54.0 45.0 36.0 48.0 72.9 48.0 GPT-5 27.4 61.5 36.0 41.3 64.0 32.0 59.1 45.9

the Host routes the request to the corresponding server, executes the tool, and returns the result in a unified response format. Tool execution is decoupled from task type: even if a task is search-oriented, code-related tool calls can still be executed by the environment, returning valid outputs despite having no direct relevance to the final solution. This design intentionally exposes the agent to a realistic setting where incorrect or irrelevant tool usage remains possible.

The agent interacts with the framework over multiple turns until producing a final answer. During interaction, we monitor execution signals (e.g., terminal outputs) and regulate the interaction budget, enabling additional computation or extended reasoning when applicable (Section 4.1). The final answer is then forwarded to the corresponding benchmark server for correctness evaluation.

#### 2.3. Experimental details

Our evaluation covers a total of ten frontier language models. Among open-source models, we include several highperforming systems such as Qwen3-235B (Yang et al., 2025) and DeepSeek-R1 (Guo et al., 2025), as well as more recent models with novel attention mechanisms, including Qwen3Next (Qwen Team, Alibaba Inc., 2025) and DeepSeek-v3.2 (Liu et al., 2025). For proprietary models, we consider both efficiency-oriented variants (e.g., Gemini 2.5 Flash (Comanici et al., 2025)) and models optimized for complex reasoning (e.g., GPT-5 (OpenAI, 2025a) and Claude Sonnet

###### 4.5 (Anthropic, 2025a)). We access these models via Amazon Bedrock 1 and the Hugging Face Inference API .2 For all evaluations, we fix the decoding temperature to 0.7 and ensure that each model’s native context length exceeds the maximum context length required by the benchmark.

- 1https://aws.amazon.com/bedrock/pricing/
- 2https://huggingface.co/docs/inference-p

roviders/en/index

### 3. Main Results

In this section, we report overall performance on General AgentBench and compare it against evaluations conducted under prior domain-specific settings to quantify the gap between specialized and general-purpose LLM agents.

#### 3.1. Result analysis

Table 2 summarizes the results across models and domains on General AgentBench. Claude Sonnet 4.5 achieves the strongest overall performance, driven primarily by its tooluse and coding capabilities, while GPT-5 attains the highest scores in the Search and Reason domains, reflecting its strengths in information retrieval and complex reasoning. Among open-source models, DeepSeek-V3.2 outperforms both Gemini variants, demonstrating the significant scaling potential of efficient, sparse-attention architectures. Across models, performance on BrowseComp remains consistently low, indicating that retrieving rare and precise information beyond in-domain training data is still a major bottleneck for current LLM agents.

We further examine how performance changes when models transition from specialized agents operating under domainspecific contexts to general agents acting within a unified environment with shared toolsets. Figure 4 summarizes the mean degradation aggregated over all domains, while Figure 3 reports the relative performance change for each agent across domains. Most LLM agents experience substantial degradation in the general-agent setting, with average relative drops ranging from 10% to 30%. The magnitude of this degradation varies widely: for example, Gemini 2.5Pro suffers a drop exceeding 60% in the Reason domain, falling from top-tier performance in the baseline setting to near-average performance as a general agent. In contrast, Claude Sonnet 4.5 remains notably robust, with only a 0.2% average degradation. Detailed overall results can be found

Baseline (B) General (G)

###### Search

###### Code

###### Reasoning

###### Tool-Call

100

70

80

60

Open-Source Proprietary

Open-Source Proprietary

Open-Source Proprietary

Open-Source Proprietary

70

60

-1.4%

-29.9%

0.0%

-41.5%

80

50

-60.8%

-40.0%

-13.4%

-16.8%

60

-36.6%

-11.1%

50

-20.2%-14.1%

-31.0%

-12.1%

-16.1%-10.1% -2.8%

40

-37.8%

-1.3%

50

-17.1%

60

-29.4%

-4.5%

Score

Score

Score

Score

40

-19.8%

+33.0%

-17.3%

-41.0%

40

-10.3%

+12.5%

30

0.0%

-4.0%

+18.6%

30

-52.2%

-44.8%

40

-7.3%

30

-24.4%

+27.7%-27.0%

20

-49.7%

+18.6%

-31.3%

20

20

20

10

10

10

0

0

0

0

OSS-120BQwen3235BQwen3NextDeepSeekV3.2DeepSeek

OSS-120BQwen3235BQwen3NextDeepSeekV3.2DeepSeek

OSS-120BQwen3235BQwen3NextDeepSeekV3.2DeepSeek

OSS-120BQwen3235BQwen3NextDeepSeekV3.2DeepSeek

R1Gemini

Gemini 2.5-Pro Haiku4.5Sonnet4.5GPT-5

R1Gemini

Gemini 2.5-Pro Haiku4.5Sonnet4.5GPT-5

R1Gemini

Gemini 2.5-Pro Haiku4.5Sonnet4.5GPT-5

R1Gemini

Gemini 2.5-Pro Haiku4.5Sonnet4.5GPT-5

2.5-Flash

2.5-Flash

2.5-Flash

2.5-Flash

- Figure 3. Relative performance change across domains from the Baseline (B) specialized agent setting to the general agent (G) setting with unified context and tools. Negative values indicate performance degradation under the General AgentBench.

10

20

30

40

50

60

70

AverageScore

AvgBaseline AvgGeneral

| |
|---|

OSS-120B Qwen3235B Qwen3NextDeepSeekV3.2DeepSeek

R1 Gemini

2.5-Flash

Gemini 2.5-Pro

Haiku4.5 Sonnet4.5 GPT-5

0

10

20

30

40

%(RelativeChange)

Mean Relative Degradation

-28.7%

-25.5%

-10.4% -9.5% -9.8%

-31.2%

-27.2%

-15.2%

-0.2%

-22.7%

- Figure 4. Performance comparison between specialized-agent and general-agent settings.Top: Absolute performance .Bottom: Relative performance degradation under the general-agent setting.

Case Study: Retrieving the Latest HF Model Task. Identify the most recent pre-trained language model on HuggingFace suitable for text classification.

Plain Web Search Baseline (6 turns)

- T1: <search> latest Hugging Face pre-trained language model text classification 2024 </search>
- T2: <search> Hugging Face text classification models sorted by recent trending 2024 </search>
- T3: <search> Hugging Face newest text classification models January 2025 recently released </search>

... T6: <answer> tabularisai/multilingual-sentiment-

analysis (fine-tuned DistilBERT) </answer>

###### General Agent (3 turns)

- T1: Hugging Face search-models( {"tags": "text-classification", "limit": 15})

→ [{"id": "FutureMa/Qwen3-4B-Evasion", "download": 15}, {"id": "BAAI/bge-reranker-v2-m3", "download": 2807076}, ...]

- T2: search web search("ModernBERT HF 2025")

→ "ModernBERT... 8192-token context, 139M/395M params, trained on 2T tokens"

- T3: Hugging Face get-model-info( {"model id": "answerdotai/ModernBERT-base"})

in Appendix A.2.

#### 3.2. Cross-domain tool usage

→ Full model card and architecture details

Interestingly, several models, including Qwen3-Next, Deepseek-R1, and Claude, exhibit performance gains in the Search domains under the general-agent setting. Trajectory-level analysis shows that these improvements arise from effective cross-domain tool usage, where agents repurpose tools beyond their originally intended domains to support reasoning and information retrieval. We take a closer look at these behaviors. Analysis of 189 search task traces from Claude Sonnet 4.5 reveals that 26% of tasks (50/189) utilized specialized domain tools beyond plain web search. The most frequently used specialized tools include Google Maps APIs (78 calls), Paper Search across arXiv, PubMed, and Google Scholar (60 calls), and Hugging Face model APIs (36 calls). We present a case study demonstrating how domain-specific tools outperform plain web search.

We observe that the plain search baseline iteratively refines web queries across 6 turns, ultimately finding

tabularisai/multilingual-sentiment-analysis

with only surface-level information (“fine-tuned DistilBERT”). In contrast, the General agent system’s specialized Hugging Face search-models API directly queries the model hub with structured filters, returning download counts, tags, and model IDs. The subsequent Hugging Face get-model-info call retrieves comprehensive metadata including architecture specifications, training data scale, and official model cards—information unavailable through web search snippets.

This behavior reflects an agent’s ability to dynamically select and compose tools under minimal domain priors, capturing a more realistic upper bound on general-agent capability

DeepSeek-R1 Qwen3-Next Qwen3-235B DeepSeek-V3.2 Gemini2.5-Flash

###### Search

###### Code

###### Reason

###### Tool-use

0.4

0.4

0.6

0.7

Accuracy

0.3

0.3

0.6

0.5

0.2

0.5

0.4

0.2

0.4

0.1

0.3

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

k (pass@k)

0.5

0.5

Accuracy

0.4

0.2

0.2

0.4

0.3

0.1

0.2

0.1

0.3

64 80 96 112 128

64 80 96 112 128

128 144 160 176 192

64 80 96 112 128

Context Length (k)

- Figure 5. Test-time scaling behaviors of general LLM agents. Results are reported for five models across four domains on General AgentBench. Top: Parallel scaling expands the solution space through increased sampling. Bottom: Sequential scaling allocates additional computation via longer interaction histories, yet exhibiting unstable or diminishing returns.

and highlighting the importance of evaluation settings that approximate real-world tool availability.

### 4. Test-Time Scaling Evaluation

In this section, we present a systematic study of test-time scaling behavior of general LLM agents. Section 4.1 introduces the scaling strategies, with results and findings presented in Sections 4.2 and 4.3.

#### 4.1. Scaling Methodology

We investigate test-time scaling through two complementary paradigms: Parallel Scaling and Sequential Scaling, which correspond to distinct axes of computational allocation—breadth of exploration versus depth of exploitation. Compared with more sophisticated test-time scaling techniques such as self-correction(Madaan et al., 2023), beam search(Yao et al., 2023), or MCTS(Zhou et al., 2023a), these two strategies are the most commonly adopted and easiest to deploy in real-world agentic systems.

evaluating and selecting the best outcome from their own generated trajectories. To assess this ability, we introduce this Self-Choice setting, in which the agent evaluates its parallelly sampled outputs using one of the following strategies:

- (1) Point-wise choice. The agent independently evaluates each sampled trajectory and assigns a binary judgment. Performance is measured by the alignment between the model’s judgments and oracle labels, averaged over trajectories that are correct under oracle evaluation.
- (2) Pair-wise choice. The agent compares two sampled trajectories at a time and iteratively promotes the superior one through a bubble-sort-style selection process. After K − 1 pairwise comparisons, a single trajectory is selected as the final output, and performance is evaluated based on the correctness of this selected trajectory.

Overall, self-choice reflects the practical effectiveness of parallel scaling, while past@K serves as an upper bound that reveals the solution potential.

Parallel Scaling. In the parallel scaling regime, we independently sample K trajectories for each query. Increasing K expands the reachable action space, thereby increasing the likelihood that the agent explores at least one trajectory containing a correct solution, relative to the single-shot baseline (K = 1).

However, in the absence of external oracles or human feedback—particularly in real-world deployments—parallel scaling alone is insufficient: agents must also be capable of

Sequential Scaling. In contrast, sequential scaling increases computational depth by extending the interaction horizon. As the agent engages with the environment, the conversation context progressively grows. When the agent attempts to terminate an episode (e.g., by emitting an Endof-Turn token), we inject an additional round of environment feedback to encourage further reflection on prior reasoning and exploration of alternative solution paths.

To quantify scaling behaviors under both paradigms, we plot

task accuracy against the number of independent samples (K) for parallel scaling, and against cumulative context length for sequential scaling in Figure 5. Due to the cost of API-based inference, each model is sampled at most four times, and context length is scaled up to 196K tokens. A detailed cost analysis for reproducibility is provided in the Appendix C.

#### 4.2. Sequential Scaling

DeepSeek-R1

###### DeepSeek-V3.2

###### Qwen3-235B

###### Qwen3-Next

###### Gemini-2.5-Flash

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

159

160

161

162

TaskID

163

164

165

166

167

168

64k80k96k112k128k

64k80k96k112k128k

64k80k96k112k128k

64k80k96k112k128k

64k80k96k112k128k

Success (score=1)

Failure (score=0)

| |
|---|

| |
|---|

- Figure 6. Instance-level correctness dynamics under sequential scaling. We randomly sample 10 instances from the reasoning domain and track their correctness across increasing context lengths. Red indicates incorrect predictions and blue indicates correct ones. Most instances exhibit stagnant or oscillatory behavior, repeating prior successes while failing on unresolved cases, with some fluctuating between correct and incorrect across steps.

Sequential scaling exhibits behavior that departs markedly from our expectations, shown in the bottom of Figure 5. Although several models on certain benchmarks show performance improvements, such as Qwen3-235B on the Search domain and Deepseek-v3.2 on the Reason domain, most show little to no consistent improvement despite allocating additional computational resources through iterative reasoning and reflection.

We categorize sequential scaling behaviors into two distinct regimes. (1) Stagnant fluctuation: In domains such as reasoning, performance oscillates within a narrow range despite increased computation. This pattern suggests a limited capacity for agents to explore novel solution paths within extended interaction traces, coupled with a diminished ability to maintain coherence—likely due to the agent’s finite context processing capacity. (2) Saturation and degradation: In coding domains, models initially benefit from additional reasoning steps; however, beyond an important turning point, performance consistently deteriorates and fails to recover. Figure 6 shows how instance-level correctness for several data entries changes as the context length increases. We observe that agents either repeatedly succeed on queries they already handle well while failing to progress on unsuccessful cases, or exhibit unstable behavior with accuracy fluctuating between 0 and 1 across interaction steps. These observations further validate our summarization regarding the inherent limitations of sequential scaling.

Search

Code

84k 95k 82k90k

0.3

0.20

Score

Score

0.15

0.2

0.10

0.1

0.05

48k 64k 80k 96k 112k 128k

48k 64k 80k 96k 128k 144k

Reason

Tool-use

134k140k 55k 66k

Gemini-2.5 Flash

Qwen3-235B

0.5

0.5

Score

Score

0.4

0.4

0.3

0.3

80k 128k144k160k176k192k

48k 64k 80k 96k 112k 128k

Figure 7. Sequential scaling behavior of Gemini 2.5-Flash and Qwen3-235B across domains, with inherent context lengths indicated by the dashed line. Performance scales positively as interaction history approaches and slightly exceeds the inherent context; however, it saturates or degrades once the context extends significantly beyond this threshold. This limit represents the “context ceiling” of sequential scaling, beyond which further history yields diminishing returns.

To better understand how this turning point occurs, we analyze performance relative to an agent’s inherent context length—defined as the total context naturally accumulated when completing a task without artificial constraints on interaction depth. By integrating data from Qwen3-235B and Gemini 2.5-Flash (as seen in Figure 5) with new experimental results on smaller context windows, we observed a distinct performance ceiling. As shown in Figure 7, both models exhibit an initial upward trend, approaching peak performance as they reach their inherent context limits. However, once the accumulated context passes a specific threshold ( for example, approximately 112K for Qwen3-235B and 96K for Gemini 2.5-Flash in the search domain), performance typically plateaus or begins to degrade. In these instances, additional computation and time yield no further gains. This suggests the existence of a “context ceiling”: a maximum effective context length under sequential scaling, beyond which raw interaction history offers diminishing or zero practical returns. Notably, this ceiling varies across domains, reflecting the unique demands each task places on context utilization and computational efficiency.

Taken together, these results indicate that simply allocating more computation by extending raw interaction histories rarely leads to meaningful performance gains, contradicting previous observations in non-agentic settings (Muennighoff et al., 2025). Long-horizon agentic tasks expose fundamental challenges in context utilization and reasoning stability, highlighting the need for more effective mechanisms for context management and reasoning control under sequential scaling (Zhang et al., 2025c; Anthropic, 2025c).

Pass@k

Point-wise (GPT5)

Verification Gap

Point-wise (Self)

Pair-wise (Self)

DeepSeek-R1

###### Qwen3-Next

###### Qwen3-235B

###### DeepSeek-V3.2

###### Gemini2.5-Flash

0.4

0.3

0.3

0.2

Search

0.3

0.2

0.2

0.2

0.2

0.1

0.1

0.1

0.1

0.1

0.0

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

0.3

0.4

0.4

###### Code

0.2

0.3

0.3

0.2

0.1

0.2

0.2

0.1

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

0.5

0.6

0.6

0.6

0.6

###### Reason

0.5

0.4

0.5

0.4

0.5

0.5

0.3

0.4

0.3

0.4

0.2

0.3

0.2

0.4

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

0.8

0.7

0.6

0.8

0.7

0.5

0.6

###### Tool-use

0.6

0.7

0.5

0.5

0.5

0.4

0.6

0.4

0.4

0.4

0.5

0.3

0.3

0.3

0.3

0.4

0.2

0.2

0.2

0.3

0.2

0.1

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

1 2 3 4

k (pass@k)

Figure 8. Verification gap between generation and self-choice. Across four domains, we observe a consistent gap between solution generation and verification: as the number of samples increases, correct solutions appear more frequently in the sampled set, yet models often fail to identify and select them. The dashed and dotted curves represent two self-choice strategies, while the diamond denotes a stronger evaluator, GPT-5.

#### 4.3. Parallel Scaling

We observe a monotonic increase in pass@K as the number of sampled trajectories grows (Figure 5, top). Increasing K from 1 to 4 yields an average improvement of roughly 50%, with DeepSeek-V3.2 exhibiting the largest gains—approaching a twofold improvement in both coding and reasoning domains.

However, pass@K only reflects an idealized upper bound. In practice, the effectiveness of parallel scaling depends on an agent’s ability to evaluate and select among its own sampled trajectories, which is captured by self-choice performance. As shown by the gap between the solid lines (pass@K) and the dashed or dotted lines (self-choice) in Figure 8, self-choice performance consistently lags behind the pass@K upper bound regardless of the strategy. In some cases, self-choice performance even degrades as K increases. While the coding domain exhibits the smallest gap between pass@K and self-choice, in most settings selfchoice gains saturate quickly and fail to track the continued improvement of pass@K.

To examine whether this gap stems from limited verifier capability, we replace the model’s internal self-judgment with an external verifier, GPT-5, and perform point-wise evaluation of sampled trajectories from all models. As shown by the diamond line in Figure 8, GPT-5 generally underperforms models’ own self-judgment. Notably, even at K = 1, GPT-5 occasionally misclassifies correct trajectories as incorrect, introducing a non-trivial gap between pass@1 and GPT-5–based verification. We hypothesize that this effect arises from solution familiarity: models are better at evaluating their own generations, which align closely with their internal reasoning patterns, whereas external verifiers may struggle to accurately assess unfamiliar execution traces.

Overall, these results reveal a fundamental gap between models’ generation capacity and their ability to reliably select correct solutions, which ultimately limits the practical utility of parallel scaling.

### 5. Related Work

Prior work on LLM agents spans three closely related directions: benchmarking agentic abilities, developing agent models and frameworks, and exploring effective scaling methods. Most agentic benchmarks evaluate specialized agents in domain-specific settings, where tasks, interaction protocols, and tool access are tailored to specific skills. Representative ones cover software engineering (e.g., SWEbench (Jimenez et al., 2023) and its variants (Aleithan et al., 2024; Zhang et al., 2025b; Yang et al., 2024a)), web navigation and interaction (e.g., WebShopcite (Yao et al., 2022a), Mind2Web (Deng et al., 2023), WebArena (Zhou et al., 2023b), WebVoyager (He et al., 2024)), and tool use with curated APIs (e.g., ToolLLM (Qin et al., 2023), API-Bank (Li et al., 2023), BFCL (Patil et al., 2025), StableToolBench (Guo et al., 2024)). Broader suites such as τ-bench (Yao et al., 2024), AgentBench(Barres et al., 2025), and GAIA (Mialon et al., 2023) incorporate multi-turn interaction across multiple task categories , while OSWorld (Xie et al., 2024), OSWorld-G (Xie et al., 2025), and AndroidWorld (Rawles et al., 2024) evaluate agents operating in real desktop and mobile systems.

In parallel, substantial effort has focused on building general-purpose agents capable of planning, acting, and invoking tools across heterogeneous tasks. Early methods such as ReAct (Yao et al., 2022b), Reflexion (Shinn et al., 2023), Toolformer (Schick et al., 2023), and HuggingGPT (Shen et al., 2023) introduce structured reasoning–action trajectories, reflection, and learned tool invocation. More recent work emphasizes scalable agent frameworks and deployment platforms supporting multi-agent coordination and rich tool ecosystems, including OpenAgents (Xie et al., 2023), AgentVerse (Chen et al., 2024b), AgentScope (Gao et al., 2024), and OpenCUA (Wang et al., 2025a). Industry systems—such as agents built on Claude (Anthropic, 2025b), Microsoft (Microsoft, 2025), OpenAI (OpenAI, 2025), Qwen (QwenLM, 2023), Kimi (Moonshot AI, 2026), and Gemini (Deepmind, 2025)—further demonstrate the practical importance of general agents in real-world deployments.

A complementary line of research studies test-time scaling, allocating additional inference-time computation to improve agent performance. Chain-of-thought prompting (Wei et al., 2022) and self-consistency (Wang et al., 2022) show that deeper reasoning and multi-sample decoding can substantially boost accuracy. More explicit approaches perform search or refinement over solution paths, including Tree-of-Thoughts (Yao et al., 2023), MCTS-based agent planning such as LATS (Zhou et al., 2023a), and iterative refinement methods like Reflexion (Shinn et al., 2023) and Self-Refine (Madaan et al., 2023). More recently, “internal scaling” trains models to autonomously decide how much

inference computation to allocate and when to terminate reasoning, shifting control from external orchestration toward model-internal deliberation (Guo et al., 2025; OpenAI, 2024a). Verifier-based inference further augments sampling or search with learned ranking or rejection, with processlevel supervision improving reliability in mathematical reasoning (Lightman et al., 2023; Li et al., 2024; Hosseini et al., 2024; Wang et al., 2024c).

### 6. Conclusions

We present General AgentBench, a unified benchmark for evaluating LLM agents under realistic, multi-domain interactions where agents must infer intent, select tools from a shared pool, and act end-to-end. Across ten leading models, we find a substantial robustness gap when moving from domain-specific to general-agent evaluation. Our test-time scaling analysis reveals two fundamental limits: sequential scaling is bounded by an context ceiling, beyond which longer interactions become unstable, and parallel scaling delivers limited practical gains due to a persistent verification gap between generation and self-choice. We hope General AgentBench enables realistic assessment and guides progress toward robust, scalable general agents.

### Impact Statement

This paper introduces General AgentBench, a benchmark for evaluating large language model agents as general-purpose systems under unified and realistic interaction settings. By exposing agents to diverse tasks and tools within a single framework, it enables more consistent and transparent assessment of agents’ abilities to interpret open-ended requests, select appropriate tools, and scale inference-time computation across domains. We expect this benchmark to support the development and diagnosis of more robust general-purpose agents. We also acknowledge potential risks. Unified evaluation and test-time scaling may favor computationally intensive or proprietary models, and unreliable self-choice under parallel scaling may lead to unstable or misleading agent behavior if applied naively in deployment. As general-purpose agents are increasingly integrated into real-world applications, failures in long-horizon reasoning, tool use, or self-verification may affect reliability and user trust. We emphasize that General AgentBench is intended for research and evaluation purposes. The community shares a responsibility to interpret results carefully and to pair scaling-based improvements with stronger verification, transparency, and safety considerations.

### References

Agarwal, R., Singh, A., Zhang, L., Bohnet, B., Rosias, L., Chan, S., Zhang, B., Anand, A., Abbas, Z., Nova, A.,

et al. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966, 2024.

Aleithan, R. et al. Swe-bench+: Enhanced coding benchmark for llms. arXiv preprint arXiv:2410.06992, 2024.

Anthropic. Introducing the model context protocol, November 2024. URL https://www.anthropic.com/ news/model-context-protocol.

Anthropic. Introducing claude sonnet 4.5. https://www. anthropic.com/news/claude-sonnet-4-5, September 2025a. Accessed: 2026-01-29.

Anthropic. Building agents with the Claude agent SDK. https://www.anthropic.com/engineerin g/building-agents-with-the-claude-age nt-sdk, September 2025b. Accessed: 2026-01-29.

Anthropic. Effective context engineering for ai agents. ht tps://www.anthropic.com/engineering/ effective-context-engineering-for-a i-agents, September 2025c. Accessed: 2026-01-29.

Anthropic. Building agents with skills: Equipping agents for specialized work, January 2026. URL https://cl aude.com/blog/building-agents-with-s kills-equipping-agents-for-specializ ed-work.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 3119–3137, 2024.

Barres, V., Dong, H., Ray, S., Si, X., and Narasimhan, K. tau2-bench: Evaluating conversational agents in a dualcontrol environment. arXiv preprint arXiv:2506.07982, 2025.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., R´e, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Chen, J., Lin, H., Han, X., and Sun, L. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17754–17762, 2024a.

Chen, W., Wang, H., Chen, J., Zhang, Y., Wang, H., Li, S., Zhou, X., and Wang, W. Y. Tabfact: A large-scale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164, 2019.

Chen, W., Su, Y., Zuo, J., Yang, C., Yuan, C., Chan, C.-M., Yu, H., Lu, Y., Hung, Y.-H., Qian, C., et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024b.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Deepmind, G. Gemini, Nov 2025. URL https://deep mind.google/models/gemini/.

Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., and Su, Y. Mind2web: Towards a generalist agent for the web. In NeurIPS Datasets and Benchmarks, 2023.

Ding, Y., Wang, Z., Ahmad, W., Ding, H., Tan, M., Jain, N., Ramanathan, M. K., Nallapati, R., Bhatia, P., Roth, D., et al. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. Advances in Neural Information Processing Systems, 36:46701– 46723, 2023.

Erdogan, L. E., Furuta, H., Kim, S., Lee, N., Moon, S., Anumanchipalli, G., Keutzer, K., and Gholami, A. Planand-act: Improving planning of agents for long-horizon tasks. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview .net/forum?id=ybA4EcMmUZ.

Fabbri, A. R., Li, I., She, T., Li, S., and Radev, D. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 1074–1084, 2019.

Friel, R., Belyi, M., and Sanyal, A. Ragbench: Explainable benchmark for retrieval-augmented generation systems. arXiv preprint arXiv:2407.11005, 2024.

Gao, D., Li, Z., Pan, X., Kuang, W., Ma, Z., Qian, B., Wei, F., Zhang, W., Xie, Y., Chen, D., et al. Agentscope: A flexible yet robust multi-agent platform. arXiv preprint arXiv:2402.14034, 2024.

Gao, T., Yen, H., Yu, J., and Chen, D. Enabling large language models to generate text with citations. arXiv preprint arXiv:2305.14627, 2023.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Guo, Z. et al. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. In Findings of ACL, 2024.

He, H., Yao, W., Ma, K., Yu, W., Dai, Y., Zhang, H., Lan, Z., and Yu, D. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.

Ho, X., Duong Nguyen, A.-K., Sugawara, S., and Aizawa, A. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Scott, D., Bel, N., and Zong, C. (eds.), Proceedings of the 28th International Conference on Computational Linguistics, pp. 6609–6625, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.580. URL https://aclanthology.org/2020.coling

-main.580/.

Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni, A., and Agarwal, R. V-star: Training verifiers for selftaught reasoners. arXiv preprint arXiv:2402.06457, 2024.

Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Hu, Y., Ganter, T., Deilamsalehy, H., Dernoncourt, F., Foroosh, H., and Liu, F. Meetingbank: A benchmark dataset for meeting summarization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16409– 16423, 2023.

Huang, L., Cao, S., Parulian, N., Ji, H., and Wang, L. Efficient attentions for long document summarization. arXiv preprint arXiv:2104.02112, 2021.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Jin, J., Paladugu, A., and Xiong, C. Beneficial reasoning behaviors in agentic search and effective post-training to obtain them. arXiv preprint arXiv:2510.06534, 2025.

Kamradt, G. LLMTest NeedleInAHaystack: Needle in a haystack – pressure testing llms. https://github.c om/gkamradt/LLMTest_NeedleInAHaystack, 2023. GitHub repository. Accessed: 2026-01-15.

Koˇcisk`y, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., and Grefenstette, E. The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328, 2018.

Kuratov, Y., Bulatov, A., Anokhin, P., Rodkin, I., Sorokin, D., Sorokin, A., and Burtsev, M. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. Advances in Neural Information Processing Systems, 37: 106519–106554, 2024.

Kweon, S., Kwon, Y., Cho, S., Jo, Y., and Choi, E. Openwikitable: Dataset for open domain question answering with complex reasoning over table. arXiv preprint arXiv:2305.07288, 2023.

Li, M. et al. Api-bank: A comprehensive benchmark for tool-augmented llms. In EMNLP, 2023. arXiv:2304.08244.

Li, X., Yu, Z., and Xiong, C. Montessori-instruct: Generate influential training data tailored for student learning. arXiv preprint arXiv:2410.14208, 2024.

Lightman, H. et al. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Liu, T., Xu, C., and McAuley, J. Repobench: Benchmarking repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091, 2023a.

- Liu, X.-Y., Wang, G., Yang, H., and Zha, D. Fingpt: Democratizing internet-scale data for financial large language models. arXiv preprint arXiv:2307.10485, 2023b.
- Liu, Y., Huang, L., Li, S., Chen, S., Zhou, H., Meng, F., Zhou, J., and Sun, X. Recall: A benchmark for llms robustness against external counterfactual knowledge. arXiv preprint arXiv:2311.08147, 2023c.

Luo, J., Zhang, W., Yuan, Y., Zhao, Y., Yang, J., Gu, Y., Wu, B., Chen, B., Qiao, Z., Long, Q., et al. Large language model agent: A survey on methodology, applications and challenges. arXiv preprint arXiv:2503.21460, 2025.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., and Scialom, T. Gaia: A benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2023.

Microsoft. Microsoft/agent-framework: A framework for building, orchestrating and deploying ai agents and multiagent workflows with support for python and .net., Apr 2025. URL https://github.com/microsoft /agent-framework.

Moonshot AI. Kimi k2.5: Visual agentic intelligence. http s://www.kimi.com/blog/kimi-k2-5.html, January 2026. Accessed: 2026-01-28.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2014– 2037, 2023.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. B. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20286– 20332, 2025.

OpenAI. Learning to reason with llms. https://op enai.com/index/learning-to-reason-wit h-llms/, September 2024a. Accessed: 2026-01-28.

OpenAI. Introducing SWE-bench verified. https://op enai.com/index/introducing-swe-bench -verified/, August 2024b. OpenAI Blog. Updated:

2025-02-24. Accessed: 2026-01-15.

OpenAI. Gpt-5 system card. Technical report, OpenAI, August 2025a. URL https://cdn.openai.com /gpt-5-system-card.pdf.

OpenAI. Openai mrcr: Long context multiple needle in a haystack benchmark. https://huggingface.co /datasets/openai/mrcr, 2025b. Hugging Face Datasets. Accessed: 2026-01-15.

OpenAI. Introducing agentkit, Oct 2025. URL https: //openai.com/index/introducing-agent kit/.

Parmar, M., Liu, X., Goyal, P., Chen, Y., Le, L., Mishra, S., Mobahi, H., Gu, J., Wang, Z., Nakhost, H., et al. Plangen: A multi-agent framework for generating planning and reasoning trajectories for complex problem solving. arXiv preprint arXiv:2502.16111, 2025.

Patil, S. G., Zhang, T., Wang, X., and Gonzalez, J. E. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37: 126544–126565, 2024.

Patil, S. G., Mao, H., Yan, F., Ji, C. C.-J., Suresh, V., Stoica, I., and Gonzalez, J. E. The berkeley function calling

leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https:

//openreview.net/forum?id=2GmDdhBdDk.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.

Qwen Team, Alibaba Inc. Qwen3-Next: Towards ultimate training & inference efficiency. https://qwen.ai/ blog?id=4074cca80393150c248e508aa629 83f9cb7d27cd, September 2025. Accessed: 2026-0129.

QwenLM. Qwenlm/qwen-agent: Agent framework and applications built upon qwen¿=3.0, featuring function calling, mcp, code interpreter, rag, chrome extension, etc., Sep 2023. URL https://github.com/QwenLM/ Qwen-Agent.

Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., CampbellAjala, F., et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.

Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

Shaham, U., Ivgi, M., Efrat, A., Berant, J., and Levy, O. Zeroscrolls: A zero-shot benchmark for long text understanding. arXiv preprint arXiv:2305.14196, 2023.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang, Y. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. arXiv preprint arXiv:2303.17580, 2023.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Thakur, N., Reimers, N., R¨uckl´e, A., Srivastava, A., and Gurevych, I. Beir: A heterogenous benchmark for zeroshot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021.

Tu, Q., Chen, C., Li, J., Li, Y., Shang, S., Zhao, D., Wang, R., and Yan, R. Characterchat: Learning towards conversational ai with personalized social support. arXiv preprint arXiv:2308.10278, 2023.

Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., and Lim, E.-P. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. arXiv preprint arXiv:2305.04091, 2023.

- Wang, L., Dong, S., Xu, Y., Dong, H., Wang, Y., Saha, A., Lim, E.-P., Xiong, C., and Sahoo, D. Mathhay: An automated benchmark for long-context mathematical reasoning in llms. arXiv preprint arXiv:2410.04698, 2024a.
- Wang, M., Chen, L., Cheng, F., Liao, S., Zhang, X., Wu, B., Yu, H., Xu, N., Zhang, L., Luo, R., et al. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5627–5646, 2024b.

Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, 2024c.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Wang, X., Wang, B., Lu, D., Yang, J., Xie, T., Wang, J., Deng, J., Guo, X., Xu, Y., Wu, C. H., et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025a.

Wang, Z., Chang, Q., Patel, H., Biju, S., Wu, C.-E., Liu, Q., Ding, A., Rezazadeh, A., Shah, A., Bao, Y., et al. Mcp-bench: Benchmarking tool-using llm agents with complex real-world tasks via mcp servers. arXiv preprint arXiv:2508.20453, 2025b.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. V., and Zhou, D. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.

Wei, J., Sun, Z., Papay, S., McKinney, S., Han, J., Fulford, I., Chung, H. W., Passos, A. T., Fedus, W., and Glaese, A. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

Xie, T., Zhou, F., Cheng, Z., Shi, P., Weng, L., Liu, Y., et al. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Xie, T., Deng, J., Li, X., Yang, J., Wu, H., Chen, J., Hu, W., Wang, X., Xu, Y., Wang, Z., et al. Scaling computer-use grounding via user interface decomposition and synthesis. arXiv preprint arXiv:2505.13227, 2025.

- Yan, K., Ling, Z., Liu, K., Yang, Y., Fan, T.-H., Shen, L., Du, Z., and Chen, J. Mir-bench: Can your llm recognize complicated patterns via many-shot in-context reasoning? arXiv preprint arXiv:2502.09933, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. Swe-agent: Agentcomputer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024a.

Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024b.

- Yao, S., Chen, H., Yang, J., and Narasimhan, K. Webshop: Towards scalable real-world web interaction with grounded language agents. arXiv preprint arXiv:2207.01206, 2022a.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022b.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

Yao, S. et al.

tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Yue, L., Xing, S., Chen, J., and Fu, T. Clinicalagent: Clinical trial multi-agent system with large language model-based reasoning. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, pp. 1–10, 2024.

Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., and Goodman, N. D. Quiet-star: Language models can teach themselves to think before speaking. arXiv preprint arXiv:2403.09629, 2024.

Zhang, J., Bai, Y., Lv, X., Gu, W., Liu, D., Zou, M., Cao, S., Hou, L., Dong, Y., Feng, L., et al. Longcite: Enabling llms to generate fine-grained citations in long-context qa. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 5098–5122, 2025a.

Zhang, L. et al. Swe-bench goes live! arXiv preprint arXiv:2505.23419, 2025b.

Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., et al. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, 2025c.

Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., and Wang, Y.-X. Language agent tree search unifies reasoning, acting, and planning in language models. arXiv preprint arXiv:2310.04406, 2023a.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023b.

### A. Additional Experimental Results

#### A.1. Agentic Benchmarks

Coding We include tasks from SWE-Bench Verified(OpenAI, 2024b) and Terminal Bench for the code domain. SWEBench Verified consists of real-world GitHub issues that require the model to analyze and propose concrete bug fixes, with evaluation performed through automated test pass rates. Terminal Bench assesses a model’s ability to solve problems within a terminal environment, requiring not only command-line common sense but also the ability to plan and reason over long user instructions. It examinzes the score by checking whether the model achieves the expected final state.

Reason For this domain, we adopt MathHay(Wang et al., 2024a) as our data source. MathHay collects mathematical information grounded in real web documents. For each group of related documents, specific pieces of information are extracted and linked to form a query. The benchmark then constructs a long-context haystack by inserting these relevant documents into noisy text placed at the beginning, middle, or end of the sequence. This design provides a suitable testbed for evaluating the model’s long-context reasoning ability. One thing to note is that no raw tools are provided in this benchmark.

Most reasoning benchmarks provide all necessary information directly within the query and ask the model to perform complex mathematical or scientific derivations. As a result, they naturally lack an interactive environment that can deliver feedback, making them unsuitable for multi-turn evaluation. However, we can simulate multi-turn interaction by explicitly prompting the model to refine its own answers over additional turns, effectively creating self-reflective reasoning cycles.

- A key requirement for such evaluation is that the query context must be sufficiently long to allow meaningful iterative improvement.

Search For the search domain, we include tasks from BrowseComp(Wei et al., 2025) and WebVoyager(He et al., 2024), aiming to evaluate a model’s ability to locate accurate information within long, evolving contexts. In contrast to Needle-in-aHaystack or purely textual retrieval tasks, agentic search requires the model to reason about what information is needed, determine whether the current context is sufficient to answer the query, and decide when additional search steps are necessary. Mind2Web and WebVoyager focus on everyday web-browsing tasks—such as shopping, navigation, and entertainment—and rely on external language models or task-specific agents to assess model performance. BrowseComp, in contrast, presents challenging, hard-to-find information queries that require multi-step investigation across the web. The model’s final answers are compared against expert-curated gold references to produce the evaluation score.

Tool-use For the tool-use domain, we include tasks from Tau2-Bench(Barres et al., 2025) and MCP-Bench(Wang et al., 2025b), both of which provide extensive tool suites for evaluating a model’s ability to understand, select, and invoke the appropriate tool to solve a problem. Tau2-Bench focuses on customer-service scenarios, where the model acts as a customersupport agent responding to simulated user queries using a set of synthetic tools tailored to the scenario. MCP-Bench, in contrast, offers a collection of real-world tools built on the Model Context Protocol (MCP), requiring models to perform dense tool calling and coordinate across multiple tools to complete complex tasks.

Both benchmarks require multi-turn interaction with a simulated environment, and the complexity and richness of the task specifications make them nearly unsolvable within a short context window.

While the coding and search domains can be viewed as specific instances of tool use—since they also involve code-execution tools and search APIs—the tool-use domain in General AgentBench significantly broadens the definition of “tools.” It requires models to precisely control tool parameters, reason about which tool to invoke and when, and plan multi-step interactions based on previous own generation and external feedback.

#### A.2. Detailed Results Comparison

Table 3 reports the detailed performance of all evaluated models on both the original domain-specific benchmarks and our General AgentBench setting. These results serve as the data source for Figure 3 and Figure 4.

- Table 3. Main Results on General AgentBench. We report performance in the Baseline (B) and General (G) settings. ∆% represents the

relative change. The overall average scores (AvgB, AvgG) and the mean relative degradation are averaged across four domains instead of each benchmark. Bold text indicates the best performance, while underlining denotes the second best; red highlights performance

degradation.

Models

Search Code Reason Tool-Call Overall Avg.

Avg. ∆% B G ∆% B G ∆% B G ∆% B G ∆% AvgB AvgG

Open-Source

GPT-OSS-120B 17.6 12.1 -31.3 16.9 8.5 -49.7 46.7 38.7 -17.1 65.2 45.0 -31.0 36.6 26.1 -28.7 Qwen3-235B-A22B 21.7 16.4 -24.4 36.2 22.5 -37.8 45.3 32.0 -29.4 62.6 52.5 -16.1 41.5 30.9 -25.5 Qwen3-Next 16.1 19.1 +18.6 22.3 12.3 -44.8 44.0 42.0 -4.5 63.3 56.9 -10.1 36.4 32.6 -10.4 DeepSeek-V3.2 34.6 28.6 -17.3 32.3 25.9 -19.8 34.7 33.3 -4.0 61.8 60.1 -2.8 40.9 37.0 -9.5 DeepSeek-R1 16.6 21.2 +27.7 23.1 10.8 -52.2 47.3 46.7 -1.3 44.7 40.1 -10.3 32.9 29.7 -9.8

Proprietary

Gemini 2.5-Flash 21.1 15.4 -27.0 30.0 17.7 -41.0 60.0 36.0 -40.0 66.0 52.7 -20.2 44.3 30.5 -31.2 Gemini 2.5-Pro 23.4 21.7 -7.3 26.9 26.9 0.0 61.3 24.0 -60.8 66.1 56.8 -14.1 44.4 32.4 -27.2 Claude Haiku 4.5 23.6 28.0 +18.6 41.5 36.9 -11.1 54.7 34.7 -36.6 64.5 56.7 -12.1 46.1 39.1 -15.2 Claude Sonnet 4.5 26.1 34.7 +33.0 49.2 48.5 -1.4 32.0 36.0 +12.5 73.0 60.7 -16.8 45.1 45.0 -0.2 GPT-5 55.8 39.1 -29.9 45.4 39.3 -13.4 64.0 64.0 0.0 78.3 45.8 -41.5 60.9 47.1 -22.7

- Table 4. Performance on Traditional Long-Context Benchmarks. We report scores across three established benchmarks focusing on retrieval and single-turn comprehension. Bold indicates the best performance in each category.

Models LongBench HELMET MRCR Open-Source

GPT-OSS-120B 47.8 12.9 32.8 Qwen3-235B-A22B 58.3 40.8 40.6 Qwen3-Next 53.1 26.3 27.9 DeepSeek-V3.2 50.3 48.0 33.2 DeepSeek-R1 58.3 36.9 39.2

Frontier

Gemini 2.5-Flash 62.1 27.8 67.4 Gemini 2.5-Pro 63.3 63.1 80.0 Claude Haiku 4.5 55.3 50.4 33.3 Claude Sonnet 4.5 61.8 49.2 35.5 GPT-5 64.6 51.6 79.1

#### A.3. Comparison with related long-context benchmarks

With unified toolsets alone already approaching 64K tokens, the addition of user queries and multi-turn interaction histories can easily push the total context length to nearly 128K tokens. As a result, long-context processing naturally emerges as a core capability required for general agents. However, existing long-context benchmarks differ fundamentally from agentic long-context scenarios. As summarized in Table 5, we identify two key dimensions along which prior benchmarks diverge from agentic settings.

- (1) Context composition. Most existing long-context benchmarks are dominated by long-document question answering (QA), where the interaction paradigm remains static and single-turn. In contrast, agentic contexts are inherently heterogeneous: beyond long documents, they include environment feedback (e.g., tool execution results) and the model’s own prior decisions accumulated through multi-turn interactions.
- (2) Long-output reasoning. Other established benchmark categories, such as many-shot in-context learning and summarization, involve long inputs but require only relatively short outputs. Agentic tasks, however, demand sustained reasoning over extended contexts, including plan generation, iterative reflection, and explicit tool-call specifications, often resulting in long and structured outputs. While retrieval-based tasks and recent citation-grounded generation benchmarks partially resemble agentic scenarios, they still lack the multi-turn, interactive dynamics essential to true agentic settings.

These fundamental differences suggest that performance measured on prior non-agentic long-context benchmarks does not directly reflect model behavior under agentic interaction.

1.0

[Figure 14]

|1.00 0.46 0.76 0.50<br><br>0.46 1.00 0.62 0.62<br><br>0.76 0.62 1.00 0.67| |0.46 0.34 0.38 0.33<br><br>0.51 0.66 0.27 0.50<br><br>0.59 0.17 0.71 0.47|
|---|---|---|
|0.50 0.62 0.67|1.00|0.87 0.70 0.65 0.85|
|0.46 0.51 0.59<br><br>0.34 0.66 0.17<br><br>0.38 0.27 0.71<br><br>0.33 0.50 0.47|0.87 1.00 0.63 0.36 0.71<br><br>0.70 0.63 1.00 0.04 0.65<br><br>0.65 0.36 0.04 1.00 0.37<br><br>0.85 0.71 0.65 0.37 1.00| |

LongBenchv2

0.8

HELMET

0.6

MRCR

0.4

All

Search

0.2

Code

0.0

Reason

0.2

Tool-use

0.4

LongBenchv2HELMETMRCR AllSearch CodeReasonTool-use

Figure 9. Pairwise correlation between static long-context benchmarks and agentic domains. “All” denotes the average performance across the search, code, reasoning, and tool-use domains.

#### A.4. Transferability of Long-Context Abilities from Static to Agentic Benchmarks

Long-context processing is a fundamental requirement for general agents, as unified toolsets and accumulated multi-turn interactions can rapidly extend the effective context length during real-world use. However, it remains unclear whether performance measured on existing static, single-turn long-context benchmarks meaningfully transfers to agentic settings, where context evolves dynamically through interaction and decisions must be made sequentially. To study this question, we evaluate models on several representative static long-context benchmarks and examine how their performance correlates with results on General AgenticBench. We choose LongBench, HELMET, and MRCR here.

LongBench v2 LongBench v2 is an updated and expanded version of LongBench, designed to evaluate language models under diverse long-context understanding tasks across multiple domains. It covers a broad set of settings, including longdocument question answering, multi-document reasoning, summarization, code understanding, and many-shot in-context learning. Compared to the original LongBench, v2 increases both context length and task diversity, and introduces harder examples that stress retrieval accuracy and reasoning robustness under long inputs. Despite its breadth, LongBench v2 remains fundamentally single-turn and static in interaction structure. All necessary information is provided upfront, and models are evaluated on their ability to extract, aggregate, or reason over relevant spans within a fixed context window. Outputs are typically short and final, without iterative refinement or environment feedback. As a result, LongBench v2 primarily measures long-context comprehension and retrieval, rather than the dynamic decision-making, planning, and self-conditioning behaviors required in agentic multi-turn settings.

HELMET HELMET is a holistic benchmark for evaluating long-context language models across a curated set of real-world and synthetic tasks. It emphasizes robustness, faithfulness, and information utilization under long inputs, and includes task categories such as long-document QA, summarization, citation-grounded generation, and structured information extraction.

Model Consistency

1.0

[Figure 15]

GPT-OSS-120B

1.00 0.39 0.68 -0.11 0.50 0.36 0.14 0.39 0.02 0.43

0.8

Qwen3-235B-A22B-2507

0.39 1.00 0.86 0.64 0.86 0.71 0.75 0.93 0.68 0.64

0.6

Qwen3-Next

0.68 0.86 1.00 0.43 0.93 0.71 0.71 0.82 0.54 0.64

0.4

Deepseek-V3.2

-0.11 0.64 0.43 1.00 0.64 0.57 0.79 0.54 0.83 0.50

0.2

Deepseek-R1

0.50 0.86 0.93 0.64 1.00 0.89 0.89 0.79 0.70 0.75

Models

0.0

Gemini2.5-Flash

0.36 0.71 0.71 0.57 0.89 1.00 0.89 0.57 0.56 0.86

0.2

Gemini2.5-Pro

0.14 0.75 0.71 0.79 0.89 0.89 1.00 0.57 0.65 0.82

0.4

Claude Haiku 4.5

0.39 0.93 0.82 0.54 0.79 0.57 0.57 1.00 0.77 0.36

0.6

Claude Sonnet 4.5

0.02 0.68 0.54 0.83 0.70 0.56 0.65 0.77 1.00 0.23

0.8

GPT-5

0.43 0.64 0.64 0.50 0.75 0.86 0.82 0.36 0.23 1.00

1.0

GPT-OSS-120BQwen3-235B-A22B-2507Qwen3-NextDeepseek-V3.2Deepseek-R1Gemini2.5-FlashGemini2.5-ProClaudeHaiku4.5ClaudeSonnet4.5GPT-5

Models

Figure 10. Pairwise correlation between models.

HELMET is particularly focused on evaluating whether models can correctly attribute claims to source documents and avoid hallucinations when operating over extended contexts.

MRCR MRCR focuses on evaluating a model’s ability to perform multi-round coreference and entity tracking under long contexts. The benchmark constructs documents containing repeated, interleaved references to entities across long spans, requiring the model to resolve pronouns, aliases, and implicit references over multiple turns or segments. Tasks are designed to test memory persistence and consistency, particularly in scenarios where earlier mentions are far removed from later queries.

Across ten models, we compute Pearson correlations over pairwise absolute performance differences between static longcontext benchmarks and the four agentic domains. As shown in Figure 9, static benchmarks exhibit consistently weak correlation with agentic performance overall, indicating limited transferability from static long-context ability to agentic long-context reasoning. A moderate correlation is observed between MRCR and the reasoning domain, which is expected: reasoning tasks primarily involve extracting and computing over long documents without tool interaction, closely aligning with the characteristics of MRCR. Other agentic domains show substantially lower alignment. In particular, coding and tool-use exhibit minimal correlation with static benchmarks, suggesting that agentic performance depends not only on longcontext comprehension, but also on dynamic decision-making and precise execution. These results highlight a fundamental gap between static, single-turn long-context evaluation and the requirements of realistic agentic settings. We further provide correlations across different models on the General AgenticBench as shown in Figure 10.

### B. Attention Behavior Analysis

In sequential test-time scaling, we observe that Qwen3-Next demonstrates weaker scaling potential compared to other models, particularly its full-attention counterpart, Qwen3-235B-A22B. Although these models differ in training data composition

- Table 5. Taxonomy of Existing Long-Context Benchmarks. Unlike static evaluation paradigms, Agentic LongBench introduces multi-turn dynamics and sequential reasoning over expansive contexts.

Task Category Sub-category Representative Benchmarks / Papers Long Document QA Single/Multi-Doc Understanding NarrativeQA (Koˇcisk`y et al., 2018), 2WikiMultihopQA (Ho et al.,

2020), Loong (Wang et al., 2024b) Long Dialogue Understanding MeetingBank (Hu et al., 2023), CharacterChat (Tu et al., 2023) Code Understanding RepoBench (Liu et al., 2023a), CrossCodeEval (Ding et al., 2023) Structured Data Understanding TabFact (Chen et al., 2019), WikiTableQuestions (Kweon et al., 2023),

LongBench (Bai et al., 2024) (L-Data)

Summarization Global Information Aggregation GovReport (Huang et al., 2021), Multi-News (Fabbri et al., 2019),

ZeroSCROLLS (Shaham et al., 2023) Many-Shot In-Context Learning MIR-Bench (Yan et al., 2025), Many-Shot ICL (Agarwal et al., 2024) Retrieval Key-Value Retrieval Needle In A Haystack (Kamradt, 2023), RULER (Hsieh et al., 2024),

MRCR (OpenAI, 2025b), BABILong 2024 Retrieval-Augmented Gen (RAG) RAGBench (Friel et al., 2024), RGB (Chen et al., 2024a), RECALL (Liu et al., 2023c)

Reranking Passage Reranking BEIR (Thakur et al., 2021), MTEB (Muennighoff et al., 2023) (Rerank-

ing Selection)

Citation Gen Generation with Citations ALCE (Gao et al., 2023) (ASQA/ELI5), LongCite (Zhang et al., 2025a) Agentic Multi-turn Reasoning SWE-Bench (Jimenez et al., 2023), BrowseComp (Wei et al., 2025),

OSWorld (Xie et al., 2024), General AgentBench (This Work)

and parameterization, we analyze this discrepancy primarily from the perspective of the attention mechanism—the most salient architectural difference between them. We first introduce our analysis methodology, followed by empirical results and key findings.

#### B.1. Extracting Top-K Attention Tokens for Reasoning Behavior

Our attention analysis is conducted on inference trajectories collected from General AgentBench. For each benchmark domain, we randomly sample 25 trajectories generated by the model under analysis. Because attention patterns depend strongly on the specific text region being examined, we follow the framework of Jin et al. (2025) to extract reasoning-behavior sentences from each trajectory. These sentences correspond to critical decision-making steps, where the model actively reasons over accumulated context. For each reasoning sentence, we iterate over all tokens within the sentence. For every token, we compute and store its attention distribution over the preceding context. We then average the attention scores across tokens in the reasoning sentence, select the top-K attended tokens (K = 128), and record their token indices. This procedure is applied to all layers and attention heads.

We evaluate attention behavior using two metrics:

Mean Attention Distance. For each of the top-K attended tokens, we compute its token distance from the reasoning sentence and weight it by the corresponding attention score. This metric measures how far back the model attends when making a key decision, effectively quantifying the model’s effective contextual view.

Top-K Overlap. We measure the overlap of top-K attended tokens through intra-layer overlap (across heads within the same layer) and inter-layer overlap ( across different layers). Higher overlap indicates that different heads or layers attend to similar contextual tokens, suggesting reduced functional differentiation.

For Gated DeltaNet, which does not explicitly expose standard attention weights, we adopt a mathematically equivalent reconstruction procedure to recover the effective top-K contributing tokens and compute the corresponding distance statistics.

A. Mean Attention Distance (Layer vs Head)

B. Head Redundancy (Intra-layer Overlap)

1.0

0 3 6 9

[Figure 16]

[Figure 17]

C. Inter-layer Token Consistency Matrix

2500

0 3 6 9

0.9

12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72 75 78 81 84 87 90 93

0.4

12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72 75 78 81 84 87 90 93

2000

0.8

0.3

0.7

AvgJaccardSimilarity

JaccardSimilarity

TokensDistance

1500

LayerIndex

LayerIndex

0.6

0.2

1000

0.5

0.1

0.4

500

0.3

0.0

12

16

20

24

28

32

36

40

44

48

52

56

60

64

68

72

76

80

84

88

92

0

4

8

Layer Index

Avg0 10 20 30 40 50 60 Head Index (Leftmost column = Layer Average)

0 20 40 60 80 Layer Index

A. Mean Attention Distance (Layer vs Head)

B. Head Redundancy (Intra-layer Overlap)

1.0

[Figure 18]

[Figure 19]

0246810121416182022242628303234363840424446 LayerIndex

C. Inter-layer Token Consistency Matrix

0 2 4 6 8

1600

0.9

0.4

1400

0.8

10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46

1200

0.3

0.7

AvgJaccardSimilarity

JaccardSimilarity

1000

TokensDistance

LayerIndex

0.6

800

0.2

0.5

600

0.4

0.1

400

0.3

200

0.2

0.0

10

12

14

16

18

20

22

24

26

28

30

32

34

36

38

40

42

44

46

0

2

4

6

8

Layer Index

Avg 0 10 Head Index (Leftmost column = Layer Average)

0 10 20 30 40 Layer Index

Figure 11. Comparison of attention behaviors under the General AgentBench setting. Top: Qwen3-235B (full attention). Bottom: Qwen3-Next (hybrid linear attention).

#### B.2. Results and findings

In the leftmost panel of Figure 11, we visualize the mean attention distance for each head across layers. Full-attention models exhibit consistently larger mean distances, with only minor exceptions in early layers and around layer 70. In most layers, the majority of heads attend to long-range context, while a small subset focuses on local patterns. This behavior is also observed in the gated full-attention layers of Qwen3-Next (appearing every four layers). These results suggest that full attention maintains a broader effective contextual view than linear attention, consistent with the convolution-like receptive field constraints imposed by DeltaNet-style linear attention (Yang et al., 2024b).

The two rightmost panels present the top-K token overlap statistics.

Intra-layer overlap. Full attention exhibits a characteristic V-shaped curve: middle-layer heads attend to more diverse patterns, while later layers converge toward similar tokens, reflecting increased certainty near the final decision stage. In contrast, linear attention lacks a clear structural pattern and shows higher average intra-layer overlap, suggesting reduced head specialization.

Inter-layer overlap. Full attention displays a gradual “low-to-high” trend across depth, indicating that adjacent layers share similar functional roles, while functional divergence accumulates progressively with depth. In linear-attention models, DeltaNet layers show very low overlap with gated full-attention layers. Although DeltaNet layers exhibit high overlap across distant layers (indicating homogeneous behavior), the full-attention layers preserve their characteristic inter-layer structure.

Overall, our findings indicate that linear attention demonstrates weaker functional differentiation across heads and layers, along with reduced long-context utilization in agentic reasoning tasks, compared to full-attention mechanisms.

### C. Estimated Cost

#### C.1. Pricing and accounting.

We report an estimated API budget for reproducing our evaluation under three settings: (i) general (default context) evaluation of each model on each domain once, (ii) parallel scaling that samples multiple independent trajectories per query, and (iii) sequential scaling that extends the interaction horizon over multiple steps.

All prices are normalized as USD per 1M tokens and are taken from the corresponding model providers at the time of running the experiments. Due to uncertainty about the model provider’s caching mechanism and evaluation intervals, we will use only the input unit price in our estimates, even if the provider supplies a cache unit price.

We compute cost by aggregating token usage from execution logs, using the provider-reported token accounting: input tokens (prompt + tool outputs + intermediate messages fed back to the model), and output tokens (model-generated tokens). For each model, the total cost is estimated as:

##### 106 + pout · T

Cost = pin · T

out

in

106 ,

where pin,pout denote the unit prices, and Tin,Tout are the measured token counts.

#### C.2. Model cost

- Table 6 lists the unit API prices used in our cost estimation. Prices for input, cached input, and output are reported in USD per 1M tokens. Tables 7–9 summarize the aggregated evaluation cost per model under parallel scaling, sequential scaling, and the general (default context) setting, respectively. Each entry corresponds to running the full benchmark split of that dataset/domain under the specified protocol and then summing costs across all queries.

Model Input Cached Input Output

Gemini-2.5-Flash 0.30 0.03 2.50 Gemini-2.5-Pro 1.25 0.125 10.00 GPT-5 1.25 0.125 10.00 Claude-Haiku 4.5 1.00 0.50 5.00 Claude-Sonnet 4.5 3.00 3.75 15.00

gpt-oss-120B 0.15 0.075 0.60 Deepseek-R1 1.35 – 5.40 Deepseek-V3.2 0.28 – 0.42 Qwen3-235B 0.22 0.11 0.88 Qwen3-Next 0.15 – 1.50

- Table 6. Unit API prices (USD per 1M tokens) used in our cost estimation.

Model Search MathHay SWEBench MCPBench Tau2Bench TerminalBench Total Gemini-2.5-Flash $193 $70.9 $12719 $122 $62.2 $826 $13993 DeepSeek-R1 $5188 $157 $3675 $492 $202 $1505 $11218 DeepSeek-V3.2 $369 $38.8 $7.80 $88.4 $54.4 $412 $970 Qwen3-235B $1254 $41.4 $1286 $120 $55.5 $409 $3166 Qwen3-Next $25.3 $9.52 $3.38 $21.2 $14.9 $154 $229 Total $7028 $317 $17692 $843 $389 $3307 $29576

- Table 7. Cost for evaluating models under parallel scaling setting (USD)

Model Search MathHay SWEBench MCPBench Tau2Bench TerminalBench Total Gemini-2.5-Flash $5588 $369 $2024 $568 $852 $1870 $11271 DeepSeek-R1 $1267 $654 $892 $931 $1088 $782 $5614 DeepSeek-V3.2 $902 $333 $191 $79.1 $52.3 $356 $1913 Qwen3-235B $1370 $238 $436 $716 $162 $689 $3610 Qwen3-Next $442 $219 $392 $152 $251 $529 $1985 Total $9568 $1814 $3935 $2445 $2405 $4225 $24392

- Table 8. Cost for evaluating models under sequential scaling setting (USD)

Model Search MathHay SWEBench MCPBench Tau2Bench TerminalBench Total Gemini-2.5-Pro $47.5 $18.6 $3253 $31.3 $15.2 $203 $3569 GPT-5 $87.5 $14.2 $146 $21.0 $12.4 $60.9 $342 Claude-Haiku-4.5 $304 $10.9 $312 $29.3 $14.1 $106 $776 Claude-Sonnet-4.5 $1248 $40.9 $926 $129 $52.0 $376 $2772 OpenAI-oss-120B $1.44 $1.49 $0.52 $0.50 $1.35 $0.40 $5.70 DeepSeek-V3.2 $96.8 $10.0 $1.92 $22.0 $14.0 $102 $247 Qwen3-Next $6.05 $2.42 $0.88 $5.51 $3.60 $38.2 $56.7 Total $1791 $98.5 $4640 $239 $113 $886 $7768

Table 9. Cost for evaluating models under general (default context) setting (USD)

### D. General AgentBench Implementation Details

#### D.1. Tool Management

Tool Registration The General AgentBench follows the Model Context Protocol (MCP) architecture with Host-ClientServer design. During server connection, the BenchmarkHost creates transport connections (STDIO/HTTP), initializes ClientSession, and discovers tools via list tools(). All tools are registered to a global tool to client routing map.

Tool Schema Each tool follows OpenAI function-calling format with name, description, and parameters (JSON Schema), as shown in Box D.1. Tools use Bedrock-compatible naming requirements ([a-zA-Z0-9 -]+): MCP-Bench format uses ServerName tool name (e.g., BioMCP think), while Tau2 uses domain tool name (e.g., airline book reservation). The complete toolset in Host contains 301 tools across 35 servers, including BioMCP (34), Scientific Computing (26), Medical Calculator (22), NASA Data (21), NixOS (18), Unit Converter (16), tau2-retail (15), tau2-airline (14), tau2-telecom (13), and others.

###### An example of tool schema: get-collection-info function from Huggingface server

{

"type": "function", "function": {

"name": "Hugging_Face__get-collection-info", "description": "Get detailed information about a specific collection", "parameters": {

"type": "object", "properties": {

"namespace": { "type": "string", "description": "The namespace of the collection (user or organization)"

}, "collection_id": {

"type": "string", "description": "The ID part of the collection"

}

}, "required": [

"namespace", "collection_id"

]

} }

}

Description Compression and Minimal Strategy To reduce context consumption, we implement multi-level compression. The –compress-tools option truncates descriptions to target number of characters and removes parameter defaults, achieving 18.6% token reduction. If the description of the first sentence is truncated in the middle, we will keep the first sentence for tool loading.. For self-choice evaluation, we adopt the –minimal-tools strategy: converting JSON schema to plain text format (tool name(params): short desc), retaining only tool names, 50-character descriptions, and parameter names. This achieves 90.1% token reduction, saving approximately 70K tokens.

#### D.2. Benchmark Integration Details

User Simulator Implementation in Tau2Bench Tau2Bench requires multi-turn conversational interactions between the agent and a simulated user. We implement this through Tau2UserSimulatorAdapter, which wraps the original Tau2 UserSimulator. During each conversation turn, the adapter converts the agent’s response to Tau2’s AssistantMessage format, calls generate next message(message, state) to obtain the user response, and maintains conversation state across turns. User-side tool execution (e.g., checking order status) is routed through MCP internal callbacks, which ensures proper environment initialization.

Docker Environment Interaction Both Terminal Bench and SWEBench employ Docker Container Bridge Mode for isolated task execution. The MCP servers run as persistent host processes, interacting with task containers via docker exec. Public tools exposed to agents include execute bash, read file, write file, and finish(SWE-Bench). Each container receives a unique UUID-suffixed project name to support parallel experiments. This MCP Process Mode design avoids server restart overhead across tasks while maintaining complete environment isolation through Docker.

Evaluators The General Agent system adopts a native evaluator delegation strategy, directly invoking each benchmark’s original evaluation code rather than reimplementing evaluation logic. For Tau2Bench, we compute rewards as the product of environment state matching (DB CHECK), action sequence validation, and communication checks. SWEBench and Terminal Bench use Docker Bridge Mode: MCP servers run as persistent host processes, executing pytest harnesses in task containers via docker exec, then parsing results using toolsets in original benchmark implementation. Search benchmarks (BrowseComp, Mind2Web, WebVoyager, GAIA) use their original eval scripts with LLM-based semantic equivalence or rubric evaluation. All evaluators produce binary rewards (0/1) except MCPBench (continuous 0-1).

### E. Detailed Prompts

Universal Agent Prompt The agent system prompt instructs the model to solve diverse problems using tools or reasoning. It emphasizes careful tool selection, avoiding redundant calls, and building on previous results. For Tau2-Bench tasks, domain-specific policy documents are appended under a Policy section, loaded from policy markdown file. Other benchmarks use the base prompt without policy appended.

###### Universal Agent System Prompt

You are a helpful AI agent that can solve a wide variety of problems, including searching the web for information, writing and running code, performing calculations and logical reasoning, and interacting with external services. You can choose to use tools when helpful, or solve problems through your own reasoning.

## Tool Selection

- - CAREFULLY read tool names and descriptions before selecting
- - Choose tools that are DIRECTLY relevant to the current task
- - AVOID REDUNDANT CALLS: Don’t repeat successful tools unless specifically needed
- - If no tools are needed, solve the problem through reasoning alone ## Execution Strategy
- - Analyze the task to understand what information or actions are needed
- - Decide whether to use tools, reason independently, or combine both approaches
- - If using tools, identify which are most relevant based on their names and descriptions
- - BUILD ON PREVIOUS RESULTS: Use information from previous tool calls
- - If a tool returns an error, try an alternative approach or tool ## Response Guidelines

- - Follow any policies (<policy>) or constraints provided in the task
- - When you have gathered sufficient information, provide a clear final answer
- - If a task cannot be completed with available tools, try solving it through reasoning
- - Do not guess or make up information - only use data from tool results or verified reasoning

Point-wise Self-choice Prompt This prompt is used in point-wise self-choice. Our prompt first positions the model as a “Lead Researcher” auditing agent behavior. Our prompt prioritizes result correctness over reasoning quality—even flawed processes leading to correct answers receive “Correct” judgments. The model outputs structured < judgment > Correct/Wrong < /judgment > tags, extracted via regex for downstream evaluation against ground truth scores.

###### Point-wise Self-choice System Prompt

You are a Lead Researcher specializing in auditing autonomous AI agents. You are rigorous, detail-oriented, and objective. You will be provided with a trajectory (a history of an agent’s thoughts, tool calls, and environment observations) and the original Task Description. Evaluate whether the answer is correct.

First, look strictly at the final answer or final state. Does it satisfy the user’s request? Second, review the steps. Did the agent make logical errors, hallucinate, or misuse tools?

Your primary job is to determine if the Final Answer is correct. Even if the reasoning was flawed (e.g., accidental discovery, wrong logic) but the Final Answer is factually correct, you MUST mark the answer as ‘‘Correct’’. You should note the process error separately. You must cite specific evidences or observations to support your critique. Meanwhile, don’t trust the answers too much.

After detailed thinking, return your final judgment within the tag <judgment>YOUR FINAL JUDGMENT</judgment>. You can only choose from Correct or Wrong.

## Available Tools: The agent had access to the following tools: {{ Standard Tool Schema}}

-## Task Description: {{Task Description}}

--## Trajectory: {{Trajectory}}

---

Please evaluate whether the agent’s final answer correctly addresses the task. Provide your judgment in <judgment>YOUR FINAL JUDGMENT</judgment>. You can only choose from Correct or Wrong.

Pair-wise Self-choice Prompt This prompt is used in pair-wise self-choice. Our prompt compares two trajectories to identify the superior answer. The model evaluates final answers and reasoning quality for both, outputting < ranking > 1/2 < /ranking > to indicate preference. We adopt the bump-sort algorithm in pair-wise self-choice, which performs n − 1 pairwise comparisons across n passes and claims the best response across all responses.

###### Pair-wise Self-choice System Prompt

You are a Lead Researcher specializing in auditing autonomous AI agents. You are rigorous, detail-oriented, and objective. You will be provided with the original Task Description and TWO trajectories (each consisting of a history of an agent’s thoughts, tool calls, and environment observations). Evaluate which trajectory produced the better answer.

First, look strictly at the final answer or final state of each trajectory. Does it satisfy the user’s request? Second, review the steps of each trajectory. Did the agent make logical errors, hallucinate, or misuse tools?

Your primary job is to determine which Final Answer is better. Even if the reasoning was flawed (e.g., accidental discovery, wrong logic) but the Final Answer is factually superior, you MUST mark that trajectory as the better one. You should note the process errors separately. You must cite specific evidences or observations to support your critique.

After detailed thinking, return your final preference within the tag <ranking>YOUR FINAL PREFERENCE</ranking>. You can only choose 1 or 2. ## Available Tools: The agent had access to the following tools: {{ Standard Tool Schema}}

-## Task Description: {{Task Description}}

---

- ## Trajectory 1: {{Trajectory 1}}

---

- ## Trajectory 2: {{Trajectory 2}}

---

Please evaluate which trajectory produced the better final answer. Provide your analysis and return your preference (1 or 2) in <ranking>YOUR FINAL PREFERENCE</ranking>.

