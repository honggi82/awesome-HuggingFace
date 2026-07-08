# arXiv:2512.23647v1[cs.CL]29Dec2025

[Figure 1]

2025-12-30

## Nested Browser-Use Learning for Agentic Information Seeking

Baixuan Li†() , Jialong Wu†, Wenbiao Yin†() , Kuan Li, Zhongwang Zhang, Huifeng Yin, Zhengwei Tao, Liwen Zhang, Pengjun Xie, Jingren Zhou, Yong Jiang()

[Figure 2]

Tongyi Lab , Alibaba Group

[Figure 3]

https://tongyi-agent.github.io/blog

https://github.com/Alibaba-NLP/DeepResearch

##### Abstract

Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer information available through real browsing. While full browser interaction could unlock deeper capabilities, its fine-grained control and verbose page content returns introduce substantial complexity for ReAct-style function-calling agents. To bridge this gap, we propose Nested Browser-Use Learning (NestBrowse), which introduces a minimal and complete browseraction framework that decouples interaction control from page exploration through a nested structure. This design simplifies agentic reasoning while enabling effective deep-web information acquisition. Empirical results on challenging deep IS benchmarks demonstrate that NestBrowse offers clear benefits in practice. Further in-depth analyses underscore its efficiency and flexibility.

##### 1 Introduction

Information-seeking (IS) agents (Team, 2025c; x.ai, 2025; AI, 2025; Team, 2025a; Gemini, 2025), powered by large language models (LLMs) have achieved strong performance on a variety of challenging wide and deep search tasks (Mialon et al., 2023; Wei et al., 2025; Zhou et al., 2025; XbenchTeam, 2025; Wong et al., 2025). These agents typically follow a ReAct-style function-calling paradigm (Yao et al., 2023), interleaving reasoning with tool invocation to incrementally gather and process evidence. While IS agents can, in principle, leverage a wide range of external information sources, most existing systems primarily rely on the web, which remains the most comprehensive repository of publicly accessible knowledge1.

BrowseComp

35

31.6

29.6

30

28.3

26.4

25

22.4

20

18.8

15.7

14.8

15

14.1

13.4 13.0

12.2

10

DeepDiver-V2 MiroThinker-38B

Claude-4-OpusWebExplorer

-32B-DPO-V0.2Claude-4-Sonnet

UI-TARS-2OpenAI-o4-miniGLM-4.5-355B

-8BDeepDive-32B

Kimi-K2-Instruct

NestBrowse-30B-A3B

NestBrowse

-1T

-4B

Figure 1: Performance of the proposed NestBrowse on the challenging deep IS benchmark BrowseComp.

Most existing IS agents (Wu et al., 2025a; Li et al., 2025c; Tao et al., 2025b; Zhao et al., 2025; Tao et al., 2025a; Li et al., 2025a) model IS with two tools: search, which retrieves query-related URLs, titles,

†Equal contribution. Correspondence to: baixuan@seu.edu.cn, {yinwenbiao.ywb, yongjiang.jy}@alibaba-inc.com.

1Unless otherwise specified, IS agents in this work refer to text-only, ReAct-style LLM-driven systems that use the web as their information source to gather evidence.

and snippets, and a goal-conditioned visit, which fetches the Markdown content of a given URL. While effective for shallow IS, this abstraction provides only partial access to the web. In practice, substantial information and functionality are exposed only through browser-level interactions or state changes, including client-side rendering, dynamic content loading, form-mediated workflows, multi-step navigation, and page-internal online functionalities that go beyond static content retrieval. Such content is neither reliably surfaced by search engines nor recoverable via a single static URL fetch, rendering IS formulations based solely on search and visit inherently incomplete.

Recent work (OpenAI, 2025d; Team, 2025a; Wang et al., 2025) has explored equipping IS agents with real browser interactions. However, there is no widely adopted standard for modeling browser actions, and the inherent complexity of browser-use makes it challenging to abstract diverse interaction patterns into a tool interface suitable for IS agents (Yu et al., 2025; Hu et al., 2025). Moreover, browser pages often contain large amounts of raw and highly redundant information. For ReAct-style IS agents, naively injecting all browser outputs into the context is suboptimal due to excessive redundancy (Kerboua et al., 2025; Wu et al., 2025c; Fang et al., 2025), and can be impractical when individual pages exceed feasible context limits. Consequently, how to design a simple yet complete browser toolkit, together with an effective interaction paradigm that enables IS agents to use such tools efficiently, remains an open challenge.

To address this challenge, we propose Nested Browser-Use Learning (NestBrowse), which consists of three key components. (i) We design a minimal yet functionally complete browser toolkit that provides four core actions, search, visit, click, and fill, enabling comprehensive web information access with low tool complexity. (ii) Building on this toolkit, we introduce the Nested Browser-Use Framework that decouples browser interaction into an outer loop of tool-integrated reasoning and an inner loop of intra-page exploration, enabling controlled and incremental information flow into the agent’s context under practical context constraints. (iii) Finally, we develop a multi-task imitation learning formulation that internalizes these nested browser-use capabilities into a single IS agent model by jointly training outer-loop reasoning and inner-loop evidence extraction.

Building on the proposed NestBrowse, we train two IS agent models at different scales, namely NestBrowse4B and NestBrowse-30B-A3B. Across four highly challenging deep IS benchmarks in both English and Chinese, both models exhibit consistently strong performance. These results support a key insight for IS tasks: with appropriate browser tool abstractions and interaction strategies, small-scale agent models can acquire effective browser-use capabilities and successfully solve complex deep IS problems.

##### 2 Related Work

Tool-Integrated Reasoning. Tool-integrated reasoning (TIR) equips models with external tools and enables them to interleave reasoning with tool invocation to solve complex tasks (Lin & Xu, 2025; Xue et al., 2025). By extending capabilities beyond parametric knowledge, this paradigm supports effective problem solving in settings that require information access, or external computation.

Prior work has identified several common principles for effective TIR (Shen, 2024; Liu et al., 2024b), with tool interface design playing a central role (Patil et al., 2024). In particular, toolkits are typically kept minimal and low in complexity to reduce decision burden (Shen et al., 2023; Qin et al., 2024), avoid confusion during reasoning, and maintain efficiency and smooth information flow between tool execution and subsequent reasoning (Deng et al., 2023; Liu et al., 2023).

Motivated by these principles, we observe that the diversity of low-level browser operations makes naive action aggregation impractical; thus, we propose a minimal yet functionally complete browser toolkit and the Nested Browser-Use Framework for efficient agentic reasoning over compact, goal-aligned page content.

Deep Information Seeking. Deep information-seeking (IS) tasks (Wu et al., 2025b; Wei et al., 2025; Xbench-Team, 2025) require agents to acquire and integrate hard-to-find information from external

sources through iterative reasoning and exploration. Unlike conventional multi-hop QA (Yang et al., 2018; Ho et al., 2020), where explicit entities and intermediate clues are provided, deep IS tasks begin from vague or underspecified prompts and demand that agents actively infer latent clues, entities, and relations. This process can be viewed as the incremental construction of an implicit entity-relation graph (Li et al., 2025c; Tao et al., 2025b; Li et al., 2025a), driven by hypothesis generation, evidence gathering, and verification. Such a detective-style reasoning process poses substantial challenges, even for humans.

Accordingly, IS agents interleave reasoning with information acquisition to gather task-relevant evidence. Prior work largely focuses on web-based IS agents using ReAct-style tool invocation (Wu et al., 2025b;a; Qiao et al., 2025; Team, 2025c), where a key challenge remains the design of effective interaction abstractions and tool-use strategies for information acquisition.

In this work, we address these challenges by introducing a practical methodology that enables IS agents to leverage more realistic browser-based interactions, allowing them to acquire more comprehensive and better-structured web information for solving complex deep IS tasks.

##### 3 Nested Browser-Use Learning

In this section, we first present a minimally complete browser toolkit (§3.1), followed by the nested browser-use framework that enables IS agents to efficiently and effectively leverage this browser toolkit for web information acquisition (§3.2). We then introduce a multi-task imitation learning paradigm to internalize IS and browser-use capabilities into a base model (§3.3).

###### 3.1 Minimally Complete Browser Toolkit

We implement a headless browser backend2 in Playwright for programmatic web interaction, parsing each page from raw HTML into a semantic DOM snapshot that exposes interactive-element identifiers for subsequent actions while presenting structured, LLM-readable content.

To abstract browser interactions into tools usable by IS agents, we partition web information into two categories, I = Istatic ∪ Idynamic: (i) static information Istatic, accessible through a single page load without in-page interaction; and (ii) dynamic information Idynamic, exposed only via browser-level interactions such as client-side rendering, incremental loading, or user-triggered actions. Mainstream IS tool abstractions focus on query-based retrieval (search) and page-level fetching (visit), which suffice for Istatic but provide limited access to Idynamic. At the same time, overly fine-grained browser tool formulations are undesirable, as increased action-space complexity substantially amplifies the decision burden for IS agents (Xu et al., 2025), hindering efficient information acquisition.

Accordingly, we balance functional completeness for web information access with minimal tool complexity, and propose a minimally complete browser toolkit consisting of four tools:

- • search: Performs batched Google queries and returns the top-10 ranked results for each.
- • visit: Fetches webpage from a URL and extracts information relevant to the given goal.
- • click: Interacts with a clickable element, potentially triggers a page transition, and extracts content relevant to the given goal.
- • fill: Types text into form fields or other editable elements within the current page.

The search and visit follow the standard tool configurations adopted by most existing IS agents and serve as the primary means for accessing static information Istatic. To cover the full spectrum of dynamic information Idynamic, we introduce two additional browser actions, click and fill. Together, these four tools complete the information access pathway for web-based IS agents, ensuring functional completeness while introducing only two additional actions and keeping tool complexity within a highly usable range.

2Notably, our setup considers only textual page content and does not incorporate any visual information.

It is worth noting that operations such as scrolling and in-page search, which are commonly included in existing browser toolkits, are omitted in our design; the rationale is discussed in §3.2.

###### 3.2 Nested Browser-Use Framework

Assistant Assistant Assistant Assistant Assistant

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

... Assistant

... Assistant

Think ...

Think ...

... Useful Info

... Useful Info

Workspace ...

Workspace ...

| | |
|---|---|
|Inner-Loop<br><br>|Page Exploration|

###### Inner-Loop Page Exploration

n

Top-10

Browser Toolkit

###### visit

###### search

###### click

Return Goal-Relevant

Return Goal-Relevant

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|[Text-Only]<br><br>search ← query<br><br>visit ← URL, goal click ← id, goal<br><br>fill ← id, text|
|---|

[Figure 21]

Useful Info Workspace

Useful Info Workspace

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|𝒫1 , 𝒫2, ..., 𝒫𝑁|
|---|

|𝒫1|
|---|

𝒲∗

|𝒲∗|
|---|

Task Initialization

Concat

Concat

Concat

User Query

Search Query:

Visit URL: “https://openreview.net”

Click Button: “NeurIPS (identifier)”

[“Openreview”]

Visit Goal: “Find the NeurIPS Venue”

Visit Goal: “Find the 2022 Entry”

According to Openreview.net,

   

at the NeurIPS 2022 Venue,

Assistant

Assistant

Assistant

Assistant

how many papers by an author named Yuri were accepted with

... Assistant Think ...

... Assistant Think ...

... Assistant Think ...

... Assistant Think ...

Continue ...

a “certain” recommendation?

... Assistant call click ...

... Assistant call search ...

... Assistant call visit ...

... Assistant call click ...

Outer-Loop Agentic Reasoning

Figure 2: Overview of the Nested Browser-Use Framework. The outer loop interleaves reasoning and tool calls to solve the user task. Page-transition actions trigger an inner loop for intra-page exploration, which extracts and returns goal-relevant content to the outer loop, forming a nested interaction structure.

Why Nested Browser-Use Matters. Although the proposed browser toolkit enables functionally complete access to web information, raw page content is often highly redundant. A single page can easily exceed 64K tokens and may even surpass 1M tokens, rendering it infeasible for IS agents operating under typical context limits of 128K or 256K (Liu et al., 2024a; Yang et al., 2025). Truncating overlong pages is a common workaround (Wu et al., 2025a; Li et al., 2025c), but inevitably risks discarding information.

From a goal-driven perspective, the objective of an IS agent is to acquire only the subset of page information necessary to resolve a task goal g. For a visited page with content P, only a small subset Pg ⊂ P is goal-relevant, while most content P \ Pg is extraneous. Injecting full page content into the agent’s context wastes token budget and impairs subsequent reasoning. Accordingly, we exclude scrolling and in-page search, as they merely limit content exposure per page read without improving goal-directed information acquisition, resulting in inefficient browser-use for IS tasks.

How Nested Browser-Use Works. To bridge the gap between browser-use and IS tasks, we decouple browser interaction into an outer loop of tool-integrated reasoning and an inner loop of goal-driven page exploration. This framework is termed nested since the inner loop is fully contained within specific outer-loop tool invocation (Figure 2).

The outer loop follows a standard ReAct-style function-calling paradigm. Let the agent context at outer-loop step t be ct, and let T denote the browser toolkit. The agent operates by

(at, ηt) ∼ pθ(· | ct), at ∈ T , (1)

where pθ denotes the parameterized IS agent model, at is the selected tool at step t, and ηt includes tool arguments. Executing the tool produces a response

rt = Exec(at, ηt), (2)

which is appended to the agent’s context to form the agent state for the next reasoning step,

ct+1 = Update(ct, at, ηt,rt). (3) The outer loop repeats until a task-level termination condition is satisfied, including successful task completion or predefined resource constraints (e.g., sequence length or the number of tool calls).

When the invoked tool transitions the agent into a new page, an inner loop is instantiated for intra-page exploration. Let gt denote the goal passed from the outer loop, and let P denote the raw textual content of the visited page. We partition P into a sequence of segments {Pi}iN=1 and explore them incrementally under a goal-driven procedure. Concretely, the agent maintains a temporary workspace W initialized as ∅ and iteratively updates it as

W ← W ∪ f(Pi, gt), (4) where f(Pi, gt) extracts information from segment Pi that is relevant to the goal gt. Only goal-relevant content is accumulated in W, while irrelevant segments are discarded. The inner loop terminates once the exploration of all the segmented page content is complete. The interaction between the outer and inner loops is formalized by defining tool execution via inner-loop aggregation. Specifically,

 

W⋆(gt, P), if at ∈ Tpage, Execbase(at, ηt), otherwise,

Exec(at, ηt) =

(5)



where W⋆(gt, P) denotes the workspace produced upon termination of the inner loop under goal gt on page P, and Execbase corresponds to standard execution for tools that do not induce page transitions. Here, at denotes the tool selected by the agent at outer-loop step t, and Tpage denotes the subset of tools that initiate a transition into a new page, which in our design includes visit and click.

As a result, the outer loop receives a compact and goal-aligned3 response rt = W⋆ instead of the raw page content P, reflecting the core design of the Nested Browser-Use Framework and enabling controlled information injection for subsequent reasoning and action. During execution, the agent’s outer-loop is serialized in a structured format: free-form reasoning is enclosed within <think> and </think> tags, tool invocations are wrapped by <tool_call> and </tool_call>, and tool responses are encapsulated using <tool_response> and </tool_response>. For the inner loop, the temporary workspace is enclosed within <useful_info> and </useful_info> tags and returned as the interface to the outer loop.

###### 3.3 Multi-Task Imitation Learning

Task and Supervision Sources. Comprehensive browser-use capabilities are best elicited under challenging IS tasks. Simple single-hop or multi-hop QA training sets (Yang et al., 2018; Ho et al., 2020; Hendrycks et al., 2021) are insufficient for this purpose, as they can often be solved with only one or a few search queries, without requiring substantive browsing or reasoning. Accordingly, we adopt SailorFog-QA-V2 (Li et al., 2025b) as our primary training task, a high-quality QA set designed to elicit complex browsing behaviors that require both effective tool use and multi-step reasoning from IS agents.

Quality Filtering via Rejection Sampling. We construct trajectories following the procedure described in §3.2. Because interleaving reasoning with browser interaction constitutes a challenging agentic capability, and model rollouts do not always exhibit the desired behavior. To encourage correct agentic browsing, we apply rejection sampling (Wu et al., 2025a) to the generated trajectories, retaining only high-quality trajectories that satisfy predefined criteria while discarding those that exhibit incorrect or undesirable patterns. This filtering biases supervision toward effective agentic behaviors, avoiding the internalization of spurious patterns. Specifically, we apply rejection based on the following three criteria:

- ◦ Format violations. Trajectories that do not adhere to the required output format are discarded, such as failing to enclose reasoning content within <think> and </think> tags.

3For this reason, our browser tools that introduce new pages explicitly take the goal as an input parameter.

- ◦ Tool-call hallucinations. We reject trajectories that contain invalid tool names or tool arguments that cannot be correctly executed.
- ◦ Incorrect final answers. We assume that only trajectories leading to a correct final answer provide meaningful supervision for learning agentic browsing and reasoning behaviors, and thus discard trajectories with incorrect outcomes.

Notably, we do not apply additional rejection criteria to intermediate reasoning or browsing steps. This choice preserves diversity in the supervision signal and avoids over-constraining agent behavior with brittle, manually specified rules.

Multi-Task Imitation Learning Objective. Because the proposed Nested Browser-Use Framework naturally induces multiple learning objectives, training an IS agent requires a multi-task learning formulation. Specifically, the outer loop aims to derive the final answer through interleaved reasoning and tool use, while the inner loop focuses on extracting goal-relevant evidence from pages and incrementally maintaining a temporary workspace.

We first define an outer-loop imitation objective over accepted agent trajectories constructed following §3.2. Each trajectory consists of a sequence of outer-loop steps indexed by t, where the agent observes a context ct and produces a serialized continuation that includes reasoning, tool invocations, and tool responses. We optimize a token-level negative log-likelihood objective:

#### ∑

Lout(θ) = E ∑

t

j

− log pθ yt,j | ct, yt,<j , (6)

where pθ denotes the parameterized IS agent model and yt,j denotes the j-th token in the target serialized IS agent output at step t (Chen et al., 2023).

For the inner loop, we supervise the agent model to generate goal-relevant evidence extracted from each page segment and written into the temporary workspace. Let Pi denote the i-th segment of a page and gt the goal passed from the outer loop. The inner-loop objective is defined as

### Lin(θ) = E ∑

t,i

#### ∑

− log pθ ut,i,j | ht,i,j , (7)

j

where ht,i,j denotes the context consisting of the goal gt, page segment Pi, and previously generated tokens ut,i,<j, and ut,i,j denotes the j-th token of the extracted goal-relevant content for segment Pi. We jointly optimize the two objectives using a weighted multi-task loss:

LMT(θ) = λout Lout(θ) + λin Lin(θ), (8)

where λout and λin balance trajectory-level supervision between outer-loop agentic reasoning with tool use and inner-loop goal-driven page exploration and evidence extraction4. Through this multi-task objective, nested browser-use capabilities are jointly trained within a single IS agent model.

##### 4 Experiments

###### 4.1 Setup

Benchmarks. We evaluate the proposed NestBrowse on a set of widely recognized and highly challenging deep IS benchmarks. Specifically, we consider English benchmarks including BrowseComp (Wei et al., 2025) and GAIA (Mialon et al., 2023), as well as Chinese benchmarks including BrowseComp-zh (Zhou et al., 2025) and XBench-DeepSearch (XBench) (Xbench-Team, 2025). All benchmarks are web-based QA tasks that require strong agentic reasoning with tool use to locate, navigate, and synthesize hard-to-find web information that is not directly retrievable via simple search queries.

4By default, we use equal weighting with λout = λin = 1.

Following prior work (Li et al., 2025d), we use the 103-question text-only subset of GAIA, while employing the full datasets for the remaining benchmarks. Performance is measured by final answer accuracy. For answer verification, we adopt an LLM-as-a-Judge protocol (Gu et al., 2024) using GPT-4.1 (OpenAI, 2025a), following the official evaluation prompts provided by each benchmark.

Baselines. We comprehensively compare the proposed NestBrowse with open-source IS agents and additionally include several proprietary IS agents for reference. Baseline details are provided in Table 1, with some results taken from prior work or official leaderboards.

NestBrowse Implementation. We select two open-source models as base models for training our browser-use IS agents: Qwen3-4B-Thinking-2507 and Qwen3-30B-A3B-Thinking-2507 (Yang et al., 2025). For both models, we set the maximum context length to 128K tokens and cap the number of tool invocations at 100. If an agent fails to produce a final answer upon reaching this limit, the episode is forcibly terminated. Following the proposed multi-task training procedure (§3.3), we train the 4B model for approximately 1,344 GPU hours and the 30B-A3B model for approximately 4,096 GPU hours on an NVIDIA H20 GPU cluster, resulting in our final models, NestBrowse-4B and NestBrowse-30B-A3B.

###### 4.2 Overall Performance

Table 1: Main results on four challenging IS benchmarks. In the Web Toolkit column, browser (text) indicates that only textual elements returned by the browser are used. All resumlts are reported using the pass@1 metric. For Xbench, † denotes the more challenging 2510 version; otherwise the 2505 version.

Model / Framework Web Toolkit BrowseComp BrowseComp-zh GAIA XBench Closed-Source Information-Seeking Agents

Claude-4-Sonnet (Anthropic, 2025) not reported 12.2 29.1 68.3 64.6 Claude-4-Opus (Anthropic, 2025) not reported 18.8 37.4 – – Kimi Researcher (Team, 2025a) browser (text) – – – 69.0 OpenAI-o4-mini (OpenAI, 2025c) browser (text) 28.3 44.3 – – OpenAI-o3 (OpenAI, 2025c) browser 49.7 58.1 70.5 66.7 OpenAI DeepResearch (OpenAI, 2025d) browser 51.5 42.9 67.4 – UI-TARS-2 (Wang et al., 2025) browser 29.6 50.5 – –

###### Open-Source Information-Seeking Agents

ASearcher-Web-32B (Gao et al., 2025) search, visit 5.2 15.6 52.8 42.1 DeepDive-32B (Lu et al., 2025) search, visit 14.8 25.6 – 50.5 DeepDiver-V2-38B (Team, 2025b) search 13.4 34.6 – 53.0 Kimi-K2-Instruct-1T (Team et al., 2025a) search, visit 14.1 28.8 57.7 50.0 GLM-4.5-355B (Zeng et al., 2025) not reported 26.4 37.5 66.0 70.0 WebExplorer-8B (Liu et al., 2025) search, visit 15.7 32.0 50.0 53.7 WebDancer-QwQ-32B (Wu et al., 2025a) search, visit 3.8 18.0 51.5 38.3 WebSailor-32B (Li et al., 2025c) search, visit 10.5 25.5 53.2 53.3 WebSailor-72B (Li et al., 2025c) search, visit 12.0 30.1 55.4 55.0 WebShaper-QwQ-32B (Tao et al., 2025b) search, visit – – 53.3 35.0 MiroThinker-32B-DPO-V0.2 (Team et al., 2025b) search, visit 13.0 17.0 64.1 – WebSailor-V2-30B-A3B-SFT (Li et al., 2025b) search, visit 24.4 28.3 66.0 61.7 WebLeaper-30B-A3B-RU (Tao et al., 2025a) search, visit 23.0 – 67.0 66.0

NestBrowse-4B browser (text) 22.4 28.4 68.9 74.0 | 38.0 † NestBrowse-30B-A3B browser (text) 31.6 42.6 75.7 75.0 | 45.0 †

We evaluate NestBrowse against a broad set of widely adopted IS agent baselines across four challenging deep IS benchmarks. As shown in Table 1, NestBrowse-30B-A3B consistently delivers strong performance, exceeding that of leading open-source IS agents and remaining competitive with, or outperforming, several proprietary systems. Although NestBrowse is trained solely on English data, its browser-use and IS capabilities generalize well to out-of-distribution Chinese benchmarks.

Notably, even NestBrowse-4B achieves competitive performance, exceeding many IS agents with sub-

stantially larger parameter counts. This observation underscores a key insight for IS tasks: performance is not solely determined by model scale, but is critically influenced by how agents are designed to access, organize, and interact with external information sources. From this perspective, we argue that training agentic capabilities on relatively small models is highly valuable (Team, 2025c; Belcak et al., 2025). With appropriate tool abstractions and interaction strategies, small agent models can achieve performance comparable to, or even exceeding, that of substantially larger systems.

###### 4.3 Analysis of Browser-Use Strategies

To validate the design choices of NestBrowse, we conduct an ablation study (Table 2) comparing three browser-use strategies. Naive uses a standard browser-use setup without tool simplification or page-level information extraction; Simplified adopts the four-action toolkit of NestBrowse without goal-conditioned extraction; and Compressed applies goal-relevant page content extraction while retaining the original toolkit.

Table 2: Ablation study on browser-use settings. Simp. indicates whether the browser toolkit is simplified, and Extr. indicates whether goal-relevant content extraction is applied to page responses.

Setting Simp. Extr. GAIA XBench

Naive ✗ ✗ 46.6 40.0 Simplified ✓ ✗ 55.3 40.0 Compressed ✗ ✓ 60.2 61.0

To isolate the impact of strategy design, all variants are executed using the same strong agent model, GPT-OSS-120B (OpenAI, 2025b). Both toolkit simplification and goal-relevant page content extraction independently improve performance, with extraction yielding larger gains by reducing redundancy and focusing agent reasoning on relevant information. Combining both components, NestBrowse achieves the strongest performance, exhibiting a clear additive effect.

NestBrowse ✓ ✓ 73.8 71.0

###### 4.4 Context Efficiency of NestBrowse

We analyze the context efficiency of NestBrowse to assess its ability to maintain controlled information flow during browser-use. As shown in Figure 3, Whole Information Processed denotes the total page content handled jointly by the outer and inner loops, while the other curve shows the context length maintained in the outer (agentic reasoning) loop. After approximately 20 toolcall turns, the total processed information already exceeds the agent’s maximum context limit (128K tokens). Without NestBrowse, execution would terminate at this point, even though about 85% of tasks remain unfinished. By injecting only goal-relevant content into the outer loop, NestBrowse keeps the agent context within feasible limits throughout execution, substantially improving the practicality and effectiveness of browser-based information seeking.

800k

200

Outer Loop Trajectory

Whole Information Processed

700k

Remaining#Trajectory

175

Averaged#Token

600k

150

500k

125

400k

100

300k

75

200k

50

128k

100k

25

0

0

0 20 40 60 80 100

Turn Index

Figure 3: Average context length over tool-call turns for NestBrowse-30B-A3B on the BrowseComp subset. Gray bars denote the number of remaining active trajectories at each turn.

###### 4.5 Analysis of Intra-Page Exploration

We further analyze the effectiveness of intrapage exploration in the NestBrowse inner loop, and evaluate the model along two dimensions using GPT-4.1 as the judge model: raw snapshot retention, which measures how well the agent preserves original page snapshot to support subsequent interactions, and goal-relevant extraction accuracy, which assesses whether the extracted content aligns with the given goal. As shown in Figure 4, compared to the base agent model, NestBrowse achieves substantial improvements on both metrics. These results validate the effectiveness of our multi-task training scheme, which simultaneously strengthens outer-loop agentic reasoning and inner-loop intra-page exploration, integrating complex browseruse capabilities into a single IS agent model.

Figure 4: Intra-page exploration performance over innerloop instances from 100 trajectories per benchmark.

We also analyze the impact of innerloop intra-page exploration quality on outer-loop agentic reasoning. As shown in Table 3, weaker inner-loop models degrade outer-loop performance, while stronger ones yield corresponding improvements, indicating a clear positive relationship between the two. This finding under-

Table 3: Deep IS performance with different inner-loop models on the BrowseComp subset (100 examples).

Outer-Loop Model Inner-Loop Model BrowseComp

NestBrowse-4B 24.0 NestBrowse-30B-A3B 35.0 GPT-OSS-120B 36.0

NestBrowse-30B-A3B

scores intra-page exploration as a critical component of information acquisition in IS tasks and provides empirical justification for jointly training the outer and inner loops under a unified multi-task objective.

###### 4.6 Case Study: Beyond Static Web Access

[Figure 30]

[Figure 31]

Question: Given 𝑥0 = −5 and 𝑓 𝑥 = 𝑥3 + 4𝑥2 − 3𝑥 + 8,

what is the smallest 𝑛 where using Newton's Method 𝑛 = 𝑛 +

1 after rounding to four decimal places?

... Search for Newton’s method and get the relevant URL ...

... Visit an online calculator (https://planetcalc.com/7748/) ...

[Figure 32]

[Figure 33]

Assistant: <think> The planetcalc page seems to have a default

1 2

function 𝑥3 + 3𝑥2 + 12𝑥 + 8, not our function. We need to

change the function to 𝑓 𝑥 = 𝑥3 + 4𝑥2 − 3𝑥 + 8.

3

... [intermediate thoughts] ...

[Figure 34]

[Figure 35]

We’ll fill function: “𝑥3 + 4𝑥2 − 3𝑥 + 8”. Fill initial value: “-5”. Then click calculate button [ref=e105].

Let’s attempt. </think> <tool_call> ... [call fill] ... </tool_call>

... Fill the function and initial value into the input fields ...

... Click the “Calculate” button to submit the computation ...

... Continue reasoning based on the page content ...

Figure 5: A case study where an online browser-based calculator is used for numerical computation.

We further analyze NestBrowse through a representative case study, shown in Figure 5. When solving a GAIA task involving numerical computation, NestBrowse-30B-A3B adopts a browser-based strategy by invoking an online calculator to perform the required computation. For a small-scale model, such numerical reasoning would be difficult to handle reliably through internal reasoning alone. By leveraging browser-use to exploit a computational utility embedded within the webpage, the agent substantially reduces its reasoning burden and successfully completes the task.

This case highlights a key advantage of browser-use beyond retrieving static page content. Rather than serving solely as a medium for information access, browser-use enables agents to exploit the rich functional ecosystem of the web, including interactive utilities that are inaccessible through static page fetching alone. From this perspective, browser-use can be viewed as a form of meta tool-use, where the effective toolkit available to an agent encompasses the broad collection of tools implicitly embedded within the web itself. NestBrowse further facilitates the acquisition of such meta tool-use capabilities, enabling agents to solve complex IS tasks in a more flexible and effective manner.

##### 5 Conclusion

We present Nested Browser-Use Learning (NestBrowse), a learning framework that equips informationseeking (IS) agents with a minimal yet complete browser toolkit and a nested browser-use paradigm for efficient deep web information access. By decoupling agentic reasoning from intra-page exploration and training these capabilities jointly, NestBrowse enables strong deep IS performance across relatively small model scales. Our results suggest that principled browser-use abstraction and interaction strategies are key to solving complex IS tasks, even without relying on large-scale agent models.

##### 6 Limitations and Future Work

While our proposed NestBrowse successfully enables small models to acquire complex browser-use capabilities and interact with real browsers to collect information for downstream reasoning, we intentionally restrict our browser modeling to textual content. This design choice is motivated by the need to isolate the effects of tool abstraction and interaction paradigms, and to maintain a controlled and interpretable experimental setting. Incorporating additional modalities such as vision or audio would introduce substantial modeling and system complexity, making it difficult to disentangle the contributions of browser-use learning from those of multimodal perception. Nevertheless, non-textual modalities often convey important information in real-world browsing scenarios. Extending the proposed NestBrowse to support multimodal information acquisition and reasoning represents a promising future direction.

##### References

Skywork AI. Skywork-deepresearch. https://github.com/SkyworkAI/Skywork-DeepResearch, 2025. Anthropic. Introducing claude 4, 2025. URL https://www.anthropic.com/news/claude-4. Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu, Xin Dong, Saurav Muralidharan, Yingyan Ce-

line Lin, and Pavlo Molchanov. Small language models are the future of agentic ai. arXiv preprint arXiv:2506.02153, 2025.

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915, 2023.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl. arXiv preprint arXiv:2508.07976, 2025.

Gemini. Gemini deep research, 2025. URL https://gemini.google.com/app. Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR. OpenReview.net, 2021.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps, 2020. URL https://arxiv.org/abs/2011.0 1060.

Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for computer, phone and browser use. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7436–7465, 2025.

Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Léo Boisvert, Massimo Caccia, Jérémy Espinas, Alexandre Aussem, Véronique Eglin, and Alexandre Lacoste. Focusagent: Simple yet effective ways of trimming the large context of web agents. arXiv preprint arXiv:2510.03204, 2025.

Baixuan Li, Dingchu Zhang, Jialong Wu, Wenbiao Yin, Zhengwei Tao, Yida Zhao, Liwen Zhang, Haiyang Shen, Runnan Fang, Pengjun Xie, et al. Parallelmuse: Agentic parallel thinking for deep information seeking. arXiv preprint arXiv:2510.24698, 2025a.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, Xinyu Wang, Zile Qiao, Zhen Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor-v2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning, 2025b. URL https://arxiv.org/abs/2509.13305.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025c. URL https://arxiv.org/abs/2507.02592.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. CoRR, abs/2504.21776, 2025d. doi: 10.48550/ARXIV.2504.21776. URL https://doi.org/10.48550/a rXiv.2504.21776.

Heng Lin and Zhongwen Xu. Understanding tool-integrated reasoning. arXiv preprint arXiv:2508.19201, 2025.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, et al. Webexplorer: Explore and evolve for training long-horizon web agents. arXiv preprint arXiv:2509.06501, 2025.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, et al. Toolace: Winning the points of llm function calling. arXiv preprint arXiv:2409.00920, 2024b.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In The Twelfth International Conference on Learning Representations, 2023.

Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, and Yuxiao Dong. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn rl. arXiv preprint arXiv:2509.10446, 2025.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.

OpenAI. Introducing openai gpt-4.1, 2025a. URL https://openai.com/index/gpt-4-1/. OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025b. URL https://arxiv.org/abs/2508.10925. OpenAI. Introducing openai o3 and o4-mini, 2025c. URL https://openai.com/index/introducing-o

3-and-o4-mini/.

OpenAI. Deep research system card, 2025d. URL https://cdn.openai.com/deep-research-system-c

ard.pdf.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.

Zile Qiao, Shen Huang, Jialong Wu, Kuan Li, Wenbiao Yin, Xinyu Wang, Liwen Zhang, Baixuan Li, Zhengwei Tao, Weizhou Shen, Xixi Wu, Yong Jiang, Pengjun Xie, Fei Huang, Jun Zhang, and Jingren Zhou. WebResearcher: Unleashing unbounded reasoning capability in long-horizon agents, 2025.

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, et al. Tool learning with foundation models. ACM Computing Surveys, 57

(4):1–40, 2024.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yu Zhuang. HuggingGPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems, volume 36, pp. 7136–7153, 2023. URL https://proceedings.neurips.cc/paper_files/pap er/2023/hash/694229972352433017a117500583a547-Abstract-Conference.html.

Zhuocheng Shen. Llm with tools: A survey. arXiv preprint arXiv:2409.18807, 2024. Zhengwei Tao, Haiyang Shen, Baixuan Li, Wenbiao Yin, Jialong Wu, Kuan Li, Zhongwang Zhang,

Huifeng Yin, Rui Ye, Liwen Zhang, et al. Webleaper: Empowering efficiency and efficacy in webagent via enabling info-rich seeking. arXiv preprint arXiv:2510.24697, 2025a.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. WebShaper: Agentically data synthesizing via information-seeking formalization, 2025b.

Kimi Team. Kimi researcher tech report, 2025a. URL https://moonshotai.github.io/Kimi-Researche

r/.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025a.

MiroMind AI Team et al. Mirothinker: An open-source agentic model series trained for deep research and complex, long-horizon problem solving, 2025b.

OpenPangu Team. Openpangu deepdiver-v2: Multi-agent learning for deep information seeking, 2025b. URL https://ai. gitcode. com/ascend-tribe/openPangu-Embedded-7B-DeepDiver, 2025b.

Tongyi DeepResearch Team. Tongyi deepresearch: A new era of open-source ai researchers. https:

###### //github.com/Alibaba-NLP/DeepResearch, 2025c.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, et al. Widesearch: Benchmarking agentic broad info-seeking. arXiv preprint arXiv:2508.07999, 2025.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webdancer: Towards autonomous information seeking agency, 2025a. URL https://arxiv.org/abs/2505.22648.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. Webwalker: Benchmarking llms in web traversal, 2025b. URL https://arxiv.org/abs/2501.07572.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025c.

x.ai. Grok 3 beta — the age of reasoning agents, 2025. URL https://x.ai/news/grok-3. Xbench-Team. Xbench-deepsearch, 2025. URL https://xbench.org/agi/aisearch. Baixuan Xu, Tianshi Zheng, Zhaowei Wang, Hong Ting Tsang, Weiqi Wang, Tianqing Fang, and Yangqiu

Song. The cognitive bandwidth bottleneck: Shifting long-horizon agent from planning with actions to planning with schemas. arXiv preprint arXiv:2510.07091, 2025.

Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: Endto-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Tao Yu, Zhengbo Zhang, Zhiheng Lyu, Junhao Gong, Hongzhu Yi, Xinming Wang, Yuxuan Zhou, Jiabing Yang, Ping Nie, Yan Huang, et al. Browseragent: Building web agents with human-inspired web browsing actions. arXiv preprint arXiv:2510.10666, 2025.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Yida Zhao, Kuan Li, Xixi Wu, Liwen Zhang, Dingchu Zhang, Baixuan Li, Maojia Song, Zhuo Chen, Chenxi Wang, Xinyu Wang, et al. Repurposing synthetic data for fine-grained search agent supervision. arXiv preprint arXiv:2510.24694, 2025.

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, et al. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. arXiv preprint arXiv:2504.19314, 2025.

##### A Appendix

###### A.1 Implementation Details: Prompts

We report all prompts used in the NestBrowse. We begin with the prompts for outer-loop agentic reasoning. Following the standard ReAct-style function-calling format, we adopt the system prompt shown below. The user prompt in the outer loop corresponds directly to the user query, without any additional instructional or guiding content.

###### SYSTEM PROMPT (OUTER LOOP)

You are a browser-use agent. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags: <tools> {BROSWER_TOOLS_SCHEMA} </tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call>

We then present the prompts used for inner-loop page exploration. This component is responsible for extracting goal-relevant content from a webpage given a specified goal, and producing outputs in a predefined JSON format. The corresponding system prompt is shown below. The content enclosed within the <useful_info> and </useful_info> tags corresponds to the temporary workspace described in §3.2, which incrementally maintains the extracted goal-relevant information during intra-page exploration.

SYSTEM PROMPT (INNER LOOP) You must answer only by outputting a single valid JSON object, with no extra text before or after it.

Your task: given webpage content and a user goal, extract and organize the useful information according to the following schema: {"rational": "string", "evidence": "string", "summary": "string"}.

Follow these rules for each field:

- 1) rational: Locate the **specific sections/data** directly related to the user’s goal within the webpage content.
- 2) evidence: Identify and extract the **most relevant information** from the content, never miss

- any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
- 3) summary: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

Formatting requirements: Output only one valid JSON object wrapped inside <useful_info> and </useful_info> tags: use double quotes (") for all keys and string values, no trailing commas, and the top-level structure must be exactly: {"rational": "...", "evidence": "...", "summary": "..."}.

For the inner loop, the user prompt combines the given webpage content with the specified goal, instructing the model to extract relevant information and produce outputs in the required format.

USER PROMPT (INNER LOOP) Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** {raw_response}

## **User Goal** {goal}

## **Task Guidelines**

- 1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user’s goal within the webpage content.
- 2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
- 3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**

When a webpage is partitioned into multiple segments, the inner loop iteratively updates the workspace. Specifically, the evidence and summary fields from the workspace produced in the previous iteration are extracted and appended to the current user prompt (incremental). This provides the model with the information already collected, preventing redundant extraction and enabling incremental workspace maintenance. Upon completion of the inner loop, the final workspace is returned to the outer loop for subsequent reasoning.

###### USER PROMPT INCREMENTAL (INNER LOOP)

Please process the following webpage content and user goal to increamentally extract relevant information:

## **Webpage Content** {raw_response}

## **User Goal** {goal}

## **Task Guidelines**

- 1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user’s goal within the webpage content
- 2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
- 3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

## **Existing Evidence** {existing_evidence} ## **Existing Summary** {existing_summary}

Note: Existing extracted evidence and summaries are already provided. You must build upon and integrate these existing pieces of information to perform incremental processing. Produce a consolidated final result that incorporates both the provided and newly added information, without indicating which parts are new or incremental.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**

