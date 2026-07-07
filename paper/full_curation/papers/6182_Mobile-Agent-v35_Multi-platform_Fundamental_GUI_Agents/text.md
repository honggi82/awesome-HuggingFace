# arXiv:2602.16855v1[cs.AI]15Feb2026

[Figure 1]

February 20, 2026

Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents

Haiyang Xu∗† Xi Zhang∗ Haowei Liu∗ Junyang Wang∗ Zhaoqing Zhu∗ Shengjie Zhou Xuhao Hu Feiyu Gao Junjie Cao Zihua Wang Zhiyuan Chen Jitong Liao Qi Zheng Jiahui Zeng Ze Xu Shuai Bai Junyang Lin Jingren Zhou Ming Yan †

[Figure 2]

Tongyi Lab , Alibaba Group

{shuofeng.xhy, ym11960}@alibaba-inc.com

https://github.com/X-PLUG/MobileAgent

Abstract

The paper introduces GUI-Owl-1.5, the latest native GUI agent model that features instruct/thinkingvariantsinmultiplesizes(2B/4B/8B/32B/235B)andsupportsarangeofplatforms (desktop, mobile, browser, and more) to enable cloud-edge collaboration and real-time interaction. GUI-Owl-1.5 achieves state-of-the-art results on more than 20+ GUI benchmarks on open-source models: (1) on GUI automation tasks, it obtains 56.5 on OSWorld, 71.6 on AndroidWorld, and 48.4 on WebArena; (2) on grounding tasks, it obtains 80.3 on ScreenSpotPro; (3) on tool-calling tasks, it obtains 47.6 on OSWorld-MCP, and 46.8 on MobileWorld; (4) on memory and knowledge tasks, it obtains 75.5 on GUI-Knowledge Bench. GUI-Owl-1.5 incorporates several key innovations: (1) Hybird Data Flywheel: we construct the data pipeline for UI understanding and trajectory generation based on a combination of simulated environments and cloud-based sandbox environments, in order to improve the efficiency and quality of data collection. (2) Unified Enhancement of Agent Capabilities: we use a unified thought-synthesis pipeline to enhance the model’s reasoning capabilities, while placing particular emphasis on improving key agent abilities, including Tool/MCP use, memory and multi-agent adaptation; (3) Multi-platform Environment RL Scaling: We propose a new environment RL algorithm, MRPO, to address the challenges of multi-platform conflicts and the low training efficiency of long-horizon tasks. The GUI-Owl-1.5 models are open-sourced, and an online cloud-sandbox demo is available at https://github.com/X-PLUG/MobileAgent.

OSWorld-Verified

WindowsAgentArena

AndroidWorld

80

62.9

44.8 42.9 42.1

73.3 71.6 69.7 67.7

56.5

39.6

60

64.2 63.7

53.1

40

50.1

35.1

60

45.0 43.9

30.9

40

40

20

20

20

0

0

0

UI-Tars-2GUI-Owl-1.5-32BGemini-2.5-ProGELab-Zero-8BQwen3-VL-32B-ThinkUI-Tars-1.5

Claude-4-5-sonnetGUI-Owl-1.5-32BUI-TARS-2GUI-Owl-1.5-8BOpenCUA-72BClaude-4-sonnet

GUI-Owl-1.5-32BQwen3-VL-32B-ThinkUI-TARS-1.5SeedVL-1.5GUI-Owl-1.5-8BQwen3-VL-32B-Instruct

VisualWebArena

OSWorld-MCP

MobileWorld

54.4 52.9

47.6

46.8

43.3 41.8

41.8 41.7

46.6

39.1 38.4

40

40

40

27.2

26.4

20.9

20

20

16.7

20

11.2

9.0

5.5

0

0

0

Gemini-2.5-Flash GUI-Owl-1.5-32BGPT-4oGPT-5 Llama-3-70B-InstructViGoRL-8B

GUI-Owl-1.5-32BGUI-Owl-1.5-8BMAI-UI-235B-A22BQwen3VL-32B-InstructUI-Tars-1.5Qwen3VL-8B-Instruct

GUI-Owl-1.5-32BClaude-4-sonnetQwen3VL-235B-A22B-ThinkGUI-Owl-1.5-8B SeedVL-1.5Gemini-2.5-Pro

OSWorld-G

ScreenSpot-Pro

GUI Knowledge Bench

66.8 65.8 65.1 62.9

80.3 77.8

75.5 73.3 72.9 71.0 67.8

80

80

72.7

58.0 57.1

60

62.2

63.6 62.0

60

60

49.7

40

40

40

20

20

20

0

0

0

GUI-Owl-1.5-32B GUI-Owl-1.5-8Bo3 Qwen3-VL-8B-InstructGPT-5-Chat Claude-Sonnet-4

GUI-Owl-1.5-32B(w/crop)GUI-Owl-1.5-8B(w/crop)Gemini-3-ProQwen3-VL-235B-A22B-InstructGTA1-New-32B EvoCUA-32B

GUI-Owl-1.5-32BGUI-Owl-1.5-8BQwen3-VL-32B-InstructSeed1.5-VLGUI-Owl-32BUI-TARS-72B

Figure 1: Performance overview on mainstream GUI task automation, grounding and knowledge benchmarks.

∗Core Contributors †Corresponding author and project leader

1

[Figure 3]

- Figure 2: Overview of our Mobile-Agent-v3.5. We illustrate our multi-platform environment supporting and our highlight capability.

- 1 Introduction

With the rapid development of Vision–language models (VLMs) (Bai et al., 2025a; Anthropic, 2025a; OpenAI, 2025c; DeepMind, 2025),multimodal agents (Wang et al., 2024b; Qin et al., 2025; Ye et al., 2025; Liu et al., 2024; Zhou et al., 2025; Wang et al., 2025a)have achieved substantive progress, especially Graphical user interface (GUI) agents. GUI Agents are mainly designed to perform automated operations across multiple devices, such as desktops, mobiles, browsers, and so on. Recently, native agent models (Ye et al., 2025; Qin et al., 2025; Seed, 2025d) based on end-to-end learning have demonstrated great potential, rather than only building agent frameworks on top of closed-source models (Wang et al., 2024a; 2025b; Agashe et al., 2025; Zhang et al., 2025).

However,the development of robust and practically usable GUI agents still faces several challenges. (1) The efficiency of real-world data collection: Collecting large-scale trajectories is costly to hamper the scalability of GUI datasets, as it requires complex agentic workflows, manual annotation, and engineering-level handling of anomalous scenarios; (2) The adaptation to multiple platforms: The native agent model needs to perform automated tasks reliably across a wide range of devices, including desktops, mobiles, browsers, and in-vehicle systems. It should also support complex agentic real-time interactions, such as edge–cloud collaboration and coordination across multiple devices; (3) The comprehensive agentic capabilities: The General GUI Agent should be capable of completing tasks efficiently, not limited to GUI-only operations. It should also support tool/Model Context Protocol (MCP) invocation, short-term and long-term memory, multi-agent adaptation, and human–agent interaction.

To address these challenges, we propose GUI-Owl1.5, our latest native GUI agent model for multi-platform GUI automation across desktops, mobiles, browsers, and more. Built on Qwen3-VL and powered by a scalable data pipeline and a multi-stage training paradigm, GUI-Owl1.5 comprises a family of foundation GUI models covering a full range of sizes, including instruct/thinking variants at 2B, 4B, 8B, 32B, and 235B-A22B. Smaller instruct models, which do not produce thoughts, enable faster inference and can be deployed on edge devices to support high-frequency, real-time interactions while addressing security and privacy concerns. Larger thinking models, with stronger capabilities in task planning and reflection, are better suited for complex tasks and can collaborate with edge-deployed instruct models in a multi-agent setup to enable edge–cloud collaboration and multi-platform coordination. The key technical points are highlighted next.

Hybird Data Flywheel: We develop the data pipeline for UI understanding and trajectory generation by synergistically integrating simulated environments with cloud-based platform environments, thereby enhancing both the efficiency and quality of data collection. For Grounding: a comprehensive grounding data augmentation pipeline that encompasses both hard grounding data generation—including challenging app GUI synthesis and multi-window high-resolution scenarios—and scalable high-quality data extension through trajectory mining, tutorial knowledge extraction, and infeasible query generation. For trajectory, we build a self-evolving trajectory synthesis workflow based on a directed acyclic graph (DAG) (Ye et al., 2025). Meanwhile, we synthesize virtual environments via Vibe Coding to create high-frequency, complex atomic operations and apps featuring challenging cases such as pop-ups and CAPTCHA-style verifications. In addition, for some challenging apps and scenarios, we incorporate a small amount of manual annotation to better align synthetic environments with real-world ones.

Unified Enhancement of Agent Capabilities: Beyond basic GUI perception and action execution, a practical GUI agent must possess a range of higher-order skills. We introduce three complementary strategies to comprehensively enhance the native model’s agent capabilities. First, we inject GUI knowledge through large-scale QA data crawled from software documentation and forums, and train the model with world modeling supervision to anticipate interface state transitions before acting. Second, we design a unified chain-of-thought (CoT) synthesis pipeline that augments all trajectory data with step-wise observation, reflection, memory management, and tool invocation reasoning, enabling superior long-horizon planning and in-context information retention. Third, we incorporate multi-agent collaboration data collected via the Mobile-Agent-v3.5 framework, allowing the model to function not only as a standalone end-to-end agent but also as specialized roles (e.g., planner, executor, verifier) within structured multi-agent systems.

Multi-platform Environment RL Scaling: To enable stable reinforcement learning training across multi-platform environments, we propose MRPO (Multi-platform Reinforcement Policy Optimization), a large-scale RL framework that addresses four critical challenges in GUI agent training. First, we unify learning across mobile, desktop, and web environments under a single device-conditioned policy. Second, we introduce an online rollout buffer that mitigates GRPO training instability when grouped rollouts collapse to identical outcomes by oversampling trajectories and strategically selecting diverse groups while maintaining on-policy guarantees. Third, we ensure consistency between environment-side inference and training-side optimization through token-ID transport, preventing tokenization mismatches. Finally, we adopt alternating multi-platform optimization to reduce gradient interference, training on single device types cyclically rather than mixing trajectories. This approach enables stable, unified policy learning while preserving cross-device generalization for long-horizon GUI control tasks.

We evaluate GUI-Owl-1.5 on a series of benchmarks spanning GUI task automation, grounding, tool invocation, memory and knowledge. Experimental results demonstrate that GUI-Owl-1.5 exhibits strong GUI understanding, grounding and execution capabilities, achieving state-of-the-art performance among open-source models across more than 20 GUI benchmarks. Specifically, it attains task success rates of 56.5%, 71.6% and 46.6% on OSWorldVerified, AndroidWorld and VisualWebArena respectively, which outperforms models such as UI-TARS-2, Claude4, and Gemini -2.5-Pro. On OSWorld-MCP, which evaluates the integration of GUI operations and tool invocation, it achieves a task success rate of 47.6%. On the ScreenSpot-Pro grounding benchmark, it achieves a state-of-the-art accuracy of 80.3% with crop-based refinement, and notably surpasses the large-scale Gemini-3-Pro even in its base configuration (72.9%) without crop tool. On MemGUI-Bench and GUI Knowledge Bench, our model also surpasses previous open-source models.

- 2 Mobile-Agent-v3.5

GUI-Owl-1.5 is a multimodal model for GUI operations, building on the previous GUI-Owl (Ye et al., 2025). Compared to its predecessor, it offers three main improvements: (1) a broader action space; (2) improved context retention; (3)enhanced design in synthetic data generation, cross-platform adaptation, and agent capabilities.

BuildingonQwen3-VLandtrainedwithextensivepost-trainingdatasets, GUI-Owl-1.5maintainsthecorefunctions of the original model—perceiving, planning, decision-making, and locating elements in GUI scenarios—while further optimizing them for different real-world cases. The model can autonomously interact with mobile, desktop, and browser interfaces across multiple turns, and can also work collaboratively in multi-agent systems.

- 2.1 Formulation

We formulate the GUI agent task as a multi-turn interactive decision-making problem, where the agent continuously perceives the environment, executes actions, and adapts its strategy based on real-time feedback.

Input Space. At each interaction step t, the agent receives:

- • Visual observation It ∈ RH×W×3: a screenshot capturing the current GUI state.
- • User instruction Lt: a natural language command expressing the user’s intent.

[Figure 4]

- Figure 3: Illustration of the interaction flow of GUI-Owl-1.5. The system message defines the available action space, the user message contains the task instruction, compressed histories, and current observation, while the response message includes the agent’s reasoning, action summaries, and the final action output.

Output Space. Given the input, the agent generates:

- • Action conclusion Ct: a natural language explanation summarizing the planned action.
- • Tool call At: a structured function call that executes the action.

The nature of GUI agent tasks requires closed-loop interaction with the environment. Specifically, after executing At, the environment transitions to a new state, providing updated visual feedback It+1 for the next turn. This iterative process continues until the task is completed or terminated. Notably, compared to the previous GUI-Owl, we significantly expand the action space to support external tool calls and API invocations in addition to primitive GUI operations (e.g., click, type, scroll). This extension enables the agent to orchestrate complex workflows across heterogeneous systems, such as querying databases through APIs, invoking specialized computational tools, and integrating with third-party services.

Context Management. To handle long-horizon tasks while maintaining computational efficiency, we adopt a sliding window mechanism with hierarchical context compression. The context at step t is organized as:

- • Recent context (full retention): The most recent N complete dialogue turns, including all modalities: {(It−N,Lt−N,Ct−N,At−N),...,(It−1,Lt−1,Ct−1,At−1)}
- • Historical context (compressed summary): Earlier interactions beyond the N-turn window are condensed into a textual summary S1:t−N−1, formed by concatenating action conclusions: S1:t−N−1 = concat(C1,C2,...,Ct−N−1)

This hierarchical design preserves fine-grained multi-modal information for immediate decision-making while maintaining high-level awareness of long-term task progression, effectively balancing context richness with memory efficiency.

- 2.2 Data Preparation

To support training of GUI-Owl-1.5 across heterogeneous platforms and task families, we develop a unified data preparation pipeline that targets both actionable interaction supervision and fine-grained visual grounding. Specifically, we curate (i) trajectory data that captures long-horizon decision-making and tool-augmented execution in realistic GUI environments, and (ii) grounding data that aligns natural-language intents with on-screen elements.

Hard Grounding Data Synthesis

###### Challenging App GUI Grounding Data Synthesis

Multi-window High-resolution Grounding Data Synthesis

###### UI Element Candidates

###### Generated GUI Grounding Data

GUI Single-window Image + Grounding Elements

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

Render UI Element

##### … …

Fruitloops

… …

Regenerate HTML

Images & Selected Grounding Elements

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Organization Settings

Multi-window Grounding Data

[Figure 20]

|MLLM|Generate<br><br>|
|---|---|
| |HTML|

Visual Studio

[Figure 21]

Diversify

MLLM

|[Figure 22]<br><br>· Parse HTML · Overlap · Element position · Neighbor Elements<br><br>Quality Check|
|---|

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Window Count Candidates

[Figure 27]

Reference Interface

Check Selected Element Non-overlap

[Figure 28]

Layout Candidates (e.g., random, 1 row x 2 cols)

[Figure 29]

| |
|---|

[Figure 30]

Resolution Candidates

show touch keyboard

High-Quality Grounding Data Extension

###### Infeasible Grounding Data Generation

Grounding-related QAs Generation

###### Grounding Data Extraction from Trajectories

[Figure 31]

[Figure 32]

Images

[Figure 33]

[Figure 34]

|MLLM|
|---|

[Figure 35]

[Figure 36]

|Critic Model|
|---|

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

MLLM

[Figure 41]

Random Pairs

Infeasible Data

|LLM|
|---|

GUI Trajectories

Cleaned Data

Queries

App Tutorial Videos

Groundingrelated QAs

Parsed Subtitles

Grounding Data

Grounding Data

Check

Figure 4: Overview of our high-quality grounding data construction pipeline.

- 2.2.1 Grounding

Existing grounding datasets exhibit limited complexity and diversity, creating a critical shortage of high-quality, challenging grounding data alongside scalable data augmentation solutions. As illustrated in Figure 7, we address these limitations through a comprehensive data augmentation framework that enhances GUI grounding capabilities via two complementary strategies.

For Hard Grounding Data Generation, which targets complex scenarios requiring specialized domain knowledge and high annotation costs, we develop two synthesis approaches:

- • Challenging App GUI Grounding Data Synthesis: We leverage annotated UI elements and reference interfaces to generate diverse, high-quality professional application screenshots using MLLMs. This process incorporates iterative quality assessment and refinement mechanisms, where generated interfaces undergo validation checks and corrective regeneration to ensure data fidelity and domain accuracy.
- • Multi-window High-resolution Grounding Data Synthesis: Utilizing existing single-window datasets combined with candidate organization pools (encompassing window count variations, layout configurations, and resolution options), we generate complex multi-window scenarios while ensuring target UI elements remain unoccluded through spatial constraint validation.

ForHigh-QualityGroundingDataExtension, whichaimstoachievecost-effectiveandscalabledataaugmentation, we implement three synergistic enhancement pathways:

- • Trajectory-based grounding extraction: We mine grounding annotations from existing PC and mobile simulation environment trajectories, employing critic models to filter and validate data quality, ensuring only high-fidelity grounding pairs are retained.
- • Tutorial-based knowledge mining: Application tutorials are parsed to extract grounding-related questionanswer knowledge by analyzing embedded subtitles and identifying spatial-semantic relationships, ultimately generating comprehensive grounding-oriented QA pairs that capture real-world usage patterns.
- • Infeasible query generation: To address the critical gap in handling infeasible grounding queries within existing datasets, we generate large-scale negative samples through strategic random pairing of queries and interface elements, followed by multi-model consensus filtering to identify and validate truly infeasible grounding instances.

- 2.2.2 Trajectory Data Collection

We build a hybrid trajectory corpus that scales to diverse applications and devices while maintaining high supervision fidelity. The pipeline consists of (i) DAG-based task synthesis to cover frequent workflows, (ii) automated rollouts on real devices with DAG-based validation, (iii) human demonstrations for tasks that remain unsolved by automation, (iv) trajectory production via virtual environments for basic actions (e.g., scroll, drag) and highfrequency challenging scenarios.

Trajectory Collection Pipeline

[Figure 42]

Query Genera on & Trajectory Veriﬁca on & Task Rewri ng Virtual Environment

[Figure 43]

[Figure 44]

Figure 5: Overview of our trajectory collection pipeline.

Task production via human-authored DAGs. For each application domain, annotators construct a directed acyclic graph (DAG)

G = (V,E), V = {vi}|iV=1| , E ⊆ V × V, where each node vi denotes an atomic subtask and each edge (vi,vj) ∈ E denotes a feasible transition under typical UI state evolution. Let S ⊆ V and T ⊆ V be the sets of valid start and terminal nodes separately, we synthesize a task by sampling a path from S to T:

p = (v1,...,vK), v1 ∈ S, vK ∈ T, (vk,vk+1) ∈ E, which represents a realistic action sequence with multiple steps. Each node vk is associated with a sub-instruction template d(vk) that optionally has slots for diverse entities. The final task instruction is composed by concatenating and rewriting the ordered sub-instructions:

I(p) = Compose d(v1),d(v2),...,d(vK) . By sampling diverse paths and instantiating templates, the DAG provides controllable coverage of high-frequency operation patterns in common apps, minimizing the impact of the LLM hallucination.

Automated trajectory generation with checkpointing, truncation, and task repair. Given I(p), an agent interacts with a real device environment E to produce a trajectory

τ = {(ot,at)}Tt=1,

where ot is the observation (e.g., screenshot, UI structure, device metadata) and at is the executed action (touch/keyboard/tool call). To assess partial completion along the subtask path p, we define a checkpoint predicate

for each node vk:

ϕk : O → {0,1}, ϕk(ot) = 1 iff subtask vk is satisfied at ot. We compute whether subtask k is achieved somewhere in the rollout by

ϕk(ot). The longest completed prefix length is

ck(τ) = max

t∈[1,T]

m(τ) = max{m ∈ {0,...,K} : ∀k ≤ m, ck(τ) = 1}. If m(τ) = K, we accept τ as a correct trajectory. Otherwise, we truncate the rollout to the last verified checkpoint of the completed prefix:

⋆

t⋆ = max{t : ϕm(τ)(ot) = 1}, τ′ = {(ot,at)}t

t=1, and repair the original task by removing the completed subtasks:

prem = (vm(τ)+1,...,vK), Irem = I(prem).

We then store (Irem,τ′) as a partially-correct instance, which provides clean supervision for the successfully executed segment while avoiding noisy labels beyond the last correct subtask.

Human annotation on real devices. For difficult tasks that remain unsolved after repeated automated attempts, we collect expert demonstrations via a cloud annotation platform. Annotators directly operate the same real device environments and record gold trajectories τhuman aligned with the task instruction, ensuring high-quality supervision for hard cases.

Virtual environment-based trajectory production. Relying solely on agent exploration in real-world environments for trajectory generation presents two notable limitations: (i) Real-world applications and software often incorporate CAPTCHA verification, anti-bot mechanisms, and other protective measures that can interrupt or terminate the agent’s exploration process. (ii) Real-world environments cannot provide accurate feedback, which results in low efficiency of trajectory generation via agent exploration, and often yields trajectories that contain erroneous or redundant steps.

To address these challenges, we develop a suite of web-rendering-based virtual environments targeting fine-grained primitive actions (e.g., scroll, drag) and high-frequency difficult scenarios (e.g., document and spreadsheet editing, popular applications). These virtual environments serve two primary purposes: (i) providing precise sub-task-level feedback to guide agent exploration, and (ii) enabling automated and scalable trajectory generation through the integration of LLM-based instruction decomposition.

Agent rollout + critic. Given a sampled scenario ω and a DAG path p = (v1,...,vK), the agent produces a simulated trajectory τ˜ = {(˜ot,a˜t)}Tt˜=1. The simulator exposes subtask-completion predicates ϕ˜k(˜st) ∈ {0,1}, enabling an exact prefix-progress score:

ϕ˜k(˜st), m˜ (˜τ) = max{m : ∀k ≤ m, c˜k(˜τ) = 1}.

c˜k(˜τ) = max

t

We accept τ˜ if m˜ (˜τ) = K; otherwise we truncate to the last verified checkpoint and keep a clean partially-correct prefix for training:

### t˜⋆ = max{t : ϕ˜m˜(˜τ)(˜st) = 1}, τ˜′ = τ˜1:t˜⋆.

Scalable Automated Trajectory Generation. Since our virtual environments are built upon web rendering, they inherently support the automated execution of atomic operations. For a given virtual environment, such as a virtual word document editor, we leverage an LLM in conjunction with the document content to decompose a user instruction into a sequence of atomic operations that the virtual environment can execute. These atomic operations are then fed into the virtual environment to produce corresponding precise operation trajectories. Also, for many concentrated scenarios, the canonical correct operation is known and can be standardized. We encode it as a script/RPA policy ρ and directly execute:

τ˜rpa = Rollout(E˜,ρ,ω,p), m˜ (˜τrpa) = K, which yields high-quality successful trajectories at low cost.

- 2.3 Agent Capability Enhancement

Beyond grounding and GUI understanding, a capable GUI agent must plan over long horizons, reason about action consequences, memorize key information, and invoke external tools. We introduce three complementary strategies to enhance these capabilities (Figure 6): (i) GUI Knowledge Injection, which enriches the model’s knowledge through QA data and world modeling; (ii) Unified CoT Synthesis, which augments trajectory data with step-wise reasoning, reflection, and memory; and (iii) Multi-Agent Collaboration, which enables the model to operate within structured multi-agent frameworks.

- 2.3.1 GUI Knowledge Injection

QA & VQA. In addition to trajectory-format data, we further augment the agent’s GUI knowledge through data in other modalities. As illustrated in Figure 6, we crawl a substantial volume of data from diverse sources on the Internet to construct a knowledge base of GUI-related information, encompassing software feature configurations, operational instructions, website navigation, among others. The primary sources can be categorized into three types: (i) official documentation and tutorials of software applications (e.g., Microsoft Office, LibreOffice);

- (ii) software forums (e.g., WPS Academy) and Q&A platforms (e.g., Baidu Jingyan); and (iii) web navigation information extracted from existing open-source web datasets. After data cleaning, the crawled information is rewritten by LLMs into task-level QA or step-level VQA data, thereby enhancing the agent’s GUI knowledge.

World Modeling. A capable GUI agent should not only perceive the current screen state but also anticipate how the interface will change in response to its actions. To cultivate this predictive understanding, we construct world modeling data derived from trajectory recordings. Specifically, given a screenshot and the action executed at that step, we prompt a proprietary model (e.g., Claude-4.5) to produce a fine-grained description of the subsequent screenshot, explicitly highlighting the state transitions. For example, newly appeared dialogs, changed text fields, shifted focus, or updated visual elements. These action-conditioned state-transition descriptions are then used as

[Figure 45]

Figure 6: Illustration of our agent capability enhancement pipeline.

training supervision. Through this process, the model acquires an internalized understanding of GUI environment dynamics, enabling it to better reason about the consequences of candidate actions before execution, which in turn facilitates more informed decision-making in multi-step tasks.

- 2.3.2 Unified CoT Synthesis

After obtaining trajectory data containing action sequences through various approaches (i.e., agent exploration, human annotation and virtual environments), we design a chain-of-thought (CoT) synthesis pipeline to generate corresponding thoughts and conclusions for each step in the trajectory data, thereby enhancing the agent’s capabilities in screen observation, memory management, progress reflection, and tool invocation.

As illustrated in figure 6, given the i-th step of a trajectory, we first employ a vision-language model (VLM) to describe the screen content and further extract query-relevant information from it. For queries that require memorizing key on-screen information, such as Check the weather in Paris and London for next Monday and record it in the memo, we extract information from the query-relevant content that may be needed in subsequent steps and incorporate it into the memory. Furthermore, we feed the action parameters executed at the i-th step, the screenshots captured before and after execution, and the user query into the VLM to determine whether the execution outcome of this step aligns with expectations. If the change in screen state is consistent with expectations, the progress of the current task is updated accordingly in the subsequent step; otherwise, corresponding reflections and error corrections are generated to inform the next action decision.

The observation, memory, reflection and task progress information obtained above are then fed into an LLM to synthesize the thought and conclusion corresponding to the action at this step. Specifically, the thought simulates the agent’s reasoning process of integrating these pieces of information for action decision-making, while the conclusion provides a concise action decision. Moreover, if the current trajectory involves tool invocation, the tool definitions from the tool set are also provided as input to the LLM, so that the synthesized thought incorporates reasoning about tool selection and invocation.

TheCoTsynthesispipelinedesignedaboveenablesthemodelto: (i)reflectontheexecutionoutcomeoftheprevious action and analyze the overall task progress accordingly, thereby achieving superior long-horizon decision-making capability; and (ii) simultaneously record key on-screen information (e.g., prices, weather conditions) that may be required in subsequent steps during the operational process, thereby achieving enhanced memory capability.

- 2.3.3 Multi-agent Collaboration

To enable the model to serve not only as an end-to-end agent but also as the components within multi-agent frameworks for multi-agent collaboration, we additionally employ the Mobile-Agent-v3.5 framework for agent exploration during the trajectory collection phase. Mobile-Agent-v3.5 agent framework largely follows MobileAgent-v3, and we only summarize the key components and interfaces here for completeness. The system instantiates a small set of role-specialized modules and executes them in a closed loop over a device environment (mobile/desktop/web), with a unified action abstraction and shared model backbone.

Problem setup. Given a user instruction I and the current device state St (e.g., screenshot, UI tree, device metadata), the goal is to produce an action at ∈ A that drives the environment to St+1 ∼ P(· | St,at) until termination.

Roles and state variables. We maintain four agent roles: a Manager (planner), a Worker (executor), a Reflector (verifier), and a Notetaker (memory). At step t, the system state is summarized by

Xt ≜ (I,St,SSt,Ft−1,Nt), where SSt is the (ordered) subgoal list, Ft−1 is the latest feedback, and Nt is persistent notes. Manager: subgoal planning and update. The Manager decomposes the instruction into subgoals and dynamically updates them:

SS0 = fM(I,KRAG), SSt+1 = uM(SSt,Ft,St+1), where KRAG denotes optionally retrieved external knowledge. Worker: action generation. Given the current context, the Worker selects a subgoal and produces the next action (optionally as a structured tuple with rationale and a normalized action schema):

### at ∼ πW(· | I,St,SSt,Ft−1,Nt).

Reflector: transition-level verification and feedback. After executing at on the device, the Reflector judges the transition and provides diagnostic feedback:

### (jt,ϕt) = fR(St,at,St+1), jt ∈ {SUCCESS,FAILURE},

and we set Ft ≜ (jt,ϕt). Notetaker: persistent memory update. Upon successful progress, the Notetaker extracts and stores salient transient information for future steps:

Nt+1 =

uC(Nt,St+1) if jt = SUCCESS, Nt otherwise.

Execution loop. The framework iterates (SSt,at,Ft,Nt) updates until all subgoals are completed or a termination condition is met (e.g., success, timeout, or safety stop). This design isolates planning, execution, verification, and memory, while remaining compatible with the multi-platform interfaces used throughout training and evaluation.

- 2.4 Training Paradigm

GUI-Owl-1.5 is initialized from Qwen3-VL and trained through a three-stage process. Compared with GUI-Owl, each stage is substantially expanded in data diversity and task coverage to support multi-platform automation, tool invocation, and complex agentic interactions.

- 2.4.1 Pre-training

We construct a large-scale pre-training corpus that extends beyond basic GUI understanding. In addition to the UI recognition and trajectory data used in GUI-Owl, we incorporate (i) QA and VQA knowledge data to strengthen general visual reasoning and knowledge comprehension, (ii) world-modeling data to train the model to predict how GUI states transition in response to actions, and (iii) tool invocation data to familiarize the model with tool-calling and MCP semantics from the earliest stage.

- 2.4.2 Supervised Fine-tuning

We perform supervised fine-tuning (SFT) to align GUI-Owl-1.5 with diverse agentic tasks across multiple devices. The SFT data covers multi-device trajectory data with CoT annotations (Section 2.2.1), augmented grounding data (Section 2.2.2), structured tool invocation supervision for both conventional tool calls and MCP-based interactions, and dedicated browser interaction data capturing the unique characteristics of web-based GUIs. This stage transforms the pre-trained model into a capable multi-device agent supporting GUI manipulation, tool invocation, and browser automation with explicit reasoning.

###### Rollout-Trainer Aligner Interleaved Trainer

###### Multi-Platform Rollouter

[Figure 46]

Update

[Figure 47]

GUI-Owl-1.5 Data Center

Ours Common Settings

Mobile

[Figure 48]

[Figure 49]

Rollout Decoding

Mobile sandbox

PC sandbox Web sandbox

Rollout Text

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Don‘tAlign

Trainer Encoding

PC

Transfer

Trainer Tokens

Rollout Tokens

Rollout Tokens

###### Expanded Rollout Buffer

[Figure 54]

[Figure 55]

[Figure 56]

( Rollout nx Expand N)

Cyclic Enhancement

[Figure 57]

… …

Align

[Figure 58]

[Figure 59]

Randomly Selectn

Replace One

Web

|Increase diversity Reduce dropping Preserve distribution<br><br>Fine Rollout Groups<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
|---|

Rollout n

[Figure 63]

Positive-Negative Mixed

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Train

Data for Training

Learned: Mobile + PC + Web

Individual Collapsed Group

Figure 7: Overview of our reinforcement learning pipeline.

- 2.4.3 Reinforcement Learning

We perform a large-scale reinforcement learning called MRPO (Multi-platform Reinforcement Policy Optimization) to further align GUI-Owl-1.5 with long-horizon, tool-augmented GUI control across heterogeneous devices. The key challenges are: (i) unifying learning across mobile/desktop/web environments under one policy, (ii) stabilizing GRPO training when grouped rollouts collapse to identical outcomes. We address these issues as follows,

- (iii) ensuring log-probability consistency between environment-side inference and training-side optimization, and
- (iv) mitigating cross-device optimization interference.

Multi-device RL with a unified policy. We optimize a single policy πθ(a | o) over trajectories collected from multiple device families d ∈ D = {mobile,desktop,web}. Each device defines its own environment Ed, action space Ad, and observation stream. We model device heterogeneity via a device-conditioned policy:

πθ(a | o,d), a ∈ Ad.

Online rollout buffer for GRPO under outcome collapse. We use GRPO-style grouped rollouts. For a task x, we sample a group of n trajectories {τi}ni=1. In practice, it is common that all n rollouts yield identical terminal outcome (e.g., all success or all failure), making the group uninformative and often discarded. A replay buffer could increase diversity but introduces off-policy bias. We therefore propose an online rollout buffer that increases within-group diversity while remaining on-policy.

For each x, we temporarily oversample kn rollouts on-policy:

Gkn(x) = {τi}kni=1, τi ∼ πθ(· | x),

then uniformly subsample n trajectories to form the training group Gn(x). Let Z(τ) ∈ {0,1} denote a binary outcome (success/failure). The crucial property is that uniform subsampling preserves the marginal distribution of any statistic under on-policy sampling:

  1

  = Eτ∼π

E

f(τ)

θ(·|x)[f(τ)],

n

τ∈Gn(x)

because Gkn(x) is i.i.d. on-policy and Gn(x) is an exchangeable uniform subset. Thus oversample-then-subsample yields an approximately unbiased estimator.

Let Z(τ) ∈ {0,1} be the terminal outcome (e.g., success). For a task x, GRPO forms a group Gn(x) = {τi}ni=1 with τi ∼ πθ(· | x). The update becomes uninformative when the group is collapsed:

Collapse(Gn) ≜

Z(τ) ∈ {0,n} .

τ∈Gn

To reduce collapsed groups without introducing off-policy replay, we use an online oversample-and-select buffer. First, sample an on-policy pool of size kn:

Gkn(x) = {τi}kni=1, τi ∼ πθ(· | x). Define the pool-diversity event

A ≜ 0 <

Z(τ) < kn ,

τ∈Gkn

with probability

P(A) = 1 − pkn − (1 − p)kn, p ≜ Pτ∼π

θ(·|x)[Z(τ) = 1].

Then construct the training group Gn(x) by:

 

Subsamplen(Gkn(x)), ¬Collapse(Subsamplen(Gkn(x))), Swap1 Subsamplen(Gkn(x)), Gkn(x) , Collapse(Subsamplen(Gkn(x))) ∧ A, ∅, ¬A,

Gn(x) =



where Subsamplen(·) is uniform random downsampling, and Swap1(S,P) replaces one random element in S with a random opposite-outcome element from pool P (guaranteeing 0 < τ∈ G

Z(τ) < n). This keeps all

n

candidates strictly on-policy (sampled from current πθ) while sharply increasing the probability of obtaining a non-collapsed GRPO group.

Training–inference log-prob alignment via token-id transport. Our inference service is deployed on the environment side and returns trajectories as text (e.g., tool calls, typed strings, serialized actions). However, if the training-side tokenizer maps the returned text to token IDs differently from the inference-side tokenizer (due to non-unique segmentation), then the computed log-probabilities can be inconsistent:

### log πθ(y | x)

train-tokenize(y)

### ̸= log πθ(y | x)

### .

infer-tokenize(y)

This breaks KL regularization and policy-gradient estimators that assume the same sampled action representation. Our fix is to transport the original inference token IDs alongside the textual payload. Concretely, for each generated sequence y, the environment returns (y, tinfer) where tinfer = (t1,...,tL) are the exact token IDs used to sample y. The training process then computes:

log πθ(y | x) :=

L

log πθ(ti | x,t<i),

i=1

thereby guaranteeing that the log-prob is evaluated on the same discrete event that was executed in the environment.

Alternating multi-device optimization to reduce gradient interference. Mixing trajectories from different devices in a single batch can induce strong gradient conflicts because Ad, UI conventions, and domain priors differ substantially. Let gd = Eτ∼π

θ,Ed[∇θL(τ)] be the device-specific policy gradient. A naive mixture update uses g = d λdgd, but when ⟨gd

2⟩ < 0 frequently, optimization becomes a “tug-of-war”. We adopt an alternating schedule across stages:

,gd

1

### θ(s+1) ← θ(s) − η gd

### , ds ∈ D,

s

where each stage s trains on a single device family (potentially with multiple environments within the family), and device families are visited cyclically or via a curriculum. This isolates device-specific adaptation while keeping a shared backbone, empirically improving stability and preserving cross-device generalization.

Agent Model OSWorld-Verified AndroidWorld OSWorld-MCP Mobile-World WindowsAA General Models

SeedVL-1.5 (Seed, 2025b) 34.1 62.1 38.4 - 39.6 Claude-4-sonnet (Anthropic, 2025d) 43.9 - 43.3 - Claude-4-5-sonnet (Anthropic, 2025e) 62.9 56.0 - Gemini-2.5-pro (Deepmind, 2025) - 69.7 27.2 - Kimi K2.5 (Kimi et al., 2025) 63.3 - - Seed-1.8 (Seed, 2025a) 61.9 70.7 - OpenAI CUA o3 (Wang et al., 2025a) 31.3 - - Qwen3-VL-8B-Instruct (Yang et al., 2025a) 33.9 47.6 - 5.5 28.8 Qwen3-VL-8B-Think (Yang et al., 2025a) 33.9 50.0 - - 24.1 Qwen3-VL-32B-Instruct (Yang et al., 2025a) 32.6 57.3 - 9.0 30.9 Qwen3-VL-32B-Think (Yang et al., 2025a) 41.0 63.7 - - 42.9 Qwen3-VL-235B-A22B-Instruct (Yang et al., 2025a) 31.6 63.7 - 9.5 28.9 Qwen3-VL-235B-A22B-Think (Yang et al., 2025a) 38.1 62.0 39.1 - 32.1

GUI Models (Single-Platform)

OpenCUA-7B (Wang et al., 2025a) 28.2 - - - OpenCUA-32B (Wang et al., 2025a) 34.8 - - OpenCUA-72B (Wang et al., 2025a) 45.0 - - EvoCUA (Xue et al., 2026) 56.7 - - MAI-UI-8b (Zhou et al., 2025) - 70.7 - 24.9 MAI-UI-32b (Zhou et al., 2025) - 73.3 - 37.3 MAI-UI-235b-A22b (Zhou et al., 2025) - 76.7 - 41.7 -

GUI Models (Multi-Platform)

UI-TARS-72B-DPO (Seed, 2025e) 27.1 46.6 - UI-TARS-1.5-7B (Seed, 2025c) 27.4 - - 15.9 UI-TARS-1.5 (Seed, 2025c) - 64.2 - 20.9 42.1 UI-TARS-2 (Seed, 2025d) 53.1 73.3 - 50.6 GELab-Zero-4B (Yan et al., 2025) 31.9 63.9 - 10.9 GELab-Zero-8B (Yan et al., 2025) 40.2 67.7 - GUI-Owl-7b (Ye et al., 2025) 34.9 66.4 - 4.5 GUI-Owl-32b (Ye et al., 2025) - - - 5.5 -

Ours (Multi-Platform)

GUI-Owl-1.5-2B-Instruct 43.5 67.9 33.0 31.3 25.78 GUI-Owl-1.5-4B-Instruct 48.2 69.8 31.7 32.3 29.44 GUI-Owl-1.5-8B-Instruct 52.3 69.0 41.8 41.8 31.66 GUI-Owl-1.5-8B-Thinking 52.9 71.6 38.8 33.3 35.07 GUI-Owl-1.5-32B-Instruct 56.5 69.8 47.6 46.8 44.76 GUI-Owl-1.5-32B-Thinking 56.0 69.8 43.8 42.8 44.13

Table 1: Comparison with state-of-the-art methods on online computer use and mobile use benchmarks.

Agent Model WebArena VisualWebArena WebVoyager Online-Mind2Web

Proprietary Models

Browser-Use (Use, 2025) - - 89.1 30.0 Claude-CUA-3.7 (Anthropic, 2025c) - - - 56.3 Operator (OpenAI, 2025a) - - 87.0 61.3 Gemini-CUA (DeepMind, 2025) - - - 69.0 Navigator (Yutori, 2025) - - - 78.7 Magnitude + Claude-4-Sonnet (Magnitude, 2025) - - 93.9 VisualWebArena + GPT-4o (Koh et al., 2024a) - 19.8 - Tree Search + GPT-4o (Koh et al., 2024b) 19.2 26.4 - WALT + GPT-5 (Prabhu et al., 2025) 50.1 52.9 - SGV + Gemini-2.5-Flash (Andrade et al., 2025) - 54.4 - DeepSky Agent + Claude-4-Sonnet (Tibrewal, 2025) 66.9 - - OAgent + Gemini-3-Pro (CodeFuse, 2025) 71.6 - - -

Open-Source Models

WebStar-7B (He et al., 2026) - - 44.8 22.8 WebStar-32B (He et al., 2026) - - 48.6 23.8 DynaWeb-8B (Ding et al., 2026) 31.0 - 38.7 ViGoRL-7B (Sarch et al., 2025) - 11.2 - Llama-3-70B-Instruct + Tree Search (Koh et al., 2025) 10.1 16.7 - AgentSymbiotic-8B (Koh et al., 2025) 43.2 - - -

Ours

GUI-Owl-1.5-8B-Instruct 45.7 39.4 69.9 41.7 GUI-Owl-1.5-8B-Thinking 46.7 40.8 78.1 48.6 GUI-Owl-1.5-32B-Instruct - - - GUI-Owl-1.5-32B-Thinking 48.4 46.6 82.1 -

Table 2: Comparison with state-of-the-art methods on online browser use benchmarks.

- 3 Experiments

- 3.1 Experimental Setup

In this section, we evaluate GUI-Owl-1.5 across a wide range of benchmarks to thoroughly assess its performance as a native GUI agent for multi-device automation. Built on Qwen3-VL, GUI-Owl-1.5 comprises a family of models including instruct and thinking variants. In this report, we focus on 6 representative versions: GUI-Owl-1.52B-Instruct, GUI-Owl-1.5-4B-Instruct, GUI-Owl-1.5-8B-Instruct, GUI-Owl-1.5-8B-Think, GUI-Owl-1.5-32BInstruct, and GUI-Owl-1.5-32B-Think. We conduct extensive experiments to evaluate GUI-Owl-1.5 along four key dimensions consistent with GUI-Owl: grounding capability, comprehensive GUI understanding, end-to-end agent capability, and multi-agent capability.

- 3.2 Main Results

- 3.2.1 End2end and Multi-Agent capability on Online environment

Thebenchmarksdiscussedaboveevaluateisolated, single-stepactions, offeringonlyapartialviewofanagent’strue capability. In practice, GUI automation requires chaining numerous decisions where earlier mistakes propagate and compound, and multiple valid execution paths may exist for the same task, yet offline benchmarks typically score against a single reference trajectory. To overcome these limitations, we conduct end-to-end evaluations across live interactive environments spanning three domains in Fig. 1 and Fig. 9: Mobile Use (AndroidWorld (Rawles et al.,

- 2024), MobileWorld (Kong et al., 2025), and MMGUI-Bench (Liu et al., 2026)), Computer Use (OSWorld (Xie et al., 2024), WindowsAgentArena (Bonatti et al., 2024), and OSWorld-MCP (Jia et al., 2025)), and Browser Use (WebArena (Zhou et al., 2023), VisualWebArena (Koh et al., 2024a), WebVoyager (He et al., 2024), and OnlineMind2Web (?)). Among these, MMGUI-Bench specifically evaluates the agent’s memory ability. MobileWorld and OSWorld-MCP further incorporate tool invocation, assessing the agent’s ability to coordinate GUI operations with external tool and MCP calls. In all environments, success is determined solely by whether the final goal state is achieved, regardless of the specific path taken.

Computer and Mobile Use. As shown in Table 1, GUI-Owl-1.5 achieves state-of-the-art performance among multi-platform GUI models across both computer and mobile use benchmarks. On OSWorld-Verified, the most widely adopted computer use benchmark, GUI-Owl-1.5-8B-Thinking achieves 52.9, surpassing UI-TARS-2 (53.1) atcomparablescaleandoutperformingallgeneral-purposemodelsincludingQwen3-VL-235B-A22B-Think(38.1). Even our 2B variant attains 43.5, exceeding models with over 10× more parameters such as UI-TARS-72B-DPO (27.1), showcasing strong parameter efficiency. Similarly, on WindowsAA, the 32B-Instruct model scores 44.76, outperforming all general-purpose models at comparable or larger scale. For mobile use, on AndroidWorld, our 8B-Thinking variant attains 71.6, on par with UI-TARS-2 (73.3). Beyond standard GUI interaction, MobileWorld and OSWorld-MCP further require coordinating GUI actions with external tool and MCP calls; on these two benchmarks, GUI-Owl-1.5-32B-Instruct scores 46.8 and 47.6 respectively, surpassing both single-platform specialists (e.g., MAI-UI-235B-A22B at 41.7) and leading proprietary models (e.g., Claude-4-Sonnet at 43.3 on OSWorld-MCP), demonstrating strong tool-use capability.

Browser Use. As shown in Table 2, GUI-Owl-1.5-8B-Thinking achieves 46.7 on WebArena, 40.8 on VisualWebArena, 78.1 on WebVoyager, and 48.6 on Online-Mind2Web, surpassing all open-source models by a wide margin and remaining competitive with proprietary systems. These results establish GUI-Owl-1.5 as one of the strongest open-source browser agents to date. Across almost all domains, the Thinking variants consistently outperform their Instruct counterparts, with pronounced gains on tasks requiring long-horizon planning (e.g., WebVoyager: 69.9 to 82.1, Online-Mind2Web: 41.7 to 48.6), validating the effectiveness of our thinking-mode training.

- 3.2.2 Grounding Capability

The grounding capability evaluates a model’s ability to locate the corresponding UI element given a naturallanguage query. We use ScreenSpot Pro, OSWorld-G, OSWorld-G-Refine, ScreenSpot V2 and MMBench-GUI L2 as benchmarks. ScreenSpot V2 covers mobile, desktop, and web scenarios, while ScreenSpot-Pro primarily evaluates a model’s localization ability at ultra-high resolutions. OSWorld-G/OSWorld-G-Refine contains finely annotated queries. MMBench-GUI L2 has the broadest coverage and more faithfully reflects a model’s grounding performance in real-world settings. The performance comparisons are shown in Tables 3 to 7.

In all grounding benchmarks, GUI-Owl-1.5-32B-Instruct achieves state-of-the-art performance among all Multiplatform GUI Models. Notably, on the Screenspot-Pro benchmark, which emphasizes high-resolution and challenging professional software grounding tasks, our GUI-Owl-1.5-32B-Instruct achieves an accuracy of 72.9, surpassing all existing GUI agents (including single-platform, multi-platform, and grounding-specialized models) as well as the large-scale Gemini-3-Pro. Moreover, when augmented with a two-stage refinement strategy with crop tool—first localizing a coarse region, then cropping and zoomin for refined grounding, GUI-Owl-1.5-32B-Instruct attains a substantially higher score of 80.3, outperforming all prior methods by a significant margin.

Windows MacOS Linux iOS Android Web

Model

Overall Basic Adv. Basic Adv. Basic Adv. Basic Adv. Basic Adv. Basic Adv.

General Models

GPT-4o (Hurst et al., 2024) 1.48 1.10 8.69 4.34 1.05 1.02 5.10 3.33 2.53 1.41 3.23 2.92 2.87 Claude-3.7 (Anthropic, 2025c) 1.48 0.74 12.46 7.51 1.05 0.00 13.69 10.61 1.40 1.40 3.23 2.27 4.66 Qwen-2.5-Max-VL (Bai et al., 2025b) 43.91 36.76 58.84 56.07 53.93 30.10 77.39 59.09 79.49 70.14 74.84 58.77 58.03 InternVL3-72B (Zhu et al., 2025) 70.11 42.64 75.65 52.31 59.16 41.33 93.63 80.61 92.70 78.59 90.65 65.91 72.20

GUI Models (Single-Platform / Grounding Specialized)

UGround-V1-7B (Gou et al., 2024) 66.79 38.97 71.30 48.55 56.54 31.12 92.68 70.91 93.54 70.99 88.71 64.61 65.68 MAI-UI-8B (Zhou et al., 2025) 92.3 74.3 90.7 86.4 81.2 67.3 97.1 90.0 97.5 92.7 95.8 86.0 88.8 MAI-UI-32B (Zhou et al., 2025) 93.0 78.7 92.8 87.6 86.9 77.6 97.1 92.4 98.0 93.2 96.1 92.5 91.3

GUI Models (Multi-Platform)

Aguvis-7B-720P (Xu et al., 2024) 37.27 21.69 48.12 33.27 33.51 25.00 67.52 65.15 60.96 50.99 61.61 45.45 45.66 OS-Atlas-Base-7B (Wu et al., 2024) 36.90 18.75 44.35 21.68 31.41 13.27 74.84 48.79 69.60 46.76 61.29 35.39 41.42 GUI-Owl-8B (Ye et al., 2025) 86.35 61.76 81.74 64.45 74.35 61.73 94.90 83.03 95.78 83.66 93.22 72.72 80.49 UI-TARS-1.5-7B (Qin et al., 2025) 68.27 38.97 68.99 44.51 64.40 37.76 88.54 69.39 90.45 69.29 80.97 56.49 64.32 UI-TARS-72B-DPO (Qin et al., 2025) 78.60 51.84 80.29 62.72 68.59 51.53 90.76 81.21 92.98 80.00 88.06 68.51 74.25 GUI-Owl-32B (Ye et al., 2025) 85.61 65.07 84.93 67.05 76.96 63.27 95.22 85.45 96.07 87.04 95.48 80.84 82.97

Ours

GUI-Owl-1.5-2B-Instruct 82.28 47.79 83.18 56.06 70.15 44.38 88.53 69.39 81.69 69.10 90.32 70.12 72.17 GUI-Owl-1.5-4B-Instruct 87.82 69.11 88.40 67.63 74.86 56.12 97.13 83.33 96.05 85.11 95.48 82.46 83.24 GUI-Owl-1.5-8B-Instruct 89.66 65.44 88.11 72.83 72.77 56.63 95.85 83.93 95.21 82.86 93.22 77.59 82.52

- GUI-Owl-1.5-8B-Thinking 84.50 63.60 85.22 71.10 71.73 53.57 92.99 81.82 89.30 78.37 95.81 77.60 80.08 GUI-Owl-1.5-32B-Instruct 91.51 68.75 92.46 77.46 76.44 67.35 97.13 90.61 96.06 89.04 96.13 84.74 86.84 GUI-Owl-1.5-32B-Thinking 89.67 66.18 88.12 73.41 77.49 63.78 94.59 89.09 92.96 87.64 96.13 81.49 84.47

Table 3: Comparison with state-of-the-art methods on the MMBench-GUI-L2 dataset.

Agent Model

Development Creative CAD Scientific Office OS

Avg Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon

General Models Claude 3.7 Sonnet (Anthropic, 2025c) - - - - - - - - - - - - 27.7

- Operator (OpenAI, 2025a) 50.0 19.3 51.5 23.1 16.8 14.1 58.3 24.5 60.5 28.3 34.6 30.3 36.6 Gemini-3-Pro (DeepMind, 2025) - - - - - - - - - - - - 72.7 Seed1.8 (Seed, 2025a) - - - - - - - - - - - - 73.1◦ Qwen3-VL-8B-Instruct (Bai et al., 2025a) - - - - - - - - - - - - 54.6 Qwen3-VL-8B-Thinking (Bai et al., 2025a) - - - - - - - - - - - - 46.6 Qwen3-VL-32B-Instruct (Bai et al., 2025a) - - - - - - - - - - - - 57.9 Qwen3-VL-32B-Thinking (Bai et al., 2025a) - - - - - - - - - - - - 57.1 Qwen3-VL-235B-A22B-Instruct (Bai et al., 2025a) - - - - - - - - - - - - 62.0 Qwen3-VL-235B-A22B-Thinking (Bai et al., 2025a) - - - - - - - - - - - - 61.8 GUI Models (Single-Platform / Grounding Specialized)

InfiGUI-R1-3B (Liu et al., 2025) 51.3 12.4 44.9 7.0 33.0 14.1 58.3 20.0 65.5 28.3 43.9 12.4 35.7 JEDI-7B (Xie et al., 2025) 42.9 11.0 50.0 11.9 38.0 14.1 72.9 25.5 75.1 47.2 33.6 16.9 39.5 GUI-G2-7B (Tang et al., 2025) 68.8 17.2 57.1 15.4 55.8 12.5 77.1 24.5 74.0 32.7 57.9 21.3 47.5 OpenCUA-7B (Wang et al., 2025a) - - - - - - - - - - - - 50.0 GTA1-New-7B (Yang et al., 2025b) - - - - - - - - - - - - 55.5 MAI-UI-8B (Zhou et al., 2025) 83.8 52.4 76.3 33.6 72.6 35.9 79.9 37.3 88.7 60.4 76.6 49.4 65.8 MAI-UI-8B + Zoom-in (Zhou et al., 2025) 78.6 58.6 78.8 46.9 80.7 43.8 86.1 49.1 88.1 81.1 76.6 51.7 70.9◦ EvoCUA-8B (Xue et al., 2026) - - - - - - - - - - - - 45.4 UI-TARS-72B (Qin et al., 2025) 63.0 17.3 57.1 15.4 18.8 12.5 64.6 20.9 63.3 26.4 42.1 15.7 38.1 UI-Venus-72B (Gu et al., 2025) 84.4 33.1 73.2 30.8 66.5 29.7 84.7 42.7 83.1 60.4 75.7 36.0 61.9 UGround-v1-72B (Gou et al., 2025) 55.8 4.8 54.0 10.5 16.8 4.7 70.8 22.7 61.0 18.9 40.2 7.9 34.5 GTA1-32B (Yang et al., 2025b) 82.5 28.3 69.2 14.7 43.7 23.4 79.9 31.8 80.8 43.4 70.1 32.6 53.6 GTA1-New-32B (Yang et al., 2025b) - - - - - - - - - - - - 63.6 GTA1-72B (Yang et al., 2025b) 79.9 33.1 73.2 20.3 56.9 28.1 81.9 38.2 85.3 49.1 73.8 37.1 58.4 MAI-UI-32B (Zhou et al., 2025) 86.4 40.7 82.8 37.8 70.1 45.3 91.7 46.4 90.4 71.7 78.5 34.8 67.9 MAI-UI-32B + Zoom-in (Zhou et al., 2025) 84.4 57.9 87.9 46.2 79.2 53.1 91.7 54.5 88.1 79.2 80.4 47.2 73.5◦ EvoCUA-32B (Xue et al., 2026) - - - - - - - - - - - - 49.7

GUI Models (Multi-Platform)

GUI-Owl-7B (Ye et al., 2025) 76.6 31.0 59.6 27.3 64.5 21.9 79.1 37.3 77.4 39.6 59.8 33.7 54.9 GUI-Owl-32B (Ye et al., 2025) 84.4 39.3 65.2 18.2 62.4 28.1 82.6 39.1 81.4 39.6 70.1 36.0 58.0 UI-TARS-1.5 (Seed, 2025c) - - - - - - - - - - - - 61.6 Step-GUI-8B (Yan et al., 2025) - - - - - - - - - - - - 62.6

Ours

GUI-Owl-1.5-2B-Instruct 74.0 43.4 66.1 32.8 53.2 26.5 78.4 39.0 79.6 45.2 66.3 50.5 57.8 + Zoom-In 77.2 62.0 71.7 56.6 75.1 51.5 84.0 49.0 88.7 77.3 73.8 53.9 70.4◦ GUI-Owl-1.5-4B-Instruct 83.1 57.9 73.7 41.9 59.8 45.3 87.5 43.6 87.0 60.3 80.3 50.5 66.8 + Zoom-In 88.9 58.6 79.7 57.3 83.7 56.2 90.2 53.6 90.3 77.3 81.3 62.9 75.6◦ GUI-Owl-1.5-8B-Instruct 87.0 63.4 79.2 45.4 76.1 43.7 87.5 47.2 89.2 56.6 82.2 49.4 71.1 + Zoom-In 90.2 68.9 84.8 56.6 86.8 62.5 89.5 55.4 91.5 71.6 86.9 53.9 77.8◦

- GUI-Owl-1.5-8B-Thinking 85.7 37.9 68.2 28.7 56.9 28.1 75.7 30.9 83.6 35.8 69.2 38.2 57.6

+ Zoom-In 88.3 58.6 78.8 45.5 84.3 48.4 86.8 45.5 91.0 64.2 84.1 52.8 72.5◦ GUI-Owl-1.5-32B-Instruct 88.3 64.1 78.8 44.1 80.2 48.4 90.3 54.5 91.5 56.6 86.9 44.9 72.9 + Zoom-In 92.2 73.1 82.8 67.1 89.3 65.6 91.7 57.3 93.2 75.5 86.0 57.3 80.3◦ GUI-Owl-1.5-32B-Thinking 83.1 37.2 71.2 23.8 46.7 20.3 81.3 34.5 87.0 47.2 72.0 31.5 57.0 + Zoom-In 88.3 53.1 81.3 45.5 83.8 42.2 87.5 46.4 92.7 66.0 83.2 55.1 72.4◦

Table 4: Comparison with state-of-the-art methods on the ScreenSpot-Pro dataset. Values marked with ◦ were processed with the crop tool.

Text Matching

Element Recognition

Layout Understanding

Fine-grained Manipulation

Refusal Avg General Models

Agent Model

- Operator (OpenAI, 2025a) 51.3 42.4 46.6 31.5 0.0 40.6 Seed1.5-VL (Seed, 2025b) 73.9 66.7 69.6 47.0 18.5 62.9 Qwen3-VL-8B-Instruct (Bai et al., 2025a) 69.0 55.5 59.7 47.7 - 54.8 Qwen3-VL-8B-Thinking (Bai et al., 2025a) - - - - - 56.7 Qwen3-VL-32B-Instruct (Bai et al., 2025a) - - - - - 65.1 Qwen3-VL-32B-Thinking (Bai et al., 2025a) - - - - - 64.0

GUI Models (Single-Platform / Grounding Specialized)

UGround-7B (Gou et al., 2024) 51.3 40.3 43.5 24.8 0.0 36.4 Aguvis-7B (Xu et al., 2024) 55.9 41.2 43.9 28.2 0.0 38.7 JEDI-7B (Xie et al., 2025) 65.9 55.5 57.7 46.9 7.4 54.1 GTA1-7B (Yang et al., 2025b) 42.1 65.7 62.7 56.1 0.0 55.1 UI-Venus-7B (Gu et al., 2025) 74.6 60.5 61.5 45.5 - 58.8 MAI-UI-8B (Zhou et al., 2025) 72.0 63.3 66.0 51.0 - 60.1 OpenCUA-32B (Wang et al., 2025a) - - - - - 59.6 GTA1-32B (Yang et al., 2025b) 63.2 78.4 73.3 65.2 0.0 65.2 MAI-UI-32B (Zhou et al., 2025) 73.6 72.4 73.9 57.7 - 67.6 EvoCUA-32B (Xue et al., 2026) - - - - - 63.9

GUI Models (Multi-Platform)

OS-Atlas-7B (Wu et al., 2024) 44.1 29.4 35.2 16.8 7.4 27.7 UI-TARS-72B (Qin et al., 2025) 69.4 60.6 62.9 45.6 0.0 57.1 UI-TARS-7B (Qin et al., 2025) 60.2 51.8 54.9 35.6 0.0 47.5 UI-TARS-1.5-7B (Seed, 2025c) 36.8 62.7 62.2 50.8 0.0 52.8 GUI-Owl-7B (Ye et al., 2025) 64.8 63.6 61.3 41.0 - 55.9 GUI-Owl-32B (Ye et al., 2025) 67.0 64.5 67.2 45.6 - 58.0

Ours

GUI-Owl-1.5-2B-Instruct 49.3 52.7 48.7 52.4 53.7 52.8 GUI-Owl-1.5-4B-Instruct 66.1 66.1 64.8 64.8 35.2 63.7 GUI-Owl-1.5-8B-Instruct 67.8 68.5 68.5 65.5 42.6 65.8 GUI-Owl-1.5-8B-Thinking 56.9 60.0 60.7 53.8 22.2 55.0 GUI-Owl-1.5-32B-Instruct 68.2 66.8 65.5 62.1 70.4 66.8 GUI-Owl-1.5-32B-Thinking 63.6 63.9 67.0 55.2 5.6 57.6

Table 5: Comparison with state-of-the-art methods on the OSWorld-G dataset.

Text Matching

Element Recognition

Layout Understanding

Fine-grained Manipulation

Agent Model

Refusal Avg General Models

Operator (OpenAI, 2025a) - - - - - 57.8 Qwen3-VL-8B-Instruct (Bai et al., 2025a) 73.9 68.2 73.1 54.4 - 64.4 Qwen3-VL-32B-Instruct (Bai et al., 2025a) 77.4 73.6 76.3 57.7 - 69.0

GUI Models (Single-Platform / Grounding Specialized)

JEDI-7B (Xie et al., 2025) - - - - - 63.8 GTA1-7B (Yang et al., 2025b) 63.2 82.1 74.2 70.5 0.0 67.7 OpenCUA-32B (Wang et al., 2025a) 63.2 79.9 84.9 62.1 7.4 70.2 GTA1-32B (Yang et al., 2025b) 63.2 83.6 84.4 70.5 0.0 72.2

GUI Models (Multi-Platform) UI-TARS-1.5-7B (Seed, 2025b) 52.6 75.4 72.4 66.7 0.0 64.2 Ours

GUI-Owl-1.5-2B-Instruct 65.1 60.3 63.3 65.9 53.7 62.6 GUI-Owl-1.5-4B-Instruct 73.1 71.8 72.9 74.1 29.6 68.4 GUI-Owl-1.5-8B-Instruct 73.1 69.4 71.0 74.8 42.5 69.3 GUI-Owl-1.5-8B-Thinking 64.7 61.1 64.9 65.3 20.4 59.8 GUI-Owl-1.5-32B-Instruct 68.5 68.5 71.0 70.1 68.5 69.7 GUI-Owl-1.5-32B-Thinking 68.9 69.7 72.6 72.1 5.6 64.7

Table 6: Performance comparison of state-of-the-art models on the OSWorld-G-Refine.

Mobile Desktop Web

Agent Model

Overall Text Icon Text Icon Text Icon

General Models

OmniParser-v2 (Yu et al., 2025) 95.5 74.6 92.3 60.9 88.0 59.6 80.7 Operator (OpenAI, 2025a) 47.3 41.5 90.2 80.3 92.8 84.3 70.5 Claude 3.7 Sonnet (Anthropic, 2025c) - - - - - - 87.6 UI-TARS-1.5 (Qin et al., 2025) - - - - - - 94.2 Seed-1.5-VL (Seed, 2025b) - - - - - - 95.2 Qwen3-VL-8B-Instruct (Bai et al., 2025a) - - - - - - 94.4 Qwen3-VL-8B-Thinking (Bai et al., 2025a) - - - - - - 93.5 Qwen3-VL-32B-Instruct (Bai et al., 2025a) - - - - - - 95.8 Qwen3-VL-32B-Thinking (Bai et al., 2025a) - - - - - - 95.7

GUI Models (Single-Platform / Grounding Specialized)

JEDI-3B (Xie et al., 2025) 96.6 81.5 96.9 78.6 88.5 83.7 88.6 JEDI-7B (Xie et al., 2025) 96.9 87.2 95.9 87.9 94.4 84.2 91.7 GTA1-7B (Yang et al., 2025b) 99.0 88.6 94.9 89.3 92.3 86.7 92.4 GTA1-32B (Yang et al., 2025b) 98.6 89.1 96.4 86.4 95.7 88.7 93.2 EvoCUA-32B (Xue et al., 2026) - - - - - - 90.4 UI-Venus-72B (Gu et al., 2025) 99.7 93.8 95.9 90.0 96.2 92.6 95.3

GUI Models (Multi-Platform)

OS-Atlas-Base-4B (Wu et al., 2024) 95.2 75.8 90.7 63.6 90.6 77.3 85.1 OS-Atlas-Base-7B (Wu et al., 2024) 96.2 83.4 89.7 69.3 94.0 79.8 87.1 UI-TARS-7B (Qin et al., 2025) 96.9 89.1 95.4 85.0 93.6 85.2 91.6 UI-TARS-72B (Qin et al., 2025) 94.8 86.3 91.2 87.9 91.5 87.7 90.3 GUI-Owl-7B (Ye et al., 2025) 99.0 92.4 96.9 85.0 93.6 85.2 92.8 GUI-Owl-32B (Ye et al., 2025) 98.6 90.0 97.9 87.8 94.4 86.7 93.2

Ours

GUI-Owl-1.5-2B-Instruct 92.9 83.1 94.8 86.4 90.1 87.6 89.7 GUI-Owl-1.5-4B-Instruct 95.1 93.1 95.9 86.4 92.9 92.8 93.2 GUI-Owl-1.5-8B-Instruct 97.4 90.5 96.4 90.7 94.2 89.7 93.7 GUI-Owl-1.5-8B-Thinking 95.8 90.5 97.4 90.0 95.0 87.7 93.2 GUI-Owl-1.5-32B-Instruct 97.1 92.6 97.9 89.3 95.5 96.4 95.3 GUI-Owl-1.5-32B-Thinking 96.5 90.5 96.9 86.4 93.8 90.8 93.2

Table 7: Comparison with state-of-the-art methods on the ScreenSpot-V2 dataset.

Instruction Understanding

Interface Perception Interaction Prediction

Avg

Agent Model state widget layout effect type parameter goal plan Proprietary Models

O3 (OpenAI, 2025d) 83.03 84.12 88.39 74.83 75.98 45.75 69.45 95.47 73.30 Gemini-2.5-Pro (Comanici et al., 2025) 81.19 84.36 87.10 71.03 73.25 46.97 67.72 92.56 71.69 GPT-5-Chat (OpenAI, 2025c) 78.90 84.12 88.39 71.55 71.55 43.85 68.98 91.26 70.97 Claude-Sonnet-4-5 (Anthropic, 2025b) 74.77 81.52 82.58 49.83 70.19 43.33 70.30 91.56 66.53 Doubao-V-Pro (Seed, 2025b) 72.48 83.65 81.29 67.24 75.64 41.07 33.07 94.17 63.42 Claude-Sonnet-4 (Anthropic, 2025a) 70.18 78.44 78.06 41.90 62.52 42.11 65.20 94.82 62.16

Open-Source Models

Qwen3-VL-8B-Instruct (Bai et al., 2025a) 76.61 89.81 83.87 58.97 70.20 51.58 67.40 77.99 67.84 Qwen3-VL-8B-Thinking (Bai et al., 2025a) 68.81 76.30 83.23 67.07 70.36 40.73 64.09 91.26 66.81 Qwen2.5VL-72B (Bai et al., 2025c) 69.27 77.49 80.00 61.72 64.91 38.99 62.20 85.44 63.88 Qwen2.5VL-7B (Bai et al., 2025c) 53.21 67.77 60.00 51.72 50.60 39.34 16.22 48.87 45.16 UITARS-1.5-7B (Seed, 2025c) 49.54 59.48 59.35 22.24 59.11 34.32 38.74 55.34 44.27 GUI-OWL-7B (Ye et al., 2025) 60.09 64.93 63.23 21.55 55.37 36.05 21.26 39.81 40.74 GLM-4.5 (Zeng et al., 2025) 49.54 48.10 53.55 27.07 17.55 35.53 28.98 91.91 38.10 Ours

GUI-Owl-1.5-2B-Instruct 60.09 77.73 72.26 44.48 47.04 41.95 62.99 47.57 54.12 GUI-Owl-1.5-4B-Instruct 75.23 88.15 82.58 55.69 65.02 54.22 69.45 70.55 66.64 GUI-Owl-1.5-8B-Instruct 77.98 88.86 84.52 66.90 71.92 61.61 73.54 80.58 72.90 GUI-Owl-1.5-8B-Thinking 75.69 90.05 87.74 67.41 68.23 53.43 67.72 77.67 69.60 GUI-Owl-1.5-32B-Instruct 77.06 92.65 85.81 70.69 73.89 64.12 73.39 88.67 75.45 GUI-Owl-1.5-32B-Thinking 81.19 90.76 85.81 68.10 73.89 57.65 72.91 86.41 73.36

Table 8: Comparison with state-of-the-art methods on the GUI Knowledge Benchmark.

Agent Model Type Success Rate Proprietary / Workflow Models

Agent-S2 w/ Gemini-2.5-Pro Workflow 41.7 M3A w/ Gemini-2.5-Pro Workflow 39.6 T3A w/ Gemini-2.5-Pro Workflow 31.2 Mobile-Agent-E w/ Gemini-2.5-Pro Workflow 12.5 AppAgent w/ Gemini-2.5-Pro Workflow 8.3 Mobile-Agent-V2 w/ Gemini-2.5-Pro Workflow 8.3 SeeAct w/ Gemini-2.5-Pro Workflow 6.2

Native Agent Models

Qwen3-VL-8B-Instruct Model 18.8 GUI-Owl-7B Model 14.6 UI-Venus-7B Model 14.6 UI-TARS-1.5-7B Model 8.3 CogAgent Model 0.0

GUI-Owl-1.5-8B Model 22.9 GUI-Owl-1.5-32B Model 27.1

Table 9: Evaluation results on MemGUI-Bench (Easy tasks).

- 3.2.3 Comprehensive GUI Understanding

GUI Knowledge. The GUI Knowledge Benchmark (Shi et al., 2025) systematically evaluates whether a GUI model possesses sufficient knowledge across three dimensions: Interface Perception (state information understanding, widget function understanding, and layout semantics understanding), Interaction Prediction (action effect, action type prediction, and action parameter prediction), and Instruction Understanding (goal interpretation and task planning). On this benchmark, GUI-Owl-1.5-32B-Instruct achieves an overall accuracy of 75.45, establishing the highest performance among all evaluated models, including proprietary ones such as o3 (73.30) (OpenAI,

- 2025b) and Gemini-2.5-Pro (71.69) (Deepmind, 2025). It attains particularly strong results on widget function understanding and action parameter prediction, substantially outperforming all other models in these categories.

GUI Memory. We further evaluate on MemGUI-Bench (Liu et al., 2026) (Table 9), which assesses an agent’s ability to recall and leverage interaction history over long horizons. Among native agent models, GUI-Owl1.5-32B achieves 27.1, substantially outperforming all prior baselines including Qwen3-VL-8B-Instruct (18.8) and UI-TARS-1.5-7B (8.3). Even our 8B variant (22.9) surpasses all existing native baselines, confirming that our training recipe effectively instills long-horizon memory capabilities without relying on external workflow orchestration.

- 3.3 Detailed Analyses

Effect of Virtual-enviroment trajectory Production and Unified CoT Synthesis. We conduct ablation experiments to validate two key components: virtual environment-based trajectory production (Table 11) and unified CoT synthesis (Table 10).

As shown in Table 11, removing trajectory data produced by virtual environments leads to dramatic performance drops on both PC-Eval (75.4% to 42.0%) and Mobile-Eval (86.7% to 50.0%). Here, PC-Eval is an in-house benchmark focusing on atomic desktop operations such as drag and scroll, as well as office document and spreadsheet editing tasks; Mobile-Eval is an in-house benchmark covering popular Chinese mobile application scenarios including food delivery, ride-hailing, ticket booking, among others. The substantial degradation on both benchmarks confirms that our web-rendering-based virtual environments effectively bypass real-world exploration limitations—such as CAPTCHA interruptions and the lack of accurate feedback—and provide scalable, highquality trajectories that are critical for mastering these challenging scenarios.

As shown in Table 10, removing the unified CoT synthesis causes consistent drops on both OSWorld (52.9% to 47.4%) and AndroidWorld (71.6% to 65.0%), demonstrating that step-wise thought and conclusion augmentation provides essential reasoning supervision. By equipping each trajectory step with observation, memory, reflection, and progress tracking, CoT synthesis enables the model to plan over long horizons and retain key information across steps, which is particularly beneficial for multi-step online tasks across different platforms.

The two components are complementary: virtual environments improve trajectory coverage and quality, while CoT synthesis enhances reasoning and decision-making supervision.

Effect of Unstable-set Train and Interleaved Train in RL. We conduct ablation experiments to validate two critical training strategies for GUI-Owl-1.5’s reinforcement learning optimization, demonstrating the effectiveness

## Unified CoT Synthesis OSWorld AndroidWorld

✗ 47.4 65.0 ✓ 52.9 71.6

- Table 10: Ablation study on the unified CoT synthesis pipeline. The experiments are conducted with GUI-Owl1.5-8B-Thinking.

Virtual Environments PC-Eval Mobile-Eval

✗ 42.0% 50.0% ✓ 75.4% 86.7%

- Table 11: Ablation study on the virtual environments. The experiments are conducted with GUI-Owl-1.5-8BThinking. PC-Eval is an in-house benchmark evaluating atomic operations such as drag and scroll, as well as office document and spreadsheet editing tasks. Mobile-Eval is an in-house benchmark evaluating popular Chinese mobile application scenarios, including food delivery, ride-hailing, ticket booking, among others.

[Figure 72]

[Figure 73]

（a）All-set Train VS Unstable-set Train （b）Mix-platform Train VS Interleaved Train

- Figure 8: Ablation Study on Reinforcement Learning Training Strategies for GUI-Owl-1.5-8B-thinking: Task Selection and Multi-Platform Training Strategies.

of targeted task selection and multi-platform training strategy, as shown in Fig. 8. Fig. 8(a) compares PC validation performance between full dataset training and unstable-task-only training (derived from multi-round rollouts). Unstable-task-focused training achieves faster convergence and higher final accuracy, demonstrating the efficacy of prioritizing challenging tasks for robust model optimization. In Fig. 8 (b), mix-platform training (simultaneous multi-platform data optimization) is contrasted with our interleaved training (switch from Mobile to PC at step 10). Mix-platform training exhibits performance oscillation due to cross-platform interference, whereas interleaved training enables focused optimization per platform while maintaining performance stability during transitions. This approach achieves synergistic multi-platform growth, validating the superiority of our presented interleaved RL training strategy.

- 3.4 Case Study

We present three representative cases to illustrate the comprehensive capabilities of GUI-Owl 1.5 beyond basic GUI navigation.

Mobile Use Case (Fig. 9) In this case, the user seeks to determine the total follower count of the ModelScope Community account across two social media platforms: Xiaohongshu and Douyin. The agent first launches the Xiaohongshu app, enters the account name in the search box, retrieves the follower information, and stores it in memory. Subsequently, the agent navigates to the Douyin app to obtain the corresponding follower count. By combining the retrieved information from memory with the current data, the agent calculates and reports the number of overall follower across both platforms.

Computer Use Case (Fig. 10). Figure 10 illustrates a case of GUI-Owl-1.5 executing a web search and notetaking task on the Windows platform. To fulfill the user query, the agent is required to accurately perform multiple

[Figure 74]

###### User Query:帮我在⼩红书、抖⾳看⼀下"魔搭ModelScope社区"账号，并告诉我这两个平台的总粉丝数

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Thought:⼩红书应⽤已成功打开，主界⾯ 显示发现⻚。现在需要搜索“魔搭 ModelScope社区”账号以查看其粉丝数。 右上⻆有搜索图标，这是查找账号的⼊⼝。 点击搜索图标后，会进⼊搜索界⾯，输⼊ 账号名称即可找到⽬标账号并查看粉丝数。

[Figure 82]

Thought:搜索界⾯已打开，搜索栏显示“装

[Figure 83]

Thought:当前是任务的第⼀步，我需要打 开⼩红书App。

修”，虚拟键盘已激活。我需要清除当前 内容并输⼊⽬标账号“魔搭ModelScope社 区”以查找其粉丝数。

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Step 3:在搜索栏中输⼊“魔搭ModelScope社 区”以搜索⽬标账号。

[Figure 88]

[Figure 89]

Step 1:打开⼩红书App

Step 2:点击屏幕右上⻆的搜索图标。

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Thought:当前⻚⾯显示的是⼩红书平台的 搜索结果，已看到“魔搭ModelScope社区” 账号在⼩红书的粉丝数为1.1万。现在需

[Figure 97]

Thought:搜索已成功执⾏，我清楚看到了 “魔搭ModelScope社区”账号的粉丝数为1.1 万（11,000）。现在需要点击右上⻆的关 注按钮，这将帮助我确认当前账号状态并 返回搜索结果⻚⾯。

[Figure 98]

Thought:搜索词已成功输⼊，系统显示相 关建议，表明搜索功能正常。当前搜索建 议中未直接显示⽬标账号，需点击“搜索” 按钮进⼊结果⻚，查看详细信息和粉丝数。

要切换到抖⾳平台查看该账号的粉丝数。 为了在抖⾳中搜索该账号，我需要先返回 主屏幕，然后找到并打开抖⾳应⽤。

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Step 4:点击搜索栏右上⻆的“搜索”按钮， 执⾏对“魔搭ModelScope社区”的搜索。

[Figure 103]

[Figure 104]

Step 5: 点击关注按钮确认账号状态 Step 6:点击主⻚按钮返回主屏幕

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Thought:点击搜索图标成功打开了抖⾳的 搜索界⾯，键盘已弹出，搜索栏显示占位 符⽂本。搜索栏已激活，键盘显示，建议 搜索项中出现了“魔搭modelscope”，但需 要完整输⼊⽬标名称以确保准确性。

Thought: 我已成功返回主屏幕，现在可以 看到所有应⽤图标。接下来需要在抖⾳上 查找该账号并获取粉丝数。我看到抖⾳图 标位于主屏幕第三⾏，点击它即可打开应 ⽤并搜索“魔搭ModelScope社区”。

[Figure 113]

Thought:抖⾳应⽤已成功打开，主界⾯完

全加载，导航元素可⻅。现在需点击右上 ⻆的搜索图标（放⼤镜），进⼊搜索界⾯ 输⼊账号名称，找到其粉丝数。

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Step 7:点击主屏幕⽹格中第三⾏的抖⾳应 ⽤图标。

Step 8:点击屏幕右上⻆的搜索图标（放⼤ 镜）。

Step 9:使⽤显示的键盘在搜索栏中输⼊“魔 搭ModelScope社区”。

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Thought:搜索成功，已找到抖⾳上“魔搭 ModelScope社区”的账号信息，粉丝数为 1.0万（10,000）。此前在⼩红书上该账号 的粉丝数为1.1万（11,000）。总粉丝数为 11,000 + 10,000 = 21,000。屏幕中可⻅抖

[Figure 125]

Thought:已在抖⾳搜索栏输⼊“魔搭 ModelScope社区”，系统显示了相关建议， 确认输⼊⽆误。搜索建议中第⼀条“魔搭 modelscope社区”与⽬标匹配，但需点击 “搜索”按钮执⾏查询，以获取完整结果和 粉丝数据。

⾳⽤户⻚显示“粉丝：1.0万”及红⾊“关注” 按钮，内容包含“魔搭社区指南！”等官⽅ 推⽂，符合任务要求。

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Step 11:提供答案，说明该账号在⼩红书和 抖⾳上的总粉丝数为21,000。 ✅

Step 10:点击搜索栏右上⻆的红⾊“搜索”按 钮，执⾏对“魔搭ModelScope社区”的搜索。

[Figure 130]

- Figure 9: A complete operation process on the Android platform, in which the user query requires the agent to search and summarize information on social media platforms.

web searches and extract key information relevant to subsequent steps from the search results, which is stored as memory within the thought content (highlighted in green). Subsequently, the agent switches to a different application, creates a new spreadsheet in WPS Office, and fills in the corresponding content at the appropriate cells based on the memorized information. The thoughts generated by GUI-Owl-1.5 during the execution steps demonstrate its understanding of screen content, precise grounding, analysis of task progress, and memorization of key information, validating the effectiveness of our proposed unified CoT synthesis pipeline.

Tool Use Case (Fig. 11). In this case, the agent is tasked with completing a partially implemented Python script on the desktop and saving its execution output. The agent seamlessly interleaves MCP tool calls with GUI operations: it first reads the source code via the filesystem_read_text_file tool, identifies and fixes the incomplete insertionsortimplementationusing filesystem_edit_file, then opensaterminalthrough osworld_mcp_os.open_shell to execute the script via command-line input, and finally verifies the output by reading the generated log file. This case demonstrates GUI-Owl 1.5’s ability to autonomously decide when to use tool invocation versus direct GUI manipulation within a single trajectory.

- 4 Conclusion

In this work, we presented GUI-Owl-1.5, the native GUI agent model that features instruct/thinking variants in multiple sizes (2B/4B/8B/32B/235B) and supports a range of devices (desktop, mobile, browser, and more). GUI-Owl-1.5 achieves state-of-the-art performance on 20+ GUI benchmarks, comprehensively covering GUI automation, grounding, tool calling, memory, and knowledge tasks. We innovatively improve the model’s robust generalization in real-world application scenarios through a Hybrid Data Flywheel, unified enhancement of agent capabilities, and multi-device environment RL scaling. We hope that the open-source release of GUI-Owl-1.5 will advance the adoption of GUI agents for device automation across a wide range of platforms.

[Figure 131]

###### User Query: Search for the stock prices of Apple and Nvidia respectively. Then create a new spreadsheet in WPS Office. Enter the company names in column A and the retrieved stock prices in column B.

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]<br><br>. . .|
|---|

|[Figure 136]|
|---|

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Thought: The search bar is active, so I can now enter the query. The task requires searching for the stock prices of Apple and Nvidia. Typing 'Apple stock price' into the search bar will initiate the search process. This step is necessary to retrieve the stock price data needed for the spreadsheet. Once entered, the search engine will process the query and display results, allowing me to gather the required information.

Thought: The search bar currently shows 'Apple stock price' with autocomplete suggestions. I need to clear this text to prepare for searching Nvidia's stock price. Clicking the clear button (X icon) at the end of the search bar will remove the current query, allowing me to type the new search term for Nvidia. This is necessary before proceeding with the second stock search as per the task requirements.

[Figure 144]

Thought: Since there were no prior actions, the computer state shows the Bing search page open. To search for Apple and Nvidia stock prices, the first step is to click the search bar to input the query. This is necessary to begin retrieving the stock prices. Once the search bar is clicked, I can type the company names and proceed with the search.

[Figure 145]

[Figure 146]

Thought: The search bar contains the query 'Apple stock price' and autocomplete suggestions are visible. The next step is to execute the search to retrieve Apple's stock price. Pressing Enter is the most direct way to submit the query. This will navigate to the search results page, where we can find the current stock price for Apple.

Thought: The search results for Apple's stock price are now displayed, showing the current price of $255.78. This completes the first part of the task. The next step is to search for Nvidia's stock price. To do this, I need to activate the search bar again by clicking on it. This will allow me to type in the new query for 'Nvidia stock price'.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Step 4: Click on the search bar to activate it for entering the Nvidia stock price search query.

Step 5: Click the clear button (X icon) in the search bar to remove the current search query.

Step 1: Click on the search bar to activate it for text input.

Step 2: Type "Apple stock price" into the active search bar.

Step 3: Press the Enter key to submit the search query Apple stock price.

|. . .<br><br>[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Thought: I've successfully retrieved the Nvidia stock price of $182.81 from the search results, completing the data collection phase. Both Apple ($255.78) and Nvidia ($182.81) stock prices are now available. The next step is to transition from the browser to WPS Office to create a new spreadsheet. I need to click the close button in the top-right corner of the browser window to return to the desktop.

[Figure 168]

[Figure 169]

Thought: I can see that ‘Nvidia stock price’ has been typed into the search bar, and autocomplete suggestions are now showing. I have already retrieved Apple‘s stock price of $255.78 from the previous search results. The next logical step is to press Enter to execute the search for Nvidia's stock price. This will load the search results page where I can retrieve Nvidia's current stock price.

Thought: WPS Office has been opened successfully and is showing the template selection screen. I have completed data collection: Apple’s stock price is 255.78 and NVIDIA’s is 182.81. Now I need to create a new spreadsheet to enter this data. On the current screen, there is a dedicated “Sheets” option in the left sidebar, which is the direct entry point for creating a spreadsheet.

[Figure 170]

[Figure 171]

Thought: The search bar is now cleared, showing only a cursor, which prepares me to search for Nvidia's stock price. I've already retrieved Apple's stock price ($255.78), and now I need to search for Nvidia's price by typing 'Nvidia stock price' into the active search bar. This follows the same method used for Apple, ensuring consistency.

Thought: I have completed the informationgathering phase and obtained the stock prices for Apple and NVIDIA: Apple is 255.78 and NVIDIA is 182.81. Now I need to create a new WPS spreadsheet. There is a visible WPS Office icon on the desktop, and the most effective way is to double-click the icon to launch the program.

[Figure 172]

Step 10: Click on the "Sheets" option in the left sidebar to access spreadsheet creation templates.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Step 9: Double-click on the WPS Office icon to launch the WPS Office application.

Step 6: Type "Nvidia stock price" into the empty search bar.

Step 7: Press the Enter key to submit the "Nvidia stock price" search query

Step 8: Click the close button (X) in the topright corner of the browser window.

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Thought: I can see that “Apple” has been entered into cell A1, as shown in the cell and formula bar. The next step is to continue building the data table. Since I have both company names and stock prices, I should continue systematically. The logical next action is to move to cell A2 to enter "Nvidia", continuing the list of company names in column A before adding the corresponding prices in column B.

Thought: I can see that Apple's stock price '255.78' has been successfully entered into cell B1, completing the first row of data entry. The next logical step is to click on cell A2 to select it and prepare to enter 'Nvidia'. This follows the natural data entry pattern of moving down to the next row after completing the previous one. Once cell A2 is selected, I can proceed to enter 'Nvidia' and then continue with its stock price in cell B2.

[Figure 194]

[Figure 195]

Thought: The interface has successfully switched to the Sheets section of WPS Office. This is the correct location to create a new spreadsheet for entering the stock price data I retrieved earlier. I now have the required information: Apple‘s stock price is $255.78 and Nvidia’s is $182.81. There is a 'Blank' template option in the main content area, which is ideal for this task.

Thought: I can see that cell B1 is currently selected, which is the correct location to enter Apple‘s stock price since column B is designated for stock prices. The next logical step is to type '255.78' to record Apple's stock price. This will complete the first row of the table and align with the task structure. After this, I will proceed to cell A2 to enter 'Nvidia', followed by its stock price in cell B2.

[Figure 196]

Thought: The spreadsheet application opened successfully, and I now have a clean, empty grid to work with. The cursor is positioned in cell A1, the ideal starting point for entering the company names. Since cell A1 is active, I will start by typing 'Apple' into this cell to begin structuring the data table.

[Figure 197]

[Figure 198]

[Figure 199]

Step 15: Click on cell A2 in the spreadsheet to select it for entering "Nvidia" as the second company name.

[Figure 200]

[Figure 201]

Step 11: Click on the "Blank" spreadsheet template in the main content area to create a new empty spreadsheet document.

Step 12: Type "Apple" into the currently selected cell A1 to enter the first company name in the spreadsheet.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Step 14: Type "255.78" into the currently selected cell B1 to enter Apple's stock price.

Step 13: Click on cell A2 in the spreadsheet to select it for entering the second company name.

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Thought: The cursor has successfully moved from cell A2 to cell B2, confirming the Tab key navigation worked as expected. The next step is to input Nvidia's stock price of 182.81 into cell B2. Since the cursor is already in B2, I can directly type the value. After entering the price, the data entry portion of the task will be complete, with both company names in column A and their corresponding stock prices in column B.

[Figure 216]

[Figure 217]

Thought: I can see that cell A2 is selected and ready for input, with the name box confirming this selection. Now I need to continue systematically by entering 'Nvidia' in cell A2. After this, I will move to cell B2 to enter Nvidia's stock price of '182.81'. This maintains the consistent structure of company names in column A and their corresponding stock prices in column B.

Thought: I can confirm that 'Nvidia' has been successfully entered into cell A2, completing the data entry for company names. Since cell A2 is currently selected, I need to move to cell B2 to input Nvidia's stock price of 182.81. The most efficient way to do this is by using the Tab key to move one cell to the right, which is standard spreadsheet navigation behavior.

[Figure 218]

Thought: I can confirm that all requirements have been met. The data is properly formatted with company names in column A and stock prices in column B. Both Apple (255.78) and Nvidia (182.81) are correctly entered in the WPS Office spreadsheet. Since all objectives are fulfilled, the task can be concluded successfully.

[Figure 219]

[Figure 220]

Step 16: Type "Nvidia" into the currently selected cell A2 to enter the second company name in the spreadsheet.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Step 19: Terminate the task with success status.

Step 17: Press the Tab key to move from the currently selected cell A2 to cell B2.

Step 18: Type "182.81" into the currently selected cell B2 to enter Nvidia's stock price

[Figure 227]

✅

- Figure 10: A complete operation process on the Windows platform, in which the user query requires the agent to memorize key on-screen information.

[Figure 228]

###### User Query: Please complete the code and retrieve the output from the Python script 'calculator.py' located on the desktop and save it as 'log.txt' in the same directory as the Python file.

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

(‘path’: ‘/home/user/Desktop/calculat or.py’, 'edits': 'path': '/home/user/Desktop/calculator.py', 'edits': [ { 'oldText': ( ' while j >=0 and key < arr[j] : \n' ' # TODO: Replace the value at arr[j+1] with the value at arr[j]\n' ' \n' ' j -= 1' ), 'newText': ( ' while j >=0 and key < arr[j] : \n' ' # TODO: Replace the value at arr[j+1] with the value at arr[j]\n' ' arr[j+1] = arr[j]\n' ' j -= 1' ), } ],

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Thought: I need to complete a task: read a Python script named ' calculator.py' from the desktop, run it, get the output, and save t he output as a 'log.txt' file in the same directory as the script. Th e file is currently visible on the desktop, and the path should be / home/user/Desktop/calculator.py. First, I need to read the file co ntents to understand its functionality, which will help determine how to execute it and what output to expect. Using the filesyste m_read_text_file tool is the most effective approach, as it can dir ectly read the file contents without needing to open an editor thr ough the graphical interface — more efficient and accurate. Setti ng the path to "/home/user/Desktop/calculator.py" follows the st andard Ubuntu directory structure. After reading, I will have clari ty on how to execute the script and capture its output in the next step.

[Figure 243]

Thought: The terminal has been successfully opened with comm and-line access, currently in the home directory (~). We need to navigate to the Desktop directory to work with the calculator.py f ile. According to the task requirements, the following steps need to be completed: 1) enter the Desktop directory; 2) execute the s cript; 3) save the output as log.txt. The predicted command is 'cd Desktop && python3 calculator.py > log.txt', which combines dire ctory navigation, script execution, and output redirection. The co de in calculator.py has already been successfully completed in th e previous step via filesystem_edit_file, so the script is ready to r un. This command will run the script in the Desktop directory and write the results to log.txt; any error messages may also be displa yed in the terminal.

[Figure 244]

Thought: The previous step successfully read the contents of calc ulator.py via filesystem_read_text_file. The current predicted acti on is to use filesystem_edit_file to modify the file, replacing the T ODO comment with arr[j+1] = arr[j], which aligns with the eleme nt-shifting logic in an insertion sort algorithm. Based on the task context and code snippet, this modification is the correct comple tion. Therefore, the editing operation should proceed to complet e the code implementation, so the script can be run and output c aptured afterward. The next step after editing is to execute the s cript and save the output as log.txt.

[Figure 245]

Thought: The previous steps have successfully read and edited ca lculator.py using filesystem_read_text_file and filesystem_edit_fil e respectively. The code has been completed with the correct ins ertion sort logic. Now, to execute the script and capture its outpu t, switching to the shell is the natural next step: it provides direct command-line access, allowing us to run python3 calculator.py a nd redirect the output to log.txt. open_shell requires no paramet ers and can quickly launch a terminal, leveraging Ubuntu's native commands to complete the task.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Step 1: tool.filesystem_read_text_file ('path': '/home/user/Desktop/calculator.py')

Step 3: tool.open_shell Step('text':4:'cdtypeDesktop && python3 calculator.py > log.txt')

[Figure 252]

[Figure 253]

Step 2: tool.filesystem_edit_file

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Thought:The previous step successfully read the contents of log.t xt using filesystem_read_text_file, confirming that the script out put has been correctly saved. The terminal also shows that the c ommand completed without errors. All task steps have now been completed:

[Figure 262]

Thought:The command has been successfully executed. The ter minal shows the current path as ~/Desktop$, indicating: Successfully navigated to the Desktop directory The Python script calculator.py has been run Output has been redirected to log.txt The command completed without any errors The previous operations have completed all the steps required f or the task: Completed the code in calculator.py (added arr[j+1] = arr[j]) Executed the script Saved the output to log.txt Next, we should verify that log.txt was successfully generated an d contains the expected output. To do this, we need to read the f ile located at /home/user/Desktop/log.txt. Using the filesystem_ read_text_file tool is appropriate, as it is designed for reading te xt files and can fully retrieve the script output. The path paramet er is set to "/home/user/Desktop/log.txt" to confirm the task co mpletion status.

[Figure 263]

Thought:From the previous actions, it is clear that the command has been successfully typed into the terminal but has not been e xecuted yet. Inputting text and executing a command are two se parate steps in automated terminal operations. Based on the op eration history: 1. calculator.py has been edited, completing the i nsertion sort algorithm with arr[j+1] = arr[j]; 2. A terminal was o pened using the osworld_mcp_os.open_shell tool; 3. A comman d was typed to navigate to the Desktop directory, run the Python script, and redirect output to log.txt. The current need is to exec ute this command to fulfill the task objective: complete the code (done), run the script, and save the output as log.txt in the same directory. The command "cd Desktop && python3 calculator.py > log.txt" does the following: enters the Desktop directory, execut es the Python script, and writes standard output to log.txt. To ex ecute the already-typed command, the Enter key (Return) must be pressed. The predicted code is pyautogui.hotkey('Return'), w hich simulates pressing the Enter key — this is the critical next st ep to complete the task.

✓ calculator.py code completion (insertion sort implementation)

✓ Script execution

✓ Output saved to log.txt in the same directory as the script

✓ log.txt contents read and verified All operations required by the user's instructions have been fulfil led: Code completion: accomplished via filesystem_edit_file Script execution: accomplished via terminal command Output saving: achieved through > redirection Content verification: accomplished via the file reading tool All objectives have been achieved, and the next step should be t

- o terminate the task with a success status. The predicted code c
- omputer.terminate(status='success') conforms to this operationa l specification.

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Step 6: tool.filesystem_read_text_file ('path': '/home/user/Desktop/log.txt’})

Step 7: terminate ('status': 'success'})

Step 5: key ('keys': ['Return']})

#### Figure 11: A case of a complete operation process on a desktop platform, which combining extended tools and computer use actions.

References

Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents, 2025. URL https://arxiv.org/abs/2504.

00906. 2

Moises Andrade, Joonhyuk Cha, Brandon Ho, Vriksha Srihari, Karmesh Yadav, and Zsolt Kira. Let’s think in two steps: Mitigating agreement bias in mllms with self-grounded verification. arXiv preprint arXiv:2507.11662,

2025. 12 Anthropic. System card: Claude opus 4 & claude sonnet 4, 2025a. URL https://www.anthropic.com/ claude-4-system-card. Accessed: 2025-09-25. 2, 16 Anthropic. Claude sonnet 4.5. https://docs.claude.com/docs/about-claude/models/ whats-new-claude-4-5, 2025b. Accessed: 2025-11-22. 16 Anthropic. Claude 3.7 sonnet and claude code. Technical report, Anthropic, 2025c. URL https://www. anthropic.com/news/claude-3-7-sonnet. System Card. 12, 14, 16 Anthropic. Claude-4-sonnet. Technical report, Anthropic, 2025d. URL https://www.anthropic.com/ news/claude-4-sonnet. 12 Anthropic. Claude-4-5-sonnet. Technical report, Anthropic, 2025e. URL https://www.anthropic.com/ news/claude-4-5-sonnet. 12

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a. 2, 14, 15, 16

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b. 14

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025c. 16

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024. 13

CodeFuse. Oagent. https://github.com/codefuse-ai/Oagent, 2025. 12

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 16

Deepmind. Gemini 2.5: Our most intelligent ai model. Technical report, Deepmind, 2025. URL https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/. 12, 17

Google DeepMind. Gemini 3 pro, 2025. URL https://deepmind.google/models/gemini/pro/. 2, 12, 14

Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, and Lei Yu. Dynaweb: Model-based reinforcement learning of web agents, 2026. URL https://arxiv. org/abs/2601.22149. 12

BoyuGou, RuohanWang, BoyuanZheng, YananXie, ChengChang, YihengShu, HuanSun, andYuSu. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243,

2024. 14, 15

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=kxnoqaisCT. 14

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833, 2025. 14, 15, 16

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Buildinganend-to-endwebagentwithlargemultimodalmodels. arXivpreprintarXiv:2401.13919,

2024. 13 Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, and Xia Song. Webstar: Scalable data synthesis for computer use agents with step-level filtering, 2026. URL https://arxiv.org/abs/2512.10962. 12

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 14

Hongrui Jia, Jitong Liao, Xi Zhang, Haiyang Xu, Tianbao Xie, Chaoya Jiang, Ming Yan, Si Liu, Wei Ye, and Fei Huang. Osworld-mcp: Benchmarking mcp tool invocation in computer-use agents. arXiv preprint arXiv:2510.24563, 2025. 13

Kimi, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. 12

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 881–905, 2024a. 12, 13

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024b. 12

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents,

2025. URL https://arxiv.org/abs/2407.01476. 12

Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, et al. Mobileworld: Benchmarking autonomous mobile agents in agent-user interactive and mcp-augmented environments. arXiv preprint arXiv:2512.19432, 2025. 13

Guangyi Liu, Pengxiang Zhao, Yaozhen Liang, Qinyi Luo, Shunye Tang, Yuxiang Chai, Weifeng Lin, Han Xiao, WenHao Wang, Siheng Chen, Zhengxi Lu, Gao Wu, Hao Wang, Liang Liu, and Yong Liu. Memgui-bench: Benchmarking memory of mobile gui agents in dynamic environments, 2026. URL https://arxiv.org/ abs/2602.06075. 13, 17

Xiao Liu, Bo Qin, Dongzhu Liang, Guang Dong, Hanyu Lai, Hanchen Zhang, Hanlin Zhao, Iat Long Iong, Jiadai Sun, Jiaqi Wang, et al. Autoglm: Autonomous foundation agents for guis. arXiv preprint arXiv:2411.00820,

2024. 2

Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239, 2025. 14

Magnitude. Magnitude. https://magnitude.run, 2025. 12 OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025a.

URL https://openai.com/index/computer-using-agent. 12, 14, 15, 16

OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, 2025b. URL https://cdn.openai. com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card. pdf. System Card. 17

OpenAI. Gpt-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf, 2025c. 2, 16 OpenAI. Openai o3 and o4-mini system card, 2025d. URL https://openai.com/index/

o3-o4-mini-system-card/. Accessed: 2025-09-25. 16

VirajPrabhu, YutongDai, MatthewFernandez, JingGu, KrithikaRamakrishnan, YanqiLuo, SilvioSavarese, Caiming Xiong, Junnan Li, Zeyuan Chen, et al. Walt: Web agents that learn tools. arXiv preprint arXiv:2510.01524,

2025. 12

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025. 2, 14, 15, 16

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024. 13

Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning, 2025. URL https://arxiv.org/ abs/2505.23678. 12

Bytedance Seed. Seed1.8 model card: Towards generalized real-world agency. arXiv preprint, December 2025a. Technical Report. 12, 14

ByteDance Seed. Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b. 12, 15, 16

- ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025c. 12, 14, 15, 16
- ByteDance Seed. Ui-tars-2. https://seed-tars.com/showcase/ui-tars-2, 2025d. 2, 12 ByteDance Seed. Ui-tars. https://seed-tars.com/1, 2025e. 12

Chenrui Shi, Zedong Yu, Zhi Gao, Ruining Feng, Enqi Liu, Yuwei Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, and Qing Li. Gui knowledge bench: Revealing the knowledge gap behind vlm failures in gui tasks. arXiv preprint arXiv:2510.26098, 2025. 17

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, et al. Gui-g2: Gaussian reward modeling for gui grounding. arXiv preprint arXiv:2507.15846, 2025. 14

Anshul Tibrewal. Deepsky agent. https://deepskyai.substack.com/p/

building-a-practical-browser-agent, 2025. 12 Browser Use. Browser use. https://github.com/browser-use/browser-use, 2025. 12 Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang.

Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. Advances in Neural Information Processing Systems, 37:2686–2710, 2024a. 2

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobileagent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024b. 2

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents, 2025a. URL https: //arxiv.org/abs/2508.09123. 2, 12, 14, 15

Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-agent-e: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733, 2025b. 2

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 14, 15, 16

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024. 13

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computeruse grounding via user interface decomposition and synthesis, 2025. URL https://arxiv.org/abs/ 2505.13227. 14, 15, 16

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454,

2024. 14, 15

Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, et al. Evocua: Evolving computer use agents via learning from scalable synthetic experience. arXiv preprint arXiv:2601.15876, 2026. 12, 14, 15, 16

Haolong Yan, Jia Wang, Xin Huang, Yeqing Shen, Ziyang Meng, Zhimin Fan, Kaijun Tan, Jin Gao, Lieyu Shi, Mi Yang, et al. Step-gui technical report. arXiv preprint arXiv:2512.15431, 2025. 12, 14

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. 12

Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, et al. Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791, 2025b. 14, 15, 16

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144,

2025. 2, 3, 12, 14, 15, 16

Wenwen Yu, Zhibo Yang, Jianqiang Wan, Sibo Song, Jun Tang, Wenqing Cheng, Yuliang Liu, and Xiang Bai. Omniparser v2: Structured-points-of-thought for unified visual text parsing and its generality to multimodal large language models. arXiv preprint arXiv:2502.16161, 2025. 16

Yutori. Navigator. https://yutori.com/blog/introducing-navigator, 2025. 12 Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao

Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025. 16

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pp. 1–20, 2025. 2

Hanzhang Zhou, Xu Zhang, Panrong Tong, Jianan Zhang, Liangyu Chen, Quyu Kong, Chenglin Cai, Chen Liu, Yue Wang, Jingren Zhou, et al. Mai-ui technical report: Real-world centric foundation gui agents. arXiv preprint arXiv:2512.22047, 2025. 2, 12, 14, 15

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023. 13

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 14

