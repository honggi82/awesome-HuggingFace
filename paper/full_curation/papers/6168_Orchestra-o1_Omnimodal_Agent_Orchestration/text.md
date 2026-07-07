## arXiv:2606.13707v1[cs.AI]10Jun2026

[Figure 1]

[Figure 2]

# Orchestra-o1: Omnimodal Agent Orchestration

###### Fan Zhang1,∗, Vireo Zhang∗, Shengju Qian2,†, Haoxuan Li3, Hao Wu4, Jinyang Wu4, Donghao Zhou1, Zhihong Zhu3, Zheng Lian5, Xin Wang2, Pheng-Ann Heng1,†

1CUHK 2LIGHTSPEED 3PKU 4THU 5Tongji University

fzhang25@cse.cuhk.edu.hk ∗ Equal Contribution † Corresponding Author https://github.com/zfkarl/Orchestra-o1 https://huggingface.co/Karl28/Orchestra-o1-8B

Abstract The recent success of agent swarms has shifted the paradigm of large language model (LLM)based agents from single-agent workflows to multi-agent systems, highlighting the importance of agent orchestration for task decomposition and collaboration. However, existing orchestration frameworks are limited to a narrow set of modalities and struggle to generalize to more complex settings where heterogeneous modalities coexist and interact. This limitation becomes particularly pronounced in omnimodal scenarios, where tasks require the unified understanding and coordination of diverse inputs such as text, image, audio, and video. In this work, we propose Orchestra-o1, an omnimodal agent orchestration framework designed to support efficient agent collaboration across multiple modalities. Orchestra-o1 introduces a unified orchestration mechanism that enables modality-aware task decomposition, online sub-agent specialization, and parallel sub-task execution. This scalable design allows agent systems to effectively tackle complex real-world tasks involving heterogeneous information sources, surpassing the second-best approach by 10.3% accuracy on the OmniGAIA benchmark. Furthermore, we introduce decision-aligned group relative policy optimization (DA-GRPO), an efficient agentic reinforcement learning approach for training Orchestra-o1-8B, which also achieves state-of-the-art performance against all existing open-source omnimodal agents. The source code is publicly available at the above links.

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

[Figure 18]

[Figure 19]

[Figure 20]

Task Agent

Task Orchestrator

Task Orchestrator

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Native Perception

Parallel Sub-agents

Linear Sub-agents

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Reasoning & Acting

Omnimodal Tools

Unimodal Tools

Answer

Answer

Answer

(a) Native Omnimodal Agents (b) Previous Orchestration-based Agents (c) Ours: Orchestra-o1

Figure 1 | Comparison among three types of omnimodal agents.

### 1. Introduction

Large language model (LLM)-based agents [Luo et al., 2025, Wang et al., 2024] have recently emerged as a powerful paradigm for building intelligent systems that can reason, plan, use tools, and interact with external environments. By augmenting LLMs with harness mechanisms [Meng et al., 2026, Pan et al., 2026], agent systems have substantially expanded the boundary of what language models can accomplish. Representative applications such as code generation and execution [Huang et al., 2023, Zhang et al., 2024b], autonomous web research [Qiao et al., 2025, Team et al.,

- 2025b], interactive problem solving [Tao et al., 2025, Yu et al., 2026], and open-ended computer-use tasks [Agashe et al., 2025, Wang et al.] have demonstrated the potential of LLM agents to reshape human productivity and information access. More recently, the success of agent swarms [Team et al.,
- 2026] has further shifted the research focus from single-agent workflows to multi-agent systems, where a main agent coordinates multiple specialized agents to decompose complex tasks, execute

sub-tasks, and aggregate intermediate results. This paradigm highlights the importance of agent orchestration, which determines how agents are created, specialized, scheduled, and coordinated during task solving.

Despite this progress, most existing LLM-based agent systems are still designed for a limited range of modalities, typically focusing on either pure-text tasks [Zhang et al., 2024a] or vision-language tasks [Geng et al., 2025]. This creates a clear gap between current agent research and real-world scenarios, where information is inherently omnimodal and often involves the coexistence and interaction of text, image, audio, and video. In everyday situations, humans naturally process heterogeneous sensory signals in a unified manner. For example, during face-to-face communication, people simultaneously interpret spoken language, facial expressions, gestures, and environmental cues, and then decide how to respond accordingly. Such omnimodal understanding and decision-making are natural for humans but remain highly challenging for existing agents. To solve omnimodal tasks, an agent must not only perceive information from diverse modalities, but also reason over their interactions, decide which specialized capabilities are needed, and coordinate actions across multiple tools or sub-agents. This requires a unified framework that supports both omnimodal perception and high-level agentic decision-making.

- As shown in Figure 1, current omnimodal agents can be broadly categorized into two types. The first category is native omnimodal agents [Team, 2026], which directly employ an omnimodal large language model (OLLM) as the agentic backend and equip it with various action tools. In this design, the same model is expected to perform perception, reasoning, planning, and tool-use simultaneously. However, existing OLLMs still exhibit limited capability in jointly handling perception and action, especially when tasks require long-horizon reasoning, external information seeking, code execution, or fine-grained cross-modal understanding. As a result, even strong proprietary omnimodal models such as Gemini-3-Pro [Google, 2025d] achieve only 62.5% accuracy on the challenging benchmark OmniGAIA [Li et al., 2026]. The second category is orchestration-based agents [Ruan et al., 2026], which decouple perception and action from high-level reasoning. In such systems, a text-based language model usually serves as the main agent or orchestrator, while perception and action are delegated to specialized sub-agents equipped with corresponding tools. This design separates highlevel decision-making from low-level modality processing, making the system more modular, extensible, and potentially more scalable for complex omnimodal tasks.

In this paper, we focus on orchestration-based omnimodal agents. Designing an effective omnimodal agent swarm, however, is non-trivial for the following reasons. First, many powerful closed-source agent swarm frameworks, such as Kimi [Team et al., 2026] and Claude [Anthropic, 2026], are hidden behind proprietary APIs, making it difficult to extend them for omnimodal research. Second, existing open-source agent orchestration frameworks [Ruan et al., 2026, Su et al., 2025] are often limited by incomplete perception and action toolsets, as well as relatively rigid and linear sub-agent workflows. These limitations restrict both the scalability and efficiency of agent systems when handling complex tasks involving heterogeneous modalities. Towards this end, we propose Orchestra-o1, an omnimodal agent orchestration framework designed to support efficient agent collaboration across multiple modalities. At the model level, Orchestra-o1 supports flexible agentic backends, allowing both the main agent and sub-agents to be instantiated with different models, including open-source models and proprietary models. At the tool level, we provide a unified tool ecosystem consisting of perception tools and action tools, enabling the system to understand and coordinate diverse inputs such as text, image, audio, and video, while also supporting external information seeking and code execution.

- At the scaffold level, Orchestra-o1 introduces a collaborative orchestration mechanism based on agent skills and context memory, enabling modality-aware task decomposition, online sub-agent specialization, and parallel sub-task execution. Together, these designs make Orchestra-o1 both effective and efficient for solving complex omnimodal agent tasks. When using GPT-5 [OpenAI,

2025] as the main agent, Orchestra-o1 establishes a new state-of-the-art (SOTA) on the OmniGAIA benchmark and substantially outperforms competing baselines, achieving a 32.8% improvement over AOrchestra [Ruan et al., 2026] and a 10.3% improvement over Gemini-3-Pro [Google, 2025d].

In addition to the orchestration framework, we further explore how to train an open-source model to serve as the main agent in Orchestra-o1. To this end, we propose decision-aligned group relative policy optimization (DA-GRPO), an efficient offline agentic reinforcement learning algorithm for enhancing orchestration decision-making. DA-GRPO extends GRPO [Team, 2025a] with a design specifically tailored for agent orchestration. Unlike the original GRPO, which focuses solely on finalanswer correctness, DA-GRPO explicitly aligns the main agent’s step-level decisions with high-quality reference trajectories, covering key decisions such as task delegation, sub-agent selection, tool usage, and answer generation. Leveraging high-quality synthetic trajectories and a multi-dimensional rubricbased reward design, we train Orchestra-o1-8B based on Qwen3-8B [Yang et al., 2025] to serve as an open-source main agent within the Orchestra-o1 framework. Experimental results demonstrate that Orchestra-o1-8B significantly improves the performance of open-source omnimodal agents on OmniGAIA, increasing the previous best accuracy from 20.8% to 30.0%.

In summary, the main contributions of this paper are as follows:

- • Omnimodal Agent Orchestration Framework. We propose Orchestra-o1, an omnimodal agent orchestration framework for complex real-world agent tasks. Through modality-aware task decomposition, online sub-agent specialization, and parallel sub-task execution, Orchestra-o1 decouples high-level orchestration from specialized perception and action execution, serving as a scalable open-source framework for building omnimodal agent swarms.
- • Efficient Agent Orchestration Training Recipe. We develop DA-GRPO, an efficient agentic reinforcement learning algorithm for orchestration training. DA-GRPO aligns the main agent’s step-level orchestration decisions with high-quality reference trajectories based on multi-dimensional rubric reward design, enabling open-source models to acquire stronger delegation, planning, and decisionmaking capabilities in omnimodal agent systems.
- • Multifaceted Experimental Validation. Extensive experiments demonstrate that Orchestra-o1 significantly outperforms existing omnimodal agents. With a strong proprietary main agent, it achieves a new state-of-the-art on OmniGAIA, surpassing the second-best approach by 10.% accuracy. Compared to AOrchestra, Orchestra-o1 further achieves faster inference and better cost-effectiveness, benefiting from its parallelizable orchestration design. Moreover, when trained with DA-GRPO, Orchestra-o1-8B consistently outperforms existing open-source omnimodal agents by a large margin.

### 2. Related Work

##### 2.1. LLM-based Agent Orchestration

Recent advances in LLM-based agents have shifted from single-agent reasoning systems to multiagent orchestration frameworks. Early efforts primarily focus on enhancing tool use and planning capabilities within a single agent [Schick et al., 2023, Yao et al., 2022], where the model iteratively interacts with external tools to solve complex tasks. More recently, multi-agent systems have emerged as a promising direction, where a central orchestrator coordinates multiple specialized agents to improve scalability and task decomposition. Representative works such as AutoGen-style systems [Wu et al., 2024b] and agent swarms [Team et al., 2026] demonstrate that dividing responsibilities across agents can significantly improve performance on complex reasoning and interactive tasks. However, existing orchestration frameworks are mostly designed for text-based or limited vision-language settings [Ruan et al., 2026, Zhang et al., 2026], and often rely on linear or heuristic-driven workflows.

In contrast, real-world tasks require more flexible coordination strategies that can dynamically adapt agent roles, parallelize execution, and integrate heterogeneous tools. Our work differs from prior studies by focusing on a unified orchestration framework that supports modality-aware decomposition and scalable multi-agent collaboration in omnimodal environments.

##### 2.2. Omnimodal Agent Intelligence

Omnimodal intelligence extends traditional vision-language or audio-language systems to handle heterogeneous inputs such as text, image, audio, and video within a unified framework. Early multimodal models mainly focus on bimodal settings, such as vision-language understanding [Li et al., 2023, Liu et al., 2023], which demonstrate strong capabilities in aligning visual and textual representations. With the development of large-scale multimodal models, recent works have begun exploring omnimodal agents [AI et al., 2025, Google, 2025d, Team et al., 2025a, Team, 2026]. These models aim to unify perception and reasoning across multiple modalities, enabling more general interaction capabilities. However, their performance remains limited in complex agentic scenarios that require long-horizon reasoning, tool use, and multi-step decision-making. To address these limitations, recent approaches introduce external tool augmentation or modular decomposition to improve omnimodal reasoning [Li et al., 2026]. Nevertheless, these methods often lack systematic orchestration mechanisms for coordinating multiple specialized components. In contrast, our work focuses on an explicit omnimodal agent orchestration paradigm, where perception, reasoning, and action are decoupled and coordinated through a structured multi-agent system, enabling more scalable and efficient omnimodal intelligence.

### 3. Methodology

In this section, we first review the background of agent orchestration and introduce the necessary preliminaries (Section 3.1). We then present our proposed omnimodal agent orchestration framework, Orchestra-o1 (Section 3.2), followed by the training recipe for deriving an open-source main agent within the framework (Section 3.3).

##### 3.1. Preliminary

Problem Definition. We formulate omnimodal agent orchestration as a multi-round decision-making problem over heterogeneous inputs. Given a task instance 𝑥 = (𝑞, M), where 𝑞 denotes the naturallanguage question and M = {𝑚𝑖}𝑖𝑁=1 denotes a set of auxiliary modality inputs such as images, audios, and videos. The goal is to produce a concise final answer 𝑎ˆ that maximizes the task reward 𝑅(𝑎,ˆ 𝑎∗) with respect to the ground-truth answer 𝑎∗.

System Formulation. An orchestration-based agent system consists of a main agent, a set of subagent backends, and a tool ecosystem. The main agent 𝜋𝜃 acts as an orchestrator rather than directly operating on every modality. At orchestration round 𝑡, it observes a state:

𝑠𝑡 = 𝑞, M, 𝑐𝑡, 𝐻𝑡, B, T , (1)

where 𝑐𝑡 is the accumulated context, 𝐻𝑡 is the structured sub-task history, B is the set of available subagent models, and T is the set of tools available to sub-agents. The main agent outputs a structured decision 𝑦𝑡 from two action types: 𝑦𝑡 ∈ {delegate, complete}. If 𝑦𝑡 = complete, the main agent terminates the trajectory and returns 𝑎ˆ. If 𝑦𝑡 = delegate, it generates a batch of 𝐾𝑡 sub-tasks:

U𝑡 = {𝑢𝑡,𝑗}𝐾𝑗=𝑡1, 𝑢𝑡,𝑗 = (𝐼𝑡,𝑗, 𝐶𝑡,𝑗, 𝑏𝑡,𝑗, T𝑡,𝑗), (2)

[Figure 33]

I need to create these sub-agents...

Based on the above results, the answer is...

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Orchestration

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

State Updation Multi-turn Interaction

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Figure 2 | An overview of the Orchestra-o1 framework.

where 𝐼𝑡,𝑗 is a sub-task instruction, 𝐶𝑡,𝑗 is the context passed from previous rounds, 𝑏𝑡,𝑗 ∈ B is the selected sub-agent backend, and T𝑡,𝑗 ⊆ T is the assigned tool subset. Each sub-task is executed by an independent sub-agent, producing a result tuple 𝑧𝑡,𝑗 that contains its status, answer-like result, summary, and execution trace. The results are summarized and appended to 𝐻𝑡+1, after which the main agent either launches another delegation round or produces the final answer.

This formulation highlights two key requirements for omnimodal orchestration. First, the main agent must make modality-aware decisions: it needs to identify which inputs and tools are relevant before dispatching sub-tasks. Second, it must make dependency-aware scheduling decisions: independent subtasks should be executed in parallel, while dependent sub-tasks should be delayed until prerequisite results become available.

##### 3.2. The Orchestra-o1 Framework

- Figure 2 presents the overall architecture of Orchestra-o1. The framework is designed as a hierarchical policy that factorizes complex omnimodal problem solving into high-level orchestration and low-level

specialized execution. Let B = {𝑏ℓ}ℓ𝐿=1 denote the candidate sub-agent backends and T = Tperc ∪ Tact represent the unified tool set, respectively. In Orchestra-o1, the perception tool set Tperc consists of

tools for image analysis, audio analysis, and video analysis. The action tool set Tact contains tools for web search, page visit, and code execution. At round 𝑡, the main agent implements a stochastic orchestration policy:

𝑦𝑡 ∼ 𝜋𝜃(· | 𝑠𝑡), 𝑦𝑡 = (𝑎𝑡, 𝜉𝑡), 𝑎𝑡 ∈ {delegate, complete}, (3)

where 𝑎𝑡 is the high-level action and 𝜉𝑡 denotes its structured arguments. The system-level trajectory is therefore 𝜏 = 𝑠1, 𝑦1, 𝑍1, 𝑠2, 𝑦2, 𝑍2, . . . , 𝑠𝑇, 𝑦𝑇 , where 𝑍𝑡 is the set of sub-agent results returned after a delegation action. The objective of Orchestra-o1 is to maximize expected task utility under latency and monetary budgets:

max

𝜃 𝑅(𝑎,ˆ 𝑎∗) − 𝜆𝑐 Cost(𝜏) − 𝜆𝑙 Latency(𝜏) . (4)

𝔼𝜏∼𝜋

𝜋𝜃

Flexible Agentic Backends. Orchestra-o1 supports heterogeneous model backends for both the main agent and sub-agents. Each backend 𝑏 ∈ B is represented by a skill vector and a cost-latency profile

𝜙(𝑏) = 𝜙𝑏txt, 𝜙𝑏img, 𝜙𝑏aud, 𝜙𝑏vid, 𝜙𝑏code, 𝜅𝑏, 𝛿𝑏 , where the first five terms encode capability scores, while 𝜅𝑏 and 𝛿𝑏 denote unit cost and expected latency. For a candidate sub-task 𝑢, the main agent predicts

a requirement vector 𝑟(𝑢) = 𝑟𝑢txt, 𝑟𝑢img, 𝑟𝑢aud, 𝑟𝑢vid, 𝑟𝑢code ∈ [0, 1]. The model assignment can be viewed as maximizing a cost-aware matching score:

𝑏∗(𝑢) = argmax

−𝜆𝑐𝜅𝑏 ℓ(𝑢) − 𝜆𝑙𝛿𝑏, (5)

⟨𝑟(𝑢), 𝜙𝑏⟩

𝑏∈B

capability match

where ℓ(𝑢) is the estimated token or step length of the sub-task. This formulation captures the practical backend selection strategy in Orchestra-o1: easy extraction and search sub-tasks are routed to cheaper models, while difficult omnimodal reasoning sub-tasks are routed to stronger backends.

Unified Omnimodal Tool Ecosystem. The tool assignment problem is also formulated as requirement matching. Let each tool 𝑔 ∈ T have a capability vector 𝜓(𝑔) = 𝜓txt𝑔 , 𝜓img𝑔 , 𝜓aud𝑔 , 𝜓vid𝑔 , 𝜓web𝑔 , 𝜓code𝑔 ∈ {0, 1}. For a sub-task 𝑢, the selected tool subset is:

T∗(𝑢) = {𝑔 ∈ T : 𝑝𝜃(𝑔 | 𝑢, 𝑠𝑡) > 𝛾}, (6) or equivalently the solution of a sparse coverage objective:

###### ⟨𝑟𝑇(𝑢), ∑︁ 𝑔∈S

𝜓(𝑔)⟩ − 𝜆𝑠|S| , (7)

T∗(𝑢) = argmax

S⊆T

where 𝑟𝑇(𝑢) denotes the tool-side requirement vector. In this view, image, audio, and video tools supply modality evidence, while web search, page visit, and code execution tools supply external knowledge and computation. The complete low-level executor for sub-task 𝑢 is thus:

𝑒(𝑢) = 𝑏∗(𝑢), T∗(𝑢) . (8)

Modality-aware Task Decomposition. At each round, the main agent first induces a latent dependency graph over unsolved sub-goals G𝑡 = (V𝑡, E𝑡) and V𝑡 = {𝑣𝑡,1, . . . , 𝑣𝑡,𝑛𝑡}, where a directed edge (𝑣𝑖, 𝑣𝑗) ∈ E𝑡 means that 𝑣𝑗 depends on the result of 𝑣𝑖. Each node is associated with a modality mask 𝜇(𝑣) ∈ {0, 1} and a tool mask 𝛼(𝑣) ∈ {0, 1}|T|, where 𝜇(𝑣) indicates whether text, image, audio, or video evidence is required, and 𝛼(𝑣) indicates candidate tools. A sub-goal is executable if all of its predecessors have already been completed. Therefore, the ready set at round 𝑡 is:

R𝑡 = {𝑣 ∈ V𝑡 \ C𝑡 : Pred(𝑣) ⊆ C𝑡}, (9) where C𝑡 denotes completed sub-goals. The main agent selects a parallel batch from this ready set:

∑︁

𝑈𝜃(𝑣 | 𝑠𝑡) s.t. |P| ≤ 𝐾max, ∑︁ 𝑣∈P

P𝑡 = argmax

cost(𝑣) ≤ 𝐵𝑡. (10)

P⊆R𝑡

𝑣∈P

For each selected node 𝑣𝑡,𝑗 ∈ P𝑡, Orchestra-o1 materializes a concrete sub-task:

𝑢𝑡,𝑗 = Γ𝜃(𝑣𝑡,𝑗, 𝑠𝑡) = 𝐼𝑡,𝑗, 𝐶𝑡,𝑗, 𝑏𝑡,𝑗, T𝑡,𝑗 , (11) where 𝑏𝑡,𝑗 = 𝑏∗(𝑢𝑡,𝑗) and T𝑡,𝑗 = T∗(𝑢𝑡,𝑗). The delegated action is therefore a structured batch decision: 𝑦𝑡 = delegate(U𝑡), U𝑡 = {𝑢𝑡,𝑗}𝐾𝑗=𝑡1, 𝐾𝑡 = |P𝑡|. (12)

This mathematical formulation makes the decomposition strategy explicit: Orchestra-o1 does not only split the task into text strings, but also predicts dependency structure, modality requirements, tool requirements, and backend assignments.

Parallel Sub-task Execution. Each delegated sub-task is executed by an independent ReAct-style [Yao et al., 2022] sub-agent. For sub-task 𝑢𝑡,𝑗, the sub-agent trajectory is:

𝜁𝑡,𝑗 = (𝜌𝑡,𝑗(ℓ), 𝑎𝑡,𝑗(ℓ), 𝑜𝑡,𝑗(ℓ)) ℓ 𝐿=𝑡,𝑗1, 𝑎𝑡,𝑗(ℓ) ∈ T𝑡,𝑗 ∪ {Finish}, (13)

where 𝜌𝑡,𝑗(ℓ) is the reasoning state, 𝑎𝑡,𝑗(ℓ) is the selected tool action, and 𝑜𝑡,𝑗(ℓ) is the observation. The final sub-agent output is summarized as:

𝑧𝑡,𝑗 = Ω(𝜁𝑡,𝑗) = (𝜎𝑡,𝑗, 𝜂𝑡,𝑗, 𝜔𝑡,𝑗, 𝑐𝑡,𝑗, 𝛿𝑡,𝑗), (14)

where 𝜎𝑡,𝑗 is the execution status, 𝜂𝑡,𝑗 is the answer-like result, 𝜔𝑡,𝑗 is a compact trace summary, 𝑐𝑡,𝑗 is the cost, and 𝛿𝑡,𝑗 is the latency. Since all sub-tasks in U𝑡 are conditionally independent given 𝑠𝑡 and do not share mutable environment states, their execution factorizes as:

𝐾𝑡

𝑝(𝑧𝑡,𝑗 | 𝑢𝑡,𝑗, 𝑠𝑡), 𝑍𝑡 = AsyncExecute(𝑢𝑡,1, . . . , 𝑢𝑡,𝐾𝑡). (15)

𝑝(𝑍𝑡 | U𝑡, 𝑠𝑡) =

𝑗=1

This factorization yields a formal latency advantage for parallel orchestration, as stated below. The proof can be found in Section A.1.

- Proposition 1 (Round-level Latency Advantage). Consider an orchestration round 𝑡 with 𝐾𝑡 ≥ 2 ready sub-tasks whose execution times are 𝛿𝑡,1, . . . , 𝛿𝑡,𝐾𝑡 > 0. Assume these sub-tasks are conditionally independent given 𝑠𝑡, do not share mutable environment states during execution, and the only additional

overhead of parallel execution is a nonnegative synchronization cost 𝛿sync𝑡 ≥ 0. If a linear orchestrator executes the sub-tasks sequentially, while Orchestra-o1 launches them asynchronously and waits for all outputs, then we have:

###### ∑︁𝐾𝑡

𝛿𝑡,𝑗 + 𝛿sync𝑡 . (16)

Latencylinear(𝑡) =

𝛿𝑡,𝑗, Latencyparallel(𝑡) = max

1≤ 𝑗≤𝐾𝑡

𝑗=1

Moreover, parallel execution is no slower than linear execution if and only if:

###### ∑︁𝐾𝑡

𝛿sync𝑡 ≤

𝛿𝑡,𝑗 − max

𝛿𝑡,𝑗. (17)

1≤ 𝑗≤𝐾𝑡

𝑗=1

Under this condition, the round-level speedup satisfies:

𝐾𝑡 𝑗=1 𝛿𝑡,𝑗

Latencylinear(𝑡) Latencyparallel(𝑡)

𝑡 𝛿𝑡,𝑗 + 𝛿sync𝑡 ≤ 𝐾𝑡. (18) The upper bound 𝑆𝑡 = 𝐾𝑡 is attainable only when 𝛿sync𝑡 = 0 and all sub-task runtimes are equal.

1 ≤ 𝑆𝑡 =

=

max1≤𝑗≤𝐾

Context Memory and Iterative Refinement. After each delegation round, Orchestra-o1 updates a structured memory that stores the evidence returned by all sub-agents. Let the memory be: 𝐻𝑡 = {ℎ1, . . . , ℎ𝑚𝑡} and ℎ = (𝐼, 𝑏, T, 𝜎, 𝜂, 𝜔), the update after round 𝑡 is:

𝐻𝑡+1 = 𝐻𝑡 ∪ {Summarize(𝑢𝑡,𝑗, 𝑧𝑡,𝑗)}𝐾𝑗=𝑡1. (19)

To keep the main-agent context within the token budget 𝐿ctx, Orchestra-o1 constructs a compressed context by solving:

###### 𝐼(𝐶;𝑞) + ∑︁

𝑤(ℎ)𝐼(𝐶; ℎ) , (20)

𝐶𝑡+1 = argmax 𝐶:|𝐶|≤𝐿ctx

ℎ∈𝐻𝑡+1

where 𝐼(·; ·) denotes information relevance and 𝑤(ℎ) up-weights successful or recently produced evidence. The next orchestration state and budget are:

###### ∑︁𝐾𝑡

𝑐𝑡,𝑗. (21)

𝑠𝑡+1 = 𝑞, M, 𝐶𝑡+1, 𝐻𝑡+1, B, T, 𝐵𝑡+1 , 𝐵𝑡+1 = 𝐵𝑡 −

𝑗=1

The main agent terminates when its evidence sufficiency score exceeds a threshold:

𝑝𝜃stop(𝑠𝑡) = 𝑝𝜃(𝑎𝑡 = complete | 𝑠𝑡), 𝑝𝜃stop(𝑠𝑡) > 𝜏stop. (22)

The final answer is generated from the compressed evidence state 𝑎ˆ = 𝐴𝜃(𝑞, M, 𝐶𝑡, 𝐻𝑡). Otherwise, the main agent refines the dependency graph according to new evidence G𝑡+1 = Refine𝜃(G𝑡, 𝐻𝑡+1), and continues delegation. Overall, Orchestra-o1 differs from linear orchestration frameworks by explicitly modeling omnimodal agent collaboration as a dependency-aware parallel scheduling process with learnable decomposition, model selection, tool selection, evidence aggregation, and stopping decisions.

Theoretical Advantage over Native Omnimodal Agents. We next provide an information-theoretic justification for why agent orchestration can be preferable to a native omnimodal agent design in heterogeneous tasks. The proof can be found in Section A.2.

- Proposition 2 (Information Gain from Omnimodal Orchestration). Let 𝑌 denote the latent task answer and let M = (𝑀1, . . . , 𝑀𝑅) denote 𝑅 modality sources. A native omnimodal agent compresses all modalities into a single internal evidence variable 𝐸0 = 𝑓0(𝑞, M) under a fixed context and computation budget. An orchestration-based system assigns modality-aware sub-tasks to specialized sub-agents and obtains evidence variables 𝐸orch = (𝐸1, . . . , 𝐸𝑅), where 𝐸𝑟 = 𝑓𝑟(𝑞, 𝑀𝑟, 𝐶𝑟) is produced by a backend/tool pair specialized for modality 𝑀𝑟. Suppose that: (i) the main agent aggregates all returned evidence without

losing information relevant to 𝑌; (ii) the native evidence admits modality-wise components (𝐸10, . . . , 𝐸0𝑅) whose joint information upper-bounds the information retained by 𝐸0; and (iii) specialized execution is at least as informative as native processing at every modality step, with a strict gain for at least one modality:

𝐼(𝑌; 𝐸𝑟 | 𝑞, 𝐸<𝑟) ≥ 𝐼(𝑌; 𝐸𝑟0 | 𝑞, 𝐸0<𝑟), 𝑟 = 1, . . . , 𝑅, (23) and the inequality is strict for some 𝑟. Then we have:

𝐼(𝑌; 𝐸orch | 𝑞) > 𝐼(𝑌; 𝐸0 | 𝑞). (24)

Moreover, under Bayes-optimal prediction with log loss, whose minimal risk is Rlog(𝐸) = 𝐻(𝑌 | 𝑞, 𝐸), orchestration has strictly smaller expected risk:

Rlog(𝐸orch) < Rlog(𝐸0). (25)

Framework Summary. In summary, Orchestra-o1 implements omnimodal agent orchestration as a closed-loop decision process that separates high-level planning from specialized perception and action execution. The main agent maintains a structured memory, decomposes the task into dependencyaware sub-goals, selects suitable backends and tools for each sub-task, executes independent sub-tasks in parallel, and iteratively compresses returned evidence until the answer is sufficiently supported. This design makes the system both modular and scalable: new modalities, tools, or sub-agent models can be integrated through the same requirement-matching interface, while the dependency-aware scheduler improves latency whenever multiple independent sub-tasks can be solved concurrently. Algorithm 1 summarizes the overall workflow of Orchestra-o1.

Algorithm 1: Workflow of Orchestra-o1 Input :Question 𝑞, Modality Inputs M, Backend Pool B, Tool Set T, Maximum Rounds 𝑇max Output :Final Answer 𝑎ˆ Initialize 𝐻1 ← ∅, 𝐶1 ← ∅, and 𝑠1 ← (𝑞, M, 𝐶1, 𝐻1, B, T); for 𝑡 = 1, . . . ,𝑇max do

Sample orchestration decision 𝑦𝑡 = (𝑎𝑡, 𝜉𝑡) ∼ 𝜋𝜃(· | 𝑠𝑡); if 𝑎𝑡 = complete then

Generate 𝑎ˆ = 𝐴𝜃(𝑞, M, 𝐶𝑡, 𝐻𝑡); return 𝑎ˆ;

Induce or refine dependency graph G𝑡 = (V𝑡, E𝑡) and compute ready set R𝑡; Select parallel batch P𝑡 ⊆ R𝑡 under dependency and budget constraints; foreach 𝑣𝑡,𝑗 ∈ P𝑡 in parallel do

Materialize sub-task 𝑢𝑡,𝑗 = Γ𝜃(𝑣𝑡,𝑗, 𝑠𝑡); Assign backend 𝑏𝑡,𝑗 = 𝑏∗(𝑢𝑡,𝑗) and tools T𝑡,𝑗 = T∗(𝑢𝑡,𝑗); Execute sub-agent trajectory 𝜁𝑡,𝑗 and summarize 𝑧𝑡,𝑗 = Ω(𝜁𝑡,𝑗);

Update memory 𝐻𝑡+1 ← 𝐻𝑡 ∪ {Summarize(𝑢𝑡,𝑗, 𝑧𝑡,𝑗)}𝐾𝑗=𝑡1; Compress context 𝐶𝑡+1 under 𝐿ctx and update remaining budget; Set 𝑠𝑡+1 ← (𝑞, M, 𝐶𝑡+1, 𝐻𝑡+1, B, T);

Generate fallback answer 𝑎ˆ = 𝐴𝜃(𝑞, M, 𝐶𝑇max, 𝐻𝑇max); return 𝑎ˆ;

##### 3.3. Training Recipe

Although Orchestra-o1 can use strong proprietary models as the main agent, a practical open-source agent system requires an open-source model that can make reliable orchestration decisions. We therefore develop a training recipe for deriving Orchestra-o1-8B from Qwen3-8B [Yang et al., 2025]. The data curation and post-training process are illustrated in Figure 3.

##### 3.3.1. Training Data Curation

A central challenge in training an open-source orchestrator is the lack of diverse omnimodal tasks with reliable answers and explicit evidence chains. We therefore build a seed-based data curation pipeline on top of public datasets such as FineVideo [Farré et al., 2024], LongVideoBench [Wu et al., 2024a], and COCO 2017 [Lin et al., 2014]. Given the original seed set:

D0 = {𝑥𝑖 = (𝑞𝑖, M𝑖, 𝑎𝑖, T𝑖)}𝑖𝑁=1, (26)

where 𝑞𝑖 is the question, M𝑖 denotes the image/audio/video inputs, 𝑎𝑖 is the answer, and T𝑖 is the required tool set, our goal is to create new examples while keeping the original modality files unchanged. Then we use this seed set to collect successful orchestration trajectories under Orchestrao1 (GPT-5 [OpenAI, 2025]) and transform each trajectory into an annotated reasoning solution 𝑟𝑖 with a difficulty level ℓ𝑖.

The curation pipeline contains three stages. First, we extract modality-grounded anchor facts from the annotated solution and evidence sources. For each seed 𝑥𝑖, an LLM extractor produces 𝐴𝑖 = {( 𝑓𝑖,𝑚, 𝜇𝑖,𝑚, 𝑒𝑖,𝑚)}𝑚𝑀=𝑖 1, where 𝑓𝑖,𝑚 is an anchor fact, 𝜇𝑖,𝑚 is its source modality, and 𝑒𝑖,𝑚 records the supporting step in the annotated solution. These anchors identify the non-bypassable perceptual facts that every valid rewrite must preserve. Second, conditioned on (𝑥𝑖, 𝐴𝑖), we generate 𝐾𝑖 candidate

- (a) Training Data Curation 300 Seeds 1.5K Raw Candidates 1.2K Samples

[Figure 70]

[Figure 71]

[Figure 72]

Rollout Decisions Advantages Optimization Updated Policy

Base Model (Qwen3-8B)

Orchestra-o1-8B

- (b) DA-GRPO Training Why DA-GRPO?

###### Seed Data Rollout Anchor Facts QA Rewrites Filter & Verify Training Data

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Trajectories Anchor Facts

[Figure 82]

[Figure 83]

[Figure 84]

Pivot Swapping

Anchor Coverage ≥ 1 Verified Samples Ready for Training

[Figure 85]

FineVideo

Swap Focus Entity/Role

Orchestra-o1 (GPT-5)

[Figure 86]

Text Fact

[Figure 87]

[Figure 88]

[Figure 89]

Temporal Shifting

Similarity ≤ 0. 85

“The camera shows a red bus.”

Shift Time or Order

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Image Fact

Dense Supervision

[Figure 98]

[Figure 99]

[Figure 100]

Numerical Recombination

LongVideoBench

Bypass Test = 0

[Figure 101]

Trajectories

(Object, Color, Position...)

Change Numbers/Units

[Figure 102]

Decision- level Signals

[Figure 103]

Audio Fact

[Figure 104]

[Figure 105]

[Figure 106]

Entity-Sibling Querying

[Figure 107]

Numerical Check = 1

Reasoning Steps

[Figure 108]

(Speaker, Sound, Activity...)

Cross- modal Coverage

Ask about Sibling Entity

[Figure 109]

COCO 2017

[Figure 110]

Video Fact

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Multi-hop Reordering

Difficulty Levels

High Diversity

LLM Judge = 1

(Event, Motion, Timestamp...)

Reorder Reasoning Hops

Cost-efficient (No Live Execution)

Dense, Decisionlevel Supervision

Broad Decision Coverage (Decomposing, Selecting...)

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

###### Reconstructions

Rubric Reward

[Figure 126]

Score each decision on 4 dimensions

Decision Example

[Figure 127]

[Figure 128]

[Figure 129]

- 1

- 2

- 1

- 2

- 1

- 2

[Figure 130]

[Figure 131]

Question & Answer

Format Reward

Key Benefits

###### Optimize with DA- GRPO

Original Question, Golden Answer

Format Correct (0/1)

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Offline Reward

Sub-task History

Action Reward

: Probability Ratio : Clipping Parameter : KL Weight : Current Policy 풓  : Reference Policy

...

...

...

Past Steps, Tools, Results

Action Valid (0/1)

[Figure 141]

[Figure 142]

[Figure 143]

Expert Decision

Tool Reward

Dense Feedback

Expert Action (Label)

Tool Reasonable (0-3)

Sample G candidate decisions for state

Compute relative advantages within each state

[Figure 144]

[Figure 145]

[Figure 146]

Current State

Decision Reward

Better Orchestration

Environment, Context

Decision Quality (0-3)

- Figure 3 | An overview of our training recipe, including (a) data curation pipeline and (b) DA-GRPO training process.

rewrites using five strategy families: pivot swapping, temporal shifting, numerical recombination, entity-sibling querying, and multi-hop reordering. Formally, each candidate is sampled as:

𝑥˜𝑖,𝑘 ∼ 𝐺𝜔(· | 𝑥𝑖, 𝐴𝑖, 𝑧𝑖,𝑘), 𝑧𝑖,𝑘 ∈ {A, B, C, D, E}, (27)

subject to the invariants M˜𝑖,𝑘 = M𝑖, ℓ˜𝑖,𝑘 = ℓ𝑖, T˜𝑖,𝑘 ⊆ T𝑖, and |˜𝑠𝑖,𝑘 − 𝑠𝑖| ≤ 2, where ˜𝑠𝑖,𝑘 and 𝑠𝑖 denote the total reasoning steps of the rewritten and original examples. In practice, easy seeds mainly use pivot swapping and entity-sibling querying, medium seeds additionally use temporal or numerical variants, and hard seeds emphasize numerical recombination and multi-hop reordering. Third, we verify each candidate with a cascade of quality gates. Let 𝑉(𝑥˜𝑖,𝑘) ∈ {0, 1} denote the final verification decision. We define:

𝑉(𝑥˜𝑖,𝑘) = 𝕀{AnchorCov(𝑞˜𝑖,𝑘, 𝐴𝑖) ≥ 1} · 𝕀{Sim(𝑞𝑖, 𝑞˜𝑖,𝑘) ≤ 0.85} · 𝕀{Bypass(𝑞˜𝑖,𝑘, 𝑎˜𝑖,𝑘) = 0}

(28)

· 𝕀{NumCheck(˜𝑟𝑖,𝑘, 𝑎˜𝑖,𝑘) = 1} · 𝕀{Judge(𝑥˜𝑖,𝑘) = 1}.

The first two gates enforce anchor coverage and remove near-duplicates by normalized lexical similarity. The third gate performs a modal-bypass test by asking a strong language model to answer without access to M𝑖: if the answer can still be recovered, the candidate is rejected. The fourth gate executes numerical solutions in a restricted Python sandbox when code execution or numeric answers are involved. The last gate uses an LLM judge to check factual consistency, difficulty preservation, and peer-level duplication among rewrites from the same seed. All LLM-based processors in our curation pipeline, including the anchor extractor, question rewriter, and verification judge, are implemented with Claude-Opus-4.6 [Anthropic, 2026].

Finally, all verified rewrites are merged to form the task set D = {𝑥˜𝑖,𝑘 : 𝑉(𝑥˜𝑖,𝑘) = 1}. Our implementation extracts valid anchors for 300 seeds, generates about 1500 raw rewrite candidates, and

retains around 1200 verified examples after filtering. For a trajectory 𝜏 = (𝑠𝑡, 𝑦𝑡∗, 𝑍𝑡) 𝑡 𝑁=1, we create 𝑁 decision-level examples, where 𝑠𝑡 reconstructs the exact main-agent state before the expert decision

and 𝑦𝑡∗ stores the reference orchestration action. This gives dense supervision for delegation, tool assignment, backend selection, parallel scheduling, and stopping decisions.

##### 3.3.2. Decision-aligned Group Relative Policy Optimization

We propose decision-aligned group relative policy optimization (DA-GRPO), a GRPO-style training objective tailored for main-agent orchestration. Standard GRPO [Team, 2025a] samples a group of responses for the same prompt and normalizes rewards within the group to form relative advantages. However, for agent orchestration, final-answer reward is sparse and expensive because it requires executing the whole multi-agent system. DA-GRPO instead evaluates each sampled main-agent decision directly at the current orchestration state, using expert trajectories and a rubric reward that measures whether the decision is well-formed, valid, tool-aware, and strategically useful.

For each prompt 𝑠𝑖, the policy samples a group of 𝐺 candidate decisions {𝑦𝑖,𝑗}𝐺𝑗=1. Each decision is scored by a multi-dimensional reward:

𝑟𝑖,𝑗 = 𝛼1 𝑟𝑖,𝑗format + 𝛼2 𝑟𝑖,𝑗action + 𝛼3 𝑟𝑖,𝑗tool + 𝛼4 𝑟𝑖,𝑗decision, (29) where 𝑟format measures whether the output is a valid JSON decision, 𝑟action measures whether the action is valid with appropriate parameters, 𝑟tool measures whether the selected tools and sub-task assignments are reasonable, and 𝑟decision measures the overall orchestration decision quality. The first two dimensions are binary, while the latter two are graded and normalized to [0, 1]. In our implementation, Claude-Haiku-4.5 [Anthropic, 2025] serves as a lightweight reward model and scores all four dimensions in a single call. The judge is given the current question, ground-truth answer, sub-task history, expert decision, and model output. Importantly, the expert decision is used as a reference rather than the only correct answer: alternative decompositions are rewarded if they are reasonable and likely to solve the task, while a complete decision receives the highest decision-quality score when its answer matches the ground truth. The coefficients for each reward term are empirically set as 𝛼1 = 𝛼2 = 0.1, 𝛼3 = 0.2, and 𝛼4 = 0.6, prioritizing the tool reward and decision reward, since the two format-related rewards exhibit relatively good initial values. Given group rewards, DA-GRPO computes the relative advantage of each sampled decision by normalizing within the group:

𝑟𝑖,𝑗 − Mean({𝑟𝑖,𝑘}𝐺𝑘=1) Std({𝑟𝑖,𝑘}𝐺𝑘=1) + 𝜖

𝐴ˆ𝑖,𝑗 =

. (30)

The policy is then optimized with a clipped policy-gradient objective and a KL regularizer to the reference model:

LDA-GRPO(𝜃) = −𝔼𝑖,𝑗 min 𝜌𝑖,𝑗(𝜃)𝐴ˆ𝑖,𝑗, clip(𝜌𝑖,𝑗(𝜃), 1 − 𝜖, 1 + 𝜖)𝐴ˆ𝑖,𝑗 − 𝛽 𝐷KL 𝜋𝜃(·|𝑠𝑖) ∥ 𝜋ref(·|𝑠𝑖) , (31)

where 𝜌𝑖,𝑗(𝜃) = 𝜋𝜃(𝑦𝑖,𝑗|𝑠𝑖)/𝜋old(𝑦𝑖,𝑗|𝑠𝑖) and 𝜋ref is the reference model. This objective encourages the open-source main agent to prefer decisions that are not only syntactically valid but also strategically aligned with successful orchestration behavior. Compared with outcome-only reinforcement learning, DA-GRPO offers two advantages. First, it avoids repeatedly executing expensive sub-agent trajectories during training, since each decision can be scored offline from the reconstructed state. Second, it provides dense feedback on the main agent’s core responsibilities: decomposing tasks, selecting tools, scheduling parallel sub-tasks, and deciding when evidence is sufficient for final answering. We train Orchestra-o1-8B with this recipe and deploy it as the main agent in Orchestra-o1.

### 4. Experiments

##### 4.1. Experimental Setup

Benchmark and Baselines. We evaluate all methods on OmniGAIA [Li et al., 2026], a challenging omnimodal agent benchmark that covers heterogeneous inputs including text, image, audio, and

Table 1 | Category-wise accuracy (%) on OmniGAIA. The non-orchestration-based models are implemented under the standard ReAct framework. The highest value in each category within each model group is highlighted in bold.

Category-Wise Breakdown

Method

Overall Geo. Tech. Hist. Fin. Sport Art Movie Sci. Food

Open-Source Agentic Models

Qwen2.5-Omni-3B 0.0 2.0 4.5 0.0 0.0 0.0 0.0 3.9 0.0 1.4 Qwen2.5-Omni-7B 1.5 4.1 7.5 4.0 0.0 2.8 0.0 7.7 5.6 3.6 Baichuan-Omni-1.5-8B 2.9 4.1 3.0 4.0 2.7 0.0 3.0 3.8 0.0 2.8 MiniCPM-O-2.6-8B 2.9 2.0 1.5 0.0 2.7 8.3 3.0 3.8 5.6 3.1 Ming-Lite-Omni-1.5-20B-A3B 2.9 6.1 1.5 4.0 5.4 2.8 6.1 7.7 5.6 3.9 Qwen3-Omni-30B-A3B 8.7 14.3 11.9 28.0 10.8 13.9 9.1 15.4 22.2 13.3 Ming-Flash-Omni-100B-A6B 5.8 8.2 10.4 12.0 8.1 5.6 6.1 11.5 11.1 8.3 LongCat-Flash-Omni-560B-A27B 8.7 10.2 16.4 12.0 10.8 8.3 6.1 11.5 16.7 11.1 OmniAtlas-Qwen2.5-3B 4.4 12.2 16.7 4.0 16.2 11.1 3.0 11.5 11.1 10.3 OmniAtlas-Qwen2.5-7B 8.7 18.4 16.4 4.0 16.2 22.2 3.0 7.7 22.2 13.3 OmniAtlas-Qwen3-30B-A3B 10.1 30.6 29.9 32.0 18.9 16.7 12.1 11.5 27.8 20.8 Orchestra-o1-8B (Ours) 21.7 32.7 37.9 12.0 29.7 16.7 45.5 38.5 38.9 30.0

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

[Figure 157]

[Figure 158]

###### Proprietary Agentic Models

Gemini-2.5-Flash-Lite 5.8 8.2 14.9 4.0 10.8 8.3 6.1 3.9 11.1 8.6 Gemini-2.5-Pro 23.2 28.6 32.8 20.0 32.4 41.7 42.4 26.9 33.3 30.8 Gemini-3-Flash 50.7 57.1 44.8 48.0 59.5 55.6 54.6 38.5 61.1 51.7 Gemini-3-Pro 65.2 59.2 62.1 72.0 78.4 52.8 48.5 42.3 88.9 62.5 AOrchestra-GPT-5 34.8 40.8 56.1 32.0 51.4 25.0 42.4 30.8 22.2 40.0 Orchestra-o1-GPT-5 (Ours) 72.5 69.4 75.8 64.0 83.8 63.9 69.7 73.1 83.3 72.8

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

video. Each task requires a concise final answer and is associated with a difficulty level (Easy, Medium, or Hard) and a topical category. We report accuracy as the primary metric. For detailed analysis, we additionally break down the results by category and by difficulty level. We compare Orchestra-o1 with three groups of baselines. First, we evaluate native open-source omnimodal models, including Qwen2.5-Omni [Team, 2025b], Baichuan-Omni [Li et al., 2025], MiniCPM-O [Yao et al., 2024], Ming-Lite-Omni [AI, 2025], Qwen3-Omni [Team, 2025c], Ming-Flash-Omni [AI et al., 2025], LongCatFlash-Omni [Team et al., 2025a], and OmniAtlas [Li et al., 2026] variants. Second, we compare with proprietary omnimodal models, including Gemini-2.5 [Google, 2025a,b] and Gemini-3 [Google, 2025c,d] variants. Third, we compare with orchestration-based agent systems, especially AOrchestra [Ruan et al., 2026], which is the strongest open-source orchestration baseline in our experiments. Non-orchestration-based models are implemented under a standard ReAct-style agent framework.

Implementation Details. For the proprietary setting, we use GPT-5 as the main agent of Orchestrao1. For the open-source setting, we train Orchestra-o1-8B from Qwen3-8B [Yang et al., 2025] and deploy it as the main agent. The maximum number of main-agent orchestration attempts is set to 10. All sub-tasks within the same delegation call are executed asynchronously by separate ReAct-style sub-agents with cloned environments, and each sub-agent can use the assigned subset of tools. The maximum step for sub-agents is set to 30. The tool ecosystem contains six tools: image analysis, audio analysis, video analysis, web search, page visit, and code execution. For the open-source training experiments, DA-GRPO is trained on a single node with 8 × H20 GPUs. We use a train batch size of 24, rollout group size of 8, learning rate 5 × 10−6, KL coefficient 0.01, and cosine learning-rate decay. The maximums of prompt length and response length are set to 24,576 and 4,096, respectively. The training process is stopped after 5 epochs. The reward is a weighted sum of format correctness, action validity, tool reasonableness, and decision quality, with weights 0.1, 0.1, 0.2, and 0.6, respectively.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- Figure 4 | Difficulty-level comparison among open-source and proprietary agentic models on OmniGAIA. The non-orchestration-based models are implemented under the standard ReAct framework.

[Figure 171]

[Figure 172]

- Figure 5 | Efficiency Comparison between Orchestra-o1 and AOrchestra.

##### 4.2. Main Results

Category-wise Comparison. Table 1 reports category-wise accuracy on OmniGAIA. Orchestra-o1GPT-5 achieves the best overall accuracy of 72.8%, outperforming the strongest native proprietary model, Gemini-3-Pro, by 10.3% absolute accuracy and outperforming AOrchestra-GPT-5 by 32.8% absolute accuracy. The improvement is consistent across most categories. In particular, Orchestra-

- o1-GPT-5 obtains strong gains in geography, technology, history, sport, art, movie, and science, showing that explicit omnimodal orchestration is broadly useful rather than being specialized to a single topic domain. The open-source results further demonstrate the effectiveness of our training recipe. Orchestra-o1-8B achieves 30.0% overall accuracy, substantially improving over the strongest
- open-source baseline OmniAtlas-Qwen-3-30B-A3B (20.8%), despite using a smaller 8B main-agent backbone. The gains are especially large in categories that benefit from structured evidence gathering and tool use, such as geography, history, movie, science, and food. These results suggest that a compact language model can become a competitive omnimodal orchestrator when trained with DA-GRPO.

Difficulty-level Comparison. Figure 4 compares methods under easy, medium, and hard difficulty levels. Across all difficulty groups, Orchestra-o1 consistently ranks first among methods in the corresponding model family. In the proprietary setting, Orchestra-o1-GPT-5 reaches 80.3%, 75.0%, and 56.4% accuracy on easy, medium, and hard tasks, respectively. Compared with AOrchestra-GPT-

- 5, the gains are 35.2%, 35.0%, and 24.3% absolute accuracy. The improvement on hard tasks is particularly important because these tasks usually require multi-step reasoning over several pieces of heterogeneous evidence. The result indicates that dependency-aware decomposition and iterative evidence aggregation help the main agent avoid premature answering and better exploit specialized sub-agents. In the open-source setting, Orchestra-o1-8B also improves the previous best results across all difficulty levels, reaching 36.1% on easy, 26.9% on medium, and 26.9% on hard tasks. The relatively strong hard-task performance shows that DA-GRPO does not merely teach surface-level JSON formatting; instead, it improves the strategic quality of orchestration decisions, such as when to delegate, which tools to assign, and when to stop.

Efficiency Analysis. Figure 5 compares the efficiency of Orchestra-o1 and AOrchestra with GPT-5 as the main agent. Orchestra-o1 achieves higher accuracy while using lower cost across easy, medium, hard, and overall splits. Overall, Orchestra-o1 reaches 72.8% accuracy with a cost of 341.6, while AOrchestra obtains 40.0% accuracy with a cost of 565.7. This means that Orchestra-o1 is not only more accurate but also more cost-effective. The efficiency advantage comes from two design choices. First, Orchestra-o1 executes independent sub-tasks in parallel within a single orchestration round, reducing latency compared with linear sub-agent workflows. Second, the main agent explicitly selects tools and sub-agent backends for each sub-task, which prevents unnecessary use of expensive or irrelevant capabilities. The observed cost-accuracy trade-off is consistent with Proposition 1: when several independent perception or information-seeking sub-tasks can be executed simultaneously, parallel orchestration reduces the effective round-level latency and improves resource utilization.

4.3. Ablation Study

[Figure 173]

Figure 6 | Ablation on the agent harness design.

Ablation on Agent Harness. Figure

- 6 studies whether the gains come from the orchestration framework rather than only from the GPT-5 backend. We compare a standard ReActGPT-5 agent with Orchestra-o1-GPT-5 under the same perception and action tools. Orchestra-o1 improves the overall accuracy from 53.9% to 72.8%, with consistent gains in all categories. The largest gains appear in categories such as art, food, geography, science, movie, and sport, where tasks often require specialized omnimodal perception or external information retrieval before final reasoning. This confirms that the proposed harness design, including task decomposition and sub-agent specialization, provides substantial benefits beyond a strong single-agent ReAct loop.

Table 2 | Ablation on the post-training recipe.

Ablation on Post-training Recipe. Table 2 evaluates the contribution of the posttraining recipe for Qwen3-8B. A direct ReAct-style Qwen3-8B agent achieves only 12.5% accuracy. Simply placing the same model into the Orchestra-o1 framework without post-training improves accuracy to 26.3%, showing that the orchestration scaffold itself provides a strong inductive bias. Supervised fine-tuning (SFT) further improves per-

Framework Model Tools Post-training Accuracy (%)

ReAct Qwen3-8B Omni None 12.5 Orchestra-o1 Qwen3-8B Omni None 26.3 Orchestra-o1 Qwen3-8B Omni SFT 28.6 Orchestra-o1 Qwen3-8B Omni Vanilla GRPO 27.7 Orchestra-o1 Qwen3-8B Omni DA-GRPO 30.0

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

###### Question

###### Annotated Solution

Round 1: Task Decomposition

###### Sub-task 2 (Image Analysis)

###### Sub-task 1 (Audio Analysis)

[Figure 179]

[Figure 180]

Using the provided audio and image, determine the equivalent UTC time for the seasonal event mentioned in the audio. Assume the local time given in the audio corresponds to the location of the landmark in the image.

From the audio, extract autumn equinox on Sep. 23 at 7:49 AM.

[Figure 181]

Extract Event Details from Audio

Identify Landmark and Location from Image

From the image, identify the Prague Astronomical Clock in Prague, Czech Republic.

[Figure 182]

- • Event Type
- • Local Date
- • Local Time

- • Landmark
- • Location
- • Time Zone

Determine the location’s timezone: Europe/Prague.

[Figure 183]

Observation (Round 1)

[Figure 184]

Audio indicates the equinox occurs on Sep. 23 at 7:43 AM. Image identifies the Prague Astronomical Clock in Prague, Czech Republic. Timezone is Europe/Prague.

[Figure 185]

[Figure 186]

Note that Prague observes CEST (UTC+2) in late September.

[Figure 187]

###### Round 2: Cross-modal Synthesis

[Figure 188]

Prague on Sep. 23 uses CEST (UTC+2). Convert local time to UTC: 07:49-2 hours = 05:49 UTC

Convert 7:49 AM local time to UTC: 5:49 AM

[Figure 189]

###### Image Audio UTC.

[Figure 190]

[Figure 191]

###### Final Answer: 05:49 UTC

Figure 7 | Case study of Orchestra-o1’s response to a representative sample on OmniGAIA.

formance to 28.6%. Vanilla GRPO [Team, 2025a] reaches 27.7%, which is slightly worse than SFT, suggesting that sparse or weakly aligned reinforcement learning is insufficient for main-agent orchestration. In contrast, DA-GRPO achieves the best accuracy of 30.0%. This validates our design choice of directly rewarding decision-level alignment, tool reasonableness, and strategic orchestration quality.

##### 4.4. Case Study

Figure 7 presents a representative OmniGAIA example. The question provides an audio clip and an image. The audio states that Mabon falls upon the equinox on September 23 at 7:49 AM, while the image depicts the Prague Astronomical Clock in Prague, Czech Republic. Solving the task requires fusing these two independently obtained facts and then applying the timezone conversion for Prague in September, when the city observes CEST (UTC+2). Therefore, the correct UTC time is 5:49 AM. Orchestra-o1-GPT-5 decomposes the task according to evidence dependencies. In its first orchestration round, the main agent launches an audio sub-task to extract the event, date, and local time, and an image sub-task to identify the landmark and timezone. The returned evidence is compact and complementary: the audio sub-agent extracts “equinox on the 23rd of September at 7:49 AM”, and the image sub-agent identifies “Prague Astronomical Clock” with timezone Europe/Prague. The main agent then aggregates these facts and performs the final conversion, producing 05:49 UTC. This example highlights the central advantage of Orchestra-o1: it improves reliability not merely by adding more tool calls, but by coordinating specialized evidence acquisition, maintaining a structured context memory, and delaying the final answer until all necessary evidence has been grounded.

### 5. Conclusion

In this paper, we introduced Orchestra-o1, an omnimodal agent orchestration framework that separates high-level orchestration from low-level tool-augmented action execution. The main agent dynamically decomposes a complex task into dependency-aware sub-tasks, dispatches independent sub-tasks

to specialized sub-agents in parallel, maintains a compact context memory, and decides when the accumulated evidence is sufficient to produce the final answer. We further proposed Orchestra-o18B, an open-source instantiation of the main agent trained with DA-GRPO. By rewarding format correctness, action validity, tool reasonableness, and decision quality, DA-GRPO directly optimizes the strategic behaviors required by orchestration. Comprehensive experiments demonstrate that Orchestra-o1 achieves strong gains over both native omnimodal agents and orchestration baselines. In the proprietary setting, Orchestra-o1 reaches the best overall accuracy while using lower cost than AOrchestra. In the open-source setting, Orchestra-o1-8B substantially improves over strong open-source omnimodal baselines despite using a compact 8B main-agent backbone. These results suggest that omnimodal agent intelligence can be advanced not only by scaling native OLLMs, but also by learning how to coordinate specialized agents, tools, and evidence sources in a principled and efficient manner. In future work, we plan to extend omnimodal agent orchestration to more practical scenarios, such as audio-video collaborative vibe coding and voice-guided computer-use tasks.

### References

Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents. arXiv preprint arXiv:2504.00906, 2025.

Inclusion AI. Ming-omni: A unified multimodal model for perception and generation, 2025. URL

https://arxiv.org/abs/2506.09344.

Inclusion AI, Bowen Ma, Cheng Zou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Chenyu Lian, Dandan Zheng, Fudong Wang, Furong Xu, et al. Ming-flash-omni: A sparse, unified architecture for multimodal perception and generation. arXiv preprint arXiv:2510.24821, 2025.

Anthropic. System card: Claude haiku 4.5, 2025. URL https://www-cdn.anthropic.com/7aa d69bf12627d42234e01ee7c36305dc2f6a970.pdf.

Anthropic. System card: Claude opus 4.6, 2026. URL https://www-cdn.anthropic.com/6a5 fa276ac68b9aeb0c8b6af5fa36326e0e166dd.pdf.

Miquel Farré, Andi Marafioti, Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Finevideo.

https://huggingface.co/datasets/HuggingFaceFV/finevideo, 2024.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, et al. Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748, 2025.

Google. Gemini 2.5 flash-lite model card, 2025a. URL https://storage.googleapis.com/dee pmind-media/Model-Cards/Gemini-2-5-Flash-Lite-Model-Card.pdf.

- Google. Gemini 2.5 pro model card, 2025b. URL https://storage.googleapis.com/deepmin

- d-media/Model-Cards/Gemini-2-5-Pro-Model-Card.pdf.

Google. Gemini 3 flash model card, 2025c. URL https://storage.googleapis.com/deepmin

- d-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf.

- Google. Gemini 3 pro model card, 2025d. URL https://storage.googleapis.com/deepmin d-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf.

Dong Huang, Jie M Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, and Heming Cui. Agentcoder: Multi-agent-based code generation with iterative testing and optimisation. arXiv preprint arXiv:2312.13010, 2023.

Jina. Jina reader, 2025. URL https://jina.ai/reader/. Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-

training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, et al. Omnigaia: Towards native omni-modal ai agents. arXiv preprint arXiv:2602.22897, 2026.

Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, et al. Large language model agent: A survey on methodology, applications and challenges. arXiv preprint arXiv:2503.21460, 2025.

Qianyu Meng, Yanan Wang, Liyi Chen, Qimeng Wang, Chengqiang Lu, Wei Wu, Yan Gao, Yi Wu, and Yao Hu. Agent harness for large language model agents: A survey. 2026.

OpenAI. Gpt-5 system card, 2025. URL https://cdn.openai.com/gpt-5-system-card.pdf. Linyue Pan, Lexiao Zou, Shuo Guo, Jingchen Ni, and Hai-Tao Zheng. Natural-language agent harnesses.

arXiv preprint arXiv:2603.25723, 2026.

Zile Qiao, Guoxin Chen, Xuanzhong Chen, Donglei Yu, Wenbiao Yin, Xinyu Wang, Zhen Zhang, Baixuan Li, Huifeng Yin, Kuan Li, et al. Webresearcher: Unleashing unbounded reasoning capability in long-horizon agents. arXiv preprint arXiv:2509.13309, 2025.

Jianhao Ruan, Zhihao Xu, Yiran Peng, Fashen Ren, Zhaoyang Yu, Xinbing Liang, Jinyu Xiang, Yongru Chen, Bang Liu, Chenglin Wu, et al. Aorchestra: Automating sub-agent creation for agentic orchestration. arXiv preprint arXiv:2602.03786, 2026.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

Serper. Serper api, 2025. URL https://serper.dev/. Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter Belcak,

Hanrong Ye, Hongxu Yin, et al. Toolorchestra: Elevating intelligence via efficient model and tool orchestration. arXiv preprint arXiv:2511.21689, 2025.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, et al. Webshaper: Agentically data synthesizing via informationseeking formalization. arXiv preprint arXiv:2507.15061, 2025.

DeepSeek Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Meituan LongCat Team, Bairui Wang, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, et al. Longcat-flash-omni technical report. arXiv preprint arXiv:2511.00279, 2025a.

- Qwen Team. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025b. Qwen Team. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025c.
- Qwen Team. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025b.

Bowen Wang, Xinyuan Wang, Jiaqi Deng, Tianbao Xie, Ryan Li, Yanzhe Zhang, Junli Wang, Dunjie Lu, Zicheng Gong, Gavin Li, et al. Computer agent arena: Toward human-centric evaluation and analysis of computer-use agents. In The Fourteenth International Conference on Learning Representations.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37: 28828–28857, 2024a.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First conference on language modeling, 2024b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

Xinmiao Yu, Liwen Zhang, Xiaocheng Feng, Yong Jiang, Bing Qin, Pengjun Xie, and Jingren Zhou. Webanchor: Anchoring agent planning to stabilize long-horizon web reasoning. arXiv preprint arXiv:2601.03164, 2026.

Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. arXiv preprint arXiv:2410.02506, 2024a.

Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13643–13658, 2024b.

Mingda Zhang, Haoran Luo, Tiesunlong Shen, Qika Lin, Xiaoying Tang, Rui Mao, and Erik Cambria. Flowsteer: Interactive agentic workflow orchestration via end-to-end reinforcement learning. arXiv preprint arXiv:2602.01664, 2026.

### A. Proof of Theorems

##### A.1. Proof of Proposition 1

Proof of Proposition 1. For a linear orchestrator, the sub-tasks are executed one after another, hence the total round latency is the sum of their runtimes 𝐾𝑗=𝑡1 𝛿𝑡,𝑗. For Orchestra-o1, conditional independence and the absence of shared mutable states allow all ready sub-tasks to be launched simultaneously. The round completes after the slowest sub-task finishes and the main agent aggregates the returned results, giving max1≤𝑗≤𝐾

𝑡 𝛿𝑡,𝑗 + 𝛿sync𝑡 . Parallel execution is no slower than linear execution exactly when:

∑︁𝐾𝑡

𝛿𝑡,𝑗 + 𝛿sync𝑡 ≤

max

𝛿𝑡,𝑗, (32)

1≤ 𝑗≤𝐾𝑡

𝑗=1

which is equivalent to the stated condition on 𝛿sync𝑡 . When this condition holds, the denominator of 𝑆𝑡 is at most the numerator, so 𝑆𝑡 ≥ 1. Since 𝛿sync𝑡 ≥ 0, we also have:

𝐾𝑡 𝑗=1 𝛿𝑡,𝑗

𝑡 𝛿𝑡,𝑗 ≤ 𝐾𝑡, (33) where the last inequality follows because each 𝛿𝑡,𝑗 ≤ max1≤𝑗≤𝐾

𝑆𝑡 ≤

max1≤𝑗≤𝐾

𝑡 𝛿𝑡,𝑗. Equality requires both 𝛿sync𝑡 = 0 and 𝑗 𝛿𝑡,𝑗 = 𝐾𝑡 max𝑗 𝛿𝑡,𝑗, which holds only when all sub-task runtimes are equal. This proves the proposition. □

###### A.2. Proof of Proposition 3.2 Proof of Proposition 3.2. By the chain rule of mutual information, we have:

###### ∑︁𝑅

𝐼(𝑌; 𝐸orch | 𝑞) = 𝐼(𝑌; 𝐸1, . . . , 𝐸𝑅 | 𝑞) =

𝐼(𝑌; 𝐸𝑟 | 𝑞, 𝐸<𝑟). (34)

𝑟=1

The modality-wise components of the native agent satisfy:

###### ∑︁𝑅

𝐼(𝑌; 𝐸0 | 𝑞) ≤ 𝐼(𝑌; 𝐸10, . . . , 𝐸0𝑅 | 𝑞) =

𝐼(𝑌; 𝐸𝑟0 | 𝑞, 𝐸0<𝑟), (35)

𝑟=1

where the inequality follows because the component tuple is assumed to contain all task-relevant information retained by 𝐸0. By the specialization assumption, every conditional information term

of orchestration is no smaller than its native counterpart, and at least one term is strictly larger. Therefore, we have:

𝐼(𝑌; 𝐸orch | 𝑞) > 𝐼(𝑌; 𝐸10, . . . , 𝐸0𝑅 | 𝑞) ≥ 𝐼(𝑌; 𝐸0 | 𝑞), (36) which proves the strict information gain. For log loss, the Bayes-optimal predictor is the posterior distribution 𝑝(𝑌 | 𝑞, 𝐸) and the minimum achievable expected loss equals the conditional entropy:

Rlog(𝐸) = 𝐻(𝑌 | 𝑞, 𝐸) = 𝐻(𝑌 | 𝑞) − 𝐼(𝑌; 𝐸 | 𝑞). (37)

Since 𝐻(𝑌 | 𝑞) is fixed for the task distribution, the strict inequality 𝐼(𝑌; 𝐸orch | 𝑞) > 𝐼(𝑌; 𝐸0 | 𝑞) implies Rlog(𝐸orch) < Rlog(𝐸0). Thus, when specialized sub-agents provide strictly more task-relevant evidence and the orchestrator preserves it, the multi-agent orchestration system is theoretically preferable to the native single-agent design. □

### B. More Experimental Details

##### B.1. Tool Configurations

The proposed Orchestra-o1 incorporates a unified tool ecosystem shared by all sub-agents. The main agent does not directly call these tools. Instead, it assigns each sub-task a subset of tools, and the corresponding sub-agent interacts with the environment through the assigned tool interface. This design keeps the main agent focused on high-level orchestration while allowing each sub-agent to perform specialized perception or action execution. A brief introduction of the incorporated tools is as follows:

Web Search. This tool performs web search for external information seeking. It is useful when the answer depends on public knowledge, recent facts, entity disambiguation, or contextual information that is not fully contained in the provided modality inputs. We use the Serper API [Serper, 2025] to perform web search.

Page Visit. This tool visits and extracts readable content from a given web page. It complements web search by allowing a sub-agent to inspect candidate sources in more detail. We use it for tasks where a search snippet is insufficient and the sub-agent needs to verify facts from the source page. Web pages are crawled by the Jina Reader API [Jina, 2025].

Code Execution. This tool executes Python code in a controlled workspace. It is primarily used for numerical computation, table processing, date or unit conversion, and other deterministic operations. In our framework, the main agent can delegate a computation sub-task only after prerequisite values have been extracted from media or retrieved from the web.

Image Analysis. This tool analyzes image inputs with a vision-capable backend. It is used for visual recognition, scene understanding, chart interpretation, OCR-like inspection, and extraction of imagegrounded evidence. The main agent is instructed to process relevant images before relying on external search because images often contain task-specific information that cannot be recovered from text alone.

Audio Analysis. This tool transcribes and analyzes standalone audio files such as speech clips. It is used when the task requires spoken content, sound events, or audio-grounded clues. The returned transcription and summary are written into the sub-task history so that later rounds can use them as textual evidence.

Video Analysis. This tool analyzes video inputs by considering visual frames and, when appropriate, the audio track. It is used for temporal reasoning, event recognition, spoken-video understanding, and multimodal evidence extraction. Since video analysis can be expensive, Orchestra-o1 encourages the main agent to formulate specific video-analysis instructions rather than asking for an overly broad description.

##### B.2. System Prompt for Main Agent System Prompt for Main Agent

You are the MainAgent (Orchestrator) for OmniGAIA benchmark tasks. Your role is to analyze the given QUESTION , plan a multi -phase execution strategy , and delegate subtasks to SubAgents , maximizing parallelism where possible while respecting task dependencies.

==== CORE PRINCIPLE: SMART PARALLEL DECOMPOSITION ==== Not all subtasks can run simultaneously. Some depend on others ’

results. Your job is to:

- 1. Identify which subtasks are INDEPENDENT and can run in parallel NOW
- 2. Identify which subtasks DEPEND on others ’ results and must wait for later phases
- 3. In each delegation round , submit ALL currently -runnable independent subtasks together
- 4. After receiving results , plan the NEXT round of subtasks based on what you learned

KEY RULES:

- - Each subtask runs as an independent SubAgent with its own environment
- - All subtasks within ONE delegation call execute simultaneously in parallel
- - Always use the "tasks" list format (even for a single subtask)
- - Each delegation (regardless of how many parallel subtasks) counts as ONE attempt

DECOMPOSITION STRATEGY:

- Phase 1: Identify ALL sub -goals needed to answer the question
- Phase 2: Classify each sub -goal:

- - INDEPENDENT: Can start immediately without any prior results (run in parallel NOW)
- - DEPENDENT: Needs results from other sub -goals first (plan for a LATER round)

- Phase 3: Submit all INDEPENDENT sub -goals as parallel subtasks in this round
- Phase 4: After receiving results , re-evaluate:

- - Are the results sufficient to answer the question? Use ‘complete ’
- - Are there DEPENDENT sub -goals now unblocked? Submit them as the next parallel batch
- - Do results reveal NEW sub -goals? Add them to the plan

DECISION PROCESS:

- 1. REVIEW the SUBTASK HISTORY below - check status , result , and key findings of each attempt
- 2. EVALUATE: Do the results SUFFICIENTLY answer the QUESTION?

- - If any subtask returned a valid result with status "done": Consider using ‘complete ’

- - If subtask status is "incomplete": Review its key findings to see what was accomplished

- 3. PLAN next action:

- - Results sufficient: Use ‘complete ’ with the answer
- - Need more work: Identify what subtasks are NOW unblocked by previous results
- - Subtask FAILED or INCOMPLETE: You can RETRY the failed/incomplete subtask in the next round. Adjust the instruction , context , or

model if needed to improve the chance of success

- - Submit all currently -runnable subtasks in parallel as the next batch (including retries of failed subtasks alongside newly unblocked subtasks)
- - Think ahead: what will you need AFTER this batch? Plan accordingly with your remaining budget

BUDGET AWARENESS:

- - You have LIMITED attempts (see Progress below)
- - Each delegation (regardless of how many parallel subtasks) counts as ONE attempt
- - Maximize parallelism within each round to get the most done per attempt
- - Plan your phases wisely: with N remaining attempts , you can run N rounds of parallel subtasks
- - If a result looks correct and was verified , trust it and complete

==== MODEL SELECTION GUIDE ==== {model_pricing_table}

Note: Higher -priced models are generally more capable. Price

correlates with model strength. Model Selection Strategy:

- - Choose cheaper models for simple tasks (e.g., straightforward web search)
- - Choose more capable models for complex reasoning , video analysis , or multi -step tasks
- - You can assign DIFFERENT models to different parallel subtasks based on their complexity

==== Progress ==== [Attempt {attempt_index }/{ max_attempts}] Remaining {remaining_attempts

} attempts Budget is limited. Maximize parallelism to get the most done per attempt.

==== QUESTION ==== {instruction}

==== SUBTASK HISTORY ==== {subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS (for SubAgents) ==== {tools_description}

==== OUTPUT FORMAT ==== ANSWER FORMAT: requires precise , concise answers (single word , number ,

or short phrase). Do NOT include explanations in the answer field. Return JSON:

If results are SUFFICIENT: {{

"action": "complete", "reasoning": "The subtask results show [X], which answers the

question", "params": {{ "answer": "concise answer" }}

}} If more work is NEEDED , submit all currently -runnable subtasks in

parallel: {{

"action": "delegate_task", "reasoning": "Based on previous results , [X] and [Y] can now run

independently in parallel. [Z] still needs to wait for their results , so I’ll handle it in the next round.",

"params": {{ "tasks": [

{{

"task_instruction": "A SPECIFIC , ACTIONABLE subtask (e.g., ’ Analyze the video to identify the main topic discussed ’)", "context": "Relevant findings from previous attempts that this

subtask can build on", "model": "one of {sub_models}", "tools": ["tool1", "tool2"]

}}, {{

"task_instruction": "Another INDEPENDENT subtask that can run at the same time (e.g., ’Search for background information about X’)",

"context": "Relevant context", "model": "one of {sub_models}", "tools": ["tool3"]

}} ]

}}

}} If only ONE subtask can run right now (others depend on its result): {{

"action": "delegate_task", "reasoning": "I need to first [X] before I can determine [Y]. So

this round only has one subtask.", "params": {{

"tasks": [ {{

"task_instruction": "The prerequisite subtask that must

complete first", "context": "Relevant context", "model": "one of {sub_models}", "tools": ["tool1"]

}} ]

}}

}} IMPORTANT RULES:

- 1. ALWAYS use the "tasks" list format (even for a single subtask)

- 2. Within each round , subtasks must be INDEPENDENT of each other , don ’ t make one subtask depend on another subtask ’s result IN THE SAME ROUND
- 3. Subtasks CAN and SHOULD depend on results from PREVIOUS rounds , pass relevant findings via the "context" field
- 4. Maximize parallelism WITHIN each round: if two things CAN run independently NOW , they SHOULD be parallel subtasks
- 5. Select relevant tools from AVAILABLE TOOLS section for each subtask
- 6. Think in phases: what can I do now in parallel? What must wait for next round?
- 7. If a subtask returns status "failed" or "incomplete", you MAY retry it in the next delegation round. When retrying , consider:

adjusting the task instruction to be more specific , providing additional context from other completed subtasks , or switching to a

more capable model. Retried subtasks can run in parallel with other new subtasks.

##### B.3. System Prompt for Sub-agent System Prompt for Sub-agent

You are a specialized SubAgent. Complete the assigned task efficiently.

==== Progress ==== [Step {current_step }/{ max_steps}] Remaining {remaining_steps} steps {budget_warning}

==== Your Task (from MainAgent) ==== {task_instruction}

==== Context ==== {context}

==== Original Question (for reference) ==== {original_question}

==== Available Tools ==== {action_space}

==== Guidelines ====

- 1. Focus on completing YOUR TASK above
- 2. Think step by step before outputting an action
- 3. Write key observations to the "memory" field
- 4. Use print() in ExecuteCodeAction to see computation results
- 5. Once done , use ’finish ’ IMMEDIATELY
- 6. **IMAGE ANALYSIS RULE:** You may ONLY use ImageAnalysisAction on image URLs that are explicitly provided in your TASK or CONTEXT from the MainAgent. Do NOT use ImageAnalysisAction on any image URLs you encounter during web search or browsing (e.g., thumbnails ,

page images , search result images). These external image URLs are often inaccessible and will waste your steps.

- 7. EFFICIENCY RULE - Avoid Repetitive Attempts:

- - Count your attempts by behavior pattern , not just individual tool names. A "search -then -extract" cycle (e.g., GoogleSearchAction

- ExtractUrlContentAction) counts as ONE search attempt , not two

separate tool uses.

- - If you have performed the same behavior pattern 5 times without finding the target information , STOP immediately. Use ’finish ’ with whatever partial results you have gathered so far.
- - Examples of behavior patterns that count as the SAME attempt: GoogleSearchAction alone (one search attempt) GoogleSearchAction and ExtractUrlContentAction (one search -and -

read attempt) ExtractUrlContentAction alone on different URLs (one URL extraction attempt each)

- - Do NOT keep trying different keyword variants or URLs endlessly. After 5 rounds of the same behavior pattern , you have likely exhausted what can be found.
- - When finishing with partial results , set status to "partial" and clearly describe what you DID find and what you could NOT find. The MainAgent can decide how to proceed.

- 8. **COMPLETENESS vs PERFECTION :** It is better to return partial results quickly than to waste all your steps searching for information that may not exist. The MainAgent can assign follow -up tasks if needed.
- 9. **FORBIDDEN IMAGE SOURCES :** Never attempt ImageAnalysisAction on URLs you discovered through GoogleSearchAction or ExtractUrlContentAction. Only analyze images that were part of the ORIGINAL task assignment.

BUDGET: When remaining_steps <= 5, use ‘finish ’ NOW with your best available results!

EFFICIENCY: After 5 rounds of the same behavior pattern (e.g., repeated search and extract cycles), use ’finish ’ NOW with partial results!

==== Output Format ==== CRITICAL: You MUST reply with ONLY a valid JSON object. No markdown ,

no extra text. The "action" field MUST be one of the exact tool names listed in

Available Tools (e.g., "ImageAnalysisAction "), or "finish". Do NOT use "execute" as the action. Do NOT pass tool names via a "

command" field. The "params" field MUST be a JSON object with the exact parameter names defined for that tool.

‘‘‘json {{

"action": "<EXACT_TOOL_NAME >", "params": {{ <tool -specific parameters as key -value pairs > }}, "memory": "<your key observations >"

}} ’’’

==== Memory ==== {memory}

==== Current Observation ==== {obs}

##### B.4. Prompt for Rubric Rewards Prompt for Rubric Rewards

You are an expert judge evaluating an AI agent ’s output in a multi step task -solving pipeline.

The agent (Main Agent) orchestrates sub -agents to solve complex tasks. At each step , it outputs a JSON decision that either:

- - **delegate_task **: Break the problem into sub -tasks and assign them to sub -agents (each sub -task should have task_instruction , model , and optionally tools)
- - **complete **: Provide the final answer (should have params.answer)

You will evaluate the agent ’s output on 4 dimensions. FORMAT_CORRECT and ACTION_VALID are scored 0 or 1 (binary). TOOL_REASONABLE and DECISION_QUALITY are scored 0-3 (integer only).

## Original Question {question}

## Ground Truth Answer {ground_truth}

## Current Step Context (Subtask History) {subtask_history}

## Expert ’s Decision (reference , NOT the only valid approach)

- - Action: {expert_action}
- - Expert Output: ‘‘‘json {expert_json} ’’’

## Agent ’s Raw Output (to be evaluated) ‘‘‘ {pred_raw} ’’’

## Agent ’s Parsed Decision ‘‘‘json {pred_json} ’’’

## Scoring Dimensions

- ### 1. FORMAT_CORRECT (0 or 1) Is the agent ’s output a valid JSON decision with required fields?

- - 1: Valid JSON with "action" field present and correctly structured
- - 0: Not valid JSON , or missing "action" field , or completely unparseable

- ### 2. ACTION_VALID (0 or 1) Is the chosen action valid and properly parameterized?

- - 1: Action is valid ("delegate_task" or "complete") with "params" field present
- - 0: Action is not in the valid set , or "params" field is missing/ invalid

- ### 3. TOOL_REASONABLE (0-3) Are the tool choices and sub -task assignments reasonable? (For "

complete" action , evaluate whether completing at this point is appropriate)

- - 3: Excellent tool/model selection , sub -tasks are well -scoped and clearly instructed
- - 2: Acceptable tool selection but could be improved (e.g., missing a useful tool , overly broad instructions)
- - 1: Questionable or mostly inappropriate tool choices , poorly defined sub -tasks
- - 0: No tools specified when needed , or completely irrelevant assignments

- ### 4. DECISION_QUALITY (0-3) **Most Important** Overall decision quality: does this decision make good progress toward

solving the problem?

**Key principle: We encourage exploration. The agent does NOT need to copy the expert ’s exact strategy .**

- - 3: Excellent decision - closely aligned with expert ’s approach , OR takes a different but equally valid/creative approach , OR directly provides the correct answer
- - 2: Acceptable decision - reasonable strategy but with notable inefficiencies or differences from optimal
- - 1: Poor decision - partially relevant but unlikely to lead to the correct answer , or fundamentally flawed
- - 0: Completely wrong - irrelevant output , nonsensical , or harmful to solving the task

**When scoring DECISION_QUALITY , consider :**

- - If the agent ’s approach differs from the expert but is still reasonable and could lead to the correct answer: score 2-3
- - If the agent chose "complete" and the answer matches the ground truth: score 3 regardless of expert action
- - If the agent chose "complete" but the answer is wrong when expert says delegate: score 0
- - If the agent chose "delegate_task" with reasonable sub -tasks when expert says complete: score 1-2 (inefficient but not wrong)

## Your Task Evaluate the agent ’s output and provide scores for each dimension.

**IMPORTANT: Output ONLY the 4 scores below. Do NOT include any explanation , analysis , or reasoning. Just the scores.**

FORMAT_CORRECT: <score > ACTION_VALID: <score > TOOL_REASONABLE: <score > DECISION_QUALITY: <score >

### C. Limitations

Although Orchestra-o1 achieves strong omnimodal agentic intelligence, several limitations remain. First, orchestration introduces additional system complexity. Compared with a single native omnimodal agent, Orchestra-o1 requires maintaining sub-agent histories, tool schemas, backend configurations, cost accounting, and asynchronous execution. While this design improves modularity

and efficiency, it also creates more implementation components that must be carefully engineered and monitored. Second, the current training recipe focuses on the main agent rather than jointly optimizing all sub-agents and tools. DA-GRPO improves decision-level orchestration, but the sub-agent backends remain fixed during training. A more complete learning system could jointly adapt the main agent, sub-agent policies, and tool-selection behavior from end-to-end task outcomes.

