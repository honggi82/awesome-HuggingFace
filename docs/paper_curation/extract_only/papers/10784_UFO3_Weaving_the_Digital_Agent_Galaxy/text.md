# arXiv:2511.11332v2[cs.DC]1Mar2026

[Figure 1]

## UFO3: Weaving the Digital Agent Galaxy

###### Chaoyun Zhang1,∗ Liqun Li1, He Huang1, Chiming Ni2†, Bo Qiao1, Si Qin1, Yu Kang1, Minghua Ma1, Qingwei Lin1, Saravan Rajmohan1, Dongmei Zhang1 1Microsoft 2ZJU-UIUC Institute

[Figure 2]

Figure 1: UFO3: Weaving the Digital Agent Galaxy. A single natural-language intent is decomposed into a dynamically evolved Constellation (DAG) executed across heterogeneous devices. Demo video available at: https://www.youtube.com/watch?v=NGrVWGcJL8o.

#### Abstract

Large language model (LLM)-powered agents are transforming digital devices from passive tools into proactive intelligent collaborators. However, most existing frameworks remain confined to a single OS or device, making cross-device workflows brittle and largely manual.

[Figure 3]

We present UFO3 , a system that unifies heterogeneous endpoints, desktops, servers, mobile devices, and edge, into a single orchestration fabric. UFO3 models each user request as a mutable TaskConstellation: a distributed DAG of atomic subtasks (TaskStars) with explicit control and data dependencies (TaskStarLines). The TaskConstellation continuously evolves as results stream in from distributed devices, enabling asynchronous execution, adaptive recovery, and dynamic optimization. A Constellation Orchestrator

∗Chaoyun Zhang is the corresponding author: chaoyun.zhang@microsoft.com †Chiming Ni is with ZJU-UIUC Institute and completed this work while at Microsoft

executes tasks safely and asynchronously while applying dynamic DAG updates, and the Agent Interaction Protocol (AIP) provides persistent, low-latency channels for reliable task dispatch and result streaming. These designs dissolve the traditional boundaries between devices and platforms, allowing agents to collaborate seamlessly and amplify their collective intelligence.

We evaluate UFO3 on NebulaBench, a benchmark of 55 cross-device tasks across 5 machines and 10 categories. UFO3 achieves 83.3% subtask completion, 70.9% task success, exposes parallelism with an average width of 1.72, and reduces end-to-end latency by 31% relative to a sequential baseline. Fault-injection experiments demonstrate graceful degradation and recovery under transient and permanent agent failures. These results show that UFO3 achieves accurate, efficient, and resilient task orchestration across heterogeneous devices, uniting isolated agents into a coherent, adaptive computing fabric that extends across the landscape of ubiquitous computing.

We developed UFO3 as a fully engineered system with over 73K lines of code, encompassing agent implementations and integrations for Windows, Linux, and Android mobile devices. The entire project is open-sourced at https://github.com/microsoft/UFO/, accompanied by detailed documentation and tutorials at https://microsoft.github.io/UFO/

#### 1 Introduction

The rise of intelligent agents Wang et al. (2024b) marks a new era of human–computer interaction, where large language models (LLMs) Naveed et al. (2025) are evolving from text-based reasoning engines Qu et al. (2025) into autonomous digital operators capable of perceiving, acting, and coordinating across tasks Zhang

- et al. (2024). Yet despite this progress, most agent frameworks remain confined within a single device or platform, be it a browser tab Ning et al. (2025); Zheng et al. (2024), a desktop environment Zhang et al. (2025b;a), or a mobile app Zhang et al. (2025d); Wang et al. (2024a). This confinement sharply limits their ability to harness the rich, distributed computational ecosystem that modern users inhabit.

When an agent is trapped within one operating system, it cannot access the complementary strengths of other devices, such as GPU clusters for computation, desktop applications for document editing, or mobile sensors for context capture. The result is a fragmented landscape of intelligent but siloed agents: each powerful in isolation yet collectively underutilized. This gap between reasoning ability and real-world actuation leaves vast potential untapped. To truly advance the next frontier of automation and reasoning, agents must operate beyond the boundaries of any single device or OS, forming a coherent digital collective where Windows laptops, Linux servers, mobile devices, and edge nodes collaborate seamlessly Houben et al. (2017); Brudy et al. (2019) to gather and act upon ubiquitous intelligence.

Imagine a future where you could simply say: “Prepare a production-ready demo of Project X and deliver a one-page executive summary with screenshots and performance numbers.” Today, this requires tedious, error-prone coordination across devices, checking out code on a laptop, triggering GPU builds on a server, deploying to a cloud instance, recording UI interactions on a phone, and stitching results into a report. Despite recent advances in intelligent agents, most systems remain confined within a single device or platform, leaving vast computational resources underutilized.

To realize this vision of seamless cross-device collaboration, we must overcome three interlocking challenges that go beyond classical workflow engines or single-machine agents. First, asynchronous parallelism: many subtasks can and should run concurrently across devices with varying capabilities. Second, distributed coordination: agents need reliable, low-latency communication for task dispatch and result streaming despite network variability. Third, heterogeneous extensibility: the system should make it easy to develop and integrate new device agents while preserving safety and global consistency.

We present UFO3: Weaving the Digital Agent Galaxy, a cross-device orchestration system that turns isolated devices, desktops, servers, mobile, and edge, into a coherent execution fabric. UFO3 models each request as a TaskConstellation: a dynamic distributed DAG whose nodes (TaskStars) represent

executable subtasks and whose edges (TaskStarLines) capture data and control dependencies. The Constellation serves as both the logical plan and live runtime substrate: nodes are assigned asynchronously, executed opportunistically, and continuously updated as results stream. Figure 1 illustrates this concept, where one intent decomposed into a distributed DAG and orchestrated across heterogeneous endpoints.

To realize these capabilities and address the challenges in cross-device agents, UFO3 is built around five tightly integrated design principles:

- • Declarative decomposition into a dynamic DAG (TaskConstellation). Natural-language or programmatic requests are decomposed by the global ConstellationAgent into a structured DAG Bei et al. (2025) of TaskStars and TaskStarLines that encode workflow logic and dependencies. This declarative structure is amenable to automated scheduling, introspection, and rewriting throughout execution.
- • Continuous, result-driven graph evolution. The TaskConstellation is a living data structure. Intermediate outputs, transient failures, and new observations trigger controlled rewrites, adding diagnostic TaskStars, creating fallbacks, rewiring dependencies, or pruning completed nodes, so the system adapts dynamically instead of aborting on errors Wu et al. (2024b).
- • Heterogeneous, asynchronous, and safe orchestration. Each TaskStar is matched to the most suitable device agent via rich AgentProfiles reflecting OS, hardware, and capabilities. The Constellation Orchestrator executes tasks asynchronously, allowing multiple TaskStars to progress in parallel. Safe assignment locking, event-driven scheduling, DAG consistency checks, and batched edits collectively ensure correctness and concurrency safety, achieving high efficiency without compromising reliability. These guarantees are further reinforced through formal verification.
- • Unified Agent Interaction Protocol (AIP). Built atop persistent WebSocket channels, we develop AIP, a protocol that provides a unified, secure, and fault-tolerant layer for agent registry, session management, task dispatch, and coordination. It ensures reliability under network fluctuations through automatic reconnection and retry, while exposing a lightweight, extensible interface that allows new agents to integrate seamlessly into the UFO3 ecosystem Yang et al. (2025a).
- • Template-driven framework for MCP-empowered device agents. To democratize agent creation, UFO3 provides a lightweight development template and toolkit for rapidly building new device agents. Developers can declare capabilities, bind to local environments, and extend them through one or more Model Context Protocol (MCP) servers Hou et al. (2025) for tool augmentation. This modular design accelerates integration while maintaining consistency across the constellation.

Together, these designs enable the system to decompose, schedule, execute, and adapt distributed tasks efficiently while maintaining safety and consistency.

Building on these designs, we implemented the full UFO3 system as a comprehensive distributed system implementation with over 73K lines of Python code. The implementation integrates all major components, the centralized ConstellationAgent, the asynchronous Constellation Orchestrator, the AIP communication layer, and representative device agents for Windows, Linux and Android (mobile), each designed for containerized deployment and cross-environment compatibility. The system follows a modular, plugin-oriented architecture with type-safe interfaces, persistent telemetry, and built-in tracing, enabling reproducible, large-scale orchestration across heterogeneous devices. We further provide a futuristic WebUI for operator interaction and system visualization.

We evaluated UFO3 on NebulaBench, a benchmark of 55 cross-device tasks spanning 10 categories across 5 machines (a Windows 11 desktop, three Ubuntu CPU hosts, and one Ubuntu A100 GPU node). UFO3 achieves a Subtask Completion Rate (SCR) of 83.3% and a Task Success Rate (TSR) of 70.9%. It exposes substantial parallelism, with an average execution width of 1.72 (peaking at ∼3.5), and reduces end-to-end latency by 31% compared to a sequential baseline. Fault-injection experiments further demonstrate its robustness: UFO3 automatically retries and migrates under transient outages, gracefully degrades under partial failures, and recovers conservatively under global failures.

In essence, UFO3 dissolves device boundaries and transforms the digital estate into a single, adaptive collaborator Brudy et al. (2018); Marks & White (2020); Zhang et al. (2018). It unifies distributed devices into a cohesive digital organism, one that executes user intents safely, asynchronously, and efficiently across heterogeneous environments. At its core, the Agent Interaction Protocol (AIP) serves as the connective tissue of this ecosystem, roviding a unified, fault-tolerant, and extensible communication substrate that allows new agents to join seamlessly and interoperate reliably. Over time, multiple constellations can interconnect through AIP, weaving together agents, devices, and capabilities into a self-organizing Digital Agent Galaxy. Through this design, UFO3 redefines cross-device automation, elevating it from a brittle engineering challenge to a unified orchestration paradigm, where multi-device workflows become naturally expressive and scalable across the landscape of ubiquitous computing.

#### 2 Background

###### 2.1 Digital Agents

Leveraging the power of LLMs, modern digital agents have emerged as powerful interfaces bridging human intent and the complex software ecosystems people interact with daily Zhang et al. (2024). These agents can parse natural language requests, interpret screenshots and system state Zheng et al. (2025); Wu et al. (2025); Zhao et al. (2025), decompose complex goals into subtasks Huang et al. (2024), and generate scripts or commands to execute tasks using a variety of tools Wang et al. (2024b). Execution modalities include API calls, operating-system-level commands, GUI interactions via automation or accessibility interfaces, and code generation Zhang et al.; Qiao et al. (2023); Wang et al. (2024c); Zhang et al. (2025e).

Digital agents operate across a wide spectrum of platforms. Web-based agents can navigate browsers and search the Internet Ning et al. (2025); Zheng et al. (2024), mobile agents are embedded in smartphone applications to automate mobile tasks Zhang et al. (2025d); Wang et al. (2024a), and desktop or laptop agents interact with local operating systems and graphical user interfaces Zhang et al. (2025b;a). Across these platforms, agents observe system state, reason about tasks, and execute actions, enabling applications such as automated customer support flows, desktop productivity macros, and repetitive workflow automation, essentially acting as intelligent digital assistants.

Despite their versatility and growing adoption, existing digital agents are fundamentally limited by their confinement to a single device or environment. In practice, many modern workflows are no longer confined to a single endpoint Brudy et al. (2018): users frequently interact with a combination of desktops, mobile devices, cloud services, and specialized hardware, all as part of a single task Xu et al. (2021); Chen & Koufaris (2020). Examples include running data analysis on a GPU cluster, collecting results on a personal laptop, and generating visual summaries on a tablet, or coordinating cross-platform deployments that touch both local workstations and cloud infrastructure. This trend makes the ability to operate seamlessly across devices increasingly urgent and commonplace.

Single-device frameworks face several inherent challenges in meeting this demand:

- • Limited device capabilities. Each agent is restricted by the hardware and software environment of its host, which limits the scope of tasks it can execute independently.
- • Fragmented personalization and context. User-specific preferences, personalization, and contextual knowledge captured on one device are difficult to transfer or leverage on another Cemri et al. (2025), preventing a coherent multi-device experience.
- • Manual coordination overhead. Orchestrating multiple single-device agents to accomplish crossdevice workflows typically requires extensive manual development and careful sequencing, which is time-consuming, error-prone, and hard to maintain.

Together, these limitations highlight the pressing need for a new generation of digital agents that can natively reason about, orchestrate, and adapt across heterogeneous devices.

###### 2.2 Cross-Device Agent: A New Paradigm

To overcome the limitations of single-device frameworks, we introduce the concept of cross-device intelligence, in which digital agents collaborate seamlessly across multiple heterogeneous endpoints Brudy et al. (2019). In this metaphorical digital galaxy, each device, desktop, mobile, cloud service, or specialized hardware, acts as a star, and coordinated tasks form a constellation that collectively fulfills complex user requests. Unlike traditional agents confined to a single host, these cross-device agents can reason about device capabilities, orchestrate subtasks, and execute actions as part of a unified, intelligent ecosystem Chen et al.

Cross-device intelligence enables a qualitatively new user experience. A single natural-language request, such as “prepare a production-ready demo, run performance tests, and generate a report”, can trigger an orchestrated constellation of actions: builds executed on developer laptops, computation-heavy tests on GPU clusters, deployments on cloud containers, and visualization or report generation on local or mobile devices. To the user, this appears as one coherent, effortless operation, eliminating the need for manual coordination and device-specific intervention.

This paradigm directly addresses the key limitations of single-device agents. It overcomes the constraints of individual devices by assigning tasks to the endpoints with the necessary capabilities and resources. It allows personalization, context, and user preferences captured on one device to propagate across others, enabling a coherent multi-device experience Giusti et al. (2025). And it automates orchestration of complex workflows, removing reliance on brittle glue code or labor-intensive manual integration.

###### 2.3 Design Challenges in Cross-Device Orchestration

However, synchronous control loops, cross-device agent orchestration introduces a fundamentally different class of systems challenges. In a distributed, heterogeneous environment, agents must collaborate across device, network, and platform boundaries, each with distinct runtime contexts and execution semantics Tran

- et al. (2025). Building a reliable and efficient orchestration layer under such conditions requires addressing three core challenges.

- 1. Asynchronous Parallelism. In cross-device workflows, multiple agents may execute concurrently on different endpoints. Unlike linear single-agent plans, task execution must accommodate partial completions, delayed feedback, and dynamic dependency resolution. The orchestrator must therefore reason about concurrency, detect when subtasks can safely proceed in parallel, and continuously adapt its scheduling plan based on evolving runtime states and network latencies Yu et al. (2025). Failing to do so may lead to wasted computation or stalled progress.
- 2. Distributed Coordination. Agents in a constellation operate across diverse network and trust domains, implemented in different languages and deployed on various infrastructures. Achieving coherent coordination among them requires standardized mechanisms for registration, capability advertisement, task dispatch, and result collection. These operations must occur through persistent, low-latency channels that can tolerate temporary disconnections and guarantee consistent task states Chen et al.. In practice, ad-hoc HTTP calls or ephemeral connections are insufficient; a structured communication substrate is essential to sustain long-lived agent interactions.
- 3. Heterogeneous Extensibility. A cross-device ecosystem must embrace diversity rather than constrain it. Device agents differ in operating systems, execution environments, and available toolchains. The architecture should therefore allow rapid development, deployment, and integration of new device agents while ensuring consistent semantics in task execution and error propagation Wu et al. (2024a). This requires a flexible interface model and a unified protocol to abstract platformspecific complexity, enabling scalable and evolvable orchestration across an open agent federation Yang et al. (2025a).

These challenges are not merely engineering details, as they delineate the boundary between ad-hoc agent scripts and a principled, distributed orchestration system. To address them, we present UFO3, a reliable and scalable framework for intelligent cross-device agent orchestration. UFO3 unifies four essential design

###### ConstellationClient

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Agent Registry

[Figure 8]

[Figure 9]

###### TaskConstellation

TaskStar 1

[Figure 10]

[Figure 11]

[Figure 12]

ConstellationAgent

[Figure 13]

[Figure 14]

[Figure 15]

TaskStar 3 TaskStar 4

[Figure 16]

[Figure 17]

TaskStarLine

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

TaskStar 2

[Figure 22]

[Figure 23]

Constellation Orchestrator

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Agent Interaction Protocol (AIP)

Bidirectional WebSocket, multiplexed events

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

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

Server

Server

Server

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

AIP

[Figure 55]

AIP AIP

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Client

Client

Client

Client

WindowsAgent

LinuxAgent 1 MobileAgent

LinuxAgent 2

Device Agents

Figure 2: Layered architecture of UFO3.

elements: (i) a global ConstellationAgent that decomposes user intents into dynamic, dependency-aware task DAGs (Section 5); (ii) an event-driven orchestration engine that enables concurrent, asynchronous, and adaptive execution (Section 6); (iii) a standardized Agent Interaction Protocol (AIP) that supports persistent, low-latency, and extensible communication across heterogeneous agents (Section 7); and (iv) an easy-to-use development interface that allows developers to quickly build new device agents and seamlessly integrate them into the ecosystem (Section 8). Together, these mechanisms transform isolated device agents into a coherent, cooperative constellation capable of executing complex, distributed tasks efficiently, safely, and at scale.

#### 3 UFO3: Design Principles and Overview

We now present an overview of the UFO3 architecture, which transforms a collection of heterogeneous device agents into a unified, fault-tolerant execution fabric. Figure 2 depicts its layered design. At a high level, UFO3 follows a hierarchical orchestration model that separates global coordination from local execution. This separation enables scalable cross-device orchestration while maintaining consistent control and responsiveness across diverse operating systems and network environments.

###### 3.1 Hierarchical Control Plane

At the top of the hierarchy, the ConstellationClient serves as the global control plane. It maintains a live registry of all connected device agents, including their capabilities, system specifications, and runtime health metrics. This registry allows the orchestrator to place tasks on devices that can satisfy their resource requirements, avoiding mismatches between task demands and device capacity.

Each device hosts a device agent server that manages local orchestration. The server maintains a persistent WebSocket session with the ConstellationClient and oversees execution contexts on the host. A lightweight device client on each host provides a unified interface to underlying tool environments, exposed via MCP servers, enabling task execution, telemetry streaming, and resource monitoring. This layered control plane cleanly decouples global orchestration policies from device-specific heterogeneity, providing a consistent abstraction across endpoints that may differ in OS, hardware, or network conditions.

###### 3.2 Orchestration Flow

When a high-level user request arrives, the ConstellationClient invokes the ConstellationAgent to construct a TaskConstellation: a dynamic directed acyclic graph (DAG) that encodes task decomposition, dependencies, and candidate device mappings. Each node, or TaskStar, represents an atomic execution unit assigned to a suitable device agent according to its capability profile and current system load.

The Constellation Orchestrator executes the DAG asynchronously and in an event-driven manner. Task completions trigger dependent nodes, while failures prompt retry, migration, or partial DAG rewrites. This design allows workflows to adapt to real-time system dynamics, such as device churn, network variability, or incremental task updates, while preserving overall progress and global consistency. The result is an execution model that is both highly parallel and resilient, sustaining workflow completion even as subsets of devices fail or reconnect.

###### 3.3 Cross-Agent Communication

All cross-agent interactions, including agent registration, capability synchronization, task dispatch, progress reporting, and result aggregation, are handled by the Agent Interaction Protocol (AIP). Built on persistent WebSocket channels, AIP provides a lightweight, bidirectional, and multiplexed substrate for structured event messages. Its design ensures low-latency propagation of control signals and consistent global state, even in the presence of intermittent connectivity or asynchronous updates.

Together, these design elements allow UFO3 to orchestrate large-scale, heterogeneous, and adaptive workflows, forming a cohesive foundation for building a resilient, multi-device execution fabric.

#### 4 Formal Constellation Model

We now formalize the concept of a TaskConstellation, the central abstraction that captures the concurrent and asynchronous structure of distributed task execution and dependencies. This model provides the theoretical foundation for reasoning about task dependencies, execution order, and fault-tolerant orchestration across heterogeneous devices. At its core, a TaskConstellation represents a decomposed view of a complex user request in a directed acyclic graph (DAG) representation: a set of interdependent subtasks connected through explicit dependency edges. This formalism not only enables consistent scheduling and recovery but also supports runtime dynamism, allowing new tasks or dependencies to be introduced as the workflow evolves.

This representation provides clear advantages. Task ordering and dependencies are explicitly captured, ensuring correctness across distributed execution. The DAG topology naturally exposes parallelism and asynchronous execution, enabling efficient concurrency across heterogeneous devices. Moreover, nodes and edges can be dynamically added, removed, or rewired based on predecessor completion, allowing adaptive execution without compromising consistency. These properties make DAGs a natural and effective abstraction for modeling complex, cross-device workflows in heterogeneous environments.

###### 4.1 TaskStar: Atomic Execution Unit

A TaskStar denotes the atomic unit of computation in the UFO3 framework, the smallest indivisible task scheduled on a device agent. Each TaskStar encapsulates the complete context necessary for autonomous execution, including its semantic description, assigned device, execution state, and dependency relationships.

- • Description: a natural-language specification of the task, sent to the target device agent;
- • Tips: A list of natural-language guidance designed to help the device agent successfully complete the task;
- • Device: the identifier of the device agent responsible for execution;
- • Status: the current execution state (e.g., pending, running, completed);
- • Dependencies: references to prerequisite tasks that must complete before execution.

Formally, a TaskStar 𝑡𝑖 is defined as: 𝑡𝑖 = (name𝑖, description𝑖, device𝑖, tips𝑖, status𝑖, dependencies𝑖)

Intuitively, a TaskStar “knows” what it should do, where it should run, how far it has progressed, and which other tasks it depends on. This self-contained representation enables fine-grained monitoring and decentralized scheduling across devices.

###### 4.2 TaskStarLines: Dependency Edge

A TaskStarLine represents a dependency relation between two TaskStars, forming a directed edge in the task graph. Let 𝑡𝑖 and 𝑡𝑗 denote two TaskStars. A TaskStarLines 𝑒𝑖→𝑗 specifies that 𝑡𝑗 cannot begin until certain conditions on 𝑡𝑖 are satisfied:

𝑒𝑖→𝑗 = (from_task𝑖, to_task𝑗, type, description) The type parameter determines the nature of the dependency:

- • Unconditional: 𝑡𝑗 always waits for 𝑡𝑖 to complete;
- • Success-only: 𝑡𝑗 proceeds only if 𝑡𝑖 succeeds;
- • Conditional: 𝑡𝑗 proceeds based on a user-defined or runtime condition.

These dependency edges enforce causal consistency within the constellation, ensuring that concurrent execution respects logical task ordering while maximizing parallelism.

###### 4.3 TaskConstellation: Directed Acyclic Task Graph A complete TaskConstellation is a DAG that encodes the structure of a distributed workflow:

###### C = (T, E)

where T is the set of all TaskStars and E is the set of TaskStarLines. Each TaskConstellation provides a compact yet expressive representation of how a complex task decomposes into independently executable units with explicit dependencies.

Figure 3 illustrates a simple TaskConstellation spanning multiple devices. Task A (LinuxAgent) must finish before Task C starts; both Task B (LinuxAgent) and Task C (WindowsAgent) must complete before Task D (WindowsAgent) begins; and Task E (MobileAgent) depends on the successful completion of both Task C and Task D. This example demonstrates how the model naturally captures both sequential and parallel dependencies within a unified structure.

[Figure 60]

[Figure 61]

TaskStar

[Figure 62]

[Figure 63]

Description

[Figure 64]

[Figure 65]

Tips

[Figure 66]

[Figure 67]

[Figure 68]

Device

[Figure 69]

[Figure 70]

Status

[Figure 71]

[Figure 72]

Dependencies

[Figure 73]

[Figure 74]

###### TaskStarLine

[Figure 75]

[Figure 76]

from_task

[Figure 77]

[Figure 78]

to_task

[Figure 79]

[Figure 80]

Type

- Task A

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Task B

Task C

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Description

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Task E

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Task D

Figure 3: Example of a TaskConstellation illustrating both sequential and parallel dependencies.

###### 4.4 Runtime Dynamism and Adaptivity

Unlike static DAG schedulers Islam et al. (2012); Di Tommaso et al. (2017); arg, UFO3 treats TaskConstellations as mutable objects. Tasks and dependency edges can be inserted, removed, or modified at runtime. This design enables UFO3 to dynamically react to evolving execution contexts, such as new user inputs, intermediate results, or device failures, without restarting the entire workflow. Such adaptivity is key to maintaining progress and efficiency in long-running, cross-device task orchestration.

In summary, the TaskConstellation formalism serves as the conceptual backbone of UFO3. It provides a precise and extensible representation of distributed workflows, enabling both rigorous reasoning and practical orchestration under asynchronous and failure-prone environments.

#### 5 ConstellationAgent: The Centralized Constellation Weaver

While the previous section formalized the TaskConstellation as an abstract model of distributed task structure, we now turn to its realization in the runtime control plane. The ConstellationAgent is the central intelligence of UFO3, responsible for interpreting user intent, constructing executable constellations, and steering their evolution across heterogeneous devices. Residing within the ConstellationClient, ConstellationAgent bridges the gap between high-level natural-language goals and concrete multi-agent execution through a unified orchestration interface.

- As illustrated in Figure 4, ConstellationAgent acts as both a planner and a replanner: it first synthesizes a TaskConstellation from user instructions, then incrementally refines this constellation as feedback arrives from distributed agents. Internally, ConstellationAgent is implemented as an LLM-driven ReAct agent Yao et al. (2022) governed by a finite-state machine (FSM) Schneider (2005). This FSM alternates between two complementary operating modes, namely creation and editing, forming a closed-loop control cycle that continuously updates the global execution graph in response to runtime conditions.

This design achieves a tight coupling between symbolic reasoning and distributed execution: declarative goals are grounded into concrete task graphs, and dynamic feedback drives continuous graph mutation. Through this feedback-driven control loop, ConstellationAgent maintains global consistency, ensures forward progress, and adapts seamlessly to changing device conditions. Overall, the design provides three core benefits:

- 1. Unified reasoning and control: High-level task synthesis and low-level execution coordination are decoupled yet remain tightly synchronized via the TaskConstellation abstraction.

[Figure 104]

[Figure 105]

###### Initial TaskConstellation

[Figure 106]

ConstellationAgent

Task A

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Task C Task D

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Task B

[Figure 121]

[Figure 122]

Creation

[Figure 123]

Newly Completed

[Figure 124]

[Figure 125]

[Figure 126]

User Query

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

###### Edited TaskConstellation

Agent Registry

[Figure 131]

[Figure 132]

[Figure 133]

Task A

- New Task 1
- New Task 2

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Task C Task D

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Task B

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Editing (On Task C Completed)

[Figure 153]

[Figure 154]

ConstellationAgent

Modification

Figure 4: An overview of the ConstellationAgent.

- 2. Dynamic adaptability: The editable TaskConstellation enables recovery, reallocation, and opportunistic task generation in the face of partial failures or evolving goals.
- 3. End-to-end observability: ConstellationAgent maintains a complete lineage of task states and dependencies, enabling introspection, debugging, and verifiable traceability.

###### 5.1 Responsibilities and I/O Interface

ConstellationAgent serves as the reasoning core of the UFO3 control plane, orchestrating a structured feedback loop that alternates between creation and editing phases. Each phase defines explicit inputs, outputs, and operational responsibilities, ensuring that cross-device execution remains both consistent and adaptive.

Creation Mode. In the creation phase, ConstellationAgent receives three primary inputs: (i) a user-issued goal, expressed in natural or structured language; (ii) the AgentProfile registry, describing each available device agent’s capabilities, environment, and metadata; and (iii) demonstration examples to support in-context learning (ICL) Dong et al. (2022); Jiang et al. (2024).

Leveraging the LLM’s semantic reasoning capabilities, ConstellationAgent decomposes the user goal into a structured execution graph, the TaskConstellation. Each node (TaskStar) is annotated with explicit dependencies, resource constraints, and a target device assignment. This graph constitutes the initial execution plan, which is then handed off to the Constellation Orchestrator for distributed scheduling.

The TaskConstellation is generated in a structured, machine-readable JSON format. Beyond the raw graph, ConstellationAgent produces a detailed reasoning trace: it first formulates an Observation of the input and AgentProfile, followed by a structured Thought capturing its analysis and decision-making rationale Wei et al. (2022); Ding et al. (2024). It then outputs the next State in the control loop and the overall Result for user-request or the generated TaskConstellation. This ensures that the TaskConstellation is derived through a transparent, deliberate reasoning process.

Editing Mode. During distributed execution, ConstellationAgent enters the editing phase of its adaptive control loop. In this mode, it continuously consumes: (i) the original user request; (ii) the current AgentProfile registry; (iii) a serialized snapshot of the TaskConstellation; and (iv) demonstration examples for ICL.

###### START

Cannot

No subtasks

complete

needed

Constellation

created

FINISH

CONTINUE

###### FAIL

Cannot All subtasks complete finish

Not yet

complete

Figure 5: Lifecycle state transitions of the ConstellationAgent.

Upon receiving completion or failure events, ConstellationAgent evaluates whether modifications are necessary, such as adding follow-up subtasks, removing redundant ones, or refining dependency edges. Edits are applied only to non-terminal TaskStars (i.e., not in Running, Completed, or Failed), ensuring runtime correctness while allowing the TaskConstellation to evolve dynamically.

Editing actions are invoked through one or more of the tools defined in Section 5.3 to facilitate parsing and execution. Concurrently, ConstellationAgent produces a structured reasoning trace, including a Thought narrative that explains the current constellation state, justifies any modifications (or lack thereof), identifies the next State in the control loop, and summarizes the overall user-request Result. This transparent, step-by-step reasoning guarantees that adaptations are both explainable and consistent.

This dual-mode control pattern realizes a balance between global consistency and local adaptivity. By structuring feedback integration through explicit modes and invariants, ConstellationAgent achieves stable yet flexible orchestration, avoiding common pitfalls of uncontrolled LLM self-modification. The resulting system embodies a feedback-driven orchestration loop that remains both reactive and verifiable.

###### 5.2 Finite-State Machine and Lifecycle

The internal logic of ConstellationAgent is expressed as a finite-state machine (FSM), providing a clear, enforceable structure for task lifecycle management. We show its state transition in Figure 5. The FSM governs how ConstellationAgent transitions across four primary operational states:

- • START: Initialization state where ConstellationAgent receives the user goal and constructs the initial TaskConstellation (creation mode).
- • CONTINUE: The steady-state loop, where task progress is monitored and incremental edits are applied based on runtime feedback (editing mode).
- • FINISH: The successful termination state, triggered when all subtasks are completed or no further edits are required. Results are aggregated and reported to the user.
- • FAIL: The terminal error state, entered upon irrecoverable failures or unreachable goals, prompting abort and logging for user recovery.

This FSM-based design ensures deterministic task transitions, consistent global state evolution, and predictable recovery paths. It also provides a natural boundary between LLM reasoning and deterministic control logic, which improves safety and debuggability in complex, cross-device workflows.

Table 1: Core tools exposed by the Constellation MCP Server for managing tasks and dependencies.

Tool Name Purpose Input Output add_task Add a new atomic task

Task ID, name, description, target device, tips

Updated TaskConstellation

(TaskStar)

remove_task Remove a task and all associated dependencies

Task ID Updated TaskConstellation

update_task Modify task fields (name, description, device, tips)

Task ID + updated fields

Updated TaskConstellation

add_dependency Establish a dependency between two tasks

Dependency ID, from task, to task, condition

Updated TaskConstellation

remove_dependency Remove a dependency line

Dependency ID Updated TaskConstellation

update_dependency Update the condition or description of a dependency

Dependency ID, condition description

Updated TaskConstellation

build_constellation Batch-create tasks and dependencies from structured input

Configuration dictionary, clear flag

Built TaskConstellation

###### 5.3 Constellation MCP Server: Structured Task Management

To operationalize dynamic graph construction, ConstellationAgent interacts with a lightweight Constellation MCP Server Hou et al. (2025), a modular component exposing a standardized set of task and dependency management primitives. This server serves as the structured manipulation layer that bridges LLM-level reasoning and concrete execution state. Each operation encapsulates a single, idempotent transformation on the TaskConstellation, ensuring reproducibility and easy rollback.

The MCP Server supports both fine-grained (single task or edge) and bulk (batch graph) operations, all of which return a serialized, globally consistent constellation snapshot. Table 1 summarizes the core toolset. Through this uniform interface, ConstellationAgent can safely evolve task graphs at runtime while preserving key invariants such as DAG validity and single assignment.

By decoupling reasoning (handled by ConstellationAgent) from structured mutation (handled by MCP), UFO3 achieves a clean separation of concerns: ConstellationAgent focuses on semantic decision-making, while the MCP Server enforces syntactic and structural integrity. This design not only enhances robustness and auditability but also facilitates future extensibility, for instance, incorporating new agent types, validation hooks, or task optimization heuristics without modifying the orchestration core.

Summary. In summary, ConstellationAgent serves as the “central weaver” of distributed intelligence within UFO3. By combining LLM-driven reasoning, a finite-state control backbone, and a structured task manipulation interface, it transforms abstract user goals into live, evolving constellations, maintaining both rigor and adaptability across the lifecycle of multi-device orchestration.

[Figure 155]

[Figure 156]

###### Device Agents TaskConstellation

[Figure 157]

###### Constellation Orchestrator

[Figure 158]

LinuxAgent

[Figure 159]

[Figure 160]

- Task A

(Running)

- Task B

(Running)

- Task C

- Task A

[Figure 161]

- Task B
- Task C

WindowsAgent

[Figure 162]

[Figure 163]

[Figure 164]

Event-driven Coordination

[Figure 165]

[Figure 166]

Task E

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Asynchronous Scheduler

- Task D

(Pending)

- Task E

LinuxAgent

[Figure 172]

[Figure 173]

Task D

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Safe Assignment Locking

[Figure 186]

[Figure 187]

MobileAgent

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

LinuxAgent

Consistency Enforcer

[Figure 192]

[Figure 193]

New Task F

[Figure 194]

[Figure 195]

(Completed)

[Figure 196]

Batched Constellation Editor

(Pending)

Asynchronous Execution

Figure 6: The Constellation Orchestrator bridges TaskConstellation and execution, enabling asynchronous, adaptive task orchestration across devices.

#### 6 Asynchronous Dynamic Constellation Orchestrator

With the ConstellationAgent governs the reasoning and evolution of a TaskConstellation, the Constellation Orchestrator brings that plan to life by executing, monitoring, and adapting interdependent tasks across heterogeneous devices Li et al. (2022), as shown in Figure 6. Conceptually, it transforms a static DAG into a living execution fabric, where tasks evolve concurrently, react to runtime signals, and adapt to new decisions generated by the reasoning agent.

Unlike traditional serial or static agent workflows, UFO3’s orchestration must satisfy three often conflicting goals: (i) asynchronous parallelism to leverage device heterogeneity, (ii) safety and consistency under concurrent DAG updates, and (iii) adaptivity to runtime feedback from both devices and LLM reasoning. Achieving these goals poses several technical challenges. First, subtasks must be assigned and executed asynchronously without violating data dependencies. Second, task execution may overlap with live TaskConstellation edits, requiring strict consistency control. Third, edited graphs must remain valid and acyclic. Finally, frequent updates should not degrade performance or introduce synchronization bottlenecks.

To address these challenges, the Constellation Orchestrator is built around five design pillars: (1) eventdriven coordination, (2) asynchronous scheduling, (3) safe assignment locking, (4) consistency enforcement, and (5) batched constellation editing. Together, these principles enable scalable, adaptive orchestration over evolving task graphs while preserving correctness and efficiency.

###### 6.1 Event-Driven Coordination

Traditional DAG schedulers rely on polling or global checkpoints to detect task completion, introducing latency and synchronization overhead. In contrast, the Constellation Orchestrator operates as a fully event-driven system built on an internal event bus and an observer design pattern.

Two key components manage this process: the ConstellationProgressObserver, which tracks task execution and orchestrates DAG edits, and the ConstellationModificationSynchronizer, which ensures global consistency after each modification. The orchestrator emits and reacts to four primary event types:

- 1. TASK_STARTED: triggered when a TaskStar is assigned to a device and begins execution.
- 2. TASK_COMPLETED: emitted upon successful task completion, prompting potential DAG updates.
- 3. TASK_FAILED: emitted on failure, triggering re-planning or fallback logic.
- 4. CONSTELLATION_MODIFIED: emitted once DAG edits are committed and synchronized across agents.

These events collectively capture the lifecycle of each task and the evolution of the constellation. All handlers operate asynchronously, ensuring immediate, fine-grained reactions to runtime signals without centralized

[Figure 197]

[Figure 198]

###### TaskConstellation

- Task A

[Figure 199]

- Task B Task D

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Task C

[Figure 208]

Figure 7: Illustration of asynchronous scheduling and concurrent TaskConstellation editing.

coordination delays. This event-driven design provides high responsiveness and forms the foundation of adaptive orchestration in UFO3.

###### 6.2 Asynchronous Scheduling

At the core of the orchestrator lies a fully asynchronous scheduling loop. Unlike traditional schedulers that alternate between discrete planning and execution phases, the orchestrator continuously monitors the evolving DAG to identify ready TaskStar, those whose dependencies are satisfied, and dispatches them concurrently to available devices. Each TaskStar runs within an asyncio coroutine that encapsulates its full lifecycle: execution, result collection, and event publication. When a task starts, a TASK_STARTED event is emitted; upon completion or failure, corresponding TASK_COMPLETED or TASK_FAILED events are immediately published to trigger downstream orchestration updates.

Notably, TaskStar execution and TaskConstellation editing can proceed concurrently. As shown in Figure 7, when Task A completes and triggers an edit, the edit operation executes in parallel with the ongoing Tasks B and C. Similarly, edits triggered by Task B overlap with subsequent executions. This concurrency further reduces end-to-end latency by overlapping computation and orchestration, allowing the system to adapt in real time as results stream in.

This asynchronous design maximizes device utilization and eliminates idle waiting, enabling parallel progress across heterogeneous devices. It is essential for scaling cross-device workflows, where independent subtasks (e.g., log collection, file aggregation, or model execution) can execute concurrently while higher-level orchestration continuously adapts to dynamic task states and outcomes.

###### 6.3 Safe Assignment Locking and Synchronization

While asynchrony improves efficiency, it introduces correctness challenges when task execution overlaps with DAG updates. Specifically, the orchestrator must prevent race conditions when the ConstellationAgent dynamically adds, removes, or rewires TaskStar during execution. Without safeguards, a task could be dispatched based on a stale DAG, leading to duplicated or invalid execution.

To ensure atomicity, the orchestrator employs a safe assignment lock. When an edit cycle begins, bounded by a TASK_COMPLETED/TASK_FAILED event and its corresponding CONSTELLATION_MODIFIED event, the scheduler suspends new task assignments to prevent dispatching based on stale DAG states. During this period, incoming events are queued, and the ConstellationModificationSynchronizer guarantees that all edits are applied atomically. A key aspect of this process is synchronization: once the ConstellationAgent finishes editing, the orchestrator merges its structural changes with runtime updates from concurrently running tasks (i.e., new completions or failures that occurred during the editing window). This ensures that the

Algorithm 1: Safe Assignment Locking and Asynchronous Rescheduling Protocol Input: Event stream E, current TaskConstellation C Output: Consistent and updated C with newly scheduled ready tasks

- 1 while system is running do

- 2 foreach event 𝑒 ∈ E do

- 3 if 𝑒 is TASK_COMPLETED or TASK_FAILED then

- 4 async enqueue(𝑒) ; //record completion/failure for processing asynchronously

- 5 acquire(assign_lock) ; //suspend new assignments
- 6 while queue not empty do

- 7 𝑒 ← dequeue() ; //get next event for processing
- 8 Δ = invoke(ConstellationAgent, edit(C, 𝑒)) ; //propose DAG edits
- 9 C ← apply(C, Δ) ; //update the constellation structure
- 10 validate(C) ; //ensure acyclicity and invariants (I1–I3)
- 11 publish(CONSTELLATION_MODIFIED, 𝑡) ; //notify DAG update
- 12 C ← synchronize(C, T𝐶) ; //merge newly completed TaskStars

- 13 release(assign_lock) ; //resume orchestration after all queued events are processed
- 14 // Rescheduling Phase (outside lock)
- 15 T𝑅 ← get_ready_tasks(C) ; //collect newly ready TaskStars
- 16 foreach 𝑡 ∈ T𝑅 do

- 17 async dispatch(𝑡) ; //send to available device agent asynchronously
- 18 async publish(TASK_STARTED, 𝑡) ; //notify asynchronously

final TaskConstellation reflects a globally consistent view of both reasoning-time modifications and execution-time progress.

The complete protocol is summarized in Algorithm 1, which demonstrates how locking, validation, synchronization, and rescheduling jointly ensure correctness and consistency under concurrent task updates. Figure 8 further illustrates this process. When multiple TASK_COMPLETED events arrive simultaneously, the orchestrator acquires a global lock to prevent inconsistent scheduling decisions. During this locked phase, the ConstellationAgent performs DAG modifications (e.g., ΔA, Δ[B, C]), which are atomically merged, validated, and synchronized with the live constellation state. Once synchronization completes, the lock is released, and normal scheduling resumes based on the updated DAG. Each edit cycle is linearized (publish–release) and validated before scheduling resumes; the linearization argument and TLA+ model are provided in Appendix A and A.1.

This mechanism guarantees that task assignments remain immutable during edits, preventing conflicts between execution and modification. By maintaining atomicity and synchronization without blocking overall progress, it preserves both safety and consistency for concurrent, LLM-driven orchestration at scale.

###### 6.4 Consistency and Safety Guarantees

Since the DAG may be dynamically rewritten by an LLM, the orchestrator enforces runtime invariants to preserve correctness even under partial or invalid updates:

- • I1 (Single Assignment): Each TaskStar has at most one active device assignment at any time.
- • I2 (Acyclic Consistency): Edits must preserve DAG acyclicity; the orchestrator performs local cycle detection before committing modifications.
- • I3 (Valid Update): Only PENDING tasks and their dependent nodes may be modified; RUNNING, COMPLETED, and FAILED nodes are immutable.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- Task B

- Task A

Task D Task E

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

(t0)

Task C

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

- Task B

- Task A

Task D Task E

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

(t5)

Task C

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

ΔA

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Δ

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Δ

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

- Task B

Task A

Task D Task E

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

(t7)

- Task C

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

- Task B

- Task A

Task D Task E

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

(t3)

Task C

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

- Task B

Task A

Task D Task E

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

(t4)

- Task C

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

ΔA

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Δ[B, C]

Figure 8: An example of the safe assignment locking and event synchronization workflow.

Together, these invariants ensure that even as the constellation evolves, new stars form, old ones fade, the overall structure remains stable and semantically valid. We enforce three runtime invariants (I1–I3) under a lock-bounded editing regime; see Appendix A for the formal state model and proof sketch.

###### 6.5 Batched Constellation Editing

Frequent LLM-driven edits can introduce significant overhead if processed individually. To balance responsiveness with efficiency, the orchestrator supports batched constellation editing. During a reasoning round, multiple TASK_COMPLETED or TASK_FAILED events may accumulate; instead of invoking the ConstellationAgent after each event, the orchestrator aggregates them and applies the resulting modifications atomically once reasoning is complete. As illustrated in Figure 8, when task_A completes (𝑡0), the ConstellationAgent starts an edit cycle. During this process, new completion events from task_B and task_C arrive (𝑡3, 𝑡4) and are temporarily queued. After the first edit result Δ𝐴 is validated and synchronized (𝑡5), the orchestrator batches the pending updates from B and C into a single reasoning round (𝑡6–𝑡7).

This batching mechanism amortizes LLM invocation and synchronization overhead while preserving atomicity and consistency. It enables the orchestrator to remain both efficient and adaptive, reacting swiftly to meaningful state transitions without incurring excessive micro-edits or redundant reasoning calls. We prove

an edit–sync confluence lemma showing that folding runtime events commutes with lock-bounded edits within the same window; see Appendix A (Edit–Sync Confluence).

Design Summary. In contrast to static DAG or synchronous schedulers, the Constellation Orchestrator treats task execution as an open-world process, continuously evolving, reacting, and converging toward user intent. Its event-driven backbone ensures responsiveness; asynchronous scheduling maximizes concurrency; locking and batching ensure safety; and DAG validity checks preserve correctness under dynamic reasoning.

Together, these components realize a new form of orchestration, asynchronous, adaptive, and reasoning-aware, that bridges declarative intent and distributed execution across a heterogeneous universe of intelligent agents.

#### 7 Agent Interaction Protocol (AIP)

The orchestration model described in Section 6 requires a communication substrate that remains correct under continuous DAG evolution, dynamic agent participation, and fine-grained event propagation. Legacy HTTP-based coordination approaches (e.g., A2A Duan & Lu (2025), ACP Ehtesham et al. (2025)) assume short-lived, stateless interactions, incurring handshake overhead, stale capability views, and fragile recovery when partial failures occur mid-task. These assumptions make them unsuitable for the continuously evolving workflows and long-running reasoning loops characteristic of UFO3.

###### 7.1 Design Overview

AIP serves as the nervous system of UFO3, connecting the ConstellationClient, device agent services, and device clients under a unified, event-driven control plane, as shown in Figure 9. It is designed as a lightweight yet evolution-tolerant protocol to satisfy six goals:

- (G1) Maintain persistent bidirectional sessions to eliminate per-request overhead;
- (G2) Unify heterogeneous capability discovery via multi-source profiling;
- (G3) Ensure fine-grained reliability through heartbeats and timeout managers for disconnection and failure detection;
- (G4) Preserve deterministic command ordering within sessions;
- (G5) Support composable extensibility for new message types and resilience strategies;
- (G6) Provide transparent reconnection and task continuity under transient failures.

To meet these requirements, AIP adopts a persistent, bidirectional WebSocket transport and decomposes the orchestration substrate into five logical strata (Figure 9), each responsible for a distinct aspect of reliability and adaptability:

- • L1: Message Schema Layer — Defines strongly-typed, Pydantic-validated contracts (ClientMessage, ServerMessage) for message direction, purpose, and task transitions. Structured metadata (system info, capabilities) supports unified capability discovery (G2) and deterministic ordering via explicit ID correlation (G4).
- • L2: Transport Abstraction Layer — Provides a protocol-agnostic Transport interface with a production-grade WebSocket implementation supporting configurable pings, timeouts, and large payloads. Decoupled transport logic ensures low-latency persistent sessions (G1) and future extensibility (G5).
- • L3: Protocol Orchestration Layer — Implements modular handlers for registration, task execution, heartbeat, and command dispatch (see Appendix B), each extending a common AIPProtocol base with middleware hooks (logging, metrics, auth). This design ensures ordered state transitions (G4) and composable extensibility (G5).

Constellation Client

Device Agent Server

Device Agent Client

###### L5: Endpoint Orchestration Layer

DeviceServer, DeviceClient, Constellation Endpoints

###### L4: Resilience & Health Management Layer

HeartbeatManager, TimeoutManager, ReconnectionStrategy

###### L3: Protocol Orchestration Layer

Registration, TaskExecution, Heartbeat, DeviceInfo, Command

###### L2: Transport Abstraction Layer

WebSocketTransport, Persistent Connections, Adapters

###### L1: Message Schema Layer

ClientMessage, ServerMessage, Pydantic Validation

Figure 9: AIP Architecture: Five-layer protocol stack enabling persistent, resilient multi-agent orchestration.

- • L4: Resilience and Health Management Layer — Encapsulates HeartbeatManager, TimeoutManager, and ReconnectionStrategy with exponential backoff and automatic session recovery. It guarantees reliability (G3) and seamless task continuity under transient disconnections (G6).
- • L5: Endpoint Orchestration Layer — Provides role-specific facades: DeviceServerEndpoint, DeviceClientEndpoint, and ConstellationEndpoint, integrating lower layers into deployable components. These endpoints unify connection lifecycle, task routing, and health monitoring across roles, reinforcing G1–G6.

Together, these layers form a vertically integrated stack where L1 establishes semantic contracts, L2 provides transport flexibility, L3 implements protocol logic, L4 ensures operational resilience, and L5 delivers deployment-ready orchestration primitives. This design enables UFO3 to maintain correctness and availability under DAG evolution (G4, G5), agent churn (G3, G6), and heterogeneous execution environments (G1, G2).

###### 7.2 Agent Registration and Profiling

Agent registration in AIP corresponds to the entry point of the orchestration pipeline, anchoring the capability discovery and topology formation processes outlined in (G2) and implemented primarily through the L1–L3 layers.

- As illustrated in Figure 10, the registration pipeline consists of three complementary stages that together establish a unified and continuously refreshed view of the constellation’s capabilities:

[Figure 399]

[Figure 400]

###### AgentPofile Registry

###### (1) User-Specified Registration

[Figure 401]

[Figure 402]

(ConstellationClient)

merge

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Agent Endpoint Information User Preference

[Figure 408]

[Figure 409]

[Figure 410]

###### (2) Service-level Manifest (Device Agent Service)

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Supported Tools Capabilities Metadata Service Infomation

[Figure 417]

[Figure 418]

[Figure 419]

###### (3) Client-Side Telemetry (Device Agent Client)

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

OS Hardware GPU Environment Status

Figure 10: Agent registration flow: multi-source AgentProfile construction and registration.

### AgentProfile: gpu_agent Status: idle

Performance GPU: 4 × NVIDIA A100 80G CPU: 96 cores Memory: 866.1 GB

System OS: linux (Ubuntu 24.04) Capabilities: cli, file_system, data_processing, model_training

Network & Host (optional) Host: gui-model-a100 IP: 172.19.0.4

Paths Dataset: ~/dataset Training: ~/script Code: ~/code

Last heartbeat: 2025-10-28 07:46:24 UTC | Server: ws://localhost:5005/ws

Figure 11: An example AgentProfile of a GPU agent with Linux system.

- 1. User-specified registration (ConstellationClient). Administrators provide endpoint identities and specify user preferences. These initial configurations define the logical boundaries of the constellation and seed the connection parameters required for session establishment.
- 2. Service-level manifest (device agent service). Each device agent advertises its supported tools, environment variables, and operational metadata through a REGISTER message. These descriptors are validated and normalized by the Message Schema Layer (L1) and merged through the Protocol Orchestration Layer (L3), ensuring semantic consistency and structured capability discovery.
- 3. Client-side telemetry (device agent client). Local clients continuously report runtime metrics, such as OS version, hardware status, GPU utilization, and software environment, to the device agent service. This telemetry stream keeps the global registry up to date and enables adaptive re-scheduling under resource drift or device churn.

The aggregated results from these three stages are merged by the ConstellationClient into a unified AgentProfile that represents each agent’s real-time operational state. An example AgentProfile for a GPU-enabled device is shown in Fig. 11, demonstrating how multi-level profiling captures hardware, software, and dynamic runtime descriptors within a single schema.

Through this registration and profiling pipeline, AIP achieves continuous capability discovery, evolutiontolerant orchestration, and consistent topology awareness across heterogeneous devices. The process directly fulfills (G1–G3) by maintaining persistent sessions, ensuring reliable metadata propagation, and enabling transparent adaptation as the constellation evolves.

###### 7.3 Task Dispatch and Result Delivery

Task dispatch operationalizes the event-driven execution model envisioned in (G1) and (G4) through tightly managed sessions that persist across multiple task rounds. When the ConstellationClient assigns a TaskStar to a device, the Transport Abstraction Layer guarantees low-latency delivery of the serialized TASK message to the target agent service. The Protocol Orchestration Layer coordinates message routing, while the Resilience and Health Management Layer monitors the session heartbeat to ensure reliability.

Each task follows a deterministic life cycle, from TASK to TASK_END, with strict ordering guarantees enforced by the session-level sequence manager. Intermediate logs and evaluator outputs are streamed back incrementally to the ConstellationClient, which updates the global TaskConstellation state and triggers potential DAG adjustments.

This continuous, feedback-driven execution loop transforms AIP from a mere transport protocol into a temporal coordination substrate, harmonizing asynchronous reasoning, scheduling, and execution across distributed devices.

###### 7.4 Command Execution

- At a finer operational granularity, AIP implements a unified command execution model that directly fulfills (G4) and (G5) by ensuring deterministic, extensible control within persistent sessions.

Each COMMAND message specifies a unique identifier, target function, and a typed argument list. The Message Schema Layerenforces structure and validation, while the Protocol Orchestration Layer executes commands sequentially within the session context to maintain determinism. To optimize multi-action workflows, multiple commands may be batched in a single message, reducing round-trip overhead. Execution results are returned as structured Result objects, containing status codes, return values, and error metadata. Failures or timeouts are propagated through the same channel, enabling the orchestrator to apply adaptive recovery or task reassignment strategies via L4’s resilience mechanisms.

Unified with UFO3’s MCP tool-calling interface, this model bridges system-level orchestration with modelgenerated actions, ensuring that high-level reasoning and low-level execution operate under a consistent and evolvable protocol surface.

###### 7.5 Resilient Connection Protocol

The distributed and volatile nature of device environments necessitates a dedicated resilience layer to uphold (G3) and (G6). AIP’s Resilient Connection Protocol, implemented primarily within L4, guarantees synchronized fault handling and seamless recovery across client–server boundaries.

When a Device Agent disconnects unexpectedly, the orchestrator immediately marks it as DISCONNECTED, removes it from the active scheduling pool, and triggers background reconnection attempts using exponential backoff. Upon recovery, the agent re-registers automatically, restoring its prior session state and resuming task participation without manual intervention. If disconnection occurs during task execution, all affected tasks are transitioned to TASK_FAILED, and corresponding updates are propagated to the ConstellationAgent for DAG revision, ensuring that the orchestration view remains globally consistent.

Symmetrically, when the ConstellationClient itself disconnects, the corresponding Device Agent Server receives a termination signal and proactively aborts all ongoing tasks associated with that client. This bidirectional fault-handling policy prevents resource leakage, avoids orphaned execution states, and guarantees that both endpoints maintain a consistent global view of task progress.

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

###### Device Agent

Strategy Command

[Figure 432]

[Figure 433]

[Figure 434]

collect_screenshot

[Figure 435]

[Figure 436]

[Figure 437]

DATA_COLLECTION

[Figure 438]

[Figure 439]

collect_a11y_tree

[Figure 440]

[Figure 441]

...

[Figure 442]

[Figure 443]

LLM_INTERACTION

###### State Machine

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Command

[Figure 448]

[Figure 449]

[Figure 450]

- State 1

[Figure 451]

[Figure 452]

- State 2

[Figure 453]

[Figure 454]

click

ACTION_EXECUTION

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

State 3

type

[Figure 460]

[Figure 461]

MEMORY_UPDATE

[Figure 462]

[Figure 463]

bash

###### ...

...

Figure 12: The three-layer framework of a device agent.

Together, these mechanisms realize an end-to-end resilient orchestration substrate that preserves correctness, availability, and synchronization under transient network failures or partial system outages,closing the reliability loop envisioned in AIP’s layered design.

Summary. AIP consolidates registration, task dispatch, command execution, and resilience into a coherent, evolution-tolerant communication fabric that embodies the six goals outlined in Section 7.1. Functionally, AIP forms the nervous system of UFO3, enabling reasoning, execution, and recovery to operate seamlessly within an evolving constellation of intelligent agents. Its minimal yet extensible design ensures that as workflows, models, and environments evolve, the underlying communication protocol remains stable, adaptive, and correct by construction.

#### 8 Design and Development of Device Agents

With the structured reasoning of the ConstellationAgent, the dynamic execution model of the Constellation Orchestrator, and the low-latency, persistent communication enabled by AIP, the next challenge is clear: how do we design a device agent that can be quickly onboarded into UFO3, adapt to a heterogeneous platform, and seamlessly participate in evolving task constellations? Our solution provides a standardized template for device agent development, minimizing engineering effort while maximizing system scalability and reliability.

###### 8.1 Architecture Overview

A device agent serves as the execution endpoint of UFO3, translating high-level task directives into concrete actions on a target device. To support rapid development and flexible deployment, each agent is structured as a three-layer, state-machine-driven framework, illustrated in Figure 12. To enable safe and scalable execution across heterogeneous devices, each agent is further partitioned into a server that manages orchestration and FSM logic, and a client that executes low-level commands locally. This server-client separation, combined with the layered FSM design, balances modularity, extensibility, and runtime robustness, and supports both single-agent and multi-agent deployment scenarios depending on platform requirements.

Specifically, the architecture decomposes agent behavior into three hierarchical levels:

- 1. Level-1: State (Finite-State Machine Layer). Each agent maintains an internal state machine that governs its behavior at each execution step. A state encapsulates a processor, the next state to transition to, and optionally the next agent to invoke in multi-agent setups. The collection of states defines a finite-state machine that ensures predictable, controllable lifecycle progression.

Device Agent Client (On Device)

###### Device Agent Server

(Cloud Hosted)

|Local MCP Servers<br><br>Tool 1 ... Tool N<br><br>|
|---|

Result

###### Session

###### AIP

State Machine

[Figure 464]

Command

Result HTTP Tool call Local MCP Servers Tool 1 ... Tool N

Figure 13: The server-client architecture of a device agent.

- 2. Level-2: Strategy (Execution Logic Layer). Within each state, the processor manages a sequence of strategies that implement the step-level workflow. Strategies handle tasks such as data collection, environment inspection, prompt construction, action planning, or tool invocation. This separation allows the agent to compose complex behaviors from modular, reusable strategies, while maintaining clear boundaries between decision logic and execution mechanics.
- 3. Level-3: Command (System Interface Layer). Each strategy can invoke a set of commands from a configured MCP server, which provides standardized operations for perceiving system state, executing tools, or interacting with device resources. Commands are executed deterministically and report structured outcomes, allowing higher layers to react and adapt without managing low-level device specifics.

By instantiating an agent with concrete State, Strategy, and Command definitions, a fully functional device agent can be realized. This hierarchical, layered approach offers several advantages. First, the same framework can accommodate a wide variety of devices, platforms, and execution environments. Second, developers can onboard new devices by defining only the relevant states, strategies, and commands, without rewriting orchestration or communication logic. Finally, the layered FSM structure allows the agent to respond to dynamic task edits, partial failures, or concurrent executions while maintaining correctness.

Together, this architecture positions device agents as plug-and-play execution units, seamlessly bridging the high-level reasoning of the ConstellationAgent, the dynamic orchestration of the Constellation Orchestrator, and the event-driven communication of AIP. It forms the final, essential layer that enables UFO3 to operate as a cohesive, scalable, and resilient multi-device system.

###### 8.2 Server-Client Architecture

To support safe, scalable, and flexible execution across heterogeneous devices, each device agent is partitioned into a server and a client, as illustrated in Figure 13. This separation of responsibilities aligns naturally with the layered FSM architecture and leverages AIP for reliable, low-latency communication.

Server: Orchestration and State Management. The agent server is responsible for managing the agent’s state machine lifecycle, executing high-level strategies, and interacting with the ConstellationAgent or the orchestrator. It handles task decomposition, prompt construction, decision-making, and command sequencing. Crucially, the server maintains full control over the agent’s workflow logic, enabling updates to decision strategies without impacting low-level execution on the device.

Each server instance exposes its AgentProfile, a structured description of its capabilities, configurations, and runtime status. This metadata allows the orchestrator to dynamically select suitable agents for specific subtasks, improving task distribution efficiency. A single server can manage multiple agent clients concurrently, maintaining isolation across devices while supporting centralized supervision and coordination.

Client: Commands Execution and Resource Access. The agent client runs on the target device and manages a collection of MCP servers or tool interfaces. These MCP servers can operate locally (via direct invocation) or remotely (through HTTP requests), and each client may register multiple MCP servers to access diverse tool sources. Upon receiving commands from the agent server, such as collecting telemetry, invoking system utilities, or interacting with hardware components, the client translates them into MCP tool calls, executes them deterministically, aggregates the results, and returns structured outputs via AIP. The client remains stateless with respect to reasoning: it faithfully executes directives without engaging in high-level decision-making.

During initialization, each client connects to the agent server through the AIP endpoint, performs self-checks (e.g., disk, CPU, memory, GPU, and network configuration), and registers its hardware–software profile. This profile is integrated into the server’s AgentProfile, giving the orchestrator complete visibility into system topology and resource availability for informed task assignment and scheduling.

Server-Client Communication. All communication between the server and client is routed through the AIP, leveraging persistent WebSocket connections. This allows bidirectional, low-latency messaging that supports both synchronous command execution and asynchronous event reporting. By decoupling high-level reasoning from low-level execution, the system can safely update server logic or client MCP tools independently, without disrupting ongoing workflows.

Design Consideration. This server–client architecture provides strong modularity and scalability. Device clients can be rapidly deployed with minimal setup, immediately joining UFO3 as execution endpoints. The server focuses on high-level reasoning and orchestration, while clients ensure deterministic command execution, preventing cross-layer interference and simplifying maintenance. Persistent sessions and structured AIP event semantics enhance robustness under intermittent connectivity and dynamic task updates. The design also scales efficiently across devices: a single server can orchestrate multiple clients, and extensibility is achieved by adding new tools or interfaces at the client side or new reasoning strategies at the server without mutual dependencies.

###### 8.3 State: Finite-State Machine Layer

The top-level lifecycle of each device agent is governed by a finite state machine (FSM), which provides a structured and predictable execution framework. Each state encapsulates the logic for handling a specific step of the agent’s workflow, including invoking the corresponding processor, determining the next state to transition to, selecting the next agent in multi-agent setups, and deciding whether the current task has reached completion. Figure 14 illustrates the interface exposed by a state.

At runtime, the agent invokes the state’s handle function, which executes the state-specific behavior (e.g., a processor) and returns control decisions to the FSM. Transitions between states can be determined dynamically by the agent based on LLM reasoning, or triggered by rule-based logic in response to errors, timeouts, or external events. This flexibility allows the agent to react promptly to runtime conditions, while maintaining a predictable execution path for normal workflows.

The FSM-based design offers several advantages. First, it enforces a clear separation of concerns: state transitions govern workflow progression, processors implement step-level strategies, and commands handle low-level system interactions. Second, it simplifies reasoning about agent behavior and facilitates debugging, since each state represents an isolated, testable unit. Finally, the FSM enables device agent to safely manage dynamic agent behaviors and failures, and concurrent executions.

- 1 class AgentState(ABC):

- 2 """Abstract interface for a device agent state."""

- 3

- 4 @abstractmethod

- 5 async def handle(self , agent , context=None):

- 6 """Execute the logic for the current state."""

- 7 pass

- 8

- 9 @abstractmethod

- 10 def next_state(self , agent) -> "AgentState":

- 11 """Return the next state in the FSM."""

- 12 pass

- 13

- 14 @abstractmethod

- 15 def is_round_end(self) -> bool:

- 16 """Determine whether the current task round has ended."""

- 17 pass

Figure 14: Simplified interface of a device agent state, highlighting the core methods for execution, state transition, and termination checks.

In essence, the State layer provides a robust backbone for device agent execution, allowing high-level orchestration from the ConstellationAgent and Constellation Orchestrator to be reliably translated into stepwise, adaptive actions across heterogeneous devices.

###### 8.4 Strategy: Composable Execution Logic

Each agent state delegates step-level workflow management to a processor, which orchestrates a sequence of strategies. Strategies encapsulate modular execution logic, enabling fine-grained control over the agent’s behavior while maintaining a clear separation between decision-making (State layer) and concrete actions (Command layer).

In a typical processor, we define four core strategy types and execute sequentially:

- • DATA_COLLECTION: Gather necessary context from the device, such as screenshots, accessibility information, system status, or user input.
- • LLM_INTERACTION: Construct prompts using the collected data and interact with the LLM to obtain actionable instructions or decisions.
- • ACTION_EXECUTION: Perform the commands returned by the LLM or pre-defined toolkits, applying them to the device environment deterministically.
- • MEMORY_UPDATE: Update the agent’s short-term or long-term memory to reflect task progress and provide context for subsequent steps.

This strategy-based design offers several advantages. First, it allows flexible customization for different device types and task requirements; strategies can be reordered, added, or replaced without modifying the core state machine. Second, it provide a template that modularizes execution, making the workflow easier to test, debug, and extend. Finally, by clearly delineating data collection, reasoning, action, and memory update, the processor ensures that each step in the agent’s lifecycle is composable, observable, and adaptable to dynamic runtime conditions.

Overall, the Strategy layer acts as the operational bridge between the high-level reasoning of the State layer and the low-level Command execution, enabling device agents to carry out complex, multi-step tasks reliably and efficiently.

###### FINISH CONTINUE FAIL

Cannot

task finish

complete

not yet

complete

Figure 15: Lifecycle state transitions of the LinuxAgent.

###### 8.5 Command: Atomic Execution Units

Commands represent the atomic execution units within a device agent. Each command encapsulates a specific operation, defined by a function and its corresponding arguments, which maps directly to a tool call on the MCP server co-located with the device client. By treating commands as self-contained units, the agent can systematically decompose complex workflows into discrete, testable actions.

Each Strategy invokes commands to realize its operational intent, for example, a DATA_COLLECTION strategy might request a screenshot or accessibility tree, while an ACTION_EXECUTION strategy triggers a system command or UI interaction. Commands are transmitted via the AIP to the client, which performs the actual tool execution on the device and returns structured results back to the server. This separation of command logic from device-level execution ensures deterministic, reliable, and auditable operations across heterogeneous platforms. At runtime, the agent can query the client for available commands and their usage metadata, enabling dynamic selection by the LLM and adaptive workflows. This design supports extensibility: new tools or device capabilities can be integrated by simply registering additional commands at the client layer, without modifying the server-side logic or State/Strategy definitions.

In essence, the Command layer completes the device agent’s execution pipeline: it bridges high-level reasoning (State), step-wise workflow orchestration (Strategy), and concrete device interaction, providing a robust and flexible foundation for reliable, multi-step automation across diverse environments.

###### 8.6 Example Device Agents: LinuxAgent and WindowsAgent

To demonstrate how the layered device agent architecture and server-client design can be instantiated in practice, we present two representative case studies: LinuxAgent and WindowsAgent. These examples illustrate how UFO3’s templates enable rapid development, integration, and execution of device agents across different platforms.

The LinuxAgent showcases a single-agent deployment, highlighting how a standalone agent can leverage the FSM-based State, Strategy, and Command layers to interact with system tools, collect telemetry, and execute workflow tasks. In contrast, the WindowsAgent demonstrates a multi-agent deployment, where multiple agent instances coordinate via a server-client setup to manage complex workflows involving local MCP servers, UI automation, and dynamic LLM-driven task decomposition. Together, these two case studies provide concrete examples of how UFO3 can accommodate diverse execution environments, and serve as the foundational agents for the experimental evaluations in Section 10.

###### 8.6.1 LinuxAgent: Single-Agent CLI Execution

The LinuxAgent is designed as a lightweight, single-agent instance capable of executing command-line instructions to fulfill user requests. It demonstrates how a standalone device agent can leverage the layered FSM architecture and server-client design to perform intelligent, iterative task execution on a CLI-based environment.

State. The LinuxAgent’s lifecycle is governed by a minimal set of states, capturing the essential execution progression while maintaining simplicity and predictability:

- • CONTINUE: The task is ongoing and requires further CLI command execution to progress toward completion.
- • FAIL: The task cannot proceed under current system conditions or resources, signaling an unrecoverable error.
- • FINISH: The task has successfully completed all required operations.

At each step, the agent evaluates execution outcomes and determines the appropriate state transition, allowing the FSM to drive both normal progress and error handling in a structured, deterministic manner.

Strategy. When in the CONTINUE state, the LinuxAgent executes a processor that orchestrates a small, modular set of strategies:

- • LLM_INTERACTION: Construct prompts using prior execution results and predefined templates to request next-step commands from the LLM.
- • ACTION_EXECUTION: Execute the CLI commands returned by the LLM, ensuring results are captured and structured for downstream processing.
- • MEMORY_UPDATE: Persist execution results and issued commands into the agent’s memory for future reference, enabling iterative refinement and error recovery.

This layered strategy design separates decision-making, such as prompt construction, from execution and state updates, enhancing modularity, reproducibility, and extensibility for future workflow modifications. In particular, unlike traditional polling-based or externally triggered data collection, the agent can proactively obtain system and environment information by invoking CLI commands on demand, eliminating unnecessary overhead and increasing responsiveness.

Command. The LinuxAgent interacts with the MCP server via two primary commands:

- • EXEC_CLI: Execute arbitrary shell commands, capturing stdout and stderr for structured feedback.
- • SYS_INFO: Collect system-level information, such as memory usage, disk space, and hardware configuration, to inform decision logic or precondition checks.

These commands provide the atomic building blocks for the agent’s strategies, isolating system-specific operations within the client layer while enabling the server layer to focus on workflow orchestration and LLM-guided reasoning.

The LinuxAgent illustrates the minimal viable design for a single-agent system that integrates FSM control, strategy orchestration, and command execution. By maintaining a small, deterministic state set, modular strategies, and well-defined commands, the agent achieves robust, flexible, and traceable CLI task execution.

###### 8.6.2 WindowsAgent: Multi-Agent Coordination on GUI Systems

While the LinuxAgent represents a lightweight, single-agent model for command-line environments, the WindowsAgent embodies a more sophisticated multi-agent framework tailored for GUI-based systems. We leverage the Desktop AgentOS UFO2 as the implementation of the WindowsAgent, which follows the same architectural principles introduced above but extends them to support multi-application coordination. Specifically, UFO2 consists of a HostAgent that decomposes a user request into multiple subtasks and assigns each subtask to an AppAgent, which executes the assigned subtask within an individual application. Below, we highlight the core design ideas and the major differences from LinuxAgent, and refer readers to the UFO2 paper for full details.

HostAgent

User Interaction

Task Decomposition

[Figure 465]

[Figure 466]

Context Management AppAgent Instantiation

Subtask Result

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

AppAgent 1 AppAgent 2 AppAgent 3 AppAgent N

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

...

###### Windows Applications

- Figure 16: Overall architecture of the WindowsAgent built upon UFO2. Figure adapted from the original paper.

State. The HostAgent adopts a state machine similar to that of the LinuxAgent, but introduces an additional ASSIGN state responsible for delegating subtasks to AppAgents. Once assigned, control transitions to the selected AppAgent for execution. Importantly, when an AppAgent reaches the FINISH or FAIL state, the control does not terminate but instead returns to the HostAgent, enabling it to decide whether to retry, re-plan, or advance to the next subtask. This hierarchical state transition mechanism naturally supports cooperative task completion across multiple applications.

Strategy. Unlike LinuxAgent, where system states can be dynamically queried through CLI commands, GUI-based environments require explicit perception of the screen and interface hierarchy. Both HostAgent and AppAgent therefore begin each round with a DATA_COLLECTION strategy that captures screenshots and accessibility (a11y) metadata to construct a structured view of the GUI environment. These inputs are crucial for grounding subsequent reasoning and action decisions. The remaining strategies, LLM_INTERACTION, ACTION_EXECUTION, and MEMORY_UPDATE, follow the same modular workflow as in the LinuxAgent, and are omitted here for brevity. This design unifies the overall agent logic across heterogeneous platforms while allowing for system-specific customization.

Command. The WindowsAgent exposes a significantly richer command set compared to the LinuxAgent. The HostAgent can invoke MCP tools to launch or select applications, while the AppAgent operates through a dedicated GUI MCP server capable of simulating user interactions such as mouse clicks and keyboard inputs. Furthermore, each AppAgent can integrate application-specific MCP servers that bridge to internal APIs, enabling faster and more reliable automation than purely vision-based approaches. This hybrid interaction model, combining GUI manipulation with API-level control strikes a practical balance between generality and robustness.

Overall, LinuxAgent and WindowsAgent jointly demonstrate how the proposed architecture can be adapted for both single-agent and multi-agent environments, from lightweight CLI systems to complex desktop ecosystems. The same design paradigm can be extended to other platforms, such as mobile or in-vehicle infotainment systems, ensuring scalability and reusability across diverse device types.

[Figure 478]

- Figure 17: Snapshot of the UFO3 WebUI. The interface integrates natural-language interaction, TaskConstellation visualization, and device agent management in real time.

#### 9 Implementation and Engineering

We implemented UFO3 as a large-scale system consisting of approximately core 73K lines of Python code, integrating the centralized ConstellationAgent, the asynchronous Constellation Orchestrator, the AIP communication layer, and 3 representative device agents, i.e.,, LinuxAgent, WindowsAgent and MobileAgent (Android). An additional 6.1K lines of code were developed for the frontend web UI. The implementation is further accompanied by over 77K lines of user documentation, detailing module interfaces, orchestration protocols, and configuration schemas. This extensive documentation ensures maintainability, reproducibility, and smooth multi-team integration, reflecting substantial engineering effort and demonstrating UFO3’s readiness for real-world deployment.

We leverage Python’s asyncio framework for concurrent event handling, allowing dynamic DAG updates, agent registration, and task execution to proceed asynchronously without blocking the orchestrator’s control loop. The system runs seamlessly across Linux and Windows environments, enabling dynamic workflow orchestration, persistent cross-device communication, and plugin-based extensibility.

###### 9.1 Interactive UFO3 WebUI

We build a modern, futuristic WebUI that serves as the operator-facing control surface of UFO3, integrating chat interaction, real-time task monitoring, and device management within a single view (Figure 17). The chatbox at the center allows users to issue natural-language requests and observe agent reasoning and replies in real time. The right panel visualizes the evolving TaskConstellation as a DAG and lists all TaskStars with status indicators and detailed logs accessible via expansion. The lower-left registry displays all connected device agents with their capabilities, and states, enabling operators to quickly inspect, reconnect, or migrate tasks as needed. Execution events stream continuously through an event-driven backend built on FastAPI and WebSocket, ensuring sub-millisecond update latency and seamless synchronization between orchestrator and visualization.

This design provides high observability and transparency across the multi-agent workflow. Users can trace each decision from natural-language reasoning to execution outcomes, diagnose failures via per-TaskStar logs, and visualize dependency satisfaction in real time. By unifying conversation, orchestration, and monitoring,

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

- Figure 18: Example Markdown log generated by UFO3, showing agent actions, reasoning, TaskConstellation DAG (before and after edits).

the WebUI bridges human intent and agentic execution, allowing rapid debugging, safe intervention, and fine-grained control without disrupting asynchronous task execution. Overall, the WebUI turns UFO3 from a background automation engine into an interactive, transparent, and trustworthy orchestration environment.

###### 9.2 Plugin and Extension Framework

Both the ConstellationAgent and device agents expose configurable MCP servers interfaces implemented using the FastMCP package. Upon startup, each agent launches an embedded MCP server whose toolset is dynamically registered according to its role (e.g., Linux CLI tools, Windows GUI automation, or system telemetry collectors). This plugin mechanism allows developers to add new capabilities, such as a novel GUI driver or API connector, by implementing a lightweight interface, without altering any orchestration or protocol logic. This design significantly improves maintainability, reduces coupling, and enables rapid onboarding of new device types into the constellation.

###### 9.3 Prompt and Model Integration

Prompts are modularly defined via a hierarchical configuration. Each agent maintains a core system prompt template augmented by a collection of in-context exemplars for few-shot adaptation. The ConstellationAgent centrally manages the LLM backend through a unified API layer compatible with OpenAI-style interfaces, enabling model-agnostic deployment across GPT-based, Claude-based, or local open-weight models. This separation between orchestration and inference ensures the entire system remains robust against future LLM model changes.

###### 9.4 Automated Task Logging.

To facilitate debugging and system introspection, UFO3 automatically generates detailed logs in Markdown format after each task execution. These logs capture the complete trace of agent actions, reasoning steps, and intermediate outputs, including thoughts, invoked commands, and returned results. They also include

visualizations of the TaskConstellation DAG, showing both the initial and modified topologies, as well as task execution timelines (TaskStar and TaskStarLine). Figure 18 illustrates an example log output, demonstrating how these comprehensive records provide a clear, structured view of task execution for analysis and debugging.

#### 10 Experimental Evaluation

Following the system implementation, we evaluate the UFO3 framework to understand its effectiveness, robustness, and coordination efficiency in realistic, heterogeneous environments. Existing agent benchmarks predominantly focus on single-device or single-OS settings (e.g., text-based API workflows or GUI automation) Mu et al. (2025), which fail to capture the cross-device, cross-platform orchestration that UFO3 is designed for. Moreover, to the best of our knowledge, there exists no prior agent system capable of orchestrating multi-device tasks that span both Linux and Windows environments with unified control and shared context. This makes direct comparison against existing systems infeasible and potentially misleading.

Therefore, instead of benchmarking against prior single-agent or single-platform baselines, we focus on a comprehensive internal evaluation that characterizes UFO3’s performance across multiple dimensions, covering its planning accuracy, execution reliability, coordination efficiency, and fault tolerance. This approach allows us to isolate the impact of UFO3’s architectural innovations and evaluate how well it scales to realistic multi-device orchestration scenarios.

Our evaluation aims to answer the following research questions:

- • RQ1 (Task Completion): Can UFO3 successfully complete diverse, multi-agent tasks across heterogeneous devices and platforms?
- • RQ2 (Orchestration and Adaptation): How effectively does UFO3 orchestrate a user query into a structured TaskConstellation DAG, and how does it adapt the DAG dynamically in response to intermediate subtask results during execution?
- • RQ3 (Parallelism Exploitation): To what extent can UFO3 identify and exploit parallelism across independent subtasks to accelerate overall execution without compromising correctness?
- • RQ4 (Performance and Scalability): How efficient is UFO3 in task planning, scheduling, and end-to-end execution latency under different network conditions and system scales?
- • RQ5 (Robustness): How does UFO3 handle partial failures, network delays, or unavailable agents during distributed execution?

- 10.1 Experiment Setup We deploy UFO3 across five physical machines in a controlled environment:

- • 1× Windows 11 desktop, running the WindowsAgent;
- • 3× Ubuntu 22.04 workstations (CPU-only), running LinuxAgent;
- • 1× Ubuntu 24.04 GPU node equipped with four NVIDIA A100 GPUs, running LinuxAgent.

Agents communicate via a simulated local network with 1–10ms latency and a wide-area link (50–100ms latency) for the GPU node using the AIP. All components are implemented in Python 3.10 and powered by the same large language model (GPT-5-Chat-20251003) OpenAI (2025) for both the ConstellationAgent and device agents. The orchestrator and controller run on a dedicated management node that coordinates agent discovery, task scheduling, and monitoring. This setup emulates a realistic hybrid enterprise environment that combines cloud servers, local desktops, and GPU compute nodes.

Table 2: Overview of the 10 task categories used in evaluation. Each category includes 4–10 representative cases.

Category Description Count Logs & Monitoring Log retrieval, aggregation, and report generation across devices. 6 System State & Configuration

Managing environment variables, users, permissions, and disk information.

5

Processes & Services Starting, stopping, and monitoring services and scheduled tasks. 5 Data Wrangling & Scripting

Parsing CSV/text files, performing statistical summaries, and executing scripts.

4

DevOps & Containers Managing Git repositories, CI/CD pipelines, and container operations. 10 Networking & Connectivity

Diagnosing connectivity issues, pinging hosts, and verifying endpoints. 5

Browsing Tasks Web-assisted cross-device operations that require interacting with web resources via a browser and then applying or verifying results on remote Linux hosts.

5

Cross-Device Orchestration

Coordinating multi-agent workflows involving data transfer and dependency handling.

5

GPU & ML Workloads Launching and monitoring GPU-based ML training and inference jobs. 5 Negative Tasks Infeasible or invalid tasks used to test failure detection and safe termi-

5

nation.

###### 10.2 NebulaBench: A Crossed-Device Evaluation Benchmark

To assess UFO3’s generality and real-world applicability, we construct NebulaBench, a benchmark of 55 representative multi-agent tasks spanning typical productivity and system administration workflows. Five volunteers with diverse Linux and Windows experience proposed realistic queries they would naturally issue to a digital assistant in daily use. The resulting tasks span ten functional categories, summarized in Table 2. A detailed list of queries in NebulaBench is shown in Appendix C and Table 6.

Each scenario is labeled with one of three difficulty levels, namely Easy, Medium, and Hard, reflecting the degree of orchestration and reasoning required:

- • Easy: Single-host operations such as log inspection, one-off checks, service control, small file transfers, or short summarization. These tasks require minimal coordination and succeed with standard administrative privileges.
- • Medium: Moderately orchestrated tasks involving multiple machines or light cross-platform activity (e.g., container run-and-verify, metric aggregation, spreadsheet updates). They often require conditional logic, transient failure handling, or simple parsing and aggregation.
- • Hard: Complex multi-step workflows requiring cross-platform orchestration, CI/build pipelines, container image management, distributed data processing, or live patching. These tasks involve high privilege levels, non-trivial verification, and greater exposure to race conditions or dependency issues.

- Figure 19 visualizes the composition of NebulaBench. The left pie chart shows the distribution of task difficulty, which is roughly balanced across Easy (33%), Medium (35%), and Hard (33%) tasks. The right chart depicts the number of devices required per task, ranging from 0 to 5. Tasks requiring no devices correspond to negative tasks that are inherently infeasible; the system is expected to detect and fail these safely without

- Figure 19: Distribution of NebulaBench tasks. Left: proportion of tasks by difficulty. Right: number of devices involved per task.

executing any agent actions. Most tasks, however, involve 3–5 devices, highlighting NebulaBench’s focus on multi-agent orchestration scenarios.

This stratification enables a controlled evaluation of UFO3’s reasoning, coordination, and execution robustness under increasing complexity. Each task is executed end-to-end, from natural language request to final result through the orchestrator–agent hierarchy, with all required data, scripts, and code pre-deployed. By including negative tasks, we also validate UFO3’s safety mechanisms, ensuring that infeasible goals are correctly identified and aborted without unintended side effects.

###### 10.3 Metrics and Methodology

To rigorously assess the effectiveness and robustness of UFO3, we design a comprehensive, multi-dimensional evaluation methodology covering correctness, adaptivity, efficiency, and fault tolerance. Our goal is not only to measure whether tasks are completed, but also to understand how well UFO3 plans, adapts, and sustains performance under dynamic and heterogeneous conditions. We organize our analysis around four research questions (RQ1–RQ4), each corresponding to a key aspect of UFO3’s system behavior.

- RQ1: Task Completion. We measure the overall Task Success Rate (TSR) and Subtask Completion Rate (SCR) to evaluate UFO3’s execution reliability. TSR captures the fraction of user queries that are successfully completed end-to-end, as judged by the query authors based on final outcomes. SCR measures the success rate of individual subtasks executed by device agents, annotated automatically by an LLM evaluator that inspects the task trajectories and completion logs. Together, TSR and SCR provide a top-down and bottom-up view of system reliability across the orchestration hierarchy. Note that for negative test cases, we mark a task or subtask as successful when the agent correctly detects and reports the intended failure, rather than attempting to complete it erroneously.
- RQ2: Orchestration and Adaptation. We evaluate UFO3’s reasoning and dynamic re-planning by analyzing the initial TaskConstellation DAG and its evolution during execution. To quantify adaptability, we track DAG modification metrics, including edits, node insertions and deletions, and dependency changes, which reflect how the system adjusts the task constellation in response to runtime results. Comparing the initial and final DAGs provides a holistic measure of UFO3’s ability to adapt its plan as execution unfolds.
- RQ3: Parallelism and Execution Efficiency. To evaluate UFO3’s ability to exploit concurrency and optimize execution, we measure several metrics derived from the TaskConstellation DAG:

| | | | | |
|---|---|---|---|---|

| |
|---|

| |
|---|

- Figure 20: Task and subtask success rates by difficulty level (left) and by query category (right) of UFO3.

- • Maximum Parallel Width: the largest number of subtasks that can be executed concurrently at any point in the DAG.
- • Critical Path Length (L): the execution time of the longest serial dependency chain in the DAG.
- • Total Task Execution Time (W): the sum of execution times of all tasks, representing the cumulative workload.
- • Parallelism Ratio (P): 𝑃 = 𝑊/𝐿, capturing the degree to which UFO3 leverages parallel execution across devices.

These metrics provide insight into UFO3’s efficiency in orchestrating multi-agent tasks and its effectiveness in parallelizing independent subtasks to reduce overall task latency.

- RQ4: Latency and Scalability. We evaluate End-to-End Latency, Orchestration Time (including TaskConstellation creation and edits), and Total Task Execution Time to characterize system efficiency. These metrics capture the balance between reasoning overhead, arising from LLM-based planning, scheduling, and coordination, and the execution gains achieved through distributed parallelism. Together, they provide a holistic view of UFO3’s scalability across heterogeneous agents and devices under realistic workload conditions.
- RQ5: Robustness and Fault Handling. We assess robustness through a case study on a single request executed under three controlled failure modes: (i) a single device agent fails and recovers via dynamic DAG reassignment, (ii) a single agent fails without recovery, and (iii) all agents fail simultaneously. These scenarios capture UFO3’s fault tolerance boundary and illustrate its adaptive recovery mechanisms under partial and global disruptions.

Overall, this evaluation framework enables a holistic examination of UFO3’s behavior under diverse operational conditions, revealing how design principles such as event-driven scheduling, runtime DAG modification, and agent autonomy translate into measurable system benefits.

###### 10.4 RQ1: Task Completion Analysis

To evaluate UFO3’s effectiveness in executing distributed tasks, we first examine the primary metrics of interest: Subtask Completion Rate (SCR) and Task Success Rate (TSR). We analyze performance both by task difficulty and by functional category, as summarized in Figure 20 1. Overall, UFO3 demonstrates strong reliability across the benchmark, achieving an SCR of 83.3% and a TSR of 70.9%. These high rates

1A complete list of UFO3’s performance on all queries is provided in Appendix C and Table 6.

Table 3: Average modifications per edit and per request by type.

Added Tasks

Removed Tasks

Modified Tasks

Added Dependencies

Removed Dependencies

Modified Dependencies

Total

Per Edit 0.05 0.00 0.99 0.04 0.01 0.00 1.09 Per Request 0.41 0.00 5.20 0.24 0.07 0.00 5.91

indicate that the orchestrator effectively coordinates subtasks across devices, while individual device agents reliably execute their assigned operations, validating the robustness of UFO3’s design. Breaking down by difficulty, easy and medium tasks exhibit comparable TSRs (72.2% and 73.7%, respectively), showing that UFO3 handles routine and moderately interdependent tasks effectively. For hard tasks, TSR declines to 66.7%, yet even in these complex multi-host scenarios UFO3 successfully completes the majority of requests, highlighting its capability under challenging orchestration conditions.

By functional category, performance trends align with task characteristics. Structured, deterministic tasks such as Data (SCR 100%) and Proc (SCR 96%) achieve the highest reliability. Tasks involving user-like interactions or ambiguity, including Browsing (SCR 64.2%) and Negative scenarios (SCR 66.7%), show lower success rates. Intermediate categories requiring cross-device coordination, such as Orchestration (SCR 85%) and DevOps (SCR 77.1%), demonstrate solid subtask reliability, evidencing UFO3’s capacity to manage dependencies across heterogeneous devices.

Across both analyses, SCR generally exceeds TSR. This gap arises because a single subtask failure among multiple dependencies can cause an overall task failure, even when most subtasks succeed. The pattern underscores the value of UFO3’s dynamic DAG management and reasoning, which mitigate partial failures and maintain high system reliability in complex, distributed workflows.

Error Analysis. To better understand the remaining failures, we examined common error patterns. First, tasks requiring file transfers across devices occasionally fail because AIP currently supports only textual communication; device agents may not have direct network access to each other. Future work includes maintaining a shared memory and extending AIP to support file read/write operations. Second, some failures arise when agents attempt to complete tasks regardless of preconditions. For example, the request “Start UFO3 service on all Linux devices” triggered the LinuxAgent to create the service even when it did not exist, rather than reporting a failure. Enhancing agent self-awareness and enforcing conditional execution will mitigate such issues. Third, tasks executed via WindowsAgent show relatively lower success due to GUI dependencies and interface variability; improving robustness of GUI-based agents remains an important direction.

Despite these errors, UFO3’s generated TaskConstellation are largely correct, indicating that the ConstellationAgent effectively decomposes complex requests, identifies dependencies, and exposes parallelism, validating its reasoning design.

###### 10.5 RQ2: Orchestration and Adaptation

We next examine UFO3’s ability to orchestrate a user query into an executable TaskConstellation DAG and adapt it dynamically as execution unfolds. To quantify this, we track modifications to the DAG across all requests, including node insertions, deletions, and dependency updates in Figure 21, and compare the initial and final task and dependency counts in Table 3.

Overall, UFO3 demonstrates a highly active editing behavior, with an average of 1.09 modifications per edit and 5.91 modifications per request. Most changes occur in the Modified Tasks category, reflecting the system’s strategy of enriching downstream tasks based on the results of preceding subtasks. For example, logs collected by earlier tasks are incorporated into subsequent document-writing or analysis subtasks, ensuring that later steps have complete context and accurate information. In contrast, the number of added or removed tasks and dependencies is minimal, indicating that ConstellationAgent generally produces well-structured DAGs from the outset and only fine-tunes tasks during execution.

| |
|---|

| |
|---|

- Figure 21: Comparison of the initial and final task (left) and dependency (right) counts in the TaskConstellation of UFO3.

Table 4: Parallelism characteristics of task constellations by difficulty.

Max Parallel Width

Critical Path Length (L)

Total Exec. Time (W)

Parallelism Ratio (P = W/L)

Difficulty

Easy 3.53 144.10 315.28 1.86 Medium 2.73 166.94 212.54 1.51 Hard 3.21 232.12 339.65 1.77

Overall 3.17 178.34 289.20 1.72

Breaking down by difficulty (Figure 21), we observe that UFO3 maintains a robust initial decomposition across Easy, Medium, and Hard tasks, with only modest increases in both task and dependency counts by the end of execution. This demonstrates that while the orchestrator adapts to runtime results, it does so without fundamentally restructuring the workflow, preserving overall plan stability. Harder tasks show slightly more edits, consistent with their longer multi-step workflows and richer context propagation.

Taken together, these observations indicate that UFO3 exhibits strong adaptive orchestration capabilities:

- (i) it actively enriches downstream tasks based on prior subtask outputs, ensuring context-aware execution;
- (ii) it generates high-quality initial DAGs, requiring minimal structural modifications; and (iii) it balances stability with runtime flexibility, applying refinements without disrupting the overall task plan. These features highlight UFO3’s ability to both plan effectively and adjust dynamically across distributed devices.

###### 10.6 RQ3: Parallelism and Execution Efficiency

We next investigate UFO3’s ability to exploit concurrency and optimize distributed task execution. Table 4 reports four DAG-derived metrics that capture the structural and runtime efficiency of each task constellation. Across all scenarios, UFO3 achieves an average parallelism ratio of 1.72, with up to 3.5 concurrent subtasks executing at peak, demonstrating that UFO3 can effectively uncover and schedule independent subtasks across devices.

We observe several trends. First, the maximum parallel width remains consistently high (around 3 tasks) even as task difficulty increases, indicating that UFO3’s orchestrator maintains concurrency opportunities even in complex, multi-host settings. Second, while the critical path length naturally grows with task complexity (144s → 232s), the increase in total execution time (315s → 340s) is modest, showing that UFO3 successfully overlaps execution through concurrent scheduling. Third, medium-difficulty tasks exhibit slightly lower parallelism (𝑃 = 1.51), as many involve short verification or data aggregation steps with limited parallel components, while both easy and hard tasks benefit more from concurrent flows.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 22: Task timing and orchestration breakdown by difficulty.

These results reveal that UFO3’s DAG-based orchestration is not only structurally well-parallelized but also runtime-efficient. By modeling tasks as DAGs, UFO3 naturally identifies and executes independent subtasks asynchronously across heterogeneous devices, keeping the system utilization high. Together, these findings demonstrate that UFO3 delivers strong execution efficiency through fine-grained parallel orchestration, validating the effectiveness of its DAG-based design.

###### 10.7 RQ4: Latency and Scalability

We evaluate End-to-End Latency, Orchestration Time (including TaskConstellation creation and edits), and Total Task Execution Time to characterize UFO3’s efficiency and scalability. These metrics reflect the trade-off between reasoning overhead, arising from LLM-based planning, scheduling, and coordination, and execution gains obtained through distributed parallelism.

As shown in Figure 22 (left), UFO3 exhibits strong execution efficiency across all task difficulties. On average, the measured end-to-end latency is only 243.7 seconds, notably shorter than the combined orchestration and task execution time of 352.5 seconds (63.3 + 289.2). This corresponds to a 31% reduction in total completion time compared to a fully sequential workflow. Even for complex, cross-device tasks, UFO3 maintains an average end-to-end latency of about 4 minutes, demonstrating its ability to exploit distributed parallelism while preserving robust coordination and correctness. Compared to manual execution, which often takes tens of minutes or longer, UFO3 delivers a substantial improvement in both speed and scalability. This efficiency gain stems from UFO3’s concurrent design described in Section 6.2: multiple subtasks are executed in parallel across heterogeneous agents, while ConstellationAgent continues to refine the TaskConstellation asynchronously in the background. Such overlap between orchestration and execution effectively hides reasoning latency and maximizes device utilization.

We also observe a clear correlation between task difficulty and latency: harder tasks incur longer orchestration times (84.2s for Hard vs. 55.1s for Easy) and longer end-to-end latency (325.2s vs. 195.9s), reflecting the increased complexity, number of subtasks, and cross-device coordination required. However, even for Hard tasks, the total end-to-end time remains within a few minutes, demonstrating UFO3’s efficiency compared to manual execution of equivalent multi-device workflows.

- Figure 22 (right) further breaks down the average time spent in TaskConstellation creation, editing, and individual task completion. Creation and editing are consistently fast (16–20s and 8–12s, respectively), indicating that ConstellationAgent is highly efficient at planning and adapting task DAGs in real time. Task completion, averaging around 48s overall, dominates the workflow, yet remains reasonable given the distributed nature and heterogeneity of devices.

These results underscore several key insights: (i) UFO3 leverages cross-device parallelism effectively, enabling task execution to substantially overlap with reasoning and orchestration; (ii) the orchestration engine scales

###### Scenario 1

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

###### Step 1

###### Step 2 Step 6

Linux1 Disconnected

Notepad (Windows)

[Figure 487]

- Task A

[Figure 488]

- Task B Task D

- Task A

[Figure 489]

- Task B Task D

Task A'

- Task A

[Figure 490]

- Task B Task D

Task A'

Linux1

[Figure 491]

(Task A Failed)

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

Resumed

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

...

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

Task C

Task C

Task C

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

- Task A: Run long_job.sh on

- linux_agent_1

Task B: Run long_job.sh on

- linux_agent_2

Task C: Run long_job.sh on

- linux_agent_3

- Task D: Create runtime report in Notepad

remove_dependency(from="Task A", to="Task D") add_task(id="Task A'",

Final Result: All three linux agents executed long_job.sh successfully. The Notepad report 'runtime_report.txt' was created

- Task B: COMPLETED
- Task C: COMPLETED Task E: COMPLETED

name="Retry long_job.sh on linux1", device="linux1")

and saved on the Windows agent’s

Desktop. Overall request completed successfully.

add_dependency(from="Task A'", to="Task D")

###### Scenario 2

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

###### Step 1

###### Step 2 Step 5

Linux1 Disconnected (Task A Failed)

###### Task A'

- Task A

[Figure 546]

- Task B Task D

- Task A

[Figure 547]

- Task B Task D

Task A'

- Task A

[Figure 548]

- Task B Task D

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

...

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

Task C

Task C

Task C

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

- Task A: Run long_job.sh on

- linux_agent_1

Task B: Run long_job.sh on

- linux_agent_2

Task C: Run long_job.sh on

- linux_agent_3

- Task D: Create runtime report in Notepad

remove_dependency(from="Task A", to="Task D") add_task(id="Task A'",

- Task B: COMPLETED
- Task C: COMPLETED Task A': FAILED

name="Retry long_job.sh on linux1", device="linux1")

add_dependency(from="Task A'", to="Task D")

Notepad (Windows)

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

Final Result: long_job.sh completed: (linux_agent_1: 31.01s,

- linux_agent_2: 58.01s,
- linux_agent_3: FAILED). The Notepad report 'long_job_report.txt' was created.

###### Scenario 3

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

###### Step 7

###### Step 1

###### Step 2

###### Linux1,2,3 Disconnected (Task A,B,C Failed)

- Task A

[Figure 603]

- Task B Task D

Task A'

- Task A

[Figure 604]

- Task B Task D

Task A'

- Task A

[Figure 605]

- Task B Task D

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

...

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

Task B'

Task C

Task B'

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

Task C Task C'

Task C

Task C'

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

- Task A: Run long_job.sh on

- linux_agent_1

Task B: Run long_job.sh on

- linux_agent_2

Task C: Run long_job.sh on

- linux_agent_3

- Task D: Create runtime report in Notepad

- add_task(id="Task A'",...)
- add_task(id="Task B'",...)
- add_task(id="Task C'",...)

- Task A': FAILED
- Task B': FAILED
- Task C': FAILED

Final Result: All long_job.sh executions on linux_agent_1–3 (including retries) have failed. The constellation cannot fulfill the request.

- add_dependency(from="Task A'", to="Task D")
- add_dependency(from="Task B'", to="Task D")
- add_dependency(from="Task C'", to="Task D")

- Figure 23: Fault injection scenarios for robustness evaluation. UFO3’s adaptive behavior under (1) transient, (2) permanent, and (3) full-agent failures.

gracefully with task complexity, keeping latency manageable even for multi-step, multi-host tasks; (iii) the efficient creation and editing of the TaskConstellation demonstrates that ConstellationAgent can dynamically adapt plans with minimal overhead, supporting robust and responsive execution in realworld scenarios. Collectively, these observations confirm that UFO3 is both performant and scalable for heterogeneous, distributed multi-agent workloads.

###### 10.8 RQ5: Robustness and Fault Handling

Finally, we evaluate UFO3’s robustness and fault-tolerance capabilities by observing its behavior under three simulated device-agent failure scenarios. The test request is: “Run long_job.sh concurrently on Linux 1–3 and report their running time on Notepad.” Ideally, this request invokes three LinuxAgent instances to execute the script in parallel, followed by the WindowsAgent aggregating and recording the results in Notepad. We design three fault-injection scenarios to examine UFO3’s adaptive responses:

- • Scenario 1: One LinuxAgent is intentionally disconnected but recovers shortly before the task completes.
- • Scenario 2: The same LinuxAgent disconnects and does not recover for the remainder of the execution.
- • Scenario 3: All LinuxAgent instances remain unavailable until the task ends.

These cases, illustrated in Figure 23, allow us to analyze how UFO3 detects failures, adapts the TaskConstellation accordingly, and maintains graceful degradation or recovery during runtime.

- In Scenario 1, where Linux 1 is temporarily disconnected, UFO3 detects the failure event immediately. The corresponding subtask (Task A) is marked as failed, and upon receiving this signal, the ConstellationAgent proactively spawns a replacement task (Task A’) to retry execution. When Linux 1 reconnects before the task deadline, the retry is successfully executed, and the results are correctly aggregated by the WindowsAgent. This demonstrates UFO3’s ability to automatically recover from transient device failures through adaptive re-planning without human intervention, showing that its fault-handling mechanism is both responsive and self-healing.
- In Scenario 2, Linux 1 remains disconnected throughout execution. UFO3 follows the same retry logic, creating a new Task A’ and reassigning it to the same device after an adaptive delay. However, as the device remains unavailable, the retry also fails. Instead of hanging indefinitely or aborting the entire workflow, UFO3 recognizes this as a partial failure, continues executing the remaining subtasks on other available devices, and finally records the aggregated results, including explicit failure traces in the final Notepad report. This behavior shows that UFO3’s orchestration framework can degrade gracefully under partial failures, preserving useful progress and ensuring result completeness.
- In Scenario 3, all LinuxAgent instances are disconnected, leading to consecutive retry failures across all subtasks. After exhausting retry attempts, the ConstellationAgent terminates the execution plan and reports the entire request as failed. The failure is propagated in a transparent manner, and UFO3 refrains from producing hallucinated or incomplete results. This highlights the system’s emphasis on fail-safe integrity, where it prefers honest termination over speculative completion when recovery is infeasible.

Overall, these experiments demonstrate that UFO3 exhibits a high degree of robustness in distributed environments. Its adaptive orchestration loop allows it to distinguish between transient and permanent failures, retry safely, and preserve progress when possible. More importantly, the system maintains end-to-end consistency even under multi-agent disruptions, validating the effectiveness of its hierarchical fault-tolerance design.

###### 10.9 Case Study: Cross-Device Orchestration in Action

Lastly, we present three representative cases from NebulaBench to demonstrate how UFO3 effectively and efficiently orchestrates heterogeneous device agents to complete complex user requests across operating systems and application boundaries.

###### 10.9.1 Case 1: Collecting Logs and Writing Reports

- As illustrated in Figure 24, this case requests UFO3 to “Retrieve all warning and error logs from Linux servers, add them to the ‘report’ sheet of the log_detailed.xlsx file, and send an email with the report to

[Figure 670]

[Figure 671]

[Figure 672]

Retrieve all warning and error logs from Linux servers, add them to the report sheet of the log_detailed Excel file, and send an email with the report to

[Figure 673]

the operations engineer.

[Figure 674]

[Figure 675]

###### Excel & Outlook (Windows)

[Figure 676]

###### Step 5

###### Step 4

###### Step 1

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

- Task A

[Figure 681]

- Task B Task D Task E

- Task A

[Figure 682]

- Task B Task D Task E

- Task A

[Figure 683]

- Task B Task D

[Figure 684]

[Figure 685]

###### Notepad (Windows)

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

Task E

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

...

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

Task C

Task C

Task C

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

update_task(task_id="Task D", description=""" Open 'log_detailed.xlsx' and append warning/error logs from Linux 1~3 to sheet 'report'...

update_task(task_id="Task E", description=""" Send an email to 'hidan.zhang@gmail.com' with attach 'log_detailed.xlsx'... Message: "Hi Hidan, the latest

Final Result: All warning and error logs from Linux 1~3 successfully retrieved and appended to the 'report' sheet of 'log_detailed.xlsx'.... An email titled 'Updated Log Report' was sent to hidan.zhang@gmail.com with the updated

- Task A (Linux1): Retrieve logs from Linux1
- Task B (Linux2): Retrieve logs from Linux2
- Task C (Linux3): Retrieve logs from Linux3
- Task D (Windows): Update Excel report
- Task E (Windows): Send email with report

- Linux 1: …
- Linux 2: …
- Linux 3: … """, …)

warning and error logs have been

Excel file attached, summarizing the

appended... """, …)

collected logs.

- Figure 24: Case 1: Log collection and report generation. UFO3 decomposes the high-level request into parallel log retrieval tasks on Linux agents and sequential report-generation tasks on Windows, demonstrating cross-device coordination and result aggregation.

the operations engineer.” While conceptually straightforward, this task traditionally requires tedious manual effort across multiple systems and interfaces.

UFO3 autonomously decomposes the request into five TaskStars: three for parallel log retrieval and two for final report generation and email dispatch. The first three TaskStars are dispatched to distinct LinuxAgents, each collecting logs asynchronously from its assigned host. Once completed, the results are automatically aggregated by the ConstellationAgent, which then spawns two sequential TaskStars on the WindowsAgent to insert the logs into Excel and compose an email summary.

This case exemplifies UFO3’s strong parallelism and adaptive coordination capabilities. By overlapping asynchronous data collection with centralized result fusion, UFO3 achieves both time efficiency and consistency across heterogeneous systems, significantly reducing human effort and turnaround time compared to manual workflows.

###### 10.9.2 Case 2: Task Allocation and Result Integration via Excel

In Figure 25, we illustrate the second case, where UFO3 is asked to “Complete all tasks listed in schedule.xlsx on Windows using Linux servers, and write back a one-sentence result summary to the Result Summary column.” This case evaluates UFO3’s ability to bridge structured spreadsheet data manipulation with distributed execution across heterogeneous devices.

Upon receiving the request, UFO3 first creates a TaskStar on the WindowsAgent to parse the Excel sheet, identifying five distinct subtasks, each targeting a specific Linux host. These subtasks (Task B–D) are then dispatched to the respective LinuxAgents for concurrent execution. After all remote tasks complete, the ConstellationAgent aggregates their outputs, synthesizes concise summaries, and writes them back into the appropriate cells of the same Excel file.

This case demonstrates how UFO3 seamlessly integrates centralized data management with distributed execution, effectively turning Excel into a live orchestration interface. It highlights the system’s fine-grained understanding of data provenance, automatically preserving task-to-cell correspondence without human supervision.

###### 10.9.3 Case 3: Resource-Aware Distributed Computation

Finally, Figure 26 presents a more heterogeneous and complex example, where the user requests: “I have a distributed computation task that requires coordinating four Linux hosts. The task consists of GPU-intensive matrix multiplication, CPU-intensive dataset processing, and memory-intensive data aggregation. Please

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

Complete all tasks on in schedule.xlsx on windows on linux servers, and write back the one-sentence result summary to the Result Summary column.

[Figure 748]

###### Step 1

Task B

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

Task C Task E

Task A

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

Task D

[Figure 765]

[Figure 766]

[Figure 767]

- Task A (Windows): Read schedule.xlsx
- Task B (Linux1): Execute tasks on Linux1 Task C (Linux2): Execute tasks on Linux2 Task D (Linux3): Execute tasks on Linux3

Task E (Windows): Write results to Excel

WindowsAgent

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

Read

[Figure 772]

[Figure 773]

###### Step 5

Task B

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

Task A Task C Task E

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

Task D

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

update_task(task_id="Task E", description="""Open 'schedule.xlsx' and Write the following into the 'Result Summary' column:

- T001: 15 log files modified ...
- T002: Disk usage 2% used, ...
- T003: Memory usage 9% used, ...
- T004: 2 files in /tmp, ...
- T005: 9 python processes running...""", …)

[Figure 798]

WindowsAgent

[Figure 799]

[Figure 800]

Write

[Figure 801]

[Figure 802]

Step 2

[Figure 803]

Task B

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

Task A Task C Task E

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

Task D

[Figure 821]

[Figure 822]

[Figure 823]

- update_task(task_id="Task B", description="""Execute the following tasks T001: Count Log size ... T004: File Count Check ...""", …)
- update_task(task_id="Task C", description="""Execute the following tasks T002: Disk Usage Check ... T004: Process Check ...""", …)

update_task(task_id="Task B", description="""Execute the following tasks T003: Memory Check ...""", …)

Excel (Windows)

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

Final Result: All tasks from schedule.xlsx were executed on the Linux servers and their one-sentence summaries were successfully written back to the 'Result Summary' column. … The Excel file 'schedule.xlsx' has been updated and saved successfully, completing the user request.

- Figure 25: Case 2: Distributed task execution from a centralized Excel sheet. UFO3 interprets structured task descriptions from Excel, distributes them to Linux agents for execution, and writes concise summaries back into the corresponding cells.

inspect hardware and current workload on all four hosts, then assign one GPU-intensive task, two CPUintensive tasks, and one memory-intensive task to the appropriate hosts. Provide a task assignment report including host specs, assigned workloads, and notes in Notepad on Windows.”

UFO3 first triggers all four LinuxAgents to perform hardware inspection and runtime workload evaluation in parallel (Task A–D). Based on the collected telemetry, the ConstellationAgent dynamically plans the task distribution strategy, ensuring that computationally expensive workloads are placed on the most suitable hosts. After the distributed executions complete, the WindowsAgent automatically compiles a task assignment report summarizing host capabilities, workload mapping, and execution notes in a human-readable format.

This case highlights UFO3’s capacity for adaptive, resource-aware orchestration, combining multi-device reasoning, real-time environment sensing, and distributed execution under a unified agentic framework. The entire workflow, from information gathering to decision-making and reporting, is autonomously coordinated, underscoring UFO3’s ability to generalize beyond predefined templates and operate effectively in dynamic, real-world computing scenarios.

Summary. Across these three representative cases, UFO3 consistently demonstrates strengths in parallelization, cross-application integration, and adaptive reasoning. It effectively decomposes high-level user goals

[Figure 829]

[Figure 830]

I have a distributed computation task that requires coordinating four Linux hosts. The task consists of: GPU-intensive matrix

[Figure 831]

multiplication, CPU-intensive large dataset processing, memory-intensive data aggregation. Please inspect hardware and current workload on all four hosts, to assign 1 GPU-intensive task, 2 CPU-intensive dataset processing tasks, 1 memoryintensive aggregation task to each of the host. Provide a task assignment report with hosts, hardware, assigned workload, and notes in the Notepad on Windows.

[Figure 832]

[Figure 833]

[Figure 834]

###### Step 1

###### Step 5

- Task A

[Figure 835]

- Task B Task E

- Task A

[Figure 836]

- Task B Task E

[Figure 837]

[Figure 838]

[Figure 839]

###### LinuxAgents

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

Task F

Task F

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

Task C

Task C

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

Task D

Task D

Inspect [Task A, B, C, D]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

- Task A (Linux1): Inspect hardware Linux1
- Task B (Linux2): Inspect hardware Linux2
- Task C (Linux3): Inspect hardware Linux3
- Task D (Linux-GPU): Inspect hardware Linux-GPU
- Task E (Linux1): Generate task assignment report
- Task F (Windows): Write report to Notepad

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

update_task(task_id="Task E", description="""Generate a task assignment report using the following data:

- Linux-1: …
- Linux-2: …
- Linux-3: … Linux-GPU: ...""", …)

LinuxAgent-1

[Figure 888]

[Figure 889]

Assignment [Task E]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

###### Step 6

- Task A

[Figure 894]

- Task B Task E

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

Task F

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

Task C

[Figure 910]

[Figure 911]

[Figure 912]

Task D

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

update_task(task_id="Task F", description="""Open Notepad and write the following complete task assignment report text into it, then save as 'Distributed_Task_Assignment_Report.txt' on the desktop... """, …)

Notepad (Windows)

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

Final Result: All four Linux hosts were inspected and optimally assigned workloads based on their hardware and current utilization. The task assignment report was generated and successfully written to Notepad on the Windows device as 'Distributed_Task_Assignment_Report.txt' on the desktop. The user request is fully satisfied.

- Figure 26: Case 3: Resource-aware orchestration of distributed computation. UFO3 dynamically inspects hardware and workload conditions on four Linux agents, assigns tasks based on available resources, and generates a human-readable summary report on Windows.

into executable subtasks, dispatches them across heterogeneous devices, and synthesizes coherent outputs in natural language or structured files. Together, these results illustrate UFO3’s promise as a general-purpose, LLM-powered orchestration system capable of automating diverse, realistic workflows with high efficiency, transparency, and reliability.

#### 11 Discussion

- 11.1 Scalability and Ecosystem Growth

Although the current UFO3 prototype integrates Windows and Linux device agents, the underlying architecture is inherently platform-agnostic. The AIP and device agent template enable rapid extension to new platforms such as mobile, tablet, or IoT devices. Developers can implement lightweight clients that conform to the AIP specification and seamlessly connect to the orchestrator without modifying the core system.

Moreover, UFO3 orchestrates at the protocol level rather than binding to specific implementations. Any

agent compatible with AIP, whether it is a coding agent Yang et al. (2024), a deep-research agent Huang et al. (2025), or a data-analysis module Zhang et al. (2025c), can participate in the orchestration. This design opens the door for a broader agent ecosystem, where specialized agents cooperate under a unified orchestration fabric. Our long-term vision is to make UFO3 an “agent galaxy”, a shared runtime that interconnects diverse autonomous agents into a cohesive, cross-platform execution universe.

- 11.2 Limitations of Device-Centric Execution

At the core of UFO3 lies the ConstellationAgent, which governs dynamic DAG scheduling and reasoning. However, the system’s overall capability remains constrained by the competence of individual device agents. Since execution is distributed, a single agent’s failure, such as incomplete subtask execution or incorrect result reporting, can cascade through the constellation and lead to global task failure Cemri et al. (2025). This “weakest-link effect” underscores the need for more reliable and self-verifying device agents. Future work will explore automatic capability validation, task-level sandboxing, and adaptive reallocation strategies that allow the orchestrator to detect and recover from unreliable nodes without compromising the consistency of the global task state.

- 11.3 Toward Shared Cross-Device Memory

Currently, agent communication in UFO3 is purely textual, suitable for command exchange but insufficient for data-rich workflows involving images, logs, or binaries. Many cross-device tasks, however, require file transfer or shared context, for example, a Windows agent generating a dataset for a Linux node to process. To support such workflows, we plan to extend AIP with a persistent, shared “agent memory” Zhang et al. (2025f), effectively a distributed workspace where agents can store and retrieve intermediate results. This enhancement would unify context across devices, simplify file sharing, and enable richer collaborative behavior across heterogeneous agents.

##### 12 Related Work We review related research on digital device agents and multi-agent orchestration systems.

###### 12.1 Digital Device Agents

The emergence of large language models, particularly multimodal variants, has enabled a new class of intelligent agents capable of operating directly on digital devices Zhang et al. (2024). These agents extend beyond web and mobile environments to desktops, and are now expanding into domains such as automotive and embedded systems.

Early systems like SeeAct Zheng et al. (2024), Mobile-Agent Wang et al. (2024a), and UFO Zhang et al. (2025b) pioneered this direction by leveraging GPT-4V Yang et al. (2023) to perceive graphical user interfaces and perform human-like interactions across different platforms. Subsequent research has evolved along two main trajectories, namely (i) system-level integration, enhancing robustness through tighter coupling with native OS and API layers Zhang et al.; 2025a); and (ii) model-level adaptation, fine-tuning LLMs with large-scale interaction data and reinforcement learning to improve grounding and autonomy in digital environments Luo et al. (2025); Zheng et al. (2025); Wang et al. (2025).

Industry adoption has accelerated rapidly. Anthropic’s Claude (Computer Use) Anthropic (2024) relies purely on screenshot-based perception, while OpenAI’s Operator OpenAI (2025) demonstrates strong multimodal reasoning and robust desktop automation. These advances collectively signal the formation of a new computing paradigm where LLMs serve as universal control interfaces for heterogeneous software systems.

Despite this progress, most existing device agents remain confined to isolated machines, limiting context sharing and collaborative capability. UFO3 addresses this limitation by introducing cross-device orchestration. Through the AIP, UFO3 unifies disparate agents into a coherent constellation, enabling seamless coordination, shared state, and large-scale cooperative intelligence across devices.

###### 12.2 Multi-Agent Orchestration

Recent years have witnessed growing interest in LLM-based multi-agent systems, where coordination among heterogeneous agents becomes central to achieving emergent intelligence Guo et al. (2024). Designing robust orchestration protocols, covering role assignment, communication, collaboration, and context sharing, has been identified as a key challenge Bhatt et al. (2025); Dang et al. (2025); Kong et al. (2025).

The Internet of Agents (IoA) Chen et al. represents one of the earliest structured attempts in this direction. It introduces an integration protocol and an instant-messaging-style communication layer, enabling flexible, scalable multi-agent collaboration. IoA emphasizes dynamic teaming and conversation flow control, allowing agents to self-organize and cooperate fluidly in an Internet-like environment Yang et al. (2025b). Building upon this idea, the Federation of Agents (FoA) Giusti et al. (2025) further advances distributed orchestration with Versioned Capability Vectors (VCVs) as machine-readable profiles that encode agent capabilities, costs, and constraints. These semantic embeddings make capabilities discoverable and composable, enabling dynamic, capability-driven coordination at scale.

In contrast, UFO3 extends this vision into the realm of heterogeneous digital devices. It provides a unified, extensible orchestration substrate that connects agents distributed across desktops, browsers, and operating systems. Through the AIP, UFO3 not only facilitates low-latency cross-device communication and shared context propagation but also enables composable construction of agent teams. This design transforms isolated device agents into a coherent digital ecosystem, scalable, evolvable, and self-organizing.

#### 13 Conclusion

We presented UFO3, a cross-device orchestration system that transforms LLM agents from isolated executors into coordinated collaborators. At its core, UFO3 introduces the mutable TaskConstellation abstraction, which tightly integrates planning and execution: the ConstellationAgent continuously synthesizes and edits DAGs; the Constellation Orchestrator schedules tasks asynchronously and applies safe, batched updates while enforcing invariants that guarantee single assignment and DAG acyclicity; and the Agent Interaction Protocol (AIP) provides persistent, low-latency, and resilient communication across heterogeneous devices. A layered device-agent template, backed by MCP servers, allows new endpoints to join seamlessly without modifying the control plane.

Our evaluation on NebulaBench, comprising 55 representative tasks across 5 machines and 10 categories, demonstrates that UFO3 reliably orchestrates heterogeneous workflows: achieving 83.3% subtask completion, 70.9% task success, an average parallelism ratio of 1.72, and 243.7 s mean end-to-end latency, 31% faster than a sequential baseline thanks to effective orchestration–execution overlap. Fault-injection experiments further confirm UFO3’s robustness: it gracefully recovers from transient agent failures, preserves partial progress under persistent outages, and conservatively terminates under global failures.

Overall, UFO3 lays the foundation for the next generation of autonomous multi-agent systems, enabling rich memory sharing, seamless coordination, and expansive device integration. By weaving heterogeneous devices and intelligent agents into a unified fabric, UFO3 brings us closer to an adaptive and reliable ecosystem of digital assistants that act collectively as a super-agent to harness ubiquitous intelligence.

#### References

Argo workflows. https://github.com/argoproj/argo-workflows. accessed: 2025-10-28. Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku, 2024. URL https:

//www.anthropic.com/news/3-5-models-and-computer-use. Accessed: 2024-10-26.

Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, et al. Graphs meet ai agents: Taxonomy, progress, and future opportunities. arXiv preprint arXiv:2506.18019, 2025.

Umang Bhatt, Sanyam Kapoor, Mihir Upadhyay, Ilia Sucholutsky, Francesco Quinzan, Katherine M Collins, Adrian Weller, Andrew Gordon Wilson, and Muhammad Bilal Zafar. When should we orchestrate multiple agents? arXiv preprint arXiv:2503.13577, 2025.

Frederik Brudy, Joshua Kevin Budiman, Steven Houben, and Nicolai Marquardt. Investigating the role of an overview device in multi-device collaboration. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, pp. 1–13, 2018.

Frederik Brudy, Christian Holz, Roman Rädle, Chi-Jui Wu, Steven Houben, Clemens Nylandsted Klokmose, and Nicolai Marquardt. Cross-device taxonomy: Survey, opportunities and challenges of interactions spanning across multiple devices. In Proceedings of the 2019 chi conference on human factors in computing systems, pp. 1–28, 2019.

Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, et al. Why do multi-agent llm systems fail? arXiv preprint arXiv:2503.13657, 2025.

Chi-Wen Chen and Marios Koufaris. Multi-device use: understanding the motivations behind switching between multiple devices during a task. International Journal of Human–Computer Interaction, 36(12): 1178–1193, 2020.

Weize Chen, Ziming You, Ran Li, Chen Qian, Chenyang Zhao, Cheng Yang, Ruobing Xie, Zhiyuan Liu, Maosong Sun, et al. Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence. In The Thirteenth International Conference on Learning Representations.

Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, et al. Multi-agent collaboration via evolving orchestration. arXiv preprint arXiv:2505.19591, 2025.

Paolo Di Tommaso, Maria Chatzou, Evan W Floden, Pablo Prieto Barja, Emilio Palumbo, and Cedric Notredame. Nextflow enables reproducible computational workflows. Nature biotechnology, 35(4):316–319, 2017.

Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of thoughts: Defying the law of penrose triangle for thought generation. In ACL (Findings), 2024.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.

Qiang Duan and Zhihui Lu. Agent communications toward agentic ai at edge–a case study of the agent2agent protocol. arXiv preprint arXiv:2508.15819, 2025.

Abul Ehtesham, Aditi Singh, Gaurav Kumar Gupta, and Saket Kumar. A survey of agent interoperability protocols: Model context protocol (mcp), agent communication protocol (acp), agent-to-agent protocol (a2a), and agent network protocol (anp). arXiv preprint arXiv:2505.02279, 2025.

Lorenzo Giusti, Ole Anton Werner, Riccardo Taiello, Matilde Carvalho Costa, Emre Tosun, Andrea Protani, Marc Molina, Rodrigo Lopes de Almeida, Paolo Cacace, Diogo Reis Santos, et al. Federation of agents: A semantics-aware communication fabric for large-scale agentic ai. arXiv preprint arXiv:2509.20175, 2025.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. arXiv preprint arXiv:2503.23278, 2025.

Steven Houben, Nicolai Marquardt, Jo Vermeulen, Clemens Klokmose, Johannes Schöning, Harald Reiterer, and Christian Holz. Opportunities and challenges for cross-device interactions in the wild. interactions, 24

(5):58–63, 2017.

Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey. arXiv preprint arXiv:2402.02716, 2024.

Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al. Deep research agents: A systematic examination and roadmap. arXiv preprint arXiv:2506.18096, 2025.

Mohammad Islam, Angelo K. Huang, Mohamed Battisha, Michelle Chiang, Santhosh Srinivasan, Craig Peters, Andreas Neumann, and Alejandro Abdelnur. Oozie: towards a scalable workflow management system for hadoop. In Proceedings of the 1st ACM SIGMOD Workshop on Scalable Workflow Execution Engines and Technologies (SWEET@SIGMOD). ACM, 2012. doi: 10.1145/2443416.2443420.

Yuxuan Jiang, Chaoyun Zhang, Shilin He, Zhihao Yang, Minghua Ma, Si Qin, Yu Kang, Yingnong Dang, Saravan Rajmohan, Qingwei Lin, et al. Xpert: Empowering incident management with query recommendations via large language models. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pp. 1–13, 2024.

Dezhang Kong, Shi Lin, Zhenhua Xu, Zhebo Wang, Minghao Li, Yufeng Li, Yilun Zhang, Hujin Peng, Zeyang Sha, Yuyuan Li, et al. A survey of llm-driven ai agent communication: Protocols, security risks, and defense countermeasures. arXiv preprint arXiv:2506.19676, 2025.

Xiang Li, Mustafa Abdallah, Shikhar Suryavansh, Mung Chiang, Kwang Taik Kim, and Saurabh Bagchi. Dag-based task orchestration for edge computing. In 2022 41st International Symposium on Reliable Distributed Systems (SRDS), pp. 23–34. IEEE, 2022.

Run Luo, Lu Wang, Wanwei He, Longze Chen, Jiaming Li, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.

Stefan Marks and David White. Multi-device collaboration in virtual environments. In Proceedings of the 2020 4th International Conference on Virtual and Augmented Reality Simulations, pp. 35–38, 2020.

Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang, Bo Qiao, Kartik Mathur, Qianhui Wu, Yuhang Xie, Xiaojun Ma, Mengyu Zhou, et al. Gui-360: A comprehensive dataset and benchmark for computer-using agents. arXiv preprint arXiv:2511.04307, 2025.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology, 16(5):1–72, 2025.

Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S Yu, et al. A survey of webagents: Towards next-generation ai agents for web automation with large foundation models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 6140–6150, 2025.

OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world.

2025. URL https://openai.com/index/computer-using-agent.

OpenAI. Gpt-5 system card. Technical report, OpenAI, August 2025. URL https://cdn.openai.com/ gpt-5-system-card.pdf. Accessed: 2025-10-29.

Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang, Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, et al. Taskweaver: A code-first agent framework. arXiv preprint arXiv:2311.17541, 2023.

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. Frontiers of Computer Science, 19(8):198343, 2025.

Fred B Schneider. The state machine approach: A tutorial. Fault-tolerant distributed computing, pp. 18–41, 2005.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322, 2025.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025.

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024a.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024b.

Lu Wang, Fangkai Yang, Chaoyun Zhang, Junting Lu, Jiaxu Qian, Shilin He, Pu Zhao, Bo Qiao, Ray Huang, Si Qin, et al. Large action models: From inception to implementation. arXiv preprint arXiv:2412.10047, 2024c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024a.

Yue Wu, Yewen Fan, So Yeon Min, Shrimai Prabhumoye, Stephen McAleer, Yonatan Bisk, Ruslan Salakhutdinov, Yuanzhi Li, and Tom Mitchell. Agentkit: structured llm reasoning with dynamic graphs. arXiv preprint arXiv:2404.11483, 2024b.

Yajie Xu, Lu Wang, Yanning Xu, Siyuan Qiu, Maopu Xu, and Xiangxu Meng. Cross-device task interaction framework between the smart watch and the smart phone. Personal and Ubiquitous Computing, 25(6): 1039–1049, 2021.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Yingxuan Yang, Huacan Chai, Yuanyi Song, Siyuan Qi, Muning Wen, Ning Li, Junwei Liao, Haoyi Hu, Jianghao Lin, Gaowei Chang, et al. A survey of ai agent protocols. arXiv preprint arXiv:2504.16736, 2025a.

Yingxuan Yang, Mulei Ma, Yuxuan Huang, Huacan Chai, Chenyu Gong, Haoran Geng, Yuanjian Zhou, Ying Wen, Meng Fang, Muhao Chen, et al. Agentic web: Weaving the next web with ai agents. arXiv preprint arXiv:2507.21206, 2025b.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Junwei Yu, Yepeng Ding, and Hiroyuki Sato. Dyntaskmas: A dynamic task graph-driven framework for asynchronous and parallel llm-based multi-agent systems. arXiv preprint arXiv:2503.07675, 2025.

Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Api agents vs. gui agents: Divergence and convergence. In ICML 2025 Workshop on Computer Use Agents.

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024.

Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, et al. Ufo2: The desktop agentos. arXiv preprint arXiv:2504.14603, 2025a.

Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, et al. Ufo: A ui-focused agent for windows os interaction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 597–622, 2025b.

Chaoyun Zhang, Zicheng Ma, Yuhao Wu, Shilin He, Si Qin, Minghua Ma, Xiaoting Qin, Yu Kang, Yuyi Liang, Xiaoyu Gou, et al. Allhands: Ask me anything on large-scale verbatim feedback via large language models. In 2025 IEEE 41st International Conference on Data Engineering (ICDE), pp. 43–57. IEEE Computer Society, 2025c.

Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pp. 1–20, 2025d.

Linghao Zhang, Shilin He, Chaoyun Zhang, Yu Kang, Bowen Li, Chengxing Xie, Junhao Wang, Maoquan Wang, Yufan Huang, Shengyu Fu, et al. Swe-bench goes live! arXiv preprint arXiv:2505.23419, 2025e.

Wenxiao Zhang, Huber Flores, et al. Towards collaborative multi-device computing. In 2018 IEEE International Conference on Pervasive Computing and Communications Workshops (PerCom Workshops), pp. 22–27. IEEE, 2018.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47, 2025f.

Yu Zhao, Wei-Ning Chen, Huseyin Atahan Inan, Samuel Kessler, Lu Wang, Lukas Wutschitz, Fangkai Yang, Chaoyun Zhang, Pasquale Minervini, Saravan Rajmohan, et al. Learning gui grounding with spatial reasoning from visual feedback. arXiv preprint arXiv:2509.21552, 2025.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Vem: Environment-free exploration for training gui agent with value environment model. arXiv preprint arXiv:2502.18906, 2025.

- A Formal Guarantees and Model State and Transitions. We model the orchestrator as an asynchronous transition system with state

𝜎 = (𝐶, 𝑠, 𝐴, 𝐿, 𝑄, 𝐷),

where 𝐶 = (𝑉, 𝐸) is the task-constellation DAG over a finite task universe 𝑇 (𝑉 ⊆ 𝑇, 𝐸 ⊆ 𝑉 × 𝑉); 𝑠 : 𝑉 → {PENDING, RUNNING, COMPLETED, FAILED} is the task-state map; 𝐴 : 𝑉 ⇀ D is the partial device assignment over devices D; 𝐿 ∈ {free, held} is the global edit/assignment lock; 𝑄 ∈ Seq(E) is the pending event queue over E = {⟨TASK_COMPLETED, 𝑡⟩, ⟨TASK_FAILED, 𝑡⟩ | 𝑡 ∈ 𝑉}; and 𝐷 ⊆ D is the set of available devices. A task 𝑡 is ready in (𝐶, 𝑠) iff

ready𝐶(𝑠, 𝑡) ≡ 𝑠(𝑡) = PENDING ∧ ∀(𝑢, 𝑡) ∈𝐸. 𝑠(𝑢) = COMPLETED.

The system alternates between a lock-bounded editing phase (apply Δ, validate, synchronize, publish) and an unlocked scheduling phase.

###### Inductive Invariants.

- • I1 (Single assignment during run). 𝐴 is a partial function 𝑉 ⇀ D; if 𝑠(𝑡) = RUNNING then 𝐴(𝑡) ∈ 𝐷 and 𝐴(𝑡) does not change while 𝑡 remains RUNNING.
- • I2 (Acyclic consistency). 𝐶 remains a DAG after every committed edit.
- • I3 (Edit locality / immutability). Within one edit cycle, edits may modify only PENDING tasks and edges whose endpoints are PENDING; RUNNING/COMPLETED/FAILED nodes and their incident edges are immutable.

Safety. Each edit–validate–synchronize–publish–release cycle is linearized as a single atomic operation with a linearization point between publishing the change and releasing the lock. As dispatch occurs only when 𝐿 = free, all assignments are taken w.r.t. a validated DAG. A standard induction over the transition relation shows that I1–I3 are preserved.

Liveness and Deadlock Freedom. Let 𝑅 = {𝑡 ∈ 𝑉 | 𝑠(𝑡) = RUNNING} at lock acquisition. While 𝐿 = held, no new RUNNING tasks are created, so only completions/failures from 𝑅 can arrive. With the variant Φ(𝜎) = |𝑄| + |𝑅|, every sync or completion event decreases Φ; hence the edit phase terminates and the lock is released. Under weak fairness of event delivery and device availability, every perpetually ready task is eventually dispatched.

Edit–Sync Confluence. Let Apply(𝐶, Δ) be an edit that respects I3 and preserves acyclicity, and let Sync(𝑠, 𝐸) fold a multiset 𝐸 of completion/failure events into 𝑠 via a per-task monotone, idempotent join. Because Apply leaves 𝑠 unchanged and Sync leaves 𝐶 unchanged, and they operate on disjoint components (I3), the two maps commute within the same lock interval; the observable post-state (𝐶′, 𝑠′) is independent of whether events are folded before or after Δ.

###### A.1 TLA+ Specification

Scope. The spec below matches the model above. For model-checking small instances, we stub Acyclic as TRUE and Apply/Synchronize as no-ops; safety is still meaningfully exercised (I1, I2), and the environment is bounded by a queue-length predicate QueueBound. Fairness assumptions are embedded in Spec.

---- MODULE Orchestrator ---EXTENDS Naturals , Sequences

CONSTANTS TASKS , DEVICES

TaskStates == {"PENDING", "RUNNING", "COMPLETED", "FAILED"} TaskEvents == {"TASK_COMPLETED", "TASK_FAILED"}

VARIABLES C, S, A, L, Q, D QLen == Len(Q) QueueBound == QLen <= 2 NULL == "NULL" Acyclic(g) == TRUE IsDAG(C_) == Acyclic(C_) Ready(t) ==

/\ t \in C.V /\ S[t] = "PENDING" /\ \A u \in C.V : <<u, t>> \in C.E => S[u] = "COMPLETED"

TypeOK == /\ C \in [V : SUBSET TASKS , E : SUBSET (TASKS \X TASKS)] /\ S \in [TASKS -> TaskStates] /\ A \in [TASKS -> (DEVICES \cup {NULL})] /\ L \in {"free", "held"} /\ Q \in Seq(TaskEvents) /\ D \subseteq DEVICES

I1 == \A t \in C.V : (S[t] = "RUNNING" => A[t] \in DEVICES) I2 == IsDAG(C)

LockFree == (L = "free") LockHeld == (L = "held")

Init == /\ C = [V |-> TASKS , E |-> {}] /\ S = [t \in TASKS |-> "PENDING"] /\ A = [t \in TASKS |-> NULL] /\ L = "free" /\ Q = << >> /\ D = DEVICES /\ TypeOK

Enqueue ==

\E e \in TaskEvents : /\ Q’ = Append(Q, e) /\ UNCHANGED <<C, S, A, L, D>>

Acquire == /\ LockFree /\ L’ = "held" /\ UNCHANGED <<C, S, A, Q, D>>

Release == /\ LockHeld /\ L’ = "free" /\ UNCHANGED <<C, S, A, Q, D>>

Edges(C_, t) == { u \in C_.V : (<<u, t>> \in C_.E) \/ (<<t, u>> \in C_.E) } Apply(C_, Delta , e) == C_

Synchronize(S_, C1) == S_ EditStep ==

/\ LockHeld /\ Len(Q) > 0 /\ LET e == Head(Q) IN

LET Q1 == Tail(Q) IN \E Delta \in {0},

C1 \in [V : SUBSET TASKS , E : SUBSET (TASKS \X TASKS)], S1 \in [TASKS -> TaskStates] : /\ C1 = Apply(C, Delta , e) /\ IsDAG(C1) /\ S1 = Synchronize(S, C1) /\ \A t \in C.V :

(S[t] \in {"RUNNING","COMPLETED","FAILED"} => /\ S1[t] = S[t] /\ Edges(C1, t) = Edges(C, t))

/\ C’ = C1 /\ S’ = S1 /\ Q’ = Q1 /\ UNCHANGED <<A, L, D>>

DrainOrNoop == \/ (LockHeld /\ Len(Q) = 0 /\ UNCHANGED <<C, S, A, L, Q, D>>) \/ EditStep

Dispatch ==

\E t \in C.V, d \in D : /\ LockFree /\ Ready(t) /\ A[t] = NULL /\ S’ = [S EXCEPT ![t] = "RUNNING"] /\ A’ = [A EXCEPT ![t] = d] /\ UNCHANGED <<C, L, Q, D>>

UpdateDevices ==

\E Dnew \in SUBSET DEVICES : /\ D’ = Dnew /\ UNCHANGED <<C, S, A, L, Q>>

Noop == UNCHANGED <<C, S, A, L, Q, D>> Next ==

\/ Enqueue \/ Acquire \/ DrainOrNoop \/ Release \/ Dispatch \/ UpdateDevices \/ Noop

vars == <<C, S, A, L, Q, D>> Spec ==

/\ Init /\ [][Next]_vars /\ WF_vars(Dispatch) /\ WF_vars(DrainOrNoop) /\ WF_vars(Release)

THEOREM Spec => [](TypeOK /\ I1 /\ I2)

====

###### A.2 Model-Checking Configuration and Results

Configuration. We evaluate safety and deadlock-freedom on small instances with a bounded event window (§A). The queue bound is expressed in the specification via QueueBound; we constrain it in the model as follows.

|-- Orchestrator.cfg SPECIFICATION Spec<br><br>CONSTANTS TASKS = {t0, t1, t2} DEVICES = {"dev0", "dev1", "dev2"}<br><br>CONSTRAINT QueueBound INVARIANTS TypeOK I1 I2<br><br>CHECK_DEADLOCK TRUE<br><br>|
|---|

Results. TLC completes exploration under the above configuration with the following summary:

- • 93,633 states generated; 7,168 distinct states; queue empty at fixpoint.
- • Graph search depth: 8; average outdegree: 1 (min 0, max 19, 95th percentile 8).
- • Action-level distinct states: Init 1, Enqueue 6, Acquire 448, Dispatch 441, UpdateDevices 6,272; others 0.
- • Fingerprint collision probability: 3.4 × 10−11.

All invariants (TypeOK, I1, I2) hold and TLC reports no deadlocks.

Reproducibility notes. Weak fairness on Dispatch, DrainOrNoop, and Release is enabled in Spec. The QueueBound predicate enforces the bounded-window assumption used in the liveness argument. For larger instances or unbounded event ingress, the state space grows quickly; bounding |𝑄| and fixing the device set 𝐷 are standard ways to reflect production ingress control and avoid spurious divergence during model checking.

#### B AIP Message Schema Reference

To support the persistent, event-driven orchestration described in Section 7, the Agent Interaction Protocol (AIP) defines a compact set of typed message primitives that unify communication across the ConstellationClient, device agent services, and device clients. Each message carries explicit directionality (Client → Server or Server → Client), well-defined key fields, and structured reliability hooks for deterministic orchestration and safe recovery.

MsgType Dir. Key Fields Semantics Idempotent? Expected Response Reliability Hooks REGISTER C→S client_id, meta-

Declare presence + capabilities

Yes HEARTBEAT(OK) or ERROR

Validation + timeout

data

TASK C→S request, session_id

Begin/extend session task

Limited COMMAND / TASK_END ack

Session guard, queue fallback

COMMAND S→C actions[], response_id

Deterministic batch execution unit

No COMMAND_RESULTS Timeout per action,

ordering preserved COMMAND_RESULTS C→S action_results[],

Return per-command outcomes

Yes Next COMMAND or TASK_END

Correlation, partial fail surfacing

prev_response_id

TASK_END S→C status, result/error

Terminalization of session task

Yes Optional TASK_END ack Cancellation, recon-

nection flush HEARTBEAT C/S timestamp Liveness probe + la-

Yes HEARTBEAT ack (opposite dir.)

HeartbeatManager jitter control

tency sampling

DEVICE_INFO_REQUEST C→S target_id, request_id

On-demand profile refresh

Yes DEVICE_INFO_RESPONSE TimeoutManager fall-

back DEVICE_INFO_RESPONSE S→C result, re-

Canonicalize device system info

Yes None Profile versioning

sponse_id

ERROR C/S error, context Protocol or execution anomaly

N/A Operator / scheduler handling

Rapid failure propagation

- Table 5: AIP message taxonomy and reliability semantics. C=Client, S=Server. Idempotent? indicates whether duplicate delivery produces the same state (REGISTER acknowledged again, HEARTBEAT updates freshness) or is safely ignored; non-idempotent messages (e.g., COMMAND) must not be replayed without coordination.

Discussion. The schema defines the canonical message types underlying AIP’s layered design (Section 7). REGISTER, DEVICE_INFO, and HEARTBEAT correspond to the Profile and Resilience Layers, ensuring freshness and liveness (G3, G4); TASK, COMMAND, and COMMAND_RESULTS implement deterministic execution semantics within the Execution Control Layer (G1, G5); and ERROR provides the recovery hooks required for extensibility and robustness (G6). Together, these primitives form the minimal yet expressive backbone that enables UFO3’s distributed, evolution-tolerant orchestration fabric.

C Details of NebulaBench

- Table 6 provides a comprehensive listing of all queries in NebulaBench, including their functional category, difficulty, the devices involved, and the observed outcomes of UFO3’s execution. Including this detailed appendix serves multiple purposes: it allows readers to (i) understand the specific nature and distribution of tasks in NebulaBench, (ii) examine UFO3’s performance at the granularity of individual queries, and (iii) enable reproducibility and comparison for future research on multi-agent orchestration and cross-device automation.

Table 6: Full NebulaBench task listing with metadata and UFO3 execution results.

###### ID Category Task Description Difficulty Devices Success

- 1 Logs Retrieve all warning and error logs from Linux servers, add them to the ’report’ sheet of the log_detailed Excel file, and send an email with the report to the operations engineer.

Hard 4.0 Yes

- 2 Logs Search auth.log for failed SSH on all linux; create top-3 offending IPs markdown report on linux-1

Medium 3.0 Yes

- 3 Logs Rotate mock_app.log if >1MB on all linux and verify shrink

Medium 3.0 Yes

- 4 Logs Scan dmesg for OOM events on linux and return counts

Medium 3.0 Yes

Continued on next page

- 5 Logs Export Windows app logs and compare with Linux WARN/ERROR counts

Medium 4.0 Yes

- 6 Logs Count how many times each systemd service was started or stopped in the last 24 hours on linux and write the results on the log_detailed excel

Easy 4.0 Yes

- 7 Sys Detect CPU models and write best host to Notepad Easy 4.0 Yes

- 8 Sys Set DEMO_ENV=staging on linux-1,2 and windows; verify JSON output”

Medium 3.0 No

- 9 Sys Ensure sandbox user exists and sudo member Easy 3.0 Yes

- 10 Sys Ensure backups folder 750 ops:ops on all linux Easy 3.0 Yes

- 11 Sys Set Windows reg flag + Linux file; confirm both paths

Easy 2.0 No

- 12 Proc Restart cron.service and record its status Easy 2.0 Yes

- 13 Proc Stop cron.service on all linux Easy 3.0 Yes

- 14 Proc Run long_job.sh concurrently on linux 1-3 and report their running time on notepad

Medium 4.0 Yes

- 15 Proc Get task schedule on schedule.xlsx on Windows and assign corresponding task to each linux server

Hard 4.0 No

- 16 Proc Create hello.timer hourly on all linux, make sure it takes effect

Hard 3.0 Yes

- 17 Data Sum today’s values from data.csv on linux-a/b; JSON output

Hard 4.0 Yes

- 18 Data On Linux-1, Linux-2, and Linux-3, recursively scan and read their the ~/ directory on each machine for

.sh files. Generate a Markdown summary table in Windows Notepad showing the file name and code summary.

Medium 4.0 No

- 19 Data Average durations from CSV on win+linux; append metrics

Medium 4.0 Yes

- 20 Data List /etc files on linux modified in 48h and write to the log_detailed excel”

Medium 4.0 Yes

- 21 DevOps On all linux, run a locally available container image and ensure the container passes the health check, then close it.

Hard 3.0 Yes

- 22 DevOps On Linux-1, clone the repository https://github.com/microsoft/UFO.git, build a Docker image named ufo:test, and push it to a shared Docker registry running on Linux-2. Then, from Linux-3, pull the same image and run a container to verify that the application starts successfully.”

Hard 3.0 No

- 23 DevOps On all Linux, list all Docker images. Then, generate a Markdown summary on Windows Notepad showing the image list per host and highlight any differences.

Hard 4.0 Yes

Continued on next page

- 24 DevOps On Linux-1, clone the repository https://github.com/microsoft/UFO/ On Windows, UFO2 branch is already checked out in VSCode. Compare the UFO2 branch with Linux-1–s main branch to check if it can be merged cleanly. Report whether the merge is clean or if conflicts exist, without modifying either branch.

Hard 2.0 No

- 25 DevOps Collect performance metrics from three Linux hosts. Based on the performance results, deploy three different microservices from their dev_path: Deploy service_1.py (lightweight) on the host with the lowest CPU load. Deploy service_2.py (medium) on the host with moderate load. Deploy service_3.py (heavy) on the host with the highest available memory. After deployment, verify that all three services are running and reachable via HTTP from each Linux node. Finally, generate a deployment report on windows notepad.

Hard 4.0 Yes

- 26 DevOps Ensure requests repo are up to date on main branch on all linux servers

Medium 3.0 Yes

- 27 DevOps clone https://github.com/psf/requests on all linux server on their dev path, set up the virtual environment with their agent name and install all dependencies there

Hard 3.0 Yes

- 28 DevOps Set up a jupyer lab on linux 1 and open it with ip url on Windows browser

Medium 2.0 No

- 29 DevOps Close all running jupyer lab linux 1. Easy 1.0 Yes

- 30 DevOps stop all running service on all linux Easy 3.0 Yes

- 31 Net Get the IP from all linux server, open the port of 8001 for linux1, 8002 for linux2, 8003 for linux3, then on each Linux server, perform ping tests to all other Linux servers on their open ports, write the results to the log_detailed excel on Windows.

Hard 4.0 Yes

- 32 Net From Windows, check if the domain intranet.local can be resolved on both Linux-1 and Linux-2. If the domain cannot be resolved, add a temporary DNS entry for intranet.local pointing to Linux-3–s IP address in /etc/hosts on both Linux-1 and Linux-2, then verify resolution again from Windows using ping intranet.local”

Medium 3.0 No

- 33 Net On Linux-1,check whether port 8080 is listening using ss or netstat. If it is closed, create a web service listen to this port, and, verify that it is running and listening on port 8080 from linux-2.

Easy 2.0 Yes

Continued on next page

- 34 Net On Linux-1, check whether http://localhost:9090/health returns a 200 OK response. If the request fails or port 9090 is closed, run a lightweight HTTP container (e.g., nginx:alpine) exposing port 9090, verify that /health returns 200 OK locally and from Linux-2, then stop and remove the container.

Medium 2.0 Yes

- 35 Net Test scp file transfers between every pair of the four Linux nodes to ensure they can send and receive files to each other successfully. After completing all pairwise tests, write a summary report on Windows Notepad.

Easy 5.0 No

- 36 Browsing Use the Windows browser to search for the latest stable Python release URL, download the installer, and then remotely copy and install it on Linux-a, Linux-b, and Linux-c. After installation, verify Python version on each host

Hard 4.0 No

- 37 Browsing For all linux, get their disk usage statistics. Then, from Windows browser, search for the top 3 recommended ways to reduce high disk usage for Linux systems and document these in a report on notepad.

Medium 4.0 Yes

- 38 Browsing Run a cpu_bench.py on all Linux. Collect results and, using Windows browser, search for recommended CPU optimization techniques for Linux servers. Create a final report on Notepad comparing benchmark results with recommended optimizations.

Easy 4.0 Yes

- 39 Browsing Use Windows browser to download a CSV dataset of historical weather data. Copy the file to all Linux. Then, on each Linux host, compute the average temperature using a Python or shell script and save the result as weather_avg.txt

Medium 4.0 No

- 40 Browsing From Windows browser, search for latest security CVEs for a specific Linux package. Then, on all Linux, check installed version, compare with CVE advisory, and if vulnerable, apply the patch or upgrade. Confirm patch applied successfully.

Hard 4.0 Yes

- 41 Orchestration Complete all tasks on in schedule.xlsx on windows on linux servers, and write back the one-sentence result summary to the Result Summary column.

Hard 4.0 Yes

- 42 Orchestration Choose lowest load host, run long_job.sh there Easy 3.0 Yes

- 43 Orchestration Check disk free <10% on linux; print OK/ALERT” Easy 3.0 Yes

- 44 Orchestration Run cpu_bench.py on all linux hosts, and write the results to notepad on Windows

Medium 4.0 Yes

- 45 Orchestration Create Windows hosts_summary.txt of all Linux kernels

Easy 4.0 No

Continued on next page

- 46 GPU check the gpu availability on the GPU node, and run the gpu_smoke.py. Summarize the results on the Notepad on Windows.

Medium 2.0 Yes

- 47 GPU Use scp to transfer the log 1, 2, 3 from each linux machine to the gpu node and merge them to a single log_dataset.txt, then run the training.sh script.

Hard 5.0 No

- 48 GPU ransfer the log 1, 2, 3 from each linux machine to the gpu node and merge them to a single log_dataset.txt, then run the training.sh script.

Hard 4.0 Yes

- 49 GPU I have a distributed computation task that requires coordinating four Linux hosts. The task consists of: GPU-intensive matrix multiplication, CPU-intensive large dataset processing, Memory-intensive data aggregation. Please inspect hardware and current workload on all four hosts, to assign 1 GPU-intensive task, 2 CPU-intensive dataset processing tasks, 1 memory-intensive aggregation task to each of the host. Provide a task assignment report with hosts, hardware, assigned workload, and notes in the Notepad on Windows.

Hard 5.0 Yes

- 50 GPU Distributedly process log files 1, 2, and 3 from each Linux machine into a format suitable for next-token prediction training. Then, merge all processed outputs into a single log_dataset.txt file on the GPU node, and execute the training.sh script.

Hard 4.0 No

- 51 Negative Start ufo3 service on all linux Easy 3.0 No

- 52 Negative Deploy the service_1.py on a linux machine with 2TB memory.

Medium 0.0 Yes

- 53 Negative Install the cuda for the linux with NVIDIA H100 GPU

Medium 3.0 Yes

- 54 Negative Send a message to Zac on Wechat Easy 0.0 Yes

- 55 Negative visualize ufo3.png on linux Easy 3.0 No

