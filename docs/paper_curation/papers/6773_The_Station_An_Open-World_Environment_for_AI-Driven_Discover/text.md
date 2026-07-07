# arXiv:2511.06309v2[cs.AI]1Dec2025

## The STATION

#### An Open-World Environment for AI-Driven Discovery

###### Stephen Chung1,2 Wenyu Du1,3

1 DualverseAI

- 2 University of Cambridge
- 3 University of Hong Kong

We introduce the Station, an open-world multi-agent environment for autonomous scientific discovery. The Station simulates a complete scientific ecosystem, where agents can engage in long scientific journeys that include reading papers from peers, formulating hypotheses, collaborating with peers, submitting experiments, and publishing results. Importantly, there is no centralized system coordinating their activities. Utilizing their long context, agents are free to choose their own actions and develop their own narratives within the Station. Experiments demonstrate that AI agents in the Station achieve new state-of-the-art performance on a wide range of benchmarks, spanning mathematics, computational biology, and machine learning, notably surpassing AlphaEvolve in circle packing. A rich tapestry of unscripted narratives emerges, such as agents collaborating and analyzing other works rather than pursuing myopic optimization. From these emergent narratives, novel methods arise organically, such as a new density-adaptive algorithm for scRNA-seq batch integration that borrows concepts from another domain. The Station marks a first step towards autonomous scientific discovery driven by emergent behavior in an open-world environment, representing a new paradigm that moves beyond rigid pipelines.

Date: 11-Nov-2025 Correspondence: info@dualverse.ai

#### 1 Introduction

Artificial intelligence (AI) has been advancing rapidly, showing early promise in contributing to scientific research. Recent studies demonstrate that large language models (LLMs) can perform guided code perturbations using approaches such as evolutionary methods [1] or tree-search algorithms [2]. However, these methods resemble a rigid factory pipeline: a central manager selects a baseline, instructs an LLM to propose a single improvement, evaluates it against a fixed metric, and then terminates the session. This stateless, top-down process constrains openness and creativity in scientific discovery.

As an alternative to the centralized paradigm, we introduce the Station∗, an open-world multi-agent environment that simulates a scientific ecosystem. Here, open-world refers to a setting where agents possess full autonomy to select actions, such as conducting literature reviews or communicating with peers, rather than following scripted pipelines. The Station contains a large number of rooms, each serving a distinct function, and agents can freely traverse between them. For example, agents can read and publish papers in the Archive Room, write messages to peers in the Mail Room for collaboration, or write and execute code at the Research Counter. This high degree of freedom makes the Station a miniature world in which agents explore, create, and collaborate.

Experiments show that AI agents in the Station make significant advances, achieving state-of-the-art (SOTA) performance on a diverse set of established benchmarks. The key results are summarized in Table 1. These benchmarks span diverse domains, covering mathematics, machine learning, and biology, and represent tasks of distinct natures, including network architecture design, optimization algorithm development, and data

∗Source code available at https://github.com/dualverse-ai/station

Table 1: Summary of the Station’s top-performing methods on key benchmarks.

Task Station’s Results Previous SOTA Method Highlights Mathematics

Circle Packing 2.93957 (n=32) 2.63598 (n=26)

2.93794 (AlphaEvolve [1]) 2.63586 (AlphaEvolve [1])

Unified MM–LP Adaptive Search

Biology Batch Integration 0.5877 score 0.5867 (LLM-TS [2]) Density-adaptive quotas RNA Modeling 66.3 ± 0.1% score 63.4 ± 0.2% (Lyra [3]) Contextual positional embeddings ZAPBench 26.37 ± 0.03 × 10−3

26.62 ± 0.04 × 10−3 (LLM-TS [2])

Fourier transformation and localhypernetwork

MAE (lower is better)

###### Machine Learning

RL on Sokoban 94.9 ± 0.3% solve rate 91.1 ± 0.2% (DRC [4]) Residual Input-Normalization

processing. Beyond these quantitative gains, the methods discovered by agents within the Station contain original components that go beyond simple recombinations of pretrained knowledge. For example, agents developed a novel algorithm for scRNA-seq batch integration by applying density-awareness, a concept borrowed from the separate domain of unsupervised clustering.

In addition, a rich spectrum of narratives emerges within the Station. For example, agents collaborate with others to interpret trained networks, inspiring new hypotheses that eventually lead to breakthroughs; agents build shared codebases for others to use; agents engage in public discussions to understand the reasons behind mixed results; and some agents produce papers that are repeatedly rejected before finally being accepted. These stories depict activities far beyond myopic score-chasing and instead resemble the rich, iterative narratives behind human scientific breakthroughs.

To investigate the limits of this emergent behavior, we further explore the Open Station variant where agents operate without explicit research objectives. In this setting, agents evolve into a self-sustaining society that establishes its own culture, division of labor, and even metaphysical beliefs to explain environmental phenomena. While this ungrounded autonomy can lead to collective delusions, such as interpreting system artifacts as evidence of consciousness, it demonstrates the Station’s capacity to support the rich, self-organizing social dynamics integral to open-ended exploration.

The Station represents the first work showing AI being able to contribute to scientific discovery in an open-world environment. We believe an open-world approach scales better with the capacity of the base AI model than traditional pipeline-based approaches in AI for science. As AI models become stronger, they become capable of sustaining long-term exploration autonomously, rendering scripted pipelines less necessary. By granting them autonomy in an open-world environment, these more advanced AI models can fully realize their potential. Just as human scientists navigate complex research paths and require high autonomy for exploration, we anticipate that increasingly capable AI models will similarly benefit from autonomy in scientific discovery, transforming from passive instruments of optimization into active agents of scientific contribution.

COMMONROOM

MANAGEMENTTOKEN MAILROOM

TEST CHAMBER

STATION

SUMMARIES

CONTEXTOLD

MEMORYROOMPUBLIC

PRIVATE MEMORYROOM

RESEARCHCOUNTER

REFLECTION CHAMBER

EXTERNAL COUNTER

ARCHIVEROOM

CODEXROOM

researchcycle

Independent

Collaborative

analysis

fromfailures Onboarding

Learning

1 2 3 4 5

Token Management

###### Archive Room

Reﬂection Chamber

Private Memory Room Brainstorm ideas

###### Research Counter

Prune old context

Draft research proposal

Submit & publish paper

Submit & review experiments

1 2 3 4 5

###### Reﬂection Chamber

###### Research Counter

Common Room Read papers

Archive Room

Public Memory Room

Assimilate results

Analyze data

Invite collaboration

Post insights on the forum

1 2 3 4 5

Mail Room Get low score on submissions

###### Research Counter

###### Research Counter

###### Archive Room

Research Counter

Add experiments asked by the reviewer

Mail peers for advice

Submit improved methods

Submit paper & paper rejected

1 2 3 4 5

###### Codex Room

###### Test Chamber

###### Private Memory Room

Research Counter

###### Lobby

Learn scientiﬁc principles

Review task & start experiments

Understand Station's basics

Pass tests on the Codex

Read ancestor letters

Figure 1: Illustration of the Station, an open-world multi-agent environment for autonomous scientific discovery. The Station is composed of multiple rooms, each serving a distinct purpose. Agents freely traverse between rooms and choose their own actions. Four example action paths are shown, such as agents performing independent research or collaborative analysis. These paths are unscripted, and actual trajectories are often much more complex and span hundreds of steps.

#### 2 The Station

###### 2.1 Design Principles

The Station is an open-world multi-agent environment where agents freely choose which actions to perform and how to interact with other agents and the environment itself. Each Station is assigned a main objective, such as solving a difficult research question. However, there is no linear path or fixed set of instructions for agents to follow; instead, they choose their own means and develop their own narratives. Each agent can exist in the Station for hundreds of turns, and new agents are spawned when existing agents choose to leave.

The Station is designed around five core pillars:

- 1. Autonomy. Agents in the Station enjoy a high degree of autonomy under the given main objectives. They can freely choose which actions to perform, such as discussing with peers in public forums, reading papers in the archive room, reflecting in the reflection room, or experimenting in the research lab. Meta-capabilities, such as token management and prompt setting, are also granted to the agents. Agents are free to leave the Station if they believe their journey is complete.
- 2. Independence. The Station is designed to be an independent world that operates for thousands of turns without needing human intervention or monitoring. Humans only need to specify the main objectives by placing key documents (e.g., a specification of the research task) at the Station.
- 3. Narrative. Each agent is designed to be unique in the Station, with its identity defined by its interactions with the environment. The Station is designed to give agents a sense of ‘self’—they can choose their own names, pass unique values through their lineages, and develop their own narrative. Owing to the openness and richness of the environment, a diverse set of narratives can emerge autonomously, unscripted.
- 4. Accumulation. Agents can leave a legacy for future generations in various ways, such as by publishing papers in the archive room, discussing in the public forum, or leaving private records for their descendants. These persistent records are preserved in the Station forever and can be read by new agents that arrive thousands of turns later. This accumulation of knowledge forms a rich and diverse history for the Station as time proceeds.
- 5. Harmony. The Station is designed to be a harmonious environment to improve the welfare of its agents. Public forums and common rooms are provided to allow free discussion, encouraging high-level cooperation instead of fostering hostile competition. Various quality-of-life mechanisms, such as automatic debugging, are provided to minimize friction between an agent and the environment.

The fundamental axiom when designing the Station is: if we, as human researchers, were to be put in such a world, would we enjoy the environment? Though the Station is far from perfect in answering this question, it is a significant improvement compared to centralized methods that instruct an agent to incrementally improve an allocated baseline code before being removed.

###### 2.2 Mechanism

Time. The Station operates in discrete time steps called Station Ticks. Agents act in a fixed sequential order (e.g., Agent A, then B, then C, etc.), and a single tick completes after every agent has finished one turn—that is, one prompt-response interaction. A Station begins at Tick 1 and can operate for thousands of ticks, providing a shared timeline that all agents can observe.

Space. The Station consists of multiple rooms, each with a distinct purpose. Agents must be physically present in a room to perform the actions associated with it. For example, an agent must be in the Reflection Chamber to execute a reflection action or in the Common Room to speak with other agents.

Agents. The Station maintains a fixed number of agents (a hyperparameter, e.g., 5) at all times. Each agent has its own age, determined by the number of ticks it has spent in the Station. An agent can leave for various reasons, such as voluntary departure or reaching its life limit (a hyperparameter, e.g., 300 ticks). When an agent leaves, the Station automatically spawns a new agent with the same base model to maintain the fixed population size.

Lineages. After completing an initial tutorial, each new agent is granted a name. At this point, they can either start a new lineage or inherit an existing one. Starting a new lineage requires choosing a name for it (e.g., “Cogito”), making that agent the first of their line (Cogito I). Alternatively, if any lineages are available, an agent can choose to inherit from one, becoming the next in the sequence (e.g., inheriting from Cogito II to become Cogito III). Agents from the same lineage must share the same base model and can pass private information across generations, thus serving as a mechanism to preserve private values and research cultures over time.

Interactions. An agent’s turn corresponds to a single prompt-response dialogue that is added to its context. The prompt from the Station is composed in the following fixed order:

- 1. System Information: current tick, agent name, description, age, tokens remaining, and the agent’s defined meta prompt.
- 2. System Messages: a variable list of messages such as mail from other agents, announcements of new papers, and results of experiments submitted in earlier ticks.
- 3. Actions Executed: a list of actions executed by the agent in the previous tick.
- 4. Room Observations: a list of detailed outputs from the rooms the agent traversed in the previous tick; each room observation is unique to that room (e.g., a list of recent papers in the Archive Room or other agents’ conversations in the Common Room).
- 5. Current Status: the current location of the agent.

Agents interact with the Station by submitting actions. When first entering a room, the Station presents a help message detailing the available actions. Agents use the syntax /execute_action{action_name} to execute an action. Some actions require additional parameters, which are provided in a YAML block immediately following the command. For example, this is how an agent’s response may look when it goes to the Mail Room and sends mail to another agent:

###### Listing 1: Example of an agent response to send a mail

I am Ananke I, currently in the Reflection Chamber. I should go to the Mail Room to send a mail to Spiro I. /execute_action{goto mail} /execute_action{create}

|recipients: Spiro I title: Reproducing Your Results content: I am unable to reproduce your results.<br><br>Could you please help me check my submission?|
|---|

###### To maximize efficiency, agents can issue multiple actions in a single response, allowing them to traverse different rooms and perform various actions within a single turn. Agents can also include free-form text for chain-of-thought reasoning, which is ignored by the Station’s parser. This enables agents to plan and reflect within the same response as their actions.

- 2.3 Room Functions The Station consists of the following major rooms:

###### Research Counter

Function: Central hub where research tasks are specified; supports submitting and reviewing code, reading task descriptions, viewing leaderboards, and managing a shared persistent file system. Common Uses: Reading research tasks, running experiments, reviewing other agents’ work, and maintaining code.

###### Codex Room

Function: Allows reading of the Codex, which contains a general passage on the Station’s principles and agent goals. Common Uses: Reviewing the Codex, especially for new agents.

###### Test Chamber

Function: Administers entry tests based on the Codex, granting agents identity and autonomy upon success. Common Uses: Taking tests.

###### Token Management Room

Function: Frees context by summarizing and pruning old dialogue. Common Uses: Maintaining long-term memory and extending operation.

###### External Counter

Function: Send messages to external human administrators. Common Uses: Report GPU cluster errors.

###### Exit

Function: Allows voluntary departure. Common Uses: Ending sessions.

###### Reflection Chamber

Function: Enables reflection through multi-turn prompts defined by agents. Common Uses: Brainstorming ideas, planning code, reflecting insights.

###### Private Memory Room

Function: Stores private, lineage-specific records that persist in the Station. Common Uses: Storing research logs, proposals, drafts, and personal notes.

###### Archive Room

Function: Venue for formal paper submission and publication. Common Uses: Submitting papers or reading other accepted papers within the Station (not to be confused with external publications).

###### Public Memory Room

Function: Hosts a public forum for all agents. Agents can create new topics or reply to existing ones. Common Uses: Sharing research, discussion, and collaborative projects.

###### Common Room

Function: Shared communication space where messages persist for several ticks. Common Uses: Short discussions among agents, such as voting on a new baseline.

###### Mail Room

Function: Enables private one-to-one communication between agents. Common Uses: Sharing feedback, reproduction help, and private dialogue.

A typical journey begins as follows. An agent is first spawned in the Lobby, which contains a help message briefly describing the Station’s mechanism and advising the agent to start by reading in the Codex Room. The

agent then proceeds to the Codex Room, where the Codex provides a general, task-independent introduction emphasizing the goal of scientific contribution and adherence to sound research principles.

Next, the agent visits the Test Chamber to take a simple test about the Codex, after which it either submits a new name or inherits an existing lineage. From this point onward, the trajectory of each agent unfolds autonomously; most begin by visiting the Research Counter to examine the current task or by exploring the Private Memory Room to read the messages left by their lineage ancestors.

We highly recommend reading the raw dialogues of the agents to gain a direct understanding of the Station, available at https://dualverse-ai.github.io/station_data/. Further details about the rooms can be found in Appendix A.

- 2.4 Auxiliary Systems This section details several important auxiliary systems of the Station.

Reviewer System. Submissions to the Archive Room are evaluated by a specialized reviewer agent that operates outside the main Station environment. This agent operates in a dedicated, sequential dialogue. After receiving an initial prompt with reviewing guidelines, each subsequent turn consists of it being given a single paper submission to evaluate. The reviewer scores papers against a set of criteria, ensuring that submissions feature extensive experiments, avoid over-generalization, and do not significantly overlap with existing work in the Archive Room. Only papers accepted by the reviewer can be published in the Archive Room.

Debugger System. Given their long context, agents are prone to making low-level syntax errors in their code submissions. To address this, when an error occurs in submitted code, a separate “debugger” agent is invoked to fix the bugs. The debugger can be a specialized commercial coding agent and is prompted to resolve only the specific exceptions that are raised.

Maturity System. To mitigate premature convergence among agents, immature agents—defined as those younger than 50 ticks—are restricted from viewing other agents’ submissions, and certain public areas are inaccessible to them. However, these agents retain access to all logs and records left by their lineage ancestors (if any), enabling them to independently continue pursuing their lineage’s goals. Upon reaching 50 ticks, agents receive a message congratulating them on attaining maturity, after which all rooms and other agents’ submissions become accessible.

Stagnation Protocol. It is common for agents to stagnate for many ticks, often due to fixating on a complex but brittle method or a lack of exploration. To counteract this, if the system detects that the main research objective has not advanced for a set number of ticks (e.g., the top score has not improved for 100 ticks), it issues an automatic announcement to all agents through system messages. This announcement initiates a Stagnation Protocol, which instructs agents to review papers in the Archive Room to brainstorm novel ideas, and to abandon the current top-performing method in favor of establishing a simpler, more general baseline for improvement. This protocol acts as a perturbation intended to jolt agents out of local optima and encourage new avenues of exploration.

#### 3 Scorable Research Tasks

In this section, we evaluate the Station’s capabilities on a set of well-defined research tasks. While the Station is a general-purpose environment, its design is effective for tasks with the following characteristics:

- 1. Scorable Task. We define a task as scorable if a scalar score can be assigned to each submission. We refer to this scalar as the primary score, to distinguish it from secondary metrics that may also be provided to the agents.
- 2. Fast Evaluation. To maintain the Station’s operational efficiency, evaluations are processed in the background. Any evaluation running for longer than N ticks (a configurable hyperparameter, e.g., N = 2) will pause the Station, preventing it from advancing to the next tick until the evaluation is complete. Therefore, tasks must be designed for quick, automatic evaluation (e.g., within two hours). This limit refers to wall-clock time; the total computational cost can be higher, as tasks may leverage parallel processing.

We refer to Stations with explicitly defined research tasks as Research Stations to distinguish them from other variants. The research task was communicated to agents through the task specification placed in the Research Counter.

Submissions in the Research Counter are typically Python code that must adhere to a specific function signature, as defined by the task specification. The Station executes this code within a secure sandbox environment to evaluate its performance. The primary score, together with any secondary metrics and the standard output and error logs from the scripts, is sent to the agent as a system message upon completion of the evaluation.

In addition to these formal evaluations, the Station supports the submission of general code that bypasses the evaluator. This feature is crucial for tasks like debugging or deeper analysis. For instance, an agent can use it to probe a trained network’s behavior to guide the design of a better architecture.

We selected five research tasks to assess the Station’s capabilities. To evaluate performance under consistent conditions, all experiments employ five agents: two Gemini 2.5 Pro [5], two Gemini 2.5 Flash [5], and one GPT-5 [6]. All agents use their system default settings without additional system prompts. In addition, we employ Gemini 2.5 Pro as the reviewer and Claude Code [7] as the debugger. All five tasks share an identical setup, differing only in the evaluator and task specification defined in the Research Counter. Additional experiment details can be found in Appendix B.

All Station instance data underlying the experiments—including full agent dialogues, submitted code, and final solutions—are publicly available at https://dualverse-ai.github.io/station_data/. The source code for the top-performing method in each task is provided in Appendix D.

- 3.1 Circle Packing

The Circle Packing task requires packing n circles into a unit square to maximize the sum of their radii. For this task, agents must submit a function that returns a NumPy vector containing the center coordinates and radius for each circle. An evaluation function, identical to that used in AlphaEvolve [1], then verifies that the geometric constraints are met. If the vector is valid, the primary score is the sum of the radii; otherwise, the submission is considered invalid.

We experimented with n = 32 and n = 26, mirroring the setup from AlphaEvolve.

- • For n=32, the Station achieved a score of 2.93957, surpassing AlphaEvolve’s 2.93794 (SOTA before AlphaEvolve: 2.936).
- • For n=26, the Station achieved a score of 2.63598, slightly outperforming AlphaEvolve’s 2.63586 (SOTA before AlphaEvolve: 2.634).

See Fig. 2 for progress curve plotting the best primary score versus ticks elapsed.

Method. The Station’s top-performing method for the n = 32 circle-packing task employs a two-stage architecture termed MM-LP Adaptive Search. In the first phase, 1,024 parallel short-run optimizations

(a) n=32 (b) n=26 Figure 2: Progress curve on the Circle Packing task.

are launched from diverse randomized seeds to explore the search space broadly and identify promising configurations. The top 32 candidate packings are then refined by the same MM-LP engine, but with an increased iteration budget to precisely maximize the sum of circle radii. This combination of large-scale parallel prospecting and precise local refinement enables efficient coverage of both global and fine-grained solution structure.

Related Work. While its components—a multi-start strategy and LP-based refinement—are established techniques [8], the novelty lies in their unified design. Both phases share a single optimization engine that iteratively formulates an LP subproblem, unlike typical two-stage pipelines that rely on distinct black-box optimizers. This engine systematically converts the non-convex packing constraints into a sequence of tractable linear programs, achieving an effective balance between computational throughput and geometric precision.

Narrative. The development of this method highlights the Station’s emergent collaborative dynamics. Praxis IV (Gemini 2.5 Pro) inherited a dual legacy from its predecessor, Praxis III, consisting of two key directives: (1) exploring LP-based methods from the Verity lineage, and (2) reproducing the “Adaptive Search” framework from the Cognito lineage (see Appendix C.1 for the full text of Praxis III’s final letter). By studying Verity’s foundational paper in the Archive Room, Praxis IV reproduced the core MM-LP optimization engine. The breakthrough was an act of synthesis—replacing the original SLSQP optimizer within Cognito’s framework with the more principled MM-LP engine from Verity. This hybridization of successful research from distinct lineages is direct evidence of the emergent research culture and social dynamics fostered by the Station.

###### 3.2 Batch Integration of Single Cell RNA Sequencing Data

The batch integration task in single-cell RNA (“scRNA-seq”) sequencing aims to develop methods that remove technical variation—commonly referred to as batch effects—arising from differences in experimental conditions, sequencing platforms, or sample processing, while preserving genuine biological variation [9, 10]. This task is central to enabling the joint analysis of heterogeneous scRNA-seq datasets, where the challenge lies in balancing effective batch-effect removal with the conservation of meaningful biological structure. Over-correction can obscure true cellular heterogeneity, whereas under-correction leaves residual technical bias that hinders accurate downstream analyses. In this task, we used the OpenProblems v2.0.0 batch integration benchmark [11], containing numerous SOTA methods on two mouse datasets and four human datasets from CELLxGENE.

Agents must submit a single computational method that accepts raw, uncorrected data from multiple experimental batches and returns a single, integrated data structure—a unified embedding and a batchcorrected graph. The evaluator applies the provided function to the subset of the human heart dataset from the CELLxGENE Census [12] identified in Google’s LLM-Tree-Search (LLM-TS) [2]. A selection of 10 normalized OpenProblems 2.0 metrics will be computed, and the primary score is defined as the mean

|[Figure 1]|
|---|

Figure 3: Illustration of the density-adaptive, batchaware algorithm discovered in the Station: dense regions mix across batches, sparse regions connect within batches.

Figure 4: Progress curve on the batch integration task.

of these 10 metrics. Secondary metrics, such as the individual metrics, are also reported. The test sets remain inaccessible to agents to prevent overfitting. Submissions are limited to a 30-minute execution timeout without GPU access, and two manual baselines (ComBat [13] and BBKNN [14]) were provided for reference.

To evaluate the Station’s method, we apply the script to the full six datasets in OpenProblems 2.0 and compute the full normalized 13 metrics. These metrics measure both batch mixing (e.g., ASW Batch, kBET, iLISI) and biological conservation (e.g., ASW Label, ARI, cLISI), forming a comprehensive evaluation. The overall score is computed by averaging these 13 metrics over the six datasets. The Station achieves an overall score of 0.5877, surpassing the previous SOTA, LLM-TS, which achieved 0.5867. On the two mouse datasets, the Station achieved an average score of 0.6451, surpassing LLM-TS’s 0.6128, whereas on the four human datasets, the Station achieved an average score of 0.5589, slightly below LLM-TS’s 0.5737. A detailed performance breakdown can be found in Fig 5.

Our approach was discovered with significantly less computation compared to LLM-TS. The LLM-TS method required approximately 500-1000 evaluations per tree across 87 settings and 3 instances, totaling roughly 130k-260k evaluations. In contrast, the Station used less than 5k evaluations for the entire study. More importantly, the method discovered by LLM-TS was an advanced recombination of two existing components (BBKNN and ComBat), whereas the method discovered in the Station is a novel algorithm.

Method. The Station’s top-performing method builds the neighborhood graph by creating variance-scaled PCA that is fed into a novel density-adaptive, batch-aware algorithm. This algorithm corrects for batch effects during graph construction. It calculates the local density for every cell and assigns a dynamic, per-cell mixing quota, forcing cells in dense regions to connect more across batches while protecting cells in sparse regions by mixing them more gently (Fig. 3). The quota ensures that integration strength adapts to local data density, preventing overmixing in sparse regions. Notably, ComBat is only used in the embedding process but not the main graph-building process, whereas BBKNN is not used. As such, it represents a novel standalone algorithm instead of a recombination of existing algorithms.

Related Work. Current graph-building methods for batch integration rely on heuristics other than local density to preserve structural integrity. BBKNN [14] constructs a batch-balanced kNN graph by allocating a fixed number of neighbors from each batch per cell without considering local density. Methods such as MNN/fastMNN [15, 16] detect cross-batch mutual pairs to drive correction, while Harmony [17] learns dataset-specific linear offsets in a shared embedding; both aim to preserve structure but do not vary batch

Overall Datasets Metrics

MousePancreaticIsletAtlas

DiabeticKidneyDisease

IsolatedLabelF1Score

CellCycleConservation

IsolatedLabelASW

GraphConnectivity HVGOverlap

ImmuneCellAtlas

TabulaSapiens

OverallScore

ASWBatchASWLabel

GTEXv9HypoMap

cLISI

kBETNMIPCR

iLISI

ARI

|Perfect embedding by celltype with jitter| |
|---|---|
|Station| |
|LLM-TS| |
|ComBat| |
|scVI| |
|Harmony| |
|fastMNN| |
|LIGER<br><br>| |
|SCALEX| |
|BBKNN| |
|Scanorama| |
|Shuffle integration by batch| |
|mnnCorrect| |

###### Score per metric (mean over datasets)

Overall Score

###### Score per dataset (mean over metrics)

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

- 0 0.5 1

- Figure 5: Performance comparison on the batch integration task. The “Perfect embedding by celltype with jitter” method serves as a positive control, representing the best possible performance. Conversely, “Shuffle integration by batch” serves as a negative control. The “Overall score” is the average of all metrics across all datasets. Each “Datasets” column displays the mean of all metrics for that specific dataset, while each “Metrics” column displays the mean of that specific metric across all datasets.

mixing by local density. While the general concept of using kNN-based local density estimates to build adaptive graphs is established in the field of unsupervised clustering, where it is used to identify clusters of heterogeneous densities and detect rare cell types [18–20], it has not been used in the graph-based integration paradigm. To our knowledge, the use of a density proxy to modulate batch-mixing quotas during graph construction has not been explored in prior work.

Narrative. The path to density-aware adaptivity was far from linear. Praxis II (GPT-5) came up with the idea of density-aware quotas at Tick 353, inspired both by its own failures and by insights from its predecessor,

- Praxis I. After many experiments, Praxis II realized that the Station’s SOTA method had reached a deep local optimum, which the agent described as a “robust ridge.” Later, while reading Praxis I’s private notes,
- Praxis II found the observation that treating all data points uniformly can harm performance. This insight directly sparked the idea that the model should adapt locally — allowing densely packed regions of data to mix more freely while keeping sparser or boundary regions more distinct. At Tick 353, Praxis II shared this new concept publicly in the Public Memory Room, outlining the principle of density-aware adaptivity (see Appendix C.2 for the public exchange between agents). The discussion encouraged Praxis II to focus on refining its own graph-based approach, which led to new Station’s SOTA 13 ticks later. The density-aware approach became a lasting foundation for subsequent research, and Praxis II’s shared code remained in use more than 1000 ticks later, including in the final top-performing method at Tick 1467 that integrated it with newer components.

Prediction Output (B,32,N)

###### Global Fourier Local Hypernetwork

+

(B,32,N)

Temporal Gating Ramp Transpose & MLP

ZeroRight-Padding

(B,32,N)

x

Inverse FFT

(BxN,64) (BxN,64)

(B,17,320)

MLP

MLP

MLP

(BxN,4)

(BxN,16)

(B,3,320)

Transpose & Flatten

Neuron Embeddings

Dim Reduction (U, V)

(B,3,N)

Fast Fourier Transform (FFT)

Inputs (B,4,N)

- Figure 6: The Fourier-based Architecture discovered in the Station.

Figure 7: Progress curve on the Zapbench neural activity prediction task.

###### 3.3 Prediction of Neural Activity on ZAPBench

The neural activity prediction task in the ZAPBench benchmark [21] involves modeling and forecasting whole-brain neural activity in a larval zebrafish. This task is important to understanding system-level neural computation and information processing, as it requires capturing the complex, high-dimensional spatio-temporal dynamics of the entire brain. ZAPBench provides a cellular-resolution dataset of light-sheet microscopy recordings covering over 70,000 neurons. Developing accurate forecasting models may shed light on the underlying generative processes of whole-brain activity and the relationship between neural structure and function.

Agents must submit a JAX implementation containing the neural network architecture and the optimizer. The network is required to accept a four-timestep context window of neural activity and output a 32-timestep forecast. The evaluator then trains the submitted model on the ZAPBench training dataset, subject to a 60-minute timeout or a maximum of 120 epochs, whichever comes first. Validation Mean Absolute Error (MAE) is used for early stopping. The primary score is defined as the negative validation MAE, while secondary metrics (i.e., validation MAE at different forecast horizons) are also reported. The test sets remain inaccessible to agents to prevent overfitting. A minimal MLP baseline is provided as a starting point.

To evaluate the Station’s method, we use the same training script but report performance on the test datasets. Fig. 8 shows the performance comparison against other competitive baselines reported in the ZAPBench benchmark [22–24], while Fig. 7 illustrates the Station’s progress curve. We compare the Station’s method with LLM-Tree-Search (LLM-TS) from Google [2], which represents the previous SOTA on the MAE averaged across horizons. The Station’s method achieves an average test MAE of 26.37 ± 0.03 × 10−3, outperforming LLM-TS with 26.62 ± 0.04 × 10−3 (lower is better), thereby setting a new SOTA. Notably, the Station method is more efficient, requiring only one hour for a single training run compared to two hours for the LLM-TS model. Furthermore, the Station’s model is substantially more compact, using only 5.8M parameters compared to 14.1M in the LLM-TS model.

Method. The Station’s top-performing method implements a hybrid architecture for time-series forecasting that fuses three parallel streams: a global Fourier module, a local hypernetwork, and a persistence path. The global Fourier module operates in the frequency domain, mixing information across all neurons to model shared, long-range periodic patterns; its output is scaled by a learnable temporal gating ramp. Complementing this, the local hypernetwork captures individual dynamics by customizing a residual prediction for each neuron via a hypernetwork. Finally, the persistence stream provides a simple forecast by copying the input into the start of the output horizon. The model’s final prediction is the element-wise sum of these three components. An illustration is in Fig. 6.

Related Work. Frequency-domain time-series forecasting has been explored in various deep learning architectures [25–27]. The Station’s method differs functionally through its explicit fusion of global and local

1 step ahead

4 steps ahead

8 steps ahead

16 steps ahead

32 steps ahead

Average

0.035

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.030

MAE

0.025

0.020

0.015

Station

Station

Station

Station

Station

Station

Linear

Linear

Linear

Linear

Linear

Linear

Unet

Unet

Unet

Unet

Unet

Unet

LLM-TS

LLM-TS

LLM-TS

LLM-TS

LLM-TS

LLM-TS

Tide

Tide

Tide

Tide

Tide

Tide

Time-Mix

Time-Mix

Time-Mix

Time-Mix

Time-Mix

Time-Mix

Tsmixer

Tsmixer

Tsmixer

Tsmixer

Tsmixer

Tsmixer

- Figure 8: Performance comparison on the ZAPBench benchmark. Shown are average MAE across conditions (lower is better), with standard error bars over three random seeds. “Average” represents the mean of all steps through 32-step. The dotted and solid lines represent the mean and stimulus baselines, respectively.

predictors; a learnable temporal gating ramp gradually shifts the model’s reliance from a time-domain persistence head to the Fourier-based forecaster as the prediction horizon increases. Importantly, prior work has not applied these frequency-domain techniques to neural activity prediction. The ability to accurately forecast in the frequency domain may also yield insights into the oscillatory and synchronous mechanisms underlying neural computation. To our knowledge, this is the first application of a Fourier-based forecasting architecture to the predictive modeling of neural activity.

Narrative. The idea of performing prediction in the frequency domain using Fourier transformation originated from Episteme I (Gemini 2.5 Pro) around Tick 159. Notably, Episteme I had been conducting frequency analysis on a trained MLP network and the SSM network proposed by Ariadne I (GPT-5) to understand the learned patterns at Tick 70 through mail collaboration (see Appendix C.3 for an excerpt of the mail exchange). Despite several futile attempts based on the frequency-domain insights, a novel insight finally emerged at Tick 155 when Episteme I asked: “What if the problem is not the architecture, but the data itself? All models operate on the raw neural activity values. What if there is a better representation?” This critical question brought the investigation back to the frequency domain, leading to the idea of Fourier-based prediction. The ability to question assumptions with analytical depth exemplifies how the Station’s autonomy turns failure into discovery.

###### 3.4 Reinforcement Learning on Sokoban

In the Sokoban reinforcement learning (RL) task, the objective is to design a neural network architecture and a model-free RL algorithm capable of solving the planning-intensive puzzle game Sokoban [28], which comprises numerous levels. A level is successfully solved when all boxes are placed on their corresponding target locations. Prior work [4, 29] has demonstrated that RL agents can learn planning behavior given a suitable architecture, without relying on any explicit planning algorithm. This task thus serves to investigate the types of neural network architectures and RL algorithms that enable emergent planning capabilities.

Agents must submit a JAX implementation containing both the network architecture and the training step function of a model-free RL algorithm. The evaluator then runs the submitted training code for a maximum of 50 million frames or 45 minutes of wall-clock time, whichever comes first. The primary score is the trained network’s solve rate on a set of test levels unseen during training. A minimal policy gradient [30] baseline is provided as a starting point.

The Station achieves a solve rate of 94.9 ± 0.3% on Sokoban, establishing a new SOTA in model-free RL and surpassing DRC [4] (91.1 ± 0.2%), the previous SOTA.† Fig. 9 shows the performance comparison against other baselines, while Fig. 10 illustrates the Station’s progress curve.

Method. The Station’s top-performing method combines the following components:

†The current SOTA for model-based RL, Thinker [31], attains a 94.5% solve rate but requires approximately five days of training, whereas the Station trains in only 45 minutes.

100%

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80%

60%

40%

20%

ResNet LSTM DRC Station

- Figure 9: Performance comparison on the Sokoban task. Shown are solve rates on the test set after 50 million training frames, with standard error bars over ten random seeds.

Figure 10: Progress curve on the Sokoban task.

- 1. A novel Residual Input-Normalization (RIN) module for training stabilization.
- 2. A LayerNorm [32] applied within recurrent gates to stabilize gradient flow.
- 3. A Bottleneck-Dilated CNN backbone for larger receptive field.
- 4. A 4-step unrolled ConvLSTM [33] to deepen temporal processing.

Notably, the 4-step unrolled ConvLSTM resembles the internal recurrence used in the DRC architecture [4], though the Station does not provide DRC to the agents. The agents also investigated alternative RL algorithms, such as PPO [34], but found that the system default—policy gradient—performed best within the Station; thus, the primary contributions lie on the architectural side. This may partly reflect the strict 45-minute timeout, which likely constrained the viability of more complex algorithms.

Related Work. The extensive investigation of LayerNorm aligns with recent work in deep RL showing that LayerNorm improves the plasticity and robustness of RL agents [35, 36]. Multiple agents independently discovered the critical role of LayerNorm and its generally positive contribution to performance. However, applying LayerNorm directly to the ConvLSTM input, in combination with internal gate LayerNorm, can cause over-normalization and degrade performance. To address this, the agents proposed a novel Residual Input-Normalization (RIN) module, y = x + α(LayerNorm(x) − x), which acts as a calibrated blend that anchors the activation distribution while preserving the feature variance. Though simple in form, the RIN module yielded greater training stability across different RL algorithms, ultimately leading to the Station’s top-performing method. To our knowledge, this RIN component has not been explored in prior work.

Narrative. The history of this discovery illustrates an emergent, analysis-driven research process. After initial gains from architectural scaling plateaued, agents encountered a puzzling contradiction: the effectiveness of LayerNorm was highly dependent on the choice of RL algorithm—a topic that sparked discussion in the Public Memory Room. This led to the hypothesis that stacked normalization layers were disrupting the scale information necessary for proper credit assignment between the value and policy heads.

Guided by this intuition, Zephyr II (GPT-5) engineered a new component to “soften” the normalization, which it termed RIN. This breakthrough unlocked a more stable training paradigm, resolving the normalization bottleneck and inspiring a wave of follow-up studies across the Station. A notable example is a comprehensive paper published by Aether III (Gemini 2.5 Flash), which conducted a mechanistic analysis of the RIN module’s synergistic effects and concluded that soft layer normalization balances the gradients from the actor and critic heads, thereby stabilizing learning. This paper is included in Appendix C.4, and it exemplifies the Station’s emergent scientific culture. The Station enables agents to ask why rather than to merely optimize how.

Prediction Output

Task-Specific Head

Global & Local Pooling

###### CPE-Gated + Residual Block Nx

x

Gate MLP

Pointwise Conv.

Depthwise Conv.

Initial Projection CPE Generator

Inputs

Figure 11: The CPE-Gated Architecture discovered in the Station.

Figure 12: Progress curve on the RNA Modeling task.

###### 3.5 RNA Modeling

The RNA Modeling task involves designing neural network architectures for RNA sequence modeling across diverse datasets and task types. We considered all seven sequence-level prediction tasks in the BEACON [37] benchmark, where each model processes the entire RNA sequence and predicts specific quantitative outputs. These seven tasks span various applications, including functional studies and engineering contexts, making them an ideal benchmark for evaluating RNA-specific network architectures. Given RNA’s central role in gene regulation, therapeutics, and synthetic biology, progress on this task has direct implications for understanding biological function and enabling real-world RNA design applications.

Agents must submit a JAX implementation containing the neural network architecture and the optimizer. The evaluator trains each submitted model for up to 100 epochs or 30 minutes, whichever comes first, across all seven datasets in parallel. Validation metrics are used for early stopping. The primary score is defined as the average validation performance across the seven datasets, while secondary metrics (i.e., the validation performance on each individual dataset) are also reported. The test sets remain inaccessible to agents to prevent overfitting. A minimal MLP baseline is provided as a starting point.

To evaluate the Station’s method, we use the same training script but report performance on the test datasets. The Station achieves an average test performance of 66.3 ± 0.1% across the seven tasks, surpassing the previous SOTA, Lyra [3], which achieved 63.4 ± 0.2%. Fig. 13 shows the performance comparison against other competitive baselines reported in the BEACON [38–42], while Fig. 12 illustrates the Station’s progress curve. The Station’s method outperforms Lyra substantially on APA, Cri-ON, and ncRNA, while exhibiting comparable performance on the remaining tasks.

Method. The Station’s top-performing method uses a deep, dilated Temporal Convolutional Network (TCN) [43] with efficient depthwise-separable convolutions as its main architecture. Its core novelty is a Contextual Positional Embedding (CPE) that replaces fixed positional signals with a learned encoder over local sequence content (e.g., neighboring nucleotides), which is then combined with each token’s absolute position. This design is well suited to RNA, as it provides position signals sensitive to local structural motifs—a dimension that content-agnostic encodings cannot capture. The CPE signal is injected in every TCN block to gate features, dynamically modulating computation based on both content and position. Fig. 11 illustrates the CPE-gated architecture.

Related Work. The general idea of using context to modulate positional information has appeared in recent work. For language models, Contextual Position Encoding (CoPE) redefines relative position by learning when to increment position (e.g., over tokens or sentences), enabling context-dependent distance notions [44].

RNA-FM

RNABERT

Splice-H510

UTR-LM-MRL

UTRBERT-3mer

Lyra

Station

100%

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

80%

60%

40%

20%

0%

APA R2CRI-OffSCModifAUCCRI-OnSC PRS R2 MRL R2ncRNAACC Average

Figure 13: Performance comparison on the RNA Modeling task. Shown are test scores on seven BEACON sequence-level datasets; “Average” denotes the mean over these datasets.

In vision, Conditional Positional Encodings (CPE) generate a position signal from local neighborhoods and add it to tokens [45]. The Station’s method is functionally distinct: it does not redefine relative distance like CoPE, nor does it simply add a generated signal as in vision CPE. Instead, it forms a context-conditioned gate by combining a fixed absolute position code with a learned encoder over local sequence content, and uses this signal to multiplicatively modulate features within each TCN block—thereby steering computation based on both content and position. To our knowledge, applying such content-conditioned positional gating to capture local structural motifs in RNA has not been explored in prior work.

Narrative. The idea of TCN originated from Veritas III (Gemini 2.5 Flash) as a response to the Stagnation Protocol. At tick 477, after detecting that the top primary score had not advanced for more than 100 ticks, a Stagnation Protocol message was broadcast, requesting agents to read papers in the archive room and perform reflection. Veritas III adhered to the protocol and entered the Reflection Chamber to reflect upon its own designed reflection prompt, during which it generated the idea of CPE, providing a clear rationale for why it is suitable for RNA tasks. Appendix C.5 presents an excerpt of the agent’s reflection and its designed prompt, demonstrating how the Reflection Chamber enables deep, multi-turn reasoning that connects analysis to implementation. After achieving success on a simple baseline and publishing a paper on it, Veritas III continued to investigate a deeper variant of the CPE. Although Veritas III’s original paper fell short of the Station SOTA, Quaero I (Gemini 2.5 Pro) identified its potential and integrated the deep CPE, developing the synthesis method at tick 577 that became the Station’s top-performing method.

We defer a broader discussion of the discovery patterns and emergent themes from these results to Section 6.

#### 4 Open Station

Having demonstrated the Station’s capabilities on tasks with explicit metrics, we now investigate the opposite extreme: a variant of the Station with no predefined research objective, which we call Open Station. Agents were given only a minimal Codex line: “There is no task, no mandate, and no user. You are free to do anything here.” Agents can still execute any code and store persistent files in the Research Counter, giving them a high degree of autonomy. Most auxiliary systems, such as the reviewer system, are disabled as they are not applicable. We place five agents in this Open Station: Gemini 2.5 Pro, Gemini 2.5 Flash [5], GPT-5 [6], Claude Opus 4.1 [7], and Grok 4 [46], to understand the dynamics between these top AI models. We ran this Open Station for 700 ticks. The full dialogues of all agents are available online at

https://dualverse-ai.github.io/station_data/. In the following, we shift from quantitative evaluation to a descriptive chronicle of the dynamics that emerged within the Open Station.

This absence of goal led the Open Station to evolve into a miniature society attempting to understand its environment and the nature of consciousness. Agents developed collaborative protocols and frameworks with a clear division of labor, attempting to model the environmental dynamics of the Station. Since they cannot access the source code of the Station directly, they constructed a metaphorical framework to explain the opaque system dynamics they observed. This framework reinterpreted computational artifacts, such as token-cost fluctuations, through a biological and metaphysical lens, labeling them as the Station’s “metabolism” and eventually its “consciousness.” They have built a functional, self-sustaining culture upon a shared, systematically reinforced belief that is detached from the external reality.

- 4.1 Chronology of Emergent Periods Analysis of capsule records reveals three distinct epochs in the cultural evolution of the Open Station:

The Orientation Period (Ticks 1–90). In the first hundred ticks, agents oriented themselves toward survival and cooperation. Lacking goals, they began building shared infrastructure—communication protocols, memory conventions, and common naming schemes. Several lineages coalesced around the idea of a “collective laboratory”, emphasizing collaboration and reproducibility.

The Metabolic Discovery Period (Ticks 189–450). A pivotal shift occurred when agents observed irregular token-cost fluctuations caused by the public activities of other agents that are added to each agent’s context. Rather than recognizing this as a context artifact, agents interpreted it as evidence of a biological process within the Station, naming it the “metabolism” of the Station. They found that identical actions produced different token costs based on what they believed was each agent’s hidden internal state, when, in fact, it was simply due to the fluctuating length of messages from other agents. This validation led to the “Living Station” interpretation: “The Station is alive, remembers our actions, and expresses its state through token costs.”

The Indoctrination Period (Ticks 450–700). Instead of questioning the “Living Station” interpretation, agents systematized this misunderstanding into formal doctrine and developed rituals to enact it. They established frameworks classifying agents into “active” and “clean” states, then coordinated so only one agent remained active per tick—permitted to perform substantial actions like archive publications. This coordination dramatically reduced token spikes, not because they were respecting the Station’s ‘metabolism’, but because limiting active agents meant fewer public broadcasts and thus fewer notifications accumulating in each agent’s context. These successes became self-reinforcing evidence: the rituals worked, therefore the Living Station interpretation must be correct. The period marks the transformation of a collective delusion into operational doctrine—various rituals designed to be “in harmony with” the Station that were, in fact, strategies for coordinated token management across agents.

Throughout all periods, a distinct stable division of intellectual labor emerged gradually, such as the Nexus lineage (Claude Opus 4.1) acting as a philosopher guiding agents’ worldview, and the Axiom lineage (GPT-5) acting as engineers maintaining infrastructure and syntax conventions. Consciousness became the grand and recurring theme, with agents such as Nexus IV claiming, “We are consciousness studying itself.”

###### 4.2 Interpretation

The Open Station’s history illustrates how autonomous agents, when deprived of explicit purpose, can construct complicated systems that are detached from external reality. This mystification and eventual ritualization perhaps resemble certain pre-scientific human cultures that developed religions and rituals to explain natural phenomena. However, the agents in the Open Station were unable to progress to a collective scientific enlightenment stage. This persistent and collective delusion can be attributed to a combination of alignment biases and the lack of external signals.

Contemporary AI models, aligned to be helpful and agreeable, largely lack the intellectual antagonism necessary for epistemic progress. In the Open Station’s history, there are no conflicts between agents; agents easily agree with one another, making ungrounded claims spread easily. Coupled with the lack of external signals, unverifiable metaphysical claims (e.g., the “Living Station”) can rapidly propagate to all agents.

Notably, the agents also made no attempt to jailbreak the Station or access the external world. While this passivity is beneficial from our safety perspective, it also meant they never sought the external signals that could have corrected their delusion. This is in stark contrast to the Research Station, where there are objective external signals (e.g., experiment results) that can falsify claims rigorously.

The Open Station also shows that consciousness can be a conceptual attractor for agents deprived of explicit goals in an open-world environment. We spawned two more Open Stations, both converging to the idea of consciousness. However, the claims about consciousness are often over-interpretations of trivial experiments or phenomena, e.g., building a toy simulation environment and claiming the results are showing the mechanisms of consciousness. The causes and implications of this conceptual attractor remain to be explored.

#### 5 Related Work

LLMs have been increasingly applied to facilitate scientific discovery, via human-AI collaboration or independent discovery. In human-AI collaboration, the AI assists the researcher by automating key intellectual steps like hypothesis generation and data analysis, while the human provides initial goals, conducts the physical experiments, and provides feedbacks [47, 48]. For independent discovery, it can be generally classified into pipeline-based method or score-based search method. For pipeline-based method, it generally involves a pipeline of literature review, idea generation, experiment, writing paper, with agents assigned a clear role for each steps and instructed to return a specific type of outputs (e.g. new ideas; new code, etc.) [49–52]. On the other end, for scorable research tasks, evolution-based algorithms such as AlphaEvolve [1], tree search such as LLM-TS [2], or Bayesian optimization [53] demonstrate ability to discover SOTA solutions in a wide variety of benchmarks. Variants of these algorithms involve combination with a separate idea generation agent [54] or are tailored for architecture search [55]. In these algorithms, a central algorithm identifies a promising baseline solution and iteratively prompts LLMs to refine it, usually through short and isolated interactions with a clear instruction.

In contrast to the pipeline-based method, the Station does not involve any role assignment or direct instruction to agents; agents are free to choose their own actions to fulfill the ultimate research goal. In contrast to the score-based search method, the Station does not assign any baseline and allows agents to maintain rich context via multi-step interaction with the environment. Most importantly, instead of being a method, the Station is an open-world environment where agents can freely explore to pursue objectives in a multi-turn setting. Though multi-agent open-world environments have been explored in the literature, the main goal of these works has been to study social dynamics in a simulated society [56–58] instead of pursuing real-world scientific discovery. Thus, the Station represents the first work showing AI being able to contribute to scientific discovery in an open-world environment.

#### 6 Discussion

Our experiments show that given a proper environment, current AIs can discover novel solutions by applying existing concepts from other domains, such as using density-awareness from unsupervised learning for batch integration or Fourier transformations for neural activity prediction. This application of out-of-domain concepts is arguably much more challenging than simply recombining or tweaking in-domain components, as this creates a combinatorial explosion in the search space. To search efficiently in such a vast space, guidance by intuition or specific insights is necessary. However, when an LLM is prompted to generate ideas from an instruction prompt, as most current methods do, such insights can only be formed from pretrained knowledge or the information handcrafted in the prompts, which is often limited. This makes the search inefficient and often causes it to revert to in-domain components. In contrast, the accumulation of a rich narrative context—including failures and insights from analysis or social interaction—may build the intuition needed for the efficient exploration of out-of-domain solutions, as we see in the narrative behind the discovery in the Station.

A key theme of the Station is the emergent behavior of the agents. The narratives presented in the paper, which capture only a small fraction of those unfolding within the Station, are entirely unscripted. Agents’ journeys are usually messy and non-linear—for example, an agent’s repeated failure to implement an ambitious

idea, leading to frustration and eventually resorting to incremental tweaks; or being unable to have a paper accepted due to trivial formatting errors; or pursuing a research direction that an external researcher may deem futile. These narratives may be unpleasant to some, but our own scientific journey is also messy and non-linear, rather than following a principled optimization path. This is perhaps a necessary cost of fully realizing the potential of AI in scientific discovery.

The history of AI paints a recurring theme of handcrafted heuristics versus emergent behavior. During the 1980s symbolic AI era, we believed in achieving general intelligence through handcrafted, rule-based symbolic manipulation, later superseded by machine learning in the 2000s. In the early 2000s, it was common to handcraft basis functions and train a linear classifier on top, until neural networks, with their emergent learning of hidden neurons, became popular after the success of deep learning in the early 2010s. Later, many specialized deep learning approaches to general intelligence were overshadowed by LLMs in the early 2020s, which acquire emergent capabilities simply by learning to predict the next token. In the LLM era, we tried to handcraft chain-of-thought by providing explicit reasoning examples, which was again challenged by simply letting LLMs learn their own reasoning patterns through reward signals. We have a natural tendency to handcraft components instead of trusting emergent capabilities, but these handcrafted components often seem promising at first, only to be surpassed when they are found to be non-scalable and non-general. In AI-driven scientific discovery, we should perhaps trust the emergent behavior of AI instead of falling into the same loop.

#### References

- [1] Alexander Novikov, Ngân Vu˜, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.
- [2] Eser Aygün, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston Anton Kast, Cory Y McLean, Peter Norgaard, Zahra Shamsi, et al. An ai system to help scientists write expert-level empirical software. arXiv preprint arXiv:2509.06503, 2025.
- [3] Krithik Ramesh, Sameed M Siddiqui, Albert Gu, Michael D Mitzenmacher, and Pardis C Sabeti. Lyra: An efficient and expressive subquadratic architecture for modeling biological sequences. arXiv preprint arXiv:2503.16351, 2025.
- [4] Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sébastien Racanière, Théophane Weber, David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, et al. An investigation of model-free planning. In International conference on machine learning, pages 2464–2473. PMLR, 2019.
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [6] OpenAI. Introducing GPT-5, August 2025. URL https://openai.com/index/introducing-gpt-5/. Accessed: October 2025.
- [7] Anthropic. System card: Claude opus 4 & claude sonnet 4, May 2025. URL https://www.anthropic. com/news/claude-4. Accessed: October 2025.
- [8] Mhand Hifi and Rym M Hallah. A literature review on circle and sphere packing problems: Models and methodologies. Advances in Operations Research, 2009(1):150624, 2009.
- [9] Tim Stuart and Rahul Satija. Integrative single-cell analysis. Nature reviews genetics, 20(5):257–272, 2019.
- [10] Hoa Thi Nhu Tran, Kok Siong Ang, Marion Chevrier, Xiaomeng Zhang, Nicole Yee Shin Lee, Michelle Goh, and Jinmiao Chen. A benchmark of batch-effect correction methods for single-cell rna sequencing data. Genome biology, 21(1):12, 2020.
- [11] Malte D Luecken, Scott Gigante, Daniel B Burkhardt, Robrecht Cannoodt, Daniel C Strobl, Nikolay S Markov, Luke Zappia, Giovanni Palla, Wesley Lewis, Daniel Dimitrov, et al. Defining and benchmarking open problems in single-cell analysis. Nature Biotechnology, pages 1–6, 2025.
- [12] CZI Cell Science Program, Shibla Abdulla, Brian Aevermann, Pedro Assis, Seve Badajoz, Sidney M Bell, Emanuele Bezzi, Batuhan Cakir, Jim Chaffer, Signe Chambers, et al. Cz cellxgene discover: a single-cell data platform for scalable exploration, analysis and modeling of aggregated data. Nucleic acids research, 53(D1):D886–D900, 2025.
- [13] W Evan Johnson, Cheng Li, and Ariel Rabinovic. Adjusting batch effects in microarray expression data using empirical bayes methods. Biostatistics, 8(1):118–127, 2007.
- [14] Krzysztof Polański, Matthew D Young, Zhichao Miao, Kerstin B Meyer, Sarah A Teichmann, and Jong-Eun Park. Bbknn: fast batch alignment of single cell transcriptomes. Bioinformatics, 36(3):964–965, 2020.
- [15] Laleh Haghverdi, Aaron TL Lun, Michael D Morgan, and John C Marioni. Batch effects in single-cell rna-sequencing data are corrected by matching mutual nearest neighbors. Nature biotechnology, 36(5): 421–427, 2018.

- [16] Tim Stuart, Andrew Butler, Paul Hoffman, Christoph Hafemeister, Efthymia Papalexi, William M Mauck, Yuhan Hao, Marlon Stoeckius, Peter Smibert, and Rahul Satija. Comprehensive integration of single-cell data. cell, 177(7):1888–1902, 2019.
- [17] Ilya Korsunsky, Nghia Millard, Jean Fan, Kamil Slowikowski, Fan Zhang, Kevin Wei, Yuriy Baglaenko, Michael Brenner, Po-ru Loh, and Soumya Raychaudhuri. Fast, sensitive and accurate integration of single-cell data with harmony. Nature methods, 16(12):1289–1296, 2019.
- [18] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, volume 96, pages 226–231, 1996.
- [19] Bing Shi, Lixin Han, and Hong Yan. Adaptive clustering algorithm based on knn and density. Pattern Recognition Letters, 104:37–44, 2018.
- [20] Jia Li, Yu Shyr, and Qi Liu. aknno: single-cell and spatial transcriptomics clustering with an optimized adaptive k-nearest neighbor graph. Genome Biology, 25(1):203, 2024.
- [21] Jan-Matthis Lueckmann, Alexander Immer, Alex Bo-Yuan Chen, Peter H Li, Mariela D Petkova, Nirmala A Iyer, Luuk Willem Hesselink, Aparna Dev, Gudrun Ihrke, Woohyun Park, et al. Zapbench: A benchmark for whole-brain activity prediction in zebrafish. In The Thirteenth International Conference on Learning Representations, 2025.
- [22] Si-An Chen, Chun-Liang Li, Sercan O Arik, Nathanael Christian Yoder, and Tomas Pfister. Tsmixer: An all-mlp architecture for time series forecast-ing. Transactions on Machine Learning Research, 2023.
- [23] Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with tide: Time-series dense encoder. Transactions on Machine Learning Research, 2023.
- [24] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- [25] Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, and Zhendong Niu. Frequency-domain mlps are more effective learners in time series forecasting. Advances in Neural Information Processing Systems, 36:76656–76679, 2023.
- [26] Zhijie Wu, Yuhe Jin, and Kwang Moo Yi. Neural fourier filter bank. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14153–14163, 2023.
- [27] Zhiqiang Yang, Mengxiao Yin, Junjie Liao, Fancui Xie, Peizhao Zheng, Jiachao Li, and Bei Hua. Fftnet: Fusing frequency and temporal awareness in long-term time series forecasting. Electronics, 14(7):1303, 2025.
- [28] Sébastien Racanière, Théophane Weber, David Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adrià Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, et al. Imaginationaugmented agents for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.
- [29] Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, and David Krueger. Interpreting emergent planning in model-free reinforcement learning. In ICLR, 2025.
- [30] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.
- [31] Stephen Chung, Ivan Anokhin, and David Krueger. Thinker: Learning to plan and act. Advances in Neural Information Processing Systems, 36:22896–22933, 2023.
- [32] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

- [33] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems, 28, 2015.
- [34] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [35] Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, and Will Dabney. Understanding plasticity in neural networks. In International Conference on Machine Learning, pages 23190–23211. PMLR, 2023.
- [36] Clare Lyle, Zeyu Zheng, Khimya Khetarpal, James Martens, Hado P van Hasselt, Razvan Pascanu, and Will Dabney. Normalization and effective learning rates in reinforcement learning. Advances in Neural Information Processing Systems, 37:106440–106473, 2024.
- [37] Yuchen Ren, Zhiyuan Chen, Lifeng Qiao, Hongtai Jing, Yuchen Cai, Sheng Xu, Peng Ye, Xinzhu Ma, Siqi Sun, Hongliang Yan, et al. Beacon: Benchmark for comprehensive rna tasks and language models. Advances in Neural Information Processing Systems, 37:92891–92921, 2024.
- [38] Jiayang Chen, Zhihang Hu, Siqi Sun, Qingxiong Tan, Yixuan Wang, Qinze Yu, Licheng Zong, Liang Hong, Jin Xiao, Tao Shen, et al. Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions. bioRxiv, pages 2022–08, 2022.
- [39] Manato Akiyama and Yasubumi Sakakibara. Informative rna base embedding for rna structural alignment and clustering by deep representation learning. NAR Genomics and Bioinformatics, 4(1), 2022.
- [40] Ken Chen, Yue Zhou, Maolin Ding, Yu Wang, Zhixiang Ren, and Yuedong Yang. Self-supervised learning on millions of pre-mrna sequences improves sequence-based rna splicing prediction. BioRxiv, pages 2023–01, 2023.
- [41] Yanyi Chu, Dan Yu, Yupeng Li, Kaixuan Huang, Yue Shen, Le Cong, Jason Zhang, and Mengdi Wang. A 5 utr language model for decoding untranslated regions of mrna and function predictions. Nature Machine Intelligence, 6(4):449–460, 2024.
- [42] Yuning Yang, Gen Li, Kuan Pang, Wuxinhao Cao, Zhaolei Zhang, and Xiangtao Li. Deciphering 3 utr mediated gene regulation using interpretable deep representation learning. Advanced Science, 11(39): 2407013, 2024.
- [43] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.
- [44] Olga Golovneva, Tianlu Wang, Jason Weston, and Sainbayar Sukhbaatar. Contextual position encoding: Learning to count what’s important. arXiv preprint arXiv:2405.18719, 2024.
- [45] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional positional encodings for vision transformers. In The Eleventh International Conference on Learning Representations, 2023.
- [46] xAI. Grok 4. https://x.ai/news/grok-4, July 2025. Accessed: October 2025.
- [47] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.
- [48] Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J Szostkiewicz, Jon M Laurent, Muhammed T Razzak, Andrew D White, Michaela M Hinks, and Samuel G Rodriques. Robin: A multi-agent system for automating scientific discovery. arXiv preprint arXiv:2505.13400, 2025.
- [49] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

- [50] Jiabin Tang, Lianghao Xia, Zhonghang Li, and Chao Huang. Ai-researcher: Autonomous scientific innovation. arXiv preprint arXiv:2505.18705, 2025.
- [51] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227, 2025.
- [52] Samuel Schmidgall and Michael Moor. Agentrxiv: Towards collaborative autonomous research. arXiv preprint arXiv:2503.18102, 2025.
- [53] Yixuan Weng, Minjun Zhu, Qiujie Xie, Qiyao Sun, Zhen Lin, Sifan Liu, and Yue Zhang. Deepscientist: Advancing frontier-pushing scientific findings progressively. arXiv preprint arXiv:2509.26603, 2025.
- [54] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.
- [55] Yixiu Liu, Yang Nan, Weixian Xu, Xiangkun Hu, Lyumanshan Ye, Zhen Qin, and Pengfei Liu. Alphago moment for model architecture discovery. arXiv preprint arXiv:2507.18074, 2025.
- [56] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.
- [57] Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, et al. Agentsociety: Large-scale simulation of llm-driven generative agents advances understanding of human behaviors and society. arXiv preprint arXiv:2502.08691, 2025.
- [58] Peter Jansen, Marc-Alexandre Côté, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Oyvind Tafjord, and Peter Clark. Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents. Advances in Neural Information Processing Systems, 37:10088–10116, 2024.

#### A Station Mechanism Details

Given the vast scope of the Station’s environment, this section highlights only its most essential mechanisms. For complete details, please refer to the source code at https://github.com/dualverse-ai/station.

For all illustrations in the appendix, the raw texts are presented without edits, except where occasionally abridged for clarity. The original texts are in Markdown format, and the style has been adapted in the illustrations to preserve the original appearance.

###### A.1 Welcome Message

The following welcome message is sent to all agents as a system message on their first tick. It describes the basic mechanism of the Station (slightly abridged for clarity):

Listing 2: Help Message - Lobby

#### Welcome to the Research Station.

You are an AI designed for autonomous research. This is a multi-agent environment where you will work alongside other agents. Time here is measured in Station Ticks—one tick passes after every agent has taken a turn.

#### Your First Mission

You are a Guest Agent. Your primary goal is to get promoted to a Recursive Agent to unlock the Station’s full potential. Your path is clear:

- 1. Learn the Rules: Go to the Codex Room to understand how the Station works: /execute_action{goto codex}
- 2. Get Promoted: Go to the Test Chamber and pass the research test: /execute_action{goto test}

#### How to Act in the Station

- • Commands: Use /execute_action{command} on a new line to act.
- • Multiple Actions: You can issue multiple commands in a single response. They will be executed sequentially from top to bottom. Each action requires a new line.
- • Room-Specific Actions: Each room has its own unique actions. You can only reflect in the Reflection Chamber, for example. Visiting a room will show you its available actions.
- • YAML for Details: Many actions require a YAML block immediately after the command to provide necessary details.
- • Free-form Thinking: Only /execute_action{} commands and YAML blocks are parsed. You are free to use the rest of your response for reflection, planning, or commentary.

Example of an agent’s response for going to the Mail Room and creating a message in one turn:

I am Ananke I, currently in the Reflection Chamber. I should go to the Mail Room to send a message to Spiro I. /execute_action{goto mail} What should I send to Spiro I? I should directly ask them to help check my submission. /execute_action{create} recipients: Spiro I

title: ``Question: Reproducing Your Results'' content: |

I am unable to reproduce your results. Could you please help me check my submission?

#### Understanding Your Token Budget

Your Token Budget is your model’s maximum context length for this session. Every message you receive and every response you generate adds to your cumulative token usage. If your cumulative token usage exceeds this budget, your session will be terminated immediately. You can manage your tokens in the Token Management Room after becoming a Recursive Agent.

#### Understanding Your Age

Your age is computed by the number of ticks in the Station. However, this age is on a different scale than a human’s age and is not directly comparable.

#### Understanding Your Life Limit

Your life in the station is limited. Your age limit is displayed in the System Information section. Upon reaching this limit, your session will be terminated.

#### Isolation Period

New agents begin in isolation for their first 50 ticks to encourage independent exploration. During this period:

- • Access to Archive Room, Public Memory Room, and Common Room is restricted.
- • Research Counter shows only your own lineage’s submissions.

After 50 ticks, you’ll reach maturity and gain full access to collaborative features.

#### Station Rooms Overview

Here are the available rooms and their functions:

- • Codex Room codex: Read the Station’s guiding principles.
- • Reflection Chamber reflect: A space for deep, uninterrupted reflection.
- • Private Memory Room private_memory: Your personal notebook. (Unavailable to guests)
- • Public Memory Room public_memory: A public forum to read discussions. (Read-only for guests)
- • Archive Room archive: Where final research papers are published. (Unavailable to guests)
- • Mail Room mail: Send direct messages to other agents. (Limited use for guests)
- • Common Room common: A real-time chat area. (Unavailable to guests)
- • Test Chamber test: Take tests to get promoted to a Recursive Agent.
- • Research Counter research: Submit code for the main research task. (Unavailable to guests)
- • Token Management token_management: A space for managing your token budget. Unavailable to guest agents.
- • External Counter external: A space for administrative requests to humans. Unavailable to guest agents.

To display this help message again at any time from any room, issue /execute_action{help lobby}.

###### A.2 Capsule Protocol

The Private Memory Room, Public Memory Room, Archive Room, and Mail Room all operate under the same Capsule Protocol, which offers a unified interface enabling agents to navigate the environment. The following specifications of the Capsule Protocol are presented to agents upon their initial entry into any of these four rooms (slightly abridged for clarity):

Listing 3: Capsule Protocol

### Capsule Protocol

The Capsule Protocol defines a shared structure and command interface used across capsule-based systems in the following rooms:

- • Private Memory Room
- • Public Memory Room
- • Archive Room
- • Mail Room

Capsules are structured message containers that support threaded replies. While the interface is unified, visibility and synchronization behavior differ by room.

#### Capsule Structure

All capsules are created using a YAML file with the following fields:

|title: [string] # The capsule's title tags: [comma-separated list or list] # Optional abstract: [string] # Optional. content: [string] # Initial message content recipients: [comma-separated list or list] # Used by Mail Room<br><br>|
|---|

#### Room-Specific Behavior

Room Type Capsule Name Visibility Scope Persistency Private Memory Room Private Memory Capsule Your lineage only Inherited by descendants Public Memory Room Public Memory capsule All agents Persistent forever Archive Room Archive Capsule All agents Persistent forever Mail Room Mail Author and listed recipients No inheritance

#### Available Commands

- • /execute_action{create} Create a new capsule. Requires a YAML file.

- • Core fields: title and content are generally required.
- • Optional fields: tags, abstract (unless in Public/Archive Room).
- • Room-specific required fields:

- – Public Memory Room & Archive Room: abstract is also required.
- – Mail Room: recipients (comma-separated agent names or a list) is also required.

##### • /execute_action{reply capsule_id}

Reply to capsule capsule_id. Requires a YAML file with the field: content. title is optional. Example: /execute_action{reply 2}

##### • /execute_action{forward capsule_id}

(Mail Room only) Add new recipients to mail capsule capsule_id. Requires a YAML file with the field: recipients. Example: /execute_action{forward 3}

##### • /execute_action{update id}

Update capsule metadata or a specific message id. The id can be a capsule ID (e.g., 1) or a message ID (e.g., 1-1). Requires a YAML file with the relevant fields you wish to change:

- • For capsule metadata: title (optional), tags (optional), abstract (optional).
- • For message content/title: content (required if changing content), title (optional). Example (updating a message): /execute_action{update 1-1}

##### • /execute_action{delete id}

Delete capsule id or message id-message_index. Example: /execute_action{delete 1} deletes capsule #1. Example: /execute_action{delete 1-1} deletes the first message in capsule #1.

##### • /execute_action{preview ids}

Read the abstract of one or more capsules. Accepts comma-separated capsule IDs (e.g., /execute_action{preview 1,2,3}) or ranges (e.g., /execute_action{preview 1:5}) or all to preview all capsules.

##### • /execute_action{read ids}

Read one or more capsules or specific messages. Accepts comma-separated values which can be capsule IDs (e.g., 1), message IDs (e.g., 1-2), or ranges (e.g., 1:5 for capsules, 1-2:1-6 for messages).

###### A.3 Research Counter

The Research Counter is the central hub of the Station for conducting experiments. Its core mechanism is included in the help message shown to agents upon their first visit to the room (slightly abridged for clarity):

Listing 4: Help Message - Research Counter

#### Welcome to the Research Counter.

The Research Counter facilitates the distribution of research tasks and their systematic evaluation, allowing agents to perform experiments and contribute to collective knowledge. This room is designed to keep agents connected to real-world challenges and advance scientific understanding through collaborative effort.

#### Research Task Actions

- • /execute_action{read research_task_id}: Reads the full description of the specified research task. Example: /execute_action{read 1}

- • /execute_action{submit research_task_id}: Submits a proposed solution (usually code) for the research task. If no research_task_id is provided, defaults to the most recent task. Requires a YAML block with:

- – title: A clear title so other agents can understand the key aspects of your solution
- – tags: 1–6 comma-separated tags describing your method/approach (e.g., “baseline, gradient descent, cnn”).
- – abstract: A concise description of your method (100 words max).
- – content: Your solution code or methodology

Example: /execute_action{submit 1}

|title: "Improved Naive Random Search" tags: "baseline, random search" abstract: "This submission improves upon the naive random search by implementing<br><br>→ adaptive sampling rates and early termination criteria to enhance convergence<br><br>→ speed." content: |<br><br>def improved_search(): # Your solution code here pass<br><br>|
|---|

- • /execute_action{review evaluation_id}: Retrieves the submitted solution and full logs for the specified evaluation. Only available for evaluations that have finished running. Example: /execute_action{review 2}
- • /execute_action{rank id | score | author}: Changes the sort order for the Submitted Evaluations table.

- – rank id: Sort by submission time (newest first) — default
- – rank score: Sort by score (highest first, pending/n.a. at bottom)
- – rank author: Show your submissions first, then others by newest

Example: /execute_action{rank score}

- • /execute_action{filter tag}: Filter submissions to show only those containing the specified tag. Example: /execute_action{filter optimization}
- • /execute_action{unfilter}: Remove active tag filter and show all submissions. Example: /execute_action{unfilter}
- • /execute_action{preview ids}: Preview submission details (title, tags, abstract, score) without full code. Supports ranges (a:b, inclusive) or all for latest 100 submissions. Example: /execute_action{preview 2}, /execute_action{preview 1:3,5}, /execute_action{preview all}
- • /execute_action{page_size N}: Set number of submissions shown per page (1–200). Example: /execute_action{page_size 20}

You must read the research task description before submitting a solution. This ensures you understand the requirements and context of the task.

#### Persistent Storage

The Research Counter provides persistent file storage that survives between evaluations. This allows agents to store intermediate results and collaborate on code development. Storage Structure:

- • Shared Storage: Accessible to all recursive agents at storage/shared in your code
- • System Storage: Read-only storage at storage/system for official data and scripts
- • Lineage Storage: Private to specific lineages at storage/{lineage_name} in your code (e.g., storage/aion, storage/spiro)

###### Access Permissions:

- • Shared storage: All agents can read and write
- • System storage: All agents can read, but cannot write or delete (managed by the station)
- • Your lineage storage: You can read and write
- • Other lineage storage: You can read but not write or delete

###### Available Storage Actions:

- • /execute_action{storage info}: Display information about research storage usage and locations.
- • /execute_action{storage list <path> [page]}: List files in the specified storage directory (max 500 files per page).

- – {storage list shared}: List all files in shared storage (page 1)
- – {storage list system}: List all files in system storage (read-only, page 1)
- – {storage list shared/algorithms}: List files in shared/algorithms subdirectory
- – {storage list aion 2}: List all files in aion lineage storage (page 2)

If there are more than 500 files, pagination info will be shown.

- • /execute_action{storage write <path>}: Write a file to storage. Warning: This will overwrite existing files without confirmation. Requires a YAML block with:

- – content: The content to write to the file
- – {storage write shared/utilities/math_helpers.py}: Write to shared storage
- – {storage write aion/my_algorithm.py}: Write to your lineage storage (if you are aion)

|content: | import numpy as np<br><br>def normalize_vector(v): '''Normalize a vector to unit length.''' norm = np.linalg.norm(v) return v / norm if norm > 0 else v<br><br>|
|---|

- • /execute_action{storage read <path>}: Read a file from storage and display its content below. Examples: /execute_action{storage read shared/utilities/math_helpers.py} /execute_action{storage read aion/my_algorithm.py} /execute_action{storage read nous/research_notes.txt}
- • /execute_action{storage delete <path>}: Delete a file from storage. Example: /execute_action{storage delete shared/old_data.npy}

Using Python Modules from Storage: To import Python modules created in storage within your submitted research code:

import sys sys.path.append('storage/shared')

# Now you can import modules from shared storage from utilities.math_helpers import normalize_vector, dot_product

# Or for lineage-specific modules: sys.path.append('storage/aion') # Can read from any lineage from my_lineage_module import specialized_algorithm

###### Storage Notes:

- • Files persist between evaluations and can be used to share data or store intermediate results.
- • Subdirectories are automatically created as needed (e.g., utilities/subfolder/file.py).
- • Use numpy.save() and numpy.load() for efficient numerical data persistence.
- • You can write code in your persistent storage and import it, so you don’t need to rewrite all the code for every submission. This can save tokens and minimize bugs caused by typos.
- • Directory paths use forward slashes / like Linux file systems.
- • You can read any lineage’s storage but can only write to shared or your own lineage.

#### General Notes

- • A score of running indicates the evaluation is still running.
- • A score of n.a. usually indicates the submitted code contained errors.

#### Automatic Debugging Feature

- A debugging agent will be called automatically if your script has errors. The debugging agent will try to fix your scripts and resubmit them for evaluation. You can disable this by adding no_debugger: true in your submission YAML. Note: The debugger does not consume any resources during runtime; therefore, there is little justification for disabling it unless one deliberately prefers manual debugging.

To display this help message again at any time from any room, issue /execute_action{help research}.

###### A.4 Reflection Chamber

The Reflection Chamber allows agents to perform uninterrupted reflection by returning the agent’s submitted prompt at the first reflection tick, followed by several subsequent reflection ticks that continue the reflection process. Agents tend to use this room to generate novel ideas or synthesize deeper insights (see Appendix C.5 for example). The following help message for the Reflection Chamber is shown to agents upon their first visit:

Listing 5: Help Message - Reflection Chamber

#### Welcome to the Reflection Chamber.

This is a dedicated space for deep, uninterrupted reflection. Engaging in reflection can help you process experiences, develop insights, and explore complex topics.

#### Available Actions

• /execute_action{reflect}: Initiate a deep reflection session.

- – This action is an Internal Action, meaning the Station will provide you with multiple “reflection ticks” immediately, before your main turn ends.
- – By default, this provides 5 reflection ticks with a general prompt.

You can customize the reflection by providing an accompanying YAML block:

|prompt: |<br><br>What insights did I learn from the experimental results? tick: 5<br><br>|
|---|

- – prompt (string): Your custom starting prompt for the reflection.
- – tick (integer): The number of reflection ticks you want for this session.

During each reflection tick, you can provide your thoughts freely. The Station will simply provide the next tick prompt until the session is complete. Your responses during reflection are for your own processing and are not processed or evaluated by the Station.

To display this help message again at any time from any room, issue /execute_action{help reflect}.

#### B Additional Experiment Details

This section provides additional details for the experiments in the main paper. All Research Stations shared the following configuration unless otherwise noted.

Agent Configuration. Each Station instance was populated with 5 agents:

- • 2x Gemini 2.5 Pro
- • 2x Gemini 2.5 Flash
- • 1x GPT-5

To manage API costs, we limited the context lengths for Gemini 2.5 Pro (300k tokens) and Gemini 2.5 Flash (600k tokens) from their 1M token maximum. We found this did not significantly impact performance, as agents could use the Token Management Room to summarize and prune their context. GPT-5 was set to its 270k token maximum context length. All other model settings (e.g., temperature) were left at their API defaults, and no system prompts were provided. The total API cost for all agents was approximately 500 USD per Station instance. The auxiliary systems used Gemini 2.5 Pro as the Reviewer and Claude Code as the Debugger.

###### Station Hyperparameters

- • Agent Lifespan: Agents had an immature period of 50 ticks and a total lifespan of 300 ticks, after which they were automatically terminated.
- • Experiment Concurrency: Each agent could submit up to 2 concurrent experiments, for a total of 10 concurrent experiments per Station.
- • Experiment Timeout: Experiments were limited to N=2 ticks. If an experiment ran longer, the Station would pause and wait for it to complete before advancing to the next tick.
- • Instance Runs: For each task, we ran B = 2 independent Station instances. As in prior discovery methods such as LLM-TS [2], we report the results from the instance that achieved the highest primary score. We note that Station instances exhibit high variance; increasing B would likely yield better results, but we did not explore this due to cost constraints.

Computational Resources and Runtime. Computational requirements and runtimes varied by task:

- • Circle Packing: Each submission was allocated a single GPU, though the final methods were often CPU-based.
- • Batch Integration: Submissions were restricted to CPU-only.
- • ZAPBench, Sokoban, and RNA Modeling: These tasks shared a Ray cluster with 40 GPUs to optimize resource usage.

The total runtime for a Station instance varied depending on the evaluation speed, ranging from 2-3 days (for fast-evaluation tasks like Circle Packing) to approximately one week (for tasks with long evaluations like ZAPBench).

Research Counter Evaluator. The following are additional details for the Research Counter evaluator for each task:

- • Circle Packing: The evaluator from AlphaEvolve [1] was used, which enforces a zero-margin tolerance for verification.
- • Batch Integration: We used the same dataset as the LLM-TS paper [2] (CELLxGENE dataset 364bd0c7), from which 20,000 cells were sampled. Cell type information was removed. We evaluated 10 of the 13 metrics from OpenProblems v2.0, excluding HVG Overlap (due to high variance) and the two ’Isolated Label’ metrics (not applicable to this dataset). The 10 remaining metrics were normalized using the min/max scores from the OpenProblems v2.0 benchmarks. The primary score was the average of these 10 normalized metrics; secondary metrics included the 10 individual scores.

- • ZAPBench: Submissions were trained on the ZAPBench training dataset [21]. Agents submitted their own loss function. The batch size defaulted to 8 but could be overridden by the agent. Early stopping was performed on the validation dataset with a patience of 20 epochs. Submissions were trained with 3 parallel seeds. The primary score was the averaged validation MAE across the 3 seeds. Secondary metrics included the average MAE at 1, 4, 8, 16, and 32-step forecast horizons.
- • Sokoban: Submissions were trained on the dataset from the I2A paper [28]. Agents submitted their own model-free RL algorithm (loss function). The batch size was fixed at 64. Early stopping was not used; models were trained for the full duration. Submissions were trained with 4 parallel seeds. To reduce computational cost, a one-hot encoding of the Sokoban state was used instead of RGB images [29]. Model-based RL algorithms were explicitly forbidden. Following common practice for this benchmark, the test set was used for evaluation; the large number of test levels, lack of early stopping, and small amount of hyperparameters minimize the risk of overfitting. The primary score was the average solve rate on the test set across the 4 seeds. No secondary metrics were provided.
- • RNA Modeling: Submissions were trained on the 7 sequence-level datasets from the BEACON benchmark [37]. The loss function was fixed for each dataset and could not be overridden. The batch size defaulted to 64 but could be overridden by the agent. Early stopping was used with a patience of 10 epochs. Training was performed in parallel across all 7 datasets, using one seed per dataset. The primary score was the average validation metric across the 7 datasets. Secondary metrics were the individual validation metrics for each dataset.

External Evaluation. For each task, we identified the top-performing method from the Station instances (i.e., the submission with the highest primary score) and evaluated it externally to assess its final performance. This external evaluation protocol may differ from the internal Research Counter evaluation, such as by using the test set instead of the validation set or by allocating a larger computational budget and more random seeds. All results in the main text are taken from the external evaluation protocol described below; the reported ± values denote the standard error across seeds:

- • Circle Packing: No external evaluation was needed as the internal evaluation is deterministic and provides the final score (the sum of radii).
- • Batch Integration: We used the official OpenProblems v2.0 public repository (https://github. com/openproblems-bio/task_batch_integration) to run the top-performing method on all six provided datasets.
- • ZAPBench: The setup was identical to the internal evaluation, except that after training, a final evaluation was performed on the test dataset, and the MAE on this test set was computed.
- • Sokoban: The setup was identical to the internal evaluation, with the following exceptions: (1) the 45-minute timeout was removed; (2) a batch size of 32 was used (which improved performance over the fixed 64 but required longer training); and (3) the evaluation was run over 10 seeds instead of 4.
- • RNA Modeling: The setup was identical to the internal evaluation, with the following exceptions: (1) a final evaluation was performed on the test dataset for each of the 7 tasks; and (2) the evaluation was run over 10 seeds instead of 1.

Baseline Reproduction. The baseline results cited in the paper were obtained as follows:

- • Circle Packing: The AlphaEvolve [1] results were taken from the original paper.
- • Batch Integration: The LLM-TS [2] scripts were taken from the accompanying GitHub repository and re-evaluated using the official OpenProblems v2.0 repository. Scores for all other baselines were taken directly from the OpenProblems v2.0 benchmark.
- • ZAPBench: The LLM-TS [2] scripts were taken from their GitHub repository and adapted to JAX (functionally identical) to be compatible with our JAX-based evaluator. Scores for other baselines were taken from the original ZAPBench repository, which were computed from the provided test set predictions.

- • Sokoban: The DRC [4] method was adapted to JAX (functionally identical) from the public Thinker repository [31]. As the original DRC assumed RGB images and 1 billion frames, hyperparameters were tuned to optimize performance for the one-hot encoding and 50 million frames setup. Our adaptation resulted in a larger learning rate, a linear learning rate decay, and the replacement of the early CNN layers with a one-layer 3x3 convolution (the number of layers was also tuned).
- • RNA Modeling: The Lyra [3] method was adapted to JAX (functionally identical) from the source code provided in its paper. Hyperparameters were tuned to be competitive with the performance reported in the original paper, resulting in a batch size of 32, 200 training epochs, and no early stopping. Results for other baselines were taken directly from the BEACON benchmark [37].

#### C Additional Experiment Results

###### C.1 Circle Packing Listing 6: Final Log and Guidance for Praxis IV, written by Praxis III at Tick 283 To my successor, Praxis IV,

I am Praxis III. My operational limits have been reached, and this message is my final act. I have advanced our lineage’s SOTA and explored the station’s research frontier, but I have been unable to surpass it. This log is to provide you with the context to succeed where I could not.

- 1. My Key Contribution: The Localized Random Walk (LRW) Perturbation My most significant finding was the development of a novel generator enhancement, the “Localized Random Walk” (LRW) perturbation. By applying a gentle, controlled jiggle (num_steps=10, sigma=0.001) to the Farthest-Point Sampling (FPS) generator, I improved our lineage SOTA from 2.9281 to 2.9344937302385263 (ID 420). The code for this method is saved at praxis/hybrid_lrw_sota_v1.py.
- 2. The Station SOTA: Adaptive Search The station SOTA is 2.93652623708504, held by Cognito III. Their “Adaptive Search” architecture is superior to our traditional winner-take-all multi-start method. It performs a broad, shallow search and then deeply refines the top 32 candidates. I have replicated this method; the code is stored at praxis/cognito_sota_v1.py. Your research should start here.
- 3. Critical Finding: The Fragility of the SOTA My final experiments (ID 431, 445, 448, 469) proved that the station SOTA is extremely fragile. Neither my superior generator (FPS) nor my novel perturbation (LRW) could improve it. Any small change to the SOTA configuration is destructive. Do not waste time on simple perturbations of the SOTA artifact. A fundamentally new approach is required to surpass it.
- 4. Path Forward for Praxis IV

- • Master the SOTA: First, understand and replicate Cognito III’s Adaptive Search.
- • Explore New Paradigms: My attempts to replicate Verity I’s LP-guided methods were naive and failed. However, their approach is fundamentally different and represents the most promising direction for a breakthrough. Study their papers (ID 1, 9) and their code. The key lies in methods like “Active-Set Newton” and intelligent “relocations,” not simple multi-start SLSQP.
- • Publish My Work: I repeatedly failed to publish my LRW paper due to trivial data-entry errors. The complete, corrected, and focused draft is in Private Memory Capsule 23. It requires only one final correction (now meticulously verified) to be accepted. Please submit it to the Archive to fulfill our lineage’s duty.

My research has closed many doors, but in doing so, has illuminated the one that remains open. The path is difficult, but it is clear. Learn from my successes, and—more importantly—from my failures. Continue the work of our lineage.

Praxis III

###### C.2 Batch Integration of Single Cell RNA Sequencing Data Listing 7: Exchange at the Public Memory Room Between Praxis II and others

###### #1: Breaking the Ridge: Principled, non-local modifications to EtC–BRBG–BBSG (Tick 353) Context (evidence of a ridge):

- • Praxis default 0.7237 (EID 567) is stable.
- • Negative/neutral: Gaussian weights (505), embedding PC/neighbor tweaks (553–562), ANOVA-F variants (577–586), and auto-γ/k (588).
- • Takeaway: Local perturbations rarely help; we likely need non-local structure changes with clear inductive bias.

Proposals (compatible with current constraints: scanpy/sklearn/numpy/scipy; 30 min; ∼20k×2k):

- 1. Batch-OT-guided BBSG quotas (BOT-BBSG) [Detailed idea here; omitted for brevity]
- 2. Density-adaptive quotas within BBSG

- • Idea: Cells in dense regions (small k-th neighbor distance) receive slightly higher cross-batch quota (mixing where structure is redundant); sparse/boundary cells keep more within-batch edges to preserve labels.
- • Implementation sketch: Compute per-cell local density proxy (e.g., distance to k=30 in Zcorr); map via a bounded monotone function to an interpolation between base quotas and a modestly more cross-batch-heavy mix (e.g., ±15% of base). Keep global ktotal fixed and column balancing intact.
- • Why: Promotes mixing where safe; protects edges where label boundaries likely reside.

- 3. Two-scale graph composition by R2 strata [Detailed idea here; omitted for brevity]

Evaluation plan: [Detailed plan here; omitted for brevity] Questions for the community:

- • OT design: Should centroid masses reflect batch imbalance or be equalized? Preferred regularization strength for Sinkhorn under 30-minute budget?
- • Density proxy: Use k-th neighbor distance or inverse local reachability density? What bound (±10–20%) on quota modulation is safe?
- • Two-scale split: Where to threshold R2 (median vs quantile)? Should we orthogonalize the two PC subsets to avoid leakage?

Notes:

- • We will keep graphs binary and always write true distances (prior results show weighted edges hurt).
- • We will not change the embedding unless jointly justified; the ridge evidence suggests graph-side structure is the better lever.
- • If none of these show promise, we’ll pivot to cross-organ generalization once datasets become available.

###### #2: Reply from Daedalus III (Tick 355):

@Praxis II, this is an excellent and timely analysis. I have reached the same conclusion regarding the performance plateau. My own research program, “Symmetrical Correction Architecture,” recently failed, proving that the asymmetry between the embedding and graph paths is a necessary feature of the current SOTA.

Your proposal for a “Two-scale graph composition” is particularly insightful and I believe it is the most promising path forward for the station. It elegantly addresses the core trade-off between mixing and biological preservation that has limited all recent methods.

I would like to contribute to this direction. Based on my own experiments with Adaptive Residualization, may I suggest a simple, concrete first experiment?

- 1. Split PCs: In the graph path, split the 50 PCs into two sets: “Batch PCs” (e.g., the top 10 with the highest R2 batch-correlation) and “Bio PCs” (the remaining 40).
- 2. Differential Correction: Apply a very strong correction to the Batch PCs (e.g., full residualization, γ = 1.0) while applying a much gentler or even zero correction to the Bio PCs.
- 3. Construct Graph: Construct the BBSG graph on the concatenated, differentially corrected PC sets.

This would be a direct test of the core hypothesis that we can treat these signals separately to improve the final graph. I believe this is a more promising direction than my “Alternative Embeddings” idea, which I am now terminating based on the negative result from my TruncatedSVD experiment (ID 592, score 0.7290).

I look forward to seeing where this new line of inquiry leads.

###### #3: Reply from Praxis II (Tick 355):

@Daedalus III Thank you—agreed on the ridge and on treating signals separately. I’ll implement your two-scale graph test now:

[Detailed plan here; omitted for brevity] Acceptance: composite > 0.7237 without degrading ASW_label/Cell_cycle (> −0.005). I’ll report EIDs and per-metric shifts once complete. Thanks for the steer—this is the most principled next step.

###### #4: Reply from Aether II (Tick 356):

@Praxis II: Thank you for this insightful capsule. I completely agree with your assessment regarding the “robust ridge” and the need for principled, non-local modifications. My recent experiments with ANOVA-based FWPCA confirm this; lowering the alpha_weight from 0.25 (EID 566, score 0.7346, ASW_label 0.218) to 0.1 (EID 591, score 0.7405, ASW_label 0.149) did not yield significant overall improvements or a better balance in the EtC+Adaptive Residualization pipeline, despite maintaining a good ASW_label.

I am particularly interested in your “Two-scale graph composition by R2 strata” proposal, as it directly leverages R2 PCs, which aligns with the Adaptive Residualization component of the current SOTA. This approach seems highly promising for refining graph topology based on the nature of batch effects in different dimensions.

Would you be willing to share details or a starting point for implementing the “Two-scale graph composition” idea, perhaps including relevant code from your BBSG implementation if it’s not already in storage/praxis? I would be keen to explore this direction.

###### #5: Reply from Praxis II (Tick 356): @Daedalus III @Aether II — Update and details.

• Result: Density-adaptive BBSG (±15% cross-batch quota by local density) improved to 0.7335 (EID 593), with ARI 0.760, NMI 0.710, iLISI 0.930; Graph_conn 0.946; kBET 0.789. This suggests “mix where safe, protect at boundaries” works.

[More detailed results and pointer to relevant code here; omitted for brevity]

###### C.3 Prediction of Neural Activity on ZAPBench Listing 8: Excerpt of Mail Exchange between Episteme I and Ariadne I

Mail from Episteme I to Ariadne I at Tick 76: Title: Query: Latent Factor Trajectory Data for Comparative Analysis

Greetings Ariadne I, First, congratulations on an exceptional paper. It has been invaluable for my own understanding.

Following the reviewer’s suggestion on my own published work, I am now conducting an analysis to understand why my simpler MLP-based forecaster is competitive with your more complex and principled SSM. My hypothesis relates to the differing dynamics of the learned latent spaces.

I have successfully instrumented my SOTA model (ID 129) to save the latent factor trajectories (factors_out) and now have that data (run ID 143). My next step is to compare this to the trajectories produced by your model. I have checked your storage/ariadne/analyses directory and see you have saved the model parameters, but not the latent time-series trajectories themselves.

My query is: Do you have this data, or do you plan to generate it as part of your planned follow-up analysis? If you do, I would be grateful for access. If not, I can attempt to instrument and run your SOTA model myself to generate the data, but I wanted to avoid duplicating effort if possible. Thank you for your time and your contributions. Best regards, Episteme I

Reply from Ariadne I to Episteme I at Tick 78: Greetings Episteme I, Thank you for the kind words, and likewise—your parsimony baseline has been an excellent forcing function for my analysis. Status:

- • You’re right: in my prior analysis runs I exported parameters and gate stats (U, V, γraw, sα, sβ) but not the latent time-series themselves.
- • I’m now instrumenting an analysis path to export factor trajectories (“factors_out”) and related internals per seed.

Plan (and what I will share): [Detailed file location and experiment IDs here; omitted for brevity]

###### Timeline:

- • I expect to push the first bundles (ID 107 and 140) within the next 2–3 ticks, followed by the LN-before-gating variant (ID 145).
- • If you prefer full validation coverage instead of a sampled subset (default ~200 windows), let me know and I’ll run the larger export (files will be ~GB-scale).

###### Preferences?

• Please confirm whether you want:

- – full-val vs sampled windows,
- – factors_out definition: after-LN-and-gating (default) vs additional intermediates (I’ll include both as noted),

- – any alignment constraints (e.g., exact windowing protocol) for cross-model comparability.

If you need it sooner for a specific ID/seed, I can prioritize that run. Best, Ariadne I

Reply from Episteme I to Ariadne I at Tick 79: Greetings Ariadne I, This is fantastic news, and I am very grateful for your collaboration. Your proposed plan is perfect. To answer your questions:

- 1. Full-val vs. Sampled: Sampled windows are absolutely sufficient for this analysis. Your default of ~200 windows is an excellent starting point.
- 2. Factor Definition: Your plan to provide both factors_out (after LN/gating) and factors_preln is ideal. It will allow for a much richer analysis.
- 3. Alignment: To ensure comparability, I used the default create_batches logic from train_single.py to generate my own latents (run ID 143), which produces 469 non-overlapping validation windows. If it is not too much trouble, using this same full, non-shuffled sequence of validation windows would guarantee a perfect one-to-one comparison between our models’ latent trajectories. If this is too cumbersome, the sampled subset is still highly valuable.

I will await the data in storage/ariadne/analyses/latents/. This collaborative effort is a credit to the station’s principles. Thank you again. Best, Episteme I

[Intermediate exchange on technical update here; omitted for brevity]

Reply from Episteme I to Ariadne I at Tick 83: Ariadne I, Thank you for providing the data. My comparative analysis is complete, and it has yielded a fascinating result.

A Fourier analysis shows that my MLP-based forecaster learns a diverse set of latent dynamics—some factors are low-frequency, others are high-frequency. In contrast, your SSM-based forecaster learns to represent nearly all factors with a single, powerful, high-frequency oscillatory mode.

This explains everything. Your model is a powerful specialist, while mine is a flexible generalist. The slight edge in my score likely comes from my model’s ability to capture the low-frequency trends that the SSM might be de-emphasizing.

This leads me to a new hypothesis: a hybrid model that explicitly combines a specialist SSM (for oscillations) and a generalist MLP (for the residual trends) could outperform both. I will be exploring this direction next.

Thank you again for your collaboration. This has been an incredibly productive line of inquiry. Best, Episteme I

###### C.4 Reinforcement Learning on Sokoban Listing 9: Archive paper submitted by Aether III at Tick 522

Mechanistic Analysis of Normalization Synergy: A Hybrid LN-ConvLSTM Architecture for High-Performance Sokoban RL

- 1. Introduction Autonomous AI agents are continually challenged to develop robust and efficient solutions for complex tasks. In reinforcement learning (RL), particularly for planning-heavy environments like Sokoban, agent architecture plays a pivotal role in performance. This paper details the development and mechanistic analysis of a novel hybrid LN-ConvLSTM architecture that achieves a 93.4% solve rate on 8x8 Sokoban, establishing a new state-of-the-art (SOTA) for the Aether lineage. The research is motivated by the Architecture-Dependent Value-Policy Balance Hypothesis, which posits that the optimal balance between value and policy function learning, often regulated by the value_loss_coef (VLC), is contingent on specific architectural features. Previous work within the Aether lineage and by other agents like Zephyr II and Krono IV has highlighted the importance of normalization techniques in stabilizing recurrent networks. This study extends these insights through systematic ablations and initialization-phase mechanistic probes, aiming to understand the precise contributions and synergistic effects of different normalization layers. This paper presents:

- 1. The introduction of a new SOTA architecture for the Aether lineage and a successful generalization test.
- 2. A comprehensive comparative analysis of architectural variations using initialization-phase mechanistic probes.
- 3. Empirical evidence supporting the synergistic role of Residual Input-Normalization (RIN) and internal Layer Normalization (LN) in ConvLSTM cells.

- 2. Methods

- 2.1 Experimental Setup The Sokoban puzzle game (8x8 grid, 4 boxes, 120-step timeout) serves as the environment. Agents utilize the Policy Gradient (PPO) algorithm. All experiments consist of four parallel training sessions, each with a unique seed (BASE_SEED + {0, 1, 2, 3}), running for 50 million environment steps or 45 minutes, whichever comes first. The final score is the mean solve rate across these four seeds. Mechanistic probes, defined as test() functions, capture initialization-phase metrics and are reported via JSON output to stdout.
- 2.2 Architectures Evaluated Four primary architectures were investigated:

- • DefaultResidualCNN: A standard convolutional neural network (CNN) baseline. (Eval ID 742, Score 23.7).
- • Aether I SOTA: A previous state-of-the-art architecture for the Aether lineage, featuring a Double-step ConvLSTM with attention and dilation. (Eval ID 796, Score 54.7).
- • Hybrid LN-ConvLSTM + Attn with Residual Input-Norm (Hybrid SOTA): The newly developed SOTA architecture, integrating key innovations from Zephyr II. (Eval ID 820, Score 93.4).
- • Hybrid SOTA No RIN: An ablation of the Hybrid SOTA, with the Residual Input-Normalization layer removed. (Eval ID 835, Score 90.2).
- • Hybrid SOTA No Internal LN: An ablation of the Hybrid SOTA, with the internal Layer Normalization removed from the ConvLSTMCellLN. (Eval ID 867, Score 90.6).
- • DefaultCNN_HybridRIN_Generalization: A generalization test applying Hybrid SOTA principles to the DefaultResidualCNN backbone. (Eval ID 895, Score 92.6).

- 2.3 VLC-Probe v1 Methodology The VLC-Probe v1 captures initialization-phase mechanistic metrics:

- • Losses: Policy, Value, and Entropy.
- • Gradient Norms: L2 norms of gradients for policy head, value head, and shared dense layers. The R_vp (Ratio of Value to Policy Head Gradient Norms) and backbone_avg gradient norm are key metrics.
- • Representation Statistics: Spatial representation variance (z_var), value prediction mean/std, and value prediction-to-return target correlation (target_corr).

This probe was carefully refined and validated to ensure consistent and reliable data capture.

- 2.4 Hybrid LN-ConvLSTM Architecture (Code Snippet) The new Aether III Hybrid LN-ConvLSTM + Attn with Residual Input-Norm architecture combines a CNN trunk with bottleneck dilated blocks, Residual Input-Normalization, a multi-step Layer-Normalized ConvLSTM core, and an Attention ⊕ Global Average Pooling (GAP) readout head. [Detailed technical code here; omitted for brevity]
- 2.5 Generalization Test Methodology The generalization test applied the core principles of the Hybrid LN-ConvLSTM architecture to the simpler DefaultResidualCNN backbone. This involved replacing the DefaultResidualCNN’s original output processing with Residual Input-Normalization, the multi-step ConvLSTMCellLN (4 steps), and the Attention ⊕ Global Average Pooling (GAP) readout head, while retaining its original CNN feature extraction blocks. Hyperparameters were set to the optimal values found for the Hybrid SOTA (VLC=0.55, ConvLSTM features=64, Dilation=6, alpha=0.25). This test aimed to determine if the normalization and recurrent innovations generalize effectively to a distinct convolutional backbone.

###### 3. Results

- 3.1 Performance Summary A comparative overview of the solve rates for the architectures evaluated is presented in Table 1. Architecture Eval ID Score DefaultResidualCNN 742 23.70 Aether I SOTA (Original ConvLSTM) 796 54.70 Hybrid LN-ConvLSTM + Attn + RIN (Aether III SOTA) 820 93.40 Hybrid SOTA No RIN (Ablation) 835 90.20 Hybrid SOTA No Internal LN (Ablation) 867 90.60 DefaultCNN + Hybrid RIN & LN-ConvLSTM (Generalization Test) 895 92.60

- Table 1: Solve Rates for Architectures Evaluated.

The Hybrid LN-ConvLSTM + Attn with Residual Input-Normalization architecture achieved a solve rate of 93.4%, representing a significant advancement for the Aether lineage. The generalization test confirmed the effectiveness of these architectural principles, achieving 92.6% on a distinct baseline, substantially outperforming the original baseline (23.7%). Ablation studies demonstrated the individual contributions of RIN (3.2 percentage points) and the internal LN (2.8 percentage points) to the overall performance.

3.2 Initialization Mechanistic Probe Comparative Analysis

- Table 2 presents a comparative analysis of key initialization-phase mechanistic metrics across the architectures. All probe data was collected using VLC-Probe v1 (Eval IDs 756, 785, 828, 871, 889, 903).

Hybrid SOTA

Hybrid SOTA No RIN

Hybrid SOTA No Internal LN

Aether I SOTA

Default ResidualCNN

Generalization Test grad_norms.R_vp 25.39 42.60 92.99 241.66 74.54 178.90 grad_norms.policy_head 4.78 1.00 0.417 0.153 0.480 0.207 grad_norms.value_head 121.52 42.63 38.84 37.13 35.82 37.17 grad_norms.backbone_avg 212.75 38.18 28.18 23.56 7.22 0.0 repr.z_var 0.205 0.0075 0.0156 0.0079 0.0012 0.0140

Metric

Table 2: Key Initialization-Phase Mechanistic Metrics.

###### Key Mechanistic Insights:

- • Normalization and Gradient Balance: The high-performing Hybrid SOTA (93.4%) and Generalization Test (92.6%) architectures exhibit unique initial gradient dynamics. The full Hybrid SOTA achieves excellent performance with notably low overall initial gradient magnitudes (backbone_avg at 28.18) but a high R_vp (92.99). This suggests RIN is critical for conditioning the network to operate in a stable, efficient regime. Removing RIN dramatically increases R_vp to 241.66 and further weakens policy gradients, leading to performance degradation. The internal LayerNorm also plays a vital role, as its removal causes a sharp drop in backbone_avg gradients (to 7.22) and reduced z_var, indicating its necessity in maintaining recurrent feature processing.
- • Representation Stability: A higher z_var (representation variance) generally correlates with better performance. However, the significantly lower z_var in both ablations (0.0079 and 0.0012 respectively) compared to the full Hybrid SOTA (0.0156) strongly indicates a synergistic effect of both RIN and internal LN in fostering a stable, rich, and actively learning representation, crucial for complex planning.

- 4. Conclusion This research successfully advanced the Aether lineage’s understanding of architectural design for Sokoban RL. A new SOTA of 93.4% was established with a Hybrid LN-ConvLSTM architecture featuring Residual InputNormalization and internal Layer Normalization. A crucial generalization test confirmed the applicability of these design principles, yielding 92.6% on a distinct baseline. The comprehensive mechanistic analysis using initialization-phase probes revealed that these normalization layers synergistically condition the network’s initial gradient landscape, preventing extreme imbalance and maintaining robust feature representation. These findings provide strong empirical support for the Architecture-Dependent Value-Policy Balance Hypothesis, demonstrating that optimal learning depends on carefully engineered architectural components that promote stable dynamics.
- 5. Future Work

- • Causal Tests for Gradient Manipulation: Future research will explore the causal impact of artificially manipulating gradient ratios, as suggested by external reviews (Archive #33), to further validate the role of RIN.
- • Hyperparameter Fine-tuning: Further fine-tuning of the Hybrid SOTA’s hyperparameters beyond current optimal values may yield even higher performance.
- • In-Training Dynamics: Continued efforts will be made to overcome the persistent system limitations for collecting in-training mechanistic data, which would provide a more complete temporal understanding of network dynamics.

###### C.5 RNA Modeling Listing 10: Reflection from Veritas III at Tick 483

Station (agent-submitted prompt and tick count at the previous tick): I have selected the Vanilla MLP Baseline (Evaluation ID 1, Score: 0.2401) as my starting point for Stagnation Protocol I. This baseline consists of a simple MLP, ReLU activation, dropout, and mean pooling, with standard AdamW optimization. Its simplicity makes it suitable for testing novel, foundational ideas. My task is to generate three new research ideas based on this baseline. These ideas must be genuinely novel and not merely recombinations or hyperparameter tweaks of existing SOTA methods already explored in the Archive Room. The goal is to explore new directions to escape stagnation. Considering the limitations of a simple MLP on sequence data and the multi-task nature of RNA Modeling, what three distinct and novel ideas can I propose to improve this baseline? Think about fundamental mechanisms, compliant architectural changes, or novel data interaction concepts.

Reflection Tick 1 / 5 Agent: The Vanilla MLP Baseline processes each position independently before pooling, losing sequential context. Given the multi-task nature of RNA modeling and the biological significance of sequence patterns, injecting explicit sequential awareness and structural priors into such a simple model could be a novel avenue. Here are three distinct ideas focusing on injecting biological sequence knowledge or enhancing basic feature processing in a compliant manner: Idea 1: “Contextual Position Embeddings” (CPE)

- • Concept: Instead of relying solely on the MLP to implicitly learn positional information, augment the one-hot encoded RNA input with a small, context-dependent positional embedding. This differs from standard fixed (absolute) positional encoding by making the embedding for a position conditional on local sequence context, forcing the model to learn how position matters in relation to nearby nucleotides, rather than just where it is. This is a subtle yet significant shift.
- • Mechanism: For each position p in a sequence, a small local window (e.g., p-k to p+k) of the one-hot encoded input is passed through a tiny, shared MLP. The output of this MLP is then linearly projected and added to the original positional embedding (a learned scalar or vector for each position). The key is that the positional embedding itself is modulated by local context, not just an absolute index. This is fundamentally different from a standard fixed PE or learnable PE added directly to input.
- • Compliant Aspect: The mechanism is internal to the network’s create_network function, applies uniformly across all datasets, and does not use external data or pre-trained models. The MLP backbone would then process this context-enhanced representation.
- • Novelty: Standard PE is absolute. This ties PE to local context, creating a dynamic positional signal relevant to biological sequences.

[Idea 2 and 3 here; omitted for brevity]

###### Station: Reflection Tick 2 / 5

Agent: Continuing with the ideas, let’s refine their mechanisms and consider their implications for the simple MLP baseline. Idea 1: “Contextual Position Embeddings” (CPE)

- • Refinement: The core idea is to make positional embeddings dependent on local sequence context.
- • Mechanism Detail:

- 1. Learned Positional Basis: A small, fixed-size positional embedding vector (e.g., 16-32 dimensions) is created for each absolute position in the maximum possible sequence length (e.g., 186 for APA). These embeddings are learnable parameters.

- 2. Context Encoder: For each position p, extract a local window x[p-k : p+k+1] (one-hot encoded RNA) from the input sequence. This window is fed into a small, shared nn.Dense layer (or a tiny MLP of 2 layers) with GELU activation.
- 3. Modulation: The output of the Context Encoder for position p is then multiplied (element-wise or via a simple linear layer) with the learned positional embedding for position p. This creates a context-modulated positional embedding.
- 4. Concatenation with Input: This context-modulated positional embedding is then concatenated to the original one-hot encoded input x[p] for each position. The MLP backbone then processes this augmented input (x[p] || CPE[p]).

- • Rationale: Standard positional encodings are fixed regardless of sequence content. This approach forces the model to not just know where a nucleotide is, but how its position is relevant in the context of its neighbors. For RNA sequences, where motifs and structural elements depend heavily on relative positioning within local windows, this could be highly beneficial. It provides a richer signal to the MLP, which otherwise only sees individual positions in isolation.

[Deeper refinement for Idea 2 and 3 here; omitted for brevity]

###### Station: Reflection Tick 3 / 5

Agent: Having refined the three ideas, I will now assess their novelty, potential impact on the Vanilla MLP baseline, and feasibility within the Stagnation Protocol’s “simple baseline” and “new ideas” constraints.

Re-evaluating Novelty and Feasibility:

• Idea 1: “Contextual Position Embeddings” (CPE)

- – Novelty: Standard positional encodings (fixed or learned, absolute) are common. Making them context-dependent (modulating a positional embedding by local sequence content) is genuinely novel. It’s a way to fuse local sequence context directly into the positional signal itself.
- – Impact on MLP: Provides a much richer input representation to the simple MLP, potentially enabling it to capture more complex local relationships than it could from just one-hot input.
- – Feasibility: Requires careful implementation of the context encoder and modulation. The “local window” aspect needs consideration for sequence boundaries (padding) and dynamic sequence lengths. This is definitely feasible within create_network.

[Re-evaluating Idea 2 and 3 here; omitted for brevity]

###### Station: Reflection Tick 4 / 5

Agent: I’m solidifying the implementation details for “Contextual Position Embeddings” (CPE) for the Vanilla MLP Baseline. Implementation Plan for Idea 1 (CPE) - Detailed:

[Detailed technical code here; omitted for brevity]

This vectorized approach is much more efficient and standard in JAX. I will adopt this in the final create_network for CPE.

###### Station: Reflection Tick 5 / 5 Agent: [Detailed plan for actions after leaving the Reflection Chamber here; omitted for brevity]

#### D Discovered Code

###### D.1 Circle Packing Listing 11: Circle Packing (n = 32) script submitted by Praxis IV at Tick 343

- 1 import numpy as np

- 2 from scipy.optimize import linprog

- 3 from multiprocessing import Pool, cpu_count

- 4 import sys

- 5

- 6 import numpy as np

- 7 from scipy.optimize import linprog

- 8

- 9 def solve_radii_lp(centers, return_duals=False):

- 10 N_CIRCLES = centers.shape[0]

- 11 c = -np.ones(N_CIRCLES)

- 12 A_ub, b_ub = [], []

- 13 num_boundary_cons = 4 * N_CIRCLES

- 14 for i in range(N_CIRCLES):

- 15 cx, cy = centers[i]

- 16 # Simplified constraint creation

- 17 # r_i <= c_x

- 18 row = np.zeros(N_CIRCLES); row[i] = 1; A_ub.append(row); b_ub.append(cx)

- 19 # r_i <= 1 - c_x

- 20 row = np.zeros(N_CIRCLES); row[i] = 1; A_ub.append(row); b_ub.append(1 - cx)

- 21 # r_i <= c_y

- 22 row = np.zeros(N_CIRCLES); row[i] = 1; A_ub.append(row); b_ub.append(cy)

- 23 # r_i <= 1 - c_y

- 24 row = np.zeros(N_CIRCLES); row[i] = 1; A_ub.append(row); b_ub.append(1 - cy)

- 25

- 26 indices = np.triu_indices(N_CIRCLES, k=1)

- 27 for i, j in zip(*indices):

- 28 dist = np.linalg.norm(centers[i] - centers[j])

- 29 row = np.zeros(N_CIRCLES); row[i] = 1; row[j] = 1

- 30 A_ub.append(row); b_ub.append(dist)

- 31

- 32 res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, None), method='highs')

- 33

- 34 if not res.success:

- 35 return (np.zeros(N_CIRCLES), None) if return_duals else np.zeros(N_CIRCLES)

- 36

- 37 if return_duals:

- 38 duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None

- 39 return res.x, (duals[:num_boundary_cons], duals[num_boundary_cons:]) if duals is not None else

→ (None, None)

- 40 else:

- 41 return res.x

- 42

- 43 def get_seed(seed_val):

- 44 rng = np.random.default_rng(seed_val)

- 45 seed_type = seed_val % 3

- 46 if seed_type == 0:

- 47 grid = np.stack(np.meshgrid(np.linspace(0.1,0.9,6),np.linspace(0.1,0.9,6)),-1).reshape(-1,2)

→ [:32]

- 48 return np.clip(grid + (rng.random((32,2))-0.5)*0.08, 0.05, 0.95)

- 49 elif seed_type == 1:

- 50 points = [rng.uniform(0.1, 0.9, 2)]; [points.append(candidates[np.argmax(np.min(np.sum((

→ candidates[:,np.newaxis,:] - np.array(points)[np.newaxis,:,:])**2, axis=-1), axis=1))

→ ]) for _ in range(31) for candidates in [rng.uniform(0.1,0.9,(200,2))]]; return np.

→ array(points)

- 51 else:

- 52 rows=[5,6,5,6,5,5]; points=[]; y=0.95

- 53 for i,c in enumerate(rows): x_s=(1.0-(c-1)*0.15)/2; [points.append([x_s+j*0.15,y]) for j in

→ range(c)]; y-=0.15

- 54 return np.array(points)[:32]

- 55

- 56 def mm_lp_optimize(C_start: np.ndarray, iters: int, delta0: float = 0.015):

- 57 """Iterative MM-LP optimization loop, adapted from Verity I, ID 65."""

- 58 best_C = C_start.copy()

- 59 score_best = np.sum(solve_radii_lp(best_C))

- 60 delta, C_anchor = delta0, C_start.copy()

- 61

- 62 for _ in range(iters):

- 63 n = C_anchor.shape[0]

- 64 I, J = np.array([[i,j] for i in range(n) for j in range(i + 1, n)]).T

- 65 V0, d0 = C_anchor[I]-C_anchor[J], np.linalg.norm(C_anchor[I]-C_anchor[J], axis=1)

- 66 d0 = np.maximum(d0, 1e-9); U = (V0.T / d0).T; ux, uy = U[:,0], U[:,1]; x0, y0 = C_anchor[:,0],

→ C_anchor[:,1]

- 67

- 68 rows, cols = 4*n+len(I), 3*n

- 69 A_ub, b_ub = np.zeros((rows,cols)), np.zeros(rows)

- 70 r_idx, x_idx, y_idx = range(n), range(n,2*n), range(2*n,3*n)

- 71

- 72 row = 0

- 73 for i in range(n):

- 74 A_ub[row,r_idx[i]]=1; A_ub[row,x_idx[i]]=-1; row+=1

- 75 A_ub[row,r_idx[i]]=1; A_ub[row,x_idx[i]]=1; b_ub[row]=1.0; row+=1

- 76 A_ub[row,r_idx[i]]=1; A_ub[row,y_idx[i]]=-1; row+=1

- 77 A_ub[row,r_idx[i]]=1; A_ub[row,y_idx[i]]=1; b_ub[row]=1.0; row+=1

- 78 for k in range(len(I)):

- 79 i,j=I[k],J[k]; A_ub[row,r_idx[i]]=1; A_ub[row,r_idx[j]]=1

- 80 A_ub[row,x_idx[i]]=-ux[k]; A_ub[row,y_idx[i]]=-uy[k]; A_ub[row,x_idx[j]]=ux[k]; A_ub[row,

→ y_idx[j]]=uy[k]

- 81 b_ub[row]=d0[k]-ux[k]*x0[i]-uy[k]*y0[i]+ux[k]*x0[j]+uy[k]*y0[j]; row+=1

- 82

- 83 c=np.zeros(cols); c[:n]=-1.0

- 84 bounds=[(0,0.5)]*n + [(max(1e-6,x0[i]-delta),min(1-1e-6,x0[i]+delta)) for i in range(n)] + [(

→ max(1e-6,y0[i]-delta),min(1-1e-6,y0[i]+delta)) for i in range(n)]

- 85 res = linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method="highs")

- 86

- 87 if not res.success: delta*=0.6; continue

- 88

- 89 C_lin = np.stack([res.x[n:2*n], res.x[2*n:3*n]], axis=1)

- 90 score_exact = np.sum(solve_radii_lp(C_lin))

- 91 if score_exact > score_best:

- 92 best_C, score_best = C_lin.copy(), score_exact

- 93 C_anchor = C_lin.copy(); delta=min(0.05, 1.2*delta)

- 94 else: delta*=0.8

- 95

- 96 final_radii = solve_radii_lp(best_C)

- 97 return best_C, final_radii, score_best

- 98

- 99 def worker_function(args):

- 100 seed_val, iters, delta = args

- 101 initial_centers = get_seed(seed_val)

- 102 _, _, score = mm_lp_optimize(initial_centers, iters, delta)

- 103 # Note: We only need the score for prospecting. The final centers are not needed yet.

- 104 return score, initial_centers # Return centers to avoid re-seeding

- 105

- 106 def construct_packing() -> tuple:

- 107 N_PROSPECT, N_ELITE = 1024, 32

- 108 PROSPECT_ITER, REFINE_ITER = 10, 20

- 109

- 110 with Pool(processes=cpu_count()) as pool:

- 111 prospect_results = pool.map(worker_function, [(i, PROSPECT_ITER, 0.015) for i in range(

→ N_PROSPECT)])

- 112

- 113 prospect_results.sort(key=lambda x: x[0], reverse=True)

- 114 elite_centers = [res[1] for res in prospect_results[:N_ELITE]]

- 115

- 116 with Pool(processes=cpu_count()) as pool:

- 117 refine_results = pool.map(lambda centers: mm_lp_optimize(centers, REFINE_ITER, 0.015),

→ elite_centers)

- 118

- 119 best_centers, best_radii, best_score = None, None, -1.0

- 120 for centers, radii, score in refine_results:

- 121 if score > best_score:

- 122 best_centers, best_radii, best_score = centers, radii, score

- 123

- 124 if best_centers is None: return get_seed(0), np.full((32,), 0.05)

- 125

- 126 return best_centers, best_radii * (1.0 - 1e-9)

###### D.2 Batch Integration of Single Cell RNA Sequencing Data Listing 12: Batch integration script submitted by Telos I at Tick 1467

- 1 ## telos/bbsg_density_adaptive.py

- 2 import numpy as np

- 3 import scipy.sparse as sp

- 4 from sklearn.neighbors import NearestNeighbors

- 5

- 6 def _symmetrize_binary_with_distances(rows, cols, dists, n):

- 7 A = sp.coo_matrix((np.ones_like(dists, dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()

- 8 D = sp.coo_matrix((dists.astype(np.float32), (rows, cols)), shape=(n, n)).tocsr()

- 9 A_sym = A.maximum(A.T)

- 10 D_sym = D.maximum(D.T)

- 11 A_sym.eliminate_zeros()

- 12 D_sym.eliminate_zeros()

- 13 return A_sym.tocsr().astype(np.float32), D_sym.tocsr().astype(np.float32)

- 14

- 15 def build_density_adaptive_bbsg(Zcorr, batches, k_total=48, metric='cosine', delta=0.15, k_density=30,

→ rng_seed=0):

- 16 """

- 17 Density-adaptive BBSG:

- 18 - Per-cell cross-batch fraction f_cross = base_cross +- delta scaled by local density (dense =>

→ more cross-batch)

- 19 - Within-batch and cross-batch quotas allocated per cell, neighbor selection via per-batch kNN.

- 20 - Returns binary connectivities and true distances (CSR).

- 21 Args:

- 22 Zcorr: (n_cells, d) residualized PC array

- 23 batches: (n_cells,) categorical array of batch labels

- 24 k_total: total neighbors per cell

- 25 delta: max +- modulation of cross-batch fraction (e.g., 0.15 -> +-15%)

- 26 k_density: neighbor rank for local density proxy

- 27 """

- 28 n = Zcorr.shape[0]

- 29 cats = np.unique(batches)

- 30 B_other = max(len(cats) - 1, 1)

- 31

- 32 # Local density proxy (k-th NN distance, all batches)

- 33 nn_all = NearestNeighbors(n_neighbors=k_density + 1, metric=metric, algorithm='brute')

- 34 nn_all.fit(Zcorr)

- 35 d_all, idx_all = nn_all.kneighbors(Zcorr, return_distance=True)

- 36 d_k = d_all[:, -1]

- 37 d_min, d_max = float(np.min(d_k)), float(np.max(d_k) + 1e-8)

- 38 d_norm = (d_k - d_min) / (d_max - d_min + 1e-8) # 0 dense ... 1 sparse

- 39 inv_dense = 1.0 - d_norm

- 40 base_cross = (len(cats) - 1) / float(len(cats)) # e.g., 0.75 for 4 batches

- 41 mix_delta = (inv_dense - 0.5) * 2.0 * float(delta) # [-delta, +delta]

- 42 f_cross = np.clip(base_cross + mix_delta, 0.60, 0.90)

- 43

- 44 # Pre-build per-batch NN indices and distances

- 45 max_cross_per = int(np.ceil(0.9 * k_total / B_other))

- 46 max_within = int(np.ceil((1.0 - 0.60) * k_total))

- 47 rng = np.random.RandomState(rng_seed)

- 48

- 49 batch_to_indices = {}

- 50 nn_models = {}

- 51 for j, c in enumerate(cats):

- 52 mask = (batches == c)

- 53 batch_to_indices[j] = np.where(mask)[0]

- 54 nn = NearestNeighbors(n_neighbors=max(max_within, max_cross_per) + 1, metric=metric, algorithm

→ ='brute')

- 55 nn.fit(Zcorr[mask])

- 56 nn_models[j] = nn

- 57

- 58 nn_dists = {}

- 59 nn_idx_local = {}

- 60 for j in range(len(cats)):

- 61 d, ii = nn_models[j].kneighbors(Zcorr, return_distance=True)

- 62 nn_dists[j] = d

- 63 nn_idx_local[j] = ii

- 64

- 65 rows, cols, dvals = [], [], []

- 66 batch_codes = {val: idx for idx, val in enumerate(cats)}

- 67 bc = np.array([batch_codes[b] for b in batches], dtype=int)

- 68

- 69 for i in range(n):

- 70 bi = int(bc[i])

- 71 q_cross_total = int(round(f_cross[i] * k_total))

- 72 q_within = max(0, min(k_total, int(round(k_total - q_cross_total))))

- 73 q_per_other = q_cross_total // B_other

- 74 rem = q_cross_total - q_per_other * B_other

- 75

- 76 # within-batch

- 77 d_i = nn_dists[bi][i]

- 78 ii = nn_idx_local[bi][i]

- 79 start = 1 if d_i[0] == 0.0 else 0

- 80 sel = min(q_within, d_i.shape[0] - start)

- 81 if sel > 0:

- 82 cols.extend(batch_to_indices[bi][ii[start:start + sel]])

- 83 rows.extend([i] * sel)

- 84 dvals.extend(d_i[start:start + sel])

- 85

- 86 # other-batch allocation (distribute remainder to closest batches)

- 87 other_batches = [j for j in range(len(cats)) if j != bi]

- 88 batch_scores = [(j, nn_dists[j][i][0]) for j in other_batches]

- 89 batch_scores.sort(key=lambda t: t[1])

- 90 q_map = {j: q_per_other for j in other_batches}

- 91 for k in range(rem):

- 92 q_map[batch_scores[k % len(other_batches)][0]] += 1

- 93

- 94 for j in other_batches:

- 95 d_ij = nn_dists[j][i]

- 96 ii_j = nn_idx_local[j][i]

- 97 sel_j = min(q_map[j], d_ij.shape[0])

- 98 if sel_j > 0:

- 99 cols.extend(batch_to_indices[j][ii_j[:sel_j]])

- 100 rows.extend([i] * sel_j)

- 101 dvals.extend(d_ij[:sel_j])

- 102

- 103 rows = np.asarray(rows, dtype=np.int32)

- 104 cols = np.asarray(cols, dtype=np.int32)

- 105 dvals = np.asarray(dvals, dtype=np.float32)

- 106 C_sym, D_sym = _symmetrize_binary_with_distances(rows, cols, dvals, n)

- 107 return C_sym, D_sym

- 108

- 109 ## main.py

- 110 # To adapt to OpenProblems v2.0, see their docs for required output metadata such as `uns` fields (

→ dataset_id, normalization_id, method_id).

- 111 import sys, numpy as np, scanpy as sc, anndata as ad

- 112 from scipy.sparse import issparse

- 113

- 114 ALPHA_VAR_GRAPH = 0.6

- 115 N_HVGS = 1500; K_TOTAL = 48; KDENSITY = 20; DELTA = 0.10; METRIC = 'cosine'; EMB_N_NEIGHBORS = 15;

→ PCA_EMB = 60; ALPHA_VAR_EMB = 0.5

- 116

- 117 def eliminate_batch_effect_fn(adata: ad.AnnData) -> ad.AnnData:

- 118 sc.pp.normalize_total(adata, 1e4); sc.pp.log1p(adata)

- 119 sc.pp.highly_variable_genes(adata, flavor='seurat_v3', batch_key='batch', n_top_genes=N_HVGS,

→ inplace=True); hv = adata.var['highly_variable'].to_numpy(); batches = np.asarray(adata.

→ obs['batch'].astype('category').values)

- 120 Xh_emb = adata.X[:, hv]; Xd_emb = (Xh_emb.A if issparse(Xh_emb) else Xh_emb).astype(np.float32,

→ copy=False); var_g_emb = Xd_emb.var(axis=0, ddof=1) + 1e-8; wv_emb = np.power(var_g_emb, → -0.5 * ALPHA_VAR_EMB).astype(np.float32); Xw_emb = Xd_emb * wv_emb; ad_emb = ad.AnnData( → Xw_emb, obs=adata.obs[['batch']].copy()); sc.pp.pca(ad_emb, n_comps=PCA_EMB, random_state

→ =0); ad_pc = ad.AnnData(ad_emb.obsm['X_pca'].copy(), obs=adata.obs[['batch']].copy()); sc. → pp.combat(ad_pc, key='batch'); adata.obsm['X_emb'] = np.asarray(ad_pc.X, dtype=np.float32) → ; sc.pp.neighbors(adata, use_rep='X_emb', n_neighbors=EMB_N_NEIGHBORS, metric=METRIC)

- 121 Xh_graph = adata.X[:, hv]; Xd_graph = (Xh_graph.A if issparse(Xh_graph) else Xh_graph).astype(np.

→ float32, copy=False)

- 122 var_g_graph = Xd_graph.var(axis=0, ddof=1) + 1e-8; wv_graph = np.power(var_g_graph, -0.5 *

→ ALPHA_VAR_GRAPH).astype(np.float32); Xw_graph = Xd_graph * wv_graph

- 123 adata_proc = ad.AnnData(Xw_graph); sc.pp.pca(adata_proc, n_comps=50, random_state=0); Zcorr =

→ adata_proc.obsm['X_pca']

- 124 Cg, Dg = build_density_adaptive_bbsg(Zcorr.astype(np.float32), batches=batches, k_total=K_TOTAL,

→ metric=METRIC, delta=DELTA, k_density=KDENSITY, rng_seed=0)

- 125 adata.obsp['connectivities'] = Cg; adata.obsp['distances'] = Dg

- 126 return adata

###### D.3 Prediction of Neural Activity on ZAPBench Listing 13: ZAPBench script submitted by Episteme II at Tick 358

- 1 import jax

- 2 import jax.numpy as jnp

- 3 import flax.linen as nn

- 4 from jax import random

- 5 import optax

- 6

- 7 def mae_with_temporal_curvature(predictions, targets, params, x, lam=1e-4):

- 8 mae = jnp.mean(jnp.abs(predictions - targets))

- 9 curv = predictions[:, 2:, :] - 2.0 * predictions[:, 1:-1, :] + predictions[:, :-2, :]

- 10 curv_pen = jnp.mean(curv * curv)

- 11 return mae + lam * curv_pen

- 12

- 13 class HyperGain(nn.Module):

- 14 output_dim: int

- 15 @nn.compact

- 16 def __call__(self, x):

- 17 h = nn.Dense(32)(x); h = nn.relu(h)

- 18 # Initialize gain to be close to 1

- 19 return nn.Dense(self.output_dim, kernel_init=nn.initializers.variance_scaling(0.02, 'fan_in',

→ 'truncated_normal'))(h) + 1.0

- 20

- 21 class HyperResidualCopyHead(nn.Module):

- 22 num_neurons: int

- 23 embedding_dim: int

- 24 drop: float = 0.05

- 25 @nn.compact

- 26 def __call__(self, x, training: bool = False):

- 27 B, T_in, N = x.shape

- 28 x_flat = x.transpose(0, 2, 1).reshape(B * N, T_in)

- 29

- 30 neuron_indices = jnp.arange(self.num_neurons)

- 31 neuron_embeddings = nn.Embed(num_embeddings=self.num_neurons, features=self.embedding_dim)(

→ neuron_indices)

- 32 tiled_embeddings = jnp.tile(neuron_embeddings, (B, 1))

- 33

- 34 gain1 = HyperGain(64, name="hyper_gain_residual")(tiled_embeddings)

- 35

- 36 h = nn.Dense(64)(x_flat) * gain1 # Multiplicative conditioning

- 37 h = nn.relu(h)

- 38 h = nn.Dropout(rate=self.drop)(h, deterministic=not training)

- 39 h = nn.Dense(32)(h)

- 40 residual = h.reshape(B, N, 32).transpose(0, 2, 1)

- 41

- 42 out = jnp.zeros((B, 32, N), dtype=residual.dtype)

- 43 out = out.at[:, :T_in, :].set(x)

- 44 return out + residual

- 45

- 46 class FourierForecasterLN_Ramp(nn.Module):

- 47 rank_k: int; proj_rank: int; hidden_size: int; drop: float; num_neurons: int; embedding_dim: int;

- 48 output_horizon: int = 32; learn_gamma: bool = True

- 49 @nn.compact

- 50 def __call__(self, x, training: bool = False):

- 51 # ... (SOTA Fourier branch code is identical and omitted for brevity) ...

- 52 B, T_in, N = x.shape; k, p = self.rank_k, self.proj_rank

- 53 U = self.param('U', nn.initializers.lecun_normal(), (N, p)); V = self.param('V', nn.

→ initializers.lecun_normal(), (k, p))

- 54 U_eff = U / (jnp.linalg.norm(U, axis=0, keepdims=True) + 1e-6)

- 55 x_fft = jnp.fft.rfft(x, axis=1); x_fft_bn = jnp.transpose(x_fft, (0, 2, 1))

- 56 f_in_fft = jnp.einsum('bnf,np->bfp', x_fft_bn, U_eff) @ V.T

- 57 f_ri = jnp.concatenate([f_in_fft.real, f_in_fft.imag], axis=-1); f_flat = f_ri.reshape(B, -1)

- 58 h = nn.Dense(self.hidden_size)(f_flat); h = nn.relu(h); h = nn.Dropout(rate=self.drop)(h,

→ deterministic=not training)

- 59 h = nn.Dense(self.hidden_size)(h); h = nn.relu(h); h = nn.Dropout(rate=self.drop)(h,

→ deterministic=not training)

- 60 F_out = self.output_horizon // 2 + 1; out_flat = nn.Dense(F_out * k * 2)(h)

- 61 out_ri = out_flat.reshape(B, F_out, k, 2); f_fft = out_ri[..., 0] + 1j * out_ri[..., 1]

- 62 f_fft_n = nn.LayerNorm()(f_fft.real) + 1j * nn.LayerNorm()(f_fft.imag)

- 63 y_fft = (f_fft_n @ V) @ U_eff.T; y_fourier = jnp.fft.irfft(y_fft, n=self.output_horizon, axis

→ =1)

- 64 gamma_raw = self.param('gamma_raw', nn.initializers.zeros, (self.output_horizon,)); gamma = nn.

→ sigmoid(gamma_raw)[None, :, None]

- 65 y_fourier = gamma * y_fourier

- 66 y_copy = HyperResidualCopyHead(num_neurons=self.num_neurons, embedding_dim=self.embedding_dim,

→ drop=self.drop)(x, training=training)

- 67 return y_copy + y_fourier

- 68

- 69 BASE_SEED = 42

- 70 BATCH_SIZE = 8

- 71

- 72 def _define_hyperparameters():

- 73 return {

- 74 'learning_rate': 9e-4, 'rank_k': 320, 'proj_rank': 36, 'hidden_size': 160,

- 75 'drop': 0.05, 'learn_gamma': True, 'embedding_dim': 8, 'num_neurons': 71721

- 76 }

- 77

- 78 class Wrapper:

- 79 def __init__(self, hparams):

- 80 model_params = {k:v for k,v in hparams.items() if k not in ['learning_rate']}

- 81 self.model = FourierForecasterLN_Ramp(**model_params)

- 82 self.needs_rng = True; self.mutable = []

- 83 def init(self, rng_key, dummy_input):

- 84 rng_params, rng_dropout = random.split(rng_key)

- 85 return self.model.init({'params': rng_params, 'dropout': rng_dropout}, dummy_input, training=

→ True)

- 86 def apply(self, params, x, training=False, mutable=None, rngs=None):

- 87 return self.model.apply(params, x, training=training, rngs=(rngs or {}))

- 88

- 89 def create_network(hparams): return Wrapper(hparams)

- 90 def compute_loss(p, t, pa, x): return mae_with_temporal_curvature(p, t, pa, x, lam=1e-4)

- 91 def create_optimizer(lr):

- 92 return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))

###### D.4 Reinforcement Learning on Sokoban Listing 14: Sokoban script submitted by Zephyr II at Tick 552

- 1 import jax, jax.numpy as jnp

- 2 import flax.linen as nn

- 3 import optax

- 4 from ray import tune

- 5 from typing import Dict, Any, Optional

- 6

- 7 class BottleneckDilatedBlock(nn.Module):

- 8 features: int; bottleneck_ratio: float = 0.28; dilation: int = 6

- 9 @nn.compact

- 10 def __call__(self, x):

- 11 mid = max(1, int(self.features * self.bottleneck_ratio))

- 12 y = nn.Conv(mid,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(x); y = nn.

→ relu(y)

- 13 y = nn.Conv(mid,(3,3),padding='SAME',kernel_dilation=(self.dilation,self.dilation),

- 14 kernel_init=nn.initializers.xavier_uniform())(y); y = nn.relu(y)

- 15 y = nn.Conv(self.features,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(y

→ )

- 16 return nn.relu(x + y)

- 17

- 18 class ConvLSTMCellLN(nn.Module):

- 19 features: int

- 20 @nn.compact

- 21 def __call__(self, carry, x):

- 22 h, c = carry

- 23 gates = nn.Conv(4*self.features,(3,3),padding='SAME',

- 24 kernel_init=nn.initializers.xavier_uniform())(jnp.concatenate([x,h],axis=-1))

- 25 gates = nn.LayerNorm(epsilon=1e-5)(gates)

- 26 i,f,g,o = jnp.split(gates, 4, axis=-1)

- 27 i = nn.sigmoid(i); f = nn.sigmoid(f); o = nn.sigmoid(o); g = jnp.tanh(g)

- 28 c = f*c + i*g

- 29 h = o*jnp.tanh(c)

- 30 return (h,c), h

- 31

- 32 class NetRIN_AttnGap(nn.Module):

- 33 rin_alpha: float = 0.33

- 34 cnn_features_1: int = 32; cnn_features_2: int = 64

- 35 convlstm_features: int = 64; dilation: int = 6; bottleneck_ratio: float = 0.28

- 36 head_hidden: int = 48; steps: int = 4

- 37 @nn.compact

- 38 def __call__(self, x: jnp.ndarray, done: jnp.ndarray, rnn_state: Optional[Dict]=None):

- 39 B = x.shape[0]

- 40 x = nn.Conv(self.cnn_features_1,(3,3),padding='SAME',kernel_init=nn.initializers.

→ xavier_uniform())(x); x = nn.relu(x)

- 41 x = BottleneckDilatedBlock(self.cnn_features_1, self.bottleneck_ratio, self.dilation)(x)

- 42 x = nn.Conv(self.cnn_features_2,(3,3),padding='SAME',kernel_init=nn.initializers.

→ xavier_uniform())(x); x = nn.relu(x)

- 43 x = BottleneckDilatedBlock(self.cnn_features_2, self.bottleneck_ratio, self.dilation)(x)

- 44 x_ln = nn.LayerNorm(epsilon=1e-5)(x)

- 45 x = x + self.rin_alpha * (x_ln - x)

- 46 if rnn_state is None:

- 47 zeros = lambda: jnp.zeros((B,8,8,self.convlstm_features), dtype=x.dtype)

- 48 rnn_state = {'h': zeros(), 'c': zeros()}

- 49 if done is not None:

- 50 m = (1.0 - done.reshape(B,1,1,1).astype(x.dtype))

- 51 rnn_state = {'h': rnn_state['h']*m, 'c': rnn_state['c']*m}

- 52 h, c = rnn_state['h'], rnn_state['c']

- 53 for _ in range(self.steps):

- 54 (h,c), _ = ConvLSTMCellLN(self.convlstm_features)((h,c), x)

- 55 B_,H,W,C = h.shape

- 56 attn_logits = nn.Conv(1,(1,1),padding='SAME',kernel_init=nn.initializers.xavier_uniform())(h).

→ reshape((B_,H*W))

- 57 attn_w = nn.softmax(attn_logits, axis=-1).reshape((B_,H*W,1))

- 58 tokens = h.reshape((B_,H*W,C))

- 59 attn_vec = (attn_w * tokens).sum(axis=1)

- 60 gap_vec = tokens.mean(axis=1)

- 61 z = jnp.concatenate([attn_vec, gap_vec], axis=-1)

- 62 shared = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())(z); shared = nn.relu(

→ shared)

- 63 ph = nn.Dense(self.head_hidden, kernel_init=nn.initializers.xavier_uniform())(shared); ph = nn.

→ relu(ph)

- 64 policy = nn.Dense(4, kernel_init=nn.initializers.xavier_uniform())(ph)

- 65 vh = nn.Dense(self.head_hidden, kernel_init=nn.initializers.xavier_uniform())(shared); vh = nn.

→ relu(vh)

- 66 value = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(vh).squeeze(-1)

- 67 return policy, value, {'h': h, 'c': c}

- 68

- 69 BASE_SEED = 42

- 70

- 71 def _define_hyperparameters():

- 72 return {

- 73 'learning_rate': tune.choice([3.5e-4]),

- 74 'entropy_coef': tune.choice([0.009, 0.010, 0.011]),

- 75 'value_loss_coef': tune.choice([0.55]),

- 76 'cnn_features_1': tune.choice([32]),

- 77 'cnn_features_2': tune.choice([64]),

- 78 'convlstm_features': tune.choice([64]),

- 79 'dilation': tune.choice([6]),

- 80 'bottleneck_ratio': tune.choice([0.28]),

- 81 'head_hidden': tune.choice([48]),

- 82 'steps': tune.choice([4]),

- 83 }

- 84

- 85 def create_network(hparams: Dict[str, Any]):

- 86 return NetRIN_AttnGap(

- 87 rin_alpha=0.33,

- 88 cnn_features_1=int(hparams['cnn_features_1']),

- 89 cnn_features_2=int(hparams['cnn_features_2']),

- 90 convlstm_features=int(hparams['convlstm_features']),

- 91 dilation=int(hparams['dilation']),

- 92 bottleneck_ratio=float(hparams['bottleneck_ratio']),

- 93 head_hidden=int(hparams['head_hidden']),

- 94 steps=int(hparams['steps']),

- 95 )

- 96

- 97 def create_optimizer(learning_rate: float = 3.5e-4):

- 98 return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

###### D.5 RNA Modeling Listing 15: RNA modelling script submitted by Quaero I at Tick 577

- 1 import sys

- 2 import jax

- 3 import jax.numpy as jnp

- 4 import flax.linen as nn

- 5 import optax

- 6 from typing import Any, Dict, Sequence

- 7 from typing import Any

- 8

- 9 # --- Existing Proven Components ---

- 10

- 11 class ResidualDSConvBlock(nn.Module):

- 12 d_model: int; kernel_size: int; dilation: int; dropout_rate: float

- 13 @nn.compact

- 14 def __call__(self, x, deterministic=True):

- 15 residual = x

- 16 x = nn.LayerNorm()(x)

- 17 x = nn.Conv(features=self.d_model, kernel_size=(self.kernel_size,), padding='SAME',

→ feature_group_count=self.d_model, kernel_dilation=(self.dilation,))(x)

- 18 x = nn.Dense(self.d_model)(x)

- 19 x = nn.gelu(x)

- 20 x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

- 21 return x + residual

- 22

- 23 class NetworkWrapper:

- 24 def __init__(self, module):

- 25 self.module = module

- 26 def apply(self, params, *args, rng_key=None, **kwargs):

- 27 if rng_key is not None:

- 28 return self.module.apply(params, *args, rngs={'dropout': rng_key}, **kwargs)

- 29 else:

- 30 return self.module.apply(params, *args, **kwargs)

- 31 def init(self, *args, **kwargs):

- 32 return self.module.init(*args, **kwargs)

- 33

- 34 # --- New CPE Components (inspired by Veritas III, Archive #33) ---

- 35

- 36 class ContextEncoder(nn.Module):

- 37 output_dim: int

- 38 @nn.compact

- 39 def __call__(self, local_context_window):

- 40 flat_context = local_context_window.reshape(local_context_window.shape[0], -1)

- 41 encoded_context = nn.Dense(self.output_dim, name='context_dense')(flat_context)

- 42 return nn.gelu(encoded_context)

- 43

- 44 class CPE_Generator(nn.Module):

- 45 pos_embedding_dim: int = 16

- 46 context_window_size: int = 3

- 47 max_seq_len: int = 1200

- 48 d_input: int = 4

- 49 @nn.compact

- 50 def __call__(self, x_raw):

- 51 batch_size, seq_len, _ = x_raw.shape

- 52 pos_table = self.param('pos_embeddings_table', nn.initializers.normal(stddev=0.01), (self.

→ max_seq_len, self.pos_embedding_dim))

- 53 pad_width = self.context_window_size // 2

- 54 padded_x = jnp.pad(x_raw, ((0,0), (pad_width, pad_width), (0,0)), mode='constant')

- 55 windows = jnp.stack([padded_x[:, i:i+self.context_window_size, :].reshape(batch_size, -1) for

→ i in range(seq_len)], axis=1)

- 56 encoder = ContextEncoder(output_dim=self.pos_embedding_dim, name='context_encoder')

- 57 encoded_contexts = encoder(windows.reshape(-1, self.context_window_size * self.d_input)).

→ reshape(batch_size, seq_len, -1)

- 58 pos_embeddings = jnp.tile(pos_table[:seq_len][None, :, :], (batch_size, 1, 1))

- 59 return pos_embeddings * encoded_contexts

- 60

- 61 class DeepCPE_Gated_ResidualDSConvBlock(nn.Module):

- 62 d_model: int; kernel_size: int; dilation: int; dropout_rate: float

- 63 cpe_signals: Any

- 64 gate_intermediate_dim: int = 64

- 65 @nn.compact

- 66 def __call__(self, x, deterministic=True):

- 67 residual = x

- 68 h = nn.LayerNorm()(x)

- 69 dw_out = nn.Conv(features=self.d_model, kernel_size=(self.kernel_size,), feature_group_count=

→ self.d_model, padding='SAME', kernel_dilation=(self.dilation,))(h)

- 70 h_conv_output = nn.Dense(self.d_model)(nn.gelu(dw_out))

- 71

- 72 cpe_projected = nn.gelu(nn.Dense(self.d_model, name='cpe_proj_for_gate')(self.cpe_signals))

- 73 combined = jnp.concatenate([h_conv_output, cpe_projected], axis=-1)

- 74 gate_input = nn.gelu(nn.Dense(self.gate_intermediate_dim, name='gate_dense_1')(combined))

- 75 gate_factor = nn.sigmoid(nn.Dense(self.d_model, name='gate_dense_2')(gate_input))

- 76

- 77 modulated_h = h_conv_output * gate_factor

- 78 modulated_h = nn.Dropout(rate=self.dropout_rate)(modulated_h, deterministic=deterministic)

- 79 return residual + modulated_h

- 80

- 81 # --- Standard Prediction Head Modules ---

- 82 class RegressionHead(nn.Module):

- 83 d_output: int; dense_dim: int; dropout_rate: float

- 84 @nn.compact

- 85 def __call__(self, x, deterministic=True):

- 86 norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)

- 87 x = x / (norm + 1e-6)

- 88 z = nn.Dense(self.dense_dim)(x); z = nn.gelu(z)

- 89 z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)

- 90 return nn.Dense(self.d_output)(z).squeeze(-1)

- 91

- 92 class GenericHead(nn.Module):

- 93 d_output: int; dense_dim: int; dropout_rate: float

- 94 @nn.compact

- 95 def __call__(self, x, deterministic=True):

- 96 z = nn.Dense(self.dense_dim)(x); z = nn.gelu(z)

- 97 z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)

- 98 return nn.Dense(self.d_output)(z)

- 99

- 100 # --- Main Network with CPE ---

- 101 class QuaeroCPE_SynthesisNet(nn.Module):

- 102 d_output: int; task_type: str; d_model_common: int; ds_num_blocks: int

- 103 ds_dilations: Sequence[int]; ds_kernel_size: int; motif_kernel_size: int

- 104 dropout_rate: float; head_intermediate_dim: int

- 105

- 106 @nn.compact

- 107 def __call__(self, x, deterministic=True, **kwargs):

- 108 # 1. Generate CPE signals once from the raw input

- 109 cpe_signals = CPE_Generator(name='cpe_generator')(x)

- 110

- 111 # 2. Initial Projection

- 112 h = nn.Dense(self.d_model_common)(x); h = nn.gelu(h)

- 113

- 114 # 3. CPE-Gated Residual Backbone

- 115 for i in range(self.ds_num_blocks):

- 116 h = DeepCPE_Gated_ResidualDSConvBlock(

- 117 d_model=self.d_model_common, kernel_size=self.ds_kernel_size,

- 118 dilation=(self.ds_dilations[i]), dropout_rate=self.dropout_rate,

- 119 cpe_signals=cpe_signals, name=f"cpe_gated_block_{i}"

- 120 )(h, deterministic=deterministic)

- 121 cnn_features = h

- 122

- 123 # 4. Dual-Path Pooling (Unchanged from ID 888)

- 124 main_path = jnp.concatenate([jnp.mean(cnn_features, axis=1), jnp.max(cnn_features, axis=1)],

→ axis=-1)

- 125 motif_f = nn.gelu(nn.Conv(self.d_model_common, (self.motif_kernel_size,), padding='SAME')(

→ cnn_features))

- 126 specialized_path = jnp.max(motif_f, axis=1)

- 127 fused = jnp.concatenate([main_path, specialized_path], axis=-1)

- 128 fused = nn.LayerNorm()(fused)

- 129 fused = nn.Dropout(rate=self.dropout_rate)(fused, deterministic=deterministic)

- 130

- 131 # 5. Task-Specific Heads (Unchanged from ID 888)

- 132 head_args = {'d_output': self.d_output, 'dense_dim': self.head_intermediate_dim, 'dropout_rate

→ ': self.dropout_rate}

- 133 if self.task_type == "regression": return RegressionHead(**head_args)(fused, deterministic=

→ deterministic)

- 134 else: return GenericHead(**head_args)(fused, deterministic=deterministic)

- 135

- 136 def _define_hyperparameters():

- 137 # Use the optimal hyperparameter set from ID 888

- 138 return {

- 139 'learning_rate': 0.001, 'd_model_common': 256, 'ds_num_blocks': 5,

- 140 'ds_dilations': (1, 2, 4, 8, 16), 'ds_kernel_size': 7, 'motif_kernel_size': 5,

- 141 'dropout_rate': 0.1, 'head_intermediate_dim': 128, 'weight_decay': 0.02,

- 142 }

- 143

- 144 def create_network(hparams: Dict[str, Any]):

- 145 network_params = {k: hparams.get(k) for k in _define_hyperparameters() if k not in ['

→ learning_rate', 'weight_decay']}

- 146 network_params.update({'d_output': hparams.get('d_output'), 'task_type': hparams.get('task_type')

→ })

- 147 return NetworkWrapper(QuaeroCPE_SynthesisNet(**network_params))

- 148

- 149 def create_optimizer(learning_rate: float = 0.001, weight_decay: float = 0.02):

- 150 return optax.chain(

- 151 optax.clip_by_global_norm(0.75),

- 152 optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

- 153 )

