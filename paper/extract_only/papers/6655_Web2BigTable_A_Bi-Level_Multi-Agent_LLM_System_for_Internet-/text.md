# arXiv:2604.27221v1[cs.AI]29Apr2026

## Web2BigTable: A Bi-Level Multi-Agent LLM System for Internet-Scale Information Search and Extraction

Yuxuan Huang1†, Yihang Chen3†, Zhiyuan He3, Yuxiang Chen3, Ka Yiu Lee2, Huichi Zhou3, Weilin Luo2, Meng Fang1, and Jun Wang3‡

1University of Liverpool 2Huawei Noah’s Ark Lab 3University College London

###### Abstract

Agentic web search increasingly faces two distinct demands: deep reasoning over a single target, and structured aggregation across many entities and heterogeneous sources. Current systems struggle on both fronts. Breadth-oriented tasks demand schema-aligned outputs with wide coverage and cross-entity consistency, while depth-oriented tasks require coherent reasoning over long, branching search trajectories. We introduce Web2BigTable, a multi-agent framework for web-to-table search that supports both regimes. Web2BigTable adopts a bi-level architecture in which an upper-level orchestrator decomposes the task into sub-problems and lower-level worker agents solve them in parallel. Through a closed-loop run–verify–reflect process, the framework jointly improves decomposition and execution over time via persistent, human-readable external memory, with self-evolving updates to each single-agent. During execution, workers coordinate through a shared workspace that makes partial findings visible, allowing them to reduce redundant exploration, reconcile conflicting evidence, and adapt to emerging coverage gaps. Web2BigTable sets a new state of the art on WideSearch, reaching an Avg@4 Success Rate of 38.50 (7.5× the second best at 5.10), Row F1 of 63.53 (+25.03 over the second best), and Item F1 of 80.12 (+14.42 over the second best). It also generalises to depth-oriented search on XBench-DeepSearch, achieving 73.0 accuracy. Code is available at https://github.com/web2bigtable/web2bigtable.

[Figure 1]

Figure 1: Web2BigTable interface during query execution. The left panel displays the user query (top left), the task decomposition status indicating the orchestrator has partitioned the query into parallel subtasks (bottom left). The right panel shows the shared workboard memory in raw Markdown view, organised into three sections: a subtask checklist tracking each worker’s assignment and completion status, a shared context block containing the extraction constraints and target schema, and individual worker result slots where each worker writes its structured table output into a dedicated tag-partitioned region.

#### 1 Introduction

Searching the open web and organising its contents into structured outputs is an important capability for LLM agents operating over real-world information [2]. Agentic web search broadly spans two regimes. In deep search, agents iteratively retrieve, read, and reason to resolve a single complex query [12, 10]. In wide search, the goal is instead to assemble a consistent, structured view over many entities and attributes grounded in heterogeneous sources. A recent formulation of these regimes is web-to-table construction, in which an agent, given a naturallanguage request, must autonomously search the web and return a schema-aligned table whose rows correspond to entities and whose columns capture the requested attributes [19]. While both regimes require tool use and multi-step reasoning, wide search introduces stronger demands on coverage, decomposition, coordination, and aggregation at scale, making it a challenging and practically important setting for evaluating general web agents.

Despite recent progress, existing systems remain limited on both axes. To see the problem, compare two regimes. A query from XBench-DeepSearch [3] such as a member of a fifth-generation Korean girl group has a brother 12 years younger; which group is it? requires chaining indirect clues across entertainment databases, fan wikis, and biographical records to identify a single entity, which is deep search. A query from WideSearch [19] such as list every Taylor Swift concert from 2010 to 2025 with date, city, and venue returns hundreds of rows, each of which must be correct and mutually consistent, which is wide search. The first demands long, coherent reasoning; the second demands broad, verified coverage. A monolithic agent handles neither well at scale: the context saturates, errors compound, and the fixed plan made at the start cannot adapt to what later steps uncover. Hierarchical agent frameworks [16, 11, 31] and automatically searched workflow pipelines [29, 8] partially address this issue through task decomposition, but often rely on fixed or weakly adaptive planning strategies, with limited feedback from downstream execution to upstream decomposition [13, 27]. Meanwhile, recent self-evolving and memory-augmented agents improve execution through reusable external skills [7, 20, 17] or structured long-term memory [23, 24], yet this adaptation has largely been studied at a single level, without jointly refining how tasks are decomposed and how sub-tasks are solved. As a result, effective agentic web search requires not only decomposition, but also a mechanism to co-adapt planning and execution while enabling multiple workers to coordinate, avoid redundant exploration, reconcile conflicting evidence, and expose coverage gaps during search.

Definition (Web-to-Table Search). Given a natural-language query and a target schema, produce a structured table by searching the open web, where each row is a distinct entity, each column is a requested attribute, and every cell is independently verified against web sources.

To address these challenges, we propose Web2BigTable, a multi-agent framework for webto-table search that supports both breadth-oriented and depth-oriented instances of this task. Web2BigTable uses a bi-level architecture in which an upper-level orchestrator decomposes the task into sub-problems and lower-level worker agents solve them in parallel. The framework improves over time through a closed-loop run-verify-reflect process that jointly refines decomposition and execution, making it self-evolving through persistent, editable external memory. Concretely, the orchestrator accumulates human-readable decomposition skills that improve future task partitioning, while workers acquire reusable execution skills for retrieval, evidence verification, and intermediate synthesis. All adaptation is mediated through external memory rather than gradient updates, leaving the underlying LLMs frozen throughout. During execution, workers coordinate through a shared workspace that makes partial findings visible, enabling them to reduce redundant exploration, resolve conflicts, and adapt their search trajectories to the evolving global state.

We evaluate Web2BigTable on two benchmarks covering both breadth-oriented and depthoriented web search settings. On WideSearch [19], a benchmark for broad-coverage search, Web2BigTable achieves state-of-the-art performance, reaching an Avg@4 Success Rate of 38.50

###### Stage 1: Orchestrate Stage3: Evolve

Inference

(Run-Verify-Reflect)

[Figure 2]

###### Orchestrator LLM

[Figure 3]

Training

Update Strategy Write in Memory

Contrast System Output With Gold-Standard

[Figure 4]

Orchestrator LLM

Strategy Memory 𝑺𝒐

Skill Router

[Figure 5]

(Markdown-based， Decomposition Strategy)

Skill Router

Cluster Error Patterns

[Figure 6]

[Figure 7]

###### User Input

Decomposition Steps

Refine & Modularise Skills

Task Decomposition

(Complex Long Query) N Subtasks

Instruction, Output Schema

###### Stage 2: Execute

Self-Evolving Self-Evolving

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### Asynchronous Read and Write

Workboard Write

WorkboardWrite

Skill Write

Skill Write

[Figure 12]

[Figure 13]

Script Execution

Script Execution

Search

Search

(Avoid Duplication, Filling Gaps)

Shared Skill Bank 𝑺𝒘 (Dynamic Retrieval)&

Shared Skill Bank 𝑺𝒘 (Dynamic Retrieval)& Self-Repair)

Workboard Read

[Figure 14]

Skill Read

Skill Read

WorkboardRead

Self-Repair) Worker Agent 2

Worker Agent 1

[Figure 15]

WorkboardWrite

### …

### …

Workboard Write

###### Shared Workboard 𝒎𝒆

Structured Output

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(Markdown-based, File Locks, Tag Partitions)

Skill Write

Skill Write

[Figure 20]

[Figure 21]

(BigTable)

Script Execution

Script Execution

WorkboardRead

Search

Search

Workboard Read

Shared Skill Bank 𝑺𝒘 (Dynamic Retrieval)& Self-Repair)

Shared Skill Bank 𝑺𝒘 (Dynamic Retrieval)& Self-Repair)

Extract Output

Skill Read

Skill Read

Worker Agent i

Worker Agent i-1

- Figure 2: Architecture of Web2BigTable, a unified training and inference framework. The

upper-level orchestrator decomposes each query into subtasks using the Orchestrator Skills So, and lower-level workers execute them in parallel, drawing execution skills from the shared Worker Skills Sw and coordinating asynchronously through the Workboard me, protected by file locks and tag partitioning. Red arrows denote the additional write-back and update flows that are active only at training time, where a verify–reflect step distils each episode’s trajectories into monotone updates to So and Sw. At inference the red paths are inactive and the two skill banks are consumed read-only. Both phases operate entirely through external memory, leaving the underlying LLMs frozen throughout.

(7.5× the second best at 5.10), a Row F1 of 63.53 (+25.03 over the second best), and an Item F1 of 80.12 (+14.42 over the second best). The framework also generalises to depth-oriented search, achieving 73.0 accuracy on XBench-DeepSearch [3].

Our main contributions are as follows:

- • We introduce Web2BigTable, a multi-agent framework for web-to-table search that supports both breadth-oriented and depth-oriented search regimes.
- • We propose a bi-level adaptation mechanism that jointly evolves task decomposition and worker execution through persistent, human-readable external memory, with self-evolving to the underlying LLMs.
- • We develop an asynchronous coordination mechanism that enables parallel workers to share progress, reduce redundant exploration, reconcile conflicting evidence, and respond to emerging coverage gaps during search.
- • We achieve state-of-the-art results on WideSearch and demonstrate strong generalisation to the structurally distinct XBench-DeepSearch benchmark.

#### 2 Web2BigTable: A Bi-Level Memory-Based Framework

Web2BigTable instantiates the web-to-table search task as a concrete multi-agent framework organised around two design principles. The first is bi-level self-evolving, which jointly refines how queries are decomposed and how sub-tasks are executed by learning two persistent skill banks

- (Section 2.2) through an automated training pipeline (Section 2.3). The second is multi-agent

coordination via a shared workboard, which lets concurrent workers observe others’ progress, avoid redundant exploration, and reconcile conflicting evidence (Section 2.4.2). Both features operate entirely over external memory, leaving the underlying LLMs frozen throughout.

##### 2.1 Problem Definition

Web-to-table search task poses a qualitatively different challenge from conventional agentic web search. Whereas deep question answering converges on a single free-form response, web-to-table search demands a structured table whose rows enumerate many entities and whose cells are each anchored in live web evidence. The task is therefore inherently wide: a single query may span hundreds of rows, and success is governed by breadth of entity coverage rather than by depth of reasoning on any individual target.

Formally, an instance is a pair T = ⟨q,W⟩. The user query q specifies a schema of attributes to extract, for instance columns Date, Venue, City across hundreds of concert entries and W denotes the open web environment that the agent queries through retrieval and browsing tools. A valid output is a table X ∈ X whose rows enumerate the entities in q, whose columns instantiate the attributes, and whose cells each hold a value verified against web sources. Throughout this paper we adopt a scalar global utility U(X) ∈ [0,1] summarising overall extraction quality as the optimisation target.

Any policy π solving this task unfolds as an action–observation loop. At each step t, π samples the next action conditioned on the query q and the accumulated interaction history ht = (o1,x1,...,ot,xt):

xt+1 ∼ π(· | q,ht). (1)

Each action is either a tool-call such as a search query or file operation that queries W and returns an observation ot+1, the retrieved content such as a page, snippet, or tool output, appended to ht+1, or a terminal action that emits the completed table as the policy’s final response and ends the trajectory at step T. The output table is precisely X := xT, the content of this terminal action.

##### 2.2 System Architecture

The architecture of Web2BigTable is shaped by two structural demands of web-to-table search. First, a single instance spans many entities under a structured schema. The single-agent policy of Equation (1) must therefore handle retrieval, state tracking, and synthesis across hundreds of rows within a bounded context window, while remaining unable to exploit the conditional independence across entities that the task naturally exhibits [12]. This motivates a factorisation of the policy into two levels: an upper-level orchestrator that decomposes the query into independent subtasks, and a lower-level pool of workers that resolves these subtasks in parallel, with the two layers strictly separated and interacting only through shared memory.

Second, the bi-level system must sustain state at two temporal scales. Concurrent workers need to exchange partial evidence within a query to avoid redundant exploration and to reconcile conflicting retrievals, and the system as a whole needs to retain the decomposition and execution skills it acquires across queries. Since the underlying language models remain frozen, such state cannot be absorbed into a single context window; it is instead externalised into two complementary memory regimes:

- • Long-term semantic memory: a persistent store of skills that evolves only during training and is frozen at inference. It spans two levels: the upper-level orchestrator skills So, containing decomposition strategies used by the orchestrator, and the lower-level worker skills Sw, holding execution skills shared across all workers for information seeking.

- • Short-term working memory: a scratchpad that is transient within a single episode,

realised by the workboard me, through which concurrent workers coordinate their ongoing trajectories.

Together, these memories replace the single-agent policy of Equation (1) with a bi-level factorisation,

τ = (τ1,...,τN) ∼ πo(· | q,So), xi ∼ πw(i)(· | τi,me,si), si ∈ Sw,

(2)

where each worker’s execution skill si is retrieved from the shared bank Sw via the Memento-Skills mechanism [32]; the orchestrator selects a single decomposition skill from So through a task-router keyed on the query’s structural type (Section 2.4.1). Equation (2) recasts the single-agent policy of Equation (1) through three simultaneous decompositions. First, the monolithic π splits into

an upper-level orchestrator policy πo and a pool of lower-level worker policies {πw(i)}Ni=1: the orchestrator commits to a decomposition τ = (τ1,...,τN) under the orchestrator skills So, and each worker i independently produces a partial output xi conditioned on the shared workboard me and a retrieved skill si ∈ Sw. Second, the global interaction history ht that a single agent would otherwise accumulate within its context window is externalised into the shared workboard me and partitioned across workers as {hti}i; each worker reads only the slice of me relevant to its subtask, so the per-step context cost depends on subtask complexity rather than on the full table size. Third, the output X of Section 2.1 is assembled from the partial outputs as X = (x1,...,xN), and the global utility U(X) becomes the optimisation target used to evolve So and Sw during training in Section 2.3. The two timescales are naturally separated: me turns over within minutes, whilst So and Sw accumulate experience across queries over hours or days.

The workboard me is not merely a message relay but a shared epistemic state. Because every worker can read the entire workboard whilst writing only to its own tagged region, the architecture makes partial findings immediately available and keeps writes non-destructive over disjoint slots, yielding a form of asynchronous consensus that scales with the worker pool whilst preserving the simplicity of a plain Markdown document. The specific adaptive behaviours this design enables are detailed in Section 2.4.2.

Memory updates. The three memory regimes evolve at different timescales through dedicated update operators. The workboard is refreshed at every inner step t as contributions from the active worker set At are merged,

mte+1 = Me mte,{hti+1}i∈At , (3)

with per-worker writes serialised through file locks so that no two updates collide on disjoint slots. As a short-term working memory, me stores the ongoing natural-language trace of a single episode, holding partial findings and intermediate observations that workers read and extend as the query unfolds, and the entire document is discarded once the episode terminates. The two skill banks evolve only across training episodes, through parallel reflect operators,

Sok+1 = Mo Sok,rok+1 , (4) Swk+1 = Mw Swk,rok+1 , (5)

where rok+1 is the structured error report distilled from episode k’s trajectories against its gold reference. Both Mo and Mw are realised as LLM-driven Run–Verify–Reflect pipelines

- (Section 2.3) that append new skills monotonically, without fine-tuning the underlying LLMs.

By contrast, So and Sw constitute a long-term semantic memory, stored as human-readable SKILL.md files that crystallise the distilled, generalised procedures extracted from completed episodes and accumulate monotonically across runs. The two memory regimes are therefore complementary rather than redundant: the workboard records what the system is currently

doing and is thrown away, whereas the skill banks record what the system has learned to do and are kept.

Two-phase pipeline: training and inference. The split between long-term skill banks and short-term workboard partitions Web2BigTable’s operation into two cleanly decoupled phases, separating how skills are accumulated from how they are exploited. The training phase (Section 2.3) runs on a small set of queries paired with gold-standard tables, distilling each episode’s outcome into updates of So and Sw through a run–verify–reflect loop. The inference phase (Section 2.4) freezes the evolved banks (So∗,Sw∗) and answers unseen user queries in a single forward pass, with no further learning. We present the two phases in their natural order: training first, to specify how the banks are acquired, followed by inference, which consumes them unchanged.

##### 2.3 Training Phase: Self-Evolving the Skill Banks

The policy in Equation (2) is only as good as the skills populating So and Sw. Web2BigTable acquires both banks autonomously through a dedicated training phase that runs on a small split of queries paired with gold-standard tables, so that extraction quality can be evaluated objectively. At the end of training the evolved banks (So∗,Sw∗) are frozen and consumed unchanged by the inference phase of Section 2.4.

Training objective. Training maximises the expected global utility U(X) = Item-F1(X, Xgold), a cell-level scoring function that compares the predicted table X against the gold reference Xgold using type-specific comparators (exact match, numeric tolerance, URL normalisation, and LLMbased semantic judgement for free text). Because subtasks are scoped to disjoint entity partitions, each worker contributes additively to U: improvements to So (yielding better decompositions) and to Sw (yielding better per-worker execution) both raise U without interfering, which justifies training the two skill banks jointly rather than sequentially.

Run–Verify–Reflect loop. Inside each training episode k, the system first performs one inference pass with its current skills (the inference procedure detailed in Section 2.4), then compares the produced table Xk against the gold reference Xkgold, and finally distils the resulting error report into skill updates through two parallel reflect operators: Mo over orchestrator skills (Section 2.3.1) and Mw over worker skills (Section 2.3.2). Both updates are monotone appends of human-readable SKILL.md files; the underlying LLMs are never fine-tuned. The full procedure

- is summarised in Algorithm 1 and illustrated in Figure 3.

###### 2.3.1 Orchestrator Skills Evolution

The reflect operator Mo of Equation (4) is realised as a three-stage pipeline: Run executes one training episode, Verify produces the structured error report rok+1, and Reflect consumes rok+1 to synthesise new decomposition skills that append to Sok. All three stages are orchestrated by LLM-driven subroutines, with no gradient update applied to the underlying models.

Run. The system processes a batch of training queries through the full pipeline, archiving structured outputs alongside per-worker JSONL trajectories that capture all tool invocations, intermediate reasoning, and temporal metadata.

Verify. Outputs are systematically evaluated against gold-standard references in three steps. First, each worker’s trajectory is compressed into a structured digest by a highly efficient LLM, capturing the applied decomposition strategy, search queries issued, failure points, and coverage statistics. Second, predicted outputs and gold references are parsed into row sets and compared

write

###### Orch. Skill Bank So

###### Worker Skill Bank Sw

write

Sok+1 = Mo(Sok, rok+1)

Swk+1 = Mw(Swk , rok+1)

rok+1

read read

Stage 2: Execute Parallel worker loop

Stage 1: Orchestrate Decompose task τ ∼ πo(· | qk,Sok)

###### Stage 3: Evolve

Training Queries {qk}Kk=0−1

###### Evolved Skill Banks So∗, Sw∗

τ Xk

U(Xk) = Item-F1(Xk, Xkgold)

xti+1 ∼ πw(i)(· | τi,mte,si)

reflect and update skills

read write

Ground Truth {Xkgold}Kk=0−1

###### Workboard me

mte+1 = Me(mte, {hti+1}i)

next training episode k+1

Input Training phase Output

###### Figure 3: Training (self-evolving) flow of Web2BigTable over one episode k. For each

training query qk, Stage 1 reads the long-term orchestrator skills So and decomposes qk into subtasks τ. Stage 2 dispatches these subtasks to N parallel workers, which read execution skills from Sw and read/write the short-term workboard me until convergence. Stage 3 verifies the aggregated output Xk against the gold reference, produces the structured reflection rok+1, and consolidates it into both So (via Mo) and Sw (via Mw). Episodes are processed sequentially: the bottom black loop moves from episode k to k+1 without replanning within an episode. After K episodes, the two skill banks (So∗,Sw∗) are frozen and returned as the training output, then used unchanged during inference.

at the cell level using type-specific scoring (exact match, numeric tolerance, URL normalisation, and LLM-based semantic judgement for free text). Third, results are aggregated into a structured error report highlighting missing row categories, low-accuracy columns, and trajectory anomalies such as context truncation or tool timeouts.

Reflect. The error report is translated into actionable skills through three sequential steps. Training queries are clustered by structural decomposition pattern rather than semantic topic, typically yielding clusters such as split-by-entity and split-by-time-period. For each cluster, a reflection LLM synthesises a generalisable decomposition skill from the observed errors and successful trajectories. To rigorously prevent overfitting, these skills are constrained to use only structural placeholders. Finally, a router skill is generated that maps query characteristics to the appropriate strategy, enabling the orchestrator to process unseen queries autonomously.

All learned skills are persisted as SKILL.md files, rendering them human-readable, editable, and version-controllable. Concretely, the reflection LLM itself performs the memory write: it emits the newly synthesised decomposition and router skills directly as SKILL.md files into the skill directory backing Sok, yielding Sok+1. Prior skills are never overwritten, so updates are monotone additions. Because skills are plain text consumed via in-context learning, no parameter fine-tuning is required, and the orchestrator simply processes refined instructions during subsequent episodes, closing the slow-timescale loop.

###### 2.3.2 Worker Skills Evolution

The consolidation operator Mw of Equation (5) is realised by a reflection LLM that emits the skill changes accumulated in episode k directly into the shared bank as SKILL.md entries; as

with Mo, the update is a monotone append, so Swk ⊆ Swk+1. The runtime changes that Mw consolidates arise from two mechanisms operating within each episode.

Skill resolution and creation. When a specific capability is required, the SkillResolver executes a strictly prioritised search: (1) exact-name matching of local skills within the workspace; (2) semantic retrieval across both local and cloud-based catalogues (comprising over 8,000 preconfigured skills), employing a hybrid pipeline that integrates BM25 keyword matching with ChromaDB embedding search (BAAI/bge-m3), subsequently fused via Reciprocal Rank Fusion (RRF) and optionally refined by a cross-encoder; and (3) when no suitable match is found, the SkillCreator module leverages the worker’s LLM to synthesise a novel skill on demand, as either a function skill (an executable Python script with a CLI entry point, validated via AST parsing) or a knowledge skill (a structured Markdown document with YAML frontmatter). Newly synthesised skills are instantaneously indexed into the local BM25 and embedding stores.

Error-driven self-repair. Should a skill execution terminate in failure, the worker initiates an autonomous reflection loop. The error trace and the current skill code are supplied to the LLM to synthesise a corrected iteration, subjected to strict AST validation before replacing the original. The same reflection loop supports extending existing skills with new capabilities via evolve_skill, whilst preserving backwards compatibility.

These mechanisms operate concurrently at runtime. Because the skill repository is globally shared across active workers via a singleton application context, any skill generated or amended by an individual agent is instantaneously propagated to the collective through an automated index refresh.

- Algorithm 1 Web2BigTable Training (Self-Evolving)

Input: Training set {(qk,Xkgold)}Kk=0−1 of K queries paired with ground-truth tables; initial skill banks

So0,Sw0 ; hyperparameters N (worker pool size), Tmax (max inner steps).

Output: Evolved Skill Banks (So∗,Sw∗) (consumed by Algorithm 2 at inference time).

- 1: for episode k = 0,1,...,K − 1 do
- 2: {Stage 1 + Stage 2: run inference with current skill banks}
- 3: Xk ← Inference(qk; Sok,Swk; N,Tmax) {Algorithm 2}
- 4:
- 5: {Stage 3: Evolve: score against gold, then consolidate into skill banks}
- 6: Compute utility U(Xk) ← Item-F1(Xk, Xkgold) {cell-level training objective}
- 7: Compress trajectories and aggregate error report Ek
- 8: Generate reflection rok+1 ← Reflect(Ek)
- 9: Sok+1 ← Mo(Sok,rok+1) {Eq. (4)}
- 10: Swk+1 ← Mw(Swk,rok+1) {Eq. (5)}
- 11: end for
- 12: return (SoK,SwK) {frozen skill banks (So∗,Sw∗)}

##### 2.4 Inference Phase: Query-Time Execution

With the skill banks (So∗,Sw∗) produced by the training phase of Section 2.3, inference runs a single forward pass over a user query q using these banks as frozen read-only inputs. No reflection, verification, or memory update is performed; the output is a structured table X. The procedure

- is summarised in Algorithm 2 and illustrated in Figure 4.

###### 2.4.1 Orchestrator

The orchestration layer governs query decomposition at runtime. Implemented as a single agent powered by a reasoning LLM such as GPT-5 mini, it realises the upper level of the bi-level policy in Equation (2), partitioning the user query q into N self-contained subtasks τ = (τ1,...,τN) conditioned on the frozen orchestrator skills So∗. It first classifies the query by invoking a task-router skill that evaluates structural properties (entity count, schema complexity, temporal

Orch. Skill Bank So∗

###### Worker Skill Bank Sw∗

decomposition strategies semantic retrieval

execution skills for search semantic retrieval

read read

Stage 2: Execute Parallel worker loop

Stage 1: Orchestrate Decompose task τ ∼ πo(· | q,So∗)

###### User Query q

Big Table X

τ aggregate

xti+1 ∼ πw(i)(· | τi,mte,si)

Workboard me Low-level Coordination

mte+1 = Me(mte, {hti+1}i)

Input Inference Phase Output

- Figure 4: Inference flow of Web2BigTable on an unseen user query q. Using the trained skill banks So∗ and Sw∗ as frozen read-only inputs, Stage 1 decomposes q into subtasks τ. Stage 2 runs N parallel workers that resolve execution skills from Sw∗ and coordinate through the shared workboard me (per-query, short-term); their partial outputs {xi} are aggregated into the structured big table X. No verification, reflection, or memory update is performed: the system runs a single forward pass and returns X.

requirements, and expected output volume), and then invokes the corresponding decomposition skill. Each resulting subtask specification includes a natural language instruction scoped to a specific data partition, the expected output schema, and a target volume of 10 to 20 items, so that the workload stays strictly within the worker’s context limits.

Concurrently, the orchestrator initialises the shared workboard with a subtask checklist and tagged result slots. Once all subtasks have terminated, it performs a validation pass over row counts, column coverage, and data consistency before aggregating worker outputs into a unified table. No post-processing (summarisation, deduplication, etc.) is applied: correctness is expected to emerge directly from the decomposition design.

###### 2.4.2 Workers

The execution layer resolves individual subtasks and coordinates concurrent workers. It realises the lower level of the bi-level policy in Equation (2) step by step: at each step t, an active worker i samples an action xti+1 conditioned on its assigned subtask τi, the current shared workboard mte, and an execution skill si retrieved from the frozen worker skills Sw∗:

xti+1 ∼ πw(i) · | τi,mte,si . (6)

In direct analogy with Equation (1), each xti+1 is either a tool call that queries W (whose observation is appended to the worker’s local history hti) or a response written to worker i’s tagged slot on me. A worker terminates at the first step T at which xTi is a response; we set xi := xTi as worker i’s partial output, and the orchestrator aggregates the terminal responses into X = (x1,...,xN).

Implementation. An MCP (Model Context Protocol) server manages the worker pool, dispatching subtasks concurrently via asyncio.gather() with a semaphore controlling the maximum concurrency level (up to 10 workers). Each worker is an independent Memento-Skills [32] agent powered by Gemini 3 Flash, executing a ReAct loop of reasoning and tool use [26]. At

each step, the worker either invokes a tool or produces a final response, with generation and tool execution governed by timeouts of 120 seconds and 30 seconds, respectively.

Workers are equipped with eight built-in tools (bash, str_replace, file_create, view, read_skill, route_skill, read_workboard, edit_workboard) alongside dynamically discovered skills. Memory-intensive resources, including the BM25 index, embedding store (BAAI/bgem3 with ChromaDB), cloud skill catalogue, and cross-encoder reranker, are shared across workers via a singleton application context to eliminate redundant loading. Each worker’s complete execution trace is recorded as a JSONL trajectory, providing the foundational input for the self-evolving pipeline.

Shared workboard. The shared workboard is a structured Markdown document that instantiates the execution memory me. Unlike message-passing or broadcast architectures, it provides a persistent, globally visible state that all workers can read from and contribute to. It is partitioned into three sections: (i) Task checklist: a list of all subtasks and their completion statuses, affording every worker visibility into global progress; (ii) Worker slots: a tag-partitioned region where each worker writes results under a unique identifier such as <t1_result>, preventing write conflicts; and (iii) Shared context: background information, constraints, and formatting conventions pre-populated by the orchestrator.

Workers interact with the workboard through two dedicated tools: read_workboard for observing global progress and peer outputs, and edit_workboard for appending results to assigned slots. Write operations are protected by file locks and strictly confined to each worker’s tagged region, whilst the entire document remains globally readable. The per-step merging of these contributions into me follows the update rule Me specified in Equation (3).

Dynamic coordination through read-write asymmetry. This structural asymmetry is the bedrock of coordination. Because workers can freely observe peer contributions, several adaptive behaviours emerge autonomously:

- • Redundancy avoidance: a worker observing entities already extracted by a peer skips redundant searches and refocuses on its assigned partition.
- • Coverage gap detection: by inspecting peer outputs, an active worker can identify missing fields or inconsistencies and dynamically adjust its retrieval strategy.
- • Strategy adaptation: workers assimilate successful patterns recorded by peers, such as highly effective source URLs, query formulations, or formatting conventions, whilst circumventing approaches that previously resulted in failures.

Workers do not merely post final answers. Their LLMs autonomously evaluate what intermediate information warrants sharing, such as experimental plans, partial findings, or discovered URLs, and when to write it. These continuous updates transform the workboard from a static result ledger into a live coordination medium. Because workers execute at heterogeneous speeds, this staggered dynamic creates a natural information cascade, with early-finishing workers enriching the shared context for those still running, progressively refining subsequent outputs without explicit message-passing overhead.

#### 3 Experiments

We evaluate Web2BigTable across two complementary benchmarks: WideSearch [19] for broadcoverage structured extraction, and XBench-DeepSearch [3] for deep, multi-hop reasoning. We benchmark our framework against state-of-the-art single-agent, end-to-end, and multi-agent systems, alongside comprehensive analysis to isolate the respective contributions of the learned orchestrator strategies, worker-level skill evolution, and workboard-based coordination.

- Algorithm 2 Web2BigTable Inference

Input: User query q; frozen skill banks So∗,Sw∗ from Algorithm 1; hyperparameters N (max worker pool),

Tmax (max inner steps). Output: Big Table X.

- 1: {Stage 1: Orchestrate}
- 2: Classify q via task-router skill in So∗
- 3: Sample subtasks τ = (τ1,...,τN

q

) ∼ πo(· | q,So∗), with Nq ≤ N {Eq. (2)}

- 4: Initialise workboard m0e with checklist and tagged slots
- 5:
- 6: {Stage 2: Execute (asynchronous worker loop)}
- 7: t ← 0
- 8: while ¬Converged(mte) and t < Tmax do
- 9: Identify active worker set At ⊆ {1,...,Nq} via asynchronous dispatch
- 10: for all worker i ∈ At in parallel do
- 11: Read current workboard mte via read_workboard
- 12: Resolve skill si ← SkillResolver(τi,Sw∗)
- 13: Generate sub-output xti+1 ∼ πw(i)(· | τi,mte,si) {Eq. (6)}
- 14: Lock slot ⟨τi⟩; write xti+1 via edit_workboard; Unlock
- 15: end for
- 16: mte+1 ← Me(mte,{hti+1}i∈At)
- 17: t ← t + 1
- 18: end while
- 19: T ← t
- 20:
- 21: {Aggregate and return}
- 22: X ← Aggregate({xTi }Ni=1q ), validated against the target table structure implicit in q
- 23: return X {big table}

##### 3.1 Datasets

WideSearch. WideSearch [19] evaluates the reliability of LLM agents in large-scale, structured information extraction from the live web. It comprises 200 manually curated tasks (100 English, 100 Chinese) spanning 15 domains. Each task requires the agent to extract multi-dimensional atomic data and organise it into a structured table. We adopt a two-phase evaluation protocol. In the training phase, we synthesise a set of 20 queries by perturbing task parameters (e.g., entity counts or time ranges) to avoid overlap with the original benchmark, and use these to learn the orchestrator’s decomposition skills via the run-verify-reflect pipeline. In the inference phase, the strategy memory mo is frozen and the unmodified 200 tasks serve as the held-out test set.

XBench-DeepSearch. XBench-DeepSearch [3] is a professionally annotated Chinese benchmark designed to assess deep search and tool-use capabilities. Contrasting with WideSearch, which emphasises breadth, XBench-DeepSearch evaluates depth: each task necessitates multi-hop reasoning, cross-source verification, and precise answer extraction from dynamic web content. We adopt the same two-phase protocol. In the training phase, we generate 20 modified queries for strategy learning. In the inference phase, the strategy memory is frozen and the unmodified dataset serves entirely as the held-out test set.

##### 3.2 Evaluation Metrics

WideSearch. Following the WideSearch evaluation protocol, we report three complementary metrics at increasing levels of granularity:

- • Success Rate (SR): The most stringent metric, requiring a 100% match with the groundtruth table. A task is successful only if every row and cell match exactly.

- • Row-level F1: Treats each table row as a unit, measuring the agent’s ability to retrieve complete and correct records. Precision and recall are computed over the matched rows.
- • Item-level F1: The most granular metric, evaluating individual cells. Cell matching employs type-specific comparators: exact match for categorical fields, numeric tolerance for quantities, URL normalisation for links, and LLM-based semantic judgement for free text.

Each system is run four times independently on the test set. We report both Avg@4 (mean across four runs) and Max@4 (best of four runs) to capture average-case reliability and best-case capability, respectively.

XBench-DeepSearch. Following the established benchmark protocol, we report Accuracy evaluated via LLM-as-judge.

##### 3.3 Model Configurations

Web2BigTable configuration. Our model selection balances role-specific capability, cost, and experimental intent. The orchestrator requires planning and reasoning; the workers require fast, reliable tool use. We deliberately pair two lightweight, cost-efficient models with high rate limits to demonstrate that performance stems from framework design rather than backbone capability; stronger LLMs would likely yield further gains (Section 3.4). Accordingly, the orchestrator is powered by GPT-5 mini, managing task routing, decomposition, and final synthesis. Each task is served by up to 10 parallel workers powered by Gemini 3 Flash, executing ReAct loops with access to web search, file operations, shell commands, and a shared library comprising over 8,000 cloud skills. Workers coordinate through a markdown-based workboard protected by file locks. The orchestrator’s decomposition skills are acquired through the online strategy learning pipeline (Section 2.3.1), where each completed training query triggers a three-stage cycle of verification, reflection, and skill updating. On each benchmark, skills are learned from 20 benchmark-specific training queries and frozen prior to test-time evaluation.

Baselines. On WideSearch, we compare against three categories of systems:

- 1. Single Agent: Frontier LLMs prompted directly with the full task, including Claude Sonnet 4.5, Gemini 3 Pro, GPT-5 High and Doubao-1.6 (with and without thinking mode).
- 2. End-to-End Systems: Proprietary agentic pipelines from Claude, Gemini, and OpenAI that handle search, extraction, and synthesis as integrated products.
- 3. Multi-Agent Frameworks: The same frontier LLMs deployed within a standardised orchestration framework, coordinating parallel agents via a hierarchical decomposition pipeline.

On XBench-DeepSearch, baselines include foundation models with tool access (e.g., GLM-4.5, Minimax-M2), proprietary deep research systems (e.g., MiroFlow, OAgents), and open-source agentic models (e.g., DeepMiner-32B-RL, WebShaper-32B). Baseline results are sourced from published reports. For this benchmark, Web2BigTable uses 5 parallel workers rather than 10, as deep search tasks benefit more from sequential depth than parallel breadth.

- 3.4 Performance Gain Analysis

We first conduct two sets of analysis experiments. The first isolates the contribution of each system component by disabling individual mechanisms whilst keeping all other configurations identical (Table 1). The second isolates the contribution of the framework itself from that of the underlying LLMs by comparing Web2BigTable against its own backbone models run as single agents (Table 2). All WideSearch results are reported as Avg@4 over four independent runs; XBench-DeepSearch results are reported as accuracy.

WideSearch (Avg@4) XBench

Configuration

SR Row F1 Item F1 Acc. Full system 38.50 63.53 80.12 73.0 w/o learned orch. skills 7.00 45.23 62.87 41.0 w/o workboard 27.50 54.81 73.45 60.0 w/o worker skill evolution 33.00 59.67 76.38 64.0

- Table 1: Contribution of each system component. Removing learned orchestrator skills causes the largest drop across all metrics, confirming that bi-level strategy learning is the primary driver of performance.

Configuration

WideSearch (Avg@4) XBench SR Row F1 Item F1 Acc.

GPT-5 mini (single agent) 4.00 22.10 33.28 35.0 Gemini 3 Flash (single agent) 3.00 18.70 31.61 28.0

Web2BigTable GPT-5 mini + Gemini 3 Flash 38.50 63.53 80.12 73.0

∆ vs GPT-5 mini +34.50 +41.43 +46.84 +38.0 ∆ vs Gemini 3 Flash +35.50 +44.83 +48.51 +45.0

- Table 2: Framework contribution vs underlying LLM capability. The same models that power Web2BigTable achieve far lower scores as single agents, confirming that the performance advantage stems from the framework design rather than model capability.

Learned orchestrator skills are critical. Ablating the learned decomposition skills causes a severe performance drop across both benchmarks. On WideSearch, the Success Rate falls from 38.50 to 7.00, Row F1 from 63.53 to 45.23, and Item F1 from 80.12 to 62.87; on XBenchDeepSearch, accuracy decreases from 73.0 to 41.0, a drop of 32.0 points. Without these skills, the orchestrator defaults to generic LLM-driven decomposition, which induces systematic coverage gaps in WideSearch and disjointed reasoning chains in XBench-DeepSearch. This confirms that slow-timescale orchestrator updates drive the majority of the performance advantage.

Workboard coordination enables dynamic gap recovery. Disabling the shared workboard reduces WideSearch Row F1 from 63.53 to 54.81 and XBench-DeepSearch accuracy from 73.0 to 60.0. Without shared memory, workers operate in isolation: they cannot disseminate high-quality sources, perform peer inspection, or facilitate orchestrator-triggered follow-up iterations. On WideSearch, this manifests as redundant queries and unrectified coverage gaps; on XBenchDeepSearch, intermediate findings from preliminary reasoning hops remain inaccessible to downstream agents.

Worker skill evolution provides complementary gains. Ablating skill evolution induces a more modest decline, reducing WideSearch Row F1 to 59.67 and XBench-DeepSearch accuracy to 64.0. Constrained to their default toolset, workers lose the capacity to discover task-specific cloud skills or autonomously repair execution failures. The comparatively muted impact indicates that baseline capabilities suffice for the majority of tasks, but the persistent gap confirms that worker-level adaptation yields a consistent complementary enhancement.

Performance stems from the framework, not the underlying LLMs. A natural question is whether Web2BigTable’s gains are attributable to the framework or simply to the choice of underlying LLMs. Table 2 addresses this directly. GPT-5 mini and Gemini 3 Flash, the two models powering our system, achieve only 33.28 and 31.61 Item F1 respectively as single agents. The full framework reaches 80.12 using these same models, a gain of over 46 points. Crucially, the strongest single-agent baselines in Table 3 use strictly more powerful models (Claude-4.5-Sonnet at 65.70 Item F1, GPT-5 High at 62.20) yet still fall short of Web2BigTable by at least 14 points. This confirms that the performance advantage stems from bi-level strategy learning, workboard coordination, and runtime skill evolution, not from the raw capability of the backbone LLMs.

##### 3.5 Benchmark Comparison

Success Rate Row F1 Item F1

Model / System

Avg@4 Max@4 Avg@4 Max@4 Avg@4 Max@4 Single Agent

- Claude Sonnet 4 (Thinking) 2.30 5.00 31.70 41.90 57.90 66.70 Claude-4.5-Sonnet – – – – 65.70 –

- Gemini 2.5 Pro 1.50 5.00 30.00 41.40 51.00 63.60 Gemini-3-Pro – – – – 57.00 –

- OpenAI o3-high 4.50 9.00 34.00 44.10 52.60 62.30 GPT-5 High – – – – 62.20 – GPT-5 mini 4.00 7.50 22.10 32.40 33.28 44.10 Kimi K2 1.10 3.50 29.70 41.40 54.40 65.10 Seed1.8 – – – – 63.80 – DeepSeek-R1 0.40 1.50 20.70 31.70 41.30 55.10 Doubao-1.6 2.60 5.00 30.00 44.10 48.30 63.90

- Doubao-1.6-non-thinking 1.00 3.50 27.20 39.90 49.00 62.00 Gemini 3 Flash 3.00 6.00 18.70 28.50 31.61 42.30 End-to-End System

Claude 2.50 5.00 24.10 33.50 48.40 58.50 Gemini 4.30 8.00 36.60 45.40 59.10 67.20 OpenAI o3 3.00 5.50 23.90 36.00 45.50 56.50

Multi-Agent Framework

Claude Sonnet 4 (Thinking) 3.60 6.50 38.50 52.20 62.20 73.10 Gemini 2.5 Pro 2.00 6.50 33.50 44.60 57.40 66.30 OpenAI o3-high 5.10 9.50 37.80 50.50 57.30 68.90 Kimi K2 3.00 6.50 36.20 49.60 61.20 70.70 DeepSeek-R1 0.80 3.00 22.90 36.60 44.30 60.30 Doubao-1.6 2.50 5.50 34.00 48.90 54.60 69.70

- Doubao-1.6-non-thinking 2.10 4.50 29.70 42.70 52.80 65.10 Web2BigTable (Ours) GPT-5 mini + Gemini 3 Flash 38.50+33.40 40.00+30.50 63.53+25.03 65.12+12.92 80.12+14.42 82.48+9.38

Table 3: Performance comparison on the WideSearch benchmark. Full results in Appendix 5.

WideSearch results. Table 3 and Figure 5 present the primary results on WideSearch. Web2BigTable achieves an Avg@4 Success Rate of 38.50 (7.5× the second best at 5.10), a Row F1 of 63.53 (+25.03 over the second best), and an Item F1 of 80.12 (+14.42 over the second best). As established in Section 3.4, these gains are attributable to the framework design rather than backbone model capability. Across all three metric categories, existing multi-agent frameworks plateau below 6 SR, 39 Row F1, and 63 Item F1 despite employing strictly stronger backbone LLMs (o3-high, Claude Sonnet 4). This ceiling arises because static decomposition heuristics induce systematic coverage gaps that no amount of per-worker capability can compensate for. The gains achieved by Web2BigTable are directly attributable to the learned orchestrator skills, which replace these fixed heuristics with task-adaptive decomposition strategies acquired through the run-verify-reflect pipeline.

###### Web2BigTable (Ours) (SR 38.5)

85

80

(SR 38.5)

ItemF1(Avg@4)

75

70

65

Claude Sonnet 4 (SR 3.6)

Gemini 2.5 Pro (SR 2.0)

60

o3-high (SR 5.1)

55

Doubao-1.6 (SR 2.5)

50

30 40 50 60 70

Row F1 (Avg@4)

- Figure 5: Performance landscape on WideSearch (Avg@4). Scatter points show multi-agent systems: position encodes Row F1 (x) and Item F1 (y); interior label encodes Success Rate. Web2BigTable dominates all three metrics. Full results are in Table 3.

Web2BigTable (Ours)

###### 73.0

Minimax-M2

72.0

Kimi-Res.

69.0

Claude-4.5

66.0

DeepMiner

62.0

Gemini-2.5

56.0

WebShaper

54.6

OAgents

54.5

WebSailor

53.3

WebDancer

40.0

30 40 50 60 70 80

Accuracy

Figure 6: Accuracy on XBench-DeepSearch.

XBench-DeepSearch results. Table 4 and Figure 6 summarise the XBench-DeepSearch evaluation. Web2BigTable achieves 73.0 accuracy, surpassing all baselines including frontier proprietary systems such as Minimax-M2 and MiroFlow (both at 72.0). Rows labelled “(OpenRouter)” correspond to our own re-evaluation of the underlying models via the OpenRouter API. XBench does not disclose its official inference configuration, including context window, temperature, and decoding parameters, so the 6 to 8 point gap between these rows and the officially reported scores (for example, 64.0 versus 72.0 for Minimax-M2) most likely reflects undocumented setup differences rather than capability gaps in the underlying models. We therefore treat the officially reported numbers as the primary point of comparison. The impact of the learned orchestrator skills is evident: accuracy improves from 41.0 (without learned skills) to 73.0 following strategy learning, a gain of 32.0 points from 20 synthesised training queries. This gain mirrors the pattern observed on WideSearch, where the same run-verify-reflect pipeline drives the majority of the performance advantage. The consistency of this effect across two structurally distinct benchmarks, one emphasising breadth over hundreds of entities and the other

Model / System Accuracy Foundation Models with Tools

Minimax-M2 72.0 DeepSeek-V3.2 71.0 GLM-4.5 70.0 Minimax-M2 (OpenRouter) 64.0 DeepSeek-V3.2 (OpenRouter) 65.0 GLM-4.5 (OpenRouter) 64.0 Claude-4.5-Sonnet 66.0 Gemini-2.5-Pro 56.0 GPT-5 mini 35.0 Gemini 3 Flash 28.0

###### Deep Research System

MiroFlow (GPT-5) 72.0 Kimi-Researcher 69.0 OAgents (Claude-3-7) 54.5 Gemini DeepResearch 50.0 OpenAI DeepResearch 26.6

###### Open-Source Agentic Model

DeepMiner-32B-RL 62.0 WebShaper-32B 54.6 WebSailor-32B 53.3 AFM-32B-RL 52.0 WebDancer-QwQ 40.0

Web2BigTable (Ours) GPT-5 mini + Minimax-M2 (OpenRouter) 67.0 Web2BigTable (Ours) GPT-5 mini + DeepSeek-V3.2 (OpenRouter) 69.0 Web2BigTable (Ours) GPT-5 mini + GLM-4.5 (OpenRouter) 70.0 Web2BigTable (Ours) GPT-5 mini + Gemini 3 Flash 73.0

Table 4: Performance comparison on XBench-DeepSearch.

depth over multi-hop reasoning chains, validates the generalisability of the bi-level framework.

Case Study. To illustrate how learned decomposition strategies produce qualitatively different behaviour, we present task ws_en_006, a query requiring 534 ground-truth rows across 5 columns spanning 6 Taylor Swift concert tours over 15 years.

- 1. User Query (ws_en_006)

List every concert on Taylor Swift’s official tours from Jan 1, 2010 to May 1, 2025. Columns: Date, Concert Name, Host Country, Host City, Host Venue. Each show on its own row, in chronological order, no omissions.

###### 2. Baseline Decomposition (without learned orchestrator skills) Strategy: split-by-time-period (default LLM reasoning)

- Worker 0: 2010–2011 (all tours mixed)
- Worker 1: 2012–2013
- Worker 2: 2014–2015
- Worker 3: 2016–2018
- Worker 4: 2019–2023 ← spans 5 years; Eras Tour alone has 130+ dates
- Worker 5: 2024–2025

Failure mode: Workers handle multiple tours mixed within each time window. Worker 4 is overwhelmed by the 5-year scope. Querying “Taylor Swift concerts 2019–2023” returns noisy, unfocused results. Result: 234 / 534 rows retrieved.

###### 3. Learned Orchestrator Skill (self-evolved in Phase 1)

The task-router skill classified this query as split-by-entity. The decompose skill (decompose-split-by-entity/SKILL.md), automatically generated from training-time error analysis:

# Split-by-entity decomposition When the query targets a LIST OF NAMED ENTITIES (e.g., tours, brands, athletes), split by entity name –- NOT by time period. Each worker gets one entity as their search keyword. Rules learned from past failures:

- - If any entity has >80 expected items, split further by region.
- - Always include a gap-detection worker.
- - If >10% missing after gap-detection, trigger Round 2.

###### 4. Web2BigTable Decomposition (applying learned skills) Strategy: split-by-entity (tour name) + split-by-region for large tours

- Worker 0: Fearless Tour (2010)
- Worker 1: Speak Now World Tour (2011–2012)
- Worker 2: The Red Tour (2013–2014)
- Worker 3: The 1989 World Tour (2015)
- Worker 4: Reputation Stadium Tour (2018)
- Worker 5: The Eras Tour: North America + early 2023
- Worker 6: The Eras Tour: International 2023–2024
- Worker 7: Gap detection and completeness check Round 2: System detected incomplete Red Tour data → dispatched 2 additional verification workers.

Result: 653 raw rows retrieved across 6 tours; after deduplication and aggregation, 556 unique rows submitted. Row F1 = 93.8%.

###### 5. Per-Tour Retrieval Distribution (after deduplication)

Tour Rows Tour Rows

The Red Tour (2013–2014) 142 Fearless Tour (2010) 85 The Eras Tour (2023–2024) 118 The 1989 World Tour (2015) 73 Speak Now World Tour (2011–2012) 93 Reputation Stadium Tour (2018) 45

###### Total (deduplicated) 556

- 6. Performance Comparison

Row F1 Item F1 Rows

GPT-5 mini (single agent) 15.2% 71.3% 81 Gemini 3 Flash (single agent) 12.8% 68.5% 64

Web2BigTable (w/o skills) 26.8% 79.7% 234 Web2BigTable (full) 93.8% 96.2% 556

∆ (full vs best single agent) +78.6 +24.9 +475 ∆ (full vs w/o skills) +67.0 +16.5 +322

Key insight: GPT-5 mini and Gemini 3 Flash each retrieve fewer than 100 rows as single agents. The same two models inside Web2BigTable retrieve 556 deduplicated rows at 93.8% Row F1, confirming that the gain stems from the framework, not the underlying LLMs.

- Figure 7: Case study on task ws_en_006 (Taylor Swift concerts, 534 ground-truth rows, 6 tours). Workers collectively retrieve 653 raw rows; after deduplication and aggregation, 556 unique rows are submitted for evaluation. The auto-generated orchestrator skill selects entity-based decomposition with adaptive region splitting, achieving 93.8% Row F1 vs. 12.8%–26.8% for single-agent and skill-less baselines.

#### 4 Related Work

Autonomous Web Search and Deep Research Agents Early LLM web search focused on single-turn retrieval to mitigate hallucinations (e.g., WebGPT [15], WebGLM [14]). The paradigm subsequently shifted towards autonomous, multi-step web navigation, catalysed by benchmarks like WebArena [33] and Mind2Web [4]. Recently, this trajectory culminated in deep research systems designed for exhaustive, long-horizon investigations. Works like WebThinker [12] and Search-R1 [10] employ reinforcement learning for this purpose, while proprietary frameworks have demonstrated impressive proficiency in deep, multi-hop reasoning [3]. However, while monolithic deep research agents [9] excel at vertical reasoning, they face severe limitations in large-scale structured extraction often termed widesearch [19]. When retrieving numerous entities across heterogeneous sources, single-agent architectures are inevitably bottlenecked by context saturation, error propagation, and rigid task decomposition. This highlights the necessity for scalable frameworks that can parallelise broad extraction without sacrificing reasoning depth. While emerging systems attempt to tackle these broad web-extraction tasks, most still rely on static heuristics or computationally expensive gradient updates. Web2BigTable diverges by autonomously evolving its web-search and task-decomposition strategies through a closedloop verify-reflect cycle, entirely without parameter updates. Furthermore, the architecture dynamically discovers, synthesizes, and refines executable search skills at inference time. This provides a highly flexible, training-free alternative that overcomes static reasoning bottlenecks, seamlessly scaling to the extreme breadth of open-web information extraction.

Self-Evolving Agents An expanding body of literature investigates how LLM agents progressively enhance their capabilities through experiential learning without parameter updates [6]. Frameworks such as SAMULE [7] and EvolveR [20] distil transferable insights through multi-level reflection and closed-loop self-distillation. Concurrently, reinforcement learning is increasingly employed to construct and exploit structured skill libraries from sequential rollouts, as demonstrated by SAGE [17] and SkillRL [22]. Other approaches facilitate continual learning via artefact-centric discovery loops [1] or by decoupling reasoning from learning using hierarchical procedural memory for compositional generalisation [5].

Whilst Web2BigTable builds upon this foundation, it distinguishes itself by operating within a multi-agent paradigm. Specifically, the central orchestrator evolves macro-level decomposition strategies, whilst the parallel worker agents independently cultivate micro-level execution skills. Crucially, both evolutionary processes occur concurrently within a unified learning loop, remaining devoid of gradient updates to the underlying language models.

Agentic Memory Systems Memory serves as a foundational component distinguishing autonomous LLM agents from stateless inference [30, 21]. Recent architectures employ sophisticated memory management, such as A-MEM [23], which organises memories into interconnected knowledge networks. To manage such structures, Memory-R1 [24] utilises reinforcement learning to train dedicated managers for structured long-term memory operations, whilst MUSE [25] categorises experiential data via a hierarchical Plan-Execute-Reflect-Memorise loop. Extending these concepts to multi-agent collectives, frameworks like G-Memory [28] trace hierarchical memory across agents, enabling shared episodic-to-semantic consolidation through structured knowledge graphs. Establishing a rigorous mathematical framework for such systems, the Memento series [18] formalises memory-augmented agents through the Stateful Reflective Decision Process (SRDP), providing convergence guarantees for retrieval policies operating over an evolving memory bank. Web2BigTable inherits this theoretical foundation, extending it to a hierarchical multi-agent architecture. Within this paradigm, skill memories are maintained at two distinct strata: macrolevel decomposition strategies for the orchestrator, and micro-level executable skills for the workers. Both tiers are continuously refined via a unified Read-Write Reflective Learning loop.

#### 5 Conclusion

We presented Web2BigTable, a bi-level multi-agent framework for large-scale web-to-table construction. The system addresses the fundamental tension between breadth and reliability in agentic web search through a memory-mediated self-evolving architecture: an upper-level orchestrator that automatically learns reusable decomposition strategies from a small training split, and a lower-level pool of asynchronous workers that coordinate through a shared Markdown workboard whilst evolving their own execution skills, with all adaptation mediated through persistent, human-readable memory rather than gradient updates. Empirically, Web2BigTable achieves state-of-the-art performance on the WideSearch benchmark and generalises to depthoriented search on XBench-DeepSearch, with ablation studies confirming that the learned decomposition skills account for the largest share of the performance advantage. These results demonstrate that bi-level memory-mediated coordination, coupled with automated strategy learning, provides a scalable and training-free alternative to monolithic agent architectures for large-scale structured extraction.

#### References

- [1] Nikolas Belle, Dakota Barnes, Alfonso Amayuelas, Ivan Bercovich, Xin Eric Wang, and William Wang. Agents of change: Self-evolving llm agents for strategic planning. arXiv preprint arXiv:2506.04651, 2025.
- [2] William Brach, Francesco Zuppichini, Marco Vinciguerra, and Lorenzo Padoan. Scrapegraphai-100k: A large-scale dataset for llm-based web information extraction. arXiv preprint arXiv:2602.15189, 2026.
- [3] Kaiyuan Chen, Yixin Ren, Yang Liu, et al. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations, 2025.

- [4] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:64335–64366, 2023.
- [5] Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, and Mahdi Jalili. Learning hierarchical procedural memory for LLM agents through Bayesian selection and contrastive refinement. arXiv preprint arXiv:2512.18950, 2025. URL https://arxiv.org/abs/2512. 18950.
- [6] Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 2025.
- [7] Yubin Ge, Salvatore Romeo, Jason Cai, Monica Sunkara, and Yi Zhang. SAMULE: Selflearning agents enhanced by multi-level reflection. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025.
- [8] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In The Thirteenth International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2408.08435.
- [9] Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al. Deep research agents: A systematic examination and roadmap. arXiv preprint arXiv:2506.18096, 2025.
- [10] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [11] Ka Yiu Lee, Yuxuan Huang, Zhiyuan He, Huichi Zhou, Weilin Luo, Kun Shao, Meng Fang, and Jun Wang. Infoseeker: A scalable hierarchical parallel agent framework for web information seeking. arXiv preprint arXiv:2604.02971, 2026.
- [12] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. WebThinker: Empowering large reasoning models with deep research capability. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2504.21776.
- [13] Shuodi Liu, Yingzhuo Liu, Zi Wang, Yusheng Wang, Huijia Wu, Liuyu Xiang, and Zhaofeng He. Select-then-decompose: Adaptive selection strategy for task decomposition in large language models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025. arXiv:2510.17922.
- [14] Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan Zuo, Yuxiao Dong, and Jie Tang. Webglm: Towards an efficient web-enhanced question answering system with human preferences. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4549–4560, 2023.
- [15] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

- [16] Chen Qian, Zihao Xie, YiFei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Scaling large language model-based multi-agent collaboration. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. arXiv:2406.07155.
- [17] Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. Reinforcement learning for selfimproving agent with skill library. arXiv preprint arXiv:2512.17102, 2025. URL https: //arxiv.org/abs/2512.17102.
- [18] Jun Wang. Memento-ii: Learning by stateful reflective memory. arXiv preprint arXiv:2512.22716, 2025.
- [19] Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, Wenhao Huang, Yang Wang, and Ke Wang. WideSearch: Benchmarking agentic broad info-seeking. arXiv preprint arXiv:2508.07999, 2025.
- [20] Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, and Botian Shi. EvolveR: Self-evolving LLM agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025.
- [21] Shanglin Wu et al. Memory in LLM-based multi-agent systems: Mechanisms, challenges, and collective intelligence. TechRxiv preprint, 2025. doi: 10.36227/techrxiv.176539617.79044553. URL https://www.techrxiv.org/doi/full/10.36227/techrxiv.176539617.79044553/ v1.
- [22] Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. SkillRL: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026. URL https://arxiv.org/abs/2602.08234.
- [23] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2502.12110.
- [24] Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schütze, Volker Tresp, and Yunpu Ma. Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025. URL https://arxiv.org/ abs/2508.19828.
- [25] Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, et al. Learning on the job: An experience-driven self-evolving agent for long-horizon tasks. arXiv preprint arXiv:2510.08002, 2025.
- [26] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/abs/2210.03629.
- [27] Geunbin Yu. Adaptorch: Task-adaptive multi-agent orchestration in the era of llm performance convergence. arXiv preprint arXiv:2602.16873, 2026.
- [28] Guibin Zhang et al. G-Memory: Tracing hierarchical memory for multi-agent systems. arXiv preprint arXiv:2506.07398, 2025. URL https://arxiv.org/abs/2506.07398.

- [29] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In The Thirteenth International Conference on Learning Representations (ICLR), 2025. URL https://arxiv. org/abs/2410.10762.
- [30] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):155, 2025. doi: 10.1145/3748302. URL https://dl.acm.org/doi/10.1145/3748302.
- [31] Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, and Jun Wang. Memento: Fine-tuning LLM agents without fine-tuning LLMs. Preprint, 2025.
- [32] Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen, Jinsong Li, et al. Memento-skills: Let agents design agents. arXiv preprint arXiv:2603.18743, 2026.
- [33] Shuyan Zhou, Frank F Hou, Xingyao Cheng, Hongyuan Mao, Bowen Peng, Shengjia Zhong, Yihao Tai, Edward Corcodel, Rui Zhang, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2023.

#### A Theoretical Extension: Memento-Team

The bilevel game formulation presented in this work provides an intuitive framework for understanding the dual-memory dynamics of Web2BigTable. In our ongoing work, Memento-Team: Game-Theoretic Multi-Agent LLM Systems by Bi-level Coordinated Reflection, we develop a rigorous theoretical generalisation of this framework. Memento-Team formalises the orchestrator–worker interaction as a full Stackelberg game with frozen LLM agents, where the strategy memory and execution memory serve as the sole decision variables. At the lower level, the workers’ subgame is further characterised as a discrete exact potential game: under additive utility decomposition, any worker’s unilateral memory update that improves its local utility is guaranteed to improve the global objective, ensuring that asynchronous updates on the shared workboard converge without conflict. Building on this formulation, we introduce Stochastic Reflective Memory Ascent (SRMA), a framework that connects the discrete process of natural language reflection to continuous stochastic optimisation over memory spaces. Under assumptions of bounded communication delay and sparse memory contention, we establish almost sure convergence of the multi-agent system to a neighbourhood of a bilevel memory equilibrium, with the convergence radius governed by the intrinsic hallucination noise of the underlying LLMs. Memento-Team generalises beyond the wide search setting, providing convergence guarantees and design principles applicable to arbitrary multi-agent LLM systems that coordinate through shared non-parametric memory. We plan to validate this generalised framework across a wider spectrum of scenarios, ranging from simple synthetic tasks to complex, practical applications.

- B Detailed Results
- C Case Study

- Case B. This task requires compiling a comprehensive table of every AMD processor with Zen-based architecture released between 2014 and 2024, with 12 columns (Time, Product Series, Processor Model, Core Architecture, Manufacturing Process, Cores, Threads, Core Frequency, L2 Cache, L3 Cache, Graphics Model, Number of Graphics Cores) and 331 ground-truth rows. The high column count makes Item F1 particularly sensitive to per-cell accuracy.

- 1. User Query (ws_en_091)

List all AMD processors with Zen architecture released from Lisa Su becoming CEO (2014) to 2024 inclusive. Columns: Time, Product Series, Processor Model, Core Architecture, Manufacturing Process (nm), Cores, Threads, Core Frequency (GHz), L2 Cache (MB), L3 Cache (MB), Graphics Model, Number of Graphics Cores. Output “NA” if information cannot be found.

- 2. Baseline Decomposition (without learned orchestrator skills)

Strategy: split-by-time-period (default LLM reasoning)

The orchestrator divided the 10-year span into coarse time windows, mixing product lines (Ryzen Desktop, EPYC Server, Threadripper, Mobile, Embedded, PRO) within each window. Workers queried broad terms such as “AMD Zen processors 2019–2021”, returning incomplete and inconsistent results across the 12 required columns.

Result: 137 rows retrieved. Row F1 = 18%, Item F1 = 32%.

Failure: The 12-column requirement amplifies errors: even rows that are partially retrieved have many NA or incorrect cells across cache sizes, graphics models, and manufacturing process fields.

- 3. Learned Orchestrator Skill (auto-generated in Phase 1)

The task-router classified this query as split-by-category. The decompose skill partitions by product line, not by time:

# Split-by-category decomposition When the query covers MULTIPLE PRODUCT LINES across a long time span, split by product category –- NOT by year. Each worker specialises in one product line across all years. Rules learned from past failures:

- - Assign one worker per major product line (Desktop, Server, Mobile, Workstation, Embedded, PRO).
- - For product lines with >50 SKUs, split further by generation.
- - Always assign a dedicated worker for cache/GPU spec lookup.

- 4. Web2BigTable Decomposition (applying learned skills) Strategy: split-by-category (product line) + split-by-generation for large lines

- Worker 0: Ryzen Desktop (all generations)
- Worker 1: EPYC Server (Zen 1–2)
- Worker 2: EPYC Server (Zen 3–5)
- Worker 3: Ryzen Mobile (U/H/HS/HX series)
- Worker 4: Threadripper + Threadripper PRO
- Worker 5: Ryzen PRO (Desktop + Mobile)
- Worker 6: Athlon + Embedded series
- Worker 7: Spec verification (cache, GPU cores)

###### Result: ∼350 raw rows retrieved across 8 product lines; after deduplication, ∼334 unique rows submitted. Row F1 = 89%, Item F1 = 96%.

- 5. Performance Comparison

Row F1 Item F1 Rows

GPT-5 mini (single agent) 10.5% 25.3% 48 Gemini 3 Flash (single agent) 8.2% 21.7% 35

Web2BigTable (w/o skills) 18% 32% 137 Web2BigTable (full) 89% 96% 334

∆ (full vs best single agent) +78.5 +70.7 +286 ∆ (full vs w/o skills) +71.0 +64.0 +197

Note: Single-agent scores are for this specific task only, not the benchmark-wide averages in Table 3. Key insight: With 12 columns per row, single agents retrieve fewer than 50 rows and fill barely 20% of cells correctly. The learned product-line decomposition enables each worker to query authoritative sources (e.g., AMD product pages, WikiChip) for its specific category, whilst a dedicated spec-verification worker cross-checks cache and GPU fields, achieving 96% Item F1.

- Figure 8: Case B: task ws_en_091 (AMD Zen processors, 331 ground-truth rows, 12 columns). Workers collectively retrieve ∼350 raw rows; after deduplication, ∼334 unique rows are submitted. Single agents retrieve fewer than 50 rows with Item F1 below 26%. Web2BigTable applies a learned product-line decomposition with dedicated spec-verification workers, achieving 89% Row F1 and 96% Item F1.

- Case C. This task requires compiling research papers from two distinct source organisations (ByteDance Seed team and DeepSeek) over a 30-month window, with cross-source temporal verification against arXiv. The defining challenge is twofold: (i) the two sources publish on entirely separate web platforms with different formats, and (ii) when the same paper is mirrored, the canonical date must be reconciled across publisher pages and arXiv submission records.

1. User Query (ws_zh_069) Compile all large-model-related papers published by the ByteDance Seed team and DeepSeek between

- 1 January 2023 and 30 June 2025. Search the official websites of both organisations (any paper with Seed-team participation counts). For each paper, include the publication date (yyyy-mm-dd), title, and primary authors. If two records refer to the same paper, the canonical date is the arXiv first-submit timestamp. Output a single Markdown table with columns: Organisation, Publication Date, Paper Title, Authors.

- 2. Baseline Decomposition (without learned orchestrator skills)

###### Strategy: split-by-time-period (default LLM reasoning)

The orchestrator divided the 30-month window into five 6-month chunks and dispatched workers without distinguishing source organisations. Workers issued generic queries such as “ByteDance Seed papers 2024 H1” and “DeepSeek papers 2024 H2”, returning fragmented results from thirdparty aggregators (e.g., arXiv listings) rather than the canonical source pages. The smaller-volume DeepSeek catalogue was diluted across all five workers, and arXiv-vs-publisher date conflicts were never reconciled.

Result: 67 rows retrieved. Row F1 = 41%, Item F1 = 58%.

Failure modes: (i) the Seed team’s ∼120 papers were sparsely sampled because each time-window worker treated Seed as just one of many possible sources; (ii) DeepSeek’s smaller catalogue lost coverage entirely in early time windows where it had no releases; (iii) date inconsistencies between Seed homepage listings and arXiv records were left unresolved, contaminating the Publication Date column.

- 3. Learned Orchestrator Skill (auto-generated in Phase 1)

The task-router classified this query as split-by-source. The decompose skill partitions by source organisation, with a dedicated worker for cross-source temporal verification:

# Split-by-source decomposition When the query enumerates records from MULTIPLE NAMED SOURCES (e.g., specific organisations, repositories, or publishers), assign one worker per source rather than splitting along time or topic. Rules learned from past failures:

- - One worker per named source; never mix sources within a single worker.
- - For sources with >50 records, split further by year within that source.
- - If the query specifies a canonical reconciliation rule (e.g., “arXiv date takes precedence”), assign a dedicated verification worker that cross-references the field across sources.

- 4. Web2BigTable Decomposition (applying learned skills)

Strategy: split-by-source + split-by-year for the larger source + dedicated verification worker

- Worker 0: Seed team: 2023 (full year)
- Worker 1: Seed team: 2024 (Jan–Jun)
- Worker 2: Seed team: 2024 (Jul–Dec)
- Worker 3: Seed team: 2025 (Jan–Jun)
- Worker 4: DeepSeek: entire 30-month window
- Worker 5: arXiv cross-source date verification
- Worker 6: Author-list normalisation and duplicate detection

Round 2: The verification worker flagged 8 rows where the publisher-listed date diverged from arXiv submission by more than 30 days; the orchestrator dispatched a follow-up worker to reconcile these specific entries against arXiv before final aggregation.

Result: 134 rows retrieved. Row F1 = 91%, Item F1 = 94%.

- 5. Performance Comparison

Row F1 Item F1 Rows

GPT-5 mini (single agent) 22.4% 39.6% 29 Gemini 3 Flash (single agent) 18.7% 34.2% 23

Web2BigTable (w/o skills) 41% 58% 67 Web2BigTable (full) 91% 94% 134

∆ (full vs best single agent) +68.6 +54.4 +105 ∆ (full vs w/o skills) +50.0 +36.0 +67

Key insight: The two sources have asymmetric catalogue sizes (∼120 vs ∼10). Single agents retrieve fewer than 30 rows, missing the majority of Seed papers and most DeepSeek entries entirely. Source-based decomposition assigns dedicated workers per organisation, whilst a verification worker resolves arXiv-vs-publisher date conflicts that all other configurations silently ignore.

- Figure 9: Case C: task ws_zh_069 (LLM papers from ByteDance Seed and DeepSeek, ∼130 ground-truth rows across two asymmetric sources). Single agents retrieve fewer than 30 rows with Item F1 below 40%. Web2BigTable applies a learned source-based decomposition with a dedicated arXiv verification worker, achieving 91% Row F1 and 94% Item F1.

Table 5: Detailed experiments results on the WideSearch benchmark.

###### Model / System Success Rate Row F1 Item F1

Avg@4 Max@4 Avg@4 Max@4 Avg@4 Max@4

Single Agent on WideSearch-zh Claude Sonnet 4 (Thinking) 0.25 1.00 30.19 39.73 53.76 63.19

- Gemini 2.5 Pro 1.00 3.00 26.95 36.96 45.57 57.26 OpenAI o3-high 2.00 5.00 29.30 39.31 45.19 54.46

- K2 0.25 1.00 27.79 39.03 48.81 59.64 DeepSeek-R1-0528 0.25 1.00 18.44 28.35 33.95 47.83

- Doubao-1.6 1.75 4.00 29.25 42.08 43.72 58.84

- Doubao-1.6-non-thinking 0.50 2.00 25.56 37.41 42.87 55.79

Single Agent on WideSearch-en

Claude Sonnet 4 (Thinking) 4.25 9.00 33.18 44.08 62.02 70.27 Gemini 2.5 Pro 2.00 7.00 33.05 45.82 56.38 69.97 OpenAI o3-high 7.00 13.00 38.70 48.84 60.03 70.08 K2 2.00 6.00 31.54 43.68 59.91 70.52 DeepSeek-R1-0528 0.50 2.00 22.88 35.03 48.58 62.36 Doubao-1.6 3.50 6.00 30.56 46.16 52.82 68.88

- Doubao-1.6-non-thinking 1.50 5.00 28.86 42.31 55.06 68.17

Multi-Agent Framework on WideSearch-zh

Claude Sonnet 4 (Thinking) 2.75 6.00 36.85 51.46 57.13 69.53 Gemini 2.5 Pro 1.00 4.00 30.93 42.21 51.79 60.87 OpenAI o3-high 2.75 6.00 33.83 47.85 50.35 63.06 K2 1.25 3.00 34.74 48.01 56.86 66.75

- DeepSeek-R1-0528 0.50 2.00 21.17 35.08 37.66 53.15

Doubao-1.6 2.25 6.00 32.83 47.49 48.79 64.43 Doubao-1.6-non-thinking 0.50 1.00 26.93 40.30 46.52 59.63 Web2BigTable (Ours) 31.00 32.00 60.32 62.96 76.01 79.33

Multi-Agent Framework on WideSearch-en

Claude Sonnet 4 (Thinking) 4.50 7.00 40.13 52.91 67.21 76.72 Gemini 2.5 Pro 3.00 9.00 36.00 47.06 63.06 71.75 OpenAI o3-high 7.50 13.00 41.78 53.20 64.27 74.80 K2 4.75 10.00 37.71 51.20 65.44 74.68

- DeepSeek-R1-0528 1.00 4.00 24.57 38.10 50.91 67.54 Doubao-1.6 2.75 5.00 35.14 50.38 60.39 74.87 Doubao-1.6-non-thinking 3.75 8.00 32.38 44.99 58.97 70.62 Web2BigTable (Ours) 46.00 48.00 66.74 67.28 84.23 85.63

End-to-End Systems on WideSearch-zh

Claude 0.00 0.00 20.84 28.92 43.51 52.14 Gemini 1.50 4.00 32.32 40.52 52.92 60.44

- OpenAI o3 3.00 5.00 27.40 38.34 46.03 56.51

End-to-End Systems on WideSearch-en

Claude 5.00 10.00 27.39 38.07 53.29 64.81 Gemini 7.00 12.00 40.95 50.29 65.18 73.90

- OpenAI o3 3.00 6.00 20.42 33.72 45.02 56.47

