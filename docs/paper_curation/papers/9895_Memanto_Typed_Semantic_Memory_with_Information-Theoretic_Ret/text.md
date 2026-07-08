# Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents

## arXiv:2604.22085v1[cs.AI]23Apr2026

Seyed Moein Abtahi

Moorcheh AI EdgeAI Innovations seyedmoein.abtahi@ontariotechu.net

Rasa Rahnema

Moorcheh AI EdgeAI Innovations ryrahnem@uwaterloo.ca

Hetkumar Patel

Moorcheh AI EdgeAI Innovations hetkumardineshbhai.patel@sheridancollege.ca

Neel Patel

Moorcheh AI EdgeAI Innovations neel@edgeaiinnovations.com

Majid Fekri

Moorcheh AI EdgeAI Innovations majid.fekri@edgeaiinnovations.com

Tara Khani

Moorcheh AI EdgeAI Innovations tara.khani@edgeaiinnovations.com

Abstract—The transition from stateless language model inference to persistent, multi session autonomous agents has revealed memory to be a primary architectural bottleneck in the deployment of production grade agentic systems. Existing methodologies largely depend on hybrid semantic graph architectures, which impose substantial computational overhead during both ingestion and retrieval. These systems typically require large language model mediated entity extraction, explicit graph schema maintenance, and multi query retrieval pipelines. This paper introduces Memanto, a universal memory layer for agentic artificial intelligence that challenges the prevailing assumption that knowledge graph complexity is necessary to achieve high fidelity agent memory. Memanto integrates a typed semantic memory schema comprising thirteen predefined memory categories, an automated conflict resolution mechanism, and temporal versioning. These components are enabled by Moorcheh’s Information Theoretic Search engine, a no indexing semantic database that provides deterministic retrieval within sub ninety millisecond latency while eliminating ingestion delay. Through systematic benchmarking on the LongMemEval and LoCoMo evaluation suites, Memanto achieves state of the art accuracy scores of 89.8 percent and 87.1 percent respectively. These results surpass all evaluated hybrid graph and vector based systems while requiring only a single retrieval query, incurring no ingestion cost, and maintaining substantially lower operational complexity. A five stage progressive ablation study is presented to quantify the contribution of each architectural component, followed by a discussion of the implications for scalable deployment of agentic memory systems.

Index Terms—Agentic AI, long term memory, retrieval augmented generation, information theory, vector search, semantic memory, large language models, autonomous agents

I. INTRODUCTION

Memory has become a fundamental architectural component in the design of foundation model based agents [1], [2]. As large language models transition from single turn question answering systems to autonomous agents capable of multi step reasoning, tool utilization, and long horizon task execution [3], [4], their inherent limitation, namely the lack of persistent state

across sessions, has emerged as a central engineering challenge in the development of agentic artificial intelligence systems.

Industry projections indicate that the agentic artificial intelligence market will expand from 7.8 billion dollars to more than 52 billion dollars by 2030. Gartner further estimates that 40 percent of enterprise applications will incorporate AI agents by the end of 2026, compared to less than 5 percent in 2025. This accelerated adoption has generated a critical need for memory infrastructure that satisfies production requirements, including high accuracy, low latency, cost efficiency, and reduced operational complexity.

Current approaches within the field remain heterogeneous and increasingly complex. A range of frameworks, including Mem0 [5], Zep [6], Letta [7], and A-MEM [8], propose architectures that integrate knowledge graphs, temporal graph databases, multi query retrieval strategies, and recursive large language model driven ingestion pipelines. Although these systems demonstrate competitive benchmark performance, they introduce substantial computational and operational overhead. This paper characterizes this phenomenon as the “Memory Tax”, defined as the cumulative increase in compute cost, latency, and system complexity associated with memory ingestion and retrieval processes.

Memanto, built upon Moorcheh’s Information Theoretic Search engine, a no indexing semantic database based on Information Theoretic Vector Compression, demonstrates that highly optimized semantic retrieval combined with structured memory typing and automated conflict resolution can achieve and surpass the performance of hybrid graph and vector architectures. This is accomplished while eliminating ingestion overhead, reducing retrieval to a single query, and removing the need for schema management. Fig. 1 contextualizes Memanto within the broader evolution of memory systems for agentic artificial intelligence.

Zep Graph KG

Supermem. Graph+Vec

###### Accuracy: 89.8%

Mem0 Vec+Graph

Memanto ITS · Typed Binary

Letta (MemGPT) Hierarchical

A-MEM Zettelkasten

2024 2025 2026

###### LLM context window: 4K → 128K → 1M+ tokens

Hybrid G+Vec Other

#### Year

- Fig. 1. Evolution of agentic memory systems (2023 to 2026). Despite rapidly growing LLM context windows, architectures have grown more complex. Memanto (2026) achieves SOTA accuracy using a simpler, vector only approach that eliminates graph infrastructure and LLM mediated ingestion entirely.

A. Contributions

- 1) Architectural: We introduce a production grade agentic memory system that achieves state of the art benchmark performance using a vector based architecture with zero cost ingestion. The system incorporates a typed memory schema consisting of thirteen semantic categories and an integrated conflict resolution mechanism, without reliance on knowledge graphs, multi query retrieval, or large language model mediated ingestion.
- 2) Empirical: We present a five stage progressive ablation study conducted on LongMemEval [9] and LoCoMo [10]. The study evaluates the effects of retrieval limit tuning, similarity threshold calibration, prompt design, and inference model selection. Final results of 89.8 percent and 87.1 percent respectively establish a new state of the art among vector based systems.
- 3) Systems: We formally analyze and quantify the Memory Tax associated with hybrid graph and vector architectures. Our results demonstrate that the additional overhead yields diminishing performance gains when compared to optimized semantic retrieval, particularly under deterministic exact match search rather than approximate nearest neighbor methods.
- 4) Design Principles: We propose six design principles for production ready agentic memory systems, derived from systematic evaluation of agent requirements and informed by feedback from real world deployments.

II. BACKGROUND AND RELATED WORK

This section situates the proposed approach within the broader intellectual and systems landscape of agentic memory. We begin with cognitive foundations that inform modern memory abstractions, followed by recent taxonomies that attempt to organize the rapidly evolving design space. We then examine dominant architectural paradigms, with particular emphasis on hybrid graph based systems, before analyzing the indexing and ingestion bottlenecks that constrain current implementations.

Finally, we review the principal evaluation benchmarks used to assess long term memory in agentic systems.

- A. Cognitive Foundations of Memory Taxonomy

Cognitive science provides a principled framework for structuring memory in artificial agents. Tulving’s foundational work [11] distinguishes between episodic memory, which captures event specific and temporally situated experiences, semantic memory, which encodes general knowledge and factual information, and procedural memory, which governs skills and behavioral rules. These distinctions have become directly relevant to the design of memory systems for large language model based agents.

Baddeley’s working memory model [12], consisting of the phonological loop, visuospatial sketchpad, and central executive, exhibits a strong conceptual correspondence with modern retrieval augmented generation architectures. In this mapping, the phonological loop aligns with in context token buffers, the visuospatial sketchpad with structured retrieval representations, and the central executive with the reasoning and control mechanisms of the agent.

Recent work has emphasized the importance of episodic memory for long horizon agent behavior. MacPherson et al. [13] argue that episodic representations enable temporal specificity and contextual binding that cannot be achieved through semantic retrieval alone. ENGRAM [14] operationalizes this insight by implementing three distinct memory types with a unified routing and retrieval mechanism, demonstrating that typed memory separation significantly improves performance on both LOCOMO and LONGMEMEVAL. Memanto extends this principle through a more granular schema comprising thirteen memory categories.

- B. Memory Surveys and Taxonomies (2024 to 2026)

A growing body of survey literature seeks to impose structure on the increasingly heterogeneous memory landscape. Zhang et al. [1] categorize memory systems along three dimensions: forms, functions, and dynamics, identifying token level,

parametric, and latent memory as primary representations. Abou Ali et al. [15] propose a dual paradigm framework that distinguishes symbolic or classical approaches from neural or generative paradigms.

Arunkumar et al. [2] describe a four layer architecture encompassing perception, memory, the agent core, and action. Nisa et al. [16] position memory as the substrate enabling coherent reasoning and planning across time, while Wang et al. [4] identify long term memory as a central unresolved challenge in large language model based agents. Sumers et al. [17] further formalize the correspondence between cognitive architectures and agent system design, mapping perception, memory, learning, and decision making to computational components.

C. Knowledge Graph Based Memory Systems (2024 to 2026)

The dominant paradigm for production grade agent memory systems has converged on hybrid architectures that integrate dense vector representations with structured knowledge graphs.

MemGPT / Letta [7] introduces a virtual memory abstraction inspired by operating systems, in which information is dynamically paged between context and external storage. While conceptually influential, this approach relies on recursive summarization and hierarchical compression, which can introduce latency variability and loss of information fidelity, particularly when precise textual recall is required.

Mem0 [5] implements a three tier memory hierarchy spanning user, session, and agent scopes. The system combines vector retrieval, graph based relational storage, and key value indexing. Although it demonstrates strong empirical performance, its ablation results indicate that the graph augmented variant yields only marginal improvements over the base vector configuration. This raises questions regarding the necessity of graph infrastructure relative to its associated computational and operational costs.

In practice, each memory insertion in the graph augmented configuration triggers a synchronous multi stage pipeline consisting of large language model driven entity extraction, vector embedding and index updates, and graph synchronization. This process transforms low latency write operations into multi second procedures, thereby exemplifying the accumulation of computational overhead associated with complex memory architectures.

Zep / Graphiti [6] extends the graph based paradigm by incorporating temporal versioning and bi temporal indexing to support enterprise grade auditing and compliance requirements. However, the reliance on synchronous extraction pipelines introduces ingestion latency, delaying the availability of newly stored information for retrieval.

A-MEM [8] adopts a Zettelkasten inspired design in which memories are represented as interconnected notes enriched with contextual metadata. While this enables associative retrieval, it requires a full inference step for each memory insertion, increasing both latency and cost.

Hindsight [18] and subsequent reflective memory frameworks [19] achieve high benchmark accuracy through multi stage retrieval and reflection mechanisms. These systems rely on parallel queries and iterative reasoning passes, resulting in significantly higher system complexity relative to single query retrieval approaches.

Emerging evidence challenges the necessity of such architectural complexity. Merrill et al. [20] demonstrate that comparatively simple retrieval based systems can outperform more elaborate memory hierarchies on existing benchmarks, suggesting that current evaluation protocols may not fully capture the benefits of structured memory organization.

- D. The Indexing and Ingestion Bottleneck

Retrieval augmented generation [21] establishes the canonical paradigm for augmenting language models with external memory. Subsequent systems extend this framework through hierarchical memory abstractions and hybrid storage mechanisms. However, traditional vector databases rely on approximate nearest neighbor indexing structures such as hierarchical navigable small world graphs [22], which introduce non negligible delays between data ingestion and query availability.

For agentic systems operating in interactive or iterative settings, this delay is problematic. An agent may need to store information and immediately retrieve it within the same reasoning trajectory. Any latency in indexing directly impairs this capability.

LONGMEMEVAL [9] provides a structured analysis of memory system design, decomposing performance into indexing, retrieval, and reading stages. The study identifies key factors including granularity of stored information, key construction, query formulation, and reading strategies. Empirical results demonstrate that fine grained session decomposition, enriched key representations, temporally aware query expansion, and structured reading techniques substantially improve accuracy.

Additional work highlights limitations in long context processing. Liu et al. [23] identify a degradation effect in which models exhibit reduced accuracy for information located in the middle portions of extended contexts. This finding reinforces the importance of targeted retrieval mechanisms that prioritize relevance over raw context length.

Alternative architectures such as HippoRAG [24] and RAPTOR [25] address long range dependencies through hierarchical or graph based representations, but introduce additional system complexity. REPLUG [26] demonstrates that combining high recall retrieval with post retrieval verification improves robustness, aligning with the principle of prioritizing recall in memory systems.

- E. Evaluation Benchmarks

LONGMEMEVAL [9] is a large scale benchmark comprising 500 curated questions spanning six categories, including user specific information, assistant responses, preferences, knowledge updates, temporal reasoning, and multi session interactions. The dataset is embedded within extended dialogues that can scale to over one million tokens across hundreds of

sessions, providing a comprehensive test of long term memory capabilities.

LOCOMO [10] consists of long form multi session dialogues with diverse reasoning requirements, including single hop, multi hop, open domain, and temporal queries. It serves as a complementary benchmark emphasizing conversational continuity and reasoning depth.

Additional benchmarks, including MemoryBank [27], PerLTQA [28], DialSim [29], MemoryAgentBench [30], and long context evaluation frameworks [31], further expand the evaluation landscape. However, recent analyses suggest that as model context windows increase, benchmark performance increasingly reflects underlying language model reasoning capabilities rather than the quality of the memory architecture itself, motivating the development of more targeted evaluation protocols.

III. DESIGN PRINCIPLES AND ARCHITECTURE

Before describing Memanto’s architecture, we articulate the six design principles that guided its development. These principles emerged from systematic analysis of agent operational requirements, including structured feedback from production AI agent deployments, and directly from the failure modes documented in the benchmark literature.

A contributing source was a structured dialogue with Claude (Anthropic) [32], in which the model was asked to articulate the limitations of its own memory architecture. The model identified passive context injection as the root failure mode, and independently surfaced seven specific gaps: the inability to query memory by relevance, the absence of temporal decay signals, the lack of confidence and provenance tagging, the flattening of episodic, semantic, and procedural memory into a single undifferentiated store, the absence of contradiction handling and versioning, the lack of scope and permissioning, and the absence of human-readable audit logs. This self-diagnosis maps directly onto desiderata D1 through D6, and informed both the framing and prioritization of Memanto’s architectural requirements. The use of a frontier language model as a requirements elicitation source reflects a broader methodological principle: the systems most qualified to identify the failure modes of agent memory are the agents themselves. For more information, please visit Moorcheh.ai1 and Memanto.ai2.

A. Six Desiderata for Production Agentic Memory

D1. Queryable, not injectable. Agents need the ability to query memory based on relevance to the current task, not receive a static blob of context injected at conversation start. The distinction is between providing an agent with a preassembled dossier versus giving it a librarian it can consult on demand. Static injection fails when the injected context exceeds the context window, contains irrelevant content, or misses recently stored facts that are not yet in the injected snapshot.

- 1https://moorcheh.ai
- 2https://memanto.ai

- D2. Temporally aware with decay. Not all memories should have equal weight. A deadline mentioned yesterday carries different urgency than a preference stated six months ago. Memory must support temporal queries, versioning, and relevance decay signals that agents can reason about. This requirement maps directly to the Knowledge Update and Temporal Reasoning categories in LongMemEval [9], where systems without temporal awareness perform significantly below average.
- D3. Confidence and provenance tracking. A production memory system must distinguish between explicitly stated facts, inferred patterns, and potentially outdated information. Memory entries should carry provenance metadata that agents use to calibrate their confidence in retrieved context and avoid overconfident assertions on stale data.
- D4. Typed and hierarchical. Episodic memory (e.g., in our November conversation the user discussed X), semantic memory (e.g., the user is building a vector database startup), and procedural memory (e.g., when asked for reports the user prefers this format) serve fundamentally different retrieval purposes [11] and should be stored and queried with appropriate type semantics.
- D5. Contradiction aware. When new information contradicts existing memory, the system must flag the conflict rather than silently overwrite or create inconsistency. For long-running agents, unresolved contradictions accumulate into what we term constraint drift, a gradual erosion of the coherence of the agent’s world model. MemoryAgentBench [30] confirms that conflict resolution represents one of the most significant unsolved challenges, with all evaluated methods failing on multi-hop conflict scenarios.
- D6. Zero overhead ingestion. Every millisecond of ingestion latency is a millisecond where the agent cannot access its own recent experience. For real-time agentic workflows, memory must be available for retrieval at write time, with no indexing delay, no mandatory LLM extraction step, and no graph construction bottleneck.

Fig. 2 visualises coverage across systems, while Table I summarises it quantitatively.

TABLE I DESIDERATA COVERAGE ACROSS AGENTIC MEMORY SYSTEMS

###### System D1 D2 D3 D4 D5 D6

Memanto ✓ ✓ ✓ ✓ ✓ ✓ Mem0 [5] ✓ ✓ ∼ ∼ ✗ ✗ Zep [6] ✓ ✓ ∼ ∼ ✗ ✗ Letta [7] ✓ ✗ ✗ ✗ ✗ ∼

✓ Full support ∼ Partial ✗ Not supported

B. System Overview

Memanto is designed as a local agentic platform that operates as a persistent FastAPI service, functioning as a dedicated memory agent in support of other AI agents. The system exposes three primary endpoints:

• /remember: Commits items to memory with automatic typing, tagging, timestamping, conflict detection, and optional namespace scoping.

D6 Ingest D1 Query

| | |
|---|---|
|Mid Low| |

D2 Temporal

D5 Conflict

High

Memanto Mem0 Zep Letta

D4 Typed D3 Provenance

- Fig. 2. Desiderata coverage radar (D1 to D6). Memanto achieves full coverage across all six production desiderata. Key gaps: Zep and Mem0 are weak on D5–D6 (no conflict detection, high ingestion cost); Letta is weak on D2–D5; ENGRAM lacks D3, D5, and D6; A-MEM lacks D5–D6 with only partial D2–D4.

- • /recall: Retrieves items from memory through Moorcheh’s ITS-powered semantic search with configurable similarity thresholds and retrieval limits.
- • /answer: Performs full Retrieval-Augmented Generation (RAG) with LLM intelligence applied on top of retrieved memory context.

- Fig. 3 presents the Memanto Frontend Architecture, which

comprises two layers. (1) Agent Ecosystem. IDE integrations, agent CLIs, custom agents built with Python, JavaScript, or LangChain, and a local web dashboard communicate with the gateway via CLI commands, REST API requests, and status controls. (2) Memanto Gateway. The gateway receives all incoming requests and routes them through two internal components: the Memanto CLI Engine, which handles commandline interactions, and the Memanto FastAPI Server, which serves REST API and dashboard traffic. All outbound calls from the gateway are forwarded to the backend shared services layer.

- Fig. 4 presents the Memanto Backend Architecture, which

also comprises two layers. (1) Shared Services. The gateway routes requests to nine internal services: Daily Summary, Conflict Resolution, Answer, Recall, Remember, Agent Manager, Session and Authentication Manager, Memory Sync via MEMORY.MD injection, and Tool Connect. The first six services communicate directly with the Moorcheh cloud layer via SDK calls, while the remaining three operate internally. (2) Moorcheh.ai Cloud Layer. All storage and retrieval pass through the Moorcheh Engine API to three components: a zero-indexing semantic database, an agent-optimized RAG pipeline, and native LLM access.

C. The Moorcheh Foundation: Information Theoretic Search

Memanto’s retrieval capabilities are built on Moorcheh’s semantic engine, which departs fundamentally from the HNSW plus cosine distance paradigm that dominates traditional vector

databases. The engine employs three core algorithmic innovations:

Maximally Informative Binarization (MIB): Compresses high-dimensional floating-point embedding vectors into compact binary representations while preserving the informationtheoretic content relevant to retrieval. This achieves 32× compression with no measurable loss in retrieval-relevant signal.

Efficient Distance Metric (EDM): Replaces cosine similarity with an information-theoretic distance measure that scores memory chunks by their ability to reduce uncertainty in the query context, rather than by surface-level geometric proximity in embedding space.

Information Theoretic Score (ITS): A universal relevance score on a normalized [0,1] scale that quantifies the decisiontheoretic value of each retrieved chunk for the current query. ITS enables deterministic, threshold-based retrieval: the same query always produces the same results, a critical property for reproducible agent behavior in regulated environments.

Together, these innovations eliminate the need for index construction, enabling instant write-to-search availability. The Moorcheh engine has been independently validated on the MAIR (Massive AI Retrieval) benchmark, achieving 64– 74% NDCG@10 with 9.6ms distance calculation latency (compared to 37–86ms for PGVector and Qdrant), sustaining 2,000+ queries per second with zero accuracy degradation, and delivering a 6.6× end-to-end speedup versus Pinecone plus Cohere reranking pipelines [33].

- D. Typed Memory Schema

Memanto implements a typed memory schema with 13 built-in semantic categories, each carrying distinct retrieval semantics and priority weighting. This design is directly motivated by the cognitive science literature on memory type distinctions [11] and the empirical findings of ENGRAM [14] that typed separation significantly outperforms undifferentiated storage. A comprehensive list of all available memory types and categories is provided in Table II.

This typing system serves a dual purpose. First, it enables type-filtered retrieval, allowing agents to query specifically for commitments, decisions, or any other category without polluting the result set. Second, it provides implicit priority and decay signals that the retrieval engine uses to weight results appropriately.

- E. Conflict Resolution

A distinguishing feature of Memanto is its built-in conflict resolution mechanism. When a new memory is committed that semantically contradicts an existing memory, for example, Project deadline is April 15 followed by Project deadline is May 1, Memanto detects the conflict and notifies the agent, requesting explicit resolution before the contradiction is persisted.

This mechanism directly addresses the constraint drift failure mode identified in long-running agent deployments. Without conflict detection, agents gradually accumulate contradictory memories that erode the coherence of their reasoning.

###### Agent Ecosystem (User Environment)

Memanto Frontend

memanto Gateway

Agent CLIs (Claude Code, Cline, ..)

memanto CLI Engine

CLI Commands

IDE Integrations (Cursor, Windsurf, Roo, ..)

### U

Shared Services

API / CLI calls

Custom Agents (Python, JS, LangChain)

REST API Requests

User

memanto FastAPI Server

Local Web Dashboard (memanto ui)

Status & Controls

Fig. 3. Memanto Frontend Architecture: User, Agent Ecosystem, and memanto Gateway

##### Memanto Backend

Shared Services

Daily Summary

Conflict Resolution

Moorcheh.ai

Answer

Zero-Indexing Semantic Database

Recall

Moorcheh Engine API Agentic Optimized RAG

Remember

SDK Calls

Agent Manager

Native Access to LLM

Session & Auth Manager Memory Sync (MEMORY.md inject) Tool Connect

Fig. 4. Memanto Backend Architecture: Shared Services and Moorcheh.ai Integration

MemoryAgentBench [30] confirms that conflict resolution remains one of the most significant unsolved challenges in current memory systems, with all evaluated methods failing on multi-hop conflict scenarios.

Conflict resolution operates through semantic similarity matching against existing memories of the same type within the agent’s namespace, using a configurable contradiction threshold. When triggered, the system presents the agent with the conflicting memories and three resolution options: supersede (replace the old memory), retain (keep the old memory), or annotate (preserve both with a conflict flag for human review).

- F. Temporal Versioning Memanto supports three temporal query modalities:

- • As-of queries: Retrieve the memory state as it existed at a specific historical timestamp, enabling audit trail reconstruction.
- • Changed-since queries: Retrieve all memories created or modified within a time range, supporting incremental context updates.
- • Current-only queries: Retrieve only non-superseded memories, providing ground-truth state without historical noise.

Memory supersession is non-destructive: superseded entries are marked accordingly but retained in the store, enabling full temporal reconstruction. This design is critical for compliancesensitive deployments in regulated industries and directly addresses the knowledge update evaluation category in LongMemEval [9].

TABLE II MEMANTO TYPED MEMORY SCHEMA WITH 13 SEMANTIC CATEGORIES.

Type Description Example Priority Signal fact Objective, verifiable

User is in PST timezone Stable, high

confidence preference User or system preferences Prefers dark mode Moderate decay decision Choices made affecting future

information

Chose PostgreSQL for DB High persistence

commitment Promises or obligations Deliver report by Friday Time critical goal Objectives to achieve Reach 10K users by Q4 Active until

achieved event Historical occurrences Meeting with CEO at 2pm Episodic, decaying instruction Rules and guidelines Always validate input Procedural,

persistent relationshipEntity connections Alice manages Bob Graph-like, stable context Situational information Currently in budget review Highly temporal learning Lessons from experience Users need simpler

Accumulating observation Patterns noticed Traffic peaks on Fridays Statistical,

onboarding

evolving error Mistakes to avoid Do not use deprecated API Persistent guard artifact Document or code

Q3 budget spreadsheet Reference pointer

references

- G. Session and Namespace Management

Memanto isolates agent memory through a namespace architecture in which each agent maintains an independent memory namespace. Sessions are time-bounded windows with a default duration of six hours that provide temporal grouping without restricting cross-session retrieval. All memories within a namespace remain accessible regardless of session boundaries, enabling the persistent cross-session context that constitutes the primary use case for agentic memory.

- H. Daily Intelligence

Memanto generates automated daily intelligence artifacts including session summaries, contradiction detection reports, and interactive conflict resolution prompts. These artifacts are persisted as local Markdown files and optionally synced to Moorcheh’s cloud store, providing both human-readable audit trails and machine-queryable context for agents operating on daily planning cycles.

IV. EXPERIMENTAL EVALUATION AND RESULTS

Memanto is evaluated on two established agentic memory benchmarks through a five-stage progressive ablation study in which the independent contribution of each architectural decision is systematically isolated and quantified. Per-category accuracy results are subsequently reported for the final configuration, followed by a comprehensive comparison against all publicly available competing systems and a quantitative analysis of the operational overhead imposed by alternative architectures. Memanto is publicly accessible as a Python package via pip install memanto. A. Benchmarks and Evaluation Protocol

LONGMEMEVALS [9]: A large-scale benchmark comprising 500 manually curated questions distributed across six

categories, designed to evaluate five core memory abilities including information extraction, multi-session reasoning, temporal reasoning, knowledge update, and abstention. The standard evaluation setting uses approximately 115K tokens across approximately 50 sessions. All evaluations employ Claude Sonnet 4 as the LLM judge.

LOCOMO [10]: A multi-modal dialogue benchmark spanning four reasoning categories, namely Single-Hop, MultiHop, Open Domain, and Temporal reasoning. Individual dialogues extend to 35 sessions, 300 turns, and approximately 9K tokens on average, providing a rigorous test of long-horizon conversational memory.

To ensure evaluation consistency across systems, answer generation and judge prompts are adapted from the Hindsight repository [18], which is specifically designed to mitigate two systematic evaluation artifacts: answerer refusal on ambiguous questions, and rigid judge rejection of responses that are semantically correct but lexically divergent from the reference answer. All experiments employ Memanto’s vector-only architecture with the Moorcheh ITS engine as the sole retrieval backend.

B. Progressive Ablation Study

Rather than reporting only final results, the contribution of each architectural decision is evaluated in isolation through a controlled, sequential ablation. Each stage is defined by a specific configuration change, and its effect is measured against the preceding stage. The five stages are presented below, each accompanied by its configuration, quantitative outcomes, and the principal empirical finding it yields.

1) Stage 1: Naive Baseline: Configuration: Standard semantic search with a retrieval limit of k=10, an ITS similarity threshold of 0.15, and Claude Sonnet 4 as the inference model.

TABLE III STAGE 1: NAIVE BASELINE

Benchmark Accuracy

LONGMEMEVAL 56.6% LOCOMO 76.2%

Config: k=10, threshold=0.15, Claude Sonnet 4

As shown in Table III, this configuration yields 56.6% on LONGMEMEVAL and 76.2% on LOCOMO, establishing the performance floor for a minimally parameterised retrievalaugmented generation implementation on Memanto. The 19.6 percentage point gap between the two benchmarks does not reflect a difference in session length, as the two datasets are comparable in this regard. Rather, it reflects a difference in question structure and information accessibility. LONGMEMEVAL queries tend to be longer and span multiple topics simultaneously, which distributes the semantic signal across a broader embedding space and reduces similarity scores for relevant chunks. Under a threshold of 0.15, this information frequently falls below the retrieval cutoff and is not surfaced. The gap therefore serves as an early indicator of the sensitivity of retrieval performance to threshold calibration, a dynamic that Stage 2 directly exploits.

2) Stage 2: Recall Expansion: Standard RAG practice constrains retrieval to a fixed top-k of 10, applying aggressive precision filtering at the retrieval layer. For agentic memory, however, multi-session questions frequently require the synthesis of facts that are distributed across temporally disjoint sessions [9], rendering this constraint a critical architectural bottleneck.

Configuration: Retrieval limit increased to 40 chunks; ITS similarity threshold relaxed to 0.10.

As shown in Table IV, this single parameter adjustment yields a gain of 20.4 percentage points on LONGMEMEVAL and 6.6 percentage points on LOCOMO, constituting the largest single improvement observed across all five ablation stages.

TABLE IV STAGE 2: RECALL EXPANSION (k = 10 → 40)

###### Benchmark Accuracy ∆

LONGMEMEVAL 77.0% +20.4 percentage points LOCOMO 82.8% +6.6 percentage points

Finding. The precision-recall tradeoff in agentic memory skews decisively toward recall. Providing the LLM with a broader, noisier candidate set and relying on its in-context reasoning to filter irrelevant content is substantially more effective than constraining retrieval to a narrow, high-precision window that risks excluding critical fragments.

3) Stage 3: Prompt Optimization: Configuration: Retrieval parameters unchanged from Stage 2 (k=40, threshold=0.10); answer generation and judge prompts replaced with optimized variants from the Hindsight repository [18].

As shown in Table V, this modification yields marginal improvements of 2.2 percentage points on LONGMEMEVAL and 0.1 percentage points on LOCOMO.

TABLE V STAGE 3: PROMPT OPTIMIZATION

###### Benchmark Accuracy ∆

LONGMEMEVAL 79.2% +2.2 percentage points LOCOMO 82.9% +0.1 percentage points

Finding. Prompt engineering yields only marginal performance improvements. When the retrieval layer fails to surface relevant content, no degree of prompt refinement compensates for the resulting structural deficit. This finding is consistent with the broader observation that as foundational model capabilities continue to advance, the relative contribution of prompt design to overall system performance diminishes in proportion to the quality of the underlying retrieval mechanism.

4) Stage 4: Maximum Recall: Error analysis of Stage 3 failures established that incorrect answers were attributable not to LLM confusion arising from expanded context [23], but to semantic search consistently failing to retrieve critical sentences embedded within otherwise largely irrelevant chunks.

Configuration: Dynamic retrieval limit expanded to a maximum of 100 chunks, governed by ITS-threshold gating rather than a fixed-k constraint; similarity threshold lowered to 0.05.

As shown in Table VI, this configuration yields improvements of 5.8 percentage points on LONGMEMEVAL and 3.4 percentage points on LOCOMO.

TABLE VI STAGE 4: MAXIMUM RECALL (k = 100)

###### Benchmark Accuracy ∆

LONGMEMEVAL 85.0% +5.8 percentage points LOCOMO 86.3% +3.4 percentage points

Finding. Modern large language models exhibit a high degree of tolerance for noisy retrieval context. High-dimensional vector search is susceptible to distortion by multi-topic chunks in which a single critical detail is co-located with predominantly irrelevant content. Extending the retrieval budget to accommodate such cases consistently outperforms engineering for retrieval precision, confirming that recall is the dominant lever for agentic memory performance.

5) Stage 5: Inference Model Upgrade: Configuration: Inference model upgraded from Claude Sonnet 4 to Gemini 3, establishing parity with the leading competing systems against which Memanto is benchmarked. This stage isolates the contribution of inference model capability from that of the memory architecture, ensuring that comparative results reflect architectural differences rather than model selection artifacts.

The complete ablation progression across all five stages, together with the cumulative accuracy gains at each step, is visualized in Fig. 5. The relationship between retrieval limit

87.1

S5 Model

89.8

86.3

S4 MaxRecall

85

82.9

- S1 Baseline
- S2 Recall
- S3 Prompt

79.2

82.8

77

LONGMEMEVAL LOCOMO

76.2

56.6

50 60 70 80 90 100

Accuracy (%)

- Fig. 5. Progressive ablation waterfall. Stage 2 (recall expansion, k = 10 → 40) delivers the largest single gain (+20.4 percentage points on LONGMEMEVAL), confirming that retrieval tuning rather than architectural complexity is the dominant performance driver. Stage 3 (prompt optimisation) contributes only +2.2 percentage points, while Stages 4 and 5 contribute +5.8 and +4.8 percentage points respectively.

k and accuracy, overlaid with average token cost per query, is presented in Fig. 6.

10 20 40 60 80 100

60

80

Inflection k = 40

Retrieval limit k

Accuracy(%)

LME LOCOMO

10K

20K

30K

40K

50K

·104

Avg.tokens/query

- Fig. 6. Accuracy versus retrieval limit k (left axis) and average tokens consumed per query (right axis, dotted). Both accuracy curves plateau above k = 60, with a clear inflection at k = 40. The accuracy gain from k = 10 → 40 (+20.4 percentage points on LME) far outweighs the approximately fourfold token cost increase, validating the recall over precision principle.

C. Final Results by Category

Per-category accuracy results at Stage 5 under Gemini 3 inference are reported in Tables VII and VIII for LONGMEMEVAL and LOCOMO, respectively. Memanto achieves 100.0% on the Single-session Assistant category and 95.7% on Single-session User, with the lowest performance on Multisession queries at 81.2%, reflecting the inherent difficulty of synthesizing information distributed across extended interaction histories.

- D. Comparative Results

Table IX presents a comprehensive comparison of Memanto against all publicly reported results from competing agentic memory systems on both benchmarks. All figures for competing systems are drawn from their respective published papers or official benchmark reports. As shown in Table IX, Memanto achieves the highest accuracy among all vector-only systems

TABLE VII LONGMEMEVAL FINAL RESULTS BY CATEGORY (STAGE 5, GEMINI 3)

###### Category Accuracy

Single-session User 95.7% Single-session Assistant 100.0% Single-session Preference 93.3% Knowledge Update 93.6% Temporal Reasoning 88.0% Multi-session 81.2%

###### Overall 89.8%

TABLE VIII LOCOMO FINAL RESULTS BY CATEGORY (STAGE 5, GEMINI 3)

###### Category Accuracy

Single-Hop 78.7% Multi-Hop 70.8% Open Domain 92.4% Temporal 85.4%

Overall 87.1%

on both benchmarks, surpassing Mem0 by 22.9 percentage points on LONGMEMEVAL and 20.2 percentage points on LOCOMO. Hindsight attains higher overall accuracy on both benchmarks, but does so at a complexity score of 4 out of 4, requiring dynamic multi-query retrieval and structured reflection passes, as visualized in Fig. 8. The architectural complexity score is computed as the sum of four binary indicators: (1) requires a graph database, (2) invokes an LLM at ingestion time, (3) employs multi-query retrieval, and (4) uses recursive or reflective querying. Each indicator contributes one point, yielding a scale from 0 (minimal overhead) to 4 (maximum complexity). The grouped accuracy comparison across all systems is presented in Fig. 7.

87.1

Memanto

89.8

Hindsight

89.6

91.4

EmergMem

80

- 85.2
- 86

Supermem.

75.8

80

ENGRAM

78

75.8

Memobase

75.8

Zep

- 74
- 75.1

71.2

Letta

74

72.9

Full Context

60.2

Mem0g

68.4

68.4

66.9

Mem0

LONGMEMEVAL LOCOMO

66.9

LangMem

58.1

58.1

50 60 70 80 90 100

Accuracy (%)

Fig. 7. Benchmark comparison. Memanto leads vector-only systems. Hindsight achieves higher accuracy via multi-query parallel retrieval + reflection layers (complexity score 4/4).

TABLE IX SYSTEM COMPARISON ON LONGMEMEVAL AND LOCOMO

Bestbenchmarkaccuracy(%)

System LoCoMo LMEval Architecture Retrieval Query Strategy

Memanto (ours) 87.1% 89.8% Vector Only RAG Single Query Hindsight [18] 89.6% 91.4% Hybrid (Reflection and

Parallel Multi-Query EmergenceMem — 86.0% Hybrid (Graph and

Vector)

Parallel Multi-Query Supermemory — 85.2% Hybrid (Graph and

Vector)

Parallel Multi-Query Memobase 75.8% — Hybrid (Graph and

Vector)

Parallel Single Query Zep [6] 75.1% 71.2% Hybrid (Graph and

Vector)

Parallel Single Query

Vector)

Letta [7] 74.0% — Local Filesystem RAG Recursive Full Context 72.9% 60.2% Full Context Full Not applicable Mem0g [5] 68.4% — Hybrid (Graph and

Parallel Single Query

Vector)

Mem0 [5] 66.9% — Vector Only Parallel Single Query LangMem 58.1% — Vector Only RAG Single Query

TABLE X MEMORY TAX: OPERATIONAL OVERHEAD COMPARISON

Ideal Memanto zone

Hindsight

90

EmergMem

System LLM

LLM per Ret.

Infrastructure

Ingest Latency

Idle Cost

Supermem.

per Write

80

Zep

Memanto 0 1 Moorcheh

<10ms Zero

70

Vector DB + API key

Mem0

Mem0g

Mem0 1 1 Vector DB ≈500ms Fixed Mem0g ≥2 ≥2 Vector and

≈2s Fixed Zep ≥2 ≥2 Vector and

60

Neo4j

≈3s Fixed

0 1 2 3 4

Graph

Architectural complexity score (0–4)

Fig. 8. Architectural complexity vs. accuracy. Complexity score = sum of: requires graph DB, LLM at ingestion, multi-query retrieval, recursive querying (each 0/1). Memanto occupies the ideal upper-left quadrant. Hindsight achieves higher accuracy only at maximum complexity (score 4).

- E. The Memory Tax: Overhead Analysis

Beyond raw accuracy, sustainable production deployment requires careful consideration of the operational overhead imposed by each memory architecture. The Memory Tax is characterized along four dimensions, and its quantitative impact is summarised in Table X.

Ingestion overhead. Systems requiring LLM-mediated entity extraction, including Mem0g, Zep, and A-MEM, consume tokens and incur non-trivial latency at every write operation. As shown in Table X, Mem0g and Zep invoke two or more LLM calls per write, resulting in ingestion latencies of approximately 2 and 3 seconds, respectively. For a customer support agent processing 1,000 messages per day, this overhead accumulates to a substantial operational cost. Memanto ingests raw conversational content with zero LLM invocations at write time, eliminating this cost category entirely.

Retrieval latency. Multi-query and recursive retrieval strategies multiply inference calls per user interaction, compounding end-to-end latency. As shown in Table X, Memanto

achieves sub-10ms ingestion and sub-90ms retrieval using a single query, compared to the multi-second round-trip characteristic of graph-traversal systems.

Infrastructure complexity. Hybrid systems require the independent provisioning, scaling, monitoring, and maintenance of separate vector and graph database instances. As shown in Table X, Memanto requires only the Moorcheh Vector DB and API key, with no additional infrastructure to configure or operate.

Idle cost. Traditional vector databases mandate continuously provisioned compute resources regardless of query volume. As shown in Table X, Moorcheh’s serverless architecture scales to zero during idle periods, eliminating fixed infrastructure costs for intermittent agent workloads, in contrast to the fixed idle costs incurred by all competing systems.

For an agent executing 10K daily memory operations, estimated daily costs amount to $0.50 for Memanto, $2.32 for Mem0-Graph, and $1.70 for Zep, yielding annual savings of approximately $662 per agent relative to Mem0-Graph. This difference compounds substantially across enterprise deployments with large agent fleets.

V. DISCUSSION

- A. Why Optimized Retrieval Outperforms Graph Complexity

Our results challenge the prevailing industry consensus that knowledge graphs are necessary for high-quality agentic memory. Three factors explain Memanto’s strong performance, visible in Fig. 8’s upper-left quadrant.

- Factor 1: Retrieval reasoning decomposition. Modern

LLMs are exceptionally capable in-context reasoners when provided with relevant raw context [34]. Graph-based systems attempt to pre compute reasoning pathways through entity relationship structures, but this pre computation is inherently lossy because it must commit to a schema at index time and is therefore schema dependent. By contrast, providing the LLM a broader set of semantically relevant raw chunks and relying on its in context reasoning produces more flexible and accurate answers. Mem0’s own ablation [5] showing ≈2% graph gain directly supports this interpretation.

- Factor 2: Semantic matching quality matters more than

structure. Moorcheh’s ITS engine provides deterministic, exact match semantic search [33] rather than the approximate and jitter prone results of HNSW based systems [22]. When the underlying search is precise and relevance scoring is information theoretically grounded in uncertainty reduction rather than geometric proximity, the structural overhead of a knowledge graph provides only diminishing marginal returns. This is consistent with ENGRAM’s finding [14] that simple dense retrieval can match complex architectures.

- Factor 3: Ingestion simplicity enables faster iteration.

Eliminating the LLM extraction step at ingestion enables sub second write to retrieval feedback loops, accelerating development and debugging of agentic workflows. The synchronous extraction pipelines in Zep and Mem0g convert memory writes into multi second blocking calls, a latency profile incompatible with tight reasoning chains.

- B. The Recall over Precision Principle

Fig. 6 quantifies a counter intuitive principle: in agentic memory, recall systematically outperforms precision as a retrieval objective. Expanding k from 10 to 100 produces a cumulative +28.4pp improvement on LongMemEval, while prompt optimisation contributes only +2.2pp. Systems that invest engineering effort in precise structured retrieval such

- as graph traversal, multi hop entity resolution, and recursive query decomposition may be solving the wrong problem. The LLM is a more capable filter than any pre computed retrieval structure, at the cost of a few extra tokens of input context. This is consistent with LongMemEval’s finding [9] that performance continues improving beyond 20K retrieved tokens with GPT-4o, and with Liu et al.’s [23] finding that the lost in the middle effect is a function of where information appears rather than how much is retrieved.

- C. Conflict Resolution as a Production Necessity

While neither LongMemEval nor LoCoMo systematically tests contradictory memories, conflict resolution is critical for

production deployments. Long running agents accumulate contradictions through user corrections, updated preferences, and evolving project contexts. Without explicit conflict detection, these produce memory poisoning, which results in increasingly incoherent agent behaviour over time. Memanto’s proactive conflict detection provides a guardrail absent from all evaluated competing systems. MemoryAgentBench [30] confirms all current systems fail on multi hop conflict scenarios.

- D. Determinism for Agentic Stability

LLMs and autonomous agents exhibit high sensitivity to retrieval variability, where even minor changes in retrieved context can trigger divergent reasoning paths. ANN based systems introduce non determinism through probabilistic graph traversal, meaning identical queries may return different results depending on index state. This volatility compounds in multi turn agent interactions, propagating inconsistencies through conversation history. Memanto’s exhaustive search architecture provides deterministic retrieval, eliminating this source of instability. Furthermore, new documents become immediately queryable without degrading search quality or requiring batch reprocessing, contrasting with HNSW systems that must balance between stale indices and computationally expensive rebuilds.

- E. Limitations and Future Work

Benchmark scope. Both LongMemEval and LoCoMo target conversational settings. Non-conversational agentic workflows, such as research agents, code generation, and multiagent coordination, remain untested. Merrill et al. [20] identify the need for benchmark testing high-level memory organization beyond factual recall.

Benchmark saturation and label quality. Manual inspection of individual questions suggests that approximately 5% of LONGMEMEVAL questions and 6–7% of LOCOMO questions contain labelling inconsistencies, including ambiguous reference answers and questions whose ground truth cannot be unambiguously determined from the provided dialogue context. This label noise establishes a practical performance ceiling that is independent of memory architecture quality. Compounding this concern, competing systems are rapidly approaching the accuracy levels reported here, suggesting that both benchmarks may soon be insufficient to distinguish meaningfully between strong memory architectures. The development of more targeted evaluation protocols, particularly those that stress-test conflict resolution, multi-agent coordination, and non-conversational agentic workflows, represents an important direction for the field.

Inference model dependence. Final results use Gemini 3, contributing +4.8pp on LongMemEval. As foundational models improve, retrieval quality will likely become an even larger differentiator relative to inference capability.

Scale evaluation. While Moorcheh’s engine has been validated at 10M+ documents and 2,000+ QPS on MAIR [33], large scale memory benchmarks testing Memanto with thousands of concurrent agents remain future work.

Multi agent memory sharing. Memanto’s namespace architecture currently isolates agent memories by design. Shared memory across agent teams with appropriate access control and consistency protocols is under active development.

CONCLUSION

We have presented Memanto, a universal memory layer for agentic AI achieving state of the art results on LONGMEMEVAL (89.8%) and LOCOMO (87.1%) using a vector only architecture with zero cost ingestion, a 13 category typed semantic memory schema, and built in conflict resolution. A five stage ablation study demonstrated that retrieval recall, rather than architectural complexity, is the dominant performance driver, and that modern LLMs perform the reasoning and filtering that graph based systems attempt to pre compute

- at ingestion time. By eliminating the Memory Tax, defined

- as the compounding cost of LLM mediated ingestion, multi query retrieval pipelines, and graph infrastructure management, Memanto enables production grade agentic memory at a fraction of the cost and complexity of hybrid alternatives. Memanto’s design embodies a principled trade in which the structural expressiveness of knowledge graphs is exchanged for operational simplicity, determinism, and zero latency ingestion of a single, highly optimised semantic search backend. The empirical results validate this trade decisively.

REFERENCES

- [1] Y. Hu, S. Liu, Y. Yue, G. Zhang, B. Liu, F. Zhu, J. Lin, H. Guo, S. Dou, Z. Xi, S. Jin, J. Tan, Y. Yin, J. Liu, Z. Zhang, Z. Sun, Y. Zhu, H. Sun, B. Peng, Z. Cheng, X. Fan, J. Guo, X. Yu, Z. Zhou, Z. Hu,

- J. Huo, J. Wang, Y. Niu, Y. Wang, Z. Yin, X. Hu, Y. Liao, Q. Li,
- K. Wang, W. Zhou, Y. Liu, D. Cheng, Q. Zhang, T. Gui, S. Pan, Y. Zhang, P. Torr, Z. Dou, J.-R. Wen, X. Huang, Y.-G. Jiang, and S. Yan, “Memory in the age of ai agents,” 2026. [Online]. Available: https://arxiv.org/abs/2512.13564

- [2] A. V, G. G. R., and R. Buyya, “Agentic artificial intelligence (ai): Architectures, taxonomies, and evaluation of large language model agents,” 2026. [Online]. Available: https://arxiv.org/abs/2601.12560
- [3] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in Proceedings of ICLR, 2023. [Online]. Available: https: //arxiv.org/abs/2210.03629
- [4] L. Wang et al., “A survey on large language model based autonomous agents,” Frontiers of Computer Science, vol. 18, no. 6, 2024.
- [5] P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav, “Mem0: Building production-ready ai agents with scalable long-term memory,”

2025. [Online]. Available: https://arxiv.org/abs/2504.19413

- [6] P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef, “Zep: A temporal knowledge graph architecture for agent memory,” arXiv preprint arXiv:2501.13956, 2025.
- [7] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, “Memgpt: Towards llms as operating systems,” 2024. [Online]. Available: https://arxiv.org/abs/2310.08560
- [8] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, “A-MEM: Agentic memory for LLM agents,” arXiv preprint arXiv:2502.12110, 2025.
- [9] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, “LongMemEval: Benchmarking chat assistants on long-term interactive memory,” in Proceedings of the International Conference on Learning Representations (ICLR), 2025, arXiv:2410.10813. [Online]. Available: https://arxiv.org/abs/2410.10813
- [10] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, “Evaluating very long-term conversational memory of LLM agents,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024, pp. 13851–13870, arXiv:2402.17753. [Online]. Available: https://arxiv.org/abs/2402.17753

- [11] E. Tulving, “Episodic and semantic memory,” in Organization of Memory, E. Tulving and W. Donaldson, Eds. Academic Press, 1972, pp. 381–403.
- [12] A. D. Baddeley, “Working memory,” Science, vol. 255, no. 5044, pp. 556–559, 1992.
- [13] M. Pink, Q. Wu, V. A. Vo, J. Turek, J. Mu, A. Huth, and M. Toneva, “Position: Episodic memory is the missing piece for long-term llm agents,” 2025. [Online]. Available: https://arxiv.org/abs/2502.06975
- [14] D. Patel and S. Patel, “Engram: Effective, lightweight memory orchestration for conversational agents,” 2026. [Online]. Available: https://arxiv.org/abs/2511.12960
- [15] M. Abou Ali, F. Dornaika, and J. Charafeddine, “Agentic ai: a comprehensive survey of architectures, applications, and future directions,” Artificial Intelligence Review, vol. 59, no. 1, Nov. 2025. [Online]. Available: http://dx.doi.org/10.1007/s10462-025-11422-4
- [16] U. Nisa, M. Shirazi, M. A. Saip, and M. S. M. Pozi, “Agentic ai: The age of reasoning—a review,” Journal of Automation and Intelligence, vol. 5, no. 1, pp. 69–89, 2026. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2949855425000516
- [17] T. Sumers, S. Yao, K. Narasimhan, and T. L. Griffiths, “Cognitive architectures for language agents,” arXiv preprint arXiv:2309.02427, 2023.
- [18] C. Latimer, N. Boschi, A. Neeser, C. Bartholomew, G. Srivastava, X. Wang, and N. Ramakrishnan, “Hindsight is 20/20: Building agent memory that retains, recalls, and reflects,” 2025. [Online]. Available: https://arxiv.org/abs/2512.12818
- [19] Z. Tan, J. Yan, I.-H. Hsu, R. Han, Z. Wang, L. Le, Y. Song, Y. Chen, H. Palangi, G. Lee, A. R. Iyer, T. Chen, H. Liu, C.-Y. Lee, and T. Pfister, “In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 8416–8439. [Online]. Available: https://aclanthology.org/2025.acl-long.413/
- [20] A. Shutova, A. Olenina, I. Vinogradov, and A. Sinitsin, “Evaluating memory structure in llm agents,” 2026. [Online]. Available: https: //arxiv.org/abs/2602.11243
- [21] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨uttler, M. Lewis, W. tau Yih, T. Rockt¨aschel et al., “Retrievalaugmented generation for knowledge-intensive NLP tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
- [22] Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824–836, 2020.
- [23] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle: How language models use long contexts,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 157–173, 2024.
- [24] B. J. Guti´errez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, “Hipporag: Neurobiologically inspired long-term memory for large language models,” 2025. [Online]. Available: https://arxiv.org/abs/2405.14831
- [25] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, “Raptor: Recursive abstractive processing for tree-organized retrieval,” 2024. [Online]. Available: https://arxiv.org/abs/2401.18059
- [26] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W. tau Yih, “REPLUG: Retrieval-augmented black-box language models,” in Proceedings of NAACL, 2024, pp. 8371–8384.
- [27] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, “Memorybank: Enhancing large language models with long-term memory,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 17, 2024, pp. 19724–19731.
- [28] Y. Du, H. Wang, Z. Zhao, B. Liang, B. Wang, W. Zhong, Z. Wang, and K.-F. Wong, “PerLTQA: A personal long-term memory dataset for memory classification, retrieval, and fusion in question answering,” in Proceedings of SIGHAN 2024, 2024, pp. 152–164.
- [29] J. Kim, W. Chay, H. Hwang, D. Kyung, H. Chung, E. Cho, Y. Jo, and E. Choi, “DialSim: A real-time simulator for evaluating longterm dialogue understanding of conversational agents,” arXiv preprint arXiv:2406.13144, 2024.
- [30] Y. Hu, Y. Wang, and J. McAuley, “Evaluating memory in llm agents via incremental multi-turn interactions,” 2026. [Online]. Available: https://arxiv.org/abs/2507.05257

- [31] A. Terranova, B. Ross, and A. Birch, “Evaluating long-term memory for long-context question answering,” 2025. [Online]. Available: https://arxiv.org/abs/2510.23730
- [32] Anthropic, “Claude model card,” https://www.anthropic.com/claude-3

-model-card, Anthropic, Tech. Rep., 2026, accessed: 2026.

- [33] S. M. Abtahi, M. Fekri, T. Khani, and A. Azim, “From hnsw to information-theoretic binarization: Rethinking the architecture of scalable vector search,” 2025. [Online]. Available: https://arxiv.org/abs/ 2601.11557
- [34] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” Advances in Neural Information Processing Systems, vol. 35, 2022.

D. Final Retrieval Parameters

The final evaluation configuration employs a dynamic retrieval budget of up to 100 chunks, governed by ITS threshold gating rather than a fixed top-k constraint. The ITS similarity threshold is set to 0.05, and a single retrieval query is issued per question, with no multi-query or recursive retrieval strategies applied.

APPENDIX

All benchmark evaluations are fully reproducible using the configuration described below. Code and evaluation scripts are publicly available on GitHub3, and datasets are hosted on Hugging Face4.

- A. Software Configuration

All benchmark evaluations are conducted using Memanto version 2.1.4 in conjunction with the latest production release of the Moorcheh SDK. Claude Sonnet 4 serves as the inference model for Stages 1 through 4, while Gemini 3 is employed at Stage 5 to establish parity with leading competing systems. Evaluation prompts for both answer generation and LLM judging are inspired by and partially adapted from the Hindsight repository [18]; however, because Memanto’s retrieval context is structurally different from Hindsight’s chunk-plus-reflection pipeline, the prompts have been modified accordingly. The LONGMEMEVAL assessment uses the full 500-question suite under the standard S setting, and LOCOMO is evaluated on its standard split. Claude Sonnet 4 serves as the LLM judge throughout all evaluation stages.

- B. Hardware Requirements

All experiments require a minimum of four CPU cores and 8GB of RAM, with 16GB recommended for parallel evaluation workloads. Approximately 10GB of disk space is required to accommodate the benchmark datasets. GPU access is not a prerequisite, as inference is performed via the managed Gemini 3 API. The software environment requires Python 3.10 or later, with Docker employed for containerised deployment.

- C. Memory Type Assignment

Memory type assignment is currently performed by the user at write time, who selects the appropriate category from the 13type schema described in Table II. Automated type assignment via a rule-based decision tree is planned for a future release and will eliminate this manual step entirely.

- 3https://github.com/moorcheh-ai/memanto-evaluation
- 4https://huggingface.co/moorcheh

