## Rethinking Memory as Continuously Evolving Connectivity

Jizhan Fang1,2, Buqiang Xu1, Zhixian Wang1, Haoliang Cao2, Xinle Deng1, Baohua Dong2, Hangcheng Zhu2, Ruohui Huang2, Gang Yu2, Ying Wei1, Guozhou Zheng1, Feiyu Xiong3, Haofen Wang4, Huajun Chen1, Ningyu Zhang1* 1Zhejiang University 2Alibaba Group 3MemTensor 4Tongji University

# arXiv:2605.28773v1[cs.CL]27May2026

Abstract

Existing memory-augmented LLM agents often treat memory as a static repository with pre-defined representations and fixed retrieval pipelines, which is brittle in dynamic agentic environments where feedback, task variation, and heterogeneous signals continuously reshape what should be remembered and how it should be connected. To address this, we propose FluxMem, a connectivity-evolving memory framework that models memory as a heterogeneous graph and progressively refines its topology through three stages: initial connection formation, feedback-driven refinement, and long-term consolidation. During execution, FluxMem repairs missing links, prunes interference, aligns abstraction granularity, and distills recurrent successful trajectories into reusable procedural circuits, guided by one metric for memory generalizability and evolutionary maturity. Across three fundamentally distinct benchmarks including LoCoMo, Mind2Web, and GAIA, FluxMem achieves consistent state-of-the-art performance, demonstrating strong adaptation and generalization in complex agentic environments. The code will be open-sourced in the near future1.

### 1 Introduction

For long-horizon agents, memory mechanism (Zhang et al., 2025c; Hu et al., 2026b) plays a central role (Mei et al., 2025), by distilling useful factual information, reusable experiences and skills from the agent’s past interaction trajectories (Packer et al., 2024; Wei et al., 2025; Zhang et al., 2026a), storing them in diverse memory forms, and retrieving relevant memories when similar tasks arise to support downstream problem solving and agent evolving (Qi et al., 2026; ang Gao et al., 2026; Qiu et al., 2025; Wang et al., 2026; Ye et al., 2026). For long-horizon

* Corresponding author. 1https://github.com/zjunlp/LightMem.

[Figure 1]

Figure 1: The failures of static memory systems.

agents, memory effectiveness ultimately depends on whether the most useful memories can be accessed at each decision step, as sufficiently useful memory context substantially improves subtask success.

We formalize such usefulness as a problem of memory connectivity. Drawing from cognitive science (Hebb, 2005; Frankland and Bontempi, 2005), we define memory as the long-term sedimentation of memory units and their connections, continuously shaped through environmental interaction. Mirroring human cognitive processes, this structural evolution operates on two levels. At the unit level, the brain generates new units for novel information and continuously reshapes existing units by modifying their internal content. This ensures that each memory unit dynamically integrates new experiences and refines its semantic representation. At the connection level, operations are strictly taskcentric, the system establishes links between coactivated units to form functional associations, and prunes links that prove irrelevant, maintaining an efficient associative network. Through repeated task execution and environmental feedback, these localized updates gradually consolidate into stable, large-scale regions of interconnected nodes and edges. Rather than static storage, memory thus emerges as a self-organizing structure that its mem-

ory units and connections continuously adapt and evolve over time (Kelly and Garavan, 2005).

Challenges. First, Failure of Adaptive Memory Connectivity. Existing methods predominantly rely on static, hand-crafted pipelines (Yang et al., 2026; Chhikara et al., 2025; Fang et al., 2025b; Suzgun et al., 2026). By hardcoding memory operations, they assume rigid designs and fixed operations generalize across tasks. However, such static paradigms cannot establish optimal memory structures for diverse scenarios or dynamically refine them based on environmental feedback (Zhang et al., 2025b; Chen et al., 2026b). This inflexibility creates bottlenecks at both the connection and unit levels: (1) Inaccurate Memory Connections. This inaccuracy primarily manifests during memory retrieval. It leads to two concrete failures: under-connection, where critical links are missed due to retrieval imprecision, depriving the agent of essential context, and over-connection, where irrelevant associations are indiscriminately retrieved, introducing noise and hallucinations (Jiang et al., 2025; Chen et al., 2026a). Fundamentally, static pipelines lack the dynamic adaptability required for precise connection formation and access. (2) Inflexible Memory Unit Content. Existing systems represent memory units at a single, predefined level of abstraction. When unit content is misaligned, either excessively coarse, losing critical execution details, or overly fine, obscuring high-level structural patterns, the memory unit fails to adaptively integrate new experiences.

Second, Failure of Memory Connection Consolidation. While existing systems preserve task trajectories (Fang et al., 2026; Ouyang et al., 2025; Tang et al., 2025a), they treat memories as isolated instances rather than progressively consolidating them. True consolidation requires localized updates to coalesce through feedback into stable, large-scale associative regions. Lacking this mechanism, agents repeatedly reconstruct similar associations instead of internalizing enduring structural patterns, preventing memory networks from selforganizing into optimal configurations.

Method. To address these challenges, we propose FluxMem, a connectivity-evolving framework that models memory as a dynamically editable heterogeneous graph across semantic, episodic, and procedural layers. Context is formalized as an activated subgraph refined through a three-stage evolutionary pipeline. (1) Initial

Connection Formation rapidly establishes tentative cross-layer associations for novel tasks. (2) Feedback-Driven Refinement employs a closedloop mechanism to iteratively edit subgraph topology, creating missing links, pruning interference, or conditionally bypassing memory until execution succeeds. (3) Long-Term Consolidation clusters successful trajectories to induce stable procedural circuits, monitored by a convergence maturity metric. As high-utility pathways crystallize, recurring tasks bypass redundant retrieval and directly activate mature subgraphs. This pipeline transforms static memory storage into a self-optimizing connectivity substrate that continuously adapts to evolving task demands.

Results. We evaluate FluxMem on three benchmarks covering distinct task scenarios to evaluate generalization: LoCoMo (Maharana et al., 2024) (long-context reasoning), Mind2Web (Deng et al., 2023) (real-world web navigation), and GAIA (Mialon et al., 2023) (general assistant tasks). FluxMem achieves state-of-the-art performance across all three benchmarks. On LoCoMo, FluxMem reaches 95.06 average accuracy, above the Full Context baseline (81.23). On Mind2Web in the realistic setting (no manual element filtering), FluxMem improves Cross-Task success rate to 8.1, more than AWM (Wang et al., 2024c) (3.6). On GAIA, FluxMem increases the average success rate from 52.12 to 64.85 on Kimi K2 (+12.73% absolute) compared with the Flash-Searcher (Qin et al., 2025) baseline, and also surpasses the strong MemEvolve (Zhang et al., 2025b) baseline.

### 2 FluxMem Memory Architecture

2.1 Three-Layer Memory Graph We model FluxMem as a heterogeneous graph G = (V,E), where the node set V comprises three functional layers. ⃝1 Semantic Knowledge Vsem stores static factual knowledge that provides evidential support (e.g., knowledge documents or their corresponding chunks). ⃝2 Episodic Experiences Vepi records concrete state-action trajectories (e.g., debugging logs or tool-use sequences). ⃝3 Procedural Skills Vproc encapsulates distilled reasoning templates (e.g., multi-step planning heuristics).

Among the 3 layers, Vepi serves as the operational nexus that orchestrates the interplay between static knowledge and distilled skills. Each node

vepi(q) ∈ Vepi represents a specific task q and records

[Figure 2]

- Figure 2: The FluxMem architecture. Stages I and II operate online at a step-wise granularity. Stage III is conducted offline, aiming for immediate performance optimization and long-term memory consolidation, respectively.

its full step-by-step trajectory τq = {(ot,at)}Tt=1. The three layers are linked in a bottom-up order through two types of edges in E. First, during task execution, the agent retrieves relevant facts from Vsem to explain its current observation and decide the next step. This creates the edge set Eground ⊆ Vsem × Vepi, where an edge (vsem,vepi(q)) simply indicates that a specific fact provides evidential support for a step in task q. Second, after completing one or more similar tasks, the agent identifies common patterns in these trajectories and summarizes them into reusable skills. This forms the edge set Edistill ⊆ Vepi × Vproc, where an edge (vepi(q),vproc) shows that a skill is distilled from past experiences. Once created, vproc ∈ Vproc can be used to guide the agent in future tasks.

In FluxMem, the semantic layer is sourced directly from the environment, encompassing raw inputs such as dialogue histories and tool API documentation. Episode nodes are instantiated individually for each task, while procedural skill nodes are subsequently induced in section 3.3.

- 2.2 Context as Dynamically Induced Connectivity

At each step t of task q, FluxMem constructs the agent’s context St(q). The system dynamically selects a task-specific subset of nodes to form a local subgraph Gt(q) = (Vt,Et) ⊂ G, where Vt = Vtsem ∪ Vtepi ∪ Vtproc contains the activated memory nodes from the three layers.

St(q) = Concat(q, Obst, Vtsem, Vtepi, Vtproc),

In this formula, Obst describes the observations. Under this formulation, optimizing the working context is equivalent to performing targeted topological edits on Gt(q), as the prompt content is strictly determined by the activated node set and edge connections. Consequently, the adaptation pipeline systematically evolves Gt(q) from fragile tentative links into robust, task-optimized circuits through three sequential stages.

### 3 Three-Stage Memory Evolution

FluxMem comprises three stages. Stages I and II operate online at a step-wise granularity during task execution. At each time step t, Stage I is executed first to generate St(q), which is immediately processed by Stage II to yield the refined context St′(q). Stage III is conducted offline.

#### 3.1 Stage I: Initial Connection Formation

Semantic Connection Retrieval. At time step t, given the current observation ot, the system establishes initial associations between ot and supporting factual knowledge by querying the semantic layer Vsem. We compute a hybrid relevance score for each candidate v ∈ Vsem by fusing dense embedding similarity, sparse lexical matching, and LLM-based verification:

v · ot ∥v∥∥ot∥

Score(v,ot) =

+ BM25(v,ot)

+LLMver(v,ot).

The top-k nodes instantiate Vtsem, with directed edges Etsem = {(vt,v) | v ∈ Vtsem} established to

link them to the current step anchor vt.

Episodic Connection Retrieval. To draw on experience from past similar tasks, we query the episodic layer Vepi for the k most relevant past episodes using embedding similarity.

Vtepi = TopKu∈V

##### cos(u,ot) .

epi

Procedural Connection Inheritance. Based on the retrieved episodes Vtepi, we collect applicable skills by traversing existing distillation connections. Specifically, we select all skill nodes vproc that are linked to any retrieved episode via Edistill:

Vtproc =

vproc | (vepi,vproc) ∈ Edistill .

vepi∈Vtepi

The retrieved facts, episodes, and skills together form the initial step-local subgraph Gt = (Vt,Et), where Vt = Vtsem ∪ Vtepi ∪ Vtproc. This subgraph is serialized into the initial step context St. Although this provides a complete starting point for the current step’s reasoning, the selected connections are preliminary and will be refined in Stage II.

- 3.2 Stage II: Feedback-Driven Connectivity Refinement

Following the initial retrieval, the system addresses structural misalignments through a feedback-driven refinement loop. At step t, upon receiving execution feedback ft (from environmental signals or self-verification), the agent attributes reasoning failures to either connection-level or unit-level flaws and applies targeted edits to Gt.

Connection-Level Refinement. To resolve inaccurate memory connections, the system dynamically adjusts the associative topology based on feedback attribution. (i) Link Expansion for UnderConnection. If ft indicates missing critical context, the system identifies semantically proximate but unactivated nodes vnew ∈ V \ Vt and establishes new task-centric edges via Et ← Et ∪ {(vt,vnew)}. (ii) Link Pruning for Over-Connection. If ft reveals context congestion or hallucinated guidance, the system identifies distractor edges Enoise ⊂ Et and severs them via Et ← Et \ Enoise, isolating vt from irrelevant associations.

Unit-Level Refinement. To overcome inflexible memory unit content, the system dynamically reshapes internal representations when granularity misalignment impedes step-level reasoning. (iii)

Content Reshaping for Granularity Alignment. When retrieval is sufficient but the unit abstraction mismatches current demands (e.g., overly coarse for precise execution or overly fine for high-level planning), the system adaptively modifies the internal content of vold ∈ Vt. This involves either expanding vold with finer-grained execution details or abstracting redundant components to elevate its semantic level, yielding a refined unit valign. The local subgraph is updated by replacing vold with valign while preserving established connections.

After applying the targeted edits, the refined subgraph Gt′ = (Vt′,Et′) is serialized into the updated context St′ for subsequent reasoning. The loop terminates upon execution success or reaching a predefined refinement rounds T.

3.3 Stage III: Long-Term Connection Consolidation

Episodic Clustering and Skill Induction. Upon task completion, trajectories are committed as episodic nodes vτ ∈ Vepi. During offline consolidation, the system first partitions Vepi into M clusters {Cm}Mm=1 based on semantic trajectory similarity, computed via cosine distance between episode embeddings uτ. For each cluster Cm, an LLM-based induction operator extracts the skills or reasoning pattern shared across episodes, abstracting them into a new procedural skill node vproc(m) ∈ Vproc.

PEMS-Guided Iterative Consolidation. Since previous initial skill induction is one-way and may produce invalid skills, we verify and optimize them through a closed-loop refinement process guided by iterative evolution. At each iteration k, the system re-runs the source episodes Cm that generated each skill vproc, using the current skill version as guidance. We then compute the Procedure Evolution Maturity Score (PEMS) for every skill:

η Vproc(k) log ℓ Vproc(k)

× 1−δ Gcons(k) ,Gcons(k−1) ,

PEMS(k) =

where η(k) is the average success rate of the source episodes under the current skill, ℓ(k) is the token length of the skill text, and δ(k) measures the embedding difference between the current and previous skill versions. Based on the execution results, the LLM directly rewrites low-scoring skills to fix logical errors or remove redundant content. This test-score-refine cycle repeats until the score improvement ∆PEMS(k) falls below ϵ. At that point,

Method Single Hop

Multi Hop

Temp Open

AVG

Domain

GPT-4.1-mini

Full Context 87.99 80.50 71.03 58.33 81.23 Zep 66.90 53.70 60.20 43.80 61.60 Mem0 71.40 68.20 56.90 47.90 66.30 A-Mem 77.41 61.35 71.03 50.00 71.43

- MemoryOS 78.95 66.67 55.45 60.42 70.65 Nemori 87.16 77.30 76.32 55.21 81.10 LightMem 87.87 76.95 80.37 57.29 82.40 MIRIX 85.11 83.70 88.39 65.62 85.38 EverMemOS 96.67 91.84 89.72 76.04 93.05

- FluxMem 95.95 93.26 95.64 90.62 95.06 Qwen3-30B-A3B-2507-Instruct

Full Context 87.40 69.86 51.71 57.29 74.87 A-Mem 67.90 57.45 27.73 43.75 56.10 MemoryOS 79.51 63.12 32.09 48.96 59.81 LightMem 82.76 68.09 50.16 52.08 71.36

- FluxMem 96.19 91.49 89.72 87.50 93.44

- Table 1: Evaluation results on LoCoMo benchmark. Bold and underlined entries indicate the best and second-best results within each group, respectively.

the skills are validated as both highly useful and concise, and the offline consolidation ends.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets & Baselines. We evaluate the proposed framework across three challenging benchmarks. LoCoMo (Maharana et al., 2024) provides a comprehensive evaluation for long-context reasoning, we compare FluxMem against several representative baselines of conversational memory modeling: ⃝1 Zep (Rasmussen et al.,

- 2025), ⃝2 Mem0 (Chhikara et al., 2025), ⃝3 AMem (Xu et al., 2025), ⃝4 MemoryOS (Kang et al., 2025), ⃝5 Nemori (Nan et al., 2025), ⃝6 LightMem (Fang et al., 2025b), ⃝7 MIRIX (Wang and Chen, 2025), and ⃝8 EverMemOS (Hu et al.,
- 2026a). Mind2Web (Deng et al., 2023) serves as a testbed for web navigation, we compare FluxMem against representative baselines: ⃝1 AWM (Wang

- et al., 2024c) and ⃝2 Reasoning Bank (Ouyang
- et al., 2025). For general assistant tasks, we employ GAIA (Mialon et al., 2023) to benchmark against a wide array of frameworks: ⃝1 OpenAI Deep Research (OpenAI, 2024), ⃝2 Langfun (Peng, 2023), ⃝3 Magnetic-1 (Fourney et al., 2024), ⃝4 Agent KB (Tang et al., 2025b), ⃝5 smolagents (Roucher et al., 2025), ⃝6 Alita (Qiu et al.,

2025), ⃝7 Flash-Searcher (Qin et al., 2025), and ⃝8 MemEvolve (Zhang et al., 2025b).

Metrics. For LoCoMo, we report the LLM-as-ajudge (LMJ) score. For Mind2Web, we evaluate action-level accuracy with Element Accuracy (EA), Action F1 (AF1), Step Success Rate (SSR), and report overall Success Rate (SR) for completing a full navigation task. For GAIA, we use Success Rate across Level 1–3 to measure end-to-end task completion under increasing difficulty. Further statistics and experimental details about baselines are provided in Appendix A.1.

4.2 Main Results Superiority in Long-Context Reasoning. As

- shown in Table 1, FluxMem sets a new state-of-theart across all sub-categories on the LoCoMo benchmark. With the GPT-4.1-mini backbone, FluxMem achieves an outstanding average LMJ score of 95.06, significantly surpassing Full Context (81.23) and the strongest specialized memory system EverMemOS (93.05). This performance gap is even more pronounced when using Qwen3-30B-A3B2507-Instruct backbone, where FluxMem maintains a high average LMJ of 93.44, while the next best baseline Full Context drops to 74.87. Robust Performance in Web Navigation. As
- shown in Table 2, the evaluation on Mind2Web highlights the adaptability of FluxMem in noisy, real-world web environments. In the realistic setting without manual element filtering ( ‡ ), our framework demonstrates consistent improvements across both backbone models. With GPT-4.1-mini, FluxMem achieves a Success Rate (SR) of 8.1 in Cross-Task scenarios, more than AWM baseline (3.6). This trend is further reinforced with Gemini-2.5-flash, where FluxMem reaches an even higher SR of 9.6 in Cross-Task evaluation, substantially outperforming AWM (5.6). Across all sub-categories and model backbones, FluxMem consistently yields highest SSR and AF1 scores.

State-of-the-Art on Generalist Assistant Tasks. On GAIA benchmark, FluxMem demonstrates exceptional gains over the high-performance FlashSearcher baseline and the meta-evolutionary system MemEvolve in Table 3. When utilizing Kimi K2, our framework boosts the average success rate from 52.12 to 64.85, achieving a remarkable absolute improvement of 12.73%. In high-complexity tasks (Level 3), FluxMem reaches a success rate

Cross-Task Cross-Website Cross-Domain AVG EA AF1 SSR SR EA AF1 SSR SR EA AF1 SSR SR EA AF1 SSR SR GPT-4.1-mini

Method

No Memory‡ 34.5 64.7 28.0 2.8 33.0 60.4 24.6 1.1 30.1 55.9 24.6 0.7 31.3 58.1 25.2 1.1 AWM‡ 37.4 71.4 31.6 3.6 34.5 62.8 26.1 1.1 30.3 56.9 24.9 1.0 32.2 60.4 26.3 1.5 AWM† 64.6 82.3 56.3 9.1 62.9 79.7 52.7 4.5 50.5 69.3 43.9 2.0 54.8 73.1 47.4 3.7

FluxMem‡ 60.7 77.5 56.7 8.1 60.5 75.6 55.4 7.9 56.2 69.4 52.1 4.3 57.6 71.7 53.4 5.5 FluxMem† 70.4 84.2 60.1 13.5 70.2 82.1 58.7 9.2 65.2 75.3 55.3 7.2 66.8 77.9 56.6 8.6

Gemini-2.5-flash

No Memory‡ 40.7 69.5 35.6 2.8 35.7 64.7 28.3 1.7 22.1 37.2 18.6 1.8 20.5 46.9 23.1 2.0 AWM‡ 53.1 76.6 48.0 5.6 48.9 71.5 44.3 3.4 41.6 60.2 38.1 1.5 44.7 64.8 40.8 2.5 AWM† 67.3 82.2 60.9 8.7 63.0 77.0 55.3 4.5 52.0 67.0 47.9 3.4 56.3 71.2 51.3 4.5

ReasoningBank† 52.1 60.4 44.9 4.8 44.3 52.6 33.9 3.3 40.6 41.3 36.6 1.6 43.2 46.4 37.8 2.4

FluxMem‡ 61.6 77.4 57.0 9.6 59.4 72.8 56.1 7.5 58.5 70.1 55.9 5.0 59.2 71.8 56.1 6.2 FluxMem† 71.3 83.0 61.3 11.1 65.9 77.0 59.7 8.5 64.8 70.3 54.6 7.0 66.2 73.6 56.5 5.9

- Table 2: Evaluation results on Mind2Web. † denotes the filtered setting with manual removal of non-golden HTML elements; ‡ denotes the realistic setting without any pre-filtering (see Appendix A.2 for details). Bold and underline denote the best and second-best results.

Framework Model Avg. Level 1 Level 2 Level 3 Closed-source Agent Frameworks

Langfun Claude 3.7 etc. 71.52 83.02 68.60 57.69 OpenAI Deep Research OpenAI o1, o3 etc. 67.36 74.29 69.06 47.60

Open-Source Agent Frameworks

Magnetic-1 OpenAI o1 etc. 46.06 56.60 46.51 23.08 Agent KB GPT-4.1 61.21 79.25 58.14 34.62 Alita Claude-3.7-Sonnet,GPT-4o 72.73 81.13 75.58 46.15

Smolagents GPT-4.1 55.15 67.92 53.49 34.62 Smolagents GPT-5-mini 55.75 69.81 54.65 30.77

Flash-Searcher GPT-5-mini 69.09 79.25 69.77 46.15 Flash-Searcher Kimi K2 52.12 58.49 52.33 34.62 Flash-Searcher DeepSeek V3.2 60.61 79.25 53.49 46.15

MemEvolve GPT-5-mini 69.09 → 73.33 ↑4.24 79.25 → 83.02 ↑3.77 69.77 → 73.26 ↑3.49 46.15 → 53.85 ↑7.70 MemEvolve Kimi K2 52.12 → 61.21 ↑9.09 58.49 → 67.92 ↑9.43 52.33 → 63.95 ↑11.62 34.62 → 38.46 ↑3.84 MemEvolve DeepSeek V3.2 60.61 → 67.88 ↑7.27 79.25 → 83.02 ↑3.77 53.49 → 63.95 ↑10.46 46.15 → 50.00 ↑3.85

FluxMem GPT-5-mini 69.09 → 76.36 ↑7.27 79.25 → 88.68 ↑9.43 69.77 → 75.58 ↑5.81 46.15 → 53.85 ↑7.70 FluxMem Kimi K2 52.12 → 64.85 ↑12.73 58.49 → 77.36 ↑18.87 52.33 → 62.79 ↑10.46 34.62 → 46.15 ↑11.53 FluxMem DeepSeek V3.2 60.61 → 70.30 ↑9.69 79.25 → 81.13 ↑1.88 53.49 → 69.77 ↑16.28 46.15 → 50.00 ↑3.85

- Table 3: Evaluation results on GAIA benchmark. Color-coded arrows denote improvements relative to the FlashSearcher baseline, with deeper green shades indicating larger score improvements. Bold and underline indicate the best and second-best final results across all methods, respectively.

of 53.85 with GPT-5-mini, effectively matching or exceeding the capabilities of much larger closedsource agent frameworks.

#### 4.3 Ablation Study

We conduct ablation studies on LoCoMo and Mind2Web to evaluate the contribution of three stages, as shown in Figure 3 (a),(b) and (c).

On LoCoMo dataset, Stage II (Feedback-Driven Refinement) proves to be the most critical component. For GPT-4.1 mini, removing Stage II leads to a substantial decrease in the average LMJ score, dropping from 95.06% to 85.32%. A similar trend

is observed with Qwen3-30B-A3B, where the average score falls from 93.44% to 84.74% upon the exclusion of Stage II, while other two ablations show relatively smaller impacts. In memory-centric scenarios like LoCoMo, where all required evidence can be directly retrieved or simply inferred from the provided context, tasks rely more on accurate recall than complex reasoning. Consequently, the Stage II refinement mechanism, which expands retrieval or prunes irrelevant memory nodes, yields substantial performance gains by helping the agent find the correct facts. In such settings, refining the semantic knowledge layer proves highly effective, while the

[Figure 3]

- Figure 3: Detailed analysis of FluxMem components and evolution dynamics: (a) Ablation study of different stages on LoCoMo using GPT-4.1-mini. (b) Ablation study of different stages on LoCoMo using Qwen3-30B-A3B-2507Instrcut. (c) Ablation study of different stages on Mind2Web using GPT-4.1-mini. (d) Performance improvement across sub-tasks relative to the number of refinement rounds in Stage II. (e) Model accuracy trends and convergence of the Procedure Evolution Maturity Score (PEMS) across evolution rounds.

procedural skill layer contributes relatively less.

In contrast, on the Mind2Web dataset, Stage III (Long-Term Consolidation) emerges as the primary performance driver. For instance, removing Stage III for GPT-4.1-mini causes a drastic performance drop (e.g., the success rate on the first sub-category falls from 8.1% to 3.2%), while the impact of removing Stage II is relatively moderate. This disparity suggests that for complex, multi-step web navigation tasks requiring strong reasoning, the extraction of skills and evolution of skill nodes in Stage III are more vital than short-term refinement.

#### 4.4 Analysis of Iterative Refinement

We analyze the impact of the number of refinement iterations in Stage II on performance. The number of refinement rounds (T) in Stage II serves as a critical scaling factor. We evaluate this effect on the LoCoMo by varying T from 0 to 5, as shown in Figure 3(d). The results demonstrate a consistent and monotonic improvement across all sub-categories and the overall score. Without refinement (T = 0), the agent achieves an average score of 85.32%. By T = 5, the average performance reaches 95.06%. This steady gain suggests that the refinement mechanism allows the agent to refine connections and find more useful factual evidences. The diminishing returns observed between T = 4 and T = 5 (an improvement of only 0.54%) indicate that the agent’s performance begins to saturate as it approaches the optimal evidence path.

#### 4.5 Analysis of Memory Evolution and Convergence

At this point, the sensitivity threshold ϵ can be utilized to terminate the evolution process.

As shown in Figure 3(e), on the LoCoMo dataset, while Stage III provides a performance boost (from 91.16% at round 0 to 95.06% at round 5), the gains are more moderate compared to Stage II. This aligns with our observation that for fact-oriented tasks, the primary role of Stage III is to summarize and stabilize background knowledge. More importantly, we observe a clear convergence trend in the PEMS metric (scaled by a factor for visibility). The PEMS increases from 0.072 to 0.158 within the first four rounds and stabilizes at 0.159 by round 5. This convergence indicates that the memory maturity mechanism effectively identifies when the anchor nodes have reached a stable state of knowledge representation. At this point, the sensitivity threshold ϵ can be utilized to terminate the evolution process, thereby preventing redundant computations once the memory has captured the essential task-related insights.

#### 4.6 Case Study

Figure 4 presents a GAIA tabular reasoning task, where the agent must identify the country with the highest average medals per athlete from a CSV file. Following Stage I, FluxMem initializes the working context S0(q) by activating a local subgraph containing semantic knowledge (e.g., CSV parsing and spreadsheet APIs), a relevant episodic trajec-

[Figure 4]

Figure 4: Case Study. The key points have been highlighted in red.

tory for tabular ranking, and a coarse procedural skill for generic table QA. The first execution step succeeds, correctly parsing the file schema from environmental observation Obs1.

However, at the next step, the agent incorrectly invokes a spreadsheet visualization API for aggregation, which triggers an environmental failure due to rendering limitations. FluxMem attributes this failure to a connectivity mismatch and performs targeted topological edits: it prunes the ineffective semantic connection while expanding connectivity toward a more suitable Python data-analysis API, yielding a refined context S1′(q). Subsequently, although aggregation succeeds, selfverification reveals that the inherited procedural skill is overly coarse-grained: it supports ranking existing statistics but fails to compose the required metric (medals per athlete). This triggers Node Reshape, replacing the skill with a finer-grained statistical aggregation procedure.

### 5 Related Work

Hierarchical Structured Memory Systems. Hierarchical memory systems organize memory units through specific topologies. Tree-based memories abstract data into hierarchical levels (A et al., 2024; Ye et al., 2025), while graph-based structures offer greater connectivity dynamics (Long et al., 2025). Pyramid mechanisms construct multilevel abstractions to facilitate coarse-to-fine querying (Han et al., 2025; Rasmussen et al., 2025). Alternatively, heterogeneous multi-layer structures partition memory into distinct modules (Zhang et al., 2025b; Xu et al., 2026) or levels specialized for specific information types or functions (Zhang et al., 2025a; Gutiérrez et al., 2025a,b).

Self Evolving Agent Memory. Agent selfevolution is typically driven by memory evolving (ang Gao et al., 2026; Fang et al., 2025a; Shinn et al., 2023), which can be categorized into three paradigms. (i) Works such as Expel (Zhao et al.,

- 2024), AWM (Wang et al., 2024c), and ReasoningBank (Ouyang et al., 2025) maintain a contextual memory repository (Liu et al., 2025; Wei et al., 2025; Cai et al., 2025), distilling experiences from historical trajectories to enhance agent capabilities (Qiu et al., 2025). These methods vary in the granularity of trajectory processing (Fang et al., 2026). In addition to successful experiences, some approaches (Tang et al., 2025b; Cao et al., 2025) incorporate failure cases, while others (Wu et al., 2025) introduce iterative evolution mechanisms. (ii) Other works focus on parametric memory. SEAL (Zweiger et al., 2025) explores the potential of agent self-training, while AgentEvolver (Zhai et al., 2025) and Agent0 (Xia et al.,
- 2025) concentrate on data acquisition and reward design. The evolutionary mechanisms primarily include SFT (Zhou et al., 2025), RL (Zhang et al.,
- 2026b) and other paradigms such as early experience (Shi et al., 2025). (iii) Some works modify model architectures to enable deeper memoryaugmented evolving. This is achieved by either introducing additional parameters as external memory (Wang et al., 2024b,a) to manage information acquisition and forgetting, or proposing novel architectures (Behrouz et al., 2024, 2025a,b) to enhance the inherent memory capacity of base models. 6 Conclusion

We introduced FluxMem, an evolutionary framework conceptualizing agent memory as dy-

namic connectivity. By a three-phase evolution, FluxMem enables autonomous memory adaptation. SOTA results across LoCoMo, Mind2Web and GAIA, provides a principled foundation for self-evolving agents in dynamic environments.

### Limitations

Despite the demonstrated effectiveness, several limitations in our experimental design warrant acknowledgment: Computational Overhead of Closed-Loop Operations. Stages II and III rely on iterative LLM calls for context verification, topological editing, and skill induction. Our current evaluation prioritizes task success and convergence, but does not systematically measure the associated latency, API cost, or token consumption, which are critical for real-time or resource-constrained deployments. Static Benchmark Protocols. Experiments are conducted on pre-collected, static datasets (LoCoMo, Mind2Web, GAIA). While diverse, these benchmarks do not fully simulate continuous, open-world distribution shifts or streaming environments where task boundaries blur and memory decay must be actively managed alongside evolution. Hyperparameter Sensitivity. The framework introduces several control thresholds (e.g., refinement rounds T, PEMS convergence threshold ϵ, retrieval top-k). Our ablations focus on component efficacy but lack a comprehensive sensitivity analysis across different model backbones and highly heterogeneous domains. Future work should systematically evaluate the robustness of these parameters under varying computational budgets. Offline Consolidation Scheduling. Stage III is executed offline in periodic batches. The current experimental setup does not evaluate dynamic scheduling strategies or the trade-off between consolidation frequency and online performance degradation, which are essential for practical lifelong agent deployment.

### Use of AI Assistants

During the preparation of this manuscript, AI language models were utilized exclusively for linguistic refinement, grammar correction, and stylistic polishing. The core research concepts, experimental design, methodological implementation, narrative structure, and all figures/illustrations were independently conceived, developed, and verified by the human authors. The authors retain full responsibility for the scientific accuracy, originality,

and integrity of the work.

### References

Aadharsh Aadhithya A, Sachin Kumar S, and Soman K. P. 2024. Enhancing long-term memory using hierarchical aggregate tree for retrieval augmented generation. Preprint, arXiv:2406.06124.

Huan ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, and 8 others. 2026. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. Preprint, arXiv:2507.21046.

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. 2025a. It’s all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. Preprint, arXiv:2504.13173.

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. 2025b. Nested learning: The illusion of deep learning architectures. Preprint, arXiv:2512.24695.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. 2024. Titans: Learning to memorize at test time. Preprint, arXiv:2501.00663.

Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Jinsong Su, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. 2025. Flex: Continuous agent evolution via forward learning from experience. Preprint, arXiv:2511.06449.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. 2025. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. Preprint, arXiv:2512.10696.

Ding Chen, Simin Niu, Kehang Li, Peng Liu, Xiangping Zheng, Bo Tang, Xinchi Li, Feiyu Xiong, and Zhiyu Li. 2026a. Halumem: Evaluating hallucinations in memory systems of agents. Preprint, arXiv:2511.03506.

Yining Chen, Jihao Zhao, Bo Tang, Haofen Wang, Yue Zhang, Fei Huang, Feiyu Xiong, and Zhiyu Li. 2026b. Memprivacy: Privacy-preserving personalized memory management for edge-cloud agents. Preprint, arXiv:2605.09530.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. Preprint, arXiv:2504.19413.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web.

Advances in Neural Information Processing Systems, 36:28091–28114.

Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, and Zaiqiao Meng. 2025a. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. Preprint, arXiv:2508.07407.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, and 1 others. 2025b. Lightmem: Lightweight and efficient memory-augmented generation. arXiv preprint arXiv:2510.18866.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2026. Memp: Exploring agent procedural memory. Preprint, arXiv:2508.06433.

Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang, Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, and Saleema Amershi. 2024. Magentic-one: A generalist multi-agent system for solving complex tasks. Preprint, arXiv:2411.04468.

Paul W Frankland and Bruno Bontempi. 2005. The organization of recent and remote memories. Nature reviews neuroscience, 6(2):119–130.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2025a. Hipporag: Neurobiologically inspired long-term memory for large language models. Preprint, arXiv:2405.14831.

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025b. From rag to memory: Non-parametric continual learning for large language models. Preprint, arXiv:2502.14802.

Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and Jiliang Tang. 2025. Retrieval-augmented generation with graphs (graphrag). Preprint, arXiv:2501.00309.

Donald Olding Hebb. 2005. The organization of behavior: A neuropsychological theory. Psychology press.

Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, and 1 others. 2026a. Evermemos: A self-organizing memory operating system for structured long-horizon reasoning. arXiv preprint arXiv:2601.02163.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, and 28 others. 2026b. Memory in the age of ai agents. Preprint, arXiv:2512.13564.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gregory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and Camillo Jose Taylor. 2025. Personamem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. Preprint, arXiv:2512.06688.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory os of ai agent. arXiv preprint arXiv:2506.06326.

AM Clare Kelly and Hugh Garavan. 2005. Human functional neuroimaging of brain changes associated with practice. Cerebral cortex, 15(8):1089–1102.

Yitao Liu, Chenglei Si, Karthik Narasimhan, and Shunyu Yao. 2025. Contextual experience replay for self-improvement of language agents. Preprint, arXiv:2506.06698.

Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. 2025. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. Preprint, arXiv:2508.09736.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753.

Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, and Shenghua Liu. 2025. A survey of context engineering for large language models. Preprint, arXiv:2507.13334.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations.

Jiayan Nan, Wenquan Ma, Wenlong Wu, and Yize Chen. 2025. Nemori: Self-organizing agent memory inspired by cognitive science. arXiv preprint arXiv:2508.03341.

OpenAI. 2024. deepresearch.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei

Han, Chen-Yu Lee, and Tomas Pfister. 2025. Reasoningbank: Scaling agent self-evolving with reasoning memory. Preprint, arXiv:2509.25140.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.

Daiyi Peng. 2023. Langfun.

Shihao Qi, Jie Ma, Rui Xing, Wei Guo, Xiao Huang, Zhitao Gao, Jianhao Deng, Jun Liu, Lingling Zhang, Bifan Wei, Boqian Yang, Pinghui Wang, Jianwen Sun, Jing Tao, Yaqiang Wu, Hui Liu, Yu Yao, and Tongliang Liu. 2026. Beyond individual intelligence: Surveying collaboration, failure attribution, and selfevolution in llm-based multi-agent systems. Preprint, arXiv:2605.14892.

Tianrui Qin, Qianben Chen, Sinuo Wang, He Xing, King Zhu, He Zhu, Dingfeng Shi, Xinxin Liu, Ge Zhang, Jiaheng Liu, Yuchen Eleanor Jiang, Xitong Gao, and Wangchunshu Zhou. 2025. Flash-searcher: Fast and effective web agents via dag-based parallel execution. Preprint, arXiv:2509.25301.

Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, Xing Zhou, Dongrui Liu, Ling Yang, Yue Wu, Kaixuan Huang, Shilong Liu, Hongru Wang, and Mengdi Wang. 2025. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. Preprint, arXiv:2505.20286.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. Preprint, arXiv:2501.13956.

Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. 2025. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/ huggingface/smolagents.

Yuchen Shi, Yuzheng Cai, Siqi Cai, Zihan Xu, Lichao Chen, Yulei Qin, Zhijian Zhou, Xiang Fei, Chaofan Qiu, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Guocan Cai, Yong Mao, Yunsheng Wu, Ke Li, and Xing Sun. 2025. Youtu-agent: Scaling agent productivity with automated generation and hybrid policy optimization. Preprint, arXiv:2512.24615.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. 2026. Dynamic cheatsheet: Test-time learning with adaptive memory. In Proceedings of the 19th Conference of the European

Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7080–7106, Rabat, Morocco. Association for Computational Linguistics.

Xiangru Tang, Tianyu Hu, Muyang Ye, Yanjun Shao, Xunjian Yin, Siru Ouyang, Wangchunshu Zhou, Pan Lu, Zhuosheng Zhang, Yilun Zhao, Arman Cohan, and Mark Gerstein. 2025a. Chemagent: Selfupdating library in large language models improves chemical reasoning. Preprint, arXiv:2501.06590.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, Hao Cheng, Chi Wang, and Wangchunshu Zhou. 2025b. Agent kb: Leveraging cross-domain experience for agentic problem solving. Preprint, arXiv:2507.06229.

Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, and Shumin Deng. 2026. Skillx: Automatically constructing skill knowledge bases for agents. Preprint, arXiv:2604.04804.

Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2024a. Wise: Rethinking the knowledge memory for lifelong model editing of large language models. Advances in Neural Information Processing Systems, 37:53764–53797.

Yu Wang and Xi Chen. 2025. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, and Julian McAuley. 2024b. Memoryllm: Towards self-updatable large language models. Preprint, arXiv:2402.04624.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2024c. Agent workflow memory. arXiv preprint arXiv:2409.07429.

Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, and Derek Zhiyuan Cheng. 2025. Evo-memory: Benchmarking llm agent test-time learning with selfevolving memory. Preprint, arXiv:2511.20857.

Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, and Botian Shi. 2025. Evolver: Self-evolving llm agents through an experience-driven lifecycle. Preprint, arXiv:2510.16079.

Peng Xia, Kaide Zeng, Jiaqi Liu, Can Qin, Fang Wu, Yiyang Zhou, Caiming Xiong, and Huaxiu Yao. 2025. Agent0: Unleashing self-evolving agents from

zero data via tool-integrated reasoning. Preprint, arXiv:2511.16043.

Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du, and Shumin Deng. 2026. Structmem: Structured memory for longhorizon behavior in llms. CoRR, abs/2604.21748.

Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110.

Ke Yang, Zixi Chen, Xuan He, Jize Jiang, Michel Galley, Chenglong Wang, Jianfeng Gao, Jiawei Han, and ChengXiang Zhai. 2026. Plugmem: A task-agnostic plugin memory module for llm agents. Preprint, arXiv:2603.03296.

Chongrui Ye, Yuxiang Liu, Yu Wang, Haofei Yu, Yining Zhao, Ge Liu, Julian McAuley, and Jiaxuan You. 2026. Auto-dreamer: Learning offline memory consolidation for language agents. Preprint, arXiv:2605.20616.

Shicheng Ye, Chao Yu, Kaiqiang Ke, Chengdong Xu, and Yinqi Wei. 2025. H2r: Hierarchical hindsight reflection for multi-task llm agents. Preprint, arXiv:2509.12810.

Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, and Jingren Zhou. 2025. Agentevolver: Towards efficient self-evolving agent system. Preprint, arXiv:2511.10395.

Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. 2025a. G-memory: Tracing hierarchical memory for multi-agent systems. Preprint, arXiv:2506.07398.

Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. 2025b. Memevolve: Metaevolution of agent memory systems. Preprint, arXiv:2512.18746.

Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. 2026a. Memskill: Learning and evolving memory skills for self-evolving agents. Preprint, arXiv:2602.02474.

Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang, and Muning Wen. 2026b. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. Preprint, arXiv:2601.03192.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025c. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, and Jun Wang. 2025. Memento: Fine-tuning llm agents without fine-tuning llms. Preprint, arXiv:2508.16153.

Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, and Pulkit Agrawal. 2025. Self-adapting language models. Preprint, arXiv:2506.10943.

### A Experimental Details

#### A.1 Detailed Dataset Statistics

LoCoMo. The LoCoMo benchmark provides a specialized evaluation for long-context reasoning through 10 extensive conversations, featuring an average of 588 turns and 16,618 tokens per dialogue. Our evaluation utilizes a total of 1,540 human-annotated questions, covering 841 singlehop, 282 multi-hop, 321 temporal reasoning, and 96 open-domain questions. This distribution ensures a balanced assessment of both simple retrieval and complex logical synthesis.

Mind2Web. This dataset serves as a large-scale testbed for generalist web agents, featuring 2,350 open-ended tasks harvested from 137 real-world websites across 31 domains. The environmental complexity is significant, with an average page size of 1,135 DOM elements and tasks requiring an average of 7.3 discrete actions to complete. We specifically evaluate the model’s generalization across three dimensions: cross-task, cross-website, and cross-domain scenarios.

GAIA. Evaluation on the GAIA benchmark is conducted across 165 curated tasks spanning three levels of increasing operational difficulty. The set includes 53 Level-1 tasks focused on basic tool usage and retrieval, 86 Level-2 tasks requiring multistep planning and intermediate reasoning, and 26 Level-3 tasks involving long-horizon execution and multi-modal integration.

#### A.2 Implementation Details

LoCoMo. For the LoCoMo benchmark, all retrieval-based methods utilize the text-embedding-3-small model for generating embeddings. The specific retrieval configurations for each baseline are as follows: ⃝1 Zep and ⃝2

Mem0 utilize their respective commercial APIs to retrieve memory contexts, extracting the top-10 relevant memories for each speaker, which are then integrated for response generation. ⃝3 A-MEM performs a global search to retrieve the top-40 overall entries. ⃝4 MemoryOS is implemented as a three-tier hierarchical system featuring exhaustive recall of all Short-Term Memory (STM) pages, a two-stage selection for Mid-Term Memory (MTM) (comprising the top-5 segments and top-10 dialogue pages), and the extraction of the top-10 relevant entries from Long-term Personal Memory (LPM). ⃝5 Nemori retrieves the top-10 episodic memories and top-20 semantic memories (m = 2k). ⃝6 LightMem extracts the top-40 total entries through its retrieval pipeline. ⃝7 MIRIX utilizes a multi-agent active retrieval mechanism, generating automated topics to extract the top-10 entries from each of its six memory banks, totaling up to 60 entries. ⃝8 EverMemOS employs a hierarchical architecture based on memory scenes (MemScene) and memory cells (MemCell), utilizing hybrid retrieval (Dense+BM25) combined with a reranking strategy to extract the top-10 thematic scenes and top-10 narrative snippets.

Mind2Web. For the Mind2Web benchmark, all baseline methods are evaluated under an offline protocol and follow their respective native evaluation pipelines,using GPT-4.1-mini and Gemini-2.5flash as the base models.: ⃝1 AWM is evaluated under two settings. In the original setting, we apply top-k = 5 candidate filtering on the current page. However, due to the large number of page elements, restricting the current page to top-k candidates discards substantial information and effectively reduces the action selection difficulty, which deviates from realistic web interaction. Therefore, in the realistic setting, we disable top-k filtering for the current page and instead provide the model with all candidate elements, including both positive and negative elements, to better simulate realworld conditions. In both settings, we keep the same configuration for visited-history elements, using top-k = 3 retrieval from previously visited pages. ⃝2 ReasoningBank The agent predicts actions with retrieved reasoning memories injected into the prompt, and scores are computed by the standard evaluation script over the test set.

GAIA. For the GAIA benchmark, all methods are evaluated with a single run per task: ⃝1 Langfun operates with Claude 3.7 Sonnet and associ-

ated models through Google’s functional programming interface; ⃝2 OpenAI Deep Research leverages o1/o3 series models with integrated chain-ofthought and reinforcement learning optimization; ⃝3 Magnetic-1 employs an o1-based open-source agentic infrastructure; ⃝4 Agent KB incorporates an offline knowledge base pre-indexed with domain expertise; ⃝5 Alita uses multi-model orchestration (Claude 3.7 Sonnet + GPT-4o) with dynamic model selection; ⃝6 Smolagents implements a lightweight two-agent architecture (manager + tool agent). Our evaluations of ⃝7 Flash-Searcher (baseline), ⃝8 MemEvolve, and ⃝9 FluxMem are conducted within the Flash-Searcher framework, configured with 40 maximum steps, 8-step planning intervals. Flash-Searcher operates without memory augmentation (32,768 completion tokens per call). MemEvolve employs meta-evolved architectures through 3 iterations, 60 trajectories per round with 40 new + 20 reused tasks, (K=1) survivor, (S=3) descendants.

