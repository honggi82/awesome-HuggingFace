## Deep Tabular Research via Continual Experience-Driven Execution

# arXiv:2603.09151v2[cs.AI]12Mar2026

Junnan Dong*1 Chuang Zhou*1 Zheng Yuan1 Yifei Yu1 Qiufeng Wang1 Yinghui Li1 Siyu An†1 Di Yin1 Xing Sun1 Feiyue Huang†2 Ruijin Hospital, Shanghai Jiaotong University Tencent Youtu Lab

### Abstract

Large language models often struggle with complex long-horizon analytical tasks over unstructured tables, which typically feature hierarchical and bidirectional headers and non-canonical layouts. We formalize this challenge as Deep Tabular Research (DTR), requiring multi-step reasoning over interdependent table regions. To address DTR, we propose a novel agentic framework that treats tabular reasoning as a closed-loop decisionmaking process. We carefully design a coupled query and table comprehension for path decision making and operational execution. Specifically, (i) DTR first constructs a hierarchical meta graph to capture bidirectional semantics, mapping natural language queries into an operation-level search space; (ii) To navigate this space, we introduce an expectation-aware selection policy that prioritizes high-utility execution paths; (iii) Crucially, historical execution outcomes are synthesized into a siamese structured memory, i.e., parameterized updates and abstracted texts, enabling continual refinement. Extensive experiments on challenging unstructured tabular benchmarks verify the effectiveness and highlight the necessity of separating strategic planning from low-level execution for long-horizon tabular reasoning.

### 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in reasoning over structured data, leading to their widespread adoption in tabular question answering (Gong et al., 2020; Zhao et al., 2024b; Katsis et al., 2022). By serializing tables into textual formats, prior research has shown that LLMs can effectively resolve fac-

1Tencent Youtu Lab 2Ruijin Hospital, Shanghai Jiaotong University. Correspondence to: Feiyue Huang <Ruijin Hospital, Shanghai Jiaotong University>, Siyu An <Tencent Youtu Lab>.

Preprint. March 13, 2026.

tual and numerical queries over well-structured schemas. These advances have positioned LLMs as promising generalpurpose interfaces for interacting with tabular data (Ren et al., 2025; Somvanshi et al., 2024). Despite this progress, most existing approaches rely on clean schemas, flat headers, and single-pass reasoning pipelines. Such assumptions severely limit their applicability to practical downstream scenarios where tables are frequently irregular, incomplete, and semantically implicit for comprehension.

Real-world tabular data, particularly spreadsheets, exhibit a wide range of unstructured properties that defy traditional TableQA pipelines. As illustrated in Figure 1, these tables often feature hierarchical and bi-directional headers, merged cells, and values that are missing or contextually defined. Navigating these complex structures remains a persistent challenge (Tang et al., 2025; Wu et al., 2025). Furthermore, beyond structural complexity, practical analytical tasks are inherently long-horizon and multi-hop. A single query may necessitate a sequence of factual checks, numerical computations, and aggregations across disparate table regions. Answering such analytical queries requires more than simple retrieval; it demands iterative verification and conditional branching, where intermediate results must be scrutinized and revised before reaching a validated conclusion.

We formalize this problem as Deep Tabular Research (DTR), i.e., long-horizon complex tabular reasoning tasks that require coordinated data acquisition, computation, and analytical synthesis. While conventional approaches primarily rely on in-context learning, treating tables as text for direct LLM reasoning—such a paradigm is inherently limited by token constraints and struggles with precise numerical operations over large, irregular headers (Sarkar & Lausen, 2023; Singha et al., 2023). To overcome these limitations, we advocate for a programmatic execution approach, leveraging tools like DataFrames to handle data processing and structural navigation. However, transitioning from static text-level reasoning to a code-driven agent introduces two significant challenges: (i) Prohibitive search space for programmatic planning. Unlike static language reasoning for conventional table QA, translating high-level analytical intent (e.g., ’summarize by department’) into concrete code

[Figure 1]

[Figure 2]

Existing Table QA pipeline

###### Unstructured Tabular Properties Multi-hop Long-horizon Tasks

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Fact Reasoning

Numerical Computation

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Short, Direct, Simple Answer

[Figure 12]

[Figure 13]

Bi-directional Hierarchy in Rows/Columns

[Figure 14]

[Figure 15]

LLM

Split Merged Cells

Handle Missing Values

Chart Visualization

Report Analysis

(a) (b) (c)

- Figure 1. Existing Table QA pipelines (a) are limited to well-structured tables and shallow queries, and fail to handle unstructured tabular properties (b) and long-horizon analytical tasks (c), motivating our Deep Tabular Research.

- • Closed-Loop Agentic Framework: We introduce a principled framework that decouples macro planning from micro execution, treating reasoning as an iterative decisionmaking process.
- • Experience-Driven Optimization: We propose an expectation-aware selection mechanism and a structured memory graph that enable the agent to learn from execution feedback and mitigate error propagation.
- • Empirical Validation: We provide extensive evaluations on unstructured tabular benchmarks, demonstrating the effectiveness and efficiency of DTR in handling complex, real-world data layouts.

operations (e.g., df.groupby(), pd.pivot_table()) over unstructured tables involves a massive space of potential execution paths. Given the ambiguity of hierarchical headers and missing values, identifying the optimal sequence of operators is non-trivial. (ii) Errors could inevitably propagate during complex long execution. Execution exposes concrete errors and ambiguities, while there is a limited mechanism to properly learn from past execution outcomes, especially for failures, to guide future decisions.

In this paper, we propose a novel agentic framework for DTR that treats tabular reasoning as a continual decision process driven by execution experience. Our framework explicitly decouples high-level strategic planning from lowlevel execution, ensuring that reasoning is informed by accumulated feedback rather than rigid heuristics. Specifically, our approach comprises three key components. (i) A query-decomposed operator module maps natural language queries into a structured space of analytical operators, facilitating flexible composition; (ii) We then design an expectation-aware policy that identifies promising execution trajectories under uncertainty, balancing exploration and utility without exhaustive simulation; (iii) Finally, a siamese structured memory module records execution outcomes and failures, enabling the system to refine its planning strategy through experience. The memory is carefully evolved in a siamese mode inlucding both parameterized updates and abstracted textual experience. By grounding reasoning in verified micro operations and continuously adapting to feedback, our framework achieves robust error isolation and recovery across diverse tabular settings. Extensive experiments demonstrate that our approach consistently outperforms strong baselines on challenging unstructured benchmarks, establishing continual experience-driven execution as a superior foundation for deep tabular research.

### 2. Task Definition

Deep Tabular Research (DTR) generalizes the problem of answering complex, multi-step analytical queries over noncanonical tables. Formally, a DTR task is defined by the tuple (T ,Q,E,Y), representing the tabular domain, the query space, the execution environment, and the output space, respectively.

|Let T ∈ T be an unstructured table with flexible schemas, T may contain hierarchical headers, bidirectional row-column spans, and implicit semantic relations. Given a natural language query q ∈ Q that requires long-horizon reasoning (e.g., trend analysis, cross-region comparison, or recursive aggregation), the task is to produce a response y ∈ Y that is both factually accurate and computationally grounded.|
|---|

#### 2.1. Execution-Grounded Paradigm

Unlike traditional TableQA with a pure linguistic mapping f(q,T) → y, DTR defines reasoning as an interactive exploration within an execution environment E, where the agent incrementally executes operations and adapts its strategies.

#### Contributions:

• Task Formalization: We define the Deep Tabular Research (DTR) task, shifting the focus from simple TableQA to long-horizon, multi-hop analytical reasoning over unstructured, non-canonical tables.

• Latent Structural State: The true semantic structure of T (e.g., the exact scope of a hierarchical header) is latent and must be inferred through interaction since raw data rarely presents explicit hierarchical dependencies.

Meta Operation Decomposition

Execution Experience Memory

Seed Operation Bank Operation Map

Path-guided Code Execution

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

| | |
|---|---|
|AGG|GROUP|

[Figure 23]

[Figure 24]

Group by Dept, Product

User Query

Unstructured Tabular Data

Table Meta Information

[Figure 25]

User Query

|FILTER|
|---|

JOIN

Calculate Q1&Q4 Sales

[Figure 26]

[Figure 27]

LLM

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Table Meta

Filter Growth Rate <10%

…

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Operation Map

[Figure 38]

[Figure 39]

Macro Path Planner

P(π) ← P(π) ⋅ exp(*) * Penalize GROUP → AGG without validation

{r_query, r_cost, r_exec}

Abstracted Experience Condition:

|Path A: [GROUP, AGG, FILTER…]|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Expectation: +7.2

Candidate Paths Operation Map

Path Selection LLM with p-UCB

AGG before validation → failure

|Path B: [CLEAN, FILTER, GROUP, AGG…]|
|---|

[Figure 45]

Suggested Fix:

Optimal Path

Expectation: +8.5

Insert CHECK/CLEAN before AGG

- Figure 2. A sketched overview for our proposed Deep Tabular Research framework for complex unstructured tabular reasoning. DTR decomposes analytical intent into meta operations, plans executable macro paths via expectation-guided search, and executes them with an experience-aware memory that records structured execution feedback and updates planning policies across iterations.

- • Analytical Trajectory: To resolve q, a sequence of atomic analytical actions a = (a1,a2,...,aH) must be performed. Each ai represents a data-level transformation that maps an input data state to an intermediate result oi.
- • State Transition: The reasoning state at step t is defined

as St = {q,T,(a1,o1),...,(at−1,ot−1)}. The objective is to find a trajectory of actions such that the final outcome oH supports the generation of correct output y.

- 3. Deep Tabular Research

Bi-directional Header Identification. Real-world tables often contain headers along both rows and columns, frequently with multi-level spans. We identify header regions along each axis and resolve their scopes via span alignment, producing a bidirectional header structure. Each data cell is associated with both row-wise and column-wise semantic descriptors, jointly defining its contextual meaning. These headers may form nested hierarchies, which are explicitly organized into a graph structure to capture interactions between row and column semantics.

Meta Graph Construction. After extracting metadata from irregular table formats, the unstructured entities are organized into a structured graph denoted as GT = (VT,ET). Each node in the graph corresponds to a header or content element, and edges represent containment or hierarchical relationships. Due to the bi-directional nature of table headers, the same sub-item can simultaneously belong to both row-wise and column-wise parent nodes, forming a overlapping tree-like hierarchical structure. This graph explicitly preserves the organizational layout of the table and serves as a structured representation for downstream reasoning tasks. 3.2. Query-Guided Operation Mapping

DTR tackles long-horizon analytical reasoning over unstructured tables by formulating tabular reasoning as a closedloop decision process. Figure 2 provides an overview of the framework. We demonstrate the core idea and technical details from five main aspects hereunder.

#### 3.1. Tabular Comprehension and Structural Modeling

Given a raw table T (e.g., an Excel sheet) and a natural language query q, the first stage of DTR constructs a structured representation that captures both explicit and implicit table semantics. This representation serves as the foundation for downstream reasoning, enabling robust navigation of complex row–column relationships and hierarchical headers.

Given a natural language query q and the constructed table graph GT, we next determine a sequence of operations that can be executed over the graph to answer the query. Rather than directly reasoning over raw table cells, our approach relies on a predefined seed operation bank that encapsulates a diverse set of atomic operations commonly required for tabular reasoning. The seed operation bank contains a collection of basic operations such as filtering, group aggregation, and numerical sorting: O = {CLEAN,FILTER,GROUP,AGG,JOIN,SORT,...}.

Meta Information Extraction. Effective tabular reasoning requires a global understanding of table structure and salient semantic signals. We extract column and row headers, including sub-headers that encode hierarchical organization, as they provide concise summaries of how data is arranged. Beyond explicit headers, we identify implicit metadata such as measurement units, temporal or categorical markers, and aggregation indicators. Integrating both explicit and implicit signals yields a structured representation that captures the table’s overall organization and supports subsequent reasoning.

To align the query with executable operations, we leverage an LLM-based agent to perform decision making over both the decomposed sub-queries and the structured table graph. The graph GT is linearized and provided to

the agent in the form of relational triples, such as (table, has_column_header, column_header_description) and (column_header, has_child, child_header), which explicitly encode the hierarchical and containment relations within the table. Given each sub-query and the corresponding graph description, the agent selects a set of candidate operations from the seed operation bank that are most relevant to the reasoning intent. The selected operations for query q are denoted as O(q) = {o1,o2,...,oK}, forming the basis for subsequent execution and reasoning over the table graph. The following text box provides a concrete input example.

The first term encourages exploitation by favoring paths that have produced reliable intermediate or final results in previous executions. The second term promotes exploration by assigning higher scores to paths that are structurally plausible but have been executed fewer times, with the logarithmic normalization accounting for the overall execution budget. The hyperparameter α controls the trade-off between exploitation and exploration. Our framework selects and executes the path that maximizes the expectation-aware score E(π), enabling a balance between enhancing reasoning strategies and discovering alternative paths.

Operation Map Construction. We further construct an operation map that encodes dependencies and admissible orderings among operations. Rather than treating candidate actions as an unordered set, the operation map organizes them into a sequential path that respects logical and semantic constraints. Certain operations impose prerequisite contexts; for instance, AGG requires a well-defined grouping scope, while FILTER may be applied either before or after aggregation depending on whether it constrains raw values or aggregated results. These initial operation paths are subsequently refined through more sophisticated selection mechanisms to identify the most coherent sequence.

#### 3.3. Path Planning with Expectation-Aware Selection

Given the feasible operation paths constructed from the operation map, a path is defined as π = (o1,o2,...,oL), representing an ordered sequence of operations. DTR performs path planning to identify the most promising execution strategy for a given query. Rather than treating all admissible paths equally, we introduce an expectation-aware selection that evaluates candidate paths based on their anticipated utility. We enumerate a finite set of candidate paths {πi} by composing operations under dependency constraints and pruning invalid or redundant sequences. Importantly, path selection is not performed in a single pass. As operations are executed, DTR reflects on the intermediate execution results and iteratively updates its preference over candidate paths. This closed-loop refinement allows the planner to revise earlier decisions, and progressively focus on operation sequences that are more likely to yield correct outcomes.

Expectation-Aware Scoring. For each candidate path π, we maintain a set of path-level statistics that summarize its historical execution behavior, including an estimate of its expected return Rˆ(π), the number of times the path has been executed N(π), and a prior term P(π) that reflects structural plausibility or domain knowledge. Based on these terms, we define an expectation-aware score for each path:

E(π) = Rˆ(π) + α · P(π)

log π′ N(π′) 1 + N(π)

. (1)

Theoretical Boundedness. Assume the realized execution reward is bounded such that R(π) ∈ [0,Rmax], and the structural prior satisfies P(π) ∈ [0,1]. Since Rˆ(π) is an empirical estimate of R(π), it follows that Rˆ(π) ≤ Rmax. Therefore, the expectation score is upper bounded by:

N(π′). (2)

E(π) ≤ Rmax + α log

π′

This bound ensures that expectation values remain scaled and prevents unbounded optimism during exploration. Moreover, the exploration term for a fixed global execution budget is monotonically decreasing with respect to N(π):

α · P(π)

lim

N(π)→∞

log π′ N(π′) 1 + N(π)

= 0. (3)

As a result, paths that are sufficiently explored gradually shift from exploration-driven selection to exploitation based on empirical performance obtained from current executions.

Path Selection. The planner selects the top-k paths with the highest expectation scores for execution. Each selected path is then instantiated into an executable analytical program and applied to the table, producing intermediate or final results. Importantly, expectations are evaluated at the path level, since rewards in long-horizon tabular reasoning are inherently not decomposable and cannot be attributed to individual intermediate operations. Execution outcomes are evaluated against the query intent to obtain a scalar feedback signal, which is subsequently used to update the path-level statistics. Through this iterative process, DTR progressively refines its preference over operation paths, enabling more accurate and reliable planning for complex tabular tasks.

Iterative Interaction. During the execution of a selected operation path, DTR enables intermediate interactions with the LLM agent between consecutive operations. Specifically, before and after executing each operation, the agent is prompted to produce a discrete flag signal that characterizes the latest execution stage, i.d., [THINK] / [CODE]. The flag indicates whether the agent is engaged in coding, i.e. running executable code, or in thinking, such as summarizing

intermediate results and validating the ongoing analytical logic. Depending on the flag type, the corresponding execution code or reasoning summary is recorded as part of the execution trace. Prior to executing the next operation, this context is fed back to the agent, providing an explicit description of the current execution state and reasoning trajectory. This design allows the agent to adapt subsequent decisions based on both concrete execution results and evolving analytical understanding, facilitating coherent progression along the operation path.

#### 3.4. Siamese Experience-Guided Reflection

Beyond path-level planning, DTR further incorporates a siamese experience-guided execution mechanism that leverages feedback at two complementary levels to perform cross-path reasoning and consolidation. The first type is parameterized execution feedback, which captures concrete execution signals produced by running a selected operation path on the table. These signals include execution success or failure, and intermediate result validity. Such feedback provides fine-grained and specific supervision that reflects how well a particular operation performs on the given table.

In parallel, DTR maintains an abstracted experience channel that summarizes execution outcomes accumulated up to each timestamp. Instead of retaining raw execution traces, this channel distills higher-level patterns, such as which classes of operation paths tend to be effective under certain query structures or table organizations. These abstracted experiences are agnostic to specific table values, enabling transfer across instances and supporting more robust decision making in future executions. These two streams operate in a siamese manner: parameterized execution feedback informs immediate path refinement for the current query, while abstracted experience guides longer-term preference. Together, they form a closed-loop execution framework.

Parameterized Execution Feedback Signals. Given a selected operation path π = (o1,...,oL) and its instantiated execution program Pπ, DTR executes Pπ on the table and collects parameterized execution feedback signals that reflect the observed execution behavior. We denote the execution feedback as the following expression:

f(π) = fexec(π), ftime(π), ftype(π) , (4)

where each component evaluates a different aspect of execution behavior. The execution validity signal fexec(π) ∈ {0,1} indicates whether the program runs without errors. The execution time signal ftime(π) ∈ R+ records the time required to complete the execution of Pπ, reflecting the computational efficiency of the operation path. The type consistency signal metric ftype(π) ∈ {0,1} evaluates whether the intermediate and final results match the expected output format implied by the query and operation sequence, such

as whether a ranking operation yields an ordered list, or a numerical aggregation produces a scalar value. These signals are parameterized by the operation sequence, allowing DTR to distinguish between structurally similar paths that differ in execution behavior. The overall execution reward is computed as r(π) = ϕ f(π) , which is used to update path-level expectations and guide subsequent decisions.

Abstracted Experience Feedback. In addition to quantitative evaluation, abstracted experience is used to guide subsequent path execution. Rather than encoding concrete signals such as running time or output validity, this feedback summarizes semantic and strategic observations distilled from executed actions, preserving flexibility. For example, recurring aggregation failures may trigger the insertion of validation or cleaning operations prior to aggregation.

Query: What are the top-selling categories in Q3? Feedback Summary:

- - Path π1 first grouped seasons to compute total sales, then filtered Q3; filtering operation should be executed first to avoid computation of irrelevant data.
- - Error: category not found. Action: re-read table samples and derive sub-column product category.
- - Repeated sorting operations before and after grouping do not contribute and can be skipped. Execution Reflection: maintaining the correct filter-thenaggregate order; reading the sub-column product category; sorting operation placed at the end.

#### 3.5. Reflection-Driven Path Adaption

With real-time experience-guided reflection from the siamese mechanism, candidate operation paths are dynamically updated, leading to corresponding changes in their expectation scores. This allows DTR to progressively favor paths that demonstrate consistent effectiveness, while enabling subsequent operation selection to learn from accumulated reflections.

Continual Expectation Update. After executing an operation path π, DTR aggregates execution outcomes into a path-level reward R(π), reflecting the overall effectiveness of the reasoning trajectory. The estimated expected return is updated incrementally as

Rˆ(π) ← (1 − η) Rˆ(π) + η · R(π), (5)

where η ∈ (0,1] is a learning rate controlling the influence of newly observed execution outcomes. While R(π) is only defined for executed paths, Rˆ(π) is continuously updated for paths to be executed. Through this mechanism, feedback from one path could influence the expectations of other related paths. As defined in Equation 4, the real record of

Table 1. Full Comparisons over DTR-Bench to evaluate over accuracy, quality, and efficiency.

Accuracy Analysis Depth Feasibility Aesthetics Avg.

Total Output Tokens↓

Avg. LLM Calls

Runtime (s)↓

Win Rate Score Rate Win Rate Score Rate Win Rate Score Rate Win Rate Score Rate ↓ (Tie=0)↑ (Tie=0.5)↑ (Tie=0)↑ (Tie=0.5)↑ (Tie=0)↑ (Tie=0.5)↑ (Tie=0)↑ (Tie=0.5)↑

Methods

###### Table specific LLM

TableGPT2-7B 0.20 8.41 5.12 5.12 4.33 4.35 6.21 6.21 3.42 12,450 1.0 TableLLM-7B 0.15 6.22 3.84 3.84 3.10 3.12 4.55 4.55 3.15 10,820 1.0 StructGPT 0.10 4.15 2.10 2.10 1.85 1.86 2.30 2.30 4.88 8,420 1.0

###### Common LLM

DeepSeek-V3 1.21 30.2 21.72 21.72 20.60 20.60 31.23 31.23 38.64 49,271,1 1.0 DeepSeek-V3.2 1.28 33.52 25.22 25.22 24.42 24.43 36.63 36.83 39.02 52,446,3 1.0

###### Workflow

ST-Raptor 0.62 22.40 6.00 6.00 7.41 7.41 12.40 12.40 999.16 31,077,2 9.2 TreeThinker 1.83 31.00 22.82 22.82 21.42 21.43 36.83 36.83 140.78 17,067,00 5.1 Code Loop 1.32 27.50 9.40 9.51 14.81 14.92 20.42 20.42 175.84 75,204,2 8.8

DTR

DTR (DS-v3) 1.93 37.53 30.23 30.23 27.62 27.64 42.74 42.64 62.09 81,754,2 4.7

execution-level factors such as running time and output format consistency is reflected in the reward r(π). Besides, abstracted experience is utilized to guide adjustments in operation selection for other candidate paths, thereby modifying their expected returns. For instance, structural modifications including operation insertions or removals lead to corresponding changes in execution count N(π). This allows our framework to dynamically prioritize paths that are structurally aligned with previously successful signals.

Closed-Loop Optimization. DTR performs closed-loop optimization by path planning, execution, and expectation update in a progressive manner. At each iteration t, the agent selects the candidate path with the highest expectation scores under real-time estimate Et(π). These paths are executed to produce intermediate or final results, whose execution feedback is then used to update both parameterized signals and abstracted experience. To determine the final answer, DTR aggregates outputs from multiple executed paths. Let A = {a1,...,am} denote the set of candidate answers and the final answer a∗ is selected by majority agreement:

m

a∗ = arg max a∈A

I(ai = a), (6)

i=1

where only answers satisfying the query requirements and correct formats are considered. This voting-based selection improves robustness against individual execution errors and reinforces the reliability supported by multiple trials.

### 4. Experimental Analysis

We evaluate Deep Tabular Research (DTR) on a diverse set of unstructured tabular reasoning benchmarks, covering factual lookup, numerical computation, structural understanding, long-horizon data analysis, and visualization tasks. Our evaluation focuses on both task accuracy and end-to-end analytical effectiveness, reflecting the real-world demands of complex spreadsheet-based workflows. The

details of the adopted benchmarks, state-of-the-art baselines and evaluation protocols are introduced in Appendix C.

#### 4.1. Main Results

Tables 4 and Table 2 report the main experimental results on DTR-Bench and RealHitBench, respectively. Though both benchmarks target table-centric reasoning, they emphasize fundamentally different aspects of model capability. For fair comparison, we allow non-code mode for the RealHitBench dataset since it focuses on answer-level correctness with concise formats. On DTR-Bench, DTR achieves the strongest overall performance across all dimensions, including accuracy, analysis depth, feasibility, and aesthetics. Notably, the improvements are consistent under both strict win-rate and more tolerant score-rate evaluation, indicating that DTR does not merely outperform baselines in marginal cases, but produces systematically higher-quality outputs.

These gains cannot be attributed solely to stronger backbone models. Compared to pure LLM baselines, DTR demonstrates substantially deeper analysis and higher feasibility. This suggests that unconstrained language generation, even in large models, has difficulty reliably organizing multi-step table operations into executable workflows. Conversely, agent-based frameworks such as ST-Raptor, TreeThinker, and Code Loop exhibit improved structural reasoning but incur significant computational overhead and instability due to extensive branching and repeated trial executions. DTR stands out in this trade-off space. By explicitly planning over macro-level operation paths and selecting candidates based on learned expectations derived from historical experience, DTR avoids exhaustive search while still preserving long-range reasoning coherence. As a result, it produces more complete and visually coherent analytical reports than pure LLMs, while remaining substantially more efficient than tree-based or loop-based agent frameworks.

Table 2. Comparisons over the RealHitBench dataset for five different task types.

Fact Checking

Numerical Reasoning

Structure Comprehending

Data Analysis

Chart/Report Generation

Methods

EM↑ F1↑ EM↑ F1↑ EM↑ F1↑ LLM-EVAL↑ ROUGE↑ PASS@1↑ ECR↑

TableGPT2-7B 46.10 53.80 29.31 39.81 48.23 56.68 62.76 33.25 32.47 67.53 TableLLM-7B 38.25 44.12 22.40 31.65 41.10 49.34 55.80 28.42 18.20 42.15 StructGPT 25.40 32.15 14.55 20.80 30.25 38.60 42.33 19.50 5.12 12.44

GPT4o 43.39 51.87 27.63 36.68 42.68 52.89 65.24 33.10 10.39 25.32 DeepSeek-v3 57.21 53.42 47.05 50.61 43.31 74.63 61.40 34.76 9.09 24.68

Code Loop (DeepSeek-v3) 48.19 56.49 42.93 49.68 44.19 51.95 62.51 33.73 20.78 39.60 Code Loop (Qwen3-1.7B) 6.91 7.65 4.02 5.75 49.24 53.08 17.97 17.29 0.00 0.60 Code Loop (Qwen3-4B) 16.61 19.76 10.25 13.00 32.85 30.30 25.83 18.17 3.25 14.30 DTR (Qwen3-1.7B) 17.93 23.50 13.75 19.01 17.17 27.88 40.28 16.44 5.16 21.94 DTR (Qwen3-4B) 30.42 37.44 22.05 30.13 32.6 43.32 53.88 20.15 8.39 30.32 DTR (DeepSeek-v3) 58.22 64.47 55.51 61.98 56.57 77.95 70.90 38.67 52.69 100.00

Beyond quality, DTR also demonstrates favorable efficiency–performance balance. While workflow-based baselines often require an order of magnitude more LLM calls and runtime to achieve competitive analytical depth, DTR attains superior results with fewer calls and more predictable execution cost. This suggests that reasoning at the level of operation sequences, rather than token-level or step-level exploration, leads to more stable and scalable analytical behavior for complex table reasoning tasks.

Table 3. Architecture ablation study of each components in DTR.

Configuration Meta QDO Exp. Abst. Acc. Anal. Feas. Aesth.

+ Meta Info ✓ × × × 34.8 27.1 25.6 38.2 + QDO ✓ ✓ × × 36.2 28.8 26.7 40.5 + Expectation ✓ ✓ ✓ × 37.1 29.6 27.2 41.8 DTR Full ✓ ✓ ✓ ✓ 37.5 30.2 27.6 42.7

#### 4.2. Ablation Studies

- 4.2.1. COMPONENT CONTRIBUTIONS

We conduct a systematic ablation study to quantify the marginal contribution of each constituent module within our framework. Beginning with a pure LLM baseline (DeepSeek-V3), we progressively integrate (i) tabular meta information, (ii) query-to-operation decomposition, (iii) expectation-aware macro path selection informed by historical execution feedback, and (iv) abstracted execution experience. Table 3 summarizes the results across multiple evaluation dimensions, including Accuracy, Analysis Depth, Feasibility, Aesthetics, and the mean number of LLM calls.

The results demonstrate that DTR yields consistent performance gains over the baseline, achieving a total accuracy improvement of 4.0 percentage points (33.5% → 37.5%). Among the individual components, tabular metainformation and query decomposition provide the most significant uplift (+1.3 and +1.4 points, respectively), which suggests that explicit structural grounding and operation-

level intent modeling are fundamental to unstructured tabular reasoning. The incorporation of historical feedback for macro path selection further enhances accuracy by 0.9 points, indicating that high-level planning benefits substantially from past experience even in the absence of low-level code simulations. Finally, the inclusion of abstracted execution experience provides an additional 0.4 point gain, validating the utility of distilling raw execution outcomes into reusable knowledge for novel problem instances.

Table 4. Exploration over prompting strategy for think and code.

Prompt Strategy Accuracy Analysis Code Error Avg Depth Rate Calls

No [THINK] (Direct) 35.2 27.8 42.3% 5.8 Simple [THINK] Hint 36.4 28.9 35.6% 5.2 [THINK]+[CODE] 37.5 30.2 28.4% 4.78

Multi-stage Reflection 37.2 29.8 26.1% 5.5

4.2.2. PROMPTING STRATEGY ANALYSIS

We observe diminishing returns as the system complexity increases. While each module contributes positive gains, the incremental improvements become smaller with additional components. This trend aligns with our objective of decoupling foundational capabilities, such as table comprehension and decomposition, from iterative refinement processes like experience-aware planning and memory abstraction.Given DTR’s step-wise paradigm, code reliability is critical to both correctness and computational efficiency. We therefore evaluate various strategies during the execution phase. Table 4 compares four configurations, including direct code generation without explicit reasoning steps, a lightweight reasoning hint, our default structured [THINK]+[CODE] prompting mode, and a multi-stage reflection strategy.

The structured [THINK]+[CODE] scheme well balances between performance and efficiency. It attains the highest accuracy (37.5%) and analysis depth (30.2) while simulta-

40

2.00

DTR Performance

Plateau (6 calls)

DTR Sweet Spot (4.78 calls, 37.5%)

Plateau Threshold

DTR Sweet Spot

1.75

Code Loop

38

Marginal Gain

1.50

36

| |
|---|

AccuracyScoreRate(%)

1.25

MarginalGain(%/call)

34

1.00

32

| |
|---|

0.75

Code Loop (8.8 calls, 27.5%)

30

0.50

| |
|---|

0.25

28

| |
|---|

| |
|---|

0.00

26

0 2 4 6 8 10

LLM Call Budget

Figure 3. LLM call budget analysis. The blue curve shows performance (left y-axis) and the purple curve shows marginal gain (right y-axis). DTR’s 4.78-call configuration red star) achieves optimal efficiency by avoiding the plateau region.

neously reducing the code error rate from 42.3% to 28.4% compared to the direct generation baseline. Furthermore, this approach optimizes the average number of LLM calls (4.78 versus 5.8), which implies that isolating semantic reasoning from code emission enhances execution stability and mitigates redundant retry cycles. Although multi-stage reflection achieves a slightly lower error rate (26.1%), it introduces substantial runtime overhead (48.6s versus 42.1s), suggesting that additional reflection yield marginal benefits.

#### 4.3. Efficiency and Scalability

We assess the computational efficiency of DTR by comparing model performance and the frequency of LLM calls, using the call budget as a proxy for inference-time cost. Figure 3 presents the performance trajectory as the budget increases and the marginal utility per additional call.

The analysis reveals three distinct regimes. In the rapid growth phase (1–3 calls), performance scales sharply with high marginal gains averaging approximately +1.45% per call, demonstrating that even sparse iterative interactions significantly bolster long-horizon tabular reasoning. In the transitional regime (3–6 calls), improvements persist but decelerate to approximately +0.45% per call as the system nears its performance ceiling. Beyond 6 calls, each additional call yields less than 0.15% performance gain. DTR operates at an average of 4.78 calls, effectively positioning it within the optimal transition region where quality and computation are balanced. In contrast, the CodeLoop baseline exhibits a failure mode characterized by over-iteration. Despite exhausting a significantly higher budget (8.8 calls), it achieves only 27.5% accuracy, suggesting that unconstrained execution without a strategic selection mechanism can propagate errors and degrade overall performance. These findings justify the importance of DTR’s budgeted design and its expectation-aware path selection.

#### 4.4. Case Study: Planning Dynamics

To further investigate the internal dynamics of DTR, we visualize the evolution of macro path selection over 500

35

|[Figure 46]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 47]

8 10 15 19 27 32 36 40 43 46

L F G

30

14 17 18 19 15 13 11 10 9 7

L S Se

25

11 12 16 18 20 21 21 21 20 19

- L F A

- L G A

SelectionFrequency(%)

OperatorPath

20

17 19 15 11 9 7 6 4 4 3

14 12 9 6 5 3 1 1 0 0

L M F

15

11 10 11 11 12 13 14 16 16 14

L F S

10

19 12 9 6 3 1 1 0 0 0

L Se F

5

6 6 7 8 9 9 9 7 9 11

L P A

0

Batch10 (Q451-500)

Batch1 (Q1-50)

Batch2 (Q51-100)

Batch3 (Q101-150)

Batch4 (Q151-200)

Batch5 (Q201-250)

Batch6 (Q251-300)

Batch7 (Q301-350)

Batch8 (Q351-400)

Batch9 (Q401-450)

Batch Steps

Figure 4. Path selection evolution across 10 batches. Colors changing from light blue to deep purple indicate both exploration and exploitation, respectively.

queries partitioned into 10 sequential batches. Figure 4 depicts the selection frequency across eight operator paths via a heatmap, where increased density represents higher preference. During the initial batches (1–3), the system engages in broad exploration by selecting candidate paths with near-uniform frequency (3–7% each). This behavior reflects high initial uncertainty and the necessity of empirical evaluation under diverse tabular conditions. There exhibits a clear convergence pattern by the fifth batch. Path 0 (LOAD → FILTER → GROUPBY) increases in frequency from 3% to 28%, while paths with consistently low rewards are automatically pruned, e.g., Path 7 which declines to near zero.

Notably, DTR avoids collapsing into a single deterministic strategy. In the final batches (8–10), the primary path stabilizes at roughly 31% while Path 5 remains a robust secondary option at 11%. The remaining probability mass is distributed across various alternative paths to maintain approximately 10–15% exploration. This equilibrium indicates that DTR successfully adapts through execution experience by prioritizing high-return strategies while retaining sufficient diversity for context-sensitive reasoning, demonstrating a balanced exploitation-exploration trade-off.

### 5. Conclusions

In this paper, we formally define Deep Tabular Research as a new paradigm of long-horizon analytical reasoning tasks over unstructured tables. We propose a principled agentic framework, i.e., DTR, that treats tabular reasoning as a closed-loop decision-making process grounded in executable operations. DTR jointly optimizes strategic planning and operational execution through query-aware operator abstraction, expectation-driven path selection, and experience-based memory refinement, enabling robust reasoning under structural ambiguity and execution uncertainty. Extensive experiments on challenging unstructured tabular benchmarks demonstrate consistent improvements over SOTA baselines in reasoning accuracy, execution stability, and efficiency. These results highlight the necessity of separating high-level planning from low-level execution and

establish execution-driven, experience-aware reasoning as a foundational paradigm for deep tabular research.

### Broader Impact

We present a framework for advancing machine learning systems in complex tabular reasoning and analytical tasks for social good based on large language models. Our primary goal is to improve the robustness, interpretability, and effectiveness of automated reasoning over structured data, which is a core and well-established research direction in machine learning. Potential positive societal impacts of this work include improved automation and decision support in data-driven domains such as scientific analysis, business intelligence, and public data reporting, where accurate and transparent table-based reasoning is essential. By enabling models to better understand complex table structures and perform multi-step analytical reasoning, we may help reduce manual effort and errors in data analysis workflows.

At the same time, we do not anticipate significant new ethical risks beyond those commonly associated with large language models and automated data analysis systems. As with prior work in this area, misuse could arise if such systems are deployed without appropriate human oversight, particularly in high-stakes settings involving sensitive or biased data. These risks are not unique to our method and are best addressed through established practices such as responsible deployment, data governance, and human-in-the-loop validation. We believe this work contributes incrementally and responsibly to the broader field of machine learning, without introducing novel ethical concerns that would require special mitigation beyond existing standards.

### References

Akhtar, M., Shankarampeta, A., Gupta, V., Patil, A., Cocarascu, O., and Simperl, E. Exploring the numerical reasoning capabilities of language models: A comprehensive analysis on tabular data. arXiv preprint arXiv:2311.02216, 2023.

Dagan, G., Keller, F., and Lascarides, A. Dynamic planning with a llm. arXiv preprint arXiv:2308.06391, 2023.

Ge, Y., Romeo, S., Cai, J., Sunkara, M., and Zhang, Y. Samule: Self-learning agents enhanced by multi-level reflection. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 16602–16621, 2025.

Gong, H., Sun, Y., Feng, X., Qin, B., Bi, W., Liu, X., and Liu, T. Tablegpt: Few-shot table-to-text generation with table structure reconstruction and content matching. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 1978–1988, 2020.

Huang, X., Liu, W., Chen, X., Wang, X., Wang, H., Lian, D., Wang, Y., Tang, R., and Chen, E. Understanding the planning of llm agents: A survey. arXiv preprint arXiv:2402.02716, 2024.

Katsis, Y., Chemmengath, S., Kumar, V., Bharadwaj, S., Canim, M., Glass, M., Gliozzo, A., Pan, F., Sen, J., Sankaranarayanan, K., et al. Ait-qa: Question answering dataset over complex tables in the airline industry. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Track, pp. 305–314, 2022.

Kulkarni, A., Dixit, K., Srikumar, V., Roth, D., and Gupta, V. Llm-symbolic integration for robust temporal tabular reasoning, 2025. URL https://arxiv.org/abs/2506. 05746.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Lu, J., Qin, J., Qiao, L., Li, Y., Dai, X., Ke, B., He, J., Qiao, R., Yin, D., Sun, X., et al. Youtu-llm: Unlocking the native agentic potential for lightweight large language models. arXiv preprint arXiv:2512.24618, 2025.

Perzina, R. and Ramík, J. Microsoft excel as a tool for solving multicriteria decision problems. In International Conference on Knowledge-Based Intelligent Information & Engineering Systems, 2014. URL https: //api.semanticscholar.org/CorpusID:34854159.

Ren, W., Zhao, T., Huang, Y., and Honavar, V. Deep learning within tabular data: Foundations, challenges, advances and future directions. arXiv preprint arXiv:2501.03540, 2025.

Renze, M. and Guven, E. Self-reflection in llm agents: Effects on problem-solving performance. arXiv preprint arXiv:2405.06682, 2024.

Sarkar, S. and Lausen, L. Testing the limits of unified sequence to sequence llm pretraining on diverse table data tasks. arXiv preprint arXiv:2310.00789, 2023.

Shi, W., Xu, R., Zhuang, Y., Yu, Y., Zhang, J., Wu, H., Zhu, Y., Ho, J. C., Yang, C., and Wang, M. D. Ehragent: Code empowers large language models for complex tabular reasoning on electronic health records. ArXiv, abs/2401.07128, 2024. URL https://api. semanticscholar.org/CorpusID:273324122.

Singha, A., Cambronero, J., Gulwani, S., Le, V., and Parnin, C. Tabular representation, noisy operators, and impacts on table structure understanding tasks in llms. arXiv preprint arXiv:2310.10358, 2023.

Somvanshi, S., Das, S., Javed, S. A., Antariksa, G., and Hossain, A. A survey on deep tabular learning. arXiv preprint arXiv:2410.12034, 2024.

Song, C. H., Wu, J., Washington, C., Sadler, B. M., Chao, W.-L., and Su, Y. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2998–3009, 2023.

Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., and Huang, J. A survey on large language model-based agents for statistics and data science. ArXiv, abs/2412.14222,

- 2024. URL https://api.semanticscholar.org/ CorpusID:274859246.

Tang, Z., Niu, B., Zhou, X., Li, B., Zhou, W., Wang, J., Li, G., Zhang, X., and Wu, F. St-raptor: Llm-powered semi-structured table question answering. Proceedings of the ACM on Management of Data, 3(6):1–27, 2025.

Wang, Z., Zhang, H., Li, C.-L., Eisenschlos, J. M., Perot, V., Wang, Z., Miculicich, L., Fujii, Y., Shang, J., Lee, C.-Y., et al. Chain-of-table: Evolving tables in the reasoning chain for table understanding. arXiv preprint arXiv:2401.04398, 2024.

Wölflein, G., Ferber, D., Truhn, D., Arandjelovic, O., and Kather, J. N. Llm agents making agent tools. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26092–26130, 2025.

Wu, P., Yang, Y., Zhu, G., Ye, C., Gu, H., Lu, X., Xiao, R., Bao, B., He, Y., Zha, L., et al. Realhitbench: A comprehensive realistic hierarchical table benchmark for evaluating llm-based table analysis. arXiv preprint arXiv:2506.13405, 2025.

Wu, X., Yang, J., Chai, L., Zhang, G., Liu, J., Du, X., Liang, D., Shu, D., Cheng, X., Sun, T., Niu, G., Li, T., and Li, Z. Tablebench: A comprehensive and complex benchmark for table question answering. ArXiv, abs/2408.09174, 2024. URL https://api. semanticscholar.org/CorpusID:271902839.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., and Huang, G. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024a.

Zhao, Y., Mi, B., Qi, Z., Nan, L., Guo, M., Cohan, A., and Radev, D. Openrt: An open-source framework for reasoning over tabular data. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 336– 347, 2023.

Zhao, Y., Chen, L., Cohan, A., and Zhao, C. TaPERA: Enhancing faithfulness and interpretability in long-form table QA by content planning and execution-based reasoning. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 12824–12840, August 2024b.

Zheng, M., Feng, X., Si, Q., She, Q., Lin, Z., Jiang, W., and Wang, W. Multimodal table understanding. arXiv preprint arXiv:2406.08100, 2024.

Zhu, J.-P., Cai, P., Xu, K., Li, L., Sun, Y., Zhou, S., Su, H., Tang, L., and Liu, Q. Autotqa: Towards autonomous tabular question answering through multi-agent large language models. Proc. VLDB Endow., 17:3920–3933, 2024. URL https://api.semanticscholar.org/CorpusID: 272727897.

### A. DTR Algorithm

Algorithm 1 Deep Tabular Research Require: Query q ∈ Q, Table T ∈ T , Execution Environment E, Learning Rate α, Number of candidate paths K Ensure: Answer y ∈ Y

- 1: // Step 1: Meta Operation Decomposition (MOD)
- 2: Extract table metadata Tmeta from T
- 3: Mq ← MOD(q, Tmeta) ▷ Set of high-level meta operations
- 4: // Step 2: Candidate Macro Path Construction
- 5: Initialize candidate path set Π ← {}
- 6: for k = 1 to K do
- 7: πk ← generate_candidate_path(Mq) ▷ Ordered sequence of meta operations
- 8: Π ← Π ∪ {πk}
- 9: end for
- 10: // Step 3: Iterative Planning and Execution
- 11: while not all paths executed or converged do
- 12: for each path π ∈ Π do
- 13: Compute expectation score E(π):

E(π) = Rˆ(π) + c · P(π)

log π′ N(π′) 1 + N(π)

- 14: end for
- 15: Select top-ranked path(s) π∗ according to E(π)
- 16: context ← initial execution state in E
- 17: for each meta operation m in π∗ do
- 18: o,r ← execute(m, context, E) ▷ Execute and observe outcome and reward
- 19: Update Execution Experience Memory: D ← D ∪ {(m,context,o,r)}
- 20: Update path reward:

Rˆ(π∗) ←

N(π∗)Rˆ(π∗) + r N(π∗) + 1

, N(π∗) ← N(π∗) + 1

- 21: if execution failed or outcome invalid then
- 22: Replan: π∗ ← Macro Path Planner(Mq, D)
- 23: break ▷ Restart path execution with revised plan
- 24: end if
- 25: Update context according to o
- 26: end for
- 27: end while
- 28: // Step 4: Return final answer
- 29: y ← extract_answer(π∗, D)
- 30: return y

### B. DTR-Bench Dataset Curation

We introduce DTR-Bench (Deep Tabular Research Benchmark), a specialized benchmark for evaluating deep analytical reasoning capabilities over tabular data. Unlike existing table QA benchmarks that focus on simple fact retrieval or singlestep reasoning, DTR-Bench emphasizes complex statistical analysis tasks that reflect real-world data science workflows. The benchmark comprises 500 scenario-driven question-answer pairs from diverse domains, each requiring sophisticated analytical reasoning such as correlation analysis, inequality measurement, anomaly detection, and statistical hypothesis testing. This benchmark is derived from a curated selection of Excel spreadsheets sourced from RealHitBench, encompassing various domains including Economy, Business, and Education.

#### B.1. Scenario-Driven Question Generation

A key innovation of DTR-Bench is the scenario-driven question generation approach. Each question is grounded in a realistic user persona with domain-specific analytical needs, ensuring that questions reflect authentic data science tasks rather than artificial academic exercises.

User Persona Design We define 8 distinct user personas, each representing a real-world role that regularly performs deep tabular analysis:

Table 5. User Persona Distribution in DTR-Bench

User Persona Questions Primary Analysis Focus

Social Researcher 105 Inequality analysis (Gini coefficient, Lorenz curves), demographic patterns Student/Researcher 88 Statistical hypothesis testing (ANOVA), effect size calculation (Cohen’s d, η2) Government Staff 62 Longitudinal policy impact assessment, equity-efficiency trade-offs Data Analyst 61 Dimensionality reduction, cohort analysis, z-score-based anomaly detection Business Owner 55 Profitability decomposition, BCG matrix classification, break-even analysis Procurement Manager 53 Supplier concentration (HHI), risk-performance matrices Sales Manager 40 Multi-dimensional performance analysis, trend detection Investor 36 Risk-return analysis, efficient frontier identification, valuation ratios

Question Template Design Each persona has 4 specialized question templates requiring analytical reasoning. Templates are mainly designed to: a. Reference specific columns using quoted identifiers (e.g., "Revenue") b. Specify analytical methods (correlation, ANOVA, Gini coefficient, HHI, etc.) c. Request structured outputs (rankings, statistical measures, interpretations). Table 6 illustrates representative question templates for each persona type.

Table 6. Representative Question Templates by Persona

Persona Analysis Type Template Example Investor Risk-Return Analysis Conduct a risk-return analysis across all {Category} by examining the rela-

tionship between {NumericCol1} (returns) and {NumericCol2} (risk metrics). Identify the efficient frontier segments.

Social Researcher Inequality Analysis Measure the distribution of {NumericCol} across {Category} using Gini coefficient and decile ratios. Investigate structural factors driving inequality. Data Analyst Anomaly Detection Use statistical methods (z-scores) to identify unusual patterns in {NumericCol} across {Category}. Investigate anomalies and assess impact.

Student/Researcher Statistical Testing Conduct rigorous statistical analysis of {NumericCol} across {Category}: test for normality, perform ANOVA/Kruskal-Wallis tests, and calculate effect sizes.

##### B.2. Quality Assurance To ensure benchmark quality, we implement multiple validation steps:

- • Column Validity Filtering: Columns with invalid names (e.g., “Unnamed:”, purely numeric headers, CJK characters) are excluded
- • Template-Table Compatibility: Templates requiring specific column types (numeric, categorical, temporal) are only instantiated when suitable columns exist
- • Time-Series Validation: Templates involving trend analysis are skipped for tables without detectable temporal columns
- • Duplicate Prevention: Questions are deduplicated using normalized string comparison
- • Answer Verification: All reference answers are computed programmatically from actual table data, ensuring reproducibility

Answer Key Points (KeyPoints) for Evaluation In addition to validating that reference answers are computed from the underlying table data, we attach an AnswerKeyPoints field to each instance as explicit, machine-checkable grading criteria.

This enables fine-grained evaluation (e.g., key-point coverage) beyond string matching.

Answer: Gini coefficient of total Revenue across Region: 0.342... AnswerKeyPoints: "Aggregate the numeric metric by category.", "Compute an inequality metric such as the Gini coefficient.", "Report concentration (e.g., top vs bottom shares) and top categories."

### C. Implementation Details

#### C.1. Benchmark Datasets

DTR-Bench: Long-Horizon Analytical Queries. To profoundly evaluate the performance for complex and long-horizon tabular reasoning datasets, we construct an additional DTR-Bench based on tables in RealHitBench. Using table meta information and expert-designed templates combined with DeepSeek-3.2, we generate 500 long-form analytical queries that require multi-step reasoning and execution. These queries span five categories: Analysis: multi-stage aggregation and interpretation. Visualization: chart generation and visual comparison. Calculation: chained numerical computation. Comparison: cross-group or temporal comparison. Conditional Calculation: conditional aggregation and filtering.

Each query typically requires planning over multiple operations and intermediate execution states, making them unsuitable for shallow or single-pass reasoning approaches.

For the evaluation, we report two following metrics. Win Rate: proportion of instances where a model outperforms baselines (ties counted as 0). Score Rate: proportion of instances where a model is not worse than baselines (ties counted as 0.5).

RealHitBench. We adopt RealHitBench (Wu et al., 2025) as our primary benchmark, a large-scale dataset designed for evaluating reasoning over real-world, unstructured tables. RealHitBench contains tables with heterogeneous layouts, including merged cells, bidirectional and hierarchical headers, missing values, and implicit semantic regions. The benchmark categorizes tasks into multiple reasoning types and provides fine-grained evaluation protocols.

We evaluate DTR on the following RealHitBench task categories: Fact Checking: verifying factual statements grounded in table entries. Numerical Reasoning: performing arithmetic and aggregation over table values. Structure Comprehension: understanding table organization, header hierarchy, and alignment. Data Analysis: multi-step analytical reasoning requiring aggregation, comparison, and synthesis. Visualization: generating charts or structured visual outputs based on tabular data.

For Fact Checking, Numerical Reasoning, and Structure Comprehension, we report Exact Match (EM) and F1 scores following the benchmark protocol. Data Analysis tasks are evaluated using LLM-based evaluation and ROUGE metrics to assess semantic correctness and completeness. Visualization tasks are evaluated using Execution Correctness Rate (ECR) and Pass@1, measuring whether the generated visualization code executes successfully and produces the expected output.

#### C.2. Baselines

We compare DTR against a diverse set of baselines spanning table-specialized models, general-purpose large language models, and agent-based reasoning frameworks (Zheng et al., 2024; Lu et al., 2025). Table-specific Models. TableGPT (Gong et al., 2020): a representative table-focused language model designed for structured table question answering. Generalpurpose LLMs. DeepSeek-3.2: a strong open-source large language model with competitive reasoning capabilities; Agentic Frameworks. ST-RAPTOR (Tang et al., 2025): a retrieval-augmented agent framework for structured reasoning; Tree Thinker (Wu et al., 2025): a tree-based reasoning agent that explores multiple reasoning branches; CodeLoop: we craft a straightforward execution-centric agentic framework that iteratively generates and debugs code. All baselines are evaluated under comparable settings, with access to the same table inputs and query information. Unless otherwise specified, we use publicly released model checkpoints and default decoding configurations. The base LLMs we adopted is Qwen3 1.7B&4B (Yang et al., 2025) and DeepSeek V3 (Liu et al., 2024).

### D. Related Work

- D.1. Deep Research and Agentic Reasoning

Recent advances in large language models have aroused growing interest in agentic systems that perform multi-step reasoning through interaction with external tools, environments, or intermediate states (Zhao et al., 2024a; Huang et al., 2024; Gong et al., 2020). A line of deep research work often focuses on long-horizon problem solving, where models iteratively plan and revise their strategies based on intermediate feedback (Wölflein et al., 2025). Typical examples include agents augmented with external tools, systems that separate planning from execution, and methods that rely on self-reflection or self-correction across multiple steps (Renze & Guven, 2024; Dagan et al., 2023). Notably, many existing agent-based approaches emphasis on language-level planning and reasoning (Ge et al., 2025; Song et al., 2023). While such approaches have demonstrated strong performance on structured benchmarks, they typically assume reliable intermediate steps and make limited use of execution feedback. In contrast, our work treats deep search as a continual process, where accumulated action experience guides path selection and enables more robust behavior.

- D.2. Tabular Reasoning and Table Question Answering

Tabular reasoning has been extensively studied in the context of table question answering, semantic parsing, and data analysis (Zhao et al., 2023). Early approaches typically focused on mapping natural language queries to logical forms or executable programs over well-structured tables with clean schemas and regular layouts (Akhtar et al., 2023; Perzina & Ramík, 2014). Recent methods leverage large language models to perform tabular reasoning in a more flexible manner through straightforwardly interpreting table contents. Such pipelines reason over tables using code generation, textual serialization, or prompting strategies (Wang et al., 2024; Zhu et al., 2024; Shi et al., 2024). Existing work on tabular reasoning spans a diverse set of tasks, including fact checking, numerical reasoning, data analysis, and table-based visualization (Wu et al., 2024), where models are required to interpret structured or semi-structured information and perform reasoning over rows, columns, and cell values (Sun et al., 2024). Despite substantial progress, chain-of-thought prompting and single-pass code generation can be sensitive to early reasoning decisions in complex tabular analysis. While hybrid symbolic approaches improve robustness, they lack mechanisms for iterative decision making with accumulated experience (Kulkarni et al., 2025).

Our work bridges this gap by introducing a unified framework for Deep Tabular Research that integrates operator-level execution with continual experience-driven decision making. By explicitly modeling path selection, execution, and memory as interacting components, the proposed approach extends prior reasoning systems to handle complex, unstructured tables requiring iterative analysis and revision.

### E. Case Studies: Real Trajectories

- E.1. Case Study 1: Pie Chart - Employment Distribution

- E.1.1. TASK

Question: Using the data, draw a pie chart to illustrate the distribution of employment between agriculture and nonagriculture sectors in 1984.

Question Type: Visualization Table Context: Employment statistics table with Year, Agriculture employment, and Nonagricultural industries columns.

- E.1.2. GENERATED CODE

import pandas as pd import matplotlib.pyplot as plt

Extract data for 1984 (row index 5) agriculture = 2668 non_agriculture = 56423

Prepare data for pie chart

sectors = [’Agriculture’, ’Non-Agriculture’] employment = [agriculture, non_agriculture]

Verify extraction print("Sectors:", sectors) print("Employment:", employment) assert len(employment) > 0, "Employment data cannot be empty!" assert all(isinstance(e, (int, float)) for e in employment), "Employment values must be numbers!"

Create pie chart plt.figure(figsize=(8, 8)) plt.pie(employment, labels=sectors, autopct=’%1.1f%%’, startangle=90, colors=[’ff9999’,’66b3ff’]) plt.title(’Employment Distribution in 1984 ’ + ’(Agriculture vs Non-Agriculture)’) plt.tight_layout() plt.show()

- E.1.3. GENERATED VISUALIZATION

[Figure 48]

Figure 5. Employment Distribution in 1984: Agriculture (4.5%) vs Non-Agriculture (95.5%)

- E.1.4. QUALITY ANALYSIS Strengths:

- • ✓Correctly extracted data from 1984 row
- • ✓Clean, executable code with data validation (assertions)
- • ✓Appropriate visualization aesthetics (colors, sizing, labels)

- • ✓Added percentage labels using autopct for clarity
- • ✓Clear title explaining what the chart shows

Key Insight: The chart clearly shows non-agricultural employment dominated 1984 with 95.5%, while agriculture accounted for only 4.5%, illustrating the economic transformation towards industrial and service sectors.

- E.2. Case Study 2: Line Chart - Employment Trends

- E.2.1. TASK

Question: Please help me create a line chart to compare the trends of total employed population and total unemployed population from 1983 to 1990.

Question Type: Visualization Table Context: Employment statistics with yearly data including total employed and unemployed counts.

- E.2.2. GENERATED CODE

import pandas as pd import matplotlib.pyplot as plt

Extract data from table years = [1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990] employed = [56787, 59091, 59891, 60892, 62107, 63273, 64315, 65104] unemployed = [6260, 4744, 4521, 4530, 4101, 3655, 3525, 3906]

Verify extraction print("Years:", years) print("Employed:", employed) print("Unemployed:", unemployed) assert len(years) == len(employed) == len(unemployed), "Data length mismatch!" assert all(isinstance(y, (int, float)) for y in employed + unemployed), "Values must be numbers!"

Create visualization plt.figure(figsize=(10, 6)) plt.plot(years, employed, marker=’o’, label=’Employed’) plt.plot(years, unemployed, marker=’o’, label=’Unemployed’)

plt.xlabel(’Year’) plt.ylabel(’Population (thousands)’) plt.title(’Employment Trends (1983-1990)’) plt.legend() plt.grid(True) plt.tight_layout() plt.show()

- E.2.3. GENERATED VISUALIZATION

[Figure 49]

Figure 6. Employment Trends 1983-1990: Employed population grew from 56.8M to 65.1M while unemployed declined from 6.3M to 3.9M

- E.2.4. ANALYSIS Data Extraction Quality:

- • Accurately extracted 8-year time series across multiple columns
- • Comprehensive data validation with assertions
- • Proper data type checking

#### Visualization Design:

- • Different series with markers for clarity
- • Grid enabled for easier value reading
- • Proper legend placement
- • Clear axis labels with units (thousands)

#### Economic Insights:

- • Chart reveals steady employment growth (56.8M → 65.1M, +14.6%)
- • Declining unemployment (6.3M → 3.9M, -37.6%)
- • Indicates strong economic recovery period during 1980s
- • Unemployment rate dropped from 10% to 5.6% over this period

- E.3. Case Study 3: Scatter Plot - Correlation Analysis

- E.3.1. TASK

Question: Create a scatter plot to visualize the relationship between the unemployment rate and the total civilian labor force for all years in the dataset.

Question Type: Visualization + Data Analysis Table Context: Long-term employment statistics (1983-1998) with civilian labor force size and unemployment rates.

- E.3.2. GENERATED CODE

import pandas as pd import matplotlib.pyplot as plt

Extract data from table

- x_values = [1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998]
- y_values = [9.9, 7.4, 7.0, 6.9, 6.2, 5.5, 5.2, 5.7, 7.2, 7.9, 7.2, 6.2, 5.6, 5.4, 4.9, 4.4] civilian_labor_force = [63047, 63835, 64411, 65422, 66207, 66927, 67840, 69011, 69168, 69964, 70404, 70817, 71360, 72087, 73261, 73959]

Verify extraction print("Years:", x_values) print("Unemployment rates:", y_values) print("Civilian labor force:", civilian_labor_force) assert len(y_values) > 0, "Y values cannot be empty!" assert all(isinstance(y, (int, float)) for y in y_values), "Y values must be numbers!"

Create visualization plt.figure(figsize=(12, 7)) plt.scatter(civilian_labor_force, y_values, c=’blue’, alpha=0.7)

plt.xlabel(’Civilian Labor Force (thousands)’) plt.ylabel(’Unemployment Rate (%)’) plt.title(’Relationship Between Unemployment Rate and Civilian ’ + ’Labor Force (1983-1998)’) plt.grid(True) plt.tight_layout() plt.show()

- E.3.3. GENERATED VISUALIZATION

[Figure 50]

Figure 7. Scatter plot showing the relationship between unemployment rate and civilian labor force size over 16 years (1983-1998)

- E.3.4. ADVANCED ANALYSIS Multi-variable Extraction:

- • Successfully extracted three related data series
- • 16 years of continuous data (1983-1998)
- • Proper alignment between years, rates, and labor force size

#### Correlation Insights:

- • Inverse Relationship: As labor force grows, unemployment tends to decrease
- • High unemployment (9.9%) at lower labor force size (63M) in 1983
- • Unemployment stabilizes around 4-6% as labor force expands to 74M
- • Demonstrates economic expansion pattern: growing labor force absorption

#### Economic Interpretation:

- • 1983-1990: Steep decline from 9.9% to 5.2% unemployment despite labor force growth of 9.4% - indicates strong job creation
- • 1990-1992: Unemployment spike to 7.9% (recession period)
- • 1993-1998: Steady improvement to 4.4% with continued labor force expansion - economic boom period

#### Code Features:

- • Alpha transparency (0.7) for better point visibility
- • Appropriate figure sizing (12x7) for wide scatter plots
- • Grid enabled for easier coordinate reading
- • Comprehensive data validation before plotting

### F. Data Analysis Case Study

#### F.1. Case Study 4: Comprehensive Statistical Analysis

- F.1.1. TASK Question Type: Data Analysis (Rudimentary Analysis) Question: What is the average unemployment rate across all years, and what is its standard deviation? Table Context: 71 years of unemployment data (1953-2023) Expected Answer: Mean = 5.80%, Std = 1.62%
- F.1.2. GENERATED CODE AND ANALYSIS

import pandas as pd import numpy as np

Extract unemployment rate data unemployment_rates = df[’Percent of labor force’].values

Calculate statistics mean_rate = np.mean(unemployment_rates) std_rate = np.std(unemployment_rates, ddof=1)

Create result DataFrame df = pd.DataFrame( ’Mean’: [round(mean_rate, 2)], ’Std Dev’: [round(std_rate, 2)] )

print(f"Average unemployment rate: mean_rate:.2fprint(f"Standard deviation: std_rate:.2f

- F.1.3. MODEL RESPONSE (DETAILED ANALYSIS)

Data Overview: Analyzed unemployment rates across 72 years (1953-2023). The "Percent of labor force" column contains annual unemployment percentages for the U.S. civilian labor force.

#### Calculation Process:

- 1. Extracted all 72 values from "Percent of labor force" column
- 2. Calculated mean: Sum(411.5) / Count(71) = 5.80%
- 3. Calculated std: Σ(x_i−x¯)

2

n−1 = 1.62% Statistical Details:

- • Mean: 5.80%
- • Standard Deviation: 1.62%
- • Minimum: 2.9% (1953)
- • Maximum: 9.7% (1982)
- • Range: 6.8 percentage points
- • Sample size: 71 valid data points

Insights: The average unemployment of 5.80% represents typical U.S. unemployment over 70 years. Standard deviation of

1.62% indicates moderate variability, with most years in 4.2-7.4% range (±1σ). Higher values occurred during recessions (1982, 2010, 2020).

Context: Peak unemployment during 1982 recession (9.7%) and 2008 financial crisis (9.6% in 2010). Recent decade shows lower volatility with rates mostly under 6% except COVID-19 spike.

- F.1.4. QUALITY INDICATORS

- • ✓Correct numerical results (5.80%, 1.62%)
- • ✓Detailed calculation methodology with formulas
- • ✓Rich contextual analysis with historical events
- • ✓Statistical interpretation (±1σ range)
- • ✓Identification of outliers and anomalies
- • ✓Time-series context (70+ years of data)

