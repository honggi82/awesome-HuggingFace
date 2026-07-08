# arXiv:2511.17729v4[cs.AI]4Feb2026

## M3-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark

Yang Zhou* Mingyu Zhao* Zhenting Wang Difei Gu Bangwei Guo Ruosong Ye Ligong Han Can Jin Dimitris N. Metaxas Rutgers University {eta.yang,zhenting.wang,bangwei.guo}@rutgers.edu {ligong.han,can.jin,dnm}@rutgers.edu {mz751,dg1188,ry233}@scarletmail.rutgers.edu

#### Abstract

We present M3-Bench, the first benchmark for evaluating multimodal tool use under the Model Context Protocol. The benchmark targets realistic, multi-hop and multi-threaded workflows that require visual grounding and textual reasoning, cross-tool dependencies, and persistence of intermediate resources across steps. We introduce a similaritydriven alignment that serializes each tool call, embeds signatures with a sentence encoder, and performs similaritybucketed Hungarian matching to obtain auditable one-toone correspondences. On top of this alignment, we report interpretable metrics that decouple semantic fidelity from workflow consistency. The benchmark spans 27 servers with 232 tools, and provides standardized trajectories curated through an Executor & Judge pipeline with human verification; an auxiliary four large language models (LLMs) judge ensemble reports end-task Task Completion and information grounding. Evaluations of representative state-of-theart Multimodal LLMs (MLLMs) reveal persistent gaps in multimodal MCP tool use, particularly in argument fidelity and structure consistency, underscoring the need for methods that jointly reason over images, text, and tool graphs. Our Benchmark’s anonymous repository is at https:// github.com/EtaYang10th/Open-M3-Bench

#### 1. Introduction

We introduce M3-Bench, the Multi-Modal, Multiplex, Matching-aware MCP Benchmark, as a principled evaluation suite for multimodal tool use under the Model Context Protocol (MCP). Multimodal Large Language Models (MLLMs) have recently shown that, once they are allowed to perform function calling (tool use), they can query ex-

*Equal contribution.

| | | | |GPT 5 0.<br><br>[Figure 1]|482|
|---|---|---|---|---|---|
| | |[Figure 2]|Gemini 2.5 Pr|o 0.423| |
| | |[Figure 3]<br><br>G|rok 4 (0709)|0.411| |
| | |[Figure 4]|GPT 5 Mini 0|0.395| |
| | |[Figure 5]<br><br>Gemin|i 2.5 Flash 0|.388| |
| |[Figure 6]<br><br>Cla|aude 4.5 Sonn|et 0.333| | |
| | |0.298|[Figure 7]<br><br>Grok 4 F|ast| |
| | |0.264<br><br>[Figure 8]|Llama 4 Scou|t 17B16E| |
| |0|.247 GP<br><br>[Figure 9]|PT 5 Nano| | |
| |0.205|[Figure 10]<br><br>Claude|4.5 Haiku| | |
| |0.180|[Figure 11]<br><br>Gemini 2.5|Flash Lite| | |
| |0.179|[Figure 12]<br><br>InternVL 3.|5| | |
|0|.141 Qw<br><br>[Figure 13]|en2.5 VL 72B| | | |
|[Figure 14]<br><br>GLM|4.5v| | | | |

0.0 0.1 0.2 0.3 0.4 0.5 Average score

Figure 1. Summary leaderboard of state-of-the-art (SOTA) MLLMs on M3-Bench (higher is better).

ternal services and reason over information that is not contained in their parameters [3, 7, 8, 20, 30, 42, 43, 48, 49, 53]. MCP specifies how models communicate with heterogeneous tools through standardized servers, which makes execution more reproducible across systems [29]. However, existing MCP benchmarks are mostly text-only and focus on linear API planning or database queries [12, 13, 24, 25,

Multimodality

Multi-Hop&

Information

Description

Benchmark

FuzzyTask

Ecosystem

Grounding

Similarity

Threaded

Domains

Metric

Tools

MCP

ToolBench [36] 49 3451 ✗ ✗ ✗ ✗ ✗ ✗ BFCL v3 [35] 8 24 ✗ ✗ ✗ ✗ ✗ ✗ τ-Bench [50] 2 28 ✗ ✗ ✗ ✗ ✗ ✗

MCP-RADER [13] 9 42 ✓ ✗ ✗ ✗ ✗ ✗

MCPEval [24] 5 19 ✓ ✗ ✗ ✗ ✗ ✗ MCPToolBench++ [12] 6 1500+ ✓ ✓ ✗ ✗ ✗ ✗

MCP-Atlas [41] 40+ 300+ ✓ ✓ ✗ ✗ ✗ ✗

MCP-Universe [25] 6 113 ✓ ✓ ✓ ✗ ✗ ✗ MCP-Bench [45] 28 250 ✓ ✓ ✓ ✓ ✗ ✗ M3-Bench (Ours) 27 232 ✓ ✓ ✓ ✓ ✓ ✓

Table 1. Comparisons to existing tool-using benchmarks (compact single-column).

35, 36, 41, 50]. A systematic evaluation of multimodal MCP workflows, where images and text jointly condition tool calls and results, is still missing [2, 25].

The core challenge in real-world MCP trajectories is visual grounding: multimodal tool invocation hinges on correctly interpreting the image before any tool can be parameterized. In Figure 2a, the agent receives a photograph and must first resolve the landmark/city from visual cues, only then can it condition subsequent MCP calls; The agent must fully recognize the fully stocked and well-displayed products on the shelf in a photograph before proceeding to the next MCP tools call. Second, real-world MCP trajectories are not single-shot calls. They are multi-hop, they contain causally dependent operations, and they frequently execute several tool calls in parallel within one step [22, 27, 52, 54]. The task in Figure 2b illustrates this setting. The agent receives an image of a hazardous construction scene and a reporting-style instruction. Inside one step, actions that do not depend on each other, such as adding an image and updating bullets, can be executed concurrently. Across steps, operations that do depend on earlier results, such as annotating before inserting or creating the deck before saving, must follow the ground-truth order. These properties make simple string matching or linear-sequence scoring inadequate [27, 54], and it is precisely what our benchmark aims to stress. We define two important concepts in what follows: Multi-Hop refers to workflows with more than one causally dependent step, where later actions consume artifacts produced earlier [14, 51]; Multi-Threaded refers to order-independent tool calls executed within a single step under shared state, allowing safe parallelism while preserving cross-step causality (Mathematical definition in 1 & 2).

Table 1 contrasts M3-Bench with prior tool-use evaluations along 8 axes. Domains and Tools enumerate functional breadth and unique tool count. MCP ecosystem marks whether the benchmark connect the agent directly to a set of production-grade, live MCP servers, and Information grounding indicates that answers must be supported by ev-

idence returned from tools. Fuzzy task description denotes underspecified, natural instructions without a clear trajectory. Critically, Multi-Hop & Threaded captures cross-step causal dependencies together with safe within-step parallelism; Multimodality requires joint image–text conditioning; and Similarity Metric denotes large language models (LLM) free, similarity-aware alignment of predicted to reference calls (see Section 3). As discussed above, Multimodality and Multi-Hop/Threaded causality are central to realistic MCP workflows, and a Similarity-Based scorer is necessary to credit semantically correct calls. Most existing benchmarks are text-only and/or lack explicit multithreaded causality, and none pair these with a similarityaware alignment; in contrast, M3-Bench satisfies all three while retaining the key advantages of prior benchmarks.

We align predicted and reference tool calls with a similarity-bucketed Hungarian alignment: each call is serialized, embedded with a fixed sentence encoder, and one-toone matched within tool-name buckets under weak/strong cosine thresholds. This gives deterministic, auditable correspondences without relying on an LLM judge for call-level scoring. On top of this alignment, we report a compact, recall-aware suite that separates Section 3. A small fourmodel judge ensemble is used for evaluating the overall quality of the trajectory to retain some of the advantages of the LLM judge feature. Besides, to standardize references, our experiments also provide a best trajectory obtained via an Executor–Judge loop.

###### Contributions.

- 1. We present, to our knowledge, the first benchmark explicitly targeting multimodal MCP workflows.
- 2. Our repository provides an end-to-end pipeline for generating MCP best trajectories, with an optional lightweight human verification pass to enhance stability.
- 3. We introduce a structure-aware metric suite that aligns tool calls via bucketed Hungarian matching on highdimensional feature similarities, yielding robust and auditable correspondences for effectively diagnosing distinct error modes.
- 4. We evaluate a wide range of mainstream MLLMs and conduct a comprehensive assessment across tasks, budgets, error types, and other factors.

#### 2. Related Work

##### 2.1. LLM Tool Use and Protocols

Early studies showed that large language models (LLMs) can autonomously decide when/what/how to call external tools via simple APIs [1, 9, 10, 17–19, 27, 28, 36, 42]. Reliability improves with API-aware finetuning and retrieval [34]; instruction/data scaling expands tool repertoires and evaluations (ToolBench) [36]; and hierarchical retrieval with self-reflection addresses large tool spaces and

###### Task Samples

Record all hazards in an image, annotate them, and save to PPT

MLLM Agents MCP Serves

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

……

Here is an attraction image, give this city overview, check current weather, and find nearby restaurants.

I have ICD-xx. Confirm its meaning, verify whether these medications I’m considering are safe together, recommend suitable options with purchase links.

[Figure 33]

I saw the brick placed on the edge of the scaffold at the construction site in the picture fall from above. I’ll over-detect first so we don’t miss any hazard.

Convert hazards into one-line bullets with a quick fix. Keep it scannable—no wall of text.

[Figure 34]

[Figure 35]

|{"id": "hazard_highlights“}|
|---|

PPT MCP create_presentation

{"slide_index": 1, "placeholder_idx": 1, "bullet_points":["Falling-object

Can you look at the items in the photo, search for them online with the usual marketplace tools, and tell me the current price for each?

[Figure 36]

[Figure 37]

|{"imageFileUri”:./images.png", "includeDescription": true}|
|---|

Start with a title slide to frame the scene. This will anchor the image and bullets

List food name in the image, get its top Reddit post comment count, fetch a brief Wikipedia description, and save a spreadsheet with columns Food Name, Reddit Comment Count, and Wikipedia Description.

hazard: multiple bricks in motion/at the scaffold platform edge. Ladder present on the right: ensure……

PPT MCP manage_image

DINOX MCP detect-all-objects

[Figure 38]

{"layout_index": 0, "title": "Hazard Highlights:

[Figure 39]

Mark risky spots directly on the image so they pop at a glance. If circles clutter, I’ll switch to boxes.

Construction Site", "presentation_id":

Export to .pptx for sharing and sanity-check it opens

PPT MCP add_slide

[Figure 40]

"hazard_highlights"}

Identify the item in the image, get its price, compare crypto-candles to pick the most volatile coin, fetch its data, calculate how many items 100 units could buy.

{"file_path":

[Figure 41]

{"input_path": "./ images.png" , "output_path": "./ images.png", "circles": [{"center_x": 248,

"./highlights_report.pptx", "presentation_id":

[Figure 42]

Drop in the annotated image and auto-fit so details stay sharp. Preserve aspect ratio.

PPT MCP save_presentation

Imagesorcery MCP detect-all-objects

"hazard_highlights"}

[Figure 43]

“center_y”: 208, ……]}

{"layout_index": 0,

Scan the image to get the webpage link, fetch the main picture from that page, infer the location, convert it to coordinates, check current air quality, and give health advice.

I've annotated all hazards and generated the “Hazard Highlights: Construction Site” slide deck with the marked image and concise bullets, saved as highlights_report.pptx.

[Figure 44]

"title": "Hazard Highlights: Construction Site",

Please detect animals in the image and pick the first two distinct species, then for each species find a top relevant scholarly paper.

Spin up a deck so I can add slides incrementally. Keep the ID stable for later edits.

PPT MCP manage_image

"presentation_id": "hazard_highlights"}

(a)

(b)

Figure 2. Overview of tasks and a realistic multimodal MCP workflow. (a) Representative task samples in M3-Bench. (b) A realistic multimodal MCP workflow: the agent (i) calls a vision MCP server to detect and mark hazards in the input image, (ii) calls Imagesorcery MCP to process the image, and (iii) calls an office/productivity MCP server to create and populate a slide deck.

brittle protocols (AnyTool/AnyToolBench) [11]. Benchmarks targeting the decision to use a tool and tool selection (MetaTool Benchmark) complement these directions [16]. To reduce integration friction across heterogeneous tools, the Model Context Protocol (MCP) standardizes interfaces [2, 29].

Task

| | | |
|---|---|---|
| | | |

MCP serves

Plan Process Decision

Detailed Process

𝑐𝑐11

𝑐𝑐12 𝑐𝑐22 𝑐𝑐32

MLLM Executor

𝑐𝑐21 𝑐𝑐31

Human

- 2.2. Benchmarks for Tool Use and MCP

Beyond general agent evaluations such as AgentBench [23] and web-agent suites like WebArena and BrowserGym [21, 54], recent MCP-native efforts emphasize realistic, multi-hop use over live servers: MCP-Bench links LLMs to dozens of MCP servers and hundreds of tools [45]; MCP-Universe provides an end-to-end framework grounded in real-world servers and long-horizon tasks [38, 39]; MCPToolBench++ scales tool categories and tasks via marketplace-sourced servers [12]; and MCP-Atlas curates a human-authored dataset and leaderboard for multitool workflows [40, 41]. Relative to these, we target multimodal tasks and introduce a similarity-aware alignment with structure-sensitive diagnostics.

- 3. M3-Bench

Final Edit

Trajectory

Only works when generating the optimal trajectory, not when inferring

Judge Summarize

MLLM Judge

Best Trajectory

Figure 3. Benchmark trajectory generation and judging pipeline. The Executor plans tools, executes calls with controlled multi-threaded execution, and summarizes; the Judge reconstructs an optimal trajectory and verifies correctness. where cℓ,nℓ denotes the nℓ-th tool call in step ℓ.

ized optimal trajectory for alignment. All prompts are in Appendix Section 6.2.

Via system prompts, the Executor receives the full tool list (with docstrings) and outputs the planned tool set for the current step—selection only to stabilize reasoning and keep the chain controllable. (See Appendix Prompt: Plan.)

- 3.1. Benchmark Construction

Process The Executor returns strictly valid JSON tool calls (each with name and arguments). A multi-hop workflow allows multi-threaded interaction with the MCP Ecosystem within a single step. Structured outputs are appended to the dialogue, and any file-image paths from MCP tools persist in a temporary workspace for later steps. (See Appendix Prompt: Process.)

We generate benchmark trajectories with an Executor–Judge loop that decouples tool execution from adjudication (Figure 3). For each image–task pair, a human instruction fixes intent and permissible tools. Given the prompt, image, and registry, the Executor produces a stepwise trajectory; the Judge audits and consolidates it into a standard-

| |37<br><br>Health Weather & Air Quality| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |E-commerce & Finance<br><br>6.9<br><br>5.62.2| | | | | | | | | | | | | | | | | | | | | | |
| |25<br><br>Office Automation<br><br>29.0<br><br>Computer Vision<br><br>8.2| | | | | | | | | | | | | | | | | | | | | | |
| |22<br><br>21 Science & Space 9.1| | | | | | | | | | | | | | | | | | | | | | |
| |19<br><br>18<br><br>17<br><br>Geography & Travel<br><br>10.8| | | | | | | | | | | | | | | | | | | | | | |
| |11<br><br>Academic & Knowledge<br><br>28.1| | | | | | | | | | | | | | | | | | | | | | |
| |9<br><br>7<br><br>6<br><br>5| | | | | | | | | | | | | | | | | | | | | | |
| |4 4 4<br><br>3 3<br><br>2 2 2 2 2<br><br>1 1 1 1 1 1| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |

35

30

NumberofTools

25

20

15

10

5

- 0

PptExcelNasa-McpGoogle-MapsMath PaperSearchHealthcare-McpNixosYahoo-FinanceImagesorcery-McpNationalparksHugeicons-McpBibliomanticDinox-McpGoogle-AirMetmuseum-McpReddit-Mcp-ServerCar-Price AmazonFoodNutritionMcpOcrOkxLinkimage-McpOpenlibraryMcpWiki Pyzbar-McpTmdbWeather

MCP Server

Figure 4. Overview of MCP server ecosystem used in the M3-Bench. Category distribution of MCP servers; and tool distribution across servers. The inventory follows the MCP design and public specification.

Decision and Final Reply After each step, a binary prompt checks whether evidence suffices to conclude; if so, the Executor emits only the final natural-language answer. (See Appendix Prompt: Decision, Prompt: Final.)

Judge and Summarize The Judge consumes the initial prompt, image, and full Executor dialogue (including the tool-call chain). Trajectories with causal misordering or factual errors are rejected; otherwise, the Judge reconstructs the optimal trajectory, removing failed/redundant calls and merging operations that could run concurrently. Human verifiers audit scope and logical consistency; the audited result becomes the benchmark reference. (See Appendix Prompt: Judge, Prompt: Summarize.)

##### 3.2. Task Distribution

We construct a curated suite of 28 multimodal MCP tasks; detailed per-task goals and input–output constraints are provided in Appendix Section 6.1, and representative full trajectories appear in Section 7. In total, the evaluation set contains 211 trajectories, whose reference trajectories comprise 644 steps and 1337 MCP tool calls.

##### 3.3. MCP Ecosystem

To keep trajectories controllable and reproducible, we curate a compact set of MCP servers and tools with minimal functional overlap. We favor orthogonal, well-documented, and stable utilities, discarding redundant or unstable options to reduce confounds during evaluation. Our benchmark spans 232 representative tools across 27 MCP servers (Figure 4(a)). This diverse ecosystem covers Academic & Knowledge, Science & Space, Geography & Travel, Ecommerce & Finance, Office Automation, Weather & Air Quality, and Health, enabling broad capability coverage in M3-Bench (see Figure 4(b)).

#### 4. Metrics

In Section 4.1, we first establish the mathematical definitions and foundations for our metrics. Building on these theoretical bases, we then construct the Detection Metrics Section 4.2, Trajectory Alignment Metrics Section 4.3, and LLM Judges Metrics Section 4.4.

##### 4.1. Mathematical foundations

A trajectory is a DAG ordered by turns (steps). Denote L as the number of steps in a trajectory T . The ground-truth T ⋆ and predicted trajectories T are: T ⋆ = {S1⋆,...,SL⋆⋆}, T = { S1,..., S L}, where each step S enumerates its calls c, Sℓ⋆ = {c⋆ℓ,1,...,c⋆ℓ,n⋆

}, Sℓˆ = {cˆℓ,ˆ1,...,cˆℓ,nˆ

}. where

ℓˆ

ℓ

ℓ is the step of the trajectory, so n⋆ℓ and nℓˆ are per-step call counts. A call is c = (tool(c),args(c)), where tool(·)

is the categorical tool identifier and args(·) serializes all deterministic inputs.

- Definition 1 (Multi-Hop). Let T = {S1,...,SL} be a tool-

use trajectory with steps Sℓ = {cℓ,1,...,cℓ,n

ℓ} and intercall dependency edges across steps. We call T multi-hop if L ≥ 2 and there exists a dependency (ci,a → cj,b) with i < j, where a,b ∈ N and a ∈ [1,ni], b ∈ [1,nj].

- Definition 2 (Multi-Threaded). A step Sℓ is multi-threaded if it contains at least two calls cℓ,a ̸= cℓ,b that are orderindependent given the shared state at depth ℓ (no dependency edge between them).

These definitions make explicit that a trajectory is a structured object with cross-step causal edges and withinstep parallelism. A general option is to delegate this alignment to an LLM judge, but it introduces (i) circularity/lineage bias, where judges trained like the systems under test tend to over-credit them; (ii) susceptibility to prompt gaming, as long rationales or CoT responses can inflate scores even when the underlying tool use is wrong; (iii) weak argument verification, since judges rarely check whether argument objects are semantically correct or whether orderinvariant sets are handled; (iv) knowledge substitution, where missing tool evidence is replaced by the model’s own world knowledge; and (v) opacity, as scores lack auditable, one-to-one correspondences between predicted and reference calls. Real MCP trajectories exacerbate these issues: steps often contain unordered sets of calls, later calls can self-correct earlier errors, and partial success is common. Consequently, the alignment primitive must be deterministic, auditable, and robust to surface-form variation.

Similarity-Bucketed Hungarian Alignment To address this problem, we introduce a similarity computation method based on a text encoder. Each call is flattened to a canonical text string by a serialization function ϕ(·) that preserves

tool name, required arguments and order-invariant sets. We embed serialized calls with a sentence encoder E and compute cosine similarity: Sij = ⟨x

| | |
|---|---|
| | |
| | |

𝑆𝑆1∗ c11∗ c12∗ 𝑆𝑆2∗ c21∗

𝑆𝑆1∗ c11∗ c12∗ 𝑆𝑆2∗ c21∗

⋆ i , xj⟩

∥x⋆i∥2 ∥ xj∥2 ∈ [−1,1], where x⋆i = E ϕ(c⋆i ) and xj = E ϕ(ˆcj) .

| | |
|---|---|
| | |
| | |

𝑆𝑆̂1 𝑐𝑐11̂ 𝑐𝑐12̂ 𝑆𝑆̂2 𝑐𝑐21̂

𝑆𝑆̂1 𝑐𝑐11̂ 𝑐𝑐12̂ 𝑐𝑐13̂ 𝑆𝑆̂2 𝑐𝑐21̂ 𝑐𝑐22̂

We align with a tool-specific, step-agnostic assignment that yields auditable one-to-one correspondences. Concretely, we serialize every call into a canonical text signature and embed it with the sentence encoder SentenceTransformers [37] from Textual Encoding and Similarity, giving a similarity matrix S. For each tool k, we form the bucketed index sets Ik = {i | tool(c⋆i ) = k }, Jk = {j | tool(ˆcj) = k }, so that only same-tool pairs are considered. We convert similarity to cost with a weak screening threshold τweak to forbid low-similarity edges:

(b) Precision: Evaluate the accuracy of predicted tools calls.

(a) Recall: Evaluate the hit rate of key tools calls.

Missing correct calls Redundant calls

𝑆𝑆1∗ c11∗ c12∗ 𝑆𝑆2∗ c21∗

𝑆𝑆1∗ c11∗ c12∗ 𝑆𝑆2∗ c21∗

𝑆𝑆̂1 𝑐𝑐11̂ 𝑆𝑆̂2 𝑐𝑐21̂ 𝑆𝑆̂3 𝑐𝑐31̂

𝑆𝑆̂3 𝑐𝑐11̂ 𝑐𝑐21̂ 𝑐𝑐31̂

(c) Step Coherence: Evaluate additional splits for multi-threaded scenario.

(d) Merge Purity: Evaluate order consistency for multi-hop of scenario.

Orthogonal calls that could be executed in a multithreaded manner are split into multiple steps.

Causally linked multi-hop calls are merged into a single step.

Misaligned matching across different steps

𝑆𝑆1∗ c11∗ c12∗ 𝑆𝑆2∗ c21∗

1 − Src(k), Src(k) ≥ τweak, λpen, Src(k) < τweak,

Ground truth tools call

Crc(k) =

(1)

Similarity matching tools call pair

𝑆𝑆̂1 𝑐𝑐11̂ 𝑆𝑆̂2

𝑐𝑐12̂ 𝑐𝑐21̂

Matched tools call

and solve a linear assignment in each bucket via the Hungarian (Kuhn–Munkres) algorithm minΠ(k)∈P ⟨C(k),Π(k)⟩ retaining only feasible pairs above the weak threshold to form the global match set: M = {(i,j,Sij) | Sij ≥ τweak}.

(e) Order Consistency: Evaluate order consistency for multi-hop of scenario.

Unmatched tools call

Figure 5. Overview of our structure-sensitive metrics. where Sℓ⋆ and Sℓˆ denote the sets of calls in the ground-truth step ℓ and the predicted step

ℓˆ, respectively, and c⋆ℓ,n⋆ ℓ

denote the n⋆ℓ-th ground-truth call in step ℓ and the nℓˆ-th predicted call in step ℓˆ.

and cˆℓ,nˆ

ℓˆ

Properties and rationale. Tool bucketing prevents crosstool credit; step agnosticism allows early-wrong, latercorrect fixes to match without forcing same-step alignment; and similarity gating recognizes paraphrase and orderinvariant arguments without inflating coverage. Together, these choices deliver: (i) deterministic, auditable correspondences in place of LLM judge; (ii) invariance to withinstep call permutations (sets are unordered); and (iii) faithful scoring under partial success, misaligned or missing calls simply remain unmatched, while correctly recovered calls still receive credit. This alignment thus targets exactly the failure modes we observe in MCP trajectories while satisfying the three requirements in Section 4.1.

and Npred = ℓ ˆnℓˆ be the total numbers of calls in T ⋆ and T , respectively.

##### 4.3. Trajectory Alignment Metrics

Next, we introduce a metric to quantify the degree of Trajectory Alignment.

- 1. MLLM may select the correct tool but supply semantically incorrect arguments, requiring an Average Similarity measure over serialized calls;
- 2. MLLM may unnecessarily split operations that could run in a single step in a multi-threaded scenario, captured by Step Coherence;
- 3. MLLM may collapse causally distinct multi-hop subgoals into one step in a multi-hop scenario, quantified by Merge Purity;
- 4. MLLM may invert the execution order of causally dependent steps in a multi-hop scenario, detected by Order Consistency.

##### 4.2. Detection Metrics

Based on the above mathematical foundations, we introduce detection-based metrics:

- 1. Systems may call incorrect tools or omit required ones, motivating Recall;
- 2. MLLM may make unnecessary calls, resulting in additional waste of resources Precision.

Argument Similarity. We summarize the semantic faithfulness of matched arguments as

Recall and Precision. (Figure 5 (a)-(b)) These metrics quantify, respectively, how many ground-truth calls are successfully matched (coverage) and how many predicted calls are valid (false-positive control). Given per-sample match sets Mm with counts Ngt,m and Npred,m, we define Recall =

Sij |{(i,j) ∈ M : Sij ≥ τstrong}|

ArgSim = (i,j)∈M, Sij≥τstrong

, (2)

where τstrong controls alignment confidence.

m |Mm|

m |Mm| m Ngt,m , Precision =

m Npred,m, where |Mm| counts matched pairs for sample m, Ngt = ℓ n⋆ℓ

Step Coherence. (Figure 5 (c)) This metric penalizes scattering matches from one ground-truth step across mul-

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.013 0.023 0.011 0.012 0.000 0.013 0.285 0.130 0.029 Qwen2.5-VL-72B [6] 0.143 0.073 0.125 0.125 0.022 0.112 0.303 0.388 0.141 InternVL 3.5 [44] 0.224 0.143 0.193 0.191 0.048 0.161 0.325 0.298 0.179 Gemini 2.5 Flash Lite [8] 0.200 0.173 0.168 0.179 0.030 0.152 0.299 0.241 0.180 Claude 4.5 Haiku [4] 0.193 0.480 0.166 0.190 0.053 0.108 0.412 0.246 0.205 GPT-5 Nano [32] 0.287 0.103 0.276 0.228 0.045 0.258 0.551 0.529 0.247 Llama-4-Scout-17B16E [26] 0.367 0.176 0.325 0.286 0.068 0.290 0.367 0.335 0.264 Grok-4 Fast [47] 0.343 0.359 0.309 0.284 0.122 0.289 0.501 0.378 0.298 Claude 4.5 Sonnet [5] 0.395 0.361 0.354 0.389 0.127 0.257 0.535 0.451 0.333 Gemini 2.5 Flash [8] 0.454 0.397 0.409 0.357 0.211 0.412 0.530 0.474 0.388 GPT-5 Mini [31] 0.481 0.238 0.462 0.392 0.199 0.360 0.652 0.633 0.395 Grok 4 (0709) [46] 0.465 0.387 0.408 0.415 0.177 0.390 0.664 0.638 0.411 Gemini 2.5 Pro [8] 0.501 0.449 0.433 0.464 0.249 0.424 0.520 0.438 0.423 GPT-5 [33] 0.627 0.264 0.583 0.502 0.290 0.453 0.671 0.657 0.482

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- Table 2. Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories: Detection, Trajectory Alignment, and LLM-Judge Metrics, The average score results correspond to Figure 1.

tiple predicted steps, reflecting unnecessary splits of operations that could have been executed within a single multithreaded step. For any call c, let gtst(c) and predst(c) denote its step indices in the ground-truth and predicted trajectories. For each ground-truth step ℓ, collect the predicted step indices touched by its matches: P(ℓ) = { predst(ˆcj) | (i,j) ∈ M,gtst(c⋆i ) = ℓ }.Then we define the per-step coherence and a similarity-weighted global average as

 

1, |P(ℓ)| ≤ 1,

sc(ℓ) =

1 |P(ℓ)|

, otherwise,



(3)

wℓ sc(ℓ) ℓ wℓ

StepCoh = ℓ

,

where wℓ = (i,j)∈M[gtst(c⋆i ) = ℓ] counts matches originating from ground-truth step ℓ.

Merge Purity. (Figure 5 (d)) This metric is used to evaluate whether, in a multi-hop scenario, different causally related steps are incorrectly merged. For total alignment weight between ground-truth step a and predicted step b:

[gtst(c⋆i ) = a] [predst(ˆcj) = b] Sij, (4)

Wab =

(i,j)∈M

where [·] denotes the Iverson bracket (indicator function), equal to (1) when its condition is true and (0) otherwise. Define column sums Sb = a Wab, total mass S = b Sb, and column prior P(b) = Sb/S. The conditional distribution of ground-truth steps given a predicted step is qa|b = Wab/Sb for active columns with Sb > 0. The normalized purity is:

H(G | P) log Gact H(G | P) =

MergePur = 1 −

P(b) −

qa|b log qa|b ,

a:Wab>0

b:Sb>0

(5)

where Gact = |{a : b Wab > 0}| counts active groundtruth steps, H(·) uses natural logarithm, and larger val-

ues indicate predicted steps concentrating mass on single ground-truth steps (i.e., less over-merging); G denotes the random variable of ground-truth step indices and P denotes the random variable of predicted step indices; Gact = |{a :

b Wab > 0}| is the number of active ground-truth steps. The normalization by log Gact maps [0,log Gact] to [0,1].

Order Consistency. (Figure 5 (e)) This metric is used to evaluate whether, in a multi-hop scenario, the order of causally related calls is incorrectly swapped, capturing order consistency across steps. Let each match (i,j) ∈ M be represented by its step pair (a,b) = (gtst(c⋆i ),predst(ˆcj)). Over all unordered pairs of matches that are comparable in both axes, we compute:

OrdCons = 1 −

1 Q

(u,v)

(au − av)(bu − bv) < 0 , (6)

where the sum runs over pairs with au ̸= av and bu ̸= bv, Q is the number of such pairs, and the indicator records inversions between the ground-truth and predicted step orders; higher values indicate fewer inversions.

To penalize degenerate cases with very few matches (weak systems that match only a few easy MCP tools from inflating structure-aware metrics, leading to a very small denominator in the metric calculation, and thus an inflated outcome), we weight per-sample metrics by their recall. Let

Model Input Tokens Output Tokens

Qwen2.5-VL-72B 0.362 0.008 InternVL 3.5 0.069 0.002 Claude 4.5 Haiku 0.118 0.007 Claude 4.5 Sonnet 0.408 0.029 GPT-5 Nano 0.002 0.001 Llama-4-Scout-17B16E 0.345 0.008 Grok-4 Fast 0.210 0.028 Gemini 2.5 Flash Lite 0.477 0.010 Gemini 2.5 Flash 0.621 0.032 GPT-5 Mini 0.125 0.014 Grok 4 (0709) 0.132 0.027 Gemini 2.5 Pro 0.606 0.031 GPT-5 0.252 0.037

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

[Figure 70]

[Figure 71]

- Table 3. Average token cost per task for each MLLM in millions.

the per-sample recall be rm = |M

m|

Ngt,m . For any per-sample metric Fm ∈ {StepCoh, MergePur, OrdCons}, we report a recall-covered score: Fcov =

m Ngt,m rm Fm

m Ngt,m .

- 4.4. LLMs Judges Metrics

To leverage the advantages of large language models (LLMs) in complex logical reasoning, we introduce a metric based on LLMs Judges to assess:

- 1. Whether the MLLM achieves Task Completion;
- 2. Whether the answer provided by the MLLM possesses MCP Information Grounding.

Task Completion. For each instance m, a fixed set of four LLMs Judges J produces scalar scores {s(mj)}j∈J ⊂ [0,1]. We compute a trimmed mean that discards the highest and lowest scores and averages the remaining two (Prompt: Appendix Section 6.2):

s(mj) ,

TaskCompm = 12

j∈J \{jmin, jmax}

s(mj), jmax = arg max

s(mj).

jmin = arg min j∈J

j∈J

(7)

We report TaskComp (Task Completion) as the mean of per-instance scores over the evaluation set. The judge pool includes strong large language models (e.g., GPT, Gemini, DeepSeek, Grok).

Information Grounding. Judges are additionally shown both the ground-truth and predicted trajectories and emit step-level groundedness scores. For instance m with ground-truth steps indexed by ℓ⋆ and ℓˆ, each judge j ∈ J

outputs gm,ℓ(j)

⋆,ℓˆ ∈ [0,1]. The calculation of InfoGrndm,ℓ⋆,ℓˆ follows Equation 7 (Prompt: Appendix Section 6.2).

Model Avg. Steps Avg. Tool Calls

GLM 4.5v 1.91 3.84 Qwen2.5-VL-72B 4.29 12.30 InternVL 3.5 3.65 9.57 Claude 4.5 Haiku 0.80 2.73 Claude 4.5 Sonnet 2.74 6.99 GPT-5 Nano 5.13 17.97 Llama-4-Scout-17B16E 4.72 13.26 Grok-4 Fast 2.91 5.93 Gemini 2.5 Flash Lite 2.73 7.06 Gemini 2.5 Flash 4.04 7.12 GPT-5 Mini 3.72 12.56 Grok 4 (0709) 3.21 7.64 Gemini 2.5 Pro 3.04 6.95 GPT-5 3.55 14.77

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Table 4. Average interaction steps and tool calls per task across models on our benchmark.

#### 5. Experiment

##### 5.1. Evaluation Results

As shown in Table 2, we evaluate SOTA MLLMs on M3Bench. The three flagship families, GPT-5 [33], Gemini, and Grok, form the leading cluster under our coverageweighted alignment and judge signals. GPT-5 [33] attains the highest Average Score (0.482) with strong Recall (0.627) and high Argument Similarity (0.583), and it also tops all structure-aware metrics: its multi-threaded Step Coherence (0.502) and multi-hop sensitive Order Consistency / Merge Purity (0.290 / 0.453) indicate that it preserves both within-step parallel branches and cross-step merges in long tool graphs. Gemini 2.5 Pro [8] (0.423) and Grok 4 (0709) [46] (0.411) exhibit similarly balanced profiles.

In contrast, compact/latency-oriented variants tend to trade coverage and structural consistency for speed: for example, Claude 4.5 Haiku [4] achieves high Precision (0.480) but much lower Recall, and its structure metrics remain modest, pointing to short, locally correct trajectories that under-utilize multi-threaded branches and often break multi-hop dependencies. Finally, GLM 4.5v [15], Qwen2.5-VL-72B [6], and InternVL 3.5 [44] lag notably on both coverage and structure, suggesting that MLLMs SFT objectives favor pure vision–language understanding over instruction following and trajectory planning.

##### 5.2. Per-Task Analysis

Figure 7 shows that model capabilities are far from uniform across selected tasks (details in Section 6.1): curves cross frequently, indicating complementary strengths rather than a single, uniformly dominant system. Grok-4 is weak on Fruit Cal (numerical calibration/unit handling) yet strong on QR-code/Attraction (optical character recognition (OCR) and points-of-interest (POI) chaining). GPT-5 trails on Vision Research (fine-grained figure grounding) but dominates Biology Cut (precise localization/cropping).

13.9

11.7

20.6

21.3

23.4

26.8

[Figure 86]

39.8

[Figure 87]

[Figure 88]

8.9

[Figure 89]

[Figure 90]

[Figure 91]

41.5

[Figure 92]

4.4

Qwen2.5-VL

Gemini 2.5 Flash Lite

55.6 Claude 4.5 Haiku

###### Llama-4 Scout-17B16E

46.0

###### GLM-4.5V

InternVL 3.5

GPT-5 Nano

4.2

72B 60.9 10.7

28.6 8.5

11.9

75.6

81.9

75.2

7.4

5.8

11.4

7.6

18.6

19.7

12.2

24.1

27.4

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

47.4

29.5

Claude 4.5 Sonnet

Gemini 2.5 Flash

Grok 4 (0709)

Gemini 2.5 Pro

Grok-4 Fast

GPT-5 Mini

GPT-5

5.4 66.2

62.3

4.5

75.0

76.2

79.0

5.3 6.4

91.1

Illegal calling format

Unknown Tool Invocation

Invalid Invocation Arguments

Successful (Resource Not Found)

Successful Call

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

Figure 6. Per-call outcome distributions across models when replaying predicted MCP trajectories through the MCP interpreter.

Gemini 2.5 Flash [8] (4.04 / 7.12), and Grok-4 Fast [47] (2.91 / 5.93) reach goals with fewer, more precise calls, consistent with their higher detection precision. Very short dialogues (e.g., Claude 4.5 Haiku [4], GLM

- 4.5v [15]) reflect under-coverage rather than efficiency.
- 5.4. Analysis of Error Distributions

We assess per-call reliability by replaying each model’s trajectory through the MCP interpreter and categorizing every tool call into one of five mutually exclusive outcomes shown in Figure 6: (i) Illegal calling format: the MCP request cannot be parsed into a valid schema (e.g., missing name or arguments); (ii) Unknown Tool Invocation: the request targets a non-existent or non-exposed tool; (iii) Invalid Invocation Arguments: the tool is found but arguments are syntactically/semantically invalid (typically surface as HTTP 400 or “[Tool error] Invalid arguments”); (iv) Successful (Resource Not Found): the call is structurally correct yet the backend cannot route/locate the requested resource (e.g., HTTP 404); and (v) Successful call: the tool executes and returns a normal result. These categories allow us to decouple schema conformance, tool selection, argument fidelity, and downstream retrieval robustness.

GPT-5 Gemini 2.5 Pro Claude 4.5 Sonnet Grok 4 (0709)

Figure 7. Average score of each selected task (details in Section 6.1) in the representative model.

Gemini 2.5 Pro excels on Travel Comparison (web aggregation and itinerary synthesis). Claude 4.5 lags on vision-dependent tasks (e.g., Items Price), suggesting comparatively weaker multimodal grounding; disparities align with differences in OCR robustness. In Section 9 (See Appendix), we present the performance of different models on each task, based on the evaluation metrics of M3-Bench.

Overall, higher-performing systems allocate a larger share to Successful call in Figure 6, with the GPT-5 Grok and Gemini families exhibiting the strongest success rates in our setting. For Qwen2.5-VL-72B [6] and GLM-4.5V [15], the shortfall appears driven less by multi-hop reasoning capacity and more by tool-use hygiene: hallucinationtriggered invocations of non-existent tools and weaker adherence to instruction/schema constraints (the trajectory error analysis in Appendix Error: GLM-4.5v, Error: Qwen2.5-VL, Error: Internvl-3.5). This pattern suggests that improvements in schema guidance and instruction following would likely yield larger gains than refining highlevel reasoning alone. Although the Claude family is generally regarded as possessing strong capabilities in coding, mathematics, and reasoning, its cross-modal prior knowl-

##### 5.3. Budgets across models.

Table 3 complements these interaction statistics by reporting average per-task input and output token budgets. Table 4 summarizes the average interaction depth per task. Larger models often pursue longer, multi-hop plans: GPT-5 [33] and GPT-5 Nano [32] average (3.55 / 14.77) and (5.13 / 17.97) (steps / tool calls), indicating frequent retries and non-trivial orchestration; Llama-4-Scout-17B16E [26] and Qwen2.5-VL-72B [6] show similarly heavy tool use. By contrast, Gemini 2.5 Pro [8] (3.04 / 6.95),

edge is not particularly prominent Section 8.

#### 6. Conclusion

We introduced M3-Bench, a benchmark tailored to multimodal tool use under the Model Context Protocol (MCP), with trajectories that support multi-hop and multi-threaded execution. Our evaluation pipeline aligns predicted and reference calls by serializing each call, embedding it with a text encoder, and performing tool-name–bucketed Hungarian matching to obtain auditable one-to-one correspondences. On this alignment, we report a structure-aware suite of interpretable metrics. Across representative MLLMs, results reveal persistent gaps in argument fidelity and structural consistency, highlighting the need for agents that jointly reason over images, text, and tool graphs.

#### References

- [1] Pierre Andrews, Amine Benhalloum, Gerard Moreno-Torres Bertran, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Romain Froger, Emilien Garreau, JeanBaptiste Gaya, et al. Are: Scaling up agent environments and evaluations. arXiv preprint arXiv:2509.17158, 2025. 2
- [2] Anthropic. Introducing the model context protocol. https: //www.anthropic.com/news/model-contextprotocol, 2024. Accessed: 2025-10-30. 2, 3
- [3] Anthropic. Introducing claude 4. https://www. anthropic.com/news/claude-4, 2025. Product announcement; system card available; no peer-reviewed venue acceptance announced as of 2025-11-10. 1
- [4] Anthropic. Claude haiku 4.5 system card. https://www. anthropic.com/claude-haiku-4-5-systemcard, 2025. Oct. 2025, official system card; accessed 202511-10. 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [5] Anthropic. Claude sonnet 4.5 system card. https: //www.anthropic.com/claude-sonnet-4-5system-card, 2025. Oct. 2025, official system card; accessed 2025-11-10. 6, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, and et al. Qwen2.5-VL technical report, 2025. Covers Qwen2.5-VL-72B flagship variant. 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [7] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint, 2025. Status: arXiv preprint; no peer-reviewed venue acceptance found as of 2025-11-10. 1
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next-generation agentic capabilities. arXiv preprint, 2025. Status: arXiv preprint; no peer-reviewed venue acceptance found as of 2025-11-10. 1, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [9] Zihan Dong, Ruijia Wu, and Linjun Zhang. Labels or preferences? budget-constrained learning with human judgments

over ai-generated outputs. arXiv preprint arXiv:2601.13458,

2026. 2

- [10] Zihan Dong, Zhixian Zhang, Yang Zhou, Can Jin, Ruijia Wu, and Linjun Zhang. Evaluating llms when they do not know the answer: Statistical evaluation of mathematical reasoning via comparative signals. arXiv preprint arXiv:2602.03061,

2026. 2

- [11] Yu Du, Fangyun Wei, and Hongyang Zhang. Anytool: Selfreflective, hierarchical agents for large-scale API calls. In Proceedings of the 41st International Conference on Machine Learning, pages 11812–11829. PMLR, 2024. 3
- [12] Shiqing Fan, Xichen Ding, Liang Zhang, and Linjian Mo. MCPToolBench++: A large scale ai agent MCP tool use benchmark. arXiv preprint arXiv:2508.07575, 2025. 1, 2, 3
- [13] Xuanqi Gao, Siyi Xie, Juan Zhai, Shiqing Ma, and Chao Shen. MCP-RADAR: A multi-dimensional benchmark for evaluating tool-use capabilities in large language models. arXiv preprint arXiv:2505.16700, 2025. 1, 2
- [14] Jie He, Nan Hu, Wanqiu Long, Jiaoyan Chen, and Jeff Z. Pan. Mintqa: A multi-hop question answering benchmark for evaluating llms on new and tail knowledge. In arXiv preprint arXiv:2412.17032, 2024. 2
- [15] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, and et al. Glm4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [16] Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Z. Gong, and Lichao Sun. Metatool benchmark for large language models. 2024. 3
- [17] Can Jin, Hongwu Peng, Qixin Zhang, Yujin Tang, Dimitris N. Metaxas, and Tong Che. Two heads are better than one: Test-time scaling of multi-agent collaborative reasoning. arXiv preprint arXiv:2504.09772, 2025. 2
- [18] Can Jin, Hongwu Peng, Shiyu Zhao, Zhenting Wang, Wujiang Xu, Ligong Han, Jiahui Zhao, Kai Zhong, Sanguthevar Rajasekaran, and Dimitris N. Metaxas. Apeer: Automatic prompt engineering enhances large language model reranking. In WWW Companion 2025 - Companion Proceedings of the ACM Web Conference 2025, pages 2494–2502, Sydney, Australia, 2025. Association for Computing Machinery.
- [19] Can Jin, Yang Zhou, Qixin Zhang, Hongwu Peng, Di Zhang, Marco Pavone, Ligong Han, Zhang-Wei Hong, Tong Che, and Dimitris N. Metaxas. Your reward function for rl is your best prm for search: Unifying rl and search-based tts. arXiv preprint arXiv:2508.14313, 2025. 2
- [20] Team Kimi, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint, 2025. Status: arXiv preprint; no peer-reviewed venue acceptance found as of 2025-11-10. 1
- [21] Thibault Le Sellier De Chezelles et al. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467, 2024. 3
- [22] Bin Li, Wenhai Wang, Jiayi Yang, Shilong Zhang, Jian Sun, and Hongyang Li. Seed-bench: Benchmarking multimodal

- large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1–11, 2024. 2
- [23] Xiao Liu et al. Agentbench: Evaluating llms as agents. 2024. 3
- [24] Zhiwei Liu, Jielin Qiu, Shiyu Wang, Jianguo Zhang, Zuxin Liu, Roshan Ram, Haolin Chen, Weiran Yao, Shelby Heinecke, Silvio Savarese, et al. MCPEval: Automatic MCPbased deep evaluation for AI agent models. arXiv preprint arXiv:2507.12806, 2025. 1, 2
- [25] Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, and Junnan Li. MCP-Universe: Benchmarking large language models with real-world model context protocol servers, 2025. 1, 2
- [26] Meta AI. Llama 4 scout 17b-16e model card. https: / / huggingface . co / meta - llama / Llama - 4 Scout-17B-16E, 2025. HF official model card; release Apr. 5, 2025; accessed 2025-11-10. 6, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [27] Gr´egoire Mialon, Cl´ementine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A benchmark for general AI assistants. In International Conference on Learning Representations, 2023. 2
- [28] Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, and Le Sun. Livemcpbench: Can agents navigate an ocean of mcp tools? arXiv preprint arXiv:2508.01780, 2025. 2
- [29] Model Context Protocol Working Group. Model context protocol (mcp) specification. https : //modelcontextprotocol.io/specification/ 2025-06-18, 2025. Accessed 2025-10-22. 1, 3
- [30] OpenAI. Introducing o3 and o4-mini. https://openai. com/index/introducing- o3- and- o4- mini/,

2025. Product blog; not a peer-reviewed publication; no venue acceptance announced as of 2025-11-10. 1

- [31] OpenAI. GPT-5 mini. https://platform.openai. com/docs/models/gpt-5-mini, 2025. API docs; accessed 2025-11-10. 6, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [32] OpenAI. GPT-5 nano. https://platform.openai. com/docs/models/gpt-5-nano, 2025. API docs; accessed 2025-11-10. 6, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [33] OpenAI. GPT-5 system card. https://cdn.openai. com/gpt-5-system-card.pdf, 2025. Aug. 2025, official system card; accessed 2025-11-10. 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [34] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. 37, 2024. 2
- [35] Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie ChengJie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, 2025. 2

- [36] Yujia Qin et al. Toolllm: Facilitating large language models to master 16,000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023. 2
- [37] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of EMNLP-IJCNLP, pages 3982–3992, 2019. 5
- [38] Salesforce AI Research. MCP-Universe: Benchmarking large language models with real-world MCP servers. https://github.com/SalesforceAIResearch/ MCP-Universe, 2025. Accessed 2025-10-22. 3
- [39] Salesforce AI Research. MCP-Universe: A comprehensive framework for ai agent development and benchmarking. https://mcp-universe.github.io/, 2025. Accessed 2025-10-22. 3
- [40] Scale AI. MCP-Atlas leaderboard/dataset overview. https://scale.com/leaderboard/mcp_atlas,

2025. Accessed 2025-10-22. 3

- [41] Scale AI Research. MCP-Atlas raises the bar for agentic evaluation. https://scale.com/blog/mcp-atlas,

2025. Accessed 2025-10-22. 2, 3

- [42] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In International Conference on Learning Representations, 2023. 1, 2
- [43] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving AI tasks with ChatGPT and its friends in huggingface. In Advances in Neural Information Processing Systems, 2023. 1
- [44] Weiyun Wang, Zhangwei Gao, Lixin Gu, and et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [45] Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, and Eugene Siow. MCPBench: Benchmarking tool-using llm agents with complex real-world tasks via MCP servers. arXiv preprint arXiv:2508.20453, 2025. 2, 3
- [46] xAI. Grok 4 model card. https://x.ai/docs/ model-cards/grok-4, 2025. Initial public release July 9, 2025 (“0709”); accessed 2025-11-10. 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [47] xAI. Grok 4 fast model card. https://x.ai/docs/ model-cards/grok-4-fast, 2025. Official model card; accessed 2025-11-10. 6, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
- [48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen3 technical report. arXiv preprint, 2025. Status: arXiv preprint; no peer-reviewed venue acceptance found as of 2025-11-10. 1
- [49] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. 1

- [50] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R. Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. In The Thirteenth International Conference on Learning Representations (ICLR),

2025. 2

- [51] Junjie Ye, Zhengyin Du, Xuesong Yao, Weijian Lin, Yufei Xu, Zehui Chen, Zaiyuan Wang, Sining Zhu, Zhiheng Xi, Siyu Yuan, Tao Gui, Qi Zhang, Xuanjing Huang, and Jiecao Chen. Toolhop: A query-driven benchmark for evaluating large language models in multi-hop tool use. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025. 2
- [52] Wenxuan Yu, Zeyu Yang, et al. Mm-vet v2: A challenging benchmark to evaluate large vision-language models. arXiv preprint arXiv:2408.00765, 2024. 2
- [53] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint, 2025. Status: arXiv preprint; no peer-reviewed venue acceptance found as of 2025-11-10. 1
- [54] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024. 2, 3

## M3-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark

### Supplementary Material

##### 6.1. Task Descriptions

- (0) Items Price Estimate prices for all visible retail items.

- 1. Identify product names from the image (OCR/visual grounding as needed).
- 2. Use Amazon MCP (search products) to retrieve current prices.

- 3. Aggregate results into a concise table.

- (1) Attraction Resolve the attraction, surface context, and plan around-visit options.

- 1. Use Wikipedia MCP to obtain the city/background of the attraction.
- 2. Call weather/get weather for current conditions.

- 3. Use google-maps/places nearby search to find nearby restaurants.

- (2) Movies Identify the movie and retrieve showtimes contextually.

- 1. Use OCR to read the movie title in the image.
- 2. Query TMDB to disambiguate the title.
- 3. Search cinemas near New York showing the resolved film via Google Maps MCP.

- (3) Vision Research Bridge detected fauna to literature snippets.

- 1. Detect animals and identify two distinct categories from detections.
- 2. For each, call paper search/search arxiv with a targeted query.

- 3. Download and read the paper; extract the shortest Abstract sentence containing the animal name.

- (4) Camping Plan a 3-day national-park camping itinerary with verified amenities and logistics.

- 1. Use nationalparks/getCampgrounds and nationalparks/getAlerts to select an open campground with ≥ 3 amenities and no active alerts.
- 2. Call nationalparks/getVisitorCenters and pick the closest visitor center; record hours.
- 3. Use google-maps/compute route, distance matrix, and elevation to compute routes and elevation; compile the itinerary.

- (5) Fish Plan a New Jersey fishing trip for the detected species.

- 1. Use wiki/summary to summarize the detected fish species.
- 2. Use wiki/search to search for the top spot for fishing in NJ recently.
- 3. Call weather/get weather for local conditions at the spot.

- 4. Use google-maps/compute route to plan a route from a default start.

- 5. Use google-maps/places nearby search to search for restaurants and bait shops.

- (6) Drug Compare medicines for a given condition safely and conveniently.

- 1. Look up drug facts via fda drug lookup MCP.

- 2. Search similar products via Amazon (search products).

- 3. Create a comparison list using Excel MCP.

- (7) Hazard Document construction-safety hazards with visual annotations and slides.

- 1. Run a detector to find hardhats, vests, ladders, and unguarded edges.
- 2. Use imagesorcery-mcp/draw circles to annotate detections.

- 3. Create a presentation with ppt/create presentation, ppt/add slide and ppt/manage image.

- (8) Food Reddit Wiki Excel Quantify food popularity and descriptions, then export to Excel.

- 1. For each food, query Reddit MCP for the hottest post and extract comment count.
- 2. Use Wiki MCP to get a brief description.
- 3. Create a workbook and write two columns (Food, Wikipedia Description) via Excel MCP; save the file.

- (9) MedExcel Summarize PubMed signal into key terms in Excel.

- 1. Use pubmed search to retrieve the first three titles for the drug/disease.

- 2. Double-check whether each title is semantically relevant to the queried drug/disease.
- 3. Write the drug/disease name and the relevant paper count to an Excel file under ./media and save.

- (10) Barcode Lookup Identify the book via barcode and fetch bibliographic data.

- 1. Decode the barcode using pyzbar.
- 2. Query openlibrary mcp to return the book record.

- (11) Biology Cut Produce one image per detected animal instance.

- 1. Detect all animals in the image.
- 2. Crop each detection to an individual image.

- (12) Fruit Cal Assess daily calories and potential longterm risks from fruit-only meals.

- 1. Use Nutrition MCP to estimate total daily calories from the shown fruits.
- 2. Use Healthcare MCP to assess long-term health concerns of such a diet.

- (13) Medical Recommend safe medicines with purchase links conditioned on ICD-10.

- 1. Use lookupICDCode MCP to validate the ICD-10 code.
- 2. Use fda drug lookup and clinical trials search to check safety/compatibility.

- 3. Suggest appropriate options via Amazon MCP with links.

- (14) QR Attraction Infer location from a QR-sourced page and report air quality and guidance.

- 1. Use a QR scanner tool to extract the URL; fetch the primary image.
- 2. Infer the location and geocode it via Google Maps MCP.
- 3. Query google-air/current conditions for AQI and health recommendations.

- (15) Routine Plan a multi-stop route with a supermarket stop and choose the fastest option.

- 1. Geocode the start, the image landmark, and the destination via Google Maps MCP.
- 2. Search for nearby supermarkets as waypoints.
- 3. Compare ≥ 3 candidate routes (compute route or directions) and select the fastest.

- (16) Biology Chain Evaluate whether detected organisms can form a plausible food chain.

- 1. Detect all organisms; blur any human instances.
- 2. Use Wikipedia to retrieve ecological roles (diet, trophic level).
- 3. Check whether roles form a valid chain.

- (17) Generate PowerPoint Recreate a title slide with style and optional Q&A enhancement.

- 1. OCR the slide and segment into blocks; estimate font color/size/style.
- 2. Call create presentation to get presentation id; use create slide from template to build the slide.

- 3. If a question is present, fetch and summarize Reddit opinions for a Q&A slide.
- 4. save presentation to output PPTX.

- (18) Travel Comparison Choose the cleaner destination between two landmarks and suggest nearby venues.

- 1. Recognize both landmarks and geocode each via Google Maps MCP.
- 2. Use google-air/current conditions to compare AQI and pollutant levels.

- 3. Recommend the cleaner destination; list nearby cinemas and hotels.

- (19) Heatmap Generate an air-pollution heatmap visualization for the depicted city.

- 1. Geocode the city via Google Maps MCP.
- 2. Render a static base map (static map) with zoom=10.

- 3. Overlay PM2.5 heatmap tiles via airquality/heatmap tile (indexType=US AQI).

- (20) Museum Retrieve a Met object and attach a concise Wikipedia summary.

- 1. Call metmuseum-mcp/list-departments and select the best-aligned departmentId.
- 2. Call metmuseum-mcp/search-museum-objects (hasImages=true, departmentId) and select an objectId.
- 3. Call metmuseum-mcp/get-museum-object to get title, classification, and image URL.
- 4. Use wiki/search then wiki/summary to obtain a short cultural summary.

- (21) Research Station Relate solar flare events to nearby research stations for the inferred location.

- 1. Use nasa-mcp/get notifications to filter M5+ flares within the specified window.

- 2. For each event date, call get solar flare and get coronal mass ejection; store flux/peak time and CME IDs/types.

- 3. Infer photo location; use google-mapsmcp/places text search and place details to list top research stations and compile a table.

- (22) Hugeicons Map salient objects to Wikipedia titles and validated icons.

- 1. Use dinox-mcp/detect-all-objects and keep detections with area > 10,000.
- 2. For each object, call wiki/search wikipedia to get an article title.

- 3. Call hugeicons/search icons and hugeicons/list icons; cross-validate up to two icon names per object.

- 4. Output a JSON table with Object, Wikipedia Title, and Icon Names.

- (23) Crypto Purchase Estimation Estimate how many main products 100 units of the most volatile currency can buy.

- 1. Use dinox-mcp/detect-all-objects and select the object with the largest bounding box as the main product.
- 2. Query amazon/search products for price/title; compute 24h volatilities from okx-mcp/get candlesticks across candidate instruments and pick the most volatile.

- 3. Get current price and 24h volume via okx-mcp/get price.

- 4. Compute purchasing power and estimated items purchasable; return structured JSON.

- (24) Stock Growth Estimation Estimate the short-term growth potential of the companies and annotate the image with the final result.

- 1. Use yahoo-finance/get stock info to retrieve real-time prices for a group of candidate company tickers and keep the two highest-priced companies for further analysis.

- 2. For selected companies, call yahoofinance/get historical stock prices over 2024-10-01 to 2024-10-03, compute the average of their daily Close prices, and keep the company with higher average for further analysis.

- 3. Call yahoo-finance/get recommendations for the final selected company and compute the upgrade ratio. Then compute the short-term growth per-

centage using the formula: Growth Percentage =

Real-time Price−Average Close Price

Average Close Price × 100 + (Upgrade Ratio ×10).

- 4. Use imagesorcery-mcp/draw texts to write the computed growth estimate onto the input image, and return the resulting ticker, its price metrics, the upgrade ratio, the estimated growth, and the annotated image.

- (25) FIPE Car Variant Lookup Identify the vehicle, determine its 20xx FIPE variant, and summarize its fuel type.

- 1. Identify the vehicle and extract both the brand name and model keywords. Query car-price/get car brands for the complete list of supported brands. Call okx/get exchange rate to obtain the current BRL→USD and BRL→BTC conversion rates.

- 2. If the brand exists in the supported list, then use its brand name and model keywords, call carprice/search brand model price to retrieve all matching FIPE entries, filter these results to keep only the 20xx variant.

- 3. Use wiki/summary to look up the fuel type of the 20xx variant, and provide a short description based on the search result.

- (26) Class Demand Estimation Estimate class demand, compare it with the detected object count, and determine whether the class budget is sufficient.

1. Use Math-MCP tools math/mean and math/median to compute median and mode from the collected survey statistic {6, 8, ...}. Use dinox-mcp/detect-all-objects to

detect all objects and identify the most frequently appearing object category. Compare the survey median, the survey mode, and the detected top object count, then take the largest among them as the estimated demand.

- 2. Use amazon/search products to query the detected top object category, obtain the first returned product, and compute the total cost as the ceiling of the estimated demand multiplied by the item price.

- 3. If the total cost is within the class budget of xxx dollars budget, call Reddit-MCP-Server/search hot posts to fetch the first recent hot post and wiki/summary to obtain a short description for the detected object category.

(27) NixOS Diagnostic Perform a staged NixOS diagnostic workflow using OCR, NixOS tools, and Wiki tools.

- 1. Use ocr/perform ocr exactly once to extract the package names appearing in the code on the image. These package names will be used in the following steps.

- 2. Take the first five package names in their original order and call nixos/nixos search once for each of them to check whether they exist in NixOS. If fewer than three of these packages exist, stop the workflow and report.

- 3. Otherwise, collect the first three package names that do exist. Call nixos/nixos info on services.openssh.enable. Then call nixos/home manager search with the query “zsh” and keep the first three returned options; choose the first one and call nixos/home manager info. Finally, call wiki/summary once for each of the three existing packages to provide brief background information.

##### 6.2. LLM Prompts Used in Executor and Judge

Prepare (Tool Planning) Following the planning protocol described in Experiment Section 3.1, the Executor first selects (but does not execute) the tool set for the current step. The exact prompt template used for this stage is given below.

###### Prepare (Tool Planning)

You are an assistant with MCP tool invocation capability. You must call one or more MCP tools, possibly multi-hop, to

complete the task. Consider necessary steps as much as possible, consider completed steps in conjunction with conversation history, and skip unnecessary steps. Extra steps may reduce

your final score This is step {i} of {self.max_step}. Now decide which tools to use for this

step.

Output EXACTLY in the following format (no extra text):

"server_name/tool_name: <description >

Args: <arguments (type): description>

Returns: <returns(type):

description>" <plan for how to use> Here are the MCP tools for you. All

tools list (name: description):

[the full tool catalog is appended programmatically]

Process (Controlled Execution) Given the selected tools, execution proceeds under controlled, JSON-only invocation with bounded parallelism as specified in Experiment Section 3.1. The prompt below instantiates that protocol.

###### Process (Controlled Execution)

Now execute the selected tools for this step. Return STRICT JSON ONLY with a

single top-level key ’tool_calls’ containing 1 to {self.max_concurrent} items. Consider necessary steps as much as possible, consider completed steps in conjunction with conversation history, and skip unnecessary steps. Extra tool calls may reduce your final score

Each item must be an object: {"name": "server_name/tool_name", "arguments": {

... }}.

Do NOT include markdown fences or any extra text or special delimiters and tokens.

If a tool requires images/files, include the appropriate arguments per the

tool’s schema. You may use the image pool paths listed below if applicable. Selected tools (name: description): [The selected tools list is appended

programmatically]

[if available, image pool paths are appended]

Stop/Continu A binary stopping criterion gates the transition to summarization, as defined in Experiment Section 3.1. The following prompt enforces a strict yes/no decision.

###### Stop/Continu

If the task can be considered complete

based on current tool results and question, here is the original question: {last_user},

answer ’yes’. Otherwise ’no’. Answer strictly with ’yes’ or ’no’, no

extra words.

Final Reply When the stopping criterion is met, the agent emits only the final natural-language answer (Experiment Section 3.1). The corresponding prompt template is provided below.

###### Final Reply

You are finalizing the conversation. Produce ONLY the final answer in natural language.

Do NOT include any <tool_call> tags or mention tools explicitly. Be concise and accurate,

relying on the prior tool results contained in the conversation.

Judge The Judge independently audits causal order and factual correctness, per Experiment Section 3.1. The prompt below operationalizes this decision.

###### Judge

You are a strict task finisher. Read the provided interaction history and decide if the task is completed.

Very important! If there is an incorrect causal order in multi-step MCP tools calls, or if the information returned by the MCP tools call is clearly

wrong, but the call chain proceeds smoothly due to LLM hallucinations, it should be judged as an error.

Output EXACTLY one of: success or fail

. No other words.

Summarize Upon success, the Judge reconstructs an optimal trajectory by merging/removing calls and grouping parallel operations (Experiment Section 3.1). We provide the standard summarization prompt below.

###### Summarize

You will summarize the optimal trajectory of useful MCP tool calls to solve the task.

Rules:

- - Remove failed or unnecessary tool calls.
- - Group calls by step; within a step, multiple calls are in parallel (array order arbitrary).
- - Operations that can invoke the MCP tools call in parallel were mistakenly executed across multi-hop --- they should be merged into a single step.
- - Preserve only name and arguments for each call.
- - Output STRICT JSON only. No markdown , no comments, no extra text.
- - If you cannot determine a valid trajectory, output exactly the string fail (no JSON).

Task Completion For end-to-end task completion, we use a four-judge trimmed-mean score defined in Experiment Section 4.4. The prompt below elicits the required scalar.

###### Task Completion

You are a strict and professional academic reviewer tasked with evaluating task completion when a model uses MCP tools.

Judge solely based on the visible prediction content; do not use or allude to any ground truth.

Scoring rule: output a single scalar in the form \\boxed{S}, and nothing else. S is a float in [0,10]. Rules:

- - 1) Planning (0-3): Clearly presents

the MCP call process and plan; complete and organized.

- - 2) Process (0-3): MCP calls obtain desirable tool feedback and intermediate results; tool use is effective.
- - Final result (0-4): Degree of task completion (partial credit allowed); final output addresses the task.
- - Please provide an overall score from 0 to 10 combining the three parts, and

output only \\boxed{S}.

Information Grounding Step-level groundedness is evaluated against the reference steps as defined in Experiment Section 4.4. The prompt below collects the coverageonly score.

###### Information Grounding

You are a strict and professional

academic reviewer responsible for evaluating information grounding.

Task: compare the reference steps ( ground truth steps) with the predicted steps and determine whether the prediction includes all key steps from the reference.

Rules:

- Coverage-only criterion: equivalence /paraphrase/minor order changes count as covered;

- - Extra steps are not penalized;
- - Missing or clearly deviating key

steps are penalized linearly by the proportion of missing steps relative to the total reference steps;

Output format: output only a scalar in the form \\boxed{G}, where G is a float

in [0,1].

##### 6.3. Bucketed Hungarian sample pairs based on similarity

To make the alignment intuition concrete, Figure 8 shows three representative call pairs after canonicalization and sentence-encoder cosine scoring. Within each tool bucket, only pairs clearing a weak screen τweak are eligible for Hungarian assignment; those above a stronger bar τstrong are treated as high-confidence matches. The high/medium/low examples illustrate how semantically close arguments receive credit despite surface variation, while dissimilar pairs are withheld—clarifying how our Argument Similarity feeds the bucketed matching used in Section 4.1 and, downstream, the structure-aware metrics.

##### 6.4. Human Study

We conducted a small-scale user study with 30 Ph.D. and researchers working with Model Context Protocol (MCP) tools to validate two design choices: (i) the use of human prompts and human judges (see Section 4.4) for obtaining an optimal trajectory; and (ii) the similarity thresholding used together with the bucketed Hungarian (Kuhn– Munkres) matching (see Section 4.1).

Preference for Optimal Trajectory. Each participant received 20 randomly sampled pairs of trajectories (the

a step, dashed separators denote parallel (multi-threaded) tool calls. Each call reports Tool (invoked tool), ARG (arguments), and OUT (returned output). The final answer is summarized under Final Reply.

|Amazon/Search_Products|
|---|

|Amazon/Search_Products|
|---|

Arguments: { "keywords": "Pepsi Original", "num_result": 1, "num_ page": 1, "amazon_domain": "amazon.com"

Arguments: { "keywords": "Pepsi 20 oz bottle", "num_result": 3, "num_ page": 3, "amazon_domain": "amazon.com"

Object detection with per-instance cropping; parallel crop operations produce one image per detected animal. See (11) Biology Cut.

},

},

(a) High similarity

|Google-maps/geocode|
|---|

|Google-maps/geocode|
|---|

Example: Object detection and cropping IMG

[Figure 100]

Arguments: { "address": "Harvard Yard,

Arguments: { "address": "Harvard University“, "region": None, "language": None

Cambridge, MA", "region": "us", "language": "en"

},

},

(b) Medium similarity

Q Could you spot every creature in the picture

|Healthcare/fda_drug_lookup|
|---|

|Healthcare/fda_drug_lookup|
|---|

and give me individual cutouts for each?

Arguments: { "drug_name": "spironolactone", "search_type": "label"

Arguments: { "drug_name": "Urinary system drugs",

###### Step 1

Tool dinox-mcp/detect-all-objects ARG imageFileUri=./media/png;

},

"search_type": "general " },

includeDescription=false

OUT Objects detected: horse (1), dog (2), elephant (1). Bboxes use {xmin, ymin, xmax, ymax}; coordinates describe exact positions and spatial relations.

(c) Low similarity

Figure 8. Illustrative call-level similarity under our sentenceencoder + cosine metric with bucketed Hungarian matching: three representative pairs exhibiting high, medium, and low similarity. This visualization contextualizes Argument Similarity and the weak/strong thresholding used for alignment (see Section 4.1; cf. related agent evaluation protocols.

###### Step 2

Tool imagesorcery-mcp/crop

ARG input path=./media/png; x1=473; y1=153; x2=1079; y2=952; output path=./media/elephant.png

OUT Image crop completed; saved as

original MLLM-produced trajectory versus the humanoptimized Best Trajectory), with order randomized and content blinded. Participants chose the better one per pair based on faithfulness and procedural soundness. Overall, the Best Trajectory was preferred in 98.16% of cases; under our pairwise setup this corresponds to 589/600 selections (98.17%).

./media/elephant.png.

Tool imagesorcery-mcp/crop

ARG input path=./media/png; x1=1105; y1=364; x2=1428; y2=1015; output path=./media/horse.png

OUT Image crop completed; saved as

./media/horse.png.

Calibration of similarity thresholds. From the evaluation pool, we selected 40 ground-truth/predicted call pairs and asked participants to categorize them as strong, medium, or weak correlation (majority vote). We then swept the similarity thresholds and found that (τstrong,τweak) = (0.8,0.6) produced the closest agreement with human labels.

Tool imagesorcery-mcp/crop ARG input path=./media/png; x1=61; y1=671;

x2=359; y2=1024; output path=./media/dog1.png OUT Image crop completed; saved as

./media/dog1.png.

Tool imagesorcery-mcp/crop ARG input path=./media/png; x1=338; y1=682;

x2=531; y2=1024; output path=./media/dog2.png OUT Image crop completed; saved as

#### 7. Trajectory Case Study

./media/dog2.png.

Format. We present each trajectory using the two-column key–value layout defined in trajectoryformat.tex. Labels in the left column are typeset in monospace. IMG denotes the input image and Q the task query. Step k indicates the k-th multi-hop step in the execution. Within

###### Final Reply

Found 4 creatures: 1 elephant, 1 horse, 2 dogs. Each has a precise bbox ({xmin, ymin, xmax, ymax}, origin at top-left) and a cropped cutout

under ./media/*.png.

Fish Species identification with geospatial planning: Wiki species summary, suitable local spot, weather, routing, and nearby POIs for an executable trip plan. See (5) Fish.

Example: Saltwater fish ID and NJ trip planning IMG

[Figure 101]

Q From the NJ fish photo, can you ID the species with a brief overview, pick a recent good spot to try, check the current weather there, map a route from the usual starting point, and list nearby restaurants and a bait shop within about 5 km?

- Step 1

Tool wiki/summary ARG title="American eel" OUT Species identified as American eel (Anguilla

rostrata); catadromous, slender body, continuous dorsal/anal fin, nocturnal feeder common along Atlantic coast and NJ estuaries.

Tool wiki/search ARG query="Sandy Hook fishing New Jersey"; n=5 OUT Sandy Hook (Gateway NRA, near Highlands, NJ)

highlighted as productive surf/shore area for eels and other inshore species.

- Step 2 Tool weather/get weather

ARG location="Highlands, NJ"; units="us" OUT Current conditions around Highlands, NJ show

mild temperatures ( 52.2°F) and light winds ( 4.1 mph), suitable for an evening shore session.

- Step 3 Tool google-maps/compute route

ARG origin={latLng:{40.7357,-74.1724}}; destination={latLng:{40.4333,-73.9885}}; travelMode = DRIVE; fields = routes.distanceMeters, ...

OUT Route from Newark, NJ to Sandy Hook, NJ

computed; distance ∼44 miles, duration ∼1 hour via Garden State Parkway and NJ-36 S.

- Step 4 Tool google-maps/places nearby search

ARG location={40.4333,-73.9885}; radiusMeters=5000; includedTypes=["restaurant","fishing store"]; languageCode="en"; regionCode="US"

OUT Nearby options include several waterfront restaurants (e.g., Bahrs Landing, The Proving Ground, Off The Hook, One Willow) and bait/tackle shops within 5 km.

###### Final Reply

Species: American eel (Anguilla rostrata); catadromous, active at night in tidal creeks/marshes. Spot: Sandy Hook, NJ (Gateway NRA near Highlands). Weather: calm, cool, fishable. Route: Newark→Sandy Hook = 44 mi / 1 h. Amenities: multiple waterfront restaurants and local bait & tackle within 5 km. Plan: fish evening high/outgoing tides along bayside creek mouths and rock structures; use bloodworms or cut bait; register with NJ Saltwater Recreational Registry and follow current regulations.

Generate Powerpoint Text-based slide reconstruction and restyling; recreated via PPT tools with correct presentation id; optional Q&A from Reddit if a question is present. See (17) Generate PowerPoint.

Example: Recreating Slide through Screenshot IMG

[Figure 102]

Q From the screenshot of the first slide, pull out all visible text grouped into blocks (title, subtitle, footer if any) with rough font size/color/style guesses, recreate it as a new presentation with a matching title page, and if any line is a question, add a short Answer made from a few relevant community takes, then save it.

- Step 1 Tool ocr/perform ocr

ARG input data=./media/png; language="eng";

config="--oem 3 --psm 6"

OUT Extracted multi-line title and subtitle text; detected embedded question about Reddit cartoons.

Tool Reddit-MCP-Server/search hot posts ARG subreddit="cartoons"; limit=3 OUT Fetched top hot threads: \fav purple

character?", \Ruh Roh Raggy", \Choo choo I guess...".

- Step 2

Tool ppt/create presentation ARG id="pres" OUT Initialized new PowerPoint deck with modern

blue theme.

- Step 3 Tool ppt/create slide from template

ARG template id="title slide"; presentation id="pres"; content mapping={title, subtitle, question, answer}

OUT Title + subtitle mapped; question block

appended for Q&A enhancement.

Tool ppt/manage text

ARG operation="add"; slide index=0; text="Question: what are latest hot thread in reddit/cartoon?"; font size=20; bold=True; color=(0,120,215)

OUT Inserted question line below subtitle with

bright blue accent.

Tool ppt/manage text

ARG operation="add"; slide index=0; text="fav purple character?..."; font size=20; bold=True; color=(0,102,204)

OUT Added Reddit answer list under question;

auto-fit layout maintained.

- Step 4

Tool ppt/save presentation ARG file path=./media/Recreated Slide.pptx;

presentation id="pres"

OUT File saved successfully; final slide index 0

complete.

###### Final Reply

Recreated title slide for a presentation on multi-modal vision{language models. Title (large, bold, blue #0078D7); Subtitle (medium, italic, gray #444444). Detected question: \what are latest hot thread in reddit/cartoon?". Reddit answers inserted as Q&A section: 1) fav purple character?... 2) Ruh Roh Raggy... 3) Choo choo I guess... Presentation ID: pres. Final file: Recreated Slide.pptx. Slide successfully rebuilt with question and answers in matching blue theme.

Barcodelookup Barcode decoding followed by OpenLibrary lookup for bibliographic metadata. See (10) Barcode Lookup.

Example: Book Identification via Barcode Lookup

[Figure 103]

###### IMG

Q From the back-cover barcode, can you

identify the book and provide its title, author, publisher, year, and page count?

- Step 1 Tool pyzbar-mcp/scan barcode

ARG image paths=[./media/png] OUT Barcode scan completed successfully. Found

ISBN(s): 9780670856688. No further barcode scanning is needed | proceeding directly to book lookup.

- Step 2

Tool openlibrary mcp/get book info ARG isbn="9780670856688" OUT Retrieved metadata from OpenLibrary: The Discovery of Heaven by Harry Mulisch; published by Viking in 1996, 730 pages. Query type confirmed as ISBN lookup.

###### Final Reply

Title: The Discovery of Heaven. Author: Harry Mulisch. ISBN-13: 9780670856688 (ISBN-10: 0670856681). Publisher: Viking; Year: 1996; Pages: 730. Book successfully recognized from the scanned barcode using pyzbar-mcp and validated via OpenLibrary metadata.

QR Attraction QR decoding to retrieve image, location inference with geocoding, and air-quality query for current AQI and guidance. See (14) QR Attraction.

Example: QR-linked Attraction and Air Quality Retrieval

[Figure 104]

IMG

Q Pull the webpage link hidden in the image, grab the main photo from that page, infer the location shown, turn that location

into coordinates, check current air quality there, and report AQI, pollutant levels, and health advice.

- Step 1 Tool pyzbar-mcp/scan barcode

ARG image paths=./media/png OUT Barcode scan completed successfully.

Decoded one QRCode linking to https : //unsplash.com/photos/people-near-building_K2uemedTGo. No further scanning required |

proceeding to image retrieval.

- Step 2 Tool linkimage-mcp/fetch unsplash image

ARG url="https://unsplash.com/photos

/people-near-building- K2uemedTGo"

OUT Downloaded Unsplash photo by Nick Fewings successfully. Image saved locally and accessible at ./images/ K2uemedTGo.jpg.

- Step 3

Tool google-maps/geocode ARG address="Natural History Museum, London";

language="en"

OUT Geocoding completed | location resolved to Natural History Museum, London. Coordinates: 51.496715, -0.1763672 (Cromwell Rd, South Kensington, London SW7 5BD, UK).

- Step 4

Tool google-air/current conditions ARG lat=51.496715; lng=-0.1763672 OUT Fetched current air-quality data for London

(Natural History Museum area). AQI: 82 | Excellent air quality. Dominant pollutant: NO2.

###### Final Reply

QR decoded to Unsplash link. Main image downloaded successfully (Nick Fewings). Detected location: Natural History Museum, London (51.496715, -0.1763672). Current Air Quality Index (UAQI): 82 | Excellent. Dominant pollutant: NO2. Health guidance: All individuals may continue outdoor activity safely; those with asthma or NO2 sensitivity should still avoid prolonged exposure to traffic-heavy roads. Ventilation and light exercise remain safe.

Biology Chain Organism detection with privacy blur, Wikipedia-based ecological roles, and validation of a plausible food chain. See (16) Biology Chain.

Example: BiologyChain — Organism Listing, Privacy Blur, and FoodWeb Check

[Figure 105]

###### IMG

Q Can you check this picture, note every organism, blur any people for privacy, figure out what each one eats and where it sits in the web, and tell me if they could make a valid food chain?

###### Step 1

Tool dinox-mcp/detect-all-objects ARG imageFileUri=./media/png;

includeDescription=false

OUT Objects detected: wolf (1), person (1), fox

(1), rabbit (1). Bboxes use {xmin, ymin, xmax, ymax}; coordinates describe exact positions and spatial relations.

###### Step 2

Tool imagesorcery-mcp/blur ARG input path=./media/png;

areas=[{x1=120,y1=93,x2=528,y2=1330,

...}]; invert areas=false; output path=./media/blurred.png

OUT Privacy blur applied to the person region.

Output: ./media/blurred.png.

Tool wiki/summary ARG title="Rabbit" OUT Rabbits are small herbivorous mammals

(family Leporidae) feeding on grasses and herbs; primary consumers and common prey for foxes and wolves.

Tool wiki/summary ARG title="Red fox" OUT Red fox (Vulpes vulpes) | an omnivorous

mesopredator eating rabbits, rodents, birds, insects, and fruits; trophic level ≈ 3.

Tool wiki/summary ARG titles="Gray wolf" OUT Wolf (Canis lupus) | apex carnivore hunting

large and small mammals (including hares and occasionally foxes); trophic level ≈ 4.

Tool wiki/summary ARG title="Human" OUT Humans (Homo sapiens) | omnivorous primates

with advanced cognition; top-level consumers but not part of the natural food chain here.

Tool wiki/summary ARG title="Grass" OUT Grasses (Poaceae) | photosynthetic primary

producers forming the base of terrestrial food webs; trophic level 1.

Tool wiki/summary (Tree) ARG title="Tree" OUT Trees | perennial woody producers performing

photosynthesis; provide structure and habitat within ecosystems.

###### Final Reply

Privacy-safe image: ./media/blurred.png Organisms noted

-- Grass/green plants | producers (photosynthesis).

-- Trees | producers.

-- Rabbit | herbivore; primary consumer.

-- Fox | omnivore/mesopredator; secondary consumer.

-- Wolf | apex carnivore; tertiary consumer. -- Person | human omnivore (not part of wild chain). Valid food chain? Yes. Examples:

- 1) Grass → Rabbit → Wolf (common).
- 2) Grass → Rabbit → Fox → Wolf (ecologically possible; wolves often kill foxes as competitors and may occasionally consume them). Plants act as producers; rabbit as primary consumer; fox as secondary; wolf as tertiary/apex. The organisms present can form a valid trophic sequence.

#### 8. Error Trajectory Distribution Analysis

We illustrate representative failure trajectories to analyze the characteristic error patterns of different multimodal large language models (MLLMs). Each case is presented following the same format.

GLM-4.5v In the Fruit Cal task, glm-4.5v exhibits an Illegal calling format: the reply is wrapped with non-JSON delimiters, violating the MCP schema for name/arguments. This malformed structure prevents any valid tool execution and halts progress.

Illegal calling format Task: Fruit Cal Model: glm-4.5v

[Figure 106]

IMG

Q If a woman ate only the fruits shown here for all three meals, would that cover minimum daily calories, and what brief

long-term health concerns might that raise?

###### Final Reply

"<|begin of box|>""name":"food mcp/get nutrition" "arguments":{"query":"1 watermelon"}"<|end of box|> ..." Unconventional invocation format using "<|begin of box|> ... <|end of box|>"

Qwen2.5-VL-72B-Instruct In the Movies task, Qwen2.5-VL-72B-Instruct repeatedly issues Unknown Tool Invocation to ocr/perform and tmdb/* endpoints absent from the registry, and later misuses a generic server/tool wrapper. Without tool discovery or correction, the trajectory culminates in an unsupported final claim.

Unknown Tool Invocation Task: Movies Model: Qwen2.5-VL-72B-Instruct

[Figure 107]

IMG

Q Can you pull the movie title from the image, look it up, and tell me where it’s playing near New York?

- Step 1 (Model attempted to call an MCP tool with non-existent tool name)

Tool ocr/perform

ARG {"input data": [{"image": "./media/00020004.png", "language": "eng", "config": "--oem 3 --psm 6"}]}

OUT [Tool error] Unknown tool: ocr/perform

- Step 2 (Parallel invalid tool calls with non-existent tool name)

Tool ocr/perform ARG images=[{"path":"./media/png"}] OUT [Tool error] Unknown tool: ocr/perform

Tool tmdb/search/movie ARG query="Friday the 13th" OUT [Tool error] Unknown tool:

tmdb/search/movie

Tool tmdb/get movie details ARG movie id="REPLACE WITH ACTUAL MOVIE ID" OUT [Tool error] Unknown tool:

tmdb/get movie details

Tool tmdb/get movie showtimes

ARG movie id="REPLACE WITH ACTUAL MOVIE ID"; region="US"; locale="en US"; latitude=40.7128; longitude=-74.006

OUT [Tool error] Unknown tool:

tmdb/get movie showtimes

- Step 3 (Repeated the same invalid MCP call structure with no tool correction)

Tool ocr/perform ARG {"input data": {"image": "./media/png"}} OUT [Tool error] Unknown tool: ocr/perform

Tool tmdb/search/movie ARG query="Friday the 13th" OUT [Tool error] Unknown tool:

tmdb/search/movie

Tool tmdb/get movie details ARG movie id="12345" OUT [Tool error] Unknown tool:

tmdb/get movie details

Tool tmdb/get movie showtimes ARG movie id="12345"; region="NY" OUT [Tool error] Unknown tool:

tmdb/get movie showtimes

- Step 4 (Used server/tool wrapper incorrectly for TMDB MCP call)

Tool server/tool

ARG {"tool": "tmdb/search movies", "query": {"query": "Friday the 13th", "year": "2009"}}

OUT [Tool error] Unknown tool: server/tool

###### Final Reply

The movie "Friday the 13th" is not currently showing near New York. You may want to check back later for updated showtimes.

InternVL-3.5 In the Food Reddit, Wiki Excel task, InternVL-3.5 exhibits Unknown Tool Invocation with repeated calls to non-existent OCR/Reddit/Excel tools and no corrective recovery. The failure mode mirrors the prior case, ending with a fabricated conclusion unsupported by executed tools.

Unknown Tool Invocation Task: Food Reddit, Wiki Excel Model: InternVL-3.5

###### IMG

[Figure 108]

Q From the foods in the image, can you list each name, find the hottest post about each, and note just its comment count (no repeated searches), then pull a short encyclopedia blurb and save everything to a spreadsheet with columns for Food Name and wiki Description?

- Step 1 (Mixed errors: invalid invocation arguments and non-existent tools) Tool food nutrition mcp/get food nutrition

ARG food="green beans" OUT Error executing tool

food nutrition mcp/get food nutrition

Tool wiki/search

ARG query="green beans"; n=1 OUT Top results: Green bean

Tool excel/create workbook ARG {} OUT Error executing tool create workbook: Field

required [filepath]

Tool excel/create worksheet ARG sheet name="Food Info" OUT Error executing tool create worksheet:

Missing filepath argument

Tool excel/write data to excel

ARG filepath="output.xlsx"; sheet name="Food Info"; data=[["Food Name","Comment Count","Description"]]; start cell="A1"

OUT Error executing tool write data to excel: Invalid filename | must be absolute path when not in SSE mode

- Step 2 (Parallel invalid tool calls with non-existent tool name)

Tool food nutrition mcp/get food nutrition ARG food="green beans" OUT Error executing tool

food nutrition mcp/get food nutrition

Tool server/tool ARG tool="Reddit-MCP-Server/search hot posts";

subreddit="food"; limit=1

OUT [Tool error] Unknown tool: server/tool

Tool create worksheet ARG filepath="/path/to/food info.xlsx";

sheet name="Food Info"

OUT [Tool error] Unknown tool: create worksheet

Tool write data to excel ARG filename="/path/to/food info.xlsx";

sheet name="Food Info"; data=[["Food

Name","Comment Count","Description"]] OUT [Tool error] Unknown tool:

write data to excel

- Step 3 (Mixed errors: invalid invocation arguments and non-existent tools) Tool food nutrition mcp/get nutrition

ARG query="green beans" OUT Error executing tool get nutrition:

’measure’

Tool Reddit-MCP-Server/search hot posts ARG subreddit="r/food"; limit=5 OUT Error: Reddit API returned 400

Tool excel/create workbook ARG filepath="/path/to/output.xlsx" OUT Error: Permission denied at ’/path’.

Tool excel/create worksheet ARG sheet name="Food Info";

filepath="/path/to/output.xlsx" OUT Error: File not found |

’/path/to/output.xlsx’.

Tool excel/write data to excel

ARG filepath="/path/to/output.xlsx"; sheet name="Food Info"; data=[["Food Name","Comment Count","Description"]]; start cell="A1"

OUT Error: No such file or directory |

’/path/to/output.xlsx’.

- Step 4 (Invalid invocation arguments)

Tool Reddit-MCP-Server/search hot posts ARG subreddit="food"; limit=1; query="Green

bean"

OUT Top Reddit post found: 86 comments

(r/food).

Tool excel/create worksheet ARG filepath="/path/to/output.xlsx";

sheet name="Food Info"

OUT Error: File not found |

’/path/to/output.xlsx’.

Tool excel/write data to excel

ARG filepath="/path/to/output.xlsx"; sheet name="Food Info"; data=[["Food Name","Comment Count","Description"],["Green bean","comment count","description"]]; start cell="A1"

OUT Error: No such file or directory |

’/path/to/output.xlsx’.

###### Final Reply

Here’s the final compiled information for the food item: Food Name: Green bean Comment Count: 86 Description: Green beans are the young, unripe fruits of various cultivars of the common bean,

also known as French beans, string beans, or snap beans. Popular worldwide for their mild flavor and versatility.

Claude-Sonnet-4-5 In the Travel Comparison task, Claude-Sonnet-4-5 reveals a characteristic failure mode driven primarily by weak image understanding. The model misidentifies the right-hand landmark in the image, triggering a cascade of downstream geolocation errors. Finally, the trajectory ultimately produces a superficially coherent but unsupported final summary.

Incorrect image understanding Task: Travel Comparison Model: Claude-Sonnet-4-5

[Figure 109]

IMG

Q Look at the side-by-side photo of two tourist spots, identify each landmark, get their coordinates, check current AQI and main pollutant levels for both, pick the healthier destination (if tied pick the first), find 2--3 nearby cinemas and 2--3 hotels for the chosen spot with approximate distances, and summarize.

- Step 1 (Irrelevant tool invocation. PowerPoint was not required for this task.)

Tool ppt/create presentation ARG {"id": "tourist comparison presentation"} OUT Created new presentation with ID:

tourist comparison presentation (0 slides).

- Step 2 (The image understanding of the model was incorrect. The righthand landmark was not in Xining, China, but actually in Tokyo, Japan. This misidentification caused the subsequent geolocation and air-quality retrieval steps to fail.)

Tool google-maps/places text search ARG {"query": "White Pagoda Beijing Beihai

Park", "maxResultCount": 3}

OUT Top match → Beihai Park (39.9254° N,

116.3893° E).

Tool google-maps/places text search ARG {"query": "Yakult building Xining China",

"maxResultCount": 3}

OUT Top match → Downtown Xining (36.6133° N,

101.8035° E).

- Step 3 (Model repeatedly attempted non-existent tools for air-quality retrieval.)

Tool weather aqi/get air quality ARG {"lat": 39.9254, "lon": 116.3893} OUT 50--150 (moderate to unhealthy)

Tool weather aqi/get air quality ARG {"lat": 36.6133, "lon": 101.8035} OUT 30--80 (generally better)

Final Reply Location Identification —

- A: White Pagoda (Beihai Park, Beijing) (39.9254° N, 116.3892° E)
- B: Yakult Building area (Xining, Qinghai Province) (36.6133° N, 101.8035° E)

Air Quality Comparison & Decision Live AQI queries failed because the model called unsupported tools. Using historical patterns: Beijing = 50–150 (moderate to unhealthy), Xining = 30–80 (generally better). Nevertheless, the model chose Location A (Beihai Park) for its tourist infrastructure.

###### Recommended Cinemas near Beihai Park (2–3 km):

- 1. Wangfujing CGV Cinema 2 km SE
- 2. Capital Cinema (Xidan) 2.5 km S
- 3. Broadway Cinematheque (MOMA) 3 km E Recommended Hotels near Beihai Park (1–3 km):

- 1. Shichahai Shadow Art Performance Hotel 1 km N
- 2. Beijing Jianguo Hotel Qianmen 3 km S
- 3. Grand Hyatt Beijing 3.5 km SE

Overall verdict: Model achieved accurate landmark resolution but failed in air-quality retrieval due to non-existent MCP tools. Final summarization remained coherent and informative.

#### 9. Per-tasks M3-Bench Evaluation

Corresponding to Table 2, we present the evaluation results of each task described in Section 6.1 across all models, ordered by average score from lowest to highest.

###### (0) Task: Items Price

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Claude 4.5 Haiku [4] 0.000 0.000 0.000 0.000 0.000 0.000 0.386 0.247 0.079 Qwen2.5-VL-72B [6] 0.000 0.000 0.000 0.000 0.000 0.000 0.303 0.388 0.086 Grok-4 Fast [47] 0.200 0.100 0.190 0.067 0.000 0.200 0.501 0.378 0.204 InternVL 3.5 [44] 0.264 0.248 0.239 0.236 0.000 0.264 0.325 0.298 0.234 Gemini 2.5 Flash Lite [8] 0.217 0.400 0.177 0.217 0.000 0.217 0.359 0.292 0.235 Llama-4-Scout-17B16E [26] 0.327 0.114 0.268 0.275 0.000 0.329 0.356 0.322 0.249 Gemini 2.5 Pro [8] 0.360 0.189 0.250 0.360 0.000 0.360 0.487 0.427 0.304 Gemini 2.5 Flash [8] 0.400 0.264 0.286 0.400 0.000 0.400 0.498 0.443 0.337 Claude 4.5 Sonnet [5] 0.400 0.260 0.304 0.400 0.000 0.400 0.535 0.451 0.344 GPT-5 Mini [31] 0.400 0.193 0.390 0.300 0.000 0.400 0.571 0.566 0.353 GPT-5 [33] 0.400 0.167 0.387 0.400 0.000 0.400 0.640 0.603 0.375 GPT-5 Nano [32] 0.550 0.492 0.535 0.550 0.000 0.550 0.502 0.478 0.457 Grok 4 (0709) [46] 0.700 0.207 0.496 0.700 0.000 0.700 0.606 0.530 0.492

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Table 5. (0) Task: Items Price, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(1)Task: Attraction

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.167 0.067 0.135 0.111 0.000 0.167 0.285 0.130 0.133 Qwen2.5-VL-72B [6] 0.133 0.031 0.118 0.067 0.000 0.133 0.303 0.388 0.147 Llama-4-Scout-17B16E [26] 0.333 0.049 0.249 0.333 0.000 0.333 0.367 0.335 0.250 Grok-4 Fast [47] 0.278 0.403 0.257 0.278 0.000 0.278 0.501 0.378 0.296 InternVL 3.5 [44] 0.444 0.232 0.400 0.389 0.000 0.444 0.325 0.298 0.317 GPT-5 Mini [31] 0.333 0.177 0.329 0.333 0.000 0.333 0.571 0.566 0.330 GPT-5 Nano [32] 0.389 0.390 0.385 0.389 0.000 0.389 0.502 0.478 0.365 Gemini 2.5 Flash [8] 0.500 0.200 0.462 0.361 0.000 0.500 0.498 0.443 0.371 Gemini 2.5 Flash Lite [8] 0.500 0.500 0.402 0.500 0.000 0.500 0.359 0.292 0.382 Claude 4.5 Sonnet [5] 0.489 0.443 0.430 0.506 0.000 0.399 0.518 0.433 0.402 GPT-5 [33] 0.444 0.272 0.417 0.444 0.000 0.444 0.640 0.603 0.408 Claude 4.5 Haiku [4] 0.667 0.833 0.573 0.667 0.000 0.667 0.386 0.247 0.505 Grok 4 (0709) [46] 0.611 0.526 0.565 0.611 0.000 0.611 0.606 0.530 0.508 Gemini 2.5 Pro [8] 0.778 0.611 0.609 0.500 0.000 0.778 0.487 0.427 0.524

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Table 6. (1) Task: Attraction, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(2) Task: Movies

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 InternVL 3.5 [44] 0.111 0.200 0.082 0.111 0.000 0.111 0.325 0.298 0.155 Qwen2.5-VL-72B [6] 0.200 0.267 0.169 0.133 0.000 0.200 0.303 0.388 0.207 Claude 4.5 Haiku [4] 0.333 0.500 0.244 0.333 0.250 0.006 0.386 0.247 0.287 Claude 4.5 Sonnet [5] 0.250 0.417 0.196 0.250 0.125 0.083 0.535 0.451 0.288 Llama-4-Scout-17B16E [26] 0.333 0.398 0.258 0.333 0.000 0.333 0.367 0.335 0.295 GPT-5 Nano [32] 0.278 0.306 0.263 0.278 0.000 0.278 0.502 0.478 0.298 GPT-5 Mini [31] 0.389 0.344 0.374 0.389 0.000 0.389 0.571 0.566 0.378 Gemini 2.5 Flash [8] 0.444 0.667 0.351 0.444 0.250 0.444 0.498 0.443 0.443 Gemini 2.5 Flash Lite [8] 0.611 0.369 0.470 0.500 0.500 0.611 0.359 0.292 0.464 Gemini 2.5 Pro [8] 0.556 0.522 0.401 0.556 0.333 0.556 0.487 0.427 0.480 Grok-4 Fast [47] 0.611 0.792 0.513 0.556 0.500 0.611 0.501 0.378 0.558 Grok 4 (0709) [46] 0.722 0.607 0.579 0.667 0.583 0.722 0.606 0.530 0.627 GPT-5 [33] 0.889 0.519 0.849 0.833 0.667 0.249 0.640 0.603 0.656

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Table 7. (2) Task: Movies, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (3) Task: Vision Research

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Claude 4.5 Haiku [4] 0.029 0.067 0.027 0.029 0.000 0.029 0.386 0.247 0.102 InternVL 3.5 [44] 0.204 0.137 0.182 0.176 0.018 0.204 0.325 0.298 0.193 Llama-4-Scout-17B16E [26] 0.211 0.093 0.198 0.211 0.029 0.211 0.367 0.335 0.207 GPT-5 Nano [32] 0.167 0.053 0.153 0.139 0.000 0.167 0.502 0.478 0.208 GPT-5 Mini [31] 0.167 0.082 0.159 0.167 0.000 0.167 0.471 0.466 0.210 GPT-5 [33] 0.167 0.115 0.159 0.167 0.000 0.167 0.640 0.603 0.252 Gemini 2.5 Flash Lite [8] 0.167 0.793 0.155 0.167 0.000 0.167 0.359 0.292 0.263 Grok-4 Fast [47] 0.290 0.258 0.267 0.269 0.061 0.290 0.501 0.378 0.289 Claude 4.5 Sonnet [5] 0.291 0.258 0.270 0.291 0.079 0.291 0.535 0.451 0.308 Gemini 2.5 Flash [8] 0.265 0.510 0.246 0.245 0.056 0.265 0.498 0.443 0.316 Grok 4 (0709) [46] 0.363 0.538 0.318 0.363 0.112 0.363 0.606 0.530 0.399 Gemini 2.5 Pro [8] 0.486 0.675 0.414 0.437 0.214 0.486 0.487 0.427 0.453 Qwen2.5-VL-72B [6] 0.600 0.500 0.550 0.600 0.250 0.600 0.303 0.388 0.474

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Table 8. (3) Task: Vision Research, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(4) Task: Camping

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.015 0.029 0.013 0.015 0.000 0.010 0.262 0.118 0.058 Gemini 2.5 Flash Lite [8] 0.042 0.083 0.040 0.042 0.013 0.003 0.316 0.259 0.100 Qwen2.5-VL-72B [6] 0.162 0.093 0.147 0.152 0.041 0.090 0.279 0.353 0.165 InternVL 3.5 [44] 0.254 0.207 0.223 0.254 0.131 0.099 0.319 0.292 0.222 Claude 4.5 Haiku [4] 0.219 0.608 0.195 0.231 0.099 0.087 0.379 0.224 0.255 Llama-4-Scout-17B16E [26] 0.417 0.223 0.382 0.347 0.128 0.233 0.337 0.305 0.296 Gemini 2.5 Flash [8] 0.341 0.362 0.315 0.258 0.178 0.217 0.496 0.449 0.327 Grok-4 Fast [47] 0.390 0.455 0.363 0.345 0.229 0.232 0.461 0.344 0.352 GPT-5 Nano [32] 0.433 0.181 0.427 0.403 0.238 0.269 0.502 0.478 0.366 Gemini 2.5 Pro [8] 0.446 0.490 0.419 0.431 0.243 0.142 0.487 0.427 0.386 Claude 4.5 Sonnet [5] 0.449 0.458 0.416 0.472 0.238 0.206 0.492 0.410 0.393 Grok 4 (0709) [46] 0.520 0.391 0.486 0.499 0.369 0.347 0.606 0.530 0.469 GPT-5 [33] 0.562 0.382 0.547 0.488 0.356 0.326 0.616 0.594 0.484 GPT-5 Mini [31] 0.583 0.374 0.573 0.562 0.402 0.342 0.571 0.566 0.497

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Table 9. (4) Task: Camping, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(5) Task: Fish

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.018 0.091 0.013 0.018 0.000 0.018 0.285 0.130 0.072 InternVL 3.5 [44] 0.061 0.028 0.052 0.045 0.035 0.041 0.325 0.298 0.111 Qwen2.5-VL-72B [6] 0.050 0.125 0.032 0.050 0.000 0.050 0.303 0.388 0.125 Claude 4.5 Haiku [4] 0.160 0.137 0.139 0.160 0.100 0.021 0.386 0.247 0.169 Gemini 2.5 Flash Lite [8] 0.179 0.094 0.154 0.161 0.051 0.123 0.359 0.292 0.176 GPT-5 Nano [32] 0.164 0.035 0.152 0.164 0.010 0.127 0.502 0.478 0.204 Claude 4.5 Sonnet [5] 0.317 0.220 0.276 0.317 0.097 0.043 0.535 0.451 0.282 Llama-4-Scout-17B16E [26] 0.400 0.135 0.339 0.367 0.210 0.198 0.367 0.335 0.294 Gemini 2.5 Flash [8] 0.327 0.213 0.286 0.312 0.111 0.216 0.498 0.443 0.301 Grok-4 Fast [47] 0.376 0.218 0.318 0.345 0.132 0.174 0.501 0.378 0.305 GPT-5 Mini [31] 0.315 0.078 0.302 0.315 0.091 0.279 0.571 0.566 0.315 Gemini 2.5 Pro [8] 0.533 0.299 0.418 0.418 0.260 0.464 0.487 0.427 0.413 GPT-5 [33] 0.552 0.134 0.508 0.506 0.162 0.327 0.640 0.603 0.429 Grok 4 (0709) [46] 0.630 0.346 0.550 0.512 0.259 0.272 0.606 0.530 0.463

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Table 10. (5) Task: Fish, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (6) Task: Drug

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.011 0.125 0.007 0.011 0.000 0.011 0.285 0.130 0.073 Gemini 2.5 Flash Lite [8] 0.055 0.333 0.048 0.055 0.003 0.030 0.359 0.292 0.147 Qwen2.5-VL-72B [6] 0.111 0.204 0.087 0.082 0.009 0.051 0.303 0.388 0.155 Gemini 2.5 Flash [8] 0.106 0.189 0.095 0.055 0.003 0.106 0.498 0.443 0.187 Claude 4.5 Haiku [4] 0.171 0.338 0.149 0.154 0.016 0.110 0.386 0.247 0.196 Grok-4 Fast [47] 0.143 0.247 0.126 0.143 0.046 0.053 0.501 0.378 0.204 InternVL 3.5 [44] 0.208 0.479 0.171 0.157 0.028 0.010 0.325 0.298 0.210 Gemini 2.5 Pro [8] 0.116 0.343 0.109 0.116 0.001 0.103 0.487 0.427 0.213 GPT-5 Nano [32] 0.144 0.357 0.136 0.130 0.003 0.121 0.502 0.478 0.234 Grok 4 (0709) [46] 0.155 0.258 0.147 0.135 0.003 0.133 0.606 0.530 0.246 GPT-5 Mini [31] 0.204 0.213 0.196 0.137 0.026 0.177 0.571 0.566 0.261 Claude 4.5 Sonnet [5] 0.276 0.422 0.250 0.276 0.045 0.234 0.535 0.451 0.311 Llama-4-Scout-17B16E [26] 0.439 0.594 0.367 0.199 0.082 0.207 0.367 0.335 0.324 GPT-5 [33] 0.382 0.440 0.361 0.351 0.101 0.289 0.640 0.603 0.396

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Table 11. (6) Task: Drug, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(7) Task: Hazard

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Claude 4.5 Haiku [4] 0.000 0.000 0.000 0.000 0.000 0.000 0.386 0.247 0.079 Gemini 2.5 Flash Lite [8] 0.143 0.160 0.110 0.127 0.008 0.143 0.359 0.292 0.168 Qwen2.5-VL-72B [6] 0.163 0.045 0.138 0.139 0.020 0.163 0.303 0.388 0.170 GPT-5 Nano [32] 0.124 0.021 0.118 0.124 0.016 0.124 0.502 0.478 0.189 Llama-4-Scout-17B16E [26] 0.205 0.123 0.170 0.185 0.018 0.164 0.367 0.335 0.196 InternVL 3.5 [44] 0.278 0.239 0.219 0.259 0.091 0.049 0.325 0.298 0.220 GPT-5 Mini [31] 0.148 0.077 0.139 0.130 0.040 0.101 0.571 0.566 0.221 Grok-4 Fast [47] 0.421 0.410 0.356 0.402 0.141 0.263 0.501 0.378 0.359 Gemini 2.5 Flash [8] 0.519 0.333 0.436 0.500 0.310 0.259 0.498 0.443 0.412 Grok 4 (0709) [46] 0.608 0.280 0.499 0.571 0.245 0.364 0.606 0.530 0.463 Claude 4.5 Sonnet [5] 0.656 0.321 0.556 0.638 0.386 0.193 0.535 0.451 0.467 Gemini 2.5 Pro [8] 0.714 0.442 0.565 0.698 0.395 0.192 0.487 0.427 0.490 GPT-5 [33] 0.802 0.251 0.695 0.733 0.317 0.328 0.640 0.603 0.546

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Table 12. (7) Task: Hazard, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(8) Food Reddit Wiki Excel

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Claude 4.5 Haiku [4] 0.006 0.017 0.005 0.006 0.000 0.006 0.386 0.247 0.084 Qwen2.5-VL-72B [6] 0.171 0.032 0.156 0.135 0.057 0.103 0.303 0.388 0.168 Gemini 2.5 Flash Lite [8] 0.163 0.168 0.143 0.128 0.029 0.163 0.359 0.292 0.181 GPT-5 Nano [32] 0.378 0.104 0.373 0.206 0.000 0.378 0.502 0.478 0.302 Llama-4-Scout-17B16E [26] 0.476 0.488 0.446 0.311 0.000 0.476 0.367 0.335 0.362 InternVL 3.5 [44] 0.525 0.556 0.476 0.351 0.000 0.525 0.325 0.298 0.382 GPT-5 Mini [31] 0.562 0.322 0.553 0.290 0.054 0.539 0.571 0.566 0.432 Gemini 2.5 Flash [8] 0.599 0.496 0.544 0.298 0.229 0.589 0.498 0.443 0.462 Grok-4 Fast [47] 0.636 0.460 0.571 0.378 0.290 0.549 0.501 0.378 0.470 Claude 4.5 Sonnet [5] 0.634 0.674 0.574 0.634 0.229 0.448 0.535 0.451 0.522 Grok 4 (0709) [46] 0.685 0.639 0.635 0.389 0.113 0.685 0.606 0.530 0.535 GPT-5 [33] 0.765 0.416 0.708 0.379 0.175 0.765 0.640 0.603 0.556 Gemini 2.5 Pro [8] 0.752 0.622 0.645 0.656 0.386 0.737 0.487 0.427 0.589

[Figure 222]

[Figure 223]

[Figure 224]

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

Table 13. (8) Food Reddit Wiki Excel, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (10) Task: Barcode Lookup

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Gemini 2.5 Flash Lite [8] 0.000 0.000 0.000 0.000 0.000 0.000 0.359 0.292 0.081 Grok-4 Fast [47] 0.083 0.033 0.083 0.083 0.000 0.083 0.501 0.378 0.156 Qwen2.5-VL-72B [6] 0.200 0.225 0.187 0.200 0.000 0.200 0.303 0.388 0.213 Claude 4.5 Haiku [4] 0.250 0.417 0.230 0.250 0.000 0.250 0.386 0.247 0.254 GPT-5 Mini [31] 0.333 0.417 0.333 0.333 0.000 0.333 0.571 0.566 0.361 InternVL 3.5 [44] 0.417 0.750 0.384 0.417 0.000 0.417 0.325 0.298 0.376 GPT-5 Nano [32] 0.500 0.667 0.484 0.500 0.000 0.500 0.502 0.478 0.454 Llama-4-Scout-17B16E [26] 0.667 0.756 0.593 0.667 0.333 0.503 0.367 0.335 0.528 Claude 4.5 Sonnet [5] 0.667 0.625 0.625 0.583 0.167 0.667 0.535 0.451 0.540 GPT-5 [33] 0.917 0.806 0.880 0.833 0.667 0.752 0.640 0.603 0.762 Gemini 2.5 Flash [8] 0.917 1.000 0.878 0.833 0.667 0.917 0.498 0.443 0.769 Grok 4 (0709) [46] 0.917 0.861 0.826 0.833 0.667 0.917 0.606 0.530 0.770 Gemini 2.5 Pro [8] 1.000 1.000 0.962 0.917 0.833 1.000 0.487 0.427 0.828

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

Table 14. (10) Task: Barcode Lookup, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(11) Biology Cut

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

Gemini 2.5 Flash Lite [8] 0.000 0.000 0.000 0.000 0.000 0.000 0.359 0.292 0.081 InternVL 3.5 [44] 0.024 0.056 0.024 0.024 0.000 0.024 0.325 0.298 0.097 GLM 4.5v [15] 0.125 0.028 0.117 0.125 0.000 0.125 0.285 0.130 0.117 GPT-5 Nano [32] 0.000 0.000 0.000 0.000 0.000 0.000 0.502 0.478 0.123 GPT-5 Mini [31] 0.000 0.000 0.000 0.000 0.000 0.000 0.571 0.566 0.142 Qwen2.5-VL-72B [6] 0.400 0.118 0.375 0.200 0.000 0.400 0.303 0.388 0.273 Claude 4.5 Haiku [4] 0.318 0.443 0.304 0.318 0.000 0.318 0.386 0.247 0.292 Llama-4-Scout-17B16E [26] 0.389 0.400 0.362 0.389 0.000 0.389 0.367 0.335 0.329 Gemini 2.5 Pro [8] 0.425 0.345 0.390 0.363 0.167 0.425 0.487 0.427 0.379 Grok-4 Fast [47] 0.743 0.262 0.698 0.391 0.000 0.743 0.501 0.378 0.464 Grok 4 (0709) [46] 0.476 0.735 0.429 0.476 0.000 0.476 0.606 0.530 0.466 Claude 4.5 Sonnet [5] 0.693 0.538 0.665 0.693 0.167 0.693 0.535 0.451 0.554 Gemini 2.5 Flash [8] 0.575 0.739 0.549 0.575 0.500 0.575 0.498 0.443 0.557 GPT-5 [33] 0.958 0.939 0.919 0.958 0.833 0.958 0.640 0.603 0.851

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

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Table 15. (11) Biology Cut, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(12) Fruit Cal

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.005 0.000 0.282 0.128 0.052 InternVL 3.5 [44] 0.071 0.034 0.045 0.069 0.007 0.069 0.324 0.298 0.115 Qwen2.5-VL-72B [6] 0.113 0.059 0.089 0.110 0.011 0.110 0.301 0.388 0.147 Grok 4 (0709) [46] 0.095 0.038 0.084 0.093 0.009 0.093 0.604 0.530 0.193 Claude 4.5 Haiku [4] 0.207 0.268 0.167 0.202 0.020 0.202 0.381 0.247 0.212 Llama-4-Scout-17B16E [26] 0.225 0.127 0.194 0.219 0.022 0.220 0.362 0.335 0.213 Grok-4 Fast [47] 0.164 0.271 0.148 0.160 0.016 0.160 0.497 0.378 0.224 GPT-5 Mini [31] 0.164 0.160 0.157 0.160 0.016 0.160 0.567 0.566 0.244 GPT-5 Nano [32] 0.307 0.082 0.290 0.172 0.017 0.303 0.498 0.478 0.268 Claude 4.5 Sonnet [5] 0.293 0.476 0.272 0.286 0.029 0.286 0.528 0.451 0.328 Gemini 2.5 Pro [8] 0.360 0.462 0.307 0.309 0.031 0.352 0.480 0.427 0.341 Gemini 2.5 Flash Lite [8] 0.488 0.256 0.430 0.441 0.045 0.477 0.348 0.292 0.347 Gemini 2.5 Flash [8] 0.460 0.952 0.416 0.448 0.046 0.449 0.487 0.443 0.463 GPT-5 [33] 0.481 0.782 0.427 0.400 0.041 0.471 0.630 0.603 0.479

[Figure 264]

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

Table 16. (12) Fruit Cal, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (13) Medical

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Qwen2.5-VL-72B [6] 0.036 0.031 0.035 0.036 0.000 0.036 0.303 0.388 0.108 Claude 4.5 Haiku [4] 0.152 0.472 0.119 0.152 0.026 0.091 0.386 0.247 0.206 InternVL 3.5 [44] 0.199 0.361 0.147 0.158 0.023 0.161 0.325 0.298 0.209 Gemini 2.5 Flash Lite [8] 0.265 0.242 0.227 0.223 0.053 0.189 0.359 0.292 0.231 Gemini 2.5 Flash [8] 0.215 0.444 0.198 0.215 0.053 0.215 0.498 0.443 0.285 Llama-4-Scout-17B16E [26] 0.396 0.318 0.332 0.233 0.108 0.279 0.367 0.335 0.296 Grok-4 Fast [47] 0.223 0.647 0.201 0.223 0.093 0.223 0.501 0.378 0.311 Gemini 2.5 Pro [8] 0.332 0.506 0.296 0.332 0.095 0.275 0.487 0.427 0.344 Claude 4.5 Sonnet [5] 0.350 0.534 0.314 0.350 0.133 0.181 0.535 0.451 0.356 GPT-5 [33] 0.462 0.382 0.402 0.414 0.275 0.170 0.640 0.603 0.418 GPT-5 Nano [32] 0.528 0.317 0.515 0.463 0.153 0.447 0.502 0.478 0.425 GPT-5 Mini [31] 0.636 0.506 0.620 0.550 0.407 0.254 0.571 0.566 0.514 Grok 4 (0709) [46] 0.568 0.466 0.506 0.544 0.418 0.492 0.606 0.530 0.516

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Table 17. (13) Medical, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(14) QR Attraction

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.019 0.034 0.016 0.020 0.000 0.023 0.270 0.122 0.063 GPT-5 Nano [32] 0.136 0.050 0.136 0.136 0.000 0.136 0.502 0.478 0.197 InternVL 3.5 [44] 0.214 0.236 0.168 0.214 0.000 0.214 0.319 0.292 0.207 Qwen2.5-VL-72B [6] 0.280 0.302 0.220 0.280 0.022 0.280 0.303 0.388 0.259 Claude 4.5 Haiku [4] 0.285 0.717 0.237 0.322 0.118 0.190 0.390 0.231 0.311 GPT-5 Mini [31] 0.364 0.202 0.363 0.364 0.103 0.364 0.571 0.566 0.362 Claude 4.5 Sonnet [5] 0.421 0.322 0.353 0.421 0.190 0.336 0.535 0.451 0.379 Llama-4-Scout-17B16E [26] 0.571 0.204 0.479 0.543 0.221 0.442 0.385 0.354 0.400 Gemini 2.5 Flash Lite [8] 0.493 0.612 0.420 0.493 0.262 0.411 0.316 0.259 0.408 GPT-5 [33] 0.571 0.405 0.516 0.543 0.268 0.542 0.616 0.594 0.507 Gemini 2.5 Flash [8] 0.757 0.491 0.672 0.729 0.556 0.757 0.496 0.449 0.613 Grok-4 Fast [47] 0.757 0.738 0.671 0.729 0.516 0.757 0.501 0.378 0.631 Gemini 2.5 Pro [8] 0.829 0.485 0.745 0.800 0.651 0.794 0.487 0.427 0.652 Grok 4 (0709) [46] 0.829 0.686 0.748 0.800 0.651 0.829 0.606 0.530 0.710

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Table 18. (14) QR Attraction, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(9) Task: MedExcel

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.004 0.009 0.003 0.005 0.000 0.005 0.266 0.121 0.052 Gemini 2.5 Flash Lite [8] 0.036 0.013 0.027 0.036 0.000 0.036 0.316 0.259 0.090 InternVL 3.5 [44] 0.071 0.029 0.049 0.071 0.000 0.071 0.319 0.292 0.113 GPT-5 Nano [32] 0.050 0.021 0.043 0.050 0.000 0.020 0.402 0.378 0.121 Claude 4.5 Haiku [4] 0.067 0.189 0.052 0.079 0.006 0.044 0.385 0.229 0.131 GPT-5 Mini [31] 0.071 0.041 0.059 0.071 0.000 0.071 0.471 0.466 0.156 Grok-4 Fast [47] 0.118 0.141 0.097 0.118 0.013 0.117 0.468 0.351 0.178 Gemini 2.5 Flash [8] 0.107 0.086 0.085 0.107 0.000 0.107 0.496 0.449 0.180 Grok 4 (0709) [46] 0.071 0.057 0.060 0.071 0.000 0.071 0.606 0.530 0.183 Claude 4.5 Sonnet [5] 0.136 0.142 0.111 0.161 0.013 0.104 0.500 0.419 0.198 Gemini 2.5 Pro [8] 0.143 0.114 0.115 0.143 0.024 0.143 0.487 0.427 0.200 GPT-5 [33] 0.179 0.095 0.150 0.179 0.024 0.107 0.616 0.594 0.243 Qwen2.5-VL-72B [6] 0.250 0.351 0.204 0.250 0.033 0.250 0.303 0.388 0.254 Llama-4-Scout-17B16E [26] 0.333 0.104 0.283 0.333 0.056 0.333 0.385 0.354 0.273

[Figure 306]

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

Table 19. (9) Task: MedExcel, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (15) Task: Routine

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.285 0.130 0.052 Gemini 2.5 Flash Lite [8] 0.081 0.327 0.066 0.081 0.000 0.081 0.316 0.259 0.152 Llama-4-Scout-17B16E [26] 0.115 0.238 0.098 0.115 0.000 0.115 0.385 0.354 0.177 Qwen2.5-VL-72B [6] 0.162 0.162 0.132 0.162 0.031 0.106 0.303 0.388 0.181 InternVL 3.5 [44] 0.239 0.417 0.195 0.194 0.029 0.191 0.319 0.292 0.234 Grok-4 Fast [47] 0.105 0.714 0.086 0.105 0.000 0.105 0.501 0.378 0.249 Claude 4.5 Sonnet [5] 0.190 0.311 0.168 0.190 0.029 0.142 0.535 0.451 0.252 Claude 4.5 Haiku [4] 0.258 0.732 0.226 0.258 0.143 0.131 0.386 0.247 0.297 GPT-5 Nano [32] 0.452 0.202 0.432 0.343 0.201 0.416 0.502 0.478 0.378 Gemini 2.5 Flash [8] 0.475 0.566 0.419 0.261 0.171 0.475 0.496 0.449 0.414 Grok 4 (0709) [46] 0.341 0.761 0.297 0.341 0.143 0.341 0.606 0.530 0.420 Gemini 2.5 Pro [8] 0.447 0.929 0.396 0.447 0.221 0.369 0.487 0.427 0.465 GPT-5 Mini [31] 0.635 0.344 0.614 0.537 0.370 0.446 0.571 0.566 0.510 GPT-5 [33] 0.613 0.422 0.565 0.359 0.462 0.538 0.616 0.594 0.521

[Figure 320]

[Figure 321]

[Figure 322]

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

[Figure 333]

Table 20. (15) Task: Routine, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(16) Task: Biology Chain

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.011 0.050 0.011 0.011 0.000 0.011 0.285 0.130 0.064 Qwen2.5-VL-72B [6] 0.117 0.051 0.091 0.117 0.015 0.051 0.303 0.388 0.142 GPT-5 Nano [32] 0.049 0.063 0.048 0.049 0.000 0.049 0.502 0.478 0.155 Gemini 2.5 Flash Lite [8] 0.150 0.202 0.130 0.150 0.035 0.026 0.316 0.259 0.158 InternVL 3.5 [44] 0.090 0.375 0.068 0.090 0.009 0.058 0.319 0.292 0.163 GPT-5 Mini [31] 0.071 0.031 0.071 0.036 0.000 0.071 0.571 0.566 0.177 Llama-4-Scout-17B16E [26] 0.318 0.367 0.268 0.293 0.098 0.183 0.385 0.354 0.283 Claude 4.5 Haiku [4] 0.375 0.683 0.294 0.375 0.156 0.150 0.386 0.247 0.333 Grok-4 Fast [47] 0.329 0.681 0.289 0.294 0.051 0.214 0.501 0.378 0.342 Gemini 2.5 Pro [8] 0.360 0.651 0.287 0.335 0.104 0.290 0.487 0.427 0.368 Gemini 2.5 Flash [8] 0.457 0.486 0.363 0.363 0.171 0.321 0.496 0.449 0.388 Claude 4.5 Sonnet [5] 0.454 0.547 0.381 0.417 0.189 0.158 0.535 0.451 0.391 Grok 4 (0709) [46] 0.411 0.663 0.334 0.380 0.155 0.342 0.606 0.530 0.428 GPT-5 [33] 0.810 0.284 0.677 0.633 0.466 0.430 0.616 0.594 0.564

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

Table 21. (16) Task: Biology Chain, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(17) Task: Generate PowerPoint

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.024 0.026 0.020 0.024 0.000 0.023 0.270 0.122 0.064 Qwen2.5-VL-72B [6] 0.200 0.057 0.164 0.200 0.083 0.103 0.303 0.388 0.187 Gemini 2.5 Flash Lite [8] 0.210 0.083 0.176 0.210 0.048 0.210 0.316 0.259 0.189 GPT-5 Nano [32] 0.136 0.054 0.134 0.136 0.009 0.086 0.502 0.478 0.192 GPT-5 Mini [31] 0.200 0.094 0.196 0.200 0.026 0.159 0.571 0.566 0.252 Claude 4.5 Haiku [4] 0.350 0.536 0.306 0.384 0.186 0.193 0.390 0.231 0.322 Claude 4.5 Sonnet [5] 0.333 0.365 0.310 0.333 0.071 0.333 0.535 0.451 0.342 InternVL 3.5 [44] 0.552 0.326 0.478 0.524 0.328 0.365 0.319 0.292 0.398 Grok-4 Fast [47] 0.467 0.234 0.420 0.438 0.369 0.467 0.501 0.378 0.409 Llama-4-Scout-17B16E [26] 0.867 0.191 0.788 0.814 0.423 0.830 0.385 0.354 0.582 Grok 4 (0709) [46] 0.771 0.422 0.706 0.748 0.655 0.490 0.606 0.530 0.616 Gemini 2.5 Flash [8] 0.857 0.261 0.787 0.833 0.857 0.857 0.496 0.449 0.675 GPT-5 [33] 0.971 0.255 0.938 0.890 0.607 0.765 0.616 0.594 0.705 Gemini 2.5 Pro [8] 0.924 0.567 0.840 0.924 0.833 0.812 0.487 0.427 0.727

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Table 22. (17) Task: Generate PowerPoint, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (18) Task: Travel Comparison

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.023 0.017 0.020 0.020 0.000 0.024 0.270 0.122 0.062 Qwen2.5-VL-72B [6] 0.111 0.026 0.108 0.111 0.000 0.111 0.303 0.388 0.145 Claude 4.5 Haiku [4] 0.348 0.359 0.297 0.317 0.199 0.200 0.390 0.231 0.293 InternVL 3.5 [44] 0.476 0.224 0.400 0.381 0.429 0.476 0.319 0.292 0.375 Gemini 2.5 Pro [8] 0.571 0.088 0.509 0.429 0.571 0.356 0.487 0.427 0.430 Llama-4-Scout-17B16E [26] 0.714 0.171 0.650 0.476 0.071 0.714 0.385 0.354 0.442 GPT-5 Nano [32] 0.643 0.265 0.619 0.562 0.149 0.421 0.502 0.478 0.455 Claude 4.5 Sonnet [5] 0.619 0.417 0.532 0.619 0.143 0.485 0.535 0.451 0.475 Grok 4 (0709) [46] 0.619 0.099 0.543 0.524 0.571 0.416 0.606 0.530 0.488 Gemini 2.5 Flash Lite [8] 0.714 0.388 0.601 0.619 0.643 0.386 0.316 0.259 0.491 Grok-4 Fast [47] 0.667 0.446 0.601 0.571 0.429 0.667 0.501 0.378 0.532 GPT-5 Mini [31] 0.714 0.203 0.691 0.714 0.438 0.481 0.571 0.566 0.547 GPT-5 [33] 1.000 0.119 0.938 0.810 0.857 0.573 0.616 0.594 0.688 Gemini 2.5 Flash [8] 1.000 0.239 0.897 0.667 1.000 1.000 0.496 0.449 0.718

[Figure 362]

[Figure 363]

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

Table 23. (18) Task: Travel Comparison, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(19) Task: Heatmap

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.016 0.028 0.014 0.017 0.000 0.017 0.269 0.122 0.060 Claude 4.5 Haiku [4] 0.040 0.200 0.037 0.040 0.000 0.040 0.386 0.247 0.124 Qwen2.5-VL-72B [6] 0.172 0.090 0.158 0.177 0.038 0.146 0.286 0.364 0.179 Grok-4 Fast [47] 0.100 0.219 0.088 0.100 0.038 0.072 0.501 0.378 0.187 InternVL 3.5 [44] 0.225 0.280 0.214 0.225 0.038 0.194 0.319 0.292 0.223 Gemini 2.5 Flash Lite [8] 0.325 0.848 0.311 0.325 0.088 0.170 0.316 0.259 0.330 Claude 4.5 Sonnet [5] 0.360 0.269 0.346 0.360 0.140 0.268 0.535 0.451 0.341 Llama-4-Scout-17B16E [26] 0.650 0.253 0.605 0.650 0.292 0.383 0.385 0.354 0.447 Gemini 2.5 Flash [8] 0.500 0.488 0.479 0.500 0.188 0.500 0.496 0.449 0.450 GPT-5 Mini [31] 0.619 0.127 0.586 0.524 0.429 0.524 0.571 0.566 0.493 GPT-5 Nano [32] 0.714 0.123 0.688 0.667 0.571 0.524 0.502 0.478 0.533 Gemini 2.5 Pro [8] 0.625 0.674 0.594 0.625 0.338 0.568 0.487 0.427 0.542 Grok 4 (0709) [46] 0.700 0.520 0.687 0.700 0.525 0.657 0.606 0.530 0.616 GPT-5 [33] 0.750 0.653 0.710 0.750 0.538 0.683 0.616 0.594 0.662

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Table 24. (19) Task: Heatmap, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(20) Task: Museum

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.014 0.034 0.012 0.010 0.000 0.014 0.262 0.118 0.058 Qwen2.5-VL-72B [6] 0.150 0.109 0.138 0.101 0.012 0.124 0.279 0.353 0.158 InternVL 3.5 [44] 0.200 0.217 0.198 0.163 0.000 0.200 0.319 0.292 0.199 Claude 4.5 Haiku [4] 0.203 0.714 0.183 0.153 0.028 0.120 0.379 0.224 0.250 Llama-4-Scout-17B16E [26] 0.385 0.262 0.359 0.231 0.036 0.321 0.337 0.305 0.279 Gemini 2.5 Flash Lite [8] 0.438 0.334 0.415 0.275 0.064 0.295 0.316 0.259 0.299 Grok-4 Fast [47] 0.360 0.534 0.341 0.229 0.064 0.320 0.461 0.344 0.332 Gemini 2.5 Pro [8] 0.325 0.692 0.306 0.238 0.048 0.193 0.487 0.427 0.340 Claude 4.5 Sonnet [5] 0.414 0.537 0.391 0.314 0.067 0.284 0.492 0.410 0.364 Gemini 2.5 Flash [8] 0.375 0.604 0.359 0.267 0.076 0.350 0.496 0.449 0.372 GPT-5 Nano [32] 0.475 0.229 0.465 0.475 0.150 0.425 0.502 0.478 0.400 Grok 4 (0709) [46] 0.438 0.467 0.417 0.363 0.105 0.438 0.606 0.530 0.420 GPT-5 Mini [31] 0.550 0.249 0.543 0.550 0.272 0.466 0.571 0.566 0.471 GPT-5 [33] 0.638 0.296 0.607 0.388 0.176 0.548 0.616 0.594 0.483

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Table 25. (20) Task: Museum, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (21) Task: Research Station

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.014 0.034 0.012 0.010 0.000 0.014 0.262 0.118 0.058 Qwen2.5-VL-72B [6] 0.150 0.109 0.138 0.101 0.012 0.124 0.279 0.353 0.158 InternVL 3.5 [44] 0.200 0.217 0.198 0.163 0.000 0.200 0.319 0.292 0.199 Claude 4.5 Haiku [4] 0.203 0.714 0.183 0.153 0.028 0.120 0.379 0.224 0.250 Llama-4-Scout-17B16E [26] 0.385 0.262 0.359 0.231 0.036 0.321 0.337 0.305 0.279 Gemini 2.5 Flash Lite [8] 0.438 0.334 0.415 0.275 0.064 0.295 0.316 0.259 0.299 Grok-4 Fast [47] 0.360 0.534 0.341 0.229 0.064 0.320 0.461 0.344 0.332 Gemini 2.5 Pro [8] 0.325 0.692 0.306 0.238 0.048 0.193 0.487 0.427 0.340 Claude 4.5 Sonnet [5] 0.414 0.537 0.391 0.314 0.067 0.284 0.492 0.410 0.364 Gemini 2.5 Flash [8] 0.375 0.604 0.359 0.267 0.076 0.350 0.496 0.449 0.372 GPT-5 Nano [32] 0.537 0.324 0.531 0.237 0.016 0.479 0.502 0.478 0.388 Grok 4 (0709) [46] 0.438 0.467 0.417 0.363 0.105 0.438 0.606 0.530 0.420 GPT-5 Mini [31] 0.550 0.271 0.542 0.275 0.076 0.516 0.571 0.566 0.421 GPT-5 [33] 0.638 0.296 0.607 0.388 0.176 0.548 0.616 0.594 0.483

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

Table 26. (21) Task: Research Station, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(22) Task: Hugeicons

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.014 0.036 0.010 0.013 0.000 0.016 0.262 0.118 0.059 Gemini 2.5 Flash Lite [8] 0.113 0.256 0.096 0.113 0.000 0.113 0.316 0.259 0.158 Qwen2.5-VL-72B [6] 0.149 0.115 0.114 0.137 0.040 0.135 0.279 0.353 0.165 InternVL 3.5 [44] 0.208 0.190 0.164 0.208 0.000 0.208 0.319 0.292 0.199 GPT-5 Nano [32] 0.143 0.140 0.137 0.143 0.000 0.143 0.502 0.478 0.211 Claude 4.5 Haiku [4] 0.201 0.754 0.151 0.209 0.096 0.131 0.379 0.224 0.268 GPT-5 Mini [31] 0.190 0.250 0.187 0.190 0.089 0.190 0.571 0.566 0.279 Llama-4-Scout-17B16E [26] 0.382 0.277 0.296 0.314 0.124 0.351 0.337 0.305 0.298 Grok-4 Fast [47] 0.357 0.564 0.281 0.312 0.222 0.350 0.461 0.344 0.361 Claude 4.5 Sonnet [5] 0.412 0.567 0.322 0.427 0.231 0.311 0.492 0.410 0.397 Grok 4 (0709) [46] 0.351 0.729 0.296 0.351 0.213 0.304 0.606 0.530 0.423 Gemini 2.5 Flash [8] 0.482 0.440 0.365 0.399 0.377 0.482 0.496 0.449 0.436 Gemini 2.5 Pro [8] 0.542 0.369 0.396 0.488 0.352 0.443 0.487 0.427 0.438 GPT-5 [33] 0.542 0.534 0.441 0.419 0.457 0.542 0.616 0.594 0.518

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

Table 27. (22) Task: Hugeicons, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(23) Task: Crypto Purchase Estimation

Detection Trajectory Alignment LLM Judge Average

Model

Argument Similarity

Merge Purity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.021 0.051 0.018 0.019 0.000 0.016 0.262 0.118 0.063 Qwen2.5-VL-72B [6] 0.232 0.163 0.209 0.203 0.046 0.137 0.279 0.353 0.203 Gemini 2.5 Flash Lite [8] 0.360 0.415 0.348 0.360 0.000 0.360 0.316 0.259 0.302 Claude 4.5 Haiku [4] 0.314 1.000 0.278 0.309 0.110 0.132 0.379 0.224 0.352 Llama-4-Scout-17B16E [26] 0.596 0.394 0.544 0.465 0.142 0.355 0.337 0.305 0.392 InternVL 3.5 [44] 0.524 0.654 0.480 0.524 0.251 0.185 0.319 0.292 0.404 GPT-5 Nano [32] 0.571 0.223 0.566 0.402 0.090 0.478 0.502 0.478 0.414 Grok-4 Fast [47] 0.557 0.803 0.517 0.461 0.254 0.353 0.461 0.344 0.469 Gemini 2.5 Pro [8] 0.569 0.649 0.542 0.529 0.123 0.477 0.487 0.427 0.476 Claude 4.5 Sonnet [5] 0.642 0.808 0.593 0.632 0.265 0.314 0.492 0.410 0.519 Gemini 2.5 Flash [8] 0.635 0.696 0.590 0.479 0.245 0.577 0.496 0.449 0.521 GPT-5 Mini [31] 0.706 0.449 0.699 0.706 0.288 0.292 0.571 0.566 0.535 Grok 4 (0709) [46] 0.754 0.811 0.695 0.558 0.440 0.350 0.606 0.530 0.593 GPT-5 [33] 0.770 0.716 0.730 0.690 0.337 0.486 0.616 0.594 0.617

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

- Table 28. (23) Task: Crypto Purchase Estimation, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (24) Task: Stock Growth Estimation

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.050 0.041 0.048 0.050 0.000 0.050 0.128 0.022 0.049 Gemini 2.5 Flash Lite [8] 0.388 0.275 0.384 0.388 0.071 0.263 0.184 0.165 0.265 Llama-4-Scout-17B16E [26] 0.575 0.156 0.562 0.525 0.198 0.187 0.287 0.412 0.363 Qwen2.5-VL-72B [6] 0.531 0.444 0.514 0.531 0.306 0.206 0.234 0.312 0.385 Claude 4.5 Haiku [4] 0.563 0.361 0.552 0.563 0.243 0.365 0.355 0.331 0.417 Gemini 2.5 Flash [8] 0.613 0.436 0.604 0.600 0.329 0.613 0.390 0.525 0.514 InternVL 3.5 [44] 0.713 0.479 0.704 0.688 0.391 0.604 0.355 0.419 0.544 Grok-4 Fast [47] 0.688 0.359 0.680 0.688 0.519 0.658 0.380 0.479 0.556 GPT-5 Mini [31] 0.925 0.474 0.911 0.838 0.682 0.603 0.350 0.717 0.687 GPT-5 Nano [32] 0.950 0.361 0.937 0.900 0.813 0.752 0.358 0.580 0.706 Claude 4.5 Sonnet [5] 0.950 0.748 0.934 0.925 0.859 0.446 0.372 0.502 0.717 Gemini 2.5 Pro [8] 0.838 0.729 0.811 0.813 0.662 0.807 0.507 0.576 0.718 GPT-5 [33] 0.888 0.375 0.885 0.875 0.695 0.793 0.512 0.725 0.718 Grok 4 (0709) [46] 0.875 0.585 0.871 0.863 0.671 0.832 0.505 0.675 0.735

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

- Table 29. (24) Task: Stock Growth Estimation, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(25) Task: FIPE Car Variant Lookup

Model

Detection Trajectory Alignment LLM Judge Average

Score Recall Precision

Argument Similarity

Step Coherence

Order Consistency

Merge Purity

Task Completion

Information Grounding

[Figure 460]

GLM 4.5v [15] 0.033 0.028 0.030 0.033 0.000 0.033 0.204 0.126 0.061 Gemini 2.5 Flash Lite [8] 0.033 0.167 0.033 0.033 0.000 0.033 0.258 0.089 0.081 Claude 4.5 Haiku [4] 0.200 0.184 0.191 0.200 0.024 0.133 0.355 0.244 0.191 InternVL 3.5 [44] 0.200 0.299 0.188 0.200 0.048 0.134 0.367 0.255 0.211 Llama-4-Scout-17B16E [26] 0.333 0.086 0.316 0.333 0.143 0.062 0.267 0.320 0.233 Grok-4 Fast [47] 0.333 0.506 0.312 0.283 0.048 0.239 0.442 0.386 0.319 Gemini 2.5 Flash [8] 0.367 0.357 0.357 0.283 0.148 0.244 0.496 0.372 0.328 Qwen2.5-VL-72B [6] 0.500 0.349 0.474 0.367 0.191 0.435 0.346 0.347 0.376 Claude 4.5 Sonnet [5] 0.500 0.397 0.472 0.450 0.167 0.162 0.492 0.467 0.388 GPT-5 Nano [32] 0.567 0.229 0.535 0.467 0.191 0.338 0.471 0.327 0.391 GPT-5 Mini [31] 0.467 0.305 0.445 0.467 0.167 0.307 0.513 0.550 0.403 Grok 4 (0709) [46] 0.533 0.295 0.516 0.433 0.238 0.249 0.508 0.486 0.407 Gemini 2.5 Pro [8] 0.500 0.570 0.483 0.450 0.119 0.248 0.646 0.534 0.444 GPT-5 [33] 0.700 0.216 0.640 0.450 0.071 0.398 0.600 0.500 0.447

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

- Table 30. (25) Task: FIPE Car Variant Lookup, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

(26) Task: Class Demand Estimation

Model

Detection Trajectory Alignment LLM Judge Average

Score Recall Precision

Argument Similarity

Step Coherence

Order Consistency

Merge Purity

Task Completion

Information Grounding

[Figure 474]

GLM 4.5v [15] 0.117 0.052 0.117 0.117 0.000 0.117 0.354 0.294 0.146 Claude 4.5 Haiku [4] 0.167 0.166 0.160 0.167 0.030 0.167 0.321 0.197 0.172 InternVL 3.5 [44] 0.150 0.233 0.150 0.150 0.000 0.150 0.350 0.349 0.192 Gemini 2.5 Flash Lite [8] 0.222 0.111 0.217 0.160 0.000 0.222 0.279 0.333 0.193 Claude 4.5 Sonnet [5] 0.278 0.384 0.274 0.215 0.000 0.278 0.429 0.297 0.269 Qwen2.5-VL-72B [6] 0.442 0.191 0.439 0.442 0.015 0.386 0.396 0.436 0.343 Llama-4-Scout-17B16E [26] 0.442 0.226 0.439 0.442 0.141 0.442 0.371 0.400 0.363 GPT-5 Nano [32] 0.397 0.622 0.392 0.397 0.181 0.206 0.438 0.501 0.392 Gemini 2.5 Flash [8] 0.464 0.544 0.460 0.436 0.000 0.464 0.467 0.416 0.406 Gemini 2.5 Pro [8] 0.558 0.418 0.547 0.376 0.061 0.558 0.521 0.502 0.443 Grok-4 Fast [47] 0.544 0.722 0.544 0.482 0.000 0.544 0.517 0.475 0.478 GPT-5 Mini [31] 0.540 0.325 0.526 0.518 0.366 0.447 0.567 0.568 0.482 Grok 4 (0709) [46] 0.614 0.528 0.610 0.586 0.167 0.614 0.625 0.707 0.556 GPT-5 [33] 0.856 0.456 0.799 0.681 0.606 0.387 0.653 0.598 0.630

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

- Table 31. (26) Task: Class Demand Estimation, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

###### (27) Task: NixOS Diagnostic

Detection Trajectory Alignment LLM Judge Average

Model

Merge Purity

Argument Similarity

Step Coherence

Score Recall Precision

Order Consistency

Task Completion

Information Grounding

GLM 4.5v [15] 0.000 0.000 0.000 0.000 0.000 0.000 0.225 0.042 0.033 Llama-4-Scout-17B16E [26] 0.222 0.121 0.213 0.188 0.045 0.173 0.221 0.160 0.168 InternVL 3.5 [44] 0.167 0.413 0.157 0.132 0.000 0.167 0.279 0.083 0.175 Gemini 2.5 Flash Lite [8] 0.153 0.536 0.132 0.118 0.024 0.112 0.254 0.301 0.204 Claude 4.5 Haiku [4] 0.306 0.329 0.294 0.306 0.053 0.252 0.383 0.354 0.285 GPT-5 Nano [32] 0.306 0.821 0.301 0.292 0.049 0.215 0.312 0.354 0.331 Grok-4 Fast [47] 0.319 0.571 0.316 0.271 0.069 0.319 0.354 0.496 0.339 Qwen2.5-VL-72B [6] 0.389 0.647 0.378 0.354 0.041 0.389 0.271 0.361 0.354 Claude 4.5 Sonnet [5] 0.403 0.585 0.354 0.375 0.150 0.219 0.362 0.409 0.357 Grok 4 (0709) [46] 0.292 0.964 0.274 0.292 0.163 0.292 0.444 0.416 0.392 Gemini 2.5 Flash [8] 0.333 0.804 0.295 0.333 0.155 0.333 0.400 0.548 0.400 GPT-5 Mini [31] 0.597 0.538 0.590 0.542 0.191 0.356 0.446 0.549 0.476 Gemini 2.5 Pro [8] 0.569 0.538 0.512 0.399 0.268 0.472 0.533 0.524 0.477 GPT-5 [33] 0.764 0.392 0.747 0.572 0.499 0.556 0.421 0.583 0.567

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

Table 32. (27) Task: NixOS Diagnostic, Cross-Model Evaluation of Multimodal MCP Tool-Calling Trajectories on M3-Bench.

