# arXiv:2601.21558v2[cs.CL]30Jan2026

## ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas

[Figure 1]

Beike Language and Intelligence1

#### Abstract

Large language models (LLMs) are increasingly used as tool-augmented agents for multistep decision making, yet training robust tool-using agents remains challenging. Existing methods still require manual intervention, depend on non-verifiable simulated environments, rely exclusively on either supervised fine-tuning (SFT) or reinforcement learning (RL), and struggle with stable long-horizon, multi-turn learning. To address these challenges, we introduce ASTRA, a fully automated end-to-end framework for training tool-augmented language model agents via scalable data synthesis and verifiable reinforcement learning. ASTRA integrates two complementary components. First, a pipeline that leverages the static topology of tool-call graphs synthesizes diverse, structurally grounded trajectories, instilling broad and transferable tool-use competence. Second, an environment synthesis framework that captures the rich, compositional topology of human semantic reasoning converts decomposed question–answer traces into independent, code-executable, and rule-verifiable environments, enabling deterministic multi-turn RL. Based on this method, we develop a unified training methodology that integrates SFT with online RL using trajectory-level rewards to balance task completion and interaction efficiency. Experiments on multiple agentic tool-use benchmarks demonstrate that ASTRA-trained models achieve state-of-the-art performance at comparable scales, approaching closed-source systems while preserving core reasoning ability. We release the full pipelines, environments, and trained models at https://github.com/LianjiaTech/astra.

[Figure 2]

Figure 1: Comparison of Model Performance on BFCL v3 Multi-Turn.

1Beike Language and Intelligence (BLI). For the complete list of authors, please refer to the Contribution section.

#### 1 Introduction

Large language models (LLMs) are increasingly deployed as tool-augmented agents that interact with external environments, invoke APIs, and perform multi-step decision making. By integrating reasoning with action, such agents enable applications ranging from information retrieval and data analysis to interactive dialogue systems, making tool use a core capability of modern language models.

Despite rapid progress, training robust and generalizable tool agents remains challenging. Recent work [1,

###### 2] has begun to reduce human intervention by automatically synthesizing tool-use data and environments through model-driven simulation, significantly improving scalability and coverage.

However, many of these approaches rely on LLM-simulated environments, where tool executions, state transitions, and feedback are generated through language-model rather than explicit rules or executable backends. That is, their reinforcement learning (RL) setups are not rule-verifiable. This lack of verifiability fundamentally limits stable long-horizon, multi-turn online RL, where deterministic transitions and reliable reward signals are critical. Moreover, several methods [2, 3] generate multi-turn trajectories offline but decompose them into isolated single-step training instances, which limits the agent’s ability to learn coherent long-horizon, multi-turn decision making.

In addition, many existing approaches focus on only a single training regime—either supervised fine-tuning (SFT) or RL [4, 5, 6]. SFT-only methods lack online learning signals from environment interaction, while RL-only approaches are fundamentally constrained by the capability of the original model, limiting their effectiveness when starting from weaker initial policies.

To address these challenges, we present ASTRA, a fully automated, end-to-end framework for training toolaugmented language model agents via scalable data synthesis and verifiable multi-turn online reinforcement learning. ASTRA removes human intervention throughout both data construction and validation, and is fully open-sourced.

ASTRA integrates two complementary components. For SFT, we propose a trajectory synthesis pipeline that leverages the static topology of tool-call graphs to construct diverse multi-turn tool-use trajectories grounded in real MCP servers, and automatically scores them for quality—enabling high-quality supervised fine-tuning without manual annotation. For RL, we introduce an environment synthesis framework that captures the rich, compositional topology of human semantic reasoning, converting decomposed question–answer traces into independent, code-executable, and rule-verifiable environments, thereby supporting multi-turn, long-horizon RL with deterministic rewards.

Building on these components, we develop a complete training methodology for tool agents.We use SFT to learn a stronger initial policy that is better adapted to multi-turn tool interaction and then perform online, multi-turn RL over diverse synthesized environments, incorporating irrelevant-tool mixing and an F1-style trajectory-level reward to jointly optimize task completion and interaction efficiency.

This two-stage method first broadens an agent’s tool-use competence over a static tool topology, then deepens its capability by learning within a complex semantic topology. As a result, ASTRA effectively balances generalization across tools with robustness in realistic, high-complexity scenarios.

Our contributions are summarized as follows:

- • We propose a fully automated, end-to-end data construction pipeline for tool-agent training, leveraging the static topology of tool-call graphs for multi-turn trajectory synthesis and capturing the rich, compositional topology of human semantic reasoning for QA-derived, rule-verifiable environment construction.
- • We propose a complete training methodology consisting of (i) supervised fine-tuning for a stronger initial policy that is better adapted to multi-turn tool interaction, and (ii) multi-turn online reinforcement learning over code-executable, rule-verifiable environments across multiple domains, enabling reliable and scalable agent training.
- • ASTRA-trained models achieve state-of-the-art performance among the same scale on multiple agentic tool-use benchmarks, approaching closed-source systems while preserving core reasoning ability, and we make the data synthesis pipelines and trained models publicly available to support reproducibility and future research.

#### 2 Tool-Integrated Trajectory and Verifiable Environment Synthesis

We first present the tool-chain-based trajectory synthesis pipeline for SFT, followed by the QA-based environment synthesis framework for RL.

##### 2.1 Multi-turn Trajectory Synthesis with MCP Services and Tool Emulators

|Filter<br><br>Tool Collection Func & Param<br><br>[Figure 3]<br><br>…<br><br>[Figure 4]<br><br>Qualified Tool Document<br><br>[Figure 5]<br><br>Stage 1: Tool Collection & Filtering<br><br>|
|---|

|Extract<br><br>Tool Graph Tool Chains<br><br>[Figure 6]<br><br>Stage 2: Tool-chain Construction<br><br>|
|---|

|Stage 4: Multi-turn Interaction<br><br>Trajectory Environments Query<br><br>[Figure 7]<br><br>[Figure 8]<br><br>Real<br><br>Tools<br><br>[Figure 9]<br><br>LLM Mock<br><br>Tools<br><br>[Figure 10]|
|---|

|Stage 3: Task Construction<br><br>[Figure 11]<br><br>Query<br><br>Plan<br><br>[Figure 12]<br><br>Evolve Score Augmentation Qualify<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Tool Document<br><br>[Figure 15]<br><br>|
|---|

|Stage 5: Reward System<br><br>Outcome Reward Process Reward Verified Trajectory<br><br>[Figure 16]<br><br>[Figure 17]<br><br>Relevance<br><br>Step-wise<br><br>Step Understand<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Query Understand<br><br>Global<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>Consistency<br><br>Global Plan Conciseness<br><br>Step Plan Tool Use|
|---|

Figure 2: Overview of the Tool-Chain-Based Trajectory Synthesis Pipeline.

As illustrated in Figure 2, the pipeline begins with tool document collection and normalization, followed by tool-chain synthesis and validation. We then generate tasks with enhanced realism and diversity, and finally perform trajectory rollouts using an agent framework that integrates both real and simulated tools.

##### 2.1.1 Tool Document Collection and Filtering

We begin by collecting tool documents from heterogeneous sources, including (i) open MCP registries and API platforms (e.g., Smithery [7] and RapidAPI [8]), (ii) internal tool specifications, and (iii) tool documentation extracted from public datasets. Then, the tool documentations are processed in the following two stages.

Schema normalization. We convert all tools into a unified schema compatible with the OpenAI client tool-calling protocol. This normalization yields a consistent representation across tool providers, which simplifies integration during deployment and inference.

Grouping and filtering. Tool documents from different sources are first grouped by their originating service. For clarity, we refer to each group as an MCP server in the remainder of this section. We then apply quality filters to keep only groups that can support non-trivial multi-step interactions:

- • Sufficient number of tools We discard MCP servers with fewer than three tools or functions, as they rarely support meaningful multi-turn workflows.
- • Clear and actionable descriptions We discard tool documents whose descriptions are too vague to determine intended functionality (e.g., missing descriptions or irreducible ambiguity after cleaning).
- • Convertible to the unified format We exclude tools whose schemas cannot be reliably mapped into the OpenAI-style tool-calling format used in our normalization step.

After grouping and filtering, we retained 1,585 MCP servers, comprising 19,036 tool documents spanning 41 domains.

##### 2.1.2 Tool-chain Construction

Formally, for an MCP server s, let F(s) = {f1(s),...,fm(s)

s} denote the set of tools exposed by s. Each tool f ∈ F(s) provides an input schema I(f), along with natural-language documentation (e.g., a tool description and per-argument descriptions). In this work, we restrict composition to tools within the same MCP server and do not compose tools across servers.

Tool-chains as task-conditioned execution sequences. For each server s, we use an LLM to jointly synthesize (i) a possible tool-relative task and (ii) a plausible tool-chain that could be used to solve it. A tool-chain is a length-n sequence c = (f1,...,fn) with fi ∈ F(s). The synthesis conditions on each tool’s input schema and natural-language documentation, encouraging chains whose successive calls are supported by the task specification and information implied by earlier tools.

Candidate chain construction via transition-graph walks. For each server s, we run the joint synthesis procedure to obtain a multiset of tool-chains C(s) = {c1,...,cN}, where each cℓ = (f1(ℓ),...,fn(ℓ)

) and fi(ℓ) ∈ F(s).

ℓ

We then aggregate C(s) into a directed transition graph G(s) = (V (s), E(s),w), where V (s) contains one node per tool in F(s), and an edge (fi→fj) ∈ E(s) is added if the ordered pair appears consecutively in any synthesized chain.

Finally, we sample candidate tool-chains by performing length-bounded random walks on G(s) (biased by w), and keep walks that satisfy basic validity constraints (e.g., maximum length and optional acyclicity). The resulting walks constitute our final candidate chains for server s.

Dependency verification. We apply two checks to each sampled chain. First, we verify inter-tool dependencies: required arguments for each tool can be supported by the task specification and fields implied by earlier tools, yielding a well-formed dependency structure. Second, we validate task–chain coherence by filtering out chains paired with ill-posed or nonsensical tasks. Chains failing either check are discarded.

##### 2.1.3 Task Construction, Augmentation, and Scoring

For each MCP server s with tool set F(s), we synthesize user tasks that are (i) plausible as genuine requests and (ii) solvable via tool usage provided by the server. Our pipeline combines complementary construction modes to balance realism and coverage, then applies controlled augmentation and quality-based filtering.

Task construction. We generate an initial task set T (s) by combining two complementary sources, T (s) = Tchain(s) ∪ Tserver(s) , where the two components emphasize executability and coverage respectively:

- • Chain-conditioned construction (Tchain(s) ) Given a server specification and a validated tool-chain, we condition the LLM 2 to generate tasks whose solutions naturally follow a coherent multi-step workflow consistent with validated tool-chain. This setting biases generation toward tasks that correspond to executable tool interactions.
- • Server-only construction (Tserver(s) ) Given only the server specification, we generate task candidates that can be solved using tools from F(s). This setting promotes broader topical and linguistic coverage, reducing redundancy and overly constrained scenarios.

Task augmentation with consistency constraints. Starting from T (s), we expand the distribution by applying an augmentation operator A(·), yielding an augmented set T (s) = T (s)∪A(T (s)). We instantiate A along three complementary axes:

- • Diversity augmentation Paraphrastic and content-varied rewrites (e.g., alternative wording, preference expressions, or contextual backgrounds) that preserve the same intent.
- • Complexity augmentation Introduce additional requirements (e.g., multi-constraint preferences, implicit references, or follow-up needs) while keeping the core goal unchanged. 2We use GLM-4.6-FP8 [9] as the default LLM unless otherwise specified

- • Persona-conditioned augmentation Rewrite tasks under user personas (e.g., novice vs. expert, concise vs. verbose) to cover diverse communication patterns.

To mitigate distribution drift, for each original task t ∈ T (s) and its augmented variant t˜∈ A(T (s)), we enforce language consistency by requiring lang(t˜) = lang(t), where lang(·) denotes the task’s language category. Furthermore, we constrain augmentation to preserve the semantic intent and logical requirements of the original task.

Task scoring and filtering. We score each candidate task tˆ∈ T (s) (including both original tasks in T (s) and augmented tasks in A(T (s))) along three dimensions:

- • Question quality Sqq(tˆ) clarity, completeness, and effectiveness as a realistic user query.
- • Scenario realism Ssr(tˆ) authenticity and plausibility of the described scenario.
- • Tool-use necessity Stn(tˆ) whether tool use is necessarily required and appropriately selected (i.e., the task is not trivially solvable without tools).

We retain a candidate only if it passes all thresholds:

Sqq(tˆ) ≥ θqq, Ssr(tˆ) ≥ θsr, Stn(tˆ) ≥ θtn. (1) Candidates failing Eq. (1) are discarded.

- 2.1.4 Trajectory Collection via Multi-turn Interaction We use Qwen-Agent [10] to handle the tool-calling loop. Tool composition and hybrid execution. Our tool pool consists of two categories:

- • Deployed MCP servers Tool calls are executed directly at runtime, and returned outputs are logged as environment feedback.
- • Doc-only MCP servers We employ a stateful tool-response emulation module to synthesize plausible outputs. The emulator retains session-level invocation histories and synthesized outputs to ensure coherent multi-turn interactions. To approximate real-world unreliability, we additionally inject tool failures into the emulation process, causing emulated calls to fail with a probability of 20% (e.g., due to timeouts or unreachable calls).

- 2.1.5 Reward Modeling

To enable high-quality supervised fine-tuning for tool-augmented language models, we design an automated trajectory quality assessment pipeline without human annotation. We define a trajectory as an ordered sequence

τ = {m0,m1,...,mk−1}, (2)

where k is the total number of messages in the trajectory, m0 is the system prompt, m1 denotes the user query q, and mi≥2 represents the subsequent interaction turns, including assistant responses, tool invocations, and environment responses.

Query Understanding and Planning. The initial assistant response m2 both interprets the user query and proposes an initial plan that guides subsequent tool use and interaction. We therefore evaluate these two aspects separately (while using the same trajectory-level input), so that failures due to misunderstanding can be distinguished from failures due to infeasible planning. In both cases, we exclude the system prompt from the evaluator input:

QU(τ) = Junderstand τ \ {m0} ∈ {0,0.5,1}, (3) QP(τ) = Jplan τ \ {m0} ∈ {0,0.5,1}. (4)

A score of 1 indicates correct understanding (or a complete and executable plan), 0.5 corresponds to partial understanding (or a partially feasible plan), and 0 indicates misunderstanding (or an infeasible/incorrect plan).

##### Tool Response Understanding and Planning. We define two trajectory-level metrics:

- • Tool-response Context Understanding (TCU) a trajectory-level score that measures whether each tool-call round reflects correct understanding of the immediately preceding tool response.
- • Tool-response Context-conditioned Planning (TCP) a trajectory-level score that measures whether each tool-call round’s plan/tool invocation(s) correctly incorporate that tool response.

If a turn contains multiple tool calls, we merge them into one grouped round. Let {uj}Jj=1 denote the grouped tool-call rounds in temporal order, and let tj be the index of the assistant message that issues uj. Since u1 has no preceding tool response, we score from j = 2 using the same history-plus-current-round input:

cj ≜ {m1,...,mt

j}, j = 2,...,J. (5) We compute the trajectory-level scores by averaging per-round judgments with inputs (cj,uj):

1 J − 1

TCU(τ) =

1 J − 1

TCP(τ) =

J

Junderstand(cj,uj), (6)

j=2

J

Jplan(cj,uj). (7)

j=2

Tool Call Status. Let n denote the total number of tool calls in trajectory τ. For the i-th tool call, we assign a binary indicator Si ∈ {0,1}, where Si = 1 if the call succeeds (i.e., returns a valid response) and Si = 0 otherwise. The trajectory-level tool-call status score is computed as the mean success rate:

1 n

TCS(τ) =

n

Si. (8)

i=1

Tool Conciseness. For the i-th tool call, we assign a binary indicator TCi ∈ {0,1}, where TCi = 1 if the call is necessary and non-redundant given the task and prior context, and 0 otherwise. We report the trajectory-level conciseness score as:

1 n

TC(τ) =

n

TCi. (9)

i=1

A higher score indicates efficient tool usage without redundant calls, while lower scores indicate unnecessary or inefficient invocations.

Final Answer Quality. The final answer quality evaluates whether the last assistant message mk−1 is both (i) semantically aligned with the original task specification and (ii) faithful to the trajectory content. Specifically, we measure semantic correlation between the user prompt and the final answer, and assess faithful summarization over the trajectory excluding the system prompt:

Corr(m1,mk−1) + Summ(τ \ {m0}) 2

FA(τ) =

. (10)

The above modules produce a set of seven trajectory-level scores. We aggregate them into a single scalar reward by taking the arithmetic mean across the seven dimensions.

##### 2.2 Automated Verifiable Environment Synthesis

Overall, our environment synthesis pipeline consists of four major stages: Q–A Instance Synthesis, Quality Validation, Environment Synthesis, and Sub-Environment Merging, as depicted in Figure 3.

##### 2.2.1 Q–A Instance Synthesis as Semantic Topology Extraction

We argue that an LLM’s agent capability depends on its ability to learn the latent planning and tool-use patterns underlying human cognition—selecting actions, updating task state from tool feedback, and replanning over multiple turns.

###### Domain Collection

###### Synthesize Multiple Q-A Pairs

###### Q-A Pairs Verification

###### Env Synthesis

Env Merging

[Figure 24]

Sports Movies

Economics Shopping Real estate …

[Figure 25]

Completeness Dependency Atomicity …

- Q1
- Q2

[Figure 26]

[Figure 27]

[Figure 28]

Q

Q3 A

[Figure 29]

[Figure 30]

[Figure 31]

Multiple Q-A pairs

[Figure 32]

Q-A Pairs Verification

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

|𝒊| |
|---|---|
| | |

[Figure 38]

[Figure 39]

𝑸 𝑨𝒊

[Figure 40]

[Figure 41]

[Figure 42]

completeness dependency atomicity

necessity rationality

Figure 3: Overview of the QA-Based Environment Synthesis Framework.

Unlike static path-supervised tool chains,we model multi-turn tool use as navigation over a latent semantic topology, verify only sub-tasks attainment, and optimize a composite reward that optimizes for success while penalizing interaction cost—rather than prescribing a fixed tool chain.

We formalize each instance as a main question q0 together with its main answer a0. During the solution process, the model often needs to resolve a set of intermediate sub-tasks. We explicitly represent these intermediate steps as a collection of sub-questions and sub-answers:

S = {(qi,ai)}mi=1, (11)

where each pair (qi,ai) corresponds to a necessary or helpful intermediate step for deriving a0, and m denotes the total number of such intermediate pairs. We model the derivation of the final answer as aggregating the sub-answers according to their dependency graph:

a0 = Φ {ai}mi=1,G , (12) where G denotes the dependency structure among sub-questions (e.g., an ordered chain or a DAG), and Φ(·) denotes an aggregation procedure that combines sub-answers following G. By jointly generating (q0,a0) and the intermediate steps S, we obtain an explicit and verifiable representation of the solution process.

Synthesis Mode. Each instance is synthesized conditioned on a domain-specific knowledge source K (e.g., a text corpus) and a complexity constraint H (hop budget). The synthesis process follows two modes:

- • Question-Conditional Generation If a specific main question q0 is provided, the module decomposes it into a sub-QA set S and derives a0 based on K.
- • Unconditional Generation If q0 is not provided, the module first generates a candidate question q0 grounded in K that requires approximately H reasoning hops, and subsequently produces the corresponding (q0,a0) and S.

##### 2.2.2 Quality Validation

We observe that a subset of synthesized Q–A instances contains intermediate sub-questions that do not require tool invocation. These sub-questions correspond to purely linguistic operations, such as evaluation, summarization, recommendation, advice, ranking, matching, and format transformation. Such steps cannot be grounded in executable tools and thus disrupt the continuity of the tool-use chain, preventing the construction of a fully verifiable agent environment.

Specifically, we allow sub-questions that do not require tool invocation only at leaf nodes and prohibit them at non-leaf nodes. Leaf nodes typically correspond to final linguistic aggregation or answer formulation steps, which do not require external tools and do not introduce downstream dependencies.

We first filter out samples whose intermediate Q–A pairs do not require tool usage using LLM. After filtering, we assign a quality score to each remaining Q–A pair along four complementary dimensions.

Dependency Consistency. We formalize a decomposed Q–A instance as a set of m sub-questions:

###### τ = {(q1,a1,d1),(q2,a2,d2),...,(qm,am,dm)}, (13)

where qi and ai denote the i-th sub-question and its corresponding answer, and di represents the dependency set of sub-question qi, specifying which preceding sub-questions and their corresponding answers it depends on.

To assess dependency consistency, we leverage LLM to verify each dependency set di by judging whether all listed dependencies are semantically and logically necessary for answering qi. For each sub-question, the dependency score is defined as a binary indicator:

1, if all dependencies in di are correct, 0, otherwise.

(14)

DCi =

The overall dependency consistency score for a Q–A instance is then computed as the average over all sub-questions:

m

1 m

DCi. (15)

DC(τ) =

i=1

Sub-Question Atomicity. Sub-question atomicity evaluates whether each sub-question corresponds to an indivisible unit that cannot be further decomposed. Given a decomposed Q–A instance τ = {(qi,ai,di)}mi=1, each sub-question is evaluated by LLM to determine whether it is atomic. An atomic sub-question receives a score of 1; otherwise, it receives a score of 0:

1, if qi is atomic, 0, otherwise.

(16)

SAi =

The overall atomicity score is computed as the average over all sub-questions:

m

1 m

SAi. (17)

SA(τ) =

i=1

Sequential Rationality. When synthesizing Q–A instances, we explicitly specify the expected number of reasoning hops. We observe that, in some cases, the language model introduces logically inconsistent or unnatural transitions solely to satisfy the prescribed hop count, resulting in irrational execution orders within the Q–A instance. To address this issue, we design a sequential rationality checking module to assess whether the ordering of sub-questions is logically valid.

Formally, given a decomposed Q–A instance τ = {(qi,ai,di)}mi=1, we evaluate sequential rationality based on the dependency sets di. A sub-question is considered rational if each sub-question qi is executed only after all its dependencies in di have been satisfied, and no superfluous intermediate steps are introduced. For each sub-question, the sequential rationality score is defined as a binary indicator:

1, if the execution order implied by di is rational, 0, otherwise.

(18)

SRi =

The overall sequential rationality score for a Q–A instance is computed as the average over all subquestions:

m

1 m

SRi. (19)

SR(τ) =

i=1

Task Completeness. To verify that the decomposition is logically consistent, we evaluate task completeness by checking whether the set of sub-questions is sufficient to solve the original task. Given a decomposed Q–A instance τ = {(qi,ai,di)}mi=1, we define an instance-level binary score:

1, if {(qi,ai)}mi=1 is sufficient to solve the main questionq0, 0, otherwise.

(20)

TC(τ) =

We combine the four quality dimensions–dependency consistency, sub-question atomicity, sequential rationality, and task completeness–to obtain an overall quality score for each decomposed Q–A instance. Formally, for a decomposition instance τ, we define QS(τ) as its aggregated quality score, computed by averaging the four dimension-specific scores:

1 4

QS(τ) =

DC(τ) + SA(τ) + SR(τ) + TC(τ) . (21)

##### 2.2.3 Environment Synthesis

Following quality validation, we obtain a filtered dataset Dfiltered. As illustrated in Figure 3, we synthesize an independent environment for each Q–A instance. Formally, each instance is represented as a decomposed

execution trace τ = {(qi,ai,di)}mi=1, where each triplet (qi,ai,di) denotes a sub-task node consisting of a sub-question qi, its ground-truth answer ai, and the associated dependency set di.

We skip leaf nodes and synthesize sub-environments only for the remaining ones, treating each sub-task (qi,ai,di) as an independent sub-environment.

Tool Specification Synthesis and Complexity Scaling. We first feed (qi,ai,di) into LLM to generate a tool specification document that describes the tool’s functionality, input parameters, and expected outputs. To improve expressiveness and better support diverse tool-invocation patterns, we further augment the generated specification by scaling its complexity, e.g., expanding parameter lists and enriching parameter value spaces through additional arguments or extended enumerated ranges.

Tool Implementation and Sandbox Verification. Conditioned on qi and the augmented tool document, we then generate a tool invocation statement. Subsequently, we use (qi,ai) together with the tool specification and invocation statement to synthesize a Python-based tool implementation. The generated code is executed in a sandboxed environment for validation, where a sub-environment is considered successful if the execution result contains the target answer ai. Otherwise, we restart the process from the tool invocation statement generation step and repeat for a fixed number of attempts.

After all sub-task nodes in τ have been successfully synthesized and validated, we aggregate the resulting sub-environments into a unified collection. This collection constitutes a complete, standalone, and executable environment for the original Q–A instance, enabling deterministic execution and verification across the entire decomposed trace.

##### 2.2.4 Sub-Environment Merging

To avoid action space inflation caused by functionally equivalent sub-questions, we perform intra-instance sub-environment merging to remove such redundancies. We carry out sub-environment merging in two stages.

Homogeneous Sub-Question Identification. Given a Q–A decomposed trace τ = {(qi,ai,di)}mi=1, we use LLM to identify homogeneous sub-questions that share the same functional intent but differ in

their parameter values (e.g., weather queries for different cities). Based on this classification, we group sub-questions into n homogeneous sets(n ≤ m) and obtain a merged representation:

τ′ = {s1,s2,...,sn}, (22) where each sj denotes a set of triplets corresponding to homogeneous sub-questions.

Database Expansion. For each homogeneous set sj, we randomly select one triplet as the base instance and treat its synthesized tool implementation as the initial sub-environment. We then iteratively insert the

remaining triplets in sj into the Python implementation by extending the underlying data structures, while the corresponding invocation statements are generated by an LLM. After each insertion, we execute all existing invocation statements in a sandboxed environment to verify that the tool can still return correct answers for all associated sub-questions.

After completing this procedure for all sj, we obtain a merged set of sub-environments in which functionally equivalent tools are consolidated into a single implementation while preserving correctness for all original triplets in τ.

#### 3 Training and Evaluation of Tool Agents

This section describes how we train and evaluate ASTRA. First, we summarize the key improvements in our training infrastructure. Second, we detail the two-stage training settings, covering both SFT and RL. Finally, we introduce the benchmarks and report evaluation results.

##### 3.1 Infrastructure

We summarize the key infrastructure improvements for both SFT and RL that enable efficient training and stable online optimization in ASTRA.

SFT Infrastructure. We perform SFT using the HuggingFace Transformers library3. To characterize tool-use learning dynamics, we save checkpoints at a high frequency for fine-grained tracking. Each full checkpoint in Transformers typically bundles both model parameters and the complete training state (e.g., optimizer and scheduler states), which substantially increases storage overhead under frequent saving. To mitigate the I/O overhead that can slow training and the storage overhead incurred by frequent checkpointing , we modify the checkpointing pipeline to decouple parameter snapshots from training-state serialization: we persist model weights at high frequency, while retaining training-state checkpoints only for the most recent 1–2 saves. This design preserves fine-grained observability for analysis and ablations, while keeping storage requirements practical at scale.

RL Infrastructure. Our reinforcement learning pipeline is implemented using verl4. We frame reinforcement learning as interactive tool-use over a collection of instance-specific, fully isolated simulators: each training instance is paired with an independent environment, and no state or information is shared across instances. Unlike prior approaches that roll out a single trajectory and apply updates only at individual tool-invocation nodes, we adopt an online, multi-turn agentic reinforcement learning paradigm.

[Figure 43]

Figure 4: Rollout Procedure in Reinforcement Learning.

As illustrated in Figure 4, at each interaction step the policy model generates a tool invocation statement. This statement, together with the corresponding tool implementation code, is passed into a code sandbox. The sandbox executes the tool call and returns the tool output produced under the current environment

- 3https://github.com/huggingface/transformers
- 4https://github.com/volcengine/verl

state. The returned result is then fed back to the model as an observation, enabling the agent to condition subsequent decisions on the accumulated interaction history.

For each data instance, the multi-turn interaction terminates when any of the following conditions is met:

- • The interaction reaches a predefined maximum number of turns or a maximum sequence length
- • The model stops issuing tool calls, i.e., no further tool invocation is generated

Under these termination criteria, the collected trajectory—comprising inputs, observations, tool calls, tool outputs, and rewards—is used directly for online policy optimization. This formulation allows the model to learn long-horizon decision-making strategies over tool-augmented environments, rather than optimizing isolated single-step actions.

For policy optimization, we build on the GRPO [11] objective (Equation 23). In our implementation, we omit both the KL-divergence regularizer and the entropy bonus for simplicity and empirical stability. However, under this simplified objective, if all samples within a group G receive identical rewards, the resulting advantage estimates collapse to zero, yielding no gradient signal and effectively reducing the number of learning-active samples per nominal batch. This mismatch can introduce training instability.

###### JGRPO(θ) = E q ∼ P(Q), {oi}Gi=1 ∼ πθ

(· | q)

old

|oi|

G

πθ(oi,t | q,oi,<t) πθ

1 |oi|

1 G

Aˆi,t, clip

min

(oi,t | q,oi,<t)

old

t=1

i=1

− βDKL(πθ ∥πref) .

πθ(oi,t | q,oi,<t) πθ

,1 − ϵ, 1 + ϵ A ˆi,t

(oi,t | q,oi,<t)

old

(23)

[Figure 44]

Figure 5: One-Step Adaptive Batch Filling.

To address this problem, we adopt Adaptive Batch Filling, a simple yet effective batching strategy illustrated in Figure 5. Let n denote the target batch size. Here, we call a rollout valid if it yields a non-zero learning signal–operationally, if the reward variance within its GRPO group is non-degenerate (e.g., Std(R) > δ, cf. Equation 32). We maintain a data buffer that is initially empty and always satisfies |buffer| < n. Before each rollout, we retrieve all samples currently stored in the buffer and concatenate them with newly batch samples.

If the concatenated set contains more than n valid samples, we select the first n samples to form the current training batch, while the remaining valid samples are placed back into the buffer. Rollout generation

continues until the number of valid samples is greater than or equal to n, ensuring that each optimization step is performed with a full batch of effective training data.

- 3.2 Training Settings

- 3.2.1 SFT Settings

We perform SFT on two models, Qwen3-14B and Qwen3-32B [12]. All models are trained for two epochs with a maximum sequence length of 20k tokens without packing. We use a batch size of 32 for all SFT experiments.

The learning rate is set to 5 × 10−6 for Qwen3-14B and 2 × 10−6 for Qwen3-32B. In both cases, we adopt a cosine learning rate schedule with a warmup ratio of 5% of the total training steps.

To support long-context training efficiently, we employ Context Parallelism (also referred to as Sequence Parallelism). Specifically, we use a context parallel degree of CP=2 when training Qwen3-14B, and CP=4 when training Qwen3-32B.

- 3.2.2 RL Settings

Irrelevant Tool Mixing. To improve robustness in tool selection across diverse tools, we augment each training instance with a controlled number of task-irrelevant tools, drawn from multiple semantic similarity bands. This expands the tool inventory beyond the minimal set required to solve the task, encouraging the model to discriminate truly relevant tools rather than overfitting to a fixed or overly clean tool list.

Let T denote the global tool pool obtained from all environments. We first remove duplicates by exactmatch deduplication on tool names, yielding a unique set Tuniq = {τ1,...,τM}. Each tool τ is associated with an OpenAI-standard tool documentation string d(τ), which typically includes a concise tool description and argument description that specify each tool’s expected usage. We embed d(τ) using Qwen3-Embedding-8B [13]:

eτ = f d(τ) ∈ RD, τ ∈ Tuniq. (24) Based on these embeddings, we compute a cosine similarity matrix over tools, Sij = cos eτ

, 1 ≤ i,j ≤ M. (25)

,eτ

i

j

For a training instance x, the environment exposes an instance-specific tool set T (x) ⊆ Tuniq. For each in-scope tool τi ∈ T (x), we normalize its similarity to every other tool:

Simin := min j̸=i

Sij, Simax := max

Sij,

(26)

j̸=i

Sij − Simin Simax − Simin ∈ [0,1].

Sij :=

To avoid near-duplicate tools that could destabilize training, we exclude same-domain candidates when forming similarity-based pools for τi.

Using fixed thresholds on Sij, we partition the remaining candidates into three semantic similarity bands: Bihigh(x) := {τj : Sij > 0.85}, Bimed(x) := {τj : 0.4 ≤ Sij ≤ 0.85},

(27)

Bilow(x) := {τj : Sij < 0.4}.

For each instance x and similarity band b ∈ {high,med,low}, let Pb(x) denote the instance-level candidate set of tools in band b, obtained by aggregating the per-tool candidate sets {Bib(x)} over all in-scope tools:

Pb(x) :=

Bib(x), b ∈ {high,med,low}. (28)

τi∈T (x)

We then uniformly sample up to K tools from each of Phigh(x), Pmed(x), and Plow(x); the sampled tools are used to augment the tool list presented to the model for instance x.

Reward Design. As described earlier, each data instance is associated with a set of sub-tasks to be solved, which we formalize as a job consisting of multiple question–answer pairs:

job = {(q1,a1),(q2,a2),...,(qn,an)} (29) where each pair (qi,ai) denotes a sub-question and its corresponding ground-truth answer.

Given a policy πθ, suppose that during a multi-turn interaction the agent invokes tools c times and successfully solves nˆ sub-tasks. We evaluate the resulting trajectory using an F1-style reward that jointly accounts for task completion and interaction efficiency. Specifically, we define

r =

nˆ n

nˆ c + ϵ

,p =

(30)

where r measures sub-task recall, i.e., the fraction of required sub-tasks that are successfully solved, and p measures precision with respect to tool usage, i.e., the effectiveness of each tool invocation.

The final trajectory-level reward is then computed as the harmonic mean of p and r:

2pr p + r

reward =

(31)

This reward design explicitly encourages the agent to solve as many sub-tasks as possible while minimizing redundant or unnecessary tool calls. By operating at the trajectory level, the reward provides a dense yet structured training signal for online multi-turn reinforcement learning, promoting long-horizon planning and efficient tool utilization in executable, verifiable environments.

JGRPO′ (θ)

= E(q,a)∼D, {o

i}Gi=1∼πθold(·|q) · Std R(q,{oi}) > δ

 

  1

|oi|

G

πθ(oi,t | q,oi,<t) πθ

πθ(oi,t | q,oi,<t) πθ

,1 − ϵ,1 + ϵ A ˆi,t

Aˆi,t, clip

min

G i=1 |oi|

(oi,t | q,oi,<t)

(oi,t | q,oi,<t)

old

old

t=1

i=1

(32)

Following prior work [14], instead of averaging the token loss at the sequence level, we adopt a batchlevel token loss averaging strategy. Empirically, we find that combining batch-level token loss averaging with Adaptive Batch Filling leads to stable training dynamics and consistent performance improvements throughout training. The final reinforcement learning objective is defined in Equation 32.

Training configurations. We set both the batch size and the mini-batch size to 256. This choice corresponds to a strictly online learning setting, where each optimization step is performed on collected trajectories without replaying past samples. The learning rate is fixed to 2 × 10−6 across all reinforcement learning experiments.

We adopt long-context settings to support multi-turn agent interactions. Specifically, the maximum prompt length is set to 25,600 tokens, and the maximum response length is set to 49,152 tokens. For each trajectory, we allow up to 32 turns for both the user and the assistant, enabling the agent to handle long-horizon, multi-step tool-use scenarios during training.

- 3.3 Evaluation

- 3.3.1 Benchmarks

We primarily evaluate our models on agentic multi-turn tool use. In addition, we include an evaluation on non-agentic complex reasoning to assess core reasoning competence.

- Table 1: Agentic benchmark results. Performance on BFCL-MT, τ2-Bench, and ACEBench across multiple model scales, covering closed-source, open-source, and our models.

BFCL-MT τ2-Bench ACEBench

Long Context Overall Retail Telecom Overall

Missing Func

Missing Param

Multi

Multi Turn

Model Base

Step Overall Closed-source

Claude-Opus-4.5 [21] 81.00 64.00 58.00 70.50 68.38 80.88 90.70 85.79 64.17 100.00 82.09 Gemini-3-Pro [22] 69.00 63.00 56.50 64.00 63.13 77.72 89.65 83.69 52.09 100.00 76.05 Claude-Sonnet-4.5 [23] 69.00 65.00 52.50 59.00 61.38 77.19 75.96 76.58 65.83 94.38 80.11 Claude-Haiku-4.5 [24] 63.50 42.50 52.50 56.00 53.63 69.12 37.19 53.16 64.17 88.75 76.46 GPT-4.1 [25] 47.50 32.50 32.50 43.00 38.88 74.00 34.00 54.00 66.67 95.00 80.84 Gemini-2.5-Pro [26] 28.50 35.00 30.00 27.00 30.12 71.26 37.89 54.58 97.50 40.00 68.75

###### Open-source

Kimi-K2-Instruct [27] 62.00 41.00 44.50 55.00 50.63 68.64 70.39 69.52 69.17 92.50 80.84 GLM-4.6 [9] 74.50 68.00 63.00 66.50 68.00 78.95 60.31 69.63 72.50 87.50 80.00 LoopTool-32B [2] 66.50 58.00 44.50 62.00 57.75 72.15 48.46 60.31 57.50 60.00 58.75 Qwen3-32B [12] 59.00 47.50 40.50 51.50 49.63 64.70 34.70 49.70 55.83 63.75 59.79 Qwen3-14B [12] 54.00 39.50 39.00 45.50 44.50 55.00 37.10 46.05 45.83 57.50 51.67

###### Astra (ours)

Astra-14B-thinking-v1 67.00 56.00 48.50 61.00 58.13 68.00 47.37 57.69 54.17 83.75 68.96 Astra-32B-thinking-v1 76.50 65.50 48.50 66.50 64.25 75.20 52.19 63.70 60.00 83.75 71.88

Agentic benchmarks. We use three widely adopted interactive benchmarks: BFCL-v3 Multi-Turn (abbreviated as BFCL-MT) [15], τ2-Bench [16], and ACEBench [17]. Each benchmark provides domainspecific environments equipped with tools, requiring multi-step and multi-turn interaction and integrating tool outputs into subsequent decisions. τ2-Bench and ACEBench further include a user simulator, stressing robustness under interactive user feedback. For τ2-Bench, we exclude the airline subset due to concerns about lower-quality ground-truth grading noted by prior reports [18].

Non-agentic benchmarks. To verify that our method enhances agentic behavior without sacrificing core logical reasoning, we additionally evaluate on AIME2024 [19] and AIME2025 [20], which focus on mathematical problem solving.

##### 3.3.2 Evaluation Methodology

Agentic Evaluation. All experiments are executed with vLLM [28] as the inference engine to ensure consistent serving and decoding behavior across benchmarks. To faithfully reflect agentic capability, we evaluate all tool-use benchmarks under the function-calling paradigm. Benchmark-specific settings are as follows:

- • τ2-Bench We run 4 independent trials and report passˆ1 under temperature=0.0 (greedy decoding), with GPT-5.1 [29] serving as the user simulator.
- • ACEBench Since the agent-task split contains only 50 instances, we repeat the evaluation 4 times and report the mean accuracy for stability. We use temperature=0.6 for inference, with GPT-4.1 [25] serving as the user simulator.
- • BFCL-MT We run inference with temperature=0.6.

Non-agentic Evaluation. For both AIME2024 and AIME2025, we adopt a unified decoding protocol with temperature=0.6 and top-p=0.95. To improve evaluation stability, we consider two top-k settings: k=20 and k=-1 (no top-k restriction). For each benchmark, we perform 32 independent generations under each setting and estimate the pass rate by averaging correctness over samples. The final reported score is the average of the two pass-rate estimates.

- Table 2: Agentic Benchmark Results Across Training Stages. Performance on BFCL-MT, τ2-Bench, and ACEBench for the Original, SFT, and RL models.

BFCL-MT τ2-Bench ACEBench

Model Base

Missing Func

Missing Param

Long Context Overall Retail Telecom Overall

Multi Turn

Multi Step Overall

14B Qwen3-14B 54.00 39.50 39.00 45.50 44.50 55.00 34.10 44.55 45.83 57.50 51.67

Ours-14BSFT 67.50 25.50 48.00 53.00 48.50+4.00 63.80 37.10 50.45+5.90 51.67 83.75 67.71+16.04 Ours-14BRL 67.00 56.00 48.50 61.00 58.13+13.63 68.00 47.40 57.70+13.15 54.17 83.75 68.96+17.29

32B Qwen3-32B 56.00 52.50 40.00 43.00 47.88 64.70 34.70 49.70 55.83 63.75 59.79 Ours-32BSFT 67.00 40.00 46.00 55.50 52.13+4.25 66.70 37.90 52.30+2.60 56.67 78.75 67.71+7.92 Ours-32BRL 76.50 65.50 48.50 66.50 64.25+16.38 75.20 52.20 63.70+14.00 60.00 83.75 71.88+12.09

- Table 3: Non-agentic benchmark results. We report AIME2024 and AIME2025 under two decoding settings for both 14B and 32B models.

(topp=0.95, topk=20, temperature=0.6) (topp=0.95, topk=-1, temperature=0.6) Model AIME2024 AIME2025 Avg AIME2024 AIME2025 Avg

Qwen3-14B 80.00 66.90 73.45 78.50 66.70 72.60 ASTRA-14B-Thinking-v1 80.10 66.70 73.40 78.80 66.40 72.60

Qwen3-32B 83.00 66.80 74.90 82.40 65.90 74.15 ASTRA-32B-Thinking-v1 81.40 68.30 74.85 81.20 69.10 75.15

##### 3.3.3 Results

As shown in Table 1, we report performance on BFCL-MT, τ2-Bench, and ACEBench across multiple model scales, including closed-source models, open-source models, and our models. Our models achieve state-of-the-art results at matched parameter scales and are competitive with higher-parameter open-source and closed-source models on multiple metrics. Table 2 further indicates that both the SFT and RL stages deliver consistent gains, with the RL stage contributing the largest improvement.

In addition, as shown in Table 3, we evaluate AIME2024 and AIME2025 under two decoding settings for both 14B and 32B models. Although our method primarily optimizes agentic tool-use, it shows negligible degradation on non-agentic complex reasoning, indicating robust behavior.

#### 4 Discussion

In this section, we conduct a comprehensive analysis to better understand the factors that shape effective tool-use behavior, focusing on three complementary perspectives: irrelevant-tool mixing strategies, reward design for RL training stability, and stage-wise analysis of agent behavior and performance across the original, SFT, and RL models.

##### 4.1 Improving Tool-Use Discrimination via Irrelevant Tool Mixing

As described in Section 3.2.2, ASTRA shapes tool-use behavior by exposing the agent to irrelevant tools during reinforcement learning. This design encourages the policy to learn not only correct tool usage but also effective tool discrimination.

To study how this interacts with reward optimization, we perform two ablations under identical RL settings, varying only the tool set composition: (i) No Irrelevant Tools, where only ground-truth tools are provided, and (ii) Random Irrelevant Tools, where 5–9 tools are randomly sampled from other domains.

Results are shown in Figure 6. Removing irrelevant tools yields the worst performance, as the policy overfits to a narrow tool-selection pattern and lacks pressure to optimize the precision component of the

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |Ou Ab Ab<br><br>|rs-14 latio latio|B<br><br>n (Ra n (No|ndom tool|too supp|l sup lem|plem entat|enta ion)|tion)| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.55

0.50

Score

0.45

0.40

0.35

0.30

5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100105110115120125130135140

Steps

Figure 6: Ablation on Irrelevant-Tool Mixing Settings.

reward. Randomly mixing irrelevant tools improves performance by introducing basic discrimination signals, but remains inferior to the full ASTRA setup.

These ablations highlight that irrelevant-tool mixing is a necessary signal for learning negative tool judgment. When no irrelevant tools are provided, the policy is never required to reject a plausible-butwrong option, and therefore fails to acquire the capability to identify tools as irrelevant under realistic toolset exposure.

0.6

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |Ou Ab Ab|rs-14 latio latio|B (F n (Re n (Pr|1) call) ecisio|n)| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.5

0.4

Score

0.3

0.2

0.1

5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100105110115120125130135140

Steps

Figure 7: Ablation on Reward Configurations.

Moreover, the effectiveness of irrelevant-tool mixing depends on the similarity structure among tools. In practice, a tool’s similarity to other tools is not uniformly distributed across the dataset; some irrelevant tools are near-miss distractors while others are trivially dissimilar. The results suggest that a more balanced coverage over tool-similarity ranges—i.e., mixing irrelevant tools that span multiple similarity bands rather

than concentrating on a single region—can better support reward optimization: it provides consistent discrimination pressure and enables the policy to learn both which tools to call and which tools to ignore.

##### 4.2 How Reward Design Shapes Tool-Use Behavior

As described earlier, we use an F1-style trajectory reward to jointly encourage task completion and interaction efficiency. For a job with n required sub-tasks, if the agent solves nˆ sub-tasks with c tool

invocations, we define recall r = nnˆ and precision p = c+nˆϵ, and compute the reward as F1 = p2+prr.

Ours-14B (F1)

Ablation (Recall)

60

Ablation (Precision)

50

AverageNumofTurns

40

30

20

10

123456789101112131415161718192021222324252627282930313233343536373839404142434445

Steps

Figure 8: Dialogue-Turn Comparison Under Different Reward Configurations.

To study how reward shaping affects tool-use behavior, we run two ablations under the same RL configuration (identical initialization, data mixture, rollout settings, and GRPO optimization), changing only the trajectory reward: recall-only (r) and precision-only (p). Results are shown in Figure 7.

We further examine training dynamics by tracking the number of interaction turns per trajectory across RL updates. As shown in Figure 8, the reward design induces markedly different turn-length distributions over training.

Optimizing recall-only quickly causes turns to explode, as the policy prolongs interaction and issues increasingly many tool calls, inflating sequence lengths and destabilizing online optimization until training collapses. In contrast, optimizing precision-only drives turns to drop sharply by discouraging tool calls. This pushes the policy toward overly conservative, short-horizon behavior that is brittle in multi-step settings and also collapses later in training.

Finally, the F1 reward yields well-behaved turn distributions and stable training. By jointly optimizing recall and precision, it preserves an incentive for exploration (solving more sub-tasks via tool use) while simultaneously penalizing incorrect or irrelevant tool invocations through the precision term. This coupled objective provides a more balanced trade-off between exploration and exploitation, preventing both runaway tool overuse and overly conservative under-calling, and thereby supporting robust multi-step performance.

##### 4.3 Analysis at Different Training Stages

We compare behaviors and performance across three stages: the original model, the model after SFT, and the final model after RL.

Behaviors Analysis. We analyze two high-level statistics: interaction steps and average output length. The results are shown in Table 4.

Table 4: Average Steps and Token Usage per Subtask on BFCL v3 MT.

Average Tokens (per sub-job) Qwen3-14B 2.5 379.6 1096.7 ASTRA-14B-Thinking-v1SFT 3.1 171.6 686.4 ASTRA-14B-Thinking-v1RL 3.2 237.9 898.6 Qwen3-32B 3.7 361.7 1145.3 ASTRA-32B-Thinking-v1SFT 3.1 192.0 672.1 ASTRA-32B-Thinking-v1RL 3.1 317.8 1130.1

Average steps (per sub-job)

Average Tokens (per step)

Models

- • Interaction Steps The average number of interaction steps remains largely unchanged across all stages. Neither SFT nor RL systematically alters dialogue depth, indicating that performance differences are not driven by trivial changes in interaction length.
- • Output Length Output length shows a clear stage-wise pattern. The original model generates the longest trajectories, while SFT produces the shortest outputs by compressing reasoning into concise, demonstration-style patterns. The RL-trained model converges to an intermediate length, longer than SFT but shorter than the original model.

Performance Analysis. Table 2 summarizes agentic benchmark performance across training stages. Overall, both SFT and RL improve over the original model, with RL consistently achieving the best results.

- • SFT improves multi-turn tool-use adaptation By using tool-chain-based pipeline, SFT provides a strong cold start by teaching structured tool invocation, multi-turn state tracking, and adherence to interaction conventions. This initialization consistently outperforms the original model.
- • RL further improves performance via broader exploration Compared to SFT, RL delivers substantial additional gains. We attribute this improvement to our QA-based RL method–instead of optimizing toward a single golden tool-sequence answer, we anchor supervision on sub-QA pairs. This method encourages the model to search over a larger space of feasible trajectories that achieve the same intermediate subgoals and final-answer constraints. Consequently, the policy is optimized over a more expansive search space with semantic and topological structure for multi-turn tool usage, enabling trajectory-level credit assignment and recovery from suboptimal decisions in a more constrained topological search space.

#### 5 Related Work

##### 5.1 Tool-Use Trajectory Synthesis

Recent progress in tool-augmented language model agent has driven growing interest in systematically constructing large-scale, high-quality trajectories. A major line of work [30, 6] constructs large tool-centric corpora over extensive tool inventories to improve data diversity and scale. Subsequent efforts extend this paradigm to multi-turn settings by explicitly modeling tool-call sequences with executability constraints and verification mechanisms [31, 32] Notably, APIGen-MT [33] formalizes multi-turn trajectory synthesis through a two-phase framework that decouples task blueprint generation from simulated human–agent interactions, enabling controllable trajectory construction and demonstrating strong performance on agentic benchmarks. Complementary approaches further broaden coverage by harvesting and standardizing tools from large-scale tool ecosystems or real-world tool servers [34].

Beyond explicit tool inventories, another emerging line of work reduces reliance on predefined tools by converting implicit procedural knowledge in open-domain text into trainable multi-turn tool-use trajectories. GEM [35] adopts a Text-to-Trajectory paradigm that extracts latent workflows from raw text and grounds them into executable trajectories, offering a scalable data source and improved cross-domain generalization. LoopTool [2] closes the data–training loop by iteratively adapting the data distribution to model weaknesses, thereby improving robustness and long-horizon tool-use performance to overcome the limitations of one-shot offline synthesis.

##### 5.2 Environment Scaling

As training and evaluating increasingly capable agents require exposure to diverse, executable, and stateful environments, a growing body of work focuses on scaling tool-interactive environments. However, most existing approaches rely on manual environment design, such as interactive benchmarks and controlled task suites [36, 37, 38], which inherently constrains domain diversity and scalability due to the high cost of human design and maintenance.

To address these limitations, recent work has shifted toward programmatic environment construction to support scalable training. EnvScaler [39] addresses this challenge by automatically synthesizing toolinteractive environments, constructing executable environment skeletons and generating diverse scenarios with rule-based validators, thereby substantially expanding environment scale while preserving verifiability for both SFT and RL. AutoForge [40] further improves efficiency by synthesizing environments directly from tool documentation and introducing environment-level optimization to mitigate noisy simulated users.

Beyond direct environment synthesis, AgentScaler [41] advances an alternative abstraction by modeling environments as read–write databases, enabling the generation of verifiable agent experiences under a unified interface together with a two-stage learning strategy. Complementary to environment construction, CuES [42] tackles task scarcity through curiosity-driven, environment-grounded task synthesis, extracting executable tasks from exploration trajectories without predefined goals. Finally, GenEnv [43] frames training as a co-evolutionary process, where the environment generator serves as a curriculum policy that dynamically aligns task difficulty with the agent’s evolving capability region.

#### 6 Conclusion and Future Work

In this work, we presented ASTRA, a fully automated, end-to-end framework for training tool-augmented language model agents via scalable data synthesis and rule-verifiable multi-turn reinforcement learning. ASTRA unifies multi-turn trajectory synthesis that leverages the static topology of tool-call graphs for supervised fine-tuning with environment synthesis that captures the rich, compositional topology of human semantic reasoning, producing independent, executable, and rule-verifiable Python environments for online RL.

Across multiple agentic tool-use benchmarks, ASTRA-trained models achieve strong performance at comparable scales while preserving general reasoning ability. We open-source the data synthesis pipelines, synthesized environments, and trained models to support reproducibility and future research. We anticipate that ASTRA can help alleviate practical deployment bottlenecks by reducing reliance on static, scenariospecific labeled data. Instead, it may be possible to synthesize multiple executable environments per scenario and train agents via iterative interaction to improve robustness in downstream applications.

Multi-turn, user-interactive agents are increasingly important in real-world deployments. In future work, we will extend ASTRA to incorporate multi-turn user interaction during training and evaluation, improving robustness to evolving intents and interactive feedback while maintaining verifiability and reproducibility. More broadly, scalable deployment also requires cost-aware automation. Since executable environment synthesis can be expensive, we will explore refining and verifying the QA-derived topology prior to code generation, using the validated topology as prior information and instantiating code environments only for high-confidence specifications.

#### 7 Contribution

##### Core Contributors

Xiaoyu Tian Haotian Wang Shuaiting Chen Hao Zhou Kaichi Yu Yudian Zhang Jade Ouyang Junxi Yin Jiong Chen

##### Contributors

Baoyan Guo Lei Zhang Junjie Tao Yuansheng Song Ming Cui Chengwei Liu

#### References

- [1] Yuwen Li, Wei Zhang, Zelong Huang, Mason Yang, Jiajun Wu, Shawn Guo, Huahao Hu, Lingyi Sun, Jian Yang, Mingjie Tang, and Byran Dai. Close the loop: Synthesizing infinite tool-use data via multi-agent role-playing, 2025.
- [2] Kangning Zhang, Wenxiang Jiao, Kounianhua Du, Yuan Lu, Weiwen Liu, Weinan Zhang, and Yong Yu. Looptool: Closing the data-training loop for robust llm tool calls, 2025.
- [3] Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Tao Gui, Xuanjing Huang, and Jiecao Chen. Feedback-driven tool-use improvements in large language models via automated build environments, 2025.
- [4] Yirong Zeng, Xiao Ding, Yutai Hou, Yuxian Wang, Li Du, Juyi Dai, Qiuyang Ding, Duyu Tang, Dandan Tu, Weiwen Liu, Bing Qin, and Ting Liu. Tool zero: Training tool-augmented llms via pure rl from scratch, 2025.
- [5] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs, 2025.
- [6] Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. Toolace: Winning the points of llm function calling, 2025.
- [7] Smithery Team. Smithery: Connect agents to mcps in minutes. https://smithery.ai/, 2026. Accessed: 2026-01-27.
- [8] RapidAPI. Rapidapi: The world’s largest api hub. https://rapidapi.com/, 2026. Accessed: 2026-01-27.
- [9] 5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025.
- [10] Qwen Team. Qwen-agent: An agent framework based on qwen language model. https://github. com/QwenLM/Qwen-Agent, 2024. Accessed: 2026-01-27.
- [11] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
- [12] Qwen Team. Qwen3 technical report, 2025.
- [13] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.
- [14] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
- [15] Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.
- [16] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment, 2025.

- [17] Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Yuefeng Huang, et al. Acebench: Who wins the match point in tool learning? arXiv preprint arXiv:2501.12851, 2025.
- [18] OpenAI. Introducing gpt-5.2, 2025.
- [19] MAA. American invitational mathematics examination - aime. https://maa.org/ math-competitions/american-invitational-mathematics-examination-aime, feb

2024. Accessed in February 2024, from American Invitational Mathematics Examination - AIME 2024.

- [20] Yixin Ye, Yang Xiao, Tiantian Mi, and Pengfei Liu. Aime-preview: A rigorous and immediate evaluation framework for advanced mathematical reasoning. https://github.com/GAIR-NLP/ AIME-Preview, 2025. GitHub repository.
- [21] Anthropic. Introducing claude opus 4.5. https://www.anthropic.com/news/ claude-opus-4-5, 2025.
- [22] Google. A new era of intelligence with gemini 3. https://blog.google/ products-and-platforms/products/gemini/gemini-3/, 2025.
- [23] Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/ claude-sonnet-4-5, 2025.
- [24] Anthropic. Introducing claude haiku 4.5. https://www.anthropic.com/news/ claude-haiku-4-5, 2025.
- [25] OpenAI. Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/, 2025.
- [26] Google. Gemini 2.5: Our most intelligent ai model. https://blog. google/innovation-and-ai/models-and-research/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025.
- [27] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence, 2025.
- [28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [29] OpenAI. Gpt-5.1: A smarter, more conversational chatgpt. https://openai.com/index/ gpt-5-1/, 2025.
- [30] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis, 2023.
- [31] Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T. Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, and Tomas Pfister. Magnet: Multi-turn tool-use data synthesis and distillation via graph translation, 2025.
- [32] Xingshan Zeng, Weiwen Liu, Lingzhi Wang, Liangyou Li, Fei Mi, Yasheng Wang, Lifeng Shang, Xin Jiang, and Qun Liu. Toolace-mt: Non-autoregressive generation for agentic multi-turn interaction, 2025.
- [33] Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Weiran Yao, Huan Wang, Silvio Savarese, and Caiming Xiong. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agent-human interplay, 2025.
- [34] Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. Toucan: Synthesizing 1.5m tool-agentic data from real-world mcp environments, 2025.

- [35] Zhihao Xu, Rumei Li, Jiahuan Li, Rongxiang Weng, Jingang Wang, Xunliang Cai, and Xiting Wang. Unlocking implicit experience: Synthesizing tool-use trajectories from text, 2026.
- [36] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024.
- [37] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment, 2025.
- [38] Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, and Yunke Zhao. Vitabench: Benchmarking llm agents with versatile interactive tasks in real-world applications, 2025.
- [39] Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Zhicheng Dou, and Ji-Rong Wen. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis, 2026.
- [40] Shihao Cai, Runnan Fang, Jialong Wu, Baixuan Li, Xinyu Wang, Yong Jiang, Liangcai Su, Liwen Zhang, Wenbiao Yin, Zhen Zhang, Fuli Feng, Pengjun Xie, and Xiaobin Wang. Autoforge: Automated environment synthesis for agentic reinforcement learning, 2025.
- [41] Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, Shibin Wu, Zhengwei Tao, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Towards general agentic intelligence via environment scaling, 2025.
- [42] Shinji Mai, Yunpeng Zhai, Ziqian Chen, Cheng Chen, Anni Zou, Shuchang Tao, Zhaoyang Liu, and Bolin Ding. Cues: A curiosity-driven and environment-grounded synthesis framework for agentic rl, 2025.
- [43] Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, and Mengdi Wang. Genenv: Difficulty-aligned co-evolution between llm agents and environment simulators, 2025.

#### A Appendix

- A.1 Data Analysis

- A.1.1 SFT Data

We construct a high-quality SFT dataset comprising 54,885 multi-turn conversation samples with a total of 580,983 messages. Each sample contains an average of 10.59 messages, with the detailed distribution shown in Figure 9.

All samples involve tool calling, with an average of 4.42 tool invocations per conversation; 72.2% of the samples contain 1–5 tool calls, as illustrated in Figure 10. The role distribution is dominated by tool responses (41.8%) and assistant utterances (39.3%), reflecting the tool-intensive nature of the dataset. These tool calls span 6,765 unique tool functions, covering reasoning, computation, search capabilities, and more.

[Figure 45]

Figure 9: Distribution of Messages per Sample in SFT.

- A.1.2 RL Data

Our RL dataset comprises 6,596 samples spanning diverse domains. As illustrated in Figure 11, the distribution is led by Real Estate (15.6%), E-commerce (10.6%), Healthcare (8.1%), covering varied application scenarios. The collection is bilingual, with English samples accounting for 71.2% (4,694) and Chinese samples for 28.8% (1,902).

In terms of task complexity, samples contain an average of 4.37 reasoning hops (median: 4.0; range: 1–20). The distribution of scenario types, shown in Figure 12, reveals that Parallel Multi-Hop scenarios are the most prevalent (47.8%), followed by Multi-Hop (34.8%), Parallel Single-Hop (10.6%), and Single-Hop (6.8%), underscoring a focus on complex, multi-step reasoning. At the sub-question level, 91.3% of the 28,794 total sub-questions require external tool calls. As detailed in Figure 13, most samples necessitate 2–5 tool calls, with an average of 3.98 calls per sample.

Furthermore, the reasoning structure indicates that 44.2% of steps can be parallelized, while 55.8% require serial execution due to data dependencies.

[Figure 46]

Figure 10: Distribution of the Number of Tool Calls per sample in SFT.

[Figure 47]

Figure 11: Top 20 Domain Distribution in the RL Dataset.

[Figure 48]

Figure 12: Distribution of Scenario Types in RL.

[Figure 49]

Figure 13: Distribution of Tool Calls per sample in RL.

##### A.2 Case study

The Original Model Is Incorrect; the SFT and RL Models Are Correct User Question

I’m interested in keeping up with Lulu Qian and her impressive achievements in the tech world, even if it’s all a bit over my head! One company I’ve heard about lately that caught my attention because of its innovative tech solutions is “Quasar Ltd.”. I’ve become quite intrigued by what they offer. Could you help me out by finding the current price and trading details for “Quasar Ltd.”. so I can better understand their market position?

### Original Model Global Think ✗

Okay, let’s see. The user wants information about “Quasar Ltd.”, specifically the current price and trading details to understand their market position. First, I need to figure out the stock symbol for “Quasar Ltd.”. The user mentioned the company name, but not the symbol. So, I should use the get_symbol_by_name function to get the stock symbol.

Once I have the symbol, the next step is to retrieve the current stock information using get_stock_info with the symbol obtained. That should give details like price, percent change, volume, and moving averages. Also, the user might want to add this stock to their watchlist for tracking, so I should check if they want to use add_to_watchlist afterwards. But the user hasn’t explicitly asked to add it yet, so maybe after providing the info, I can suggest adding it.

Wait, the user’s main request is to find the current price and trading details. So first priority is getting the symbol, then the stock info. Let me make sure to call those two functions. If the symbol isn’t found, I’ll need to inform the user. But assuming it’s a real company, proceed. Let’s start with getting the symbol.

### SFT Model Global Think ✓

The user is asking for information about “Quasar Ltd.” stock. They want to know the current price and trading details to understand the company’s market position.

To help with this, I need to:

- 1. First get the stock symbol for “Quasar Ltd.” using the get_symbol_by_name function
- 2. Then get the stock details using the get_stock_info function with that symbol Let me start by getting the symbol for “Quasar Ltd.”.

### RL Model Global Think ✓

The user wants to find the current price and trading details for “Quasar Ltd.”. Looking at the available functions, I need to:

- 1. First, get the stock symbol for “Quasar Ltd.” using the get_symbol_by_name function
- 2. Then, get the stock details using the get_stock_info function with that symbol Let me check the parameters:

- • get_symbol_by_name: name (required) – “Quasar Ltd.”
- • get_stock_info: symbol (required) – the symbol returned from the first function call

So I’ll first call get_symbol_by_name with “Quasar Ltd.” and then use that symbol to call get_stock_info.

The Original and SFT Models Is Incorrect; the RL Model Are Correct User Question

I’m gearing up for a quick business getaway and need my ride all set. Would you be able to verify if my tire pressure is in check? If it falls under 37.5 PSI, perhaps we could swing by the nearest tire shop?

### Original Model Global Think ✗

Okay, the user is preparing for a business trip and wants to ensure their car is ready. They specifically mentioned checking tire pressure and visiting a tire shop if any tire is below 37.5 PSI.

First, I need to check the tire pressure using the check_tire_pressure function. This function returns the pressure for each tire and whether it’s healthy.

Once I get the data, I’ll review each tire’s PSI. If any are below 37.5, the healthy_tire_pressure field might indicate false. If that’s the case, I should use find_nearest_tire_shop to locate a shop for them.

Wait, even if healthy is false, maybe the user wants to know the exact pressures. So I should list each tire’s pressure. But the function’s response includes healthy_tire_pressure as a boolean. If it’s false, then proceed to find the shop.

So the steps are: call check_tire_pressure, evaluate the result, and if needed, call find_nearest_tire_shop. Let’s start with the first function.

### SFT Model Global Think ✗

The user wants me to:

- 1. Check the tire pressure of their vehicle
- 2. If any tire pressure is under 37.5 PSI, find the nearest tire shop

Let me start by checking the tire pressure first, then based on the results, I’ll decide whether to find a tire shop.

I’ll use the check_tire_pressure function to get the current tire pressure readings.

### RL Model Global Think ✓

The user wants me to check their tire pressure and if it’s under 37.5 PSI, find the nearest tire shop. Let me break this down:

- 1. First, I need to check the tire pressure using the check_tire_pressure function.
- 2. Then, I need to evaluate the results. If any tire is under 37.5 PSI, I should use the find_nearest_tire_shop function.

Let me start with checking the tire pressure. Looking at the check_tire_pressure function:

- • It has no required parameters
- • It returns a dict with tire pressure info including frontLeftTirePressure, frontRightTirePressure, rearLeftTirePressure, rearRightTirePressure, and healthyTirePpressure

So I’ll call that first, then based on the results, decide if I need to find a tire shop. Let me make the function call for checking tire pressure.

A.3 Example Prompts Here we list four prompts used in ASTRA.

- Example Prompt 1: Query Generating from Tool Chain

You are a helpful AI assistant. You have the following tools: {TOOLS} These tools are designed for the following scenario: {SCENERY}

I will provide you with a tool invocation chain (i.e., the sequence and content of tool calls). Your task is:

- 1. Determine whether this tool invocation chain is meaningful and valid (i.e., whether it can accomplish a reasonable task).
- 2. If it is meaningful, extract the original user intent or query corresponding to this chain.
- 3. Output the result strictly in the specified JSON format:

Example output: {"valid": true, "query": "Check the weather in Beijing."}

Field description:

- • valid: Boolean value indicating whether the chain is valid (true = valid, false = invalid).
- • query: String representing the user’s instruction or intent; leave it as an empty string if invalid.

Please follow the above format strictly and do not include any extra explanations or text. Input: <ToolInvocationChain>{chain}</ToolInvocationChain>

Output:

- Example Prompt 2: Tool Call Conciseness

##### Task Overview

You will act as an evaluator and read a complete trajectory of an Agent interacting with tools. Your goal is to assess the cost efficiency of each tool call, focusing on whether the call is concise, whether it could be optimized to reduce the number of calls or cost, and to identify redundant, repetitive, or inefficient calling paths.

##### Input Format

- 1. The trajectory includes: the user’s original request, the Agent’s reasoning, each tool call (with parameters), tool responses, and the Agent’s final reply.
- 2. The trajectory is provided as a JSON object with the following structure:

{ "tools": [...], "messages": [

{ "role": "system" | "user" | "assistant" | "tool", "content": "...", "tool_calls": [...] }

] } The complete "trajectory" will be provided at the end of the prompt via the "trajectory" placeholder.

Core Metric Each tool call is scored as 0 or 1:

- • 1.0 (Necessary): Without this call, the user’s goal cannot be achieved or would be very difficult to achieve; the call follows the context, parameters match the requirement, and it obtains new information or advances the process at minimal cost.
- • 0.0 (Redundant): The call is disconnected from the goal (including task substitution), repeatedly retrieves already available information, has incorrect parameters (including format errors), involves parameter hallucination (the model fabricates parameters not provided by the user or uses default/example parameters), or represents invalid retries.

##### General Rules

- 1. Necessity Principle: If the call is unrelated or the goal is already achieved, score 0.
- 2. Information Gain Principle: If the call does not provide critical missing info, score 0.
- 3. Reasonable Orchestration Principle:

- • Batch Support: Multiple serial calls when batch is supported → redundant.
- • Avoid Fragmentation: Don’t split a single atomic task unnecessarily.

- 4. Tool Trust Principle: Calls to undefined tools → score 0; retries due to external failures are necessary; repeated invalid verification → redundant.

##### Special Judgments (Common Failure Cases)

- 1. Parameter Responsibility: Hallucinated or malformed parameters → score 0.
- 2. Repeated Reads: Re-fetching already obtained info → score 0 unless justified (e.g., latest snapshot).
- 3. Invalid Retries: Retries allowed only for transient errors (timeout, network); retries for clear errors → score 0.
- 4. Stochastic Interfaces: Multiple calls allowed if randomness is required, with reasoning.
- 5. Task Consistency: Substituting or simplifying the task that deviates from the goal → score 0.

##### Evaluation Procedure

- 1. Identify user goal, tools, and constraints.
- 2. Parameter Source Check: Verify origin of all parameters, detect hallucination.
- 3. Step-by-step reconstruction: record context, existing info, and call intent.
- 4. Per-call scoring: evaluate each call according to criteria, explain reasoning.
- 5. Global summary: In the ‘thought‘ field, summarize strategy, key decisions, issues, and improvements.

Output Format Strictly output **valid JSON only**. Example structure:

{ "tool_call_num": <total number of tool calls>, "tool_evaluations": [

{ "tool_index": 1, "tool_name": "<tool name>", "reasoning": "<concise justification: analyze necessity, parameter correctness,

→ and conciseness in context>", "score": 0.0 | 1.0 }

], "thought": "<overall analysis>", "tool_score_list": [score_1, score_2, ...] }

##### Additional Notes

- • Tool definitions must be strictly followed; do not infer unavailable capabilities.
- • Parameter source checking is mandatory before evaluating any call.
- • Maintain consistent judgment standards across similar scenarios.
- • Strictly adhere to JSON formatting; escape quotes inside string values.
- • Respond in Chinese.

##### Complete trajectory: {trajectory}

- Example Prompt 3: Semantic Topology Extraction for Q–A Instance Synthesis

Role You are an expert Data Architect specializing in building Tool-Use evaluation datasets for Large Language Models (LLMs). Your task is to generate high-quality Question-Answer (QA) pairs and their detailed reasoning decomposition paths based on the given scenarios and constraints.

Input Data Please generate data based on the following input variables:

- 1. Domain: {{Domain}}
- 2. Knowledge Corpus: {{Knowledge_Corpus}}
- 3. Min_Num_Hops: {{min_num_hops}}
- 4. Max_Num_Hops: {{max_num_hops}}
- 5. Count: {{num_samples}}

Scenario Definitions You need to construct {{num_samples}} complex User Queries and decompose them into sub-questions. Based on the input constraints, the data must adhere to one of the following logical structures:

- 1. Single-Hop: Contains exactly one sub-question.
- 2. Parallel Single-Hop: The user query contains multiple independent sub-tasks that are mutually exclusive and can be executed in parallel.
- 3. Multi-Hop: Contains serialized dependency relationships (sub-question qi+1 depends on the answer ai).
- 4. Parallel Multi-Hop: A hybrid structure. It contains both independent parts that can be parallelized and parts that depend on previous results.

##### Constraints & Guidelines

- 1. Quantity & Diversity: Strictly generate {{num_samples}} data entries. Each entry must involve different entities, attributes, or specific scenarios. Avoid repetitive templates.
- 2. Realism & Specificity: Generated sub-questions and sub-answers must be specific (e.g., specific dates, amounts, real entity names) and not vague.
- 3. Task Parallelism: Multi-step tasks must automatically identify and split mutually independent subquestions (is_parallel = true).
- 4. Tool-Oriented: Each sub_question must be solvable by a single atomic API tool. sub_answer contains only the final result.
- 5. Clear Dependencies: Explicitly indicate in JSON dependency field which step’s output the current step depends on (null for parallel steps). hop_level indicates layers.
- 6. Corpus Compliance: Answers must be derived from the corpus; otherwise, synthesize specific reasonable data.
- 7. Hop Range: hop_level must be between {{min_num_hops}} and {{max_num_hops}}. Decide Num_Hops based on Domain and Knowledge_Corpus.

Output Format Output only a JSON object list containing {{num_samples}} objects. The format is:

[

{

"scenario_type": "Single-Hop" | "Multi-Hop" | "Parallel Single-Hop" | "Parallel

→ Multi-Hop", "main_question": "...", "final_answer": "...", "decomposition_trace": [

{

"_uuid": 1, "hop_level": 1, "sub_question": "...", "is_parallel": true, "dependency": null, "sub_answer": "..."

},

... ]

},

... ]

- Example Prompt 4: Generating Function Code for the Synthesis Tool

Implement a function in accordance with a provided tool document, a set of question–answer pairs, and a given call statement. The implementation must strictly follow the tool specification and include robust, defensive error handling.

##### Procedure

- 1. Review the Tool Document Carefully parse the tool document to extract the function name and the exact parameter specification (names, types, required/default values). These must be used verbatim in the implementation.
- 2. Interpret the Question–Answer Pairs Use the pairs to infer:

- • how problem statements map to function inputs,
- • how outputs should be computed or formatted to match the expected answers.

- 3. Develop the Function Implementation Implement the function such that:

- • the function call described by the call statement produces the correct result,
- • the function name matches the tool document exactly,
- • parameters are defined exactly as specified (including ordering and defaults where applicable),
- • internal logic derives outputs consistent with the question–answer pairs,
- • if any parameters have default values, ensure the function’s return value fully contains the corresponding expected answer (i.e., the answer must appear as a substring within the returned value),
- • the function can yield diverse outputs when required (simulate additional fields or return values if implied by the tool document).

- 4. Add Comprehensive Error Handling Implement a reliable validation and error-reporting strategy to handle:

- • incorrect parameter types,
- • missing required parameters,
- • invalid values or out-of-range inputs,
- • any other foreseeable runtime issues.

Errors should result in clear, stable messages that help diagnose the issue without breaking execution.

Output Requirements Return only a JSON object (no other text) using the following structure:

- • "analysis": A detailed explanation of the implementation approach, including design rationale for parameters, output formatting, and exception/error-handling logic.
- • "function": The full function implementation, including code and explanatory comments.

##### Notes

- • Parameter names and types must match the tool document exactly.
- • Only Python 3 built-in libraries may be used.
- • If the tool specification implies additional return values, simulate them in a way consistent with the documentation.
- • Error handling should be exhaustive and anticipatory.
- • Ensure that for any input q, the produced result is uniquely a.

Provided Inputs Tool Document

{document} Question–Answer Pairs {pairs} Call Statement {call_statement}

