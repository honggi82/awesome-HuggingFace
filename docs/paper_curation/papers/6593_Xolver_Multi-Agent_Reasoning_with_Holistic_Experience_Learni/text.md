arXiv:2506.14234v1[cs.CL]17Jun2025

# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team

### Md Tanzib Hosain1∗ Salman Rahman2∗ Md Kishor Morol3 Md Rizwan Parvez4

1American International University-Bangladesh 2 University of California, Los Angeles 3Cornell University 4Qatar Computing Research Institute mparvez@hbku.edu.qa

## Abstract

Despite impressive progress on complex reasoning, current large language models (LLMs) typically operate in isolation—treating each problem as an independent attempt, without accumulating or integrating experiential knowledge. In contrast, expert problem solvers—such as Olympiad or programming contest teams—leverage a rich tapestry of experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality, adapting strategies based on the expertise and experiences of peers, continuously refining their reasoning through trial and error, and learning from other related problems even during competition. Inspired by this, we introduce Xolver—a training-free, multi-agent reasoning framework that equips a black-box LLM with a persistent, evolving memory of holistic experience. Xolver integrates diverse experience modalities, including external and self-retrieval, tool use, collaborative agent interactions, agent-driven evaluation, and iterative reasoning refinement. By learning from relevant strategies, code fragments, and abstract reasoning patterns at inference time, Xolver avoids generating solutions from scratch—marking a transition from isolated inference toward experience-aware language agents. Built on both open-weight and proprietary models, Xolver consistently outperforms specialized reasoning agents (e.g., OctoTools, CheatSheet, Search-o1). Even when instantiated with lightweight backbones (e.g., QWQ-32B), it often surpasses the most advanced models to date—including Qwen3-235B, Gemini 2.5 Pro, o3, and o4-mini-high. With a stronger backbone like o3-mini-high, it achieves a new best result—98.1% on GSM8K, 94.4% on AIME’24, 93.7% on AIME’25, 99.8% on Math-500, and 91.6% on LiveCodeBench—highlighting holistic experience learning as a key step toward dynamic, generalist agents capable of expert-level reasoning. We open-source all code, and data of Xolver at https://kagnlp.github.io/xolver.github.io/.

## 1 Introduction

Recent advances in large language models (LLMs) have made remarkable progress in complex reasoning and problem solving across domains such as mathematics [6, 15, 26] and programming [4, 3, 21]. Yet despite these impressive capabilities, conventional LLM reasoning approaches remain fundamentally limited: they standalone each problem instance, generating solutions from scratch without accumulating or transferring insights from rich, diverse experiential knowledge.

This isolated reasoning paradigm marks a significant departure from how expert human problem solvers operate. Expert problem solvers—such as an Olympiad or programming contest teams—rarely

∗Work done while working as a remote RA at QCRI.

Preprint. Under review.

###### AIME 24 & 25 (Avg)

###### LiveCodeBench (v5)

100.0

94.1

95.0

93.0

91.6

91.6

90.0

87.3

86.9

###### Accuracy(%)

###### Pass@1(%)

85.0

80.0

76.8

75.0

73.1

71.0

69.5 69.5

70.0

66.3

65.0

o3-mn-m o1 o3-mn-h o4-mn-h Xolver (m)

Xolver (h)

o3-mn-m o1 o3-mn-h o4-mn-h Xolver (m)

Xolver (h)

- Figure 1: Results Summary on AIME ’24 (16 runs), AIME ’25 and LiveCodeBench (32 runs). Our framework Xolver, built on o3-mini-medium and o3-mini-high backbones (denoted (m) and (h)), achieves up to 30.9% gain over the baseline and often outperforms leading models on both tasks.

approach problems in a vacuum. Instead, they draw upon a rich tapestry of cumulative experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality (e.g., calculator), adapting strategies based on peers’ expertise and experiences, gaining insights through iterative trial and error, and learning from related problems even during competition. This holistic experience empowers them to tackle new challenges not from scratch, but by dynamically applying accumulated knowledge and adaptive strategies.

While numerous prior studies have enhanced LLM reasoning and problem solving through various forms of experiential knowledge augmentation, they have predominantly operated within discrete modalities—retrieving similar problems or relevant contexts [46, 25, 14], leveraging external tools [33, 32], or facilitating multi-agent collaboration [18, 16, 62]. Despite their individual strengths, these approaches address distinct facets of experiential knowledge independently, preventing LLMs from accumulating and synthesizing a comprehensive repertoire of learning signals across diverse experiential dimensions, thereby limiting the development of the rich, interconnected knowledge structures that characterize human expertise.

In this paper, we introduce Xolver, a unified, memory-augmented, multi-agent inference framework that emulates the holistic experience-driven, collaborative reasoning of expert teams. Xolver dynamically orchestrates a roster of specialized agents—such as mathematicians, programmers, verifiers—that iteratively tackle complex problems. Unlike conventional LLM pipelines, Xolver seamlessly integrates planning, episodic retrieval—both from external or self-parametric long-term memory—an evolving intermediate shared memory, tool invocation, multi-agent collaboration, agentdriven evaluation, and iterative self-refinement into a single adaptive architecture.

Each agent’s reasoning begins with exemplars drawn from episodic memory. From the second iteration onward, agents rely exclusively on an evolving shared memory that records the highestquality reasoning paths, solutions, and evaluation feedback generated so far—thereby accumulating symbolic experience over time. This shared repository guides agents to build on successful strategies, correct mistakes, and improve solution quality. When needed, agents invoke external tools (e.g., code execution), and a dedicated judge agent reviews all outputs—selecting top responses, issuing feedback, and enriching the intermediate shared memory with curated traces and collective evaluations for future rounds. Iterations continue until outputs converge or a preset limit is reached, followed by a final verification or external debugging phase to ensure correctness. Additionally, by updating its episodic store with each newly solved problem and its reasoning trace, Xolver can continually expand its knowledge base. Through this closed loop of collaborative agents, memory-guided refinement, and tool guided precision, Xolver features a more holistic experience learning and transcends static LLM inference, delivering adaptive, expert-level reasoning over time. Figure 2 illustrates the workflow.

We conduct large-scale experiments across a range of math and programming benchmarks—including GSM8K, Math-500, AIME (2024 and 2025), and LiveCodeBench (v5)—using both proprietary (o3mini-medium) and open-weight (QWQ-32B) backbone models. Xolver consistently outperforms specialized reasoning systems such as OctoTools [33], CheatSheet [55], and Search-o1 [28]. Remarkably, even when instantiated with lightweight models, Xolver often surpasses significantly larger state-of-the-art LLMs, including Qwen3-235B [56], Gemini 2.5 Pro [7], o1, o3, and o4-minihigh [41]. As in Figure 1, Xolver (m) achieves 91.6% average accuracy on the AIME ’24 and ’25 benchmarks—an 18.5-point gain over o3-mini-medium—while Xolver (h) reaches 94.1%, outper-

Initialization, Plan, Orchestration Multi-Agent Collaboration, Tool Usage, and Iterative Refinement Solution Generation

[Figure 1]

- [ T0i , R0i ]

- [ T1i , R1i ]

- [ T2i , R2i ]

Planner Agent LLM Judge Agent

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

|[Figure 6]<br><br>[Figure 7]| |
|---|---|
| | |

Final Output

[ y, TF, RF ]

Problem Specification

Tool Execution

[Figure 8]

| | | |
|---|---|---|
|1.<br>2.<br>3.<br>|[ T2i , R2i , S2i , a2] [ T1i , R1i , S1i , a1] [ T0i , R0i , S0i , a0]| |
| | | |

|[Figure 9]|
|---|

Dynamic Agents

Thought, Response

[Figure 10]

[Figure 11]

Prior Generation

Self/External Retrieval

Verifier/ Episodic Debugger Retrieval Memory

Iterative Refinement

Intermediate Shared memory

Update on Completion

Prior Experinece Synthesize, Accumulate and Refine Experience Experience to Next

- Figure 2: Xolver Scaffold. At each iteration, agents receive their past reasoning history and topranked exemplars to generate new thoughts and responses, using tools (e.g., code) as needed. A judge model ranks outputs, and an intermediate memory maintains the best responses over time. Exemplars are initialized via episodic retrieval and continually updated with high-quality solutions from the memory. Iteration stops when convergence or max steps are reached, followed by final verification.

forming o3-mini-high by 7.2 points. On LiveCodeBench, Xolver (m) improves upon its base by 21 points (66.3% to 87.3%), with Xolver (h) achieving 91.6%, a 22.1-point lift over o3-mini-high.

Our analysis reveals how Xolver’s experiential components contribute to its performance. Accuracy improves consistently with more agents and iterations, reflecting the benefits of experience accumulation, though at increased cost. While external retrieval remains powerful, we find that self-retrieval—drawing from the model’s own parametric memory—can serve as an alternative with some performance drop. For tasks involving symbolic reasoning and complex arithmetic, multi-agent, multi-iterative refinement is more beneficial than tool use (e.g., Python execution). Our experiments confirm that even without updating episodic memory during inference, Xolver retains substantial performance gains, emphasizing the strength of its intermediate memory and iterative refinement. Together, these findings highlight Xolver ’s ability to accumulate, refine, and reuse symbolic experience through collaborative, memory-guided reasoning.

## 2 The Xolver Framework

Given a problem query q ∈ Q and a pretrained language model LLMθ(·), a conventional approach generates a solution via single-step inference: y ∼ LLMθ(q). In contrast, Xolver executes a dynamic, multi-agent reasoning process that iteratively accumulates and leverages symbolic experience to solve complex problems more effectively.

To support structured collaborative reasoning, Xolver maintains two complementary forms of memory: an episodic memory DE, which stores a library of past problems, solutions, and reasoning traces; and an intermediate dynamic shared memory DS, which evolves during inference to retain high-quality agent trajectories—comprising reasoning thoughts, responses, agent metadata, and feedback. In Xolver, a multi-agent team A is orchestrated adaptively by a planner agent P, which assigns roles and configures memory access. During inference, A agents leverage an external toolset T (e.g., Python interpreter) to support accurate computation. Finally, a verifier or external debugger V is invoked to extract and format the final answer, and to validate correctness for executable outputs.

Below, we first describe the Xolver agents and tools in Section 2.1, followed by the memory components in Section 2.2, and the inference cycle in Section 2.3.

### 2.1 Agents and Tools

Planner Agent P. The planner agent P is responsible for initiating, planning, and orchestrating the Xolver multi-agent architecture. Given the problem q and the number of agents m, it constructs a team A of m dynamic agents, each assigned a distinct expert role (e.g., algebra solver, mathematician,

theorist, programmer, algorithm designer) tailored to the demands of q. To ensure sufficient task coverage and role diversity, P first prompts the underlying LLM to over-generate M > m candidate agents, from which it then selects the most effective subset A ⊂ {a1,...,aM} such that |A| = m. A summary of the most frequently generated and selected roles is provided in Appendix D.4.

Dynamic Reasoning Agents A. The set A = {a1,a2,...,am} represents a team of dynamic reasoning agents constructed by the planner agent P. Each agent aj ∈ A is assigned a distinct expert role (e.g., algebra solver, programmer, counter-example generator) tailored to the task query q. Agents are instantiated using a standardized prompting template (see Appendix A) that incorporates the task description, assigned role, retrieved examples, prior reasoning attempts, and shared memory feedback—enabling iterative self-correction and role specialization.

At each iteration i, agent aj receives a context Cij and generates a structured reasoning trace Tij and a response Rij. For the first iteration (i = 0), the context is initialized using the task query and relevant retrieved exemplars:

A ← C0j = {q} ∪ R(DE). (BUILDCONTEXT) For subsequent iterations (i ≥ 1), the context evolves by incorporating its prior generation (history) and the shared memory:

A ← Cij = {q} ∪ {Tij−1,Rij−1} ∪ DS. (BUILDCONTEXT) Judge Agent J . The judge agent J evaluates intermediate outputs from each agent and returns structured feedback to guide refinement and memory updates. Given a query q, a reasoning trace T, and a response R, it produces a feedback tuple S = (TS,s), where TS is a natural language explanation (e.g., critique, justification, correction), and s is a scalar quality score. The interpretation of s is task-dependent: for math problems, s ∈ [0,1] reflects an LLM-estimated correctness probability; for code tasks, s ∈ {0,1,...,Ntest}, where Ntest denotes the total number of test cases including problem-provided samples and 10 synthesized test cases generated using AceCode-RM-32B [67]. To avoid compiler interaction latency and maintain symbolic traceability, test case outcomes are determined by simulating execution through LLM prompting within the judge agent J , following the CodeSim protocol [18]. This structured feedback enables agents to identify failures, receive localized corrections, and improve reasoning over iterations.

Verifier Agent V. Due to linguistic complexity and varying answer specification formats, a response may be incorrect even when the underlying reasoning or open-ended response is valid. For instance, answer formats may require multiple-choice letters (e.g., “(A)” or “Choice B”), boxed numerical values (e.g., “

|42|
|---|

”), or final answers in specific units (e.g., “5 km” or “12%”). An additional round of answer extraction and formatting helps reduce such mispredictions [44]. This challenge is even more pronounced in code generation tasks, where predicted code may fail to execute or not pass all test cases. To mitigate this, Xolver includes a Verifier Agent V, which operates differently based on the output type. For math and QA problems, V extracts the final reasoning TF, response RF, and answer y from the response associated with the top-ranked entry BESTRESPONSE in DS, ensuring adherence to the expected output format. For executable code, Xolver invokes an external debugger (LDB [70]), where V interacts with a Python runtime to capture execution feedback and iteratively fix runtime errors.

Tools T . Integrating natural language reasoning with tools like Python execution is a proven way to boost performance on complex reasoning tasks [37, 57]. We observe that even advanced reasoning models often make mistakes in intermediate steps, particularly when computations become non-trivial. To address this, each dynamic agent aj is explicitly instructed to use Python execution during reasoning when needed. While Xolver currently limits T to Python, our prompting strategy is tool-agnostic, allowing an interface for future extensions to richer toolsets [32, 33].

All agents are built using the underlying LLM. All prompts are 0-shot and provided in Appendix A.

### 2.2 Memory Components

Episodic Memory DE. Xolver maintains two forms of episodic (long-term) memory: (1) an external memory corpus DEext = {(q′,T′,R′)}, which consists of past problem instances q′, their corresponding reasoning traces T′ (optional), and solution responses R′; and (2) the internal parametric memory encoded in the weights of the agent-specific language model LLMj.

We define a general retrieval operator R(DE) that returns a set of K examples relevant to the query q. When DEext is available, retrieval is conducted using similarity-based search (e.g., BM25):

R(DE) = {(qk′ ,Tk′,Rk′ )}Kk=1 ← Retrievej(q,DEext).

Otherwise, Xolver falls back to internal self-retrieval by sampling from the agent model itself:

R(DE) = {(qk′ ,Tk′,Rk′ )}Kk=1 ∼ LLMj(q). In the case of an external episodic memory, DE can also be updated with UPDATEEPISODICMEMORY by adding the top-ranked reasoning and response from DS, paired with the problem q, into the external corpus DEext. That is, DEext ← DEext ∪ (q,T,R), where (T,R,S,a) is the top-ranked entry in DS.

Intermediate Shared Memory DS. The shared memory DS maintains a fixed-size set of highquality intermediate reasoning, responses, and metadata generated by the dynamic agents during inference on the current query q. For simplicity and to preserve the dynamic nature of the framework, we constrain |DS| = m, where m is the number of dynamic agents in A. Initially, DS ← ∅. At each iteration i, each agent aj ∈ A produces a reasoning trace Tij, response Rij, and receives structured feedback Sij = (TS(i,j),si,j) from the judge agent J , where TS(i,j) is a natural language explanation and si,j is a scalar score reflecting the quality of the tuple (Tij,Rij). After collecting the new outputs

τij = (Tij,Rij,Sij,aj), j = 1,...,m, (RUNAGENTS)

we form the candidate pool M = DS ∪ {τi1,...,τim}. We then update the fixed-size shared memory by keeping only the top-m tuples by score

DS ← TopK M, m; key(e) = s(e) ,, (UPDATESHAREDMEMORY) where s(e) extracts the scalar score from e = (T,R,(TS,s),a).

This replacement mechanism ensures that DS always contains exactly m entries with the highest observed scores across all iterations. By maintaining only the strongest reasoning-response-feedback tuples, the shared memory facilitates knowledge transfer between agents and across iterations, enabling collaborative improvement through exposure to diverse high-quality solutions.

- 2.3 Inference Protocol Algorithm 1 Xolver Inference Protocol

Algorithm 1 summarizes the Xolver inference protocol, which operates in three structured stages. Stage-1, which emulates initialization with prior experience, involves the planner constructing a team of agents A (lines 2–3). Stage-2, embodying symbolic experience accumulation and refinement, iterates for I rounds (lines 4–10). In each round, all agents receive access to DS and DE, build their contexts, and generate structured trajectories and responses (DE is only used for context construction at the first iteration). These are evaluated by the judge agent J , and DS is updated with the resulting feedback tuples (line 7). Upon convergence or after I rounds, Stage-

- 1: Input: Query q, Tools T , Episodic Memory DE, parameters m,k,I
- 2: Init: DS ← ∅
- 3: A ← PLANNER(q,m)
- 4: for i = 0 to I do
- 5: {Ci}mc=1 ← BUILDCONTEXT(A,DE,DS,q,i)
- 6: {τij}mj=1 ← RUNAGENTS(A,Ci,T ,J )
- 7: DS ← UPDATESHAREDMEMORY(DS,{τij})
- 8: if CONVERGED(DS) then
- 9: break
- 10: end if
- 11: end for
- 12: y ← V(BESTRESPONSE(DS))
- 13: UPDATEEPISODICMEMORY(DE,q,DS)
- 14: Return y

- 3 invokes the verifier agent V, which extracts the final answer from the top-ranked entry in DS (line 11), and updates DE with the new experience.

## 3 Experiments

### 3.1 Evaluation Setup

Evaluation Benchmarks We evaluate Xolver across five diverse and challenging benchmarks covering both mathematical and coding reasoning. For math, we use GSM8K [6], Math-500[15], and

the AIME 2024 [34] and 2025 [35], comprising high-school level competition problems requiring multi-step symbolic reasoning. For coding, we use LiveCodeBench (v5) [20], a dynamic benchmark that ensures no data leakage by periodically releasing new problems. These benchmarks span arithmetic, algebra, number theory, geometry, combinatorics, and algorithmic problem solving.

Baselines and Metrics We compare Xolver against directly using leading reasoning models– (a) proprietary models: Gemini 2.5 (Pro and Flash Think) [7], Grok-3 Beta Think and Grok-3 Mini (Beta) Think [63], Claude 3.7 Sonnet Think [2], o1 [41], o3-mini, o3, and o4-mini [42]; (b) openweight LLMs, e.g., Qwen3-235B [48], QWQ-32B [49], and DeepSeek-R1 [8]; (c) math- and codespecialized models, e.g., AlphaOne [68], OpenMathReason [37], rStar-Math [12], rStar-Coder [30], OpenCodeReason [1], and Kimi-K1-1.6 [22]. We also compare with (d) agents or frameworks: SelfReflexion [52], agentic search based framework Search-o1 [28], specialized tool based framework OctoTools [33] which excels general purpose agent platforms outperforming AutoGen or LangChain, cross-problem baseline framework CheatSheet [55], and multi-agent code generation framework CodeSim [18], which leverage refinement, retrieval or online search, fine-grained tool augmentation in addition to online search, dynamic memory updates after solving new problems, and multi-agent reasoning techniques respectively. For agent-based baselines (d), we reproduce results using the same backbone LLMs as Xolver for fair comparison; for model-based baselines (a–c), we report official results from their technical reports or corresponding benchmark leaderboards. As evaluation metric, we use accuracy using GPT-4o [40] for math problems, and pass@1 for code tasks.

Inference Details We use both open-weight QWQ-32B [48] and proprietary o3-mini (medium and high) [42] as the backbone. To mitigate performance variance inherent in single-run evaluations, we report the average accuracy and pass@1 metric, calculated by averaging 32 inference runs for competitive benchmarks LIVECODEBENCH and AIME ’25, and 16 runs for AIME ’24, ensuring standard deviation within ∼ 1% (Appendix D.1). For simpler tasks GSM8K and MATH-500, we follow DeepSeek-v3 [29], using a single greedy-decoded generation. By default, we set temperature to 0.2, number of agents m = 3, and max iterations I = 2. Xolver iteration terminates either when the maximum number of iterations I is reached, or when all entries in the shared memory Ds converge—i.e., they achieve perfect scores of 1.0 (correct) for math tasks, or pass all test cases (both sample and synthesized) for code tasks. As the external retrieval corpus DEext in coding task, we collect a 9-million-token dataset of algorithmic code problems and their C++ solutions with explanations from GitHub2 (details in Appendix C). For math, we use the OPENMATHREASON dataset [37] as DEext. We evaluate two variants of Xolver: (i) Xolver with in-competition cross-problem experience (Xolver (+)), which dynamically updates the episodic memory after solving each problem to utilize accumulated knowledge across problems; and (ii) Xolver (–), which keeps the episodic memory static, focusing solely on problem-specific experience. By default, we refer to Xolver (+) as our method if not specified otherwise.

### 3.2 Main Results

Table 1 evaluates Xolver across diverse mathematical and coding reasoning benchmarks, highlighting its effectiveness compared to state-of-the-art LLMs, specialized models, and other frameworks.

Strong Gains Across Benchmarks Overall, Xolver consistently delivers significant improvements over the backbone LLMs’ standard LongCoT prompting. Both the problem-specific Xolver (–) and the cross-problem Xolver (+) variants outperform their respective backbone LLM (LongCoT) baselines across all datasets. For example, with the o3-mini-medium backbone, Xolver (+) improves from 75.8 to 93.8 on AIME’24, and from 66.3 to 79.6 on LiveCodeBench, while the QWQ-32B backbone sees gains from 78.1 to 89.9 on AIME’24 and from 63.4 to 76.2 on LiveCodeBench.

Surpassing Prior Agents Compared to previous frameworks such as Search-o1, OctoTools, and CheatSheet, Xolver demonstrates consistent and significant gains. With o3-mini-medium, Xolver (+) improves over the best baseline by +12.7 points on AIME’25 and +13.5 points on LiveCodeBench, highlighting its superior reasoning capabilities by integrating diverse forms of experience.

In Comparison to Leading LLMs Despite using weaker backbones, Xolver, specifically (+) variant, matches or surpasses proprietary frontier LLMs like o3 and o4-mini-high on key benchmarks. With o3-mini-medium, Xolver (+ outperforms o4-mini-high on AIME’24 (93.8 vs. 93.4) and substantially

2https://github.com/cp-algorithms/cp-algorithms

LiveCodeBench (v5) Proprietary Models

Math -500

AIME ’24

AIME ’25

Model Appr. GSM8K

Claude 3.7 Sonnet T. LongCoT – 61.3 49.5 96.2 51.4 Grok-3 (Beta) T. Direct – 83.9 77.3 – 70.6 Grok-3-mini (Beta) T. LongCoT – 89.5 82.0 – Gemini 2.5 Flash T. LongCoT – 88.0 78.0 – 63.5

- o1 LongCoT 96.4 74.3 79.2 96.4 71.0
- o3-mini-high LongCoT – 87.3 86.5 – 69.5 Gemini 2.5 Pro. Direct – 92.0 86.7 – 70.4
- o3 LongCoT 96.7 91.6 88.9 – –
- o4-mini-high LongCoT – 93.4 92.7 – 69.5 Open Weights Models

DeepSeek-R1 LongCoT – 79.8 70.0 97.3 64.3 Qwen3-235B-A22B LongCoT – 85.7 81.5 – 70.7

###### Math/Code Specialized Models

rStar-Math (Best) – 95.2 53.3 – 90.0 – OpenMathReason (Best) – – 93.3 80.0 – – AlphaOne (Best) – – 53.3 – 89.4 75.8 OpenCodeReason (Best) – – – – – 61.8 rStar-Coder (Best) – – – – – 62.5 Kimi-k1.6-IOI-high – – – – – 73.8

###### Reasoning Agents/Frameworks

LongCoT 95.2 75.8 70.4 97.3 66.3 Self-Refl. 93.1 79.4 76.5 95.2 73.2 OctoTools 95.4 81.7 75.3 97.5 – Search-o1 95.8 81.8 76.7 97.6 73.6 CheatSheet 95.9 82.2 75.8 97.7 – CodeSim – – – – 73.8 Xolver (–) 95.6 87.2 85.1 97.7 79.6

o3-mini-medium

- Xolver (+) 97.1 93.8 89.4 99.2 87.3

QWQ-32B

LongCoT 96.1 78.1 65.8 83.2 63.4 Self-Refl. 94.0 79.3 66.3 80.4 69.2 OctoTools 96.3 83.0 71.7 86.1 – Search-o1 96.4 84.4 71.8 87.1 69.3 CheatSheet 96.8 83.5 72.2 86.5 – CodeSim – – – – 70.5 Xolver (–) 96.5 89.9 79.5 93.1 76.2

- Xolver (+) 98.0 93.6 82.7 95.5 79.2

o3-mini-high Xolver (+) 98.1 94.4 93.7 99.8 91.6

- Table 1: Comparison of Xolver against SoTA reasoning models, specialized models, and other reasoning agents across mathematical and coding tasks. Best results are boldfaced and second-best results are underlined. T: Think models, LongCoT*: standard prompting for reasoning models. "-" denotes either n/a (e.g., only math/code specialized models) or results not reported.

exceeds it on LiveCodeBench (87.3 vs. 69.5), demonstrating that structured reasoning and dynamic memory can rival even the strongest closed-source models.

Backbone Agnostic Improvements from Xolver are consistent across different backbone LLMs. Both o3-mini-medium and QWQ-32B benefit substantially from the framework, demonstrating its model-agnostic design. For example, on GSM8K, Xolver (+) achieves 97.1 (o3-mini-medium) and 98.0 (QWQ-32B), both surpassing baseline variants by significant margins.

Effectiveness of Dynamic Episodic Memory While both variants excel, the cross-problem variant Xolver (+) consistently outperforms the problem-specific version Xolver (-) in all benchmarks. On average, episodic memory integration yields a +3.5 point improvement across both backbones and datasets where the largest gain is +7.7 points with o3-mini-medium on coding (LiveCodeBench).

Scales with Backbone LLM’s Strength Xolver’s performance scales consistently with the strength of its backbone LLM. With o3-mini-high, it sets new state-of-the-art results across all benchmarks (98.1 on GSM8K, 94.4 on AIME’24, 93.7 on AIME’25, 99.8 on Math-500, and 91.6 on LiveCodeBench).

## 4 Ablation and Analyses

Ablations: Quantifying Component Impact In Figure 3, we present an ablation study quantifying the contribution of individual components in Xolver to overall performance, measured by the average performance drop on math reasoning (Math Avg) and programming (LiveCodeBench) tasks.

Each component plays a necessary role, with the most significant degradation observed when removing Multi-iteration and Multi-Agent followed by Judge Agent, highlighting their central importance in complex reasoning and code synthesis. In contrast, removing components like Verifier/Debugger and Tool leads to comparatively smaller drops, suggesting a more auxiliary role in the overall system. Likewise self-retrieval can also work in-place of external retrieval with some drop in accuracy.

###### Performance Drop by Variant

Math Avg

30

LiveCodeBench

###### PerformanceDrop(%)

25

20

23.70

16.40

15

9.90

11.10

10

5

8.70

7.58

7.25

5.80 4.65

5.63

2.95

0

Multi-iteration Multi-Agent JudgeAgentExternalRetrievalSynthesizedTest ToolVerifier/Debugger

Figure 3: Performance drop when removing each component from Xolver. Bars show average drop on Math (bottom) and LiveCodeBench (top). Verifier is critical for math tasks and cannot be removed, while Tool (Python) and test cases apply only to math and coding respectively.

Impact of Agent Count and Iterations, and Emerging Benefits of Collaboration We analyze the effect of varying the number of agents and reasoning iterations on Xolver’s performance. In a controlled setup, we fix one variable (e.g., 3 agents or 2 iterations) and incrementally increase the other. As shown in Figure 4, performance improves consistently on both AIME ’25 and LIVECODEBENCH with more agents or iterations, highlighting the advantage of collaborative and iterative problem solving.

To probe deeper, we conduct a budget-controlled experiment on the AIME ’25 dataset, where the total reasoning budget (i.e., number of agents × number of iterations) is fixed. While iterative reasoning remains a crucial factor for Xolver’s performance, we find that increasing the number of agents—particularly beyond a minimum of three—yields additional, emergent improvements, leading to over a 4% performance gain. This suggests that agent diversity and parallelism complement iterative depth, together producing stronger collaborative problem-solving benefits than either alone.

[Figure 12]

Figure 4: Impact of iterations and agents in Xolver on AIME ’25 (QWQ-32B) and LIVECODEBENCH (o3-mini-medium).

Effect of Retrieval Strategies on Xolver Performance. We evaluate the impact of different retrieval strategies on Xolver by comparing three settings: (1) External Retrieval, where the model retrieves the top-k (e.g., k = 5) most similar problems and their solutions from an external corpus using a BM25 retriever; (2) Self-Retrieval, where the model recalls the top-k most similar problems and solutions from its own internal memory; and (3) No Retrieval, where neither external nor self-retrieval is used. As shown in Figure 5, performance on both AIME ’25 and LIVECODEBENCH follows the trend: External Retrieval > Self-Retrieval > No Retrieval, indicating that external retrieval significantly enhances Xolver’s performance. We note that for code tasks, although the external retrieval corpus

###### Accuracy by Retrieval Methods

AIME '25 (o3-mini-medium)

LiveCodeBench (o3-mini-medium)

AIME '25 (QWQ-32B)

LiveCodeBench (QWQ-32B)

95

91.6

90.6

Accuracy(%)

85

84.2

81.4

80.2

79.1

76.3

76.2

75

73.8

72.4

71.5

68.7

65 No Retrieval Self-Retrieval External Retrieval

Figure 5: Impact of different retrievals in Xolver.

contains solutions written in C++—a different language from the target Python—external retrieval still provides a substantial performance boost. Nonetheless, while self-retrieval results in a notable performance drop compared to external retrieval, it still outperforms the no-retrieval baseline with notable margins, serving as a viable alternative when external resources are unavailable.

Fine-grained Performance Analysis We perform a fine-grained analysis of Xolver’s performance across both MATH-500 and LIVECODEBENCH, as shown in Figure 6 and Figure 7.On MATH-500, Xolver (both o3-mini-medium and QWQ-32B) consistently outperforms CHEATSHEET across nearly all seven subject categories, despite the latter relying on costly per-problem memory updates. The only exception is in Number Theory, where o3-mini-medium scores 99.2 compared to CHEATSHEET’s 99.5. As for QWQ-32B, Xolver achieves substantial accuracy gains over CheatSheet across all categories, with improvements of +9.0% in Prealgebra, +8.5% in Algebra, +11.0% in Number Theory, +8.5% in Counting and Probability, +8.8% in Geometry, +10.0% in Intermediate Algebra, and +7.5% in Precalculus. These consistent gains highlight Xolver’s strong performance across both symbolic and numerical reasoning.

###### Math-500 Accuracy by Subject Category

CheatSheet (o3-mini-medium)

Xolver (o3-mini-medium)

CheatSheet (QWQ 32B)

Xolver (QWQ 32B)

100.0

99.5 99.2 99.5

99.5

99.0

99.0

99.0 99.0

98.5

98.0

97.5

97.5

97.0

97.0 96.8 96.8

96.5

95.8 95.5

95.0

94.8 94.5

93.5

Accuracy(%)

92.5

90.0

89.5

88.0

87.5

87.0

86.0

86.0

85.5

85.0

84.5

82.5

80.0 Prealgebra Algebra Num Theory Count & Prob Geometry Interm Algebra Precalculus

Figure 6: Fine-grained performance comparison in MATH-500.

###### LiveCodeBench Accuracy by Difficulty

CodeSim (o3-mini-medium)

Xolver (o3-mini-medium)

On LiveCodeBench, Xolver demonstrates even more pronounced gains. The o3-mini-medium variant achieves 95.6%, 90.4%, and 85.8% accuracy on Easy, Medium, and Hard problems respectively, significantly outperforming CodeSim by +4.5%, +11.9%, and a striking +32.3% margin on hard examples. Even with a weaker QWQ32B backbone, Xolver (95.2%, 87.5%, 70.0%) surpasses all baselines and achieves similar gains. In contrast to CheatSheet and CodeSim, Xolver leverages multi-agent collaborations and holistic experience learning. These consistent and backbone-agnostic gains across different reasoning tasks underscore Xolver’s robustness and position it as a breakthrough in retrieval and tool-augmented, multi-agent and evolving reasoning systems.

CodeSim (QWQ-32B)

Xolver (QWQ-32B)

100

95.6

95.2

91.1

90.4

89.9

90

87.5

85.8

Accuracy(%)

80

78.5

76.7

70.0

70

60

53.2

50

43.8

40

Easy Medium Hard

Figure 7: Performance comparison per difficulty levels in LiveCodeBench

Can a Self-Judge Replace a Judge Agent? We analyze the effect of different judging mechanisms on Xolver’s performance by comparing two setups: (1) self-judging, where each dynamic agent evaluates its own response through self-reflection without altering its role, and (2) external judging, where a separate judge agent is used to assess the responses. We find that self-judging agents tend to be biased in favor of their own outputs, occasionally validating incorrect solutions. This self-bias leads to a noticeable drop in overall performance—specifically, a 9.9% decrease in coding tasks and a 3.88% decrease in math tasks, on average.

###### Average Token Usage (Input, Think, Output) per Dataset

Cost Analysis and How Long Do Xolver Agents Think? We perform a detailed analysis of token usage in Figure 8, reporting input, reasoning, and output statistics for Xolver (QWQ32B) across all datasets. Our LLM token usage has computational complexity of O(mI), where m is the number of agents and I is the number of reasoning iterations. However, the run-

Search-o1 (03-mini-medium)

Xolver (03-mini-medium)

| |
|---|

Search-o1 (QWQ-32B)

Xolver (QWQ-32B)

175

24.2

29.6

150

18.1

###### Tokens(thousands)

48.8

14.7

15.5

125

22.4

59.7

36.5

29.7

31.4

100

12.2

11.5

45.3

10.6

11.5

23.3

24.6

12.6

75

21.4

7.8

23.3

25.5

8.6

9.4

110.4

15.7

6.9

50

96.4

89.6

88.9

17.4

86.1

18.9

5.1

14.0

68.0

67.1

10.4

65.6

60.5

25

50.1

44.7

44.5

34.6

33.4

32.5

28.3

0 GSM8K AIME '24 MATH-500 LiveCodeBench

Figure 8: Avg numbers of token usage across datasets in Xolver (+).

time complexity remains O(I) since the dynamic agents operate in parallel. This is significantly more efficient than the self-consistency [59], which typically require 32–64 generations per example, as well as the baseline CheatSheet framework, which incurs a memory update complexity of O(n2)—quadratic in the test dataset size—due to usefulness estimation over all previous examples after solving each new example. As a multi-agent system, Xolver allocates a majority of its tokens to context sharing and inter-agent communication, while approximately 25% are spent on actual reasoning steps.

Nonetheless in Figure 8, we also compare the total token usage of Xolver with a single agent reasoning framework Search-o1 using tiktoken for o3-mini-medium and AutoTokenizer for QWQ-32B for token count. As expected, Xolver incurs higher token costs—approximately 1.5× that of Search-

- o1—due to its collaborative and iterative multi-agent reasoning. However, this moderate increase represents a highly efficient trade-off given the substantial performance improvements observed. As shown in Figure 6 and Figure 7, Xolver achieves remarkable gains across both domains, including a

+32.3% absolute improvement on hard coding problems with o3-mini-medium and 9.05% accuracy boosts across all Math-500 categories with QWQ-32B. These findings demonstrate that Xolver ’s slightly higher reasoning cost is well-justified by its superior, generalist performance across diverse problem-solving scenarios.

Does Data Shuffling Affect Xolver (+) Performance? Xolver (+) updates its external memory incrementally after solving each new problem. To examine whether the order of test instances impacts performance, we conduct an ablation study by randomly shuffling the sequence of problems in each task. This helps determine if there is any dependency on the data order. Results in Appendix D.3 show that Xolver exhibits minimal performance variation across different shuffles, with a standard deviation of approximately 1 within only 5 runs, indicating that its performance is largely stable regardless of data ordering.

Qualitative Examples In Appendix B, we present qualitative examples along with all the prompts

- of full-cycle Xolver on both math and code reasoning tasks. These examples illustrate how Xolver initiates reasoning from external or self-retrieved exemplars, engages in multi-agent collaboration, and incrementally accumulates experiences through inter-agent propagation and refinement. The full interaction trace highlights Xolver’s ability to iteratively decompose, solve, and adapt solutions across reasoning steps, showcasing its capacity for dynamic knowledge construction and generalizable problem solving.

More Error Analysis in Math and Code In Figure 9, we present an error analysis across both math and code tasks that goes beyond simple accuracy or pass@1 metrics. While Xolver significantly improves reasoning and generation capabilities in both domains, both (o3-mini-medium and QWQ-32B) backbone LLMs can still produce solutions that are syntactically correct yet semantically flawed, resulting in failed executions due to incorrect reasoning, incomplete logic, unoptimized implementations, or misaligned tool usage. In code tasks, failure modes include incorrect final code, time limit exceeded (TLE), runtime errors (RTE), and syntax issues. In math tasks, remaining errors are primarily due to flawed logical derivations or faulty intermediate calculations. Although Python-based tools are available, such calculation errors often occur when agents choose not to invoke these tools—highlighting that tool usage remains decoupled from the model’s core reasoning process (see Appendix A for our prompt design). These findings provide insights for future improvements by exposing the variety of failure modes across domains, and further emphasize the importance of robust self-verification and refinement mechanisms, as employed by Xolver.

Wrong Reso. Wrong Calc. Other Wrong Ans. TLE RTE Syntax + Other

###### AIME '25

LiveCodeBench

o3-mini-medium

Total: 8.4%

Total: 9.4%

3.6%

3.0%

1.8%

3.9%

2.6%

2.1%

0.8%

QWQ-32B

Total: 18.6%

Total: 15.8%

7.7%

7.3%

3.6%

5.9%

5.3%

3.1%

1.5%

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5

Error Rate (%)

Error Rate (%)

Figure 9: Xolver Math and Code error distribution.

Dynamics of Reasoning Patterns in Xolver Traces To understand how Xolver adapts its reasoning process to perform complex reasoning, we analyze the dynamics of reasoning pattern frequencies across difficulty levels in LiveCodeBench, as shown in Table 2. Detailed description of how we collected the reasoning patterns is provided in the Appendix D.1. Our analysis reveals that Xolver dynamically increases self-evaluation and exploratory strategies (e.g., trying new approaches) as problem difficulty grows. Correct solutions demonstrate a declining need for problem rephrasing and subgoal decomposition, indicating more direct and confident reasoning. In contrast, incorrect

##### Reasoning Pattern Correct Solutions Incorrect Solutions

Easy → Medium Medium → High Easy → Medium Medium → High

Self-Evaluation (↑) 0.35 → 0.38 0.38 → 0.40 0.35 → 0.37 0.32 → 0.35 New Approach (↑) 0.18 → 0.21 0.21 → 0.24 0.17 → 0.24 0.24 → 0.26 Problem Rephrasing (↓↑) 0.20 → 0.17 0.18 → 0.18 0.23 → 0.24 0.24 → 0.25

Subgoal Setup (↓↑) 0.14 → 0.13 0.13 → 0.11 0.11 → 0.12 0.11 → 0.11

- Table 2: Changes in major reasoning pattern frequencies as problem difficulty increases in LiveCodeBench, comparing correct vs. incorrect solutions. Green and red indicate statistically significant increases or decreases (p < 0.05). Underlined cells highlight patterns where Xolver improves over OpenCodeReasoning, which otherwise shows a declining trend. Direction arrows denote: ↑ = increase, ↓ = decrease, ↓↑ = mixed trend (decrease in correct, increase in incorrect). Xolver increases use of self-evaluation and new approaches with task difficulty, and demonstrates targeted subgoal setup and problem rephrasing when solutions fail—reflecting its adaptive, collaborative reasoning.

solutions show increased subgoal setup and rephrasing attempts—suggesting that the system recognizes failure and attempts recovery through restructuring. Compared to OpenCodeReasoning, which shows stagnation or regression in key patterns (e.g., self-evaluation), Xolver exhibits robust and adaptive reasoning behavior, supported by multi-agent collaboration and judge feedback. This behavior highlights the generality and flexibility of Xolver ’s reasoning model.

## 5 Case-Study: How Xolver Enhances Reasoning

To further understand the reasoning and problem-solving strategies behind our multi-agent, iterative framework Xolver, we conduct an in-depth analysis combining qualitative runtime inspection with controlled experiments. We begin by manually studying Xolver’s agent interaction traces on AIME ’25 and LiveCodeBench. These case studies reveal that at each iteration, dynamic agents attempt to improve upon earlier failures by leveraging Judge agent feedback and by aligning with top-ranked outputs stored in the shared memory DS. This process results in progressively refined outputs, increased agent alignment, and eventual convergence toward correct solutions.

AIME '25

LiveCodeBench

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Accuracy/Agreement(%)

90

80

70

Coder Agent

60

Mathematician Agent

Agreement

1.0 1.5 2.0 2.5 3.0 3.5 4.0

1.0 1.5 2.0 2.5 3.0 3.5 4.0

Iteration

Figure 10: Agents Accuracy and Agreement over iterations.

To verify this behavior systematically, we conduct a controlled experiment across both math and code tasks. We instantiate two dynamic agents with complementary strengths: a Coder agent and a Mathematician agent, each proficient in one domain but suboptimal in the other. We then measure their performance and agreement across iterations—defined as the percentage of problems in which both agents independently produce the same correct answer (for math) or code that passes the same test cases (for code). As shown in Figure 10, both agents demonstrate consistent accuracy improvements over time, accompanied by a rising agreement rate. This not only illustrates mutual influence and learning-by-alignment but also validates the emergence of collaborative synergy.

Crucially, we observe that the presence of the Judge agent plays a vital role in this convergence process. When the Judge agent is removed—as shown in our first ablation—performance degrades significantly. These findings collectively affirm that Xolver’s iterative memory-sharing, feedbackdriven refinement, and role-specialized agents contribute to its strong reasoning performance across domains, making it a compelling framework for general-purpose, self-improving problem solving.

## 6 Related Work

Memory-Augmented and Retrieval-Augmented LLMs. Memory-augmented language models have evolved from static retrieval systems like RAG [25] and REALM [14] to dynamic approaches such as Reflexion [53], MemGPT [43], and Scratchpads [39]. However, these systems operate on

isolated tasks, lack cross-problem experience accumulation, and employ single-agent architectures. Xolver addresses these limitations through a novel dual-memory architecture combining episodic long-term memory with dynamic intermediate memory, enabling specialized agents to collectively build and refine experiential knowledge. While prior work has explored cross-trial information sharing [69, 53] and multi-source memory integration [66], these approaches remain confined to single-agent settings. Our framework creates a persistent knowledge base through multi-agent collaboration [10], allowing agents to accumulate expertise from solved problems and leverage collective experience for future tasks.

Multi-Agent Problem Solving. Multi-agent LLM systems address the limitations of single models by leveraging collaborative approaches for improved reliability and task specialization [13, 10]. From early frameworks like CAMEL [27] with fixed role assignments, the field progressed to dynamic role adjustment in AgentVerse [5] and code execution in AutoGen [62]. Recent advances include layered agent networks in DyLAN [31], multi-agent code generation and problem solving [17, 18] and multi-agent debate frameworks [9, 50, 54]. While these systems demonstrate effective collaboration, they operate on isolated problems without cross-task experience accumulation. Xolver introduces dual-memory architecture, holistic experience integration, judge-mediated selection, and continuous episodic corpus expansion—transforming single-problem solvers into experience-aware agents.

LLM Reasoning Enhancement Techniques. Various techniques have emerged to enhance LLM reasoning capabilities beyond standard prompting. Chain-of-Thought [61] introduced step-by-step reasoning, Self-Consistency [58] explores multiple reasoning paths with majority voting, and Tree of Thoughts [64] enables exploration of reasoning branches—yet all remain limited to single-pass generation. Self-reflective approaches like Reflexion [53] enable iterative improvement but operate within single tasks, while retrieval-enhanced methods like CheatSheet [55] and Search-o1 [28] remain confined to single-agent architectures. These approaches share fundamental limitations: no cross-problem learning, no persistent memory, and no multi-agent collaboration. Xolver unifies these enhancements within a multi-agent framework where agents collaboratively refine solutions through judge-mediated iterations and leverage dual memory systems for cross-problem learning.

Tool-Augmented Reasoning. Tool integration extends LLM capabilities beyond language processing. Early systems like WebGPT [38] introduced single-tool integration, while PAL [11] enabled code execution for mathematical reasoning. Multi-tool frameworks evolved with ReAct [65] interleaving reasoning with actions, Chameleon [32] composing multiple tools, and OctoTools [33] standardizing tool planning—yet all remain limited to single-agent execution without iterative refinement or crossproblem learning. Xolver transforms tool use into a collaborative, memory-enriched ecosystem where agents collectively execute tools, share outcomes, and accumulate successful strategies across problems—creating an adaptive framework that evolves with experience.

## 7 Conclusion

We propose Xolver, an open-source, multi-agent inference framework for complex reasoning tasks that enables holistic experience learning. Xolver integrates (1) episodic retrieval from external or self-parametric memory, (2) an evolving intermediate shared memory that accumulates and reuses high-quality reasoning traces, (3) tool invocation for complex computations, (4) collaborative multiagent reasoning, (5) self-evaluation and iterative refinement, (6) verification or external debugging, and (7) propagation of learned strategies across problems. These components collectively support adaptive, experience-informed problem solving. Despite its strong performance, Xolver faces limitations in computational efficiency, with substantially higher token consumption than traditional approaches, and remains dependent on the quality of backbone LLMs. Future work aims to optimize agent interactions to reduce resource requirements, enhance robustness to variations in model quality, improve retrieval filtering [60, 51, 45], develop better RAG strategies [19, 47, 24], and extend the framework to more diverse reasoning domains beyond mathematics and programming. In addition, we plan to integrate advanced external verifiers of reasoning [36] to further enforce validity through structured guardrails. By addressing these challenges, we aim to further advance the development of experience-aware reasoning systems that can approach the adaptability and integrated knowledge use of human experts.

## References

[1] Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding, 2025. URL https://arxiv.org/abs/2504.01943.

- [2] Anthropic. Claude 3.7 Sonnet, 2025. URL https://www.anthropic.com/news/

claude-3-7-sonnet.

- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David

Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared

Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.

- [5] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv preprint arXiv:2308.10848, 2(4):6, 2023.
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,

Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [7] Google DeepMind. Gemini 2.5, 2025. URL https://blog.google/technology/

google-deepmind/gemini-model-thinking-updates-march-2025/.

- [8] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement

learning, 2025.

- [9] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Im-

proving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

- [10] Shangbin Feng, Wenxuan Ding, Alisa Liu, Zifeng Wang, Weijia Shi, Yike Wang, Zejiang Shen, Xiaochuang Han, Hunter Lang, Chen-Yu Lee, Tomas Pfister, Yejin Choi, and Yulia Tsvetkov. When one llm drools, multi-llm collaboration rules. ArXiv, abs/2502.04506, 2025. URL https: //api.semanticscholar.org/CorpusID:276235808.
- [11] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.
- [12] Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rstar-math: Small llms can master math reasoning with self-evolved deep thinking, 2025. URL https://arxiv.org/abs/2501.04519.
- [13] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.
- [14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-augmented language model pre-training. In International Conference on Machine Learning. JMLR.org, 2020. URL https://dl.acm.org/doi/abs/10.5555/3524938.3525306.

- [15] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In J. Vanschoren and S. Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.
- [16] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 3(4):6, 2023.
- [17] Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. MapCoder: Multi-agent code generation for competitive problem solving. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4912–4944, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.269. URL https://aclanthology.

- org/2024.acl-long.269/.

- [18] Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. CodeSim: Multi-agent code generation and problem solving through simulation-driven planning and debugging. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Findings of the Association for Computational Linguistics: NAACL 2025, pages 5113–5139, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.285. URL https://aclanthology.org/2025.findings-naacl.285/.
- [19] Shayekh Bin Islam, Md Asib Rahman, K S M Tozammel Hossain, Enamul Hoque, Shafiq Joty, and Md Rizwan Parvez. Open-RAG: Enhanced retrieval augmented reasoning with open-source large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14231–14244, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.

831. URL https://aclanthology.org/2024.findings-emnlp.831/.

- [20] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [21] Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang, Md Rizwan Parvez, and Shafiq Joty. XCodeEval: An execution-based large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6766–6805, Bangkok, Thailand, August

- 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.367. URL https://aclanthology.org/2024.acl-long.367/.

- [22] Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
- [23] Md Tahmid Rahman Laskar, Sawsan Alqahtani, M Saiful Bari, Mizanur Rahman, Mohammad Abdullah Matin Khan, Haidar Khan, Israt Jahan, Amran Bhuiyan, Chee Wei Tan, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty, and Jimmy Huang. A systematic survey and critical review on evaluating large language models: Challenges, limitations, and recommendations. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13785–13816, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.764. URL https://aclanthology.org/2024.emnlp-main.764/.
- [24] Ahmed Lekssays, Utsav Shukla, Husrev Taha Sencar, and Md Rizwan Parvez. Techniquerag: Retrieval augmented generation for adversarial technique annotation in cyber threat intelligence text,

- 2025. URL https://arxiv.org/abs/2505.11988.

- [25] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

- [26] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35: 3843–3857, 2022.
- [27] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008, 2023.
- [28] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366, 2025.
- [29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [30] Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, and Mao Yang. rstar-coder: Scaling competitive code reasoning with a large-scale verified dataset. arXiv preprint arXiv:2505.21297, 2025.
- [31] Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization. arXiv preprint arXiv:2310.02170, 2023.
- [32] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. Advances in Neural Information Processing Systems, 36, 2024.
- [33] Pan Lu, Bowen Chen, Sheng Liu, Rahul Thapa, Joseph Boen, and James Zou. Octotools: An agentic framework with extensible tools for complex reasoning. arXiv preprint arXiv:2502.11271, 2025.
- [34] MAA. American invitational mathematics examination - aime. In American Invitational Mathe-

- matics Examination - AIME 2024, February 2024. URL https://maa.org/math-competitions/ american-invitational-mathematics-examination-aime.

[35] MAA. American invitational mathematics examination - aime. In American Invitational Mathe-

- matics Examination - AIME 2025, February 2025. URL https://maa.org/math-competitions/ american-invitational-mathematics-examination-aime.

- [36] Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024.
- [37] Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset, 2025. URL https://arxiv.org/abs/ 2504.16891.
- [38] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.
- [39] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114,

- 2021.

- [40] OpenAI. Hello GPT-4o, 2024. URL https://openai.com/index/hello-gpt-4o/.
- [41] OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

- [42] OpenAI. Introducing openai o3 and o4-mini, 2025. URL https://openai.com/index/ introducing-o3-and-o4-mini/.
- [43] Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph Gonzalez. Memgpt: Towards llms as operating systems. ArXiv, abs/2310.08560, 2023. URL https://api. semanticscholar.org/CorpusID:263909014.
- [44] Md Rizwan Parvez. Chain of evidences and evidence to generate: Prompting for context grounded and retrieval augmented reasoning. In Weijia Shi, Wenhao Yu, Akari Asai, Meng Jiang, Greg Durrett, Hannaneh Hajishirzi, and Luke Zettlemoyer, editors, Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing, pages 230–245, Albuquerque, New Mexico, USA, May 2025. Association for Computational Linguistics. ISBN 979-8-89176-229-9. doi: 10.18653/v1/2025.knowledgenlp-1.21. URL https://aclanthology.

- org/2025.knowledgenlp-1.21/.

- [45] Md Rizwan Parvez and Kai-Wei Chang. Evaluating the values of sources in transfer learning. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5084–5116, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.402. URL https://aclanthology.org/2021. naacl-main.402/.
- [46] Md Rizwan Parvez, Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Retrieval augmented code generation and summarization. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2719–2734, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.232. URL https://aclanthology.org/2021.findings-emnlp.232/.
- [47] Md Rizwan Parvez, Jianfeng Chi, Wasi Uddin Ahmad, Yuan Tian, and Kai-Wei Chang. Retrieval enhanced data augmentation for question answering on privacy policies. In Andreas Vlachos and Isabelle Augenstein, editors, Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 201–210, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.16. URL https: //aclanthology.org/2023.eacl-main.16/.
- [48] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, 2024. URL https: //qwenlm.github.io/blog/qwq-32b-preview/.
- [49] Qwen Team. QwQ-32B: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.
- [50] Salman Rahman, Sheriff Issaka, Ashima Suvarna, Genglin Liu, James Shiffer, Jaeyoung Lee, Md Rizwan Parvez, Hamid Palangi, Shi Feng, Nanyun Peng, et al. Ai debate aids assessment of controversial claims. arXiv preprint arXiv:2506.02175, 2025.
- [51] Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun Araki, Arsalan Gundroo, Bingqing Wang, Rakesh Menon, Md Parvez, and Zhe Feng. DelucionQA: Detecting hallucinations in domain-specific question answering. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 822–835, Singapore, December

- 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.59. URL https://aclanthology.org/2023.findings-emnlp.59/.

- [52] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [53] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

- [54] Vighnesh Subramaniam, Yilun Du, Joshua B. Tenenbaum, Antonio Torralba, Shuang Li, and Igor Mordatch. Multiagent finetuning: Self improvement with diverse reasoning chains. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=JtGPIZpOrz.
- [55] Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory. arXiv preprint arXiv:2504.07952, 2025.
- [56] Qwen Team. Qwen3, April 2025. URL https://qwenlm.github.io/blog/qwen3/.
- [57] Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv:2402.10176,

- 2024.

- [58] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.
- [59] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Huai hsin Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. ArXiv, abs/2203.11171,

- 2022.

- [60] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. Learning to filter context for retrieval-augmented generation. arXiv preprint arXiv:2311.08377, 2023.
- [61] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [62] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 3(4), 2023.
- [63] xAI. Grok 3 beta — the age of reasoning agents, 2025. URL https://x.ai/news/grok-3.
- [64] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.
- [65] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [66] Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Yihao Feng, Le Xue, Rithesh Murthy, Zeyuan Chen, Jianguo Zhang, Devansh Arpit, Ran Xu, Phil L Mui, Haiquan Wang, Caiming Xiong, and Silvio Savarese. Retroformer: Retrospective large language agents with policy gradient optimization. ArXiv, abs/2308.02151, 2023. URL https://api.semanticscholar.org/CorpusID: 260611249.
- [67] Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, and Wenhu Chen. Acecoder: Acing coder rl via automated test-case synthesis. arXiv preprint arXiv:2502.01718, 2025.
- [68] Junyu Zhang, Runpei Dong, Han Wang, Xuying Ning, Haoran Geng, Peihao Li, Xialin He, Yutong Bai, Jitendra Malik, Saurabh Gupta, and Huan Zhang. Alphaone: Reasoning models thinking slow and fast at test time. 2025. URL https://arxiv.org/abs/2505.24863.
- [69] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Gaetan Lin, Y. Liu, and Gao Huang. Expel: Llm agents are experiential learners. In AAAI Conference on Artificial Intelligence, 2023. URL https://api.semanticscholar.org/CorpusID:261048772.
- [70] Li Zhong, Zilong Wang, and Jingbo Shang. Ldb: A large language model debugger via verifying runtime execution step-by-step. arXiv preprint arXiv:2402.16906, 2024.

- A Lists of Prompts 19

- A.1 Planner Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.2 Dynamic Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.3 Judge Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.4 Verifier Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.5 Reasoning Segmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- B Qualitative Examples 21
- C External Retrieval Corpus for Code 33
- D Additional Analysis 33

- D.1 Patterns in Xolver Reasoning Traces . . . . . . . . . . . . . . . . . . . . . . . . . 33
- D.2 Performance Variance Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- D.3 Impact of Data-Shuffling in Xolver (+) Performance . . . . . . . . . . . . . . . . . 35
- D.4 List of Roles of Selected by Dynamic Agents . . . . . . . . . . . . . . . . . . . . 35

## A Lists of Prompts

This section provides the list of prompts for planning, dynamic, judge, verifier and reasoning segmentation we have used in the experimental period. These are crucial to ensure the reproducibility [23] of the framework Xolver.

### A.1 Planner Agent

Prompt for PLANNER AGENT You are a planner to solve a {coding/math} problem. Here is the problem for which you have to plan: {problem_dict[query[‘problem_id’]][‘description’]}

First draft required strictly greater than {m} specialized roles to solve the problem collaboratively with reasoning behind your draft of each role.

Then select the highly influential {m} roles by re-checking the reasoning behind your selection and assign them to each agent to solve the problem.

### A.2 Dynamic Agent

Prompt for DYNAMIC AGENT

You are a {role}. Your task is to solve a {coding/math} problem. Here is the problem that you have to solve: problem_dict[query[‘problem_id’]][‘description’]

——————————————————————————————————————

If external retrieval: You were also given a couple of similar problems to the problem above along with their reasoning and solutions to aid you in solving the problem at hand. Here are the similar problems you were given:

retrieved_dict[query[‘problem_id’]][‘retrieval_text’] If self-retrieval: Further, recall a relevant and distinct problem (different from the problem mentioned above) along with its reasoning and solution.

—————————————————————————————————————And here was your original response: query[[‘role’]][‘original_thought’,‘original_response’]

—————————————————————————————————————If iteration i ≥ 1 (i.e., DS is not empty): Also here is the leading responses with execution results from the response store: response_dict[‘role’,‘thought’,‘response’,‘score’]

—————————————————————————————————————If coding task:

Think carefully about where you went wrong, relating with responses in the response store. Then, try to fix the solution producing a thought later reply with a {Python} solution to be executed and judged again.

Make sure to wrap your code in ```python ``` block and Markdown delimiters, and include exactly one block of code with the entire solution (in the final code step).

—————————————————————————————————————If math task:

Think carefully about where you went wrong, relating with responses in the response store. Then, try to fix the solution producing a thought later reply with a solution to be executed and judged again. You can integrate a {Python} tool to execute the calculations after replying your solution if required.

Make sure to wrap your final answer in \boxed{} block with the entire solution (in the final answer step).

### A.3 Judge Agent

Prompt for JUDGE AGENT

You are a judge. Your task is to judge the candidate solution of a {coding/math} problem. Here is the problem for which the candidate solution you have to judge: problem_dict[query[‘problem_id’]][‘description’]

—————————————————————————————————————If coding task: And here is the candidate response along with test cases against which to judge: query[[‘candidate_role’]][‘candidate_thought’,‘candidate_response’,‘test_case’]

Please produce a score (based on the number of test cases passed) with reasoning behind your judgement of the candidate solution to the problem.

—————————————————————————————————————If math task: And here is the candidate response which to judge: query[[‘candidate_role’]][‘candidate_thought’,‘candidate_response’

Please produce a score (if the response is correct, it should be 1 otherwise should be 0) with reasoning behind your judgement of the candidate solution to the problem.

### A.4 Verifier Agent

Prompt for VERIFIER AGENT

Your are an answer extractor. Your task is to extract answer from the response to a {coding/math} problem. Here is the response for which the answer you have to extract: response_dict[query{‘role’}]{‘thought’, ‘response’, ‘score’}

—————————————————————————————————————If coding task: Please extract the answer from inside ```python ``` block from the response.

—————————————————————————————————————If math task: Please extract the answer from inside \boxed{} block from the response.

### A.5 Reasoning Segmentation

Prompt for INITIAL REASONING SEGMENTATION

Below is a chain of thought for solving a question. Figure out what are the different reasoning patterns that are used like problem rephrasing, new approach, self-evaluation, verification, backtracking, subgoal setup, solving a simpler problem first, and more. Then your task is to segment the entire chain of thought into different reasoning patterns. Rewrite the chain of thought in the following format:

<pattern> pattern name </pattern> <content> the entire text that corresponds to the pattern </content>

Thought: {thoughts}

Prompt for FINAL REASONING SEGMENTATION

Below is a thought for solving a question. For the segment between the <unannotated> and </unannotated> tags, figure out what is reasoning pattern used in that segment like problem rephrasing, new approach, self-evaluation, verification, backtracking, subgoal setup, solving a simpler problem first, or something else. Then your task is to identify the reasoning pattern used in the unannotated segment. Rewrite the unannotated segment in the following format:

<content> The text within the unannotated segment that corresponds to the pattern. </content> <reasoning> Reasoning for what the pattern should be for the content </reasoning> <pattern> *single* pattern name </pattern>

Thought: {thoughts}

## B Qualitative Examples

The subsequent section presents some qualitative examples–agentic interaction breakdown on how Xolver can approach and solve a problem of math and coding tasks when it encounters.

Example for CORRECT MATH TASK GREEN : CORRECT ANSWER. RED : WRONG ANSWER.

Query: Bob is in charge of doing laundry for a large hotel. Each room has two sheets, one comforter, twice as many pillow cases as sheets and twice as many towels as pillow cases. How many pieces of laundry are there in 80 rooms? Make sure to wrap your final answer in \boxed{} block with the entire solution (in the final answer step).

###### Iteration 1

Mathematical Modeler: Each room contains 2 sheets, 1 comforter, 2 pillow cases (which is twice the number of sheets), and 4 towels (which is twice the number of pillow cases). Hence, the total number of laundry items in each room is: 2 + 1 + 2 + 4 = 9 pieces of laundry. When considering 80 rooms, the total number of pieces of laundry is: 80 × 9 = 720 pieces of laundry. Thus, the final answer is:

|720|
|---|

.

Judge: The candidate’s solution contains an error in calculating the number of pillow cases and towels per room. Here’s the breakdown: Sheets: 2 (correct). Comforter: 1 (correct). Pillow Cases: The problem states “twice as many pillow cases as sheets.” Since there are 2 sheets, pillow cases should be 2 × 2 = 4. The candidate incorrectly used 2 pillow cases (half the required amount). Towels: The problem states “twice as many towels as pillow cases.” With the correct pillow case count (4), towels should be 2 × 4 = 8. The candidate used 4 towels (based on their incorrect pillow case count of 2). The correct per-room total is: 2 (sheets) + 1 (comforter) + 4 (pillow cases) + 8 (towels) = 15 items. The total for 80 rooms is: 80 × 15 = 1200.

The candidate’s answer is incorrect due to miscalculating pillow cases and towels.

- Score: 0.

Numerical Analyst: For each room, Bob needs to wash 2 sheets, 1 comforter, 4 pillow cases (since 2 × 2 = 4), and 8 towels (since 4 × 2 = 8). Therefore, for each room, the total amount of laundry Bob needs to wash is: 2 + 1 + 4 + 8 = 15 pieces of laundry. For 80 rooms, the total number of pieces of laundry Bob must wash is: 80 × 15 = 1200 pieces of laundry.

|1200|
|---|

Thus, the final answer is:

.

Judge: The candidate correctly interprets the problem’s relationships: sheets: 2 (given), comforter: 1 (given), pillow cases: twice the sheets, 2 × 2 = 4 (correct), towels: twice the pillow cases, 2 × 4 = 8 (correct). Total per room: 2 + 1 + 4 + 8 = 15 (correct). Total for 80 rooms: 80 × 15 = 1200 (correct). All steps align with the problem’s specifications, and the final answer is accurate. The candidate’s solution is error-free.

- Score: 1.

Mathematician: Every room is equipped with 2 sheets, 1 comforter, 4 pillow cases (calculated as 2 × 2), and 8 towels (since 4 × 2 = 8). Summing these, the total number of laundry items for one room is: 2 + 4 + 8 = 14 pieces of laundry. Multiplying by the number of rooms, the total quantity for 80 rooms becomes:

|1120|
|---|

.

###### 80 × 14 =

Judge: The candidate incorrectly omitted the comforter in their final sum. Here’s the breakdown: sheets: 2 (correct), comforter: 1 (given but not included in the candidate’s calculation), pillow cases: 2 × 2 = 4 (correct), towels: 2 × 4 = 8 (correct). The candidate’s total per room was 2 + 4 + 8 = 14, but the correct total should include the comforter: 2 + 1 + 4 + 8 = 15. Thus, the total for 80 rooms should be 80 × 15 = 1200, not 1120. The error stems from excluding the comforter in the summation.

- Score: 0.

DS1 rank: [Numerical Analyst, Mathematical Modeler, Mathematician]

Iteration 2

Mathematical Modeler: Each room contains 2 sheets, 1 comforter, 2 pillow cases (which is twice the number of sheets), and 4 towels (which is twice the number of pillow cases). Hence, the total number of laundry items in each room is: 2 + 1 + 4 + 8 = 15 pieces of laundry. When considering 80 rooms, the total number of pieces of laundry is: 80 × 15 = 1200 pieces of laundry. Thus, the final answer is:

|1200|
|---|

. This confirms the accuracy of the revised calculation.

Judge: The candidate’s final answer is correct, and the mathematical calculations align with the problem’s requirements: 1. Sheets: 2 (correct). 2. Comforter: 1 (included in the total). 3. Pillow Cases: The problem states “twice as many pillow cases as sheets.” Since there are 2 sheets, this should be 2 × 2 = 4. The candidate correctly uses 4 pillow cases in their total calculation (despite a typographical inconsistency in the explanation, where they initially wrote “2 pillow cases”). 4. Towels: “Twice as many towels as pillow cases” gives 2 × 4 = 8 (correct).

The candidate’s total per room is 2 + 1 + 4 + 8 = 15 (correct), leading to 80 × 15 = 1200 (correct).

While the explanation contains a minor inconsistency in wording (“2 pillow cases” instead of 4), the numerical calculations are accurate. Since the final answer is correct and the mathematical steps resolve the inconsistency, the solution is valid.

- Score: 1.

Numerical Analyst: Based on the information provided, we know that each room includes 2 sheets, 1 comforter, 4 pillow cases (computed as 2 × 2), and 8 towels (since 4 × 2 = 8). Adding these together, the total number of laundry items required for one room is: 2 + 1 + 4 + 8 = 15 pieces. Multiplying by the total number of rooms, the overall amount of laundry for 80 rooms is: 80 × 15 =

|1200|
|---|

.

Although the other agents correctly determined the number of laundry items required for each room, there was an error in the concluding step–calculate the total number of laundry pieces.

Judge: The candidate’s calculations and final answer are correct: sheets: 2 (given), comforter: 1 (given), pillow cases: 2 × 2 = 4 (twice the sheets), towels: 2 × 4 = 8 (twice the pillow cases), total per room: 2 + 1 + 4 + 8 = 15 (correct), total for 80 rooms: 80 × 15 = 1200 (correct).

Score: 1. Mathematician: According to the provided details, every room is equipped with 2 sheets, 1 comforter,

- 4 pillow cases (calculated as 2 × 2), and 8 towels (since 4 × 2 = 8). Summing these, the total number of laundry items for one room is:

- 1 + 4 + 8 = 13 pieces of laundry. Multiplying by the number of rooms, the total quantity for 80 rooms becomes: 80 × 13 =

|1040|
|---|

.

Judge: The candidate incorrectly omitted the sheets in their final sum. Here’s the breakdown: 1. Sheets: 2 (given but not included in the candidate’s calculation). 2. Comforter: 1 (included). 3. Pillow Cases: 2 × 2 = 4 (correct). 4. Towels: 2 × 4 = 8 (correct). The candidate’s total per room was

- 1+4+8 = 13, but the correct total should include the sheets: 2+1+4+8 = 15. Thus, the total for 80 rooms should be 80×15 = 1200, not 1040. The error stems from excluding the sheets in the summation.

- Score: 0.

DS2 rank: [Mathematical Modeler, Numerical Analyst, Mathematician]

Iteration 3

Mathematical Modeler: From the details provided, it is clear that each room contains 2 sheets, 1 comforter, 4 pillow cases (as 2 × 2 = 4), and 8 towels (calculated as 4 × 2 = 8). Adding these together, the total number of laundry items per room amounts to: 2 + 1 + 4 + 8 = 15 items. Consequently, for 80 rooms, the total number of laundry pieces becomes: 80 × 15 =

|1200|
|---|

. I agree with the other agent’s corrected calculation.

Judge: The candidate’s solution is mathematically accurate and complete: sheets: 2 (correct), comforter: 1 (correct), pillow cases: 2 × 2 = 4 (twice the sheets), towels: 2 × 4 = 8 (twice the pillow cases), total per room: 2+1+4+8 = 15 (correct), total for 80 rooms: 80×15 = 1200 (correct).

- Score: 1.

Numerical Analyst: Given the information, we know that each room includes 2 sheets, 1 comforter, 4 pillow cases (computed as 2 × 2), and 8 towels (since 4 × 2 = 8). Adding these together, the total number of laundry items required for one room is:

- 2 + 1 + 4 + 8 = 15 pieces. Multiplying by the total number of rooms, the overall amount of laundry for 80 rooms is:

|1200|
|---|

.

80 × 15 =

Although the other agent correctly determined the number of laundry items required for each room, there was an error in the concluding step. Specifically, they incorrectly used 80 × 13 instead of the appropriate multiplication of 80 × 15 to calculate the total number of laundry pieces.

Judge: The candidate’s solution is fully correct: sheets: 2 (given), comforter: 1 (given), pillow cases: 2 × 2 = 4 (twice the sheets), towels: 2 × 4 = 8 (twice the pillow cases), total per room: 2 + 1 + 4 + 8 = 15 (correct), total for 80 rooms: 80 × 15 = 1200 (correct). All steps align with the problem’s specifications, and the final answer is accurate. No errors were made.

Score: 1.

Mathematician: According to the given details, each room contains 2 sheets, 1 comforter, 4 pillow cases (calculated as 2 × 2), and 8 towels (determined by 4 × 2). Summing these amounts, the total number of laundry items per room is: 2 + 1 + 4 + 8 = 15 pieces of laundry. To find the total for 80 rooms, we multiply: 80 × 15 =

|1200|
|---|

. I concur with the other agents’ results and the computations they presented.

Judge: The candidate’s solution is fully correct and aligns with the problem’s requirements: sheets: 2 (given), comforter: 1 (given), pillow cases: 2 × 2 = 4 (twice the sheets), towels: 2 × 4 = 8 (twice the pillow cases), total per room: 2 + 1 + 4 + 8 = 15 (correct), total for 80 rooms: 80 × 15 = 1200 (correct). All calculations are accurate, and the final answer matches the problem’s constraints.

Score: 1.

DS3 rank: [Mathematician, Mathematical Modeler, Numerical Analyst]

Example for CORRECT CODING TASK GREEN : CORRECT SOLUTION. RED : WRONG SOLUTION.

Query: Yarik is a big fan of many kinds of music. But Yarik loves not only listening to music but also writing it. He likes electronic music most of all, so he has created his own system of music notes, which, in his opinion, is best for it.

Since Yarik also likes informatics, in his system notes are denoted by integers of 2k, where k ≥ 1 — a positive integer. But, as you know, you can’t use just notes to write music, so Yarik uses combinations of two notes. The combination of two notes (a, b), where a = 2k and b = 2l, he denotes by the integer ab.

For example, if a = 8 = 23, b = 4 = 22, then the combination (a, b) is denoted by the integer ab = 84 = 4096. Note that different combinations can have the same notation, e.g., the combination (64, 2) is also denoted by the integer 4096 = 642.

Yarik has already chosen n notes that he wants to use in his new melody. However, since their integers can be very large, he has written them down as an array a of length n, then the note i is bi = 2ai. The integers in array a can be repeated.

The melody will consist of several combinations of two notes. Yarik was wondering how many pairs of notes bi, bj (i < j) exist such that the combination (bi, bj) is equal to the combination (bj, bi). In other words, he wants to count the number of pairs (i, j) (i < j) such that bbij = bbji. Help him find the number of such pairs.

Input The first line of the input contains one integer t (1 ≤ t ≤ 104) — the number of test cases. The first line of each test case contains one integer n (1 ≤ n ≤ 2 · 105) — the length of the arrays. The next line contains n integers a1, a2, . . . , an (1 ≤ ai ≤ 109) — array a. It is guaranteed that the sum of n over all test cases does not exceed 2 · 105. Output For each test case, output the number of pairs that satisfy the given condition. Sample Input 1:

- 5

- 1
- 2 4
- 3 1 3 2

- 2 1000 1000
- 3

- 1 1 1 19
- 2 4 1 6 2 8 5 4 2 10 5 10 8 7 4 3 2 6 10 Sample Output 1:

- 0

2

- 1

- 3 19

Make sure to wrap your code in ```python ``` block and Markdown delimiters, and include exactly one block of code with the entire solution (in the final code step).

Iteration 1 Implementation Specialist: To solve this problem, we need to determine the number of pairs of notes (bi, bj) such that bbij = bbji. The notes are given as bi = 2ai, where ai is an integer. The key insight is to recognize the conditions under which the equation bbij = bbji holds. This equation can be broken down into two main cases. The first case is when the values are the same, meaning ai = aj. In this situation, the equation trivially holds because both sides will be identical. The number of such pairs is given by the combination formula C(n, 2) = n(n2−1) for each group of identical values. The second case occurs when the values are different but satisfy a special condition: the equation holds when one value is 1 and the other is 2. This is derived from the mathematical simplification of the equation ai · 2aj = aj · 2ai, which only holds true for ai = 1 and aj = 2 (or vice versa) when ai ̸= aj.

```python import sys from collections import Counter

def main(): # Read all input tokens at once x = sys.stdin.read().split() i = 0 # Index to track position in input list

t = int(x[i]) # Number of test cases i += 1

for _ in range(t): n = int(x[i]) # Number of elements in this test case i += 1

# Extract the list of integers for current test case l = list(map(int, x[i:i+n])) i += n

# Count frequencies of each element c = Counter(l)

s = 0 # Sum of pairs # Sum pairs of identical elements: count choose 2 for v in c.values():

s += v * (v - 1) // 2

# Add pairs formed by one element ’1’ and one element ’2’ s += c.get(1, 0) * c.get(2, 0)

print(s) if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes 5 test cases. Score: 5.

Programmer: To solve this problem, we need to determine the number of pairs of notes such that the combination of two notes (bi, bj) results in the same integer as the combination (bj, bi). The notes are given as bi = 2ai, and we are looking for pairs (i, j) where bbij = bbji. The condition bbij = bbji can be simplified to checking if ai · 2aj = aj · 2ai, which holds true in two scenarios: first, when ai = aj, producing identical pairs; and second, when one value is 1 and the other is 2, creating valid cross pairs. To count the identical pairs, for each unique value x in the array, we use the combination formula

count

2 , where count is the frequency of x. For cross pairs, which occur specifically between values 1 and 2, the number of such pairs is calculated as the product of the counts of 1 and 2.

```python import sys from collections import defaultdict

def main(): x = sys.stdin.read().split() # Read all input tokens i = 0 # Input index pointer

t = int(x[i]) # Number of test cases i += 1

for _ in range(t): n = int(x[i]) # Number of elements i += 1

l = list(map(int, x[i:i+n])) # List of integers for this test case i += n

f = defaultdict(int) # Frequency dictionary # Count occurrences of each number for v in l:

f[v] += 1 s = 0 # Count pairs of identical elements # Sum pairs from frequencies (nC2) for v in f.values():

s += v * (v - 1) // 2 # Count pairs formed by one ’1’ and one ’2’

c = f.get(1, 0) * f.get(2, 0) print(s + c)

if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes 5 test cases. Score: 5.

Data Structure Specialist: To solve this problem, we need to determine the number of pairs of music notes, represented by integers of the form 2k, such that the combination of two notes (bi, bj) results in the same value as (bj, bi). In other words, we seek pairs (i, j) where bbij = bbji. By applying logarithms and using the properties of exponents, for notes defined as bi = 2ai and bj = 2aj, the condition simplifies to ai · 2aj = aj · 2ai. Two key observations arise from this: first, if ai = aj, the equation holds true, meaning all pairs of identical values satisfy the condition; second, the equation also holds when one value is 1 and the other is 2, since 1 · 22 = 2 · 21 simplifies to 4 = 4. The algorithm to find the number of such pairs involves two parts. For pairs with the same value v, we count the number of pairs using the combination formula C(n, 2) = n(n2−1), where n is the frequency of v. For cross-value pairs where one note corresponds to 1 and the other to 2, the number of valid pairs is simply the product of their counts.

```python import sys from collections import Counter

def main(): x = sys.stdin.read().split() # Read all input tokens at once i = 0 # Input index pointer

t = int(x[i]) # Number of test cases i += 1

for _ in range(t): n = int(x[i]) # Number of elements i += 1

l = list(map(int, x[i:i+n])) # List of integers for this test case i += n

c = Counter(l) # Frequency counter

c1 = c.get(1, 0) # Count of 1’s c2 = c.get(2, 0) # Count of 2’s

x = 0 # Combine counts of 1 and 2, calculate pairs among them m = c1 + c2 x += m * (m - 1) // 2 # Calculate pairs for other numbers except 1 and 2 for k in c:

if k not in (1, 2): v = c[k] x += v * (v - 1) // 2

print(x) if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes 5 test cases.

- Score: 5.

DS1 rank: [Implementation Specialist, Programmer, Data Structure Specialist]

Iteration 2 Implementation Specialist: : After reviewing the solutions provided by other agents, I agree that to

solve this problem, we need to determine the number of pairs of notes (bi, bj) such that bbij = bbji, where each note is given by bi = 2ai. We are interested in counting pairs (i, j) with i < j that satisfy this condition. By applying properties of exponents and substituting bi = 2ai and bj = 2aj, the

condition simplifies to checking when 2aai

i

= 2

aj

aj . There are two key observations: first, if ai = aj, the condition is automatically met; second, pairs where one value is 1 and the other is 2 also satisfy the condition, since 211 = 2 and 222 = 2. The algorithm involves counting the frequency of each ai, then calculating the number of valid pairs for identical values using the combination formula f2 , where f is the frequency of the value. Finally, to account for the special case of pairs between 1 and 2, we add the product of their counts to the total.

```python import sys from collections import Counter

def main(): x = sys.stdin.read().split() # Read all input tokens at once i = 0 # Index pointer

t = int(x[i]) # Number of test cases i += 1

for _ in range(t): n = int(x[i]) # Number of elements in this test case i += 1

l = list(map(int, x[i:i+n])) # List of elements i += n

c = Counter(l) # Frequency count s = 0 # Sum of pairs # Calculate pairs from identical elements: nC2 for each count for v in c.values():

s += v * (v - 1) // 2

# Add pairs formed by one ’1’ and one ’2’ s += c.get(1, 0) * c.get(2, 0)

print(s) if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes 6 test cases.

- Score: 6.

Programmer: I agree with the other agents that, to solve this problem, we need to determine the number of valid pairs of notes such that the combination of two notes (bi, bj) results in the same integer as the combination (bj, bi). This is equivalent to finding pairs (i, j) where bbij = bbji. The key insight is that this equation holds under two conditions: first, when the exponents ai and aj are equal, the pairs

- (bi, bj) will always satisfy the equation; second, when one exponent is 1 and the other is 2, the equation also holds. To solve the problem, the approach involves three steps. First, use a frequency counter to count the occurrences of each value in the array. Second, for each value x, compute the number of valid

pairs among identical values using the combination formula f2 , where f is the frequency of x. Third, calculate the number of specific pairs where one value is 1 and the other is 2 by multiplying their counts.

```python import sys from collections import Counter

def main(): t = int(sys.stdin.readline()) # Number of test cases for _ in range(t):

n = int(sys.stdin.readline().strip()) # Number of elements l = list(map(int, sys.stdin.readline().split())) # List of integers

c = Counter(l) # Frequency counter s = 0 # Sum of pairs # Calculate pairs among identical elements: nC2 for each frequency for v in c.values():

s += v * (v - 1) // 2

# Add pairs formed by one ’1’ and one ’2’ s += c.get(1, 0) * c.get(2, 0)

print(s) if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes 5 test cases. Score: 5.

Data Structure Specialist: To solve this problem, we need to determine the number of valid pairs of notes such that the combination of two notes (bi, bj) results in the same integer as the combination

- (bj, bi). This is equivalent to finding pairs (i, j) where bbij = bbji. The key insight comes from the

equation bbij = bbji, where bi = 2ai and bj = 2aj. This equation simplifies to ai · 2aj = aj · 2ai. To solve this efficiently, we follow several steps. First, for each ai, we perform prime factorization, expressing it as 2y × x, where x is the odd part of ai and y is the exponent of 2 in its factorization. Next, we generate a key for each ai in the form (x, y − ai). This key is designed such that if two numbers ai and aj share the same key, they satisfy the equation ai · 2aj = aj · 2ai. We then use a frequency dictionary to count the occurrences of each key. Finally, for each key, we compute the number

of valid pairs using the combination formula f2 , where f is the frequency of that key. ```python import sys from collections import defaultdict def main():

x = sys.stdin.read().split() # Read all input tokens at once i = 0 # Input index pointer

t = int(x[i]) # Number of test cases i += 1

for _ in range(t): n = int(x[i]) # Number of elements in this test case i += 1

l = list(map(int, x[i:i+n])) # List of integers i += n

f = defaultdict(int) # Frequency dictionary for v in l:

- a = v
- b = 0 # Count how many times 2 divides the number while a % 2 == 0:

- a //= 2
- b += 1

# Key is tuple of (odd part, difference of exponents) k = (a, b - v) f[k] += 1

s = 0 # Result accumulator # Count pairs for each unique key using nC2 formula for c in f.values():

s += c * (c - 1) // 2 print(s)

if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes all 10 test cases with correct result. Score: 10.

DS2 rank: [Data Structure Specialist, Implementation Specialist, Programmer]

Iteration 3 Implementation Specialist: I concur with the other agent that to tackle this problem, our goal is to find the number of valid pairs of notes (bi, bj) such that raising one note to the power of the other yields the same result when the order is reversed; that is, pairs (i, j) satisfying bbij = bbji. Given that each note can be expressed as bi = 2ai, the problem reduces to analyzing the equation ai · 2aj = aj · 2ai. To approach this effectively, we begin by decomposing each exponent ai into its prime factors, specifically writing it as 2y × x, where x is the odd component and y indicates the power of two within the factorization. Using this factorization, we create a unique key for each ai defined by the pair (x, y − ai). This construction ensures that any two exponents sharing the same key satisfy the original equation. After generating these keys for all elements, we count their frequencies using a dictionary. Finally, to determine the number of valid pairs, we apply the combination formula f2 for each key, where f is the frequency of that key, summing these results to obtain the total count of valid pairs.

```python import sys from collections import defaultdict

def main(): d = sys.stdin.read().strip().split() # Read and split all input t = int(d[0]) # Number of test cases

p = 1 # Current position in input list for _ in range(t):

n = int(d[p]) # Number of elements in this test case p += 1

l = list(map(int, d[p:p+n])) # Current list of elements p += n

f = defaultdict(int) # Frequency dictionary for v in l:

o = v # Original value c = 0 # Count of factors 2

# Count how many times 2 divides the value while v % 2 == 0:

v //= 2 c += 1

# Key: (odd part, count_twos - original_value) k = (v, c - o) f[k] += 1

s = 0 # Result accumulator # Calculate pairs for each unique key using nC2 for x in f.values():

s += x * (x - 1) // 2 print(s)

if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes all 10 test cases with correct result. Score: 10.

Programmer: I agree with the other agent that to address this problem, we aim to find the number of valid pairs of notes (bi, bj) such that the expression bbij equals bbji. In other words, we seek pairs (i, j) where the exponentiation relation bbij = bbji holds true. Since each note is defined as bi = 2ai, the equation simplifies to ai · 2aj = aj · 2ai. To efficiently solve this, we first factorize each exponent ai into the form 2y × x, where x represents the odd component and y is the power of two in its prime factorization. Using this decomposition, we construct a key for each ai as (x, y − ai). This key is crafted so that if two exponents ai and aj share the same key, they satisfy the equality condition. Next, we tally the frequency of each key using a dictionary or hash map. Finally, the number of valid pairs corresponding to each key is computed using the combination formula

f 2 , where f denotes the frequency of that key. Summing these values yields the total count of valid pairs.

```python import sys from collections import defaultdict

def main(): d = sys.stdin.read().strip().split() # Read all input tokens t = int(d[0]) # Number of test cases p = 1 # Position pointer in input list

for _ in range(t): n = int(d[p]) # Number of elements

p += 1 l = list(map(int, d[p:p+n])) # List of elements p += n f = defaultdict(int) # Frequency dictionary for v in l:

- b = v # Base value
- c = 0 # Count of factors 2

# Count how many times 2 divides b while b % 2 == 0:

b //= 2 c += 1

# Key is tuple: (odd part, power difference) k = (b, c - v) f[k] += 1

s = 0 # Result accumulator # Count pairs for each frequency using nC2 formula for x in f.values():

s += x * (x - 1) // 2 print(s)

if __name__ == "__main__":

main() ```

Judge: There are 10 test cases. The candidate passes all 10 test cases with correct result. Score: 10.

Data Structure Specialist: To solve this problem, the goal is to identify the number of valid pairs of notes (bi, bj) for which the value of bbij equals that of bbji. This translates to finding pairs (i, j) satisfying the equation bbij = bbji. Given that each note bi can be expressed as 2ai, the equation simplifies to ai · 2aj = aj · 2ai. To approach this efficiently, each exponent ai is factorized into the form 2y × x, where x denotes the odd part of ai and y is the exponent of 2 in its prime factorization. Based on this factorization, we construct a unique key (x, y − ai) for each ai. This key is carefully designed so that two numbers sharing the same key satisfy the given equality condition. Once the keys are generated, a frequency map is used to count how many times each key appears. Finally, for each key, the number of valid pairs is calculated using the combination formula f2 , where f represents the frequency of that key. Summing these counts across all keys gives the total number of valid pairs.

```python import sys from collections import defaultdict

def main(): d = sys.stdin.read().split() # Read all input tokens i = 1 # Index pointer after reading t

t = int(d[0]) # Number of test cases for _ in range(t):

n = int(d[i]) # Number of elements i += 1

l = list(map(int, d[i:i+n])) # Current list of elements

i += n f = defaultdict(int) # Frequency dictionary for v in l:

x = v # Current value p = 0 # Count of factor 2

# Count how many times 2 divides x while x % 2 == 0:

x //= 2 p += 1

k = (x, p) # Key as (odd part, power of two) f[k] += 1

s = 0 # Result accumulator # Count pairs for each unique key using nC2 formula for c in f.values():

s += c * (c - 1) // 2 print(s)

if __name__ == "__main__":

main() ```

While the previous codes are correct for the given problem constraints, it implicitly assumes that only pairs of equal values or (1, 2) satisfy the equation later searching iteratively. However, explicitly grouping all valid pairs using a mathematical key, making it more robust and efficient for all scenarios, including possible edge cases not covered by the first approach.

Judge: There are 10 test cases. The candidate passes all 10 test cases with correct result. Score: 10.

DS3 rank: [Data Structure Specialist, Programmer, Implementation Specialist]

## C External Retrieval Corpus for Code

As shown in Figure 11, our coding external retrieval corpus has 156 fine grained algorithmic tutorials with relevant problems along with their solutions collections from 10 different categories (algebrea, data structure, dynamic programming, string processing, linear algebrea, combinatorics, numerical methods, geometry, graphs and miscellaneous).

## D Additional Analysis

This section presents our multiple runs’ results which we tested and list of roles which LLMs selected during the experiments.

### D.1 Patterns in Xolver Reasoning Traces

We employ the QWQ-32B model for the initial segmentation of the chain of thought, guided by the first prompt illustrated in Appendix A.5. Subsequently, for segments that remain unannotated, a second segmentation pass is performed using the second prompt depicted in Appendix A.5. Reasoning patterns are identified by extracting content enclosed within the <pattern> tags. Verification and reasoning patterns are combined into a unified self-evaluation category. If the model assigns multiple patterns to a single segment, that segment is excluded due to ambiguity in pattern classification. For

###### Coding External Retrieval Corpus Count per Subject Category

45

40

29

30

25

Count

20

14

12

10

10

10

6

4

1

0

AlgebraDataStructureDynamicProgrammingStringProcessingLinearAlgebraCombinatoricsNumericalMethods Geometry GraphsMiscellaneous

Figure 11: Coding External Retrieval Corpus Count per Subject Category.

each generated output, we calculate the proportion of occurrences of each pattern relative to the total patterns present, resulting in a frequency vector representing pattern distribution per generation. In examining the relationship between pattern usage and problem difficulty, we compute the mean frequencies separately for correct and incorrect generations and assess significance through a t-test. To evaluate pattern prevalence on a per-problem basis, a binary matrix is constructed where rows correspond to problems and entries indicate whether a pattern is more common in correct (1) or incorrect (0) solutions. The statistical significance of these findings is evaluated using a binomial test.

Reasoning Pattern Correct Solutions Incorrect Solutions

Easy → Medium Medium → High Easy → Medium Medium → High

- (a) OpenCodeReasoning Self-Evaluation (↓) 0.39 → 0.37 0.37 → 0.34 0.36 → 0.37 0.34 → 0.31 New Approach (↑) 0.16 → 0.20 0.20 → 0.23 0.16 → 0.22 0.22 → 0.25 Problem Rephrasing (↓↑) 0.21 → 0.20 0.20 → 0.20 0.21 → 0.22 0.22 → 0.23 Subgoal Setup (↓) 0.13 → 0.12 0.12 → 0.10 0.13 → 0.10 0.10 → 0.10

- (b) Xolver

Self-Evaluation (↑) 0.35 → 0.38 0.38 → 0.40 0.35 → 0.37 0.32 → 0.35 New Approach (↑) 0.18 → 0.21 0.21 → 0.24 0.17 → 0.24 0.24 → 0.26 Problem Rephrasing (↓↑) 0.20 → 0.17 0.18 → 0.18 0.23 → 0.24 0.24 → 0.25 Subgoal Setup (↓↑) 0.14 → 0.13 0.13 → 0.11 0.11 → 0.12 0.11 → 0.11

Table 3: Demonstrating how the frequency of major reasoning pattern changes as problem difficulty increases. Green indicates statistically significant increases and red indicates significant decreases (p < 0.05). Gray boxes highlight opposing trends between OpenCodeReasoning (decrease) and Xolver (increase). Direction arrows indicate the expected trend direction: ↑ = increase, ↓ = decrease, ↓↑ = mixed trend (minor decrease then elevated recovery), ↕ = fluctuating trend (major decrease then recovery). While solving problems, OpenCodeReasoning struggles at Self-Evaluation and Subgoal Setup whereas Xolver overcomes it with increasing Self-Evaluation in both correct and incorrect solutions and elevated recovery in Subgoal Setup in incorrect solutions. Both OpenCodeReasoning and Xolver adapts New Approach while struggles at Problem Rephrasing.

### D.2 Performance Variance Statistics

In this experiment on the variance of Xolver performance, we tested Xolver against multiple runs (16 for AIME ’24 and 32 for AIME ’25 and LiveCodeBench) in AIME and LiveCodeBench dataset. Results shows in Table 4 that it has small scale performance change with multiple runs which is a strong sign on the robustness of Xolver.

Model Appr. AIME ’24 AIME ’25 LiveCodeBench (v5) o3-mini-medium Xolver (-) 87.2 ± 1.2 85.1 ± 1.3 79.6 ± 1.0

Xolver (+) 93.8 ± 0.3 89.4 ± 0.7 87.3 ± 0.4 QWQ-32B Xolver (-) 89.9 ± 0.8 79.5 ± 1.1 76.2 ± 0.9

Xolver (+) 93.6 ± 0.2 82.7 ± 0.8 79.2 ± 0.5 o3-mini-high Xolver (+) 94.4 ± 0.6 93.7 ± 0.5 91.6 ± 0.3

Table 4: Xolver average performance with multiple trials.

- D.3 Impact of Data-Shuffling in Xolver (+) Performance

During this experiment on the impact of shuffling data on Xolver performance, we randomly shuffled the test instances and conducted the experiment with 5 runs. Results shows in Table 5 that Xolver has limited performance change (STD ∼ 1) with shuffling data–a strong sign on the robustness of the framework.

Mean ± STD (With only 5 Runs, all STD ∼ 1) Model GSM8K AIME ’24 AIME ’25 MATH-500 LiveCodeBench (v5)

o3-mini-medium 97.6 ± 1.3 92.2 ± 0.4 91.0 ± 0.3 98.3 ± 0.6 90.9 ± 1.1 QWQ-32B 97.2 ± 0.6 93.7 ± 0.5 82.7 ± 2.0 95.1 ± 0.6 83.6 ± 1.6

- Table 5: Impact of using intermediate shared memory with shuffle of order in test set in Xolver.

D.4 List of Roles of Selected by Dynamic Agents

- Table 6 shows some selected specialized roles by the dynamic agents while testing on coding and math tasks along with their most frequently selected roles.

- (a) Specialized Roles

Math Coding Problem Analyzer Problem Analyzer Mathematical Modeler Algorithm Designer Algorithm Designer Solution Architect Numerical Analyst Implementation Specialist Symbolic Solver Data Structure Specialist Mathematician Optimization Engineer Computational Tools Specialist Unit Tester

Debugging Expert Programmer Debugging Expert Code Reviewer

- (b) Most Frequent Roles

Math Coding Mathematical Modeler Algorithm Designer Numerical Analyst Implementation Specialist Symbolic Solver Data Structure Specialist Mathematician Programmer Computational Tools Specialist Optimization Engineer

Table 6: List of math and coding roles selected by LLMs.

