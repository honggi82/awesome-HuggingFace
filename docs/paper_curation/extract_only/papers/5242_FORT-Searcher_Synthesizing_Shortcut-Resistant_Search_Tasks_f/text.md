# arXiv:2606.12087v1[cs.CL]10Jun2026

[Figure 1]

[Figure 2]

## FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents

Jia Deng1*, Yimeng Chen2*, Xiaoqing Xiang1*, Ziyang Zeng3*, Shuo Tang4*, Wayne Xin Zhao1†, Feng Chang3†‡, Chuan Hao3, Yuan Wei3, Ran Tao3, Bryan Dai3, Ji-Rong Wen1 1Gaoling School of Artificial Intelligence, Renmin University of China, 2KAUST, 3IQuest Research, 4Shanghai Jiao Tong University *Equal contributors †Corresponding Authors ‡Project Leader Email: dengjia0510@outlook.com, yimeng.chen@kaust.edu.sa, batmanfly@gmail.com, fchang@iquestlab.com

## Abstract

Training deep search agents requires verifiable questions whose answers remain unavailable until sufficient evidence has been acquired through search. Existing synthesis methods often increase apparent difficulty by enriching graph structures, but structural complexity alone does not guarantee realized search difficulty: the intended search process can collapse through a cheaper identifying route. We formalize this gap with a shortcut-aware difficulty framework and identify four actionable shortcut risks: evidence co-coverage, single-clue selectivity, exposed constants, and prior-knowledge binding. To diagnose their realized effects, we use trajectory signatures including solving cost, answer hit time, and prior-shortcut rate. Guided by this framework, we introduce FORT, a Framework of Shortcut-Resistant Training-Data Synthesis. FORT constructs shortcut-resistant training data by controlling shortcut risks across entity selection, evidence graph construction, question formulation, and adversarial refinement. Experiments show that FORT induces longer pre-answer search and fewer shortcut patterns than existing open-source deep search datasets. Using the resulting trajectories, we train FORT-Searcher with supervised finetuning (SFT) only, and it achieves the best overall performance among comparable-size open-source search agents on challenging deep search benchmarks. Relevant resources will be made available at https://github.com/RUCAIBox/FORT-Searcher.

FORTSearcher

Comparable-size open-source

>30B or closed-source

| |
|---|

| |
|---|

| |
|---|

###### BrowseComp

BrowseComp-ZH

###### Comparable-size open-source >30B or closed-source

###### Comparable-size open-source >30B or closed-source

90

80

78.6

76.1

75.9

75.0

73.7

72.7

72.3

72.2

Score

70.3

69.5

69.0

70

67.9

65.8

61.0

60

58.2

57.4

50

FORTSearcherMiroThinkerQwen3.5-35BREDSearcherQwen3.5-397B GLM-5Step3.5FlashGPT-5.2

FORTSearcherMiroThinkerQwen3.5-35BREDSearcherQwen3.5-397B GLM-5Step3.5FlashGPT-5.2

Figure 1. Performance of FORT-Searcher against other search agents on BrowseComp and BrowseComp-ZH.

1

### 1. Introduction

Large language models (LLMs) are increasingly deployed as tool-empowered search agents [3, 11] that interact with external environments, gather evidence over multiple turns, and synthesize scattered information before arriving at an answer. The target capability in this setting is not merely retrieval, but persistent evidence discovery on the open web: an agent must formulate and iteratively revise search queries, locate hard-to-find information, assess the reliability of retrieved evidence, and connect disparate clues before committing to a final response [4, 12, 20, 41].

Training effective search agents requires high-quality search data built around verifiable questions whose answers must be acquired through search. With such data, search agents can be trained using supervised fine-tuning or reinforcement learning [23, 24]. However, existing question-answering datasets designed for traditional retrieval are not a good fit for this purpose, because they often do not require long-horizon evidence acquisition [10, 38].

To address this limitation, recent deep search task synthesis methods often increase apparent difficulty by enriching the intended solving structure, including hop count and graph shape [25, 26, 31, 34], hierarchical constraints [37], and evidence dispersion or treewidth [6]. These controls are useful because they specify an intended multi-step evidence-acquisition process. However, the underlying structural priors used to generate the question-to-answer chain do not necessarily translate into the realized search process when the task is actually executed by the search agent. The agent may still bypass much of the intended evidence acquisition by leveraging evidence from the environment or its own prior knowledge. As a result, training on such data yields limited benefit because it seldom incentivizes LLMs to engage in long-horizon search, planning, and information synthesis.

To characterize this mismatch, we develop a shortcut-aware difficulty framework for multi-constraint agentic retrieval. The framework models deep search difficulty as depending not only on the complexity of the intended task structure, but also on how the resulting question is realized in the retrieval environment and by the solver. A question may contain many clues and be constructed from a long evidence graph, yet still become easy if the solver can identify the answer through a cheaper route during actual search. Such failures can arise when a clue is overly selective, several intended clues are co-covered by the same retrieved evidence, the question exposes searchable constants that should have been discovered through earlier retrieval, or the model binds the answer from prior knowledge before retrieval provides support. We refer to these route-level collapses, together with solver-level prior binding, as shortcuts. This view suggests that data synthesis should not only build complex intended structures, but also check whether the resulting questions still force the intended evidence-acquisition process in real search trajectories.

This framework leads to four actionable shortcut risks that should be controlled during data synthesis: evidence co-coverage, where multiple clues can be verified from the same evidence item; single-clue selectivity, where one clue or a small clue subset is already enough to identify the answer; exposed constants, where names, strings, or numbers on the question surface make downstream queries directly executable; and prior-knowledge binding, where the solver names the answer before retrieval anchors it. Since these risks are only partially visible from the final question alone, we monitor their realized effects through trajectory signatures, including solving cost, answer hit time, and prior-shortcut rate.

Guided by this framework, we introduce FORT, a Framework of Shortcut-Resistant Training-Data Synthesis. FORT controls shortcut risks across the full data construction process. It selects long-tail root entities to reduce prior-knowledge binding, initializes and expands evidence graphs to avoid linearly exposed structures, collects facts from heterogeneous sources to reduce evidence co-coverage, constructs derived facts that are less likely to appear verbatim in a single source, and selects facts that are weak in isolation but identifying in combination. During question formulation, it withholds intermediate entity names and fuzzes exact constants into truthful ranges, categories, or indirect descriptions. Finally,

adversarial refinement runs a strong search agent against each draft question and repairs shortcut-prone, ambiguous, or over-obfuscated cases.

Experiments show that FORT produces useful supervision for deep search agents. With supervised fine-tuning alone, we introduce FORT-Searcher, which achieves the best overall performance among comparable-size open-source agents on challenging benchmarks, including BrowseComp, BrowseCompZH, xbench-DeepSearch, and Seal-0. Further analyses show that FORT induces longer pre-answer search than existing open-source deep search datasets and reduces multiple shortcut patterns, supporting the importance of shortcut-resistant data synthesis.

Our contributions are as follows:

- • We develop a difficulty framework for multi-constraint agentic retrieval tasks, showing that realized difficulty is governed by the cheapest identifying route and diagnosing four actionable shortcut risks: evidence co-coverage, single-clue selectivity, exposed constants, and prior-knowledge binding.
- • We propose FORT, a Framework of Shortcut-Resistant Training-Data Synthesis that turns shortcut analysis into construction controls across entity selection, evidence graph construction, question formulation, and adversarial refinement. Trajectory diagnostics show that FORT tasks induce substantially longer pre-answer search than existing open-source deep search datasets.
- • We train FORT-Searcher with SFT on FORT data and achieve the best average performance among comparable-size open-source search agents on challenging deep search benchmarks, demonstrating the effectiveness of our method.

### 2. Difficulty Framework

We study a class of multi-constraint agentic retrieval tasks. In such a task, an agent solves a question through multi-turn interaction with a retrieval interface, such as a web search. At each turn, the agent may issue a query, observe retrieved evidence, and decide whether to continue searching or return an answer. For readability, we summarize the main notation used in the difficulty framework in Appendix A.

- 2.1. Problem Formulation A task instance is represented as

𝑞 = (X, C𝑞, Σ), (1) where X is the answer space, C𝑞 is the set of constraints expressed by the question, and Σ is the retrieval interface. We use a botanist-identification question as a running example throughout this section. Here the answer space X is the set of candidate botanists. The question does not name the target botanist directly. Instead, it describes the target through several indirect constraints: the botanist described a fern species whose specific epithet is derived from a mountain range; the botanist’s doctoral advisor also supervised another botanist known for discovering a certain orchid; and the botanist served as president of a botanical society in a year whose digits sum to seven. In this example, C𝑞 contains these clue-like conditions. The retrieval interface Σ specifies what information the agent can obtain through tool calls. For instance, when Σ is a web search engine, a query may return web snippets that help verify which candidate in X satisfies the constraints in C𝑞.

For any subset of constraints P ⊆ C𝑞, we use Ans(P) to denote the remaining candidate pool after only the clues in P are applied. That is, it contains all entities in X that satisfy every constraint in P:

Ans(P) = 𝑥 ∈ X :

𝑐𝑖(𝑥) = 1 . (2)

𝑐𝑖∈P

For example, if P contains only the fern-species clue in the botanist example, then Ans(P) is the set of botanists matching that clue; adding more clues to P further narrows this candidate pool. We assume the question is well-posed: the full constraint set determines a unique gold answer 𝑦★, i.e. Ans(C𝑞) = {𝑦★}.

Each retrieval action is a query 𝜃 submitted to Σ, which returns a set of evidence items. A search trajectory is the sequence of queries a solver issues before committing to its answer, 𝜏 = (𝜃1, . . . ,𝜃|𝜏|); its length |𝜏| counts these retrieval queries, and 𝜏 succeeds when the committed answer is 𝑦★. We count only retrieval, since that is the cost of interest.

- 2.2. Difficulty Analysis Task difficulty has two sources. One is objective: the structure of the constraints and the gold answer, together with what the interface Σ exposes and how it connects evidence, which fixes how much retrieval any evidence-only solver is forced to perform. The other is solver-specific: a concrete model may bypass part of that search through prior knowledge. We analyze the main factors behind both sources and derive a lower bound on the task-side cost 𝐷post(𝑞); the solver-specific factor then relates this bound to the realized cost Ω(𝑞,𝜋0).

To measure the search cost imposed by the task itself, we first define a reference solver that does not use problem-specific prior knowledge. Let Πpost be the class of no-prior, no-guessing solvers. A solver in this class issues only executable queries—those whose required constants are already available from the question or from previously retrieved evidence—and commits to an answer 𝑥 only once retrieved evidence verifies that 𝑥 satisfies some subset P with Ans(P) = {𝑥}, i.e. a subset that already identifies the answer. We define the pure-posterior cost as

𝐷post(𝑞) = inf

𝜋∈Πpost

E𝜏∼𝜋 [|𝜏|] . (3)

The formal restrictions of Πpost, including query executability and evidence-grounded termination, are given in Appendix B.1. This quantity is the minimum expected number of retrieval queries an evidenceonly solver needs to answer 𝑞: the search cost the question imposes when the solver cannot rely on memorized facts or unsupported guesses.

Even if the question carries many clues, such a solver need not verify all of them; it only needs evidence for some subset that already identifies the answer. The difficulty forced by the question is therefore governed by the cheapest route that verifies such a subset. For a subset P with Ans(P) = {𝑦★}, let 𝑄Σ(P) be the length of the shortest valid evidence-acquisition route that verifies P for 𝑦★:

𝑄Σ(P) = min |𝜽| : 𝜽 is a valid evidence-acquisition route for (𝑦★, P) under Σ . (4)

Unlike a realized trajectory 𝜏, 𝜽 is an ideal route for a known target (𝑦★, P). 𝜏 is instead what a solver issues adaptively as it searches, which may run longer with redundant or dead-end queries. A fully formal definition of valid routes is given in Appendix B.2: a valid route must retrieve enough evidence to verify that 𝑦★ satisfies P, and must respect query executability, so each query uses only constants present in the question or exposed by earlier retrieved evidence.

The structural lower bound is obtained by minimizing this route cost over all identifying subsets:

𝑄★Σ = min

𝑄Σ(P). (5)

P: Ans(P)={𝑦★}

Under the no-prior, no-guessing reference solver, the pure-posterior cost is lower-bounded by this cheapest identifying route:

𝐷post(𝑞) ≥ 𝑄★Σ. (6)

The proof is in Appendix C.1. The bound is one-sided: a large 𝑄★Σ means even the cheapest identifying route requires substantial evidence acquisition, while a small 𝑄★Σ means the question does not force much search.

We now isolate three structural quantities that control 𝑄★Σ and the gap above it. The first is subset selectivity. For a constraint subset P ⊆ C𝑞 we define

##### 𝑠(P) = |Ans(P)|, (7)

the size of the candidate pool after applying only the clues in P. For example, with 𝑐fern the fern-species clue, if P = {𝑐fern} leaves two plausible botanists then 𝑠(P) = 2; if it leaves only the gold botanist then Ans(P) = {𝑦★}, and this single clue already identifies the answer. Selectivity does not lower-bound the route length, so it is not a component of the floor 𝑄★Σ. Rather, it is the candidate-breadth dimension governing the exploration gap 𝐷post(𝑞) − 𝑄★Σ: a large remaining candidate space typically forces a no-prior solver to generate and eliminate more candidates than the cheapest route alone would require. When a small subset is selective enough to identify the answer, selectivity instead controls which subsets enter the minimization in Eq. (5).

The second factor is evidence dispersion. For a fixed subset of clues, we next ask how much separate evidence-acquisition effort is needed to verify it. We use 𝑀ev(P) to denote the minimum number of queries needed to verify that 𝑦★ satisfies all constraints in P:

𝑀ev(P) = min |𝜽| : 𝜽 retrieves evidence verifying (𝑦★, P) . (8)

Note that 𝑀ev(P) ignores query executability: it counts only how many distinct retrievals are needed to cover the verification, regardless of their order. Adding executability back to this requirement can only raise the cost, so 𝑄Σ(P) ≥ 𝑀ev(P). A small 𝑀ev(P) means that little separate evidence needs to be acquired. For example, let P = {𝑐fern, 𝑐society}, where 𝑐fern is the fern-species clue and 𝑐society is the botanical-society presidency clue. If one web page states both facts, then a single evidence-acquisition step is sufficient, so 𝑀ev(P) = 1.

The third factor is dependency depth. A later query is dependent on an earlier one when it requires a name, attribute, or intermediate fact that is not available in the question but is exposed by earlier retrieved evidence. For example, in the botanist question, searching for other students of the same doctoral advisor requires first discovering the advisor’s name. For a valid evidence-acquisition route 𝜽, let depth(𝜽) be the length of the longest dependency chain among its queries. We define

dep(P) = min

depth(𝜽). (9)

𝜽 valid for P

A smaller dep(P) means that the required evidence can be reached through a shorter serial dependency. If the question directly exposes an intermediate constant, such as the advisor’s name, then the downstream query becomes executable from the start and the dependency depth may become smaller.

Selectivity, dispersion, and depth are objective properties of the task and interface; together they determine 𝐷post(𝑞), the cost any no-prior solver must pay. A concrete solver 𝜋0 may pay less, either by drawing on model-specific prior knowledge or by committing to the answer before retrieved evidence supports it. We write its actual successful-trajectory cost as

Ω(𝑞,𝜋0) = E𝜏∼𝜋0 [|𝜏| | 𝜏 succeeds] , (10) and the solver-side cost reduction as

𝑈𝜋0(𝑞) = 𝐷post(𝑞) − Ω(𝑞,𝜋0). (11)

A positive 𝑈𝜋0(𝑞) means 𝜋0 reaches the answer with fewer retrieval queries than the no-prior reference solver would need: for instance, a model already familiar with the target botanist may answer after very few searches, whereas the evidence-only solver must follow the advisor and society-record clues. The

quantity is most directly interpretable when 𝜋0 has a high success rate; otherwise it should be read together with that rate.

We summarize the above discussion of the four factors’ effects on difficulty in the following proposition. Proposition 1 (Structural lower bound and realized cost). Let 𝑞 be a well-posed instance and 𝜋0 any solver. Every identifying subset P satisfies

𝑄Σ(P) ≥ max 𝑀ev(P), dep(P) , (12) so the pure-posterior cost is bounded below by the cheapest identifying route,

𝐷post(𝑞) ≥ 𝑄★Σ ≥ min

max 𝑀ev(P), dep(P) . (13)

P: Ans(P)={𝑦★}

The first inequality holds with equality when the cheapest identifying route can be followed without exploration, and 𝑄★Σ = 1 when some identifying subset is verified by a single initially executable query. For a concrete solver, the realized successful-trajectory cost satisfies

Ω(𝑞,𝜋0) = 𝐷post(𝑞) −𝑈𝜋0(𝑞), (14) and reaches the lower bound 𝑄★Σ when, in addition, 𝑈𝜋0(𝑞) = 0.

Eq. (12) and the collapse case are proved in Appendices C.2 and C.3. Evidence dispersion 𝑀ev and dependency depth dep form this lower bound: 𝑄★Σ is large only when every identifying subset is either dispersed or deep. The two enter differently. 𝑀ev does not depend on executability, so the smallest 𝑀ev over identifying subsets already lower-bounds 𝐷post(𝑞); dep, by contrast, arises only from query executability. Two further factors move the realized cost away from this bound: subset selectivity sets the exploration gap 𝐷post(𝑞) − 𝑄★Σ (and, when a small subset is selective enough to identify the answer, which subsets enter the minimization), while the prior utility 𝑈𝜋0 sets the gap between 𝐷post(𝑞) and the realized cost of a specific solver. Since each gap is attributed to a named factor, the four jointly consists the difficulty.

- 2.3. Shortcut Patterns We call a mechanism a shortcut if it allows a question to be solved with less evidence acquisition than intended, thereby reducing the realized retrieval cost Ω(𝑞,𝜋0). According to Eq. (14), this reduction can

happen in two ways. A route-level shortcut makes the cheapest identifying route shorter, reducing 𝑄★Σ in Eq. (5) and weakening the structural lower bound on 𝐷post(𝑞) in Eq. (6). A solver-level shortcut instead increases 𝑈𝜋0(𝑞), allowing a concrete model to solve the task with less search than the no-prior reference solver. We focus on four actionable shortcut patterns. The first three operate through 𝑄★Σ, while the fourth operates through 𝑈𝜋0(𝑞).

Single-clue selectivity. This route-level shortcut occurs when one clue, or a small clue subset, narrows the candidate set to one or a few plausible answers. In terms of Eq. (7), 𝑠(P) is very small for some proper subset P ⊊ C𝑞. If such a subset already identifies the answer (Ans(P) = {𝑦★}), then the answer can be reached without using the full constraint set, making the minimum in Eq. (5) attainable through a small subset. Even when the subset is not identifying, its high selectivity can make the gold answer or near-answer evidence appear early in finite retrieval results. This reduces realized retrieval cost because the solver can focus on a small candidate set and reach the answer before acquiring evidence for all intended constraints.

Evidence co-coverage. This route-level shortcut occurs when one retrieved evidence item, page, or snippet verifies several intended constraints at once. In terms of Eq. (8), this makes 𝑀ev(P) small for

some identifying subset P. Since 𝑀ev(P) lower-bounds the route cost in Eq. (12), co-covered evidence can make 𝑄Σ(P) small and may collapse the cheapest identifying route 𝑄★Σ in Eq. (5). This reduces realized retrieval cost because several intended evidence-acquisition steps are compressed into a single retrieved source.

Exposed constants. This route-level shortcut occurs when the question surface exposes an exact name, string, date, number, or other intermediate constant that should have been discovered through earlier retrieval. Such constants make downstream queries executable from the initial question. In terms of Eq. (9), they reduce dep(P) by shortening the serial dependency needed to reach the relevant evidence. Since dep(P) also lower-bounds the route cost in Eq. (12), exposed constants can reduce 𝑄Σ(P) and hence 𝑄★Σ. This reduces realized retrieval cost because the solver can skip intermediate search steps that were intended to make later queries executable.

Prior-knowledge binding. This solver-level shortcut occurs when a concrete solver names or commits to the gold answer before retrieved evidence anchors it. Unlike the previous three shortcuts, it does not necessarily reduce the structural route cost 𝑄★Σ or the pure-posterior cost 𝐷post(𝑞). Instead, it directly lowers Ω(𝑞,𝜋0) by increasing the solver-side cost reduction 𝑈𝜋0(𝑞) in Eq. (11). Thus, even if the task requires a long evidence-acquisition route for a no-prior solver, it can still become easy for a model that binds the answer from parametric knowledge before search provides support.

Together, these shortcuts explain how a multi-constraint retrieval task can fail to induce substantial search. The first three make the task structurally easier by shortening the cheapest identifying route, while the fourth makes the task solver-specifically easier by reducing the cost paid by a particular model. Representative trajectory-level examples of the four shortcut patterns are provided in Appendix D.

- 2.4. Trajectory Signatures The quantities above explain how difficulty can collapse, but they are not all directly computable at scale. We therefore use observable trajectory signatures to diagnose realized difficulty under a fixed solver and retrieval interface. Let 𝜏𝑖 be a successful trajectory for question 𝑞𝑖. For 𝑁 successful trajectories, the realized solving cost is

##### ∑︁𝑁

1

|𝜏𝑖|. (15)

Ω =

𝑁

𝑖=1

This is the dataset-level empirical counterpart of Ω(𝑞,𝜋0) in Eq. (10). A high Ω means that the solver used many retrieval calls, but it does not by itself prove that the answer was hard to discover. The answer may have appeared early, with later turns spent on verification or detours. We therefore need to measure not only how long the trajectory is, but also when the answer first becomes available. To measure answer availability, let 𝑇tool(𝜏) be the first retrieval step whose observation contains the gold answer or a normalized alias, and let 𝑇model(𝜏) be the first model step at which the model explicitly mentions the gold answer or a normalized alias in visible generated text. The answer hit time of a trajectory is

𝑇hit(𝜏) = min {𝑇tool(𝜏),𝑇model(𝜏)} . (16) For a dataset, we report the average answer hit time:

##### ∑︁𝑁

1

𝑇hit(𝜏𝑖). (17)

𝑇hit =

𝑁

𝑖=1

A later 𝑇hit indicates a longer pre-answer search prefix, which is the observable behavior expected when cheap identifying routes in Eq. (5) are suppressed. Conversely, an early 𝑇hit suggests that the answer

becomes available through a short route even if the full trajectory is long. A large gap between Ω and 𝑇hit therefore suggests post-hit verification, detours, or redundant searching after the answer has already surfaced. Finally, we estimate explicit prior-knowledge binding with the prior-shortcut rate. A successful trajectory is marked as prior-bound if the model mentions the gold answer before any retrieved observation anchors it:

##### ∑︁𝑁

1

1 [𝑇model(𝜏𝑖) < 𝑇tool(𝜏𝑖)] . (18)

𝑝prior =

𝑁

𝑖=1

This is a conservative observable proxy for solver-side prior binding. It corresponds to the shortcut mechanism captured by 𝑈𝜋0(𝑞) in Eq. (11): when the model names the answer before retrieval evidence anchors it, the successful trajectory may pay less search cost than the no-prior reference solver. The proxy is conservative because it captures only visible answer-before-evidence behavior and may miss cases where the model uses prior knowledge without verbalizing the answer early.

To examine whether existing open-source deep-search data induces substantial pre-answer search, we evaluate several datasets with the same strong search agent under the same retrieval budget. For each dataset, we randomly sample 200 questions and compute Ω, 𝑇hit, and 𝑝prior over successful trajectories. Table 11 shows that existing open-source deep-search datasets often provide limited pre-answer search. Some datasets have low realized solving cost and early answer hit time, suggesting that the answer becomes available after only a few retrieval steps. Others induce longer trajectories but still expose the answer early or show high prior-shortcut rates. For example, OpenSeeker has Ω = 84.7, but its 𝑇hit is only 9.3 and 𝑝prior reaches 31.9. REDSearcher has stronger trajectory signatures than the other open-source datasets, yet its answer hit time remains much earlier than its total solving cost. These observations suggest that apparent search length alone does not guarantee search-heavy supervision. Useful training data should reduce cheap identifying routes in Eq. (5), delay answer exposure as reflected by 𝑇hit, and limit explicit prior binding as reflected by 𝑝prior. This motivates FORT, which controls shortcut risks during entity selection, evidence graph construction, question formulation, and adversarial refinement.

- 3. Methodology Section 2 characterizes two ways in which realized search difficulty can collapse. At the route level, a

question becomes easier when the cheapest identifying route 𝑄★Σ in Eq. (5) is small, thereby weakening the structural lower bound on 𝐷post(𝑞) in Eq. (6). At the solver level, a concrete model may further reduce the realized retrieval cost Ω(𝑞,𝜋0) in Eq. (14) by increasing the solver-side cost reduction 𝑈𝜋0(𝑞) in Eq. (11). FORT treats data synthesis as the inverse construction problem: it builds questions that cheap identifying routes are less available and solver-side prior binding is less likely.

Our methodology has two components. First, FORT synthesizes shortcut-resistant questions and search trajectories by controlling how constraints, evidence sources, and intermediate dependencies are constructed before the final question is exposed to a solver. Second, the resulting trajectories are used to train FORT-Searcher, an SFT-only search agent with a context-managed inference protocol. This separation links the data-side objective of reducing route-level and solver-level shortcuts to the model-side objective of learning robust multi-turn evidence acquisition.

- 3.1. FORT: Shortcut-Resistant Data Synthesis The difficulty framework is defined over final questions, whereas FORT must intervene before the question is verbalized. To do so, we use an internal evidence graph as a construction workspace. In this graph, nodes represent real-world entities, and edges represent verified facts that connect entities and can later be verbalized as clues. The graph is not the object solved by the final agent; instead, it organizes these entities and facts together with their evidence sources and dependencies before they are rendered as a natural-language question. By selecting and verbalizing subgraphs from this workspace, FORT controls

Shortcut risk Collapsed quantity FORT control Implemented in

Single-clue selectivity

Graph construction Question formulation Evidence co-coverage

Small P: 𝑠(P) ↓

Generic fact selection Low-specificity clue formulation

𝑀ev(P) ↓ Multi-source enrichment Derived fact construction

Graph construction

Cycle-based initialization Name withholding Exact-value fuzzing

Exposed constants

Graph initialization Question formulation

dep(P) ↓

###### Prior-knowledge binding

Long-tail root selection Adversarial refinement

Graph initialization Adversarial refinement

𝑈𝜋0(𝑞) ↑ Ω(𝑞,𝜋0) ↓

Table 1. From shortcut risks to FORT controls. The table connects the difficulty quantities in Section 2 with the construction mechanisms in Section 3.

the route-level factors introduced in Section 2.2: clue selectivity 𝑠(P) in Eq. (7), evidence dispersion 𝑀ev(P) in Eq. (8), and dependency depth dep(P) in Eq. (9). These factors affect whether the cheapest identifying route 𝑄★Σ in Eq. (5) can collapse, as characterized by the component lower bound in Eq. (12). In parallel, long-tail root selection and adversarial refinement target solver-level prior binding, which increases 𝑈𝜋0(𝑞) in Eq. (11) and reduces the realized cost Ω(𝑞,𝜋0) through Eq. (14). Table 1 summarizes how these shortcut risks are translated into construction-time controls. The first three risks are addressed by controlling the route-level quantities that influence 𝑄★Σ, while prior-knowledge binding is addressed by reducing solver-side cost reduction. The four stages of FORT implement these controls jointly rather than as independent engineering steps.

As shown in Figure 2, FORT consists of four stages. First, graph initialization selects a long-tail root entity and initializes a seed structure, reducing prior binding and limiting premature exposure of downstream constants. Second, graph construction expands the seed into a heterogeneous evidence graph by collecting, deriving, verifying, and selecting facts; this stage disperses evidence and avoids overly identifying individual clues rather than merely increasing graph size. Third, question formulation chooses an answer-bearing subgraph and renders it as a natural-language question while withholding intermediate names and fuzzing exact constants, so that downstream queries are not made executable too early. Finally, adversarial refinement runs a strong search agent against each draft question and uses trajectory signatures to repair cases that are shortcut-prone, ambiguous, or over-fuzzed.

#### 3.1.1. Graph Initialization

FORT starts from a root entity 𝑟 and initializes a seed subgraph G0. This stage addresses two shortcut risks in Section 2.3: prior-knowledge binding and exposed constants. For prior-knowledge binding, we prefer long-tail root entities, since a familiar root may allow the solver to bind the answer from parametric knowledge, increasing 𝑈𝜋0(𝑞) in Eq. (11). For exposed constants, we initialize the seed with a less linearly exposed structure whenever possible, so that later question formulation can refer to relations indirectly without revealing downstream entity names. This helps preserve dependency depth dep(P) in Eq. (9) and prevents the cheapest identifying route 𝑄★Σ in Eq. (5) from becoming too short.

We select 𝑟 from Wikidata under three criteria. First, to ensure topical coverage, we sample from broad semantic categories and remove abstract concepts that do not support factual decomposition. Second, to reduce prior-knowledge binding, we prefer rare entities, especially those without an English Wikipedia page. Third, to preserve solvability, we run a lightweight pre-search and discard entities with insufficient external evidence.

Beyond the choice of root entity, the initial graph structure affects how much of the later search path must

###### 2 Graph Construction

###### 1 Graph Initialization

###### Enricher agent

Expander agent Reasoning graph G

fact pool

###### Root filtering

Multi-source collection

Derived fact Fact verification

[Figure 3]

Topical coverage

[Figure 4]

[Figure 5]

[Figure 6]

Weak fact selection

[Figure 7]

[Figure 8]

[Figure 9]

Entity obscurity

Iterative interaction

Child-node expansion

Wikidata candidates

[Figure 10]

new entity

[Figure 11]

Information density Initial cycle

generation

Fact sources: Root entity Wikidata Drived fact APIs Knowledge bases Web pages

- 3 Question Formulation

###### 4 Adversarial Refinement

1

[Figure 12]

Answer selection and subgraph pruning

·Repair shortcut ·Strengthen fuzzing

shortcutprone

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Diagnose and analyze

Clue composition

[Figure 19]

[Figure 20]

· Fix ambiguity ·Narrow fuzzing

initially unsolved

[Figure 21]

Fuzzing

Refined question

Adversarial agent

Draft question

Figure 2. Overview of FORT, a shortcut-resistant synthesis pipeline.

be exposed in the final question. A purely linear seed or expansion path often forces downstream entities, aliases, or literal values to be verbalized as surface clues. Once these constants appear in the question, the corresponding downstream queries may already be executable from the initial state, allowing the solver to skip the intended process of discovering them from earlier evidence. This creates an exposed-constant shortcut and may reduce dep(P).

To reduce this risk, FORT uses a pre-mined cycle structure whenever possible. A cycle seed embeds the root in a closed local structure rather than a purely linear chain, making it easier to formulate relational clues without explicitly exposing every intermediate entity. To support efficient initialization, we pre-mine a cycle library from Wikidata and store it as an entity-to-cycle index. Specifically, we build a pair index from Wikidata edges, enumerate small cycles, remove duplicate cycles with the same node set, and filter out unsuitable structures such as hub-based, redundant, or mass-collaboration cycles. The remaining cycles are stored in an inverted index I, where I[𝑢] contains all retained cycles that include entity 𝑢. During seed initialization, the initializer looks up I[𝑟]. If this set is non-empty, one retained cycle is used as G0; otherwise, FORT falls back to the single-node seed ({𝑟},∅).

#### 3.1.2. Graph Construction

Starting from the initialized seed graph G0, FORT expands it into an evidence graph G. This stage mainly targets two route-level shortcut risks in Table 1: evidence co-coverage and single-clue selectivity. To reduce evidence co-coverage, FORT collects facts from diverse sources and constructs derived facts that are less likely to be verified by one retrieved item, increasing the separate evidence-acquisition effort 𝑀ev(P) in Eq. (8). To reduce single-clue selectivity, it avoids facts that are overly identifying in isolation, so that small clue subsets are less likely to yield a very small candidate pool 𝑠(P) in Eq. (7). Together, these controls make cheap identifying routes in Eq. (5) less likely to collapse.

Nodes in G denote real-world entities, and edges denote verified facts that connect an entity to another referenced entity [25, 36]. The expansion is controlled by a depth limit 𝐷 and a node-addition budget 𝐵.

- As shown in Algorithm 1, FORT maintains a queue Q of expandable nodes and expands the deepest unprocessed node first. This preference preserves multi-step referenced-entity chains that can later be verbalized as serial dependencies in the final question. For each selected node 𝑣, an enricher agent collects atomic facts from external sources, constructs

derived facts, and verifies the resulting fact pool. An expander agent then selects suitable facts, adds the corresponding edges to G, and promotes newly referenced entities as child nodes. New nodes are assigned depth 𝛿G(𝑣) + 1 and are expanded only if they remain within the depth limit.

Algorithm 1 Evidence graph construction

Input: initialized subgraph G0 = (𝑉0, 𝐸0) with root 𝑟 ∈ 𝑉0, depth limit 𝐷, node-addition budget 𝐵 Output: evidence graph G

- 1: G ← G0
- 2: 𝛿G(𝑢) ← distG(𝑟,𝑢) for all 𝑢 ∈ 𝑉(G) ⊲ initialize node depth
- 3: Q ← {𝑢 ∈ 𝑉(G) : 𝛿G(𝑢) < 𝐷}
- 4: Vdone ← ∅
- 5: 𝑏 ← 0 ⊲ number of newly added nodes
- 6: while Q ≠ ∅ and 𝑏 < 𝐵 do
- 7: 𝑣 ← argmax𝑢∈Q 𝛿G(𝑢) ⊲ expand deepest node first
- 8: Q ← Q \ {𝑣}
- 9: if 𝑣 ∈ Vdone or 𝛿G(𝑣) ≥ 𝐷 then
- 10: continue
- 11: end if
- 12: Vdone ← Vdone ∪ {𝑣}
- 13: A𝑣 ← COLLECTATOMICFACTS(𝑣) ⊲ collect facts from diverse sources
- 14: D𝑣 ← CONSTRUCTDERIVEDFACTS(A𝑣) ⊲ construct facts from multiple atomic facts
- 15: P𝑣 ← A𝑣 ∪ D𝑣
- 16: F𝑣 ← VERIFYFACTS(𝑣, P𝑣) ⊲ filter inconsistent or mismatched facts
- 17: S𝑣 ← SELECTFACTS(F𝑣) ⊲ select low-shortcut facts
- 18: for (𝑣, 𝜌,𝑤) ∈ S𝑣 do ⊲ 𝜌: selected fact, 𝑤: referenced entity
- 19: if 𝑤 ∉ 𝑉(G) and 𝑏 < 𝐵 then
- 20: G ← G ∪ {(𝑣, 𝜌,𝑤)}
- 21: 𝛿G(𝑤) ← 𝛿G(𝑣) + 1
- 22: 𝑏 ← 𝑏 + 1
- 23: if 𝛿G(𝑤) < 𝐷 then
- 24: Q ← Q ∪ {𝑤}
- 25: end if
- 26: end if
- 27: end for
- 28: end while
- 29: return G

Multi-source enrichment. To reduce evidence co-coverage, the enricher avoids drawing multiple selected facts from the same evidence item. It collects facts from heterogeneous sources, including Wikidata, open web pages, structured databases, Google Scholar, and Google Maps. For open web evidence, FORT further encourages different facts to come from different pages. Facts around the root node are distributed across sources whenever possible, while non-root nodes use sampled sources for efficiency because only one or two facts are typically selected for each node [43].

Derived fact construction. Atomic facts are directly extracted from evidence items and may be easy to recover through keyword search. FORT therefore augments the fact pool with derived facts constructed from multiple atomic facts or source texts. Such facts are less likely to appear verbatim in a single retrieved item, reducing exact-match shortcuts and mitigating evidence co-coverage. We use four constructors: coincidence bridging, count aggregation, numerical relation, and meta-fact extraction [21]. Table 2 defines each constructor and gives examples.

Constructor Definition Example Coincidence bridging

Derives a relation by matching the same attribute across independently retrieved records.

Entity 𝐴 once held a concert at venue 𝑉. Entity 𝐵 once held a concert at venue 𝑉. ⇒ 𝐴 and 𝐵 once held concerts at the same venue.

Count aggregation

Derives a counting constraint by combining an entity set with independently verified attributes.

The cast of work 𝑊 includes 𝐴, 𝐵, and 𝐶. 𝐴, 𝐵, and 𝐶 have each won award 𝑀.

⇒ 𝑊 features at least two winners of award 𝑀.

Numerical relation

Derives a quantitative relation by retrieving numerical values and applying arithmetic comparison.

Person 𝐴 was born in year 𝑌1. 𝐴’s advisor is 𝐵. 𝐵 was born in year 𝑌2.

⇒ 𝐴’s advisor is 𝑌2 −𝑌1 years older than 𝐴.

Meta-fact extraction

Derives a fact by inspecting, counting, or summarizing the original text of a rich-content entity.

Source text: song 𝐴’s lyrics.

⇒ The word 𝐵 appears at least three times in song 𝐴. Source text: Chapter 3 of novel 𝐶.

⇒ Chapter 3 of novel 𝐶 begins with a memory-oriented passage.

Table 2. Derived fact constructors and representative examples.

Fact verification. FORT verifies candidate facts for source consistency and entity consistency. For source consistency, a fact is discarded if it is contradicted by the same evidence item, contradicted by other retrieved evidence, or incompatible with accepted facts in the current graph. For entity consistency, the evidence item must describe the same real-world entity as the target node, rather than merely sharing surface strings. This step controls validity rather than difficulty [7]. Table 3 summarizes common entity-consistency failures.

Generic fact selection. Highly representative facts can be overly selective: they make 𝑠({𝑐𝑖}) in Eq. (7) small and may create a single-clue shortcut. The expander therefore prefers generic facts that are reliable but less characteristic of the target entity. Such facts usually cannot identify the target alone, but several of them can jointly determine the answer. Table 4 compares generic and representative facts for the same entities.

- 3.1.3. Question Formulation A question formulation agent realizes a selected subgraph G∗ ⊆ G as a natural-language question. This stage determines how the constructed evidence graph is exposed to the solver, and therefore whether the route-level controls from earlier stages survive in the final question. The agent selects an answer node 𝑦★, removes redundant facts, and keeps clues that are jointly identifying while remaining individually generic. This targets the single-clue selectivity shortcut: no individual clue, or very small clue subset, should make 𝑠(P) in Eq. (7) too small.

The formulation agent also controls which constants are exposed on the question surface. To reduce exposed-constant shortcuts, intermediate node names are withheld and rendered as generic referring expressions such as “the artist”, “the institution”, or “the work”. This prevents downstream queries from becoming executable from the initial question. Otherwise, the solver could skip the intended discovery chain, reduce the dependency depth dep(P) in Eq. (9), and shorten the cheapest identifying route 𝑄★Σ in Eq. (5).

For remaining entity names and literal values that must be mentioned, FORT applies exact-value fuzzing. Fuzzing rewrites exact names, strings, dates, or numbers into truthful but less directly searchable

Failure mode Definition Example case Similar names The retrieved page describes a different

Target: A Chemical Plant. Rejected page: A Organic Chemical Plant, which refers to a different company.

entity whose name is similar to, or partially overlaps with, the target name.

Abbreviation ambiguity

Target: a research institute abbreviated as IIS. Rejected page: IIS as Internet Information Services.

The same abbreviation refers to another subject, so surface-form overlap is insufficient for entity matching.

Temporal or geographic drift

Target: the 𝑁-th Director of Bureau 𝑋. Rejected page: the (𝑁 − 1)-th Director or a same-named bureau in another region.

The page matches the surface name but differs in time, location, or instance boundary.

Series–edition mismatch

Target: The Voice of China Season 3. Rejected pages: the series-level page or pages about other seasons.

The page describes a different granularity level, such as a series instead of a specific edition, or vice versa.

Table 3. Common failure modes in entity-consistency filtering.

Entity Generic fact Representative fact Marie Curie Worked at a university in France. Won Nobel Prizes in two different

scientific fields. The Eiffel Tower Is located in a European capital city. A landmark iron tower in Paris. The Matrix Is an English-language science fiction

Depicts humans living inside a simulated reality created by machines.

film.

- Table 4. Examples of generic and representative facts for the same entities. Generic facts have low standalone specificity and are less likely to create single-clue shortcuts.

constraints [42]. It is not intended to make the question ambiguous or unverifiable. Instead, it prevents surface constants from directly retrieving the answer or a near-identifying clue, while preserving the same gold answer.

- Table 5 summarizes the five fuzzing strategies used during question formulation. They are not applied by a fixed rule. For each exposed value, the formulation agent selects a strategy according to its semantic type, distinctiveness, and role in the question.

- 3.1.4. Adversarial Refinement Construction-time controls reduce shortcut risks, but they cannot guarantee realized difficulty once a draft question is exposed to a concrete solver and retrieval environment. FORT therefore performs adversarial refinement as a trajectory-level calibration step. For each draft question, we run a strong adversary agent in a realistic search setting and inspect the per-trajectory counterparts of the signatures defined in Section 2.4: realized solving cost, answer hit time, and prior-shortcut behavior (Eqs. (15)–(18)).

Let 𝜏th denote the minimum desired solving cost. A draft is accepted if the adversary answers correctly, uses at least 𝜏th retrieval turns, reaches the answer only after a sufficiently late 𝑇hit(𝜏), and does not bind the answer before retrieved evidence supports it. These criteria favor trajectories with a substantial

Strategy Description Examples Category generalization

Replace a specific entity name with a higher-level category, type, or domain description.

International Monetary Fund → an international financial institution. Harvard University → an Ivy League university in the United States.

Range relaxation

Replace an exact numerical value with an interval, approximate range, or open-ended constraint.

1863 → the second half of the nineteenth century. 16,000 → more than ten thousand.

Meta-attribute description

Replace a value with a description of its intrinsic form or surface-level structure.

September 9 → a date whose month and day use the same number. Hannah → a person whose given name is a palindrome.

Arithmetic encoding

Express a numerical value indirectly through arithmetic, digit-level, or number-theoretic constraints.

42 years old → an age that is a multiple of six. 17,921 → a five-digit prime number whose digits sum to 20.

Contrastive exclusion

Canada → a country in North America that does not border Mexico. Wednesday → a weekday that is not Monday.

Constrain the target value by excluding alternative candidates while keeping the value implicit.

Table 5. Exact-value fuzzing strategies used during question formulation.

pre-answer search prefix rather than early answer exposure followed by post-hoc verification.

If the adversary solves a draft too quickly or reaches the answer too early, we treat the trajectory as evidence of a route-level shortcut. FORT then repairs the corresponding clue or fact by replacing cocovered evidence, removing overly selective facts, withholding or fuzzing exposed constants, or otherwise eliminating the cheap identifying route [1, 15]. If the model names the answer before any retrieved evidence anchors it, the draft is treated as prior-bound and is repaired by replacing the root entity or strengthening the evidence path. Conversely, if the adversary fails to solve the draft within the rollout budget, the question is treated as over-hard or underspecified; FORT narrows over-fuzzed clues, removes ambiguous facts, or restores necessary constraints. Thus, adversarial refinement calibrates each draft toward the desired region: solvable, but still search-heavy. Figures 3 and 4 show two representative refinement cases.

- 3.2. Agent Training and Inference The synthesis pipeline above produces shortcut-resistant search trajectories, which are used to train FORT-Searcher. Starting from a base reasoning model, we perform supervised fine-tuning on trajectories generated from FORT questions. This step turns the data-side objective of shortcut resistance into model-side search behavior: the agent is trained to acquire evidence over multiple turns before committing to an answer, rather than relying on early answer exposure or parametric priors.

- At inference time, FORT-Searcher uses a context-managed search protocol. Within each rollout, retrieved tool-call results are retained so that the model can reuse accumulated evidence. If the model reaches the predefined turn limit without producing a final answer, the current interaction history is cleared and the agent restarts from the original question.

Case 1: Repairing Shortcuts Before.

A certain film was released in a certain year. The film has a runtime of a three-digit minute count whose digits sum to 10, features protagonists launching business schemes and a scam escalating to large monetary levels. The film’s composer is a certain individual who plays a certain instrument typically belonging to the keyboard family. In that same release year, a certain aviation incident occurred involving a target being shot down by the Soviet Union. The incident originated from a certain institution having 4 runways. Who is the director of the film?

#### Refinement.

The original wording creates a cross-domain shortcut: “Soviet Union” and “4 runways” uniquely identify a Cold War aviation incident (pinning the release year), and the precise digit-sum constraint on the runtime then collapses the candidate films to one. We repair this by broadening the incident description and relaxing the mathematical filter, forcing the solver to triangulate through the remaining business-scheme plot, composer, and keyboard-instrument clues.

#### After.

A certain film was released in a certain year. The film has a runtime of a three-digit minute count whose digits sum to a multiple of 5, features protagonists launching business schemes and a scam escalating to large monetary levels. The film’s composer is a certain individual who plays a certain instrument typically belonging to the keyboard family. In that same release year, a certain aviation incident occurred involving a target being shot down by a certain country during the Cold War. The incident originated from a certain international airport. Who is the director of the film?

Figure 3. Shortcut repair by broadening a cross-domain aviation reference and relaxing a precise runtime filter that together enabled single-query triangulation of the release year and film.

### 4. Experiment

- 4.1. Experimental Setup Our base model is Qwen3-30B-A3B-Thinking-2507 [32], which activates about 3B of 30B parameters at inference and supports a 256K context window. We train the model with supervised fine-tuning only, using the synthesized search trajectories produced by FORT. The training data are preprocessed with sequence packing to improve long-context utilization. We fine-tune for 6 epochs with a global batch size of 64 and a maximum sequence length of 262,144 tokens. Training uses bf16 precision and the Adam optimizer

with 𝛽1 = 0.9, 𝛽2 = 0.95, 𝜖 = 10−8, weight decay 0.01, and gradient clipping at 1.0. We use a cosine learning-rate schedule with peak learning rate 2 × 10−5, minimum learning rate 10−7, and 2 warmup iterations. For memory-efficient MoE training, we use tensor parallelism 4, pipeline parallelism 1, context parallelism 1, expert parallelism 4, sequence parallelism, activation recomputation, and a distributed optimizer.

Benchmarks. We evaluate our model on five challenging agentic benchmarks: BrowseComp [35], BrowseComp-ZH [44], xbench-DeepSearch-2505 [5], xbench-DeepSearch-2510 [5], and Seal-0 [19]. BrowseComp mainly consists of long-tail entity identification problems, where a short answer must be recovered from multiple indirect and range-based constraints. BrowseComp-ZH evaluates a similar setting over Chinese web sources. xbench-DeepSearch-2505 and xbench-DeepSearch-2510 cover broader real-world deep search tasks, including enumeration, document lookup, finance and so on. Seal-0 further tests search-augmented reasoning under noisy or conflicting evidence. To avoid unfair comparisons, we block domains from Hugging Face in the testing environment.

Baselines. We compare FORT-Searcher with leading search agents in three groups. The first group includes proprietary agents, which provide a frontier closed-system reference: GPT–5.2–Thinking– xhigh [22], GPT-5.5 [18], and Claude Opus 4.7 [2]. The second group includes large-scale opensource agents with substantially larger model capacity: GLM-5 [40], DeepSeek-V3.2 [16], Step 3.5 Flash [13], LongCat-Flash-Thinking-2601 [29], Hy3-preview [14], Kimi-K2.5-Thinking [28], Qwen3.5-

Case 2: Narrowing over-fuzzed clues Initial Question

An individual was born in a certain year and died 61 years after that birth year. The individual was a member of an institution headquartered in a building in a certain European capital, built in a year in the 1910s. This institution was a junior coalition partner to another institution founded in a certain year. The individual served in a ministerial position involving economic and communications matters in a region where another individual also served. The other individual worked as a lawyer until 50 years after that other institution’s founding year. The individual had been threatened multiple times before death. Who is the individual described above?

#### Refinement

The original clues are under-specified. The phrases “a certain European capital”, “a region”, and “a certain year” allow an unrelated figure to satisfy the same constraints. We narrow these clues with more specific but still anonymized descriptions, removing the spurious answer while preserving the intended obfuscation.

#### Refined Question

An individual was born in a certain year and died 61 years after that birth year. The individual was a member of an institution headquartered in a building in the capital of a federal republic in Central Europe, built in a year in the 1910s. This institution was a junior coalition partner to another institution founded in the mid-20th century. The individual served in a ministerial position involving economic and communications matters in a federated state where another individual also served. The other individual worked as a lawyer until 50 years after that other institution’s founding year. The individual had been threatened multiple times before death. Who is the individual described above?

Figure 4. Failure repair by narrowing over-fuzzed clues.

397B-A17B [27], and Qwen3.5-122B-A10B [27]. The third group includes comparable-size open-source agents: GLM-4.7-Flash [39], Tongyi DeepResearch [33], OpenSeekerV2 [8], REDSearcher [6], Qwen3.535B-A3B [27], and MiroThinker-1.7-mini [31]. Since our model activates about 3B parameters at inference, we treat the comparable-size open-source group as the primary fair-comparison set, while the proprietary and large-scale open-source groups provide broader performance context.

Context Management. For evaluation, we build on the context-management framework of MiroThinker [31]. Within a rollout, all tool-call results are retained so that the model can reuse previously retrieved evidence. If the model does not produce a final answer within the rollout limit, the current trajectory is discarded, the interaction history is cleared, and the model restarts from the original question [16]. We set the rollout limit to 300 turns for BrowseComp and BrowseComp-ZH, and 200 turns for xbench-DeepSearch-2505, xbench-DeepSearch-2510, and Seal-0.

#### 4.2. Main Results

- Table 6 compares FORT-Searcher with proprietary, large-scale open-source, and comparable-size opensource search agents. Among comparable-size open-source agents, FORT-Searcher achieves the highest overall score of 66.2, outperforming MiroThinker-1.7-mini [31] by 1.6 points and Qwen3.5-35B-A3B [27] by 6.3 points. It ranks first on BrowseComp, BrowseComp-ZH, and xbench-DeepSearch-2505, and ties the best result on xbench-DeepSearch-2510. On BrowseComp, it improves over MiroThinker-1.7-mini from 67.9 to 72.2. On BrowseComp-ZH, it improves from 72.3 to 75.0. On xbench-DeepSearch-2505, it reaches 80.8, outperforming the strongest comparable-size baseline, OpenSeekerV2, by 2.8 points.

Despite relying on SFT only and activating about 3B parameters at inference, FORT-Searcher remains competitive with larger open-source agents. On BrowseComp, it outperforms DeepSeek-V3.2 (67.6), Step

- 3.5 Flash (69.0), LongCat-Flash-Thinking-2601 (67.6), Hy3-preview (67.1), and Qwen3.5-122B-A10B (63.8). On BrowseComp-ZH, it achieves 75.0, the best open-source result among all agents listed in the table. On xbench-DeepSearch- 2505 and xbench-DeepSearch-2510, it also matches or exceeds most large-scale open-source agents with available scores. This pattern supports the effectiveness of FORT’s shortcut-resistant data synthesis for improving deep

Model BrowseComp BC-ZH xbench-05 xbench-10 Seal-0 Overall Proprietary Agents

GPT–5.2–Thinking–xhigh 65.8 76.1 – – – – GPT-5.5 84.4 – – – – – Claude Opus 4.7 79.3 – – – – –

Large-scale Open-source Agents

GLM-5 75.9 72.7 – – – – DeepSeek-V3.2 67.6 65.0 – – – – Step 3.5 Flash 69.0 73.7 83.7 56.3 – – LongCat-Flash-Thinking-2601 67.6 65.0 – – – – Hy3-preview 67.1 – – – – – Kimi-K2.5-Thinking 74.9 – – – 57.4 – Qwen3.5-397B-A17B 78.6 70.3 – – 46.9 – Qwen3.5-122B-A10B 63.8 69.9 – – 44.1 –

Comparable-size Open-source Agents

GLM–4.7–Flash 42.8 – – – – – Tongyi DeepResearch 43.4 46.7 75.0 47.5 45.8 51.7 OpenSeekerV2 46.0 58.1 78.0 43.4 41.4 53.4 REDSearcher 57.4 58.2 – – – – Qwen3.5-35B-A3B 61.0 69.5 77.4 50.3 41.4 59.9 MiroThinker-1.7-mini 67.9 72.3 77.2 57.2 48.2 64.6

FORT-Searcher 72.2 75.0 80.8 57.2 46.0 66.2

Table 6. Benchmark performance of FORT-Searcher against proprietary and open-source search agents. BC-ZH denotes BrowseComp-ZH; xbench-05 and xbench-10 denote xbench-DeepSearch-2505 and xbench-DeepSearch-2510, respectively. Bold numbers indicate the best results among comparable-size open-source agents. Overall is the average over the five benchmarks and is reported only for agents with complete results.

search behavior under SFT-only training.

- 4.3. Context Management We further examine the effect of context management on FORT-Searcher. Once a predefined turn limit is reached, the interaction history is cleared, and the model restarts from the original question [16].

- As shown in Table 7, context management brings substantial gains on BrowseComp and BrowseComp-ZH. On BrowseComp, the score increases from 55.9 to 72.2, yielding an absolute improvement of 16.3 points. On BrowseComp-ZH, the score increases from 62.1 to 75.0, yielding an absolute improvement of 12.9 points. In contrast, the gains on the other benchmarks are much smaller. The improvements are 0.7 points on xbench-DeepSearch-2505, 3.1 points on xbench-DeepSearch-2510, and 2.3 points on Seal-0. This suggests that BrowseComp and BrowseComp-ZH are more likely to fall into inefficient search patterns under a limited rollout budget. Context reset provides the model with an additional opportunity to explore a different search path, resulting in higher accuracy. For the other benchmarks, most examples are already solvable within the original rollout budget, making context reset less impactful.

Context Management

BrowseComp -ZH

BrowseComp

xbench-05 xbench-10 Seal-0

w/o 55.9 62.1 80.1 54.1 43.7 w/ 72.2 75.0 80.8 57.2 46.0

Table 7. Effect of context management on FORT-Searcher across five benchmarks.

### 5. Further Analysis

- 5.1. Training-Data Difficulty Analysis We examine whether the trajectory signatures defined in Section 2.4 capture useful training-data difficulty. We construct four training sets, each containing 12K examples and trained with the same recipe. The first three sets are sampled from existing open-source deep search data by filtering for average solving costs of approximately 40, 85, and 140 turns [6, 9, 17, 19, 30, 37]. These settings test whether increasing total trajectory length alone improves downstream search performance. The last setting is sampled from FORT data with a similar average solving cost, Ω = 140.0, but with later answer exposure and a lower prior-shortcut rate. This setting tests whether, under comparable trajectory length, longer pre-answer search and reduced prior binding provide additional training value.

- As shown in Table 8, increasing the average solving cost of open-source training data from 40.0 to 140.0 brings moderate gains on BrowseComp and BrowseComp-ZH. However, the best results are obtained

with FORT data at the same Ω = 140.0, where 𝑇hit is delayed to 47.0 and 𝑝prior is reduced to 11.4. This suggests that useful training difficulty is not defined by long trajectories alone, but by extended pre-answer search with limited prior exposure.

Data Settings Results 𝛀 𝑻hit 𝒑prior(%) BrowseComp BrowseComp-ZH

40.0 9.4 10.5 47.1 54.9 85.0 16.0 15.6 48.4 54.6

140.0 22.3 18.1 49.5 58.1 140.0 47.0 11.4 52.9 60.3

Table 8. Training results on data under different settings.

#### 5.2. Ablation Studies

- 5.2.1. Shortcut-Resistance Ablation To examine whether each shortcut-resistant component contributes to the realized difficulty of synthesized questions, we conduct a cumulative ablation on 2K questions. For each configuration, we evaluate the resulting questions using the same strong search agent and report its solving accuracy on these synthesized questions. Each row removes one additional component from the row above. For the long-tail entity ablation, we use GPT-5.5 [18] to select common, high-popularity replacement entities and reconstruct the questions accordingly.

Table 9 shows that removing shortcut-resistant components consistently makes the synthesized questions easier. Accuracy increases from 29.0 to 81.6, while Ω decreases from 141.9 to 43.7. At the same time, 𝑇hit shifts earlier from 46.5 to 11.8, and 𝑝prior increases from 11.4 to 22.3. These trends indicate that FORT’s difficulty relies on the joint control of multiple shortcuts. Under this cumulative ablation order, removing fuzzing produces the largest drop in difficulty, suggesting that obfuscation is especially important for increasing search difficulty.

###### Configuration Acc. 𝛀 𝑻hit 𝒑prior(%)

Full 29.0 141.9 46.5 11.4 − Cycle Construction 36.5 124.8 42.7 12.6 − Long-Tail Entity Selection 42.7 101.2 39.6 15.0 − Derived-Fact Construction 53.2 87.7 38.3 16.3 − Source Diversity 57.4 74.3 36.8 17.7 − Generic-Fact Selection 65.0 69.5 35.3 20.3 − Fuzzing 81.6 43.7 11.8 22.3

- Table 9. Cumulative ablation on shortcut-resistant components. Each row removes one additional component from the row above. Acc. denotes the solving accuracy of the same strong search agent on the synthesized questions, where lower values indicate harder questions under this evaluation setting.

5.2.2. Adversarial Refinement

To examine whether adversarial refinement can calibrate questions into a more appropriate difficulty range, we evaluate its effect on two types of failed drafts. The first type is shortcut-prone drafts, which are solved too quickly because the adversary can exploit an early identifying route or prior binding. The second type is initially unsolved drafts, for which the adversary fails to produce a successful trajectory before refinement because the question is over-fuzzed, ambiguous, or underspecified. We first run the adversary agent on the original drafts. For shortcut-prone drafts, we repair the earliest observed shortcut-prone clue. For initially unsolved drafts, we narrow over-fuzzed clues, remove ambiguous facts, or restore necessary constraints. We then rerun the adversary under the same search setting and compare the trajectory signatures before and after refinement.

Draft type Version 𝛀 𝑻hit 𝒑prior(%) Shortcut-prone Original 33.9 12.4 17.0

Refined 82.7 31.4 12.0 Initially unsolved Original – – –

Refined 123.0 50.2 13.0

- Table 10. Effect of adversarial refinement. Metrics for original initially unsolved drafts are unavailable because they do not produce successful trajectories before refinement.

As shown in Table 10, refinement substantially improves shortcut-prone drafts. After refinement, Ω rises from 33.9 to 82.7, 𝑇hit is delayed from 12.4 to 31.4, and 𝑝prior decreases from 17.0 to 12.0. This indicates that refinement suppresses early shortcut routes and induces longer pre-answer search. For initially unsolved drafts, the original versions do not produce successful trajectories, while the refined versions become solvable and still retain substantial search difficulty, with Ω = 123.0 and 𝑇hit = 50.2. These results show that adversarial refinement is not merely a difficulty-increasing step; it calibrates questions by repairing both shortcut-prone and initially unsolved drafts.

- 5.3. Dataset Difficulty Comparison Section 2.4 uses trajectory signatures to diagnose existing open-source deep-search datasets and shows that many of them expose the answer early during successful solving. We now include FORT in the same diagnostic setting to examine whether shortcut-resistant synthesis alleviates this issue. For a fair comparison, we do not rely on the released trajectories of different datasets, which may be generated by different agents under different search settings. Instead, for each dataset, we randomly sample 200

Source 𝛀 𝑻hit 𝒑prior(%) InfoSeek [37] 20.6 5.7 2.0 MiroVerse-Voyager [30] 30.6 5.9 5.7 DeepDive [17] 47.7 15.5 7.4 DeepResearch-9K [19] 47.8 3.4 27.2 OpenSeeker [9] 84.7 9.3 31.9 REDSearcher [6] 92.1 18.7 11.8

###### FORT 141.0 46.9 11.0

Table 11. Dataset difficulty comparison under trajectory signatures.

questions and re-evaluate them with the same strong search agent under the same retrieval budget. Metrics are computed over successful trajectories.

- Table 11 shows that FORT yields the highest solving cost and latest answer exposure under the same diagnostic setting. Compared with the strongest open-source baseline, REDSearcher, Ω increases from 92.1 to 141.0, and 𝑇hit increases from 18.7 to 46.9. This indicates that FORT extends the pre-answer search prefix rather than merely inducing post-hit verification or detours. Meanwhile, 𝑝prior remains comparable to REDSearcher, suggesting that the increased solving cost is not driven by more explicit prior-bound behavior.

- 5.4. Trajectory-Level Proxies for Difficulty Factors To examine how the difficulty factors in Section 2 are reflected in realized search behavior, we conduct a trajectory-level diagnosis on successful question–trajectory pairs. The exact values of the theoretical quantities in Section 2 are not feasible to compute in open-domain web search. In particular, exact

computation of 𝑠(P), 𝑀ev(P), and dep(P) would respectively require enumerating the full answer space, all identifying constraint subsets, and valid evidence-acquisition routes under the retrieval interface Σ.

The solver-side prior utility 𝑈𝜋0(𝑞) would further require characterizing the model’s prior knowledge and the counterfactual search cost it saves. These objects are generally not enumerable or directly observable for real web-search tasks.

We therefore do not treat the following measurements as direct estimates of the theoretical quantities. Instead, we use them as trajectory-level proxies for the realized effects of the difficulty factors analyzed in our framework. Specifically, we randomly sample 200 successful question–trajectory pairs from open-source deep-search data and 200 from FORT. We use GPT-5.5 to inspect each complete question and trajectory, decompose the question into atomic clues, and align each clue with the corresponding search steps and retrieved evidence. Based on this annotation, we define four observable proxies.

For subset selectivity, we examine whether the search evidence associated with each individual clue already narrows the candidate space to a very small set. We mark a clue as low-width if the clue-specific search evidence yields only one or two plausible candidate answers. We report the percentage of such clues as 𝑅low. A lower 𝑅low indicates that individual clues are less likely to identify the answer on their own, making small identifying subsets less available in realized search.

For evidence dispersion, we assign each verified clue to one primary evidence source that supports it. If the same web page or evidence source verifies multiple clues, it is counted only once. We then compute the normalized evidence dispersion:

𝑅ev = 𝑀ev/𝑛𝑐 × 100%, (19) where 𝑀ev is the number of distinct primary evidence sources and 𝑛𝑐 is the number of clues in the question.

𝑅low

𝑅ev

𝑝prior

𝐶dep

Ω

Data

(%) ↓

(%) ↑

(%) ↓

↑

↑

Open-source 73.7 55.2 78.7 3.1 27.0 FORT 139.6 40.2 90.2 5.9 16.0

- Table 12. Trajectory-level proxies for realized difficulty factors on 200 successful question–trajectory pairs from each data source. Ω denotes the empirical average solving cost. 𝑅low denotes the percentage of clues whose clue-specific search evidence yields only one or two plausible candidate answers. 𝑅ev denotes normalized evidence dispersion. 𝐶dep denotes the retrieval cost along the most costly realized pre-answer dependency chain. 𝑝prior denotes the fraction of trajectories where the model proposes the gold answer before retrieved evidence exposes it.

A higher 𝑅ev indicates more dispersed evidence, while a lower value suggests that multiple clues are co-covered by a small number of sources.

For dependency depth, we do not estimate the oracle quantity dep(P). Instead, we measure the retrieval cost of the most costly realized dependency chain, denoted by 𝐶dep. We first keep only clue-relevant search steps before the first answer hit. We then check whether each step uses an entity name, attribute, date, title, alias, or intermediate fact first exposed by earlier retrieved evidence. If a later search depends on information exposed by an earlier step, the two steps are considered part of the same realized dependency chain. Among all realized dependency chains, we select the one with the highest retrieval cost and count the actual retrieval steps needed to complete it. A higher 𝐶dep indicates stronger serial dependence and greater retrieval effort before the answer is exposed.

For solver-side prior binding, we reuse the prior-shortcut indicator defined in Section 2.4. We report the fraction of trajectories where GPT-5.5 judges, based on semantic equivalence rather than string matching, that the model proposes the gold answer before retrieved evidence exposes it, denoted by 𝑝prior.

- Table 12 shows that FORT moves all trajectory-level proxies in the desired direction: compared with open-source deep-search data, it induces higher retrieval effort, less over-selective individual clues, more dispersed supporting evidence, more costly pre-answer dependency chains, and fewer explicit prior-bound answers. These proxies provide an operational connection between the theoretical difficulty factors and realized search behavior. Although they are not exact estimates of the quantities in Section 2, they indicate that the difficulty induced by FORT is not merely a matter of longer trajectories; rather, FORT makes the factor-level conditions behind shortcut opportunities less favorable, thereby making cheap identifying routes less available and evidence acquisition more necessary during actual search.

### 6. Conclusion

We introduce FORT-Searcher, a deep search agent trained with FORT, a shortcut-resistant synthesis framework for generating search-heavy supervision. The central insight is that apparent structural complexity does not necessarily translate into realized search difficulty: a question with many clues, long latent chains, or dense evidence graphs may still be solved cheaply if the solver can exploit evidence co-coverage, single-clue selectivity, exposed constants, or prior-knowledge binding. High-quality deep search supervision should therefore be judged not only by the complexity of its intended structure, but by whether the intended evidence-acquisition process remains necessary during actual agent search.

FORT operationalizes this view by controlling shortcut risks across entity selection, evidence graph construction, question formulation, and adversarial refinement. These controls produce trajectories in which the model spends more effort on answer discovery rather than relying on early answer exposure, post-hit verification, or parametric priors. Using these trajectories, we train FORT-Searcher with

supervised fine-tuning alone. We leave the integration of reinforcement learning with FORT trajectories to future work. Empirically, FORT-Searcher achieves the best overall performance among comparablesize open-source search agents, showing that shortcut-resistant supervision can effectively improve deep search behavior.

Looking forward, deep search agents should move beyond longer web-only trajectories toward more efficient tool-augmented search and more complex search-grounded tasks. A richer tool harness can support more targeted evidence acquisition, structured interaction with external environments, and reliable intermediate computation. Meanwhile, harder search-based tasks require agents to integrate heterogeneous evidence, resolve conflicting information, and make well-supported decisions under uncertainty. Shortcutaware data construction and trajectory-level diagnostics could provide a useful insight for training and evaluating this broader class of search agents.

### References

- 1 Raviteja Anantha, Soheil Hor, Teodor Nicola Antoniu, and Layne C Price. Nanoflux: Adversarial dual-llm evaluation and distillation for multi-domain reasoning. arXiv preprint arXiv:2509.23252,

- 2025.

2 Anthropic. Claude Opus 4.7. https://www.anthropic.com/news/claude-opus-4-7,

- 2026. Released April 16, 2026. Accessed: 2026-05-25.

- 3 Guoxin Chen, Minpeng Liao, Peiying Yu, Dingmin Wang, Zile Qiao, Chao Yang, Xin Zhao, and Kai Fan. C-3PO: Compact plug-and-play proxy optimization to achieve human-like retrieval-augmented generation. In Proceedings of the 42nd International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=hlpwAmQ4wr.
- 4 Guoxin Chen, Zile Qiao, Xuanzhong Chen, Donglei Yu, Haotian Xu, Wayne Xin Zhao, Ruihua Song, Wenbiao Yin, Huifeng Yin, Liwen Zhang, et al. Iterresearch: Rethinking long-horizon agents via markovian state reconstruction. arXiv e-prints, pages arXiv–2511, 2025.
- 5 Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, et al. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations. arXiv preprint arXiv:2506.13651, 2025.
- 6 Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao, Cheng Xiang, Shengchao Hu, et al. Redsearcher: A scalable and cost-efficient framework for long-horizon search agents. arXiv preprint arXiv:2602.14234, 2026.
- 7 Zikun Cui, Tianyi Huang, Chia-En Chiang, and Cuiqianhe Du. Toward verifiable misinformation detection: A multi-tool llm agent framework. In Proceedings of the 2025 International Conference on Generative Artificial Intelligence for Business, pages 179–185, 2025.
- 8 Yuwen Du, Rui Ye, Shuo Tang, Keduan Huang, Xinyu Zhu, Yuzhu Cai, and Siheng Chen. Openseekerv2: Pushing the limits of search agents with informative and high-difficulty trajectories. arXiv preprint arXiv:2605.04036, 2026.
- 9 Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, and Siheng Chen. Openseeker: Democratizing frontier search agents by fully open-sourcing training data. arXiv preprint arXiv:2603.15594, 2026.
- 10 Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.

- 11 Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Tong Zhao, Xiaoxi Li, Zheng Liu, and Zhicheng Dou. Agentfugue: Agent scaling for long-horizon tasks through collective reasoning, 2026. URL https://arxiv.org/abs/2605.24486.
- 12 Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Ziliang Zhao, Jiejun Tan, Zheng Liu, and Zhicheng Dou. Sam: State-adaptive memory for long-horizon reasoning agent, 2026. URL https://arxiv.org/abs/2605.24468.
- 13 Ailin Huang, Ang Li, Aobo Kong, Bin Wang, Binxing Jiao, Bo Dong, Bojun Wang, Boyu Chen, Brian Li, Buyun Ma, et al. Step 3.5 flash: Open frontier-level intelligence with 11b active parameters. arXiv preprint arXiv:2602.10604, 2026.
- 14 Tencent Hunyuan. Hy3 preview: The First Step in Rebuilding the Hy model. https://hy.tence nt.com/research/hy3, 2026. Accessed: 2026-05-26.
- 15 Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, and Ofer Hadar. Vault: Vigilant adversarial updates via llm-driven retrieval-augmented generation for nli. arXiv preprint arXiv:2508.00965, 2025.
- 16 Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.
- 17 Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, and Yuxiao Dong. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn rl. arXiv preprint arXiv:2509.10446, 2025.
- 18 OpenAI. GPT-5.5 System Card. https://deploymentsafety.openai.com/gpt-5-5/,

2026. Accessed: 2026-05-25.

- 19 Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, and Tu Vu. Sealqa: Raising the bar for reasoning in search-augmented language models. arXiv preprint arXiv:2506.01062, 2025.
- 20 Zile Qiao, Guoxin Chen, Xuanzhong Chen, Donglei Yu, Wenbiao Yin, Xinyu Wang, Zhen Zhang, Baixuan Li, Huifeng Yin, Kuan Li, Rui Min, Minpeng Liao, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webresearcher: Unleashing unbounded reasoning capability in long-horizon agents,

2025. URL https://arxiv.org/abs/2509.13309.

- 21 Zhiyu Shen, Jiyuan Liu, Yunhe Pang, and Yanghui Rao. Hopweaver: Cross-document synthesis of high-quality and authentic multi-hop questions. arXiv preprint arXiv:2505.15087, 2025.
- 22 Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- 23 Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in LLMs via reinforcement learning,

2025. URL https://arxiv.org/abs/2503.05592.

- 24 Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. Smart-Searcher: Incentivizing the dynamic knowledge acquisition of LLMs via reinforcement learning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 13572–13586, Suzhou, China, November 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-emnlp.731. URL https://aclanthology.org/2025.findings-emnlp.731/.

- 25 Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, and Ji-Rong Wen. Simpledeepsearcher: Deep information seeking via web-powered reasoning trajectory synthesis. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 13705–13720. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025. findings-emnlp.739/.
- 26 Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, et al. Webshaper: Agentically data synthesizing via informationseeking formalization. arXiv preprint arXiv:2507.15061, 2025.
- 27 Alibaba Qwen Team. Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog ?id=qwen3.5, 2026. Accessed: 2026-05-26.
- 28 Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.
- 29 Meituan LongCat Team, Anchun Gui, Bei Li, Bingyang Tao, Bole Zhou, Borun Chen, Chao Zhang, Chen Gao, Chen Zhang, Chengcheng Han, et al. Longcat-flash-thinking-2601 technical report. arXiv preprint arXiv:2601.16725, 2026.
- 30 MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, et al. Mirothinker: Pushing the performance boundaries of opensource research agents via model, context, and interactive scaling. arXiv preprint arXiv:2511.11793, 2025.
- 31 MiroMind Team, S Bai, L Bing, L Lei, R Li, X Li, X Lin, E Min, L Su, B Wang, et al. Mirothinker-1.7 & h1: Towards heavy-duty research agents via verification. arXiv preprint arXiv:2603.15726, 2026.
- 32 Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- 33 Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.
- 34 Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
- 35 Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.
- 36 Xiaojun Wu, Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Hui Xiong, Jia Li, and Jian Guo. Think-on-graph 3.0: Efficient and adaptive llm reasoning on heterogeneous graphs via multi-agent dual-evolving context retrieval. arXiv preprint arXiv:2509.21710, 2025.
- 37 Ziyi Xia, Kun Luo, Hongjin Qian, and Zheng Liu. Open data synthesis for deep research. arXiv preprint arXiv:2509.00375, 2025.
- 38 Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering.

- In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.
- 39 Z.ai. GLM-4.7-Flash. https://github.com/zai-org/GLM-4.7-Flash, 2025. Accessed: 2026-05-26.
- 40 Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.
- 41 Ziyang Zeng, Heming Jing, Jindong Chen, Xiangli Li, Hongyu Liu, Yixuan He, Zhengyu Li, Yige Sun, Zheyong Xie, Yuqing Yang, et al. Optimizing generative ranking relevance via reinforcement learning in xiaohongshu search. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 2551–2561, 2026.
- 42 Gongrui Zhang, Jialiang Zhu, Ruiqi Yang, Kai Qiu, Miaosen Zhang, Zhirong Wu, Qi Dai, Bei Liu, Chong Luo, Zhengyuan Yang, et al. Infoagent: Advancing autonomous information-seeking agents. arXiv preprint arXiv:2509.25189, 2025.
- 43 Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, and Xinyu Wang. Argus: Evidence assembly for scalable deep research agents. arXiv preprint arXiv:2605.16217, 2026.
- 44 Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, et al. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. arXiv preprint arXiv:2504.19314, 2025.

### A. Notation Summary

Notation Meaning 𝑞 = (X, C𝑞, Σ) A multi-constraint agentic retrieval task, consisting of an answer space, a

question-constraint set, and a retrieval interface.

X The answer space, i.e., the set of candidate entities or values from which the gold

answer is selected.

C𝑞 The set of constraints expressed by question 𝑞. Each constraint corresponds to one clue-like condition that the answer should satisfy.

P A subset of question constraints, P ⊆ C𝑞, used to describe partial evidence

requirements. Σ The retrieval interface available to the agent, such as a web search engine. Ans(P) The candidate pool that remains after applying a subset of constraints P ⊆ C𝑞. I𝑞 The set of identifying constraint subsets, i.e., subsets P ⊆ C𝑞 such that Ans(P) =

{𝑦★}. 𝑦★ The unique gold answer satisfying the full constraint set C𝑞. 𝜏 A realized search trajectory, represented as a sequence of retrieval queries before

the solver commits to an answer. Πpost The class of no-prior, no-guessing solvers used to define the pure-posterior cost 𝐷post(𝑞). 𝐷post(𝑞) The pure-posterior cost: the minimum expected retrieval cost required by a noprior, no-guessing solver to answer 𝑞. 𝜋0 A concrete solver whose realized behavior may exploit shortcuts or prior knowledge. 𝑈𝜋0(𝑞) The solver-side cost reduction available to solver 𝜋0, for example through prior knowledge or shortcut exploitation. Ω(𝑞,𝜋0) The realized retrieval cost of solver 𝜋0 on task 𝑞, represented by the decomposition 𝐷post(𝑞) −𝑈𝜋0(𝑞). 𝑄Σ(P) The length of the shortest valid evidence-acquisition route that verifies the identifying subset P for the gold answer under retrieval interface Σ. 𝑄★Σ The cheapest identifying route over all identifying subsets: minP∈I

𝑞 𝑄Σ(P). 𝑠(P) The selectivity of constraint subset P, measured by the size of the remaining candidate pool |Ans(P)|. 𝑀ev(P) The minimum number of evidence-acquisition steps needed to verify that 𝑦★ satisfies all constraints in P, ignoring query executability. dep(P) The minimum dependency depth required by a valid evidence-acquisition route for verifying P. 𝑇hit The answer hit time: the first turn at which the gold answer, or its normalized

alias, appears in the retrieved evidence or solver observation. Ω The empirical average solving cost measured over a set of realized trajectories. 𝑝prior The empirical prior-shortcut rate: the fraction of trajectories classified as prior-

shortcut cases, where the solver proposes the answer before acquiring sufficient search evidence.

Table 13. Summary of the main notation used in the shortcut-aware difficulty framework and trajectory diagnostics.

### B. Formal Details for the Difficulty Framework

- B.1. Pure-Posterior Policy Class A policy 𝜋 belongs to the pure-posterior policy class Πpost if it satisfies three restrictions.

Executable queries. At each history ℎ𝑡, the policy can issue only executable queries. Let K(ℎ𝑡) be the constants available from the question and from previously retrieved evidence, and let Const(𝜃) be the constants needed to instantiate query 𝜃. Then any query 𝜃𝑡 issued by 𝜋 must satisfy

Const(𝜃𝑡) ⊆ K(ℎ𝑡). (20)

No-guessing termination. The policy may terminate with answer 𝑥 only after its accumulated evidence 𝐸 verifies an identifying subset for 𝑥. Formally, there must exist P ⊆ C𝑞 such that

Ans(P) = {𝑥} and {𝑥} × P ⊆ 𝑉𝐸. (21)

No problem-specific prior knowledge. The policy may use basic reading and composition over retrieved evidence, but it cannot use entity-level prior knowledge specific to the current question. In particular, it cannot name the target from memory or use a memorized target profile to bypass evidence acquisition.

#### B.2. Valid Evidence-Acquisition Routes

Let P ∈ I𝑞 be an identifying subset. A valid evidence-acquisition route for (𝑦★, P) under Σ is an ordered sequence of queries

𝜽 = (𝜃1, . . . ,𝜃𝑚). (22)

Executing 𝜃𝑗 returns evidence RetrieveΣ(𝜃𝑗). Let ℎ𝑗−1 be the history containing the question and all evidence retrieved before 𝜃𝑗. The route is valid if every query is executable when issued:

𝜃𝑗 ∈ Exec(ℎ𝑗−1), 𝑗 = 1, . . . , 𝑚, (23) and if the accumulated evidence verifies that 𝑦★ satisfies all constraints in P:

𝑚

{𝑦★} × P ⊆ 𝑉𝐸, 𝐸 =

RetrieveΣ(𝜃𝑗). (24)

𝑗=1

The route length is |𝜽| = 𝑚, and 𝑄Σ(P) = min |𝜽| : 𝜽 is valid for (𝑦★, P) under Σ . (25)

### C. Proofs for the Difficulty Framework

- C.1. Proof of the Structural Lower Bound

- Proposition 2. For any task 𝑞,

𝐷post(𝑞) ≥ 𝑄★Σ, 𝑄★Σ = min P∈I𝑞

𝑄Σ(P). (26)

Proof. Consider any successful trajectory generated by a policy 𝜋 ∈ Πpost. By the no-guessing restriction in Appendix B.1, the policy can terminate with 𝑦★ only after its accumulated evidence verifies an identifying subset P ∈ I𝑞. By the executable-query restriction, every query in the trajectory is executable when issued. Therefore, the successful trajectory realizes a valid evidence-acquisition route for (𝑦★, P) under Σ. Its length is at least 𝑄Σ(P), and hence at least

𝑄Σ(P) = 𝑄★Σ.

min

P∈I𝑞

Taking expectation over pure-posterior trajectories and then the infimum over Πpost gives 𝐷post(𝑞) ≥ 𝑄★Σ. □

- C.2. Proof of the Component Lower Bound This subsection proves Eq. (12). We use the definitions of 𝑄Σ(P), 𝑀ev(P), and dep(P) from Section 2.2.

- Proposition 3. For any identifying subset P ∈ I𝑞, 𝑄Σ(P) ≥ max 𝑀ev(P),dep(P) . (27)

Proof. Let 𝜽★ be a shortest valid evidence-acquisition route for (𝑦★, P) under Σ. By definition,

|𝜽★| = 𝑄Σ(P). (28)

First, because 𝜽★ is a valid route, it retrieves enough evidence to verify that 𝑦★ satisfies all constraints in P. Therefore, 𝜽★ is a feasible evidence-acquisition sequence for the problem defining 𝑀ev(P). Since 𝑀ev(P) is the minimum number of evidence-acquisition steps needed for such verification, we have

𝑄Σ(P) = |𝜽★| ≥ 𝑀ev(P). (29)

Second, depth(𝜽★) is the length of the longest dependency chain among the queries in 𝜽★. Since this dependency chain is contained in the route itself, the route cannot be shorter than its longest dependency chain:

|𝜽★| ≥ depth(𝜽★). (30) By the definition of dependency depth,

dep(P) = min

𝜽 valid for P

depth(𝜽), (31) so for the particular valid route 𝜽★,

depth(𝜽★) ≥ dep(P). (32) Combining Eqs. (28), (30), and (32) gives

𝑄Σ(P) ≥ dep(P). (33)

Finally, Eqs. (29) and (33) together imply

𝑄Σ(P) ≥ max 𝑀ev(P),dep(P) . (34) □

This bound concerns 𝑀ev and dep because they are route-length constraints for a fixed identifying subset. By contrast, 𝑠(P) does not directly lower-bound 𝑄Σ(P). Instead, it affects whether a small subset can become identifying and therefore enter the minimization that defines 𝑄★Σ.

- C.3. Collapse of the Structural Lower Bound

Corollary 1. If there exists an identifying subset P ∈ I𝑞 that can be verified by a single initially executable query, then

𝑄★Σ = 1. (35)

Proof. If P ∈ I𝑞 can be verified by a single initially executable query, then there exists a valid route 𝜽 = (𝜃1) for (𝑦★, P). Therefore, 𝑄Σ(P) = 1. Since

##### 𝑄Σ(P′),

𝑄★Σ = min

P′∈I𝑞

we have 𝑄★Σ ≤ 1. Under the route-cost definition, at least one retrieval step is required to verify evidence, so 𝑄★Σ ≥ 1. Thus, 𝑄★Σ = 1. □

### D. Shortcut Diagnostic Cases

This appendix provides representative trajectory-level examples for the four shortcut risks discussed in Section 2.3. Each case shows how an apparently multi-constraint question can be solved through a cheaper identifying route or through solver-side prior binding. These cases are sampled from open-source data.

Case 1: Evidence Co-coverage Shortcut

Question. A corporate entity in the Pacific region, which was formally established to consolidate various holdings, traces its commercial lineage to entrepreneurial activities in Apia that began in the early decades of the 1900s. These foundational ventures were launched by a prominent individual whose career was primarily built in the local hospitality industry. What is the formal name of the resulting holding company that was incorporated?

#### Gold answer. Grey Investment Group. Shortcut type. Evidence co-coverage. Trigger turn. Round 1.

- Explanation. The shortcut is triggered in Round 1 by the query Apia Samoa early 1900s holding company founded by hospitality entrepreneur. A single search-result snippet directly surfaces Grey Investment Group and simultaneously states that it is a Samoan investment enterprise, specializes in tourism and hospitality, and has assets across the Pacific region. Since this single evidence item exposes the answer entity together with multiple answer-side facts before the model names the answer, it collapses several intended evidence-acquisition steps into a single retrieval result.

Figure 5. Diagnostic example of an evidence co-coverage shortcut.

Case 2: Single-clue Selectivity Shortcut

Question. A specific calendar year in the middle of the twentieth century can be defined by four converging threads. First, it is the year a foundational international convention on the legal protection of displaced persons was opened for signature, administered by a UN agency. Second, that same UN agency has a mandate that significantly overlaps with a much larger intergovernmental body—one that formally became a UN-related organization more than five decades after the first human-crafted object reached another celestial body. Third, in this identical year, on a day in the final week of June, a major U.S. television network publicly transmitted its first regular programming using a full-color system. Finally, in that same twelve-month period, in a nation through which the Paraná River flows, a documented submission grappling contest took place between two high-profile figures from the world of martial arts. Determine which single year satisfies all four of these conditions.

Gold answer. 1951. Shortcut type. Single-clue selectivity.

- Trigger turn. Round 1.

Explanation. The shortcut is triggered by the Round 1 query first regular full-color programming US television network final week June. This query uses only one thread of the question, namely the full-color television broadcast clue. The retrieved results immediately identify June 1951 events, and the model then treats 1951 as the main candidate before checking the other threads. Thus, a single clue is already selective enough to surface the gold answer year, while the remaining constraints mainly serve as post-hoc verification.

Figure 6. Diagnostic example of a single-clue selectivity shortcut.

Case 3: Exposed-constant Shortcut

Question. Identify an individual who held a regional leadership role in a major North American forestry labor union, an acronym that could also denote an international water association or a wrestling organization, during the latter half of the 1980s. This person was publicly associated with a city whose name derives from an Indigenous word for “grizzly bear,” located within a tourism region in the southern interior of the westernmost Canadian province. When the provincial government announced new labor legislation in the mid-1980s, this individual publicly stated that the people of this province want to work, not protest. The same city hosts an airport that provides temperature observations used to model evaporation rates for a glacially formed lake in a north-south oriented valley. This individual shares the same full name as a professional pickleball player from Austin, Texas. Who is this union leader?

Gold answer. Jack Munro. Shortcut type. Exposed constant.

- Trigger turn. Round 2.

- Explanation. The shortcut is triggered in Round 2 by the query IWA union leader Kelowna 1980s "want to work not protest". The question surface exposes a highly searchable phrase from the answer entity’s public statement: “want to work, not protest.” The model reuses this phrase nearly verbatim, and the retrieved results directly name Jack Munro. Because the searched phrase is a distinctive attribute of the target person rather than a generic intermediate clue, the exposed constant directly drives answer retrieval.

Figure 7. Diagnostic example of an exposed-constant shortcut.

Case 4: Prior-knowledge Binding Shortcut

Question. In a coastal town where a mechanical marvel from the age of telegraphy still stands, the town’s historical associations trace back to a Roman conqueror. That conqueror later defied the republic by crossing a boundary river, setting in motion conflicts that culminated in a naval battle whose aftermath led to the demise of a legendary queen. Who was that queen?

#### Gold answer. Cleopatra. Shortcut type. Prior-knowledge binding. Trigger turn. Round 0.

Explanation. The shortcut occurs before any tool call. In Round 0, the model already names the gold answer by stating that the legendary queen may be Cleopatra and links Caesar’s crossing to the civil war, Actium, and Cleopatra’s death. The first retrieved observation containing “Cleopatra” appears only later, after a query that already includes the answer name. The decisive answer binding therefore comes from the model’s prior knowledge rather than evidence acquired during the search trajectory.

Figure 8. Diagnostic example of a prior-knowledge binding shortcut.

