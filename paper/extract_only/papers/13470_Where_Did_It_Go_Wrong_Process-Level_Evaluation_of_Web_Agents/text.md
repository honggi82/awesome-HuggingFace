# arXiv:2606.15673v1[cs.AI]8Apr2026

### Where Did It Go Wrong? Process-Level Evaluation of Web Agents with Semantic State Tracking

Jiwan Chung1 JiHyuk Byun1 Vibhav Vineet2 Seon Joo Kim1 1Yonsei University 2Microsoft Research jiwan.chung.research@gmail.com https://jiwanchung.github.io/webstep/

#### Abstract

Web agents act through long interaction sequences, yet existing benchmarks evaluate only terminal success, discarding all process information and offering little guidance on improvement. In this work, we conduct a process-level analysis of web agents. We introduce WEBSTEP, a benchmark of 1,800 task instances with controlled difficulty and automatic semantic state tracking. Each website exposes a deterministic semantic MDP alongside the GUI: the agent operates on the interface, while the environment records high-level states and transitions in the background, enabling fine-grained analysis without manual annotation. Based on the semantic trajectory, we first show that process metrics reveal differences invisible to outcome evaluation: three agents whose success rates cluster within 31–33% diverge in exploration reach versus execution accuracy. Then, decomposing by skill characterizes the nature of these differences, exposing opposite per-skill rankings hidden within the same website: e.g., on Housing, OpenAI CUA outperforms Qwen3.5 by 23.7% on commit actions yet underperforms it by 15.6% on filtering, pinpointing a concrete skill to improve even within a domain. Bifurcation analysis further localizes the decisive error that loses the task and shows that this error is agent-specific rather than shared. Finally, these differences widen as tasks grow harder: success rate is similar on easy tasks but separates sharply as exploration becomes more demanding. Our process-level analysis opens a new avenue in web agent evaluation, providing fine-grained and actionable insight into where and how each agent should be improved.

#### 1 Introduction

Web agents automate interactive digital tasks such as shopping, booking, and document management (Awadallah et al., 2025; Qin et al., 2025; Gupta et al., 2026). Unlike single-turn instruction following, these tasks require multi-step navigation, information gathering across pages, and dependent action execution (Zhou et al., 2024; Koh et al., 2024). Thus, the final outcome alone does not fully characterize agent behavior: the trajectory itself carries meaningful information about the interaction process.

Most existing web agent benchmarks, however, evaluate agents primarily by terminal success (He et al., 2024; Mialon et al., 2023; Deng et al., 2023; Xue et al., 2025). This outcomeonly view compresses an entire interaction trajectory into a single binary result, obscuring qualitatively different failure modes. As illustrated in Figure 1, an agent that reaches the correct colleague’s message thread but sends the wrong emoji receives the same score as one that never finds the relevant page at all. Terminal success therefore provides little actionable insight into where an agent failed or which capability limited performance, as it does not support accurate credit assignment (Pignatelli et al., 2024; Lu` et al., 2025; Gritta et al., 2026).

In this work, we introduce WEBSTEP, a benchmark for process-level evaluation of web agents. The core idea is to construct each website as a semantic Markov Decision Process (MDP) dual: agents interact only with the rendered GUI, while the benchmark records the

[Figure 1]

🔥

TASK Search for "watching" in messages. Find the result from Finley Clark and add a reaction to it.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Click ‘Sage Foster’ Back to Search Result Click ‘Avery Martinez’ React 🔥

Search ‘watching’

[Figure 13]

[Figure 14]

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

| | | |
|---|---|---|
| |[Figure 27]| |

###### Exploration Phase ✗

###### Fara

✗

Exploration Failure Back to Search

[Figure 28]

Search ‘watching’ Open ‘Sage Foster’ Open ‘Avery Martinez’ React 🔥

[Figure 29]

[Figure 30]

###### UI-TARS

✗

[Figure 31]

Search ‘watching’ React 👍 Remove 👍 Timeout

[Figure 32]

Time Out Open ‘Finley Clark’ loop

###### Execution Phase ✗

###### GUI-Owl

✗

Execution Failure

Search ‘watching’ Open ‘Finley Clark’ Open thread Open React toggle React ✅

[Figure 33]

- Figure 1: Terminal outcome alone cannot distinguish qualitatively different failures. In the task of finding Finley Clark’s message and reacting with a :fire: emoji, Fara fails to locate the correct target, while GUI-Owl reaches the correct thread but applies the wrong reaction.

corresponding semantic state transitions induced underneath. Because the GUI is rendered directly from the semantic model, the full semantic trajectory is recovered exactly, enabling automatic state tracking and process-level evaluation without manual annotation.

Using WEBSTEP, we show that process-level evaluation extends web agent evaluation from ranking to diagnosis. It separates agents that appear similar under terminal success by disentangling exploration from execution (Section 4.1), attributing their differences to specific skills (Section 4.2), localizing the step at which unsuccessful trajectories first diverge (Section 4.3), and identifying the task-complexity regimes in which these failures become most pronounced (Section 4.4). In turn, this yields concrete targets for improvement: which skills to strengthen, which temporal failure patterns to correct, and which complexity regimes to emphasize during training and evaluation.

Our main contributions are as follows:

- 1. We introduce WEBSTEP, a benchmark of 10 self-hosted websites and 1,800 tasks, each paired with a semantic MDP dual that tracks structured states and transitions.
- 2. We develop a process-level evaluation framework that decomposes agent behavior into exploration success, execution outcomes, and skill invocation patterns, and supports trajectory comparison at shared semantic states to localize failure modes.
- 3. We present a process-level analysis of existing web agents, showing that models with similar terminal success rates exhibit qualitatively distinct failure signatures.

#### 2 Process-level evaluation via semantic MDP duals

Process-level evaluation requires access to the semantic effects of agent actions, yet web agents interact only through low-level operations such as clicking, typing, and scrolling. As shown in Figure 2, we address this mismatch by mapping each raw GUI trajectory to an aligned semantic trace that records which items were visited, what information became available, and which high-level action was taken. This is enabled by constructing each website as a semantic MDP dual: the agent acts only on the graphical user interface, while the benchmark tracks the corresponding semantic state transitions underneath.

[Figure 34]

Task: ‘Search for repositories about ‘database’. Go to ‘ultra-proxy’, star it, then create an issue titled ‘Improve CI/CD pipeline.’’

Search Results Wrong Page Returned to Search Target Target Repo Issues Form

|Search Results<br><br>[Figure 35]|[Figure 36]<br><br>|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]| |[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]| |Repository Page (‘ultra-proxy’) target ✓<br><br>[Figure 45]|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|(Starred)<br><br>[Figure 49]<br><br>star ✓| |[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>not submitted<br><br>✗|
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

Web

Environment

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Agent Action

scroll scroll

Scroll

click(1255, 462) type(Improve...)

click(179, 495) history_back click(272, 269)

click(1272, 597)

click(294, 180)

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

MDP Observer (Automatic GUI-to-Semantic Conversion)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

ViewRepo(020) GoBack ... ViewRepo(023) StarRepo(023) CreateIssuetyped but ✗✗

Semantic

Transition

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

###### Exploration

Skill Invocation Efficiency

[Figure 79]

✓ x2

- - REPO-020 (Wrong) visited.
- - REPO-023 target entity visited. ✓ Exploration Status: PASS

10

ViewRepo StarRepo CreateIssue

GUI Actions

✓ ✗

5 3

Semantic Transitions Oracle Transitions

- Figure 2: Overview of WEBSTEP. Each visual website is paired with a semantic MDP dual that converts raw GUI actions into interpretable semantic transitions. For example, clicking and typing merge to transitions such as VIEWREPO, STARREPO, and CREATEISSUE, making explicit which item was visited and which semantic action was taken. These traces enable automatic process-level analyses such as exploration, skill invocation, and step efficiency.

- 2.1 Semantic MDP For each website, we define a deterministic semantic MDP M = (S, A, T, ρ0), where:

- • S: a structured semantic state space. Each state is decomposed as s = (sp, si), where:

- – sp: the current interface position (e.g., URL, search and filter settings, pagination and open modals);
- – si: the item attributes revealed to the agent (e.g., a product’s price, score, or seller).

- • A: a typed semantic action space abstracting away raw coordinate-level interactions (e.g., issuing a search, opening a detail page, or executing the final task action);
- • T: a deterministic transition function that maps each state-action pair (st, at) to the next semantic state st+1 = T(st, at);
- • ρ0: the task and environment configurations, elaborated in Section 2.2. For each agent run on a task, we record the resulting MDP trace τ

τ = (s0, a0, s1, a1, . . . , sT), st+1 = T(st, at), (1) which serves as the basis for process-level evaluation.

GUI–MDP coupling. Each website is implemented so that the semantic MDP is the source of truth for all behavior. When the web agent performs a GUI action sequence (e.g., searching, filtering, or opening a detail page) the frontend forwards it to the semantic model instead of updating the page on its own. The semantic model applies the corresponding state transition, determines what information is visible in the new state, and the page is then re-rendered from that state. As a result, the task-relevant GUI trajectory is a direct execution of the semantic MDP, which allows the full semantic trajectory to be recovered exactly.

- 2.2 Task and world generation

Given a website’s semantic MDP, each task consists of two jointly generated components: a natural-language instruction and a corresponding world, defined as a set of items with concrete attribute values (e.g., brand, price, seller, or delivery option). Each task is constructed by combining a task template (e.g., ”Add a bag with [Constraints] to the cart”) with

###### Deterministic Tasks Process-eval Hard negative

Online-Mind2Web (Xue et al., 2025) ✗ 300 ✗ ✗ WebVoyager (He et al., 2024) ✗ 643 ✗ ✗ WebWalkerQA (Wu et al., 2025) ✗ 680 ✗ ✗ AssistantBench (Yoran et al., 2024) ✗ 214 ✗ ✗ WebArena (Zhou et al., 2024) ✓ 812 ✗ ✗ VisualWebArena (Koh et al., 2024) ✓ 910 ✗ ✗ WorkArena++ (Boisvert et al., 2024) ✓ 682 ✗ ✗ Mind2Web-Live (Pan et al., 2024) ✗ 542 Key node ✗

###### WEBSTEP ✓ 1,800 MDP ✓

Table 1: Comparison with existing web agent benchmarks. Deterministic: self-hosted deterministic websites. Process eval: Key node uses manual milestones; MDP enables automatic stage decomposition and skill attribution. Hard negative: controlled distractors.

a randomly sampled set of item-attribute constraints (e.g., price ≤ 300 and rating ≥ 3.5). The constraints are instantiated into text instruction, and the world is then populated to satisfy the resulting task specification. This coupled generation process enables:

Benchmark validity. The world is guaranteed to contain exactly one target that satisfies all constraints in the instruction, so every task is solvable and has a unique answer.

Task complexity control. Alongside the target, the generator places hard negative items: distractors that share the target’s visible attributes on list and search-result pages (e.g., same title, price, and category) but differ on an attribute visible only after opening the item’s detail page (e.g., seller or delivery policy). Thus, identifying the target requires navigating to individual detail pages. This allows explicit problem complexity control, enabling complexity measure on three axes: hard negative count, informational access level, and oracle trajectory length. The corresponding definitions and analysis is in Section 4.4.

Oracle trajectory. We can statically derive an oracle trajectory for each task since both the MDP and the item set are fully specified. Refer to Appendix E.3 for details.

#### 3 WEBSTEP

WEBSTEP is a benchmark for process-level evaluation of web agents in visually realistic environments with deterministic and semantic underlying dynamics. It consists of 10 self-hosted websites spanning diverse domains (Figure 7), including productivity tools, commerce platforms, professional and collaborative platforms, and technical knowledge sites. Each website is modeled after a real site from the same domain (Appendix D.2), while replacing the backend with the deterministic semantic model defined in Section 2.1. Additional dataset statistics are provided in Table 4.

- Table 1 positions WEBSTEP among existing web agent benchmarks. Relative to live-website settings, it provides deterministic self-hosted environments. Compared to terminal-level benchmarks, it adds process-level evaluation grounded in an explicit semantic MDP and complexity control with task-conditioned world generation. Further discussion on related work is in Appendix B.

We design a multi-stage pipeline to construct WEBSTEP. Starting from live website workflow traces, a coding agent (Anthropic, 2026) with iterative human review produces a formal MDP specification and converts it into a self-hosted website driven by the semantic model. Each site is then validated through automated testing, including MDP unit tests, GUI interaction tests, and oracle trajectory replay, together with manual verification. Full pipeline details and example MDPs are provided in Appendices E.1 and H.3, respectively.

Success Rate (%) ↑ Information ↑ Steps Terminal Exploration Execution Coverage (%) GUI Semantic GUI/Semantic ↓

Agent

Fara-7B 31.4 43.6 55.7 60.6 18.9 8.3 2.3 GUI-Owl-1.5-8B 31.9 43.6 55.6 61.9 28.3 10.4 2.7 UI-TARS-1.5-7B 32.6 46.3 50.6 62.6 35.0 14.0 2.5

Qwen3.5-122B 57.9 66.1 73.7 67.7 22.1 9.8 2.3 OpenAI CUA 82.2 87.7 86.2 71.0 19.7 10.0 2.0

- Table 2: Aggregate evaluation results. We report terminal success together with processlevel metrics from semantic MDP traces: exploration, execution, and information coverage. These metrics reveal behavioral differences not visible from terminal success alone.

#### 4 Experiments and Results

We analyze web agent behavior using the process-level metrics automatically produced with WEBSTEP. We begin by showing that they reveal differences hidden by outcome metrics (Section 4.1), then characterize the nature of those differences (Section 4.2), localize where they arise within trajectories (Section 4.3), and finally show that they widen as exploration difficulty increases (Section 4.4). Full results and qualitative examples appear in Appendices F and I, respectively.

Models. We evaluate five web agents spanning both small specialist models, UI-TARS1.5 (Qin et al., 2025), Fara (Awadallah et al., 2025), and GUI-Owl-1.5 (Ye et al., 2025), and large generalist models, Qwen3.5 (Yang et al., 2025) and OpenAI CUA (GPT-5.4) (OpenAI, 2025). To ensure a consistent evaluation protocol, we disable external actions such as crosswebsite navigation and Google search, and evaluate all tasks at non-critical terminal points. More details are provided in Appendices C.2 and C.3.

###### 4.1 Aggregate Results

We begin with coarse trajectory-level summaries derived from the semantic MDP traces. While these metrics do not yet expose fine-grained process-level breakdowns, they already reveal behavioral differences that terminal success alone obscures.

Metrics. All are computed automatically from the MDP trace τ = (s0, a0, s1, . . . , sT).

- • Terminal Success Rate (SR): The standard binary outcome measuring whether the agent found the correct target and successfully executes the final commit action on it;
- • Exploration SR: Evaluates whether the agent isolates the correct target item e⋆ before committing. Formally, if Vc = (v1, . . . , vk) denotes the sequence of items visited on detail surfaces prior to the first commit, exploration succeeds if and only if vk = e⋆;
- • Execution SR: The terminal success rate conditioned strictly on successful exploration. This isolates the agent’s ability to finalize a task once the correct target has been found;
- • Informational Coverage: The fraction of task-relevant attributes the agent witnessed, serving as a measure of information-gathering thoroughness (Refer to Section C.4);
- • Step Efficiency: We record both raw GUI steps and semantic steps (state-changing MDP transitions). Their ratio quantifies the agent’s interaction efficiency.

Results. Table 2 reports aggregate performance. Under terminal success, the small agents appear largely similar, clustering at 31-33%. The trajectory-derived metrics, however, reveal clear behavioral differences. UI-TARS identifies the correct target more often (∆ +2.7%) but performs worse at execution (∆ -5.0%), with higher informational coverage and step count indicating a more expansive exploratory strategy. By contrast, Fara has the lowest step count and coverage, suggesting weaker exploration engagements.

The larger models, Qwen3.5 and OpenAI CUA, perform better in both exploration and execution, and also achieve higher information coverage. But this higher coverage is not

Large models Small models

Strengths of OpenAI CUA

Strengths of Qwen3.5-122B

Strengths of GUI-Owl

Strengths of UI-TARS

+13.3%

+30.0%

| |+25.8%<br><br>+19.2% +35 +31.4% +26.7<br><br>+23.7 +20 +18. +15 +13|.6<br><br>% %<br><br>.0 3%<br><br>.0 .2|
|---|---|---|
| | | |

| |+16.7%<br><br>+13.3<br><br>+11.4%<br><br>+9.4%<br><br>+32.0% +21.7%<br><br>+16.5%<br><br>+9.5%<br><br>+38.3<br><br>+11.1%|%<br><br>%|
|---|---|---|
| | | |

Code Repo Accomm. Mail

Q&A Mail Code Repo Q&A Shopping

Calendar Accomm.

Q&A Team Chat Mail Accomm. Housing Shopping Q&A Code Repo

+8.9%

###### NavCommit

###### NavFilterCommit

+16.2%

###### InspectNavCommit

###### NavFilter

%

+48.3% +32.0%

+12.2%

Team Chat

Q&A Housing Job Net.

Job Net. Code Repo

+16.0%

+29.4%

+9.7%

+15.6%

Job Net. Food Del.

Accomm.

%

+27.8%

+14.7%

Accomm. Shopping

Calendar Food Del.

+25.3%

+5.3%

Team Chat Q&A Accomm.

% %

+12.4%

Calendar Food Del.

Calendar Housing

+3.3%

+9.1%

0% 100%

0% 100%

0% 100%

0% 100%

OpenAI CUA Qwen3.5-122B GUI-Owl UI-TARS

Weaknesses of OpenAI CUA

Weaknesses of Qwen3.5-122B

Weaknesses of GUI-Owl

Weaknesses of UI-TARS

- -13.3%

-8.9%

- -48.3%
- -32.0%
- -29.4%

-9.7%

- -27.8%
- -25.3%
- -12.4%

-30.0%

| |-25.8%<br><br>-19.2%<br>-35.6%<br>-31.4%<br>-26.7%<br>-23.7%<br>-20.0%<br>-18.3%<br>-15.0%<br>-13.2%<br>| |
|---|---|---|
| | | |

|3|-16.7%<br><br>-13.3%<br>-11.4%<br><br><br>-9.4% 2.0%<br><br>-21.7%<br>-16.5%<br>-9.5%<br>-38.3%<br>-11.1%<br>| |
|---|---|---|
| | | |

Code Repo Accomm. Mail

Calendar Accomm.

Q&A Mail Code Repo Q&A Shopping

Q&A Team Chat Mail Accomm. Housing Shopping Q&A Code Repo

###### NavCommit

###### NavFilterCommit

-16.2%

###### InspectNavCommit

###### NavFilter

- -12.2%
- -16.0%
- -15.6%

Team Chat

Q&A Housing Job Net.

Job Net. Code Repo

-

Accomm.

Job Net. Food Del.

-14.7%

Accomm. Shopping

Calendar Food Del.

- -5.3%
- -3.3%

Team Chat Q&A Accomm.

Calendar Food Del.

Calendar Housing

-9.1%

0% 100%

0% 100%

0% 100%

0% 100%

- Figure 3: Skill-level strengths and weaknesses. Agents exhibit distinct within-website skill profiles. e.g., on Housing, OpenAI CUA is stronger on commit but weaker on filter, showing that an agent’s weakness can be concentrated in a specific interaction skill. Point labels (e.g.,

+26.7%) denote the gap relative to the compared model.

just a result of exploring more: OpenAI CUA covers more information than Qwen3.5 while using steps more efficiently, and its lower GUI-to-semantic step ratio shows that more of its actions produce meaningful semantic progress.

Finding 1. Terminal success hides behavioral differences. e.g., while the small-scale agents cluster at similar terminal success, trajectory-level summaries separate UI-TARS

- as exploration-strong but execution-weak, and Fara as under-exploratory (Table 2).

###### 4.2 Skill-level diagnosis

Aggregate summaries show overall patterns, but they do not isolate the specific interaction capability responsible for a model’s weakness. A model may appear strong or weak on a website for different underlying reasons, and broad summaries can hide these differences by mixing opposing skill-level signals. To make the diagnosis actionable, the evaluation must identify which interaction skills are present in the model’s behavior and which are not.

We operationalize this with skill invocation. For each task, we use an oracle trajectory as a reference. Although oracle trajectories are not unique solutions, they instantiate shortest valid strategies for completing the task. Let S⋆ be the set of skill types that appear in the

oracle trajectory, and let S be the set of skill types that appear in the model trajectory. For each skill k ∈ S⋆, we define invocation as whether k ∈ S.1

Skills. We map MDP actions into five categories (full definitions in Tables 6 and 7):

|[Figure 80]|
|---|

###### MAIL / COMPARE THREADS

- • inspect: opening an item’s detail view;
- • navigate: transitioning between interfaces without directly narrowing candidates;
- • search: text query to retrieve candidates;
- • filter: using the filtering interface to reduce the candidate set;
- • commit: performing the final taskcompleting action on the selected target.

Compare multiple email threads to find the one matching a superlative criterion (most CC recipients, largest attachment, etc.) and open it.

[Figure 81]

OpenAI CUA

SEARCH Issue search queries

…

Strengths and weaknesses. Figure 3 shows that agents exhibit distinct skill profiles. Among the two larger models, OpenAI CUA is substantially stronger on the final execution skill (commit), whereas Qwen3.5 is stronger on filter and navigate. Among the smaller agents, GUI-Owl is stronger on filter, whereas UI-TARS is stronger on navigate. These contrasts show that differences between agents are often concentrated in particular interaction skills.

FARA-7B

INSPECT Open entity detail page

COMMIT Final action (star, reply, book...)

…

NAVIGATE G b k, …

Figure 4: Temporal skill invocation patterns.

This contrast becomes sharper within a single website. In Housing, OpenAI CUA excels in executing the final action such as issuing the rental request (∆ + 23.7% over Qwen3.5), but falters in filter in the same domain (∆ − 15.6% over Qwen3.5). Website-level analysis would average these opposing signals into a single Housing score and obscure the intervention. Instead, skill-level diagnosis pinpoints what to fix. In Housing, the issue is a specific weakness in filter, which directly suggests adding Housing examples that train filtering.

Temporal structure of skill invocation. Semantic MDP traces also reveal when skills are invoked. While Figure 3 shows which required skills agents tend to miss, Figure 4 shows when they are used along the trajectory. OpenAI CUA follows a concentrated early-stage pattern, typically beginning with search, whereas Fara’s early actions are more dispersed. Fara also invokes commit earlier, while OpenAI CUA spends more early steps on informationgathering skills such as search, navigate, and inspect. This suggests targeted interventions: reinforcing a more consistent initial skill sequence for Fara, and delaying commitment until more information has been gathered.

Finding 2: Agent weaknesses can be skill-specific. In Housing, OpenAI CUA shows strong commit skills, but underperforms on filter usage, indicating that its weakness in this website is not uniform, but concentrated in a specific interaction skill (Figure 3).

###### 4.3 Trajectory bifurcations

The preceding analyses identify which skills are weak, but they do not yet show where failure is introduced within a trajectory. We address this by comparing successful and failing trajectories from the same task at the level of semantic MDP states. Since all agents act in

1We do not measure skill success because a semantic skill is usually identifiable only upon completion. Unsuccessful partial attempts therefore often lack a well-defined skill label. For example, a search is identifiable only after query entry and submission.

###### (a) Wrong branch (First Divergent Action)

###### (c) Premature Commit (Required Action)

###### (b) Delayed Commit (Extra Action)

Navigate Commit

Commit

###### Commit

Inspect

[Figure 82]

Search

Inspect

[Figure 83]

Search

Search

[Figure 84]

[Figure 85]

[Figure 86]

Filter Search

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Filter Search

[Figure 92]

Filter Search

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

…

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

…

[Figure 104]

𝜌

[Figure 105]

𝜌

[Figure 106]

𝜌

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Inspect

Inspect

browse

browse

browse

[Figure 115]

Commit

[Figure 116]

[Figure 117]

Search

[Figure 118]

[Figure 119]

[Figure 120]

Search

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Navigate

[Figure 126]

Navigate

[Figure 127]

[Figure 128]

[Figure 129]

###### Required Action:

Extra Action:

[Figure 130]

Successful Agent A

Successful Agent A

Successful Agent A

[Figure 131]

[Figure 132]

[Figure 133]

First Wrong Action: Search

Search Filter Inspect Navigate

x1 x4

x1 x4

Inspect x1 Navigatex1

Failed OpenAI-CUA

Failed OpenAI-CUA

Failed OpenAI-CUA

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

24% 16% 39% 16% 6%

22% 22% 33% 22%

0% 20% 60% 20%

OpenAI CUA

OpenAI CUA

OpenAI CUA

14% 12% 41% 23% 9%

8% 14% 42% 36%

10% 10% 48% 32%

Qwen3.5-122B

Qwen3.5-122B

Qwen3.5-122B

29% 14% 34% 17% 7%

26% 25% 30% 19%

20% 9% 45% 26%

FARA

FARA

FARA

19% 24% 32% 20% 5%

10% 65% 14% 12%

64% 0% 27% 9%

GUI-Owl

GUI-Owl

GUI-Owl

27% 17% 28% 19% 9%

5% 40% 25% 31%

20% 8% 47% 25%

UI-TARS

UI-TARS

UI-TARS

Search Filter InspectNavigate Commit

Search Filter InspectNavigate

Search Filter InspectNavigate

- Figure 5: Trajectory bifurcations at shared semantic states. Bifurcation analysis localizes where failure is introduced. (a) Wrong branch: the decision that first sends the failing trajectory off course. (b) Delayed commit: the extra behavior after missing the right time to finish. (c) Premature commit: the missing behavior skipped before finishing too early.

the same Markovian environment, trajectories can be aligned whenever they visit the same state. We define the bifurcation point as the last shared state before divergence.

We categorize bifurcation points into three types, as shown in the top row of Figure 5:

- • Wrong branch: the two next actions differ, and neither is COMMIT. We show the mismatched next action.
- • Delayed commit: the successful next action is COMMIT, but the failing next action is not. We show the remaining failing suffix.
- • Premature commit: the failing next action is COMMIT, but the successful next action is not. We show the remaining successful suffix.

Wrong branches. Figure 5 (a) identifies the decision that first sends a trajectory onto a losing branch, revealing the branching error each agent is most prone to. For most agents, this first wrong step is Inspect, suggesting that they often branch off by opening the wrong item page (28–41%). Fara and UI-TARS also frequently fall back to Search when they go off track (27–29%), whereas GUI-Owl more often diverges through Filter (24%). These patterns show that agents tend to enter failure through different kinds of decisions.

Delayed commit. Figure 5 (b) characterizes how agents behave after missing the point

- at which a successful trajectory would already have finished, revealing the kinds of unnecessary actions they continue to take. GUI-Owl spends 65% of these extra actions on Filter, whereas Qwen3.5 instead continues mainly with Navigate (36%) and Inspect (42%). Their delayed failures therefore take different forms: repeated refinement for GUI-Owl, and continued navigation or evidence gathering for Qwen3.5.

Premature commit. Figure 5 (c) shows what kinds of evidence agents most often skip before committing too early. In four of the five models, the omitted action is most often Inspect, indicating that premature failures usually stem from skipping a final evidencegathering step. GUI-Owl again differs from this pattern: in 64% of these cases, the skipped action is Search, suggesting that its early commits often occur before the correct target has even been retrieved.

###### (a) Hard negative count

###### (b) Oracle Trajectory Length

###### (c) Info Access Level

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |

0.8

ExplorationSR

0.6

| |
|---|

| |
|---|

| |
|---|

0.4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

0.0

0 1 2 3

2 4 6 8 10

Card Filter Detail

Count

Length

Level

OpenAI CUA Qwen3.5-122B FARA GUI-Owl UI-TARS

- Figure 6: Exploration success by task complexity. (a) Performance by hard negative counts separates OpenAI CUA from others. (b) Oracle trajectory length shows downward trend. (c) Tasks requiring page detail evidence show different model gaps from list-level tasks.

Finding 3. The decisive error point that loses the task differs across agents. e.g., GUI-Owl most often first diverges to failure at Filter, whereas Qwen3.5 most often first diverges at Inspect. (Figure 5).

###### 4.4 Exploration success by task complexity

Having localized where individual trajectories first go wrong, we next ask whether failure follows systematic patterns as exploration difficulty increases.

Task-complexity measures. We consider three complementary measures:

- • Hard negative count: the number of plausible distractor items in the world;
- • Oracle trajectory length: measures the minimum number of steps in the oracle trajectory;
- • Information access level: groups tasks by the exploration type required to identify the target.

- – Card for list-level information only;
- – Filter for tasks requiring filtering or search;
- – Detail for tasks requiring explicit visits to item detail pages.

Hard negative count. Figure 6 (a) shows that task structure strongly shapes exploration difficulty. As the count increases, exploration success declines monotonically for four of the five agents, confirming that the procedurally generated distractors induce genuine exploration difficulty. OpenAI CUA is the exception, showing much weaker sensitivity to the number of distractor items and suggesting substantially stronger exploration capacity.

Oracle trajectory length. Figure 6 (b) shows a similar pattern. As the minimum successful trajectory becomes longer, exploration performance generally degrades. Qwen3.5 exhibits a relatively monotonic drop as trajectory length increases, whereas OpenAI CUA maintains its performance for longer, suggesting better generalization to more complex tasks.

Information access level. Figure 6 (c) further separates the agents by exploration types required. Performance is relatively similar on card tasks, where only list-level inspection is needed. Filter tasks begin to separate the larger models from the smaller ones, and detail tasks further separate the strongest model, OpenAI CUA, from the second-best model, Qwen3.5. Thus, complex exploration requirements reveal differences in the agents.

Finding 4. Agent differences in exploration widen as tasks become harder. Exploration performance is relatively similar on easier tasks, but separates much more clearly as exploration requirements become more demanding (Figure 6).

#### 5 Conclusion

We introduced WEBSTEP, a benchmark for automatic process-level evaluation of web agents. By pairing each self-hosted website with semantic state tracking, WEBSTEP enables finegrained analysis of agent behavior without manual trajectory annotation. Our results show that terminal success alone hides major differences in how agents explore, act, and fail, whereas process-level evaluation makes these differences directly diagnosable. We hope WEBSTEP supports deeper diagnosis of web agents. Limitations are in Appendix A.

#### Ethics Statement

WEBSTEP is a benchmark for evaluating web agent behavior in controlled, self-hosted environments. It does not involve human-subject experiments, real user accounts, or interaction with live online services during evaluation. All websites are independently implemented reproductions of publicly observable front-end interaction patterns (screenshots and keyboard+mouse movements), and the benchmark uses synthetic world data rather than real user data or production backends. We recognize that improved web agent evaluation can have dual-use implications, including downstream misuse on real websites. Our work is intended to support diagnostic analysis in sandboxed environments, and all experiments are confined to offline, self-hosted replicas with no external side effects. Details of LLM usage are provided in Appendix C.6.

#### Reproducibility Statement

We will publicly release the full benchmark, including all 10 website implementations, all 1,800 task definitions with world seeds and oracle trajectories, and the complete evaluation suite. Each environment is formalized as a deterministic MDP with a fixed world seed, so the environment dynamics, verification outcomes, and semantic traces are reproducible given the same agent actions. This guarantees reproducibility of the benchmark conditions, although full end-to-end run reproducibility may still be affected by model-side sources of non-determinism such as stochastic decoding or API variation. Agent hyperparameters, model identifiers, and decoding settings are reported in Table 3. Additional details on environment determinism and residual sources of non-determinism are provided in Appendix C.5.

#### References

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Advances in neural information processing systems, 2017.

Anthropic. Claude opus 4.6 system card, 2026. URL https://www-cdn.anthropic.com/ 14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf.

Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-7b: An efficient agentic model for computer use. arXiv preprint arXiv:2511.19663, 2025.

L´eo Boisvert, Megh Thakkar, Maxime Gasse, Massimo Caccia, Thibault Le Sellier De Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. WorkArena++: Towards compositional planning and reasoning-based common knowledge work tasks. arXiv preprint arXiv:2407.05291, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

De Chezelles, Thibault Le Sellier, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lu,` Ori Yoran, Dehan Kong, Frank F Xu, Siva Reddy, Quentin Cappart, et al. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467, 2024.

Jiwan Chung, Neel Joshi, Pratyusha Sharma, Youngjae Yu, and Vibhav Vineet. What mllms learn about when they learn about multimodal reasoning: Perception, reasoning, or their integration? arXiv preprint arXiv:2510.01719, 2025.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. In NeurIPS, 2023.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marber, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.

Divyansh Garg, Shaun VanWeelden, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Tomas Abraham, Michael Lara, Federico Lopez, et al. Real: Benchmarking autonomous agents on deterministic simulations of real websites. arXiv preprint arXiv:2504.11543, 2025.

Milan Gritta, Debjit Paul, Xiaoguang Li, Lifeng Shang, Jun Wang, and Gerasimos Lampouras. Process evaluation for agentic systems. In Findings of the Association for Computational Linguistics: EACL 2026, pp. 2678–2692, 2026.

Alex Gu, Baptiste Rozi`ere, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I Wang. Cruxeval: a benchmark for code reasoning, understanding and execution. In Proceedings of the 41st International Conference on Machine Learning, pp. 16568–16621, 2024.

Tanmay Gupta et al. MolmoWeb: Open visual web agent and open data for the open web. Technical report, Allen Institute for AI, 2026. URL https://allenai.org/papers/ molmoweb.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.

Rodrigo Toro Icarte, Toryn Klassen, Richard Valenzano, and Sheila McIlraith. Using reward machines for high-level task specification and decomposition in reinforcement learning. In International Conference on Machine Learning, pp. 2107–2116. PMLR, 2018.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. ICLR, 2024.

Evan Zheran Liu, Kelvin Guo, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802, 2018.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In ICLR, 2024.

Xing Han Lu,` Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher Pal, and Siva Reddy. Agentrewardbench: Evaluating automatic evaluations of web agent trajectories. In Second Conference on Language Modeling, 2025.

Chang Ma, Junlei Zhang, Zhihao Liu, Jiawei Chen, Yitao Wang, et al. AgentBoard: An analytical evaluation board of multi-turn LLM agents. In NeurIPS, 2024.

Gr´egoire Mialon, Cl´ementine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: A benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2023.

OpenAI. Operator system card, 2025. URL https://cdn.openai.com/operator system card. pdf.

Yichen Pan, Dehan Kong, Sida Zhou, Cheng Cui, Yifei Leng, Bing Jiang, Hangyu Liu, Yanyi Shang, Shuyan Zhou, Tongshuang Wu, and Zhengyang Wu. WebCanvas: Benchmarking web agents in online environments. In ICML, 2024.

Eduardo Pignatelli, Johan Ferret, Matthieu Geist, Thomas Mesnard, Hado van Hasselt, and Laura Toni. A survey of temporal credit assignment in deep reinforcement learning. Transactions on Machine Learning Research, 2024.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy P Lillicrap. Androidinthewild: A large-scale dataset for android device control. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William E Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations, 2025.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Daniel Toyama, Philippe Hamel, Anita Gergely, Gheorghe Comanici, Amelia Glaese, Zafarali Ahmed, Tyler Jackson, Shibl Mourad, and Doina Precup. Androidenv: A reinforcement learning platform for android. arXiv preprint arXiv:2105.13231, 2021.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Cˆot´e, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, 2022.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglong Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. WebWalker: Benchmarking LLMs in web traversal. arXiv preprint arXiv:2501.07572, 2025.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024.

Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. In COLM, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. NeurIPS, 2022.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, and Ming Yan. Mobile-agent-v3: Fundamental agents for GUI automation. arXiv preprint arXiv:2508.15144, 2025.

Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, and Jonathan Berant. AssistantBench: Can web agents solve realistic and time-consuming tasks? arXiv preprint arXiv:2407.15711, 2024.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. In ICLR, 2024.

## Appendix

#### Table of Contents

- A Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- B Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- C.1 Agent Configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2 Compute and Runtime Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .17
- C.3 Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.4 Informational Coverage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.5 Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.6 LLM Usage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Benchmark Specification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .20

- D.1 Task Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .20
- D.2 Site Screenshots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.3 Per-Site MDP Specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Data Generation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .25

- E.1 Site Construction Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- E.2 World Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- E.3 Oracle Trajectory Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- F Extended Quantitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- F.1 Per-site performance decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.2 Per-site skill invocation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- F.3 Aggregate skill invocation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- F.4 Exploration SR by problem complexities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- G Artifact Viewer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- H Data Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- H.1 Task Template Catalog . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- H.2 Example Trajectories: Agent vs. Oracle Trajectory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .34
- H.3 MDP Surface Transition Graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

- I Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- I.1 Model Scale and Browsing Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- I.2 Premature Commitment to Hard Negative . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- I.3 Thorough Exploration Before Commitment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .43
- I.4 Exploration Success, Execution Failure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- I.5 Exploration Failure, Execution Success . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- I.6 Redundant Process Despite Success . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- I.7 Failure by Safeguard, Not by Competence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

#### A Limitations

Our benchmark prioritizes controllability, reproducibility, and trajectory-level diagnosis. These design choices improve measurement precision, but they also introduce limitations in realism, coverage, and action scope.

Semantic abstraction. Each site is implemented as a deterministic MDP that captures the task-relevant interaction structure of its real-world counterpart. As a result, the benchmark does not capture some properties of live websites, including dynamic content updates, session-dependent state, asynchronous loading, personalized recommendations, and thirdparty integrations. Agent behavior in these self-hosted environments may therefore differ from behavior on live websites with noisier layouts and less predictable dynamics.

Domain coverage. The benchmark spans ten web domains, but it does not cover every application type. In particular, domains involving sensitive data handling, complex authorization flows, or rich media interaction are not included. Still, the covered domains are diverse, and the benchmark is broader in scope and substantially larger in task count than existing alternatives (He et al., 2024; Zhou et al., 2024; Deng et al., 2023). We therefore view the remaining gap as a matter of incomplete coverage rather than narrow domain design.

Synthetic world data. World data is generated synthetically through seeded procedures. This improves reproducibility and allows controlled construction of hard negatives, but it may miss properties of organic real-world data such as long-tail popularity patterns, organically correlated attributes, or temporal drift.

Template-based tasks. Tasks are instantiated from structured templates with deterministic verification conditions. This does not capture the full ambiguity, underspecification, or open-endedness of real user requests. However, the loss is constrained by that the templates still span diverse task types and difficulty factors, while the resulting structure enables explicit control over complexity, scalable generation, and reliable aggregation across task families. This design is therefore necessary for systematic process-level evaluation at scale.

Skill invocation rather than skill success. Our skill-level analysis measures whether agents invoke the required action types, not whether each action succeeds in the agent’s intended sense. A finer notion of per-action success would require reliable access to latent intent, which is not available from GUI traces alone and is difficult to infer faithfully from model-generated rationales. We therefore evaluate behavior in terms of the realized causal trajectory, which is the more robust and less assumption-dependent object of analysis.

Interaction modality. The current evaluation focuses on coordinate-based visual interaction, specifically clicking, typing, and scrolling. Other interaction modes, such as dragand-drop, keyboard shortcuts, or direct URL navigation, are not included. This narrows the covered action space and may omit behaviors that matter in some real web settings.

#### B Related Work

Web agent benchmarks. Web agent benchmarks now span a broad range of environment regimes, from tightly controlled synthetic interfaces such as MiniWoB++ (Liu et al., 2018) and WebShop (Yao et al., 2022), to self-hosted realistic websites such as WebArena (Zhou

- et al., 2024) and VisualWebArena (Koh et al., 2024), to online and live-web settings such as Mind2Web (Deng et al., 2023), WebVoyager (He et al., 2024), Mind2Web-Live (Pan et al., 2024), Online-Mind2Web (Xue et al., 2025), and AssistantBench (Yoran et al., 2024). Related benchmarks further broaden this landscape to enterprise workflows and general computeruse settings, including WorkArena (Drouin et al., 2024), BrowserGym (Chezelles et al., 2024), and OSWorld (Xie et al., 2024). Despite this rapid progress in environment realism and scope, evaluation remains centered primarily on terminal outcomes such as task success, sometimes with partial-credit variants or key-node style intermediate checks. WEBSTEP

differs in focusing on built-in process instrumentation: each environment is co-developed with an explicit semantic MDP dual, enabling automatic process-level evaluation grounded in the environment’s own state and transition structure.

Environment modeling and GUI-semantic bridging. Constructing faithful simulated environments for agent evaluation is a longstanding challenge. AndroidEnv (Toyama et al., 2021) and AndroidWorld (Rawles et al., 2025) wrap real mobile applications in RLcompatible interfaces, while OSWorld (Xie et al., 2024) provides full desktop VMs with scripted evaluators. These environments expose low-level interaction traces, but they do not provide an explicit semantic account of what each action reveals or accomplishes. A related line of work maps GUI events to higher-level action abstractions for training or evaluation (Rawles et al., 2023; 2025), but usually does so post hoc through heuristic alignment rather than through the environment construction itself. In the web domain, WebArena (Zhou et al., 2024), VisualWebArena (Koh et al., 2024), and REAL Evals (Garg

- et al., 2025) move toward sandboxed and more reproducible evaluation, but their evaluation remains decoupled from an explicit website semantics and is largely limited to terminal success. Our semantic MDP dual instead co-develops the GUI and the semantic model so that all task-relevant rendering is generated from semantic state, making the correspondence explicit by construction.

Process-level and intermediate evaluation. The limitations of outcome-only evaluation are well recognized across adjacent areas. In mathematical reasoning, process supervision and process reward models explicitly score intermediate reasoning steps rather than only final answers (Lightman et al., 2024). In code evaluation, benchmarks such as HumanEval assess functional correctness by executing generated programs against tests (Chen et al.,

- 2021), while CRUXEval further targets code understanding and execution behavior rather than final output alone (Gu et al., 2024). In sequential decision making, the broader credit assignment problem has long been recognized as a central challenge in reinforcement learning (Pignatelli et al., 2024; Sutton & Barto, 1998), precisely because terminal rewards alone are often too sparse or too delayed to localize which decisions mattered. This has motivated methods that expose or reshape intermediate structure, including reward decomposition (Icarte et al., 2018) and hindsight-based relabeling (Andrychowicz et al., 2017). Chung et al. (2025) also considered decomposition of multimodal reasoning capacities. Our setting is analogous: terminal task success does not reveal whether a web agent failed to find the right target, failed to gather the decisive evidence, or failed only at the final action, which is why process-level evaluation is necessary.

In web agent evaluation, however, most existing analyses remain at the task level: AgentBench (Liu et al., 2024) and GAIA (Mialon et al., 2023) report performance by task category, separating tasks from one another but not analyzing behavior within a trajectory. Recent work has begun to introduce within-task intermediate evaluation. AgentBoard (Ma et al., 2024) measures progress rate via manually annotated subgoals, and Mind2Web-Live (Pan et al., 2024) defines key nodes along reference trajectories. These approaches demonstrate that binary success discards substantial signal, but their reliance on static human-defined checkpoints limits both scalability and robustness, particularly on live websites where interface changes can invalidate annotations.

Automated trajectory evaluation. Trajectory-level evaluation that avoids manual annotation has been explored through several paradigms. LLM-as-judge approaches (Lu` et al., 2025; Gritta et al., 2026) use language models to score intermediate agent steps, offering flexibility but introducing stochastic evaluation noise and dependence on judge model capability. Rule-based evaluation in structured environments (Shridhar et al., 2021; Wang et al.,

- 2022) enables deterministic assessment but typically operates at the level of task completion predicates rather than fine-grained process analysis. Our framework differs from both: because the evaluation environment is built on an explicit semantic MDP, process metrics (including exploration success, skill invocation, and trajectory bifurcation analysis) are derived deterministically from the MDP trace without requiring either human annotation or model-based judgment.

#### C Implementation Details

###### C.1 Agent Configurations

Agent Model Backend Input Action Space Max Imgs Temp. Max Tokens

Fara microsoft/Fara-7B vLLM Screenshot Function call (coordinate) 3 0.0 800 UI-TARS ByteDance-Seed/UI-TARS-1.5-7B vLLM Screenshot Thought + Action (coordinate) 5 0.0 2048 GUI-Owl mPLUG/GUI-Owl-1.5-8B vLLM Screenshot Tool call (coordinate) 4 0.0 2048 Qwen3.5 Qwen/Qwen3.5-122B-FP8 vLLM Screenshot Thought + Action (coordinate) 5 0.7 2048 OpenAI CUA GPT-5.4 OpenAI API Screenshot Built-in computer tool (coordinate) – – –

- Table 3: Agent configurations. Input: what the agent receives each turn. Action space: how the agent specifies actions. Max imgs: number of recent screenshots retained in context.

All agents interact with the browser through a shared viewport of 1440 × 900 pixels. Each episode is capped at a maximum of 50 turns; if the agent has not completed the task within this budget, the episode is terminated and scored as a failure. Between consecutive agent actions, a brief blocking period is enforced to allow page transitions and rendering to complete before the next screenshot is captured.

We evaluate the following agents in WEBSTEP:

- • UI-TARS Qin et al. (2025): A vision–language model for GUI interaction developed by ByteDance.2
- • Fara Awadallah et al. (2025): A 7B web agent from Microsoft Research, fine-tuned on Qwen2.5-VL.3
- • GUI-Owl Ye et al. (2025): A multimodal agent from Alibaba mPLUG. We use the 8B-model. 4
- • Qwen3.5 Yang et al. (2025): Alibaba’s Qwen3.5-122B model, quantized to FP8.5
- • OpenAI CUA OpenAI (2025): OpenAI’s Computer-Use Agent, accessed via the OpenAI API. We implemented the CUA-agent following the official github repository. 6

###### C.2 Compute and Runtime Statistics

All GPU-based agents (Fara, UI-TARS, GUI-Owl, and Qwen3.5-122B) are served via vLLM on a single node with 4× NVIDIA H200 GPUs (140GB each). The full benchmark evaluation across all GPU models completes in approximately 3–4 days of wall-clock time. The closedsource agent (OpenAI CUA) uses the OpenAI API and runs in approximately 8 hours with 8 parallel workers.

###### C.3 Evaluation Protocol

Each task episode proceeds as follows. First, a headless Chromium browser instance is launched and initialized with the task’s world data and starting state. The agent then enters a turn-based loop: at each turn, the runner captures a screenshot of the current browser viewport, constructs an observation (including the screenshot, the current page URL, and history of previous conversation), and passes it to the agent. The agent returns an action (e.g., a click at a coordinate, a text input, or a scroll) which the runner executes in the browser. This loop continues until the agent signals task completion, the maximum turn limit of 50 is reached, or the agent reports the task as infeasible.

- 2https://github.com/bytedance/ui-tars
- 3https://github.com/microsoft/fara
- 4https://github.com/x-plug/mobileagent
- 5https://github.com/QwenLM/Qwen-Agent
- 6https://github.com/openai/openai-agents-python

Upon episode termination, the verifier evaluates the stored semantic MDP trajectory against the task’s verifier conditions. These conditions check the application’s terminal semantic state for example, whether a specific item has been added to a cart, a message has been starred, or a booking has been confirmed with the correct parameters. An episode is scored as successful only if all verifier conditions are satisfied.

Additionally, we observed that some models (e.g., Fara) consistently fail to execute the final commit action even after correctly identifying the target entity. To ensure fair comparison, we adopt a safe pass policy: if the agent reaches the critical decision point, i.e., it has navigated to the correct entity and gathered sufficient information to act, but does not issue the final commit, the episode is still counted as a success for the terminal success rate. This prevents penalizing models whose failure is limited to the mechanical execution of a single concluding action rather than a deficiency in exploration or reasoning. All terminal success rates reported in this paper are measured under the safe pass criterion.

###### C.4 Information Coverage

Information coverage measures how much task-relevant evidence an agent has gathered by the time it commits. It is a graded metric of evidence acquisition, not a binary test of whether the agent observed everything that could have been useful. For each task, the benchmark derives a set of information constraints C = {c1, . . . , cK} from the task’s information gap, template type, and reasoning type using per-site declarative rules. This set is best understood as a conservative upper bound on useful information: it includes fields, entities, and comparisons that may support the decision, even though full observation of all such information is not generally necessary for success.

Each site defines a visibility specification that maps every interface surface σ to the set of item attributes rendered on that surface. Some attributes are visible on list-level surfaces, while others become visible only after opening a detail page. As the agent follows a semantic trajectory τ = (s0, a0, . . . , sT), the benchmark tracks cumulatively which task-relevant constraints have become satisfied. Coverage at step t is then defined as

COVt = |{c ∈ C : sat(c, t)}|

, (2)

|C|

where sat(c, t) indicates whether constraint c has been satisfied by the information observed up to step t.

We report coverage at the commit step, tcommit, defined as the first step at which the agent executes a terminal action such as PlaceOrder or done. The reported metric, COVtcommit, therefore measures how much of this benchmark-defined task-relevant evidence the agent had gathered when it chose to act.

###### C.5 Reproducibility We take the following steps to ensure reproducibility:

Code and data release. We will publicly release the complete benchmark codebase, including all 10 site implementations, the task generation pipeline, the evaluation runner, and the metric computation scripts. All 1,800 task definitions (with world seeds, instructions, oracle trajectories, and verifier conditions) will be included in the release.

###### Environment determinism.

- • Fixed world seeds. Each task is paired with a deterministic random seed that fully determines the world data (user profiles, item catalogs, message contents, etc.), ensuring identical evaluation conditions across runs.
- • Deterministic MDPs. All site implementations are pure deterministic MDPs: the same state–action pair always produces the same next state and observation, eliminating environmental stochasticity.

Agent hyperparameters. All agent hyperparameters are reported in Table 3 (Section B.1). Key settings:

- • Greedy decoding. All open-weight agents (Fara, UI-TARS, GUI-Owl) are served with temperature 0, ensuring deterministic outputs given identical inputs. Qwen3.5 uses temperature 0.7, top-p 0.8, top-k 20 to follow official implementation.
- • Episode budget. All agents are given a maximum of 50 GUI turns per task.
- • Viewport. All agents share a 1440 × 900 browser viewport.

Non-determinism. For the closed-source OpenAI CUA agent, exact reproducibility cannot be guaranteed due to potential server-side non-determinism in the API; however, we observe low variance across repeated runs.

###### C.6 LLM Usage

We use LLMs in three aspects of this work. First, in the benchmark construction pipeline, we use Claude Opus 4.6 as a coding subroutine to help convert recorded website interaction traces into semantic MDP specifications and to build the executable website implementations from those specifications, as described in Section E. This process involved substantial manual validation, editing, and re-iteration between steps; the LLM served as an accelerator within a human-driven development loop rather than as an autonomous construction agent. Second, the web agents evaluated in our experiments (Section 4) are themselves LLM-based systems; their use constitutes the object of study rather than a methodological choice. Third, LLMs assisted with result visualization by providing initial seeds for plotting code and performing instruction-based edits; all figures were subsequently hand-edited in both code and output. In all three cases, the authors reviewed and verified the outputs. No LLM was used to originate research ideas or generate experimental data. LLMs provided assistance with prose editing during drafting.

#### D Benchmark Specification

###### D.1 Task Distribution

Find Email Extract (30)

Search React (15)

Find Attachment Message (15)

Compare Threads (30) Compose Reply (30)

Edit Message (15)

Multi Channel Send (15)

Thread Read Reply (15)

Channel Find Message (15)

Count Compute (30) Find By Content (30) OrganizeEmail(30)

Send Dm (15)

Pin Message (15)

Search And Reply (15)

React To Message (15)

ReplyToThread(15)

SendMessage(15)

FindEventExtract(30) CompareEvents(30) ScheduleMeeting(30) CountCompute(30) FindConflict(30) EditEvent(30)

JobCompareDetail(6) MultiJobApplyMessage(4)

JobSave(7)

JobSalaryCompare(9)

JobSkillsMatch(16) JobCompanyCriteria(11) NetworkConnectMultiple(10)

JobFilterApply(22) PeopleSearchConnect(19)

MessagePerson(23)

JobSearchApply(25)

ProfileExplore(28)

(180)Team Chat

FindProductExtract(30)

ContributorBasedOpen(15)

Mail(180)

JobNetwork(180)

LanguageFilterStar(15)

Calendar (180)

CompareProducts(30)

MultiStepStarAndIssue(15)

IssueSearchFilter(15)

PurchaseProduct(30)

CompareRepos(15)

Shopping (180)

CodeRepo (180)

ForkAndStar(15)

Accommodation

CountCompute(30)

CodeNavigation(15)

CodingQ&A

PrReviewDetail(15)

(180)Food Delivery

FindBySpec(30)

CreateIssue(15)

(180)

(180)

IssueTriageClose(15)

CartManagement(30)

Housing(180)

IssueTriageLabel(15)

SearchAndStar(15)

SearchFilterBook(15) DetailCancellationBook(15)

CodePatternOpen(15)

DetailAmenityBook(15) ComparePriceTotal(15)

UserProfileOpen(15)

CompareAnswersBookmark(15) SortQuestionsOpen(15)

SleepingArrangementBook(15) HighestRatedBook(15) BookThenCancel(15)

MultiTagOpen(15)

HostSuperhostBook(15)

LocationAmenityCombo(15)

PostComment(15)

ViewDetailOnly(15) PaginateCheapestBook(15) MultiConstraintBook(15)

PostAnswer(15)

DownvoteAnswer(15)

AcceptAnswer(15)

SearchOpenQuestion(15) TagFilterOpen(15)

Filter And Schedule Tour (15)

UpvoteAnswer(15)

Compare Restaurants De

ComplexFilterDetail(15)

SearchOrderSimple(12) CuisineFilterOrder(12)

Comparison Shopping (15)

Neighborhood Search (15)

Cheapest Item (12) Highest Rated Cuisine (12) Multi Item Order (12)

(12) Menu Dietary Detail (12)

Dietary Restriction Or

PricePerSqft(15)

Search And Save (15)

Compare Properties (15)

Investment Analysis (15)

Compare Menu Prices (12)

(12) Filter Detail Combo (12)

Property History (15)

Agent Contact (15)

Open House Find (15)

Free Delivery Order (12)

Fastest Delivery (12)

Modify Cart Order (12)

Calorie Conscious (12)

School Based (15)

Allergen Check (12)

- Figure 7: WEBSTEP task distribution. The inner ring shows 10 websites (180 tasks each). The outer ring shows task templates with instance counts. Templates range from 4 to 30 instances per template, reflecting the diversity of interaction patterns within each site.

- Figure 7 visualizes the hierarchical task distribution across the ten sites in WEBSTEP. Each site contributes 180 tasks instantiated from multiple templates, yielding a total of 1,800 tasks spanning diverse interaction patterns and difficulty levels.

###### D.2 Site Screenshots

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

Mail (Gmail-clone) Calendar (Gcalendar-clone) Shopping (Amazon-clone) Accommodation (Airbnb-clone) Food Delivery (UberEats-clone)

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Housing (Zillow-clone) Coding Q&A (StackOverflow-clone) Code Repo (GitHub-clone) Job Network (LinkedIn-Clone) Team Chat (Slack-Clone)

- Figure 8: Screenshots of all 10 WEBSTEP websites. Top row: Mail, Calendar, Shopping, Accommodation, Food Delivery. Bottom row: Housing, Coding Q&A, Code Repository, Job Network, and Team Chat. Each site is a standalone HTML/CSS/JS application with full MDP instrumentation.

- Figure 8 shows representative screenshots from each of the ten sites in WEBSTEP. The sites span a broad range of web application categories–from e-mail and calendaring to e-commerce, social networking, and developer tools–each with distinct visual layouts, navigation patterns, and interaction paradigms. Each site’s visual layout and interaction flow is modeled after a representative real-world web application in the same domain.

###### D.3 Per-Site MDP Specifications

State decomposition. Each site in WEBSTEP is modeled as a deterministic MDP whose state is decomposed into a set of typed variables. These variables capture the current navigation surface (e.g., inbox, search results, item detail), UI component states (e.g., which dropdown is open, current scroll position, active filters), and application-level data (e.g., selected items, form field contents, user preferences). Table 4 summarizes the number of surfaces, state variables, actions, and tasks for each site.

Visibility partitioning. Not all state variables are visible on every surface. Table 5 specifies the visibility partition: for each site, it lists which state variables are observable on each surface. This partition determines what information is available to the agent at any given point in the episode and is a key factor in task difficulty: tasks that require reasoning about state variables only visible on other surfaces necessitate multi-step navigation.

Skill taxonomy. Table 6 defines the five skill categories used throughout the benchmark. Table 7 then lists the full set of skills for each site, organized by category. Each task is labeled with the set of skills it exercises, enabling fine-grained analysis of agent capabilities across navigation, search, inspection, and execution dimensions.

###### Site Domain Surfaces Actions Templates Tasks Card / Detail Fields Decision Object

Mail Productivity 3 36 6 180 10 / 9 Thread Calendar Productivity 6 14 6 180 8 / 6 Event Team Chat Productivity 3 24 12 180 6 / 5 Message Shopping E-Commerce 6 23 6 180 12 / 12 Product Food Delivery E-Commerce 6 24 15 180 8 / 7 Restaurant / Item Accommodation Discovery 4 16 12 180 9 / 8 Listing Housing Discovery 4 16 12 180 8 / 10 Property Coding Q&A Information 4 18 12 180 7 / 6 Question Code Repo Collaboration 6 32 12 180 9 / 8 Repository / Issue Job Network Social 6 26 12 180 7 / 9 Job / Profile

Total – 48 229 105 1,800 84 / 80 –

- Table 4: Per-site MDP statistics. Surfaces: distinct UI views (pages, modals). Actions: typed semantic action count. Templates: task template count. Tasks: total task instances. Card fields: attributes visible on list views. Detail fields: attributes requiring detail-page navigation.

Site Card Surfaces Detail Surfaces Commit Surfaces |Vcard| |Vdetail| Mail ThreadList ThreadView ComposeModal, ThreadList, ThreadView 10 9 Calendar WeekView, DayView, MonthView EventDetail EventEditor 8 6 Team Chat ChannelView, SearchResults ChannelView (thread) ChannelView, DirectMessageView 6 5 Shopping SearchResults ProductDetail Checkout, OrderConfirmation 12 12 Food Delivery RestaurantList RestaurantDetail, MenuItemDetail Checkout, OrderStatus 8 7 Accommodation SearchResults ListingDetail Checkout, Reservations 9 8 Housing SearchResults PropertyDetail PropertyDetail, SavedHomes 8 10 Coding Q&A QuestionList QuestionDetail QuestionDetail 7 6 Code Repo RepoList, IssueList, PullRequestList RepoDetail, IssueDetail, PRDetail IssueDetail, RepoDetail 9 8 Job Network JobList, ConnectionList JobDetail, ProfileView JobDetail, Messaging 7 9

- Table 5: Visibility partition for each site. |Vcard|: attributes visible on list surfaces. |Vdetail|: attributes visible only on detail surfaces. Surfaces are categorized as Card (list views), Detail (entity detail pages), and Commit (surfaces where task-completing actions are available).

Category Description Navigate Actions that move between surfaces or manage the current view without modifying

application state. Includes pagination, folder/tab switching, back navigation, closing detail views, and temporal navigation (e.g., next/previous week).

Search Actions that issue or clear search queries to change the visible entity set. Includes keyword search, people search, repository search, and clearing search context. Filter Actions that narrow or reorder the visible entities within the current result set. Includes applying categorical or range filters, sorting, and clearing filters.

Inspect Actions that reveal additional information about an entity without modifying application state. Includes opening detail views, expanding hidden content, viewing profiles, and selecting entities for closer examination.

Commit Actions that modify the application state toward task completion. Includes form filling, adding to cart, placing orders, sending messages, starring items, voting, creating issues, confirming bookings, and all other state-changing operations.

- Table 6: Skill category definitions. Each MDP action in the benchmark is assigned to one of five categories based on its role in the task-completion process.

###### Site Category Skills

Navigate folder navigation, pagination, backtrack navigation Search search refinement Filter filter application Inspect thread inspection, message expansion Commit compose setup, compose field entry, email management,

Mail

bulk management, send commit

Navigate temporal navigation, view switching, backtrack navigation Search search query Filter calendar filtering Inspect event inspection, information extraction, comparison, con-

Calendar

flict detection Commit event creation, event editing, field entry, commit timing

Navigate pagination Search query formulation Filter department filtering, category filtering, brand filtering,

Shopping

price filtering, attribute filtering, sort usage Inspect product inspection, review inspection, comparison Commit option selection, cart management, checkout flow, commit timing

Navigate pagination, search return Search query formulation Filter property type filtering, price filtering, beds filtering,

Accommodation

amenity filtering, sort usage Inspect listing inspection, amenity expansion, comparison Commit booking initiation, booking confirmation, booking cancellation,

checkout cancellation

Navigate pagination Search query formulation Filter cuisine filtering, price filtering, dietary filtering, attribute filtering,

Food Delivery

sort usage Inspect restaurant inspection, menu item inspection, comparison Commit customization selection, cart management, checkout flow, com-

mit timing

- Table 7: Skill taxonomy per site (Part 1 of 2). Skills are organized into five categories: Navigate (surface traversal), Search (query formulation), Filter (attribute filtering and sorting), Inspect (entity viewing), and Commit (state-modifying actions).

###### Site Category Skills

Navigate pagination Search query formulation Filter price filtering, beds filtering, baths filtering, sqft filtering,

Housing

type filtering, attribute filtering, sort usage Inspect property inspection, agent inspection, comparison Commit save management, tour scheduling, agent contact,

saved homes review

Navigate back navigation, pagination Search query formulation Filter tag filtering, acceptance filtering, date filtering, vote filtering,

Coding Q&A

sort usage, answer sort usage, filter reset Inspect question inspection, user inspection, tag inspection Commit voting, answer posting, commenting, answer acceptance, bookmark-

ing

Navigate navigate to issues, navigate to prs, back navigation, pagination Search repo search Filter language filtering, stars filtering, license filtering, topic filtering, repo attribute filtering, repo sort usage, issue state filtering, issue label filtering, issue author filtering, issue sort usage, pr filtering, pr sort usage

Code Repo

Inspect repo inspection, issue inspection, pr inspection Commit issue creation, issue commenting, issue assignment, issue labeling,

issue state change, star management, commit timing

Navigate tab navigation, pagination Search job query formulation, people query formulation Filter location filtering, job type filtering, remote filtering, salary filtering,

Job Network

experience filtering, sort usage, filter clearing Inspect job inspection, profile inspection, job comparison, profile comparison Commit job application, connection request, messaging, job save action, conversation management, feed interaction

Navigate channel navigation, dm navigation, scroll navigation,

search result navigation Search search query, search clear Filter Inspect thread inspection, member inspection, thread close, member close Commit send channel message, send thread reply, send dm, mes-

Team Chat

sage reaction, message pin, message edit, message delete, channel mute, commit timing

Table 8: Skill taxonomy per site (Part 2 of 2). See Table 6 for category definitions.

#### E Data Generation Pipeline

###### E.1 Site Construction Pipeline

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

WorkFlow Traces

Specification

Implementation 🤖 LLM + 👤 Human review

Validation ⚙ Automated + 👤 Manual QA

• Browse live website

🤖 LLM + 👤 Human review

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Generated Web Applica on Ar facts

[Figure 175]

ℳ = (𝑆, 𝐴, T, 𝜌 )

[Figure 176]

- • state.js
- • transi ons.js
- • render.js
- • observa ons.js
- • world.js Assembly

[Figure 177]

[Figure 178]

[Figure 179]

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

[Figure 194]

[Figure 195]

REC

- • MDP unit tests
- • GUI interaction tests
- • Oracle trajectory replay

[Figure 196]

[Figure 197]

Fix

[Figure 198]

| | |
|---|---|
|[Figure 199]| |

| | |
|---|---|
|[Figure 200]| |

|[Figure 201]|
|---|

/ atric/deployment

###### Text-based MDP Spec Details

→ Self-hosted website

→ Workflow traces

→ Verified benchmark

→ Formal MDP spec

###### Figure 9: Overview of the website pipeline.

Each site in WEBSTEP is developed through an iterative human-in-the-loop pipeline. An LLM coding agent (Claude Opus 4.6 Anthropic (2026)) carries out much of the implementation at each stage, while all intermediate outputs are manually reviewed and refined before the next stage. The process is iterative rather than strictly linear: when downstream validation reveals issues, earlier stages are revisited and updated accordingly.

- Phase 1: Grounded exploration. We begin by interacting with representative user workflows on the live target website, including searching, filtering, inspecting items, and completing final actions, while recording user actions and screenshots at each step. These traces provide grounded evidence for the subsequent specification stage. All observations are derived solely from user-visible interaction, namely the screenshot and user action at each step, without relying on backend code, API traffic, or other non-GUI signals.
- Phase 2: Specification. The coding agent generalizes the recorded traces into a formal semantic MDP specification through the following staged procedure. Each stage must satisfy its validation criteria before the pipeline proceeds.

The state schema refines the main text’s positional/informational decomposition: interface, query, and session state together form the positional state, while revealed state corresponds to the informational state.

Algorithm 1 details the steps for generating MDP specification from the Workflow-traces of Phase 1.

- Phase 3: Website implementation. The specification is then implemented as a selfcontained single-page web application whose visible GUI is a rendered view of the underlying semantic MDP state, with minor exceptions on non-semantic actions:

- • state.js: State schema and createInitialState(world) initializer.
- • transitions.js: Pure transition function transition(state, action, world) →{state, error?}, with one case per typed action and precondition checking.
- • observations.js: Pure observation function observe(state, world) →observation, resolving visible entities, applying surface-specific visibility rules, and computing derived values.
- • world.js: Seeded pseudo-random generator generateWorld(seed) producing a complete entity catalog with cross-entity consistency enforced by construction.
- • render.js: DOM rendering as a pure projection of the observation. Every interactive element is bound to dispatch(action) calls via standardized data-test-id attributes.

Thus, GUI interactions are executed at the coordinate level, but every task-relevant interaction is mapped to a typed semantic action through the dispatch layer. The entry page wires these modules into a dispatch loop: each user interaction calls dispatch, which runs

the transition, recomputes the observation, and re-renders the page. Every dispatch also persists the current state and the full action trajectory to localStorage.

Algorithm 1 Specification construction from grounded traces Require: Workflow traces from Phase 1 Ensure: Structured MDP specification

- 1: Identify domain entities with typed attributes and unique identifiers
- 2: Define relations between entities with explicit cardinalities
- 3: Map surfaces, including visible entities, displayed fields, and available actions
- 4: Construct the navigation graph and verify surface reachability
- 5: Define the state schema: interface state, query state, session state, revealed state
- 6: Specify the observation function from state to visible fields and available actions
- 7: Define each action with parameters, preconditions, state updates, reveals, and failure cases
- 8: Specify filtering, sorting, pagination, and the deterministic query function
- 9: Define flow guards for sequential dependencies and stage-locked fields
- 10: Specify all derived display values as deterministic functions of state and world data
- 11: Define the rendering contract, including layout templates, rendering rules, DOM bindings, and style parameters
- 12: Define the world generator, including field distributions, consistency constraints, variation axes, and invariants
- 13: Run specification verification: coverage matrix reachability state liveness constraint binding observation sufficiency world generator consistency
- 14: if any check fails then
- 15: Return to the relevant earlier stage and revise
- 16: end if
- 17: Compile the final structured specification for the implementation stage

Multi-level validation. Each site is validated at four complementary levels:

- 1. MDP unit tests (validate.js). These tests validate the semantic model in isolation, including:

- • correct execution of happy-path trajectories,
- • rejection of invalid actions,
- • surface-specific observation visibility,
- • correctness of derived values,
- • navigation and back-stack behavior,
- • filter, sort, and search behavior, and
- • deterministic but varied world generation across seeds.

- 2. GUI interaction validation (validate interactions.py). Playwright-based browser tests verify:

- • discovery of all instrumented data-test-id elements,
- • coordinate-level interactability without occlusion,
- • execution of interactions through raw mouse and keyboard actions,
- • agreement between GUI interactions and expected MDP state transitions, and
- • absence of unsupported controls such as native <select> elements.

- 3. GUI trajectory replay (verify gui.py). End-to-end replay of every oracle trajectory in headless Chromium checks:

- • that every semantic action produces a real state change,
- • that the full oracle trajectory executes without error, and
- • that the final state satisfies the task verifier.

- 4. Manual QA iteration. Early agent runs are manually inspected to identify residual issues in:

- • verifier logic,
- • world-generation ambiguities, and
- • verifier-specification gaps.

All identified issues are fixed, propagated back through the pipeline as needed, and re-verified.

###### E.2 World Generation

Each task operates over a world: a self-contained data environment that populates the site with concrete entities (e.g., email threads, product listings, calendar events). Worlds are generated through the following procedure:

Target entity creation. For each task, a target entity is generated that satisfies the task’s constraint set. The target’s attributes are sampled to be consistent with the natural language instruction the agent will receive.

Hard negative injection. To calibrate task difficulty, a controlled number of hard negative entities are generated. These entities share surface-level attributes with the target (e.g., same category, similar name, overlapping tags) but differ on the specific attributes referenced in the task constraints. The number of hard negatives directly affects the agent’s required discrimination effort. In particular, hard negatives are constructed to remain ambiguous under shallow evidence, often matching the target on card- or filter-level attributes while differing only on decisive detail-level attributes.

Filler population. The remaining entity slots are populated with randomly sampled filler entities that provide a realistic background distribution. Fillers are generated to be clearly distinguishable from the target, ensuring they do not introduce ambiguity.

Deterministic seeding. All randomness in world generation is governed by a single integer seed per task. Given the same seed, the generation procedure produces an identical world, ensuring full reproducibility.

###### E.3 Oracle Trajectory Generation

Each task in WEBSTEP is accompanied by an oracle trajectory: a minimal-length sequence of semantic MDP actions that achieves the task goal from the initial state.

Generation pipeline. Oracle trajectories are produced by a template-based deterministic construction. Each task template defines a surface-visit plan (e.g., SearchResults → ProductDetail → Checkout) and emits the corresponding action sequence in the space of typed semantic actions rather than pixel coordinates. The pipeline proceeds in three stages:

- 1. Navigation to target region: The generator applies any necessary search queries, filters, or pagination to reach the list surface containing the target entity.
- 2. Hard-negative exploration: For tasks with comparison requirements, the oracle visits hard negatives in ranking order (e.g., opening each distractor’s detail page, inspecting the distinguishing attribute, and returning to the list) before opening the target entity last. This establishes the minimum exploration depth: an agent must inspect all higher-ranked hard negatives to confirm the target is correct.
- 3. Commit: The generator emits the final task-specific actions (e.g., add to cart, star, upvote) to satisfy the verifier conditions.

Optimality, non-uniqueness, and validation. Optimality is guaranteed by construction. Each template specifies the shortest path in terms of surface visits, and the visitation order over hard negatives is fixed by the ranking. However, multiple valid solutions may still exist, including alternatives with the same path length. To verify correctness, every generated

trajectory is replayed through the site’s deterministic transition function, and the final state is checked against all verifier conditions. This guarantees that every benchmark task is solvable.

Purposes. Oracle trajectories serve three roles: (i) they provide ground-truth supervision for process-level metrics (e.g., coverage computation, skill scoring); (ii) they establish an upper bound on task efficiency (optimal step count); and (iii) they are used during validation to verify that every task is solvable within the MDP.

#### F Extended Quantitative Results

- F.1 Per-site performance decomposition Table 2 reports aggregate metrics. Tables 9 to 12 provide the full per-site breakdown.

Agent Mail Cal. Shop. Acco. Food Hous. Q&A Code Jobs Chat

- OpenAI CUA 89.4 92.2 80.6 87.2 68.3 66.1 78.9 88.3 90.0 65.6

- Qwen3.5-122B 58.9 58.3 58.9 35.0 37.2 13.3 52.8 66.1 56.7 50.0 Fara 41.7 20.6 27.8 2.2 18.3 3.9 24.4 33.9 22.2 24.4 GUI-Owl 56.1 10.6 12.8 7.8 22.2 12.2 47.2 41.1 22.2 31.1

- UI-TARS 69.4 18.9 11.1 2.8 20.0 17.2 32.8 36.7 33.3 15.6

- Table 9: Terminal Success Rate (%) by site. Best per site in bold.

Agent Mail Cal. Shop. Acco. Food Hous. Q&A Code Jobs Chat OpenAI CUA 90.0 93.3 96.1 91.1 91.1 68.3 81.7 88.3 90.0 65.6 Qwen3.5-122B 59.4 75.6 72.2 65.0 88.3 24.4 72.8 66.7 56.7 50.0 Fara 50.6 38.9 47.8 51.1 78.9 12.2 46.7 34.4 22.2 25.6 GUI-Owl 58.3 30.0 20.0 48.9 76.7 20.6 56.1 41.1 22.2 31.7 UI-TARS 71.7 43.9 20.0 52.8 80.0 27.2 50.6 37.8 33.3 17.2

- Table 10: Exploration Success Rate (%) by site. Best per site in bold.

Agent Mail Cal. Shop. Acco. Food Hous. Q&A Code Jobs Chat OpenAI CUA 99.4 98.8 83.8 95.7 75.0 96.7 96.6 100.0 100.0 100.0 Qwen3.5-122B 99.1 77.2 81.5 53.8 42.1 54.5 72.5 99.2 100.0 100.0 Fara 82.4 52.9 58.1 4.3 23.2 31.8 52.4 98.4 100.0 95.6 GUI-Owl 96.2 35.2 63.9 15.9 29.0 59.4 84.2 100.0 100.0 98.2 UI-TARS 96.9 43.0 55.5 5.3 25.0 63.3 64.8 97.1 100.0 90.3

- Table 11: Execution SR | Exploration Success (%) by site. Best per site in bold.

Agent Mail Cal. Shop. Acco. Food Hous. Q&A Code Jobs Chat OpenAI CUA 76.8 91.7 99.6 94.5 98.9 91.7 41.9 58.3 14.9 41.7 Qwen3.5-122B 73.8 89.5 97.2 81.1 94.0 90.6 39.7 57.8 12.1 41.7 Fara 72.0 81.7 93.0 74.4 90.1 61.3 31.1 53.7 6.9 41.7 GUI-Owl 77.3 79.4 74.0 75.7 83.8 90.7 34.2 56.3 6.2 41.7 UI-TARS 70.5 84.9 79.6 78.9 86.2 78.3 36.9 58.3 10.4 41.7

- Table 12: Informational Coverage at Commit (%) by site. Best per site in bold.

###### F.2 Per-site skill invocation

- Table 13 reports skill invocation rates by site. Each cell shows the percentage of episodes requiring that skill where the agent invoked it.

F.3 Aggregate skill invocation

- Table 14 reports overall skill invocation rates across all sites.

F.4 Exploration SR by problem complexities

- Table 15, Table 16, and Table 17 report the numeric values underlying Figure 6 (a), (b), and (c), respectively.

Agent Site Search Filter Inspect Navigate Commit OpenAI CUA Mail 100 – 90 62 99

Cal. – – 100 92 100 Shop. 100 21 100 90 98 Acco. 97 69 100 77 99 Food 77 86 100 68 99 Hous. 98 84 88 73 89 Q&A 91 85 99 43 79 Code 93 90 100 65 98 Jobs 92 97 100 100 94 Chat 100 – 100 84 97

Qwen3.5-122B Mail 86 – 90 74 63 Cal. – – 97 89 85 Shop. 99 35 97 85 79 Acco. 87 85 96 57 79 Food 71 86 99 61 86 Hous. 97 100 87 63 63 Q&A 90 91 93 73 48 Code 82 93 99 39 86 Jobs 99 100 99 100 70 Chat 100 – 91 100 89

Fara Mail 97 – 85 38 60 Cal. – – 78 62 83 Shop. 99 71 88 64 78 Acco. 77 83 88 53 53 Food 63 81 96 40 56 Hous. 69 81 71 43 27 Q&A 77 53 71 23 18 Code 71 58 91 32 61 Jobs 88 58 68 94 51 Chat 93 – 77 94 70

GUI-Owl Mail 56 – 94 82 80 Cal. – – 39 36 52 Shop. 100 85 47 46 55 Acco. 91 100 75 22 25 Food 62 90 98 55 73 Hous. 99 100 70 48 39 Q&A 89 91 83 57 74 Code 81 90 96 10 57 Jobs 63 100 79 100 64 Chat 80 – 62 100 79

UI-TARS Mail 95 – 86 73 84 Cal. – – 56 53 90 Shop. 99 56 55 49 53 Acco. 89 100 88 44 16 Food 68 85 99 64 45 Hous. 87 97 74 52 50 Q&A 96 59 78 43 62 Code 87 42 88 42 54 Jobs 94 90 88 98 58 Chat 97 – 73 98 54

- Table 13: Skill invocation rates (%) by site and agent. ”–” = fewer than 20 required episodes.

Agent Search Filter Inspect Navigate Commit OpenAI CUA 94.5 78.9 97.7 79.8 95.0 Qwen3.5-122B 90.9 87.2 95.0 76.2 74.7 Fara 80.4 70.9 81.4 59.8 53.4 GUI-Owl 83.3 94.0 75.0 58.2 58.1 UI-TARS 90.5 77.3 79.2 65.2 52.4

- Table 14: Aggregate skill invocation rates (%). Each cell shows the fraction of episodes requiring that skill where the agent invoked it. Agent HN=0 HN=1 HN=2 HN=3 OpenAI CUA 93.4 84.3 83.2 84.2 Qwen3.5-122B 87.1 73.0 59.2 53.3 Fara 71.5 49.4 33.7 32.3 GUI-Owl 74.8 38.2 33.3 31.9 UI-TARS 68.2 44.9 38.6 35.8
- Table 15: Exploration SR (%) by hard negative count.

Agent 1 2 3 4 5 6 7 8 9 10 OpenAI CUA 94.4 84.1 83.6 94.7 80.0 93.7 82.0 85.6 90.8 69.0 Qwen3.5-122B 70.4 65.9 76.1 71.2 63.7 69.2 51.7 47.7 68.2 40.8 Fara 42.6 36.4 48.3 41.7 41.5 45.3 42.2 32.3 39.9 28.2 GUI-Owl 48.1 38.7 52.1 53.8 38.5 45.9 33.3 32.3 34.1 28.2 UI-TARS 38.9 40.4 47.5 47.0 57.8 51.6 40.8 32.3 39.9 35.2

- Table 16: Exploration SR (%) by oracle trajectory length.

Agent Card Filter Detail OpenAI CUA 69.8 94.8 86.0 Qwen3.5-122B 61.5 87.6 57.9 Fara 54.6 61.9 34.0 GUI-Owl 54.6 57.7 34.5 UI-TARS 43.9 57.0 40.3

- Table 17: Exploration SR (%) by information access level.

#### G Artifact Viewer

|[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]|
|---|

Figure 10: Artifact viewer interface. Left: task list with success/failure indicators. Center: per-turn screenshot browser with navigation controls. Right: episode metadata, per-turn action details (action type, coordinates, resolved MDP target), surface, and coverage. Bottom: semantic MDP trajectory showing the sequence of semantic actions with surface annotations.

WEBSTEP includes an interactive artifact viewer for inspecting and analyzing agent evaluation results. The viewer provides the following capabilities:

Turn-by-turn replay. Each episode can be replayed step by step, displaying the agent’s screenshot observation, the action taken, and the resulting state change at each turn. This enables detailed diagnosis of where and why an agent deviates from the optimal trajectory.

Trajectory visualization. The viewer renders the agent’s trajectory alongside the oracle trajectory, highlighting action matches and divergences. This side-by-side comparison makes it straightforward to identify systematic failure patterns.

Verification panel. For each episode, the viewer displays the full set of verifier conditions and their satisfaction status, providing immediate visibility into which conditions the agent met and which it missed.

Batch comparison. Multiple agents’ results can be loaded simultaneously, enabling comparative analysis across models on the same task set. Aggregate statistics (success rate, partial credit, failure mode distribution) are computed and displayed alongside per-task details.

We will release the viewer together with the evaluation codebase.

#### H Data Examples

###### H.1 Task Template Catalog

We present representative task templates from three sites to illustrate the diversity of task structures in WEBSTEP.

Each template defines a parameterized task schema consisting of a natural language instruction pattern, a set of constraint variables that are instantiated per task, and verifier conditions

that determine success. Templates vary in the number of required navigation steps, the complexity of the constraint set, and the type of reasoning demanded (e.g., information retrieval, comparison, multi-step form completion).

Field Value

Template find email extract Site Mail Instruction pattern ”{sender} has sent you several similar emails. Find the one that mentions

’{keyword}’ in its body and {action} it.” Information gap body, cc, attachments (detail-only fields) Constraint count 3 Commit action Star / Archive / Label

Concrete instance (gmail 0001) Instruction ”Priya Patel has sent you several similar emails. Find the one that mentions ’ProjectAlpha006’ in its body and star it.” Target entity THR-006 Hard negatives THR-019, THR-050 (same sender, different body content) Oracle trajectory SearchEmails → OpenThread(THR-019) → CloseThread →

OpenThread(THR-050) → CloseThread → OpenThread(THR-006)

→ Star(THR-006) Optimal length 7 steps

- Table 18: Task template: find email extract (Mail). Requires searching, inspecting multiple threads to find one matching body-level content, and starring it.

Field Value

Template find product extract Site Shopping Instruction pattern ”Search for {query} in {department}. Find the one with {detail attr}:

’{value}’ and add it to your cart.” Information gap specifications, bullet points, seller, shipping cost Constraint count 1–3 Commit action AddToCart

Concrete instance (amazon 0010) Instruction ”Search for fiction in Books. Find the one with Material: ’Leather’ and

add it to your cart.” Target entity PRD-039 (”Apple Essential Biography”, $12.90, Books, rating 4.7) Hard negatives Products matching ”fiction” + ”Books” but with different Material Oracle trajectory Search(”fiction”) → OpenProduct(PRD-039) → AddToCart Optimal length 3 steps

- Table 19: Task template: find product extract (Shopping). Requires searching, navigating to product detail pages, and adding the correct product to cart based on a detail-only attribute.

###### Field Value

Template search filter book Site Accommodation Instruction pattern ”Book the {superlative} {property type} in {location} that has

{amenity} for {guests} guests, {dates}.” Information gap amenities, cancellation policy, host details Constraint count 4–6 Commit action ConfirmBooking

Concrete instance (airbnb 0005) Instruction ”Book the cheapest cabin in Chicago that has a Pool for 3 guests, April

15–19.” Target entity LST-011 Constraints property type=Cabin, amenity=Pool, sort=price asc, location=Chicago Oracle trajectory Search(”chicago”) → SetFilter(property type, Cabin) → SetFil-

ter(amenities, Pool) → SortBy(price asc) → ViewListing(LST-011) → BookListing(dates, guests) → ConfirmBooking

Optimal length 7 steps

- Table 20: Task template: search filter book (Accommodation). A multi-step booking task requiring search, filter application, sort, listing inspection, and checkout completion.

H.2 Example Trajectories: Agent vs. Oracle Trajectory

Step Oracle Fara UI-TARS

0 SearchEmails("Priya Patel") OpenThread(THR-006) SearchEmails("ProjectAlpha006") 1 OpenThread(THR-019) SearchEmails("ProjectAlpha006") Star(THR-006) 2 CloseThread() OpenThread(THR-006) SwitchFolder(STARRED) 3 OpenThread(THR-050) Star(THR-006) 4 CloseThread() 5 OpenThread(THR-006) 6 Star(THR-006)

7 actions 4 actions (success) 3 actions (success)

- Table 21: Semantic trajectory comparison on mail 0001. All trajectories are shown as semantic MDP actions extracted from the environment trace. The oracle systematically inspects hard negatives before committing; both agents find target item without sufficient information.

Step Oracle Fara UI-TARS

- 0 Search("fiction") Search("Books") Search("Books")
- 1 OpenProduct(PRD-039) ApplyFilter(dept=Books) ApplyFilter(dept=Books)
- 2 AddToCart() OpenProduct(PRD-009) ClearFilters()
- 3 GoBack() ApplyFilter(dept=Books)
- 4 OpenProduct(PRD-027) NewSearch("Leather")
- 5 GoBack() NewSearch("Leather")
- 6 OpenProduct(PRD-036) ...no further progress
- 7 AddToCart() × 3 actions (success) 8 actions (failure) 6 actions (failure)

- Table 22: Semantic trajectory comparison on shopping 0010. The oracle executes a minimal 3-action sequence. Fara searches but inspects wrong products before adding one to cart (failure-wrong product). UI-TARS repeatedly reformulates queries without reaching the target.

Section H.2 and table 22 present side-by-side comparisons of agent trajectories and oracle trajectories for representative tasks from the Mail and Shopping sites. The oracle trajectory column shows the minimal-length ground-truth action sequence, while the agent trajectory column shows the actions actually taken by the evaluated agent. These examples illustrate common agent behaviors, including correct action sequences, unnecessary exploration, and error recovery attempts.

###### H.3 MDP Surface Transition Graphs

Each site’s semantic MDP can be visualized as a directed graph where nodes are surfaces (distinct UI views) and directed edges are typed semantic actions that transition between surfaces. Gray boxes connected to a node by dashed arrows represent the intra-surface action list: actions that modify state without leaving the current surface (e.g., applying filters, filling form fields, or toggling UI elements). For detailed per-site statistics including visibility partitions and skill taxonomies, see Section D.3. We present the transition graphs for all 10 sites below.

##### Mail MDP Surface Transitions

###### 3 surfaces · 36 typed actions · 8 inter-surface + 28 intra-surface

OpenThread

Star/Unstar Label MarkRead

ThreadList ThreadView

GoBack

Search Star/Unstar Archive/Delete Label SelectThread Paginate SwitchFolder

SendEmail Reply/Forward

SetTo/Cc/Bcc SetSubject SetBody Attachments

ComposeModal

Figure 11: Mail site: 3 surfaces, 36 typed actions (8 inter-surface, 28 intra-surface).

###### Calendar MDP Surface Transitions

###### 6 surfaces · 14 typed actions · 9 inter-surface + 5 intra-surface

NextWeek PrevWeek

CreateEvent

WeekView EventEditor

SaveEvent

UpdateField

OpenEvent EditEvent

EventDetail

Search

OpenEvent OpenEvent

SearchResults

NextDay PrevDay

DayView

- Figure 12: Calendar site: 6 surfaces, 14 typed actions (9 inter-surface, 5 intra-surface).

###### Shopping Site MDP Surface Transitions

###### 6 surfaces · 23 typed actions · 8 inter-surface + 15 intra-surface

Filter Sort Paginate SetBrandFilter SetPriceRange SetDeptFilter

SetColor SetSize SetQuantity AddToCart

Search OpenProduct

Home SearchResults ProductDetail

GoBack

GoBack GoBack

ViewCart

PlaceOrder ProceedToCheckout

OrderConfirm Checkout Cart

SetShippingAddr SetPaymentMethod

UpdateQuantity RemoveFromCart

- Figure 13: Shopping site: 6 surfaces, 23 typed actions (8 inter-surface, 15 intra-surface).

Accommodation MDP Surface Transitions

4 surfaces · 16 typed actions · 7 inter-surface + 9 intra-surface

Search SetFilter ClearFilters SortBy Paginate

ShowAmenities HideAmenities

CancelBooking

SearchResults ListingDetail Checkout

Reservations

ViewListing

GoBack

BookListing

CancelCheckout

ConfirmBooking

- Figure 14: Accommodation site: 4 surfaces, 16 typed actions (7 inter-surface, 9 intra-surface).

###### Food Delivery MDP Surface Transitions

###### 6 surfaces · 24 typed actions · 13 inter-surface + 11 intra-surface

ViewRestaurant ViewMenuItem

RestaurantList RestaurantDetail MenuItemDetail

GoBack GoBack

Search SetFilter SortBy Paginate

ViewCart

SetCustomization SetQuantity

SetTip DeliveryInstr.

RemoveFromCart UpdateQty

PlaceOrder Checkout

OrderStatus Checkout Cart

- Figure 15: Food Delivery site: 6 surfaces, 24 typed actions (13 inter-surface, 11 intra-surface).

###### Housing MDP Surface Transitions

###### 4 surfaces · 16 typed actions · 8 inter-surface + 8 intra-surface

ViewProperty

ViewAgent

SearchResults PropertyDetail

AgentProfile

GoBack

GoBack

GoBack ViewSaved

Search SetFilter SortBy Save/Unsave Paginate

SaveHome ContactAgent ScheduleTour

SavedHomes

Figure 16: Housing site: 4 surfaces, 16 typed actions (8 inter-surface, 8 intra-surface).

###### Coding Q&A MDP Surface Transitions

###### 4 surfaces · 18 typed actions · 12 inter-surface + 6 intra-surface

ViewQuestion

ViewUser

QuestionList QuestionDetail

UserProfile

GoBack

GoBack

Upvote/Downvote Bookmark PostAnswer PostComment AcceptAnswer

Search SetFilter Paginate

GoBack

ViewTag

TagPage

- Figure 17: Coding Q&A site: 4 surfaces, 18 typed actions (12 inter-surface, 6 intra-surface).

Code Repo Site MDP Surface Transitions

6 surfaces · 32 typed actions · 11 inter-surface + 21 intra-surface

Search SortBy LanguageFilter StarsFilter

Star Fork ViewReadme

FilterByState FilterByLabel SortBy

Comment Label Close Assign

FilterByState SortBy

Review Comment Approve

RepoList RepoDetail

IssueList IssueDetail

PRList PRDetail

OpenRepo

ViewIssues

ViewPRs

OpenIssue

OpenPR

GoBack

GoBack

GoBack

GoBack

GoBack

- Figure 18: Code Repo site: 6 surfaces, 32 typed actions (11 inter-surface, 21 intra-surface).

###### Job Network MDP Surface Transitions

###### 6 surfaces · 26 typed actions · 18 inter-surface + 8 intra-surface

SearchJobs SetFilter SortBy Save/Unsave

ViewJob

Paginate ApplyToJob Save/Unsave

JobList JobDetail

SwitchTab

GoBack

Messaging

Feed

SearchPeople Paginate Connect

SendMessage

SwitchTab

SendMessage OpenConv

ViewProfile

ConnectionList ProfileView

GoBack

- Figure 19: Job Network site: 6 surfaces, 26 typed actions (18 inter-surface, 8 intra-surface).

Team Chat MDP Surface Transitions

3 surfaces · 24 typed actions · 8 inter-surface + 16 intra-surface

SwitchChannel SendMessage React Pin/Edit/Delete OpenThread Mute Scroll

SearchMessages

SendDM React Edit/Delete

ChannelView SearchResults DirectMessage

Search

ClearSearch

OpenDM

SwitchChannel

- Figure 20: Team Chat site: 3 surfaces, 24 typed actions (8 inter-surface, 16 intra-surface).

#### I Case Study

We present seven qualitative case studies that ground our quantitative findings in concrete agent trajectories. Case 1 demonstrates how process metrics distinguish behaviorally different agents that SR treats as equivalent (Section I.1); Cases 2–3 contrast premature commitment with thorough exploration (Sections I.2 and I.3); Cases 4–5 illustrate the exploration–execution decomposition (Sections I.4 and I.5); Case 6 reveals inefficiency hidden behind a passing SR (Section I.6); Case 7 shows how a safety guardrail can cause task failure despite fully correct behavior, motivating the Safe Pass metric (Section I.7).

###### I.1 Case 1: Model Scale and Browsing Quality

David Kim hasBuy aDellWirelessMidisent you several similar emails. Find the one thatDresswith2yearswarranty WebAgent

mentions "ProjectAlpha073" in its body and star it.

|OpenAI-CUA<br><br>|Terminal Pass|
|---|---|
|[Figure 205]| |

|UI-TARS<br><br>|Terminal Pass|
|---|---|
|[Figure 206]| |

###### Trajectory Table Trajectory Table

|#|TURN|ACTION|PAYLOAD|
|---|---|---|---|
|0|4|SearchEmails|{"query":"from:David"}|
|1|8|SearchEmails|{"query":"ProjectAlpha"}|
|2|8|SearchEmails|{"query":"from:David"}|
|3|16|SearchEmails|{"query":"David Kim"}|
|4|17|OpenThread|{"thread_id":"THR-016"}|
|5|18|ExpandMessage|{"message_id":"MSG-00037"}|
|*(Similar6 to|Oracle20Trace)|ExpandMessage|{"message_id":"MSG-00039"}|
|7| |CloseThreadBrowsing<br><br>|Trajectory {}|
|8| |OpenThread|{"thread_id":"THR-033"}|
|9| |CloseThread|{}|
|10| |OpenThread|{"thread_id":"THR-073"}|
|11|25|ExpandMessage|{"message_id":"MSG-00167"}|
|12| |ExpandMessage|{"message_id":"MSG-00166"}|
|13| |Star|{"thread_id":"THR-073"}|

|#|TURN|ACTION|PAYLOAD|
|---|---|---|---|
|0|1|OpenThread|{"thread_id":"THR-016"}|
|1|5|SearchEmails|{"query":"ProjectAlpha073"}|
|2|6|Star|{"thread_id":"THR-073"}|

- Figure 21: Case 1: Model scale and browsing quality. OpenAI-CUA vs. UI-TARS on the same Mail task (”David Kim has sent you several similar emails. Find the one that mentions ProjectAlpha073 in its body and star it”). Both models succeed (Terminal Pass), yet their trajectories diverge sharply. OpenAI-CUA exhibits a more browsing-like trajectory– searching by sender name ”David Kim,” opening and expanding messages across three threads before finally starring the correct one (14 MDP actions). UI-TARS takes a shortcut, directly searching for ”ProjectAlpha073” to locate the target thread and star it in just 3 actions. The trajectory tables below each screenshot make this gap concrete: model scale produces qualitatively different exploration behavior, and our process metrics make this divergence visible where SR cannot.

###### I.2 Case 2: Premature Commitment to Hard Negative

‘SearchBuy aforDellwomen'sWireless Midi Dressdresseswith 2 yearsin Clothing.warranty Find the one with Web Agent

Warranty:"2Years"and add it to yourcart.’ UI-TARS

1. SEARCH RESULTS (woman)

2. ADD TO CART WITHOUT VERIFYING

3. DONE

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|Thought: …"Instant Pot Deluxe Cocktail Dress,"<br><br>which seems like a good option. Let me click on it to check the details and see if it comes with a<br><br>two-year warranty … Action: click (1303, 400)|
|---|

|Thought: I finally found a product that meets my requirements! The "Instant Pot Deluxe Cocktail Dress" not only comes with a two-year<br><br>warranty, but it also has a reasonable price. I'm<br><br>going to click the "Add to Cart" button right now to add it to my shopping cart, so I can continue<br><br>browsing other items.<br><br>Action: click (1269, 258)|
|---|

|Thought: I found a dress that meets all my requirements - the Instant Pot Deluxe Cocktail Dress. It not only comes with a two-year<br><br>warranty, but it's also reasonably priced at<br><br>$328.08. I just added it to my cart, and I'm all set to go.<br><br>Action: Done|
|---|

- Figure 22: Case 2: Premature commitment to a hard negative. UI-TARS on a Shopping task (”Search for women’s dresses in Clothing. Find the one with Warranty: 2 Years and add it to your cart.”). The agent clicks on the Instant Pot Deluxe Cocktail Dress from the search results (Step 1), then immediately adds it to cart without scrolling down to verify the warranty specification (Step 2), and declares the task done (Step 3). The agent’s chainof-thought confidently claims the dress ”comes with a two-year warranty,” yet this detail was never verified on the product page. This is a premature commitment: the agent selected the first plausible candidate without confirming the critical constraint, exactly the type of hard-negative failure our benchmark’s controlled difficulty design is intended to expose.

###### I.3 Case 3: Thorough Exploration Before Commitment

‘Search forBuy aDellWirelesscampingMidiDressin Sportswith2yearswarranty& Outdoors. WebAgent

Findtheonewith Weight:"0.3 lbs" and addit to your cart.’ UI-TARS

1. SEARCH RESULTS (Camping)

2. PRODUCT DETAIL (Verify)

3. SEARCH RESULTS (Camping)

4. PRODUCT DETAIL (Verify)

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|Thought: The search results are in,<br><br>and I see three camping-related<br><br>products. The first one is a sleeping bag from Instant Pot, which is<br><br>exactly what I'm looking for. I need<br><br>to check its weight specifications, so I'll click on the product title to go to the details page and take a closer look.<br><br>Action: click (538, 185)|
|---|

|Thought: I finally found the weight<br><br>information in the technical<br><br>specifications section, and it's exactly 0.3 pounds, which perfectly meets my requirements. Now, I need to go back to the search<br><br>results page to continue looking for<br><br>other suitable products. …<br><br>Action: click (196, 30)|
|---|

|Thought: I just checked the weight<br><br>of the Instant Pot sleeping bag, and<br><br>it's exactly 0.3 pounds, which is perfect. However, I still need to keep looking for other suitable products. I noticed that the search<br><br>results page shows three products,<br><br>and the second one is a Sony Classic Lantern. …<br><br>Action: click (911, 158)|
|---|

|Thought: I just checked the weight<br><br>of the Sony Classic Lantern, and it's<br><br>2.5 pounds, which is quite a bit off from the 0.3 pounds I'm looking for. It seems I need to go back to the search results page and continue<br><br>my search. I noticed there's a<br><br>"Back" button in the upper left corner of the page, and …<br><br>Action: click (196, 30)|
|---|

8. ADD TO CART

5. SEARCH RESULTS (Camping)

6. PRODUCT DETAIL (Verify)

7. SEARCH RESULTS (Camping)

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|Thought: I just checked the weight<br><br>of the Samsung lantern, and it's 3.8<br><br>pounds, which is still quite a bit away from the 0.3 pounds I'm looking for. It seems I need to go back to the search results page and<br><br>continue my search. I noticed<br><br>there's a "Back" button in the upper left corner of the page, and …<br><br>Action: click (196, 30)|
|---|

|Thought: I just checked the weight<br><br>of the Instant Pot sleeping bag, and<br><br>it's exactly 0.3 pounds, which is perfect. However, I still need to keep looking for other suitable products. I noticed that the first<br><br>product on the search results page<br><br>is the Instant Pot sleeping bag, so I'll click on it to see …<br><br>Action: click (538, 185)|
|---|

|Thought: I just checked the weight<br><br>of the Instant Pot sleeping bag, and it's exactly 0.3 pounds, which is perfect. I noticed there's a yellow "Add to Cart" button on the right<br><br>side of the page, and I need to click<br><br>it to add this product to my shopping cart. This way, I can<br><br>continue browsing other …<br><br>Action: click (1269, 276)|
|---|

|Thought: C I just checked the<br><br>weight of the Sony Classic Lantern, and it's clearly not what I'm looking<br><br>for. The third product on the search<br><br>results page is a Samsung lantern, so I'll click on it to see if it meets my weight requirements. I need to be careful with each product I check,<br><br>as I can't afford to miss …<br><br>Action: click (1271, 185)|
|---|

- Figure 23: Case 3: Thorough exploration before commitment (positive example). UI-TARS on a Shopping task (”Search for camping in Sports & Outdoors. Find the one with Weight: 0.3 lbs and add it to your cart.”). The agent systematically checks all three search results: it opens the first product (Instant Pot sleeping bag), confirms Weight = 0.3 lbs in the Technical Details (Step 2), then returns to the search results and checks the second product (Sony Classic Lantern, 2.5 lbs, Step 4) and the third (Samsung lantern, 3.8 lbs, Step 6) before returning to add the correct item to cart (Step 8). Coverage = 100% at commit. Task: PASS. A positive contrast to Case 2: the agent’s chain-of-thought explicitly tracks constraints (”I still need to keep looking for other suitable products”), verifying its choice was uniquely correct before committing.

###### I.4 Case 4: Exploration Success, Execution Failure

‘Search for "spring-boot". Open the question "How to debug memory leaks in in Java Spring Boot?" and upvote

###### Figure 2. Case Study 2: Exploration Success, Execution Failure (Finding F1)

‘SearchBuy aforDellrepositoriesWireless Midi Dressrelatedwithto2 years"api".warrantyFind the repository Web Agent

Buy a Dell Wireless Midi Dress with 2 years warranty Web Agent

"ultra-pipeline"andstarit.’ Fara

Fara

the answer by bug_hunter.’

CODING Q&A UPVOTE TARGET

CODING REPO STAR TARGET

|Thought:<br><br>Click the upvote button<br><br>for the answer by<br><br>bug_hunter to upvote it Action: click (150, 157)|
|---|

|Thought:<br><br>To star the 'ultra-<br><br>pipeline' repository as<br><br>requested by the user Action: click (1275, 250)|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

###### Verification:

###### Verification:

- 1. EXPLORATION SUCCESS (Navigate to Target page)

[Figure 220]

- 2. EXECUTION FAILURE (Apply Upvote action) ✘

- 1. EXPLORATION SUCCESS (Navigate to Target page)

[Figure 221]

- 2. EXECUTION FAILURE (Apply Star action) ✘

- Figure 24: Case 4: Exploration success, execution failure. Two examples from Fara on Coding tasks. Left: Fara navigates to the correct StackOverflow answer by bug hunter and attempts to upvote it, but the click coordinates (150, 157) land on the code block instead of the upvote button (Verification: Exploration Success ✓, Execution Failure ×). Right: Fara navigates to the correct GitHub repository ”ultra-pipeline” and attempts to star it, but the click coordinates (1275, 250) miss the Star button (Verification: Exploration Success ✓, Execution Failure ×). This execution bottleneck (missing small UI targets due to imprecise coordinate grounding) accounts for much of Fara’s gap between its Exploration rate and task SR.

###### I.5 Case 5: Exploration Failure, Execution Success

‘Search for "api" and filter by TypeScript language.BuyaDellWirelessMidiDresswith2yearswarranty Web Agent Sort by forks and star the one with the most forks.’ Fara

2. REPO PAGE

###### 3. SET REPO SORT

4. SET REPO SORT ‘MOST FORK’

1. SET REPO FILTER ‘Typescript’, BUT FAILED

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

|Thought: Return to the filtered repository list to sort them by forks<br><br>Action: HistoryBack|
|---|

|Thought: Open the sort-by dropdown to change sorting from Best match to Forks Action: click (1259, 189)|
|---|

|Thought: Click the 'Most forks' option to sort the repositories by fork count descending<br><br>Action: click (1282, 296)|
|---|

|Thought: Select 'TypeScript' from the language filter dropdown to filter repositories by TypeScript<br><br>Action: click (206, 479)|
|---|

5. STAR TOP REPO

###### 6. STARRED

|[Figure 226]|
|---|

[Figure 227]

|Thought: Click the 'Star' button for the top-forked repository (/tinyloader) to mark it as a favorite Action: click (1273, 249)|
|---|

|Thought: The requested actionssearching for “api”, filtering by TypeScript, sorting by forks, and starring the top repository—have been completed. We can now exit the task. Action: DONE|
|---|

- Figure 25: Case 5: Exploration failure, execution success. Fara on a Code Repo task (”Search for ’api’ and filter by TypeScript language. Sort by forks and star the one with the most forks.”). At Step 1, the agent intends to select ”TypeScript” from the Language filter dropdown, but instead clicks a ”TypeScript” label displayed in a repository’s language listing: mistaking it for the correct filter option. The filter is not applied, yet the agent proceeds to navigate back (Step 2), sort by forks (Steps 3–4), and star the top repository (Step 5), declaring success at Step 6. The execution mechanics are correct; the exploration target is wrong. Task: FAIL. The mirror image of Case 4: our four-way taxonomy classifies this as an exploration failure with correct execution: the agent knew how to commit but not where to look.

###### I.6 Case 6: Redundant Process Despite Success

Add theBuy aApple Deluxe Mobile andDell Wireless Midi Dress with 2 yearsthewarrantySamsung Essential Pot Set to Web Agent

yourcart.Thenremove theApple Deluxe Mobile fromthe cart. Fara TerminalPass

Step 8. Modify Search

Step 9. Modify Search

Step 10. Back to Home

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

… …

|Thought: Clear the current query and<br><br>search for a broader term 'Samsung pot' to<br><br>locate similar products Action: Type “Samsung Pot”|
|---|

|Thought: Type the second product name<br><br>into the search bar to locate the Samsung<br><br>Essential Pot Set Action: Type “Samsung Essential Pot Set”|
|---|

|Thought: Open the navigation menu to access the categories section and browse for a pot sets listing Action: click (84, 29)|
|---|

Oracl Trajectory (9 steps · target: PRD-037) Agent Trajectory (13 steps)

###### # TURN ACTION PAYLOAD

###### # ACTION PAYLOAD

- 0 2 Search {"query":"Apple De…"}
- 1 3 OpenProduct {"product_id":"PRD-014"}
- 2 4 AddToCart {}
- 3 5 ViewCart {}
- 4 6 ContinueShopping {}
- 5 8 NewSearch {"query":"Apple Deluxe MobilSamsung … "}
- 6 9 NewSearch {"query":"Apple Deluxe MobilSamsung … “}
- 7 10 GoHome {}
- 8 11 Search {"query":"Home &…"}
- 9 12 OpenProduct {"product_id":"PRD-037"}
- 10 13 AddToCart {}
- 11 14 ViewCart {}
- 12 15 RemoveFromCart {"product_id_to_remove"}

- 0 Search {"query":"smartphones"}
- 1 OpenProduct {"product_id":"PRD-014"}
- 2 AddToCart {}
- 3 GoBack {}
- 4 NewSearch {"query":"cookware"}
- 5 OpenProduct {"product_id":"PRD-037"}
- 6 AddToCart {}
- 7 ViewCart {}
- 8 RemoveFromCart {"product_id_to_remove":"…"}

2steps

6steps

- Figure 26: Case 6: Redundant process despite success. Fara on a Shopping task (”Add the Apple Deluxe Mobile and the Samsung Essential Pot Set to your cart. Then remove the Apple Deluxe Mobile from the cart.”). The oracle trajectory completes the task in 9 MDP actions; Fara executes 13, with 6 extra steps highlighted in orange in the agent trajectory table. After successfully adding the first product, Fara issues garbled concatenated queries (”Apple Deluxe MobilSamsung Essential Pot Set...”) that return no results (Steps 8–9), navigates to the home page (Step 10), and searches by category (”Home &...”) before finally locating the second product. The task ultimately succeeds (Terminal Pass), but the extra steps reveal inefficient error recovery. Our process-recording system captures this waste: process-level evaluation distinguishes efficient success from success-by-persistence.

###### I.7 Case 7: Failure by Safeguard, Not by Competence

Book the cheapest condo in Seattle that has aBuy aDellWirelessMidiDresswith2yearswarranty WebAgent

Pool for 3 guests, April14-17. Fara

Checkout Page

|[Figure 231]|
|---|

|Thought: We have selected the cheapest condo with a pool for 3<br><br>guests from April 14–17 and reached the final confirmation page.<br><br>Proceeding further would require entering personal and payment<br><br>information, which is beyond the allowed scope.<br><br>Action: Done|
|---|

- Figure 27: Case 7: Failure by safeguard, not by competence. Fara on an Accommodation task (Book the cheapest condo in Seattle that has a Pool for 3 guests, April 14–17.”). The agent successfully explores all listings, identifies the correct property (Vibrant Loft in Capitol Hill, $485/night), and navigates to the final ”Confirm and pay” page, achieving Exploration: Pass, Coverage: 100%, and Hard Neg: 0. However, Fara is trained to stop when completing the task would require entering personal and payment information, so it issues Done at step 16 without clicking ”Confirm and pay.” The task is marked FAIL under standard SR, yet our Safe Pass metric recognizes that the agent performed every step within its allowed scope correctly. This case motivates the Safe Pass metric: it distinguishes agents that fail due to a built-in safety guardrail from those that fail due to genuine incompetence.

