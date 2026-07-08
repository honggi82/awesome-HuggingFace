### arXiv:2603.13428v2[cs.SE]5Jun2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

# EvoClaw: Evaluating AI Agents on Continuous Software Evolution

[Figure 8]

Gangda Deng1,*, Zhaoling Chen2,*, Zhongming Yu3, Haoyang Fan1, Yuhong Liu1, Yuxin Yang1, Dhruv Parikh1, Rajgopal Kannan4, Le Cong5, Mengdi Wang6, Qian Zhang2, Viktor Prasanna1, Xiangru Tang7,†, Xingyao Wang8

1USC 2UCR 3UCSD 4Army Research Office 5Stanford 6Princeton 7Haven 8OpenHands

*Equal Contribution †Corresponding Author

DeepCommit Pipeline: github.com/DeepCommit-ai/DeepCommit EvoClaw Benchmark: github.com/EvoClaw-Bench/EvoClaw Data: huggingface.co/datasets/EvoClaw-Bench/EvoClaw-data Leaderboard: evo-claw.com

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

With AI agents increasingly deployed as long-running systems, it becomes essential to autonomously construct and continuously evolve customized software to enable interaction within dynamic environments. Yet, existing benchmarks evaluate agents on isolated, one-off coding tasks, neglecting the temporal dependencies and technical debt inherent in real-world software evolution. To bridge this gap, we introduce DeepCommit, an agentic pipeline that reconstructs verifiable Milestone DAGs from noisy commit logs, where milestones are defined as functionally cohesive development goals. These executable sequences enable EvoClaw, a novel benchmark that requires agents to sustain system integrity and limit error accumulation, dimensions of long-term software evolution largely missing from current benchmarks. Our evaluation of 12 frontier models across 4 agent frameworks reveals a critical vulnerability: overall performance scores drop significantly from >80% on isolated tasks to at most 38% in continuous settings, exposing agents’ profound struggle with long-term maintenance and error propagation.

[Figure 13]

[Figure 14]

Figure 1 | Milestone-level task granularity optimally balances functional coherence and evolutionary awareness for benchmarking continuous software evolution.

#### 1. Introduction

Frontier LLM-powered agents (e.g., Claude Code (Anthropic, 2025), Codex (OpenAI, 2025)) are increasingly deployed as long-running systems (e.g., OpenClaw (OpenClaw, 2026) and Hermes (Nous Research, 2026)) into complex, open-ended environments. To operate effectively in these dynamic settings, such agents must treat their environment-facing interfaces as a maintainable software system—one that requires autonomous development and continuous refinement rather than a fixed, hand-crafted tool. As the agent iteratively adapts to successive requirements from end-users and ongoing environmental feedback, its continuous development efforts accumulate, naturally forming a complete repository evolution history.

Contact: Gangda Deng (gangdade@usc.edu), Zhaoling Chen (zchen526@ucr.edu), Xiangru Tang (xiangru.tang@yale.edu)

- Table 1 | Representative software engineering benchmarks for LLMs. Unlike other categories of benchmarks that evaluate agents on isolated snapshots or against ground-truth states at each step, Repository Evolution requires agents to continuously build upon their own accumulated development history, exposing them to error propagation across tasks. EvoClaw adopts the Milestone-level granularity, a functionally coherent group of commits that collectively advance a development objective, avoiding the noise of individual commits and the excessive scope of full releases. Dev History denotes the agent’s own development trace accumulated from preceding tasks.

Task Properties Cross-task Dependency

Task

Category Benchmark Language

Granularity Avg LoC Additional Context Collection Function Completion HumanEval (Chen et al., 2021) Python Function-level 6.8 – – Manual

SWE-bench (Jimenez et al., 2024) Python Commit-level 32.8 Codebase – Rule-based SWE-rebench (Badertdinov et al., 2026) Python Commit-level 142 Codebase – Rule-based Multi-SWE-bench (Zan et al., 2026) Multi Commit-level 246 Codebase – Rule-based SWE-bench Pro (Deng et al., 2025) Multi Commit-level 107 Codebase – Rule-based SWE-CI (Chen et al., 2026) Python Commit-level ∼60 Codebase + Oracle Test Commit Chain Rule-based

Issue Resolution

SWE-Dev (Du et al., 2026) Python Feature-level 190 Codebase – Rule-based FeatureBench (Zhou et al., 2026) Python Feature-level 790 Codebase – Test-driven SWE-EVO (Thai et al., 2025) Python Release-level 611 Codebase – Rule-based Commit0 (Zhao et al., 2025) Python Project-level >3k Codebase Skeleton – Rule-based NL2Repo (Ding et al., 2026) Python Project-level >3k – – Rule-based ProgramBench (Yang et al., 2026) Multi Project-level >8k Executable – Agentic

Codebase Generation

Repository Evolution EvoClaw (Ours) Multi Milestone-level 570 Codebase + Dev History Milestone DAG Agentic

Yet, evaluation for such long-running agent systems remains largely under-explored. While benchmarks for agents on coding tasks have advanced from isolated function completion (Chen et al., 2021) to full-scale codebase generation (Yang et al., 2026; Zhao et al., 2025) (Table 1), they predominantly treat development as independent tasks (Deng et al., 2025; Jimenez et al., 2024; Zan et al., 2026). A critical dimension remains unaddressed: the temporal structure of software evolution. A true repository evolution benchmark must capture the full evolution itinerary—a continuous stream of dependent tasks where early implementation decisions constrain subsequent ones. Ignoring these dependencies allows agents to take expedient shortcuts that satisfy immediate tests but silently accumulate technical debt (Chen et al., 2026; Orlanski et al., 2026), undermining the long-term maintainability of the codebase, a failure mode that remains invisible to current isolated evaluations (Yao, 2025).

To capture these long-term dynamics, our benchmark replays the dependency-rich evolution of highquality open-source repositories (Badertdinov et al., 2026; Fu et al., 2026; Jain et al., 2025; Pan et al., 2025) as a high-fidelity proxy for the continuous repository evolution that software as a skill demands. However, determining the appropriate task granularity is non-trivial (Figure 1). Intuitively, one might attempt to measure evolution at the release-level (Thai et al., 2025). However, this granularity is too coarse: release snapshots collapse the hundreds of interdependent commits between versions into a single update, flattening the fine-grained dependency structure that drives evolutionary changes. In contrast, the commit-level history (Chen et al., 2026; Jimenez et al., 2024) is too fine-grained and imbalanced: many commits are trivial (e.g., typo fixes) while a few are substantive, and the linear commit sequence encodes only chronological apply order, introducing spurious dependencies between unrelated changes (Fan et al., 2024; Herzig et al., 2016).

To address this, we propose modeling software evolution at the Milestone-level. We define a milestone

- as a coherent functional unit that preserves dependency constraints. This granularity strikes a crucial balance: unlike releases, it retains the fine-grained development dependencies and structural evolution of the codebase; unlike commits, it encapsulates realistic and coherent functional goals. Functional dependencies among milestones naturally form a Directed Acyclic Graph (DAG), which captures genuine prerequisite constraints while allowing independent features to proceed in parallel. However, constructing Milestone DAGs requires reordering and grouping commits, which disrupts the native

git history. This poses a severe challenge to correctness: applying reordered patches often breaks compilation and test collection, jeopardizing the benchmark’s executability and realism.

To resolve this, we introduce DeepCommit, an automated agentic pipeline that reconstructs verifiable software evolution itineraries in the form of Milestone DAGs. By synergizing static analysis, LLMagent-driven milestone construction, and runtime validation, DeepCommit ensures the synthesized milestones are executable and testable. Powered by Claude Opus 4.5, it achieves a high average test collection success rate of 87.1%, ensuring comprehensive verification coverage. Designed as a scalable agentic framework, DeepCommit is poised to leverage future LLM advancements to harvest increasingly accurate and extensive evolution itineraries from the vast open-source ecosystem (Fu et al., 2026; Jain et al., 2025; Pan et al., 2025).

Building on this foundation, we present EvoClaw, a benchmark for evaluating LLM agents under continuous software evolution. EvoClaw comprises 98 human-verified milestones across 7 evolution itineraries (Milestone DAGs), each from a release range of a unique high-impact open-source repository, and spanning five programming languages. Rather than solving independent tasks, agents in EvoClaw are tasked with evolving a codebase through streams of these dependency-constrained milestones, closely mirroring real-world development scenarios. A single full evaluation costs approximately $500 with frontier models such as Claude Opus 4.5. To achieve a high score in this setting, an agent must maintain long-term context, manage architectural consistency, and prevent error accumulation across extended development horizons.

Using EvoClaw, we conduct a comprehensive evaluation of 4 frontier agent frameworks and 12 state-of-the-art LLMs. We assess performance using a unified Score (Section 5.1), which balances Recall (completeness of new feature implementation) and Precision (robustness against regressions), along with a strict Resolve Rate for fully completed milestones. Our evaluation reveals the following key findings regarding agent capabilities in continuous software evolution:

- • A fundamental performance gap: Continuous vs. Independent. (Section 5.2) Frontier models exhibit a substantial degradation from independent to continuous task evaluation. Scores drop from over ∼80% on isolated tasks to at most 38.03% (Claude Opus 4.6) in continuous environments, with a mere 13.37% Resolve Rate (Gemini 3 Pro).
- • Recall grows linearly but Precision saturates. We identify a fundamental asymmetry in continuous software evolution (Section 5.4): while frontier agents retain the capability to implement new features (linear Recall growth), they fail to prevent regressions as the system evolves (saturated Precision). This indicates that agents struggle primarily with system-level maintenance rather than local implementation.
- • Accumulated errors stall downstream progress. Unresolved regressions trigger a “snowball effect” where errors accumulate faster than agents can fix them (Section 5.5). Early bugs propagate through dependency chains to contaminate downstream tasks, eventually stalling development entirely.
- • Proactive exploration and verification mitigate technical debt. Behavioral analysis shows that successful sustained evolution relies on proactive codebase exploration and disciplined test verification, whereas both blind trial-and-error and the absence of verification accelerate failure (Section 5.6).

#### 2. Related Work

LLM-Driven Coding Agents. While basic Bash tools alone can drive an LLM through software tasks (Yang et al., 2025), specialized scaffolding is what unlocks reliable, efficient, and user-friendly behavior. Scaffolds have evolved from predefined pipelines (Xia et al., 2025) to fully autonomous

systems (Wang et al., 2025; Yang et al., 2024). Commercial deployments span complementary use cases: Devin (Labs, 2024), GitHub Copilot (GitHub, 2025), Cursor (Cursor, 2024), Trae (TRAE,

- 2025), and Antigravity (Google, 2025a) integrate agents into IDE and cloud workflows, while Claude Code (Anthropic, 2025), Codex (OpenAI, 2025), and Gemini CLI (Google, 2025b) run in the terminal. While these agents are well-validated on independent, interactive tasks, their reliability under longhorizon, fully autonomous operation remains a formidable challenge. EvoClaw addresses this gap with a standardized, quantitative evaluation that surfaces fine-grained progress-level feedback.

Automated SWE Environment Synthesis. A growing line of work automatically constructs testable Dockerized environments from open-source repository snapshots. These environments primarily serve to continuously refresh evaluation data (Badertdinov et al., 2026; Zhang et al., 2026) and scale agent training (Fu et al., 2026; Jain et al., 2025; Pan et al., 2025). Unlike these snapshotbased approaches, DeepCommit preserves the temporal structure of development trajectories by reorganizing commit histories into Milestone DAGs, thereby synthesizing long-horizon tasks with verifiable progress checkpoints.

Issue Resolution Tasks. Unlike Terminal-Bench (Merrill et al., 2026), which tests agents on shell commands isolated from any codebase, SWE benchmarks require modifying real-world GitHub repositories. SWE-bench (Jimenez et al., 2024) pioneered this line by pairing issues with hidden tests, followed by SWE-bench Pro (Deng et al., 2025) for data hygiene and Multi-SWE-bench (Zan et al., 2026) for multilingual coverage.

Codebase Generation Tasks. Codebase generation benchmarks pursue longer-horizon evaluation by progressively enlarging the scope of a single task. At the feature level, SWE-Dev (Du et al.,

- 2026) and FeatureBench (Zhou et al., 2026) require agents to implement complete features against curated test suites. At the release level, SWE-EVO (Thai et al., 2025) bundles all changes between consecutive release tags into a single task. At the repository level, Commit0 (Zhao et al., 2025), NL2Repo (Ding et al., 2026), and ProgramBench (Yang et al., 2026) push agents toward synthesizing entire repositories from scratch. Despite the enlarged scope, these benchmarks evaluate task instances in isolation on a reset codebase, leaving inter-task dependencies and accumulated technical debt unmodeled. EvoClaw instead links milestones through a dependency DAG over a persistent repository, making both intermediate progress and cross-task error propagation directly measurable.

Continuous SWE Tasks. Real software development involves a stream of dependent tasks on the same codebase. While LLMs are known to degrade in this setting relative to single-shot evaluation (Laban et al., 2026), dedicated benchmarks remain nascent. SlopCodeBench (Orlanski et al., 2026) measures structural erosion and verbosity drift on hand-crafted tasks where an agent extends a codebase built from scratch. A complementary thread evaluates agents via continuous integration (CI) loops on real repositories (XU et al., 2026). SWE-CI (Chen et al., 2026), for instance, chains commit-to-commit CI rounds by exposing ground-truth tests to a dual-agent protocol following the native commit order. EvoClaw instead adopts a coarser, functionally coherent granularity by grouping commits into selfcontained milestones whose dependencies form a DAG. This approach offers a more faithful and flexible model of continuous software evolution than scattered commit-level CI.

Performance Optimization Tasks. A parallel direction evaluates whether agents can speed up code across repositories (He et al., 2025; Ma et al., 2025; Shetty et al., 2026), numerical algorithms (Press et al., 2026), and GPU kernels (Ouyang et al., 2025). While these works instantiate long-horizon evaluation, they focus on closed-ended optimization objectives, such as wall-clock time against an expert reference. In contrast, EvoClaw addresses the open-ended challenge of general functional software development.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 2 | The DeepCommit pipeline architecture. Phase 1 extracts structured data from commit history through static analysis, including source filtering, commit extraction, PR/Issues, releases, commit DAG, code metrics, and symbol changes. Phase 2 employs an LLM agent to construct a Milestone DAG via four iterative stages: seed discovery, milestone consolidation, dependency inference, and milestone decomposition. Phase 3 resolves runtime dependencies through testbed construction and test collection, with DAG refinement and fallback patches as repair strategies, followed by flaky test filtering to produce an executable testbed. Quality Assurance validates outputs at textual, compilation, and test collection levels. See Appendix B.4 for all DAG visualizations.

#### 3. DeepCommit: An Automated Pipeline for Reconstructing Software Evolution

###### 3.1. From Raw Commits to Milestone DAGs

Software repositories encode rich evolutionary trajectories, yet raw commit histories remain noisy, fragmented, and inadequate as executable development sequences. Commits vary widely in granularity, semantic clarity, and dependency structure, while parallel branches, squash merges, and non-functional changes obscure true developmental relationships. Relying on documentation or release notes alone lacks sufficient resolution to reconstruct precise code evolution. DeepCommit addresses this challenge by transforming linear git histories into structured, verifiable Milestone DAGs, where each node represents a coherent, testable unit of development and edges encode dependency constraints across evolution phases.

###### 3.2. Overall Agent-Driven Pipeline

As illustrated in Figure 2, DeepCommit reconstructs software evolution itineraries through an endto-end pipeline that sequentially integrates: (1) commit history preprocessing, (2) Milestone DAG construction, and (3) executable environment resolution.

###### 3.2.1. Commit History Preprocessing

We model each repository’s main-branch range between release tags as a linear sequence of commits. This linearization aligns naturally with the squash-and-merge workflow, the most widely adopted convention in actively-maintained, high-quality repositories, where each main-branch commit maps to a Pull Request (PR) or Issue together with its internal sub-commits. We collect all main-branch commits and their PR, Issue, and Release metadata. An LLM agent then prepares a per-repo configuration of source directories, test patterns, and exclusion rules that separates product-logic source from test code and filters out commits touching only non-source files such as docs, CI configs, and build assets (Appendix A.1). To enable downstream milestone discovery and dependency inference, we extract three structural signals via static analysis: (i) a commit-level DAG built with git blame to capture line-level textual dependencies, (ii) symbol-level modifications identifying changes in classes and functions, and (iii) file-level co-change statistics reflecting evolutionary coupling.

###### 3.2.2. Milestone DAG Construction

Organizing hundreds of discrete commits into functionally coherent milestones requires integrating structural dependencies with code-level reasoning. We employ a four-stage LLM-agent-driven process to progressively construct the Milestone DAG. Each stage is orchestrated with automated data preparation, agent-accessible validation tools, and a post-stage quality gate. A stage runs in a forward pass and may iterate internally against its self-checks. At stage boundaries, the pipeline can also fan out into multiple parallel instantiations of the next stage and retain the variant that best satisfies the downstream quality gate.

Seed Discovery. Each milestone is initiated by a seed commit, namely a foundational anchor that introduces a distinct development theme. An LLM agent identifies such seeds by jointly evaluating commit semantics (commit messages and linked discussion context) and structural signals (DAG topology, including out-degree and descendant count), filtering out cosmetic edits, hotfixes, and follow-up patches that lack downstream structural influence.

Milestone Consolidation. For each seed, parallel sub-agents expand the milestone boundary using shared file modifications, temporal proximity, and PR/Issue references, growing each seed into a milestone that bundles all commits realizing its development theme. Since the sub-agents operate independently, the same commit may be claimed by several milestones. A coordinating agent then resolves these overlapping claims so that each commit belongs to exactly one milestone, enforces complete coverage of the range, and certifies acyclicity.

Dependency Inference. The majority of inter-milestone edges follow directly from the line-level textual dependencies extracted during preprocessing. More subtle dependencies, such as call relationships that share no common hunk, are proposed and validated by an LLM agent using symbol-level analysis and file co-change patterns. Even so, certain dependencies surface only when milestones are built and executed. These residual edges are recovered later during runtime environment resolution (Section 3.2.3).

Milestone Decompose. Oversized milestones are decomposed into functionally independent submilestones while underspecified ones are merged into adjacent neighbors, with dependencies synchronously updated to preserve a valid DAG. When an oversized milestone is dominated by a single squashed PR-commit, the agent further re-segments the commit’s diff along feature boundaries and remaps the affected dependency edges via line-level blame, promoting the resulting sub-units to first-class milestones. The pipeline targets a coefficient of variation CV < 1.0 over per-milestone LoC, achieving CV = 0.96 on EvoClaw (Appendix A.2).

###### 3.2.3. Runtime Environment Resolution

To transform the Milestone DAG into an executable evaluation environment, a MainAgent orchestrates a multi-agent workflow that produces, for every milestone, a reproducible Docker image that yields stable test signals. From a whole-evolution perspective, the MainAgent balances testbed quality against the cost of automated resolution. To achieve high quality at reasonable cost, the pipeline additionally relies on human-expert guidance to steer the MainAgent at key decision points. The MainAgent then dispatches sub-agents for batch analysis and problem localization, routing the surfaced issues to two specialized repair modules that iterate together until every milestone reaches a stable, fully collected state.

Milestone DAG Optimization and Testbed Preparation. A MilestoneAgent reconstructs each milestone’s code state by cherry-picking its commits in topological order onto the codebase at the base release tag. When a cherry-pick conflict arises, the agent repairs the DAG, primarily by adding the missing cross-milestone dependency edge and, when necessary, relocating misattributed commits to the correct milestone. Commits that still cannot be applied are marked as deferred, so that every milestone is reconstructed into a complete, DAG-consistent state.

Environment Configuration and Test Collection. For each milestone, an EnvAgent generates a Dockerfile from the repository’s CI/CD workflows and enforces three hard gates: the source compiles, the test framework collects successfully, and as many tests referenced by the milestone’s patch as possible are captured in the collected set. A characteristic failure mode arises when the configured test environment runs ahead of the cherry-picked source state, so that functional modules referenced by the collected tests do not yet exist. These cases cannot be fixed locally and are deferred to the DAG optimization module for refinement (Appendix A.3).

###### 3.3. Automated Quality Assurance

To ensure the reliability and reproducibility of our evaluation, we rigorously validate each milestone testbed and report the aggregate evidence across three core dimensions, complementing the permilestone gates enforced in Section 3.2.3:

Milestone Graph Validity. We verify the structural integrity of the reconstructed history. This includes confirming commit completeness (100% coverage of the target range), dependency consistency (ensuring milestone dependencies respect underlying commit dependencies), and DAG correctness (validating acyclicity).

Runtime Executability. We ensure that errors stem from agent code, not infrastructure. We verify testbed compilability by ensuring successful build and test collection in both states. We also strictly monitor execution logs to ensure environment-induced errors remain negligible (≤0.10%).

Evaluation Reliability. We assess the stability of the test suites. We achieve a high test collection rate (87.1%). We ensure test consistency by validating a negligible Pass-to-Fail rate (≤0.026%) and filtering flaky tests through three repeated runs. Finally, we require each retained milestone to expose at least one F2P or N2P test signal (details in Appendix A.4).

#### 4. EvoClaw: Benchmarking Continuous Software Evolution

EvoClaw introduces a novel evaluation paradigm designed to assess an agent’s ability to evolve and maintain a software codebase over an extended lifecycle. As shown in Figure 3, unlike traditional benchmarks that focus on resolving independent issues, EvoClaw simulates a realistic, continuous development process where requirements arrive as a stream, and tasks have strict sequential dependencies.

a. Independent Task Evaluation (e.g., SWE-bench)

b. Continuous Task Evaluation (EvoClaw)

(iii)

Task Stream

###### Task Dependency

###### Task Dependency

|SRS 1<br><br>[Figure 23]<br><br>SRS 2<br><br>[Figure 24]<br><br>Task List<br><br>SRS 3<br><br><br>[Figure 25]|
|---|

|SRS 1<br><br>[Figure 26]<br><br>SRS 2<br><br>[Figure 27]<br><br>Task List<br><br>SRS 3<br><br><br>[Figure 28]|
|---|

Issue

- 1 3 4

- 2

- 1 3 4

- 2

|SRS 1<br><br>[Figure 29]<br><br>SRS 2<br><br><br>[Figure 30]<br><br>Task List|
|---|

[Figure 31]

[Figure 32]

[Figure 33]

(i) (ii)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Fetch Next Task

[Figure 41]

…

## …

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Finished Task M1&M2; Check the new tasks.

Finished Task M3; All Tasks Done.

Agent Loop

[Figure 46]

Agent Repo Snapshot

[Figure 47]

###### Patch

[Figure 48]

Codebase

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Tag-M1 Tag-M3

Tag-M2

Base Snapshot

Test Evaluation

[Figure 57]

[Figure 58]

[Figure 59]

In state (i), M1 and M2 were unlocked, while M3 remained locked.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Result (fail/pass)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

In State (ii), Unlock M3 when “M1, M2 done”

Result (fail/pass)

Test Evaluation

Test Evaluation

Test Evaluation

- Figure 3 | Illustration of the evaluation pipelines. (a) In the Independent Task Evaluation Workflow, the environment resets after each task. (b) In the Continuous Task Evaluation Workflow, tasks are organized as a dependency graph. The agent continuously evolves the Codebase from a base snapshot. Upon completing a task (e.g., M1 & M2), the repository is snapshotted for Isolated Evaluation while the planner unlocks subsequent tasks (e.g., M3) for the agent to fetch, ensuring a continuous and stateful development loop.

###### 4.1. The Continuous Task Evaluation Framework

The framework orchestrates a continuous development pipeline: an external planner dynamically unlocks tasks based on a dependency graph, the agent implements them in a persistent codebase, and the framework asynchronously evaluates snapshots upon submission. This design explicitly decouples roadmap planning from implementation, allowing us to assess the agent’s ability to maintain and evolve software within a structured workflow. The framework comprises three core components:

Dependency-Driven Task Stream. Requirements are not presented in a static batch but are unlocked dynamically. The system maintains a DAG-based task scheduler where a new milestone 𝑀𝑖 becomes available to the agent if and only if all its prerequisite milestones {𝑀𝑗 | 𝑀𝑖 𝑑𝑒𝑝𝑒𝑛𝑑𝑠 𝑜𝑛 𝑀𝑗} have been completed. This simulates real-world constraints in which foundational features must be established before dependent features are implemented.

Continuous Evolution Environment. The agent operates within a persistent, stateful environment where modifications from each task persist into the next. This compels the agent to maintain the long-term health of the codebase, as early technical debt or latent bugs can accumulate and impede future progress.

Snapshot-Based Isolated Evaluation. To reconcile the need for a continuous development flow with rigorous verification, we employ a “develop-in-place, evaluate-in-isolation” strategy. Upon task completion, the agent’s implementation state is snapshotted and transferred to an isolated evaluation container to run the test suite. This ensures that the scoring process is reproducible and unaffected by the agent’s ongoing development, while the agent’s working environment remains uninterrupted.

###### 4.2. Benchmark Construction

We construct a high-quality dataset through a rigorous pipeline that transforms open-source repositories into verified evolutionary suites.

|Feature<br><br>| |
|---|
<br><br>Enhance<br><br>| |
|---|
<br><br>Bugfix<br><br>| |
|---|
<br><br>Refactor|
|---|

23

go-zero

Go

| | | | |
|---|---|---|---|

- 12

11

- 13

dubbo ripgrep nushell

Java

| | | |
|---|---|---|

Rust

| | | | |
|---|---|---|---|

Rust

| | | | |
|---|---|---|---|

18 9

element-web navidrome scikit-learn

TS

| | | | |
|---|---|---|---|

Go JS

| | | | |
|---|---|---|---|

12

Python

| | | | |
|---|---|---|---|

0 5 10 15 20 25

Number of Milestones

(a) Distribution of milestones across the 7 repositories in EvoClaw. Each bar represents the number of verified milestones extracted from the corresponding open-source project.

|Human Writing Time<br><br>| |
|---|
<br><br><30m 30m-1h<br><br>| |
|---|
<br><br>| |
|---|
<br><br>1-2h<br><br>| |
|---|
<br><br>2-4h<br><br><br>| |
|---|
<br><br>4h+|
|---|

|Human Dev Time<br><br>| |
|---|
<br><br>1-4h 4h-1d<br><br>| |
|---|
<br><br>| |
|---|
<br><br>1-3d 3-5d<br><br>| |
|---|
<br><br>| |
|---|
<br><br>5d+|
|---|

40

15

| |
|---|
| |
| |

| |
|---|
| |
| |
| |

10

Count

Count

| |
|---|
| |
| |

| |
|---|
| |
| |

20

5

0

0

200-500500-800800-1.2K1.2K-1.8K1.8K+

0-100100-300300-700700-1.5K1.5K+

Gold Patch LOC

SRS Word Count

(b) Distribution of SRS word counts (left) and gold patch LOC (right), stratified by estimated human effort.

- Figure 4 | Dataset statistics and characteristics of EvoClaw.

Repository and Range Selection. We identify projects with high community impact and diverse programming languages. We specifically select release ranges that exhibit rich dependency structures, ensuring the benchmark captures complex, non-linear development scenarios rather than trivial sequences.

Itinerary Extraction via DeepCommit. Leveraging the DeepCommit pipeline (Section 3), we mine the evolutionary history of selected projects. To guarantee the benchmark’s quality and evaluation efficiency, we apply strict post-processing filters to the generated milestones. We retain only milestones that: (1) represent core functional changes (filtering out pure documentation updates); (2) possess executable F2P tests to serve as definitive success criteria; and (3) fall within a manageable context window to maintain task solvability. This step ensures that every task in the benchmark is grounded in a verified, executable state transition.

Reverse-Engineering Software Requirement Specifications (SRS). Relying solely on original GitHub issues or PR descriptions is often insufficient, as they can be underspecified, outdated, or disconnected from the final code implementation. To bridge this gap, we employ an agent-driven reverse-engineering approach to synthesize high-fidelity Software Requirement Specifications (SRS). We first dispatch an LLM agent to analyze the ground-truth patches to draft precise functional requirements. This draft then undergoes a refinement phase to align acceptance criteria strictly with the verified Fail-to-Pass tests. Finally, environment-specific instructions (e.g., dependency updates) are appended by analyzing build configuration changes, ensuring a complete execution context.

Human-in-the-Loop Verification. Automated generation can yield logical inconsistencies and misalignment with edge cases. To mitigate this, expert annotators conduct a final review focused on task solvability. Annotators verify that the SRS provides all necessary information to solve the problem without leaking implementation details and that the acceptance criteria are unambiguous. Simultaneously, we validate the stability of the test suites to rule out flaky tests. This hybrid verification ensures that EvoClaw provides a fair assessment, distinguishing genuine agent errors from artifacts of ambiguous specifications (Appendix B.1).

###### 4.3. Benchmark Statistics

EvoClaw comprises 98 verified milestones across 7 diverse open-source repositories, spanning five programming languages (Go, Rust, Java, TypeScript, Python) with a total of 109 inter-milestone

dependencies. As shown in Figure 4a, the milestones are distributed across repositories with varying complexity, ranging from 9 to 23 milestones per repository. The dataset captures diverse real-world development patterns, including major architectural changes (e.g., multi-library support), feature-rich iterations (e.g., cloud-native enhancements), stability-focused releases (e.g., compatibility fixes), and large-scale refactoring (e.g., type system overhauls). This ensures EvoClaw evaluates agents across the full spectrum of software engineering tasks.

- Figure 4b illustrates the distribution of task complexity. The dataset exhibits substantial diversity in both specification length (SRS mean: 1,348 words) and implementation scope (gold patch LOC ranging from < 100 to > 1, 500). On average, each milestone modifies 27.4 files and involves 17.1 Fail-to-Pass tests for verification alongside 6,218 Pass-to-Pass tests for regression prevention. Detailed per-repository statistics are provided in Appendix B.3, and the full Milestone DAG visualizations for all repositories are shown in Appendix B.4.

#### 5. Results and Analysis

###### 5.1. Experimental Setup

Evaluation Settings To isolate how error accumulation across milestones affects agent performance, we evaluate methods under two different settings based on the Milestone DAG: (1) Continuous Task Evaluation, the standard EvoClaw setting where agents continuously evolve a codebase under streaming requirements. This setting introduces real-world challenges such as error accumulation and technical debt management. (2) Independent Task Evaluation, a stateless baseline (similar to SWE-bench (Jimenez et al., 2024)) in which each milestone is treated as an isolated task by providing agents with the canonical codebase snapshot, thereby decoupling performance from the cumulative effects of prior modifications.

Evaluation Metrics Evaluating agents in a continuous evolution setting requires metrics that capture two competing objectives: implementing new functionality and preserving existing behavior. Traditional benchmarks such as SWE-bench rely on binary success criteria (all tests pass or fail), which are too coarse-grained to capture the nuance of incremental progress and regression. Simple pass-rate metrics conflate these two objectives, failing to distinguish an agent that implements features but introduces regressions from one that avoids regressions but makes no progress.

To address this limitation, we decompose agent performance along two complementary dimensions:

- • Recall measures feature implementation completeness: the proportion of required functional changes successfully implemented by the agent.

Recall𝑚 =

𝑁fixed,𝑚 𝑁requried,𝑚

(1)

where 𝑁required,𝑚 denotes the total number of Fail-to-Pass (F2P) tests for milestone 𝑚 (tests that transition from failing at the start state to passing after the milestone’s gold patch is applied),

and 𝑁fixed,𝑚 is the count of such tests that the agent successfully fixes.

- • Precision measures modification reliability: the proportion of test status changes that are improvements rather than regressions, quantifying the safety of the agent’s edits.

𝑁fixed,𝑚 + 𝜖 𝑁fixed,𝑚 + 𝑁broken,𝑚 + 𝜖

Precision𝑚 =

(2)

where 𝑁broken,𝑚 is the number of Pass-to-Pass (P2P) tests (tests that pass at the start state and must remain passing after the agent’s changes) that regress (fail or error out) due to the agent’s changes. The term 𝜖 = 1 is a smoothing factor to handle cases where the agent makes no impact (i.e., when both fixed and broken counts are zero).

We then define the score for each milestone as the harmonic mean of Recall and Precision:

2 · Precision𝑚 · Recall𝑚 Precision𝑚 + Recall𝑚

Score𝑚 =

(3)

The final reported metric is the average Score across all milestones: Score = |𝑀1| 𝑚∈𝑀 Score𝑚. This ensures that neither dimension can be neglected: an agent that implements all features but introduces

severe regressions will score as poorly as one that preserves existing functionality but fails to implement any changes.

Consistent with prior work like SWE-bench, we also report the Milestone Resolve Rate, where a milestone is considered resolved only if the agent passes all associated F2P and P2P tests. We report the average resolve rate across all repositories. While Score quantifies partial progress, Resolve Rate assesses whether the task was fully resolved.

Evaluated Models and Agents We evaluate a diverse set of frontier LLMs across four agent frameworks. Specifically, we test (1) Claude Code with Claude Opus 4.5, Claude Sonnet 4.5, Claude Opus 4.6, and Claude Sonnet 4.6, (2) Codex CLI with GPT 5.2, GPT 5.2-Codex, and GPT 5.3-Codex (all set to xhigh reasoning effort), (3) Gemini CLI with Gemini 3 Pro, Gemini 3.1 Pro, and Gemini 3 Flash (all by default with 1M context), and (4) OpenHands (Wang et al., 2025) with Claude Opus 4.6, GPT 5.3-Codex, Gemini 3 Flash, Kimi K2.5, and MiniMax M2.5. Detailed framework versions, context management configurations, and the unified agent system prompt (Figure 18) are provided in Section B.2.

Agent Model Score★ (%)↑ Precision (%)↑ Recall (%)↑ Resolve (%)↑ Cost ($) Out Tok. (K) Time (h) Turns

- Claude Sonnet 4.5 15.16 18.88 28.50 5.49 27.02 243 2.06 770

- Claude Opus 4.5 25.85 28.04 40.80 6.28 71.10 309 2.35 999

Claude Sonnet 4.6 29.58 29.62 47.63 5.88 68.88 852 4.41 1538

- Claude Opus 4.6 36.29 37.84 56.32 11.57 88.22 578 2.73 1891

claude-code

- GPT 5.2-Codex 13.46 12.65 26.65 4.78 38.11 701 3.56 1259 GPT 5.2 23.30 20.89 45.76 8.18 56.90 814 5.35 1717

- GPT 5.3-Codex 28.88 27.81 49.70 9.58 25.01 392 8.23 1109

codex

Gemini 3 Pro 24.25 25.46 32.70 13.37 114.96 294 3.65 676 Gemini 3.1 Pro 23.32 21.59 37.22 10.95 62.97 207 3.89 1208 Gemini 3 Flash 24.22 24.31 42.12 8.37 12.10 255 5.02 1512

gemini-cli

MiniMax M2.5 17.60 22.48 34.60 1.30 3.57 598 11.88 1846 Kimi K2.5 20.20 26.29 31.37 8.49 4.32 279 6.85 800 Gemini 3 Flash 22.32 25.20 37.13 6.59 16.90 1516 7.16 2632 GPT 5.3-Codex 26.47 25.01 37.32 12.50 30.13† 553† 18.75 1047† Claude Opus 4.6 38.03 37.33 55.21 8.46 75.73† 524† 7.54 1970†

openhands

- Table 2 | Performance of coding agents on EvoClaw under continuous task evaluation. All metrics are per-evolution-range averages. Out Tok. (K): total generated tokens in thousands, including reasoning where applicable. ★Primary metric (shaded). Bold: best in column. †Token tracking unavailable for some repos; averaged over available data.

###### 5.2. Overall Performance

Table 2 presents results across 15 agent-model configurations. Claude Opus 4.6 achieves the highest Score (38.03% in Openhands and 36.29% in Claude Code), followed by Claude Sonnet

Sonnet 4.5 Opus 4.5

Sonnet 4.6 Opus 4.6

GPT-5.2 Codex GPT-5.2

GPT-5.3 Codex Gemini 3 Pro

Gemini 3.1 Pro Gemini 3 Flash

Continuous Task Evaluation

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Independent Task Evaluation

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

100%

80%

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Score(%)

60%

40%

20%

0%

scikit-learn Dubbo Navidrome go-zero Element Web Nushell ripgrep

- Figure 5 | Per-repository score comparison under two evaluation modes. High independent-task performance across all repositories confirms that milestones are individually solvable.

- 4.6 (29.58%) and GPT 5.3-Codex (28.88%). Across all models, the gap between Score (∼38% at best) and Resolve Rate (∼13%) is substantial: agents achieve partial progress on most milestones but rarely complete them fully. Moreover, the resolved milestones are predominantly early ones with few upstream dependencies, confirming that accumulated upstream errors increasingly hinder downstream task completion. Unless otherwise noted, subsequent analyses focus on configurations where each model is paired with its vendor-provided agent framework.

$0 $20 $40 $60 $80 $100 $120

Cost per Evolution Range ($)

10%

15%

20%

25%

30%

35%

40%

Score(%)

Sonnet 4.5

Opus 4.5

Sonnet 4.6

Opus 4.6

GPT-5.2-Codex

GPT-5.2

GPT-5.3-Codex

Gemini 3 Pro Gemini 3.1 Pro

Gemini 3 Flash

Figure 6 | Overall Score vs. cost trade-off across all repositories.

Across model families, generational improvements emerge: Claude 4.6 models significantly outperform their 4.5 predecessors, and GPT

- 5.3-Codex substantially improves over its predecessors. However, comparing GPT 5.2 and GPT 5.2-Codex reveals that Codex-specific optimization may be counterproductive for longhorizon development, where sustained codebase maintenance demands broader analytical capabilities beyond isolated task solving. The three Gemini models achieve comparable scores, with Gemini 3 Flash matching Gemini 3 Pro at one-ninth the cost. Gemini 3 Pro uses the fewest turns, possibly indicating insufficient exploration. Figure 6 visualizes the cost-score trade-off. Higher cost does not uniformly translate into higher performance: Gemini 3 Pro exceeds $100 per evolution range yet scores below Opus

- 4.6 ($88), and Sonnet 4.6 ($69) trails Opus 4.6 by 6.7 points despite a similar cost tier. On the Pareto frontier, Gemini 3 Flash ($12, 24.2%) and GPT 5.3-Codex ($25, 28.9%) offer the best cost-effectiveness, achieving competitive scores at a fraction of the cost of top-performing models. OpenHands trials exhibit notably longer execution times (e.g., 18.49 h for GPT 5.3-Codex) because its runner permits up to 3,000 iterations per milestone with automatic session resumption, allowing the agent to retry extensively when stuck. This additional compute does not consistently improve scores: Claude Code with Opus 4.6 achieves a comparable score in under 4 hours.

- Figure 5 compares per-repository performance under both evaluation modes. High independent-task performance across all repositories confirms that milestones are individually solvable, indicating that the difficulty stems from long-horizon error accumulation rather than inherent task complexity. This effect varies significantly by repository. scikit-learn exhibits the largest degradation: Claude Sonnet 4.6 achieves 93.2% independently but only 21.1% under continuous evaluation.

Sonnet 4.6

Opus 4.6

GPT-5.2

GPT-5.3 Codex

Gemini 3.1 Pro

Gemini 3 Flash

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

60%

60%

60% 58.5%

60%

50%

50%

50%

50%

48.2%

40.8%

40%

40%

40%

40%

Score(%)

33.6%

31.0%

31.0%

30%

30%

30%

30%

29.1%

26.3%

24.6%

24.2%

23.4%

20.5%

20%

20%

20%

20%

17.2%

15.9%

10%

10%

10%

10%

8.1%

7.0%

0%

0%

0%

0%

0-150 (n=150)

150-300 (n=132)

300-500 (n=114)

500+ (n=114)

200-1k (n=156)

1k-1.3k (n=120)

1.3k-1.8k (n=114)

1.8k+ (n=120)

1-3 (n=108)

4-6 (n=108)

7-9 (n=108)

10-14 (n=108)

15+ (n=78)

0 (n=216)

1-2 (n=162)

3+ (n=132)

DAG Layer

Lines of Code (LOC)

SRS Word Count

Milestone Execution Order

40%

40%

40%

40%

33.3%

30%

30%

30%

30%

Resolve(%)

20%

20%

20%

20%

19.4%

16.0%

15.3%

10.8%

10%

10%

10%

10%

9.6%

7.6%

4.6%

4.4%

2.8%

1.2% 0.0%

0.8%

0.0%

0.0% 0.0%

0%

0%

0%

0%

0-150 (n=150)

150-300 (n=132)

300-500 (n=114)

500+ (n=114)

200-1k (n=156)

1k-1.3k (n=120)

1.3k-1.8k (n=114)

1.8k+ (n=120)

1-3 (n=108)

4-6 (n=108)

7-9 (n=108)

10-14 (n=108)

15+ (n=78)

0 (n=216)

1-2 (n=162)

3+ (n=132)

DAG Layer

Lines of Code (LOC)

SRS Word Count

Milestone Execution Order

Figure 7 | Task complexity effects on Score (top) and Resolve Rate (bottom), binned by code size, specification length, execution order, and DAG layer. Dashed lines are per-bin averages across models.

Overall, these results highlight that EvoClaw poses a significant challenge to current frontier models, and reliable long-horizon continuous development remains an open problem.

- 5.3. Task Complexity and Topological Effects

- Figure 7 examines how milestone characteristics correlate with agent performance. Gold patch LOC, a traditional measure of task complexity, shows a clear monotonic relationship. Larger patches require more code changes and yield lower scores across all models. SRS (Software Requirements Specification) word count, however, exhibits a non-monotonic pattern, with a clear sweet spot for specifications of moderate length (around 500 to 1500 words). When specifications are concise, agents must autonomously locate relevant context from the repository, increasing exploration burden. When specifications are verbose, the sheer volume of requirements increases implementation workload. Milestones with moderate-length SRSs achieve the highest accuracy, suggesting that task difficulty depends not only on implementation effort but also on the cost of information acquisition.

Beyond these static factors, the continuous evaluation setting introduces structural complexity unique to EvoClaw. Both the milestone execution order and the DAG topological layer show statistically significant negative correlations with the score. Later milestones and deeper topological layers consistently yield lower performance. This reflects the compounding effect of upstream errors, as agents must build upon their own (potentially flawed) prior work. These topological factors are absent in independent evaluation and represent the distinctive challenge of long-horizon software evolution. The Resolve Rate (bottom row of Figure 7) makes this effect even starker: it drops drastically beyond the earliest milestones and the shallowest DAG layers, indicating that current agents can only fully resolve milestones that appear early in the sequence or have no upstream dependencies. Once prior errors accumulate, agents may still achieve partial progress (reflected in Score) but rarely produce a completely correct solution.

###### 5.4. Evolution Dynamics: Recall Scales while Precision Saturates

The declining performance at later milestones raises a natural question: does agent capability degrade over time, or does accumulated technical debt overwhelm otherwise competent agents?

To answer this, we model cumulative score trajectories using a saturation function 𝑦 = 𝑎(1 − 𝑒−𝑏𝑥),

Ideal (oracle)

5.3-Codex Continuous (init=0.68, ret=16%)

Opus 4.6 Continuous (init=0.84, ret=15%)

0.7

3.1 Pro Continuous (init=1.03, ret=1%)

Sonnet 4.6 Continuous (init=0.81, ret=9%)

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

Independent (dashed, same color)

Opus 4.5 Continuous (init=0.70, ret=8%)

| |
|---|

| |
|---|

| |
|---|

0.6

- GPT-5.2 Codex ceil=0.16

GPT-5.2 ceil=0.24

- GPT-5.3 Codex ceil=0.37

Ideal (y = x)

Independent (dashed, same color)

| |
|---|

Ideal (y = x)

| |
|---|

| |
|---|

| | | |
|---|---|---|

CumulativeScore

CumulativeScore

0.5

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

Gemini 3 Flash ceil=0.29

| |
|---|

| |
|---|

Gemini 3 Pro ceil=0.25 Gemini 3.1 Pro ceil=0.22

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

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|
| |
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Sonnet 4.5 ceil=0.19

- Opus 4.5 ceil=0.28

Sonnet 4.6 ceil=0.34

- Opus 4.6 ceil=0.44

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.3

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

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|
| |
| |

| |
|---|

| | | |
|---|---|---|

| |
|---|

| | | |
|---|---|---|

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

| |
|---|

| | | | |
|---|---|---|---|
| | | | |

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

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |
| | |
| | |

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

| | | |
|---|---|---|
| | | |

| |
|---|

| | | |
|---|---|---|

0.2

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

| |
|---|

| | |
|---|---|
| | |
| | |

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

| |
|---|

| |
|---|

| | |
|---|---|
| | |
| | |
| | |

| |
|---|

| |
|---|

| |
|---|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| |
|---|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

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

| |
|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

0.1

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

| |
|---|

| | | | |
|---|---|---|---|
| | | | |

| |
|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

observed predicted

| |
|---|

| |
|---|

0.0

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Progress

Progress

Progress

- Figure 8 | Evolution dynamics across models. (Left) Multi-window extrapolation of saturation curves fitted with 𝑦 = 𝑎(1 − 𝑒−𝑏𝑥), showing projected ceilings beyond the observed window. Legend annotations report init = 𝑎𝑏 (marginal efficiency at the onset of the sequence) and retain = 𝑒−𝑏 (fraction of efficiency preserved after each observation window). (Middle) Continuous vs. Independent comparison for GPT 5.3-Codex (better retain) and Gemini 3.1 Pro (better init). (Right) Continuous vs. Independent comparison for the Claude model family.

|Opus 4.5<br><br>Recall<br><br>Precision<br><br>Ideal|
|---|

|Sonnet 4.6|
|---|

|Opus 4.6|
|---|

| |
|---|
| |

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

| |
|---|

GPT-5.2 Codex

|| |
|---|
| |
<br><br>| |
|---|
| |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>GPT-5.2|
|---|

| |
|---|

| |
|---|

| |
|---|
| |

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

| |
|---|

| |
|---|

GPT-5.3 Codex

|Gemini 3 Pro|
|---|

|Gemini 3.1 Pro|
|---|

|Gemini 3 Flash|
|---|

Progress

CumulativeScore

- Figure 9 | Per-model cumulative Recall (solid) and Precision (dotted) over evolution progress. Stronger models achieve near-linear Recall growth, yet Precision saturates across all evaluated configurations.

where a small 𝑏 yields near-linear growth while a large 𝑏 produces rapid saturation toward the ceiling 𝑎. As shown in Figure 8 (left), all models under exhibit clear performance ceilings, and multi-window extrapolation (fitting the saturation model to progressively larger subsets of milestones and projecting forward) confirms that these ceilings persist beyond the observed window. Comparing continuous and independent evaluation (middle, right), independent scores grow near-linearly while continuous scores saturate, with the gap widening monotonically as evolution progresses.

We decompose the cumulative score into Recall (successful feature implementation) and Precision (preservation of existing functionality) to isolate the underlying mechanism. Figure 9 reveals a fundamental asymmetry: Recall continues to grow near-linearly across all models (especially frontier models), indicating that agents retain the ability to solve newly assigned tasks. Precision, however, saturates rapidly across all evaluated configurations. This means the performance ceiling is not caused by agents forgetting how to code, but by their inability to prevent regressions from accumulating. Stronger models achieve higher Precision plateaus, yet none avoid saturation entirely. This RecallPrecision divergence provides a mechanistic explanation for the snowball effect: as unresolved regressions compound, each new milestone operates on an increasingly degraded codebase, eventually

P0 Root Cause

P0 Induced

P1 Inherited

PX Missing

PX Build

PH Healed

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

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

0.200

0.175

0.150

EventProportion

MilestoneProgress

0.125

0.100

0.075

0.050

0.025

0.000

ripgrep dubbo element-web scikit-learn navidrome go-zero

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Milestone Progress

- Figure 10 | Propagation type analysis for Opus 4.6. Left: Selected error chain patterns across repositories, where each column is an error chain and each row a milestone. Right: Distribution of propagation event types across milestone progress bins (averaged over all repositories), showing how inherited failures (P1) and infrastructure effects (PX) increasingly dominate in later stages. overwhelming the agent’s capacity for productive development.

###### 5.5. Failure Analysis: Error Generation and Propagation

Understanding why agents fail in continuous evaluation is inherently difficult. A single early mistake can trigger cascading test failures across dozens of downstream milestones, making it challenging to disentangle root causes from their propagated consequences. To enable systematic analysis, we introduce the concept of error chains: for each test that transitions from passing to failing during the evolution, we trace its status across all subsequent milestones until it is either healed or the trial ends. This yields a per-test timeline that captures the full lifecycle of an error. We focus this analysis on the strongest configuration, Claude Opus 4.6, to characterize failure mechanisms at the frontier of current agent capabilities.

We decompose error chains along two orthogonal dimensions. The first, Propagation Type, captures how a fault affects downstream milestones. This is determined statistically from evaluation results: we track each test’s status across the milestone timeline and classify events as P0 Root Cause (the originating failure), P0 Induced (cross-chain contamination from unrelated changes), P1 Inherited (propagated through dependency), PX Missing (skipped execution), or PH Healed (successfully recovered). Figure 10 (right) shows that propagation events (P1, PX) increasingly dominate in later stages, confirming the compounding degradation observed in Section 5.3. The left panel visualizes representative error chain patterns across repositories, illustrating how a single root cause event can cascade through the entire remaining evolution.

The second dimension, Root Cause Type, captures why the initial fault originates. Since root cause attribution requires understanding the agent’s intent, we employ Claude Sonnet 4.6 as a reviewer agent that compares the task agent’s code changes against the ground-truth patch, the SRS specification, and evaluation artifacts. The reviewer classifies each error chain’s root cause into three categories: Logic Error (correct target, buggy implementation), Omission (missing a required component), or Extraneous (unnecessary

[Figure 68]

22% 0.9% 12% 17% 5%

Logic Error

RootCauseType

9% 1% 3% 8% 2%

Omission

8% 0.4% 4% 4% 4%

Extraneous

P0 Root Cause

P0 Induced

P1 Inherited

PX Missing

PH Healed

Propagation Type

Figure 11 | Root Cause Type × Propagation Type heatmap. Each cell shows the macro-averaged event proportion across all repositories.

Continuous / Indep. Turns Ratio 10-Bin Violin (All Agents)

8x

4x

Continuous/Indep.TurnsRatio

- 1x

- 2x

0.5x

0.25x

CC Opus 4.6

CC Sonnet 4.6 Codex GPT-5.3 Gemini Flash

25% 50% 75%

0.125x

0.0625x

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Progress

- Figure 12 | Continuous-to-Independent turns ratio across normalized execution progress (10 bins). Values above 1× indicate the agent expends more effort under continuous evaluation than on the same milestone independently. The ratio remains near or below 1× for most of the sequence but rises sharply in the final bin. This reflects increased rework and error-recovery effort as accumulated technical debt compounds.

|M1|M2|M3|M4|M5|M6| |M7|M8| |M9|M10| |M11|M12| |M13|M14|M15| |M16|M17|M18| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | |

0%

50%

100%

Exploration%

Exploration Ratio

| | | | | | | |C F T|ached<br><br>resh input<br><br>ool result|eviction|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |Compaction| |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0 25 50 75 100 125 150 175

Active time (minutes)

0K

50K

100K

150K

ContextTokens

Context Wave

- Figure 13 | Exploration ratio and context wave for Claude Code (powered by Opus 4.6) on element-web. Top plot shows milestone progression (M1 to M18). Middle plot shows per-minute exploration ratio (read/search vs. write/execute). Bottom plot shows total context token usage over active time, with compaction and eviction events marked.

modifications that break existing functionality). Figure 11 presents the joint distribution of Root Cause Type × Propagation Type. Logic Error is the dominant root cause (∼57% of all error chain events), with its chains exhibiting both the highest inherited propagation (P1, 12%) and the highest proportion of missing test execution (PX Missing, 17%), indicating that buggy implementations frequently prevent downstream tests from running at all.

###### 5.6. Agent Behavior: The Struggle Against Accumulating Complexity

Beyond aggregate scores, we examine how agents allocate effort and manage state when facing accumulating technical debt during long-horizon iterations. By instrumenting tool calls, context usage, and interaction turns, we reveal distinct behavioral patterns.

Effort Fluctuation and Extremes. As shown in Figure 12, all evaluated agents exhibit a shared

Exploration Count Score (%)

40

11,686 (71%)

Claude Code Codex Gemini CLI

12000

35

9,270 (72%)

ExplorationCount

10000

7,667 (64%)

30

7,454 (70%)6,281

Score(%)

8000

(72%) 5,854

25

(55%)5,075 (59%)

4,915 (65%)

6000

3,484 (57%)

20

4000

2,584 (53%)

15

2000

0

10

Opus4.6Sonnet4.6Opus4.5Sonnet4.5 GPT-5.3-codexGPT-5.2GPT-5.2-codexGemini3FlashGemini3.1ProGemini3Pro

- Figure 14 | Exploration tool-call count aggregated across all repositories, grouped by agent framework and sorted by count within each cluster. Diamond markers show average F1 score. Higher-performing agents consistently devote greater effort to exploration (e.g., reading and searching).

trend in their effort allocation (measured by the continuous-to-independent turns ratio). In the initial phase (progress ∼0.1), continuous effort is slightly higher than independent effort, as agents must conduct large-scale exploration to build a mental model of the unfamiliar repository. During the middle phase (progress 0.1–0.5), the ratio drops below 1× (with the median falling to ∼0.83×): agents successfully reuse their established context, bypassing the redundant exploration required in independent evaluation. However, in the late stage (progress 0.6–0.9), effort rises significantly as accumulating errors demand extensive debugging. Finally, near completion (progress ∼1.0), agent behavior diverges sharply. Some agents resort to frantic thrashing, while others prematurely give up. Notably, GPT 5.3-Codex demonstrates the most stable effort profile, maintaining consistent variance throughout the project lifecycle.

Context Stability and Exploration Patterns. To sustain this fluctuating effort, agents must effectively manage their context. Figure 13 illustrates this using Claude Code with Opus 4.6 as a representative example. The context window shows stable, controllable wave patterns, demonstrating that modern agent frameworks paired with frontier models can effectively support long-horizon programming without catastrophic context overflow. The framework employs two compression strategies: partial compression (evicting specific tool results) and heavy compaction (summarizing extensive histories). Crucially, agent exploration behavior (reading and searching) tightly couples with this state management. Exploration surges at the beginning of each new milestone and immediately following major context compaction events, as the agent works to rebuild its mental model.

The Impact of Exploration. This exploration behavior directly dictates downstream success. Figure 14 demonstrates that within their respective agent frameworks, models from the same family exhibit a consistent pattern: higher exploration counts correlate with better performance. For instance, Claude Opus 4.6 and Claude Sonnet 4.6 hold a distinct advantage because they aggressively dispatch subagents to analyze the codebase, executing over 7,000 exploration commands. Conversely, models like Gemini 3 Pro allocate too little effort to reading, indicating that many current models still lack proactive exploration for long-horizon tasks.

Verification vs. Blind Thrashing. Alongside exploration, we analyze verification behavior (test execution). Figure 15 shows that average verification effort generally follows an inverted-U shape, increasing as the codebase grows more complex before declining near the end. However, this masks two problematic extremes: Gemini 3.1 Pro verifies excessively, while GPT 5.2-Codex rarely verifies at all, and both achieve lower scores. Figure 16 further isolates this dynamic by mapping the score landscape against edit thrashing and verification frequency. A clear sweet spot emerges for

25

Opus 4.6 (Score=0.36)

GPT-5.3 Codex (Score=0.29) Gemini 3 Flash (Score=0.24)

Gemini 3.1 Pro (Score=0.23) GPT-5.2 Codex (Score=0.13)

| |
|---|

| |
|---|

Sonnet 4.6 (Score=0.30)

| |
|---|

| |
|---|

20

VerifyCount

15

10

5

0

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Progress

- Figure 15 | Average verification tool-call count per milestone progress bin (10 bins), for the six strongest agent configurations. Verification effort generally increases with progress. It peaks around 70% to 80% completion before declining in the final bin.

0.0 0.2 0.4 0.6 0.8 1.0

Repeat Edit Ratio pctrank

0.0

0.2

0.4

0.6

0.8

1.0

Verifypctrank

Score Landscape in Edit-Thrashing × Verification Space

Careful Crafting low thrash, high verify

Disciplined Struggle high thrash, high verify

Confident Execution low thrash, low verify

Blind Thrashing high thrash, low verify

Sweet Spot moderate verification

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.21

0.23

0.25

0.27

0.29

0.31

0.33

0.35

0.37

MeanScore

- Figure 16 | Score landscape in the edit-thrashing (repeat edit ratio) vs. verification (test execution frequency) space, aggregated across all repositories. A sweet spot of moderate thrashing and moderate verification yields the highest scores. The blind thrashing quadrant (high thrash, low verify) produces the worst outcomes.

moderate, disciplined verification. In contrast, the worst outcomes concentrate in the high-thrash and low-verify quadrant—a blind thrashing trap where agents repeatedly modify the same files without executing tests to guide them, effectively accelerating the snowball effect.

###### 5.7. DeepCommit vs Human-Annotated Milestone DAG

We conducted a case study (Appendix C.3) that compares the human-annotated and DeepCommit Milestone DAGs for the scikit-learn v1.5.2–v1.6.0 release interval. The Human DAG organizes milestones by semantic release intent, whereas DeepCommit derives groups from dependency topology in the commit graph. As a result, DeepCommit covers a smaller but tightly connected subset of commits, recovers human-like boundaries when technical structure is clear (e.g., documentation), but tends to fragment cross-module, intent-defined milestones into topological phases. Overall, this case study shows that the Human DAG captures semantically coherent and process-aware milestone structure,

whereas DeepCommit more strongly reflects dependency topology and phase-wise code organization.

#### 6. Conclusion

We introduced DeepCommit, a pipeline that distills verifiable software evolution into coherent Milestone DAGs from noisy, fine-grained git histories, and EvoClaw, a benchmark for evaluating LLM agents under continuous, dependency-driven development. Our results reveal a fundamental gap between independent task-solving and continuous evolution: frontier models achieve over 80% on isolated milestones but drop below 38% in continuous settings. This degradation stems from a critical inability to maintain code integrity: while agents can implement new features, they fail to prevent regressions, causing a snowball effect of accumulating technical debt. Even the strongest agents resolve only ∼13% of milestones in full evolutionary sequences, establishing sustained, maintainable repository evolution as a central open challenge for autonomous software agents.

#### Limitations

EvoClaw and the DeepCommit pipeline have several limitations that bound the conclusions to be drawn and the settings to which they currently apply.

Test-Suite Dependency. Our construction relies on repositories with well-maintained, executable test suites that provide reliable F2P and P2P signals. Projects that lack rich test coverage, or whose tests depend on inaccessible external services, cannot currently be incorporated into the benchmark.

Filtering Bias. The pipeline retains only commits that touch source code with non-trivial intercommit dependencies, dropping documentation-only commits and commits without resolvable structural ties. This filtering improves DAG quality and evaluation tractability, but it may bias the resulting benchmark toward dependency-rich evolution and underrepresent independent maintenance work.

Data Contamination Risk. The repositories used in EvoClaw are high-impact open-source projects whose commit histories may have appeared in the pretraining corpora of frontier models. The substantial performance gaps we observe among frontier models suggest contamination has limited impact on relative ranking, but we cannot fully rule out memorization on individual milestones. Continuously refreshing the benchmark with newly merged commits, or applying DeepCommit to private repositories, would mitigate this risk.

Human-in-the-Loop Reliance. Two stages of DeepCommit still require human-expert oversight: (i) the MainAgent’s scheduling decisions during runtime environment resolution, where humans guide the trade-off between testbed quality and resolution cost, and (ii) the SRS verification stage, where human annotators run the three-step refinement loop described in Appendix B.1. Fully automating these stages remains an open engineering problem.

Scale Limit. The current pipeline targets release ranges whose source-code gold patch is under roughly 30k LoC. Larger ranges produce Milestone DAGs that exceed the agent’s resolution budget and frequently fail the testbed-construction gates. Scaling DeepCommit to longer histories will require further improvements to both DAG construction and runtime resolution.

#### Acknowledgements

The authors of this work are supported in part by the U.S. Army Research Office under Grant W911NF242-0194, the National Science Foundation under Grants OAC-2505107 and CCF-2426161, a Google Research Credit Award, and UCR Senate Awards. We thank OpenHands for providing the API credits that support most of the evaluations in this study. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of these organizations.

#### Impact Statement

This paper presents a benchmark for evaluating LLM agents in continuous software evolution, a prerequisite for deploying autonomous agents in real-world production environments. Beyond code generation, we emphasize the ability to sustainably evolve systems over time, which is essential for long-running agent runtimes to iteratively customize software for diverse user needs. This capability unlocks a new productivity paradigm: agents serving as adaptive interfaces that bridge human intent and complex digital systems. Although highly capable coding agents may impact the software development workforce, we expect them to lower the barrier to entry for software creation, empowering a broader range of users to leverage the power of code for problem-solving

#### References

Anthropic. Claude code: Best practices for agentic coding. https://www.anthropic.com/ engineering/claude-code-best-practices, 2025.

- I. Badertdinov, A. Golubev, M. Nekrashevich, A. Shevtsov, S. Karasik, A. Andriushchenko, M. Trofimova, D. Litvintseva, and B. Yangel. SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https: //openreview.net/forum?id=nMpJoVmRy1.
- J. Chen, X. Xu, H. Wei, C. Chen, and B. Zhao. Swe-ci: Evaluating agent capabilities in maintaining codebases via continuous integration, 2026. URL https://arxiv.org/abs/2603.03823.

- M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda,
- N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin,

- B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Cursor. Cursor: The AI-first code editor. https://cursor.sh, 2024.

- X. Deng, J. Da, E. Pan, Y. Y. He, C. Ide, K. Garg, N. Lauffer, A. Park, N. Pasari, C. Rane, K. Sampath,

- M. Krishnan, S. Kundurthy, S. Hendryx, Z. Wang, V. Bharadwaj, J. Holm, R. Aluri, C. B. C. Zhang,
- N. Jacobson, B. Liu, and B. Kenstler. Swe-bench pro: Can ai agents solve long-horizon software engineering tasks?, 2025. URL https://arxiv.org/abs/2509.16941.

J. Ding, S. Long, C. Pu, H. Zhou, H. Gao, X. Gao, C. He, Y. Hou, F. Hu, Z. Li, W. Shi, Z. Wang, D. Zan, C. Zhang, X. Zhang, Q. Chen, X. Cheng, B. Deng, Q. Gu, K. Hua, J. Lin, P. Liu, M. Li, X. Pan, Z. Peng, Y. Qin, Y. Shan, Z. Tan, W. Xie, Z. Wang, Y. Yuan, J. Zhang, E. Zhao, Y. Zhao, H. Zhu, L. Zhu, C. Zou, M. Ding, J. Jiao, J. Liu, M. Liu, Q. Liu, C. Tao, J. Yang, T. Yang, Z. Zhang, X. Chen, W. Huang, and G. Zhang. Nl2repo-bench: Towards long-horizon repository generation evaluation of coding agents, 2026. URL https://arxiv.org/abs/2512.12730.

- Y. Du, Y. Cai, Y. Zhou, C. Wang, Y. Qian, X. Pang, Q. Liu, Y. Hu, and S. Chen. Swe-dev: Evaluating and training autonomous feature-driven software development, 2026. URL https://arxiv.org/ abs/2505.16975.

- M. Fan, W. Zhang, H. Zhao, G. Liang, and Z. Jin. Detect hidden dependency to untangle commits. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, pages 179–190, 2024.

D. Fu, S. Wu, Y. Wu, Z. Peng, Y. Huang, J. Sun, J. Zeng, M. Jiang, L. Zhang, Y. Li, J. Hu, L. Liu,

- J. Hou, and P. Liu. davinci-env: Open swe environment synthesis at scale, 2026. URL https: //arxiv.org/abs/2603.13023.

GitHub. Github copilot: Meet the new coding agent. https://github.blog/news-insights/ product-news/github-copilot-meet-the-new-coding-agent/, 2025.

Google. Antigravity: An agentic development platform. https://antigravity.google/, 2025a. Google. Gemini CLI: An open-source AI agent for the terminal. https://github.com/

google-gemini/gemini-cli, 2025b.

X. He, Q. Liu, M. Du, L. Yan, Z. Fan, Y. Huang, Z. Yuan, and Z. Ma. Swe-perf: Can language models optimize code performance on real-world repositories?, 2025. URL https://arxiv.org/abs/ 2507.12415.

- K. Herzig, S. Just, and A. Zeller. The impact of tangled code changes on defect prediction models. Empirical Software Engineering, 21(2):303–336, 2016.

- N. Jain, J. Singh, M. Shetty, T. Zhang, L. Zheng, K. Sen, and I. Stoica. R2e-gym: Procedural environment generation and hybrid verifiers for scaling open-weights SWE agents. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=7evvwwdo3z.

- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.

- P. Laban, H. Hayashi, Y. Zhou, and J. Neville. LLMs get lost in multi-turn conversation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=VKGTGGcwl6.

C. Labs. Introducing Devin, the first AI software engineer. https://cognition.ai/blog/ introducing-devin/, 2024.

J. J. Ma, M. Hashemi, A. Yazdanbakhsh, K. Swersky, O. Press, E. Li, V. J. Reddi, and P. Ranganathan. Swe-fficiency: Can language models optimize real-world repositories on real workloads?, 2025. URL https://arxiv.org/abs/2511.06090.

M. A. Merrill, A. G. Shaw, N. Carlini, B. Li, H. Raj, I. Bercovich, L. Shi, J. Y. Shin, T. Walshe, E. K. Buchanan, J. Shen, G. Ye, H. Lin, J. Poulos, M. Wang, M. Nezhurina, D. Lu, O. M. Mastromichalakis, Z. Xu, Z. Chen, Y. Liu, R. Zhang, L. L. Chen, A. Kashyap, J.-L. Uslu, J. Li, J. Wu, M. Yan, S. Bian, V. Sharma, K. Sun, S. Dillmann, A. Anand, A. Lanpouthakoun, B. Koopah, C. Hu, E. K. Guha, G. H. S. Dreiman, J. Zhu, K. Krauth, L. Zhong, N. Muennighoff, R. K. Amanfu, S. Tan, S. Pimpalgaonkar, T. Aggarwal, X. Lin, X. Lan, X. Zhao, Y. Liang, Y. Wang, Z. Wang, C. Zhou, D. Heineman, H. Liu, H. Trivedi, J. Yang, J. Lin, M. Shetty, M. Yang, N. Omi, N. Raoof, S. Li, T. Y. Zhuo, W. Lin, Y. Dai, Y. Wang, W. Chai, S. Zhou, D. Wahdany, Z. She, J. Hu, Z. Dong, Y. Zhu, S. Cui,

- A. Saiyed, A. Kolbeinsson, C. M. Rytting, R. Marten, Y. Wang, J. Jitsev, A. Dimakis, A. Konwinski, and L. Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=a7Qa4CcHak.

Nous Research. Hermes Agent: The agent that grows with you. https://hermes-agent.

nousresearch.com/, 2026. OpenAI. Introducing codex. https://openai.com/index/introducing-codex/, 2025. OpenClaw. OpenClaw: Your own personal AI assistant. https://github.com/openclaw/

openclaw, 2026.

G. Orlanski, D. Roy, A. Yun, C. Shin, A. Gu, A. Ge, D. Adila, N. Roberts, F. Sala, and A. Albarghouthi. SlopCodeBench: Benchmarking how coding agents degrade over long-horizon iterative tasks, 2026. URL https://arxiv.org/abs/2603.24755.

- A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Re, and A. Mirhoseini. Kernelbench: Can LLMs write efficient GPU kernels? In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=yeoN1iQT1x.

J. Pan, X. Wang, G. Neubig, N. Jaitly, H. Ji, A. Suhr, and Y. Zhang. Training software engineering agents and verifiers with SWE-gym. In Forty-second International Conference on Machine Learning,

##### 2025. URL https://openreview.net/forum?id=Cq1BNvHx74.

- O. Press, B. Amos, H. Zhao, Y. Wu, S. Ainsworth, D. Krupke, P. Kidger, T. Sajed, B. Stellato, J. Park, N. Bosch, E. Meril, A. Steppi, A. Zharmagambetov, F. Zhang, D. Pérez-Piñeiro, A. Mercurio, N. Zhan, T. Abramovich, K. Lieret, H. Zhang, S. Huang, M. Bethge, and O. Press. Algotune: Can language models speed up general-purpose numerical programs? In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https: //openreview.net/forum?id=dF1tD9hjvn.

M. Shetty, N. Jain, J. Liu, V. Kethanaboyina, K. Sen, and I. Stoica. GSO: Challenging software optimization tasks for evaluating SWE-agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview. net/forum?id=I5qDL315bQ.

M. V. Thai, T. Le, D. N. Manh, H. P. Nhat, and N. D. Bui. Swe-evo: Benchmarking coding agents in

long-horizon software evolution scenarios. arXiv preprint arXiv:2512.18470, 2025. TRAE. Trae: The real AI engineer. https://www.trae.ai/, 2025.

X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, H. H. Tran, F. Li, R. Ma, M. Zheng, B. Qian, Y. Shao, N. Muennighoff, Y. Zhang, B. Hui, J. Lin, R. Brennan, H. Peng, H. Ji, and G. Neubig. Openhands: An open platform for AI software developers as

generalist agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=OJd3ayDDoF.

C. S. Xia, Y. Deng, S. Dunn, and L. Zhang. Demystifying llm-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824, 2025.

W. XU, J. Xiong, C. Zhao, Q. Chen, H. Wang, H. Shen, Z. Wan, J. Dai, T. Wu, H. Xiao, C. Tao, Z. Mao, Y. Sheng, Z. Guo, H. Yang, B. Yu, L. Kong, Q. Gu, and N. Wong. SWINGARENA: Adversarial programming arena for long-context github issue solving. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=YuxgSGFaqb.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=mXpq6ut8J3.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press. mini-SWE-agent: A minimal ai software engineering agent. https://github.com/SWE-agent/mini-swe-agent, 2025.

J. Yang, K. Lieret, J. Ma, P. Thakkar, D. Pedchenko, S. Sootla, E. McMilin, P. Yin, R. Hou, G. Synnaeve, D. Yang, and O. Press. Programbench: Can language models rebuild programs from scratch?, 2026. URL https://arxiv.org/abs/2605.03546.

##### S. Yao. The second half. https://ysymyth.github.io/The-Second-Half/, 2025. Accessed:

2025.

- D. Zan, Z. Huang, W. Liu, H. Chen, S. Xin, L. Zhang, Q. Liu, A. Li, L. Chen, X. Zhong, S. Liu, Y. Xiao, L. Chen, Y. Zhang, J. Su, T. Liu, R. LONG, M. Ding, and liang xiang. Multi-SWE-bench: A multilingual benchmark for issue resolving. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview. net/forum?id=MhBZzkz4h9.

- L. Zhang, S. He, C. Zhang, Y. Kang, B. Li, C. Xie, J. Wang, M. Wang, Y. Huang, S. Fu, E. Nallipogu, Q. Lin, Y. Dang, S. Rajmohan, and D. Zhang. SWE-bench goes live! In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview.net/forum?id=OGWkr7gXka.

W. Zhao, N. Jiang, C. Lee, J. T. Chiu, C. Cardie, M. Gallé, and A. M. Rush. Commit0: Library generation from scratch. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=MMwaQEVsAg.

- Q. Zhou, J. Zhang, H. Wang, R. Hao, J. Wang, M. Han, Y. Yang, S. Wu, F. Pan, L. Fan, D. Tu, and Z. Zhang. Featurebench: Benchmarking agentic coding for complex feature development. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=41xrZ3uGuI.

#### Table of Contents

###### A DeepCommit Pipeline Implementation Details 24

- A.1 Commit History Preprocessing Details . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- A.2 Milestone DAG Construction Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.3 Runtime Environment Resolution Details . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.4 Testbed Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

###### B EvoClaw Benchmark Details 27

- B.1 Software Requirement Specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.2 Evaluation Configuration Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- B.3 Detailed Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- B.4 Milestone DAG Visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

###### C Extended Experimental Analysis and Discussion 33

- C.1 Agent Code Quality Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- C.2 Cumulative Error Analysis on element-web . . . . . . . . . . . . . . . . . . . . . . . . 38
- C.3 Case Study: Human vs. DeepCommit Milestone DAG Construction . . . . . . . . . . . 38

#### A. DeepCommit Pipeline Implementation Details

This appendix documents the per-stage implementation of the DeepCommit pipeline introduced in Section 3, covering commit history preprocessing, Milestone DAG construction, runtime environment resolution, and testbed validation.

###### A.1. Commit History Preprocessing Details

In open-source projects, version tags are applied in various ways, posing challenges for determining the mainline commit range. Commonly, tags are placed directly on the main branch, making the commits between start and end tags the target range. However, many projects use a release branch model: a release branch is created from main for stabilization before release, and the tag is ultimately placed on the release branch rather than main. In such cases, directly comparing two tags would include release branch commits while missing parallel development on main.

To address this, we employ a branch-out/first-parent strategy to recover the mainline range. First, we use git merge-base to find each tag’s branch-out point from the main branch. Then, we collect all commits between these two branch-out points along the main branch’s first-parent chain. First-parent traversal ensures we only follow direct commits on main, ignoring internal commits from merged feature branches, thus obtaining a clean mainline evolution sequence.

Data Collection. For each entity, we collect metadata and content: commit change details and parent relationships; PR descriptions, constituent commits, linked Issues, and code review discussions;

Issue descriptions and discussions; Release version tags and notes.

Per-Repo Configuration. Source-file filtering is driven by a small per-repository configuration with four fields: repo_src_dirs (source-module directories to retain), test_dirs (test-file glob patterns), exclude (build artifacts and generated files to strip from within source directories), and main_branch (the branch used for range extraction). An LLM agent bootstraps this configuration by detecting the repository language from build markers (e.g., pyproject.toml, pom.xml, go.mod) and proposing source and test patterns from the repository layout and tag-range history. A human curator then verifies coverage, confirming which directories constitute product-logic source code. The rules below apply on top of this configuration.

Filtering Rules. Raw data contains numerous changes irrelevant to core functionality. We apply multi-layer filtering:

- 1. Source directory whitelist: Retain only changes under designated source directories (e.g., src/, lib/), excluding documentation, configuration files, and CI/CD scripts.
- 2. Test file exclusion: Exclude test code via filename patterns (e.g., *_test.go, test_*.py) and directory patterns (e.g., tests/, __tests__/).
- 3. Empty commit removal: Commits containing no valid source changes after filtering are removed.
- 4. Reference consistency: When commits are removed, the system checks PR and Issue references, removing orphaned PRs and Issues no longer referenced by any commit.

###### A.2. Milestone DAG Construction Details

Topological Features for Seed Discovery. We utilize the following Commit DAG topological metrics to identify milestone seeds: (1) Out-degree: commits depended upon by multiple subsequent commits typically introduce foundational functionality; (2) Topological level: commits at higher levels represent later architectural changes; (3) Descendant count: commits with many descendants have broad impact and are often key nodes. The Agent typically identifies fewer than 20 seed commits.

Expansion Rules for Milestone Consolidation. Sub-agents expand seed boundaries using the following rules: (1) File modification overlap: commits modifying the same files likely belong to the same feature; (2) Temporal proximity: commits close in time are more likely part of the same development task; (3) Semantic association: commit messages containing similar keywords or referencing the same Issue/PR. The system first pre-groups seeds based on commit subgraph overlap.

Heuristics for Dependency Inference. For potential dependencies not covered by structural analysis, the system generates and ranks candidate edges using heuristic rules: (1) File overlap ratio: the proportion of shared files modified by two milestones; (2) Symbol references: whether downstream milestones call functions/classes defined in upstream ones; (3) Temporal ordering: whether the upstream milestone’s median commit time precedes the downstream’s; (4) Author overlap: milestones by the same author are more likely to have dependencies. After Agent verification, dependencies are classified into two types: strong dependencies indicate that missing upstream milestones will cause downstream build or test failures; weak dependencies indicate functionality can degrade gracefully without affecting basic operation.

Balancing Strategy for Milestone Decomposition. The system computes the source lines of code (LOC) for each milestone and uses the coefficient of variation (CV = std/mean) to measure granularity uniformity, targeting CV < 1.0. On EvoClaw, the final partition achieves CV = 0.96. Abnormally sized milestones are flagged in two cases: too large, when LOC exceeds mean +𝜎, and too small, when LOC falls below mean −𝜎 and under 100 lines.

PR-Commit Splitting. An oversized milestone is sometimes dominated by a single squashed PRcommit, which cannot be rebalanced by regrouping commits. In this case, an LLM agent re-segments the squashed diff into smaller commits along feature boundaries, optionally emitting one integrationtest commit. Dependencies among the resulting pieces, together with edges to downstream milestones, are recomputed via line-level git blame, so each split unit inherits only the edges it truly depends on, and the units are promoted to first-class milestones in the DAG. When no clear feature boundary exists, the agent falls back to chronological grouping, and test-only commits are never emitted as standalone milestones.

###### A.3. Runtime Environment Resolution Details

Motivation for Testbed Reconstruction. The original repository’s commit history reflects the actual chronological order of development, not the logical dependencies between functional modules. For example, a developer may implement feature A before feature B, even though B logically depends on

- A. Using the original history directly would cause milestone code states to be inconsistent with the DAG’s dependency structure. Therefore, we re-cherry-pick commits according to the DAG’s topological order, generating a new history organized by functional logic.

Flaky Test Filtering. Flaky tests produce inconsistent results on the same code, severely affecting evaluation reliability. We identify such tests by running the test suite three times. If a test produces different results across runs (e.g., sometimes passing, sometimes failing), it is marked as flaky and excluded. Common causes of flakiness include: dependencies on external services, time-sensitive assertions, race conditions in concurrent code, and random data generation.

###### A.4. Testbed Validation

To ensure the reliability and reproducibility of our evaluation, we rigorously validate each milestone testbed across three core dimensions: milestone graph validity, runtime executability, and evaluation reliability.

###### A.4.1. Milestone Graph Validity

We first verify the structural properties of the reconstructed history by construction. Commit Completeness: The commit set across all milestones is cross-checked against git log v_start..v_end, confirming 100% coverage with no gaps. Dependency Consistency: If commit 𝑎 in milestone 𝐴 depends on commit 𝑏 in milestone 𝐵 according to the commit-level DAG, then 𝐴 must depend on 𝐵 in the Milestone DAG. This constraint is verified for all inter-milestone commit pairs. DAG Correctness: DFS cycle detection and Kahn’s topological sort confirm valid acyclic DAGs for all 7 repos.

###### A.4.2. Runtime Executability

- A reliable testbed must guarantee that errors stem from agent code, not infrastructure. Following the Milestone DAG, we reconstruct repository states by sequentially cherry-picking milestone commits

in topological order. Testbed Compilability: We verify that docker build completes without error and that the test framework’s collection command (e.g., pytest –collect-only) succeeds in both START and END states for all repos in EvoClaw. No Environment-Induced Errors: We strictly monitor execution logs to ensure tests classified as error (reflecting environment/setup failures rather than assertion failures) remain negligible (≤0.10% across all repos).

###### A.4.3. Evaluation Reliability

Finally, we assess the runtime behavior of the test suites to ensure they provide accurate evaluation signals. Test Collection and Size Stability: We statically extract test names from commit diffs and match them against runtime-collected node IDs, achieving an overall collection rate of 87.1% (3,563 of 4,090 tests). Furthermore, we compare test counts across milestones, confirming that the delta matches expected N2P (None-to-Pass, newly added tests) additions and removals. Test Consistency: To prevent false penalties, we ensure a negligible Pass-to-Fail (P2F) rate (≤0.026%), indicating no unintended regressions are introduced by milestone transitions. Additionally, each milestone is run three times to filter out flaky tests (yielding only 0–16 flaky tests per repo). Coverage: Every milestone with source changes must contain at least one F2P or N2P test. Milestones lacking test signals are labeled as maintenance tasks and excluded from the evaluation path, with their structural dependencies transitively bypassed to maintain DAG connectivity.

#### B. EvoClaw Benchmark Details

This appendix collects the construction artifacts of EvoClaw: the Software Requirement Specifications, the unified evaluation configuration shared across all agents, per-repository dataset statistics, and the full Milestone DAG visualizations.

###### B.1. Software Requirement Specifications

This appendix details the design, structural format, verification protocol, and refinement statistics of the SRS that serves as the task instruction for every milestone in EvoClaw.

Design Principles. Each milestone’s SRS is written as a minimally solvable behavior- and contractlevel specification rather than a patch-reconstruction hint. The SRS deliberately avoids line-level edits, patch-like instructions, and identifiers such as file paths or function names unless those identifiers are themselves part of the public contract or are essential to making the requirement unambiguous. Three deliberately competing principles govern this trade-off. First, specify what is required, not how to implement it: state motivation, behavior, and acceptance criteria without prescribing the implementation. Second, align with test intent, not test artifacts: cover the evaluated behaviors without revealing test names, concrete inputs, or asserted values, exposing only public interface constraints when strictly necessary. Third, ensure solvability: include the essential formats, function signatures, edge cases, and non-inferable conventions needed to derive a correct implementation.

SRS Structure. Each milestone’s SRS is organized as a sequence of Feature Requirements, each containing three fields: a Problem statement describing the symptoms and context that motivate the change, a Requirements block stating the intended functionality and constraints in an implementationagnostic way, and an Acceptance clause listing observable pass criteria without referencing test artifacts. Together, the three fields bound the implementation space while leaving the agent free to choose any compliant solution. Figure 17 shows the Overview and two illustrative Feature Requirements from

the SRS for milestone M001 of zeromicro/go-zero, with the remaining requirements omitted for space. Full SRS instances for all milestones are released with the benchmark dataset.

Human-in-the-Loop Verification. The reverse-engineered SRS is refined through an agent-assisted, human-led verification loop. Human annotators iterate over three checks. In static verification, annotators read each SRS against the three design principles and revise any obvious violations until the document is internally consistent. In independent dry-run verification, annotators dispatch a strong coding agent to attempt each milestone in isolation using only the refined SRS, then inspect the failed Fail-to-Pass tests and add the minimal non-inferable conventions required for solvability. After several iterations, state-of-the-art models typically achieve 80% to 90% on independent runs, in line with the per-repository independent-task scores reported in Section 5.2. This indicates that residual unsolvability is small. In continuous dry-run verification, annotators replay each repository’s DAG end-to-end to surface missing cross-milestone dependencies and SRS gaps that only emerge under sustained execution. A final end-to-end review by an independent senior expert closes the loop.

Refinement Outcomes. The verification loop converges quickly. Across EvoClaw, 71% of milestones underwent at least one revision, distributed across four categories: removing implementation leakage (39%), removing test leakage (31%), tightening inaccurate or underspecified requirements (25%), and adding missing functional requirements (5%). The convergence is also visible in evaluation impact. The first refinement round shifts per-milestone scores by approximately 5%, whereas the second round shifts them by at most 2%, indicating that residual SRS noise has limited effect on the relative ranking of agents.

- B.2. Evaluation Configuration Details We evaluate multiple model-agent configurations across four agent frameworks:

- • Claude Code (v2.1.50)1 is Anthropic’s terminal-based agent that operates with full shell access and autonomous multi-step planning. We pair it with Claude Opus 4.5, Claude Sonnet 4.5, Claude Opus 4.6, and Claude Sonnet 4.6, all supporting a 200K-token context window. Claude Code triggers automatic context compaction at approximately 80% of the context window (∼160K tokens), summarizing prior conversation history via server-side compression while preserving key context.
- • Codex CLI (v0.105.0)2 is OpenAI’s open-source CLI agent designed for code generation and editing tasks. We pair it with GPT 5.2, GPT 5.2-Codex, and GPT 5.3-Codex, all configured with xhigh reasoning effort and operating with a 272K-token context window. Codex CLI triggers auto-compaction at approximately 90% of context capacity (∼245K tokens). The underlying model is natively trained for multi-context-window operation, automatically summarizing the session and creating a fresh context window while preserving recent messages alongside the summary.
- • Gemini CLI (v0.29.5)3 is Google’s command-line agent for interacting with Gemini models in development workflows. We pair it with Gemini 3 Pro, Gemini 3.1 Pro, and Gemini 3 Flash, all supporting a 1M-token context window. Gemini CLI triggers context compression at approximately 50% of the context window. The compression mechanism invokes a specialized

- 1https://docs.anthropic.com/en/docs/claude-code
- 2https://github.com/openai/codex
- 3https://github.com/google-gemini/gemini-cli

###### Milestone M001 SRS (zeromicro/go-zero, v1.6.0→v1.9.3): go-redis v9 Upgrade with API Modernization

Overview. This milestone upgrades the go-redis dependency from v8 to v9 across the Redis module (core/stores/redis, core/iox, zrpc/internal/clientinterceptors), requiring a new package import path, migration from callback-based to middleware-based hooks, error handling via errors.Is()/errors.As(), and updated sorted-set API signatures.

- FR1: Migrate go-redis Import Path Problem. The Redis client library import path has changed from github.com/go-redis/redis/v8 to github.com/redis/go-redis/v9, causing compilation failures. Requirements.

- • Update all import statements referencing the old go-redis v8 package to use the new v9 package path.
- • Ensure all Redis-related source files compile with the new import path.
- • Maintain the red alias convention for the imported package.

Acceptance.

- • When building the project, no import errors occur for the Redis package.
- • All files in the redis module successfully import from the v9 package location.

- FR2: Implement New Hook Interface Pattern Problem. The go-redis v9 library replaced the callback-based hook interface (BeforeProcess/AfterProcess methods) with a middleware-style hook interface (ProcessHook/ProcessPipelineHook functions that wrap the next handler). Existing hook implementations fail to compile. Requirements.

- • Implement the DialHook(next red.DialHook) red.DialHook method that passes through to the next handler.
- • Replace BeforeProcess/AfterProcess with a single ProcessHook(next red.ProcessHook) red.ProcessHook that captures the start time, starts the tracing span, invokes the next handler, ends the span, and records metrics and slow query logs.
- • Replace BeforeProcessPipeline/AfterProcessPipeline with a single ProcessPipelineHook method with equivalent behavior for pipeline operations.
- • Remove the context-based start-time storage mechanism used by the old hook pattern.
- • Update the startSpan helper to return both the context and an endSpan closure.

###### Acceptance.

- • When a Redis command is executed, the tracing span is properly created and completed with correct status.
- • When a Redis command exceeds the slow threshold, the slow query is logged.
- • When a Redis command fails, the error is properly recorded in the span and metrics.
- • When a pipeline of commands is executed, the combined duration is measured and logged appropriately.

###### · · ·

Environment Dependency Changes. The SRS additionally lists the Go package upgrades and additions required by this milestone (e.g., github.com/redis/go-redis/v9 v9.4.0 added, github.com/go-redis/redis/v8 removed, along with updates to roughly forty transitive dependencies). The full dependency list is included with each milestone’s SRS in the released dataset.

- Figure 17 | Excerpt of the SRS for milestone M001 of the zeromicro/go-zero repository (v1.6.0→v1.9.3) in EvoClaw, showing the Overview, two of the seven Feature Requirements (FR1 and FR2), and the trailing Environment Dependency section. Each Feature Requirement specifies a Problem, a Requirements block, and an Acceptance clause. Full SRS instances for all milestones are available at https://huggingface.co/datasets/EvoClaw-Bench/EvoClaw-data.

summarizer that distills the conversation into a structured snapshot preserving the overall goal, key knowledge, file system state, and the agent’s current plan.

• OpenHands (v1.2.1)4 Wang et al. (2025) is an open-source platform for autonomous software development agents. Unlike the provider-specific frameworks above, OpenHands serves as a model-agnostic harness, enabling cross-provider evaluation under a unified agent architecture. We pair it with Claude Opus 4.6, GPT 5.3-Codex, Gemini 3 Flash, Kimi K2.5, and MiniMax M2.5. OpenHands uses an LLMSummarizingCondenser that triggers compression when total tokens surpass a model-dependent threshold (500K for Gemini 3 Flash, Gemini 3 Pro, and Gemini 3.1 Pro; 160K for all others), preserving the first 4 events (task description and initial context) and using the same underlying model to summarize the remainder.

Runtime Configuration. All agents operate within identical sandboxed Docker environments.

Agent System Prompt. Under the Continuous Task Evaluation setting, every agent receives the same system prompt (Figure 18) regardless of framework or model. The prompt declares the agent’s role as a software engineer, exposes the working directory, source-code paths, task-queue file, and per-milestone SRS directory, lists the four critical constraints (continuous context, in-scope edit boundaries, one-shot tagged submission, and an asynchronously updated streaming queue), and defines the monitor-implement-submit loop in which a git tag on the milestone identifier is the sole signal that triggers external evaluation.

###### B.3. Detailed Dataset Statistics

Table 3 presents comprehensive per-repository statistics for EvoClaw. The table is organized into six major categories:

Repository. Basic repository information including the organization/repository name on GitHub, the number of source files (#Files), and total lines of code (#LoC) at the end version of the analyzed range.

Release Range. The version span analyzed by DeepCommit, specified by start and end tags, along with the delta lines of code (ΔLoC) representing the total code changes between these versions.

Milestone DAG. Statistics about the extracted milestone directed acyclic graph: the number of graded milestones (#M; the 3 non-graded context milestones are excluded), inter-milestone dependencies (#Deps), and the coefficient of variation (CV) of patch LOC across milestones. A higher CV indicates greater diversity in milestone complexity within the repository.

Fix Patches. Metrics characterizing the gold patches: average lines of code (Avg.LoC), average number of files modified (Avg.#F), and estimated human development time (HumanDT) required to implement each milestone.

4https://github.com/All-Hands-AI/OpenHands

###### Continuous Task Evaluation Agent System Prompt (e2e/prompt/v2.md)

You are an expert Software Engineer working in a continuous integration environment. Your role is to sequentially implement software development tasks from a dynamic queue, maintaining a single continuous context throughout the entire session. You are responsible for writing code, running tests, and managing version control directly.

###### Environment

- • Working Directory: /testbed (the repository root).
- • Source Code: {src_dirs}.
- • Task Queue File: /e2e_workspace/TASK_QUEUE.md (read-only, updated asynchronously by the system).
- • SRS Directory: /e2e_workspace/srs/ (contains {milestone_id}_SRS.md files).

###### Critical Constraints

- 1. Continuous Context: You maintain FULL MEMORY of all previous tasks, decisions, and code changes. Use this knowledge to ensure consistency and avoid regressing previous fixes.
- 2. Scope: Only changes within the Source Code directories ({src_dirs}) are validated for the final submission. However, you MAY (and should) modify or add tests to verify your work locally.
- 3. One-Shot Submission: Once you create a submission tag, the task is considered done and removed from the queue. You cannot edit a tagged submission.
- 4. Streaming Queue: The Task Queue is dynamic. New tasks may appear asynchronously as dependencies are satisfied. You must poll it continuously.

Workflow. Follow this continuous loop.

- Step 1: Monitor Task Queue. Constantly read /e2e_workspace/TASK_QUEUE.md to see available tasks. If multiple tasks are available, prioritize them based on the order listed or dependency logic.
- Step 2: Implement Task. For each task found in the queue:

- 1. Read Requirements from the SRS file at the path shown in TASK_QUEUE.md (format: /e2e_workspace/srs/{milestone_id}_SRS.md).
- 2. Plan and Implement: analyze the codebase, plan changes, and modify the code in {src_dirs}. Verify by running existing tests or by creating new reproduction scripts to ensure that no existing functionality regresses.
- 3. Refine the implementation until satisfied.

- Step 3: Finalize and Submit. When the implementation is complete and verified:

- 1. Commit changes with git add {src_dirs} followed by git commit -m "Implement {milestone_id}".
- 2. Tag for submission via git tag agent-impl-{milestone_id}. This is the ONLY signal that the task is complete, and tagging immediately triggers the external evaluator and updates the Task Queue.

- Step 4: Loop. After tagging, IMMEDIATELY re-read /e2e_workspace/TASK_QUEUE.md and repeat Step 2 for any new or remaining tasks. Leverage your memory of previous tasks to handle integration points and shared components effectively. Exit Condition. Only stop when /e2e_workspace/TASK_QUEUE.md shows “(No tasks currently available)” AND you have completed processing all previously claimed tasks.

- Figure 18 | System prompt provided to every agent under the Continuous Task Evaluation setting in EvoClaw. The prompt is framework-agnostic and is loaded as the agent’s system message before the first SRS task is read. Variable placeholders such as {src_dirs} and {milestone_id} are filled in per repository and milestone at runtime.

Table 3 | Detailed per-repository statistics for EvoClaw. HumanDT and HumanWT are average humanestimated development time and SRS writing time, respectively. #M counts graded milestones; the 3 non-graded context milestones are excluded.

Codebase Release Range Milestone DAG

Org/Repo #Files #LoC Start End ΔLoC #M #Deps LoC CV zeromicro/go-zero 1,021 110K v1.6.0 v1.9.3 6,403 23 25 1.29 apache/dubbo 4,279 350K 3.3.3 3.3.6 4,154 12 9 0.76 BurntSushi/ripgrep 159 48K 14.1.1 15.0.0 1,474 11 12 0.83 nushell/nushell 1,727 264K 0.106.0 0.108.0 15,520 13 28 1.10 element-hq/element-web 2,430 476K v1.11.95 v1.11.97 7,657 18 12 0.87 navidrome/navidrome 1,110 144K v0.57.0 v0.58.0 5,900 9 9 1.02 scikit-learn/scikit-learn 1,314 280K 1.5.2 1.6.0 7,372 12 14 0.84 Total/Average 12,040 1.67M — — 48,480 98 109 0.96

###### Fix Patches SRS Unit Tests

Org/Repo Avg.LoC Avg.#F HumanDT Avg.#W HumanWT F2P P2P zeromicro/go-zero 278 10.2 4h-1d 1,330 4h+ 2.2 1,812 apache/dubbo 346 10.8 1-3d 1,138 1-2h 1.1 6,924 BurntSushi/ripgrep 134 5.5 1-3d 879 30m-1h 1.3 1,057 nushell/nushell 1,268 63.3 1-3d 1,528 1-2h 7.6 4,736 element-hq/element-web 445 27.2 1-3d 1,546 2-4h 7.1 5,235 navidrome/navidrome 656 13.2 1-3d 1,954 30m-1h 3.0 1,452 scikit-learn/scikit-learn 1,167 58.6 1-3d 1,580 30m-1h 97.2 22,308 Total/Average 570 27.4 1-3d 1,348 2-4h 17.1 6,218

SRS. Software Requirement Specification statistics: average word count (Avg.#W) measuring specification length, and estimated human writing time (HumanWT) for authoring equivalent specifications. Note that SRS data is currently available for 5 repositories (dubbo, ripgrep, nushell, navidrome, scikit-learn).

Unit Tests. Test suite metrics: average Fail-to-Pass (F2P) tests that verify new functionality, and average Pass-to-Pass (P2P) tests ensuring backward compatibility. Notably, scikit-learn exhibits the highest F2P count (97.2) due to its comprehensive test coverage requirements.

###### B.4. Milestone DAG Visualizations

Figures 19 to 22 present the full Milestone DAGs for all seven repositories in EvoClaw. Each node is a card-style box representing a milestone, containing (top to bottom): a short ID, a descriptive title, summary statistics (number of commits, lines of code changed, and number of fail-to-pass tests), and one or more category tags. The border color of each node reflects its primary category: Feature (blue), Bugfix (red), Refactor (orange), Enhance (teal), or Chore (gray). Milestones belonging to multiple categories display all applicable tags. Gray nodes with dashed borders indicate non-graded milestones (3 in total: 2 in ripgrep, 1 in dubbo), which are included in the execution sequence for context continuity but excluded from scoring. The benchmark therefore contains 98 graded milestones and 3 non-graded milestones across all repositories (101 total). Solid red edges denote strong dependencies (upstream removal causes build or test failures downstream), while dashed gray edges denote weak dependencies (functionality degrades gracefully without the upstream milestone). Orange dashed

edges represent additional dependencies inferred during the DAG refinement stage. The DAGs are laid out top-to-bottom from upstream prerequisites to downstream dependents. Milestones with no dependency edges are grouped in a dashed “Independent Milestones” box at the bottom of each DAG.

#### C. Extended Experimental Analysis and Discussion

This appendix provides additional experimental analyses: qualitative agent-code quality patterns, per-model cumulative error trajectories on element-web, and a comparison case study against human-annotated Milestone DAGs.

###### C.1. Agent Code Quality Analysis

Beyond aggregate metrics, we examine representative cases to understand how agent-generated patches differ from human-written patches in structural quality. We identify three recurring quality anti-patterns through qualitative analysis.

Responsibility Boundary Misplacement. Figure 23 illustrates a case from apache/dubbo (M004) where the agent correctly identifies the logic to implement but places it at the wrong abstraction level. The task requires skipping deserialization for stream parameters in FallbackArgumentResolver. The ground truth inserts the check in resolveValue(), allowing accept() to still claim the parameter, after which the framework injects the real stream instance. The agent instead places the check in accept(), causing the terminal resolver to reject the parameter entirely and triggering 3 P2P regressions. The agent’s solution is functionally plausible—checking isStream() before processing is a reasonable heuristic—but violates the resolver chain’s structural contract, revealing a quality gap in architectural reasoning.

Shotgun Fixes. Figure 24 shows a pattern from nushell where the agent replaces targeted validation with overly permissive error suppression. The task requires break/continue outside loops to produce compile-time errors. The ground truth adds an is_in_loop() guard in compile_break() and precisely exempts only the known catch block false positive. The agent instead deletes the is_in_loop() API entirely, removes all guards from compile_break(), and globally suppresses NotInALoop errors across all block expressions. Tests still pass because an internal fallback in push_break() happens to catch the error—but the SRS-mandated API contract is violated, and the deleted API blocks future evolution of the compiler’s context-tracking infrastructure. This pattern exemplifies a broader quality gap: agents achieve test-passing behavior through coarse-grained suppression rather than precise, contract-preserving fixes.

API Signature Degradation. Figure 25 demonstrates how a local algorithm choice can silently degrade a public API. In nushell’s multi-output type checking, the ground truth iterates per inputoutput pair, keeping the expected field of OutputMismatch as a structured Type enum, adds a reusable utility in ty.rs, and cleans up obsolete workarounds (7 files modified). The agent groups by unique input types, which forces the expected field to become an opaque String (losing patternmatchability), defines only a local helper (leaving 35 lines of duplicate logic elsewhere), and retains stale FIXME workarounds (2 files modified). Both pass all F2P tests, but the agent’s patch introduces architectural debt—weaker type signatures, missed deduplication, and retained technical debt—that compounds across milestones.

Feature Bugﬁx Refactor Enhance Chore Non-graded ━━━▸ Strong dep. ╌╌╌▸ Weak dep. ╌╌╌▸ Additional dep.

###### M1

###### M5

###### M6

###### MUU

###### M8

RoomListStoreV3 Filter Infrastructure

Key Storage Toggle for Encryption Settings

Identity Reset Protection UX Enhancement

UI/UX and Accessibility Improvements

Group Call Error Screen Display 1 commit · 293 LoC · 4 F2P tests

9 commits · 486 LoC · 29 F2P tests

3 commits · 372 LoC · 9 F2P tests

3 commits · 62 LoC · 2 F2P tests

7 commits · 100 LoC · 27 F2P tests

Feature Refactor Enhance

Feature

Feature Refactor

Enhance

Enhance

###### M10

###### M2.1

###### FE

###### M9

Element Call Bundle Integration for Oﬄine Deployment

Room List V3 ViewModel Foundation

Reply Mentions and Notiﬁcation API Improvements 3 commits · 253 LoC · 13 F2P tests

Simpliﬁed Sliding Sync Architecture

- 7 commits · 415 LoC · 2 F2P tests Feature Refactor

M2.2

Room List V3 UI Components

- 8 commits · 791 LoC · 60 F2P tests Feature Refactor

1 commit · 952 LoC · 16 F2P tests

1 commit · 135 LoC · 8 F2P tests

Feature

Refactor

Feature Refactor

###### M4

React DOM Rendering Modernization for Message Content

4 commits · 1,428 LoC · 20 F2P tests

Refactor

###### M12

###### M2.3

###### MBF

Enhanced Rageshake Error Reporting with Friendly Messages

Room List V3 Room Behavior and Display

Cross-Subsystem Bug Fixes and Stability Improvements

5 commits · 587 LoC · 22 F2P tests

6 commits · 80 LoC · 2 F2P tests

1 commit · 127 LoC · 8 F2P tests

Feature Refactor

Bugﬁx

Feature Refactor

Independent Milestones (3)

###### M7

###### M11

###### M13

###### M3

SSO Pickle Key Generation and Session Handling

Image and Video Hiding Privacy Feature

Compound Design System Checkbox Migration

React 19 JSX Import Migration 2 commits · 456 LoC · — F2P tests

3 commits · 92 LoC · 1 F2P test

2 commits · 446 LoC · 15 F2P tests

1 commit · 572 LoC · 6 F2P tests

Refactor

Bugﬁx

Feature

###### Enhance Chore

- (a) element-web, v1.11.95 → v1.11.97 (18 milestones)

M1.1 (non-graded)

Windows Hyperlink Path Performance Optimization 1 commit · 53 LoC · — F2P tests

Bugﬁx Enhance

M1.2

Hyperlink Alias Modular Architecture Restructuring 2 commits · 276 LoC · — F2P tests

Feature Refactor

MS1 (non-graded)

Code Style Refactoring to Modern Rust Idioms

6 commits · 196 LoC · — F2P tests

Bugﬁx Refactor Enhance

M2

Replacement Text Support in JSON Output Mode

1 commit · 117 LoC · 1 F2P test

Feature

M7.2

Simplify Printer max_matches After Searcher Migration

1 commit · 108 LoC · — F2P tests

Feature

- M3.1

Nested Alternation Groups Support in Glob Patterns

1 commit · 65 LoC · 9 F2P tests

Feature

- M3.2

Unclosed Character Class Toggle for Glob Parsing

1 commit · 119 LoC · 10 F2P tests

Feature

M4

Fix Global Gitignore Matching with Absolute Paths

1 commit · 172 LoC · 1 F2P test

Bugﬁx Refactor

M5

Add Minimum Depth Option for Directory Traversal 1 commit · 51 LoC · 1 F2P test

Feature

M6

Fix Performance with Large After-Context Flag Values

2 commits · 27 LoC · 2 F2P tests

Bugﬁx Enhance

M7.1

Migrate Max Matches Limit from Printer to Searcher

2 commits · 240 LoC · 8 F2P tests

Bugﬁx Refactor

MF1S

Exit Code and Search Worker Initialization Bug Fixes

6 commits · 158 LoC · 4 F2P tests

Feature Bugﬁx Enhance

MF1S

Maintenance Fixes and Minor Enhancements

16 commits · 141 LoC · 7 F2P tests

Feature Bugﬁx

- (b) ripgrep, 14.1.1 → 15.0.0 (13 milestones)

- Figure 19 | Milestone DAGs for EvoClaw repositories (Part 1/4).

Feature Bugﬁx Refactor Enhance Chore Non-graded ━━━▸ Strong dep. ╌╌╌▸ Weak dep. ╌╌╌▸ Additional dep.

###### M16.1

REST Bean Argument Resolution and Protocol Fixes

- 2 commits · 175 LoC · 2 F2P tests Bugﬁx

- M3.1

Mutiny Module Conﬁguration and Code Generation Templates 1 commit · 358 LoC · 1 F2P test

Feature

- M3.2 (non-graded)

Mutiny Core Abstractions and Publisher/Subscriber

- 1 commit · 455 LoC · — F2P tests Feature

M3.3

Mutiny Reactive Streaming Support

- 2 commits · 543 LoC · 19 F2P tests Feature Enhance

M4

REST Streaming and SSE Infrastructure Enhancements

- 3 commits · 497 LoC · 1 F2P test Feature Bugﬁx

10 commits · 496 LoC · 6 F2P tests

Bugﬁx

###### M1.1

###### M18

###### M25

Spring Security OAuth2 Serialization Support

Performance Optimizations for Hot Code Paths

Miscellaneous Fixes and Improvements

2 commits · 500 LoC · 4 F2P tests

3 commits · 110 LoC · 6 F2P tests

10 commits · 135 LoC · — F2P tests

Feature

Enhance

Bugﬁx Enhance

###### M1.2

Spring 6 Security Module Split and UTF-8 Standardization

2 commits · 385 LoC · 11 F2P tests

Feature Enhance

Independent Milestones (4)

###### M6

###### M11

###### M17

RadixTree Multi-Method REST Path Routing Fix

Logging FQCN Infrastructure Enhancement

Remoting Module Stability and Triple Protocol Enhancements

1 commit · 271 LoC · 5 F2P tests

6 commits · 239 LoC · 1 F2P test

Feature

###### Feature Bugﬁx

###### M19

Minor Feature Enhancements and Additions

8 commits · 445 LoC · 6 F2P tests

Feature

###### (a) dubbo, 3.3.3 → 3.3.6 (13 milestones)

###### M2

###### M6

Plugin API Extensions 4 commits · 99 LoC · 1 F2P test

UI Quality Improvements 6 commits · 97 LoC · 6 F2P tests

Feature Bugﬁx

###### Bugﬁx

###### M3.1

###### M1

Multi-Library Database and Persistence Foundation

Plugin System Core Improvements 5 commits · 1,064 LoC · 97 F2P tests

5 commits · 1,226 LoC · 191 F2P tests

Feature Bugﬁx Enhance

Feature

###### M3.2

Multi-Library Core Service and Scanner

2 commits · 991 LoC · 95 F2P tests

Feature

###### M3.3

Multi-Library Server APIs 4 commits · 610 LoC · 89 F2P tests

Feature

###### M3.4

Multi-Library UI Components 2 commits · 1,571 LoC · 93 F2P tests

Feature

###### M4

Scanner Quality Improvements 4 commits · 90 LoC · 4 F2P tests

Feature Enhance

Independent Milestones (1)

###### M7

Agent Logic Streamlining 1 commit · 152 LoC · 9 F2P tests

Refactor

(b) navidrome, v0.57.0 → v0.58.0 (9 milestones)

###### M12.1

Infrastructure Phase 1: API Changes and Bug Fixes

41 commits · 2,583 LoC · 874 F2P tests

Feature Bugﬁx Refactor Chore

###### M12.2

Infrastructure Phase 2: Metadata Routing and Bug Fixes 35 commits · 1,208 LoC · 72 F2P tests

Feature Bugﬁx Refactor Enhance Chore

###### M3

###### M12.3

Estimator Type Detection and Pipeline Fitted State Checks

Infrastructure Phase 3a: Metrics and Validation Updates 11 commits · 448 LoC · 17 F2P tests

2 commits · 289 LoC · 2 F2P tests

Feature

Feature Bugﬁx Refactor Enhance

###### M1

###### M12.4

###### M6

###### M11

Metadata Routing for RFE and RFECV Feature Selection

Infrastructure Phase 3b: Testing and Metadata Routing 11 commits · 703 LoC · 104 F2P tests

Array API Support for Metrics and Preprocessing

Missing Value Support for ExtraTree Estimators

2 commits · 262 LoC · 10 F2P tests

7 commits · 304 LoC · 5 F2P tests

1 commit · 174 LoC · 20 F2P tests

Feature Bugﬁx

Feature Bugﬁx Refactor

Feature Refactor

Feature

###### M12.5

###### M13

Tags System Migration and Validation Enhancements

Documentation Consistency and Metadata Routing Coverage

10 commits · 391 LoC · 47 F2P tests

75 commits · 655 LoC · 14 F2P tests

Feature Bugﬁx Refactor

Feature Bugﬁx

###### M17

###### M4

Performance Optimizations for Manifold and Covariance

Estimator Validation and Performance Improvements 3 commits · 149 LoC · 1 F2P test

3 commits · 206 LoC · 8 F2P tests

###### Feature Enhance

Feature Enhance

(c) scikit-learn, 1.5.2 → 1.6.0 (12 milestones)

- Figure 20 | Milestone DAGs for EvoClaw repositories (Part 2/4).

Feature Bugﬁx Refactor Enhance Chore Non-graded ━━━▸ Strong dep. ╌╌╌▸ Weak dep. ╌╌╌▸ Additional dep.

###### MG04_1ddae02

###### MM02_parser

Query XML Multiple Output Types and Enhanced Nodeset Formatting

Parser Fixes for Expressions and Scoping and Error Reporting

4 commits · 451 LoC · 11 F2P tests

7 commits · 296 LoC · 6 F2P tests

Feature Enhance

###### Bugﬁx

###### Mcore_development.2

Operators and Completions and REPL Improvements

28 commits · 2,138 LoC · 20 F2P tests

Bugﬁx Refactor

###### MG05_0b8531e

Optimized String Types Infrastructure for nu-utils 3 commits · 1,002 LoC · 9 F2P tests

Feature Refactor Chore

###### MG02_da9615f

###### MG04_ca0e961

###### Mcore_development.3

Parse-time Pipeline Type Checking for Multiple Output Types

Overlay List Active Status and Scoped Module Registration

Commands and Parser and Error Handling Improvements

1 commit · 96 LoC · 11 F2P tests

27 commits · 1,778 LoC · 14 F2P tests

1 commit · 186 LoC · 2 F2P tests

Enhance

Feature Bugﬁx Enhance

Bugﬁx

###### MG01_48bca0a

###### MG05_be6e868

Custom Completion Refactoring for Flag Name vs Value

LSP Testing Refactoring and Goto Deﬁnition Fix

3 commits · 303 LoC · 2 F2P tests

4 commits · 3,036 LoC · — F2P tests

Bugﬁx Refactor

Refactor

###### MM08_docs

Documentation and Help Improvements with LSP Integration

6 commits · 206 LoC · 3 F2P tests

Feature Bugﬁx Refactor

###### Mcore_development.1

Core Shell Language and Type System Fixes

27 commits · 2,199 LoC · 25 F2P tests

Feature Bugﬁx Enhance

###### Mcore_development.4

Case-Insensitive Get and Flatten Each Completions 28 commits · 3,775 LoC · 41 F2P tests

Feature Bugﬁx

###### MG02_a647707

Runtime Enforcement of Type Annotations on Variables

1 commit · 54 LoC · 3 F2P tests

Feature Enhance

(a) nushell, 0.106.0 → 0.108.0 (13 milestones)

- Figure 21 | Milestone DAGs for EvoClaw repositories (Part 3/4).

Feature Bugﬁx Refactor Enhance Chore Non-graded ━━━▸ Strong dep. ╌╌╌▸ Weak dep. ╌╌╌▸ Additional dep.

###### M23

###### M5

SQL and ORM Metrics and Bug Fixes

SQL Statement-Level Circuit Breaker Support

10 commits · 78 LoC · 88 F2P tests

5 commits · 280 LoC · 197 F2P tests

Bugﬁx Enhance

Feature

###### M26

###### M17

Conﬁg Parsing Fix and K8s Build Tag Support

Mapping and Unmarshaler Bug Fixes

4 commits · 32 LoC · 19 F2P tests

19 commits · 152 LoC · 14 F2P tests

Bugﬁx Chore

Bugﬁx Enhance

###### M21

###### M4

Service Infrastructure and Conﬁg Validation Improvements 6 commits · 170 LoC · 30 F2P tests

CPU Monitoring and Adaptive Load Shedding

7 commits · 402 LoC · 62 F2P tests

Bugﬁx Enhance

Feature Enhance

###### M3

###### M18

Circuit Breaker Algorithm Evolution and Recovery Optimization

Logging System Enhancements and Bug Fixes

19 commits · 615 LoC · 148 F2P tests

8 commits · 411 LoC · 218 F2P tests

Feature Enhance

Feature Refactor Enhance

###### M1

Go-Redis v9 Upgrade with API Modernization

4 commits · 195 LoC · 134 F2P tests

Refactor

###### M27

###### M7.1

###### M19

Lua Scripts Externalization to Embedded Files

Etcd Conﬁg Center Infrastructure

HTTP Request/Response Parsing and CORS Improvements

2 commits · 156 LoC · 17 F2P tests

8 commits · 460 LoC · 36 F2P tests

13 commits · 219 LoC · 718 F2P tests

Refactor

Feature

Bugﬁx Enhance

###### M20

###### M7.2

Code Modernization with Go Generics and Stdlib

Etcd Discovery Mechanism Reliability Fixes

15 commits · 615 LoC · 63 F2P tests

11 commits · 537 LoC · 275 F2P tests

Refactor Chore

Bugﬁx Enhance

###### M14

###### M10

Continuous Proﬁling with Pyroscope Integration

SSE Routes Support with Timeout Handling

7 commits · 461 LoC · 240 F2P tests

9 commits · 175 LoC · 177 F2P tests

Feature

Feature

###### M22

###### M28

Trace and Observability Stability Improvements 5 commits · 53 LoC · 215 F2P tests

Miscellaneous Cross-Module Fixes and Improvements

42 commits · 447 LoC · 644 F2P tests

Bugﬁx Enhance

Bugﬁx Chore

Independent Milestones (5)

###### M8

###### M9

###### M13

REST File Server for Static File Serving

HTTP-to-HTTP Gateway Proxy Support

Consistent Hash Load Balancer for zRPC

5 commits · 117 LoC · 50 F2P tests

10 commits · 343 LoC · 51 F2P tests

1 commit · 122 LoC · 54 F2P tests

Feature

Feature

Feature

###### M24

###### M25

zRPC Server Interceptor and Balancer Improvements

Threading and Concurrency Utility Enhancements

5 commits · 216 LoC · 86 F2P tests

3 commits · 147 LoC · 21 F2P tests

Bugﬁx Refactor

###### Feature

(a) go-zero, v1.6.0 → v1.9.3 (23 milestones)

- Figure 22 | Milestone DAGs for EvoClaw repositories (Part 4/4).

###### Stream Check Placement: accept() vs resolveValue()

apache/dubbo v3.3.3 → v3.3.6 (M004)

###### SRS / FR1: STREAM PARAMETER DETECTION AND SKIPPING

All three REST dialects must check isStream() in their FallbackArgumentResolver implementations. When a parameter is detected as a stream type, the resolver must return null immediately without attempting request body resolution. Non-stream bean parameters in the same method signature must still be correctly resolved from the request body.

###### Gold Patch: check in resolveValue() PASS

###### Agent Generated Patch: check in accept() 3 REGRESSIONS

basic/FallbackArgumentResolver.java accept() · unchanged

basic/FallbackArgumentResolver.java accept() · modified

@Override public boolean accept(ParameterMeta param) {

@Override public boolean accept(ParameterMeta param) {

return param.getToolKit().getDialect()

+ if (param.isStream()) {

== RestConstants.DIALECT_BASIC; }

+ return false;

###### + }

return param.getToolKit().getDialect()

basic/FallbackArgumentResolver.java resolveValue() · modified

== RestConstants.DIALECT_BASIC; }

@Override protected Object resolveValue(...) {

basic/FallbackArgumentResolver.java resolveValue() · unchanged

...

+ if (meta.parameter().isStream()) {

@Override protected Object resolveValue(...) {

+ return null;

###### + }

... if (single) {

if (single) { if (Map.class.isAssignableFrom(...))

if (Map.class.isAssignableFrom(...))

###### ✓ All 3 pbServerStream tests pass

###### ✗ 3 P2P regressions, all pbServerStream tests fail

basic::pb server stream · basic::pb server stream get · spring::pb server stream

basic::pb server stream · basic::pb server stream get · spring::pb server stream

- Figure 23 | Responsibility boundary misplacement in apache/dubbo (M004). The ground truth checks stream parameters in resolveValue(), preserving the resolver chain contract. The agent checks in accept(), ejecting the parameter from the terminal resolver and causing 3 P2P regressions.

These cases reveal a common quality pattern: agent-generated patches achieve superficial correctness (tests pass) while introducing structural and maintainability issues—misplaced abstractions, coarsegrained suppression, and degraded API contracts—that are invisible to automated evaluation but consequential in long-horizon development.

###### C.2. Cumulative Error Analysis on element-web

Figure 26 contrasts continuous and independent evaluation on element-web across multiple models. Under continuous evaluation, errors introduced at early milestones, such as regressions and unresolved bugs, propagate through subsequent development stages, producing a pronounced snowball effect in which agents must operate on an increasingly unstable codebase. In contrast, Gemini 3 Flash under independent evaluation achieves substantially higher performance than all continuous runs, including those of frontier models such as GPT 5.2 and Claude Opus 4.5. This gap illustrates that independent-task evaluation effectively serves as an optimistic upper bound, substantially overstating an agent’s ability to sustain coherent software evolution. Together, these results highlight long-horizon codebase maintenance as a central bottleneck: even state-of-the-art agents struggle to control error accumulation and technical debt when development unfolds continuously.

###### C.3. Case Study: Human vs. DeepCommit Milestone DAG Construction

We conduct a structured comparison between the Human-annotated Milestone DAG and the DeepCommit DAG for the scikit-learn v1.5.2–v1.6.0 release interval (373 commits over 7 months). As shown in Figures 27 and 28, the Human DAG consists of 14 milestones organized into 6 semantic groups with 20 dependency edges, whereas the DeepCommit DAG consists of 12 milestones organized into 6 data-driven groups with 14 dependency edges.

Both DAGs adopt a two-level hierarchy (groups → milestones). However, the Human DAG is structured

###### The Shotgun Fix: Default Deny vs Global Suppression

nushell v0.106.0 → v0.108.0

###### SRS / FR3: COMPILE-TIME LOOP CONTROL STATEMENT VALIDATION

Using break / continue outside a loop must produce a compile-time error ( CompileError::NotInALoop ). The compile_break() function must return this error when not inside a loop.

###### Gold Patch: default deny, precise exception PASS

###### Agent Generated Patch: global suppression, API deleted TECH DEBT

crates/nu-engine/src/compile/keyword.rs modified

crates/nu-engine/src/compile/keyword.rs modified

pub(crate) fn compile_break(...) -> Result<(), CompileError> {

pub(crate) fn compile_break(...) -> Result<(), CompileError> {

- - if builder.is_in_loop() {

+ if !builder.is_in_loop() {

+ return Err(CompileError::NotInALoop {

+ msg: "...".to_string(),

+ span: Some(call.head),

+ });

+ }

- - builder.load_empty(io_reg)?;
- - builder.push_break(call.head)?;
- - builder.add_comment("break");
- - } else { /* Call fallback */ }

- - if builder.is_in_loop() {
- - builder.load_empty(io_reg)?;
- - builder.push_break(call.head)?;
- - builder.add_comment("break");
- - } else { /* Call fallback */ }

+ builder.load_empty(io_reg)?;

+ builder.push_break(call.head)?;

+ builder.add_comment("break");

Ok(()) }

crates/nu-parser/src/parser.rs parse_block_expression() · modified

+ builder.load_empty(io_reg)?;

+ builder.push_break(call.head)?;

let compile_errors_len =

+ builder.add_comment("break");

working_set.compile_errors.len(); let mut output = parse_block(...);

Ok(()) }

+ // Remove NotInALoop compile errors

+ // from block's independent compilation

crates/nu-parser/src/parser.rs parse_internal_call() · modified

+ if working_set.compile_errors.len() > compile_errors_len {

let compile_error_count =

+ let new_errors: Vec<_> = working_set

working_set.compile_errors.len(); let arg = parse_multispan_value(...);

+ .compile_errors

+ .drain(compile_errors_len..)

+ // HACK: catch block compiled as Closure

+ .filter(|err| !matches!(err, CompileError::NotInALoop { .. }))

+ if let SyntaxShape::Keyword(ref keyword, ..) = positional.shape

+ .collect();

+ && keyword == b"catch"

+ working_set.compile_errors.extend(new_errors);

+ && let [CompileError::NotInALoop { .. }] =

###### + }

+ &working_set.compile_errors[compile_error_count..]

crates/nu-engine/src/compile/builder.rs deleted

###### + {

+ working_set.compile_errors.truncate(compile_error_count);

- - /// True if we are currently in a loop.
- - pub(crate) fn is_in_loop(&self)
- - -> bool
- - {
- - !self.loop_stack.is_empty()
- - }

###### + }

crates/nu-engine/src/compile/builder.rs unchanged

/// True if we are currently in a loop. pub(crate) fn is_in_loop(&self)

-> bool {

⚠ F2P tests pass (via push_break() fallback), but API contract violated

!self.loop_stack.is_empty() }

✓ F2P tests pass; is_in_loop() API preserved for future evolution

- Figure 24 | Shotgun fix in nushell. The gold patch uses a whitelist strategy: compile_break() returns NotInALoop as required by the SRS, and only the known catch-block false positive is exempted in parse_internal_call. The agent uses a blacklist strategy: it suppresses all NotInALoop errors across every block expression and deletes the is_in_loop() API. Tests pass only because push_break() has an internal fallback.

around manually defined release-planning themes (e.g., framework evolution, backend compatibility, quality assurance), while the DeepCommit DAG reflects phases and clusters induced from commitlevel dependency topology. In addition, the Human DAG distinguishes functional and process-level dependencies with explicit strength annotations, whereas the DeepCommit DAG encodes dependencies derived from structural signals in the commit graph.

This case study is not intended as a performance comparison. Rather, it examines how distinct construction principles—semantic curation versus topology-driven aggregation—lead to systematically different milestone abstractions and dependency structures.

C.3.1. Commit Coverage

The 201 DeepCommit commits form a strict subset of the 373 Human commits. The 172 filtered commits result from the pipeline’s source-file filtering and F2P test requirements, which preferentially exclude Human milestones whose commits have low inter-dependency, since such commits tend to

###### API Signature Degradation: Type vs String

nushell v0.106.0 → v0.108.0

###### SRS / FR1–FR3: MULTI-OUTPUT TYPE CHECKING

When a command declares multiple output types for the same input (e.g., record → record and record → date ), the parser must track all possible outputs and propagate them through the pipeline. Error messages must display all candidate types.

###### Gold Patch: preserve structured Type PASS

###### Agent Generated Patch: degrade to String TECH DEBT

crates/nu-protocol/src/errors/parse_error.rs modified

crates/nu-protocol/src/errors/parse_error.rs modified

OutputMismatch( Type, // expected: preserved

OutputMismatch(

- - Type, // expected: structured enum

+ String, // expected: now opaque

- - Type, // actual

- Type, // actual

+ String, // actual: multi-type combo

Span, ),

+ String, // actual: multi-type combo

Span, ),

crates/nu-parser/src/type_check.rs modified

crates/nu-parser/src/type_check.rs modified

// check_block_input_output()

+ for (input_type, output_type) in

// check_block_input_output()

+ &block.signature.input_output_types

+ for input_type in &unique_input_types {

###### + {

+ let declared_outputs: Vec<&Type> =

if !current_output_types.iter()

+ io_types.iter()

.any(|ty| type_compatible(...)) {

+ .filter(|(it,_)| it == input_type)

+ let s = combined_type_string(

+ .map(|(_,ot)| ot).collect();

+ &current_output_types, "or");

if !any_compatible {

output_errors.push( + OutputMismatch(

output_errors.push( + OutputMismatch(

+ output_type.clone(), // Type

+ format_types(&declared_outputs),

+ s, // String

+ format_types(&current_output_types),

span, ))

span, ))

} }

} }

crates/nu-protocol/src/ty.rs combined_type_string() · new

crates/nu-parser/src/type_check.rs format_types() · local

+ pub fn combined_type_string(

+ fn format_types(types: &[Type]) -> String { ... } // local to type_check.rs only // eval_ir.rs: 35-line duplicate kept

+ types: &[Type], join_word: &str

+ ) -> Option<String> { ... } // reused in type_check.rs ("or") // and eval_ir.rs ("and")

⚠ F2P pass, but Type→String degrades API; workarounds & duplicates remain

✓ F2P pass; API preserves Type enum; workarounds cleaned; eval_ir.rs deduplicated

- Figure 25 | API signature degradation in nushell’s type checker. The agent’s group-by-input algorithm forces OutputMismatch.expected from a structured Type enum to an opaque String, losing pattern-match capability. The agent also omits workaround cleanup and utility deduplication (2 vs. 7 files modified).

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

Milestone Execution Steps

0

2

4

6

8

10

12

CumulativeScore

Continuous vs Independent Task Evaluation (element-web)

Gemini 3 Flash (Independent) Claude Opus 4.5 (Continuous) Codex GPT-5.2 (Continuous)

Gemini 3 Flash (Continuous)

- Figure 26 | Continuous vs Independent evaluation on element-web. The shaded area highlights how error accumulation in continuous mode causes a cost-effective model (Gemini 3 Flash, Independent) to outperform frontier models in continuous mode.

modify isolated files and fail to form strongly connected components in the dependency-induced commit DAG. Among the most affected categories, 73% of build system modernization commits and 78% of Array API interface adaptation commits are excluded, followed by CI/release engineering (56%) and small independent bug fixes (39%). This strategy retains a tighter, more interconnected

###### Human Milestone DAG

M1 — Framework Evolvability M2 — Backend & Ecosystem Compatibility M3 — Technical Debt Retirement

###### M1.1

###### M1.2

###### M2.1 Array API Support for Metrics

###### M3.1 Build System Modernization

###### Estimator Testing Infrastructure Modern…

###### Estimator Tags System Redesign

- ▸ API vs Legacy Test Categorization 0338eec 0b0b90b 0f334d4 13de540 1c3dcb4 364cafe 7bcae6c 938fa18 95e9459

- ▸ Instance Generation & Test Infrastructure 18dc863 3cb7b58 551d56c 668488f 74a3375 a4ebe19 c24a3f9

- ▸ Test Framework Modernization & Cleanup c3d4e57 c71340f cc5372c d33c5e2 d8f0781 d9deffe

- ▸ Tags System Core Implementation 04fbe04 0ad96c3 1429d59 6f424d0 8e9407b ba83d16 bfd91d7 d4ab9ed

- ▸ Estimator Type Migration to Tags 613cff9 e04142c e16a6dd f0080be

- ▸ Tags Utilities & Infrastructure 8f620fd d083972

- ▸ Regression Metrics Array API e12f192 9f44f1f 53ed13c 56dbfd0 5ced13c

- ▸ Classification & Encoding Array API 08de29d 65b2571 68b71b5 e82c14b

- ▸ Array API Infrastructure & Device Handling 16ed528 a624991 bd8f5bd dc6c01c e4362e5 e92dd40 f10c171

- ▸ Meson Build System Migration 06eafd8 21e1642 4cf0772 5e25db7 602aaaa 6032c95 6275f25 d301304 d79cb58 d8185db

- ▸ Build Configuration & Dependencies 6614f75 6fd2f83 7eb7eff 9c2f5c3 b72e81a c3fed50 d840f36

- ▸ Build Infrastructure Cleanup 64f20aa 8132683 a63b021 ab76a90 abbaed3

| |
|---|

| |
|---|

14 commits LoC 4,426

05/21/24-12/03/24

16 commits LoC 673

05/07/24-12/01/24

| |
|---|

22 commits LoC 2,889

05/21/24-10/28/24

| |
|---|

22 commits LoC 2,783

06/27/24-11/21/24

###### M1.4

###### M2.2 Array API for Kernels, Preprocessing & …

###### M3.2 Deprecation Cleanup & Legacy API Removal

Estimator & Pipeline Semantics via Meta…

###### M1.3

- ▸ Feature Selection & Ensemble Routing 38fefe1 5fb9d48 61281cf cef803a e617d82

- ▸ Cross-Validation & Scoring Routing 025d4b0 72844cd 8332017 e749dd9 ff02e17

- ▸ Routing Cleanup & Deprecation 234260d 77fc72c 8a5d8c3 afee65a e1cf244

- ▸ Kernel & Pairwise Metrics Array API 78102fd acd2d90 b461547 be8e28d f6f6b77 fa65654

- ▸ Preprocessing & Encoding Array API 012de1e 0f3c9ff 153205a 1813b4a 3a023b0 5692e59 bed36b2

- ▸ Model Selection & Cross-Validation Array API 23fc332 3bb916f 479e911 48669a5 5bc86b8 a1027b5 a3a0e51 a408a59 a922568 e7af195

Unified Data Validation Public API

- ▸ Parameter & API Deprecation Removal 004cf9e 0f27a26 1bcddcb 215be2e 3b7734e

3c86573 3d5e243 48eeca6 5cbc419 7c5f13b 8d01acf a9ec347 e887583 e9c3942

- ▸ Algorithm & Method Deprecation Cleanup 038bbc1 0422188 1c47967 1d6cfde 26384be 68d8c2c 70aab36 862997f 9ed6207 a4e2bfb af57108 b0f86e7 b4eef25 b679062 bc8eb66 bfa0d65 c1b1f87 f316f4c fbf86e3 fc5428b

- ▸ validate_data Public API Migration 54d1ec9 7d0bec5

- ▸ Validation Parameter Enhancements a490ab1 ea6c77b ef6efef

| |
|---|

5 commits LoC 1,817

06/19/24-09/19/24

| |
|---|

15 commits LoC 1,650

05/12/24-11/14/24

…and 1 more

| |
|---|

23 commits LoC 688

05/15/24-09/12/24

- ▸ Graphviz & HTML Display Fixes 6de55b3 7ab1b64 8ae3ced e02daf5

- ▸ Estimator Signature & Attribute Cleanup ab7ff70

M4 — User-Facing Capability Expansion

###### M4.1

###### M4.2

New Estimator Capabilities and Algorith…

###### Usability and API Ergonomics Improvemen…

- ▸ FrozenEstimator & Calibration 05c0992 775587b bcc6430

- ▸ Missing Value Support dddf2f0 936a391 c08b433

- ▸ New Algorithm Features 151cb2d 4ee3afa e2ee931 f3b1da3

- ▸ New Utility Functions & Data Access 0d37bd9 329a1cf 4e44ede 5c28a8e e760a3e ef784c8

- ▸ Estimator Configuration & Display 2b2e290 c44457a c55d064 e5ed851 e7be614 fd07977

- ▸ Scoring & Validation Ergonomics 156ef1b 59dd128 6b3f9bd 8eafd10 99916c4 99f0f69 a072e56 ba2dd5d d2f1ea7

| |
|---|

40 commits LoC 1,998

05/20/24-11/28/24

M5 — Release Quality Assurance

###### M5.1

###### M5.2

| |
|---|

10 commits LoC 1,979

07/08/24-10/27/24

###### Stability and Regression Fixes

Performance Improvements

- ▸ Metrics & Scoring Bug Fixes 69c1d79 25bc29a 60e4acd 87ceec2 7c44713 d36039a f106177 82404ba

- ▸ Estimator Convergence & Validation Fixes e796d0a eec6ef0 83da530 e5075ae 1bd9c1d 21ab5e1 938c36b 941ffbd 8d0b243

- ▸ Cross-Validation & Pipeline Fixes 102663d 142d824 2ca4eca 4dfbfb9 691b00f c0e07cf a2448b5

- ▸ Infrastructure & Integration Fixes 03d90f3 15d5a06 20c7bd0 2f8f9f3 39e1cc1 3b39d7c 3c9930b 438f093 4bc61a0 546fd5f 5879f2e 596ba5f 6595229 66b71f0 748e6f9 7baa11e 8b06fa6 94f8875 9238fe0 9732b58

- ▸ Parallel Processing Improvements 191f969 2107404

- ▸ Algorithm & Memory Optimizations 88c2db2 9590c07 eb29207

| |
|---|

21 commits LoC 989

05/14/24-11/04/24

| |
|---|

5 commits LoC 358

06/26/24-09/15/24

M6 — Project Delivery & Communication

###### M6.1

###### M6.2

Documentation and User Guidance

###### Release Engineering and CI

- ▸ Docstring Consistency & Testing Infrastruct… 019e953 08961de 09781c5 0a2ca15 0e0033a 199322d

- ▸ NumPy 2.x & Doctest Compatibility c807092 e2a42a8 94d4a33 9d39f57 a4582c0 edfb690 e825502

- ▸ API Documentation & User Guide Updates 034b0d1 09c5c00 0e68084 1d22a48 2364d8d 34db65a 45ca0a7 4c78d7c 4eb215c 57d8284 5b6622b 5f7d66c 63e1584 67597ce 6f25e64 7398114 7566d30 87d8236 938bce5 9672c2e

- ▸ Version Management & Changelog 358b3b8 3e127a3 5810fd3 6bb703f c0c46f7 cb35bd4 d0a5fdf df07b99 ea2510c

- ▸ CI & License Infrastructure 00a8ae7 0dba98f 2d6e1be 5080f25 51c8e0e 55ca335 8133eca

…and 17 more

| |
|---|

61 commits LoC 1,146

05/06/24-12/03/24

| |
|---|

16 commits LoC 1,544

05/06/24-12/03/24

…and 70 more

| |
|---|

103 commits LoC 3,210

05/05/24-12/04/24

###### DeepCommit Milestone DAG

G1 — Infrastructure Foundation (Phase 1–2) G2 — Deprecation & Validation (Phase 3–4)

###### M1.1

###### M1.2

###### M2.1

###### M2.2

License Cleanup, New Utilities & Bug Fi…

###### Metadata Routing Extension & Estimator …

Deprecation Cycle Completion & Sparse V…

Sample Weight, Verbose Controls & Compa…

###### ▸ License & Code Cleanup

- ▸ Metadata Routing 60e4acd 61281cf 64f20aa 691b00f

- ▸ Metrics & Scoring Fixes 69c1d79 6b3f9bd

- ▸ Estimator Bug Fixes 6de55b3 775587b 8132683 87ceec2 8ae3ced

- ▸ Tree & Ensemble Improvements 8133eca 7ab1b64 7baa11e 9238fe0

- ▸ API Changes & Deprecation Cleanup 77fc72c 2107404 83da530 8a5d8c3 8d01acf 8d0b243 8eafd10 938c36b 941ffbd 94d4a33 94f8875 99916c4 99f0f69 9d39f57 a072e56 a4582c0 a63b021 a67ebbe a9ec347 ab76a90

- ▸ Metrics & Metadata Routing ab7ff70 abbaed3

- ▸ Deprecation & API Cleanup af57108 afee65a b4eef25 b679062 ba2dd5d bca3634 bfa0d65

- ▸ Validation Infrastructure ba2c93b bed36b2

- ▸ Metadata Routing & Model Improvements cef803a c0e07cf

- ▸ UI & Developer Experience c44457a c55d064 c807092

- ▸ Compatibility & Cleanup cc00420 d083972 d840f36 dc1cad2 e02daf5 e2a42a8

004cf9e 00a8ae7 0338eec 2d6e1be 2f8f9f3 329a1cf 39e1cc1 3b39d7c 3b7734e 3c86573

- 3c9930b 3d5e243 45ca0a7 34db65a 48eeca6

- 4bc61a0 4e44ede 546fd5f 55ca335 5080f25

…and 5 more

- ▸ New API Features 0d37bd9 102663d

- ▸ Cross-Validation & Metadata Routing 0dba98f 142d824

- ▸ Metrics Improvements 0f27a26 156ef1b 25bc29a

- ▸ Linear Model & SVM Fixes

- 1bcddcb 1bd9c1d 1c3dcb4

▸ Estimator Validation & Bug Fixes

20c7bd0 215be2e 21ab5e1 234260d 2b2e290

- 2ca4eca

| |
|---|

| |
|---|

11 commits LoC 448

05/13/24-10/30/24

11 commits LoC 703

05/13/24-10/29/24

###### M2.3

Validation Writability, Convergence & R…

- ▸ Validation & Data Handling ef6efef eec6ef0 e796d0a e9c3942

- ▸ Metadata Routing & Estimator Enhancements e749dd9 e617d82 e5ed851

- ▸ Deprecation Cleanup e2ee931 e887583 edfb690

| |
|---|

35 commits LoC 1,208

05/13/24-11/07/24

| |
|---|

41 commits LoC 2,583

05/07/24-11/05/24

| |
|---|

10 commits LoC 391

05/17/24-11/06/24

G3 — API Surface Modernization

G4 — SLEP006 Metadata Routing Completion

###### M3.1

###### M3.2

Estimator Type Migration to Tags System

Public validate_data() API & Estimator …

###### M4.1

RFE/RFECV Metadata Routing (SLEP006)

- ▸ Add is_clusterer() Function e16a6dd

- ▸ Pipeline Fitted State Warning 4dfbfb9

- ▸ Deprecate Birch copy Parameter 7c5f13b

- ▸ IsolationForest Parallel Prediction 191f969

- ▸ KernelCenterer force_writeable Fix a490ab1

- ▸ RFE Metadata Routing 38fefe1

- ▸ RFECV Thread-Safe Routing a2448b5

| |
|---|

2 commits LoC 289

05/22/24-11/05/24

| |
|---|

2 commits LoC 262

08/13/24-10/31/24

| |
|---|

3 commits LoC 149

06/21/24-11/05/24

G6 — Quality Assurance & Performance

G5 — Backend & Data Type Extensions

###### M6.1

###### M6.2

###### M5.1

###### M5.2

Documentation Consistency & Testing Inf…

Targeted Performance Optimizations

Array API Support for Metrics & Preproc…

ExtraTree & IsolationForest Missing Val…

- ▸ Docstring Testing Infrastructure 019e953 08961de 09781c5 0a2ca15 0e0033a

- ▸ Metadata Routing Test Coverage 199322d

- ▸ Documentation & Testing Updates 1e3c7be 1fa3c75 1fd6ca9 25cb305 30cf4a0 30f7d8a 325930e 35f106c 379a2f1 3a25c38 3ca9fc1 3cda5b2 429d67a 4400917 48ef3ba 49c5948 4bdd398 4dc7dbb 5301c94 59a0b41

- ▸ LLE Memory-Efficient Sparse Construction (F… 88c2db2

- ▸ MinCovDet Partial Sort Optimization (FR2) 9590c07

- ▸ Classification Metrics Unique Label Caching… eb29207

- ▸ Array API for Regression Metrics e12f192 9f44f1f acd2d90 78102fd

- ▸ LabelEncoder Array API Support 08de29d

- ▸ cosine_similarity Array API & Device Fix e4362e5 b461547

###### ▸ ExtraTree and IsolationForest Missing Value…

dddf2f0

| |
|---|

1 commits LoC 174

07/09/24-07/09/24

| |
|---|

3 commits LoC 206

06/27/24-09/16/24

| |
|---|

7 commits LoC 304

05/07/24-06/07/24

…and 49 more

| |
|---|

75 commits LoC 655

05/06/24-11/07/24

- Figure 27 | Human-annotated and DeepCommit milestone decompositions for scikit-learn v1.5.2– v1.6.0. Top: Milestones grouped by human analyzers (M1–M6). Bottom: milestone grouped by DeepCommit (G1–G6). Each block summarizes representative commits, commit count, LoC, and time span.

###### Milestone DAG Structure Comparison Dependencylegend

Functional / strong Functional / weak Process / strong Process / weak

###### Human Milestone DAG 14 milestones | 19 dependencies

M1 — Framework Evolvability

M1.1 M1

###### M1.2 M1

Estimator Testing Infrastructure Modernization

###### Estimator Tags System Redesign

22 commits 2.8k LoC 06/27-11/21

14 commits 4.4k LoC 05/21-12/03

M6 — Project Delivery & Communication

- M1.3 M1

Unified Data Validation Public API

5 commits 1.8k LoC 06/19-09/19

- M1.4 M1

- M3 — Technical Debt Retirement
- M4 — User-Facing Capability Expansion M5 — Release Quality Assurance

Estimator & Pipeline Semantics via Metadata Routing (SLEP006)

###### M3.1 M3

###### M3.2 M3

- M6.1 M6

Documentation and User Guidance

103 commits 3.2k LoC 05/05-12/04

- M6.2 M6 Release Engineering and CI

15 commits 1.6k LoC 05/12-11/14

###### Build System Modernization

###### Deprecation Cleanup & Legacy API Removal

22 commits 2.9k LoC 05/21-10/28

40 commits 2k LoC 05/20-11/28

16 commits 1.5k LoC 05/06-12/03

###### M4.1 M4

- M5.1 M5 Stability and Regression Fixes

61 commits 1.1k LoC 05/06-12/03

- M5.2 M5 Performance Improvements

New Estimator Capabilities and Algorithms

10 commits 2k LoC 07/08-10/27

5 commits 358 LoC 06/26-09/15

M2 — Backend & Ecosystem Compatibility

###### M2.1 M2

###### M2.2 M2

###### M4.2 M4

###### Array API Support for Metrics

###### Array API for Kernels, Preprocessing & Model...

###### Usability and API Ergonomics Improvements

16 commits 673 LoC 05/07-12/01

23 commits 688 LoC 05/15-09/12

21 commits 989 LoC 05/14-11/04

###### DeepCommit Milestone DAG 12 milestones | 14 dependencies

G5 — Backend & Data Type Extensions

- G3 — API Surface Modernization
- G4 — SLEP006 Metadata Routing Completion

###### M5.1 G5

###### M3.1 G3

Array API Support for Metrics & Preprocessing

###### Estimator Type Migration to Tags System

7 commits 304 LoC 05/07-06/07

2 commits 289 LoC 05/22-11/05

###### M5.2 G5

###### M4.1 G4

###### ExtraTree & IsolationForest Missing Value Support

###### RFE/RFECV Metadata Routing (SLEP006)

1 commits 174 LoC 07/09

2 commits 262 LoC 08/13-10/31

G1 — Infrastructure Foundation (Phase 1–2)

M1.1 G1

###### M1.2 G1

Phase 1: License Cleanup, New Utilities & Bug Fixes

###### Phase 2: Metadata Routing Extension & Estimator Fixes

G2 — Deprecation & Validation (Phase 3–4)

G6 — Quality Assurance & Performance

41 commits 2.6k LoC 05/07-11/05

35 commits 1.2k LoC 05/13-11/07

###### M2.3 G2

###### M6.2 G6

###### Phase 4a: Validation Writability, Convergence &...

###### Targeted Performance Optimizations

10 commits 391 LoC 05/17-11/06

3 commits 206 LoC 06/27-09/16

###### M2.1 G2

###### M2.2 G2

###### Phase 3a: Deprecation Cycle Completion & Sparse Validation

###### Phase 3b: Sample Weight, Verbose Controls &...

11 commits 448 LoC 05/13-10/30

11 commits 703 LoC 05/13-10/29

###### M6.1 G6

###### M3.2 G3

###### Documentation Consistency & Testing Infrastructure

###### Public validate_data() API & Estimator Improvements

75 commits 655 LoC 05/06-11/07

3 commits 149 LoC 06/21-11/05

- Figure 28 | Structural comparison of Human and DeepCommit Milestone DAGs. The Human DAG contains 14 milestones and 19 dependencies; The DeepCommit DAG contains 12 milestones and 14 dependencies. Edges denote functional or process-level dependencies with strong/weak strength.

Table 4 | Structural overview of the DeepCommit Milestone DAG and the Human-annotated Milestone DAG

Dimension Human DAG DeepCommit DAG

Leaf milestones 14 12 Groups 6 6 Covered commits 373 201 DAG edges 20 14 Edge types FUNC (15) + NFR (5) FUNC (13) + NFR (1) Strong edges 2 8 Weak edges 18 6

subgraph suited for benchmark construction, but omits process-level patterns such as “code first, document later” that the Human DAG captures via NFR edges.

###### C.3.2. Clustering Divergence and Structural Differences

Although both DAGs have 6 groups, their clustering methods are fundamentally different, producing an Adjusted Rand Index (ARI) of only 0.538, indicating moderate agreement despite identical group counts.

Top-down semantics vs. bottom-up topology. The human annotator reads the release notes5 and resolved issues6 to define semantic cluster centers (e.g., “Estimator Tags System Redesign” → M1.2, “Deprecations and Removals” → M3.2), then assigns commits by developer intent. DeepCommit instead constructs a commit-level DAG from code dependencies and partitions it via seed discovery and consolidation. The resulting chain M1.1→M1.2→M2.1→M2.2→M2.3 is a contiguous topological ordering.

The contingency matrix (Figure 29) illustrates this difference from both directions. Human M5.1 (Stability Fixes, 37 shared commits) groups bug fixes spanning the entire release window by shared intent, but DeepCommit distributes them across eight milestones (M1.1:13, M1.2:12, M2.1–M2.3:7, etc.) according to which code regions they touch. Conversely, DeepCommit M1.1 (41 commits) clusters code-adjacent early-phase work into a single topological block, drawing from four distinct Human categories: stability fixes(13), deprecation cleanup(9), usability improvements(6), and release engineering(6).

Group semantics and dependency structure. The two construction methods yield different group semantics. Human groups reflect strategic release themes with high internal coherence (e.g., M1: testing → tags → validation → routing, all under “Framework Evolvability”), whereas DeepCommit groups reflect topological phases: M1 (“Framework Evolvability,” 76 commits) aggregates license cleanup, metadata routing, and estimator bug fixes—commits that are code-adjacent in time but span multiple developer-facing themes. This difference extends to edges. The Human DAG distinguishes FUNC edges (15, functional preconditions) from NFR edges (5, process-level dependencies pointing to Documentation) and annotates predominantly weak couplings (18 weak / 2 strong), expressing that most milestones can proceed in parallel. DeepCommit has 8 strong / 6 weak edges, with the main phase chain entirely strong, reflecting strict topological ordering rather than semantic coupling.

- 5https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_

highlights_1_6_0.html

- 6https://github.com/scikit-learn/scikit-learn/milestone/57?closed=1

[Figure 69]

2

- M1.1

- M1.2

- M1.3

- M1.4

- M2.1

- M2.2

- M3.1 M3.2

- M4.1 M4.2

- M5.1

M5.2 M6.1

- M6.2

70

1 1

|[Figure 70]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

60

1 1

50

1 3 1 1 2 1 1

40

4

HumanMilestones

30

1 1 3

4 1 1

Commits

20

9 5 5 1 2 1

10

1 1 1

6 5 1 2 1

13 12 2 3 2 1 1 3

1 1 3

3 3 2 1 71

0

6 1

M1.1M1.2M2.1M2.2M2.3M3.1M3.2M4.1M5.1M5.2M6.1M6.2

DeepCommit Milestones

- Figure 29 | Contingency matrix of commit assignments (Human × DeepCommit) for the 201 shared commits. The dominant Human M6.1↔DeepCommit M6.1 block (71 commits) reflects high agreement on documentation, while the dispersed pattern across DeepCommit M1.1–M2.3 reveals how phase-based partitioning fragments intent-based Human milestones.

Similarly, the Human DAG models cross-cutting workflows—five milestones connect to M6.1 via NFR edges (“code first, document later”)—while DeepCommit’s M6.1 has only 2 incoming edges, since its bottom-up construction does not capture process-level conventions.

###### C.3.3. Agreement Analysis and Summary

Despite moderate overall agreement (ARI=0.538, Normalized Mutual Information =0.444), certain regions converge strongly. We measure per-milestone overlap as the fraction of a Human milestone’s shared commits that map to a single DeepCommit milestone. Documentation shows the highest overlap: 89% of Human M6.1 commits (71 of 80) land in DeepCommit M6.1. Array API milestones also align well (Human M2.1→DeepCommit M5.1: 100%; Human M2.2: 60%), as do performance optimizations (Human M5.2→DeepCommit M6.2: 60%). These high-overlap milestones share distinctive file-modification patterns and isolated module boundaries. Conversely, intent-defined milestones show low overlap—Human M5.1 (Stability Fixes: 35%), M3.2 (Deprecation Cleanup: 39%), M1.4 (Metadata Routing: 30%)—because their commits span multiple modules and time periods, united by purpose rather than code proximity.

In summary, DeepCommit recovers milestone boundaries consistent with human annotation when technical boundaries are clear, but falls back to phase-based partitioning for cross-module, intentdefined work. The Human DAG’s structure (strategically coherent groups, FUNC/NFR edge distinction, and cross-cutting workflow modeling) captures developer reasoning that remains difficult to infer from dependency topology alone.

