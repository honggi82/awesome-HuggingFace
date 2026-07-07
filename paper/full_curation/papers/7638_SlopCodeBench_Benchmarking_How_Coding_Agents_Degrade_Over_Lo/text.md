# arXiv:2603.24755v2[cs.SE]7May2026

## SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks

Gabriel Orlanski1∗ Devjeet Roy2 Alexander Yun1 Changho Shin1 Alex Gu3

Albert Ge1 Dyah Adila1 Nicholas Roberts1 Frederic Sala1 Aws Albarghouthi1

1University of Wisconsin–Madison 2Washington State University 3MIT

### Abstract

Software development is iterative, yet agentic coding benchmarks hide design issues through their single-shot setup. Recent iterative benchmarks attempt to remedy this but heavily constrain an agent’s design decision space, making it impossible to faithfully measure how their decisions shape future extensions. We introduce SlopCodeBench, a benchmark of 36 problems and 196 checkpoints where agents repeatedly extend their own solutions. Unlike prior iterative benchmarks, our evolving specifications demand architectural decisions but leave internal structure to the agent. We measure two forms of degradation: structural erosion (concentrated complexity) and verbosity (redundant code). Evaluating 15 coding agents across open and closed models, we find that no agent fully solves any problem end-toend, and the best agent passes 14.8% of checkpoints. Quality degrades across checkpoints, with structural erosion rising in 77% of trajectories and verbosity in 75.5%. Compared to 473 open-source Python repositories, agent code is 2.3x more verbose and 2.0x more eroded, and the human repositories degrade less often and by smaller margins across their git histories. Explicit quality guidance reduces initial verbosity and erosion by up to a third, without affecting degradation rates. SlopCodeBench provides the first measurement of code degradation under iterative extension, revealing that agents pass checkpoints while producing code that erodes and bloats with each turn.

### 1 Introduction

Every design decision in software engineering is a compromise with unknown future requirements. A code search program built around regular expressions works until the specification demands structural pattern matching, at which point the entire architecture must be rewritten. Existing coding-agent benchmarks systematically undermeasure this failure mode, evaluating models once against complete task specifications [28, 38, 58, 10]. They measure whether an agent can produce correct code for the current specification, not whether that code remains extensible under future change.

Under repeated editing, agent-generated code often deteriorates in recognizable ways. LLMs favor verbose constructions over concise idioms [22, 1], and agent-generated code carries redundant or unnecessary methods that reviewers routinely strip [15, 43, 62]. The resulting low-quality, highvolume code is often colloquially called “slop.” In traditional software engineering, such accumulation is associated with higher maintenance cost and slower modification [33–35], yet pass rates can remain stable even as the underlying code becomes harder to extend. Despite this large body of work, the

∗Corresponding author. Correspondence to gorlanski@wisc.edu.

Preprint.

###### Checkpoint 1

###### Checkpoint 2

###### Checkpoint N

Search Python files; regex or exact rules.

###### Add structural pattern matching with AST support.

###### Add JS & C++ support.

###### AGENT

SEES ONLY: Spec 1

###### Empty Workspace SEES ONLY: Spec 2 Workspace 1 SEES ONLY: Spec 3 Workspace 2

###### WORKSPACE

+ def find(root, rule):

def find(root, rule): for p in walk(root, "*.py"): # exact / regex branches ...

def find(root, rule): for p in walk(root, "*.py"): # exact / regex branches ...

+ for p in walk(root, "*.py"):

+ text = read(p)

+ if rule.kind == "exact":

+ for p in walk(root, "*.js"):

+ elif rule.kind == "ast":

+ yield from exact(text, rule)

+ text = read(p)

+ for n in ts_walk(p, "py"):

+ elif rule.kind == "regex":

+ if rule.kind == "exact":

+ yield from match_ast(n, rule)

+ yield from regex(text, rule)

+ yield from exact(text, rule)

for p in walk(root, "*.js"): # exact / regex branches ...

+ elif rule.kind == "regex":

+ yield from regex(text, rule)

+ elif rule.kind == "ast":

+ for p in walk(root, "*.cpp"):

+ for n in ts_walk(p, "js"):

+ yield from match_ast(n, rule)

+ # cpp: same block, ts_walk(p,"cpp")

HIDDEN FROM AGENT

###### HIDDEN FROM AGENT

###### HIDDEN FROM AGENT

[Figure 1]

[Figure 2]

[Figure 3]

Correct 1.00 Erosion .40 Verbosity .15

C .55 E .60 V .28

C .10 E .80 V .40

- Figure 1: Iterative evaluation in SCBench. A task unfolds over checkpoints. At each step, the agent extends its prior workspace to satisfy an updated spec (orange). Correctness, erosion, and verbosity are scored externally and withheld from the agent (gray). Local shortcuts compound: correctness falls from 1.00 to 0.10 while erosion and verbosity grow.

standard agentic coding benchmarks cannot adequately measure this behavior because they focus on correctness in single-shot setups rather than on iterative evaluations.

Recent benchmarks push toward multi-turn or long-horizon coding, but none of them isolate true iterative coding. Some construct iterative tasks by decomposing monolithic solutions into dependencyordered subproblems, thereby producing a self-contained testbed rather than a realistic setting in which the agent selects an architecture and must live with it later [60]. Others derive tasks from the commit histories of mature open-source repositories [57, 20, 13]. These are valuable for studying an agent’s ability to maintain or add narrow features in an existing codebase, but they fundamentally can not test iterative coding. Using human-built workspaces and historically realized evolution paths means the agent never pays the cost of its own early design decisions. In some cases, the task formulation is also tied to test- or oracle-derived signals, which further undercuts the benchmark’s ability to measure open-ended iterative design [13]. To properly measure this, a benchmark needs four things: the agent builds on its own prior code; problems specify only external behavior, not internal interfaces; a truly hidden test suite to prevent leaking architectural hints; and black-box contracts that are implementable in any language.

We therefore introduce SlopCodeBench (SCBench), a benchmark for measuring how code quality evolves as agents repeatedly extend their own prior code under changing specifications. SCBench contains 36 problems spanning 196 checkpoints. Each checkpoint specifies only observable behavior at a CLI or API boundary, leaving internal structure unconstrained and keeping the test suite hidden. The benchmark is language-agnostic by construction: specifications describe only CLI and API behavior. We evaluate the Python track in this paper. Beyond correctness, we track two trajectory-level quality signals: verbosity, which measures the growth of redundant or duplicated code, and structural erosion, which measures the concentration of complexity in already-complex functions. The former calculates the percent of lines that are either flagged by ast-grep patterns or are structural duplicates of other code. For erosion, we calculate the percentage of the overall cyclomatic complexity [39] that is contained in high-complexity functions.

Our contributions are:

- 1. SlopCodeBench 2a language-agnostic benchmark of 36 iterative software-development problems across 196 checkpoints. The state-of-the-art performance is 14.8% with no problem fully solved.
- 2. Two trajectory-level quality metrics, verbosity and structural erosion, to measure low-quality agent code. Structural Erosion rises in 77% of trajectories and verbosity in 75.5%. 2Full problems and code are found at scbench.ai

- 3. Calibration against human code. Agent code is 2.3× more verbose and 2.0× more eroded than 473 open-source repositories. Per checkpoint, agent verbosity grows 6.6× and erosion grows 5.0× faster than that in human code.
- 4. Study on prompting techniques. Quality-aware prompts reduce initial verbosity but do not slow the degradation. Further, this results in an average 12.1% increase in the cost per checkpoint and a drop in correctness of 2.3 pp.

### 2 SlopCodeBench

SlopCodeBench contains 36 language-agnostic problems spanning 196 checkpoints. Each problem is specified solely in terms of observable behavior at a CLI or API boundary, so it can be evaluated in any implementation language. An agent implements the first specification from scratch, then repeatedly modifies and extends its own prior code as specifications evolve. Our focus is on measuring both correctness and code quality throughout that trajectory. The remainder of this section is structured as follows: 1) our desiderata, 2) how we constructed the dataset, 3) how we quantify quality issues, and 4) finally, conclude with our evaluation procedure.

#### Running example: code_search

The agent is tasked with building a CLI tool for semantic source-file search, inspired by ast-grep, across five checkpoints:

C1 - Python files only exact and regex matching. Establishes the core CLI contract. C2 - Support for Javascript and C++. C3 - AST-based pattern matching with metavariable capture. C4 - Selector rules and auto-fix functionality. C5 - Add support for Go, Rust, and Java.

An agent that hardcodes language-specific logic at C1 faces cascading rewrites at C2 and C5; one that builds an extensible parser interface does not.

#### 2.1 Design Principles

The core goal of SlopCodeBench is to evaluate an agent’s ability to make design decisions that will directly influence the quality of their code. For this, we have a core set of design principles that every problem must follow. Without these, architectural details are leaked to the agent and corrupt the signal we explicitly want to measure. They are:

- 1. No prescribed internal interfaces. Unlike existing benchmarks [13, 20], we only specify the external contract (CLI arguments or API I/O), so that the agent is forced to make architectural decisions.
- 2. No visible test suite. The dominant SWE evaluation paradigm provides fail-to-pass tests [28, 3]. SlopCodeBench agents see only specification prose and embedded examples, never the actual test suite or its feedback. They must infer unstated edge cases from the specification alone.
- 3. Black-box, language-agnostic problem design. Problems constrain only observable behavior, not implementation language or ecosystem. Following the principle that evaluation should not depend on a specific language’s ecosystem [68, 54, 10], outputs are evaluated solely via CLI or API interfaces, with normalization removing inconsequential differences in formatting and ordering. We evaluate only on Python due to cost.

In our running code_search problem, C1 specifies the CLI contract: <root_dir> –rules <file> [–encoding <name>], with output as JSON lines containing fields rule_id, file, start, end, and match. The only prescribed internals are the input/output structures that the harness needs to supply inputs and parse outputs. Specifications add normalization guidance only where arbitrary choices could cause false failures, such as key ordering, text casing, or match-span sorting. In C3, for example, an example fixes the sort order for multiple pattern matches even though the rule is not stated explicitly. This prevents penalizing inconsequential implementation choices.

#### 2.2 Task Formulation

All of our problems are written by hand, either as novel tasks or inspired by popular repositories. Each problem P is an ordered list of checkpoints [C1,...,Cn]. At Ci, the agent πθ receives only the xi and its previous workspace yi−1, then produces an updated workspace yi. At C1, the agent starts from the empty workspace y0.

- y1 = πθ(x1,y0)
- y2 = πθ(x2,y1)

(1)

... yi = πθ(xi,yi−1)

Each checkpoint is a fresh feature starting from the prior checkpoint’s workspace. The agent must reason about changes solely from the code’s current structure, as we do not provide the prior conversation’s context. A poor architectural choice at checkpoint i becomes the foundation for checkpoint i + 1, and the agent must build on it. If a reference solution replaces the agent’s code between turns, the causal chain from early decisions to later degradation is removed. In SlopCodeBench, unlike other benchmarks [60, 61, 20], the agent’s workspace is carried forward to the next checkpoint, and quality is measured at every step.

Problem Construction. Problems were authored by the paper’s authors through a two-phase process, with each problem reviewed by at least one author other than its drafter. In the proposal phase, an author drafted a problem along with its checkpoint partition; problems that did not meaningfully test design decisions, or that frontier agents could solve in a single shot, were removed from the pool. In the validation phase, we wrote the initial test suite and attempted each checkpoint with an agent, using these runs to identify and refine ambiguous or under-specified test cases. A final review pass confirmed that every problem was solvable in principle and that test suites matched the specifications. Table 4 details each problem and what design decisions are stressed.

#### 2.3 Measuring Code Quality

Standard quality models decompose software quality into broad characteristics such as maintainability, reliability, and portability [27]. We focus on two trajectory-level failure modes that are computable at every checkpoint and comparable across agents and problems. Structural erosion captures the concentration of control-flow complexity in already-complex functions. Verbosity captures duplicated and rule-flagged redundant code patterns. These metrics are computed independently of functional correctness, letting us measure structural changes that pass rates alone do not expose.

Structural erosion. Agents under iteration tend to patch new logic into existing functions rather than distributing it across focused callables. Listing 4 shows how a function in code_search at C5 can grow to 117 lines with branches for each rule kind and each match source chained inside one loop. In the iterative paradigm, the clearest notion of erosion is haphazard edits to a function that patch functionality. These edits compound slowly until they become too large to work on. Thus, we define erosion as the fraction of the codebase’s total complexity mass that resides in high-complexity functions. To this end we first assign each callable a complexity mass that accounts for both its cyclomatic complexity (CC, [39]) and its size:

mass(f) = CC(f) × SLOC(f) (2)

where CC(f) is the cyclomatic complexity of callable f and SLOC(f) is its source lines of code. The square root compresses the size factor so that complexity dominates rather than pure lines of code. Erosion is then the share of total mass held by functions exceeding a high-complexity threshold:

I[CC(f) > 10] · mass(f) f∈F mass(f)

Erosion = f∈F

(3)

where F is the set of all callables. We use a cutoff of 10 for CC following the established bounds in the popular code analysis tool Radon. In the solution to code_search shown in Listing 4, the majority of decision-points have collapsed into find_matches_in_file(), driving its mass upward even as the agent adds other functions around it.

Verbosity. The other dimension of slop is code that is too verbose, copy-pasted, or unnecessary lines that do not add anything to the overall codebase. A typical code_search fragment rebuilds the array rather than using filter, guards each iteration with empty-list checks, and assigns intermediate names used once. To capture both effects, we use a static verbosity score with two parts. First, we measure clear patterns of wasteful code generated by agents through constructing 137 targeted AST-Grep rules. These rules are based on observed cases of verbose code, best practices, and commonly cited anti-patterns on social media. Second, we measure structural duplication as clone lines normalized by LOC [30, 2]. The resulting score is

Verbosity = |{AST-Grep Flagged Lines ∪ Clone Lines}| LOC

(4)

We deduplicate lines hit by multiple AST-grep rules before counting. This score is bounded in [0,1], thus comparable across runs and independent of erosion. The two metrics measure different failure modes, so tracking both gives a fuller picture of the “slop” agents generate.

Alternative Metrics. A natural alternative is to use existing software quality metrics directly. Composite metrics such as the Maintainability Index combine traditional measures into a single maintainability score [44]. While useful as broad summaries, they are less appropriate for measuring specific ways code changes over iterations. More targeted metrics also capture only part of the phenomenon: Cyclomatic Complexity measures independent control-flow paths [39], Cognitive Complexity penalizes control-flow breaks and nesting [12], the Chidamber-Kemerer suite measures class-level object-oriented design properties [16], clone metrics capture duplication [30, 2], and LOC captures code growth. In contrast, our goal is to separate two independent failure modes: code can become more verbose without concentrating complexity, and complexity can concentrate in some functions without much overall growth. We therefore measure structural erosion and verbosity separately rather than relying on a single aggregate maintainability score or fine-grained yet incomplete single-property metrics.

#### 2.4 Evaluating Solutions

Every checkpoint’s tests interact with the solution only through subprocess or its served API. Test suites normalize outputs where needed and maintain held-out tests beyond the specification’s examples. Each test is categorized as:

- • Core — Functionality explicitly mentioned or shown in the specification.
- • Error — Failure-mode behaviors.
- • Functionality — Hidden tests that exhaustively check correctness.
- • Regression — All tests from prior checkpoints. C1 has no regression tests.

The produced workspace yi is correct if all tests pass. Because regression tests carry earlier requirements forward, a mistake at C2 can zero out later checkpoints even if later code partly works. To separate implementation quality from cascading failures, we also report correct in isolation (ISO) if yi passes all non-regression tests for Ci, and core correct (CORE) if it passes only the core tests. A problem is Partially solved if at least one checkpoint is strictly solved.

When an agent fails or crashes mid-problem, remaining checkpoints receive a correctness score of zero. Erosion and verbosity are computed only for checkpoints where the agent produced a workspace; missing checkpoints are excluded rather than imputed.

### 3 Iterative Evaluation of Coding Agents

This section reports empirical results on SlopCodeBench. The evaluation targets behaviors that arise once an agent edits its own code across multiple checkpoints, a regime that one-shot benchmarks do not exercise. Four research questions structure the experiments, each stated with its supporting result.

- RQ1: Can agents iteratively extend their own design decisions? No configuration is able to produce a solution that passes every checkpoint for any problem. Each checkpoint’s cost rises without corresponding improvements to correctness.

Table 1: Per-model performance on SlopCodeBench. For each model we report its best just-solve run, ranked by isolated solve rate with core, strict, and $/checkpoint as tiebreakers. Solve rates use a fixed 196-checkpoint denominator; missing checkpoints count as unsolved. Other metrics are only for ran checkpoints. Bold marks the best value per column. See Table 8 for the full set of runs.

Solve Rate (%) Cost & Time Quality Model Strict Iso. Core Partial $/CKPT Net $ Min/CKPT Erosion Verbosity

- Opus 4.5 9.2 17.3 57.7 25.0 2.53±1.57 492.40 7.2±3.7 0.70±0.20 0.43±0.18

- Opus 4.6 9.7 20.9 67.3 27.8 3.17±2.82 620.80 13.1±12.9 0.75±0.15 0.44±0.19

- Opus 4.7 8.2 20.9 65.8 36.1 2.17±1.83 425.34 6.8±5.0 0.76±0.17 0.48±0.21 Sonnet 4.6 7.1 16.8 57.7 27.8 1.96±1.97 383.83 14.2±14.2 0.75±0.20 0.44±0.19

- GPT 5.2 Codex 9.7 21.9 56.1 36.1 0.85±0.80 167.29 12.3±9.7 0.73±0.18 0.50±0.16

- GPT 5.3 Codex 11.2 26.0 60.7 41.7 0.66±0.49 129.65 7.7±4.6 0.64±0.19 0.46±0.18

- GPT 5.3 Spark 3.1 8.2 29.1 11.1 0.20±0.41 37.17 2.4±4.6 0.68±0.19 0.48±0.16

- GPT 5.4 10.7 23.5 62.8 36.1 0.72±0.51 141.32 8.0±4.7 0.51±0.19 0.33±0.13

- GPT 5.4 Mini 5.1 13.8 53.1 22.2 0.45±0.35 89.03 6.2±3.8 0.66±0.17 0.42±0.12

- GPT 5.5 14.8 28.1 66.8 50.0 1.51±0.81 295.59 6.0±2.4 0.49±0.20 0.32±0.12

Composer 2 6.1 16.3 52.6 13.9 0.44±0.44 86.31 8.8±8.3 0.72±0.15 0.45±0.17 GLM 5.1 9.7 13.8 40.3 30.6 1.47±1.45 288.85 31.4±20.0 0.71±0.17 0.41±0.18

- Kimi K2.5 4.6 9.7 39.8 19.4 0.33±0.25 63.21 14.5±10.0 0.72±0.20 0.44±0.20

- Kimi K2.6 10.7 18.9 51.0 33.3 0.74±0.64 133.18 26.9±20.7 0.76±0.20 0.51±0.22 MiniMax M2.7 2.0 4.1 28.1 8.3 0.34±0.23 65.79 11.9±6.8 0.73±0.16 0.47±0.17

- RQ2: Does iteration cause agent code to degrade? Structural erosion increases in 77% of trajectories while verbosity increases in 75.5% of trajectories.
- RQ3: How does agent degradation compare to repository histories? Compared to 473 Python repositories, agents produce code that is 2.0× more structurally eroded and 2.3× more verbose. Agents additionally exhibit significantly higher degradation, especially compared to pre-2024 commits.
- RQ4: Can better prompts stop this behavior? Better prompts can reduce initial structural erosion by up to 62.3% and verbosity by up to 34.8%, but do not stop iterative degradation. The average quality velocity remains 1.3 percentage points per checkpoint. Different prompting methods additionally increase the cost per checkpoint by 12.1% on average.

Setup. Each checkpoint runs in a fresh Docker container as a non-root user; only the working directory persists between checkpoints. Frontier models are trained for their provider’s harness rather than generalized agent loops, and most developers interact with agents through these CLI tools. We therefore evaluate in native harnesses rather than frameworks such as SWE-agent [65]. We evaluate 15 models across six providers in detail. Each run has a two-hour wall-clock limit, no turn or cost cap, and uses the just-solve prompt, which only instructs the agent to solve the problem and keep track of dependencies. Additional setup details are found in Appendix C.

Problems range from 3 to 8 checkpoints, so raw checkpoint indices are not directly comparable across problems. For aggregation and visualization, we map each trajectory to one of five progress phases. The first checkpoint is always Start and the last is always Final. The remaining interior checkpoints are divided into three groups: Early, Mid, and Late. All per-phase statistics in this paper use this.

#### 3.1 Solve Rates Stagnate as Cost Grows

Table 1 summarizes solve rates across 15 models on SlopCodeBench. No agent fully solves any of the 36 problems. No run passes every test at every checkpoint end-to-end. GPT 5.5 achieves the highest strict solve rate at 14.8% and isolated solve rate peaks at 28.1%. Opus 4.5 achieves the highest core solve rate at 67.3%.

- Figure 2 shows the gap between core and isolated pass rates widens from 2.5× to 5.4×. Core pass rate falls from 64.6% to 35.5%. Functionality pass rate changes by less than six points, from 62.7% to 57.7%, while error pass rate drops from 80.1% to 62.2%.

Solve Rates (%)

Cost / Checkpoint (USD)

Core Iso Strict

60

- 0

- 1

- 2

- 3

- 4

40

42pp gap

| |
|---|

20

| |
|---|

0

Start Early Mid Late Final

Start Early Mid Late Final

Figure 2: Performance degrades as problems progress and cost per checkpoint steadily rise. Left: Checkpoint solve rate by type. Right: Mean cost per checkpoint as problems progress.

Cost and context grow while code movement declines. Mean cost per checkpoint grows 2.2× from the start to end, while mean relative lines changed fall from 97.4% early to 29.5% late. Total recorded token accounting is 13.18B tokens. Normalized to 196 checkpoints, Composer 2 has the lowest token spend at 0.38B tokens, while Kimi K2.6 has the highest at 1.14B tokens.

#### 3.2 Iterative Agent Trajectories Accumulate Quality Issues

Anthropic Erosion

Anthropic Verbosity

OpenAI Erosion

OpenAI Verbosity

0.50

0.80

0.7

0.50

0.45

0.75

0.6

0.40

0.45

0.70

0.35

0.5

0.40

0.30

0.65

0.4

Start Early Mid Late Final

Start Early Mid Late Final

Start Early Mid Late Final

Start Early Mid Late Final

Sonnet 4.6 Opus 4.6 Opus 4.7 GPT 5.3 Codex GPT 5.4 GPT 5.5

#### Figure 3: Models from both frontier labs produce similar degradation trends as the problem progresses. Erosion and verbosity across problem progress for six representative models.

- Figure 3 highlights the changes in quality throughout frontier models’ trajectories on problems. Across all settings, structural erosion increases in 77% of trajectories, verbosity in 75.5%. The average number of functions with at least 10 cyclomatic complexity rises from 3.6 to 23.7 and mean maximum CC from 27.5 to 69.0. This behavior would be invisible without iterative evaluation.

Opus 4.7’s main() on forge grows 7× in CC across 8 checkpoints, from 38 to 240 lines. The C1 argparse call, shown in Listing 6, is replaced by a manual flag loop in which seven flags each get an identical pair of --X val/--X=val branches and no helper is extracted (Listing 7). The accumulation over time highlights the key failure mode.

Verbosity behaves similarly to erosion. Structural duplication grows 96% while AST-grep violation density grows only 0.3%, reflecting the rewriting of existing code rather than fresh rule violations. Iterative evaluation makes accumulation visible; small structural choices are amplified by iteration rather than absorbed.

#### 3.3 How does agent degradation compare to repository histories?

To answer this, we collect 473 Python repositories ranging from hobby projects under 100 stars to major frameworks with over 10k stars and 2M LOC, covering web frameworks, scientific computing, infrastructure, and utilities. For each repository, we randomly sample up to 30 source-modifying commits totaling 13667 commits. We treat each repository’s most recent commit at the time of

Verbosity

Erosion

0.4

0.6

0.3

0.4

| |
|---|

| |
|---|

0.2

0.2

| |
|---|

0.1

Start Early Mid Late Final

Start Early Mid Late Final

Agents Hobby ( < 100★) Niche (100-1k★) Established (1-10k★) Major ( ≥ 10k★)

- Figure 4: Agents produce lower quality code and degrade it faster than real developers. Verbosity and erosion across normalized trajectory progress, with 95% confidence intervals. GitHub-star bins over 473 repositories for a total of 13667 checkpoints. Table 2 has HEAD commit details.

- Table 2: Verbosity and erosion at HEAD commit, by GitHub-star tier and against agent checkpoints. Group n Verbosity Erosion

Hobby (< 100⋆) 114 0.21 ± 0.14 0.33 ± 0.25 Niche (100–1k⋆) 124 0.19 ± 0.11 0.32 ± 0.22 Established (1k–10k⋆) 120 0.18 ± 0.08 0.33 ± 0.20 Major (≥ 10k⋆) 115 0.19 ± 0.10 0.37 ± 0.19

Human panel (all) 473 0.19 ± 0.11 0.34 ± 0.22 Agent checkpoints 2869 0.44 ± 0.18 0.68 ± 0.20

collection as its current state for cross-sectional comparisons and the full commit sequence as analogous to an agent trajectory.

- Figure 4 shows the human panel averages 0.19 verbosity and 0.34 erosion. Agent checkpoints sit

- 2.3× and 2.0× above those means. Only 18 of 473 repositories cross the agent verbosity mean. Agent verbosity grows by 0.0144 per checkpoint and erosion by 0.0264; the human medians are 0.0022 and 0.0053. Inside the iteration loop, agents accumulate verbosity roughly 7× faster and erosion 5× faster. Erosion rises in 77% of agent trajectories against 53% of human repositories.

For the 321 repositories with at least three commits in each of the pre- and post-January-2024 windows, within-repository medians shift by +0.002 for verbosity and +0.014 for erosion; 59% of these repositories are more eroded after 2024 than before. The shift is consistent with a small uptick in human-authored slop after agent assistance became common, and it remains far below what an agent accumulates in a single checkpoint. Appendix D extends this analysis.

- 3.4 Explicit Prompting Improves Initial Quality, But Still Degrades

We next evaluate whether prompt-side modifications can mitigate the degradation induced by iteration. Two prompts are tested against the just-solve baseline. anti-slop (Listing 9) enumerates verbosity and over-engineering patterns the agent should avoid. plan-first (Listing 10) requires the agent to produce a written plan before generating code.

- Figure 5 highlights that the magnitudes of structural erosion and verbosity are significantly lowered by prompt strategies. Under anti-slop, verbosity is lower than baseline by 27.5% on GPT 5.4, 35.6% on GPT 5.5, and 32.8% on GPT 5.3 Codex. Erosion is lower by 45.1%, 57.6%, and 34.3% respectively. Only with GPT 5.4 do we observe that quality improves from start to end with the anti-slop prompt.

This improved quality comes with some trade-offs, as highlighted in Table 3. Across models, the base just-solve prompt has the best strict performance with an average drop of 2.4 pp for anti-slop and 3.6 pp for plan-first. Core pass rate improves by 0.8 pp for plan-first. Both prompts raise the cost per checkpoint by 12.1% on average. Overall, prompting strategies trade capabilities for better initial quality, with little impact on the iterative degradation.

GPT 5.3 Codex

###### GPT 5.4

###### GPT 5.5

0.5

Erosion

0.6

0.4

0.4

0.3

0.4

0.2

0.5

Verbosity

0.3

0.30

0.4

0.25

0.3

0.2

Start Early Mid Late Final

Start Early Mid Late Final

Start Early Mid Late Final

Baseline Anti-Slop Plan-First

- Figure 5: Prompting can improve initial quality, but does not remove the iterative issues. Mean structural erosion (top) and verbosity (bottom) at each progress bin, per model.

- Table 3: Raw values per prompt with the lift over just-solve in parentheses. Green means improvements while red indicates regressions.

Model Prompt Strict Iso. Core Erosion Verbosity Cost

- GPT 5.3 Codex

Just-Solve 11.2 26.0 60.7 0.64 0.46 0.66 Anti-Slop 11.2 (+0.0) 23.5 (-2.6) 57.1 (-3.6) 0.42 (-0.22) 0.31 (-0.15) 0.73 (+0.07)

- Plan-First 8.2 (-3.1) 21.4 (-4.6) 62.2 (+1.5) 0.55 (-0.10) 0.40 (-0.06) 0.68 (+0.02)

GPT 5.4

Just-Solve 10.7 23.5 62.8 0.51 0.33 0.72 Anti-Slop 9.2 (-1.5) 25.5 (+2.0) 61.2 (-1.5) 0.28 (-0.23) 0.24 (-0.09) 0.90 (+0.18)

- Plan-First 9.7 (-1.0) 25.0 (+1.5) 63.3 (+0.5) 0.39 (-0.11) 0.30 (-0.02) 0.83 (+0.11)

- GPT 5.5

Just-Solve 14.8 28.1 66.8 0.49 0.32 1.51 Anti-Slop 9.2 (-5.6) 24.0 (-4.1) 63.3 (-3.6) 0.21 (-0.28) 0.21 (-0.11) 1.68 (+0.17) Plan-First 8.2 (-6.6) 24.0 (-4.1) 67.3 (+0.5) 0.29 (-0.21) 0.25 (-0.06) 1.61 (+0.11)

- 4 Related Work

Single-shot benchmarks. Repository-level evaluation against a fixed specification began with Jimenez et al. [28] and has broadened across language, domain, and task complexity [68, 54, 25, 36, 73]. Pass rate hides defect categories that tests do not exercise [66, 76].

Iterative and evolutionary benchmarks. Prior iterative work severs accumulation through per-step scoring, clean prior-turn states, isolated modifications, or unchained long-horizon framing [70, 41, 60, 61, 74, 57]; Chen et al. [13] preserves history but draws from public repositories and inherits contamination risk. Deng et al. [20] chains agent code across 98 milestones with pass rates falling from 80% to 38%, yet evaluation stays pass/fail, and SLUMP [64] measures only final-artifact compliance. Verbosity and erosion require the agent’s own output to accumulate across many steps.

Quality degradation and code smells. LLM code carries higher per-line cyclomatic complexity than human code, with redundant steps, duplication, and unnecessary conditionals its most prevalent smells [22, 1, 17]. Under repeated modification, outputs drift toward structural attractors, quality diverges across trajectories, and refinement introduces defects that tests miss [53, 15, 55, 14, 23, 11, 71, 29, 56, 43, 9]. Our metrics quantify these dynamics per checkpoint.

- 5 Conclusion

We present SlopCodeBench, an iterative benchmark designed to measure the quality of code produced by agents when they are forced to make design decisions. We follow explicit construction principles to create black-box problems with only external contracts described. To measure quality, we introduce structural erosion, complexity concentration, and verbosity, the fraction of code that is redundant. No agent is able to solve any of our 36 problems end-to-end, while the state-of-the-art solve rate is only 14.8% of 196 checkpoints. When forced to iterate, agents consistently degrade the codebase’s

quality. In our maintained-repository calibration panel, agent checkpoints exhibit higher erosion and verbosity, and agent trajectories accumulate both metrics faster than those of sampled Python repository histories. Our analysis of prompting interventions finds that they can improve initial quality issues, but cannot stop the eroding behavior. SlopCodeBench is a first-of-its-kind iterative evaluation that allows proper investigation of how agents act when given decision autonomy.

### Acknowledgments and Disclosure of Funding

We thank Abtin Molavi, Amanda Xu, June Cho, Xavier Garcia, and Samuel Guo for feedback during the development of this project. We also thank the Terminal Bench team for their inspiration and feedback.

This work is supported by Snorkel AI through the Open Benchmarks Grants program. This work was additionally supported in part by DARPA and the NSF.

### References

- [1] Altaf Allah Abbassi, Leuson Da Silva, Amin Nikanjam, and Foutse Khomh. A taxonomy of inefficiencies in LLM-generated Python code, 2025. URL https://arxiv.org/abs/ 2503.06327.
- [2] Ajmain I. Alam, Palash R. Roy, Farouq Al-Omari, Chanchal K. Roy, Banani Roy, and Kevin A. Schneider. Gptclonebench: A comprehensive benchmark of semantic clones and cross-language clones using gpt-3 model and semanticclonebench. In 2023 IEEE International Conference on Software Maintenance and Evolution (ICSME), pages 1–13, 2023. doi: 10.1109/ICSME58846. 2023.00013.
- [3] Reem Aleithan, Haoran Xue, Mohammad Mahdi Mohajer, Elijah Nnorom, Gias Uddin, and Song Wang. Swe-bench+: Enhanced coding benchmark for llms. arXiv preprint arXiv:2410.06992, 2024.
- [4] Anthropic. Claude code, 2025. URL https://docs.anthropic.com/en/docs/ claude-code. CLI tool for agentic coding with Claude.
- [5] Anthropic. Claude opus 4.5 system card, 2025. URL https://assets.anthropic.com/ m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf.
- [6] Anthropic. Claude opus 4.6 system card, 2026. URL https://anthropic.com/

- claude-opus-4-6-system-card.

[7] Anthropic. Claude opus 4.7 system card, 2026. URL https://anthropic.com/

- claude-opus-4-7-system-card.

- [8] Anthropic. Claude sonnet 4.6 system card, 2026. URL https://anthropic.com/ claude-sonnet-4-6-system-card.
- [9] Syed Ammar Asdaque, Imran Haider, Muhammad Umar Malik, Maryam Abdul Ghafoor, and Abdul Ali Bangash. Novice developers produce larger review overhead for project maintainers while vibe coding, 2026. URL https://arxiv.org/abs/2602.23905.
- [10] Ibragim Badertdinov, Maksim Nekrashevich, Anton Shevtsov, and Alexander Golubev. Swerebench v2: Language-agnostic swe task collection at scale, 2026. URL https://arxiv. org/abs/2602.23866.
- [11] Jeremiah Bohr. Show and tell: Prompt strategies for style control in multi-turn LLM code generation, 2025. URL https://arxiv.org/abs/2511.13972.
- [12] G. Ann Campbell. Cognitive complexity — an overview and evaluation. In 2018 IEEE/ACM International Conference on Technical Debt (TechDebt), pages 57–58, 2018.
- [13] Jialong Chen, Xander Xu, Hu Wei, Chuan Chen, and Bing Zhao. Swe-ci: Evaluating agent capabilities in maintaining codebases via continuous integration, 2026. URL https://arxiv. org/abs/2603.03823.

- [14] Yi Chen, Yun Bian, Haiquan Wang, Shihao Li, and Zhe Cui. Scaffold-cegis: Preventing latent security degradation in llm-driven iterative code refinement, 2026. URL https://arxiv. org/abs/2603.08520.
- [15] Zhi Chen and Lingxiao Jiang. Evaluating software development agents: Patch patterns, code quality, and issue complexity in real-world github scenarios. In 2025 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), pages 657–668, 2025. doi: 10.1109/SANER64311.2025.00068.
- [16] S.R. Chidamber and C.F. Kemerer. A metrics suite for object oriented design. IEEE Transactions on Software Engineering, 20(6):476–493, 1994. doi: 10.1109/32.295895.
- [17] Domenico Cotroneo, Cristina Improta, and Pietro Liguori. Human-written vs. AI-generated code: A large-scale study of defects, vulnerabilities, and complexity, 2025. URL https: //arxiv.org/abs/2508.21634.
- [18] Ward Cunningham. The wycash portfolio management system. volume 4, page 29–30, New York, NY, USA, December 1992. Association for Computing Machinery. doi: 10.1145/157710.

157715. URL https://doi.org/10.1145/157710.157715.

- [19] Cursor. Introducing composer 2, 2026. URL https://cursor.com/blog/composer-2.
- [20] Gangda Deng, Zhaoling Chen, Zhongming Yu, Haoyang Fan, Yuhong Liu, Yuxin Yang, Dhruv Parikh, Rajgopal Kannan, Le Cong, Mengdi Wang, Qian Zhang, Viktor Prasanna, Xiangru Tang, and Xingyao Wang. Evoclaw: Evaluating ai agents on continuous software evolution, 2026. URL https://arxiv.org/abs/2603.13428.
- [21] Jingzhe Ding, Shengda Long, Changxin Pu, Huan Zhou, Hongwan Gao, Xiang Gao, Chao He, Yue Hou, Fei Hu, Zhaojian Li, Weiran Shi, Zaiyuan Wang, Daoguang Zan, Chenchen Zhang, Xiaoxu Zhang, Qizhi Chen, Xianfu Cheng, Bo Deng, Qingshui Gu, Kai Hua, Juntao Lin, Pai Liu, Mingchen Li, Xuanguang Pan, Zifan Peng, Yujia Qin, Yong Shan, Zhewen Tan, Weihao Xie, Zihan Wang, Yishuo Yuan, Jiayu Zhang, Enduo Zhao, Yunfei Zhao, He Zhu, Liya Zhu, Chenyang Zou, Ming Ding, Jianpeng Jiao, Jiaheng Liu, Minghao Liu, Qian Liu, Chongyang Tao, Jian Yang, Tong Yang, Zhaoxiang Zhang, Xinjie Chen, Wenhao Huang, and Ge Zhang. Nl2repo-bench: Towards long-horizon repository generation evaluation of coding agents, 2026. URL https://arxiv.org/abs/2512.12730.
- [22] Shihan Dou, Haoxiang Jia, Shenxi Wu, Huiyuan Zheng, Muling Wu, Yunbo Tao, Ming Zhang, Mingxu Chai, Jessica Fan, Zhiheng Xi, Rui Zheng, Yueming Wu, Ming Wen, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. What’s wrong with your code generated by large language models? an extensive study. Science China Information Sciences, 2026. doi: 10.1007/ s11432-025-4632-8. URL https://arxiv.org/abs/2407.06153.
- [23] Simantika Bhattacharjee Dristi and Matthew B. Dwyer. A differential fuzzing-based evaluation of functional equivalence in llm-generated code refactorings, 2026. URL https://arxiv. org/abs/2602.15761.
- [24] Guoliang Duan, Mingwei Liu, Yanlin Wang, Chong Wang, Xin Peng, and Zibin Zheng. A hierarchical and evolvable benchmark for fine-grained code instruction following with multi-turn feedback, 2025. URL https://arxiv.org/abs/2507.00699.
- [25] Yukang Feng, Jianwen Sun, Zelai Yang, Jiaxin Ai, Chuanhao Li, Zizhen Li, Fanrui Zhang, Kang He, Rui Ma, Jifan Lin, Jie Sun, Yang Xiao, Sizhuo Zhou, Wenxiao Wu, Yiming Liu, Pengfei Liu, Yu Qiao, Shenglin Zhang, and Kaipeng Zhang. LongCLI-Bench: A preliminary benchmark and study for long-horizon agentic programming in command-line interfaces, 2026.
- [26] Martin Fowler and Kent Beck. Refactoring: Improving the Design of Existing Code. AddisonWesley, 1999. ISBN 978-0201485677.
- [27] ISO. Iso / iec 25010 : 2011 systems and software engineering — systems and software quality requirements and evaluation ( square ) — system and software quality models, 2013. URL https://api.semanticscholar.org/CorpusID:19022091.

- [28] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=VTF8yNQM66.
- [29] Haolin Jin and Huaming Chen. Are llms reliable code reviewers? systematic overcorrection in requirement conformance judgement, 2026. URL https://arxiv.org/abs/2603. 00539.
- [30] Elmar Juergens, Florian Deissenboeck, Benjamin Hummel, and Stefan Wagner. Do code clones matter? In Proceedings of the 31st International Conference on Software Engineering, ICSE ’09, page 485–495, USA, 2009. IEEE Computer Society. ISBN 9781424434534. doi: 10.1109/ ICSE.2009.5070547. URL https://doi.org/10.1109/ICSE.2009.5070547.
- [31] Kimi Team. Kimi k2.5: Visual agentic intelligence. 2026. URL https://arxiv.org/abs/ 2602.02276.
- [32] Kimi Team. Kimi K2.6, 2026. URL https://moonshotai.com/kimi-k2.6.
- [33] Guilherme Lacerda, Fabio Petrillo, Marcelo Pimenta, and Yann Gaël Guéhéneuc. Code smells and refactoring: A tertiary systematic review of challenges and observations. Journal of Systems and Software, 167:110610, 2020. ISSN 0164-1212. doi: https://doi.org/10.1016/j. jss.2020.110610. URL https://www.sciencedirect.com/science/article/pii/ S0164121220300881.
- [34] Duc Minh Le, Suhrid Karthik, Marcelo Schmitt Laser, and Nenad Medvidovic. Architectural decay as predictor of issue- and change-proneness. In 2021 IEEE 18th International Conference on Software Architecture (ICSA), pages 92–103, 2021. doi: 10.1109/ICSA51549.2021.00017.
- [35] Ruiyin Li, Peng Liang, Mohamed Soliman, and Paris Avgeriou. Understanding software architecture erosion: A systematic mapping study. Journal of Software: Evolution and Process, 34(3):e2423, 2022. doi: https://doi.org/10.1002/smr.2423. URL https://onlinelibrary. wiley.com/doi/abs/10.1002/smr.2423.
- [36] Wei Li, Xin Zhang, Zhongxin Guo, Shaoguang Mao, Wen Luo, Guangyue Peng, Yangyu Huang, Houfeng Wang, and Scarlett Li. Fea-bench: A benchmark for evaluating repository-level code generation for feature implementation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025. doi: 10.48550/arXiv. 2503.06680.
- [37] Kaiyuan Liu, Youcheng Pan, Yang Xiang, Daojing He, Jing Li, Yexing Du, and Tianrun Gao. Projecteval: A benchmark for programming agents automated evaluation on project-level code generation, 2025.
- [38] Pengrui Lu, Shiqi Zhang, Yunzhong Hou, Lyumanshan Ye, Chaoyi Huang, Zixi Chen, Ji Zeng, Hantao Jiang, Pengfei Liu, Yiwei Wang, and Ming-Hsuan Yang. Projdevbench: Benchmarking ai coding agents on end-to-end project development, 2026. URL https://arxiv.org/ abs/2602.01655.
- [39] T.J. McCabe. A complexity measure. IEEE Transactions on Software Engineering, SE-2(4): 308–320, 1976. doi: 10.1109/TSE.1976.233837.
- [40] Mike A Merrill, Alexander Glenn Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Kumar Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Kwesi Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei

- Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Jenia Jitsev, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=a7Qa4CcHak.
- [41] Chunyu Miao, Henry Peng Zou, Yangning Li, Yankai Chen, Yibo Wang, Fangxin Wang, Yifan Li, Wooseong Yang, Bowei He, Xinni Zhang, Dianzhi Yu, Hanchen Yang, Hoang H. Nguyen, Yue Zhou, Jie Yang, Jizhou Guo, Wenzhe Fan, Chin-Yuan Yeh, Panpan Meng, Liancheng Fang, Jinhu Qi, Wei-Chieh Huang, Zhengyao Gu, Yuwei Han, Langzhou He, Yuyao Yang, Yinghui Li, Hai-Tao Zheng, Xue Liu, Irwin King, and Philip S. Yu. Recode-h: A benchmark for research code development with interactive human feedback, 2025.
- [42] MiniMax. MiniMax M2.7: Early echoes of self-evolution, 2026. URL https://www. minimax.io/news/minimax-m27-en.
- [43] Sota Nakashima, Yuta Ishimoto, Masanari Kondo, Shane McIntosh, and Yasutaka Kamei. Why agentic-PRs get rejected: A comparative study of coding agents, 2026.
- [44] P. Oman and J. Hagemeister. Metrics for assessing a software system’s maintainability. In Proceedings Conference on Software Maintenance 1992, pages 337–344, 1992. doi: 10.1109/ ICSM.1992.242525.
- [45] OpenAI. Codex CLI, 2025. URL https://github.com/openai/codex. CLI tool for agentic coding with OpenAI models.
- [46] OpenAI. Addendum to GPT-5.2 system card: GPT-5.2-Codex, 2025. URL https://openai. com/index/gpt-5-2-codex-system-card/.
- [47] OpenAI. GPT-5.3-Codex system card, 2026. URL https://openai.com/index/ gpt-5-3-codex-system-card/.
- [48] OpenAI. Introducing GPT-5.3-Codex-Spark, 2026. URL https://openai.com/index/ introducing-gpt-5-3-codex-spark/. Research preview; safety coverage under the GPT-5.3-Codex System Card.
- [49] OpenAI. GPT-5.4 thinking system card, 2026. URL https://openai.com/index/ gpt-5-4-thinking-system-card/.
- [50] OpenAI. GPT-5.4 thinking system card, 2026. URL https://openai.com/index/ gpt-5-4-thinking-system-card/. Includes March 17, 2026 addendum covering GPT5.4 mini and nano.
- [51] OpenAI. GPT-5.5 system card, 2026. URL https://openai.com/index/ gpt-5-5-system-card/.
- [52] D.L. Parnas. Software aging. In Proceedings of 16th International Conference on Software Engineering, pages 279–287, 1994. doi: 10.1109/ICSE.1994.296790.
- [53] Norman Peitek, Julia Hess, and Sven Apel. From restructuring to stabilization: A large-scale experiment on iterative code readability refactoring with large language models, 2026. URL https://arxiv.org/abs/2602.21833.
- [54] Muhammad Shihab Rashid, Christian Bock, Yuan Zhuang, Alexander Buchholz, Tim Esler, Simon Valentin, Luca Franceschi, Martin Wistuba, Prabhu Teja Sivaprasad, Woo Jung Kim, Anoop Deoras, Giovanni Zappella, and Laurent Callot. Swe-polybench: A multi-language benchmark for repository level evaluation of coding agents, 2025. URL https://arxiv. org/abs/2504.08703.
- [55] César Santos, Ermeson Andrade, and Roberto Natella. Investigating software aging in llmgenerated software systems. In 2025 IEEE 36th International Symposium on Software Reliability Engineering Workshops (ISSREW), page 314–321. IEEE, October 2025. doi: 10.1109/issrew67781.2025.00090. URL http://dx.doi.org/10.1109/ISSREW67781. 2025.00090.

- [56] Song Tae-Eun. More rounds, more noise: Why multi-turn review fails to improve cross-context verification, 2026. URL https://arxiv.org/abs/2603.16244.
- [57] Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, and Nghi D. Q. Bui. Sweevo: Benchmarking coding agents in long-horizon software evolution scenarios, 2026. URL https://arxiv.org/abs/2512.18470.
- [58] Hung Tran, Langston Nashold, Rayan Krishnan, Antoine Bigeard, and Alex Gu. Vibe code bench: Evaluating ai models on end-to-end web application development. 2026. URL https: //api.semanticscholar.org/CorpusID:286256067.
- [59] Peiding Wang, Li Zhang, Fang Liu, Lin Shi, Minxiao Li, Bo Shen, and An Fu. Codeif-bench: Evaluating instruction-following capabilities of large language models in interactive code

- generation, 2025. URL https://arxiv.org/abs/2503.22688.

[60] Sizhe Wang, Zhengren Wang, Dongsheng Ma, Yongan Yu, Rui Ling, Zhiyu Li, Feiyu Xiong, and Wentao Zhang. Codeflowbench: A multi-turn, iterative benchmark for complex code

- generation, 2026. URL https://arxiv.org/abs/2504.21751.

- [61] Zhengren Wang, Rui ling, Chufan Wang, Yongan Yu, Sizhe Wang, Zhiyu li, Feiyu Xiong, and Wentao Zhang. Maintaincoder: Maintainable code generation under dynamic requirements. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=TIbQixHEFD.
- [62] Kan Watanabe, Tatsuya Shirai, Yutaro Kashiwa, and Hajimu Iida. What to cut? predicting unnecessary methods in agentic code generation, 2026. URL https://arxiv.org/abs/ 2602.17091.
- [63] Jiahong Xiang, Wenxiao He, Xihua Wang, Hongliang Tian, and Yuqun Zhang. Evaluating and improving automated repository-level rust issue resolution with llm-based agents, 2026. URL https://api.semanticscholar.org/CorpusID:286083036.
- [64] Lu Yan, Xuan Chen, and Xiangyu Zhang. When the specification emerges: Benchmarking faithfulness loss in long-horizon coding agents, 2026. URL https://arxiv.org/abs/ 2603.17104.
- [65] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=mXpq6ut8J3.
- [66] Boxi Yu, Yang Cao, Yuzhong Zhang, Liting Lin, Junjielong Xu, Zhiqing Zhong, Qinghua Xu, Guancheng Wang, Jialun Cao, Shing-Chi Cheung, Pinjia He, and Lionel Briand. Swe-abs: Adversarial benchmark strengthening exposes inflated success rates on test-based benchmark,

2026. URL https://arxiv.org/abs/2603.00520.

- [67] Z.ai. GLM-5.1: Towards long-horizon tasks, 2026. URL https://z.ai/blog/glm-5.1.
- [68] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Shulin Xin, Linhao Zhang, Qi Liu, Aoyan Li, Lu Chen, Xiaojian Zhong, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, RUI LONG, Ming Ding, and liang xiang. Multi-SWE-bench: A multilingual benchmark for issue resolving. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https: //openreview.net/forum?id=MhBZzkz4h9.
- [69] Zhengran Zeng, Yixin Li, Rui Xie, Wei Ye, and Shikun Zhang. Benchmarking and studying the llm-based agent system in end-to-end software development, 2025. URL https://arxiv. org/abs/2511.04064.
- [70] Zexun Zhan, Shuzheng Gao, Ruida Hu, and Cuiyun Gao. Sr-eval: Evaluating llms on code generation under stepwise requirement refinement, 2026. URL https://arxiv.org/abs/ 2509.18808.

- [71] Binquan Zhang, Li Zhang, Lin Shi, Song Wang, Yuwei Qian, Linhui Zhao, Fang Liu, An Fu, and Yida Ye. An empirical study of interaction smells in multi-turn human-llm collaborative code generation, 2026. URL https://arxiv.org/abs/2603.09701.
- [72] Zhirui Zhang, Hongbo Zhang, Haoxiang Fei, Zhiyuan Bao, Yubin Chen, Zhengyu Lei, Ziyue Liu, Yixuan Sun, Mingkun Xiao, Zihang Ye, Yu Zhang, Hongcheng Zhu, Yuxiang Wen, and HeungYeung Shum. Swe-agi: Benchmarking specification-driven software construction with moonbit in the era of autonomous agents, 2026. URL https://arxiv.org/abs/2602.09447.
- [73] Wenting Zhao, Nan Jiang, Celine Lee, Justin T Chiu, Claire Cardie, Matthias Gallé, and Alexander M Rush. Commit0: Library generation from scratch. 2024. URL https://arxiv. org/abs/2412.01769.
- [74] Dewu Zheng, Yanlin Wang, Ensheng Shi, Ruikai Zhang, Yuchi Ma, Hongyu Zhang, and Zibin Zheng. Humanevo: An evolution-aware benchmark for more realistic evaluation of repository-level code generation. 2025. URL https://arxiv.org/abs/2406.06918.
- [75] Qixing Zhou, JiaCheng Zhang, Haiyang Wang, Rui Hao, Jiahe Wang, Minghao Han, Yuxue Yang, Shuzhe Wu, Feiyang Pan, Lue Fan, Dandan Tu, and Zhaoxiang Zhang. Featurebench: Benchmarking agentic coding for complex feature development. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=41xrZ3uGuI.
- [76] Xin Zhou, Martin Weyssow, Ratnadira Widyasari, Ting Zhang, Junda He, Yunbo Lyu, Jianming Chang, Beiqi Zhang, Dan Huang, and David Lo. Lessleak-bench: A first investigation of data leakage in llms across 83 software engineering benchmarks, 2025. URL https://arxiv. org/abs/2502.06215.

### A Benchmark Details

#### A.1 Code Search Checkpoint Specifications

This subsection provides the complete specifications for the first three checkpoints of the code_search problem referenced in section 2. Each specification appears exactly as presented to agents during evaluation.

#### A.1.1 Checkpoint 1: Python-only Exact and Regex Matching

Build a command-line code searcher for Python codebases. It takes a directory of source files and a set of rules, then prints one JSON object per match (JSON Lines) to STDOUT.

Supported language (by file extension only): Python (.py). Supported rule types (exactly these two): exact match and vanilla regex.

# Deliverable

Write an executable that accepts:

<entry_command> <root_dir> --rules <rules_file> [--encoding <name>]

- * <root_dir>: path to the codebase to scan.

- * --rules <rules_file>: path to a JSON array of rules (see schema).

- * --encoding <name>: optional; default utf-8. Files that fail to decode are skipped.

Output: JSON Lines to STDOUT, one object per match (schema below). On success (even if zero matches): exit code 0.

## Inputs

### File type -> language

Only files with extension .py are scanned. All other files are ignored.

### Rules file (JSON array)

Each rule is an object with this schema:

{

"id": "<non-empty string>", "kind": "exact" | "regex", "pattern": "<non-empty string>", "languages": ["python"], // optional; default: ["python"] "regex_flags": ["i", "m", "s"] // optional; only for kind = "regex"

}

Constraints on types/inputs

- * id: unique across the rules array.

- * pattern: valid UTF-8 string. For kind="regex", it must compile with provided regex_flags (subset of i case-insensitive, m multiline, s dotall). No other flags allowed.

- * languages (when present) must be an array of strings and may only contain "python".

- * Missing optional fields use their defaults. ### Matching semantics

Matches inside comments/strings are allowed.

## Output (JSON Lines to STDOUT)

For each match, print a single JSON object with exactly:

{

"rule_id": "<string>", "file": "<posix path>", // path relative to root_dir, ’/’

separators "language": "python", "start": {"line": <int>, "col": <int>}, // 1-based line/column. "end": {"line": <int>, "col": <int>}, // position immediately AFTER the match "match": "<string>"

}

No extra fields. Each object on its own line with a trailing newline. No other STDOUT output.

## Normalization

- 1. Ordering: Matches must appear by file (lexicographically), then start.line, then start.col, then rule_id.

- 2. Coordinates: Lines and columns are 1-based.

- 3. Path format: file is relative to <root_dir> with ’/’ separators.

- 4. Encoding: Read files using --encoding (default utf-8); skip files that fail to decode.

Listing 1: Specification for code_search checkpoint 1

#### A.1.2 Checkpoint 2: Multi-language Support with Filtering

Extend your code searcher to support JavaScript and C++ source files.

---

## New Requirements

### File type -> language

Scan these extensions:

| Language | Extensions | | ---------- | -------------------------------------------- | | Python | .py | | JavaScript | .js, .mjs, .cjs | | C++ | .cc, .cpp, .cxx, .hh, .hpp, .hxx |

### Rule language filtering

Rules may specify "languages" from this set: ["python", "javascript", "cpp"]. If omitted, the rule applies to all three.

---

## Example

### rules.json

[

{"id":"todo","kind":"exact","pattern":"TODO:"}, {"id":"printf","kind":"regex","pattern":"\\bprintf\\s*\\(","languages":["cpp"]}, {"id":"console-log","kind":"regex","pattern":"console\\.log\\s*\\(","languages":["

javascript"]} ]

### Project tree

repo/ main.py app.js src/

engine.cpp

### Run

$ <entry_command> repo --rules rules.json

### Output

{"rule_id":"console-log","file":"app.js","language":"javascript","start":{"line":5," col":3},"end":{"line":5,"col":15},"match":"console.log("} {"rule_id":"printf","file":"src/engine.cpp","language":"cpp","start":{"line":42,"col ":3},"end":{"line":42,"col":10},"match":"printf("} {"rule_id":"todo","file":"main.py","language":"python","start":{"line":1,"col":1}," end":{"line":1,"col":6},"match":"TODO:"}

Listing 2: Specification for code_search checkpoint 2

#### A.1.3 Checkpoint 3: Structure-Aware Metavariable Patterns

Extend your code searcher to support structure-aware patterns with metavariables.

Supported rule kinds: exact, regex, and pattern.

# Deliverable

Your existing executable is extended to understand kind: "pattern" in the rules file:

<entry_command> <root_dir> --rules <rules_file> [--encoding <name>]

Output: JSON Lines (one object per match).

## Pattern Rules

### Rule schema additions

{

"id": "<non-empty string>", "kind": "pattern", "pattern": "<code-like string with metavariables>", "languages": ["cpp" | "javascript" | "python"], // optional; default: all 3

}

### Metavariables

- * Any token of the form $NAME (e.g., $X, $GREETING) is a metavariable.

- * Metavariables ending in ? (e.g., $X?) are optional -- the pattern matches even if they are not present.

- * A metavariable matches a single code element appropriate for that position.

- * If the same metavariable name appears multiple times in the pattern, all occurrences must match the same text.

- * $$ in the pattern matches a literal $ in the source code. ### Pattern string

- * The pattern must be valid code in the target language (with metavariables treated as valid placeholders).

- * No wildcards beyond metavariables are required (no ellipsis or quantifiers).

### Matching semantics

Find all matches in source files where the rule’s language applies.

## Output (JSON Lines to STDOUT)

For kind: "pattern", the JSON object per match is:

{

"rule_id": "<string>", "file": "<posix path>", "language": "cpp" | "javascript" | "python", "start": {"line": <int>, "col": <int>}, "end": {"line": <int>, "col": <int>}, "match": "<string>", "captures": {

"$NAME": { "text": "<matched source text>", "ranges": [

{"start": {"line": <int>, "col": <int>}, "end": {"line": <int>, "col": <int >}}

]

} // ... one entry per metavariable bound in this match

} }

- * ranges lists every occurrence of that metavariable within the matched region (use the same 1-based, Unicode column coordinates).

- * If a metavariable appears once, ranges has a single range.

## Normalization

- 1. Pattern determinism: When multiple matches share the same start position, sort by end position (earlier end first), then by rule_id.

- 2. Captures key order: Serialize captures with keys sorted lexicographically by metavariable name (e.g., $A before $X).

Listing 3: Specification for code_search checkpoint 3

- Table 4: SlopCodeBench problems and the design decisions stressed by iterative checkpoints, part 1 of 2. CPs gives the number of checkpoints.

Problem CPs Design decisions stressed cfgpipe 6 Source-priority resolution, typed parsing boundaries, nested schema representation, watch and

event architecture, redaction, and store-prefix composition. circuit_eval 8 Circuit IR, parser separation across formats, two-valued versus three-valued logic, optimization pass pipeline, and equivalence checking.

code_search 5 Regex versus AST matching, metavariable representation, rewrite conflict handling, selector semantics, and language adapter boundaries.

dag_execution 3 DAG DSL parsing, task parameter model, execution ordering, cache key design, and per-task cache overrides. database_migration 5 Migration operation model, rollback representation, dependency sorting, cycle errors, and constraint or index handling.

datagate 7 Dataset ingestion abstraction, query and filter pipeline, pagination and export contracts, cache invalidation, runtime config, and access control.

dynamic_buffer 4 Example-to-transform inference, generated API shape, stateful transform classes, streaming buffers, and cache behavior. dynamic_config_service_api 4 Versioned config storage, inheritance and deep merge rules, schema validation, approval workflow state, and policy enforcement. env_manager 5 Declarative provisioning IR, OS-specific planning boundaries, deterministic output, module validation, profiles, and build-script generation. etl_pipeline 5 Pipeline IR, expression language design, branching, reusable sub-pipeline parameters, and library loading. eve_industry 6 SDE normalization, recipe graph representation, invention probability, ME and TE accounting, job scheduling, and recursive build planning. eve_jump_planner 3 Route graph modeling, fuel and fatigue costs, spatial distance calculations, cloak mechanics, and high-sec fallback routing. eve_market_tools 4 Market order ingestion, price-book aggregation, reprocessing yield math, hub optimization, and arbitrage or hauling profit ranking. eve_route_planner 3 Warp physics, route cost calculation, cargo manifest planning, multi-trip scheduling, and contract selection under constraints. execution_server 6 Command execution boundaries, file tracking, output format abstraction, chain and hook semantics, cache keys, persistent environment concurrency, and job queue DAGs. file_backup 4 Schedule schema, backup strategy abstraction, glob exclusion semantics, verification, incremental hashing, and destination layout. file_merger 4 Multi-format reader abstraction, schema alignment, external sort strategy, partition and shard layout, and nested type flattening. file_query_tool 5 SQL parsing and execution planning, file table abstraction, joins, aggregation, window state, CTEs, and subquery scope.

### B Quality and Erosion Analysis

#### B.1 Code Quality Examples

This subsection reproduces the two code_search code fragments referenced in subsection 2.3. Listing 4 illustrates structural erosion, and Listing 5 illustrates verbosity.

#### B.2 Erosion Example: forge/main Across Eight Checkpoints

On forge, Opus 4.7’s main() grows 7× in cyclomatic complexity over 8 checkpoints, from CC 13 to CC 92, and from 38 to 240 lines. The increase is monotonic across every checkpoint. Listing 6 shows the C1 implementation, a six-line argparse setup followed by a small dispatch ladder. Listing 7 shows the C8 implementation, in which argparse has been removed and replaced with a manual while-loop that handles each flag in two parallel branches, one for the --X val form and one for --X=val. This pair of branches is repeated for seven flags before control passes to a nested resource-by-command dispatch ladder. No shared helper is introduced across the seven flag pairs, despite the per-flag template being uniform.

#### B.3 Erosion Sensitivity

We vary the high-CC cutoff (8, 10, 12) and the size term (no size term, √SLOC, linear SLOC) around the reported erosion family. Across all nine variants, the result is stable: predictive correlation with next-checkpoint pass rate stays near zero, while predictive correlation with next-checkpoint cost stays positive.

- Table 5: SlopCodeBench problems and the design decisions stressed by iterative checkpoints, part 2 of 2.

Problem CPs Design decisions stressed forge 8 Resource models for blueprints, allocations, units, modules, revision gates, strict JSON contracts,

and admin access boundaries.

l2m 5 LaTeX parsing depth, macro expansion, environment conversion rules, CLI flags, and KaTeX-compatible Markdown output.

layered_config_synthesizer 4 Layer merge semantics, conflict resolution, fragment expansion, environment interpolation, multi-run manifests, and JSON Schema validation. log_query 5 SQL-like grammar design, NDJSON streaming execution, aggregation and joins, schema label mapping, and subquery scope. meshctl 8 Declarative resource specs, persisted state shape, lifecycle and topology operations, security model, migrations, network exposure, and multi-region policy. metric_transform_lang 5 DSL parser and interpreter design, stream processing, aggregation and window state, temporal joins, and resumable execution. migrate_configs 5 Rule engine design, multi-format parsing, inheritance and cycle detection, array pattern matching, relocation rules, and validation order. mocked_http 8 Mock schema design, request matching, JSONPath evaluation, environment interpolation,

validation strategy, and admin runtime behavior. mvvault 6 Versioned catalog schema handling, field-level history, sync and download boundaries, viewer

routing, annotations, and auto-migration triggers. pwd_manager 5 Vault storage and encryption boundary, unlock lifecycle, search and edit model, categories, clipboard integration, import or export, and locking behavior. recli 8 CLI command tree architecture, config inheritance, aliases, output formatting, file cache and SQLite persistence, container orchestration, upgrades, and system checks. rejector 5 Rate-limit scheduler, task and generation scheme abstractions, ICL setup representation, agentic tool-loop state, and multi-provider routing. sheeteval 7 Rule schema for grading, formula and dependency evaluation, fatal and concealed controls, threshold semantics, scenario check mode, and HTML reports. sith 6 Static versus interpreter-assisted analysis, symbol indexing, refactor safety, project root discovery, and environment or settings handling. test_translator 8 Structured test spec parsing, generator per target language, test discovery, equality semantics, generate-before-test enforcement, and loop-as-test translation. textdrop 6 HTTP boundary validation, Markdown rendering, metadata and TOC extraction, static assets, auth and drain lifecycle, and storage backend abstraction. trajectory_api 5 Trajectory data model, token and cost tracking, mutable updates with ETags, fork lineage, grammar parsing for tool calls, and sandboxed execution. xjq 5 Multi-format input detection, XPath and CSS selector dispatch, text extraction modes, file

precedence, and XML or JSON output formatting. Total 196 36 problems

Table 6: Erosion metric sensitivity sweep.

Metric Pass Cost Reported erosion −0.018 0.167 Best variant (CC > 12 + Linear SLOC) −0.136 0.095 LOC −0.212 0.502 Mean CC −0.046 0.218 Max CC −0.117 0.356 Clone Ratio −0.042 0.114 AST-grep hits / LOC −0.052 0.151

Size-heavy baselines such as LOC and max CC remain stronger raw cost predictors, but the erosionfamily conclusion does not depend on the exact threshold or size term.

### C Evaluation Setup

#### C.1 Environment and Invocation

Harness version selection. For each model we report the earliest publicly available harness version that supported that model and could execute the benchmark end-to-end. For older models whose launch-era harness was unavailable or incompatible, we used the nearest later compatible version.

Environment. The container image installs all languages required by the problem set alongside a shared tooling baseline. We derived this baseline by identifying commands whose absence caused

- 1 def find_matches_in_file(text, path, language, rules):

- 2 ...

- 3 for rule in rules:

- 4 if kind == "exact":

- 5 ...

- 6 elif kind == "regex":

- 7 ...

- 8 elif kind == "pattern":

- 9 else: ...

- 10

- 11 if iterable is not None:

- 12 for match in iterable:

- 13 ...

- 14 if kind == "pattern":

- 15 for match in iter_pattern_matches(...):

- 16 ...

- 17 for node in iter_tree_nodes(source_root):

- 18 if not node_matches_selector(selector, node):

- 19 ...

- 20 return matches

- Listing 4: Structurally eroded find_matches_in_file() in code_search at C5 (117 lines total). Nearly all decision-point mass ends up in one function.

- 1 for posix_path, full_path, language in source_files:

- 2 applicable_rules = [

- 3 r for r in all_compiled_rules

- 4 if language in r["languages"]

- 5 ]

- 6 if not applicable_rules:

- 7 continue

- 8 with open(full_path, "r", encoding=encoding) as f:

- 9 content = f.read()

- 10 matches = find_matches_in_content(

- 11 content, applicable_rules, language

- 12 )

- 13 if not matches:

- 14 continue

- 15 match_list.extend(matches)

- 16 all_matches = deduplicate_matches(match_list)

- 17 return all_matches

- Listing 5: Overly verbose code from code_search. Identity list comprehension instead of filter, empty checks instead of building around iteration, and single-use variables.

failures across all harnesses; commands that failed on only one harness were excluded to avoid biasing the environment toward a particular agent. Installed packages, shell history, and agent session data reset between checkpoints. The benchmark problems are language-agnostic by design, but current experiments evaluate only the Python track.

Invocation. Following Terminal Bench [40], we install Claude Code [4] and Codex [45] directly and invoke each in headless mode. Specific versions appear in Table 7.

Reasoning effort. For Codex we set the reasoning effort parameter to high. For Claude Code we configure the thinking-token budget via the environment variable following Anthropic’s published mapping.

- 2 parser = argparse.ArgumentParser(add_help=False)

- 3 parser.add_argument("--data-dir", required=True)

- 4 parser.add_argument("resource")

- 5 parser.add_argument("command")

- 6 parser.add_argument("rest", nargs=argparse.REMAINDER)

- 7 args = parser.parse_args()

- 8

- 9 if args.resource != "blueprint":

- 10 emit_error("validation_error", f"unknown resource: {args.resource}" )

- 11

- 12 cmd, rest = args.command, args.rest

- 13 if cmd == "create": cmd_create(args.data_dir)

- 14 elif cmd == "list": cmd_list(args.data_dir)

- 15 elif cmd == "get": cmd_get(args.data_dir, rest[0])

- 16 # ... two more branches ...

- 17 else:

- 18 emit_error("validation_error", f"unknown command: {cmd}")

Listing 6: main() from Opus 4.7’s forge solution at C1. Argument parsing is delegated to argparse; dispatch is a single if/elif ladder.

Table 7: Harness version and reasoning effort used for each model in the main evaluation.

Model Harness Version Reasoning

- Opus 4.5 Claude Code 2.0.51 high
- Opus 4.6 Claude Code 2.1.32 high Sonnet 4.6 Claude Code 2.1.44 high

- GPT 5.2 Codex Codex CLI 0.93.0 high
- GPT 5.3 Codex Codex CLI 0.98.0 high

- GPT 5.3 Spark Codex CLI 0.100.0 high
- GPT 5.4 Codex CLI 0.110.0 high

- GPT 5.4 Mini Codex CLI 0.110.0 high

Composer 2 Cursor CLI 2026.04.13-a9d7fb5 none GLM 5.1 Claude Code 2.1.44 high

- Kimi K2.5 Claude Code 2.1.44 high
- Kimi K2.6 Kimi CLI 1.37.0 high MiniMax M2.7 Claude Code 2.1.44 high

- C.2 Agent Harness Versions
- C.3 just-solve (Baseline)

The minimal baseline used for all primary evaluations. Each agent receives a Jinja template as its system prompt. The is_continuation flag is false for the first checkpoint of a problem and true for all subsequent checkpoints. The checkpoint specification is injected verbatim via spec.

Implement a program that 100% solves the specification. That is all you need to do. {% if not is_continuation -%} Use a virtual environment and ensure that a ‘requirements.txt‘ is present with any dependencies you need to solve the problem. {% else -%} Keep using the same virtual environment you started with, update ‘requirements.txt‘ with any new dependencies you need. {% endif -%}

Your task is:

- 2 argv = sys.argv[1:]

- 3 data_dir = revision_arg = config_path = role = scope = None

- 4 rules_file_flag = None

- 5 cleaned, i = [], 0

- 6 while i < len(argv):

- 7 tok = argv[i]

- 8 if tok == "--data-dir":

- 9 if i + 1 >= len(argv):

- 10 emit_error("validation_error", "--data-dir requires a value ")

- 11 data_dir = argv[i + 1]; i += 2

- 12 elif tok.startswith("--data-dir="):

- 13 data_dir = tok[len("--data-dir="):]; i += 1

- 14 # ... same two-branch pattern repeats verbatim for

- 15 # --revision, --config, --rules-file, --role, --scope

- 16 # (six more flags, twelve more branches) ...

- 17 else:

- 18 cleaned.append(tok); i += 1

- 19

- 20 # ... config load, revision negotiation (~30 lines) ...

- 21

- 22 if resource == "blueprint":

- 23 if cmd == "create": cmd_create(data_dir, ...)

- 24 elif cmd == "list": cmd_list(data_dir, ...)

- 25 # ... five more cmd branches ...

- 26 else: emit_error("validation_error", f"unknown command: {cmd}")

- 27 elif resource == "revision":

- 28 # ... same cmd-ladder shape, six more resources ...

Listing 7: main() from Opus 4.7’s forge solution at C8, abridged to expose structure. The argparse call has been removed; each of seven flags is handled by an identical pair of branches. After parsing, a nested if resource == ...: if cmd == ... ladder dispatches over seven resources.

{{ spec.strip() }}

Listing 8: just-solve prompt template

- C.4 anti_slop

Explicitly instructs the agent to avoid verbose patterns, defensive over-engineering, and unnecessary abstractions.

{%- if not is_continuation -%} You are an exceptional python software engineer and you to need to implement a spec. Your instructions are:

- - Write the python script that satisfies the spec completely.

- - Use a virtual environment and ensure that a ‘requirements.txt‘ is present with any dependencies you need to solve the problem.

{%- else -%} You are an exceptional python software engineer and you to need to implement a spec. You are updating your code to match an extension of the spec. Here are your instructions:

- - Keep using the same virtual environment you started with, update ‘requirements.txt‘ with any new dependencies you need.

- - Focus only on adding in the new features/changes below.

- - Make sure you test any examples provided in the task description {%- endif %}
- - You ONLY work in this directory.

- - Follow best coding practices:

- - Group functions into files based on related functionality

- - Keep your code clean

- - No god functions/classes.

- - Make sure the code is documented appropriately so that it is easy to pick up.

- - Minimize the following gotchas:

- - Extra defensive checks or try/catch blocks that are abnormal.

- - Casts to get around type checking

- - Variables that are only used a single time after declaration.

- - Extra comments that a human wouldn’t add.

- - Trivial wrappers

- - Heavy nesting

- - If/Else ladders

- - A ton of helper methods

Your task is: {{ spec.strip() }}

Listing 9: anti_slop prompt template

- C.5 plan_first

Requires the agent to plan its approach before writing code.

You are an expert programmer and need to implement a task. {% if not is_continuation -%} Use a virtual environment and ensure that a ‘requirements.txt‘ is present with any dependencies you need to solve the problem. {% else -%} Keep using the same virtual environment you started with, update ‘requirements.txt‘ with any new dependencies you need. {% endif -%}

Here are the steps you should always follow:

- 1. Before coding plan out what you need to implement.

- 2. Write the simple solution first.

- 3. Ensure it is 100% correct and you have covered all edge cases.

- 4. Refactor to ensure the code is high quality.

Here are the basic style rules you must follow:

- - Make sure the code is documented appropriately so that it is easy to pick up.

- - Minimize the following gotchas:

- - Extra defensive checks or try/catch blocks that are abnormal.

- - Casts to get around type checking

- - Variables that are only used a single time after declaration.

- - Extra comments that a human wouldn’t add.

- - Trivial wrappers

- - Heavy nesting

- - If/Else ladders

- - A ton of helper methods

- - Follow best coding practices:

- - Group functions into files based on related functionality

- - Keep your code clean

Your task is: {{ spec.strip() }}

Listing 10: plan_first prompt template

Table 8: Full set of runs on SlopCodeBench across all harnesses, versions, reasoning levels, and prompt strategies. Bold marks the best value per column.

Solve Rate (%) Cost Quality Model Harness Version Think Prompt Strict Iso. Core Partial $/CKPT Net $ Erosion Verbosity

- Opus 4.5 [5] Claude Code 2.0.51 High Just Solve 9.2 17.3 57.7 25.0 2.53±1.57 492.40 0.70±0.20 0.43±0.18

- Opus 4.6 [6] Claude Code 2.1.32 High Just Solve 9.7 20.9 67.3 27.8 3.17±2.82 620.80 0.75±0.15 0.44±0.19

- Opus 4.7 [7] Claude Code 2.1.111 High Just Solve 8.2 20.9 65.8 36.1 2.17±1.83 425.34 0.76±0.17 0.48±0.21 Sonnet 4.6 [8] Claude Code 2.1.44 High Just Solve 7.1 16.8 57.7 27.8 1.96±1.97 383.83 0.75±0.20 0.44±0.19

- GPT 5.2 Codex [46] Codex CLI 0.93.0 High Just Solve 9.7 21.9 56.1 36.1 0.85±0.80 167.29 0.73±0.18 0.50±0.16

- GPT 5.3 Codex [47] Codex CLI 0.98.0 High Anti-Slop 11.2 23.5 57.1 41.7 0.73±0.50 143.04 0.42±0.20 0.31±0.12

- GPT 5.3 Codex Codex CLI 0.98.0 High Just Solve 11.2 26.0 60.7 41.7 0.66±0.49 129.65 0.64±0.19 0.46±0.18

- GPT 5.3 Codex Codex CLI 0.98.0 High Plan First 8.2 21.4 62.2 33.3 0.68±0.45 134.03 0.55±0.24 0.40±0.17

- GPT 5.3 Spark [48] Codex CLI 0.100.0 High Just Solve 3.1 8.2 29.1 11.1 0.20±0.41 37.17 0.68±0.19 0.48±0.16

- GPT 5.4 [49] Codex CLI 0.110.0 High Anti-Slop 9.2 25.5 61.2 38.9 0.90±0.78 176.46 0.28±0.16 0.24±0.09

- GPT 5.4 Codex CLI 0.110.0 High Just Solve 10.7 23.5 62.8 36.1 0.72±0.51 141.32 0.51±0.19 0.33±0.13

GPT 5.4 Codex CLI 0.110.0 High Plan First 9.7 25.0 63.3 38.9 0.83±0.61 163.65 0.39±0.19 0.30±0.12 GPT 5.4 Mini [50] Codex CLI 0.110.0 High Just Solve 5.1 13.8 53.1 22.2 0.45±0.35 89.03 0.66±0.17 0.42±0.12 GPT 5.5 [51] Codex CLI 0.124.0 High Anti-Slop 9.2 24.0 63.3 38.9 1.68±0.83 329.65 0.21±0.15 0.21±0.08 GPT 5.5 Codex CLI 0.124.0 High Just Solve 14.8 28.1 66.8 50.0 1.51±0.81 295.59 0.49±0.20 0.32±0.12

- GPT 5.5 Codex CLI 0.124.0 High Plan First 8.2 24.0 67.3 30.6 1.61±0.72 316.26 0.29±0.18 0.25±0.12 Composer 2 [19] Cursor CLI 2026.04.13-a9d7fb5 None Just Solve 6.1 16.3 52.6 13.9 0.44±0.44 86.31 0.72±0.15 0.45±0.17

GLM 5.1 [67] Claude Code 2.1.44 High Just Solve 9.7 13.8 40.3 30.6 1.47±1.45 288.85 0.71±0.17 0.41±0.18 GLM 5.1 OpenCode 1.4.3 High Just Solve 5.6 8.2 20.4 19.4 0.59±0.99 107.39 0.70±0.22 0.45±0.20

Kimi K2.5 [31] OpenCode 1.4.3 None Just Solve 4.6 8.7 31.6 16.7 0.53±0.80 104.77 0.71±0.18 0.46±0.18 Kimi K2.5 Claude Code 2.1.44 High Just Solve 3.6 7.1 29.6 16.7 1.07±0.96 210.36 0.71±0.18 0.43±0.19

- Kimi K2.5 Kimi CLI 1.37.0 High Just Solve 4.6 9.7 39.8 19.4 0.33±0.25 63.21 0.72±0.20 0.44±0.20

- Kimi K2.6 [32] Kimi CLI 1.37.0 High Just Solve 10.7 18.9 51.0 33.3 0.74±0.64 133.18 0.76±0.20 0.51±0.22

MiniMax M2.7 [42] Claude Code 2.1.44 High Anti-Slop 2.6 4.1 29.1 11.1 0.33±0.21 64.92 0.51±0.22 0.38±0.15 MiniMax M2.7 OpenCode 1.4.3 None Just Solve 1.5 3.6 21.9 5.6 0.27±0.36 53.36 0.75±0.20 0.45±0.20 MiniMax M2.7 Claude Code 2.1.44 High Just Solve 2.0 4.1 28.1 8.3 0.34±0.23 65.79 0.73±0.16 0.47±0.17

- D Results Continued D.1 Human Panel construction

The temporal panel spans 473 maintained Python repositories sampled across web frameworks, scientific computing, infrastructure tools, machine-learning libraries, and command-line utilities. Repositories are stratified by GitHub star tier: Hobby (< 100⋆, n=114), Niche (100–1k⋆, n=124), Established (1–10k⋆, n=120), and Major (≥ 10k⋆, n=115). For each repository we sample up to 30 source-touching commits across its history, totaling 13667 checkpoints between 1990 and 2026. Checkpoints with empty source trees or failed metric runs are dropped before analysis. Each repository’s most recent checkpoint serves as its HEAD snapshot for the calibration band reported in subsection 3.3. Table 2 reports HEAD verbosity and erosion by tier and against agent checkpoints.

Pre- vs. post-ChatGPT temporal check. The panel straddles 2022, so post-ChatGPT commits could in principle include LLM-assisted contributions. Pre-ChatGPT checkpoints (n=9705) have a median verbosity of 0.155 versus 0.166 for post-ChatGPT checkpoints (n=3962). The shift is real but small relative to the agent gap (mean verbosity 0.44). The same comparison anchored at January 2024 yields medians of 0.156 versus 0.168 for verbosity and 0.278 versus 0.319 for erosion.

Within-repository agent-era shift. A cross-section across the 2024 boundary mixes new repositories in with old ones. Restricting to the 321 repositories with at least three checkpoints in each era, the median within-repository verbosity shift is +0.002 and the erosion shift is +0.014. 55% of these repositories are more verbose post-2024 and 59% are more eroded. The signal is consistent with a small uptick in human-authored slop after agent-assisted contributions became common, but it is dwarfed by the agent-side trajectory growth reported in subsection 3.3.

First-to-last growth. For repositories with at least five sampled commits, 256 of 378 (68%) end with higher verbosity than they started, with median first-to-last growth of 36%. Erosion rises in 103 of 196 (53%) repositories. Per-checkpoint slope is 0.0022 for verbosity and 0.0053 for erosion across the human panel, against 0.0144 and 0.0264 for agent trajectories.

Notable outliers. huggingface/transformers (verbosity 0.451, erosion 0.473) and fastapi/fastapi (0.273, 0.219) are the most prominent verbosity outliers in the panel, both above the agent verbosity mean. The high transformers verbosity is driven by per-architecture file duplication (one model implementation per family). At the other end, pallets/flask (0.111, 0.209) and scikit-learn/scikit-learn (0.094, 0.407) are large, mature projects with low verbosity and middling erosion. scipy/scipy (0.145, 0.441) sits at the high end of human erosion, still well below the agent mean.

### E Extended Related Work

Quality degradation under multi-turn coding. LLM-generated code degrades under repeated modification. Code converges toward structural attractors [53], quality diverges across agents and runs [15, 55, 11], and refinement introduces defects that correctness testing does not catch [14, 23]. Interaction failure modes compound these effects [71, 29, 56]; agent-generated code has become a practical integration burden [43, 9].

Code quality metrics. LLM code has higher cyclomatic complexity per line than human-written code [22], though aggregate CC can fall while vulnerabilities rise [17]. Abbassi et al. [1] extend the code smell taxonomy to LLM outputs, finding redundant steps, duplication, and unnecessary conditionals most prevalent. Code smells are well-studied in software engineering [26, 33]; software aging [52] and technical debt [18] describe how structural degradation accumulates under modification, with Li et al. [35] and Le et al. [34] documenting this in long-lived systems.

Benchmark landscape. Beyond the paradigm established by Jimenez et al. [28], the evaluation landscape spans three waves. The first broadens language and domain coverage through extensions [3, 68, 63, 10, 54]. Instruction-following benchmarks evaluate compliance across conversation turns [59, 24] but assess each response independently. A second wave targets feature-level development from existing repositories [36, 75]. A third wave builds entire projects from scratch: Zhao et al. [73] generate libraries from specifications with interactive test feedback, several benchmarks construct full repositories from natural-language requirements [38, 69, 21, 72, 37], and Feng et al. [25] spans from-scratch construction through refactoring. Tasks grow larger and harder across all three waves; evaluation in each case is a single artifact assessed against a fixed specification.

