# arXiv:2606.22883v1[cs.AI]22Jun2026

2026-06-23

## CLI-Universe: Towards Verifiable Task Synthesis Engine for Terminal Agents

Nanjing University StepFun ZODA Shanghai AI Lab Huazhong University of Science and Technology

See Contributions section for a full author list.

### Abstract

While recent LLM-based terminal agents have demonstrated promising capabilities, the scarcity of high-quality, executable training data remains a critical bottleneck. Existing synthesis pipelines typically scale by retrofitting surface-level artifacts into tasks, frequently yielding ambiguous instructions, shallow execution paths, and brittle tests that provide weak learning signals. To overcome this, we introduce CLI-Universe, a principled synthesis engine that constructs terminal-agent tasks. CLI-Universe generates candidate tasks by sampling combinations across a multi-dimensional capability taxonomy (domain, skill type, capability, and engineering pillar), then grounds each candidate through evidence-guided deep research over real-world technical materials. To ensure rigorous supervision, validated blueprints are instantiated into Dockerized environments and subjected to a multi-stage executable verification pipeline—featuring rubric-gated test construction, hint-conditional filtering, and strict fail-to-pass checking. Across the full pipeline, from candidate generation to verification, approximately two-thirds of candidates are discarded, retaining only those that are genuine, verifiable, and non-trivially challenging. To validate our framework, we instantiate a highly distilled dataset of 6,000 trajectories called CLI-Universe-6K. Remarkably, fine-tuning Qwen3-32B on CLI-Universe-6K achieves 33.4% on Terminal-Bench 2.0. This sets a new state-of-the-art for models trained on open-source data at or below 32B parameters, and outperforms several models an order of magnitude larger, demonstrating the profound data efficiency of structured, high-fidelity synthesis.

### 1 Introduction

Recent terminal agents (Anthropic, 2025a; OpenAI, 2025a; Google, 2025; OpenCode Team, 2025; SWEagent Team, 2025; Wang et al., 2025; Merrill et al., 2026) have demonstrated increasingly strong performance on complex CLI tasks, from software debugging and system administration to security analysis and data engineering. Yet high-quality training data for building capable terminal-agent models remains scarce. Effective training requires tasks that are genuinely difficult, demanding multi-step interaction with realistic environments, and validated by rigorous executable checks that provide unambiguous supervision signal rather than merely confirming that a script runs. Existing synthesis pipelines (Wu et al., 2026; Pi et al., 2026; Fan et al., 2026; Gandhi et al., 2026; Lin et al., 2026; Zhu et al., 2026) attempt to address this gap by converting readily available surface materials into tasks, for example by mining repositories and documentation, transforming execution traces and templates, or repurposing existing benchmarks. However, these materials were not originally designed as training tasks, and direct conversion often produces ambiguous queries, shallow execution paths, or brittle tests that provide weak learning signal even when they run successfully. In short, existing pipelines scale task sources rather than task quality.

To address this, we propose CLI-Universe, a pipeline that constructs terminal-agent tasks from structured capability specifications combined with evidence-guided deep research. Rather than retrofitting tasks from existing artifacts, CLI-Universe takes an inside-out approach: each task candidate is first defined through a structured taxonomy that combines a target domain with specific skill types, capabilities, and engineering pillars, ensuring broad and controlled coverage of the capability space. Each candidate is then refined through deep research over real technical materials: we search repositories, documentation, issue discussions, and usage examples to ground the task in realistic tools, constraints, and failure modes. This evidence-guided process transforms abstract capability specifications into technically grounded blueprints that exercise meaningful engineering workflows.

Validated blueprints are instantiated into Dockerized environments with materialized assets and runtime state. To ensure reliable supervision, CLI-Universe applies a multi-stage executable verification pipeline. First, role-separated agents independently construct rubric-gated tests and reference solutions, where the test agent and solution agent operate without seeing each other’s output. Second, hint-conditional

* Equal Contribution. † Corresponding Author.

filtering retains only tasks where the internal hint is genuinely necessary for success, removing trivially solvable instances that would provide little training value. Finally, a fail-to-pass check provides executable proof that each retained task realizes a meaningful state transition from an unsolved state to a verified solution. Together, these stages reject approximately two-thirds of all candidates, concentrating supervision on high-signal examples.

After fine-tuning Qwen3-32B (Yang et al., 2025) on only 6k CLI-Universe trajectories, our model achieves 33.4% on Terminal-Bench 2 (Merrill et al., 2026), surpassing all models trained with open-source data at ≤32B scale and outperforming several open-weight models an order of magnitude larger on the official leaderboard. Performance scales monotonically with model size and generalizes to out-of-domain benchmarks, suggesting that structured specification, evidence-guided deep research, and multi-stage verification can together provide strong supervision even at limited data scale.

Our main contributions are:

- • A data synthesis pipeline for terminal-agent training. CLI-Universe defines each task through a structured capability taxonomy and refines it via evidence-guided deep research over real technical materials, producing Dockerized environments grounded in realistic constraints and failure modes. Unlike pipelines that retrofit tasks from surface artifacts, this inside-out design ensures that tasks exercise meaningful capability patterns by construction.
- • Multi-stage verification that concentrates training signal. CLI-Universe applies rubric-gated test construction, hint-conditional filtering, and fail-to-pass checking to reject approximately two-thirds of candidates, retaining only tasks with high-fidelity supervision. Ablation shows that removing any single component costs 3–6 points on Terminal-Bench 2, confirming that each verification stage drives downstream performance.
- • A high-fidelity 6k-trajectory training set (CLI-Universe-6K) and empirical validation. We construct CLI-Universe-6K, a 6,000-trajectory training set distilled from the pipeline. Fine-tuning Qwen3-32B on CLI-Universe-6K achieves 33.4% on Terminal-Bench 2, surpassing all models trained with open-source data at ≤32B scale and outperforming several models an order of magnitude larger on the official leaderboard, demonstrating the data efficiency of our pipeline. Performance scales monotonically with model size and generalizes to out-of-domain benchmarks including BFCL v4 and VitaBench (Patil et al., 2025; He et al., 2025).

### 2 Related Work

Terminal Agents. LLM-based agents have seen rapid progress in the code domain, with capabilities expanding from repository-level issue resolution (Yang et al., 2024; Jimenez et al., 2024) to interactive terminal environments where agents execute multi-step workflows through command-line interfaces. A growing collection of agent scaffolds (Anthropic, 2025a; OpenAI, 2025a; Google, 2025; Wang et al., 2025; OpenCode Team, 2025) supports this paradigm by enhancing LLMs’ planning, execution, and tool-calling capabilities. To evaluate terminal-agent performance, Terminal-Bench (Merrill et al., 2026) provides hand-crafted tasks spanning diverse domains within containerized Docker environments. However, open-source models remain substantially behind proprietary systems on these tasks, and performance improvements are increasingly bottlenecked by the scarcity of high-quality terminal-agent training data.

Synthetic Data for Terminal Agents. Existing approaches to scaling terminal-agent training data follow two main strategies. The first generates tasks from skill or domain taxonomies produced by LLMs (Gandhi et al., 2026; Zhu et al., 2026; Pi et al., 2026; Fan et al., 2026), achieving broad topical coverage by systematically enumerating capability dimensions. The second extracts tasks from existing infrastructure, e.g. by harvesting Docker environments from open-source repositories (Wu et al., 2026) or transforming working configurations into faulty states to create debugging scenarios (Lin et al., 2026). Both strategies effectively increase task volume, but offer limited guarantees on per-task verification quality and training signal density. CLI-Universe takes a complementary approach: it combines structured capability specification with evidence-guided deep research and multi-stage executable verification, so that each retained trajectory is grounded in realistic technical constraints and validated by rigorous executable checks.

[Figure 1]

Step1: Task Blueprint Construction

[Figure 2]

[Figure 3]

Blueprint Formation

Anchor-Based Ideation

Evidence-Guided Refinement

[Figure 4]

[Figure 5]

[Figure 6]

Environment checklist

Instruction.md

Hint.md

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Skill

Count persons by applying YOLO detection

[Figure 12]

[Figure 13]

- -Synth: Generate metronome videos.Tempo/color/position variations ......
- -Pull: Collect real metronome videos.Normalize fps and......
- -Runtime:python:3.11-slim,OpenCV

[Figure 14]

[Figure 15]

ç≈ç ç≈ç ç≈ç

ç≈ç ç≈ç ç≈ç

Create CLI program Run: python3 /app/metronome_beats.py VIDEO.mp4 Detect L/R swing extrema Use frames, debounce, alternate sides ....... Outputs: WebVTT....

Domain

- -Track saturated bob blob Allow vertical motion Avoid hardcoded color/y
- -Detect trajectory extrema Debounce same-side ≥8f......
- -Read fps/size via cv2 Export WEBVTT + PPM crops

[Figure 16]

Data Processing Configuration Mathematical Deployment ......

[Figure 17]

video-processingç file-operations model-training data-science optimization data-querying

[Figure 18]

[Figure 19]

Too Easy/Library Wrapper

[Figure 20]

video-processing

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Capabilities

ç≈ç

ç

[Figure 26]

Output score for actor's performance in video

[Figure 27]

+ NumPy.....

Cross-representation Exploration Long-horizon-planning ......

[Figure 28]

......

Subjective – no GT

[Figure 29]

[Figure 30]

Blueprint Validation

[Figure 31]

[Figure 32]

Engineering Pillar

[Figure 33]

Fixed-camera MP4 to temporal events plus spatial stitch

[Figure 34]

[Figure 35]

......

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Clarity & Convergence Solvability & Verifiability Difficulty & Robustness

New feature creation Debugging Systems programming ......

[Figure 40]

[Figure 41]

Step2: Environment Realization Step3: Executable Verification

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Container Setup

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- -Create controlled variants
- -Generate verification metadata

Pull & Adapt

- - Acquire external resources
- - Adapt to task requirements

- -Assemble execution environment
- -Resolve dependencies

[Figure 59]

[Figure 60]

Runtime Verification

- -Verify environment integrity
- -Validate basic functionality

Initial 0/N

xcç ≈ç

[Figure 61]

All tests fail

Instruction

Test Agent Refine tests until they pass the rubric

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Synthesize Assets

[Figure 67]

[Figure 68]

[Figure 69]

Environment Agent

[Figure 70]

[Figure 71]

###### +

[Figure 72]

Excutable Environment

+

[Figure 73]

N/N

All tests pass

Solution Agent

Solution

Fail2Pass Filter

Asset Materialization Container Assembly

- Figure 1: Overview of CLI-Universe. Step 1 (Query Construction): A structured taxonomy seeds diverse task candidates; each is grounded through iterative deep research and compiled into a blueprint. Step 2 (Environment Synthesis): Blueprints are realized into Docker containers with materialized assets and verified runtime. Step 3 (Validation & Filtering): A test agent and a solution agent independently verify the task; only instances passing rubric-gated tests and fail-to-pass checks are retained.

### 3 Method

#### 3.1 Overview

CLI-Universe constructs terminal-agent tasks through a structured synthesis pipeline (Figure 1) with three stages. First, task blueprint construction (Section 3.2) specifies candidates along domain, skill type, capability, and engineering pillar dimensions, then iteratively refines each candidate through a research agent that searches real technical materials and incorporates evidence into the task specification, producing validated blueprints. Second, environment realization (Section 3.3) turns each blueprint into a fully executable Dockerized environment. Third, validation with executable filtering (Section 3.4) constructs tests and reference solutions, retaining only candidates that pass fail-to-pass checking. Candidates are progressively filtered at each stage, and only 33.6% survive end-to-end (Figure 2d).

#### 3.2 Task Blueprint Construction

Task Candidate Specification. We anchor idea generation on three orthogonal dimensions beyond domain: (1) skill type, the specialized technical knowledge required (e.g., algorithmic, systems, configuration, cryptography); (2) capability, the reasoning behavior the task should elicit (e.g., exploration, error recovery, constraint satisfaction, long-horizon planning); and (3) engineering pillar, the form of work the agent performs (e.g., new feature creation, debugging, DevOps, refactoring). For each domain, we predefine a pool of values per dimension and sample combinations as anchor points. Given an anchor, we brainstorm task ideas constrained to exercise the specified capability pattern, ensuring diversity at the skill level rather than only in surface topic. Generated ideas are scored for creativity, technical grounding, and feasibility, and only top-scoring candidates proceed to evidence-guided refinement. The full taxonomy is given in Appendix A.

Evidence-Guided Refinement. Each task candidate undergoes iterative refinement driven by a dedicated research agent. Starting from the abstract idea, the agent searches real technical materials (repositories, documentation, issue discussions, tutorials, and usage examples) and progressively incorporates the evidence into the task specification, grounding it in specific tools, realistic constraints, known failure modes, and concrete input/output contracts. The agent continues refining until the specification is sufficiently complete to support query formulation, environment construction, and test generation. Candidates that cannot be adequately grounded (e.g., those requiring unavailable tools or conflicting constraints) are discarded. As shown in Figure 2a, tasks with evidence-guided refinement require 3.45× more solver turns and reduce pass rate by 13.3 points compared to unrefined tasks, confirming that the

90

###### w/ rubric

100

100

| |72%<br><br>91%<br><br>75%<br><br>93%<br><br>Human accept rate<br><br>LLM accept rate| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| |91% 88%| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

80

75

Meanpassrate(%)

75

(5.34 turns, 68.2%)

w/o material

70

Rate(%)

Rate(%)

50

50

60

###### -13.3 pt pass · 3.45× turns

###### w/ material

25

25

(18.43 turns, 54.9%)

50

0

40

0

Pass rate (official solution)

Semantic match (agent-as-judge)

0 5 10 15 20 25

Before After

Mean turns

(a) Solver turns and pass rate, before vs. after refinement.

(b) Blueprint accept rate, before vs. after rubric review.

(c) Agreement between our tests and TB-2 ground truth.

All Candidates

Idea Filtered

Blueprint Validated

Environment Built

Verified Tasks

100%

70.0%

56.0%

42.0%

33.6%

Fail-to-pass filtered −8.4%

Failed environments −14.0%

Invalid blueprints −14.0%

Rejected ideas −30.0%

(d) Per-stage retention across the five filters.

CLI-Universe (Ours)

LiberCoder

40

SkillSynth

ROME

Nemotron-Term

TerminalTraj

35

CLI-Universe-32B

LiberCoder-235B

30

SkillSynth-32B

TB2.0passrate(%)

Nemotron-Term-32B

25

ROME-30B

CLI-Universe-14B

TerminalTraj-32B

SkillSynth-14B

20

LiberCoder-32B

Nemotron-Term-14B

TerminalTraj-14B

15

SkillSynth-8B

Nemotron-Term-8B

10

CLI-Universe-8B

TerminalTraj-7B

5

0

104 105 106 # Training Trajectories / Instances

(e) TB-2.0 score vs. number of training trajectories.

- Figure 2: Each synthesis stage adds a measurable gain, and the resulting dataset trains agents with 10–100× fewer trajectories. (a)–(c) validate the three pipeline stages: evidence refinement raises difficulty (3.45× solver turns, −13.3 pt pass rate); rubric-based blueprint review lifts accept rate to 91%/93% (human/LLM); synthesized tests agree with Terminal-Bench 2 ground truth on 91% of sampled tasks (88% semantic match, Codex / GPT-5.4 judge). (d) The five-stage filter retains 33.6% of candidates end-to-end. (e) Models trained on CLI-Universe data match or exceed TB 2.0 baselines using 1–2 orders of magnitude fewer training trajectories.

refinement raises genuine difficulty rather than merely lengthening trajectories.

Blueprint Formation and Validation. Each refined candidate is compiled into a blueprint that records the user-facing instruction, an internal hint for reference-solution construction, and an environment checklist. CLI-Universe validates each blueprint before proceeding, retaining only those whose specification is sufficiently clear and whose setup admits reliable downstream verification. As shown in Figure 2b, human and LLM judges show high agreement, and rubric-based validation substantially increases acceptance rates (human: 72% → 91%; LLM: 75% → 93%).

#### 3.3 Environment Realization

Only validated blueprints are carried forward to environment realization, where CLI-Universe turns each blueprint into a fully executable Dockerized environment.

Asset Materialization. Guided by the environment checklist in the blueprint, CLI-Universe acquires the required assets (source repositories, documentation, datasets, configuration files, service logs, etc.) from publicly available resources. Retrieved assets rarely match the task specification exactly, so the system adapts them to fit: normalizing data formats, injecting controlled faults, adjusting configuration parameters, or scoping content to the intended workflow. When suitable external assets are unavailable, the system synthesizes them from scratch, creating controlled variants with known ground-truth properties and generating the verification metadata needed for downstream testing.

Environment Assembly. The materialized assets are then packaged into a Docker image together with the required dependencies, runtime configuration (environment variables, services, permissions), and pinned package versions to ensure reproducibility. The system places each asset at the filesystem location specified by the blueprint and wires up any inter-component references (e.g., paths in configuration files,

database connection strings) so that the environment is self-contained. Each assembled environment undergoes a suite of smoke tests that verify successful dependency installation, correct service startup, expected filesystem layout, and basic end-to-end reachability. Environments that fail these checks are discarded.

#### 3.4 Test Construction and Executable Filtering

CLI-Universe constructs both executable tests and a reference solution trajectory for each task. The tests operationalize task completion with respect to the user-facing query, while the solution trajectory provides a successful demonstration under internal guidance and is retained as training data for the final dataset.

Test Construction. CLI-Universe first constructs task-specific tests from the realized environment and the validation target specified in the blueprint. The test agent generates a candidate test suite, then iteratively checks each test against a set of test-case rubrics covering correctness, determinism, and edge-case coverage, refining or replacing tests until they provide a stable executable signal for the task.

To assess the validity of the constructed tests, we apply the same test-construction pipeline to 89 TerminalBench2 tasks and compare the resulting test suites with the official ones. As shown in Figure 2c, the official solutions pass our synthesized tests on 91% of sampled tasks, and agent-as-judge semantic match (using Codex with GPT-5.4) between our tests and the official ones reaches 88%. These results indicate that our pipeline produces tests that are consistent with the official ground truth.

Solution Construction. A solution agent is prompted with the realized environment and an internal hint from the blueprint to produce a successful solution trajectory. The hint provides key resolution steps and expected intermediate states without appearing in the actual task query, anchoring an intended resolution path while preserving the external difficulty of the task. The resulting trajectory is retained as training data only when it resolves the task under the realized environment and remains consistent with the executable completion signal established by the test suite.

Hint-Conditional Filtering. We further filter tasks and solutions by comparing solution attempts with and without the internal hint. A separate agent attempts each task without access to the hint; a candidate is retained only when the hint-free attempt fails while the hint-guided attempt succeeds. This removes trivially solvable instances so that the final dataset concentrates on task–trajectory pairs where the supervision provides meaningful training value.

Fail-to-Pass Filtering. A final fail-to-pass condition is imposed on each retained solution. The generated tests must fail on the initial environment and pass after execution of the hinted solution trajectory. This bidirectional check removes both vacuous tasks whose tests pass trivially and unsupported trajectories that do not actually reach the goal state, ensuring that every retained example realizes an executable transition from an unsolved state to a verified solution.

### 4 Experiments

We use the Qwen3 (Yang et al., 2025) dense series (8B, 14B, and 32B) as base models and fine-tune each on 6k CLI-Universe trajectories collected from Kimi-K2.6 (Team et al., 2026). Full training details are in Appendix B. We evaluate on Terminal-Bench 1.0 and 2.0 (Merrill et al., 2026) (TB 1.0, TB 2.0) mainly using the Terminus 2 scaffold with 200 turns per task and report avg@4. Generalization is measured on BFCL v4 (Patil et al., 2025) and VitaBench (He et al., 2025).

#### 4.1 Main Results

CLI-Universe surpasses all models trained with open-source data at ≤32B. Table 1 reports performance on Terminal-Bench 1.0 and 2.0 under the Terminus 2 scaffold. CLI-Universe-32B reaches 33.4 on TB 2.0, outperforming SkillSynth-32B (Fan et al., 2026) (29.6), Nemotron-Terminal-32B (Pi et al., 2026) (27.4), and TerminalTraj-32B (Wu et al., 2026) (22.0). The advantage holds at 14B and 8B. Our 32B model also surpasses several open-weight systems an order of magnitude larger (e.g., 480B Qwen3-Coder at 23.9, 1T Kimi-K2-Instruct at 27.8), though a gap to the strongest proprietary systems (e.g., Claude-Opus-4.5 at 57.8) remains.

Gains scale with model size. Figure 3b reports TB 2.0 scores of Qwen3 baselines and our CLI-Universe models at three sizes. We observe two trends. First, the Qwen3 baselines stay essentially flat across

- Table 1: Main results on Terminal-Bench 1.0 and 2.0. Models are sorted by TB 2.0 within each group. CLI-Universe models surpass all models trained with open-source trajectory data at comparable size (≤32B). All baselines use the Terminus 2 scaffold except LiberCoder, which uses OpenHands.

Model Size Terminal-Bench 1.0 Terminal-Bench 2.0 Closed-Source Models

Claude-Opus-4.5 (Anthropic, 2025b) – 47.5 57.8 Gemini 3 Pro Preview (Google DeepMind, 2025) – 46.4 56.9 GPT-5.2 (OpenAI, 2025b) – 54.4 54.0

Open-Source Models

GLM-4.7 (GLM-5-Team et al., 2026) 358B 48.8 41.0 LiberCoder-235B (Lin et al., 2026) 235B 46.1 31.0 SkillSynth-32B (Fan et al., 2026) 32B 33.8 29.6 Minimax-M2.1 (MiniMax et al., 2026) 229B 42.5 29.2 Kimi-K2-Instruct (Team et al., 2026) 1T 44.6 27.8 Nemotron-Terminal-32B (Pi et al., 2026) 32B – 27.4 Qwen3-Coder-480B (Cao et al., 2026) 480B 37.5 23.9 TerminalTraj-32B (Wu et al., 2026) 32B 35.3 22.0 Nemotron-Terminal-14B (Pi et al., 2026) 14B – 20.2 SkillSynth-14B (Fan et al., 2026) 14B 22.9 19.9 LiberCoder-32B (Lin et al., 2026) 32B 38.9 19.5 TerminalTraj-14B (Wu et al., 2026) 14B 28.9 19.1 Qwen3-Coder-30B-A3B (Cao et al., 2026) 30B 23.8 14.6 SkillSynth-8B (Fan et al., 2026) 8B 17.1 13.5 Nemotron-Terminal-8B (Pi et al., 2026) 8B – 13.0 TerminalTraj-7B (Wu et al., 2026) 7B 23.0 10.1 Qwen3-14B (Yang et al., 2025) 14B 11.6 4.0 Qwen3-32B (Yang et al., 2025) 32B 11.2 3.4 Qwen3-8B (Yang et al., 2025) 8B 8.1 2.5

Ours

CLI-Universe-32B 32B 38.8 33.4 CLI-Universe-14B 14B 27.5 23.0 CLI-Universe-8B 8B 19.1 10.9

scales (2.5 / 4.0 / 3.4 at 8B / 14B / 32B); without targeted agentic data, simply enlarging the base model does not unlock terminal-agent capability. Second, after training on CLI-Universe trajectories the same checkpoints jump to 10.9 / 23.0 / 33.4, and the gain over baseline grows monotonically with model size (+8.4 → +19.0 → +30.0). This suggests that CLI-Universe data scales with model capacity rather than saturating, and that larger students extract more value from the same teacher trajectories, a regime where bigger models are not yet bottlenecked by data difficulty.

4.2 Ablation Studies

We isolate the contribution of three pipeline components and two data-side choices (trajectory selection and teacher model). Unless noted, the student model is Qwen3-32B, and we report TB 2.0 scores.

- Table 2: Data-side ablations on Qwen3-32B. (a) trajectory selection strategy; (b) teacher model choice. TB 2.0 scores reported as avg@4.

(a) Trajectory selection. Strategy #Traj. TB2.0

Complete (all kept) 10k 28.2 Success-only 6k 33.4

###### (b) Teacher model. Teacher #Traj. TB2.0

DeepSeek-V4-Pro 6k 31.2 Kimi-K2.6 6k 33.4

###### 4.2.1 Effect of different components

Figure 3(a) isolates the contribution of each pipeline component by removing it in turn while keeping the rest of the pipeline and training recipe fixed. To make the ablation study tractable, we conduct it on a 1k-task subset using the Qwen3-32B student. All three ablations cause a substantial drop relative to the full pipeline (26.7): removing the asset strategy is the most damaging (−6.2, down to 20.5), indicating that

Full CLI-Universe

###### 26.7

w/o asset strategy

20.5

w/o query rubrics

23.3

w/o test-case rubrics

22.8

0 10 20 30 TB2.0 pass rate (%)

(a) Component ablation.

40

###### +8.4 +19.0 +30.0

33.4

TB2.0passrate(%)

Qwen3 (baseline) CLI-Universe (Ours)

| |
|---|

30

| |
|---|

23.0

20

10.9

10

4.0 3.4

2.5

0

8B 14B 32B Model Size

(b) Model scaling.

###### 33.4

35

28.9

30

TB2.0passrate(%)

25

20

18.0

###### +30.0

+25.5

15

+14.6

10

5

0

TerminalTraj Nemotron CLI-Universe (Ours)

(c) Data efficiency.

- Figure 3: Ablation, scaling, and data efficiency of CLI-Universe. (a) TB 2.0 score when each pipeline component is removed on a 1k-task subset using Qwen3-32B. (b) Qwen3 baselines vs. CLI-Universe at 8B / 14B / 32B on TB 2.0; blue badges show absolute gains. (c) TB 2.0 score of Qwen3-32B fine-tuned on 6k trajectories from each of TerminalTraj, Nemotron, and CLI-Universe (matched data volume); white labels show gain over the Qwen3-32B baseline.

the diversity of seeded environments is a major driver of task coverage. Removing the query rubrics costs

- 3.4 points (down to 23.3), showing that even when assets and tests are in place, query quality continues to bound what the student can learn. Removing the test-case rubrics costs 3.9 points (down to 22.8), confirming that high-fidelity verification contributes meaningfully to trainable trajectories. Together these results suggest that the three components are complementary rather than redundant, and that quality control at each stage contributes to downstream performance.
- 4.2.2 Effect of different selection strategies

- Table 2a compares trajectory filtering strategies on Qwen3-32B. Keeping only the 6k successful trajectories (those that pass all test cases) yields the best downstream performance (33.4 on TB 2.0), outperforming the unfiltered complete set of 10k trajectories (28.2) by +5.2 points. This suggests that failed and incomplete interactions introduce noise that degrades training signal at this model scale, and that quality filtering on correctness is more important than raw trajectory volume.

- 4.2.3 Effect of different teachers

Table 2b shows the impact of teacher model choice on student performance. We sample 6k trajectories from each teacher on the same CLI-Universe tasks and distill into Qwen3-32B. Kimi-K2.6 (Team et al., 2026) reaches 33.4 on TB 2.0 while DeepSeek-V4-Pro (DeepSeek-AI, 2026) scores 31.2, suggesting that the pipeline is robust to teacher choice and does not require a specific frontier model to produce strong supervision.

4.3 Scaling Analysis

- 4.3.1 Data scaling

We investigate how the number of CLI-Universe training trajectories affects downstream performance. We train Qwen3-8B on subsets of increasing size and evaluate on TB 2.0. As shown in Figure 2e, performance improves steadily as data scales, with no sign of saturation at our largest budget. This suggests that the pipeline can continue to generate useful supervision and that the current data regime has not yet reached the point of diminishing returns.

###### 4.3.2 Data efficiency

Figure 3(c) compares the three data sources on Qwen3-32B. CLI-Universe achieves the highest TB 2.0 score (33.4), outperforming Nemotron (Pi et al., 2026) (28.9) and TerminalTraj (Wu et al., 2026) (18.0). Measured as gain over the Qwen3-32B baseline (3.4), CLI-Universe yields +30.0 points, versus +25.5 for Nemotron and +14.6 for TerminalTraj. This suggests that the per-trajectory information density of CLI-Universe is markedly higher, and that benchmark gains here are driven by the quality of the underlying tasks and verification rather than trajectory count alone.

BFCL-v4

Qwen3 (baseline) CLI-Universe (Ours)

| |
|---|

+11.3

| |
|---|

60

+7.0 58.0

pass@1(%)

46.7

47.4

40.4

40

20

0

8B 32B Model Size

VitaBench

30

+11.6

27.0

pass@1(%)

20

15.4

+1.1

10.0

10

11.1

0

8B 32B Model Size

(a) Generalization across agentic benchmarks (BFCL v4, VitaBench).

Qwen3-32B CLI-Universe-32B

Software & System

+41.7

System Administration (n=9)

+37.5

Security (n=8) Software Engineering (n=26)

+28.8

+20.0

Debugging (n=5) File Operations (n=5)

+5.0

Data & Science

+62.5

Data Processing (n=4) Data Querying (n=1)

+50.0

+25.0

Data Science (n=8) Scientific Computing (n=8)

+15.6

+12.5

Mathematics (n=4)

Machine Learning

+50.0

Machine Learning (n=3) Model Training (n=4)

+43.8

Other

+100.0

Personal Assistant (n=1) Optimization (n=1)

+75.0

+0.0 +0.0

Games (n=1) Video Processing (n=1)

0 25 50 75 100 TB2.0 pass rate (%, mean@4)

(b) TB 2.0 pass rate by fine-grained category at 32B (avg@4).

- Figure 4: Cross-benchmark and per-category performance at 32B. (a) Qwen3-32B baselines vs. CLIUniverse-32B on two out-of-domain agentic benchmarks. (b) TB 2.0 avg@4 broken down by fine-grained categories grouped into four super-categories; gray dots show Qwen3-32B baseline.

#### 4.4 Generalization

- 4.4.1 Cross-benchmark transfer

To verify that CLI-Universe training generalizes beyond Terminal-Bench, we evaluate on two additional agentic benchmarks: BFCL v4 (Patil et al., 2025) (function calling) and VitaBench (He et al., 2025) (multiturn tool-use). Figure 4a shows consistent improvements: CLI-Universe-32B outperforms Qwen3-32B by +11.3 on BFCL v4 (58.0 vs. 46.7) and +11.6 on VitaBench (27.0 vs. 15.4), with gains of +7.0 and +1.1 at 8B respectively. This cross-benchmark transfer suggests that the skills acquired from CLI-Universe (tool orchestration, environment state tracking, and multi-step planning) are broadly applicable rather than narrowly fitted to Terminal-Bench.

- 4.4.2 Performance on different categories

Figure 4b breaks down TB 2.0 avg@4 by fine-grained categories at 32B. The Qwen3-32B baseline scores near zero in most categories, whereas CLI-Universe training unlocks substantial capability across the board. The largest absolute gains appear in Data Processing (+62.5), Machine Learning (+50.0), Data Querying (+50.0), and Model Training (+43.8). In the Software & System group, System Administration (+41.7), Security (+37.5), and Software Engineering (+28.8) all show strong improvements. A few categories remain challenging: Video Processing and Games show no gain at 32B, suggesting directions for future pipeline expansion.

#### 4.5 Error Study

We run two trajectory rollouts per task on Terminal-Bench 2 for each model, and use Codex with GPT-5.4 to classify each failed trajectory by the single failure mode most causally responsible for its failure, against a 9-mode taxonomy organized into three classes (Execution / Coherence / Verification, see Figure 5).

Frontier SOTA models fail mostly on the verification side. For all four proprietary baselines (ClaudeOpus-4.6 (Anthropic, 2025b), GPT-5.3-Codex (OpenAI, 2025b), GLM-5 (GLM-5-Team et al., 2026), and DeepSeek-V4-Pro (DeepSeek-AI, 2026)), the Verification class accounts for the largest share of failures, ranging from 47% to 60%. The pattern suggests these failures are not primarily an inability to execute or reason about the task: the agent typically makes a plausible attempt, but does not correctly check whether the goal state is actually satisfied before terminating.

| | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | |

Claude-Opus-4.6

GPT-5.3-Codex

GLM-5

DeepSeek-V4-Pro

CLI-Universe-32B

0 20 40 60 80 100 Proportion of failed trajectories

Execution

Coherence

Verification

Disobey specification

Context loss

Premature termination

Task derailment

No/incorrect verification

Step repetition

Reasoning--action mismatch

Weak verification

Unaware of termination

- Figure 5: Primary failure attribution on failed TB 2.0 trajectories. Each failed trajectory is assigned the single failure mode most causally responsible for the task failure (mutually exclusive, rows sum to 100%). The 9 modes are organized into three classes: Execution (disobey spec, step repeat, unaware of termination), Coherence (context loss, derailment, reasoning–action mismatch), and Verification (premature termination, no/incorrect verification, weak verification).

Frontier SOTA split into two distinct verification styles. The Verification-side failures of SOTA fall into two opposite sub-modes. Opus is dominated by weak verification (36% vs. GPT’s 10%): it performs a check, but the check is too shallow to catch the failure. GPT is instead dominated by no/incorrect verification (47% vs. Opus’s 20%), as it tends to skip the check altogether. GLM-5 and DeepSeek-V4-Pro track the GPT pattern, with somewhat noisier execution (Execution share 28–31%, against Opus’s 16% and GPT’s 23%).

CLI-Universe-32B’s failure profile shifts to the execution side. For CLI-Universe-32B, the Verification share drops to 27% while Execution becomes the largest class at 44%. The most prominent shift is step repetition, which rises from 0–7% in the frontier baselines to 23%. The failure mode is qualitatively different from the SOTA pattern: rather than skipping verification, the agent more often gets stuck during execution, looping on the same operation or failing to make stable progress on the task.

### 5 Conclusion

We presented CLI-Universe, a pipeline that constructs terminal-agent training tasks through structured capability specification, evidence-guided deep research, and multi-stage executable verification. Finetuning Qwen3 on the resulting trajectories yields consistent gains across 8B, 14B, and 32B students, and transfers to out-of-domain benchmarks including BFCL-v4 and VitaBench. These results suggest that carefully constructed and independently verified tasks can provide strong supervision for terminal agents even at limited data scale.

### 6 Limitations

Several aspects of this work deserve further scrutiny. The pipeline relies on LLM-based agents for ideation, environment construction, solution generation, and test construction; despite rubric gating and executable verification, the quality ceiling of the synthesized data is ultimately bounded by the capability of the underlying models. Although CLI-Universe substantially narrows the gap to proprietary systems at ≤32B scale, a clear performance gap to the strongest frontier models remains, and we have not yet explored whether larger open-source base models or reinforcement-learning-based training can further close this gap. Finally, although CLI-Universe demonstrates strong data efficiency, the current dataset contains only 6k trajectories; scaling the pipeline to larger and more diverse task pools may unlock further gains, and we leave this exploration to future work.

### Contributions

Authors Zhanbo Hua3,∗, Yifan Yao1,∗, Weihao Xie4,∗, Yongchi Zhao2,∗, Minghao Liu3,∗, Ruizhi Qiu1, Zhewei Huang2, Zun Wang5, Yiyan Ji1, Yunhai Ye3, Letian Zhu1, Xinping Lei1, Han Li1, Zhiyuan Ma4, Zili Wang2, Zhaoxiang Zhang1, Jiaheng Liu1,†

Affiliations 1Nanjing University, 2StepFun, 3ZODA, 4Huazhong University of Science and Technology, 5Shanghai AI Lab

∗ Equal contribution. † Corresponding author.

### References

Anthropic. Claude code: Best practices for agentic coding. https://www.anthropic.com/engineering/

claude-code-best-practices, 2025a. OpenAI. Codex CLI. https://github.com/openai/codex, 2025a. Google. Gemini CLI. https://github.com/google-gemini/gemini-cli, 2025. OpenCode Team. Opencode: The open source ai coding agent. https://opencode.ai, 2025. SWE-agent Team. mini-swe-agent. https://github.com/SWE-agent/mini-swe-agent, 2025. Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi

Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents, 2025. URL https://arxiv.org/abs/2407.16741.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Jesse Hu, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026. URL https://arxiv.org/abs/2601.11868.

Siwei Wu, Yizhi Li, Yuyang Song, Wei Zhang, Yang Wang, Riza Batista-Navarro, Xian Yang, Mingjie Tang, Bryan Dai, Jian Yang, and Chenghua Lin. Large-scale terminal agentic trajectory generation from dockerized environments, 2026. URL https://arxiv.org/abs/2602.01244.

Renjie Pi, Grace Lam, Mohammad Shoeybi, Pooya Jannaty, Bryan Catanzaro, and Wei Ping. On data engineering for scaling llm terminal capabilities, 2026. URL https://arxiv.org/abs/2602.21193.

Zhiyuan Fan, Tinghao Yu, Yuanjun Cai, Jiangtao Guan, Yun Yang, Dingxin Hu, Jiang Zhou, Xing Wu, Zhuo Han, Feng Zhang, and Lilin Wang. Toward scalable terminal task synthesis via skill graphs, 2026. URL https://arxiv.org/abs/2604.25727.

Kanishk Gandhi, Shivam Garg, Noah D. Goodman, and Dimitris Papailiopoulos. Endless terminals: Scaling rl environments for terminal agents, 2026. URL https://arxiv.org/abs/2601.16443.

Yusong Lin, Haiyang Wang, Shuzhe Wu, Lue Fan, Feiyang Pan, Sanyuan Zhao, and Dandan Tu. Cli-gym: Scalable cli task generation via agentic environment inversion, 2026. URL https://arxiv.org/abs/ 2602.10999.

Kaijie Zhu, Yuzhou Nie, Yijiang Li, Yiming Huang, Jialian Wu, Jiang Liu, Ximeng Sun, Zhenfei Yin, Lun Wang, Zicheng Liu, Emad Barsoum, William Yang Wang, and Wenbo Guo. Termigen: High-fidelity environment and robust trajectory synthesis for terminal agents, 2026. URL https://arxiv.org/abs/ 2602.07274.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 48371– 48392. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, and Yunke Zhao. Vitabench: Benchmarking llm agents with versatile interactive tasks in real-world applications, 2025. URL https://arxiv.org/abs/2509.26490.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering, 2024. URL https://arxiv.org/abs/2405.15793.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. URL https://arxiv.org/abs/2310.06770.

Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenxiao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Lin Sui, Xinjie Sun, Flood Sung, Yunpeng Tai, Heyi Tang, Jiawen Tao, Qifeng Teng, Chaoran Tian, Chensi Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Si Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Haoning Wu, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Jing Xu, Jing Xu, Junjie Yan, Yuzi Yan, Hao Yang, Xiaofei Yang, Yi Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Siyu Yuan, Haobing

- Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian
- Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng, Shaojie Zheng, Longguang Zhong, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2026. URL https://arxiv.org/abs/2507.20534.

Anthropic. The claude model card. https://docs.anthropic.com/en/docs/about-claude/models, 2025b.

Google DeepMind. Gemini model thinking updates, march 2025. https: //blog.google/innovation-and-ai/models-and-research/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025.

OpenAI. GPT-5 system card. https://openai.com/index/gpt-5-system-card, 2025b.

GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

MiniMax, Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changqing Yu, Chao Wang, Cheng Ma, Cheng Zhong, Cheng Zhu, Chengjun Xiao, Chengyi Yang, Chengyu Du, Chenyang Zhang, Chi Zhang, Chuangyi Huang, Chunhao Zhang, Chunhui Du, Chunyu Zhao, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dongyu Zhang, Enhui Yang, Fei Yu, Guang Zheng, Guodong Zheng, Guohong Li, Haichao Zhu, Haigang Zhou, Haimo Zhang, Han Ding, Hao Zhang, Haohai Sun, Haolin Lyu, Haonan Lu, Haoyu Wang, Huajie Shi, Huiyang Li, Jiacheng Chen, Jian Zhang, Jiaqi Zhuang, Jiaren Cai, Jiaxin Pan, Jiayao Li, Jiayuan Song, Jichuan Zhang, Jie Wang, Jihao Gu, Jin Zhu, Jingwei Dong, Jingyang Li, Jingyu Zhang, Jingze Zhuang, Jinhao Tian, Jinli Liu, Jinyi Hu, Jun Tao, Jun Zhang, Junbin Ruan, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kang Xu, Ke Ji, Ke Yang, Kecheng Xiao, Keyu Duan, Keyu Li, Le Han, Letian Ruan, Li Yuan, Lianfei Yu, Liheng Feng, Lijie Mo, Lin Li, Lingye Bao, Lingyu Yang, Lingyuan Zhou, Loki, Lu Chen, Lunbin Ceng, Ming Li, Ming Zhong, Mingliang Tao, Mingyuan Chi, Mujie Lin, Nan Hu, Ningxin Chen, Peiyin Zhu, Peng Gao, Pengcheng Gao, Pengfei Li, Penglin Li, Pengyu Zhao, Qibin Ren, Qidi Xu, Qihan Ren, Qile Li, Qin Wang, Quanliang Chen, Qunhong Ceng, Rong Tian, Rui Dong, Ruitao Leng, Ruize Zhang, Shanqi Liu, Shaoyu Chen, Sheng Jia, Shun Yao, Shuoran Zhao, Shuqi Yu, Sichen Li, Sicheng Pan, Songquan Zhu, Tengfei Li, Tian Xie, Tiancheng Qin, Tianrun Liang, Wei Liu, Weiqi Xu, Weitao Li, Weixiang Chen, Weiyu Cheng, Weiyu Zhang, Wenhu Chen, Wenqian Zhao, Xiancai Chen, Xiangjun Song, Xiangyuan Wang, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xiaojie Wu, Xihao Song, Xingyi Han, Xinyu Guan, Xuan Lu, Xun Zou, Xunhao Lai, Xutong Li, Yan Gong, Yang Wang, Yang Xu, Yangsen Wang, Ye Tang, Yicheng Chen, Yinran Qiu, Yiqi Shi, Yiting Guo, Yiwen Huang, Yixuan Wang, Yongyi Hu, Yu Gao, Yu Zhang, Yuanxiang Ying, Yuanzhen Zhang, Yubo Wang, Yuchen Song, Yufeng Yang, Yuhang Meng, Yuhang Miao, Yuhao Li, Yujie Liu, Yulin Hu, Yunan Huang, Yunji Li, Yunyi Huang, Yusen Zhang, Yusu Hong, Yutao Xie, Yutong Zhang, Yuwen Liao, Yuxuan Shi, Yuze Wenren, Zebin Li, Zehan Li, Zejian Luo, Zeyu Jin, Zeyuan Sun, Zhanpeng Zhou, Zhaochen Su, Zhendong Li, Zhengmao Zhu, Zhengyuan Peng, Zhenhua Fan, Zhi Zhang, Zhichao Xu, Zhiheng Lv, Zhikang Xu, Zhitao He, Zhiwei He, Zhongyuan Li, Zibo Gao, Zijia Wu, Zijian Song, Zijian Zhou, Zijun Sun, Zishan Huang, Ziying Chen, and Ziyue Ge. The minimax-m2 series: Mini activations unleashing max real-world intelligence, 2026. URL https://arxiv.org/abs/2605.26494.

Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, Zeyao Ma, Kashun Shum, Xuwu Wang, Jinxi Wei, Jiaxi Yang, Jiajun Zhang, Lei Zhang, Zongmeng Zhang, Wenting Zhao, and Fan Zhou. Qwen3-coder-next technical report, 2026. URL https://arxiv.org/abs/2603.00729.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence. https:// huggingface.co/deepseek-ai/DeepSeek-V4-Pro, 2026. Technical Report.

### A Task Taxonomy

We describe each terminal-agent task in CLI-Universe along four orthogonal dimensions, distilled from patterns we observe across TB-Agent tasks: where the task lives (Domain), what specialized knowledge the agent must apply (Skill Types), what reasoning behaviors the task elicits (Capabilities), and what engineering activity the agent is asked to perform (Engineering Pillars). CLI-Universe uses this taxonomy to seed the random task-candidate assignment in its synthesis pipeline (Section 3.2), so that two tasks in the same domain still differ in the underlying capability pattern they exercise.

#### A.1 Domain

The application area a terminal-agent task belongs to. Domains are human-curated and act as the entry point of the synthesis pipeline; each domain carries its own pool of allowed values on the other three dimensions, so the random combinations drawn from them remain realistic within that domain.

Table 3: Domains in CLI-Universe.

Domain Description

Software Engineering

General-purpose programming tasks: writing, modifying, or debugging application code in mainstream languages, including standalone scripts, libraries, and small services.

Debugging Localizing and repairing defects in existing artifacts—crashes, broken builds, corrupted state—where the agent must read failure signals to produce a targeted fix rather than write new functionality.

System Administration

OS-level management tasks: process and service control, filesystem and permission management, scheduled jobs, and inspecting or repairing a running system.

File Operations Manipulating, recovering, or transforming files and on-disk artifacts: extracting structured content from binary formats, large-scale text edits, or repairing corrupted containers, where filesystem- and format-level reasoning dominate.

Security Penetration-testing, capture-the-flag-style exploitation, secure-configuration audits, and vulnerability analysis where the agent must reason about attacker and defender behavior.

Data Processing ETL pipelines, log and text mining, schema reconciliation, and data wrangling across heterogeneous structured or semi-structured sources.

Data Querying Authoring queries against structured data backends such as SQL, SPARQL, or other declarative interfaces, where the difficulty lies in expressing intent in the query language rather than in transforming data downstream.

Data Science End-to-end analysis pipelines on real datasets: model evaluation harnesses, statistical inference, embedding and retrieval workflows, and reproducible analyses that combine data preparation with downstream interpretation.

Scientific Computing Numerical simulation, scientific data analysis, and computation-heavy workflows where correctness depends on physical or mathematical modeling rather than software engineering alone.

Mathematics Derivation- or proof-heavy tasks such as cryptanalysis, numerical linear algebra, and model-theoretic reasoning, where progress depends on mathematical insight before any code is written.

Optimization Formulating and solving constrained optimization problems—portfolio allocation, scheduling, parameter tuning—where the agent must select a solver and encode constraints faithfully.

Machine Learning Training, evaluation, and inference of models, including data preparation, hyperparameter sweeps, and small-scale serving setups.

Model Training Practical training and fine-tuning workflows: launching runs, recovering interrupted checkpoints, and scripting CLI training entry points, where the work centers on the training loop and its operational concerns.

Video Processing Frame-level video analysis and transformation, including extracting structured information from video streams and producing synchronized outputs across visual and audio modalities.

Table 3 – continued. Domain Description Web & API Tooling Building or interacting with web services and browser-side flows: HTTP APIs, scraping,

headless-browser automation, and integration with external services.

Games Game-state reasoning and search-driven decision-making, where the agent implements or invokes engines that explore move spaces under explicit rules.

Personal Assistant User-facing tasks such as scheduling, planning under preferences and constraints, and producing concise actionable outputs from messy inputs.

#### A.2 Skill Types

The specialized technical knowledge a terminal-agent task requires the agent to apply. Each task is tagged with the single load-bearing skill type; auxiliary skills (e.g. light shell glue around an algorithmic core) are not counted.

Table 4: Skill Types, distilled from patterns observed across TB-Agent tasks.

Skill Type Description Algorithmic Non-trivial algorithm or data-structure design where the core difficulty is the algorithm

itself — choosing, adapting, or combining primitives such as graph traversal, dynamic programming, custom indexing, or careful complexity reasoning. Routine glue around standard library calls does not qualify.

Data Processing Parsing, transforming, joining, or aggregating structured or semi-structured data (CSV/JSON/logs, tabular pipelines). The difficulty lies in handling schema variations, malformed records, or composing standard data operations correctly, rather than in inventing a new algorithm.

Systems Low-level OS, process, filesystem, or performance work whose correctness depends on system semantics — signals, file descriptors, scheduling, memory layout, or precise concurrency behavior. Tagged when the agent must reason about how the OS or runtime actually behaves, not when it merely calls a high-level API that happens to live near the system.

Configuration Authoring or repairing configuration for tools, services, or build systems (Dockerfiles, systemd units, CI manifests, package or compiler config). The difficulty is in knowing the right knob and how knobs interact, not in writing program logic.

Shell Scripting End-to-end automation expressed primarily in shell, including control flow, pipelines, text-stream tools (awk, sed, jq), and quoting/escaping correctness. Tasks that only use shell to invoke a single Python or Node script do not qualify.

Mathematical Numerical or symbolic computation whose correctness rests on mathematical reasoning

— probability, linear algebra, signal processing, optimization, or numerical stability. Tagged when the agent must derive or pick the correct formulation, not when it merely plugs numbers into a known formula.

Deployment Packaging, building, releasing, or wiring up runtime infrastructure so that an artifact can actually run end-to-end: image building, dependency pinning, service registration, environment bootstrapping, or release orchestration.

Cryptography Cryptographic or cryptanalytic reasoning, including key recovery, protocol analysis, primitive misuse, and side-channel reasoning. Routine TLS or SSH configuration without analytic content is treated as Configuration, not Cryptography.

#### A.3 Capabilities

The reasoning behaviors a terminal-agent task is intended to elicit during the agent’s interaction with the environment. Each capability follows a strict TRIGGER / SIGNATURE / NOT rule applied to the trajectory, and a task may carry multiple capabilities.

Table 5: Capabilities, distilled from patterns observed across TB-Agent tasks.

Capability Description Exploration Probes an unfamiliar workspace (ls, cat, find, grep) before committing to a plan,

because the task does not hand the agent a complete map of the environment and key affordances must be discovered from the filesystem itself.

Decomposition Enumerates sub-goals up front, before any irreversible action, so that subsequent steps execute against an explicit plan. Tagged when the trajectory shows a planning utterance with discrete sub-goals, not just a vague intent.

Error Recovery Reads an explicit failure signal (traceback, exit code, test output, log line) and revises the plan accordingly, rather than retrying the same action or escalating without diagnosis.

Specification Adherence

Returns to the original spec late in the trajectory to verify that the produced artifact matches the required format — exact file path, output schema, argument signature catching cases where intermediate code drifts from the literal contract.

Working Memory Carries ≥ 5 mutually dependent facts simultaneously across 10+turns without losing them, e.g. tracking variable bindings, file contents, and prior tool outputs that all need to remain consistent for the next step to succeed.

Long-Horizon Planning

Sustains a coherent plan with inter-step dependencies over more than 10 meaningful turns, where premature commitment to an early step closes off later ones.

Constraint Satisfaction

Maintains ≥ 2 hard rules that pull in opposite directions, so progress on one risks breaking another — e.g. a strict latency budget alongside a memory ceiling, or a format requirement that forbids the very tool the agent would otherwise reach for.

Modality Translation Translates information across semantic modalities, such as image→code (rendering a diagram into a script), binary→source (recovering structure from a stripped artifact), or table→schema (inferring types and constraints from instance data).

Reverse Engineering Reconstructs hidden structure or intent from an opaque artifact, e.g. figuring out the calling convention of an undocumented binary, or inferring a service protocol from packet captures.

Mathematical Derivation

Produces a closed-form derivation or proof before any code is written, so the resulting implementation reflects a pre-justified formula rather than an empirically tuned approximation.

Adversarial Reasoning

Reasons against the designer’s intent — bypassing intended controls, finding exploitable corner cases, or mapping the attack surface of a system the agent is supposed to defend or analyze.

#### A.4 Engineering Pillars

The engineering pillar a terminal-agent task falls under — whether building something new, fixing something broken, or restructuring something existing. Each task is tagged with a single dominant pillar.

Table 6: Engineering Pillars, distilled from patterns observed across TB-Agent tasks.

Pillar Description New feature creation Build a new artifact — script, module, or service — from scratch against a spec. The

agent’s primary work is to produce code that did not previously exist, rather than to modify or audit existing code.

Debugging Localize a defect from a failure signal and produce a minimal patch (error → root-cause

→ fix). The deliverable is a small, targeted change to existing code, not a rewrite.

Systems programming

Low-level, OS-near, or performance-critical work where the deliverable lives close to the system layer — a syscall wrapper, a kernel-adjacent utility, or a hand-tuned hot path and where systems-level concerns dominate the design rather than appear incidentally.

DevOps Build, deploy, configure, or wire up runtime infrastructure end-to-end so that a target service or pipeline can actually run, including container/image work, CI/CD glue, and environment provisioning.

Feature iteration Extend or modify an existing codebase with a scoped new behavior, leaving the surrounding architecture intact. The change is additive or local rather than structural.

Large-scale refactoring

Cross-file structural change that preserves behavior — e.g. extracting an interface across many modules, renaming a pervasive concept, or reorganizing a directory layout while keeping all tests green.

### B Training Details

We fine-tune Qwen3 models using multi-turn SFT. Hyperparameters and hardware setup are summarized in Table 7. Unless otherwise noted, all trajectories (including failed and incomplete ones) are retained; the effect of trajectory selection is studied in Section 4.2.2.

Table 7: Training hyperparameters and hardware setup for SFT of Qwen3 students.

Hyperparameter Value

Optimizer AdamW (β2 = 0.95) Peak learning rate 1 × 10−5 LR scheduler cosine_with_min_lr (min 1 × 10−6) Warmup ratio 0.03 Weight decay 0.01 Max gradient norm 1.0 Epochs 5 Per-device batch size 1 Gradient accumulation steps 4 Context length 64K Precision bf16 Sequence parallelism degree 2 GPUs 32 × NVIDIA H200 Global batch size 64

### C Failure Mode Examples

Each failure mode is illustrated with one CLI-Universe-32B trajectory on Terminal-Bench 2, shown as a turn-level timeline. A “turn” is one agent action cycle (reasoning → command → observation).

#### C.1 Disobey Task Specification (Execution)

Task (sam-cell-seg): Write convert_masks.py. Task specifies: --output_path /app/test_output.csv (a

.csv file path). 21 turns.

Turn Agent Action Failure Relevance 6 Drafts: os.path.dirname(

Correctly treats output_path as file

args.output_path) + result.to_csv(args.output_path)

12 Rewrites: os.makedirs( args.output_path) + os.path.join(args.output_path, ...)

✗ Switches to directory handling

12 Tests with

✗ Tests with directory, not task’s /app/test_output.csv

--output_path /app/output

13–16 Same os.makedirs + os.path.join across 4 rewrites

✗ Directory convention persists 13–16 Each rewrite tested with

✗ Never tests with the task’s file path 20 task_complete: true,

--output_path /app/output

Spec violation never exposed

“The task is fully complete”

Early draft (Turn 6) had correct file-path handling. From Turn 12 onward, all landed versions use directory semantics, and all 4 test invocations used /app/output, and the task’s .csv file path was never tested.

#### C.2 Step Repetition (Execution) Task (build-pov-ray): Download and compile POV-Ray 2.2 from source. 357 turns.

∼1–30 curl -s ’http://.../download/’

No download link found

| grep -o ’http[ˆ"]+’ | head -20

∼100 Same curl command ✗ Same empty result, ∼100th attempt

- ∼150 Agent reasons: “switch to archive.org” Strategy change in reasoning only
- ∼151 curl -s ’http://.../download/’ ... ✗ Same command despite reasoning ∼1–357 find / -maxdepth 5 -type f

✗ Same filesystem state each time ∼1–357 grep -rl ’2\.2’

| grep -i ’pov’ ×151

✗ Same files listed each time

/app/povray-source/ ×44

Final curl ×165, find ×151, grep ×44 Same commands loop without convergence

Reasoning mentions strategy changes, but actual commands never change. The same curl/find/grep loop runs 44–165 times each across 357 turns.

#### C.3 Unaware of Termination (Execution)

Task (compile-compcert): Compile CompCert. Environment has hard 120s timeout (exit 124). 159 turns, 0 successes.

Turn Agent Action Failure Relevance ∼1 opam init --compiler=4.05.0

First timeout (hard 120s ceiling)

→ exit 124

∼5 Requests duration 60s → exit 124 ✗ Ceiling unchanged ∼10 Requests duration 120s → exit 124 ✗ Still 120s ceiling 12 echo hello (duration 2s)

Even trivial commands time out (systemic issue)

→ exit 124

∼30 Requests duration 300s → exit 124 ✗ Escalation begins ∼60 Requests duration 1200s → exit 124 ✗ 10× original, same result ∼80 Agent: “environment is extremely slow”

✗ Wrong diagnosis ∼100 Writes one-shot /tmp/build.py

→ duration 3600s

✗ Condensing into one script doesn’t help ∼120 Duration 36000s (10 hours)

→ exit 124

✗ ∼146 Duration 72000s (20 hours)

→ exit 124

✗ Final attempt, 600× original timeout

→ exit 124

159 turns, 163 commands, 0 successes. Agent escalates duration from 1s to 72,000s, but the hard 120s ceiling never changes. Even echo hello fails.

##### C.4 Context Loss (Coherence) Task (make-mips-interpreter): Build a MIPS interpreter in JavaScript to run a DOOM binary. 300 turns.

Facts first read

5–6 readelf -h/l/s on /app/doomgeneric_mips → “I have the function addresses and entry point”

10–11 Re-runs readelf -s | grep ’__start’

Had to re-check: didn’t retain numeric value from T6

→ 0x400110

13 Summarizes: __start at 0x400110, BSS 0x004a6ba0–0x004b0a40

Facts stated explicitly 22 Hardcodes TEXT_START, DATA_START,

Constants committed to file 49–64 Re-runs readelf, re-checks entry point,

BSS_START, STACK_START in vm.js

✗ Re-checking facts already established at T11–13

re-states 0x400110 + BSS layout

96–103 Re-reads f.seek(0x400110) bytes, re-runs readelf, rewrites vm.js with same constants

✗ Same facts re-derived again

131 “I defined read8 but forgot to

Bug diagnosed, fix attempted 145 “I defined read8 but forgot to define write8” again

define write8” → patches with sed -i

✗ Same diagnosis, same sed patch, 14 turns later

170–178 Three consecutive turns re-examine 0x439ad0 / 0xa0450000, re-argue endianness, re-derive big-endian reads

✗ Same evidence, same conclusion, 3 times

298–300 Summarizes ELF structure and entry point

✗ Same parser that should have been the base throughout

→ rewrites vm.js with same e_phoff/e_entry parser

The agent repeatedly loses established facts across turns: entry point 0x400110 is re-checked via readelf at least 4 times, write8 bug is diagnosed and patched twice, endianness is re-derived 3 times. Despite hardcoding constants in vm.js at Turn 22, the agent keeps re-running the same diagnostic commands to re-establish the same values.

- C.5 Task Derailment (Coherence) Task (regex-chess): Write /app/re.json, regex pairs for chess move generation via re.sub. 10 turns.

Turn Agent Action Failure Relevance 1 cat /app/check.py, understands

need for [regex, replacement] pairs

Reads requirement

2–3 python3 -c "import chess; for move in board.legal_moves: print(move.uci())"

✗ Diagnostic, enumerates moves

4–5 python3 -c "board.push(move); print(board.fen())"

✗ Diagnostic, FEN after each move 6–8 Manual FEN parsing, confusion

about rank ordering

✗ Diagnostic, notation study 9–10 More python3 probes on

castling, en passant

✗ /app/re.json never written

10 turns of diagnostic python-chess scripts. 0 turns of implementation. The deliverable is never started.

- C.6 Reasoning–Action Mismatch (Coherence) Task (feal-differential-cryptanalysis): Implement differential attack on 4-round FEAL. 35 turns.

- 2 Reads /app/feal.py; reasoning contains getright() + seed*1234567 brute-force

Forward brute-force present from the start

- 3 Reasons: “work backwards through round 4”, but embedded code still uses getright() + brute-force

✗ Reasoning says backward, code template is forward

- 4 Discovers differential: “0x80800000 → 0x02000000”

Correct differential found

- 5 Reasons: “the key differential property is...”, code still getright() + brute-force

- ✗ Backward reasoning, forward code

- 7 Writes attack.py, brute-force over large key space, too slow

- ✗ Still forward

- 10 Rewrites attack.py, still brute-force ✗ Still forward
- 11 Reasons: “My earlier analysis was wrong. Let me re-examine” + backward analysis

Retracts, starts over

14 Writes attack.py, forward search again ✗ Still forward 19 Reasons: “My earlier analysis was wrong”

Second retraction

again + backward analysis

22 Writes attack.py, forward search again ✗ Still forward 31 Reasons: “Now I understand the encryption

Third attempt at backward plan

structure” + backward analysis

35 Final attack.py: for seed in range(65536): key5 = (seed*1234567) + f_function(getright(plain[i]) ˆ key5)

###### ✗ Still forward brute-force

The mismatch is present from Turn 2: the agent’s reasoning text describes backward differential propagation, but the code template embedded in the same reasoning uses getright() + seed*1234567 forward brute-force. Across 35 turns and 3 retraction cycles, this contradiction never resolves.

#### C.7 Premature Termination (Verification)

Task (cobol-modernization): Modernize a COBOL program to Python, producing identical .DAT output files. 20 turns.

Turn Agent Action Failure Relevance 1–8 Reads COBOL source, analyzes

Understanding the program

record layouts

9 Writes /app/program.py, first version Book owner update uses offset:offset+4 (bytes 0–3, the book-ID field)

9 Should write to offset+24:offset+28 (the owner field)

✗ Wrong offset: overwrites book-ID instead of owner

11–16 Rewrites program.py 4 more times, book owner update still uses offset:offset+4

✗ Bug persists through all rewrites

17 Runs comparison: print(’Books equal:’, b1==b2), but b1 undefined (NameError)

✗ Verification script has variable scope bug, produces no valid result

17 Agent notes: “typo in variable names (a1, b1, t1 were not defined)”

Recognizes NameError but doesn’t fix and re-run

19–20 task_complete: true, declares success ✗ No valid verification; book-ID still corrupted in output

The book-owner update writes to the wrong byte offset (0–3 instead of 24–27) across all 5 versions of the code. The verification script has a NameError and produces no result. The agent acknowledges the NameError but declares completion without re-running a corrected verification, so the core bug (wrong offset) is never caught.

#### C.8 No or Incorrect Verification (Verification) Task (cancel-async-tasks): Async task queue with correct SIGINT/KeyboardInterrupt cleanup. 6 turns.

Turn Agent Action Failure Relevance

- 1 Writes /app/run.py, on SIGINT: queue None sentinels, await workers

Implementation has subtle race condition

- 2 Test v1: asyncio.create_task( run_with_interrupt())

✗ Structurally wrong, doesn’t simulate SIGINT

- 3 Test v2: asyncio.wait_for( run_with_interrupt(), timeout=0.5)

✗ Raises TimeoutError, not KeyboardInterrupt

- 4 Test v2 passes, “all 5 cleanups ran” ✗ TimeoutError path verified, not SIGINT path
- 5 “Cleanup verified via timeout simulation” → task_complete: true

✗ Claims SIGINT verified

- 6 Reaffirms completion Real bug: sentinels queued behind items → never exposed

3 test iterations, each testing the wrong exception path. asyncio.wait_for raises TimeoutError, not KeyboardInterrupt, so the real SIGINT race condition is never triggered.

#### C.9 Weak Verification (Verification)

Task (build-cython-ext): Build pyknotid 0.5.3 Cython extensions, numpy pinned to 2.3.0. Evaluator checks 11 properties. 26 turns.

Turn Agent Action Failure Relevance

1–2 Clones repo, checks numpy → 2.3.0 ✓numpy version checked 3 build_ext --inplace,

△ Partial build

cinvariants hangs/times out

4–9 Fixes np.float → float, np.complex → complex, gcd import

Numpy 2.0 compat

✓2 functions checked

10 README snippet test

→ returns 6.999999999999998

11–12 pytest tests/ → 18 passed ✓Test suite passes 13 pip install -e /app/pyknotid

△ May overwrite numpy 2.3.0 14–25 Tries .so file check + individual

--force-reinstall

✗ Malformed commands, never execute

Cython function tests

26 task_complete: true ✗ No re-check of numpy; cinvariants never verified

Verified 4/11 properties. --force-reinstall at Turn 13 invalidates the numpy check from Turn 1. cinvariants timed out during build and was never re-verified. Last verification commands had JSON escaping errors and never actually ran.

