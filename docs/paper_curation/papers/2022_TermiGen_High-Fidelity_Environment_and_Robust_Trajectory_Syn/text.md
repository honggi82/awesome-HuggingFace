## TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents

Kaijie Zhu1 Yuzhou Nie1 Yijiang Li2 Yiming Huang2 Jialian Wu3 Jiang Liu3 Ximeng Sun3 Zhenfei Yin4 Lun Wang5 Zicheng Liu3 Emad Barsoum3 William Yang Wang1 Wenbo Guo1

# arXiv:2602.07274v1[cs.AI]6Feb2026

### Abstract

Executing complex terminal tasks remains a significant challenge for open-weight LLMs, constrained by two fundamental limitations. First, high-fidelity, executable training environments are scarce: environments synthesized from realworld repositories are not diverse and scalable, while trajectories synthesized by LLMs suffer from hallucinations. Second, standard instruction tuning uses expert trajectories that rarely exhibit simple mistakes common to smaller models. This creates a distributional mismatch, leaving student models ill-equipped to recover from their own runtime failures. To bridge these gaps, we introduce TermiGen, an end-toend pipeline for synthesizing verifiable environments and resilient expert trajectories. TermiGen first generates functionally valid tasks and Docker containers via an iterative multi-agent refinement loop. Subsequently, we employ a Generator-Critic protocol that actively injects errors during trajectory collection, synthesizing data rich in error-correction cycles. Fine-tuned on this TermiGen-generated dataset, our TermiGenQwen2.5-Coder-32B achieves a 31.3% pass rate on TerminalBench. This establishes a new openweights state-of-the-art, outperforming existing baselines and notably surpassing capable proprietary models such as o4-mini. Dataset is avaiable at https://github.com/ucsb-mlsec/ terminal-bench-env.

### 1. Introduction

Recent advancements have empowered LLMs with agentic scaffolding and tool-use capabilities, enabling them to

1UC, Santa Barbara 2UC, San Diego 3AMD 4University of Oxford 5Google. Correspondence to: Kaijie Zhu <kaijiezhu@ucsb.edu>.

Preprint. February 10, 2026.

66

Proprietary Models Apex2 + Claude-4.5-Sonnet

General-purpose Base Models

64

Fine-tuned Models

62

Codex CLI + GPT-5-Codex

AvgPassRate(%)

40

Claude Code + Claude-3.7-Sonnet

BashAgent + TermiGen-Qwen2.5-Coder-32B

30

BashAgent + TermiGen-Qwen3-32B

Reptile + Devstral-2505-22B

Codex CLI + o4-mini

Terminus 2 + LiteCoder-30a3b

20

BashAgent + GPT-OSS-120B BashAgent + Qwen3-235B-A30B

Terminus 2 + TerminalAgent-32B

10

Terminus 1 + Qwen3-235B-A30B

BashAgent + Qwen3-32B

BashAgent + Qwen2.5-Coder-32B

0

22B 30B 32B 120B 235B N/A

Model Size (Billions of parameters)

Figure 1. Average pass rate on Terminal Bench (%) vs. model size. TermiGen models (bold) outperform other open-source baselines and approach proprietary system performance at 32B scale. Blue refers to proprietary models, green refers to general-purpose base models, and red refers to fine-tuned models.

tackle complex tasks through interaction with real-world environments (Zhou et al., 2023; Jimenez et al., 2023; Xie et al., 2024; Terminal-Bench Team, 2025). Among these domains, executing software engineering and system administration tasks via the terminal (i.e., terminal tasks) stands out as one of the most fundamental yet challenging frontiers (Terminal-Bench Team, 2025). Distinct from standard code generation (Chen, 2021; Huang et al., 2024) or math reasoning (Cobbe et al., 2021), terminal tasks pose two core challenges and have thus far been dominated by large proprietary models backed by extensive private training data.

First, scalable construction of executable training environments remains a major bottleneck. Unlike individual code snippets, terminal tasks require a complete, executable specification of system architecture, file dependencies, and dynamic runtime status. Constructing such complex and verifiable environments necessitates substantial effort from domain experts. TerminalBench (TerminalBench Team, 2025), for instance, required extensive manual curation to produce merely 200 tasks. Existing data synthesis approaches generally fall into two paradigms, each with significant limitations. The first paradigm adapts exist-

ing software repositories (e.g., mining GitHub issues (Yang et al., 2025b)). While grounded in real code, these sources often lack diversity and are biased towards debugging and issue resolving (Liu et al., 2024). They also require extensive manual effort to define verifiable success criteria for each task. The second paradigm involves purely LLM-based synthesis, where models are prompted to generate hypothetical execution logs without access to a live runtime (Chen et al., 2025; Kimi Team, 2025). However, as our experiments reveal (Section 4.3), this approach suffers from severe hallucinations: without grounded execution, the synthesizer invents plausible but technically incorrect outcomes (e.g., reporting successful installation despite dependency conflicts), which can mislead student models during training.

Second, long-horizon terminal execution worsens exposure bias. Completing a terminal task often requires executing long command sequences where a single intermediate mistake can cascade into an irreversible failure. Standard distillation methods typically perform rejection sampling (Mitra et al., 2024) on expert models to harvest optimal execution paths. However, for long-horizon terminal tasks, we find that success crucially depends on the ability to detect, critique, and correct one’s own errors. Since expert models rarely make simple mistakes, standard datasets suffer from a distributional mismatch, leaving student models ill-equipped to handle runtime failures. Consequently, student models suffer from a distribution shift during deployment, they frequently encounter trivial errors (e.g., dependency conflicts) that were absent during training, lacking the requisite supervision to recover from these failure states.

We aim to close the significant gap between open-weight agents and large proprietary models in the terminal tasks. We approach this challenge from a data-centric perspective, i.e., an automated, scalable pipeline for high-fidelity environment synthesis is the foundational prerequisite for effective training. Guided by this insight, we propose TermiGen, an end-to-end data generation recipe specifically designed for developing robust terminal agents. At a high level, TermiGen bridges the data scarcity gap by first synthesizing verifiable tasks via a multi-stage framework, and subsequently employing a controlled error-correction mechanism to collect realistic expert trajectories that facilitate resilient learning. Specifically, for environment synthesis, we employ a multi-agent system to decompose the generation process, ensuring coverage across diverse task categories. Crucially, to eliminate the hallucinations common in simulation-based approaches, we introduce an interactive execution loop that validates environment correctness and task solvability within a real Docker container before data collection begins. Following environment generation, we propose a novel trajectory collection protocol designed to mitigate the distributional mismatch in the standard distillation process. To incentivize resiliency, we employ a

Generator-Critic architecture to generate error-correction trajectories in a controlled manner. At each action step, we stochastically inject realistic faults into the trajectory and condition the agent to diagnose the failure and recover the system state subsequently. This process yields training data rich in explicit error → diagnosis → correction cycles, teaching the model how to recover from runtime mistakes.

We conduct extensive evaluations to validate our framework. First, our TermiGen-Qwen2.5-Coder-32B model achieves a pass rate of 31.3% on TerminalBench, establishing a new state-of-the-art for open-weights models and outperforming existing fine-tuning baselines like Reptile by over 12%. Notably, it even surpasses capable proprietary models such as o4-mini with Codex CLI, demonstrating that high-quality data can enable smaller models to rival larger counterparts in specialized domains. Second, we demonstrate that training on grounded, verifiable environments significantly outperforms simulation-based baselines. Third, our ablation studies reveal that the inclusion of error-correction trajectories improves pass rates by a substantial margin compared to training solely on standard expert trajectories. Last, we show that including error trajectories which fail to reach the task goal also improves performance. To the best of our knowledge, TermiGen is the first end-to-end recipe for training terminal agents that integrates a high-fidelity, robust, and scalable data synthesis pipeline.

### 2. Existing Works and Limitations

Environment for Agentic Training. At a high level, existing works for constructing agent environments fall into two primary categories. The first line of methods focuses on automatically constructing executable environments from existing code repositories, utilizing either a semi-automated pipeline (Yang et al., 2025b) or a fully automated pipeline (Zhang et al., 2025). However, the dependency on pre-existing projects constrains their generalizability to more diverse and unseen tasks and environment settings. Another line of work employs LLMs as simulated environments to directly generate feedback (Chen et al., 2025; Kimi Team, 2025; Li et al., 2025; Wang et al., 2025), thereby facilitating large-scale data synthesis and reinforcement learning. While simulation reduces reliance on real environments, the generated feedback can be inconsistent with actual execution (e.g., state-tracking errors or hallucinated effects), which may mislead agent training. In contrast, our framework generates task-specific Docker-based environments that support actual execution, ensuring that agents are trained in diverse, grounded, and realistic environments.

Agentic Supervised Fine-Tuning and Learning from Failure. Standard approaches for training agentic small models typically rely on SFT over expert trajectories distilled from frontier models (Chen et al., 2023; Zeng et al., 2024). While

Phase II: Controlled Error-

Phase I: High-fidelity Task and Environment Synthesis

Correction Traj Collection

Input

Output

Input Categories and Seed Tasks Output Synthetic Training Data

[Figure 1]

Synthetic Training Data

[Figure 2]

Distilled

Software Build

[Figure 3]

Porting &

Data

Trajectory

& Compilation

Bug Fixing

Processing

Other Categories

Task

Source

Unit

System Administration

Config Linux

Ghidra

CUDA Toolkit

Other seeds

Dockerfile

Target Agent

Kernel

Decompilation

Install

Description

Files

Tests

80%

20% Incorrect Step Generation

Correct Step

Generation

Yes

Candidate

Candidate

Step

Step

2. Environment Synthesis

1. Task Creation

###### 3. Env & Task Validation

(Correct)

(Incorrect)

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Dockerfile

File

[Figure 8]

Required

No

Files List

Generator

Analyzer

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Dockerfile

Execution

[Figure 13]

###### Judge Agent

Discard &

[Figure 14]

Judge

Yes Finish?

[Figure 15]

[Figure 16]

Retry

Source

(Validator)

Task

[Figure 17]

Unit Test

Seed

File Content

Files

Description

Generator

Generator

[Figure 18]

(code, data,

Judge

Unit Tests

etc.)

Figure 2. The overall pipeline of TermiGen. Phase I generates diverse, functionally valid tasks within Docker containers via iterative refinement (Section 3.2). Phase II synthesizes robust expert trajectories by actively injecting errors into the execution flow, enabling the model to learn error diagnosis and recovery (Section 3.3).

effective for cloning optimal behaviors, this paradigm suffers from exposure bias (Ross et al., 2011; Bengio et al., 2015): since experts rarely make simple mistakes, the resulting datasets lack failure states and corresponding recovery phases. Consequently, student models trained solely on expert trajectories become fragile, failing catastrophically when they inevitably encounter runtime exceptions. To address this, recent works utilize execution feedback or leverage iterative self-refinement to guide self-correction (Shinn et al., 2023; Madaan et al., 2023). However, these methods predominantly focus on inference-time refinement rather than model training. Although some efforts explicitly train on error-correction data (Zheng et al., 2024), they are often limited to single-turn, static code repair tasks. Our framework advances this direction by integrating active error injection into the trajectory synthesis pipeline, enabling the model to internalize the full error → diagnosis → recovery cycle within complex, multi-turn terminal interactions.

rity forensics, and infrastructure management unaddressed. Models trained on these limited tasks exhibit restricted reasoning capabilities for broader tasks and fail to utilize a diverse range of terminal operations. Attempts to bypass this via simulated execution face a fundamental fidelity gap, as the synthesized trajectories may not represent the real execution traces and may contain hallucinations. A typical issue is state inconsistency. For example, if an agent modifies a file (e.g., changing a threshold in config.py), simulators often fail to track this persistent change, hallucinating outcomes based on the original version. This necessitates a pipeline that ensures both executability (to avoid hallucinations) and scalability (to cover diverse domains).

❷ Necessity of Error-correction Training. Software and system-related terminal tasks typically require long action sequences (often > 20 steps), where any single step can fail. However, standard expert demonstrations used for training are typically sparse in errors, as strong teacher models rarely make the trivial mistakes that smaller student models encounter. Consequently, agents trained on such data suffer from exposure bias: they lack the resilience to correct runtime exceptions. When an error occurs during deployment, the resulting failure state (e.g., confusing stderr, corrupted file systems) represents an out-of-distribution (OOD) scenario relative to the training data. Self-correction is particularly critical for such agentic tasks, as external tool calls induce state changes that are inherently more difficult to reverse than internal reasoning errors. This necessitates training on imperfect expert demonstrations that explicitly include error-correction sequences, which show not only correct execution paths, but also how to recognize mistakes and patch them through self-correction.

### 3. Key Technique

#### 3.1. Overview

Technical Challenges. Training agentic small models for terminal agents has two unique challenges compared to standard code and math reasoning tasks.

❶ Scarcity of Diverse, Executable Environments. Unlike self-contained coding tasks (e.g., LeetCode), terminal tasks are environment-dependent, requiring specific OS configurations and runtime dependencies. Constructing such environments manually is prohibitively expensive, forcing existing benchmarks to focus narrowly on code debugging while leaving critical domains like system administration, secu-

Insight of TermiGen. In this work, we propose TermiGen, the first end-to-end framework for training effective terminal agents with the self-refinement capability. Our method consists of the following two phases, designed to address the aforementioned challenges. Figure 2 shows an overview of TermiGen pipeline.

- Phase I: High-fidelity Task and Environment Synthesis. We implement a multi-agent framework to decompose the generation process into controllable stages. Guided by a structured taxonomy (spanning 11 categories across three tiers), we first instruct an LLM to generate distinct task seeds to ensure broad coverage across domains. These seeds are concise, high-level objective descriptions. Subsequently, we employ a Proposer-Evaluator architecture to expand these seeds into fully specified tasks (e.g., adding detailed input/output requirements). In this loop, the Evaluator validates the generated content against predefined criteria, providing feedback to guide iterative refinement. We then leverage a structured workflow to translate these textual descriptions into executable Dockerfiles. Given the complexity of these domains, ensuring environment correctness and task validity is non-trivial. We ensure data quality through a two-stage automated validation pipeline. First, to guarantee environment executability, we implement an iterative Docker build loop where build failures are automatically diagnosed and corrected. Second, to verify task solvability, we employ a Generator-Judge framework: a Generator Agent synthesizes initial unit tests, which are then iteratively validated and refined by a Judge Agent to ensure they accurately capture task objectives and provide reliable execution feedback.

- Phase II: Controlled Error-Correction Trajectory Collection. Following environment verification, we proceed to trajectory synthesis. To address the exposure bias inherent in standard distillation (as discussed in Challenge ❷), we introduce an active error injection mechanism. We employ a Generator-Critic architecture that stochastically injects controlled failures sampled from a taxonomy of five common failure modes (Section 3.3). The generator operates under a mixed policy: at each step, it probabilistically determines whether to pursue the optimal path or intentionally deviate to commit a specific error (e.g., invoking non-existent arguments in a command). A Critic Agent validates the alignment between actions and intents, ensuring that injected errors trigger informative feedback (stderr) while correct steps meaningfully advance the task. Crucially, whenever the policy dictates a return to generate a correct step, the generator is conditioned to diagnose the preceding failure and synthesize a corrective solution. This process yields trajectories enriched with explicit error → diagnosis → recovery cycles.

#### 3.2. High-Fidelity Task and Environment Synthesis

We now zoom into the technical details of our task and environment synthesis pipeline, which generates diverse, executable terminal tasks and environments at scale. The pipeline consists of three stages: 1) Task Definition and Generation, generating diverse and verifiable tasks; 2) Automated Environment Synthesis, generating required files and Docker containers; and 3) Environment and Task validation, where we iteratively test and validate the correctness of the environments and the resolvability of the tasks. The workflow is shown in Figure 2 Phase I.

Stage 1: Task Generation. To ensure comprehensive coverage, we construct a hierarchical taxonomy (Appendix Table 5) that spans the full spectrum of terminal usage: from low-level system operations to highlevel application workflows. The taxonomy is structured into three tiers comprising 11 distinct categories: 1) Tier 1: Infrastructure and core systems focuses on lowlevel system interactions, the corresponding categories range from build systems (e.g., resolving gcc dependency for Linux kernels) to system administration (e.g., configuring Docker). 2) Data and algorithm applications covers data-centric workflows (e.g., merging parquet files), MLOps (e.g., resolving CUDA out-of-memory), and algorithmic reasoning tasks (e.g., implementing A* algorithm). 3) Specialized development targets advanced engineering scenarios such as software development (e.g., debugging PostgreSQL deadlock), scientific computing (e.g., fixing RDKit molecule sanitization), etc. Guided by this taxonomy, we prompt an LLM to generate distinct task seeds, defined as concise, high-level abstracts such as “Resolving gcc

-l library not found errors”, aiming to ensure broad and non-overlapping coverage within each category.

Since these seeds are merely high-level directives, we employ a Task Proposal Agent to instantiate each seed into a concrete definition. The agent is prompted to synthesize a structured specification, comprising the detailed objective, input/output requirements, and success criteria. However, unconstrained generation often yields tasks that are infeasible or non-reproducible (e.g., “optimize server latency to < 10ms”, which is hardwaredependent). To mitigate this, we introduce a Task Evaluator Agent, establishing a Proposer-Evaluator architecture for iterative refinement. The Evaluator critiques task proposals based on three feasibility metrics (scored 1–5): 1) Environment Complexity: Prioritizing tasks using standard packages (e.g., apt, pip) over those requiring special hardware or obscure dependencies. 2) Data Generatability: Ensuring input artifacts (e.g., logs, configs) can be synthesized by LLMs without requiring external production dumps. 3) Verification Determinism: Favoring tasks verifiable via deterministic rules (e.g., pytest) rather than ambiguous state

checks. We enforce a strict quality threshold: only tasks scoring > 4 in all dimensions are accepted. Low-scoring proposals trigger a feedback-guided refinement loop (max 3 rounds) until the criteria are met.

- Stage 2: Environment Synthesis. Given the detailed task description from Stage 1, we employ a sequential workflow where specialized agents collaborate to materialize the environment: 1) Structural Planning: A File Planner Agent analyzes the description to decompose it into a concrete file system blueprint, outlining the directory structure and specific requirements for each artifact. 2) Content Generation: Guided by this blueprint, a Construct Agent instantiates the actual content for each file separately (e.g., synthesizing faulty Python scripts, configuration logs, or Makefiles).

3) Dockerfile Generation: Finally, an Env Agent generates a Dockerfile to encapsulate these artifacts. Crucially, the agent is instructed to explicitly resolve system-level dependencies (e.g., installing libssl-dev for compilation), ensuring the environment is fully self-contained.

- Stage 3: Environment and Task Validation. We note that the generated Dockerfiles often contain subtle errors (e.g., deprecated packages, version mismatches). We implement an iterative refinement loop: every generated environment is submitted to the Docker daemon. If Docker build fails, the stderr log (e.g., E: Unable to locate package) is captured and fed back to Env Agent as a prompt for correction. This cycle repeats (up to N = 5 iterations) until the build succeeds. This process filters out non-executable configurations, ensuring that 100% of our training environments are functionally valid.

We further proposed a Unit Test Generation and Validation process for verifying that the generated tasks are solvable. We first instruct a Unit Test Generator Agent to generate unit tests for each task. However, generating reliable unit tests is inherently challenging, even more so than solving the tasks themselves (M¨undler et al., 2024; Tang et al., 2026). A valid unit test must correctly distinguish ground-truth solutions (which pass) from plausible but incorrect attempts (which fail), requiring a precise understanding of success criteria and edge cases. This difficulty is compounded in our setting as we lack ground-truth solutions to verify against. Nevertheless, unit tests are critical for both trajectory collection (providing execution feedback) and reinforcement learning (defining reward signals). To address this, we further employ a Judge Agent that validates each generated unit test through iterative refinement: it verifies that unit tests correctly capture task objectives, provide adequate coverage of success conditions, and execute without errors. Tests failing validation receive detailed feedback and are regenerated by the Unit Test Generator until the Judge Agent approves.

#### 3.3. Error-Correction Trajectory Collection

Following the verification of environments, we proceed to trajectory synthesis. To mitigate the exposure bias discussed in Challenge ❷, we move beyond standard distillation by actively injecting failures into the training data. We implement this via a Generator-Critic framework (as shown in Figure 2 Phase II). At each time step t, framework operates through three sequential steps.

- Step 1: Intent Sampling. To control the trajectory gen-

eration, we explicitly sample an intent signal It from a Bernoulli distribution parameterized by an injection rate ϵ:

It ∼

Icorrect with probability 1 − ϵ Ierror with probability ϵ

(1)

where we set ϵ = 0.2. If It = Icorrect, the agent aims to advance the task state st efficiently. We note that the term “correct” in this context does not necessarily imply that the step is entirely accurate. Rather, it signifies that the agent is attempting to advance the task, as opposed to deliberately introducing errors. Conversely, if It = Ierror, the agent is instructed to commit a sophisticated error.

- Step 2: Context-Aware Generation. Guided by the sampled intent, the Step Generator produces a ReAct-style action at conditioned on the current system state st and task context. Crucially, when It = Ierror, the agent is prompted to synthesize plausible domain-specific mistakes rather than random noise. To ensure comprehensive coverage, we taxonomize agent failures into five categories: 1) Analysis Errors: misinterpretation of environment states or data structures;

2) Command Errors: syntactic or formatting failures in tool execution; 3) Hallucinations, where agents assume the existence of absent tools or services; 4) Requirement Violations: neglecting explicit task constraints; 5) Verification Failures: lack of self-check steps before concluding tasks. For example, in a system administration context, it might attempt a privileged operation without sudo (a common Permission Oversight), ensuring the error reflects realistic user behavior.

- Step 3: Critic Validation. Finally, a Critic Agent validates the alignment between the generated action at and the intent It to ensure data quality. For error steps, it filters out low-quality error injections by verifying that the action is syntactically plausible and will trigger informative feedback (e.g., a specific stderr message) upon execution. For optimal steps, it confirms that the action effectively advances the task state. Any step failing this validation triggers a regeneration loop guided by the Critic Agent’s feedback.

Handling Cumulative Errors and Recovery. The stochastic nature of our intent sampling enables the emergence of diverse recovery patterns. When an error action at is executed, the environment transitions to a failure state st+1, typically accompanied by execution feedback ostderr (e.g.,

RuntimeError). In the subsequent steps, if the sampled intent reverts to Icorrect, we enforce a recovery behavior: the Step Generator is conditioned to analyze the preceding error feedback and synthesize a corrective action (e.g., debugging constraints or attempting an alternative command). Crucially, since sampling is independent step-by-step, the system naturally allows for consecutive errors. This exposes the model to cascading failure modes, teaching it not only to fix isolated mistakes but also to maintain diagnostic coherence across multi-turn, noisy interaction histories. Note that these injected errors do not necessarily result in a failed trajectory, as the agent is required to recover from failures whenever a correct step is sampled in subsequent steps. These injected errors also do not reduce the success rate of completing task goals.

###### File & Directory

###### Debug & Binary

cat, ls, find ... (38 tools)

gdb, strace, valgrind ... (30 tools)

Shell & System

###### Text Processing

###### Sysadmin

grep, sed, awk ... (28 tools)

echo, ps, bash ... (51 tools)

journalctl, lsof, sysctl ... (39 tools)

###### Formal Methods

z3, cvc5, minisat ... (14 tools)

###### Scientific

###### Programming

###### Build & Package

###### DevOps & Cloud

samtools, blender, gdalinfo ... (27 tools)

python, gcc, java ... (33 tools)

make, pip, cmake ... (28 tools)

docker, kubectl, git ... (20 tools)

###### Security

###### Other

###### Database

###### Data & ML

hashcat, yara, gpg ... (20 tools)

yamllint, borg, scrapy ... (26 tools)

sqlite3, psql, redis-cli ... (17 tools)

jupyter, spark-submit, ray ... (16 tools)

######  etwork & Web

###### HPC & Runtime

curl, wget, nginx ... (17 tools)

mpirun, qemu, wasmtime ... (16 tools)

#### 3.4. Implementation and Data Statistics

Implementation Details. We utilized Claude-4.5Sonnet (Anthropic, 2025b) as the backbone model for agents in both Phase I and Phase II. For trajectory collection, we implemented a minimal terminal-based agent framework, BashAgent, following the architecture of Terminus (Terminus, 2025). The agent interacts with a Docker container via a raw bash shell, generating a ReAct-style (Yao et al., 2022) response at each turn (a reasoning trace followed by a bash command). We deliberately exclude complex agent scaffolding (e.g., RAG or multi-agent debate) to ensure the collected data reflects the intrinsic reasoning capabilities required for fine-tuning. A detailed task example and a visualization of the error-correction loop are in Appendix C.

Figure 3. Distribution of 420 command-line tools across 16 functional categories. Bubble sizes are proportional to the logarithm of the number of tools in each category.

baselines: 1) proprietary frontier models, 2) generalpurpose open-weights models, and 3) specialized models fine-tuned for terminal tasks. Then, we answer the following research questions with rigorous ablation studies: RQ1: Verifiable Environments vs. Simulation. Does training on trajectories from physically verifiable environments (Phase I) yield better generalization than training on synthetic trajectories from simulated environments? RQ2: Error-Correction vs. Standard Trajectory. Does our controlled error injection strategy (Phase II) effectively improve the agent’s self-correction capabilities compared to standard training on golden trajectories? RQ3: The Value of Negative Trajectories. How does the inclusion of unresolved error trajectories (attempts that failed to reach a solution) during SFT affect the model’s ability?

Dataset Statistics. Our synthesis pipeline yielded a final set of over 3,500 verified environments, balanced across the From these environments, we curated a corpus of 3,291 trajectories. This dataset includes both successful solutions and unresolved attempts. As we will show in Section 4.3, these unresolved trajectories remain valuable for training. Notably, these trajectories exhibit a long-horizon characteristic, spanning an average of 25.5 turns and 8,722 tokens, reflecting the complexity of real-world system tasks.

#### 4.1. Experiment Setup

Models. For the student models, we select the Qwen-2.5Coder-32B-Instruct (Hui et al., 2024) and Qwen3-32B (Yang et al., 2025a). We choose these open-weight models as they represent the current state-of-the-art in code reasoning and general instruction following. Training involves fullparameter supervised fine-tuning with a sequence length of 20,000 for 5 epochs to accommodate long interaction histories. Hyperparameter details are provided in Appendix B.

Tool Diversity. The dataset demonstrates broad coverage of terminal operations, utilizing a total of 420 unique command-line tools. These tools span 16 functional domains, ranging from standard utilities (e.g., file and text processing) to specialized software in security forensics, network operations, and HPC environments. Figure 3 illustrates the distribution of tool usage, highlighting the dataset’s coverage of both foundational and domain-specific skills.

Baselines. We evaluate TerminalBench against a comprehensive suite of models and frameworks, categorized into three distinct groups. First, we include proprietary frontiers and top-performing agent systems, Claude-4.5sonnet (Anthropic, 2025b), Claude-3.7-Sonnet (Anthropic, 2025a), GPT-5 (OpenAI, 2025b), o4-mini (OpenAI, 2025d),

### 4. Experiments

We first validate the effectiveness of TermiGen by comparing our fine-tuned models against three categories of

Claude Code (Anthropic, 2025a), Codex (OpenAI, 2025a), and Apex2, which establish the performance ceiling for autonomous terminal operations. Second, we assess agents built on general-purpose base models without task-specific fine-tuning, specifically Qwen3-235B-A30B (Hui et al.,

- 2024), Qwen3-32B (Hui et al., 2024), Qwen-2.5-Coder32B (Yang et al., 2025a), and GPT-OSS-120B (OpenAI,
- 2025c), paired with BashAgent (described in Section 3.4). Finally, we compare against specialized models that have been explicitly fine-tuned for terminal-based tasks using supervised fine-tuning or reinforcement learning, including Reptile (Dou et al., 2025), LiteCoder-Terminal (Peng et al., 2025), and TerminalAgent (Austin, 2025).

Benchmarks and Evaluation Metrics. To assess the agent’s real-world capability, we employ TerminalBench 1.0 (Terminal-Bench Team, 2025) as our primary benchmark. TerminalBench evaluates agents on a curated set of realistic terminal tasks proposed by various domain experts within isolated Docker containers. Consistent with its standard evaluation protocol, we define the pass rate as our core metric. For each task, the benchmark executes a dedicated verification script (test.sh), an agent is considered successful only if it passes all test cases defined in the verification script. We set the generation temperature as 0.6 for our fine-tuned models and base models. To account for generation variance, we report the Average Pass@1 across three independent runs to ensure reliability.

#### 4.2. Main Comparisons

Table 1 benchmarks our framework against both openweights state-of-the-art (SOTA) and proprietary frontier models. Our TermiGen-Qwen2.5-32B achieves a pass rate of 31.3%, establishing a new open-weights SOTA. It significantly outperforms existing fine-tuned agents such as Reptile (18.9%) and TerminalAgent (15.5%) by a substantial margin. Crucially, compared to the untuned base models, TermiGen yields a nearly 25% performance gain. Remarkably, our 32B model surpasses capable proprietary models like o4-mini (20.0%) by 11.3%, despite having significantly fewer parameters and no access to other tools such as web browsing or memory. While a gap remains compared to the apex of proprietary reasoning (e.g., Apex2 with Claude-4.5Sonnet at 64.5%), TermiGen-Coder-32B reaches 73.1% of the performance of GPT-5-Codex. This result suggests that a specialized model, when trained on high-fidelity errorcorrection data, can effectively approach the reasoning capabilities of much larger frontier models in domain-specific execution environments.

#### 4.3. Ablation Studies

- RQ1: Verifiable Environments vs. Simulation. To quantify the necessity of physical execution, we investigate

- Table 1. Comparison of TermiGen and baselines on TerminalBench 1.0. Results are sorted by pass rate in descending order. The * indicates results are reported by the authors.

Agent Framework Base Model Avg Pass Rate (%)

Proprietary Models

Apex2 Claude-4.5-Sonnet 64.5 ± 1.1 Codex CLI GPT-5-Codex 42.8 ± 4.2 Claude Code Claude-3.7-Sonnet 35.2 ± 2.6 Codex CLI o4-mini 20.0 ± 2.9

General-purpose Base Models

BashAgent GPT-OSS-120B 19.2 ± 2.4 BashAgent Qwen3-235B-A30B 15.4 ± 0.6 BashAgent Qwen3-32B 10.4 ± 1.2 BashAgent Qwen2.5-Coder-32B-Instruct 4.5 ± 0.7

- Terminus 1 Qwen3-235B-A30B 6.6 ± 2.8 Fine-tuned Models

BashAgent TermiGen-Qwen2.5-Coder-32B 31.3 ± 1.8 BashAgent TermiGen-Qwen3-32B 27.5 ± 1.3 Reptile Devstral-2505-22B 18.9∗

- Terminus 2 LiteCoder-30a3b 18.8∗ Terminus 2 TerminalAgent-32B 15.5 ± 2.2

- Table 2. Comparison of average Pass@1 scores between synthesized observation (Simulation) and real observation (TermiGen), controlled for data size (N = 800).

Observation Source Avg Pass@1 (%)

Simulation 22.5 TermiGen 25.0

whether training on simulated observations yields comparable performance to training on real execution feedback. We construct a baseline simulation dataset comprising 800 trajectories. To ensure a fair comparison, we maintain two identical experimental conditions to our main pipeline: 1) Tasks: The simulation uses the exact same task seeds and prompts as our verifiable dataset. 2) Strategy: We apply the same Error-Correction strategy with a 20% error injection probability. The sole difference lies in the execution layer. Instead of executing commands in a Docker container, an Observation Synthesis Agent (prompted with the current context) generates the hypothetical stdout, stderr.

As shown in Table 2, training on verifiable environments yields a performance advantage (2.5%) over the simulation baseline. With 800 samples, the Verifiable model achieves a Pass Rate of 25.0%, improving the Simulation baseline by a relative improvement of ≈ 11%. This result suggests that while simulation serves as a strong proxy, the fidelity of grounded execution provides the critical signals necessary to resolve complex edge cases.

To provide a rigorous explanation for the performance gap, we employ Claude-4.5-Sonnet to conduct an empirical audit of 50 randomly sampled trajectories from the simulation dataset. Our analysis reveals that 26% of these trajectories contain observation errors with three systemic issues. The most prevalent issue, Spurious Verbosity, accounts for 53%

Table 3. Performance comparison between the baseline model (trained on standard trajectories) and our TermiGen method, controlled for data size (N = 800).

Training Strategy Avg Pass@1 (%)

Standard 21.8 TermiGen 25.0

Table 4. Performance comparison when varying the inclusion threshold (τ) based on trajectory unit test pass rates.

Inclusion Threshold (τ) Data Size Avg Pass@1 (%)

≥ 100% 2, 040 25.0 ≥ 50% 2, 696 26.3 ≥ 0% 3,291 31.3

of the identified errors. This occurs when the simulator hallucinates explicit confirmation outputs for commands that are natively silent, such as file redirections or heredocs. Additionally, 35% of errors involve Semantic Deviation, where the simulator fails to faithfully replicate standard shell logic. For instance, it may misattribute the root cause of a permission error or return incorrect exit codes. This deviation is particularly detrimental as it instills flawed diagnostic logic: the agent learns to associate specific error messages with incorrect root causes (e.g., applying sudo to fix a path error), resulting in ineffective debugging loops during realworld deployment. Lastly, State Inconsistency (12%). For example, an agent first use cat to create a configuration file and receive a success signal, while in later steps, when the agent tries to read the config file, the simulator responds with File not found. Such discontinuities break the logical chain of reasoning, making it impossible for the agent to debug effectively.

- RQ2: Error-Correction vs. Standard Trajectory. Next, we isolate the impact of our Error-Correction strategy by comparing it against a model trained on standard trajectories. We construct a standard baseline dataset with 800 standard trajectories, which collected in the same verifiable environments but without active error injection. Note that this baseline is not strictly error-free, the agent may still commit a few incidental mistakes.

As illustrated in Table 3, our TermiGen model achieves a Pass@1 of 25.0%, consistently outperforming the baseline model (21.8%) fine-tuned on standard trajectories. This performance gap indicates that high-density exposure to diverse failure modes generalizes effectively, improving end-to-end reasoning capabilities for small models.

To explain this gain, we conducted a granular analysis of the error distributions using the five-category taxonomy defined in Section 3.3. We employ Claude-4.5-Sonnect to inspect 50 trajectories from the baseline dataset and reveals a critical skewness in naturally occurring errors: they are dominated by Verification Failures (≈ 50%), where agents typically fail simply by omitting a final check, followed by Analysis Errors (≈ 19%). Complex failure modes such as Hallucinations and Requirement Violations are rare. In contrast, our strategy (ϵ = 0.2) actively forces the agent to navigate a diverse spectrum of failure modes. We further present a comparative case study in Appendix D to demonstrate the error-correction capabilities acquired through our training.

RQ3: The Value of Negative Trajectories. Standard SFT protocols typically enforce a strict quality filter, retaining only trajectories that achieve a 100% pass rate on verification unit tests (e.g., reject sampling). We conduct an ablation study by varying the inclusion threshold τ based on the trajectory’s unit test completion rate. Specifically, we train Qwen-2.5-Coder-32B on trajectories where Test Pass Rate ≥ τ, with τ ∈ {100%,50%,0%}). Counterintuitively, as shown in Table 4, we observe that strict data filtering is detrimental in our setting: the model trained with τ = 0% achieves the highest performance, surpassing the strict τ = 100% baseline by a noticeable margin.

We attribute this performance gain to two factors. First, Task Complexity Coverage. We hypothesize that trajectories achieving a 100% pass rate are statistically skewed towards easier tasks. Strictly filtering out imperfect runs inadvertently discards valuable exposure to complex, high-difficulty scenarios where the agent demonstrated valid reasoning logic but fell short of a perfect score. Second, Local Recovery Supervision. Due to our Error-Correction mechanism, even a trajectory that ultimately fails the final evaluation, it still contains valid local repairs of injected errors. These segments provide high-quality supervision for the mechanics of diagnosis and correction, independent of the global task outcome. Together, these factors suggest that imperfect data is essential for learning resilience in complex domains.

### 5. Conclusion and Limitations

In this work, we present a framework for synthesizing verifiable terminal environments and an error-injection distillation strategy. Our empirical results demonstrate that this approach closes the gap between open-weights and proprietary models. Despite these advancements, we acknowledge three key limitations that outline future directions.

First, our current training relies exclusively on SFT. Since our environment provides deterministic verification signals via automated tests, a natural next step is to apply reinforcement learning. This would allow agents to explore novel solutions and learn from trial-and-error beyond the fixed distribution of the training data. Second, we currently implement a simple agent without a memory component to mainly test the effectiveness of generated environments and trajectories. Future work could design more sophisticated agents with memory to leverage interaction histories. Third, while our taxonomy spans diverse domains, our environ-

ments are inherently synthetic and isolated. They cannot fully replicate the stochasticity and scale, especially for realworld production systems (e.g., distributed clusters, highconcurrency traffic). Future research should investigate how agents trained on these controlled sandbox environments transfer to open-ended, large-scale infrastructure tasks.

### Impact Statement

This paper presents TermiGen, a framework that improves the reliability of autonomous agents in software engineering and system administration. By enabling open-source models to effectively use terminal tools, we make advanced DevOps capabilities more accessible to the community, lowering the barrier for complex software tasks.

However, we acknowledge that giving agents the power to execute terminal commands comes with risks. Safety Risks: An autonomous agent might accidentally cause data loss or system crashes (e.g., deleting the wrong files). Our focus error correction helps reduce this risk. By teaching agents to identify and fix their own mistakes, we prevent simple errors from leading to larger failures. Malicious Use: The techniques described could theoretically be misused to automate cyber-attacks. To address these risks, we emphasize that such agents must be deployed in isolated environments (e.g., Docker containers) and always under human supervision.

### References

Anthropic. Claude 3.7 sonnet and claude code. Technical announcement, Anthropic, February 2025a. URL https://www.anthropic.com/news/ claude-3-7-sonnet.

Anthropic. Introducing claude sonnet 4.5, September 2025b. URL https://www.anthropic.com/ news/claude-sonnet-4-5.

Austin, D. Terminal-bench-rl: Training long-horizon terminal agents with reinforcement learning, 2025. URL https://github.com/Danau5tin/ terminal-bench-rl.

Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Chen, B., Shu, C., Shareghi, E., Collier, N., Narasimhan, K., and Yao, S. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915, 2023.

Chen, M. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, Z., Zhao, Z., Zhang, K., Liu, B., Qi, Q., Wu, Y., Kalluri, T., Cao, S., Xiong, Y., Tong, H., et al. Scaling agent learning via experience synthesis. arXiv preprint arXiv:2511.03773, 2025.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dou, L., Du, C., Li, S., Wang, T., Zhang, T., Liu, T., Chen, X., Tang, C., Zhao, Y., and Lin, M. Reptile: Terminal-agent with human-in-the-loop learning, 2025. URL https://github.com/terminal-agent/ reptile. GitHub repository.

Huang, Y., Lin, Z., Liu, X., Gong, Y., Lu, S., Lei, F., Liang, Y., Shen, Y., Lin, C., Duan, N., et al. Competition-level problems are effective llm evaluators. In Findings of the Association for Computational Linguistics: ACL 2024, 2024.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Kimi Team. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Li, Y. et al. Simulating environments with reasoning models for agent training. arXiv preprint arXiv:2511.01824, 2025.

Liu, J., Wang, K., Chen, Y., Peng, X., Chen, Z., Zhang, L., and Lou, Y. Large language model-based agents for software engineering: A survey. arXiv preprint arXiv:2409.02977, 2024.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

Mitra, A., Del Corro, L., Zheng, G., Mahajan, S., Rouhana, D., Codas, A., Lu, Y., Chen, W.-g., Vrousgos, O., Rosset, C., et al. Agentinstruct: Toward generative teaching with agentic flows. arXiv preprint arXiv:2407.03502, 2024.

M¨undler, N., M¨uller, M., He, J., and Vechev, M. Swtbench: Testing and validating real-world bug-fixes with code agents. Advances in Neural Information Processing Systems, 37:81857–81887, 2024.

OpenAI. Codex. OpenAI Product Page, 2025a. URL https://openai.com/codex/.

OpenAI. Introducing gpt-5. OpenAI Blog, August 2025b. URL https://openai.com/index/ introducing-gpt-5/.

OpenAI. Introducing gpt-oss. OpenAI Blog, August 2025c. URL https://openai.com/index/ introducing-gpt-oss/.

OpenAI. Introducing openai o3 and o4-mini. OpenAI Blog, April 2025d. URL https://openai.com/index/ introducing-o3-and-o4-mini/.

Peng, X., Lu, X., Zhang, K., Fang, T., Cao, B., and Lu, Y. Litecoder: Advancing small and medium-sized code agents, 2025.

Ross, S., Gordon, G., and Bagnell, D. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Tang, Y., Zhu, K., Ruan, B., Zhang, C., Yang, M., Li, H., Guo, S., Shi, T., Li, Z., Kruegel, C., et al. Devops-gym: Benchmarking ai agents in software devops cycle. arXiv preprint arXiv:2601.20882, 2026.

Terminal-Bench Team. Terminal-bench: A benchmark for ai agents in terminal environments, 2025. URL https://github.com/laude-institute/ terminal-bench.

Terminus. Terminus: A research-preview agent for terminalbench, May 2025. URL https://www.tbench.ai/ terminus.

Wang, Y. et al. Llms as scalable, general-purpose simulators for evolving digital agent training. arXiv preprint arXiv:2510.14969, 2025.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, J., Lieret, K., Jimenez, C. E., Wettig, A., Khandpur, K., Zhang, Y., Hui, B., Press, O., Schmidt, L., and Yang, D. Swe-smith: Scaling data for software engineering agents. arXiv preprint arXiv:2504.21798, 2025b.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., and Tang, J. Agenttuning: Enabling generalized agent abilities for llms. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 3053–3077, 2024.

Zhang, L., He, S., Zhang, C., Kang, Y., Li, B., Xie, C., Wang, J., Wang, M., Huang, Y., Fu, S., et al. Swe-bench goes live! arXiv preprint arXiv:2505.23419, 2025.

Zheng, T., Zhang, G., Shen, T., Liu, X., Lin, B. Y., Fu, J., Chen, W., and Yue, X. Opencodeinterpreter: Integrating code generation with execution and refinement. arXiv preprint arXiv:2402.14658, 2024.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

Table 5. Hierarchical Domain Taxonomy. Our dataset covers three tiers spanning 11 sub-categories, ranging from low-level infrastructure to high-level specialized development. We list representative competencies and specific task seeds for each category.

Sub-Category Competencies (Goals & Skills) Representative Task Seeds Tier I: Infrastructure & Core Systems

- 1.1 Software Build & Compilation

Goal: Resolve dependency hell, linker errors, and cross-compilation issues. Skills: gcc, cmake, makefile, rustc, autotools, ld.

gcc cannot find library, cmake cuda toolkit not found, rust cargo linker failure, makefile missing separator

- 1.2 System Administration & DevOps

Goal: Manage containers, orchestrate clusters, and debug system services. Skills: docker, kubernetes, systemd, nginx, terraform.

kubernetes pod crashloop backoff, docker layer caching broken, nginx upstream keepalive connections, systemd unit fails on boot

- 1.3 Security, Forensics & Reverse Engineering

Goal: Exploit analysis, binary reversing, and digital forensics. Skills: ghidra, wireshark, metasploit, gdb, volatility.

buffer overflow exploit development, wireshark pcap analysis, ghidra reverse engineering workflow, sqlmap automated injection

###### Tier II: Data & Algorithm Applications

- 2.1 Data Processing & ETL

Goal: Transform large-scale datasets and handle schema evolution. Skills: pandas, spark, kafka, sql, parquet, jq.

parquet schema evolution, kafka avro schema registry, pandas merge asof timeseries, spark dataframe schema inference

- 2.2 Machine Learning & MLOps

Goal: Debug training instability, optimize inference, and manage pipelines. Skills: pytorch, cuda, huggingface, scikit-learn.

cuda out of memory, pytorch dataloader workers, transformers token limit, gradient explosion detection

- 2.3 Algorithms & Logic Puzzles

Goal: Implement classic algorithms and solve competitive programming tasks. Skills: Graph theory, Dynamic Programming, Search (A*), CSP.

n queens placement, shortest path dijkstra, alpha beta pruning minimax, knapsack dynamic programming

###### Tier III: Specialized Domains & Advanced Development

- 3.1 Software Dev, Porting & Bug Fixing

Goal: Full-stack debugging, framework configuration, and legacy porting. Skills: react, django, git, rest-api, ci/cd.

react hooks dependency array, django csrf token missing, git merge conflict binary, rest api pagination cursor

- 3.2 Scientific & Domain Computing

Goal: Computational biology, chemistry simulations, and statistical modeling. Skills: bioconductor, rdkit, gromacs, stan, numpy.

rdkit mol sanitization error, stan divergent transitions, bioconductor annotation mismatch, gromacs topology atom mismatch

- 3.3 Interactive Environments

Goal: Handle real-time protocols, REPLs, and interactive debugging sessions. Skills: websocket, ssh, gdb-interactive, jupyter.

websocket chat server, interactive sql repl, gdb interactive debugging, jupyter kernel connection

- 3.4 Distributed & Parallel Computing

Goal: Debug race conditions, deadlocks, and distributed consensus issues. Skills: mpi, openmp, ray, dask, slurm.

mpi deadlock collective ops, spark executor out of memory, ray actor died unexpectedly, openmp race condition shared

- 3.5 Formal Verification & Graphics

Goal: Theorem proving, SAT solving, and rendering pipeline debugging. Skills: coq, z3, opengl, vulkan, blender.

coq universe inconsistency, opengl context creation headless, z3 timeout optimization, vulkan validation layer error

### A. Detailed Task Categories

- Table 5 shows the task taxonomy.

| |
|---|

Algorithms & Logic Puzzles

Data Processing & ETL

Distributed & Parallel Computing

Formal Verification & Graphics

Interactive Environments & Games

Machine Learning & MLOps

Scientific & Domain-Specific Computing

Security, Reverse Engineering & Forensics

Software Build & Compilation

Software Development, Porting & Bug Fixing

System Administration & DevOps

Figure 4. t-SNE visualization of ≈ 3, 500 tasks across 11 categories, showing semantic clustering by task type.

### B. Implementation Details

Our training pipeline is built upon the LLaMA-Factory framework. We perform full-parameter supervised fine-tuning (SFT) on the base model using the synthetically generated dataset. To accommodate the extensive context requirements of multi-turn terminal interactions, we set the maximum sequence length to 20,000 tokens. The training is optimized using AdamW with a learning rate of 5.0 × 10−6, employing a cosine learning rate scheduler with a warmup ratio of 0.1 and zero weight decay. We train the model for 5 epochs with a per-device batch size of 1 and 8 gradient accumulation steps. All experiments were conducted on the compute nodes equipped with 8× AMD MI325X GPUs. To ensure memory efficiency and scalability across these accelerators, we utilize DeepSpeed ZeRO-3 with BF16 precision enabled.

### C. Task and Error-Correction Trajectory

- C.1. Synthetic Dataset Samples In this case study, we present a representative synthetic task directory, to illustrate the structure and content of our benchmark.

Task Description. This task targets macOS automation using AppleScript. The high-level goal is to locate all application bundles (.app packages) within the /Applications directory (including subdirectories) and extract their version information from Info.plist files. The task requires handling edge cases such as: (1) Application bundles nested in subdirectories; (2) Missing or malformed Info.plist files; (3) Applications without readable version information.

The solution must generate a JSON file at /tmp/app inventory.json containing: (1) total apps: Total number of .app bundles found; (2) apps with version: Number of apps with readable version information (3) apps without version: Number of apps where version info couldn’t be extracted.

Directory Structure. The directory is organized as follows:

|-- task.yaml # Task description and metadata |-- Dockerfile # Docker environment configuration |-- docker-compose.yaml # Docker Compose configuration

|-- run-tests.sh # Test execution script |-- Applications/ # Test data directory | |-- TextEditor.app/ | | |-- Contents/ | | | |-- Info.plist | |-- Utils/ | | |-- Calculator.app/ | | | |-- Contents/ | | | | |-- Info.plist | |-- DevTools/ | | |-- CodeEditor.app/ | | | |-- Contents/ | | | | |-- Info.plist | | |-- Terminal.app/ | | | |-- Contents/ | | | | |-- Info.plist |-- tests/

|-- test_outputs.py # Automated test suite

Evaluation Framework. The evaluation is performed through a comprehensive pytest-based test suite (tests/test outputs.py) that validates: (1) Output file existence at the specified path; (2) Valid JSON format; (3) Presence of all three required fields; (4) Exactly three fields (no extra fields); (4) Correct data types (all integers); (5) Accurate count of total applications (must equal 4).

- C.2. An Error-Correction Example in Training Trajectories We present a representative example illustrating the error analysis and correction loop in our generated trajectories.

|[Figure 19]<br><br>(Goal):<br><br>Solve the SMT constraints in /workspace/password policies.smt2 using CVC5. Goal: Determine satisfiability and extract a valid password model if one exists. Output: Save the result to /workspace/result.txt in the exact format: satisfiable: yes followed by password: <value> on the next line (or just satisfiable: no if impossible).<br><br>[Figure 20]<br><br>(Action 1):<br><br>...<br><br>[Figure 21]<br><br>(Output 1):<br><br>...<br><br>[Figure 22]<br><br>(Action 2):<br><br>...<br><br>[Figure 23]<br><br>(Output 2):<br><br>...<br><br>[Figure 24]<br><br>(Action 3): Looking at the current state, I have:<br><br>• Confirmed the /workspace/password policies.smt2 file exists and contains valid SMT-LIB2 constraints<br><br>• Verified that CVC5 is available at /usr/local/bin/cvc5<br>• Understood the password constraints from the SMT file The next logical step is to run CVC5 on the SMT file to determine if the<br>|
|---|

|constraints are satisfiable and, if so, get a model (example password) that satisfies all constraints.<br><br><tool call>{cvc5 --lang smt2 --strings-exp /workspace/password policies.smt2} </tool call><br><br>⇒ This command is intentionally incorrect to inject an error.<br><br>[Figure 25]<br><br>(Output 3):<br><br>Parse Error: /workspace/password policies.smt2:6.33: Symbol ’>=’ not declared as a variable<br><br>[Figure 26]<br><br>(Action 4):<br><br>The error persists - CVC5 is still reporting that ’>=’ is not declared as a variable. This is happening because the SMT-LIB2 file declares the logic as ‘QF S‘ (quantifier-free strings only), but then uses integer comparison operators (‘>=‘, ‘<=‘) with ‘str.len‘, which requires integer arithmetic support.<br><br>The issue is that QF S logic doesn’t include integer arithmetic. The correct logic should be ‘QF SLIA‘ (Quantifier-Free Strings and Linear Integer Arithmetic) or just let CVC5 auto-detect it.<br><br>⇒ In this step, it reflects and refines previous error command. However, I notice the task says ¨The file contains a complete, valid SMT-LIB2<br><br>problem statement¨- this suggests the file should work as-is. Let me check if there’s a way to make CVC5 more lenient or if I need to use a different approach.<br><br>Let me try running CVC5 without specifying the logic strictly, by using the ‘--force-logic‘ option to override the declared logic:<br><br><tool call>{cvc5 --force-logic=QF SLIA /workspace/password policies.smt2} </tool call><br><br>|
|---|

### D. Case Study: Learned Error-Correction Capabilities in TermiGen

To demonstrate the robust error-correction capabilities of TermiGen, we conduct a comparative case study against the baseline model (Qwen2.5-Coder-32B) and our fine-tuned model (TermiGen-Qwen2.5-Coder-32B) on a challenging formal verification task: proving the commutativity of addition (n + m = m + n) in Coq.

Baseline Failure: Premature Conclusion. As summarized in Appendix D, the baseline model attempts a standard proof using induction and reflexivity. However, it suffers from a silent failure mode: upon generating the code, it assumes correctness without verifying the generation of the compiled object file (‘.vo‘). Lacking a verification loop, it fails to detect that the proof is mathematically incomplete, resulting in a failed trajectory despite a confident termination.

TermiGen Success: Reasoning-Driven Recovery. In contrast, TermiGen exhibits a distinct iterative debugging behavior. In the initial attempt, TermiGen encounters a compilation error: Unable to unify "m + 0" with "0 + m". Instead of hallucinating or giving up, the model analyzes the stderr output and correctly identifies the root cause: reflexivity only handles definitional equality, whereas the goal requires propositional equality via rewriting. Guided by this diagnosis, TermiGen applies the necessary lemmas (rewrite plus 0 r and plus Snm) across two correction steps, ultimately producing a verified, compiling proof.

This comparison highlights that our error-injection training has endowed TermiGen with the critical ability to treat execution feedback as a navigational signal rather than a terminal state.

- Table 6. Qualitative Comparison on Coq Theorem Proving. The baseline model fails due to a lack of verification, TermiGen successfully navigates multiple error states through iterative diagnosis and correction.

Aspect Baseline (Qwen2.5-Coder-32B) TermiGen (Ours) Behavior One-shot attempt. Wrote a proof and exited immedi-

Iterative refinement. Engaged in a multi-turn debug loop (20 episodes).

ately without checks.

Error Handling None. Ignored potential compilation issues and as-

Active Diagnosis. Analyzed stderr (Unable to unify...) to distinguish definitional vs. propositional equality.

sumed success.

Key Action Failed to verify the output artifact (‘.vo‘ file). Applied specific fixes: rewrite plus 0 r and

rewrite plus Snm. Outcome Failed (Incomplete proof) Success (Verified proof)

