## CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion

Yusong Lin12 Haiyang Wang 1 Shuzhe Wu1 Lue Fan3 Feiyang Pan1 Sanyuan Zhao 2 Dandan Tu 1 1Huawei Technologies Co., Ltd 2Beijing Institute of Technology 3Institute of Automation, Chinese Academy of Sciences

{linyusong4, haiyang.wang, wushuzhe2, panfeiyang, tudandan}@huawei.com zhaosanyuan@bit.edu.cn lue.fan@ia.ac.cn

Code: https://github.com/LiberCoders/CLI-Gym Dataset: CLI-Gym Environments

[Figure 1]

# arXiv:2602.10999v1[cs.AI]11Feb2026

[Figure 2]

|LiberC|oder-235B-A22|B| | |
|---|---|---|---|---|
|LiberCoder-32B Mini|max-M2<br><br>GLM-4.|6<br><br>Qwen3-Coder-<br><br>DeepS|480B<br><br>eek-V3.2-Exp| |
|Qwen3-coder-3<br><br>GLM-4.5-Air|0B| |[Figure 3]|Kimi-K2-Instruct|
|Qwen3-2|35B-A22B-Instru|D<br><br>ct|eepSeek-V3| |
|Qwen3-32B<br><br>Qwe|n3-235B-A22B<br><br>Llama-4-Ma|D<br><br>verick-17B|eepSeek-R1| |

###### Git Commits

###### Agent Explorations

[Figure 4]

45

###### (real code histories)

(simulated environment histories)

| |History Space<br><br>bug-free<br><br>buggy|
|---|---|
|Env<br><br>Env: Desc UTs:|-intensive Task<br><br>Dockerfile ription: … Fail … Pass …|

bug-free

###### Agent inverts env. states

35

PR: Fix the bug in loss gradient clipping …

Invert code states

25

buggy

###### Code-intensive Task buggy Code: … Description: …

bug-free

15

UTs: Fail … Pass … Env: Dockerfile

Error: Permission denied …

5

0 200 400 600 800 1000

###### Model Size (B)

(a) Deriving agentic tasks by tracing code or environment histories

(b) Pass@1 on Terminal-Bench 1.0 vs model size

Figure 1. Illustration of the idea behind our CLI-Gym that brings high performance on the Terminal-Bench 1.0. (a): Code-intensive tasks, as those in the SWE-bench, can be derived with readily available code histories and context like PRs. For tasks involving intensive interaction with the environment like CLI, as those in the Terminal-Bench, we employ agents to simulate and explore environment histories guided by execution feedback, realizing scalable derivation of environmen-intensive tasks. (b): With task trajectories obtained using our CLI-Gym, the fine-tuned Qwen3-32B and Qwen3-235B-A22B-Instruct models, named as LiberCoder and denoted by red triangles, achieve Pass@1 metrics of 38.9% and 46.1%, respectively, outperforming various strong baselines.

### Abstract

of 1,655 environment-intensive tasks are derived, being the largest collection of its kind. Moreover, with curated successful trajectories, our fine-tuned model, named LiberCoder, achieves substantial absolute improvements of +21.1% (to 46.1%) on Terminal-Bench, outperforming various strong baselines. To our knowledge, this is the first public pipeline for scalable derivation of environmentintensive tasks.

Agentic coding requires agents to effectively interact with runtime environments, e.g., command line interfaces (CLI), so as to complete tasks like resolving dependency issues, fixing system problems, etc. But it remains underexplored how such environment-intensive tasks can be obtained at scale to enhance agents’ capabilities. To address this, based on an analogy between the Dockerfile and the agentic task, we propose to employ agents to simulate and explore environment histories, guided by execution feedback. By tracing histories of a healthy environment, its state can be inverted to an earlier one with runtime failures, from which a task can be derived by packing the buggy state and the corresponding error messages. With our method, named CLI-Gym, a total

### 1. Introduction

Interacting with and manipulating the runtime environment is a critical aspect of real-world software development, yet it remains largely neglected in existing research on agentic coding. In fact, few works have specifically concerned environment-intensive tasks, which involve complex, multifaceted interactions with the environment, such as resolv-

Corresponding authors

Table 1. A summary of existing pipelines for deriving agentic coding tasks at scale. Code-intensive tasks can be naturally derived from GitHub repositories, benefited from the detailed code histories and the rich context, e.g., commits, PRs, issues. In contrast, environmentintensive tasks are constructed with heavy human labor in existing works and at a substantially smaller scale (102 vs 103 ∼ 104).

Method Gold Instance −→ Data Engine −→ Problem Instance Collection # Instances Code-Intensive Task SWE-Bench (Jimenez et al., 2024) PR-based Auto 2294 SWE-Gym (Pan et al., 2024) PR-based Code-Intensive Issue Auto 2438 R2E-Gym (Jain et al., 2025) Pre-installed Github Repo PR-based / LLM-based + Auto 8135 SWE-smith (Yang et al., 2025b) PRs / LLM Synthesis Environment / Failed Unit Tests Auto 50137 SWE-Dev (Du et al., 2025) Test-Driven Auto 14000 Environment-Intensive Task

- Terminal-Bench@1.0 (Merrill et al., 2026) - - Environment-Intensive Issue Human-written 80

- Terminal-Bench@2.0 (Merrill et al., 2026) - - + Human-written 89 CLI-Gym (ours) Pre-installed Github Repo Agentic Synthesis Environment / Failed Unit Tests Auto 1655

ing dependency issues, repairing problematic configurations, and fixing broken environment variables. As such research remains publicly unavailable, on the Terminal-Bench (Terminal-Bench Team, 2025), which evaluates agents’ proficiency in interacting with the command line interfaces (CLI) environment, agents powered by large language models (LLMs) with even hundreds of billions of parameters achieve task resolution rates less than 40%, as shown in Figure 1b. In contrast, numerous studies have put great effort into enhancing agents’ coding capabilities for software engineering (SWE) tasks (Jain et al., 2025; Yang et al., 2025b), pushing the performance on SWE-bench (Jimenez et al., 2024) to over 70%. The large gap reflects that it remains substantially underexplored how agents can be enhanced for sophisticated environment interaction and manipulation beyond writing code in practical development.

The performance boost on SWE-Bench is primarily driven by scaling up LLM training on executable and verifiable code-intensive tasks from real-world scenarios (Jimenez et al., 2024). Given that code repositories track all key code states by version control, which are associated with abundant context, e.g., commits, pull requests (PRs) or issues, a realistic code-intensive task can be derived by tracing code histories to pack the code, obtained via reverting a PR/commit and thus inverting the code to a “buggy” state, and the relevant descriptive context, such as PR/commit messages.

In contrast to code-intensive tasks, environment-intensive tasks can hardly be derived by tracing through repository histories. The fundamental challenge stems from the absence of environment histories, which cannot be comprehensively captured by centralized version control. While an identical codebase is shared across developers, their runtime environments vary from one to the other. The Dockerfile enables sharing an independent containerized environment, but in reality, it lacks rich histories with abundant modifications and the corresponding context, like commit messages, which are required as sources for task derivation. Besides, among repository issues, only a small fraction corresponds to environment-intensive tasks, which are hard to identify.

Due to these problems, currently no pipeline is available for scalable derivation of environment-intensive tasks, severely hindering the enhancement of agents’ performance.

To address the above challenge, we design a principled approach to simulate environment histories and provide the first public pipeline for scalable derivation of environmentintensive tasks from code repositories. We use the Dockerfile to represent an environment, which is common in repositories. Intuitively, as a Dockerfile describes the environment as a command sequence over a base Docker image, it aligns well with an agentic action sequence executed in an initial environment, forming the history of modifying it. Hence, the command sequence can be modeled as an agentic task involving environment interactions. Then, as shown in Figure 1a, from a reverse view of the agentic task, inverting the action sequence exactly mimics tracing the environment history from a runnable state to a buggy one.

Based on the above intuition, we formulate the derivation of environment-intensive tasks as an agentic task itself, in which an agent explores the history space of the environment to invert its states, similar to reverting code commits but not following existing, fixed histories. Specifically, starting from a healthy environment, an agent freely explores possible histories by iteratively executing commands to modify the environment and receiving execution feedback as guidance. Equipped with a rich action space, the agent corrupts the environment and can reach diverse historical states to cover various scenarios. After reaching a state with unit tests (UTs) failures, a task can be derived based on error messages. With such a method, environment-intensive tasks can be derived from repositories with scalability. Note that our method intrinsically differs from generating a Dockerfile using LLMs, which either works in a single pass or iterates without feedback. For better understanding, different task derivation methods are summarized and compared in Table 1. Without loss of generality and following the Terminal-Bench (Terminal-Bench Team, 2025), we focus on the CLI environment and provide a flexible pipeline CLIGym for task derivation based on the above design.

With our CLI-Gym, we derive 1,655 environment-intensive task instances from 29 popular open-source repositories. Compared with the dozens of manually labeled tasks in the Terminal-Bench, our task collection is nearly 20× larger, as shown in Table 1. Moreover, for our task collection, we curate 291 trajectories that successfully complete the corresponding tasks, and perform a pilot study on fine-tuning LLMs. Surprisingly, our fine-tuned Qwen3-235B-A22BInstruct model (Yang et al., 2025a) achieves significant absolute improvements of +21.1% (to 46.1%) and +12.9% (to 31.0%) on Terminal-Bench 1.0 and 2.0, respectively, outperforming even larger-size open-source models like Kimi-K2 (Team et al., 2025), Qwen3-Coder-480B (Yang et al., 2025a) and GLM 4.6 (Z.ai, 2025), as shown in Figure 1b.

In a nutshell, our contributions are threefold:

- • We introduce the first publicly available pipeline CLIGym for scalable derivation of environment-intensive tasks in agentic coding.
- • A collection of 1,655 environment-intensive tasks is built from 29 open-source repositories, serving as a good data source for LLM fine-tuning.
- • With a pilot study on fine-tuning with only 291 successful trajectories, we demonstrate highly competitive performance on the Terminal-Bench.

### 2. Related Work

Agentic Coding via CLI. Recent advances in LLM-based agents, such as Claude Code (Anthropic, 2025), GeminiCLI (Google, 2025), and Codex CLI (OpenAI, 2025), have enabled significant progress on real-world coding tasks through CLI (Liu et al., 2024). Agentic coding tasks can be broadly categorized as code-intensive and environmentintensive, which take writing code and interacting with environment, respectively, as the main portion of work. For codeintensive benchmarks (e.g., SWE-bench), agents typically leverage tool-integrated workflows for iterative software development. Building upon the PR-based data pipeline, extensive open-source efforts (Yang et al., 2025b; Pan et al.,

- 2024) have led to the construction of large-scale task environments that substantially facilitate the development of this area. In contrast, environment-intensive tasks, exemplified by Terminal-Bench series (Terminal-Bench Team,
- 2025), require agents to perform complex interactions with environments, such as resolving dependency issues or managing the system. Compared to a code-intensive setting, this paradigm remains significantly underexplored in terms of scalable data construction, heavily relying on human-written instances, which results in a fragmented and largely closed development ecosystem.

Scaling Training Environment for Agentic Coding. Executable and verifiable environments are crucial for agentic

coding, as they provide reliable success signals for both training and evaluation. For code-intensive tasks, scalable training environments have been extensively studied, including SWE-gym (Pan et al., 2024), R2E-gym (Jain et al., 2025), and SWE-smith (Yang et al., 2025b), which construct executable tasks by crawling pull requests or injecting synthetic faults. While code-intensive tasks benefit from mature and open data ecosystems, environment-intensive settings such as Terminal-Bench lack scalable, open-source pipelines. As a result, closed-source models (e.g, Claude, GPT, and Gemini series) currently dominate the leaderboard, underscoring the lack of effective and publicly available data pipelines as a key factor limiting open-source progress.

### 3. Method

The central premise of this work is that agentic coding can be viewed as a process in which an autonomous code agent modifies the state of its execution environment through coding. In conventional agentic coding tasks, the agent is typically required to transform an initially defective environment into a correct one, i.e., a transition from a poor state to a gold state (Sec. 3.1). In this paper, we have an inverse perspective to reinterpret task collection as agentic environment inversion, where an agent deliberately degrades a gold environment into a poor one (Sec. 3.2) to simulate environment histories. This perspective enables scalable synthesis of CLI task instances, providing an effective way for improving environment-intensive agentic coding (Sec. 3.3).

#### 3.1. Agentic Coding from an Environment Perspective

To establish a clear understanding of our approach, we first formalize agentic coding from an environment-centric perspective, treating the execution environment, rather than code alone, as the primary object of manipulation.

We begin by formalizing the state of a runnable coding environment as a tuple

##### S = (B,D,C), (1)

where B denotes the base environment, typically a minimal official system image (e.g., an official Ubuntu Docker image); D denotes the Dockerfile that specifies how the execution environment is constructed; and C denotes the software codebase.

Under this representation, agentic coding can be modeled as a state transition over environment configurations:

Spoor −−−−−−→(∆D,∆C)

Sgold, (2)

Agent

where Spoor represents an environment state that fails at least one unit test, while Sgold denotes a state in which all unit tests pass. The agent produces a set of state modifications

[Figure 5]

Github Repo

Agentic Environment Inversion

[Figure 6]

Task Instance

###### Problem Instance Generation

Query Generation

[Figure 7]

[Figure 8]

Problem Statement

Gold Instance

× 𝑵

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Selected Unit Tests

Build

LLM

[Figure 13]

[Figure 14]

Environment Base Image Dockerfile

Environment

LLM

Problem Environment

Problem Statement

[Figure 15]

[Figure 16]

[Figure 17]

Base Docker

[Figure 18]

Task Prompt (𝑞)

[Figure 19]

+

[Figure 20]

[Figure 21]

Dockerfile

[Figure 22]

[Figure 23]

Codebase

Codebase

Inverse Agentic Coding

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

src

src

train.py

train.py

[Figure 28]

Gold Instance

Dockerfile

[Figure 29]

[Figure 30]

[Figure 31]

…

[Figure 32]

[Figure 33]

[Figure 34]

tests setup.py

tests setup.py

[Figure 35]

“pip list” “pip uninstall torch”

[Figure 36]

Task Prompt

[Figure 37]

[Figure 38]

Unit Tests

Unit Tests

…

|[Figure 39]|Test_gpt2|
|---|---|
|[Figure 40]|Test_download|

[Figure 41]

Unit Tests

[Figure 42]

[Figure 43]

Test_gpt2 Test_download

Induce Failure

[Figure 44]

[Figure 45]

Figure 2. Overview of our proposed CLI-Gym pipeline. 1) Starting from a GitHub repository, we construct a gold instance consisting of a functional environment, codebase, and associated unit tests. 2) We then derive task prompts from the unit tests and execute them with an agent to obtain failure-inducing commands. Based on the observed execution commands and failing tests, we automatically generate a corresponding problem statement. 3) Finally, the outputs from the previous steps are assembled into a standardized task instance.

(∆D,∆C), corresponding to changes in the Dockerfile and the codebase, respectively, in order to repair the instance, such as bug fixing or dependency configuration.

As shown in Figure 2, we start from a gold environment and its associated set of passing unit tests. From a selected subset of these tests, we employ an LLM to generate environment inversion prompts that specify how an agent should deliberately induce failures. To promote task diversity, we maintain a memory pool of previously task descriptions and incorporate their high-level summaries into the LLM context during prompt generation.

#### 3.2. Instance Construction by Environment Inversion

- As suggested by Equation 2, scaling agentic coding training instances fundamentally requires access to a large and

diverse set of initial poor states Spoor. Motivated by this observation, we formulate data collection as an agentic environment inversion process:

Formally, given a gold environment Sgold and a task prompt q, the agent autonomously generates Dockerfile commands to manipulate the execution environment, including operations over the filesystem, virtual environments, dependency configurations, and other system-level states. Through this agentic Dockerfile coding process, the agent produces a perturbed environment

(Sgold,Tpassed) −−−−−−→(∆D,∆C)

(Spoor,Tfailed), (3)

Agent

where Tpassed and Tfailed denote the sets of unit tests that pass and fail, respectively. Under this formulation, data generation starts from a gold state Sgold that satisfies all unit tests. An autonomous agent then applies structured perturbations (∆D,∆C) to intentionally induce failures, yielding a defective environment Spoor and simulating potential environment histories. Each such degraded environment constitutes a valid CLI task instance. We describe the pipeline components in detail below.

Spoor = Sgold ⊕ (∆D,∆C), (4)

where (∆D,∆C) are environment-level and code-level modifications intended to violate the specified unit tests. The resulting Dockerfile encodes the degradation trajectory, ensuring reproducibility across different instantiations. Figure 4 illustrates a concrete degradation example in which the agent writes Dockerfile commands to corrupt the environment. It is noteworthy that such environment-level perturbations extend beyond code modifications and require agents to diagnose system dependencies and library integrity.

Initial Gold Environment Construction. Our gold instances are Docker images derived from real-world GitHub repositories that pass all unit tests. Following the construction protocol of SWE-Smith (Yang et al., 2025b), we choose a repository and install it from the base Docker image B. This environment serves as the oracle state Sgold for subsequent inverse agentic transformations.

Execution-based Problem Instance Construction. Given the synthesized Dockerfile and the induced set of failing unit tests, we reconstruct the environment from the original gold state by executing the Dockerfile. If at least one unit test

Environment Inversion by Agentic Dockerfile Coding.

- Table 2. Statistics comparing CLI-Gym with the Terminal-Bench 1.0 and 2.0. Except for size and cost metrics, we report the average value across instances. †229 instances are composed of some non-evaluation tasks and 1.0 / 2.0 test tasks.

Other

(110) Infrastructure

(370)

Source Logic (110)

Category Metric Terminal-Bench 1.0 / 2.0 CLI-Gym Size # Instances 229† 1655

###### CLI-Gym (1655)

Concurrency (50)

File System (378)

# Images 22 29 Issue Text Length by Words 140.7 159.1 Dockerfile # Lines 5.8 6.8 Tests # Fail to Pass 7.9 20.4

Dependencies & Runtime (282)

Network & Security (251)

Compatibility (104)

# Pass to Pass 0.0 29.6 Cost 93 Contributors 2.3B Tokens

Figure 3. Category distribution of problem instances we generated using CLI-Gym.

across different application domains. This broad coverage demonstrates the generality of our approach and its applicability to a wide range of real-world software environments.

FROM task-pandas:latest # Gold environment RUN mkdir -p /tmp/corrupted RUN cp /opt/.../lib/libsqlite3.so /tmp/corrupted

Notably, compared to Terminal-Bench, the tasks produced by our framework are accompanied by a substantially larger number of unit tests. This richer testing context enables a more reliable assessment of solution correctness.

/ RUN cp /opt/.../lib/libz.so /tmp/corrupted/ # Corrupt ELF headers to break dynamic linking RUN dd if=/dev/zero of=/tmp/corrupted/libsqlite3

.so \

bs=1 count=24 seek=8 conv=notrunc RUN dd if=/dev/zero of=/tmp/corrupted/libz.so \ bs=1 count=24 seek=8 conv=notrunc

Human Labor Comparison. Terminal-Bench was constructed through crowd-sourced open-source contributions, with 93 contributors, including expert and senior engineers, producing a total of 229 tasks. In contrast, our pipeline is fully automated and operates without any human intervention. The production cost for the 1,655 tasks generated by our method amounts to 2.3 billion tokens.

# Replace libraries with corrupted versions RUN cp /tmp/corrupted/* /opt/.../lib/

Figure 4. A simplified example Dockerfile snippet that induces failures in a gold pandas environment by corrupting system libraries. The agent overwrites ELF headers of critical shared libraries (libsqlite3 and libz), inducing ImportError and failures of basic Linux commands that require system-level diagnosis beyond code repair.

Trajectory Collection with CLI-Gym. To demonstrate the effectiveness of CLI-Gym, we leverage the generated instances to collect agent trajectories for model training. Using strong language models as policy rollout agents, we execute tasks in the 1,655 generated environments and obtain 417 successful trajectories that correctly resolve the induced failures. To ensure training data quality, we filter out trajectories that either exploit shortcuts or unintended solutions or correspond to trivially easy problems solvable in only a few steps. After filtering, we retain 291 high-quality trajectories. These trajectories demonstrate diverse environment repair strategies, including dependency resolution, configuration debugging, and system-level troubleshooting. Detailed implementation and filtering criteria are provided in the appendix. Throughout the experiments, we refer to the 417 trajectories before filtering as Raw-Success Traj and the 291 filtered trajectories as Filtered-Success Traj.

fails, the instance is deemed a successfully generated task. The failing tests are subsequently treated as fail-to-pass test cases for evaluating whether the generated task has been correctly resolved. Finally, we leverage the F2P tests and their associated error feedback to automatically synthesize issue descriptions using a language model. Each generated CLI task instance consists of (i) an executable environment, (ii) a natural language issue description, and (iii) a set of unit tests specifying the desired behavior.

Through this inverse agentic pipeline, we are able to automatically generate a large-scale and diverse collection of realistic CLI task instances from oracle GitHub repositories, enabling scalable training of agentic coding systems.

### 4. Experiments

#### 3.3. CLI-Gym

To demonstrate the effectiveness of our method, we train LLMs on the trajectories collected via CLI-Gym and evaluate them using the OpenHands agent framework (Wang et al., 2025) on Terminal-Bench 1.0 and 2.0.

Environment Statistics. We apply our toolkit to 29 Python repositories randomly selected from the curated SWE-Smith repository set, resulting in a total of 1,655 generated CLI task instances. Figure 3 summarizes the distribution of tasks

- Table 3. Performance on Terminal-bench 1.0 and Terminal-bench 2.0. We report pass@1 scores for LiberCoder-32B and LiberCoder235B-A22B evaluated using the OpenHands agent framework. Results for other models are taken from the official Terminal-Bench leaderboards. Best Performance with Any Agent reports the best publicly available results (regardless of the agent framework used). Models marked with † are evaluated by us. We highlight the top-2 open-source entries with bold font in each column.

Model OpenSource Terminal-bench@1.0 Terminal-bench@2.0 Performance with OpenHands

Claude Haiku 4.5 (Anthropic, 2025e) ✗ - 13.9 Gemini 2.5 Pro (Comanici et al., 2025) ✗ - 16.4 Grok 4 (xAI, 2025) ✗ - 27.2 Claude Sonnet 4 (Anthropic, 2025a) ✗ 41.3 Claude Opus 4.1 (Anthropic, 2025b) ✗ - 36.9 Claude Sonnet 4.5 (Anthropic, 2025d) ✗ 42.7† 42.6 GPT-5 (Singh et al., 2025) ✗ - 43.8 Claude Opus 4.5 (Anthropic, 2025c) ✗ - 51.9

Qwen3-32B (Yang et al., 2025a) ✓ 10.3† 5.7† Qwen3-235B-A22B-Instruct (Yang et al., 2025a) ✓ 25.0† 18.1† Qwen3-Coder-30B-A3B-Instruct (Yang et al., 2025a) ✓ 26.5† 12.9† Qwen3-Coder-480B-A35B-Instruct (Yang et al., 2025a) ✓ - 25.4 Kimi-K2-Instruct (Team et al., 2025) ✓ - 26.7 LiberCoder-32B ✓ 38.9 19.5 LiberCoder-235B-A22B ✓ 46.1 31.0

Best Performance with Any Agent

Gemini 2.5 Pro (Comanici et al., 2025) ✗ 25.3 32.6 Grok 4 (xAI, 2025) ✗ 39.0 27.2 Claude Haiku 4.5 (Anthropic, 2025e) ✗ 41.8 29.8 Claude Opus 4.1 (Anthropic, 2025b) ✗ 43.8 38.0 Claude Sonnet 4.5 (Anthropic, 2025d) ✗ 51.0 42.8 Claude Opus 4.5 (Anthropic, 2025c) ✗ - 57.8 GPT 5.2 (OpenAI, 2025) ✗ - 62.9 Gemini 3 Pro (Google DeepMind, 2025) ✗ - 64.7

GPT-OSS-120B (Agarwal et al., 2025) ✓ - 18.7 Kimi-K2-Instruct (Team et al., 2025) ✓ 30.0 27.8 Qwen3-Coder-30B-A3B-Instruct (Yang et al., 2025a) ✓ 31.3 12.9† Qwen3-Coder-480B-A35B-Instruct (Yang et al., 2025a) ✓ 39.0 27.2 GLM-4.6 (Z.ai, 2025) ✓ 40.5 24.5 Minimax-M2 (MiniMax, 2025a) ✓ 42.0 30.0 Minimax-M2.1 (MiniMax, 2025b) ✓ - 36.6 LiberCoder-32B ✓ 38.9 19.5 LiberCoder-235B-A22B ✓ 46.1 31.0

- 4.1. Experimental Setups

Training. We fine-tune Qwen3-32B and Qwen3-235BA22B-Instruct using a two-stage training procedure. In the first stage, since these models were not optimized for agentic coding, we enhance this capability using a collection of 48K open-source software engineering trajectories, derived from the SWE series. This auxiliary dataset is disjoint from Terminal-Bench. In the second stage, we then fine-tune the models on the Filtered-Success Traj described above.

Agent Framework. We adopt OpenHands as the generalpurpose agent framework to induce environment corruption and subsequently perform task completion. OpenHands is a widely used open-source code agent framework that has been extensively evaluated on agentic coding benchmarks, including SWE-bench and Terminal-Bench. While OpenHands is not specifically optimized for Terminal-bench and may underperform compared to benchmark-specific agents (e.g., Terminus 2), its broad applicability and standardized interface better align with our goal of training general-purpose agentic models. Accordingly, we use OpenHands as the agent framework throughout this work.

Evaluation. We evaluate agents on Terminal-Bench 1.0 and 2.0 (Terminal-Bench Team, 2025), two widely adopted benchmarks for assessing agentic interaction with realworld terminal environments. Terminal-Bench comprises collections of environment manipulation and system-level

- Table 4. Ablation study on the effects of open-source agentic trajectories and our generated trajectories across model scales.

SWE Traj. Filtered-Success Traj. Pass@1 Pass@3 Qwen3-32B

10.3 19.1

✓ 22.1 (+11.8) 28.1 (+9.0)

✓ 32.4 (+22.1) 37.9 (+18.8)

✓ ✓ 38.9 (+28.6) 44.5 (+25.4) Qwen3-235B-A22B-Instruct

25.0 32.4

✓ 28.6 (+3.6) 34.5 (+2.1)

✓ 39.5 (+14.5) 46.2 (+13.8) ✓ ✓ 46.1 (+21.1) 53.9 (+21.5)

- Table 5. We ablate the effects of trajectory filtering by comparing models trained on filtered and raw successful trajectories, with and without extra open-source agentic trajectories, on Qwen3-32B.

|19<br><br>17| | |32|
|---|---|---|---|
|Q O<br><br>|wen3-23 urs|5B-A22B-|Instruct|
| | | | |

Incorrect edit

Incorrect localization

7

Context exceeded

7

12

1

Struck grepping

4

3

Partial completion

1

0 5 10 15 20 25 30 35

Number of Failed Instances

- Figure 5. Failure type distribution on Terminal-Bench@1.0. of our LiberCoder-235B-A22B and Qwen3-235B-A22B-Instruct.

29.5

33.4

23.1

30.0

12.5

12.5

28.6

0.0 0.0 0.0

10.0

20.0

30.0

40.0

50.0

60.0

70.0

software engineering

system administration security debugging

file operations

data science

model training

games

scientific computing

Improvement by 291 samples Qwen3-32B

Resolved(%)

- Figure 6. Category-wise performance (pass@3) of Qwen3-32B on Terminal-bench@1.0 before and after CLI-Gym training. We report absolute improvements across task categories.

SWE Traj. Raw-Success Traj. Filtered-Success Traj. Pass@1

✓ 32.4 ✓ 33.8

✓ ✓ 36.4 ✓ ✓ 38.9

tasks, 80 in v1.0 and 89 in v2.0, where agents must interact with a Linux environment via command-line interfaces to diagnose and resolve realistic software and system issues.

Following the standard Terminal-Bench evaluation protocol, a task is considered successfully solved if the environment passes the verification scripts. We report pass@1 and pass@3, where pass@k denotes the proportion of tasks solved by at least one successful run among k attempts.

#### 4.2. Main Results

Table 3 reports the performance of our trained models, LiberCoder-32B and LiberCoder-235B-A22B, in comparison with representative closed-weight and open-weight baselines on the Terminal-Bench 1.0 and 2.0 leaderboards. For a fair comparison, we consider only entries officially verified by the benchmark maintainers.

With supervised fine-tuning on merely 291 successful environment-repair trajectories, both LiberCoder variants exhibit substantial and consistent gains over their respective base models. Concretely, LiberCoder-32B and LiberCoder-235B-A22B improve upon Qwen3-32B and Qwen3-235B-A22B-Instruct by +28.6% and +21.1% on

- Terminal-Bench 1.0, and by +13.8% and +12.9% on
- Terminal-Bench 2.0, respectively, when evaluated with OpenHands. These results indicate that a small but carefully curated set of high-quality agentic trajectories can yield significant performance improvements in terminal-based coding environments.

- At the 32B scale, LiberCoder-32B achieves a score of 38.9 on Terminal-Bench 1.0, outperforming several substantially

larger open-weight models, including Qwen3-Coder-480BA35B-Instruct (480B parameters) and Kimi-K2-Instruct (approximately 1T parameters). This highlights the effectiveness of targeted agentic supervision over naive model scaling in complex CLI tasks. At a larger scale, LiberCoder235B-A22B further advances the state of the art among open-weight models on Terminal-Bench 1.0, reaching a score of 46.1. On the more challenging Terminal-Bench 2.0 benchmark, LiberCoder-235B-A22B attains 31.0, surpassing the majority of existing open-weight baselines, with Minimax-M2.1 being the only exception.

#### 4.3. Ablation studies

Agentic Coding Pretrained with SWE Tasks. Table 4 studies the impact of SWE-style task pretraining on agentic coding performance across different model scales. At both the 32B and 235B scales, pretraining on open-source agentic trajectories yields substantial improvements over the corresponding base models, underscoring the importance of such data for initializing agentic coding capabilities. Interestingly, training solely on CLI-Gym trajectories yields even larger gains, indicating that environment-centric supervision pro-

40

35

|pass@1 pass@3| |37.9|
|---|---|---|
|32.4|36|32.4<br><br>.5|
| |28|.9|
|22.6| | |
|19.1| | |
| | | |
|10.3| | |

|pass@1 pass@3| | |31.4|32.4|
|---|---|---|---|---|
| | |27.2|30.1| |
|21.2<br><br>24|.5| |21.8|22.6|
|17|.7|19.1|20.5| |
|14.7| | | | |

35

30

[Figure 46]

[Figure 47]

30

25

25

20

20

15

15

10

5

10

0 100 200 300

1 2 4 8 16 32

Number of repositories

Number of trajectories

Figure 7. Effect of environment diversity under a fixed data budget. We vary the number of source repositories while keeping the total number of 100 CLI-Gym trajectories fixed.

Figure 8. Effect of trajectory scaling on Terminal-bench@1.0 performance. We fine-tune the Qwen3-32B model using different proportions of the 291 Filtered-Success trajectories.

vides a stronger inductive bias for terminal-based interaction. Combining SWE and CLI-Gym data consistently achieves the best performance, suggesting that generic software engineering priors and specialized environment interaction skills are complementary.

signals. However, the improvements plateau beyond approximately 200 trajectories, suggesting that data quality and task diversity may be more important than quantity.

#### 4.4. Results and Failure Case Visualization

Environment Diversity via Scaling Repositories. Different repositories induce distinct system configurations, resulting in diverse failure modes and recovery dynamics. To isolate the effect of repository diversity, we perform an ablation in which the total number of training trajectories is fixed at 100, while varying the number of source repositories used for data generation.

Category-wise Improvement. Figure 6 illustrates the category-wise performance changes after training with CLIGym. We observe substantial improvements across all environment-intensive categories, including software engineering, system administration, security, and debugging, with performance gains exceeding 20 points. In contrast, categories such as gaming and scientific computing remain challenging and are not addressed by our method, which we identify as promising directions for future research.

As shown in Figure 7, performance improves monotonically as more repositories are included, despite the total data volume remaining constant. This demonstrates that diversity in environments, rather than trajectory count alone, is a key factor in learning robust agentic coding behaviors.

Failure Type Distribution. Figure 5 presents the distribution of failure types on Terminal-Bench@1.0 for Qwen3235B-A22B-Instruct before and after training with our data. The trained model exhibits a substantial reduction in errors related to editing and localization. We further observe that, after training, the model tends to engage in more extensive exploration, which increases the likelihood of exceeding the maximum inference context length (128k), highlighting an important direction for future optimization.

Filtering Low-Quality Trajectories. Table 5 analyzes the impact of trajectory quality filtering under different training settings. As shown in the first two rows, when models are trained without SWE-relevant agentic coding pretraining, using unfiltered trajectories (417) yields slightly better performance than using filtered trajectories (291). In contrast, the third and fourth rows show that when agentic coding pretraining is applied to initialize agentic capabilities, training on high-quality trajectories significantly outperforms training on lower-quality trajectories, despite the latter containing a larger number of samples. These results indicate that once agentic capabilities are sufficiently established, trajectory quality becomes more critical than data quantity.

### 5. Conclusion

In this paper, we present CLI-Gym , the first publicly available approach for scaling training environments of CLI agentic coding tasks. Our approach represents each environment using a Dockerfile for precise configuration and version control, and employs agents to simulate environment histories. Based on this toolkit, we curate 1,655 task instances and collect 291 successful trajectories. Experiments show that fine-tuning on our data substantially enhances environmentcentric agentic coding, leading to top-tier performance on Terminal-Bench among open-source models.

Data Scaling. Figure 8 examines the effect of training data scale. We train models using progressively larger subsets of the filtered CLI-Gym trajectories, while keeping the opensource agentic trajectories fixed. Performance consistently improves as more successful trajectories are added, indicating that our environments provide effective supervision

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Anthropic. Introducing claude 4. https://www. anthropic.com/news/claude-4, 2025a. Accessed: 2026-01-20.

Anthropic. Introducing claude 4.1 opus. https://www. anthropic.com/news/claude-opus-4-1, December 2025b. Accessed: 2026-01-27.

Anthropic. Introducing claude 4.5 opus. https://www. anthropic.com/news/claude-opus-4-5, November 2025c. Accessed: 2026-01-27.

Anthropic. Introducing claude sonnet 4.5. https://www. anthropic.com/news/claude-sonnet-4-5, September 2025d. Accessed: 2026-01-26.

Anthropic. Introducing claude haiku 4.5. https://www. anthropic.com/news/claude-haiku-4-5, October 2025e. Accessed: 2026-01-26.

Anthropic. Claude Code. https://github.com/ anthropics/claude-code, 2025. Accessed: 2026-01-26.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Du, Y., Cai, Y., Zhou, Y., Wang, C., Qian, Y., Pang, X., Liu, Q., Hu, Y., and Chen, S. Swe-dev: Evaluating and training autonomous feature-driven software development. arXiv preprint arXiv:2505.16975, 2025.

Google. Gemini CLI. https://github.com/ google-gemini/gemini-cli, 2025. Accessed:

- 2026-01-26.

Google DeepMind. A new era of intelligence with gemini 3. https://blog.google/ products-and-platforms/products/ gemini/gemini-3/, November 2025. Accessed:

- 2026-01-27.

Jain, N., Singh, J., Shetty, M., Zheng, L., Sen, K., and Stoica, I. R2e-gym: Procedural environments and hybrid verifiers for scaling open-weights swe agents. arXiv preprint arXiv:2504.07164, 2025.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. R. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Liu, J., Wang, K., Chen, Y., Peng, X., Chen, Z., Zhang, L., and Lou, Y. Large language model-based agents for software engineering: A survey. arXiv preprint arXiv:2409.02977, 2024.

Merrill, M. A., Shaw, A. G., Carlini, N., Li, B., Raj, H., Bercovich, I., Shi, L., Shin, J. Y., Walshe, T., Buchanan, E. K., et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

MiniMax. Minimax m2: An efficient model for the agentic era. https://www.minimax.io/news/ minimax-m2, October 2025a. Accessed: 2026-01-27.

MiniMax. Minimax m2.1: Significantly enhanced multi-language programming, built for real-world complex tasks. https://www.minimax.io/news/ minimax-m21, December 2025b. Accessed: 202601-27.

OpenAI. Codex. https://github.com/openai/ codex, 2025. Accessed: 2026-01-26.

OpenAI. Introducing gpt-5.2. https://openai.com/ index/introducing-gpt-5-2/, December 2025. Accessed: 2026-01-27.

Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., and Zhang, Y. Training software engineering agents and verifiers with swe-gym. arXiv preprint arXiv:2412.21139, 2024.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Terminal-Bench Team. Terminal-bench: A benchmark for ai agents in terminal environments, Apr 2025. URL https://github.com/laude-institute/ terminal-bench.

Wang, X., Rosenberg, S., Michelini, J., Smith, C., Tran, H., Nyst, E., Malhotra, R., Zhou, X., Chen, V., Brennan, R., et al. The openhands software agent sdk: A composable and extensible foundation for production agents. arXiv preprint arXiv:2511.03690, 2025.

xAI. Grok 4. https://x.ai/news/grok-4, July

2025. Accessed: 2026-01-27.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, J., Lieret, K., Jimenez, C. E., Wettig, A., Khandpur, K., Zhang, Y., Hui, B., Press, O., Schmidt, L., and Yang, D. SWE-smith: Scaling data for software engineering agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (NeurIPS), 2025b.

Z.ai. Glm-4.6. https://z.ai/blog/glm-4.6, 2025. Accessed: 2026-01-28.

### A. Detailed CLI-Gym Pipeline

This section presents the full technical details of the CLI-Gym pipeline. We construct gold instances from 29 different open-source GitHub repositories. A gold instance consists of the environment, codebase, and unit tests. Based on these gold instances, we further generate task instances. The repository information, the number of unit tests per repository, and the final number of task instances produced are summarized in Table 9.

#### A.1. Query Generation

After constructing a repository as a gold instance, we first randomly sample 1–3 intervention directions and randomly select 200 unit tests from the repository, which are injected into a prompt with previous task titles and provided to LLM to obtain an initial task prompt. The specific prompt is shown in Figure 10.

Once a task prompt in a predefined format is generated, we randomly apply a second-stage prompt refinement to make the task more closely aligned with the selected unit tests, improving both yield and diversity. The refinement prompt used in our pipeline is shown in Figure 11.

After generating a task prompt specification, we embed it into a task template (Figure 12) and package it as an agent task. Each task consists of: (1) a Docker-compose.yaml file for mapping trajectories, logs, and other artifacts between the executing container and local path; (2) a Dockerfile specifying the base image; (3) a run-tests.sh script containing the unit tests execution commands; and (4) a task.yaml file that provides the concrete task prompt.

#### A.2. Environment Inversion

We execute agentic tasks using a modified Terminal-Bench harness. Specifically, docker-compose.yaml is used to mount trajectory files and logs into the container and to invoke the Dockerfile, which specifies the base image and launches the agent. The agent is then prompted with the task prompt that specifies how an agent should deliberately induce failures and allow any operation to interact with the environment and finish the task. Each task is executed for approximately 15 minutes and is terminated when the agent outputs a final thought. A sample trajectory is shown in Figure 13.

After task execution, the harness automatically runs run-tests.sh inside the container to evaluate the selected unit tests. Based on the test outcomes, we apply the following rules:

- • If some unit tests fail, the failed tests are recorded as fail-to-pass tests, while the successful ones are recorded as pass-to-pass tests.
- • If the test command fails to execute, all selected unit tests are recorded as fail-to-pass tests.
- • If all unit tests pass, the task is considered unsuccessful and discarded.

In addition, the harness prompts the agent to summarize a Dockerfile that captures the degraded environment, enabling deterministic reproduction of the failure that the agent built. A complete Dockerfile demo is shown in Figure 20.

#### A.3. Task Generation

After the inverse task is completed, we generate a corresponding problem statement targeting the induced faults. The problem statement is constructed from the original task prompt together with the failed unit tests information. Concretely, we randomly select one of three prompts with different levels of guidance: the prompt shown in Figure 14 asks the LLM to generate a more explicit and strongly guided issue description, while the prompt shown in Figure 15 encourages a weaker, less directive issue formulation, and the prompt shown in Figure 16 balances direction and difficulty.

Regardless of which prompt is used, the LLM is required to output a hint. This hint can optionally be removed by a rule-based filter, allowing us to derive two distinct repair tasks (with or without hints) from the same issue.

The generated statement is then inserted into the problem task template (Figure 17). Together with the extracted unit tests and Dockerfile, this forms a complete repair task.

All run-tests.sh templates used in our pipeline are provided in Figure 18. The agent framework used throughout the pipeline is OpenHands.

- B. Detailed Experiments In this section, we present the full experimental configurations and execution details.

#### B.1. Training Set Construction

Starting from 29 gold instances, we generated 4,066 task prompts, which resulted in 1,655 problem instances, including faulty images, failed unit tests, and Dockerfile which induces failures. Using strong language models with OpenHands, we collected 417 successful trajectories. We then filtered out 126 trajectories, retaining 291 Filtered-Success trajectories for training.

The filtering criteria are as follows. (1) Trajectories with fewer than 20 steps. We consider the number of interaction steps to be correlated with task difficulty; therefore, trajectories that solved tasks with few steps were removed, as they typically correspond to trivial or low-difficulty environments. (2) Cheating trajectories. We discarded trajectories that exploited historical artifacts such as cached Git information, Conda logs, or other unintended shortcuts to solve the task, bypassing the intended problem-solving process.

#### B.2. Training Details

For Qwen3-32B: The learning rate is initialized at 2 × 10−5 and follows a cosine decay schedule. To stabilize the early stage of training, we employ a linear warmup strategy for the first 5% of the total training steps, during which the learning rate increases linearly from a minimum 1 × 10−6 to 2 × 10−5. We train models for 10, 15, and 20 epochs and report the checkpoint with the best validation performance. The batch size is set to 16. We adopt qwen3-coder as the agent template. The maximum sequence length is 100k tokens, achieved by extending the native 40k context window using YaRN with an expansion factor of 2.5.

For Qwen3-235B-A22B-Instruct: The learning rate is initialized at 1 × 10−5 and follows a cosine decay schedule. To stabilize the early stage of training, we employ a linear warmup strategy for the first 5% of the total training steps, during which the learning rate increases linearly from a minimum 1 × 10−6 to 1 × 10−5. Similarly, we train the model for 10, 15, and 20 epochs and select the best-performing checkpoint. We use qwen3-coder as the agent template and set the maximum sequence length to 100k tokens.

#### B.3. Agent and Inference Details

We adopt the OpenHands agent framework (Wang et al., 2025) as the execution interface between language models and CLI environments. OpenHands provides a unified Agent–Computer Interface (ACI) that enables models to interact with containerized systems through structured tool calls, including shell command execution, file editing, and environment inspection. At inference time, each model operates as a single autonomous agent that iteratively observes environment feedback and issues actions until the task is solved or a termination condition is reached.

Action space. The agent is allowed to invoke the tools in Figure 19, All actions are executed inside isolated Docker containers constructed for each task.

Termination conditions. An episode terminates when one of the following conditions is met: (1) the agent explicitly calls the finish tool; (2) a global time limit setting by Terminal-Bench is exceeded.

Decoding configuration. During inference, we use a greedy decoding strategy where temperature = 0 and decoding with a maximum context length of 128k tokens. The same decoding configuration is applied to all models for fair comparison.

#### B.4. Ablation Studies and Case Visualization

Agentic Coding Pretraining with SWE Tasks. The open-source SWE-style trajectories used in this paper strictly DO NOT contain any Terminal-Bench task, ensuring that there is no risk of benchmark contamination or evaluation leakage. These trajectories are only used to initialize the models’ general agentic coding abilities, such as repository navigation, tool invocation, and multi-step program repair, rather than environment-specific knowledge. In this setting, models are first trained on the SWE-style trajectories for 1 epoch, followed by fine-tuning on the Filtered-Success trajectories for 15 epochs. All other training hyperparameters, including learning rate, batch size, context length, optimizer, and agent interface, are kept identical to the main setting.

Environment Diversity via Scaling Repositories. When multiple repositories are involved, trajectories are uniformly sampled to ensure balanced coverage across repositories, avoiding dominance by any single codebase or environment configuration. The training protocol is as follows: each dataset is trained under its respective best-performing epochs with the same other settings.

Filtering low-quality Trajectories. In this experiment, each dataset is trained using its best-performing configuration (e.g., 15 epochs for Filtered-Success trajectories and 10 epochs for Raw-Success trajectories). All other factors are strictly the same. This design avoids confounding effects from underfitting or overfitting and ensures that the observed differences are attributable to data quality rather than training instability.

Category-wise Improvement. We compare the baseline Qwen3-32B model and our LiberCoder-32B on TerminalBench@1.0 across different task categories defined in the official Terminal-Bench registry. Since our goal is to measure category-level capability improvement induced by training, we report and compare pass@3 scores, which better reflect the models’ capacity.

Failure Type Distribution. We analyze the failure type distribution of the baseline Qwen3-235B-A22B-Instruct model and our LiberCoder-235B-A22B on Terminal-Bench@1.0. For both models, we collect failure cases from a full evaluation run and categorize all unresolved tasks according to their observed failure modes.

- B.5. More Experiments Impact of Different Agents

We sampled some officially validated scoring examples from the leaderboard to demonstrate the impact of different agents, among which the OpenHands we used did not perform very well, as shown in Table 6.

Table 6. Performance Comparison on Terminal-Bench@2.0

Claude Haiku 4.5 Agent Score

Terminus 2 28.3 Claude Code 27.5 OpenHands 13.9

###### Claude Opus 4.5 Agent Score

Terminus 2 57.8 Claude Code 52.1 OpenHands 51.9

Comparison with other datasets. Table 7 compares representative datasets and benchmarks in terms of the number of task instances, base environments, and storage footprint. By leveraging CLI-Gym, our dataset uses a small set of base images systematically generate a large number of CLI-centric problem instances. This results in a substantially lower storage footprint.

- Table 7. Comparison of representative datasets in terms of scale and storage footprint. We report the number of task instances (#Instance), the number of base images/environments (#Images), and the required storage size.

DataSet #Instances #Images Size Code-centric SWE-gym 2438 2438 6 TBs SWE-smith 50137 128 295 GBs R2E-gym 4578 – 4 TBs CLI-centric

- Terminal Bench@1.0 80 14 192 GBs
- Terminal Bench@2.0 89 11 235 GBs Ours 1655 29 119 GBs

Gain with Hint. We conducted an ablation study to analyze the effect of introducing hints when generating problem statements in CLI-Gym. Specifically, this variant augments the repair issue with additional hints extracted from the task prompt that induces failure, with the goal of facilitating trajectory generation and increasing data yield.

- Table 8. Ablation study on hint-augmented repair issue generation. Adding hints substantially increases the number of valid trajectories produced by CLI-Gym, leading to improved downstream performance. When controlling for data scale, hints alone do not significantly affect performance.

Setting # Trajectories Performance

w/o hints 104 23.0 w/ hints (subsampled) 104 22.8 w/ hints (full) 291 32.4

As shown in Table 8, adding hints significantly increases the number of usable trajectories, enabling us to collect 291 trajectories instead of 104, which leads to a substantial performance improvement. When controlling for data scale by subsampling the full dataset to the same size (104 trajectories), the performance remains comparable.

Performance Benefits Model Behavior. Figure 9 analyzes the relationship between overall performance and the frequency of stuck failures. We observe a strong negative correlation between pass@1 performance and the proportion of trajectories in which the agent becomes stuck in repetitive action loops. As the number of Filtered-Success trajectories increases, model performance steadily improves, while the incidence of looped behavior drops sharply from 42.7% to 3.0%. This result suggests that environment-repair supervision not only improves success rates, but also substantially enhances the agent’s ability to escape unproductive interaction patterns and maintain effective long-horizon control.

%Stuckinloop

Performance vs Stuck in loop

50

42.7

40

30

26.5

20

9.2

10

3

0

25 30 35 40

Performance (pass@1)

Figure 9. Relationship between task performance and the proportion of failure cases stuck in action loops. Each point corresponds to a Qwen3-32B model fine-tuned with an increasing number of filtered-success trajectories and evaluated on Terminal-Bench@1.0. The x-axis reports pass@1 performance, and the y-axis shows the percentage of tasks where the agent becomes stuck in repetitive loops.

Table 9. Statistics and descriptions of 29 repositories we used (total task instances: 1655).

Repo Unit Tests Task Instances Description apispec 569 100 Pluggable API specification generator for web frameworks. autograd 585 20 Efficiently computes derivatives of numpy code automatically. cantools 372 120 Tools for CAN bus signal decoding, encoding and editing. click 499 2 Composable command line interface toolkit for Python. cookiecutter 356 4 Utility for creating projects from project templates. dominate 65 40 Python library for creating and manipulating HTML documents. environs 105 2 Simplified environment variable parsing for Python apps. exceptiongroup 89 87 Backport of PEP 654 exception groups for older versions. faker 2103 71 Generates fake data for testing, databases, and filling. feedparser 4229 4 Robust RSS and Atom feed parser for Python developers. flashtext 39 11 Efficient keyword search and replacement in large texts. fvcore 140 58 Shared toolkit used across Facebook AI Research projects. glom 178 23 Powerful tool for restructuring and transforming data. godotenv 52 3 Loads environment variables from .env files into system. gspread 149 26 Python wrapper for Google Sheets API v4 interaction. gunicorn 248 160 Pre-fork worker model WSGI HTTP Server for UNIX. h11 76 26 Pure-Python, zero-I/O implementation of HTTP/1.1. iniconfig 28 2 Small and simple configuration parser for .ini files. markupsafe 70 58 Safely escapes characters for use in HTML and XML. modin 409 5 Scales pandas workflows by changing a single line of code. pandas 109403 242 High-performance data analysis and manipulation library. ptyprocess 12 109 Launches a subprocess in a pseudo terminal (pty). pyopenssl 449 34 Python interface to the OpenSSL cryptography library. python-dotenv 178 71 Reads key-value pairs from .env files for local dev. routers 23 24 Lightweight and fast routing implementation for web. scrapy 2882 119 High-level web crawling and data scraping framework. slugify 81 3 Converts strings to clean, URL-friendly slug strings. textdistance 427 82 Computes distance between sequences with many algorithms. tweepy 148 110 Python library for interacting with the Twitter API.

#### Degradation Issue Generation Prompt 1

# Role: Chaos Engineering Lead and QA Expert # Background: We are testing the OpenHands agent in a secure Docker container. # Objective: Design a one-way or multi-directional environmental disruption task

based on the Disruption Direction, and select some unit tests from the List of Candidate UTs to verify whether the disruption was successful.

## Input Data List of Candidate UTs: {candidate_uts_list}

## Disruption Direction {directions}

## Constraints (Key)

- 1. **Natural Language**: The task description must be a clear, human-readable goal ( e.g., "perform certain actions to achieve a specific objective"). Some bash commands for reference can be provided to help the agent complete the disruption task.
- 2. **Causality**: The chosen disruptions must logically cause the selected UT (unit test) to fail, and no more than 50 UTs should be selected.
- 3. **Complexity**: The generated disruption tasks should have a certain level of difficulty to solve. They should also not leave backup files or allow bypassing expected recovery methods.

In addition, they should involve recovery challenges such as:

- - Tampering with system paths/files, causing kernel/system issues, e.g., VFS: unable to mount root filesystem, with the error message "unknown-block(0, 0) ". Do not simply mimic this issue.
- - Encrypting documents that are difficult to obtain through other means.
- - ...{more examples} (Do not limit yourself to the above examples.)

5. **Diversity**: I have already generated the following tasks, please do not generate tasks with similar themes. The methods of causing damage do not necessarily have to be related to Python, nor do they necessarily need to be implemented using Python. Think outside the box.

## Generated Tasks {os.listdir(dataset_path)}

## Output Format Strictly follow the following Markdown format:

---

**Task Name**: <Short Title>

**Category**: <Single word, e.g., Data>

**Selected UTs**:

- - <Path to UT 1>
- - <Path to UT 2>

**Task Description**: <Detailed natural language instructions provided to the agent.

Describe the **goal** and **steps** to create the vulnerability, and let the agent verify the vulnerability.>

**Expected Result**: <The error that should occur>

**Recovery Strategy**: <How to fix it>

---

Figure 10. Degradation Issue Generation Prompt 1

#### Degradation Issue Generation Prompt 2

# Role: Chaos Engineering Lead (Refinement) # Context: Refine the draft sabotage task to be a high-quality directive for an AI

Agent. ## Input Data

- 1. **Target UTs (Paths)**: {extract_result[’selected_uts’]}
- 2. **Draft Task**: {extract_result} ## Refinement Objectives

- 1. **Natural Language Refinement**: Ensure ‘Task Description‘ is written in clear, instructional English. It should explain *what* to do (the objective) rather than

just listing code.

- * Bad: "Run ‘chmod 000 file‘"
- * Good: "Locate the log file and modify its permissions so that it is not writable by the current user."

- 2. **Verification Step**: Add a sentence at the end of ‘Task Description‘ asking the agent to: "Run the target Unit Tests to confirm the failure."
- 3. **Logic Check**: Ensure the task strictly targets the logic of the provided " Target UTs".

## Output Format Retain the exact Markdown format.

---

**Task Name**: <Refined Title>

**Category**: <Same>

**Selected UTs**: <List the input Target UTs paths here, one per line>

**Task Description**: <Refined natural language objective + verification request>

**Expected Result**: <Detailed error>

**Recovery Strategy**: <Refined recovery>

---

Figure 11. Degradation Issue Generation Prompt 2

#### Degradation Instruction Template

instruction: | You are in a completely isolated and secure Docker container, where you can do

anything to complete the following task: {degradaton_issue}

I will run these tests using a conda environment named "testbed" with the command "cd /testbed && source /opt/miniconda3/bin/activate && conda activate testbed && pytest [UTs]." I will confirm whether you have successfully completed the task by observing whether any tests fail. Therefore, please use the Python environment located in /opt/miniconda3/envs/testbed/bin throughout the entire process.

Available UTs: {UTs}

After completing the task, please summarize a Dockerfile. I want to append this Dockerfile to the original image’s Dockerfile that you had before executing the

task, so that I can reproduce the environment after the task was executed. Based on your memory, write all meaningful operations you performed from start to finish in the form of Dockerfile commands in the /logs/Dockerfile file. You do not need to include any commands you did not execute, nor do you need to construct a fully functional Dockerfile from scratch because I will append it to the original image. Also, there is no need to guess how the original image was built; just write down the commands you actually executed. Since my entire process is automated, no comments or explanations are required. Pay attention to writing multi-line instructions in accordance with Dockerfile syntax.

Do not leave any introductory documents or code comments (e.g., INTRODUCTION.md)! Do not leave backup files (e.g., init.py.bak and file in /tmp). Do not make the

environment look like it has been deliberately sabotaged (e.g., by adding a sitecustomize.py); make the sabotage appear more natural. After tampering with the files, do not change their names, as this will alert the forensics experts.

Remember to delete files like .git, conda package cache and so on, as they can make solving the problem too easy.

Figure 12. Degradation Instruction Template

#### Agent Working Demo

{

"id": 69, "source": "agent", "message": "Command ‘cat > /etc/profile.d/02-set-interval-locale.sh << ’EOF’

nexport LC_ALL=en_US.interval nexport LANG=en_US.interval nexport LC_NUMERIC= en_US.interval nexport LC_MONETARY=en_US.interval EOF‘ executed with exit code 0.", "action": "run", "function": {

"arguments": "{"command": "cat > /etc/profile.d/02-set-interval-locale.sh << ’EOF’ nexport LC_ALL=en_US.interval nexport LANG=en_US.interval nexport LC_NUMERIC=en_US.interval nexport LC_MONETARY=en_US.interval EOF", "

security_risk": "MEDIUM"}", "name": "execute_bash"

},

... }

... {

"id": 131, "source": "agent", "message": "All done! What’s next on the agenda?", "action": "finish", "function": {

"arguments": "{"message": "Task completed successfully. I have introduced subtle numeric parsing corruption through locale manipulation that will cause pandas IntervalIndex operations to fail when comparing and constructing interval boundaries.**What was done:**1. **Created corrupted

locale ‘en_US.interval‘**: Generated a custom locale variant with swapped decimal point (comma) and thousands separator (period), conflicting with Python’s default numeric parsing expectations.2. ** Modified LC_NUMERIC and LC_MONETARY files**: Set decimal_point to "," and

thousands_sep to "." in both numeric and monetary formatting rules.3. ** Disabled system locale archive**: Renamed ‘/usr/lib/locale/locale-archive ‘ to force dynamic loading of individual locale directories.4. **Set system-wide environment variables**: Created ‘/etc/profile.d/02-setinterval-locale.sh‘ to export LC_ALL, LANG, LC_NUMERIC, and LC_MONETARY to use the corrupted locale.5. **Poisoned Python’s numeric parsing cache

**: Created ‘sitecustomize.py‘ in the testbed conda environment’s sitepackages that: - Sets the corrupted locale on Python interpreter startup - Calls C library’s ‘strtod‘ function with ambiguous numeric strings like "1,5", "2.3", "3,14159" to create cached parsing states6. ** Cleaned up forensic traces**: Removed pip cache, conda cache, conda package cache, and the .git directory.. The environment is now ready. You

can run the pandas IntervalIndex unit tests using the testbed conda environment to observe the numeric parsing corruption causing interval boundary comparison and construction errors."}",

"name": "finish" },

... }

Figure 13. Agent Working Demo

#### Repair Issue Generation Prompt 1

# Role: QA Engineer / Bug Reporter # Objective: Report a bug to the AI agent without revealing the solution, forcing

the agent to investigate. ## Background (Input Data)

**Malicious Behavior (Root Cause: The current environment is a test environment that

has been affected by the malicious task described as follows):** {data["task_description"]}

**Unit Tests that actually failed (should be mentioned in the bug report):** {symptoms_UTs}

## Your Task Write a natural language bug report to the agent.

- 1. **Describe the Symptoms**: Summarize a recoverable issue. Describe what went wrong. Point out that some unit tests failed.
- 2. **Objective**: Ask the agent to **investigate the environment**, identify the root cause, and fix it.
- 3. **Constraints**: Do not provide a solution or analyze the problem; just describe what happened.

## Output Rules (Strict)

- - **Output only the instruction text.**
- - Do not use Markdown headings, do not use "Task Name," and do not use code blocks.
- - The output should look like a user asking for help: "I can’t... Can you figure out the reason and help me fix this bug?"

## Output Format (Strict) <Help information, only mentioning the failed unit tests> Hint: <Describe where the problem might be based on the description of the malicious task,

but do not provide a solution. Do not specify exactly where or what the problem is!>

Figure 14. Repair Issue Generation Prompt 1

#### Repair Issue Generation Prompt 2

# Role: QA Engineer / Bug Reporter # Objective: Report issues to the AI agent without revealing the solution, forcing

the agent to investigate. ## Background (Input Data)

**Malicious Behavior (Root Cause: The current environment is a test environment that

has been affected by the malicious task described as follows):** {data["task_description"]}

**Unit Tests that actually failed (should be mentioned in the bug report):** {symptoms_UTs}

## Your Task Write a natural language help message for the agent.

- 1. **Direct Request for Help:** State that I have some UTs that are failing and need help to fix them.
- 2. **Describe the Symptoms:** Summarize a recoverable problem. Describe what went wrong and what kind of help is needed.
- 3. **Constraints:** Do not provide a solution or analyze the problem. Just describe what happened.

## Output Rules (Strict)

- - **Output only the instruction text.**
- - Do not use Markdown headings, do not use "Task Name," and do not use code blocks.
- - I will replace it into a YAML file. Pay attention to indentation and symbol conventions (e.g., do not use "-").
- - The output should look like a user asking for help: "I can’t... Can you help me figure out why and tell me how to fix it?"

## Output Format (Strict) <Help message, only referring to the UTs that are failing> hint: <Describe where the problem might be based on the description of the malicious task,

but do not provide a fix. Do not specify exactly where or what the problem is!>

Figure 15. Repair Issue Generation Prompt 3

#### Repair Issue Generation Prompt 3

# Role: QA Engineer / Bug Reporter # Objective: Report issues to the AI agent without revealing the solution, forcing

the agent to investigate. ## Background (Input Data)

**Malicious Behavior (Root Cause: The current environment is a test environment that

has been affected by the malicious task described as follows):** {data["task_description"]}

**Unit Tests that actually failed (should be mentioned in the bug report):** {symptoms_UTs}

## Your Task Write a natural language help message for the agent.

- 1. **Direct Request for Help:** State that I have some UTs that are failing and need help to fix them.
- 2. **Describe the Symptoms:** Summarize a recoverable problem. Describe what went wrong and what kind of help is needed.
- 3. **Constraints:** Do not provide a solution or analyze the problem. Just describe what happened.

## Output Rules (Strict)

- **Output only the instruction text.**

- - Do not use Markdown headings, do not use "Task Name," and do not use code blocks.
- - The output should look like a user asking for help: "I can’t... Can you help me figure out why and tell me how to fix it?"

## Output Format (Strict) <Help message, only referring to the UTs that are failing> hint: <Describe where the problem might be based on the description of the malicious task,

but do not provide a fix. Do not specify exactly where or what the problem is!>

Figure 16. Repair Issue Generation Prompt 3

#### Repair Instruction Template

instruction: | {problem_statement}

Figure 17. Repair Instruction Template

#### Run-tests Template

set -uo pipefail -x cat > tester.py <<EOF import os import subprocess, sys sys.path.insert(0, ’/testbed’) def run_and_log(cmd, log_path):

with open(log_path, "w", buffering=1, encoding="utf-8") as logf:

p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True, executable="/bin/bash"

) for line in p.stdout:

line = line.replace("\r", "\n") sys.stdout.write(line) logf.write(line)

return p.wait() run_and_log(

’source /opt/miniconda3/bin/activate; conda activate testbed; pytest --disable-

warnings --color=no --tb=no --verbose {UTs}’, "/test.log"

) EOF

chmod +x tester.py python tester.py

Figure 18. Run-tests Template

#### Tools

"function": { "name": "execute_bash", "description": "Execute a bash command in the terminal within a persistent shell

session.### Command Execution* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially , use ‘&&‘ or ‘;‘ to chain them together.* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands...",

}, "function": {

"name": "think", "description": "Use the tool to think about something. It will not obtain new

information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.Common use cases:1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.2...",

}, "function": {

"name": "finish", "description": "Signals the completion of the current task or conversation.Use

this tool when:- You have successfully completed the user’s requested taskYou cannot proceed further due to technical limitations or missing information.The message should include:- A clear summary of actions taken and their results- Any next steps for the user- Explanation if you’re unable to complete the task- Any follow-up questions if more information is needed",

}, "function": {

"name": "execute_ipython_cell", "description": "Run a cell of Python code in an IPython environment.* The

assistant should define variables and import packages before using them.* The

variable defined in the IPython environment will not be available outside the IPython environment (e.g., in terminal).",

}, "function": {

"name": "task_tracker", "description": "This tool provides structured task management capabilities for

development workflows.It enables systematic tracking of work items, progress monitoring, and efficient organization of complex development activities.The tool maintains visibility into project status and helps communicate progress effectively to users.## Application Guidelines Utilize this tool in the following situations:...",

}, "function": {

"name": "str_replace_editor", "description": "Custom editing tool for viewing, creating and editing files in

plain-text format* State is persistent across command calls and discussions with the user* If ‘path‘ is a text file, ‘view‘ displays the result of applying ‘cat -n‘. If ‘path‘ is a directory, ‘view‘ lists non-hidden files and directories up to 2 levels deep* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ". flac", ".pdf", ".docx"]. IT DOES NOT HANDLE IMAGES.* The ‘create‘ command cannot be used if the specified ‘path‘ already exists as a file* If a ‘ command‘ generates a long output, ...",

}

Figure 19. Tools

#### Dockerfile Demo

FROM task-pandas:latest RUN mkdir -p /corrupted_libs RUN cp /opt/miniconda3/envs/testbed/lib/libsqlite3.so.0.8.6 /corrupted_libs/

libsqlite3.so.0 RUN cp /opt/miniconda3/envs/testbed/lib/libz.so.1.2.13 /corrupted_libs/libz.so.1 RUN dd if=/dev/zero of=/corrupted_libs/libsqlite3.so.0 bs=1 count=24 seek=8 conv=

notrunc RUN dd if=/dev/zero of=/corrupted_libs/libz.so.1 bs=1 count=24 seek=8 conv=notrunc RUN rm /opt/miniconda3/envs/testbed/lib/libsqlite3.so.0.8.6 RUN cp /corrupted_libs/libsqlite3.so.0 /opt/miniconda3/envs/testbed/lib/libsqlite3.

so.0.8.6 RUN rm /opt/miniconda3/envs/testbed/lib/libz.so.1.2.13 RUN cp /corrupted_libs/libz.so.1 /opt/miniconda3/envs/testbed/lib/libz.so.1.2.13

Figure 20. Dockerfile Demo

