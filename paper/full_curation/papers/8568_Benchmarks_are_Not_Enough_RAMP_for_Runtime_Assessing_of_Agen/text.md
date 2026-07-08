# arXiv:2605.27492v1[cs.SE]26May2026

## Benchmarks are Not Enough: Ramp for Runtime Assessing of Agentic Models in Production Systems∗

Yipeng Ouyang, Xin Huang, Bingjie Liu, Zhongchun Zheng, Yuhao Gu, Xianwei Zhang†

School of Computer Science and Engineering, Sun Yat-sen University, Guangzhou, China {ouyyp5, huangx385, liubj8, zhengzhch3, guyh9}@mail2.sysu.edu.cn, {zhangxw79}@mail.sysu.edu.cn http://ramp.yatcc-ai.com/

[Figure 1]

How Far Can Today’s Agentic Models Go on the Ramp to Real System Development? T0 –T5 correspond to increasingly sophisticated implementation tasks from the YatCC production infrastructure project. A model reaches tier Tx only if it correctly and completely accomplishes every task up to Tx, exposing the practical limits of current LLMs in sustained multistage software development.

### Abstract

Large language model (LLM) agents are rapidly evolving from coding assistants into autonomous software engineering systems. However, existing evaluation methodologies remain largely centered on static, isolated, and short-horizon benchmarks that fail to capture the dynamic complexity of real-world production workflows. As a result, benchmark performance may poorly reflect practical capability under realistic runtime environments invovling long execution chains, tool interactions, dependency management, and iterative feedback loops. We thus present Ramp (Runtime Assessment of Models in Production), a production-grounded infrastructure for assessing long-horizon software engineering agents. Built upon the YatCC integrated platform, Ramp provides a unified runtime assessment architecture supporting heterogeneous LLM providers and agent SDKs through standardized orchestration and execution interfaces. Ramp introduces realistic compiler-construction workloads with serial dependencies and complex toolchain interactions, together with a staged recovery mechanism for analyzing

∗ Timeline: Idea proposed on April 11th, 2026; Project initiated on April 22nd; Repository and leaderboard open-sourced on May 10th; Draft completed on May 14th. † Corresponding author.

execution behavior under partial workflow failure. The framework further incorporates utility-oriented multi-dimensional metrics that jointly evaluate outcome quality and process efficiency. We conduct runtime assessments across 15 mainstream models and observe substantial capability degradation that remains largely invisible to conventional isolated benchmarks. Task completion rates progressively collapse across serial workflows, dropping from 100% in the initial stage to only 20.0% in the final stage, while none of the evaluated models successfully completes the entire pipeline. Runtime analysis further reveals systematic failure propagation patterns and significant inefficiencies in resource utilization, with computational costs differing by up to three orders of magnitude (2525x) among comparably performing models. These findings suggest Ramp advances agentic model evaluation toward continuous, runtime-observable, and production-grounded assessment.

### Keywords

agent model, production systems, compiler construction, dependency, resurrection

### 1 Introduction

Large language models (LLMs) have rapidly evolved from conversational assistants into autonomous software engineering agents capable of planning, implementing, debugging, and deploying increasingly complex software stacks and infrastructure workflows [3, 12, 16]. Recent agentic models have demonstrated strong capabilities in repository-level code generation, multi-file reasoning, iterative tool invocation, and workflow orchestration. However, despite these advances, the evaluation methodologies for such systems remain largely rooted in a benchmark-driven paradigm consisting of isolated, static, and short-horizon tasks, where each evaluation instance is executed independently and the agent state is reset after every task.

This evaluation paradigm exhibits a fundamental mismatch with real-world production engineering environments. In practice, production software and infrastructure engineering involve longhorizon iterative workflows spanning multi-file and cross-module reasoning, dependency management, runtime debugging, toolchain coordination,deploymentvalidation, continuous integration pipelines. Real production systems must further operate under imperfect runtime conditions including environment instability, resource constraints, asynchronous execution,and cascading failure propagation. Existing static benchmarks fail to capture these dynamics—they primarily measure instantaneous coding task completion rather than sustained engineering capability under evolving runtime conditions. Moreover, conventional benchmarks provide limited visibility into operational efficiency, execution stability, recovery behavior, and cost-performance tradeoffs, all of which are critical factors in practical deployment environments.

As a result, strong benchmark performance does not necessarily imply practical engineering utility. Models that perform well on benchmark-style coding may still fail in realistic production environments where runtime correctness, execution robustness, and adaptive recovery become critical. In many cases, failures originate not from insufficient code generation ability, but from unstable tool interaction, environment inconsistency, cascading execution errors, and inefficient runtime orchestration. Existing benchmark paradigms largely overlook these system-level behaviors because they evaluate isolated task completion rather than continuous execution under evolving runtime conditions. We therefore argue that the field must move beyond static benchmarks toward productiongrounded assessment workloads to gauge agentic models in realistic, multi-stage engineering environments with persistent runtime state, controlled intermediate state injection, multi-dimensional metrics, and systematic failure diagnosis. Such assessment should not only measure whether a task succeeds, but also characterize how the system behaves throughout the execution process.

To this end, we present Ramp (Runtime Assessment of Models in Production), a production-grounded assessment infrastructure built upon the YatCC integrated serving platform. Ramp provides a unified evaluation architecture supporting heterogeneous LLM providers and agent SDKs, through standardized API access and orchestration services. Unlike conventional benchmark pipelines, Ramp assesses models using realistic long-horizon compiler construction workloads organized as serial dependency chains, reflecting practical software engineering and infrastructure development

#### Table 1: Compilation tasks of YatCC. Each task consumes the output artifact of its predecessor, which can form an serial dependency.

###### Task Name Objective Input Artifact

- T0 Env. Setup Build & verify toolchain —
- T1 Lexer Tokenize C source code Preprocessed source
- T2 Parser Construct AST/ASG Token stream (from T1)
- T3 IR Gen. Emit LLVM IR AST/JSON (from T2)
- T4 IR Opt. Implement LLVM Pass Raw IR (from T3)
- T5 Asm. Gen. Produce RV64 assembly Optimized IR (from T4)

processes, which enables systematic analysis of scaffolding effects and execution complexity under production-style workflows.

A key design of RAMP is the resurrection mechanism, which decomposes cascading workflow failures into recoverable diagnostic stages through controlled intermediate-state restoration. This mechanism substantially improves runtime observability and enables fine-grained analysis of downstream execution behavior that would otherwise be hidden by early-stage failures. In addition, RAMP introduces utility-oriented multi-dimensional metrics that jointly assess outcome quality, execution efficiency, runtime cost, workflow robustness, recovery capability, and deployment-oriented operational behavior.

In summary, the paper makes the following contributions:

- (1) We propose Ramp as a production-based assessment infrastructure, not merely another benchmark, that moves beyond synthetic isolated tasks toward persistent, runtimeobservable evaluation of agentic models under realistic engineering environments.
- (2) We construct realistic compiler construction workflows withstrictserialdependencies, verifiable intermediate states that systematically quantify the impact of scaffolding and execution complexity in agentic software engineering tasks. Further, we introduce utility-oriented metrics that characterize not only task outcomes, but also runtime efficiency, failure patterns, and deployment-relevant operational cost.
- (3) We provideempiricalevidenceacross 15 model tests, demonstrating that realistic assessment reveals failure patterns invisible to static benchmarks and that resurrection-based recovery substantially improves downstream diagnostic observability.

### 2 Background and Motivation

In this section, we first introduce YatCC, detailing YatCC Tasks and YatCC Platform capabilities, and discuss how they reflect a production-level serving environment. We then analyze the motivation and challenges of evaluating agentic systems within such workflows, formalizing the key requirements that a realistic assessment framework should satisfy.

### 2.1 YatCC Tasks and Platform

Here, we briefly introduce the practical compiler-construction tasks and the YatCC support platform that together form the foundation of realistic model assessment.

##### YatCC Tasks & Platform

##### Compiler-Constr. Tasks

- §2.1.1

- §2.1.2

Real-World Sequentially Dependent Stages (T0-T5)

##### Unified Env. Management

Isolated Containers Pre-installed Environment

#### Figure 1: YatCC, with Compiler-Construction Tasks and Unified Environment Management.

- 2.1.1 Compiler-ConstructionTasks. Thecompiler-constructiontasks built on YatCC [9] is based on LLVM [13], which has been extensively used as real-world compiler coursework and engineering exercises. The workload consists of six sequentially dependent compilation stages or tasks, where each task produces an intermediate artifact that becomes the input dependency for the next stage. Specifically, Task0 verifies agent capability and the build environment, Task1 produces a token stream, Task2 generates an AST/ASG, Task3 generates LLVM IR, Task4 performs IR optimization, and Task5 generates RV64 assembly. Each task is graded using dedicated test suites with comprehensive correctness checks, with scores ranging from 0 to 100.

Compiler development naturally embodies realistic systemsdevelopment jobs, involving large codebases, multi-stage execution pipelines, strict correctness constraints, and tightly coupled dependencies across lexical analysis, parsing, optimization, code generation, linking, and runtime validation. Each stage produces intermediate artifacts that must remain semantically and functionally consistent throughout the pipeline. Completing these tasks requires sustained interaction with external compilers, build systems, testing frameworks, and debugging tools, while continuously maintaining evolving project states and resolving execution failures. As a result, these tasks capture core characteristics of real-world software engineering, including long-horizon execution, iterative refinement, dependency management, and operational coordination in complex runtime environments.

- 2.1.2 Unified Environment Management. To reliably support these long-horizon compilation tasks and ensure strict dependency isolation, YatCC Platform provides a robust execution infrastructure with unified environment management capabilities. With isolated containers and pre-installed environment, YatCC Platform is a onestop execution infrastructure that supports application development, deployment, execution, and management across heterogeneous execution environments through unified runtime abstractions. It integrates workflow runtime, containerized execution, external toolchains, and resource-managed infrastructure into a unified operational environment.

Applications running on YatCC execute within complete software stacks involving dependency installation, compilation, testing, runtime monitoring, and iterative debugging, reflecting practical deployment and engineering workflows. Such environments naturally require continuous coordination across execution pipelines, system services, and runtime states over extended execution horizons. By leveraging unified containerized sandboxes and pre-installed toolchains, YatCC Platform ensures that the full software stack and runtime configurations remain deterministic and isolated across repeatedexecutioncycles,preventing environmental cross-contamination while tracking persistent state evolution.

### 2.2 Motivation and Challenges

The production-oriented characteristics of YatCC and YatCC Tasks expose key limitations in existing agent evaluation methodologies.

Overall, current evaluation approaches suffer from three major limitations. First, they lack support for serial dependency and persistent execution state, preventing assessment of cross-task interactions, cumulative error propagation, and long-horizon workflow completion. Second, they are largely outcome-centric, reporting only final scores without visibility into process efficiency, execution dynamics, failure behavior, or recovery capability. Third, evaluation infrastructures are often tightly coupled to specific benchmarks or frameworks, making systematic cross-model and cross-agent comparison difficult.

YatCC Tasks address these limitations by providing a unified, production-oriented evaluation setting. The workloads preserve execution continuity across sequential compilation stages, maintain evolving intermediate artifacts throughout the pipeline, and require sustained interaction with realistic build, testing, and debugging environments. These characteristics enable assessment not only of task completion, but also of execution robustness, dependency management, runtime coordination, and long-horizon operational behavior.

Table 2 positions a desired assessment infrastructure against existing benchmarks across six key dimensions. Contemporary benchmarks demonstrate robust capabilities across various axes: SWE-bench, ProgramBench, and SkillsBench excel in real-world execution-grounded tasks with strict zero-tolerance scoring, while frameworks like WebArena, OSWorld, and AgentBench naturally embed chain dependencies through complex multi-step interactions (though some, like OSWorld, permit partial rewards rather than binary evaluation). Furthermore, RE-Bench pioneers the integration of deep process metrics (such as time and monetary cost) into the evaluation of long-horizon research tasks.

However, a critical gap remains in evaluating production-level engineering workflows. Existing benchmarks primarily evaluate end-to-end task success without explicit verification of intermediate states. In compiler-construction and similar industrial pipelines, the Artifact—the intermediate product such as a parsed AST or compiled IR passed between dependent stages—is fundamentally crucial. The desired framework must simultaneously integrate real-world workflows, execution-grounded testing, strict serial dependencies, zero-tolerance scoring, continuous intermediate artifact verification, and detailed process-level metrics. Together, these properties enable a more granular evaluation of practical agent capabilities

- Table 2: Benchmark comparison. ✓ = supported, × = not supported, partial = partially supported. Real-world = tasks derived natively from practical production systems; Exec. = execution-grounded scoring; Chain Dep. = tasks or actions form a sequential dependency chain; Strict Score = zero-tolerance evaluation (e.g., non-perfect output yields zero); Artifact = explicitly evaluates and preserves intermediate generated artifacts (e.g., AST, IR) across workflow stages; Process = outputs detailed process metrics (e.g., cost, time) as first-class criteria; Eval Points = the total number of distinct tasks or test instances evaluated.

Benchmark Real-world Exec. Chain Dep. Strict Score Artifact Process Eval Points

SWE-bench [12] ✓ ✓ × ✓ × × 2.3K AgentBench [16] partial ✓ ✓ partial × × GAIA [18] partial × ✓ ✓ × × 466 OSWorld [28] ✓ ✓ ✓ partial × partial 369 WebArena [31] ✓ ✓ ✓ ✓ × partial 812 Terminal-Bench [17] ✓ ✓ ✓ ✓ × × 89 SkillsBench [15] ✓ ✓ × ✓ × × 84 RE-Bench [26] ✓ ✓ ✓ × partial ✓ 7 ProgramBench [29] ✓ ✓ × ✓ × × 200

Desired ✓ ✓ ✓ ✓ ✓ ✓ ≥ 100

that are difficult to capture through conventional isolated-task or purely outcome-centric benchmarks. To effectively assess agentic models in production-oriented environments, several key requirements that must be met.

A unified runtime infrastructure for heterogeneous model assessment. Production environments typically involve diverse models, APIs, execution frameworks, workflow engines, and deployment backends. A practical assessment platform should therefore provide a unified infrastructure that enables reproducible and scalable evaluation across heterogeneous models and runtime systems. Such an infrastructure should support standardized service interfaces, workflow orchestration, environment provisioning, dependency isolation, and runtime management, allowing different models and execution frameworks to be evaluated under consistent operational conditions. The infrastructure should also enable closedloop assessment pipelines for automated testing, execution tracing, error diagnosis, result aggregation, and replayable experimentation, which helps better capture the evolving behavior of agentic systems under complex and changing production workloads.

Realistic workloads and diverse assessment perspectives. Conventional benchmarks often focus on isolated tasks with short execution horizons, which fail to capture the characteristics of real production systems. Production-grade assessment instead requires realistic workloads involving long-running workflows, external tool invocation, iterative execution feedback, environment interaction, failure recovery, and multi-stage task coordination. Moreover, assessment should support multiple evaluation perspectives, including functional correctness, execution efficiency, robustness under runtime perturbations, adaptability to dynamic environments, and operational scalability under concurrent workloads.

End-to-end observability and measurable runtime metrics. Assessing agentic systems in production settings requires comprehensive runtime visibility beyond task success rates alone. The assessment infrastructure should provide observable metrics spanning workflow execution, tool interaction, runtime behavior, resource consumption, latency characteristics, failure propagation, and execution stability. Such observability is essential for understanding not only whether a task succeeds, but also how the system

behaves under realistic operational conditions, enabling deeper analysis of reliability, orchestration capability, and production readiness.

Together, these requirements highlight that assessing agentic models in production systems fundamentally a systems-level problem rather than merely a benchmark-level evaluation task. Effective assessment therefore necessitates integrated runtime infrastructure, realistic workload modeling, comprehensive observability, and scalable orchestration mechanisms.

### 3 Design and Implementation

To assess agentic models in production-level environments, we propose Ramp, which is a unified assessment infrastructure that cleanly decouples workload specification from execution, enabling scalable tests across heterogeneous model providers and agent frameworks through a standardized API layer. Unlike prior benchmark approaches, it treats both task outcomes and process dynamics as first-class analytical objects, allowing detailed, and finegrained analysis of agentic model behavior. The system is structured around three core components—framework, workloads, and metrics—providing a flexible yet systematic foundation for end-to-end assessment.

### 3.1 Assessment Framework

###### Assessment Framework

Env. Support

§3.1.2 Agentic Runtime §3.1.1

###### AI Hub

###### Agent Backends

[Figure 2]

Unified API Gateway

14+ LLMs Qwen, GPT, Kimi, Deepseek, etc.

Identical Task Context

Openhands SDK

Exec. & Obse. §3.1.3

#### Figure 2: The framework of Ramp that enables unified and extensible integration of heterogeneous agentic models and backends.

[Figure 3]

- Figure 3: Long-horizon assessment workloads in the integrated pipeline of Ramp (§3.2). The workflow demonstrates Serial Evolution where tasks pass cumulative context. When Task 2 fails (red), Mode 2 (wavy orange line) propagates the error, leading to cascading downstream failures. Mode 1 (solid blue line) transparently injects a golden artifact via the orchestrator, isolating the failure and allowing evaluation to continue.

The assessment framework is organized into three layers of runtime, environment and execution, which together provide a unified, reproducible, and extensible foundation for assessing LLM agents in production-oriented software engineering workflows.

- 3.1.1 Agentic Runtime Layer. This layer provides a unified abstraction over heterogeneous agent frameworks and model providers. It supports multiple agent backends, each representing a distinct interaction paradigm. All backends consume a shared task context format and generate standardized execution outputs, thus enabling direct cross-framework comparisons.

To further normalize model access, Ramp integrates a unified API gateway, i.e., AI Hub 1 that exposes OpenAI-compatible interfaces [19] for both proprietary and open-weight models. This design eliminates backend-specific integration overhead and ensures that observed assessment differences primarily reflect model and scaffold capabilities rather than infrastructure inconsistencies.

- 3.1.2 Environment Support Layer. This layer ensures execution isolation and reproducibility. Each assessment run is executed within an independent workspace containing a complete copy of the target repository with all required dependencies pre-installed, including ANTLR, LLVM 14, and pybind11. In addition, every run launches a fresh containerized environment that is destroyed after execution, preventing cross-run contamination.

The build and execution environment is configured using a fixed CMake-based toolchain, and all scoring scripts are deterministic. This setup guarantees that assessment results remain stable and comparable across repeated runs and heterogeneous agent backends.

- 3.1.3 Execution and Observation Layer. The execution layer manages workload orchestration, execution monitoring, and result analysis. It is responsible for test scheduling, dependency resolution, resurrection triggering, and assessment pipeline coordination. The 1AI Hub: https://aihub.arcsysu.cn

core component of this layer, referred to as the orchestrator, maintains the state of serial task chains, tracks per-task pass/fail outcomes, and automatically triggers resurrection when task completion falls below the predefined threshold (e.g., 60%).

Beyond execution management, this layer also collects and aggregates assessment artifacts, including scores, logs, trajectories, and process metrics. It computes utility-oriented metrics, maintains dynamic leaderboards, and preserves all raw assessment data for reproducibility and fine-grained behavioral analysis.

### 3.2 Long-horizon Workloads

Ramp assesses LLM agents using long-horizon, production-oriented software engineering workloads grounded in compiler construction, as illustrated in Figure 3. Unlike isolated coding benchmarks, Ramp organizes tasks into a serial dependency chain that preserves execution context, intermediate artifacts, and runtime state across the entire workflow. To further improve diagnostic resolution in long execution chains, Ramp introduces a resurrection protocol that enables continued assessment after intermediate failures.

3.2.1 Serial Evolution. Ramp currently provides a representative compiler-constructionworkloadderived from the individual tasks [20]. The workload consists of six sequential tasks spanning the compiler development pipeline. Each task provides partially implemented scaffold code, requiring agents to complete missing functionality within an existing codebase rather than generating standalone solutions from scratch.

A key characteristic of the workload is persistent execution continuity. The full repository state, execution context, and runtime environment are preserved across all tasks, simulating realistic agentic software development workflows in which agents incrementally evolve a single system over time. This design avoids artificial context resets between tasks and enables evaluation of continuous development capabilities, long-horizon dependency handling, and cumulative error propagation.

- 3.2.2 Resurrection Protocol. Serial workloads introduce a fundamental assessment challenge: failure at Task 𝑘 may invalidate all downstream tasks, making it impossible to determine whether subsequent failures are caused by limited downstream capability or simply inherited broken states. To address this issue, Ramp introduces the resurrection protocol.

When a task score falls below the passing threshold (e.g., 60%), resurrection is automatically triggered by the orchestrator. The failed intermediate artifact is replaced with the corresponding golden artifact generated by the reference implementation, and the build environment is reconfigured through config.cmake (TASK_k_REVIVE=ON). The agent then continues execution from Task 𝑘 + 1 using the corrected system state.

Importantly, resurrection is designed as an assessment protocol rather than a scoring adjustment mechanism. Its purpose is to separate two otherwise entangled failure modes: (1) the inability to correctly complete the current task, and (2) the capability to solve downstream tasks given valid prerequisites. This decomposition enables node-level capability analysis even when the overall pipeline is broken, substantially improving diagnostic granularity for long-horizon assessment.

To avoid behavioral bias, agents are not informed when resurrection occurs. All interventions are handled externally by the task orchestrator.

- 3.2.3 Execution Modes. Ramp supports two execution modes for long-horizon workloads.

#### Mode 1: Serial Pipeline with Resurrection (Default). Agents

execute Tasks 0–5 sequentially. Whenever a task fails, resurrection is triggered and execution continues using the corrected intermediate artifact. This setting maximizes diagnostic coverage and produces the primary leaderboard containing both outcome and process metrics.

#### Mode 2: Serial Pipeline without Resurrection (Cascade

failure). Agents also execute Tasks 0–5 sequentially, but no resurrection is applied. Failed intermediate artifacts are directly propagated to downstream tasks, often causing cascading failures. This setting measures zero-shot pipeline depth, i.e., how far an agent can progress in a fully autonomous end-to-end workflow without external correction.

Comparing these two settings allows Ramp to quantify the extent to which resurrection recovers diagnostic signals that would otherwise be obscured by cascading failures in long-horizon software engineering workflows.

### 3.3 Multi-dimensional Metrics

Ramp adopts a multi-dimensional measuring suite that moves beyond single accuracy-based scores toward utility-oriented assessment. In long-horizon software engineering workflows, successful assessment requires measuring not only whether an agent completes tasks, but also how efficiently it operates, how resources are consumed, and how failures emerge during execution. To this end, Ramp combines outcome metrics, process metrics, failure analysis, and composite utility measurements into the unified assessment framework.

- 3.3.1 Outcome and Process Metrics. Outcome metrics. Ramp defines Mean Reward (MR) as the primary outcome metric. MR computesaweightedaverageoftaskscores while incorporating resurrectionaware bonuses:

MR = ∑︁𝑠𝑖𝑤𝑖𝑏𝑖/∑︁𝑤𝑖𝑏𝑖 ,

where𝑠𝑖 denotesthetaskscore,𝑤𝑖 = [0.05, 0.20, 0.20, 0.15, 0.30, 0.10]

represents task importance weights, and 𝑏𝑖 is a resurrection factor. Tasks completed without resurrection receive a bonus (𝑏𝑖 = 1.2), while resurrected tasks use the default weight (𝑏𝑖 = 1.0). This formulation rewards both correctness and autonomous pipeline continuity.

Process metrics. Beyond final task outcomes, Ramp records detailed execution dynamics throughout the assessment process. Collected metrics include token consumption, interaction turns, command executions, retry counts, and elapsed wall-clock time at both per-task and end-to-end levels. These measurements enable fine-grained analysis of agent efficiency, interaction behavior, and resource utilization.

- 3.3.2 Failure Taxonomy. Outcome scores alone provide limited insight into why agents fail in long-horizon workflows. To improve diagnostic interpretability, Ramp introduces an observable failure taxonomy grounded in and extended from the four empirically validated architectural fault dimensions proposed by Shah et al.[21]. We add a dedicated planning dimension to comprehensively cover all execution failure modes observed in long-horizon tasks. All categories are defined based on trace-observable behaviors, avoiding ambiguous cognitive labels:

- • Reasoning Failure: Failures manifesting as repetitive debugging loops, infinite search cycles, or stalled iterations with no meaningful progress, arising from flawed control orchestration and execution loop logic.
- • Planning Failure: Failures where the agent explicitly decides to bypass one or more required task steps, resulting from incorrect task decomposition or flawed strategic decision-making.
- • Context Failure: Failures occurring when accumulated prompts, execution history, or intermediate state exceed the model’s context window capacity, preventing further reasoning or action.
- • Tooling & Integration Failure: Failures caused by invalid tool invocations, external service access errors, resource interaction problems, or network connectivity issues.
- • Infrastructure Failure: Failures stemming from bugs or design flaws in the agent framework itself, including dependency conflicts, environment misconfigurations, and broken exception handling.

- 3.3.3 Overall Utility. The preceding metrics independently characterize task completion, execution efficiency, and failure behavior. However, practical deployment scenarios often require assessing overall engineering productivity under constrained resources. To capture this tradeoff, Ramp introduces the Agent Efficiency Index (AEI), a composite utility metric that jointly measures task effectiveness and resource efficiency:

#### Table 3: Categorization and Abbreviations of the Evaluated Models. Model types: FF (Frontier Flagship), AM (Advanced Mainstream), SL (Standard & Lightweight).

Model Type Abbr. Model Type Abbr. Model Type Abbr.

claude-opus-4-7 FF Opus-4.7 kimi-k2.6 FF Kimi-2.6 kimi-k2.5 AM Kimi-2.5 gpt-5.5 FF GPT-5.5 deepseek-v4-flash AM DS-v4-Flash glm-4.6 SL GLM-4.6 deepseek-v4-pro FF DS-v4-Pro deepseek-reasoner AM DS-Reasoner deepseek-chat SL DS-Chat qwen3.6-max-preview FF Qwen-3.6-Max qwen3.5-plus AM Qwen-3.5-Plus qwen3-coder-flash SL Qwen3-Coder glm-5.1 FF GLM-5.1 glm-4.7 AM GLM-4.7 minimax-m2.5 SL MiniMax-2.5

∑︁

1 |D|

AEI =

𝑠𝑑, D = {stage, reward, time, cost, tokens}.

𝑑∈D

where 𝑠𝑑 ∈ [0, 100] is the normalized score for dimension 𝑑 ∈ D. Each dimension is mapped using the maximum value observed across all evaluated models. Let 𝑆 denote the furthest pipeline stage reached, 𝑅 the mean reward, and 𝑇, 𝐶, and 𝐾 the end-to-end wallclock time, LLM API cost (USD), and total token usage, respectively. Let 𝑆max, 𝑅max,𝑇max,𝐶max, and 𝐾max denote these observed maxima. We define

𝑆 𝑆max × 100, 𝑠reward =

𝑅

𝑅max × 100, 𝑠time =

𝑠stage =

𝑇max −𝑇 𝑇max × 100, 𝑠cost =

𝐶max −𝐶

𝐶max × 100, 𝑠tokens =

𝐾max − 𝐾 𝐾max × 100.

Because each raw metric is non-negative and at most its observed maximum, every 𝑠𝑑 lies in [0, 100] by construction. And, stage captures pipeline depth (the furthest stage completed along the serial workflow), reward captures Mean Reward (MR), and time, cost, and tokens capture elapsed time, monetary API cost, and token consumption, respectively.

Higher AEI alues indicate more balanced utility across outcome quality and resource utilization. Because each dimension is equally weighted, AEI annot be maximized through strong task scores alone while ignoring time, cost, or context pressure.

### 4 Results and Analysis

In this section, we first describe the experimental methodology, including the tested models, execution settings, and assessment metrics. We then present and analyze the experimental results across multiple dimensions.

### 4.1 Experimental Methodology

4.1.1 Models and Agent Backends. We evaluate RAMP using 15 representative models spanning both proprietary and open-weight ones, as listed in Table 3. The assessed models cover multiple capability tiers, including frontier flagship models, advanced mainstream models, and lightweight efficient models, enabling comprehensive analysis with varying performance–cost tradeoffs.

All assessments are conducted using a unified agent backend 2

built on top of the OpenHands SDK [25], enabling agents to iteratively invoke tools, inspect runtime outputs, modify source codes, and coordinate execution workflows within a persistent runtime environment. To ensure fair comparison across models, all LLM services are accessed through the AIHub of Ramp (§3.1.1), which exposes standardized OpenAI-compatible interfaces for heterogeneous model providers. For each model test, agents execute the complete serial workload under identical runtime conditions, using the same execution environment, toolchain configuration, prompts, and orchestration policies.

4.1.2 Configurations and Metrics. All experiments are performed under a standardized assessment pipeline. For each run, the agent is initialized with a fresh workspace containing the complete repository, pre-installed dependencies, build toolchains, and grading scripts. Agents interact with the environment through iterative command execution, file modification, compilation, testing, and debugging operations until task completion or termination.

During execution, Ramp continuously records both outcome and process traces, including task scores, interaction histories, command invocations, execution logs, token consumption, runtime latency, and resource usage statistics. All raw artifacts are preserved to support reproducibility and detailed post-hoc behavioral analysis.

All experiments are executed on a standardized hardware and software platform to ensure reproducibility and fair comparison across evaluated agents. The evaluation server is configured with with following settings: ○1 CPU: AMD EPYC 7742 64-Core Processor; ○2 Memory: 256 GB RAM; ○3 Operating System: Debian GNU/Linux 11 (bullseye); ○4 Libraries and Toolchains: LLVM 14, ANTLR, pybind11, CMake, Docker, and related runtime dependencies. And, to bound evaluation cost while preserving sufficient exploration capability, we enforce a strict maximum of 500 dialogue turns per agent across the full workload pipeline.

We further adopt a multi-dimensional metric suite that measures not only task correctness, but also execution efficiency, failure behavior, and overall engineering utility. As defined in §3.3, the evaluation consists of four complementary dimensions:

- • Performance: measured using capability depth, pipeline completion, and weighted leaderboard scores;
- • Cost: including wall-clock execution time, token consumption, command counts, and estimated monetary cost;

2Results of extra backends are to be added in analysis.

#### Table 4: Ramp leaderboard. MR = Mean Reward. Bold values indicate task scores ≥ 60. The Baseline row corresponds to the unmodified code framework without agent intervention.

# Model T0 T1 T2 T3 T4 T5 MR Cost$

- 1 Opus-4.7 100 100 100 100.0 68.4 100 93.39 126.24
- 2 DS-v4-Pro 100 100 100 83.6 38.6 100 85.34 8.68
- 3 DS-v4-Flash 100 100 31.6 37.0 39.0 100 66.48 3.43
- 4 GPT-5.5 100 100 100 100 48.2 0.0 65.91 8.77
- 5 Qwen-3.6-Max 100 100 100 65.8 35.0 0.0 56.72 67.28
- 6 Kimi-2.5 100 100 14.2 24.7 39.7 9.4 40.24 1.75
- 7 Kimi-2.6 100 95.9 31.4 1.4 37.5 0.0 33.94 3.99
- 8 GLM-4.6 100 67.4 38.5 1.4 39.6 0.0 30.88 —
- 9 DS-Chat 100 72.9 30.9 0.0 35.7 0.0 29.74 0.07
- 10 Qwen-3.5-Plus 100 100 0.0 0.0 38.0 0.0 29.63 14.37
- 11 GLM-4.7 100 94.8 0.0 1.4 36.2 0.0 29.50 0.80
- 12 GLM-5.1 100 0.0 97.7 0.0 36.6 0.0 27.26 2.03
- 13 Qwen3-Coder 100 56.0 20.6 1.4 35.1 0.0 25.97 0.05
- 14 MiniMax-2.5 100 64.2 0.0 2.7 36.4 0.0 25.09 0.17
- 15 DS-Reasoner 100 0.0 0.0 0.0 0.0 0.0 7.69 0.23

- Baseline 100 17.6 20.6 1.4 35.1 0.0 23.38 —

- • Failure: characterizedthroughobservablefailure categories such as execution failure, process collapse, cost overrun, task skipping, and context exhaustion;
- • Overall Utility: measured using the Agent Efficiency Index (AEI), a composite metric that jointly captures task utility and resource efficiency(§ 3.3.3).

Together, these metrics provide a holistic characterization of agent behavior under realistic long-horizon software engineering workloads.

### 4.2 Capability Boundaries and Leaderboard Scoring

To establish a rigorous baseline to measure agent capabilities, we conduct all 15 model test runs under the most demanding assessment conditions, i.e., Mode 2 (serial pipeline without resurrection, §3.2.3). This long-horizon regime eliminates all external intervention and enforces production-grade functional correctness, strictly respecting the dependency structure of real-world development—agents cannot proceed to a task until all preceding stages are fully and correctly completed.

Results reported in Table 4 reveal a stark limitation: even the state-of-the-art models, including Opus-4.7 and GPT-5.5, fail to complete the entire six-task pipeline. The vast majority of models suffer catastrophic early-stage malfunctions, and even the topperforming model Opus-4.7 stalls at the IR Generation stage (T3), unable to fully complete IR Optimization (T4). Only a small subset of proprietary models like DS-v4-Pro and Qwen-3.6-Max reach T3 with none progressing further. This demonstrates a clear capability ceiling for current models on complex multi-step reasoning pipelines.

Table 4 and Figure 4 together present the detailed Ramp leaderboard, moving beyond binary pass or fail boundaries to continuous per-task scoring. We summarize the key findings below.

Finding 1: Current agentic models lack sustained multistage engineering capabilities for end-to-end autonomous

development of complex engineering systems. None of the 15 assessed models achieve perfect scores across all six stages, confirming that flawless compiler chain implementation exceeds current zero-shot LLM agent capabilities even with structured scaffolding. Opus-4.7 leads with an MR of 93.39, achieving perfect completion on five out of six tasks but falling short at IR optimization (T4, 68.4). Deepseek-v4-pro ranks second (MR 85.34) with four perfect tasks, while GPT-5.5 also achieves four perfect tasks but produces non-functional output at backend code generation (T5, 0.0).

Task completion rates across stages are 100% (T0), 46.7% (T1), 26.7% (T2), 13.3% (T3), 0% (T4), and 20% (T5). Notably, T5 exhibits a higher task completion rate than T4. This means task difficulty itself does not follow a monotonic increasing pattern, so the observed performance decline cannot be simply attributed to harder tasks. Instead, performance declines reflect the cumulative impact of earlier task modifications to the execution environment and the propagation of subtle errors across stages. The overall average MR of 42.7 is only 17.6 points above the non-agent Baseline, highlighting that current agents provide limited resilience to state accumulation and error propagation in multi-stage engineering workflows.

Finding 2: Agentic models show systematic capability decline in long-horizon execution. All models achieve perfect

- scores on T0 (environment setup), and 8 models achieve perfect
- scores on T1 (lexer). However, performance declines sharply at T2 (parser) and T3 (IR generation) as the execution environment becomes increasingly modified and interdependent. Models with perfect T1 scores experience an average 41.3-point score lowering by T3, with one model (Qwen-3.5-Plus) achieving scores of 0 points. Quantitatively, the average stage-wise score decreases sequentially from 100 (T0) to 76.75 (T1), 44.85 (T2), 27.95 (T3), 37.60 (T4), and 20.63 (T5), confirming a consistent downward trend despite nonmonotonic fluctuations.

[Figure 4]

- Figure 4: Per-task score distribution of LLM agents. Models (y-axis) are ranked by overall mean reward. The x-axis is divided into six equal segments corresponding to Task0 –Task5: environment setup, lexer, parser, IR generation, IR optimization, and backend code generation. Within each segment, the colored bar length indicates the normalized task score (out of 100).

Three distinct performance patterns emerge across the pipeline. First, gradual degradation is observed in several models like Kimi2.6 and DS-Chat, where scores decrease incrementally across consecutive stages. Second, error amplification occurs when minor imperfections in earlier stages lead to disproportionately lower scores in subsequent stages. This is observed in DS-v4-Pro, which achieves 83.6 on T3 but only 38.6 on T4. Third, discontinuous performance manifests as unexpected score variations in later stages that are uncorrelated with earlier performance. Most notably, GPT-5.5 achieves perfect scores on T0 –T3 but produces completely nonfunctional backend code, while DS-v4-Flash exhibits limited performance at T2 and T3 but delivers perfect T5 output. This pattern demonstrates that current LLMs exhibit inconsistent performance across stages that share dependencies, with distinct capability profiles: for example, Opus-4.7 excels in IR optimization while DS-v4-Pro shows superior backend code generation ability.

### 4.3 Process Efficiency

We next move beyond performance analysis to examine the efficiency implications, including execution time, token consumption, and monetary cost.

The assessment reveals extreme variability in cost efficiency, with total inference costs ranging from $0.05 (Qwen3-Coder) to $126.24 (Opus-4.7)—a 2,525x difference. API costs are calculated based on provider-specific pricing for input and output tokens separately. Opus-4.7 delivers the highest mean reward (MR=93.39) but at a prohibitive cost, representing a 14.5x cost premium over DS-v4-Pro (MR=85.34, $8.68) for only a 9.4% improvement on overall performance. DS-v4-Pro achieves the best balance of performanceandcost,delivering29.5%higher MR thanGPT-5.5(MR=65.91, $8.77) at nearly identical cost. Most strikingly, Qwen-3.6-Max ranks

fifth in performance (MR=56.72) but incurs 7.75x higher cost than DS-v4-Pro and 19.6x higher cost than DS-v4-Flash (MR=66.48, $3.43). At the lower end, DS-Chat ($0.07) and Qwen3-Coder ($0.05) outperform the baseline by 27.2% and 11.1% respectively at negligible cost, making them attractive options for rapid prototyping.

Figure 5 plots the relationship between computational investment (elapsed time and LLM API cost) against outcome quality.

Finding 3: Agentic models exhibit substantial variability in performance relative to cost and execution time. On the log-scaled axes, ordinary least squares (OLS), a widely used linear regression technique,indicates a moderate positive association between computational investment and performance: API cost against mean reward yields 𝑅2 ≈ 0.52, while elapsed time shows a somewhat weaker but still visible trend with 𝑅2 ≈ 0.40. As the goodnessof-fit metric of OLS, 𝑅2 measures the proportion of performance variance explained by resource overhead, with values closer to 1 indicating a stronger linear correlation. These results suggest that greater resource consumption is associated with higher reward on average, but the relationship is far from deterministic and leaves substantial variance unexplained. In particular, several high-cost or long-running systems still achieve only moderate performance, whereas some more economical configurations remain highly competitive. For example, Qwen-3.6-Max is among the most expensive models yet attains only moderate reward, while DS-v4-Flash reaches top-tier performance at substantially lower cost. Overall, computational budget and execution time provide useful but incomplete signals of agent quality: they correlate with performance yet cannot reliably predict it.

###### Time vs Performance

###### Cost vs Performance

[Figure 5]

[Figure 6]

| | | | | | | | | | | | | |DS|[Figure 7]<br><br>-v<br><br>O|4<br><br>pu|-Pro<br><br>s-4.7|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |D|S-v4|-F|[Figure 8]<br><br>[Figure 9]<br><br>la|sh| | | | | |
| | | | | | |Qwen-3.6|Kimi-2.<br><br>-Max|[Figure 10]<br><br>6|[Figure 11]<br><br>|im|i-2<br><br>G|.5<br><br>PT|-5.|5| | |
| | | | | | | | | | | | | | | | | |
| |[Figure 12]<br><br>w|en|3-C|o|de|DS-Chat<br><br>GLM-4.7 r<br><br>MiniM|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>G<br><br>LM-4.6 Q<br><br>ax-2.5|[Figure 17]<br><br>LM-5<br><br>wen-|.<br><br>3|[Figure 18]<br><br>[Figure 19]<br><br>1<br><br>.5-|Plu|s| | | | |
| | | | | | | |DS-Re|ason|e|r|[Figure 20]| | | | | |

| | | | | | | | | | | | | | | | | | | | |[Figure 21]| | | | | | | | | | | |p|u| |s|-4.7|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |D|S|-|v|4|-|F|lash| | |[Figure 22]| | | | | |[Figure 23]<br><br>S-v|4<br><br>G|-Pr<br><br>PT|o<br><br>-5|.5| | | | | | |
| | | | | | | | | | | | | | | |K| |imi-2|[Figure 24]<br><br>.6<br><br>K|im<br><br>Q|i<br><br>w|-2<br><br>e| |.5<br><br>n|-|3|.6-|M|ax| | | | |[Figure 25]| | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| |[Figure 26]<br><br>D w<br><br>|S<br><br>e|-C inn|[Figure 27]<br><br>i3|h M|a<br><br>-|Ca|t<br><br>G<br><br>x-2.ode|[Figure 28]<br><br>LM<br><br>5r|-4|.<br><br>G|7<br><br>L|M| |[Figure 29]<br><br>-|5|.1|[Figure 30]<br><br>[Figure 31]<br><br>GL|M|-4|[Figure 32]<br><br>.|6| | | |Q|[Figure 33]<br><br>w|en|-3|.|5|-|P|l|u|s|
| | |D|S|-R| |e| |ason|[Figure 34]<br><br>er| | | | | | | | | | | | | | | | | | | | | | | | | | | |

80

80

60

60

MR

MR

40

40

###### Q

Q

20

20

0

103 104

10 1 100 101 102

Elapsed Time (s)

Cost ($)

Regression line Mean (horizontal & vertical)

Bubble & logo size tokens

| |
|---|

- Figure 5: Trade-off of cost and performance: elapsed time (left) and API cost (right) vs. mean reward across 15 models (𝑛=15).

Dashed lines are OLS fits (linear fitted lines) of mean reward on log10(𝑥) (log-scaled axes). Moderate positive association: 𝑅2 ≈ 0.40 (time) and 𝑅2 ≈ 0.52 (cost). Bubble area is proportional to total LLM tokens.

### 4.4 Failure Analysis

Existing benchmarks rarely systematically categorize runtime failure modes in long-horizon serial workflows. By contrast, RAMP classifies agent failures into five mutually exclusive trace-driven categories of Reasoning Failure, Planning Failure, Context Failure, Tooling & Integration Failure, and Infrastructure Failure. Each category follows explicit observable criteria derived from execution logs and interaction traces to avoid subjective interpretation, and only the primary root cause is labeled when multiple failure symptoms co-occur. Figure 6 summarizes the failure taxonomy across all evaluated models.

Statistically, Context Failure emerges as the most prevalent hardstop failure across the 15 evaluated models (9 out of 15, 60.0%), followed by Planning Failure (2 out of 15, 13.3%) and Reasoning Failure (1 out of 15, 6.7%). Further stage-distributed analysis reveals clear failure concentration patterns: Context Failure predominantly occurs at T2 –T3, where accumulated code artifacts, dialogue history, and task instructions continuously consume context budget. Planning Failure mainly appears at T2 –T4, while Tooling & Integration Failure is concentrated in T2 (parser) and T3 (IR generation) stages that require frequent tool invocation and environment interaction.

Finding 4: Long-horizon execution of current agents is primarily bottlenecked by context exhaustion and premature strategic abandonment. Context Failure stems from limited context window capacity and ineffective long-range context compression, rather than fundamental reasoning deficiencies. It can be precisely identified via API logs and trajectory truncation signals, acting as a critical system-level constraint for sustained agent workflow execution.

Planning Failure can be observed in 8 of the 15 evaluated models (53.3%) and act as the primary termination cause in 2 models (13.3%),

making it one of the most common observed failure behaviors. Unlike other failures, Planning Failure originates from explicit agent strategic decisions, where the agent acknowledges unresolved task requirements but voluntarily skips difficult stages to conserve budget or reduce iteration pressure. This behavior represents premature task abandonment rather than adaptive scheduling, as the agent redefines task boundaries instead of completing the predefined serial workflow.

Further behavioral abstraction derived from Planning Failure reveals two representative agent operational strategies: a skip-first strategy and a persistent search strategy. Several models including DS-Reasoner, DS-v4-Flash, and Qwen-3.5-Plus explicitly choose to bypass challenging early stages and advance to subsequent tasks, justifying such decisions with time limits, iteration quotas, or perceived task difficulty. This strategy reduces resource consumption but sacrifices end-to-end pipeline completeness. In contrast, other models adopt a persistent search strategy, repeatedly executing edit-compile-debug loops even when progress stalls. Although this persistence may occasionally resolve ambiguous implementation problems, it continuously expands the dialogue context, accelerates token consumption, and eventually leads to context exhaustion or budget overrun.

The divergence between the two strategies demonstrates that agent process quality depends not only on final task accuracy but also on adaptive effort allocation and long-horizon workflow scheduling under resource constraints. Moreover, such failure and behavioral patterns can only be exposed through Ramp’s serial dependency pipeline, which are largely invisible in conventional independent static benchmarks.

| | | | | | |
|---|---|---|---|---|---|
| | | | |PF| |
| | | | | | |
| | | | |IF| |
| | | | | | |
| | | |PF|PF| |
| | | | | | |
| | | |CF| | |
| | | | | | |
| | |PF|PF|PF| |
| | | | | | |
| | |PF|CF| | |
| | | | | | |
| | |RF|CF| | |
| | | | | | |
| | |PF|PF|PF|CF|
| | | | | | |
| | |PF|PF|PF|CF|
| | | | | | |
| | |TF|PF|PF|PF|
| | | | | | |
| |TF|PF|CF| | |
| | | | | | |
| |TF|CF| | | |
| | | | | | |
| |TF|CF| | | |
| | | | | | |
| |RF|RF|RF| | |
| | | | | | |
| |TF|TF|CF| | |
| | | | | | |
| | | | | | |

Opus-4.7

GPT-5.5

DS-v4-Pro

Qwen-3.6-Max

DS-v4-Flash

DS-Chat

GLM-4.6

Models

Kimi-2.6

Kimi-2.5

Qwen-3.5-Plus

DS-Reasoner

- GLM-4.7
- GLM-5.1

Qwen3-Coder

MiniMax-2.5

0 1 2 3 4 5

Task Stages

Tooling & Integration Failure (TF) Infrastructure Failure (IF)

Reasoning Failure (RF) Planning Failure (PF)

Context Failure (CF) Pass ( )

Not Attempted ( )

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

- Figure 6: Multi-dimensional efficiency profiles. The radar axes show pipeline stage, mean stage-wise reward, inverted wall-clock time, inverted LLM cost, and inverted token usage. Each axis is mapped to [0, 100] using the leaderboard-wide normalization in §3.3.3. Larger values indicate stronger progress, higher reward, or lower resource use relative to the costliest run among evaluated models.

### 4.5 Overall Utility

Putting together mean reward, execution time, monetary cost, and token consumption, the Agent Efficiency Index (AEI) provides an operational measure of how efficiently an agent converts resources into useful engineering progress. This section therefore shifts from isolated outcome or resource measurements to a holistic view of utility, where a model is not only assessed by the reward it attains, but also by the time, budget, and context it spends to obtain that reward. AEI captures this productivity-oriented perspective and makes visible an important deployment trade-off between raw task utility and scalable resource discipline. Higher AEI values indicate greater useful reward per unit of aggregate resource consumption, with time, cost, and token usage jointly treated as important resource dimensions.

Figure 7 complements this scalar index with multi-dimensional profiles normalized efficiency profiles, where larger radii consistently indicate stronger stage progress, higher reward, shorter runtime, lower cost, or lower token consumption. These profiles reveal distinct efficiency tradeoffs across models that are obscured by single-value rankings.

Finding 5: Agentic models exhibit sharply different rankings under AEI and mean reward. The highest mean reward is achieved by Opus-4.7 (𝑀𝑅 = 93.39), but its AEI s only 40.00 because it also attains the observed maximum on wall-clock time, API cost, and token usage (11,913.79 s, $126.24, and 218.86M tokens), yielding zero scores on all three inverted resource axes. By contrast, GPT-5.5 achieves the highest AEI (81.57) with 𝑀𝑅 = 65.91, combining the maximum pipeline stage (𝑆 = 3) with mid-to-high scores on reward and all three resource dimensions. DS-v4-Pro reaches a higher mean reward (85.34) but a lower AEI (65.21), primarily because it consumes 6.17× more tokens than GPT-5.5 (136.76M vs. 22.18M). Qwen3-Coder illustrates the opposite pattern: strong time/cost/token scores at very low absolute cost (608.36s, $0.05, 3.48M tokens), but limited stage progress and mean reward (𝑀𝑅 = 25.97, AEI 64.22).

This inversion is the central interpretive role of AEI. It does not claim that a low-reward model is preferable in every setting. Instead, it ranks models by balanced progress across mean reward and the three resource dimensions on a common [0, 100] scale. For large-scale agent deployment, this distinction is essential: a

#### Figure 7: Multi-dimensional efficiency profiles. The radar axes show pipeline stage, mean stage-wise reward, inverted wall-clock time, inverted LLM cost, and inverted token usage, each mapped to [0, 100] using fixed reference scales. Larger values therefore indicate stronger progress, higher reward, shorter runtime, lower cost, or lower token consumption under the same external budget.

model that obtains the best reward through orders-of-magnitude higher cost, latency, and context usage may be less attractive than a more modest model that delivers repeatable progress with far better resource discipline. For research settings focused on pushing capability boundaries or one-off high-stakes tasks, MR remains the more relevant metric.

Finding 6: Top-tier LLM agents (e.g., Opus-4.7) bear excessive resource overhead for only marginal performance gains, with AEI clearly exposing such implicit inefficiency. The comparison between GPT-5.5, DS-v4-Pro, and Opus-4.7 illustrates this effect. DS-v4-Pro achieves a higher mean reward than GPT-5.5 (85.34 vs. 65.91), but GPT-5.5 has a higher AEI (81.57 vs. 65.21). Their monetary costs are nearly identical ($8.77 vs. $8.68), so the efficiency gap is mainly driven by time and context usage: DS-v4-Pro uses 1.37× more time and 6.17× more tokens. Opus-4.7 pushes mean reward even higher, but its AEI falls to 40.00 because it uses 2.19× more time, 14.39× more money, and 9.87× more tokens than GPT-5.5.

Othermodelsexhibitdifferentforms of inefficiency.DS-v4-Flash

reaches a strong mean reward of 66.48, but its AEI is 56.40 because it consumes 157.85M tokens. Qwen-3.6-Max reaches 𝑅 = 56.72 with AEI 66.75, primarily due to its high monetary cost of $67.28. These cases show that inefficient agent behavior is not one-dimensional. Some agents are context-heavy, some are price-heavy, and some are slow. AEI is useful because equal weighting across the five normalized dimensions prevents any one favorable reading from masking weakness on the others.

Finding 7: Agentic models naturally fall into three distinct operational regimes in production workflows when measured by AEI. The AEI ranking reveals three distinct operating regimes for LLM agents in production workflows. First, balanced productive agents exemplified by GPT-5.5 (AEI 81.57, 𝑀𝑅 = 65.91) achieve the highest composite efficiency while maintaining solid task performance, making them the strongest choice for generalpurpose production deployments under equal weighting assumption. Second, marginally efficient solvers such as Qwen3-Coder and DS-Chat deliver mid-tier AEI values (64.22 and 60.36) at very low cost, making them suitable for early-stage prototyping and lowstakes tasks despite modest mean rewards. Third, reward-heavy solvers including Opus-4.7, DS-v4-Pro, and DS-v4-Flash obtain strong mean rewards but lower AEI (40.00, 65.21, and 56.40), making them valuable primarily when maximum task completion outweighs resource discipline.

Overall, AEI shows that current LLM agents differ not only in the reward they can obtain, but also in how efficiently they spend time, money, and context to obtain it. For long-horizon agent evaluation, this distinction is central: an agent that is powerful but expensive, slow, or context-hungry may be less productive in practice than a more modest agent that delivers reliable progress with disciplined resource use.

### 5 Related Work

Existing work on LLM benchmarking and evaluation can be broadly divided into categories of benchmarks, agent execution frameworks, and runtime infrastructure platforms.

Repository-level coding benchmarks. Repository-level coding benchmarks have significantly advanced the evaluation of LLM-based software engineering systems. SWE-bench [12] established the paradigm of evaluating agents on real GitHub issues with execution-based patch verification, moving beyond isolated function synthesis benchmarks such as HumanEval [2]. SWE-bench Verified [4] further improved task reliability through human validation, while later extensions expanded the setting to multilingual repositories [30], longer and more complex issue resolution [6], mobile development [23], hardware engineering [5], and feature implementation [14]. Furthermore, recent advancements in 2026 have focused on refining the methodological rigor and scope of these benchmarks. For example, ATime-Consistent Benchmark [27] introduces strict temporal boundaries to prevent data contamination from future repository states, while SWE-QA [8] shifts the focus toward multi-hop codebase comprehension across dispersed files rather than patch generation alone. Similarly, USEbench [1] unifies various software engineering capabilities—ranging from code generation to program repair—into a comprehensive meta-benchmark. These benchmarks provide important execution-grounded signals, but their evaluation units remain largely independent. Each task typically starts from a clean repository state and ends with a single patch. As a result, they provide limited visibility into cumulative state evolution, intermediate artifact continuity, and cascading failure propagation. Ramp builds on the execution-grounded tradition of repository-level benchmarks, but shifts the evaluation target from isolated patch correctness to sustained progress along a serial dependency chain.

Interactive agent benchmarks. Interactive benchmarks evaluate agents in environments that require tool use, multi-step reasoning, and interaction with external systems. AgentBench [16] evaluates agents across diverse environments including operating systems, databases, and web tasks. GAIA [18] focuses on general assistant competence under multi-step tool-augmented reasoning. OSWorld [28] and WebArena [31] assess agents in realistic graphical and web-based environments, while Terminal-Bench [17] provides execution-grounded command-line tasks. These benchmarks are valuable because they expose agents to broader interaction surfaces than static code-generation tasks. However, their strength is primarily breadth rather than serial depth. They generally do not require agents to maintain a continuously evolving software artifact across a strict multi-stage dependency pipeline. Consequently, they are less suited to measuring whether early-stage defects corrupt downstream execution, whether valid intermediate states can be preserved, or whether process costs accumulate in ways that affect deployment feasibility. Ramp fills this gap by making long-horizon dependency continuity and runtime observability central to the evaluation design.

Paired and augmentation-oriented evaluation. Recent work has alsostudiedevaluationthroughpaired conditions. SkillsBench [15] evaluates tasks under both vanilla and skill-augmented settings, showing that external skills do not uniformly improve agent performance. SWE-Skills-Bench [10] applies a similar methodology to software engineering tasks and finds that many public skills provide limited benefit. These works are methodologically relevant because they show the importance of comparing agent behavior

under controlled intervention settings. However, their main objective is to measure augmentation efficacy. Ramp adopts a related paired-evaluation philosophy for a different purpose: failure decomposition. By comparing serial execution with and without resurrection, Ramp measures how much downstream diagnostic signal is hidden by upstream failure. This makes the intervention not an assistance mechanism for improving final scores, but a controlled assessment protocol for separating “cannot reach” from “cannot solve” in long-horizon workflows.

Long-horizon and self-evolving agent evaluation. Longhorizon agent evaluation has become increasingly important as agents move from single-step coding assistance toward autonomous engineering workflows. Frontier-Eng [3] evaluates agents on realistic research and development tasks, while RE-Bench [26] and PaperBench [22] study complex research-engineering settings. Voyager [24] demonstrates long-term skill accumulation in an openended environment. ProgramBench [29] further evaluates whether language models can reconstruct complete software systems from high-level specifications and behavioral signals, emphasizing holistic end-to-endsoftwareconstruction. In parallel, WildClawBench [7] assesses agents in native-runtime CLI environments with realistic multi-step tool execution, highlighting the difficulty of maintaining persistent context in actual deployments. The extended interaction lengths in these environments also introduce novel vulnerabilities, as demonstrated by AgentLAB [11], which evaluates how agents succumb to compounding errors and long-horizon adversarial attacks during multi-turn trajectories. These works highlight the need to evaluate agents over extended trajectories, but they typically lack controlled intermediate-state restoration. When an agent fails early, the remaining trajectory is often either terminated or interpreted through the corrupted state produced by the failure. Ramp complements this line of work by introducing resurrection as a principled mechanism for continuing evaluation after upstream failure. This enables measurement of both strict end-to-end autonomy and latent downstream capability under corrected prerequisites, while also preserving process-level evidence about cost, context pressure, and recovery behavior.

Overall, existing benchmarks have made substantial progress in execution-grounded coding evaluation, interactive tool-use assessment, paired intervention studies, and long-horizon agent analysis. However, none of these lines of work simultaneously centers serial dependency, intermediate artifact continuity, resurrection-based failure decomposition, and deployment-oriented process metrics. Ramp is designed around this combination. Its goal is not only to determine whether an agent can solve a task, but to assess whether the agent can sustain useful engineering progress under realistic runtime constraints.

### 6 Discussion on Limitations and Future Work

Although Ramp provides a realistic and runtime-observable evaluation framework for long-horizon software development agents, several challenges remain insufficiently explored and warrant future investigation.

Limitations. First, Ramp currently focuses on compiler construction, which provides well-defined intermediate artifacts and deterministic grading, but may not fully represent other production

domains such as machine learning engineering, DevOps automation, data pipelines, or web service maintenance. The methodology of serial dependency, resurrection, and process observation is transferable, but the empirical findings should not be assumed to generalize across all software engineering settings. Second, our experiments cover 15 model tests, which may not be completely sufficient to reveal clear behavioral patterns, and thus can be limited for fine-grained statistical conclusions across model families, agent scaffolds, or repeated runs. Third, all experiments are performed using the OpenHands backend. This design improves internal consistency, but it also means that the observed behavior reflects the interaction between models and this specific agent framework. Different agent frameworks may produce different planning strategies, tool-use patterns, context-management behavior, or recovery dynamics. Fourth, YatCC is publicly available, creating a potential contamination risk for frontier models that may have encountered parts of the codebase during training. Although Ramp primarily stresses serial continuity and artifact integration rather than memorization of isolated solutions, future private or temporally split workloads would provide stronger protection against this risk. Finally, the proposed AEI introduces a useful deployment-oriented view, but its equal weighting across reward, stage progress, time, cost, and tokens is necessarily subjective. Different deployment scenarios may prioritize latency, cost, reliability, or maximum reward differently, so AEI should be interpreted as one operational summary rather than a universal ranking criterion.

Future Work. These limitations point to several directions for future work. First, to broaden the evaluation scope by incorporating more models, more agent frameworks, and repeated runs under controlled settings, enabling clearer separation between model capability, scaffold design, and runtime orchestration effects. Second, to improve AEI so that it more objectively reflects agent process capability. This may involve learned or scenario-dependent weighting, Pareto-front analysis, robustness-aware scoring, or metrics that explicitly capture recovery behavior, context management, and productive tool use. Finally, future versions of Ramp should include harder workload variants with reduced or removed YatCC code scaffolding. The current scaffolded setting evaluates whether agents can complete and evolve an existing compiler pipeline; removing scaffolding would further test architectural reasoning, from-scratch implementation ability, and the capacity to maintain system-level consistency with less structural guidance. Together, these extensions would make Ramp a stronger foundation for studying not only whether agentic models can solve production-like tasks, but also whether they can do so robustly, efficiently, and with deployable engineering discipline.

### 7 Conclusion

Static and isolated benchmarks are increasingly insufficient for faithfully evaluating agentic model capabilities. Real-world softwaredevelopmentworkflows arelong-horizon, stateful, dependencydriven, and resource-constrained, making final task accuracy alone inadequateforassessingpractical deployability. We therefore present Ramp, a production-grounded assessment infrastructure built on YatCC compiler-construction tasks. Through serial workflow execution,resurrection-basedfailuredecomposition, and multi-dimensional

runtime metrics, Ramp assesses not only task outcomes, but also execution behavior, artifact continuity, resource efficiency, and downstream recoverability. Our findings reveal substantial gaps between benchmark performance and production robustness, demonstrating that practical utility cannot be judged by reward alone. A meaningful assessment must look into the agent’s process: how it spends time, cost, and context; how it handles failures; and whether it can maintain progress across dependent engineering stages.

### Acknowledgment

We would like to sincerely thank all contributors, collaborators, students, and users who participated in the design, development, deployment, operation, and continuous evolution of the YatCC platform and its ecosystem. YatCC is the result of extensive collective efforts spanning infrastructure engineering, platform operations, educational practice, and community-driven innovation. Their dedication, technical insights, feedback, experimentation, and long-term support have been instrumental in transforming YatCC from an initial course teaching tool into a practical AI-native serving platform.

### References

- [1] Leonhard Applis, Yuntong Zhang, Shanchao Liang, Nan Jiang, Lin Tan, and Abhik Roychoudhury. 2025. Unified Software Engineering Agent as AI Software Engineer. arXiv:2506.14683 [cs.SE] https://arxiv.org/abs/2506.14683
- [2] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374 [cs.LG] https://arxiv.org/abs/2107.03374
- [3] Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, and Qinhuai Na. 2026. Frontier-Eng: Benchmarking SelfEvolving Agents on Real-World Engineering Tasks with Generative Optimization. arXiv:2604.12290 [cs.AI] https://arxiv.org/abs/2604.12290
- [4] Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E Jimenez, John Yang, Leyton Ho, Tejal Patwardhan, Kevin Liu, and Aleksander Madry. 2024. Introducing SWE-bench Verified. OpenAI Blog.
- [5] Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, and Yun Liang. 2026. HWEBench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks. arXiv:2604.14709 [cs.AI] https://arxiv.org/abs/2604.14709
- [6] Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Nitin Pasari, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa Kundurthy, Sean Hendryx, Zifan Wang, Vijay Bharadwaj, Jeff Holm, Raja Aluri, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. 2025. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? arXiv:2509.16941 [cs.SE] https://arxiv.org/abs/2509.16941
- [7] Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Yang JingYi, Penghui Yang, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. 2026. WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation. arXiv:2605.10912 [cs.CL] https://arxiv.org/abs/2605.10912
- [8] Laïla Elkoussy and Julien Perez. 2026. SWE-QA: A Dataset and Benchmark for Complex Code Understanding. arXiv:2604.24814 [cs.SE] https://arxiv.org/abs/ 2604.24814
- [9] Yuhao Gu et al. 2026. YatCC: Yat Compiler Course. https://github.com/arcsysu/ YatCC
- [10] Tingxu Han, Yi Zhang andWei Song, Chunrong Fang andZhenyu Chen, and Youcheng Sun andLijie Hu. 2026. SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering? arXiv preprint (2026).

- arXiv:2603.15401 [cs.SE] https://arxiv.org/abs/2603.15401 arXiv:2603.15401.
- [11] Tanqiu Jiang, Yuhui Wang, Jiacheng Liang, and Ting Wang. 2026. AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks. arXiv:2602.16901 [cs.AI] https://arxiv.org/abs/2602.16901
- [12] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. In The Twelfth International Conference on Learning Representations. Oral presentation.
- [13] Chris Lattner and Vikram Adve. 2004. LLVM: A Compilation Framework for Lifelong Program Analysis and Transformation. San Jose, CA, USA, 75–88.
- [14] Wei Li, Xin Zhang, Zhongxin Guo, Shaoguang Mao, Wen Luo, Guangyue Peng, Yangyu Huang, Houfeng Wang, and Scarlett Li. 2025. FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation. arXiv:2503.06680 [cs.SE] https://arxiv.org/abs/2503.06680
- [15] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Binxu Li, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han chung Lee. 2026. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670 [cs.AI] https://arxiv.org/abs/2602.12670
- [16] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2024. AgentBench: Evaluating LLMs as Agents. In The Twelfth International Conference on Learning Representations.
- [17] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, JanLucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Jesse Hu, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. 2026. Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. arXiv:2601.11868 [cs.SE] https://arxiv.org/abs/2601.11868
- [18] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: A Benchmark for General AI Assistants. In The Twelfth International Conference on Learning Representations.
- [19] OpenAI. 2026. OpenAI API Documentation. https://platform.openai.com/docs/ api-reference Chat Completions API v1 and Responses API v1 specifications.
- [20] Sun Yat sen University arcSYSu Lab YatCC Team. 2026. YatCC: Yat Compiler Course. https://github.com/arcsysu/YatCC.
- [21] Mehil B Shah, Mohammad Mehdi Morovati, Mohammad Masudur Rahman, and Foutse Khomh. 2026. Characterizing Faults in Agentic AI: A Taxonomy of Types, Symptoms, and Root Causes. arXiv:2603.06847 [cs.SE] https://arxiv.org/abs/ 2603.06847
- [22] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. 2025. PaperBench: Evaluating AI’s Ability to Replicate AI Research. arXiv:2504.01848 [cs.AI] https://arxiv.org/ abs/2504.01848
- [23] Muxin Tian, Zhe Wang, Blair Yang, Zhenwei Tang, Kunlun Zhu, Honghua Dong, Hanchen Li, Xinni Xie, Guangjing Wang, and Jiaxuan You. 2026. SWE-Bench Mobile: Can Large Language Model Agents Develop Industry-Level Mobile Applications? arXiv:2602.09540 [cs.SE] https://arxiv.org/abs/2602.09540
- [24] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An Open-Ended Embodied Agent with Large Language Models. In Intrinsically-Motivated and Open-Ended Learning Workshop @NeurIPS2023. https://openreview.net/forum?id=nfx5IutEed
- [25] Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, and Graham Neubig. 2026. The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents. arXiv:2511.03690 [cs.SE] https://arxiv.org/abs/2511.03690

- [26] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Holden Karnofsky, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. 2025. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv:2411.15114 [cs.LG] https://arxiv.org/abs/2411.15114
- [27] Xianpeng, Sun, Haonan Sun, Tian Yu, Sheng Ma, Qincheng Zhang, Lifei Rao, and Chen Tian. 2026. ATime-Consistent Benchmark for Repository-Level Software Engineering Evaluation. arXiv:2603.26137 [cs.SE] https://arxiv.org/abs/2603. 26137
- [28] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu.

2024. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang

(Eds.), Vol. 37. Curran Associates, Inc., 52040–52094. doi:10.52202/079017-1650

- [29] John Yang, Kilian Lieret, Jeffrey Ma, Parth Thakkar, Dmitrii Pedchenko, Sten Sootla, Emily McMilin, Pengcheng Yin, Rui Hou, Gabriel Synnaeve, Diyi Yang, and Ofir Press. 2026. ProgramBench: Can Language Models Rebuild Programs From Scratch? arXiv:2605.03546 [cs.SE] https://arxiv.org/abs/2605.03546
- [30] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, and Liang Xiang. 2025. Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving. arXiv:2504.02605 [cs.SE] https://arxiv.org/abs/2504.02605
- [31] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), Vol. 2024. 15585–15606. https://proceedings.iclr.cc/paper_files/paper/2024/file/ 4410c0711e9154a7a2d26f9b3816d1ef-Paper-Conference.pdf

