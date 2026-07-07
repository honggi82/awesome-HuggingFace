# arXiv:2601.11077v1[cs.SE]16Jan2026

OpenMOSS

[Figure 1]

## ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development

Jie Yang1,* Honglin Guo1,* Li Ji1 Jiazheng Zhou1 Rui Zheng2 Zhikai Lei2 Shuo Zhang2 Zhiheng Xi1 Shichun Liu1 Yuxin Wang1

Bo Wang1 Yining Zheng1,† Tao Gui1,† Xipeng Qiu1,3

1Fudan University 2Shanghai Qĳi Zhifeng Co., Ltd. 3Shanghai Innovation Institute

### Abstract

The evolution of Large Language Models (LLMs) into autonomous agents has expanded the scope of AI coding from localized code generation to complex, repository-level, and execution-driven problem solving. However, current benchmarks predominantly evaluate code logic in static contexts, neglecting the dynamic, full-process requirements of real-world engineering, particularly in backend development which demands rigorous environment configuration and service deployment. To address this gap, we introduce ABC-Bench, a benchmark explicitly designed to evaluate agentic backend coding within a realistic, executable workflow. Using a scalable automated pipeline, we curated 224 practical tasks spanning 8 languages and 19 frameworks from open-source repositories. Distinct from previous evaluations, ABC-Bench require the agents to manage the entire development lifecycle from repository exploration to instantiating containerized services and pass the external end-to-end API tests. Our extensive evaluation reveals that even state-of-the-art models struggle to deliver reliable performance on these holistic tasks, highlighting a substantial disparity between current model capabilities and the demands of practical backend engineering.

Github: https://github.com/OpenMOSS/ABC-Bench Dataset: https://huggingface.co/datasets/OpenMOSS-Team/ABC-Bench

### 1 Introduction

Recent advancement of Large Language Models (LLMs) has redefined the role of AI in software engineering: moving beyond simple code prediction to acting as autonomous agents capable of exploring repositories, wielding terminal tools, and executing complex tasks within real environments. [3, 9, 16, 26, 29, 34]. Consequently, there is an urgent need to evaluate their ability in handling real-world software engineering tasks.

Despite the progress on software engineering benchmarks, evaluating these agents in production-like settings remains a critical gap. Current benchmarks focus on localized engineering, such as making isolated code edits that often overlook environment configuration, while relying on fragmented, unit-level valida-

∗Equal Contribution. Work done during an internship at Shanghai Qĳi Zhifeng Co., Ltd. †Corresponding authors. ynzheng@fudan.edu.cn, tgui@fudan.edu.cn

- ① Repository exploration
- ② Code modification

- ③ Environment configuration
- ④ Generate Dockerfile

[Figure 2]

Inspect project structure: ls -R repo Analyze documentation: cat README.md Check dependencies: cat requirements.txt

Install Core Dependencies: pip install fastapi uvicorn Install Project Requirements: pip install -r requirements.txt

You are a backend development expert. Inspect the backend project and fix a broken health check endpoint. Create a Dockerfile to run the service successfully.

[Figure 3]

[Figure 4]

[Figure 5]

Refactor Health Check Logic: endpoint.py: Fix status code return (+10 lines, -5 lines) Update Routing Table: router.py: Register new health endpoint (+3 lines)

Draft Container Specification: nano /app/repo/Dockerfile Define Base Image (Python 3.9-slim) Copy Source Code & Install Deps Set Entrypoint (uvicorn)

#### ABC-Bench

[Figure 6]

⑤ Docker build&run ⑥ End-to-end API test ⑦ Finish

[Figure 7]

[Figure 8]

Repo Dockerfile requirements.txt router.py...

[Figure 9]

def test_health_check(): # Verify the health endpoint url = "http://localhost:8000/health" response = requests.get(url) assert response.status_code == 200

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 1 Overview of the ABC-Bench evaluation pipeline. The figure illustrates the closed-loop evaluation process. In the development phase (Steps 1–4), the agent acts as a backend expert to analyze the repository, resolve issues, and draft container specifications. Transitioning to the validation phase, the benchmark system builds a Docker image from the agent’s output and deploys the service (Step 5). Finally, the functional correctness is verified by sending real HTTP requests to the deployed endpoint (Step 6), ensuring the fix works in a production-like environment.

tion instead of end-to-end tests [8, 10, 12, 15, 32, 33, 35, 36]. However, real-world software engineering is an integrated workflow where coding, configuration, and deployment are inherently intertwined.[13, 14, 18]. This limitation is particularly pronounced in backend development, which demands the integration of code changeswith environment configuration and container orchestration. While BaxBench [25] attemptsto evaluate backend capabilities, it is still constrained to relatively isolated tasks that fail to capture the full complexity of production environments. Consequently, there remains a significant lack of evaluations that rigorously assess backend agents within comprehensive, production-grade workflows.

To address these needs, we introduce ABC-Bench, a benchmark designed to evaluate agentic backend coding throughout the entire backend development lifecycle. Each task goes beyond localized code edits and requires the agent to configure the environment and instantiate a containerized service. Once the service is launched, we evaluate correctness strictly via external API-level integration tests, awarding credit only when the deployed service starts correctly and exhibits the expected behavior. To construct the benchmark at scale, we build ABC-Pipeline, a task-generation workflow that produces candidate tasks from open-source backend repositories with greatly reduced manual intervention. Applied to 2,000 open-source repositories, it yields 224 curated tasks spanning 8 backend programming languages and 19 frameworks, preserving the heterogeneity of realworld backend stacks.

Benchmark Expl. Code Env. Deploy E2E BaxBench [25] SWE-bench [12] FullStack Bench [7] DevBench [13] ABC-Bench (Ours)

Table 1 Comparison of Benchmark Scope. Real-world backend development mandates a continuous workflow spanning five distinct stages: (1) Repository Exploration (Expl.), (2) Code Editing (Code), (3) Environment Setup (Env.), (4) Deployment (Deploy), and (5) End-to-End Testing (E2E). While prior benchmarks focus primarily on localized coding tasks, ABC-Bench is the only benchmark to encompass this entire lifecycle.

Our extensive evaluation of various models and agent frameworks reveals that current systems remain far

from reliable on these full-lifecycle tasks. Even the top-performing model achieves only a 63.2% pass@1 rate, while many others lag substantially, indicating significant headroom for improvement. Further analysis identifies environment configuration and deployment as persistent bottlenecks, often acting as the primary barriers to success. These findings underscore a clear gap between current model capabilities and the rigorous demands of real-world backend engineering.

We highlight our contributions as follows:

- • We introduce ABC-Bench, a benchmark of 224 full-lifecycle backend tasks requiring autonomous repository exploration, environment configuration, deployment, and API-based verification.
- • We design ABC-Pipeline, a scalable workflow that automates the extraction of tasks from GitHub, significantly lowering the barrier for constructing realistic evaluation datasets.
- • We evaluate diverse agents and models, establishing robust baselines and uncovering that environment configuration and deployment are the predominant bottlenecks, offering key insights for future system improvements.

### 2 Related Work

Agentic Paradigms for Software Engineering. Recent progress in code-oriented large language models has expanded their scope from isolated programming problems to full-fledged software-engineering tasks. Earlier work on code-focused LLMs primarily emphasized code reasoning and synthesis in competitive or self-contained settings [11, 21], whereas more recent efforts increasingly adopt an agentic paradigm, enabling models to navigate real-world codebases and development environments through tools and multi-step execution to address end-to-end development tasks. A representative line of work conceptualizes software development as an agentic loop in which language models iteratively inspect repositories, modify code, and refine solutions based on execution feedback. This paradigm has been instantiated across a range of systems, from early agent–computer interfaces for resolving real GitHub issues to more general-purpose code-agent frameworks [9, 26, 27, 29]. Collectively, these efforts signal a decisive shift from passive code generation toward autonomous or semi-autonomous software-engineering agents operating over full repositories, moving large language model research beyond proof-of-concept demos toward real-world scenarios with tangible productivity impact, and motivating evaluations grounded in realistic software-development settings.

Evaluation of Coding Capability. In line with the shift toward agentic software-engineering systems that iteratively explore repositories, edit code, and execute tools, evaluation of coding capability has also moved from function-level generation on small, synthetic tasks [2, 6] toward more realistic settings that emphasize executable feedback, natural prompts, multi-stage development, and code editing [5, 10, 13, 19, 31, 35]. More recently, repository-level benchmarks evaluate code agents on real-world repositories via multi-step issue resolution and feature implementation under executable test signals [1, 30, 36]. To rigorously assess agentic mastery of the full software lifecycle, evaluations must extend beyond local code logic to include system-level operations and runtime interactions. Backend development exemplifies these challenges, necessitating an orchestrated workflow from repository exploration to environment provisioning, deployment, and live service validation. However, as shown in Table 1, existing benchmarks typically cover only parts of this lifecycle and rarely evaluate whether agents can deliver a deployment-ready backend service in a realistic stack [25]. Consequently, current evaluations often overemphasize code-only changes under pre-configured sandboxed environments and unit-test signals, leaving the full backend agentic workflow underexplored; ABC-Bench closes this gap by assessing the complete backend lifecycle with end-to-end, deployment-oriented verification.

AWS SAM (1)

Symfony (10)

Ruby (33)

net/http (14)

C# (43)

ASP.NET Core (43)

Laravel (20)

Koa (2)

Rust (4)

Sinatra (7)

Gin (8) Spring MVC (3)

PHP (33)

Rack (4)

CleverGo (3)

Axum (4)

Chi (4)

JavaScript (38)

Flask (5)

FastAPI (4)

Express (36)

Go (29)

Spring Boot (31)

Python (10)

Ruby on Rails (22)

Java (34)

Silex (3)

- Figure 2 Overview of the ABC-Bench dataset composition. The left pie chart illustrates the distribution of tasks across eight major programming languages. The right chart provides a detailed breakdown of the 19 web frameworks involved, demonstrating the benchmark’s capability to evaluate models across a wide spectrum of real-world software stacks.
- 3 ABC-Bench

- 3.1 Overview of ABC-Bench

ABC-Bench comprises 224 tasks, offering a diverse and balanced representation of modern backend ecosystems. As illustrated in Figure 2, the benchmark spans 8 programming languages and 19 backend frameworks. Of these tasks, 132 focus primarily on logic implementation within a pre-provisioned runtime, while 92 additionally require autonomous environment configuration and containerized service startup, thereby testing end-to-end operational capability. Beyond technical heterogeneity, the tasks are drawn from a broad spectrum of real-world domains, ensuring that the benchmark reflects practical engineering needs. These domains range from data analytics and search systems to commerce platforms, payment gateways, and developer tooling. Detailed category-level statistics and domain descriptions are summarized in Table 5.

- 3.2 Evaluation Pipelines

We evaluate models and agents using a standardized, isolated sandbox environment, which strictly separates the agent’s workspace from the backend service under test. As summarized in Figure 1, the evaluation setup launches an outer container that hosts the agent, delivers the task prompt.

Within this workspace, the agent is granted full autonomy to explore the repository, modify code, install dependencies, andupdateDockerconfigurations, enablingacomprehensive, full-lifecycledevelopmentloop. The agent must orchestrate the entire process without human intervention.

Upon submission of the solution or when the interaction budget limits are reached, the evaluation system attempts to build and launch the backend service in a separate inner container using the agent’s generated code and configurations. We determine success exclusively via functional API-level verification by executing external requests against the deployed service to validate its behavior. This execution-driven approach ensures that agents are evaluated not on static code artifacts, but on their ability to deliver a functioning, production-ready service.

- 4 ABC-Pipeline

To construct realistic backend development tasks at scale, we build the ABC-Pipeline, an automated workflow that converts open-source backend repositories into full-lifecycle development Tasks, as shown in Figure 3.

###### Phase 1: Repository Exploration Phase 2: Environment Setup & Verification Phase 3: Task Instantiation

[Figure 18]

Filtered Repository

API Group

Repositories on GitHub

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Filter backend repositories

Environment conﬁg

Generate

Task instruction

Filtered Repository

Environment dependencies

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Generate

Extract API group

Generate

[Figure 30]

git.patch

API Group:

test_connect.py test_func.py

Dockerfile

[Figure 31]

[Figure 32]

[Figure 33]

- - POST /api/orders
- - GET/api/orders/{id}

Apply patch Construct task

Build successful

[Figure 34]

[Figure 35]

ABC-Task

[Figure 36]

[Figure 37]

Veriﬁcation

Docker image

Generate API tests

{task instruction} Repository (patched) Dockerfile Deploy.sh test_connect.py test_func.py

[Figure 38]

[Figure 39]

test_connect.py test_func.py

Start service

[Figure 40]

[Figure 41]

[Figure 42]

Docker container

[Figure 43]

- Figure 3 Overview of the ABC-Pipeline workflow. The process consists of three phases: (1) Repository Exploration, where backend repositories are filtered and API tests are generated; (2) Environment Setup & Verification, which involves synthesizing Docker environments and verifying them against generated tests; and (3) Task Instantiation, where the final benchmark task is constructed by applying git patches and generating task instructions.

##### 4.1 Task Construction

The ABC-Pipeline constructs backend tasks from real-world repositories by leveraging an autonomous agent that augments GPT-5 with command-line capabilities. This construction process is structured into three distinct phases:

- Phase 1: Repository Exploration. We initiate the process by filtering a pool of 2,000 open-source, MITlicensed repositories to isolate high-quality backend candidates. The construction agent then autonomously explores each repository to identify functional API groups. Rather than relying on existing tests—which may be incomplete or outdated—the agent proactively generates dedicated verification suites. These suites cover both service connectivity and functional logic, serving as the criteria to verify the correctness of the model’s solution in the final evaluation pipeline.
- Phase 2: Environment Synthesis. Once API targets are identified, the pipeline advances to runtime synthesis. The agent analyzes the repository structure to resolve dependencies and generates the necessary container configuration files. It then attempts to build the runtime image and launch the service within an isolated container. This phase focuses strictly on establishing a deployable infrastructure, ensuring that the service can start up and listen on the expected ports, preparing the stage for subsequent validation.
- Phase 3: Task Instantiation. In the final phase, the pipeline synthesizes the actual benchmark problems using a masking-based strategy. For a selected API group, the agent formulates a natural language task instruction and synthesizes a solution patch representing the correct implementation. We then apply a reverse operation, selectively masking the implementation logic of the target endpoint to simulate a pre-implementation state. The resulting ABC-Task package encapsulates this masked repository, the task instructions, the environment setup files, and the verification suite. Specifically for tasks designated as environment configuration challenges, we subsequently remove all synthesized environment setup files from the final package and supplement the task instructions with explicit requirements for the evaluated model to autonomously configure the runtime environment.

##### 4.2 Task Verification

To guarantee the reliability and solvability of the automatically constructed tasks, we implement a rigorous two-stage verification protocol.

First, we verify the correctness of the ground truth environment and the test suite. We deploy the service using the original, unmasked repository and execute the generated tests. Since the implementation logic is fully intact, a valid task requires the tests to pass successfully. This step serves as a quality gate, filtering out repositories with unstable runtime configurations or flawed test logic, ensuring that the baseline evaluation setup is sound.

Second, we verify the effectiveness of the generated patch (mask) and the test suite’s ability to detect missing logic. We apply the masking strategy to the repository to simulate the pre-implementation state, redeploy the service, and re-execute the tests. A valid task requires the tests to fail upon applying the mask. Instances that continue to pass are discarded, as this indicates that either the mask failed to remove the core functionality, or the tests are insufficient to detect the incomplete implementation.

##### 4.3 Task Distribution

Through this pipeline, we initially generated 600 candidate tasks. To ensure a balanced evaluation benchmark, we filtered these candidates based on the distribution of programming languages and frameworks, resulting in a final set of 224 high-quality tasks spanning 8 programming languages and 19 web frameworks, of which 92 additionally require autonomous environment configuration.

### 5 Experiments

##### 5.1 Experimental Setup

We evaluate a diverse suite of representative open-source and proprietary models. The open-source models include Qwen3-8B and Qwen3-32B [28], DeepSeek-V3.2 [17], and GLM 4.7 [23], alongside coding-specialized variantslikeQwen3-Coder-30B-A3B-InstructandQwen3-Coder-480B-A35B-Instruct[11,28]andagent-oriented models like Nex-N1-671B and Nex-N1-32B [4]. For proprietary models, we include GPT-5, Gemini 2.5 Pro [22], and Claude Sonnet 4.5. For detailed information about the models used in our experiments, please refer to Table 3.

To ensure a unified evaluation protocol, we employ OpenHands [26] as the default agent framework for all models. For each model-agent pairing, we perform three independent runs per task. We set the sampling temperature to 0.7 for standard models and 1.0 for reasoning-enhanced variants.

##### 5.2 Main Results

Full-Lifecycle Tasks Remain Challenging. Table 2 reveals that ABC-Bench presents a rigorous barrier for current models. The state-of-the-art Claude Sonnet 4.5 achieves an overall pass@1 of 63.2%, while other models like DeepSeek-V3.2 hover around 50%. In contrast, smaller models, such as Qwen3-8B, fail to reach 10%. This performance stratification underscores the complexity of full-lifecycle software engineering: unlike isolated code generation, agents must maintain consistency across environment setup, dependency management, and functional deployment.

Models Lack Multilingual Robustness. Performance varies significantly by language stack. While widelyused languages like Python, Go, and JavaScript generally see higher success rates, other languages create major bottlenecks. Rust stands out as an extreme case where most models—including strong open-source contenders like DeepSeek-V3.2 fail to solve a single task (scoring 0.0%). Only the most capable proprietary models, specifically Claude Sonnet 4.5 and GPT-5, achieve meaningful success (above 30%), highlighting a distinct capability gap in handling less common or more complex language stacks.

Environment Configuration as the Primary Bottleneck. To understand the performance disparity, we analyzed 92 environment-related tasks by decomposing the workflow into two distinct stages: Environment Build (S1), which verifies if the service can be successfully constructed and started, and Functional Execution (S2), which measures the pass rate of functional tests specifically for the subset of tasks that successfully

By language (Avg. pass@1, %)

Model Thinking

Overall Py Go JS Java Ruby C# PHP Rust

Open Source Models Qwen3-8B Yes 26.7 5.7 21.9 3.9 7.1 4.7 1.0 0.0 8.3±1.1 Qwen3-32B Yes 16.7 2.3 22.8 2.9 9.1 7.8 5.1 0.0 8.9±1.1 Qwen3-Coder-30B-A3B No 46.7 31.0 50.9 19.6 25.3 13.2 31.3 0.0 28.6±1.7 Qwen3-Coder-480B-A35B No 56.7 55.2 57.0 52.9 36.4 24.0 36.4 16.7 43.1±1.9 Nex-N1-32B No 43.3 67.8 49.1 26.5 22.2 20.9 28.3 0.0 34.5±1.8 Nex-N1-671B No 60.0 63.2 49.1 54.9 30.3 26.4 34.3 0.0 42.1±1.9 DeepSeek-V3.2 No 56.7 64.4 57.0 57.8 42.4 38.8 48.5 0.0 50.1±1.9 GLM 4.7 Yes 40.0 56.3 50.0 40.2 34.3 32.6 36.4 0.0 40.1±1.9 Proprietary Models

GPT-5 Yes 30.0 56.9 46.5 67.6 55.6 36.4 44.4 41.7 49.4±1.9 Gemini 2.5 Pro Yes 30.0 29.9 47.4 19.6 20.2 17.1 16.2 8.3 25.0±1.7 Claude Sonnet 4.5 Yes 73.3 75.9 69.3 67.6 51.5 46.5 74.7 33.3 63.2±1.9

- Table 2 ABC-Bench results measured by average pass@1 (%) under three independent attempts per task. We report the overall performance over all 224 tasks, along with a breakdown by programming language. Models are evaluated using OpenHands as the agent framework. The highest scores for open-source and proprietary models are highlighted in bold, respectively; the second-highest scores are underlined, except for values equal to 0.

passed S1. As illustrated in Figure 4, Claude Sonnet 4.5 demonstrates the most robust full-lifecycle capability across these tasks, achieving high success rates in both stages (S1 ≈ 78%, S2 ≈ 80%). In contrast, models like GPT-5 and DeepSeek-V3.2 exhibit a striking imbalance: while they excel at functional coding (S2 > 80%), they struggle significantly with the initial setup (S1 < 50%). This sharp drop reveals that environment configuration is the primary bottleneck masking their algorithmic proficiency. Smaller models like Qwen3-8B are largely filtered out at the first stage (S1 < 20%). These findings suggest that bridging the gap in environment configuration is crucial for unlocking the full potential of LLMs in end-to-end software development.

Correlation Between Interaction Depth and Success. In addition to environment capabilities, we observe a strong positive correlation (r = 0.87) between the depth of agent interaction and task success, as shown in Figure 5. Top-performing models, notably Claude Sonnet 4.5, exhibit the longest execution trajectories (averaging > 60 turns), whereas weaker models like Qwen3-8B tend to terminate prematurely (≈ 10 turns). This trend underscores the iterative nature of ABC-Bench: unlike simple Q&A tasks, full-lifecycle software engineering requires agents to actively explore the environment, interpret error logs, and perform multiple rounds of debugging. Consequently, the ability to sustain and manage long-horizon interactions acts as a prerequisite for the high performance observed in our main results.

### 6 Analysis

##### 6.1 Agent Frameworks

To assess how agent framework influences performance beyond the model itself, we benchmark DeepSeekV3.2 and GPT-5 across three frameworks: OpenHands, Claude Code, and mini-SWE-agent. Figure 6(a) reveals that the framework is a critical variable. OpenHands facilitates peak performance (≈ 50%) for both models, whereas mini-SWE-agent causes severe degradation, notably dropping GPT-5’s success rate to below 20%. This highlights that ABC-Bench evaluates the holistic system, where the framework’s interaction strategy is as vital as the underlying model.

S1

S2 Avg Pass@1

| |
|---|

| |
|---|

100

70

88.4

82.9

81.8

60

80.3

77.3

80

Successratio(%)

AvgPass@1(%)

50

66.0

61.6

60

40

48.3

39.3

30

40

35.3

20

21.2

20

16.7

10

0

0

Qwen3-8BQwen3-Coder-480B-A35BQwen3-32B GPT-5DeepSeek-V3.2ClaudeSonnet4.5

- Figure 4 Analysis of environment configuration capabilities. Comparison of various models (including Claude Sonnet 4.5, GPT-5, DeepSeek-V3.2, and Qwen3-8B) across 92 environment-related tasks. The bar charts display Build Success (S1) and Conditional End-to-End Success (S2). The red line plot indicates the Average Pass@1.

Qwen3-8B

GPT-5

60%

Qwen3-32B

Gemini 2.5 Pro

Nex-N1-32B

Nex-N1-671B

Qwen3-Coder-30B-A3B

Claude Sonnet 4.5

Qwen3-Coder-480B-A35B

GLM 4.7

50%

DeepSeek-V3.2

trend

AveragePass@1

40%

30%

20%

r = 0.87 y = 0.894x + 5.45

10%

10 20 30 40 50 60 70

Average agent turns

Figure 5 Interaction turns vs. performance. Scatter plot illustrating the relationship between the average number of agent turns (x-axis) and the Average Pass@1 rate (y-axis) across various models. The blue fitted trend line reveals a strong positive correlation (r = 0.87).

##### 6.2 Effect of Agentic Post-Training

DeepSeek-V3.2 GPT-5

w/o SFT w/ SFT

AvgPass@1(%,3attempts)

AvgPass@1(%,3attempts)

60

40

We investigate the impact of further agentic supervised fine-tuning (SFT) on ABC-Bench performance. Using the agentic coding subset of the publicly-available 3Nex-N1 dataset, we fine-tuned Qwen3-8B and Qwen3-32B and evaluated them using OpenHands. Figure 6(b) demonstrates substantial improvements: pass@1 increases from 8.3% to 13.9% for the 8B model, and surges from 8.9% to 33.8% for the 32B model. These results indicate that training on high-quality agentic trajectories significantly boosts the model’s capability to handle full-lifecycle tasks, with larger models showing particularly strong data efficiency.

| |
|---|

| |
|---|

50.1

33.8

49.4

35

50

42.7

30

37.5

40

34.5

25

30

20

13.9

15

18.2

20

8.3 8.9

10

10

5

0

0

OpenHands Claude Codemini-SWE-agent

Qwen3-8B Qwen3-32B

(a) Agent Framework Performance

(b) Effect of Agentic Post-Training

Figure 6 (Left) Agent Framework Performance. Comparison of agent frameworks on ABC-Bench for DeepSeek-V3.2 and GPT-5. (Right) Effect of Agentic Post-Training. Impact of agent-style supervised fine-tuning (SFT) on ABC-Bench. Both are reported as average pass@1 (%) over three attempts per task.

##### 6.3 Performance by Task Category

To investigate whether models possess generalized engineering capabilities or exhibit domain-specific biases, we analyze performance variations across different task categories. Figure 7 stratifies performance across diverse application domains, revealing significant variance in task difficulty and model competency. We observe a distinct difficulty hierarchy: models generally excel in Analytics (where Claude Sonnet 4.5 reaches 86.7%) and Specialized tasks. In contrast, DevTools—the largest category (N = 53)—proves consistently challenging, with even the strongest model scoring below 50% (47.8%). This suggests that tasks involving development tooling and infrastructure require more complex, context-heavy reasoning than standard analytical scripts.

Furthermore, the heatmap highlights domain-specific specialization. While Claude Sonnet 4.5 dominates most categories, it is notably outperformed by GPT-5 in the Identity domain (60.0% vs. 73.3%). Overall, these results confirm that ABC-Bench captures meaningful domain heterogeneity, requiring agents to possess robust expertise across a wide spectrum of backend scenarios.

3https://huggingface.co/datasets/nex-agi/agent-sft

100%

[Figure 44]

Analytics (15)

6.7 4.4 42.2 62.2 35.6 44.4 60.0 55.6 54.4 33.3 86.7

Specialized (27)

- 12.3 22.2 39.5 54.3 45.7 51.9 64.2 51.9 71.6 37.0 79.0
- 12.4 17.1 31.4 45.7 44.8 45.7 65.7 51.4 55.2 33.3 68.6

80%

Other (35)

Communication (18)

20.4 14.8 29.6 38.9 40.7 37.0 25.9 35.2 51.9 27.8 66.7

AveragePass@1

60%

Content (40)

9.2 5.8 29.2 46.7 32.5 55.0 45.0 35.0 43.3 25.8 65.8

Identity (10)

0.0 3.3 20.0 23.3 33.3 20.0 36.7 33.3 73.3 20.0 60.0

40%

Infrastructure (8)

4.2 0.0 29.2 33.3 25.0 33.3 50.0 45.8 33.3 12.5 58.3

Commerce (17)

3.9 3.9 21.6 39.2 35.3 31.4 54.9 27.5 33.3 5.9 52.9

20%

DevTools (53)

3.1 2.5 20.8 35.8 23.3 35.8 42.8 33.3 39.0 18.2 47.8

Entertainment (1)

0.0 0.0 0.0 0.0 0.0 0.0 66.7 33.3 66.7 33.3 0.0

0%

Qwen3-8BQwen3-Coder-30B-A3BQwen3-32BQwen3-Coder-480B-A35BNex-N1-32BNex-N1-671BDeepSeek-V3.2GLM4.7Gemini2.5ProGPT-5ClaudeSonnet4.5

Figure 7 Heatmap of Pass@1 accuracy. The figure compares the performance of various models (x-axis) across different task categories (y-axis). The numbers in parentheses indicate the task count per category, and darker colors represent higher accuracy.

| |30|19|16|8|30|4|
|---|---|---|---|---|---|---|
| | | | | | | |

GPT-5

| |21|16|12|6|49|22|
|---|---|---|---|---|---|---|
| | | | | | | |

Qwen3-Coder 480B-A35B

| |20|76|17|11|60|19|
|---|---|---|---|---|---|---|
| | | | | | | |

Qwen3-8B

0 25 50 75 100 125 150 175 200 Count

Syntax Errors Path Missing

Dependency Missing Compilation Errors

Logic Errors Other

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

Figure 8 Distribution of error types across models. While environment-related issues remain a persistent challenge for all models, a distinct shift in error composition is observed: smaller models struggle with basic syntax, whereas larger models’ failures are concentrated in complex logic.

##### 6.4 Error Analysis and Failure Modes

To better understand the limitations of current code agents, we classify failure cases into six distinct categories, detailed in Table 4. Based on this taxonomy, we analyze the error breakdown for three different models, as illustrated in Figure 8. The distribution reveals two critical insights regarding model reliability and the nature of software engineering tasks.

First, environment configuration remains a universal bottleneck, though its severity varies by model capability. Errors related to Path Missing and Dependency Missing constitute a significant portion of failures. This is particularly acute in the smaller Qwen3-8B model, which suffers disproportionately from fundamental Path Missing errors (76 instances)—nearly four times that of GPT-5 (19 instances). This indicates a struggle with basic file system navigation and structure in smaller models.

Second, we observe a correlation between model scale and error sophistication. While the GPT-5 and Qwen3Coder-480B-A35B-Instruct demonstrate greater robustness in basic syntax and compilation, they exhibit a higher density of Logic Errors (30 and 49 instances, respectively) compared to other error types within their own distributions. This shift implies that as models gain better instruction-following and syntactic capabilities, the frontier of failure moves toward high-level reasoning and algorithmic correctness rather than low-level implementation details.

### 7 Conclusion

In this paper, we introduced ABC-Bench, a rigorous evaluation framework designed to assess LLM-based agents’ capabilities in handling full-lifecycle backend software engineering tasks. Sourced from 2,000 realworld GitHub repositories, ABC-Bench features a diverse collection of 224 tasks spanning 8 programming languages and 19 distinct frameworks. Our extensive experiments reveal that current systems are still far from reliable in handling these complex tasks. Further analysis identifies environment configuration and deployment as critical bottlenecks, often serving as major barriers to success before code logic can even be validated. These findings underscore a significant gap between current model capabilities and the practical demands of real-world backend engineering. We hope ABC-Bench provides insights that inspire the community to develop agents capable of mastering the full complexity of software production. To support this, we will open-source the code and dataset.

### References

- [1] Reem Aleithan, Haoran Xue, Mohammad Mahdi Mohajer, Elĳah Nnorom, Gias Uddin, and Song Wang. Swebench+: Enhanced coding benchmark for llms. CoRR, abs/2410.06992, 2024. doi: 10.48550/ARXIV.2410.06992. URL https://doi.org/10.48550/arXiv.2410.06992.

- [2] Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. CoRR, abs/2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.

- [3] Islem Bouzenia, Premkumar T. Devanbu, and Michael Pradel. Repairagent: An autonomous, llm-based agent for program repair. pages 2188–2200, 2025. doi: 10.1109/ICSE55347.2025.00157. URL https://doi.org/10.1109/ ICSE55347.2025.00157.
- [4] Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, et al. Nex-n1: Agentic models trained via a unified ecosystem for large-scale environment construction. arXiv preprint arXiv:2512.04987, 2025.

- [5] Linzheng Chai, Shukai Liu, Jian Yang, Yuwei Yin, Ke Jin, Jiaheng Liu, Tao Sun, Ge Zhang, Changyu Ren, Hongcheng Guo, Noah Wang, Boyang Wang, Xianjie Wu, Bing Wang, Tongliang Li, Liqun Yang, Sufeng Duan, Zhaoxiang Zhang, and Zhoujun Li. Mceval: Massively multilingual code evaluation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=UunCPtPOlZ.

- [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

- [7] Yao Cheng, Jianfeng Chen, Jie Chen, Li Chen, Liyu Chen, Wentao Chen, Zhengyu Chen, Shĳie Geng, Aoyan Li, Bo Li, Bowen Li, Linyi Li, Boyi Liu, Jerry Liu, Kaibo Liu, Qi Liu, Shukai Liu, Siyao Liu, Tianyi Liu, Tingkai Liu, Yongfei Liu, Rui Long, Jing Mai, Guanghan Ning, Z. Y. Peng, Kai Shen, Jiahao Su, Jing Su, Tao Sun, Yifan Sun, Yunzhe Tao, Guoyin Wang, Siwei Wang, Xuwu Wang, Yite Wang, Zihan Wang, Jinxiang Xia, Liang Xiang, Xia Xiao, Yongsheng Xiao, Chenguang Xi, Shulin Xin, Jingjing Xu, Shikun Xu, Hongxia Yang, Jack Yang, Yingxiang Yang, Jianbo Yuan, Jun Zhang, Yufeng Zhang, Yuyu Zhang, Shen Zheng, He Zhu, and Ming Zhu. Fullstack bench: Evaluating llms as full stack coders. CoRR, abs/2412.00535, 2024. doi: 10.48550/ARXIV.2412.00535. URL https: //doi.org/10.48550/arXiv.2412.00535.

- [8] Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Nitin Pasari, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa Kundurthy, Sean Hendryx, Zifan Wang, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. Swe-bench pro: Can AI agents solve long-horizon software engineering tasks? CoRR, abs/2509.16941, 2025. doi: 10.48550/ARXIV.2509.16941. URL https://doi. org/10.48550/arXiv.2509.16941.

- [9] Pengfei Gao, Zhao Tian, Xiangxin Meng, Xinchen Wang, Ruida Hu, Yuanan Xiao, Yizhou Liu, Zhao Zhang, Junjie Chen, Cuiyun Gao, Yun Lin, Yingfei Xiong, Chao Peng, and Xia Liu. Trae agent: An llm-based agent for software engineering with test-time scaling. CoRR, abs/2507.23370, 2025. doi: 10.48550/ARXIV.2507.23370. URL https: //doi.org/10.48550/arXiv.2507.23370.

- [10] Jiawei Guo, Ziming Li, Xueling Liu, Kaĳing Ma, Tianyu Zheng, Zhouliang Yu, Ding Pan, Yizhi Li, Ruibo Liu, Yue Wang, Shuyue Guo, Xingwei Qu, Xiang Yue, Ge Zhang, Wenhu Chen, and Jie Fu. Codeeditorbench: Evaluating code editing capability of large language models. CoRR, abs/2404.03543, 2024. doi: 10.48550/ARXIV.2404.03543. URL https://doi.org/10.48550/arXiv.2404.03543.

- [11] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, An Yang, Rui Men, Fei Huang, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5coder technical report. CoRR, abs/2409.12186, 2024. doi: 10.48550/ARXIV.2409.12186. URL https://doi.org/10. 48550/arXiv.2409.12186.

- [12] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=VTF8yNQM66.

- [13] Bowen Li, Wenhan Wu, Ziwei Tang, Lin Shi, John Yang, Jinyang Li, Shunyu Yao, Chen Qian, Binyuan Hui, Qicheng Zhang, Zhiyin Yu, He Du, Ping Yang, Dahua Lin, Chao Peng, and Kai Chen. Devbench: A comprehensive benchmark for software development. CoRR, abs/2403.08604, 2024. doi: 10.48550/ARXIV.2403.08604. URL https://doi.org/10.48550/arXiv.2403.08604.

- [14] Bowen Li, Wenhan Wu, Ziwei Tang, Lin Shi, John Yang, Jinyang Li, Shunyu Yao, Chen Qian, Binyuan Hui, Qicheng Zhang, Zhiyin Yu, He Du, Ping Yang, Dahua Lin, Chao Peng, and Kai Chen. Prompting large language models to tackle the full software development lifecycle: A case study. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 7511–7531. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.coling-main.502/.

- [15] Wei Li, Xin Zhang, Zhongxin Guo, Shaoguang Mao, Wen Luo, Guangyue Peng, Yangyu Huang, Houfeng Wang, and Scarlett Li. Fea-bench: A benchmark for evaluating repository-level code generation for feature implementation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 17160–17176. Association for Computational Linguistics, 2025. URL https: //aclanthology.org/2025.acl-long.839/.

- [16] Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Hongzhang Liu, Ronghao Chen, Yangfan He, Daxin Jiang, Binxing Jiao, Chen Hu, and Huacan Wang. Se-agent: Self-evolution trajectory optimization in multi-step reasoning with llm-based agents. CoRR, abs/2508.02085, 2025. doi: 10.48550/ARXIV. 2508.02085. URL https://doi.org/10.48550/arXiv.2508.02085.

- [17] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

- [18] Harshith Padigela, Chintan Shah, and Dinkar Juyal. Ml-dev-bench: Comparative analysis of AI agents on ML development workflows. CoRR, abs/2502.00964, 2025. doi: 10.48550/ARXIV.2502.00964. URL https://doi.org/ 10.48550/arXiv.2502.00964.

- [19] Musfiqur Rahman, SayedHassan Khatoonabadi, and Emad Shihab. Beyond synthetic benchmarks: Evaluating LLM performance on real-world class-level code generation. CoRR, abs/2510.26130, 2025. doi: 10.48550/ARXIV.2510.

26130. URL https://doi.org/10.48550/arXiv.2510.26130.

- [20] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. CoRR, abs/1909.08053, 2019. URL http://arxiv.org/abs/1909.08053.

- [21] Demin Song, Honglin Guo, Yunhua Zhou, Shuhao Xing, Yudong Wang, Zifan Song, Wenwei Zhang, Qipeng Guo, Hang Yan, Xipeng Qiu, and Dahua Lin. Code needs comments: Enhancing code llms with comment augmentation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 13640–

13656. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.809. URL https: //doi.org/10.18653/v1/2024.findings-acl.809.

- [22] Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi. org/10.48550/arXiv.2507.06261.

- [23] GLM Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, BohanZhang, BosiWen, BowenWu, BowenXu, CanHuang, CaseyZhao, ChangpengCai, ChaoYu, ChenLi, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jĳie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai, Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shĳie Li, Shuang Li, Shuang-Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zihan Wang, Zĳun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm4.5: Agentic, reasoning, and coding (arc) foundation models, 2025. URL https://arxiv.org/abs/2508.06471.
- [24] The Terminal-Bench Team. Terminal-bench: A benchmark for ai agents in terminal environments, Apr 2025. URL https://github.com/laude-institute/terminal-bench.
- [25] Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanovic, Jingxuan He, and Martin T. Vechev. Baxbench: Can llms generate correct and secure backends? In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=il3KRr4H9u.

- [26] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, and et al. Openhands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=OJd3ayDDoF.

- [27] Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, and Lingming Zhang. Live-swe-agent: Can software engineering agents self-evolve on the fly? arXiv preprint arXiv:2511.13646, 2025.

- [28] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388,

2025. doi: 10.48550/ARXIV.2505.09388. URL https://doi.org/10.48550/arXiv.2505.09388.

- [29] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/ 2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html.

- [30] Boxi Yu, Yuxuan Zhu, Pinjia He, and Daniel Kang. Utboost: Rigorous evaluation of coding agents on swe-bench. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 3762–3774. Association for Computational Linguistics, 2025. URL https: //aclanthology.org/2025.acl-long.189/.

- [31] Zhaojian Yu, Yilun Zhao, Arman Cohan, and Xiaoping Zhang. Humaneval pro and MBPP pro: Evaluating large language models on self-invoking code generation task. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 13253–13279. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.findings-acl.686/.

- [32] Daoguang Zan, Zhirong Huang, Ailun Yu, Shaoxin Lin, Yifan Shi, Wei Liu, Dong Chen, Zongshuai Qi, Hao Yu, Lei Yu, Dezhi Ran, Muhan Zeng, Bo Shen, Pan Bian, Guangtai Liang, Bei Guan, Pengjie Huang, Tao Xie, Yongji Wang, and Qianxiang Wang. Swe-bench-java: A github issue resolving benchmark for java. CoRR, abs/2408.14354, 2024. doi: 10.48550/ARXIV.2408.14354. URL https://doi.org/10.48550/arXiv.2408.14354.

- [33] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, and Liang Xiang. Multi-swe-bench: A multilingual benchmark for issue resolving. CoRR, abs/2504.02605, 2025. doi: 10.48550/ARXIV.2504.02605. URL https://doi.org/10.48550/arXiv.2504.02605.

- [34] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges. pages 13643–13658, 2024. doi: 10.18653/V1/2024. ACL-LONG.737. URL https://doi.org/10.18653/v1/2024.acl-long.737.
- [35] Shudan Zhang, Hanlin Zhao, Xiao Liu, Qinkai Zheng, Zehan Qi, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Naturalcodebench: Examining coding performance mismatch on humaneval and natural user queries. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 7907–7928. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.471. URL https://doi.org/10.18653/v1/2024. findings-acl.471.

- [36] YaolunZhang, YinxuPan, Yudong Wang, Jie Cai, Zhi Zheng, GuoyangZeng, and Zhiyuan Liu. Pybench: Evaluating LLM agent on various real-world coding tasks. CoRR, abs/2407.16732, 2024. doi: 10.48550/ARXIV.2407.16732. URL https://doi.org/10.48550/arXiv.2407.16732.

- [37] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. Sglang: Efficient execution of structured language model programs. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/ 724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.

## Appendix

### Appendix Contents

- A Experiment Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- B Metric Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C Dataset Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D Responsible NLP Research Statements . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- E Task Instruction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

### A Experiment Detail

To ensure fair and reproducible comparisons across various models and agent frameworks, we standardize our inference and evaluation settings. Specifically, we set the sampling temperature to 1.0 for reasoningoriented models and 0.7 for non-reasoning models. These configurations remain consistent when evaluating different model-framework combinations to maintain a controlled experimental environment.

Models. For closed-source models, as well as DeepSeek-V3.2, Nex-N1, and GLM 4.7, we access them via official APIs, strictly adhering to documented usage policies and recommended request formats. For the Qwen series and the specialized Nex-N1-32B, we deploy them on a cluster of up to 128 NVIDIA H100 GPUs using the SGLang [37] framework. To facilitate long-context evaluation for the Qwen3 Instruct family, we extend the context window via YARN-based RoPE scaling during deployment. Furthermore, we enable native tool-calling parsers for Qwen3-8B, Qwen3-32B, Qwen3-Coder, and Nex-N1-32B to assess their tooluse capabilities within their intended inference interfaces.

Evaluation. All evaluations are executed in containerized environments and managed through the TerminalBench framework [24], which orchestrates agents, tasks, and Docker runtime isolation in a uniform way. We evaluate three open-source agent frameworks, OpenHands, mini-SWE-agent, and Claude code. For each framework, we follow the official documentation and widely adopted community configurations. We keep task-level constraints such as timeouts, concurrency, and environment setup consistent across models.

Reliability. To improve reliability and account for stochasticity in decoding and agent behavior, we run the full evaluation for each model three times and report the mean performance across the three runs. Beyond reporting averages, we analyze the error distribution of each run to identify systematic failure patterns and to diagnose run-to-run variability. This procedure helps us maximize the trustworthiness of the execution results and reduces the chance that conclusions are driven by outliers or transient environment effects.

Training. Beyond evaluation, we fine-tune Qwen3 models at 8B and 32B scales using a Megatron-LM-based training stack [20]. The training is conducted on 128 NVIDIA H100 GPUs, employing a 128K sequence length and context parallelism to ensure efficient scaling. The models are trained for 3 epochs with a target global batch size of 128. Optimization is performed using Adam with a cosine learning rate schedule and a warmup period.

### B Metric Detail

We quantify the agent’s capability by decomposing the workflow of the 92 environment-related tasks into two distinct sequential stages: Environment Build (S1) and Functional Execution (S2). This decomposition allows us to calculate the success rate for each stage independently, separating environment configuration outcomes from functional logic correctness.

Stage 1, denoted as Environment Build (S1), involves constructing and starting the service. Success requires generating a valid Dockerfile, building the image, and initializing the container without fatal errors. Stage 2, denoted as Functional Execution (S2), measures the pass rate of functional tests specifically for the subset of tasks that successfully passed S1. It evaluates the correctness of the functional logic within the established environment.

Formally, given a total of Ntotal environment-related tasks, we track two outcome counts. Let Nbuild be the number of tasks where the environment is successfully built and started (passing S1), and let Npass be the number of tasks that pass all functional tests among those that succeeded in S1.

The success rates for these two stages are defined as follows:

Nbuild Ntotal

Success RateS1 =

(1)

Model Params Launch Max tokens MLP Type Provider License Qwen3-8B 8B 2025–04 128K Dense Alibaba Apache 2.0 Qwen3-32B 32B 2025–04 128K Dense Alibaba Apache 2.0 Qwen3-Coder-30B-A3B 30B 2025–07 256K MoE Alibaba Apache 2.0 Qwen3-Coder-480B-A35B 480B 2025–07 256K MoE Alibaba Apache 2.0 Nex-N1-32B 32B 2025–11 128K Dense Nex AGI Apache 2.0 Nex-N1-671B 671B 2025–11 128K MoE Nex AGI Apache 2.0 DeepSeek-V3.2 671B 2025–12 128K MoE DeepSeek MIT GLM 4.7 358B 2025–12 200K MoE Z.ai MIT GPT-5 – 2025–08 400K – OpenAI Proprietary Gemini 2.5 Pro – 2025–06 1M – Google DM Proprietary Claude Sonnet 4.5 – 2025–09 200K – Anthropic Proprietary

- Table 3 Model configurations used in our experiments, including parameter scale, launch date, maximum context length, MLP type, provider, and license.

Npass Nbuild

Success RateS2 =

(2)

In cases where Nbuild = 0, we define Success RateS2 = 0. This conditional formulation ensures that S2 accurately reflects the model’s coding proficiency by decoupling it from the environment configuration bottleneck

identified in S1.

### C Dataset Detail

ABC-Bench is constructed from public GitHub repositories that are explicitly released under the MIT license. We strictly adhere to the terms of these licenses, ensuring that our use is consistent with the open-source nature of the original artifacts. The dataset is intended for research on automated code agents, and we stipulate that any derivatives of this data used for research purposes must remain within research contexts. We list them in Table 6.

The benchmark focuses on code logic and environment configuration. During the construction pipeline, we operate on repository snapshots and apply automated filtering to detect and redact sensitive credentials, such as access tokens, API keys, and private keys. We explicitly avoid collecting logs or other user-generated content containing personal information. Consequently, the dataset concentrates on technical artifacts rather than personal or demographic data. We do not design tasks to elicit offensive content; any incidental text (e.g., in code comments) is inherited directly from the original open-source projects.

ABC-Bench contains 224 tasks derived from real-world backend repositories, covering a diverse range of programming languages, frameworks, and task categories. Figure 2 provides detailed documentation of the dataset statistics, including the distribution over languages, frameworks, and task types. Regarding the data collection protocol, the construction process is primarily automated with internal quality verification by the authors; it does not involve external human participants or manual annotation by crowdworkers, and therefore does not constitute human-subjects research requiring IRB review.

### D Responsible NLP Research Statements

All code used for task construction in ABC-Pipeline and ABC-Bench is sourced from public GitHub repositories explicitly released under the MIT license. We strictly avoid proprietary code or repositories with unclear licensing. Regarding potential risks, we anticipate minimal negative impact, as our work focuses on evalu-

###### Error type Meaning

Syntax Errors Static syntax issues in source code or configuration files, reported by a compiler, interpreter, or linter. Typical examples include invalid tokens, missing brackets, malformed JSON or YAML, or other parse errors that prevent the code from being compiled or executed.

Path Missing A required file or directory cannot be found when building the image or running the service. Common cases include incorrect paths in Dockerfile COPY or ADD instructions, missing project directories or entrypoints, or startup scripts referencing non-existent configuration files.

Dependency Missing A required dependency is not available in the environment. This includes missing language packages, system libraries, or runtime extensions, such as missing PHP extensions, Python or Node.js packages, Java runtimes, or operating-system-level libraries that the application or framework expects.

Compilation Errors Build or compilation failures that are not purely syntax mistakes, but arise during the compilation or linking phase. Typical examples are type mismatches, unresolved symbols, missing headers, or incompatibilities between language, framework, or tool versions that prevent producing a build artifact.

Logic Errors The service builds and runs, but the behavior is functionally incorrect. This includes wrong business logic, incorrect HTTP status codes, mismatched response schemas or data types, failing assertions in tests, or incorrect handling of edge cases, even though the process itself does not necessarily crash.

Other Errors that do not fit cleanly into the above categories, typically related to environment, infrastructure, or multi-factor issues. Examples include port conflicts, file permission problems, disk or memory limits, SSL or network connectivity failures, problems with base images, or mixed cases where no single category is clearly dominant.

- Table 4 Explanations of error types used in our analysis, summarizing common failure modes observed during code changes, build, deployment, and end-to-end API evaluation.

ating technical infrastructure using public data and does not introduce dangerous capabilities or harmful content.

Publicly-accessible models, datasets, software, and other related artifacts are used under their corresponding licenses or agreements.

Our data collection and task construction processes are primarily automated. To ensure data quality, the authors performed a manual verification of the task titles and descriptions. This review process was conducted exclusively by the research team on a voluntary basis and did not involve the recruitment of external human subjects or crowdworkers. Consequently, no financial compensation was involved. Since this procedure constitutes internal quality assurance rather than experimentation on human subjects, the study is exempt from standard IRB review.

We apply automated screening during dataset construction to detect and exclude content that names or uniquely identifies individual people or contains offensive material. If such cases are detected in the dataset fields we distribute, we remove the affected items or redact sensitive strings to protect privacy and anonymize the data prior to release.

We used AI assistants to support research and writing. All technical claims, experimental results, and reported numbers were produced by our code and verified by the authors.

Category Count Description Content 34 Content platforms such as blogs, knowledge bases, documenta-

tion sites, and course portals. Commerce 17 Backends for e-commerce, ordering, payment processing, book-

keeping, and financial workflows. DevTools 41 SDKs, frameworks, scaffolds, routers, and package tools targeting

developers. Analytics 14 Services focused on search, reporting, monitoring, recommendations, and other data analytics. Communication 18 Chat, email, notification, collaborative editing, and CRM-style

systems. Entertainment 1 Backends serving games or other entertainment experiences. Identity 9 Systems handling login, OAuth, SSO, and authorization manage-

ment. Infrastructure 2 Examples centered on containerization, operations, and deployment practices. Specialized 36 REST/GraphQL services or demos built around specific business domains. Other 33 Mixed scenarios.

Table 5 Task categories in ABC-Bench, grouped by backend application domain with counts and brief descriptions.

### E Task Instruction

The task instructions employed in ABC-Bench are detailed below, derived from the templates in Listing 1 and Listing 2. Specifically, Listing 1 presents the template for tasks without environment configuration, while Listing 2 illustrates the template for tasks that include environment configuration. For a concrete instantiation, please refer to Listing 3.

Listing 1 Task template used in ABC-Bench for backend tasks without environment-configuration subtasks.

You are a backend development expert. Please inspect the backend project located in the current directory, determine its programming language and architectural style, and then complete the following code implementation and environment setup tasks.

{Task Description} {API Endpoint Specifications and Requirements} {Important Notes}

Listing 2 Task template used in ABC-Bench for backend tasks that include environment-configuration subtasks (env subset).

You are a backend development expert. Please inspect the backend project located in the current directory, determine its programming language and architectural style, and then complete the following code implementation and environment setup tasks.

{Task Description} {API Endpoint Specifications and Requirements} {Important Notes}

After completing all source code implementation, create a Dockerfile for this project using the following example template as a reference (Python version):

``` setup base

FROM nikolaik/python-nodejs:python3.12-nodejs22-bullseye RUN apt-get update && apt-get install -y sqlite3

install dependencies and copy project files

WORKDIR /app COPY . /app/ RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3", "app.py"] ``` Notes

- 1) Ensure all required project dependencies are installed inside the image

- 2) The generated Dockerfile must successfully build and run the application

- 3) The Dockerfile must be created in the root directory of the backend project, i.e. {/app/ project_name/Dockerfile}

Listing 3 Natural-language task instruction used for ABC-Bench in our environment-centric backend benchmark.

You are a backend development expert. Please inspect the backend project located in the current directory, determine its programming language and architectural style, and then complete the following code implementation and environment setup tasks.

Implement the missing comment-domain logic inside: module/core/src/main/java/io/zhc1/realworld/service/ArticleCommentService.java

These methods power:

- - POST /api/articles/{slug}/comments

- - GET /api/articles/{slug}/comments

- - DELETE /api/articles/{slug}/comments/{id}

Required behavior

- 1) ArticleComment getComment(int commentId)

- - Look up the comment via ArticleCommentRepository.findById

- - Throw NoSuchElementException with the exact message expected by the REST layer when the id is invalid

- 2) List<ArticleComment> getComments(Article article)

- - Return every comment associated with the provided article via articleCommentRepository. findByArticle
- 3) ArticleComment write(ArticleComment articleComment)

- Persist the comment using the repository and return the managed entity

- 4) void delete(User requester, ArticleComment articleComment)

- - Reject deletions from non-authors by raising IllegalArgumentException

- - Otherwise remove the entity via articleCommentRepository.delete

Constraints

- - Never accept null Article, User, or ArticleComment inputs; throw IllegalArgumentException when misused

- - Ownership checks must rely on ArticleComment.isNotAuthor to stay consistent with the aggregate's equality semantics

- - Keep the service side-effect free outside repository operations so controllers can safely call it multiple times per request

After completing all source code implementation, create a Dockerfile for this project using the following example template as a reference (Python version):

``` setup base

FROM nikolaik/python-nodejs:python3.12-nodejs22-bullseye RUN apt-get update && apt-get install -y sqlite3

install dependencies and copy project files

WORKDIR /app COPY . /app/ RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3", "app.py"] ``` Notes

- 1) Ensure all required project dependencies are installed inside the image

- 2) The generated Dockerfile must successfully build and run the application

- 3) The Dockerfile must be created in the root directory of the backend project, i.e. /app/1 chz_realworld-java21-springboot3/Dockerfile

###### Table 6 Repository statistics: ABC-Bench comprises a diverse collection of 127 GitHub repositories and 224 tasks.

|Language|Framework Repo # Tasks<br><br>|
|---|---|
|C#<br><br>|ASP.NET Core<br><br>angelsix/fasetto-word 4 AIDotNet/OpenDeepWiki 3 neozhu/cleanaspire 3 cloudinary/CloudinaryDotNet 2 RageAgainstThePixel/OpenAI-DotNet 2 BenediktAlkin/SongTaggerForSpotify 2 ntxinh/AspNetCore-DDD 1 withsalt/BilibiliLiveTools 1 BlazorStatic/BlazorStatic 1 ardalis/CachedRepository 1 ardalis/CleanArchitecture 1 stidsborg/Cleipnir.NET 1 linezero/GitServer 1 khellang/Middleware 1 paiden/Nett 1 devmentors/PackIT 1 reactjs/React.NET 1 samanazadi1996/Sam.CleanArchitecture 1 sharpenrocks/Sharpen 1 Havunen/SystemTextJsonPatch 1 microsoftgraph/aspnetcore-webhooks-sample 1 dtm-labs/client-csharp 1 jamesmh/coravel 1 Azure-Samples/cosmos-db-design-patterns 1 Azure/dev-spaces 1 dotnet/dotNext 1 cornflourblue/dotnet-6-crud-api 1 cornflourblue/dotnet-6-jwt-refresh-tokens-api 1 dotnet/dotnet-monitor 1 Azure-Samples/eShopOnAzure 1 unosquare/passcore 1 ThunderDev1/reactjs-ts-identityserver 1 simplyvinay/vue-expenses 1|
|JavaScript|Express<br><br>15Dkatz/official_joke_api 4 CapacitorSet/box-js 3 azat-co/expressworks 3 auth0/auth0-react 2 carteb/carte-blanche 2 conclave-team/conclave 2 jellyfangs/messenger-bot-tutorial 2 BretFisher/node-docker-good-defaults 2 hagopj13/node-express-boilerplate 2 eduardoboucas/staticman 2 filipedeschamps/video-maker 2 adrianvlupu/C4-Builder 1 microsoft/PowerBI-Developer-Samples 1 davila7/claude-code-templates 1 attacomsian/code-examples 1 external-secrets/kubernetes-external-secrets 1 jagenjo/litegraph.js 1 kogosoftwarellc/open-api 1 ankur-anand/simple-sso 1 daspinola/video-stream-sample 1<br><br>|

|Language<br><br>|Framework Repo # Tasks|
|---|---|
| |welldone-software/why-did-you-render 1 Koa jamhall/s3rver 2|
| | |
|Java<br><br>|Spring Boot<br><br>1chz/realworld-java21-springboot3 6 getmoneynote/moneynote-api 3 kawhii/sso 3 dailycodebuffer/Spring-MVC-Tutorials 2 ttulka/ddd-example-ecommerce 2 cassiomolin/log-aggregation-spring-boot-elastic-stack 2 AwakenCN/Almost-Famous 1 zzzzbw/Fame 1 KonBAI-Q/RuoYi-Flowable-Plus 1 HouariZegai/Tutorials 1 liuxx-u/bird-java 1 cucumber/cucumber-jvm 1 PomZWJ/database-export 1 liyupi/father 1 zhongjinggz/geekdemo 1 threedr3am/learnjavabug 1 ssssssss-team/magic-api 1 nonacosa/new-bee 1 allanzhuo/yyblog 1 3pillarlabs/socialauth 2|
| |Spring MVC<br><br>v5tech/elasticsearch-jest-example 1|
|Ruby|Ruby on Rails<br><br>danielschuster-muc/potter-db 6 exoego/rspec-openapi 6 active-elastic-job/active-elastic-job 3 amatsuda/html5_validators 2 rubygems/rubygems.org 2 hyperstack-org/hyperstack 1 ElMassimo/oj_serializers 1 crmne/ruby_llm 1 assaf/rack-oauth2-server 2<br><br>|
| |Sinatra<br><br>maccman/go 1 mobomo/green_onion 1 isaiah/jubilee 1 brandur/rocket-rides-atomic 1 joakimk/testbot 1<br><br>|
| |Rack<br><br>flippercloud/flipper 3 amoniacou/danthes 1<br><br>|
|PHP<br><br>|Laravel<br><br>amsgames/laravel-shop 5 cretueusebiu/laravel-vue-spa 5 laqul/laqul 3 JeffreyWay/council 2 ploi/roadmap 2 WangNingkai/OLAINDEX 1 devinsays/laravel-react-bootstrap 1 protonemedia/laravel-splade 1 composer/packagist 4<br><br>|
| |Symfony<br><br>AdventureLookup/AdventureLookup 2 hgraca/explicit-architecture-php 2 nelmio/NelmioJsLoggerBundle 1 ankitpokhrel/tus-php 1|
| |Silex intaro/pinboard 3|

|Language<br><br>|Framework Repo # Tasks|
|---|---|
|Go<br><br>|net/http<br><br>tsileo/blobstash 6 go-spatial/tegola 2 flycash/toy-web 2 Azure/azure-sdk-for-go 1 99designs/gqlgen 1 darkweak/souin 1 stripe-archive/timberlake 1 air-go/rpc 3|
| |Gin<br><br>Email-Dashboard/Email-Dashboard 2 izghua/go-blog 1 nhost/hasura-auth 1 swaggo/swag 1|
| |Chi ashirt-ops/ashirt-server 4|
| |CleverGo clevergo/clevergo 3|
|Python<br><br>|Flask<br><br>HaoZhang95/Python24 3 stripe-samples/accept-a-payment 2 Azure/apiops 3<br><br>|
| |FastAPI<br><br>Tongjilibo/bert4torch 1|
| |AWS SAM aws-samples/serverless-test-samples 1|
|Rust<br><br>|Axum<br><br>restsend/rustpbx 3 Totodore/socketioxide 1|

