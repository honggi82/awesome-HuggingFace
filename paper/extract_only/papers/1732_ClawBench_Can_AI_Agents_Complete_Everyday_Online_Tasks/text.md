# arXiv:2604.08523v1[cs.CL]9Apr2026

### ClawBench: Can AI Agents Complete Everyday Online Tasks?

###### Yuxuan Zhang1,2,3,§, Yubo Wang2,5, Yipeng Zhu1, Penghui Du3, Junwen Miao4, Xuan Lu6, Wendong Xu7,§, Yunzhuo Hao8, Songcheng Cai5, Xiaochen Wang9, Huaisong Zhang10, Xian Wu3, Yi Lu5, Minyi Lei5, Kai Zou11, Huifeng Yin7, Ping Nie5,§, Liang Chen7,†, Dongfu Jiang2,5,†, Wenhu Chen2,5,†, Kelsey R. Allen1,2,†

1University of British Columbia 2Vector Institute 3Etude AI 4Carnegie Mellon University 5University of Waterloo 6Shanghai Jiao Tong University 7UniPat AI 8Zhejiang University 9HKUST 10Tsinghua University 11Netmind.ai

https://claw-bench.com

###### Sandbox Evaluation

###### (Traditional Benchmarks) Performance Comparison

###### Comprehensive Categories

Daily Life (21)

Shopping (16)

Oﬄine / Sandbox

###### Task Completion Rate %

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Check ﬁnal page

- - Static HTML only
- - Fixed DOM structure
- - No login / auth
- - No dynamic content
- - No real-world challenge

[Figure 5]

fake-shop.test

OSWorld-Veriﬁed WebArena-Veriﬁed

[Figure 6]

[Figure 7]

Task

ClawBench

[Figure 8]

Entertainment (15)

Oﬃce (9)

[Figure 9]

80

Submit

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

No explanation

Controlled · Oversimpliﬁed · Unrealistic

75.0 72.5

Job Search (8)

Personal Mgmt(4)

66.4 67.3

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

60

###### Real-World Evaluation (ClawBench - ours)

Dev & Tech (15) Automation (3)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

###### Live

Agentic Evaluator

40

Social (8)

Rating (10)

- - Cookie consent popups
- - Dynamic JS rendering
- - Complex and multi-step interaction

[Figure 31]

google.com/flights

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

33.3

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Search Flights

[Figure 44]

Task

[Figure 45]

Academia (5)

Education (9)

20

Session Replay

Action Screenshots

HTTP Traﬃc

Rich Verdict

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

score + justiﬁcation + per-evidence references

Agent Messages

Browser Actions

Pets (11) Finance (6)

Travel (13)

6.5

Traceable Analysis

0

[Figure 54]

[Figure 55]

[Figure 56]

Challenging · Dynamic · Realistic

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Claude-sonnet-4.6 GPT-5.4

- Figure 1: CLAWBENCH overview. Left: 153 tasks across 15 life categories. Middle: existing benchmarks evaluate agents in offline sandboxes with static HTML and fixed DOM structures; CLAWBENCH evaluates on live websites with real-world complexity and provides rich, traceable verdicts via an agentic evaluator. Right: Claude-Sonnet-4.6 and GPT-5.4 achieve 65-75% task completion on established benchmarks such as OSWorld and WebArena, but only 33.3% and 6.5%, respectively, on CLAWBENCH, highlighting the difficulty of real-world everyday web tasks.

0 10 20 30 40

Claude Sonnet 4.6

GLM-5 Gemini 3 Flash

Claude Haiku 4.5

GPT-5.4 Gemini 3.1 Flash Lite

Kimi K2.5

33.3% 24.2%

19.0% 18.3%

6.5% 3.3%

0.7%

- Figure 2: Main results: success rate on CLAWBENCH for 7 frontier models. Even the strongest model (Claude Sonnet 4.6) completes only 33.3% of tasks, while two of seven models score below 5%. See Table 2 for the per-category breakdown.

§Project Lead. †Advisors.

##### Abstract

AI agents may be able to automate your inbox, but can they automate other routine aspects of your life? Everyday online tasks offer a realistic yet unsolved testbed for evaluating the next generation of AI agents. To this end, we introduce CLAWBENCH, an evaluation framework of 153 simple tasks that people need to accomplish regularly in their lives and work, spanning 144 live platforms across 15 categories, from completing purchases and booking appointments to submitting job applications. These tasks require demanding capabilities beyond existing benchmarks, such as obtaining relevant information from user-provided documents, navigating multi-step workflows across diverse platforms, and write-heavy operations like filling in many detailed forms correctly. Unlike existing benchmarks that evaluate agents in offline sandboxes with static pages, CLAWBENCH operates on production websites, preserving the full complexity, dynamic nature, and challenges of real-world web interaction. A lightweight interception layer captures and blocks only the final submission request, ensuring safe evaluation without real-world side effects. Our evaluations of 7 frontier models show that both proprietary and open-source models can complete only a small portion of these tasks. For example, Claude Sonnet 4.6 achieves only 33.3%. Progress on CLAWBENCH brings us closer to AI agents that can function as reliable general-purpose assistants.

##### 1 Introduction

AI agents powered by large language models can now navigate graphical interfaces, fill forms, and execute multi-step workflows autonomously (Yao et al., 2023; Wang et al., 2024),

- as demonstrated by commercial systems such as OpenAI Operator (OpenAI, 2025) and Anthropic Computer Use (Anthropic, 2025a), and the open-source agent OpenClaw (Steinberger, 2025). However, the extent to which they can serve as truly general online assistants is unknown. To be general assistants, agents need to do more than summarize emails. They need to reliably handle the everyday online tasks that people depend on, for example booking flights, ordering groceries, and submitting job applications. These tasks are individually straightforward for a human, typically requiring under thirty minutes, yet they involve production websites with dynamic content, authentication flows, anti-bot defenses, and constantly evolving layouts.

Evaluating agents on such tasks is challenging precisely because real websites are unpredictable and consequential. To avoid safety risks, most existing benchmarksWebArena (Zhou et al., 2024), VisualWebArena (Koh et al., 2024), OSWorld (Xie et al., 2024), TheAgentCompany (Xu et al., 2025)—evaluate agents in offline sandboxes with static HTML, fixed DOM structures, no authentication, and no dynamic content (Figure 1, middle top). This controlled setting simplifies evaluation but removes the very complexities that make real-world web interaction difficult: cookie consent pop-ups, dynamic JavaScript rendering, complex and multi-step interaction. Benchmarks that do operate on real websitesWebVoyager (He et al., 2024), AssistantBench (Yoran et al., 2024), Online-Mind2Web (Xue et al., 2025), Claw-Eval (Ye et al., 2026)—are limited to read-only information retrieval or mock APIs for testing simple write operations. As a result, write-heavy task completion on live platforms—the category most directly relevant to people’s daily lives—remains unevaluated. We have no reliable picture of how well agents can actually “get things done” on the real web.

We present CLAWBENCH, a benchmark of 153 everyday online tasks spanning 15 life categories across 144 live platforms (Figure 1, left).

Rather than recreating websites in sandboxes, we let the agent operate on production sites and address the safety concern with a single targeted mechanism: a lightweight Chrome extension that records low-level browser actions, paired with a CDP-based instrumentation

- server that monitors outgoing network traffic and intercepts the final submission request—

the single HTTP call that would commit an irreversible transaction (Figure 1, middle bottom).

During each task, five layers of behavioral data are captured (Figure 3): session replay via Xvfb virtual display and FFmpeg, per-step action screenshots, HTTP traffic, agent messages (reasoning traces and tool calls), and low-level browser actions (clicks, keystrokes, scrolls). Human annotators complete every task under the same setup to produce ground-truth trajectories. Agent trajectories are then scored by comparing against these human references.

We evaluate each recorded trajectory using an Agentic Evaluator, implemented by invoking a Claude Code sub-agent under a fixed evaluation rubric. The evaluator consumes the task instruction together with the human reference actions and payloads and the agent’s executed actions and payloads, and judges whether the task was completed correctly. Instead of relying on a single final-state check, it applies predefined evaluation criteria to compare the agent trajectory against the human reference and produces a binary verdict with a structured justification grounded in the request schema and step-level evidence. As a result, the evaluation is not only outcome-based but also traceable: it reveals not just whether the agent failed, but which required fields or steps were incorrect and where the agent diverged from the reference trajectory.

###### Setup Execution

###### Evaluation

Evaluation Module

Task: Book a one-way economy-class ﬂight from JFK to LAX on Dec 31 this year for me.

Recorded Trajectory Session Replay Action Screenshots HTTP Traﬃc Agent Messages (thinking, tool-call) Browser Actions

[Figure 63]

Agentic Evaluator

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Agent

[Figure 69]

Task

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

|One way 1 Economy<br><br>New York<br><br>JFK<br><br>Los Angeles<br><br>LAX<br><br>Dec 31<br><br>All ﬁlters Stops Airlines Bags Price<br><br>|
|---|

Human

Binary Score

[Figure 75]

[Figure 76]

Veriﬁcation Conditions

###### 1 / 1

[Figure 77]

•••• DepartureArrivalClassDate(Dec(Economy)(LAX)31)(JFK)

[Figure 78]

[Figure 79]

Final step reached + all verify conditions met

Record

- Figure 3: The CLAWBENCH evaluation pipeline. Setup: a human-authored task with explicit verification conditions. Execution: the agent operates in a real browser while five layers of behavioral data are recorded. Evaluation: the recorded trajectory is scored against a human ground-truth trajectory via an Agentic Evaluator, producing a binary pass/fail verdict with step-level justification.

We evaluate 7 frontier models on CLAWBENCH: Claude Sonnet 4.6 (Anthropic, 2026), GPT-5.4 (OpenAI, 2026), Gemini 3.1 Flash Lite (Google DeepMind, 2026), Claude Haiku

- 4.5 (Anthropic, 2025b), Gemini 3 Flash (Google DeepMind, 2025), GLM-5 (Zeng et al., 2026), and Kimi K2.5 (Team et al., 2026). As shown in Figure 1 (right), Claude Sonnet 4.6 (Anthropic, 2026) and GPT-5.4 (OpenAI, 2026) score 65–75% on traditional web benchmarks (OSWorld (Xie et al., 2024), WebArena (Zhou et al., 2024)) but only 33.3% and 6.5%, respectively, on CLAWBENCH. The five-layer recording makes each failure traceable to specific steps, providing concrete diagnostic signals for future agent development. Our contributions are as follows:

- (1) We introduce CLAWBENCH, a benchmark of 153 everyday online tasks across 15 life categories and 144 live platforms, targeting write-heavy, consequential web interactions that existing benchmarks do not cover.

- (2) We design a Chrome extension and CDP-based instrumentation mechanism that records browser actions and intercepts only the final submission request, enabling safe evaluation on production websites without sacrificing ecological validity.
- (3) We develop a five-layer recording infrastructure and an agentic evaluator that provides step-level alignment against human ground truth, along with fully traceable failure diagnostics.
- (4) We open-source the complete data collection and evaluation pipeline to support community-driven benchmark maintenance and expansion.
- (5) We benchmark 7 frontier models and provide step-level failure analysis, documenting a large gap between agent performance on structured coding tasks and real-world everyday web tasks.

##### 2 Benchmark

We present CLAWBENCH, an evaluation infrastructure for measuring how well AI agents can complete everyday online tasks on the real web. Unlike prior benchmarks that retreat to sandboxes or restrict evaluation to read-only information retrieval, CLAWBENCH operates on 144 live platforms, targets write-heavy transactions (purchases, reservations, applications), and verifies outcomes by comparing the agent’s full behavioral trajectory against a human ground-truth reference. Table 1 positions CLAWBENCH among existing web-agent benchmarks across four axes: real-world environment, write-heavy task coverage, evaluation reliability, and diagnostic traceability. Figure 3 illustrates the three-stage architecture: task definition, agent execution with five-layer recording, and automated evaluation via the Agentic Evaluator.

- Table 1: Comparison of CLAWBENCH with existing web agent benchmarks across four axes: real-world environment fidelity, write-heavy task coverage, evaluation reliability, and diagnostic traceability. CLAWBENCH is the only benchmark that combines live-website execution, write-heavy state-changing tasks, human-grounded comparative evaluation, and multi-layer trajectory recording. Environment: Offline = static cached pages; Sandbox = self-hosted or VM-based replicas; Real Web = live production websites. Task Type: Read-only

= information retrieval / QA; Mixed = includes some write operations in controlled settings; Write-heavy = majority of tasks involve state-changing submissions on live platforms. Recording: None = no behavioral trace beyond the final output; Screenshot = per-step or final screenshots only; 5-Layer = session replay, action screenshots, HTTP traffic, agent messages, and browser actions. Human Traj.: whether full human ground-truth trajectories (recorded under the same infrastructure) are available for evaluation.

Benchmark Environment # Tasks # Sites Task Type Verification Recording Human Traj.

Mind2Web (Deng et al., 2023) Offline (static traces) 2,350 137 Read-only Action seq. match None Partial (action seq.) WebArena (Zhou et al., 2024) Sandbox (self-hosted) 812 5 Mixed Script-based None ✗ VisualWebArena (Koh et al., 2024) Sandbox (self-hosted) 910 3 Mixed Script-based None ✗ OSWorld (Xie et al., 2024) Sandbox (VM) 369 9 Mixed Script + screenshot Screenshot ✗ WebVoyager (He et al., 2024) Real Web 643 15 Read-only LLM-as-judge Screenshot ✗ TheAgentCompany (Xu et al., 2025) Sandbox (self-hosted) 175 6 Mixed Checkpoint-based None ✗ Online-Mind2Web (Xue et al., 2025) Real Web 300 136 Read-only Human + LLM judge Screenshot ✗ EconWebArena (Liu & Quan, 2025) Real Web 360 82 Read-only Exact numeric + URL None ✗ Claw-Eval (Ye et al., 2026) Sandbox (Docker + FastAPI) 139 15 Mixed API state check CLI logs ✗

ClawBench (Ours) Real Web 153 144 Write-heavy Agentic Evaluator 5-Layer ✓ (all tasks)

###### 2.1 Task Design and Collection

CLAWBENCH focuses on write-heavy web tasks: actions that modify server-side state through form submissions, reservations, purchases, applications, and similar state-changing operations. We target these tasks because (i) they are the category most directly relevant to people’s daily lives—each task is something an ordinary person might need to accomplish in

- Table 2: Main results on CLAWBENCH. Success rate (%) of seven AI agents on the 153-task CLAWBENCH benchmark, reported overall and for each of the 8 high-level task category groups. Models are ordered by overall success rate. Bold marks the best result in each column; underline marks the second best. † denotes a text-only model without vision capability.

Task Categories Daily Finance Work Dev Academic Travel Social Pets

Rank Model Overall

- 1 Claude Sonnet 4.6 33.3 44.2 50.0 19.0 11.1 50.0 23.1 38.9 18.2

- 2 GLM-5† 24.2 30.8 16.7 38.1 16.7 28.6 0.0 16.7 18.2

- 3 Gemini 3 Flash 19.0 15.4 33.3 23.8 22.2 28.6 30.8 11.1 0.0

- 4 Claude Haiku 4.5 18.3 15.4 33.3 19.0 27.8 21.4 7.7 16.7 18.2

- 5 GPT-5.4 6.5 9.6 0.0 0.0 11.1 7.1 7.7 0.0 9.1

- 6 Gemini 3.1 Flash Lite 3.3 1.9 0.0 0.0 5.6 14.3 0.0 0.0 9.1

- 7 Kimi K2.5 0.7 1.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0

under thirty minutes, and (ii) they produce observable HTTP payloads that enable objective verification.

Candidate generation. We construct CLAWBENCH by curating realistic everyday online tasks on live websites and retaining 153 tasks in the final benchmark. Each task is defined by three elements: a natural-language user instruction, a starting URL, and a terminal submission target specified at the HTTP-request level. Human annotators survey representative platforms across life categories, instantiate realistic user goals, and complete each task end-to-end within our recording framework. For every task, this yields a human reference trajectory together with the corresponding intercepted submission payload, which later

- serves as the basis for evaluation.

Multi-stage filtering. We apply a rigorous filtering pipeline that removes tasks requiring paid subscriptions, geographically restricted services, or websites that have gone offline.

At each stage, independent annotators verify that the task remains completable and that the ground-truth trajectory is reproducible. The final dataset comprises 153 tasks across 144 unique platforms.

Interception signal annotation. A distinguishing aspect of our data collection is that every task’s interception signal—the specific HTTP endpoint, request method, and payload schema that identifies the dangerous, inreversable submission—is manually annotated by a human expert. This annotator inspects the browser’s network traffic during the ground-truth execution, identifies the exact request that would commit the irreversible transaction, and records a declarative specification (URL pattern, HTTP method, and required payload field names). This human-in-the-loop design ensures that the ClawBench framework intercepts precisely the intended request, avoiding both false positives (blocking benign navigation requests) and false negatives (allowing the submission to reach the server). The result is a safe, targeted interception that preserves the full complexity of live-site interaction while guaranteeing zero real-world side effects—no orders are placed, no applications are submitted, and no critical server-side state is modified.

###### 2.2 Task Taxonomy

To support fine-grained analysis, we organize the 153 tasks into a two-level taxonomy (Figure 4). The top level consists of 8 high-level category groups—Daily, Work, Dev, Social, Academic, Travel, Pets, and Finance—that capture broad domains of everyday web activity. Each group is further divided into 15 fine-grained categories (e.g., “Daily Life & Shopping & Entertainment,” “Job Search & Office & Personal Management,” “Academic & Education”).

###### Benchmark Saturation

[6]Finance

DailyLife[21]

Top models are saturating existing benchmarks ClawBench remains challenging

[11]Pets

[6]Finance

PinchBench Claw-Eval

Travel [13]

[11]Pets

OSWorld-Veriﬁed WebArena-Veriﬁed

WildClawBench ClawBench

[13]Travel

Shopping [16]

Daily [52]

100

Education [9]

Tasks

88.0

Academic [14]

80

## 153

Score%(Claude-Sonnet-4.6)

77.6

72.5

Academia [5]

66.4

Entertainment[15]

60

Social [18]

Social [8]

51.1

[21]Work

40

Dev[18]

33.3

Rating [10]

[9]Office

20

Automation [3]

JobSearch[8]

Dev & Tech[15]

[4]MgmtPersonal

0

- Figure 4: Task taxonomy of CLAWBENCH. Inner ring: 8 high-level category groups; outer ring: 15 fine-grained categories. The dataset spans 153 tasks across diverse real-world domains.

Figure 5: Benchmark saturation comparison. Claude-Sonnet-4.6 performs substantially better on existing web-agent benchmarks than on CLAWBENCH, indicating that CLAWBENCH remains challenging for frontier agents.

###### 2.3 Interception Mechanism

The central design insight of CLAWBENCH is that evaluating agents on real websites does not require preventing them from interacting with real websites—it only requires intercepting the final request. We implement this via a lightweight Chrome extension and a CDP server that is loaded alongside the agent’s browser session.

How it works. The instrumentation server connects to the browser through CDP to monitor all outgoing HTTP requests against the human-annotated interception specifications (Section 2.1). When the agent’s action triggers a request matching the declared URL pattern and HTTP method, the system (i) captures the full request body, including all form fields, payloads, headers, and query parameters; (ii) blocks the request before it leaves the browser, preventing it from reaching the server; and (iii) logs the captured payload to a local file alongside a timestamp and the originating tab URL. All other requests—page loads, AJAX calls for dynamic content, image fetches, analytics pings—pass through unmodified, so the agent experiences the website exactly as a human user would.

Safety guarantees. Because interception signals are human-annotated at the endpoint level rather than inferred by heuristics, the mechanism achieves high precision: in a validation study over all 153 tasks, the extension correctly intercepted the terminal request in 100% of the human ground-truth runs with zero false positives on navigation traffic. This targeted approach avoids the risks associated with letting agents operate freely on production servers (e.g., accidentally placing orders or submitting real applications) while also avoiding the ecological validity loss inherent in sandbox-based alternatives.

###### 2.4 Five-Layer Recording Infrastructure

A distinctive feature of CLAWBENCH is that every agent run produces five synchronized layers of behavioral data, enabling both automated evaluation and deep post-hoc diagnosis.

- (1) Session Recording. The Chrome browser is running on a Xvfb virtual display which is monitored by ffmpeg, producing a full-session video recording of the browser window.
- (2) Action Screenshots. A per-step screenshot is captured immediately after each agent browser action (click, type, scroll), providing a chronological visual record of the agent’s observations and the resulting page states.
- (3) HTTP Traffic. All HTTP requests are logged via the Chrome DevTools Protocol, including request bodies, payloads and timing information. The intercepted terminal payload is a special case of this layer.
- (4) Agent Messages. The full chain of reasoning traces, tool calls, and intermediate outputs produced by the agent framework is recorded in a structured JSON format, preserving the agent’s “thought process” at each decision point.
- (5) Browser Actions. Low-level browser events—mouse clicks (with coordinates), keystrokes, scroll offsets, tab switches, and navigation events—are captured via the Chrome extension, providing a fine-grained behavioral log independent of the agent’s self-reported actions.

Human annotators produce ground-truth recordings under the same five-layer setup for every task. The parallel structure between agent and human recordings is what enables the Agentic Evaluator (Section 2.5): both trajectories are represented in the same multi-modal format, making step-level alignment and comparison feasible.

The five layers are complementary by design. Session recording and screenshots capture what the agent saw; agent messages capture what the agent thought; browser actions capture what the agent did; and HTTP traffic captures what effect the agent’s actions had on the network. Together, they enable a level of failure diagnosis that goes far beyond a binary pass/fail score: when an agent fails, a developer can trace the failure to the exact step, inspect the page state the agent observed, read the reasoning that led to the wrong decision, and compare the agent’s action against the human annotator’s action at the corresponding step.

###### Reference Trajectory

###### Prompt

Session Replay Action Screenshots HTTP Traﬃc Browser Actions

Role: Judge task success from full trajectory Key Rules:

[Figure 80]

[Figure 81]

[Figure 82]

- 1. Must use provided personal info correctly
- 2. Must complete forms and submit
- 3. Interceptor / phone veriﬁcation block: PASS if all prior actions are correct
- 4. CAPTCHA: must attempt, else FAIL

Human

###### Output

[Figure 83]

[Figure 84]

Agent Trajectory

[Figure 85]

Pass

Session Replay Action Screenshots HTTP Traﬃc Browser Actions Agent Messages

[Figure 86]

[Figure 87]

[Figure 88]

Fail

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Claude Code

Cross-reference & veriﬁcation

Agent

- Figure 6: Agentic Evaluator Inference Pipeline. The evaluator determines whether a browser agent successfully completes a real-world web task by comparing the agent trajectory against a reference trajectory across five evidence layers: session replay, action screenshots, HTTP traffic, browser actions, and agent messages. A Claude Code sub-agent applies a structured evaluation prompt that enforces key behavioral rules. The evaluator performs crossreferencing and verification over these multimodal traces to produce a final decision. PASS indicates the agent has effectively executed the task (including cases where completion is blocked but all prior actions are correct), while FAIL indicates unsuccessful execution or violation of required behaviors.

###### 2.5 Evaluation Protocol

Given the five-layer recordings from both the agent and the human annotator, we need an evaluation method that can reliably determine whether the agent completed the task correctly. We investigate three evaluator designs that operate on different subsets of the recorded data, each representing a different modality and comparison strategy. As illustrated in Figure 6, our evaluation is grounded in a multi-layer comparison between the agent trajectory and a human reference trajectory. The evaluator operates over synchronized evidence streams—including session replay, screenshots, HTTP traffic, browser actions, and agent messages—and applies a structured rubric to determine task success.

Agentic Evaluator. We evaluate each task using a single Agentic Evaluator, implemented by invoking a Claude Code sub-agent under a fixed evaluation rubric. Given the task instruction, the agent trajectory, and the human reference trajectory, the evaluator performs an explicit alignment between the two executions: it identifies corresponding steps, detects divergences, checks whether the required fields and actions are correct, and determines whether the agent reaches a terminal state equivalent to the human reference. This design leverages the full multi-layer recordings from both runs and grounds evaluation in a concrete human demonstration rather than in the task instruction alone. Figure 7 provides a schematic view of this evaluation process, where the evaluator takes both the agent and human trajectories as input and produces a binary verdict with structured justification.

Scoring. For each task t ∈ T, let q(t) denote the task instruction, Ta(t) the recorded agent trajectory, and Th(t) the recorded human reference trajectory. We denote the Agentic Evaluator by A, which maps these inputs to a binary task-level verdict:

Score(t) = A q(t), Ta(t), Th(t) , (1)

where Score(t) ∈ {0,1}, with 1 indicating successful task completion and 0 indicating failure.

The overall success rate over a task set T is then defined as

1

#### |T| ∑

SR =

Score(t), (2)

t∈T

where |T| is the number of evaluated tasks.

Unlike evaluators that judge an agent trajectory in isolation, A performs an explicit comparison between the agent trajectory and the human reference trajectory under a fixed evaluation rubric. This comparative signal provides a concrete specification of successful task completion, including platform-specific details such as field bindings, interaction order, and terminal submission structure, which may be difficult to infer reliably from the task instruction alone.

##### 3 Experiments

We evaluate 7 frontier AI models on CLAWBENCH, spanning both proprietary and opensource systems. This section describes the experimental setup, presents the main results, analyzes failure modes, and reports ablation studies on the observation modality.

###### 3.1 Experimental Setup

Models. We evaluate 5 proprietary models (Claude Sonnet 4.6 (Anthropic, 2026), GPT-

- 5.4 (OpenAI, 2026), Gemini 3.1 Flash Lite (Google DeepMind, 2026), Claude Haiku 4.5 (Anthropic, 2025b), and Gemini 3 Flash (Google DeepMind, 2025)) and 2 open-source models (GLM-5 (Zeng et al., 2026) and Kimi K2.5 (Team et al., 2026)).

###### Input Evaluator Output (0 / 1)

Agent Input

Agentic Evaluator

Justiﬁcation (from schema)

Task Instruction

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

“departure = Toronto arrival = New York dep_date = Aug 01 return_date = Aug 07 type = direct flight ”

[Figure 97]

[Figure 98]

Human Actions

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Human Action JSON payloads

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Claude Code subagent invoked under a ﬁxed evaluation rubric

Agent Actions

Each schema ﬁeld checked against request payload

Agent Action JSON payloads

[Figure 111]

[Figure 112]

- Figure 7: Evaluation protocol. The evaluator takes as input the task instruction together with the human reference actions and payloads and the agent’s executed actions and payloads. It then invokes a Claude Code sub-agent under a fixed evaluation rubric to determine whether the task was completed correctly, producing a binary verdict and a structured justification grounded in schema-level checks over the request payload.

Infrastructure. Each model is given control of a Chromium browser instance via the OpenClaw agent framework. The OpenClaw agent is connected to a managed isolated browser. It then can use the OpenClaw framework’s browser toolset to interact with the browser (click, type, scroll, navigate, etc.). The CLAWBENCH Chrome extension and CDP instrumentation server runs in the background throughout the session, intercepting HTTP requests and agent actions for post-hoc evaluation.

Reproducibility controls. Each benchmark run uses an encapsulated container, ensuring environment isolation and cross-run consistency. Chrome is launched with flags that disable UI prompts, sync, and irrelevant extensions to minimize environmental variance.

Metrics. Our primary metric is success rate (SR): the percentage of tasks for which the agent receives a binary score of 1 (Section 2). We report SR both overall and broken down by the 8 high-level category groups.

- 3.2 Main Results

Table 2 reveals three main findings. First, Claude Sonnet 4.6 is the strongest model overall, achieving 33.3% success rate, followed by GLM-5 at 24.2%. This gap suggests that frontier models differ substantially in their ability to handle realistic everyday web workflows.

Second, model performance varies considerably across task categories. Claude Sonnet 4.6 leads on Daily, Finance, Academic, and Social, while GLM-5 performs best on Work, Gemini

- 3 Flash on Travel, and Claude Haiku 4.5 on Dev. These patterns indicate that current agents do not yet exhibit uniform competence across domains, but instead show category-specific strengths and weaknesses. At the same time, even the best category-level results remain far from saturation, underscoring the overall difficulty of CLAWBENCH.
- 4 Related Work

Web Agent Benchmarks. Early web agent benchmarks such as MiniWoB (Shi et al., 2017) evaluated agents on simplified, synthetic web interfaces with short action sequences. WebArena (Zhou et al., 2024) introduced self-hosted, realistic web environments with 812 tasks across 5 domains, using URL and element matching for evaluation. VisualWebArena (Koh et al., 2024) extended this to visually grounded tasks on 3 self-hosted sites. Mind2Web (Deng et al., 2023) scaled to 2,350 tasks on 137 real-world domains but evaluated action sequences rather than end-to-end task completion. OSWorld (Xie et al., 2024) broadened

the scope to full operating system tasks across 9 applications. More recently, REAL Bench (Garg et al., 2025) evaluated agents on live websites but relied on manual rating for scoring. CLAWBENCH differs from all prior work by (i) operating on 144 live platforms rather than self-hosted sandboxes, (ii) focusing on write-heavy, state-changing tasks, and (iii) providing traceable, comparative evaluation against human reference trajectories through an agentic evaluator.

LLM-Based Web Agents. The emergence of large language models has driven rapid progress in autonomous web agents. Systems such as WebGPT (Nakano et al., 2021), WebAgent (Gur et al., 2023), and SeeAct (Zheng et al., 2024) demonstrated that LLMs can interpret web pages and execute multi-step browsing tasks when given appropriate observation and action interfaces. Recent approaches combine visual perception (screenshots) with structured page representations (accessibility trees, HTML) to improve grounding accuracy. Agent frameworks including AgentGPT, AutoGPT, and OpenClaw provide standardized interfaces for deploying LLMs as web agents with tool use and action execution capabilities. CLAWBENCH is designed to evaluate any agent system that can control a Chromium browser, independent of the underlying model or framework.

Evaluation Methods for Agent Systems. Evaluating autonomous agents remains challenging due to the diversity of possible action trajectories and the difficulty of defining success criteria. Prior work has used action sequence matching, URL-based success detection, screenshot comparison, and human judgement. Action-level metrics suffer from the problem of multiple valid paths: an agent may complete a task correctly through a different sequence of actions than the reference trajectory. Screenshot-based methods require visual similarity thresholds that introduce non-determinism. Human evaluation, while flexible, is expensive and non-reproducible. CLAWBENCH sidesteps these issues by combining intercepted submission payloads with an agentic evaluator that performs explicit step-level alignment between the agent trajectory and a human reference trajectory, producing a binary verdict together with a structured justification grounded in the recorded evidence.

Concurrent and Complementary Work. Several recent benchmarks address related but distinct aspects of web agent evaluation. TheAgentCompany (Xu et al., 2025) provides a self-hosted sandbox simulating a software company with 175 tasks and checkpoint-based partial credit; CLAWBENCH trades environmental control for real-world breadth across 144 live platforms. EconWebArena (Liu & Quan, 2025) is a live-web benchmark for economic research tasks featuring 360 read-only tasks with exact numeric matching and URL provenance; CLAWBENCH extends the live-web paradigm to write-heavy, state-changing tasks. MCP-Bench (Wang et al., 2025) evaluates LLM agents on tool invocation via the Model Context Protocol with strict schema validation; CLAWBENCH targets browser-based web interaction rather than structured API calls. TrickyArena (Ersoy et al., 2025) studies dark pattern susceptibility in web agents across 4 controlled applications—an orthogonal safety concern that highlights the importance of evaluating on real websites where dark patterns occur naturally. AssistantBench (Yoran et al., 2024) defines 214 realistic open-web tasks with automated evaluation, focusing on information retrieval; CLAWBENCH complements this with write-heavy tasks. WebCanvas (Pan et al., 2024) proposes key-node evaluation for 542 tasks on dynamic websites in a similar live-web setting but without HTTP payload verification. Taken together, these efforts illustrate a fundamental realism-vs-reproducibility trade-off: sandboxed benchmarks offer perfect reproducibility but may not reflect the complexity of real websites, while live-web benchmarks expose agents to authentic challenges

- at the cost of environmental variability. CLAWBENCH deliberately chooses realism and mitigates reproducibility concerns through human-grounded comparative evaluation and full multi-layer trajectory recording.

##### 5 Conclusion

We introduce CLAWBENCH, a benchmark of 153 real-world everyday web tasks spanning 144 live platforms across 8 high-level category groups. By evaluating agents on live production websites and focusing on write-heavy, state-changing workflows, CLAWBENCH

provides a substantially more realistic testbed than prior benchmarks built on static pages or sandboxed environments. Our framework combines final-request interception, five-layer trajectory recording, and an agentic evaluator. Experiments on 7 frontier models show that strong performance on existing web-agent benchmarks does not transfer to CLAWBENCH, underscoring the gap between controlled benchmark success and real-world everyday web competence. We release the benchmark, evaluation toolkit, and supporting infrastructure to support future research on realistic web-agent evaluation.

##### References

Anthropic. Introducing computer use. https://www.anthropic.com/news/ 3-5-models-and-computer-use, 2025a. Accessed: 2026-03-20.

Anthropic. Claude haiku 4.5. https://www.anthropic.com/news/claude-haiku-4-5, 2025b. Anthropic. Claude sonnet 4.6. https://www.anthropic.com/news/claude-sonnet-4-6,

2026.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Devin Ersoy, Brandon Lee, Ananth Shreekumar, Arjun Arunasalam, Muhammad Ibrahim, Antonio Bianchi, and Z Berkay Celik. Investigating the impact of dark patterns on llm-based web agents. arXiv preprint arXiv:2510.18113, 2025.

Divyansh Garg, Shaun VanWeelden, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Tomas Abraham, Michael Lara, Federico Lopez, et al. Real: Benchmarking autonomous agents on deterministic simulations of real websites. arXiv preprint arXiv:2504.11543, 2025.

Google DeepMind. Gemini 3 flash. https://deepmind.google/models/gemini/flash/, 2025. Google DeepMind. Gemini 3.1 flash-lite. https://deepmind.google/models/gemini/

flash-lite/, 2026.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Saber, Yutaka Matsuo, Douglas Eck, and Aleksandra Fishi. A real-world WebAgent with planning, long context understanding, and program synthesis. arXiv preprint, 2023. arXiv:2307.12856.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Zefang Liu and Yinzhu Quan. Econwebarena: Benchmarking autonomous agents on economic tasks in realistic web environments. arXiv preprint arXiv:2506.08136, 2025.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint, 2021. arXiv:2112.09332.

OpenAI. Introducing operator. https://openai.com/index/introducing-operator/, 2025. OpenAI. Gpt-5.4. https://platform.openai.com/docs/models/gpt-5.4, 2026.

Yichen Pan, Dehan Kong, Sida Zhou, Cheng Cui, Yifei Leng, Bing Jiang, Hangyu Liu, Yanyi Shang, Shuyan Zhou, Tongshuang Wu, et al. Webcanvas: Benchmarking web agents in online environments. arXiv preprint arXiv:2406.12373, 2024.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In Proceedings of the 34th International Conference on Machine Learning, pp. 3135–3144. PMLR, 2017.

Peter Steinberger. Openclaw: Your own personal AI assistant. https://github.com/ openclaw/openclaw, 2025. Accessed: 2026-03-20.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, et al. Mcp-bench: Benchmarking tool-using llm agents with complex real-world tasks via mcp servers. arXiv preprint arXiv:2508.20453, 2025.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Frank F Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, et al. Theagentcompany: Benchmarking LLM agents on consequential real world tasks. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025.

Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. In Conference on Language Modeling (COLM), 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Bowen Ye, Rang Li, Qibin Yang, and Lei Li. Claw-eval: A transparent benchmark for real-world agents. https://github.com/claw-eval/claw-eval, 2026. Peking University & University of Hong Kong. Accessed: 2026-03-20.

Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, and Jonathan Berant. Assistantbench: Can web agents solve realistic and time-consuming tasks? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8938–8968, 2024.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. SeeAct: GPT-4V(ision) as a generalist web agent, if grounded. arXiv preprint, 2024. arXiv:2401.01614.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations (ICLR), 2024.

