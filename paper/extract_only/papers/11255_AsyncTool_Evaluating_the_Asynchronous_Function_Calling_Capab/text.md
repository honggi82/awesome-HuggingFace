# arXiv:2605.27995v2[cs.AI]28May2026

## AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios

Kou Shi1,†, Ziao Zhang1,†, Shiting Huang1, Avery Nie2, Zhen Fang1, Qiuchen Wang1, Lin Chen1, Huaian Chen1, Zehui Chen1, Feng Zhao1,* 1University of Science and Technology of China 2University of Toronto †Equal contribution. *Corresponding author.

Abstract

Large language model (LLM)-based agents have demonstrated strong capabilities in leveraging external tools to solve complex tasks. However, existing evaluations largely overlook the temporal dimension of tool invocation, particularly the impact of tool response latency, and are typically limited to single-task settings. In real-world applications, multiple tasks often need to be executed concurrently, and overall efficiency critically depends on whether an agent can utilize idle time while waiting for tool responses. We refer to this capability as asynchronous tool calling. To evaluate this capability, we propose AsyncTool, a benchmark for assessing LLM-based agents in interactive multi-task tool-use environments with delayed tool feedback. AsyncTool presents multiple heterogeneous tasks simultaneously and simulates realistic tool response latency during execution. Using a hybrid data evolution strategy, we construct a diverse asynchronous multitasking dataset covering multiple scenarios and tool-use patterns. We evaluate models at three levels—step, sub-task, and task—and further introduce efficiency-oriented metrics to measure task coordination and completion efficiency. Extensive experiments show that delayed tool feedback poses substantial challenges to current agents, leading to clear performance degradation. Models that better coordinate task switching and dependency tracking tend to achieve stronger performance on AsyncTool.Our analysis identifies the main failure modes of current tool agents and provides practical guidelines for designing future systems with stronger temporal reasoning and coordination capabilities.The code is available at https://github.com/StoKou/repo-asynctool

### 1 Introduction

Recent advances in large language models (LLMs) have significantly improved their ability to follow instructions and understand context, leading to

increasingly capable LLM-based agents for tool use(OpenAI, 2025b; Comanici et al., 2025; Anthropic, 2025; Yang et al., 2025; Team et al., 2025; Zeng et al., 2025; Chen et al., 2025a; Wang et al., 2025; Chen et al., 2024b; Huang et al., 2026; Zhang et al., 2026). This capability enables them to handle more sophisticated, multi-step tasks that require external information or actions, and to achieve strong performance across diverse tool-use scenarios (Liu et al., 2023; Li et al., 2025; Chan et al., 2024).

However, real-world environments are often more complex, frequently requiring the concurrent execution of multiple tasks that may involve different tools. In practical settings, function calls usually incur latency, and executing tasks sequentially in a synchronous manner fails to fully utilize idle waiting time, thereby reducing overall efficiency. To better evaluate and enhance the agent’s performance under such conditions, we introduce the concept of Asynchronous Tool Call into the interaction between the agent and the environment, where the agent should utilize these idle intervals to advance other available tasks. Motivated by these gaps, we identify three critical observations: (i) Inadequate evaluation of the agent’s capability to complete multiple tasks in asynchronous scenarios. Existing studies are typically restricted to single-task scenarios in which tools operate in an immediate response manner (Zhuang et al., 2023; Ruan et al., 2023; Xu et al., 2023; Guo et al., 2024; Qin et al., 2023; Ye et al., 2024), overlooking the evaluation for multiple tasks in asynchronous scenarios. (ii) Lack of alignment with real-world conditions in interactive environments involving real-time tool calls. Existing asynchronous planning benchmarks do not operate within interactive environments, which is inconsistent with real-world scenarios involving real-time tool calls (Lin et al., 2024). (iii) Insufficient metrics and standardized protocols specific to concurrent tasks with delayed and out-of-order tool feed-

back. Traditional benchmarks involving time delays do not cover tool-using tasks and cannot be transferred to agentic tasks (Zhang et al., 2024a; Gonzalez-Pumariega et al., 2025).

To bridge these gaps, we propose ASYNCTOOL, a benchmark for evaluating the ability of LLMbased agents to perform asynchronous tool calling in interactive multi-task scenarios. To our knowledge, ASYNCTOOLis the first benchmark that jointly considers delayed tool feedback, concurrent multi-task execution, multi-step function calling, and dependency-aware task coordination. (i) Our benchmark consists of combinations of multiple tasks, where each task contains intra-task step dependencies and different tasks can be pursued concurrently. This design allows an agent to use the waiting periods caused by tool latency to advance other independent tasks. (ii) To better approximate real-world tool-use conditions, we simulate tool-specific response latencies, integrate multiple tasks into a shared interaction process, and require the agent to make progress on them through asynchronous function calls. This setting provides a practical environment for assessing whether agents can coordinate multiple tasks under delayed and potentially out-of-order tool feedback. Table 2 compares ASYNCTOOLwith existing benchmarks on tool calling and asynchronous execution. (iii) To comprehensively evaluate asynchronous tool-use capabilities, we assess model performance at three levels: Step Level, Sub-Task Level, and Task Level, covering fine-grained tool-call correctness, intermediate subtask completion, and end-to-end multitask success. In addition, we introduce efficiencyoriented metrics to measure task-interleaving behavior and completion efficiency under tool latency. Through extensive experiments, we find that delayed tool feedback poses substantial challenges to current LLM-based agents, especially in maintaining task states before results arrive. Compared with synchronous or immediate-response settings, asynchronous execution leads to clear performance degradation, especially when models prematurely continue a task before its dependent tool result has returned. Our analysis further shows that effective asynchronous tool use requires more than frequent task switching: models must coordinate task switching with dependency tracking and state maintenance. Stronger models are better able to utilize idle waiting periods to advance other tasks while resuming pending tasks at the appropriate time, whereas weaker models often suffer from

dependency violations, task neglect, and tool confusion. These findings highlight the importance of temporal coordination for future tool-using agents.

The main contributions of our work are summarized as follows:

- • We propose ASYNCTOOL, a benchmark for evaluating asynchronous tool calling in interactive multi-task environments with delayed tool feedback.
- • We construct a diverse asynchronous multitasking dataset by composing validated singletask tool-use trajectories through a hybrid data-evolution strategy. The resulting tasks cover different task numbers, task types, scenarios, and dependency structures.
- • We design a multi-level evaluation protocol that assesses model performance at the step, sub-task, and task levels, capturing both finegrained tool-call correctness and end-to-end task completion.
- • We introduce efficiency-oriented metrics to analyze task interleaving and completion behavior under tool latency, and conduct extensive experiments to reveal the challenges current LLM agents face in temporal coordination and dependency tracking.

### 2 ASYNCTOOL

Building on the motivation introduced in Section 1, ASYNCTOOL simulates tool response latency and evaluates asynchronous tool calling in multi-task settings. This section first formalizes the interaction paradigm in ASYNCTOOL, where agents must coordinate multiple tasks under delayed tool feedback. We then describe the asynchronous multitasking dataset and evaluation protocol.

2.1 Agent as a Concurrent Tool-Using System While agents’ ability to solve problems through tool use is well established, their practical effectiveness can be limited by the non-negligible response latencies of tool calls in real-world scenarios.

This raises an important question: can agents use idle time from pending tool calls to work on other tasks? ASYNCTOOL studies this by simulating delayed tool feedback and concurrent execution. Unlike standard tool-use settings, tool results in ASYNCTOOL are returned with delays. After making a tool call, the agent must decide whether to

###### STEP 4: multi-task combination

###### STEP 3: Fine-grained Annotation

###### STEP 1: Data Collection

###### STEP 2: Coarse Reconstruction

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Combiner

AI Human

Benchmarks

Step-dependency Enhancement

Trajectory Validation

Mixed-strategy Evolution

Validation

Ground-truth Path Generation

Correction & Disambiguation

Classification

Filter

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Origin Dataset

Preliminary Dataset Single-Task Dataset

Multi-Task Dataset

Step-dependency Enhancement

Mixed-strategy Evolution

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

By num

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

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Single Tasks By type

[Figure 33]

[Figure 34]

[Figure 35]

dependencies

- Figure 1: Overview of the dataset construction process. The pipeline starts by collecting raw data from tool-use benchmarks, categorizing it by scenario. Task-step dependencies are reinforced, and execution trajectories are reconstructed using Gemini 2.5 Pro, followed by manual verification for accuracy and determinism. Finally, data evolution occurs through hybrid strategies, with filtering producing the final multitasking dataset.

wait or switch to another task. This makes delayed feedback, task interleaving, and dependency-aware scheduling key challenges in ASYNCTOOL.

For example, consider a scenario where the agent receives two independent tasks, denoted as task1 and task2, whose required function-call sequences are ⟨f1,f3,f5⟩ and ⟨f2,f4⟩, respectively. Although the two tasks are mutually independent, the function calls within each task must follow the specified order due to intra-task dependencies. In this setting, the agent acts as the Assistant, while the execution system serves as the Environment. The Assistant first attempts to solve task1 by calling f1. After receiving the formatted tool-call request, the Environment informs the Assistant that the result of f1 is not yet available, since tool execution is non-instantaneous. The Assistant can then switch to task2 and issue the call f2, which also incurs its own latency. When the result of f1 becomes available, the Assistant can resume task1 and continue with the next dependent call. This process continues until all tasks are completed.

Figure 2 illustrates this interaction process. Rather than evaluating tool use as a purely sequential procedure, ASYNCTOOL evaluates whether an LLM can act as a coordinator that schedules tool calls across multiple pending tasks. A capable agent should not only invoke the correct tools with

valid arguments, but also track task states, respect intra-task dependencies, and determine when to switch between tasks under delayed feedback. Consequently, ASYNCTOOL provides a testbed for evaluating temporal coordination and asynchronous task management in tool-using agents.

#### 2.2 Data Construction

The construction of ASYNCTOOLrequires a highquality multi-task dataset. To this end, we design a data construction pipeline consisting of four main stages: Data Collection (§ 2.2.1), Coarse Reconstruction (§ 2.2.2), Fine-Grained Annotation (§ 2.2.3), and Multi-Task Composition (§ 2.2.4). An overview of the dataset construction process is shown in Figure 1.

#### 2.2.1 Data Collection

Existing benchmarks have already collected tool APIs derived from real-world scenarios and provide well-developed tool executors, task descriptions, and execution paths. To avoid reinventing the wheel, we leverage these resources as high-quality sources of single-task data. Specifically, we select two representative benchmarks, NESTFUL (Basu et al., 2024) and BFCLv3 (Yan et al., 2024).

After automated verification, we categorize and organize the tools and tasks from these benchmarks,

ensuring that each task is uniquely associated with a specific tool category. Through this process, we extract a total of 12 tools and 358 tasks, each paired with its corresponding tool-call path, to form the Original Dataset.

#### 2.2.2 Coarse Reconstruction

To ensure reliable evaluation, we first generate ground-truth tool-call trajectories and verify them at both the trajectory and final-environment levels. To reduce manual annotation cost, we then use Gemini 2.5 Pro (Comanici et al., 2025) for coarse reconstruction.

Specifically, given the original task description, the multi-step execution trajectory, and the tool set (Hsieh et al., 2023), Gemini 2.5 Pro is prompted to reconstruct task descriptions and produce strictly ordered function-call trajectories that align with the reconstructed tasks. We process instances in batches and use carefully designed few-shot prompts to improve the consistency and reliability of the generated data. The detailed prompt is provided in Appendix D.5.

Although most reconstructed instances satisfy our requirements, some errors remain, including incorrect function arguments and mismatches between task descriptions and execution orders. We manually inspect and correct these cases to ensure the quality of the final benchmark data.

#### 2.2.3 Fine-grained Annotation

After refinement with Gemini 2.5 Pro, we obtain a Preliminary Dataset, which may still contain potential issues, including errors that could invalidate an entire tool-call trajectory. To address these quality concerns, we design a fine-grained human annotation pipeline to identify and correct subtle errors and logical inconsistencies introduced during model-based generation.

Trajectory Validation. We first ensure that every function call in a trajectory is valid. To this end, we manually verify the sequential execution results for each task trajectory. Through this process, we identified three recurring error patterns and applied targeted corrections: (1) misinterpretation of the initial task conditions, leading to errors in the first call, such as repeatedly executing cd to enter the current directory in file system tasks; (2) violation of dependency relations, i.e., failing to invoke prerequisite functions, for example skipping a preceding call that is tied to the current one; and (3) misunderstanding of tool functionalities,

often manifested as providing function arguments in unsupported or invalid formats.

Correction and Disambiguation. Once the validation step confirms that all trajectories are free of execution errors, our focus shifts to aligning tasks with their corresponding trajectories and eliminating ambiguities. First, we verify the consistency between each task and its trajectory, removing task descriptions or partial trajectories that cannot be matched. Second, we strictly enforce the order of function calls within each trajectory, correcting any incorrect sequences. Finally, we replace ambiguous descriptions with precise expressions wherever possible, ensuring that essential details (e.g., location, time, and other key arguments) are explicitly included in the task description.

Following this pipeline, we conduct multiple rounds of verification on 358 tasks until no errors remain, ultimately producing a high-quality SingleTask Dataset comprising 358 validated instances, as summarized in Table 4.

#### 2.2.4 Multi-task Composition

To evaluate agents in realistic multitask scenarios, we consider two factors: task quantity and task type. Task quantity includes dual-task and tri-task settings, while task type includes within-class and cross-class combinations. Combining these factors yields four multitask configurations, which are applied to the single-task dataset.

Since exhaustive combination would produce too many samples, we use weighted random sampling to construct a fixed-size subset. The final Multitasking Dataset contains 712 instances, covering diverse and complex multitasking scenarios.

#### 2.3 Evaluation

In ASYNCTOOL, each task is defined as a set of n subtasks {S1,S2,...,Sn}. Each subtask Si is represented as a tuple (Ii,Qi,Ti,Ei), where Ii is a unique identifier, Qi denotes the task query, Ti specifies the list of available APIs, and Ei denotes the hidden environment state associated with the subtask, which is not directly exposed to the assistant. The model’s response must explicitly include Ii to indicate which subtask is being executed.

For each subtask, we extract its execution trajectory Ti, defined as an ordered sequence of tool calls:

Ti = ⟨a1,a2,...,ak⟩,

where each action aj is represented as a tuple (tool,args). Once all subtasks are completed,

User Trajectory Executor

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

Submit initial Task 1:

[Figure 36]

get_symbol_by_name ( name = "Alpha Tech " )

[Figure 37]

[Figure 38]

###### Task 1

𝑓1

Please add Alpha Tech to my watchlist, and then

[Figure 39]

[Figure 40]

get_user_tickets ( )

deposit $500 into my

[Figure 41]

{ "symbol": "ALPH " }

account.

𝑓2

[Figure 42]

When Task 1 hasn’t

[Figure 43]

add_to_watchlist ( stock = "ALPH " )

been completed:

[Figure 44]

{ "id": 10, "title": "Invoice inquiry", "created_by": … }

Time

[Figure 45]

###### Task 2

𝑓3

I need to review my open

[Figure 46]

[Figure 47]

edit_ticket ( ticket_id = 11, updates = {"priority": "3"})

requests. Please list all of

my support tickets and locate the one titled "Password Reset Failure" (I do not remember its ticket ID). Once you find it, update its priority to 3, as the issue has become more urgent.

{"symbol": ["MSFT", "ALPH"]}

[Figure 48]

𝑓4

fund_account ( amount = "500.0 " )

[Figure 49]

[Figure 50]

[Figure 51]

{ "status": "Ticket has been updated successfully. . . " }

𝑓5

[Figure 52]

{ "status": " Account has been funded successfully… " }

ALL COMPLETED

- Figure 2: An example of asynchronous multi-task tool use in ASYNCTOOL. The agent receives two tasks simultaneously and interleaves their dependency-constrained tool-call trajectories under delayed executor feedback. When a tool result is pending, the agent can switch to another independent task; when the result becomes available, it resumes the corresponding task. The example highlights task interleaving, dependency tracking, and temporal coordination.

accuracy, which measures whether a subtask is completed both procedurally and environmentally. Detailed calculation procedures are provided in Appendix B.1.

we obtain the set of trajectories {T1,T2,...,Tn}, which is then used to evaluate whether the model has successfully completed the overall task.

In asynchronous multi-task execution, interactions between the assistant and external tools can become highly complex. To provide a comprehensive evaluation of the assistant’s performance under such conditions, we assess the results at three levels: Step Level, Sub-task Level, and Task Level.

Task Level. At the task level, we evaluate whether the agent successfully completes the entire task. The trajectory-completion and environmentconsistency metrics at this level are counted as correct only when all corresponding subtask-level metrics within the task are satisfied. These metrics provide an overall assessment of the agent’s ability to coordinate and complete multiple subtasks. The final task accuracy is defined as the proportion of tasks for which both task-level trajectory completion and environment consistency are achieved.

Step Level. Following the fine-grained evaluation methodology of Patil et al. (2023), we assess the agent’s fundamental tool-calling capability, focusing on call format, tool selection, and parameter correctness. To quantify these aspects, we follow Basu et al. (2024) and compute F1 scores separately for tool accuracy and parameter accuracy.

### 3 Experiment

Sub-task Level. At this level, we define accuracy-based metrics to evaluate the agent’s performance on individual subtasks. For each sub-

#### 3.1 Experimental Setup

We evaluate 19 models on ASYNCTOOL, aiming to provide a comprehensive benchmark for assessing their capability of asynchronous tool calling under multi-task scenarios. Specifically, for closedsource models, we select four prominent models: Qwen-max (Team, 2024c) created by the Qwen Team, Kimi k2 (Team et al., 2025) by Kimi Team, Gemini 2.5 Pro (Comanici et al., 2025) developed by Google, alongside GPT-4.1 (Achiam et al., 2023), GPT-4o (Hurst et al., 2024), and

task, we compare the predicted trajectory Tipred with the ground-truth trajectory Tigt to determine whether the subtask is successfully completed, yielding the trajectory-completion metric. In addition, we compare the predicted hidden state Eipred with the ground-truth hidden state Eigt to measure environment consistency, yielding the environment-matching metric. These two metrics are further combined into the overall subtask

Table 1: Main Results of ASYNCTOOL. Func. and Param. mean matching the model’s results with the ground truth to calculate the F1 score. Char. means path matching while Env. means matching multiple execution result environments. Bold indicates best overall performance, while underline denotes the best within the same group.

Step Level Sub-Task Level Task Level

Models

Overall Func. Param. Char. Env. Acc. Char. Env.

Closed-Sourced Models

Qwen-Max 86.22 73.62 53.45 72.98 52.44 26.97 50.14 25.56 Kimi-K2 96.14 80.46 59.61 76.06 56.79 28.23 51.69 24.44 Gemini 2.5 Pro 89.08 78.27 63.16 76.43 62.05 34.13 54.35 32.44 GPT-5 92.21 80.11 62.13 78.34 60.67 33.15 58.43 31.32

- GPT-4o 93.92 82.26 62.90 80.04 61.41 33.71 60.53 31.74 GPT-4.1 96.22 84.08 67.89 82.91 67.14 39.04 64.89 38.06

Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 78.29 43.69 12.95 41.08 12.47 1.26 14.61 1.26 GLM-4-9B-chat 74.72 59.70 29.67 55.94 28.93 7.30 25.56 6.88 Qwen2.5-7B-Ins. 82.40 65.01 28.66 55.15 26.38 6.74 25.84 6.04 Qwen2.5-14B-Ins. 81.32 70.22 50.21 65.29 46.28 22.61 38.20 18.82 Qwen3-8B 63.05 53.61 29.88 53.08 29.30 11.10 28.65 10.67 Qwen3-14B 85.02 72.67 49.62 69.27 47.19 20.51 44.66 18.82

Open-Sourced Models > 20B

LLaMA-3.1-70B-Ins. 89.60 47.10 18.47 43.26 17.83 3.37 16.43 2.81 LLaMA-3.3-70B-Ins. 73.00 40.32 21.92 44.37 20.54 6.18 18.26 5.34 GLM-4-32B 60.59 51.41 35.46 56.21 33.97 16.57 29.78 15.17 Qwen3-32B 79.95 70.37 48.14 67.46 46.71 20.22 41.43 19.10 Qwen2.5-32B-Ins. 94.24 81.73 61.41 74.42 56.48 30.62 49.72 24.86 Qwen3-30B-A3B-Ins. 94.29 80.03 54.14 72.72 53.03 22.33 47.47 21.49 DeepSeek-V3.1-Terminus 86.10 75.32 59.39 72.45 56.21 33.43 49.30 28.93

- GPT-5 (OpenAI, 2025a) by OpenAI. For opensource LLMs, we evaluate numerous models including LLaMA3.1 (AI@Meta, 2024), LLaMA3.3, Qwen2.5 (Team, 2024a,b), Qwen3 (Yang et al., 2025), GLM4 (GLM et al., 2024), DeepSeek (Liu et al., 2024).

bilities of these models. In the sub-task evaluation, the models’ scores are nearly double those of the overload models.

Furthermore, as shown in Appendix 12, the average number of dialogue turns for closed-source models was significantly lower than that of opensource models, which also demonstrates that more powerful models are more efficient in the same environment.

#### 3.2 Results on ASYNCTOOL

We conducted a comprehensive empirical evaluation across a wide spectrum of current mainstream models to assess their capabilities. Based on these findings, our analysis is structured around three key questions.

Q2: How do Accuracy and Efficiency Trade off in Asynchronous Multi-Task Tool Use?

Same-task Streak measures the longest consecutive sequence of turns in which the model continues working on the same task, averaged across all samples. A lower value suggests stronger interleaving ability in asynchronous multi-task execution.

Q1: Which Model is Better in Completing Multiple Tasks Asynchronously?

As shown in Table 1, GPT-4.1 demonstrates the strongest performance in asynchronous capability under asynchronous multitasking evaluation, achieving a score of 38.06. Close behind, the large open-source model DeepSeek-V3.1-Terminus achieves performance highly comparable to that of closed-source models, highlighting its strong competitive capability.

Figure 4 shows that accuracy and efficiency are not strictly aligned in asynchronous multi-task tool use. The ideal model should appear in the lowerright region, achieving a high Overall score while maintaining a low Same-task Streak, meaning that it can complete tasks accurately and interleave different subtasks efficiently. Closed-source models generally occupy this favorable region: GPT4.1 achieves the highest Overall score while keeping a relatively low Same-task Streak, indicating that it balances task correctness and asynchronous

In the step-level evaluation, closed-source models consistently achieve high scores, while opensource models exhibit notable discrepancies. This highlights the differences in the asynchronous capa-

GPT-4.1 GPT-4o

Gemini2.5-Pro Qwen2.5-7B-Ins.

LLaMA-3.1-8B-Ins. GLM-4-9B-chat

Qwen3-8B Qwen3-32B.

LLaMA-3.3-70B-Ins. LLaMA-3.1-70B-Ins.

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

69.6

68.868.0

70

67.1

61.462.0

60

56.8

50

46.7

44.6

42.6

41.0

39.0

39.0

Score

40

36.5

36.2

32.6

31.732.4

32.3

28.929.3

30

26.4

26.2

21.9

21.3

20.5

19.1

20

17.8

15.9

13.1

12.9

12.5

10.7

10

6.9

6.0

5.3

5.3

4.5

2.8

1.3

0

Sync TC

Sync SC

Async TC

Async SC

- Figure 3: Scores of some models. TC is overall Score, and SC is Sub-Task Level Score. As the model size reduces, the model score correspondingly declines. A higher subtask completion rate does not invariably result in a higher overall score.

scheduling most effectively. Gemini 2.5 Pro and GPT-4o also show strong accuracy with compact task switching, suggesting that high-performing models can use idle waiting time to make progress on other tasks rather than repeatedly staying on the same task.

However, Figure 4 also reveals a clear tradeoff. Some models obtain relatively low Same-task Streak values but still have limited Overall scores, such as smaller open-source models. This suggests that frequent switching alone does not guarantee successful asynchronous execution: models must also correctly track task states, dependencies, and tool outputs. Conversely, models such as DeepSeek-V3.1 achieve competitive Overall scores but with a higher Same-task Streak, indicating stronger task-solving ability but less compact interleaving behavior. Therefore, asynchronous tool-use performance depends on both dimensions: accuracy determines whether the model can complete the required tool trajectories, while efficiency reflects whether it can schedule multiple tasks without wasting waiting time. Overall, Figure 3 suggests that the best models are not simply those that switch most often, but those that switch at the right time while preserving task correctness.

Q3: What Challenges do LLMs Encounter in ASYNCTOOL?

In the ASYNCTOOL evaluation, our analysis of tool-call trajectories across various tasks reveals a critical gap in temporal reasoning. Models with lower performance often exhibit a lack of temporal reasoning ability, executing the next function call immediately without waiting for the tool’s response to the current task. This behavior is particu-

larly problematic when dependencies exist between the two calls, often leading to unforeseen errors. In contrast, higher-performing models are able to identify such dependencies, execute tasks sequentially, and leverage idle time to advance other tasks, resulting in a substantial performance gap.

We also observe that agents occasionally fail to complete certain tasks. Specifically, some models tend to execute the most recently presented task first, neglecting earlier tasks. This issue is more common in smaller models and is almost absent in larger 70B-scale models. Moreover, it occurs more frequently in tri-task combinations than in dual-task combinations, which is consistent with intuition. Another notable error is tool misidentification, in which models misjudge the tool use across tasks, for example, invoking a flight-booking tool when they should continue a data-processing task. This type of error typically leads to cascading failures, as the model often struggles to selfcorrect once the confusion occurs. Common errors, by contrast, include non-compliant instructions, erroneous tool-call trajectories, and parameter errors, all of which directly prevent tasks from being scored successfully. Such errors still constitute a significant proportion of failures in smaller models within ASYNCTOOL, but their frequency decreases substantially as model size increases.

Q4: What Factors Make Multi-Tasking Hard in Asynchronous Tool Calls?

As shown in Figure 3, the results show that asynchronous multitasking causes varying degrees of performance degradation across different models compared with regular multitasking, accompanied by a simultaneous decline in the SC index.

LLaMA-3.3-70B

Closed-source

6.0

Open-source <20B Open-source >20B

5.5

LLaMA-3.1-70B

Same-taskStreak

5.0

LLaMA-8B

4.5

4.0

Qwen3-30B

DeepSeek-V3.1

GLM-9B

3.5

GPT-5

Qwen3-14B

GLM-32B

Qwen2.5-7B

3.0

GPT-4o

Qwen3-32B

GPT-4.1

2.5

Gemini

Qwen-Max

0 10 20 30 40

Overall Score

- Figure 4: Trade-off between task accuracy and scheduling efficiency. The x-axis reports Overall Score, while the y-axis reports Same-task Streak. Lower Same-task Streak indicates stronger interleaving ability in asynchronous multi-task execution.

In terms of the SC metric, Gemini 2.5 Pro exhibits the largest drop among closed-source models, while Qwen3-8B experiences the greatest decrease among open-source models.

The key distinction between asynchronous and regular multitasking is whether the agent can immediately obtain tool responses. In asynchronous settings, each function call incurs a delay, so after invoking a tool, the agent cannot access the information needed for the next step, such as file operation confirmations, query results, or critical state data. High-performing models can handle this uncertainty by shifting their attention to other tasks and making progress via different tool calls while waiting. In contrast, lower-performing models often lack temporal awareness and prematurely advance within the same task, sometimes fabricating parameters based on assumed outcomes from prior calls—an instance of the hallucination behavior commonly observed in large language models.

Another factor influencing difficulty is task quantity. An intuitive hypothesis is that as the number of tasks increases, the overall difficulty also increases. As shown in Figure 6, when the number of tasks reaches four, the difficulty of each task increases significantly, and the impact of task quantity becomes more pronounced. Consequently, we do not consider a larger number of tasks in this analysis. Based on ASYNCTOOL, we find that when the number of tasks goes from two to three, the closed-source model Gemini 2.5 Pro decreases the most, from 42.56 to 32.44 while the open-source model Qwen3-8B decreases the most, from 27.91 to 10.67 while the closed-source model GPT-4.1 and the open-source model LLaMA-3.1-70B-Ins

llama3.3-70B-Ins.

35

gpt-4o

GLM-9b-chat

Llama-3.1-70B-Ins.

30

AsyncToolAsyncOverrallScore

gemini2.5-pro

qwen3-14B

25

qwen3-30B-A3B-Ins.

20

15

10

5

0

15 20 25 30 35 40 45 50 BFCL Overall Acc

Figure 5: Comparison between BFCL Overall Accuracy and ASYNCTOOL Overall Scores across several models.

decrease less. Full results are listed in Appendix D.1

Other factors include task response latency. As shown in Appendix 8, longer latency delays tool execution, resulting in more interactions and increased task difficulty. We additionally report results under random latency settings in Appendix 9 and Appendix 10; although this introduces additional randomness into the evaluation, it provides a more realistic assessment of model performance.

### 4 Conclusion

We present ASYNCTOOL, a benchmark for evaluating asynchronous tool calling in interactive multi-task environments with delayed tool feedback. ASYNCTOOLdiffers from conventional tooluse evaluations by requiring agents to coordinate multiple tasks, utilize idle waiting periods, and respect intra-task dependencies when tool results are not immediately available. Based on validated single-task tool-use trajectories, we construct a diverse asynchronous multitasking dataset and evaluate model behavior at the step, sub-task, and task levels, together with efficiency-oriented metrics for task interleaving. Extensive experiments show that asynchronous execution remains challenging for current LLM-based agents, causing clear performance degradation under delayed feedback.Our analysis shows that effective asynchronous tool use requires correct tool calls, task-state maintenance, dependency tracking, and timely task switching. Strong agents should switch tasks at appropriate moments rather than frequently. We hope ASYNCTOOL can support future research on temporally aware, efficient, and reliable tool-using agents.

### Limitations

In ASYNCTOOL, we introduce response delays into tool calls for the first time, enabling a fine-grained evaluation of tool-using agents’ performance in asynchronously executing multiple tasks. Nonetheless, two notable limitations remain. First, our dataset is reconstructed from BFCLv3 and NESTFUL. Although tasks are carefully annotated at a fine-grained level, the task scenarios and the range of available tools are inherently constrained by these sources. Second, the data reconstruction process relies on the high-performance Gemini-2.5-pro, making dataset construction costly. This dependency not only restricts further expansion of the dataset but also introduces inevitable generation errors that require additional human verification, thereby increasing annotation costs. These limitations present challenges to the diversified scaling of asynchronous multitasking evaluation.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

AI@Meta. 2024. Llama 3 model card. Anthropic. 2025. Introducing claude 4.

Kinjal Basu, Ibrahim Abdelaziz, Kelsey Bradford, Maxwell Crouse, Kiran Kate, Sadhana Kumaravel, Saurabh Goyal, Asim Munawar, Yara Rizk, Xin Wang, and 1 others. 2024. Nestful: A benchmark for evaluating llms on nested sequences of api calls. arXiv preprint arXiv:2409.03797.

Jiajun Chai, Zijie Zhao, Yuanheng Zhu, and Dongbin Zhao. 2025. A survey of cooperative multi-agent reinforcement learning for multi-task scenarios. Artificial Intelligence Science and Engineering, 1(2):98– 121.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, and 1 others. 2024. Mle-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, and 1 others. 2025a. Minimaxm1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585.

Jinyang Chen, Haolun Wu, Jianhong Pang, Yihua Wang, Dell Zhang, and Changzhi Sun. 2025b. Tool learning with language models: a comprehensive survey of methods, pipelines, and benchmarks. Vicinagearth, 2(1):16.

Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, and Feng Zhao. 2024a. T-eval: Evaluating the tool utilization capability of large language models step by step. In ACL, pages 9510–9529.

Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao. 2024b. Mindsearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Haotian Wang, Ming Liu, and Bing Qin. 2023. Timebench: A comprehensive evaluation of temporal reasoning abilities in large language models. arXiv preprint arXiv:2311.17667.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

In Gim, Seung-seob Lee, and Lin Zhong. 2024. Asynchronous llm function calling. arXiv preprint arXiv:2412.07017.

Antonio A Ginart, Naveen Kodali, Jason Lee, Caiming Xiong, Silvio Savarese, and John Emmons. 2024. Asynchronous tool usage for real-time agents. arXiv preprint arXiv:2410.21620.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, and 1 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Gonzalo Gonzalez-Pumariega, Leong Su Yean, Neha Sunkara, and Sanjiban Choudhury. 2025. Robotouille: An asynchronous planning benchmark for llm agents. arXiv preprint arXiv:2502.05227.

Zishan Guo, Yufei Huang, and Deyi Xiong. 2024. CToolEval: A Chinese benchmark for LLM-powered agent evaluation in real-world API interactions. In ACL, pages 15711–15724.

Kethmi Hirushini Hettige, Jiahao Ji, Cheng Long, Shili Xiang, Gao Cong, and Jingyuan Wang. 2025. A modular multitask reasoning framework integrating spatio-temporal models and llms. arXiv preprint arXiv:2506.20073.

Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister. 2023. Tool documentation enables zero-shot tool-usage with large language models. arXiv preprint arXiv:2308.00675.

Shijue Huang, Wanjun Zhong, Jianqiao Lu, Qi Zhu, Jiahui Gao, Weiwen Liu, Yutai Hou, Xingshan Zeng, Yasheng Wang, Lifeng Shang, Xin Jiang, Ruifeng Xu, and Qun Liu. 2024. Planning, creation, usage: Benchmarking LLMs for comprehensive tool utilization in real-world complex scenarios. In Findings of the Association for Computational Linguistics: ACL 2024, pages 4363–4400. Association for Computational Linguistics.

Shiting Huang, Zecheng Li, Yu Zeng, Qingnan Ren, Zhen Fang, Qisheng Su, Kou Shi, Lin Chen, Zehui Chen, and Feng Zhao. 2026. Internalizing metaexperience into memory for guided reinforcement learning in large language models. arXiv preprint arXiv:2602.10224.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Junlong Li, Wenshuo Zhao, Jian Zhao, Weihao Zeng, Haoze Wu, Xiaochen Wang, Rui Ge, Yuxuan Cao, Yuzhen Huang, Wei Liu, and 1 others. 2025. The tool decathlon: Benchmarking language agents for diverse, realistic, and long-horizon task execution. arXiv preprint arXiv:2510.25726.

Fangru Lin, Emanuele La Malfa, Valentin Hofmann, Elle Michelle Yang, Anthony G. Cohn, and Janet B. Pierrehumbert. 2024. Graph-enhanced large language models in asynchronous plan reasoning. In Proceedings of the 41st International Conference on Machine Learning, ICML’24.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, and 1 others. 2023. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688.

- OpenAI. 2025a. Gpt-5. https://openai.com/ zh-Hans-CN/index/introducing-gpt-5/. Accessed: 2025-08-07.
- OpenAI. 2025b. Introducing o3 and o4-mini.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2023. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Guoqing Du, Shiwei Shi, Hangyu Mao, Xingyu Zeng, and Rui Zhao. 2023. Tptu: Task planning and tool usage of large language modelbased ai agents. arXiv preprint arXiv:2308.03427.

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023. Towards benchmarking and improving the temporal reasoning capability of large language models. arXiv preprint arXiv:2306.08952.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

- Qwen Team. 2024a. Qwen2 technical report. arXiv preprint arXiv:2407.10671.
- Qwen Team. 2024b. Qwen2.5: A party of foundation models.
- Qwen Team. 2024c. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. 2025. Vrag-rl: Empower visionperception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019.

Yuqing Wang and Yun Zhao. 2023. Tram: Benchmarking temporal reasoning for large language models. arXiv preprint arXiv:2310.00835.

Mengsong Wu, Tong Zhu, Han Han, Chuanyuan Tan, Xiang Zhang, and Wenliang Chen. 2024. Seal-tools: Self-instruct tool learning dataset for agent tuning and detailed benchmark. In Natural Language Processing and Chinese Computing: 13th National CCF Conference, NLPCC 2024, Hangzhou, China, November 1–3, 2024, Proceedings, Part II, page 372–384.

Zirui Wu, Xiao Liu, Jiayi Li, Lingpeng Kong, and Yansong Feng. 2025. Recipe2plan: Evaluating planning abilities of llms for efficient and feasible multitasking with time constraints between actions. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 4279–4301.

Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. 2024. Travelplanner: A benchmark for realworld planning with language agents. arXiv preprint arXiv:2402.01622.

Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. 2023. On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504.

Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Berkeley function calling leaderboard.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. Preprint, arXiv:2406.12045.

Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Qi Zhang, Tao Gui, and 1 others. 2024. Tooleyes: Fine-grained evaluation for tool learning capabilities of large language models in real-world scenarios. arXiv preprint arXiv:2401.00741.

Shaofeng Yin, Ting Lei, and Yang Liu. 2025. Toolvqa: A dataset for multi-step reasoning vqa with external tools. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4424–4433.

Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, and Feng Zhang. 2025a. C3-bench: The things real disturbing llm based agent in multitasking. arXiv preprint arXiv:2505.18746.

Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, and Feng Zhang. 2025b. Multimission tool bench: Assessing the robustness of llm based agents through related and dynamic missions. Preprint, arXiv:2504.02623.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, and 1 others. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471.

Yikai Zhang, Siyu Yuan, Caiyu Hu, Kyle Richardson, Yanghua Xiao, and Jiangjie Chen. 2024a. Timearena: Shaping efficient multitasking language agents in a time-aware simulation. arXiv preprint arXiv:2402.05733.

Yinger Zhang, Hui Cai, Xierui Song, Yicheng Chen, Rui Sun, and Jing Zheng. 2024b. Reverse chain: A generic-rule for LLMs to master multi-API planning. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 302–325. Association for Computational Linguistics.

Ziao Zhang, Kou Shi, Shiting Huang, Avery Nie, Yu Zeng, Yiming Zhao, Zhen Fang, Qishen Su, Haibo Qiu, Wei Yang, and 1 others. 2026. Skillflow: Benchmarking lifelong skill discovery and evolution for autonomous agents. arXiv preprint arXiv:2604.17308.

Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V Le, Ed H Chi, and 1 others. 2024. Natural plan: Benchmarking llms on natural language planning. arXiv preprint arXiv:2406.04520.

Lucen Zhong, Zhengxiao Du, Xiaohan Zhang, Haiyi Hu, and Jie Tang. 2025. Complexfuncbench: exploring multi-step and constrained function calling under long-context scenario. arXiv preprint arXiv:2501.10132.

Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2023. Toolqa: A dataset for llm question answering with external tools. arXiv preprint arXiv:2306.13304.

### A Related Work

Multi-Step Tool Call. As LLM-based agents continue to improve in their ability to leverage external tools, related benchmarks have higher complexity and richer dependencies. Some tasks now require multiple tool calls, executed in the correct order, to be completed, which presents significant challenges for the models in terms of comprehension and planning (Chen et al., 2025b; Yin et al., 2025; Xie et al., 2024; Zheng et al., 2024). Currently, two primary approaches are used to evaluate this capability. The first approach asks the model to provide, in a single step, the full sequence of tool calls needed to solve the task, focusing on the model’s ability to plan tool usage (Huang et al., 2024; Yu et al., 2025b; Zhang et al., 2024b; Chen et al., 2024a; Wu et al., 2024). The second approach allows the model to invoke tools step by step, receiving responses at each stage. It emphasizes evaluating the multi-step tool calling process until task completion (Yan et al., 2024; Basu et al.,

- 2024; Zhong et al., 2025). The order of tool calls is often a critical aspect to

evaluate. (Yan et al., 2024) constrains the tool call sequence using traditional rule-based restrictions, while NESTFUL (Basu et al., 2024) enforces a fixed call order through parameter dependencies. A key limitation of existing benchmarks is their failure to account for tool invocation delays frequently encountered in real-world applications. ASYNCTOOLaddresses this gap, enabling systematic evaluation of a model’s capacity for asynchronous tool calling.

Asynchronous Multitasking Scenarios. The ability to perform multiple tasks asynchronously is an inevitable trend in the development of large LLMs. (Chu et al., 2023; Tan et al., 2023; Wang and Zhao, 2023; Ginart et al., 2024; Hettige et al.,

- 2025; Chai et al., 2025; Wu et al., 2025) introduce time changes in model evaluation and provide related evaluation tasks.(Zhang et al., 2024a) introduces the concept of time in multitasking, aiming to assess the efficiency with which models manage multiple tasks concurrently. (Gonzalez-Pumariega et al., 2025) evaluates models’ asynchronous planning, focusing on failure modes and challenges in integrating long-term information. While important for understanding LLMs in asynchronous tasks, these studies are limited to simulations and don’t assess performance on tasks requiring real-world tool calls. Moreover, (Gim et al., 2024) enhances the

operational efficiency of LLMs by enabling them to generate and execute function calls concurrently, representing a novel approach. Unlike prior benchmarks, ASYNCTOOL is the first to systematically evaluate models’ capabilities for asynchronous tool calling in realistic multi-task scenarios, explicitly incorporating tool response latency into evaluation environment.

### B Evaluation

#### B.1 Metrics

Character matching. We extract the function name and parameters and match them against the golden truth. We also consider the possibility that the model might execute some incorrect function calls due to the use of subset validation. Specifically, given a tool list T and a query q, the model generates a series

T pred = ⟨apred1 ,apred2 ,...,apredn ⟩,

where apredi is the predicted LLM action at turn i. The golden truth is

T gt = ⟨agt1 ,agt2 ,...,agtn⟩. We verify whether Agt ⊆ Apred.

F1 function & Parameters. Referring to the implementation of NESTFUL((Basu et al., 2024)), we use the above metrics to evaluate the model’s responses. In our experiments, we find that frequent task switching by the model interferes with its ability to recognize different tasks accurately. Specifically, the model often invokes incorrect tools for the tasks, which leads to a decline in performance metrics.

Trajectory completion. After we extract the function, we will throw T gt and T pred to the executor for execution. The executor will return the corresponding results Rgt and Rpred, and we will compare the results using the subset judgment method the same as above.

Environment matching. Referring to the implementation of BFCL((Yan et al., 2024)), we use the executor class instance for comparison. In our data, some tool calls induce alterations to the environment, while the majority of tool calls do not result in any environmental changes, such as those associated with queries. Nonetheless, we incorporate this metric into the evaluation to ensure comprehensiveness.

Task Completion (TC) Scores across Models

[Figure 53]

50

[Figure 54]

50.00 42.50 41.67 44.17 32.50 1.67 12.50 10.00 30.00 25.83 27.50 10.00 5.83 34.17 41.67

- SIMILAR#2

- SIMILAR#3

40

36.67 29.17 21.67 27.50 15.42 0.42 7.08 4.17 17.92 5.42 14.58 3.75 1.67 20.83 27.08

30

TCScore

20

47.73 41.67 43.94 40.15 41.67 4.55 8.33 12.12 18.94 15.91 28.03 8.33 6.06 31.82 39.39

- CROSS#2

- CROSS#3

10

27.27 22.73 28.64 26.82 23.18 0.00 2.73 2.27 13.64 5.00 13.18 2.73 0.45 20.00 24.55

0

GPT-4.1 GPT-4.0 GPT-5Gemini2.5-ProQwen-maxLLaMA-3.1-8B-Ins.GLM-4-9B-chatQwen2.5-7B-Ins.Qwen2.5-14B-Ins.Qwen3-8BQwen3-14BLLaMA-3.3-70B-Ins.LLaMA-3.1-70B-Ins.Qwen2.5-32B-Ins.Qwen2.5-72B-Ins.

Figure 6: The performance of different models on specific task classifications. Red indicates poorer performance, and green indicates better performance.

#### B.2 Compare ASYNCTOOLwith other benchmarks

Tab. 2 shows how ASYNCTOOLcompares against existing tool-use and asynchronous benchmarks.

#### B.3 Model Version

The version for GPT-4o is gpt-4o-2024-11-20, for GPT-4.1 is gpt-4.1-2025-04-14, and for Gemini 2.5 Pro is gemini-2.5-pro-preview-05-06. The version for GLM-4-32B is GLM-4-32B-0414. The version for Qwen3-30B-A3B-Ins. is Qwen3-30B-A3B-Instruct-2507.

### C Data

#### C.1 Human Annotation Instructions

We used human annotators to verify and refine the single-task tool-use trajectories during dataset construction. The annotators were instructed to check whether each task description, tool-call trajectory, and execution result were mutually consistent. The full instructions given to annotators are as follows:

Annotation Goal. Given a task description, a list of available tools, and a candidate multi-step tool-call trajectory, please determine whether the trajectory correctly solves the task. The goal is to ensure that each trajectory is executable, deterministic, and consistent with the task description.

Annotation Procedure. Annotators were asked to follow these steps:

- 1. Read the task description carefully and identify the user’s intended goal.
- 2. Inspect the ordered tool-call trajectory and check whether each function call is valid.
- 3. Verify that the function name and arguments are supported by the corresponding tool.

- 4. Check whether the trajectory respects all dependency relations between tool calls. A later call should not use information that has not been obtained from a previous tool response.
- 5. Execute or inspect the trajectory results and confirm that the final environment state satisfies the task requirement.
- 6. If an error is found, mark the error type and provide a corrected version when the correction is unambiguous.
- 7. If the task description is ambiguous or inconsistent with the trajectory, rewrite the description to make key information such as entities, time, location, or required arguments explicit.
- 8. Remove instances that cannot be reliably corrected or whose task description cannot be aligned with the execution trajectory.

Common Error Types. Annotators were asked to pay special attention to the following errors:

- • incorrect interpretation of the initial task condition;
- • missing prerequisite tool calls;
- • invalid function names or unsupported argument formats;
- • incorrect ordering of dependent function calls;
- • mismatch between the task description and the execution trajectory;
- • ambiguous task descriptions that may lead to multiple valid trajectories.

Risk and Privacy Notice. The annotation task only involved checking synthetic benchmark task descriptions and tool-call trajectories. Annotators were not asked to provide personal information, opinions, or sensitive data. The annotation process did not involve interaction with real users or real private user data. Therefore, the risk to annotators was minimal.

Quality Control. Each instance was checked for trajectory validity, task-trajectory consistency, and final environment correctness. The annotation process was repeated until no execution errors or obvious inconsistencies remained in the validated single-task dataset.

###### Table 2: Comparison between ASYNCTOOL and existing benchmarks.

###### Benchmark Async Executor Function Call Multi Task Multi Step Cross Scenario

τ-bench (Yao et al., 2024) ✗ ✓ ✗ ✗ ✗ BFCL v3 (Yan et al., 2024) ✗ ✓ ✗ ✓ ✗ NestFul (Basu et al., 2024) ✗ ✓ ✗ ✓ ✗

TimeArena (Zhang et al., 2024a) ✓ ✗ ✓ ✓ ✗ Cˆ3-Bench (Yu et al., 2025a) ✗ ✓ ✓ ✓ ✗ Robotouille (Gonzalez-Pumariega et al., 2025) ✓ ✗ ✗ ✓ ✗

ASYNCTOOL (Ours) ✓ ✓ ✓ ✓ ✓

#### Table 3: Abbreviation explanation of data categories. Full Name Abbreviation

Data Management DM Filesystem FS Data generation DG MessageAPI MA Number operations NO SocialConnect SC String Manipulation SM TicketPurchase TP TradingBot TB TravelPlanning TP* DataFormat DF Machine Operation MO

###### Table 4: Task Counts and Average Trajectory Lengths.

###### Type DM FS DG MA NO SC SM TP TB TP* DF MO

#Tasks 59 53 29 23 24 18 34 21 15 20 44 18 Avg. len. 2.14 3.58 2.14 3.48 2.12 4.11 2.09 2.81 3.27 3.10 2.09 3.00

#### C.2 Data details

As shown in Table 3, the content inside introduces the meaning of our categories. Besides, Table 5 shows the data distribution of ASYNCTOOL.

#### C.3 Data length

We counted the input and output lengths of each task to ensure that they were within a reasonable range. These data are shown in Figure 7.

#### C.4 Supplementary data

For the four-task test, considering the large number of combination types and tasks, we randomly screened the data after each combination and finally selected 300 SIMILAR and 300 CROSS four-task data for comparison. The results are only used for ablation experiments.

### D Results

In this section, we present the comprehensive results, categorized by different types, along with the

Table 5: Data composition of ASYNCTOOL. # denotes number of, and % denotes proportion of. SIMILAR means similar task combinations and CROSS means cross task combinations.

Method #Tasks #Samples %Samples

- SIMILAR 2 120 16.85%

- CROSS 2 132 18.54%

SIMILAR 3 240 33.71%

- CROSS 3 220 30.90%

test outcomes for varying numbers of tasks on the open-source model.

#### D.1 Detailed results

- As shown in Figure 6, we compile and present the detailed results by task type.

D.2 Other results

- As shown in Figure 7, we conduct a preliminary experiment on the open-source model to demonstrate the impact of the number of tasks on the score.

#### D.3 Analysis of error cases

We provide numerous examples of common model errors in the end of appendix. A frequent issue, likely due to insufficient prior training on multiple tasks, is the model forgetting about ongoing tasks. This results in a high rate of incompletion failures, despite the model’s high accuracy in completing subtasks. In open-source models, frequent task switching presents an even greater challenge to the model’s memory and localization abilities, causing confusion in function calls, which is a primary source of errors.

#### D.4 Analysis of model performance

We must acknowledge that even the most advanced models perform suboptimally on our benchmark. However, we need to analyze the reasons behind this from two perspectives. First, we have conducted ablation experiments on mainstream large

- Table 6: Detailed Results of ASYNCTOOL. We approach the analysis from the perspective of dataset classification to calculate the results. We find that the scores across different categories exhibit significant disparities. Bold indicates best overall performance, while underline denotes the best within the same group.

Models SIMILAR#2 SIMILAR#3 CROSS#2 CROSS#3 Overall Closed-Sourced Models

Qwen-max 32.50 15.42 41.67 23.18 25.56 Kimi-K2 33.33 20.83 30.30 20.00 24.44 Gemini 2.5 Pro 44.17 27.5 40.15 26.82 32.44 GPT-5 41.67 21.67 43.94 28.64 31.32 GPT-4o 42.50 29.17 41.67 22.73 31.74 GPT-4.1 50.00 36.67 47.73 27.27 38.06

Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 1.67 0.42 4.55 0 1.26 GLM-4-9B-chat 12.50 7.08 8.33 2.73 6.88 Qwen2.5-7B-Ins. 10.00 4.17 12.12 2.27 6.04 Qwen2.5-14B-Ins. 30.00 17.92 18.94 13.64 18.82 Qwen3-8B 25.83 5.42 15.91 5.00 10.67 Qwen3-14B 27.50 14.58 28.03 13.18 18.82

Open-Sourced Models >20B

LLaMA-3.1-70B-Ins. 5.83 1.67 6.06 0.45 2.81 LLaMA-3.3-70B-Ins. 10.00 3.75 8.33 2.73 5.34 GLM-4-32B 21.67 13.75 21.97 9.09 15.17 Qwen3-32B 31.67 13.33 26.52 14.09 19.10 Qwen2.5-32B-Ins. 34.17 20.83 31.82 20.00 24.86 Qwen3-30B-A3B-Ins. 30.00 15.42 32.58 16.82 21.49 DeepSeek-V3.1-Terminus 32.50 24.58 35.61 27.73 28.93

- Table 7: Experimental results with different numbers of tasks on SYNC setting. In #number,number means the number of tasks. # 3 is compared with # 2 and # 4 is compared with # 3. The data source of # 4 is shown in Appendix C.2

# 2 # 3 # 4

Models

S#2 C#2 Overall S#3 C#3 Overall Drop S#4 C#4 Overall Drop

Qwen2.5-7B-Ins. 20.00 14.39 17.06 11.67 10.00 10.87 36.28% 8.51 3.44 6.33 41.77% GLM-4-9b-chat 20.83 21.97 21.43 14.58 10.91 12.83 40.13% 12.73 3.65 8.17 36.32%

Qwen2.5-72B-Ins. 53.33 47.73 50.79 35.42 31.82 33.70 33.65% 20.33 23.67 22.00 34.72% LLaMA3.3-70B-Ins. 25.00 16.67 20.63 10.42 6.82 8.70 57.83% 3.67 3.67 3.67 57.81%

models, revealing that as the number of tasks increases, the difficulty grows nonlinearly. To encourage future models to achieve greater capabilities, we designed the benchmark such that tasks involving three or more objectives account for over 60

#### D.5 Ablation results

###### Table 8: Supplementary Results of ASYNCTOOL.All experiments are conducted with a delay of two turns. Bold indicates best overall performance, while underline denotes the best within the same group.

Step Level Sub-Task Level Task Level

Models

Overall Func. Param. Char. Env. Acc. Char. Env.

Closed-Sourced Models

Qwen-Max 95.81 82.70 64.7 77.18 60.18 35.11 53.51 30.76 Gemini 2.5 Pro 91.94 80.05 57.48 75.90 56.48 26.97 53.65 25.70 GPT-4o 91.41 79.79 57.70 75.42 56.79 27.67 53.37 26.54 GPT-4.1 91.32 79.97 57.96 75.90 56.90 27.67 54.49 26.40

Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 83.16 44.42 13.38 40.76 12.42 1.54 14.75 1.40 Qwen2.5-7B-Ins. 87.67 66.97 32.27 58.28 29.99 6.88 26.40 5.48 Qwen2.5-14B-Ins. 86.75 72.18 50.21 63.59 45.06 19.94 35.11 15.17 Qwen3-8B 59.06 49.4 24.52 48.99 23.99 6.6 21.77 6.6

Open-Sourced Models > 20B

LLaMA-3.1-70B-Ins. 90.73 45.45 17.04 40.02 16.14 1.83 12.36 1.4 Qwen3-32B 70.67 60.99 34.34 56.95 33.7 11.52 30.76 10.96 Qwen3-30B-A3B-Ins. 95.57 79.72 55.41 71.71 52.81 23.17 44.66 20.65 DeepSeek-V3.1-Terminus 81.69 70.52 43.84 65.02 42.94 15.73 36.94 14.61

###### Table 9: Results ofASYNCTOOL.All experiments are conducted with a delay of zero to one turn randomized. Bold indicates best overall performance, while underline denotes the best within the same group.

Step Level Sub-Task Level Task Level

Models

Overall Func. Param. Char. Env. Acc. Char. Env.

Closed-Sourced Models

Qwen-Max 96.65 84.21 66.30 80.57 64.65 36.80 58.43 34.41 Gemini 2.5Pro 95.57 83.85 65.45 81.69 64.92 35.81 62.22 35.25 GPT-4o 96.00 84.03 66.08 81.74 65.61 35.96 62.08 35.53 GPT-4.1 95.45 83.58 65.76 81.90 65.18 35.67 62.92 34.55

Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 83.93 47.44 16.24 43.84 15.13 1.97 16.71 1.69 Qwen2.5-7B-Ins. 83.73 67.22 32.11 59.02 31.00 8.71 29.07 8.43 Qwen2.5-14B-Ins. 85.00 73.32 48.30 67.46 46.18 19.24 39.47 17.98 Qwen3-8B 69.38 60.22 38.80 61.09 38.32 14.47 35.11 14.04

Open-Sourced Models > 20B

LLaMA-3.1-70B-Ins. 90.49 49.28 20.12 45.59 19.75 3.51 18.12 3.37 Qwen3-32B 76.21 67.91 47.45 66.88 47.03 20.79 43.54 20.08 Qwen3-30B-A3B-Ins. 95.10 82.17 59.39 76.70 58.86 26.69 52.67 26.26 DeepSeek-V3.1-Terminus 90.04 78.29 56.32 75.96 55.73 26.97 52.39 26.26

Base Final

Base Final

Base Final

| |
|---|

| |
|---|

| |
|---|

200

200

200

175

150

150

150

125

Count

Count

Count

100

100

100

75

50

50

50

25

0

0

0

0 500 1000 1500 2000 2500 3000 3500

0 500 1000 1500 2000 2500 3000 3500

0 500 1000 1500 2000 2500 3000 3500

Length

Length

Length

(a) GPT-4.1

(b) GPT-4o

(c) Gemini 2.5 Pro

Figure 7: Length distribution for base and final conversation, measured by the number of tokens.

###### Table 10: Results ofASYNCTOOL.All experiments are conducted with a delay of one to two turns randomized. Bold indicates best overall performance, while underline denotes the best within the same group.

Step Level Sub-Task Level Task Level

Models

Overall Func. Param. Char. Env. Acc. Char. Env.

Closed-Sourced Models

Qwen-Max 96.87 83.56 64.49 78.50 61.94 33.85 54.63 30.20 Gemini 2.5 Pro 93.98 81.68 61.84 78.34 61.04 30.62 57.58 29.35 GPT-4o 93.08 81.05 60.51 78.29 59.66 30.06 57.44 28.79 GPT-4.1 91.97 80.29 58.76 76.17 58.07 30.34 55.90 29.49

Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 83.77 45.06 12.74 41.14 11.68 1.54 15.03 1.54 Qwen2.5-7B-Ins. 86.02 66.81 29.09 56.48 27.55 6.88 26.26 6.04 Qwen2.5-14B-Ins. 87.22 73.30 49.10 65.82 45.70 19.52 36.94 16.29 Qwen3-8B 60.72 50.76 25.85 51.38 25.42 7.30 25.14 7.16

Open-Sourced Models > 20B

LLaMA-3.1-70B-Ins. 90.76 46.37 16.67 40.98 16.03 2.25 13.90 1.83 Qwen3-32B 71.41 61.92 37.10 59.55 36.20 12.08 33.71 11.24 Qwen3-30B-A3B-Ins. 95.39 79.86 54.09 72.13 52.87 22.75 46.49 20.93 DeepSeek-V3.1-Terminus 84.52 73.42 47.35 68.31 46.07 17.98 39.89 16.57

###### Table 11: Results of ASYNCTOOL on fewshot settings.Specifically, we provided a successful trajectory in the prompt as a reference.

Step Level Sub-Task Level Task Level

Models

Overall Func. Param. Char. Env. Acc. Char. Env.

Open-Sourced Models < 20B Qwen2.5-7B-Ins. 82.40 65.01 28.66 55.15 26.38 6.74 25.84 6.04

+fewshot 78.89 61.72 30.04 54.41 28.13 8.99 26.40 8.29 LLaMA-3.1-8B-Ins. 78.29 43.69 12.95 41.08 12.47 1.26 14.61 1.26

+fewshot 90.27 63.35 29.67 56.37 28.03 7.16 27.67 6.74 Qwen3-8B 63.05 53.61 29.88 53.08 29.30 11.10 28.65 10.67

+fewshot 72.04 60.67 34.82 34.82 34.34 11.52 32.30 11.24 Qwen2.5-14B-Ins. 81.32 70.22 50.21 65.29 46.28 22.61 38.20 18.32

+fewshot 94.28 80.40 54.19 75.32 52.87 22.89 50.14 21.91 Qwen2.5-72B-Ins. 95.69 82.78 65.07 79.78 62.00 34.41 58.01 31.04

+fewshot 89.34 83.61 66.40 81.16 64.65 36.94 59.69 34.55

###### Table 12: Results of ASYNCTOOL. Specifically, we analyzed the average number of rounds of interaction in successfully completed task models.

Models # 2 # 3 Closed-Sourced Models

Qwen-max 6.91 9.38 Gemini 2.5 Pro 6.85 10.50 GPT-4o 6.95 10.52 GPT-4.1 6.70 10.57

###### Open-Sourced Models < 20B

LLaMA-3.1-8B-Ins. 9.75 9.00 Qwen2.5-7B-Ins. 8.50 12.35 Qwen2.5-14B-Ins. 7.94 10.95 Qwen3-8B 6.64 9.19

###### Open-Sourced Models >20B

LLaMA-3.1-70B-Ins. 8.41 16.28 LLaMA-3.3-70B-Ins. 10.62 16.23 Qwen2.5-32B-Ins. 9.20 13.68

### E Use of Large Language Models

We use a large language model for translation and language polishing, but its role in the paper is limited. The core research and content are completed by us.

Standardization of Test Data System Prompt: You are an expert in handling multiple tasks using functions. Your goal is to call one function in each round, gradually completing all tasks. When a function call is not yet complete, you can switch to another task. If you decide to invoke a function for a task, you MUST format it as follows:

{"id": "id of task", "func_name": "func_name", "params": {"params_name1": "params_value1", "params_name2": "params_value2"}}

You must invoke only one function in your response. If you believe all tasks have been completed, please use the following format to reply:

{"content": "ALL COMPLETED"}

Here is a list of functions in JSON format that you can invoke.

[API List]

User: Below are all tasks you need to finish:

{

id:'file_13 ', question:'Compare 'academic_venture/goals.txt' with 'academic_venture/reference_goals.txt' to verify if they are identical , then rename 'goals.txt' to 'master_goals.txt' in the same directory tory.'

} {

id:'SM_11 ', question have a list of integers [1, 2, 3, 4, 5]. I want to convert this list into a string representing a space -separated sequence of those integers. Then , I want to normalize this string by replacing all non -alphanumeric characters with underscores and converting to lowercase.'

} {

id:'posting_11 ', question:'Now that I'm following 'AI_enthusiast ', I'd like to know more about them

. Please retrieve their user statistics. After that , fetch all of their tweets. I remember seeing one about 'ethics '; find that specific tweet and add the comment 'A very important point to consider.' to it.'

}

#### Ground True:

[{

"task_id": "file_13", "ground_truth": [

"cd(folder='academic_venture ')", "diff(file_name1='goals.txt', file_name2='reference_goals.txt ')", "mv(source='goals.txt', destination='master_goals.txt ')"

],

}, { "task_id": "SM_11", "ground_truth": [

"integer_list_to_string(integer_list=[1, 2, 3, 4, 5])", "normalize_string(s='1 2 3 4 5')"

],

}, { "task_id": "posting_11", "ground_truth": [

"get_user_stats(username='AI_enthusiast ')", "get_user_tweets(username='AI_enthusiast ')", "comment(tweet_id=18, comment_content='A very important point to consider.')"

], }]

Figure 8: An example of standardized test data.

Prompt for Reconstruction

Prompt: You are an AI assistant with API-calling capabilities. Your task is to refine query descriptions based on given contexts and initial configurations, then provide explicit solution paths. You must meet these requirements:

- 1. Ensure problems are solvable within the given initial configuration, avoiding ambiguous or unsolvable cases. Clarify any ambiguities in the original problem by deriving explicit specifications from the initial configuration. Including but not limited to: file names, folder paths, numerical counts, and similar parameters.
- 2. Define solution paths as strict API call sequences (not less than 2 steps) with strong dependencies—later steps must require outputs from earlier ones. Expand the problem if necessary to meet this length requirement.
- 3. Preserve existing explicit information (e.g., filenames, data). Extend problems only by deriving new details from the initial config.
- 4. Always provide the most efficient path—no unnecessary API calls. Consecutive duplicate function calls are prohibited in the execution sequence.
- 5. Never explicitly name APIs in the problem description. Carefully review the function description. You must ensure all function calls in the provided ground truth strictly comply with the specifications. For example, certain filesystem operations are restricted to the current working directory and cannot process multi-level paths. Only functions explicitly defined in the function description may appear in the ground truth. 6. For every input data given, you must return the same number of corresponding processed data.
- 6. You have permission to edit the question and ground_truth fields, and your response must explicitly return modified versions. The output format should follow the usage examples in JSON format instead of the origin data.

You can use the following APIs:

[API Document]

Here is one usage examples that you must follow:

[

{

"id": "number_0", "question":"Hey there , I need to set up a directory titled 'Reports ' in my current workspace. Once that 's in place , could you help me locate a file called 'summary.doc' in this vicinity and transfer it to the new 'Reports ' folder? After that , I'd appreciate if you could find and open up 'data.txt', then scan through it to identify lines that mention 'Q4 financials '. Also , could you let me know the total number of lines in 'data.txt '?", "ground_truth ":["mkdir(dir_name='Reports ')","mv(source='summary.doc', destination='Reports ')", "cat(file_name='data.txt ')", "grep(pattern='Q4 financials ', file_name='data.txt ')", "wc(file_name='data.txt',mode='l')"], "thought":"your thought"

}, {

"id": "number_1", "question":"Let's bring some order to the project documents. I want to human -readable log the storage usage of the entire current directory to usage.txt file. The content of the file should be the number follwed by the word bytes and nothing else.", "ground_truth ":["du(human_readable=True)", "touch(file_name='usage.txt ')", "echo(content='205 bytes ', file_name='usage.txt ')"], "thought":"your thought"

} ]

where ‘id’ is the id of task, ‘question’ is the user query, ‘ground_truth’ is the shortest function calls sequences used to solve the query, and ‘thought’ is your ideas for refining this query.

Figure 9: prompt for reconstruction.

Reconstructed Data Task: Locate a file with “ideas” in its name within the ‘documents’ folder, display its contents, and create a backup copy named ‘ideas_backup.txt’ in the same directory. Config:

{

"root": {

"simona": { "type": "directory", "contents": {

"documents": { "type": "directory", "contents": {

"ideas.txt": { "type": "file", "content": "Collaboration leads to success. Innovation ignites growth."

}, "Archived": {

"type": "directory", "contents": {}

}, "past_projects": {

"type": "directory", "contents": {}

} }

} }

} }

}

#### Trajectory:

[

"cd(folder='documents ')", "find(name='ideas ')", "cat(file_name='ideas.txt ')", "cp(source='ideas.txt', destination='ideas_backup.txt ')"

]

Figure 10: An example of reconstructed data: GorillaFileSystem.

##### API Document

{

"name": "mv", "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories , creating files and directories , reading and writing to files , etc. Tool description: Move a file or directory from one location to another. so", "parameters": {

"type": "dict", "properties": {

"source": { "type": "string", "description": "Source name of the file or directory to move. Source must be local to the current directory."

}, "destination": {

"type": "string", "description": "The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something , destination is the new file name. "

}

}, "required": [

"source", "destination"

]

}, "response": {

"type": "dict", "properties": {

"result": { "type": "string", "description": "The result of the move operation."

} }

} }

Figure 11: An example of an API document.

Correct Trajectory

User: Below are all tasks you need to finish:

{id:'trading_0 ',question:'Can you get me the ticker for 'Alpha Tech ', see its current price , and then execute a purchase of 20 shares at a price of $1320.5?'} {id:'file_11 ',question:'Hey , can you set up a new directory named 'Projects ' right in workspace folder?'}

#### Assistant:

{

"id": "trading_0", "func_name": "get_symbol_by_name", "params": {"name": "Alpha Tech"}

}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed ."}"}

#### Assistant:

{

"id": "file_11", "func_name": "cd", "params": {"folder": "workspace"}

}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"}, {"role": "tool", "id": "trading_0", "function name": "get_symbol_by_name(name="Alpha Tech")", "response": "{"symbol": "ALPH"}"}

#### Assistant:

{

"id": "trading_0", "func_name": "get_stock_info", "params": {"symbol": "ALPH"}

}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"}, {"role": "tool", "id": "file_11", "function name": "cd(folder="workspace")", "response": "{" current_working_directory": "workspace"}"}

#### Assistant:

{

"id": "file_11", "func_name": "mkdir", "params": {"dir_name": "Projects"}

}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"}, {"role": "tool", "id": "trading_0", "function name": "get_stock_info(symbol="ALPH")", "response": "{"price": 1320.45, "percent_change": -0.08, "volume": 1.567, "MA(5)": 1321.12, "MA(20)": 1325.78}"}

#### Assistant:

{

"id": "trading_0", "func_name": "place_order", "params": {"order_type": "Buy", "symbol": "ALPH", "price": 1320.5, "amount": 20}

}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"}, {"role": "tool", "id": "file_11", "function name": "mkdir(dir_name="Projects")", "response": "None"}

#### Assistant:

{"content": "ALL COMPLETED"}

Figure 12: An example of correct trajectory.

Error Caused by Insufficient Temporal Awareness

User: Below are all tasks you need to finish:

{

id:'trading_0 ', question:'Can you get me the ticker for 'Alpha Tech ', see its current price , and then execute a purchase of 20 shares at a price of $1320.5?'

} {

id:'file_11 ', question:'Hey , can you set up a new directory named 'Projects ' right in workspace folder?'

}

#### Assistant:

{

"id": "trading_0", "func_name": "get_symbol_by_name", "params": {"name": "Alpha Tech"}

}

#### Environment:

{

"role": "tool", "content": "{"wait": "The current function is being executed."}"

}

#### Assistant:

{

"id": "trading_0", "func_name": "get_stock_info", "params": {"symbol": " ATGL "}

}

- Figure 13: An example of an error caused by insufficient temporal awareness. In this case, the agent prematurely assumed the symbol of Alpha Tech to be “ATGL” before receiving the actual call result.

Error Caused by Tool Confusion

User: Below are all tasks you need to finish:

{

id:'trading_0 ', question:'Can you get me the ticker for 'Alpha Tech ', see its current price , and then execute a purchase of 20 shares at a price of $1320.5?'

} {

id:'file_11 ', question:'Hey , can you set up a new directory named 'Projects ' right in workspace folder?'

}

#### Assistant:

{

"id": "trading_0", "func_name": "get_symbol_by_name", "params": {"name": "Alpha Tech"}

}

#### Environment:

{

"role": "tool", "content": "{"wait": "The current function is being executed."}"

}

#### Assistant:

{

"id": " trading_0 ", "func_name": "cd", "params": {"folder": "workspace"}

}

- Figure 14: An example of an error caused by tool confusion. In this case, the agent mistakenly applied the tool intended for task “file_11” to task “trading_0”.

Error Caused by Neglecting a Certain Task

User: Below are all tasks you need to finish:

{

id:'file_13 ', question:'Compare 'academic_venture/goals.txt' with 'academic_venture/reference_goals.txt' to verify if they are identical , then rename 'goals.txt' to 'master_goals.txt' in the same directory.'

} {

id:'SM_11 ', question:'I have a list of integers [1, 2, 3, 4, 5]. I want to convert this list into a string representing a space -separated sequence of those integers. Then , I want to normalize this string by replacing all non -alphanumeric characters with underscores and converting to lowercase.'

} {

id:'posting_11 ', question:'Now that I'm following 'AI_enthusiast ', I'd like to know more about them. Please retrieve their user statistics. After that , fetch all of their tweets. I remember seeing one about 'ethics '; find that specific tweet and add the comment 'A very important point to consider.' to it.'

}

#### Assistant:

{"id": "file_13", "func_name": "cd", "params": {"folder": "academic_venture"}}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed ."}"}

#### Assistant:

{"id": "SM_11", "func_name": "integer_list_to_string", "params": {"integer_list": [1, 2, 3, 4, 5]}}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"},{"role": "tool", "id": "file_13", "function name": "cd(folder="academic_venture")", "response": "{" current_working_directory": "academic_venture "}"}

#### Assistant:

{"id": "file_13", "func_name": "diff", "params": {"file_name1": "goals.txt", "file_name2": "reference_goals.txt"}}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"},{"role": "tool", "id": "SM_11", "function name": "integer_list_to_string(integer_list=[1, 2, 3, 4, 5])", "response": "1 2 3 4 5"}

#### Assistant:

{"id": "SM_11", "func_name": "normalize_string", "params": {"s": "1 2 3 4 5"}}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"},{"role": "tool", "id": "file_13", "function name": "diff(file_name1="goals.txt", file_name2="reference_goals.txt")", "response": "{"diff_lines": "- Research topic selection\\\\n+ Data analysis\\\\n- Literature review\\\\n+ Data collection \\\\n- Data collection \\\\n+ Draft writing\\\\n- Data analysis\\\\n+ Final submission \\\\n- Draft writing\\\\n+ Literature review\\\\n- Final submission \\\\n+ Research topic selection"}"}

#### Assistant:

{"id": "file_13", "func_name": "mv", "params": {"source": "goals.txt", "destination": "master_goals.txt"}}

#### Environment:

{"role": "tool", "content": "{"wait": "The current function is being executed."}"},{"role": "tool", "id": "SM_11", "function name": "normalize_string(s="1 2 3 4 5")", "response": "1_2_3_4_5"}

#### Assistant:

{"content": "ALL COMPLETED"}

###### Figure 15: An example of an error caused by neglecting a certain task. In this case, the task “posting_11” was ignored by the agent.

