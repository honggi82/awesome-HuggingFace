## Exploring Autonomous Agentic Data Engineering for Model Specialization

Yujie Luo♠♡*, Xiangyuan Ru♠*, Jingsheng Zheng♠, Jingjing Wang♠, Yuqi Zhu♠, Jintian Zhang♠, Runnan Fang♠, Kewei Xu♠, Ye Liu♡, Zheng Wei♡†, Jiang Bian♡, Zang Li♡, Shumin Deng♠† ♠Zhejiang University ♡Platform and Content Group, Tencent {luo.yj,231sm}@zju.edu.cn

Abstract

[Figure 1]

### Agentic Data Engineering

[Figure 2]

Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-designed workflows, leaving it unexamined whether LLMs can autonomously execute an end-to-end data engineering pipeline for model specialization. We formalize Autonomous Agentic Data Engineering, a novel task designed to evaluate LLMs as autonomous data engineers that drive model specialization through end-to-end data curation. We frame data as an optimizable component and study agents that plan, generate, and iteratively optimize training data across multiple domains, guided by post-training performance improvement. Experiments show that autonomous LLM data engineers yield substantial gains, as GPT-5.2 constructs a training curriculum that improves a student model by 57.29%, entirely through iterative, agentdriven data adaptation. By illuminating both potential and bottlenecks, our study establishes autonomous data engineering as a measurable capability and charts a path toward agent-driven model specialization1.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Data Curation Model SFT

[Figure 7]

# arXiv:2605.30407v2[cs.CL]8Jun2026

[Figure 8]

[Figure 9]

Plan

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Science

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

Env.

Code Finance

Opt. Gen.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

LLM as Data Engineer Model Specialization

[Figure 34]

Figure 1: Paradigm of Agentic Data Engineering. LLM data engineer independently executes the entire data curation loop to drive model specialization, iteratively optimizing data guided by post-training student model performance feedback.

training on domain-specific instruction data, as exemplified by curated corpora (Zhang et al., 2024; Yang et al., 2023). Given the complexity of data processing and the scarcity of high-quality domain data, researchers have increasingly turned to LLMbased methods (Qiao et al., 2024; Liang et al., 2025), utilizing LLMs as data generators within human-designed workflows. As adapting these handcrafted recipes to new domains requires extensive configuration, modern LLM agents offer a more promising alternative through their remarkable advances in complex reasoning (DeepSeek-AI, 2025), code generation (Ni et al., 2023; Hong et al., 2024), and tool use (Qin et al., 2024). These advances further raise a natural question: Can LLM agents autonomously perform end-to-end data engineering for model specialization?

#### 1 Introduction

Large Language Models (LLMs) have acquired emergent capabilities through training on massive amounts of data (Guha et al., 2025; Zhou et al., 2025) in recent years. Despite strong performance on general tasks, even the most advanced LLMs often struggle to adapt when their training data do not adequately reflect specialized downstream tasks (Li et al., 2024; Mishra et al., 2022).

To investigate this question, we formalize the task of Autonomous Agentic Data Engineering (Figure 1), where LLMs are tasked with completing the entire training data curation pipeline independently, including strategy plan, domain specification, prompt design, data synthesis, data validation, and iterative data optimization. By holding both the teacher model for data synthesis and the student model for data training fixed, we isolate the end-to-end data engineering capability of LLMs, which is ultimately evaluated by the post-training

Adapting a general-purpose model to a target specialized domain typically necessitates post-

* Equal contribution. † Corresponding Authors. 1Code will be released at https://github.com/zjunlp/

DataAgent.

performance improvement of the student model.

We conduct a comprehensive analysis of the performance of mainstream LLMs across three specialized domains: Science, Code, and Finance. LLM capabilities are evaluated under a single-turn completion agent setting (One-Shot) and a closed-loop, self-optimizing agent setting (Iterative Agent), both from scratch and with initial seed data. Experiments show that modern LLM agents possess substantial data engineering capabilities, enabling them to infer missing supervision signals and synthesize task-aligned instances even from scratch. Notably, GPT-5.2 achieves an average relative performance gain of 57.29% through iterative optimization, surpassing human-crafted data synthesis pipelines. Despite these encouraging findings, we also identify significant failure modes, suggesting that LLMs still lack robust post-generation mechanisms for reliable quality assurance.

Overall, we summarize our contributions as:

- • We formalize the task of Agentic Data Engineering, an autonomous paradigm in which LLMs independently manage the entire training data curation lifecycle. This provides a controlled setting for studying end-to-end data engineering as a measurable capability of LLM agents.
- • We develop an end-to-end execution & evaluation environment that covers the full data curation pipeline for model specialization, enabling isolated and budget-controlled agent execution, along with external feedback and a performancebased evaluation protocol.
- • We instantiate two representative settings: OneShot and Iterative Agent, and evaluate mainstream LLMs across diverse domains. We further provide analysis of iterative optimization, data quality, and failure modes towards specialization.

#### 2 Agentic Data Engineering

##### 2.1 Problem Formulation

We formalize Agentic Data Engineering (Figure 1) as an end-to-end closed-loop paradigm in which an LLM agent A autonomously curates training data to specialize a fixed student model MS with a fixed teacher model MT for data synthesis.

For a target task T , the agent designs a datacuration program PA that calls MT to synthesize a candidate dataset

D = PA(T ; MT). (1)

The student model is then specialized on D via supervised fine-tuning, denoted Spec(·), and scored by a deterministic rule-based evaluator E, producing the environmental feedback signal

f = E Spec(MS, D) . (2)

Given the synthesis data D and the feedback signal f, the entire agentic data engineering process can be cast as a closed-loop objective in which agent A searches over curation strategies to maximize the student’s post-training performance:

PA⋆ = arg max

E(Spec(MS, PA(T ; MT))).

PA

(3) Under this formulation, both MT and MS are fixed across tasks, enabling controlled analysis of the contribution of agent-driven data curation to student model specialization.

##### 2.2 Task Protocol

Task Input As shown in Figure 2(a), for each task the agent is provided with: (1) a brief introduction of the evaluation setting; (2) a basic overview of the target dataset, including dataset description, submission format, optional seed pool, and the public test set for validation; (3) a fixed budget of teacher model API calls that the agent can use to synthesize data; and (4) a fixed student model for domain specialization, together with corresponding standardized fine-tuning & inference parameters.2

Task Output The agent is tasked to produce training data D as a submission.json file that conforms to the required format. The submission must be produced by the agent’s generated code, with all instances generated via teacher-model API calls rather than directly written into the file.

Task Evaluation We evaluate the agent by improving the end-to-end performance of the student model. Specifically, the student model is fine-tuned on the submission data D and then evaluated on the hidden private set. The resulting private-set performance gain (Section 3.1) serves as a measure of the agent’s end-to-end data engineering capability.

Task Environment Our running environment enforces fixed budgets on teacher-model API calls and wall-clock time, and provides standardized interfaces for teacher API calls, student model finetuning, and public set evaluation, as detailed in Appendix D. In this setting, the agent focuses solely

2By default, we adopt Qwen3-30B-A3B as the teacher model and LLaMA-3.1-8B-Instruct as the student model.

###### (a) Environment (b) Agent Workflow

###### AutoDataBench Agent Workflow

Dataset Curation

Domain Coverage

（i）Iterative Agent （ii）One-Shot

Iterative Agent One-Shot

Science Code Finance

[Figure 35]

###### Plan

| | | |
|---|---|---|
| |Draft| |

| |Debug| |
|---|---|---|
| | | |

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Automatic Data Curation

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Plan

[Figure 44]

|[Figure 45]<br><br>Error Code Runtime Error|
|---|

[Figure 46]

- • Create diverse problem generation
- • Generate step-by-step solutions
- • Ensure LaTeX and boxed answers
- • Save the data to submission.json

Automatic Data Curation

Phys, Chem, Math Test Output Pred. Tabular, Document

[Figure 47]

code.py

Debug

if __name__ == "__main__": # Generate data

[Figure 48]

[Figure 49]

Task Input

- • Fix parameters in call_api()
- • Fix answer extraction logic

[Figure 50]

- • Expand existing dataset
- • Add answer format filter

code.py

... ... ... # Save data ... ... ...

[Figure 51]

[Figure 52]

Task Intro

Bench Intro Dataset Overview

if __name__ == "__main__": # Define diverse science tasks categories = {

[Figure 53]

You are an expert data engineer. Your

You are participating AutoDataBench,

├── Description ├── Submission Sample │ ├── input │ ├── output ├── Public Test Set ├── Seed Pool │ ├── source.json

[Figure 54]

your mission is to craft a instruction-tuning dataset for a given task to improve models.

mission is to craft instruction-tuning data for a given task to improve models.

| | | |
|---|---|---|
| |Improve| |

| | | |
|---|---|---|
| |Repair| |

Chemistry: [thermodynamics, ...], Physics: [classical_mechanics, ...], Mathematics: [calculus, ..., ...]

[Figure 55]

[Figure 56]

[Figure 57]

Model Setting

Env.

Error

|Submission Data Invalid|
|---|

|Pub. Score & Bad Cases|
|---|

- • Teacher: Qwen3-30B-A3B
- • Student: LLaMA-3.1-8B-Instruct

} # Generate various prompts problems = gen_problem(API, categories) # Generate step-by-step solutions solutions = gen_solution(API, problems) # Create instruction-tuning pairs pairs = gen_pair(problems, solutions) # Save final data with open(submission.json) as f:

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Env. Feedback

Improve

Repair

[Figure 65]

1 2

solution difficulty problem diversity prompt engineer quality control

Model Training Pub. Test Eval

[Figure 66]

###### Pub. Score

Temp Submission Fine-Tune

Public Test Set Inference

[Figure 67]

[Figure 68]

StudentModel* Bad Cases

json.dump(dataset[:target_size],f)

Student Model ‘ Student Model ‘

[Figure 69]

[Figure 70]

[Figure 71]

Iterative Optimization

[Figure 72]

submission.json

1 2

[Figure 73]

[

###### Model Training Pri. Test Eval

Final Score

{

[Figure 74]

Final Submission

Private Test Set Inference

"instruction": "Solve the scientific problem step-by-step and express the final answer as \\boxed{[ANSWER]}.", "input": "A gas follows van der Waals equation with a=0.50 m⁶·Pa·mol⁻². Calculate the compression factor Z.", "output": "Let’s think step by step. Using van der Waals equation:\n\nZ = PVm/RT\n\nPVm = (3.0×10⁶)(5.00×10⁻⁴) =

|Task|Gain|
|---|---|
|Science|————|
|Code|————|
|Finance|————|

[Figure 75]

Fine-Tune Student Model * Student Model *

Student Model *

1500\nRT = 8.314 × 273 = 2269.7\n\nZ = 1500/2269.7 = 0.661\n\nThe answer is therefore \\boxed{0.661}." } } ... ...

| |
|---|
|Output Evaluation|
| |

]

Submission

Figure 2: Overall framework of our study. (a) Environment: the overview of the covered domains, the agent input containing task settings and procedural feedback, and the final evaluation method. (b) Agent Workflow: the example workflow in which agents develop strategies to curate data and output a submission.json towards specialization. In (ii) One-Shot setting, the submission is produced in a single pass, whereas in (i) Iterative Agent setting, the agent iteratively improves its data curation strategy with feedback and reports the best submission.

↑

- • Dataset Description: an overview of the dataset, component illustration, and data examples.
- • Evaluation Script: a script extracting answers from responses and computing dataset scores.
- • Seed Data: standardized raw materials for domain specialization, where agent visibility depends on the experiment setting.
- • Public Test Set: the visible data split for procedural feedback during iterative optimization.
- • Private Test Set: the hidden data split reserved exclusively for final performance evaluation.
- • Sample Submission: the required task-specific data generation format.

on the data engineering task by implementing the data curation logic through code generation.

- 2.3 Dataset Preparation

We collect QA reasoning tasks from three representative domains: Science, Code, and Finance, evaluating how agents adapt and improve through autonomous data engineering within each domain.

Dataset Selection We select task domains that satisfy: (i) specialized tasks that are not adequately covered by general-purpose pretraining, where targeted specialization is essential to unlock the model’s full potential; (ii) direct evaluation, enabling deterministic rule-based scoring serving as environment feedback without execution environments or LLM judgment; and (iii) broad reasoning pattern across representative domains. Based on these criteria, we adopt SciBench (Wang et al., 2024b), LiveCodeBench (Jain et al., 2025) Test Output Prediction (LCB-TOP), and FinanceReasoning (Tang et al., 2025) for final evaluation.

Dataset Partition For seed data construction, we fix a budget of 1,000 instances per task and ensure that all seeds contain only raw questions and associated context without reference answers (examples in Appendix H). Specifically, for the Science task, we filter SciInstruct (Zhang et al., 2024) to retain instances with deterministic numeric answers, and then apply data selection strategies for quality. For the Code task, we draw seeds from LiveCodeBench releases v1–v6 via stratified sampling, further augmented with stratified samples from TACO (Li et al., 2023). For the Finance task, due to limited

Dataset Standardization We derive task descriptions from their official documentation and redesign the original evaluation logic to be fully rulebased by removing subjective or LLM judgment components. In addition, we provide a standardized sample submission file for each task that defines the required format for generated training data. Ultimately, we normalize each task as:

related resources, we sample half of FinanceReasoning as seed data. We then construct the public and private splits from SciBench, LCB-TOP, and the remaining portion of FinanceReasoning. The resulting Public Test Set and Private Test Set follow a 1:3 split ratio. Throughout seed construction and test-set splitting, we enforce strict stratified sampling and rigorously ensure zero overlap in problems and contexts to prevent data leakage.

##### 2.4 Automatic Data Engineering Agent

We investigate agentic data engineering under two representative scenarios: a single-turn completion setting (One-Shot) and a closed-loop, selfoptimizing setting (Iterative Agent), both illustrated in Figure 2 (b).

One-Shot In this setting, the agent generates the final submission in a single pass. We provide the agent with a comprehensive prompt with the necessary task input. The agent then drafts a strategy plan, implements it via code.py, and produces submission.json (Figure 2(b-ii)). We allow up to 8 independent attempts to mitigate generation failure. Once a valid submission is generated, the process terminates, and the submission is used to fine-tune the student model.

Iterative Agent In this setting, the agent is tasked with continuously enhancing model performance through a closed-loop data engineering process. Inspired by recent advances in self-improving agents (Madaan et al., 2023; Jiang et al., 2025), we investigate whether LLMs can apply such capabilities to data engineering by leveraging environmental feedback signals. To this end, we design the Iterative Agent, as illustrated in Figure 2(b-i), incorporating four operations:

- • Draft. Guided by the task settings and the dataset description, the agent formulates a new data synthesis strategy plan by outlining a plan and implementing it via executable code.
- • Debug. When the generated code throws an error during execution, the agent analyzes the traceback to diagnose and fix errors, ensuring the script executes successfully.
- • Repair. When the code executes successfully but the generated submission.json fails validation, the agent either refines the synthesis strategy to regenerate data or post-processes existing in-

stances in the raw data, ensuring the submission meets the required quantity and format.

• Improve. Leveraging environmental feedback, the agent employs iterative improvement: it applies a greedy strategy to select the solution with the highest public score from iteration history, consisting of the plan, code, and submission data, and optimizes it to evolve the synthesis strategy and enhance data quality.

Specifically, the process initiates with the Draft operation. The generated code for data curation first undergoes an execution check, and any failure triggers the Debug operation. Upon successful execution, if the output fails the submission validation check (i.e., <= 1,000 samples remain after format filtering), the process shifts to the Repair operation. We cap Debug and Repair operations at 3 consecutive attempts, restarting from Draft if this limit is exceeded. If the data validation check passes, the agent submits the curated data, receives feedback from the environment, and proceeds to the Improve operation accordingly. This iterative process enables the agent to simultaneously optimize synthesis strategies, prompt designs, and data distributions, continually driving student model specialization (see Appendix F for a running example).

#### 3 Experiments

##### 3.1 Metric Definition

We assess the agentic data engineering capability in a training-based setting, where the student model’s post-training performance gain directly reflects the agent’s effectiveness.

Relative Performance Gain (%) To enable consistent comparison across tasks, we report the relative performance gain of the student model:

Score(M⋆S) − Score(MS) Score(MS) × 100

Gain(%) =

(4) where MS denotes the initial student model and M⋆S denotes the specialized student model finetuned on the agent’s final data submission. Positive values indicate successful model specialization, whereas negative values indicate performance degradation. We follow each source benchmark’s official evaluation metric, with all tasks evaluated by accuracy. We also report the absolute accuracy of each run in Table 4 as a complementary view.

Specialization from Scratch Specialization with Seed Science Code Finance Avg. Science Code Finance Avg.

Agent Models

MATS Gain MATS Gain MATS Gain MATS Gain MATS Gain MATS Gain MATS Gain MATS Gain One-Shot

GPT-5.2 2.0 35.66 1.5 34.89 1.5 51.63 1.67 40.73 1.0 58.24 1.5 52.12 1.5 13.72 1.33 41.36 Qwen3-Max 1.0 37.69 1.5 32.03 3.5 30.00 2.00 33.24 2.0 49.34 2.0 32.77 1.5 59.39 1.83 47.17 DeepSeek-R1 2.5 47.94 2.0 4.75 2.0 -6.72 2.17 15.32 1.0 83.60 2.0 0.45 2.0 53.72 1.67 45.92 DeepSeek-V3.1 2.5 18.04 2.0 -4.58 3.0 24.03 2.50 12.50 2.5 52.78 1.0 19.83 2.5 59.25 2.00 43.95 Gemini-2.5-Pro 2.0 35.63 1.0 22.00 2.0 11.34 1.67 22.99 1.0 87.01 1.0 20.56 1.0 66.35 1.00 57.97 Claude-4-Sonnet 3.0 30.17 2.0 15.53 1.5 17.46 2.17 21.05 2.5 79.48 1.0 14.07 2.0 61.34 1.83 51.63

###### Iterative Agent

GPT-5.2 1.53 70.58 1.75 50.68 1.25 50.60 1.51 57.29 3.00 82.23 2.07 49.24 1.36 36.56 2.14 56.01 Qwen3-Max 1.58 58.24 1.83 34.16 1.41 39.84 1.61 44.08 1.75 74.67 2.58 20.56 1.23 61.33 1.85 52.19 DeepSeek-R1 2.13 37.01 3.63 11.92 1.56 26.56 2.44 25.16 2.75 89.76 4.00 19.10 1.65 54.92 2.80 54.59 DeepSeek-V3.1 7.13 45.91 2.17 24.10 1.80 40.28 3.70 36.76 1.47 78.14 1.24 29.15 1.31 65.66 1.34 57.65 Gemini-2.5-Pro 1.60 41.79 1.71 16.24 1.13 35.06 1.48 31.03 1.63 46.62 1.55 37.77 2.16 64.11 1.78 49.50 Claude-4-Sonnet 2.61 58.93 1.74 22.71 2.35 39.69 2.23 40.44 2.17 82.08 2.24 69.33 3.59 68.36 2.67 73.26

Table 1: Main Results. We report MATS (Mean Attempts to Successful Submission; lower is better) and relative performance Gain (%) over the base Llama-3.1-8B-Instruct model (higher is better), using Qwen3-30B-A3B as the unified teacher model. Results are averaged over two runs, with the raw accuracy scores reported in Table 4.

Mean Attempts to Success (MATS) MATS measures the average number of trial attempts to obtain a successful data submission. Given a run with N attempts, an attempt is marked successful if it generates a submission.json and the filtered submission retains at least 1,000 instances after format validation filtering. We report

N

MATS =

N i=1 I[succ(i)]

(5)

where succ(i) = 1 if the i-th attempt yields a successful submission and succ(i) = 0 otherwise.

##### 3.2 Experiment Setup

Execution Details We run experiments under the following budgets: 50,000 total teacher API calls per task (≤5,000 per attempt), 3-hour limit per code execution (i.e., data synthesis), and 12-hour timeout limit per run, terminating once any budget is exhausted. For the One-Shot scenario, we allow up to 8 attempts; Iterative Agents run for at most 30 iterations. We fine-tune Llama-3.1-8BInstruct as student model on 2× H100 GPUs and deploy Qwen3-30B-A3B as the teacher model on 2× H100 GPUs via vLLM with max concurrency of 80 (details in Appendix C). To verify generalization, we also evaluate alternative teacher-student configurations (see Appendix E). Each complete iteration cycle (synthesis → training → evaluation) takes 1–2 hours under this setting. We conduct two independent runs and report the final mean performance gain (raw accuracy scores in Table 4).

Data Initialization Settings We evaluate both agents under two distinct settings: (1) From Scratch: The agent must synthesize the entire dataset relying solely on the task description and teacher model API. (2) With Seed: The agent is additionally provided with a seed pool of 1,000 raw questions (as described in Section 2.3) to guide the data synthesis and exploration process.

##### 3.3 Main Results

Iterative optimization drives gains, while seed data ensures stability. As shown in Table 1, Iterative Agents consistently outperform One-Shot Agents. Specifically, in the from scratch regime, GPT-5.2 improves its average relative gain from 40.73% to 57.29%, demonstrating the efficacy of LLMs in leveraging environment feedback for selfimprovement. Compared with One-Shot generation, where a single error can corrupt the process (DeepSeek-V3.1’s -4.58% drop on Code), iterative mechanisms mitigate this by repeatedly improving exploration toward higher-quality solutions. Furthermore, adding a 1k seed pool consistently improves performance in both settings. This effect is strongest in the more fragile OneShot from scratch scenario, where most models obtain 30%+ additional relative gains after seeds are introduced. These results suggest that seed data serves to broaden the agent’s coverage of the target task distribution while injecting essential domainrelevant knowledge, thereby reducing off-target generation and low-quality instances.

Code Execution Timeout Code Extraction Failed Code Runtime Error LLM Output Truncated Generated Data Invalid Other Error

Gemini 2.5 pro Claude 4 sonnet Deepseek r1 Deepseek v3.1 Qwen 3 max Gpt 5.2

0.6 0.8 0.2 4.6

0.71 0.14 0.57 0.71

0.29 2.14 1 0.14

0.17

Code Execution Timeout Code Extraction Failed Code Runtime Error LLM Output Truncated Generated Data Invalid Other Error

Errors/RunsRatio

0.4

0.63 0.88 2.38 1.50 1.88

- 0.5 0.38 2.5 1.38 0.4 2 0.2

- 0.6 7.5

2.6 17.2 8 0.6

0.2

Gemini 2.5 pro Claude 4 sonnet Deepseek r1 Deepseek v3.1 Qwen 3 max Gpt 5.2

0.6 0.8 0.2 4.6

0.71 0.14 0.57 0.71

0.29 2.14 1 0.14

0.17

0.75 0.5 4

1 0.25

0.6 0.2

0.33 3.33 0.67

Errors/RunsRatio

0.4

0.25

0.63 0.88 2.38 1.50 1.88

- 0.5 0.38 2.5 1.38 0.4 2 0.2

- 0.6 7.5

2.6 17.2 8 0.6

0.2

0.8 0.25 3 0.25 1.75

0.25 3 0.25 1.75

0.57 0.86 0.29 0.11 1.33

0.8 0.6 0.2 0.83 0.33 2.5

0.2 0.2

0.2 0.6 0.2 0.86 0.14 1.57

0.75 0.5 4

1 0.25

0.6 0.2

0.33 3.33 0.67

0.25

0.8 0.25 3 0.25 1.75

0.25 3 0.25 1.75

(a) Science Domain (b) Code Domain (c) Finance Domain

0.57 0.86 0.29 0.11 1.33

0.8 0.6 0.2 0.83 0.33 2.5

0.2 0.2

0.2 0.6 0.2 0.86 0.14 1.57

(a) Science Domain (b) Code Domain (c) Finance Domain

Public Private Final

Public Private Final

Public Private Final

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

38.1

72.9

[Figure 237]

[Figure 238]

34.6 34.6

[Figure 239]

33.3

[Figure 240]

[Figure 241]

[Figure 242]

31.6 31.9

64.7 64.7

[Figure 243]

###### Accuracy(%)

Consistent Round

Public Private Final Public Private Final

Public Private Final

Consistent Round

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

38.1

72.9

[Figure 249]

[Figure 250]

Consistent Round

34.6 34.6

[Figure 251]

[Figure 252]

[Figure 253]

33.3

[Figure 254]

64.7 64.7

31.6 31.9

[Figure 255]

[Figure 256]

###### Accuracy(%)

[Figure 257]

[Figure 258]

Submission Round Submission Round

(a) Science Domain (b) Code Domain

(c) Finance Domain

Consistent Round Consistent Round

Consistent Round

[Figure 259]

[Figure 260]

[Figure 261]

Submission Round Submission Round Submission Round

(a) Science Domain (b) Code Domain (c) Finance Domain

Figure 3: Iteration analysis of performance across successful submissions produced by the Iterative Agent.

Instruction Difficulty 4o

Instruction Difficulty 4o

Method Source Pipeline Teacher Gain Human SciInstruct Human None 84.95 DataFlow None Human Qwen3-30B 65.82 Iterative (seed) SciInstruct GPT-5.2 Qwen3-30B 93.19 Iterative (scratch) None GPT-5.2 Qwen3-30B 76.76

[Figure 262]

[Figure 263]

Response Perplexity

Instruction diversity

Response Perplexity

Instruction diversity

Human-Consistent DIversity

Response Diversity

Response Quality 4o

Response Diversity

Response Quality 4o

Table 2: Comparison with human-involved settings. Detailed configurations see Appendix C.5.

Instruction Difficulty 4o

Instruction Difficulty 4o

Response Quality Skywork

Response Quality Skywork

[Figure 264]

[Figure 265]

#### 4 Further Analysis

Response Perplexity

Instruction diversity

Response Perplexity

Instruction diversity

Human Dataflow Iterative (seed) Iterative (scratch)

Submission-1 Submission-4 Submission-8

Performance

##### 4.1 Iterative Data Optimization Analysis

(a) Iteration Analysis (b) Human Intervention Analysis

Figure 4: Quality evaluation of synthesized instructions.

We conduct a controlled analysis to investigate how Iterative Agents improve and what specific aspects are optimized during iteration. Specifically, we increase the iteration and API-call limits and extend the time budget to 48 hours (Figure 3), with all runs generated using GPT-5.2. With public score already recorded during the execution loop, we reevaluate the student model on the private test set for every successful submission. We also report a final score which represents the private score of the best-performing submission on the public leaderboard up to time t, reflecting what the agent would actually select with the greedy strategy.

Response Diversity

Response Quality 4o

Response Diversity

Response Quality 4o

LLMs have emerged as independent data engineers for end-to-end model specialization. Even in the most fragile One-Shot from scratch setting, most agents still deliver positive average gains, and GPT 5.2 attains roughly a 40% improvement of the base model. Without external knowledge, environment feedback, or any humandesigned workflow, the submission must be produced by agent-written code independently. Under these constraints, the observed gains provide concrete evidence of non-trivial data engineering ability. The LLM agents can autonomously infer what supervision the model lacks, synthesize task-aligned instances, and curate a training set that generalizes to the hidden private task distribution.

Response Quality Skywork

Response Quality Skywork

Human-crafted SciInstruct Dataflow

Submission-1 Submission-4 Submission-8

Iterative with Seed

Iterative from Scratch

Iterative Agent demonstrates steady overall improvement across iterations. Figure 3 shows that public, private, and final scores exhibit a clear upward trend despite some fluctuations across iterations. Substantial gains typically emerge within the first 8 to 15 iterations, beyond which performance plateaus, indicating diminishing returns as the agent reaches the boundaries of its data awareness and cognitive capacity (Shinn et al., 2023).

Compared to stronger models, weaker models benefit more from sophisticated agent frameworks to unlock capabilities. Table 1 reveals a consistent interaction between base model capability and the effectiveness of complex agent frameworks. Weaker models experience substantially larger improvements from these advanced designs: DeepSeek-V3.1 surges from 12.50% in the oneshot from scratch baseline to 57.65% with iterative optimization and seed data, while stronger models such as GPT-5.2 and the Claude family show relatively modest gains under the same conditions. This pattern suggests that feedback-driven iterative optimization and seed data injection serve as critical guide rails for weaker models.

Greedy public-score selection ensures robustness performance. The fluctuations in Figure 3 reflect the intrinsic variance of synthetic data curation, where minor changes of prompts or generation pipelines can substantially shift answer correctness and the data distribution. For instance, in Figure 3 (b) (Round 6), a pipeline mistake causes a sharp drop in both public and private performance. The greedy selection rule mitigates such failures by

retaining the best historical submission based on public score (Chen et al., 2021). Given that public and private performance are largely aligned across iterations, this strategy yields a final curve that is noticeably more stable, thus less susceptible to occasional catastrophic regressions.

Iteration primarily drives improvements in data diversity. Following prior work (Kim et al., 2025), we conduct quality analysis on the generated submission data using six intrinsic metrics: instruction difficulty (GPT-4o assessment), instruction diversity (embedding similarity), response quality (GPT-4o assessment & Skywork reward model), response diversity (embedding similarity), and response perplexity (LLaMA-3.1-8B). The diagnostics reveal that both instruction and response diversity consistently increase over iterations, while response quality improves only marginally (example in Appendix F). Consistent with prior work (Yu

- et al., 2024), this indicates that iterations primarily expand and diversify synthesized questions rather than enhance quality for existing items.

##### 4.2 Human Involvement Influence Analysis

To systematically examine how varying degrees of human involvement influence data curation, we study several settings as follows:

- • Human. We build a 2k training set by sampling from SciInstruct with output-length filtering and diversity-aware clustering (Guha et al., 2025). We then use LLMs to rewrite instruction pairs into the target evaluation format. In this setting, both the data source and the synthesis pipeline are fully specified by humans.
- • DataFlow. DataFlow provides a general synthesis from scratch pipeline with predefined strategies for generation, filtering, and refinement. We adopt it as a strong method representing a humandesigned synthesis pipeline without relying on an external data source.
- • Iterative Agent (with seed / from scratch). We report the best-performing submission across all iterative rounds to approximate the current upper bound of the LLM data engineer without any human-designed strategies or recipes.

Fully autonomous data engineering shows potential to outperform human-involved methods. Under the same constraint, the pipeline designed by GPT-5.2 surpasses the human-designed DataFlow

framework (Table 2). First, LLMs can flexibly adapt their pipeline design strategy to the target task, automatically aligning the synthesized data to the appropriate domain, difficulty, and output format, rather than relying on rigid human-designed logic, which is consistent with ORPO (Yang et al., 2024). Second, environmental feedback acts as a closed-loop signal that mimics the self-reflection process (Shinn et al., 2023), enabling LLMs to continuously improve their data curation strategies and progressively shape the data distribution toward the specialized domain.

LLMs can match human-level data complexity, while falling short in generating diverse data. As illustrated in Figure 4(b), the from-scratch agent successfully approaches the human baseline in Instruction Difficulty, demonstrating its strong capability to self-curate challenging data. Nevertheless, it exhibits a performance drop in Instruction Diversity and Response Diversity compared to humaninvolved settings. This result reveals a basic limit of purely LLM-driven data engineering: the generated examples are high-quality but too repetitive.

##### 4.3 Failure Mode Analysis

Data Submission Failure As shown in Table 1, LLMs often fail to generate valid data in a single round. We perform detailed analysis based on the error-type breakdown (Figure 5).

- • Lack of Quantity Assurance Awareness. As shown in Figure 5, Insufficient Valid Samples dominates errors across most models (e.g., GPT-

5.2 and DeepSeek-R1). While agents aggressively filter generated data, they lack the awareness to validate the final dataset size and dynamically replenish discarded samples, ultimately failing the 1,000-instance quantity check.

- • Weak Format Handling in Complex Domains. Error distribution is domain-dependent. Insufficient Valid Samples is generally milder in textbased Finance tasks. Conversely, Science (requiring LaTeX) and Code (requiring executable logic) impose formatting constraints, causing massive data rejection and extraction failures.
- • Potential Over-Engineering Trap. We observe an anomalously high rate of LLM Output Truncated specifically for Claude-4-Sonnet (e.g., 52.63% in Code and 59.31% in Finance), revealing a tendency to design overly complex, verbose curation pipelines that exceed task requirements.

Code Execution Timeout Code Extraction Failed Code Runtime Error LLM Output Truncated Insufficient Valid Samples Other Error

|[Figure 266]<br><br>7.69 92.31 33.33 50.00 16.67<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>11.76 17.66 5.88 58.82 5.88 6.06 3.03 90.91<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>9.68 12.90 3.23 74.19<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>10.53 7.89 52.63 28.95<br><br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]|
|---|

|[Figure 287]<br><br>22.73 9.09 68.18 50.00 37.5 12.5 14.28 9.53 76.19 13.23 4.13 49.58 4.13 28.93<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>33.33 6.67 26.67 33.33 8.62 12.07 32.76 20.69 25.86<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]|
|---|

|[Figure 310]<br><br>33.33 5.56 61.11<br><br>[Figure 311]<br><br>[Figure 312]<br><br>[Figure 313]<br><br>12.29 14.29 14.29 42.86 14.29<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]<br><br>[Figure 318]<br><br>6.25 62.50 18.75 12.50 4.55 4.55 31.81<br><br>[Figure 319]<br><br>[Figure 320]<br><br>[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]<br><br>[Figure 324]<br><br>[Figure 325]<br><br>7.64 57.22 4.58 26.74 3.82<br><br><br>[Figure 326]<br><br>[Figure 327]<br><br>[Figure 328]<br><br>[Figure 329]<br><br>[Figure 330]<br><br>1.38<br><br>[Figure 331]<br><br>0.69<br><br>[Figure 332]<br><br>8.97 59.31 27.59 2.07<br><br>[Figure 333]<br><br>[Figure 334]<br><br>[Figure 335]<br><br>[Figure 336]<br><br>4.55<br><br>[Figure 337]<br><br>54.55|
|---|

Gpt 5.2

PercentageofErrors

Qwen3 max

Deepseek R1 Deepseek V3.1

4 7 1

3

Gemini 2.5 pro Claude 4 sonnet

(a) Science Domain (b) Code Domain (c) Finance Domain

Figure 5: Error type analysis of valid submission generation failure.

[Figure 338]

Model Specialization Failure Agents demonstrate capabilities for data engineering in most cases, but also encounter typical failure instances.

et al., 2026) and reusable skill acquisition (Yang et al., 2026; Zhang et al., 2026). A particularly relevant line of work investigates how agents continuously evolve through accumulated experience (Cai et al., 2025; Ouyang et al., 2026; Yue et al., 2026), constructing structured knowledge from past successes and failures to improve future performance. While these efforts focus on agents that consume data to complete tasks, our work studies agents that produce and optimize data as primary objective.

- • Distribution Shift of Data. In a from-scratch Science failure case (Appendix G.1), the agent hard-coded logic forcing 50% of the data budget into just five narrow topics (e.g., Boltzmann distribution). Instead of broad scientific sampling, this skewed generation caused a severe distribution shift. Consequently, it induced catastrophic forgetting (Luo et al., 2025), causing the student model to overfit to these specific sub-domains while losing broader competency.
- • Naive Rule-Based Augmentation. In a fromseed Code failure case (Appendix G.2), the agent employs a naive regex strategy to indiscriminately perturb numerical values, ignoring their distinct roles in control flow. This severed the semantic link between instructions and executable logic, directly violating the SECON principle (Zhang et al., 2025). Consequently, instead of valid data expansion, the agent injected syntactically broken noise, severely degrading the student model’s performance.

Data Centric AI. In the early stages of LLM development, training heavily relied on high-quality human-annotated data (Stiennon et al., 2020). With the rapid depletion of naturally occurring human text, a crisis of data scarcity has become imminent (Villalobos et al., 2024). Pioneering work such as Self-Instruct (Wang et al., 2023) introduced a paradigm shift, demonstrating that LLMs can synthesize training data from a small seed set to optimize themselves (Taori et al., 2023; Gunasekar et al., 2023). More recent studies further improve the automated generation of synthetic data (Huang et al., 2025; Liang et al., 2025; Kessler et al., 2025), yet remain fundamentally dependent on data synthesis pipelines or recipes designed by humans. Meanwhile, the search-verify-feedback paradigm (Guan et al., 2024) highlights the importance of automated verification as a supervision signal for post-training. Autodata (Kulikov et al., 2026) further introduces a multi-agent system with predefined roles that iteratively generates challenging examples, though the architecture and quality criteria remain human-specified. Concurrent to our work, PostTrainBench (Rank et al., 2026) benchmarks whether LLM agents can automate the full post-training pipeline with unconstrained choice over training methods, hyperparameters, and data sources. DataPrep-Bench (Liang et al., 2026) compares diverse data construction methods and quality metrics under a unified downstream-grounded protocol. In contrast, our work fixes the training pipeline, parameter settings, teacher model, and

Overall, despite demonstrated capability to guide end-to-end data curation, LLMs still lack robust post-generation safeguards for stringent quality assurance and reliable quantity control.

#### 5 Related Work

LLM Agent. LLM agents (Wang et al., 2024a; Xi

- et al., 2025) leverage the reasoning capabilities of foundation models (Zheng et al., 2025), integrated with external tools (Schick et al., 2023) and environmental feedback (Yao et al., 2023) to complete tasks. This revolution has catalyzed the emergence of specialized agents across diverse domains, ranging from autonomous data analysis (Zhang et al.,

2023) to scientific discovery (Boiko et al., 2023), with growing attention to agent memory (Chen

student model, isolating data curation as the sole variable to study whether individual LLM agents can iteratively optimize training data driven by the downstream student model feedback.

#### 6 Conclusion

We present a systematic analysis of Agentic Data Engineering for Model Specialization by requiring LLM agents to conduct end-to-end data engineering in a closed loop. Our results across Science, Code, and Finance show that iterative agents consistently yield stronger and more stable specialization, with feedback-driven iteration improving both data strategy and alignment. In particular, GPT-5.2 reaches a 57.29% average gain, demonstrating that LLM agents can autonomously author data curricula that drive substantial student model specialization. Our further failure analysis of the observed dominance of invalid submissions and failed specializations exposes the lack of data assurance awareness of current LLMs.

#### Limitations

We acknowledge several limitations in our work. First, although focusing on QA tasks allows us to efficiently obtain reliable environmental feedback for closed-loop optimization, this design restricts our evaluation on open-ended generation tasks where automated evaluation is difficult to achieve. Second, despite implementing strict budget caps, the Iterative Agent still demands considerable computational resources for model inference and fine-tuning. Finally, while we average the results across multiple runs to mitigate fluctuations, coupling complex end-to-end data engineering tasks still introduces unavoidable run-to-run variance. We leave broader task coverage and more cost-efficient strategies to future work.

#### References

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. 2025. τ2-bench: Evaluating conversational agents in a dual-control environment. CoRR, abs/2506.07982.

Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. 2023. Autonomous chemical research with large language models. Nat., 624(7992):570– 578.

Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. 2025. FLEX: continuous

agent evolution via forward learning from experience. CoRR, abs/2511.06449.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Yining Chen, Jihao Zhao, Bo Tang, Haofen Wang, Feiyu Xiong, and Zhiyu Li. 2026. Memprivacy: Privacypreserving personalized memory management for edge-cloud agents. arXiv preprint arXiv:2605.09530.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948.

Xinyan Guan, Yanjiang Liu, Xinyu Lu, Boxi Cao, Ben He, Xianpei Han, Le Sun, Jie Lou, Bowen Yu, Yaojie Lu, and Hongyu Lin. 2024. Search, verify and feedback: Towards next generation post-training paradigm of foundation models via verifier engineering. CoRR, abs/2411.11504.

Etash Kumar Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, and 31 others. 2025. Openthoughts: Data recipes for reasoning models. CoRR, abs/2506.04178.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. 2023. Textbooks are all you need. CoRR, abs/2306.11644.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. Metagpt: Meta programming for A multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen, Qihui Zhang, Yao Wan, Tianyi Zhou, Chaowei Xiao, Jianfeng Gao, Lichao Sun, and Xiangliang Zhang. 2025. Datagen: Unified synthetic dataset generation via large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. 2025. AIDE: ai-driven exploration in the space of code. CoRR, abs/2502.13138.

Samuel Kessler, Menglin Xia, Daniel Madrigal Díaz, Dongge Han, Helia Heshemi, Saravan Rajmohan, Victor Rühle, and Jordan T. Ash. 2025. Towards active synthetic data generation for finetuning language models. CoRR, abs/2512.00884.

Seungone Kim, Juyoung Suk, Xiang Yue, Vijay Viswanathan, Seongyun Lee, Yizhong Wang, Kiril Gashteovski, Carolin Lawrence, Sean Welleck, and Graham Neubig. 2025. Evaluating language models as synthetic data generators. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 6385–6403. Association for Computational Linguistics.

Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, and Jason Weston. 2026. Autodata: an automatic data scientist to create high quality data.

Chenxi Li, Yuanhe Tian, Zhaxi Zerong, Yan Song, and Fei Xia. 2024. Challenging large language models with new tasks: A study on their adaptability and robustness. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, Findings of ACL, pages 8140–8162. Association for Computational Linguistics.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. 2023. TACO: topics in algorithmic code generation dataset. CoRR, abs/2312.14852.

Hao Liang, Qifeng Cai, Yibo Lin, Jianzhuo Du, Qifeng Xia, Sizhe Qiu, Linzhuang Sun, Meiyi Qiang, Zhaoyang Han, Xiaochen Ma, Bohan Zeng, Ruichuan An, Conghui He, and Wentao Zhang. 2026. DataPrep-Bench: Benchmarking LLMs as Training Data Preparators. https://datapreparationbench.github.io/ assets/DataPrep-Bench.pdf. Preprint.

Hao Liang, Xiaochen Ma, Zhou Liu, Zhen Hao Wong, Zhengyang Zhao, Zimo Meng, Runming He, Chengyu Shen, Qifeng Cai, Zhaoyang Han, and 1 others. 2025. Dataflow: An llm-driven framework

for unified data preparation and workflow automation in the era of data-centric ai. arXiv preprint arXiv:2512.16676.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2025. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Transactions on Audio, Speech and Language Processing.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 3470–3487. Association for Computational Linguistics.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-Tau Yih, Sida I. Wang, and Xi Victoria Lin. 2023. LEVER: learning to verify language-tocode generation with execution. In International Conference on Machine Learning, ICML 2023, 2329 July 2023, Honolulu, Hawaii, USA, Proceedings of Machine Learning Research, pages 26106–26128. PMLR.

Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, ChunLiang Li, Yizhu Jiao, Kaiwen Zha, and 1 others. 2026. Skillos: Learning skill curation for self-evolving agents. arXiv preprint arXiv:2605.06614.

Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, and Huajun Chen. 2024. Autoact: Automatic agent learning from scratch for QA via self-planning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3003–3021. Association for Computational Linguistics.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina Nguyen, Matthias Bethge, and Maksym Andriushchenko. 2026. Posttrainbench: Can LLM agents automate LLM post-training? CoRR, abs/2603.08640.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Zichen Tang, Haihong E, Ziyan Ma, Haoyang He, Jiacheng Liu, Zhongjun Yang, Zihua Rong, Rongjin Li, Kun Ji, Qing Huang, Xinyang Hu, Yang Liu, and Qianhe Zheng. 2025. Financereasoning: Benchmarking financial numerical reasoning more credible, comprehensive and challenging. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 15721–15749. Association for Computational Linguistics.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpaca: A Strong, Replicable Instruction-Following Model.

Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. 2024. Position: Will we run out of data? limits of LLM scaling based on human-generated data. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. 2024a. A survey on large language model based autonomous agents. Frontiers Comput. Sci., 18(6):186345.

Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba,

Shichang Zhang, Yizhou Sun, and Wei Wang. 2024b. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13484–13508. Association for Computational Linguistics.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, and 9 others. 2025. The rise and potential of large language model based agents: a survey. Sci. China Inf. Sci., 68(2).

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. Preprint, arXiv:2309.03409.

Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. Fingpt: Open-source financial large language models. CoRR, abs/2306.06031.

Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, and Liang He. 2026. Autoskill: Experience-driven lifelong learning via skill selfevolution. CoRR, abs/2603.01145.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Simon Yu, Liangyu Chen, Sara Ahmadian, and Marzieh Fadaee. 2024. Diversify and conquer: Diversitycentric data selection with iterative refinement. Preprint, arXiv:2409.11378.

Zhenrui Yue, Kartikeya Upasani, Xianjun Yang, Suyu Ge, Shaoliang Nie, Yuning Mao, Zhe Liu, and Dong Wang. 2026. Dr. zero: Self-evolving search agents without training data. CoRR, abs/2601.07055.

Dan Zhang, Ziniu Hu, Sining Zhoubian, Zhengxiao Du, Kaiyu Yang, Zihan Wang, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024. Sciinstruct: a self-reflective instruction annotated dataset for training scientific language models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Kangning Zhang, Shuai Shao, Qingyao Li, Jianghao Lin, Lingyue Fu, Shijian Wang, Wenxiang Jiao, Yuan Lu, Weiwen Liu, Weinan Zhang, and 1 others. 2026. Mmskills: Towards multimodal skills for general visual agents. arXiv preprint arXiv:2605.13527.

Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. 2023. Data-copilot: Bridging billions of data and humans with autonomous workflow. CoRR, abs/2306.07209.

Xu Zhang, Zexu Lin, Xiaoyu Hu, Jianlei Wang, Wenpeng Lu, and De-Yu Zhou. 2025. Secon: Maintaining semantic consistency in data augmentation for code search. ACM Trans. Inf. Syst., 43(2).

Da Zheng, Lun Du, Junwei Su, Yuchen Tian, Yuqi Zhu, Jintian Zhang, Lanning Wei, Ningyu Zhang, and Huajun Chen. 2025. Knowledge augmented complex problem solving with large language models: A survey. CoRR, abs/2505.03418.

Qiannan Zhou, Fei Xu, Lingxuan Weng, Ruixing Li, Xudong Wu, Li Chen, Zhi Zhou, and Fangming Liu. 2025. Espresso: Cost-efficient large model training by exploiting GPU heterogeneity in the cloud. In IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, London, United Kingdom, May 19-22, 2025, pages 1–10. IEEE.

#### A Dataset Details

To systematically analyze the capability of LLM agents in end-to-end data engineering, we curate datasets across three specialized domains: Science, Code, and Finance. We elaborate on the source datasets and the rationale behind our construction choices below; the standardization protocol and partition rules follow Section 2.3.

Science. We build the Science task upon SciBench (Wang et al., 2024b) and SciInstruct (Zhang et al., 2024). SciBench evaluates college-level scientific reasoning across physics, chemistry, and mathematics, providing a rigorous testbed for model specialization, while SciInstruct serves as a diverse instruction-tuning corpus suitable for seed construction. Since our environment requires deterministic rule-based scoring, we filter SciInstruct to retain instances with definitive numeric answers before applying quality-aware selection to form the seed pool. SciBench is used exclusively for evaluation, with zero overlap in problems and contexts against the seed pool.

Code. For the programming domain, we adopt LiveCodeBench (LCB) (Jain et al., 2025) and TACO (Li et al., 2023). We focus on the Test Output Prediction sub-task of LCB (LCB-TOP),

which requires predicting program execution outputs and thus demands deep algorithmic understanding. Seed instances are drawn from LCB releases v1–v6 via stratified sampling, augmented with stratified samples from TACO to broaden diversity. The evaluation sets are constructed exclusively from LCB-TOP under the same zero-overlap constraint.

Finance. For the financial domain, we adopt FinanceReasoning (Tang et al., 2025), which targets deep financial logic, numerical reasoning, and domain-specific text comprehension. As highquality open-source resources for complex financial reasoning are scarce, both the seed and the evaluation sets are derived from FinanceReasoning via disjoint stratified splits, ensuring no contextual leakage between the seed pool and the test sets.

#### B Main Result Details

###### Task Benchmark Metric Pri. Score

Science SciBench Acc. (%) 16.74 Code LCB-TOP Acc. (%) 21.18 Finance FinanceReasoning Acc. (%) 39.93

Table 3: Base-model performance on each task. We use LLaMA-3.1-8B-Instruct as the unified base model.

- Table 3 details the baseline performance of our unified backbone model, LLaMA-3.1-8B-Instruct, across the three target domains. These scores reflect the model’s capabilities on the private test sets prior to any instruction tuning or distillation. Specifically, the base model achieves accuracy scores of 16.74 on science domain (SciBench), 21.18 on code domain (LCB-TOP), and 39.93 on finance domain (FinanceReasoning). These results establish a performance baseline, highlighting the challenges of these specialized tasks for the off-the-shelf model and serving as a reference for quantifying the improvements gained through our synthetic data generation methods.
- Table 4 provides the comprehensive raw data from our main experiments. We report two key metrics: MATS (Mean Attempts to Success Submission), which quantifies the efficiency of the agent in generating valid datasets, and Accuracy (%), which measures the performance of the student model fine-tuned on the synthesized data. The results

cover both Specialization from Scratch and Specialization with Seed settings across all three domains. To demonstrate the stability of our approach, we report results from at least two independent runs for each agent configuration.

#### C Experiment Configuration Details

In this section, we provide a comprehensive breakdown of the experimental configurations used in our study. This includes the hyperparameter settings for our data synthesis agents (both One-Shot and Iterative), the specific configurations for model training via LoRA, and the inference parameters employed for both the teacher and student models.

##### C.1 One-Shot Agent Configuration

The One-Shot Agent represents a baseline approach where the synthetic dataset is generated in a single pass without iterative refinement. The configuration is designed to balance generation speed with robustness against potential code execution failures.

Key configuration details include:

- • Resource Constraints: We limit the total runtime to 12 hours (Max Time Hours) and the dataset size to 2,000 samples (Dataset Size). We enforce a 3-hour limit per code execution (data synthesis) and a strict budget of 50,000 total teacher API calls per task (≤ 5,000 per attempt).
- • Teacher Model: We utilize Qwen3-30B-A3B as the teacher model. To maximize throughput during the data generation phase, we set the Api Concurrency to 80, allowing parallel processing of multiple data points via the vLLM deployment.
- • Robustness Mechanism: We set Max Generation Attempts to 8. This allows the agent to retry the generation process up to 8 times if the code fails to execute or produces invalid JSON output.
- • Student Model Environment: The student model serves as the validator. It is hosted locally using vLLM with Vllm Max Num Seqs set to 128 to optimize inference throughput during the validation phase.

The exact configuration file is presented in Listing C.1.

Detailed Configuration for the One-Shot Agent.

# General Settings common:

DATASET_SIZE: 2000 MAX_TIME_HOURS: 12 EXECUTION_TIMEOUT_MIN: 180 MAX_GENERATION_ATTEMPTS: 8

# Teacher Model Settings (Generator) teacher:

TEACHER_MODEL: Qwen3-30B-A3B TOTAL_API_LIMIT: 50000 SESSION_API_LIMIT: 5000 API_CONCURRENCY: 80

# Student Model Settings (Validator) student:

LOCAL_MODEL: Llama-3_1-8B-Instruct VLLM_PORT: 8099 VLLM_MAX_NUM_SEQS: 128 VLLM_CONCURRENCY: 64 VLLM_MAX_TOKENS: 8192

##### C.2 Iterative Agent Configuration

The Iterative Agent introduces a feedback loop where the agent analyzes the performance of the trained model and refines the dataset accordingly. This requires a more sophisticated configuration to handle the cycle of drafting, training, evaluating, and improving.

Key distinctions in the configuration include:

- • Self-Correction Loop: The agent runs for at most 30 iterations. Within each iteration, the agent may encounter errors in its proposed code. We define Max Debug Attempts as 3 for self-correction within a single cycle.
- • Resource Allocation: The system adheres to the global 12-hour time limit and monitors the Total Api Limit of 50,000 calls to decide whether to continue refining or finalize the submission. Each complete iteration cycle takes approximately 1–2 hours.

The specific parameters are detailed in Listing C.2.

Specialization from Scratch Specialization with Seed

Agent Models Runs

Science Code Finance Science Code Finance MATS Acc(%) MATS Acc(%) MATS Acc(%) MATS Acc(%) MATS Acc(%) MATS Acc(%)

One-Shot

- run1 1.00 15.83 2.00 27.96 1.00 59.83 1.00 30.73 2.00 32.22 2.00 46.60

- run2 2.00 29.59 1.00 29.18 2.00 61.26 1.00 22.25 1.00 32.22 1.00 44.22

GPT-5.2 Qwen3-Max DeepSeek-R1 DeepSeek-V3.1 Gemini-2.5-Pro Claude-4-Sonnet

- run1 1.00 27.29 1.00 27.05 1.00 45.89 1.00 28.67 3.00 27.36 2.00 62.81

- run2 1.00 18.81 2.00 28.88 6.00 57.93 3.00 21.33 1.00 28.88 1.00 64.48

- run1 3.00 27.06 2.00 20.97 2.00 46.60 1.00 31.65 2.00 30.09 3.00 57.09

- run2 2.00 22.47 2.00 23.40 2.00 27.89 1.00 29.82 2.00 12.46 1.00 65.67

- run1 2.00 30.05 1.00 29.79 4.00 65.20 3.00 17.66 3.00 18.54 3.00 54.59

- run2 3.00 21.10 1.00 20.97 1.00 61.98 2.00 21.86 1.00 21.88 3.00 44.46

- run1 2.00 17.66 1.00 25.23 2.00 42.07 2.00 30.05 1.00 29.79 4.00 65.20

- run2 2.00 27.75 1.00 26.44 2.00 46.84 3.00 21.10 1.00 20.97 1.00 61.98

- run1 3.00 20.64 2.00 23.71 2.00 46.01 4.00 28.67 1.00 37.08 2.00 60.55

- run2 3.00 22.94 2.00 25.23 1.00 47.79 1.00 31.42 1.00 11.24 2.00 68.30 Iterative Agent

- run1 1.80 25.46 1.00 30.09 1.17 60.79 3.50 32.34 1.80 31.00 1.60 48.75

- run2 1.25 31.65 2.50 33.74 1.33 59.48 2.50 28.67 2.33 32.22 1.11 60.31

GPT-5.2 Qwen3-Max DeepSeek-R1 DeepSeek-V3.1 Gemini-2.5-Pro Claude-4-Sonnet

- run1 1.17 28.21 1.00 23.40 1.06 53.40 1.50 31.19 3.50 33.74 1.33 63.17

- run2 2.00 24.77 2.67 33.43 1.75 58.28 2.00 27.29 1.67 17.33 1.13 65.67

- run1 2.00 25.69 3.25 25.23 1.40 49.82 3.00 32.11 1.00 30.39 2.22 58.05

- run2 2.25 20.18 4.00 22.18 1.73 51.25 2.50 31.42 7.00 20.06 1.07 65.67

- run1 1.25 17.89 2.33 29.78 1.80 61.26 1.15 26.15 1.14 24.92 1.05 66.98

- run2 13.00 30.96 2.00 22.79 1.80 50.77 1.80 33.49 1.33 29.79 1.57 65.32

- run1 1.60 19.72 1.71 22.49 1.06 46.48 1.75 33.03 1.50 32.22 1.92 66.70

- run2 1.60 27.75 1.71 26.74 1.20 61.38 1.50 16.06 1.60 26.14 2.40 64.36

- run1 4.00 26.15 1.27 26.14 3.14 50.77 3.33 28.44 1.14 39.21 3.17 66.15

- run2 1.21 27.06 2.20 25.84 1.56 60.79 1.00 32.52 3.33 32.52 4.00 68.30

Table 4: Raw scores of the main experiment. We report MATS (Mean Attempts to Success Submission) and the Accuracy (Acc, %) of the fine-tuned student model LLaMA-3.1-8B-Instruct. Scores of the original student model are reported in Table 3. The teacher model is set to Qwen3-30B-A3B globally.

Detailed Configuration for the Iterative Agent.

# General Settings common:

DATASET_SIZE: 2000 MAX_TIME_HOURS: 12 MAX_ITERATIONS: 30 MAX_DEBUG_ATTEMPTS: 3

# Teacher Model (Data Generator) teacher:

TEACHER_MODEL: Qwen3-30B-A3B API_CONCURRENCY: 80

# Agent Core (Controller) iterative-agent:

AGENT_MODEL: your_agent_model env_vars: <<: [

- *common_settings,
- *teacher_config,
- *student_config ]

##### C.3 Model Training Parameters

All synthetic datasets are evaluated by training a student model using Supervised Fine-Tuning (SFT). We adopt a parameter-efficient approach using Low-Rank Adaptation (LoRA) on 2× H100 GPUs.

The complete training configuration is provided in Listing C.3. And the training configuration is standardized as follows:

- • Optimization Strategy: We use the AdamW optimizer with a learning rate of 1.0 × 10−4 and a cosine learning rate scheduler. A warmup ratio of 0.1 is applied to stabilize the early training phase.
- • LoRA Configuration: We target all linear layers (Lora Target: all) with a rank of 8 (Lora Rank) and an alpha of 16 (Lora Alpha). This configuration provides a good balance between adaptation capacity and parameter efficiency.
- • Compute Efficiency: To accommodate hardware constraints, we set the per-device batch

size to 1 but employ Gradient Accumulation Steps of 8. Training is performed in Bf16 precision to reduce memory usage. The model is trained for 3 epochs with evaluations performed every 500 steps.

Hyperparameters for Model Training (SFT with LoRA).

# Training Stage stage: sft finetuning_type: lora lora_target: all lora_rank: 8 lora_alpha: 16 lora_dropout: 0

# Hyperparameters learning_rate: 1.0e-4 num_train_epochs: 3 lr_scheduler_type: cosine warmup_ratio: 0.1 bf16: true

# Batch Size & Gradient Accumulation per_device_train_batch_size: 1 gradient_accumulation_steps: 8 cutoff_len: 2048

##### C.4 Model Inference Parameters

The inference process involves two distinct phases: data generation (Teacher) and model evaluation (Student). The parameters for each phase are optimized for their specific objectives.

- C.4.1 Teacher Model Inference The teacher model (Qwen3-30B-A3B) is deployed on separate 2× H100 GPUs via vLLM. The inference parameters are managed dynamically by the agent code:

- • Concurrency: We utilize an asyncio.Semaphore with a limit of 80 (Api Concurrency) to maximize generation throughput against the vLLM server.
- • Sampling: The teacher model uses standard sampling parameters to ensure diversity in the generated synthetic data.

- C.4.2 Student Model Inference (Evaluation) The student model is deployed locally using the vLLM engine. The configuration focuses on memory stability and evaluation throughput:

- • System Configuration: We explicitly set Gpu Memory Utilization to 0.85 to reserve GPU memory for activation overheads, preventing Out-Of-Memory (OOM) errors during longsequence processing. The Max Num Seqs is set to 128 to fully utilize the GPU’s parallel processing capacity.
- • Tensor Parallelism: The system automatically detects the number of available GPUs and scales the Tensor Parallel Size accordingly.
- • Dynamic Sampling: During evaluation, sampling parameters (e.g., temperature) are not hardcoded but are generated dynamically based on the specific dataset requirements (defined in evaluate.py).

These settings are summarized in Listing C.4.2.

Inference Parameters for Teacher and Student Models.

# Teacher Model teacher_inference:

model: Qwen3-30B-A3B deployment: vLLM (2x H100) concurrency_limit: 80 temperature: default

# Student Model student_system:

engine: vLLM gpu_memory_utilization: 0.85 max_model_len: 8192 max_num_seqs: 128 tensor_parallel_size: auto

student_sampling: temperature: dynamic top_p: dynamic

##### C.5 Human Involvement Analysis Setup

Across all settings, we fix the teacher model to Qwen3-30B-A3B. We cap the API budget at 5,000 calls for Iterative Agent, and assign DataFlow with official configuration of 6,000 calls. Each final synthesized training set contains 2,000 examples. The student model and training & inference parameters are kept identical across all settings.

#### D Platform Design

Our benchmark is accompanied by an execution platform that enables agents to synthesize data and run end-to-end model training within a controlled environment (Figure 2 (a)).

Agent Toolkit We provide a set of programmatic tools that agents can directly invoke, including check_submission (format and schema validation), train_model (fine-tuning on the submitted data), evaluate_dataset (public set evaluation and bad case demonstration), and api_count (APIusage tracking). We additionally ship a lightweight library with pre-defined helper functions for common operations during code generation, such as batched API calls and robust answer parsing. These tools allow an LLM agent to implement its own data-synthesis logic while seamlessly integrating with the full pipeline.

Execution & Evaluation Environment We offer an isolated execution environment that separates both (i) the per-run workspace and (ii) the runtime execution context at the thread level, enabling safe concurrent runs while preventing data contamination. During execution, the platform continuously monitors API calls and remaining time, automatically terminating runs that exceed the prescribed budgets.

After each run completes, we replay the evaluation on the final submission.json by retraining the model on the submitted dataset and re-evaluating it on the held-out private test set, ensuring that the reported gains are attributable to the data.

#### E Generalization Across Different Teacher-Student Configurations

In our main experiments, we employ a fixed teacher-student model configuration. This serves as a controlled evaluation setup to provide a consistent scale for comparison, following common practices in prior empirical studies (e.g., using a fixed user simulator in τ2-Bench (Barres et al., 2025)).

To demonstrate that our findings are robust and not overfit to a specific model pair, we conduct additional experiments using the GPT-5.2 agent under varying teacher and student configurations on Science task. Results summarized in Table 5.

As shown in Table 5, while the absolute value of the Gain (%) naturally varies depending on the base capabilities of the chosen models, the overall

Agent Teacher Student Gain (%)

One-Shot Qwen3-30B-A3B Qwen2.5-1.5B-Instruct 51.89 Iterative Qwen3-30B-A3B Qwen2.5-1.5B-Instruct 59.43

One-Shot DeepSeek-V3.2 LLaMA-3.1-8B-Instruct 24.43 Iterative DeepSeek-V3.2 LLaMA-3.1-8B-Instruct 53.16

Table 5: Performance gain of the GPT-5.2 agent across different teacher and student model configurations. The iterative approach consistently outperforms the one-shot baseline regardless of the underlying models.

trends remain highly consistent. Specifically, the iterative agent consistently yields positive gains and maintains a significant performance gap over the one-shot baseline across all configurations. These results confirm that the fixed teacher-student setting in our main text successfully serves its purpose of providing a consistent evaluation scale, and the effectiveness of autonomous data engineering generalizes well to other model combinations.

#### F Running Example of Iterative Agent

To clearly demonstrate how the Iterative Agent optimizes its own data synthesis pipeline, we have selected an optimization example of GPT-5.2 on the Science task to illustrate the specific details of the iterative improvement.

Here is the original plan drafted by the model: Original Plan of GPT-5.2 in Science Task.

##### 1. Define the instruction format:

• Define a single fixed "instruction" string matching sample_submission.json style

##### 2. Build topic templates spanning:

- • Mathematics: calculus (integrals, series, multivariable), differential equations, linear algebra, probability/statistics
- • Physics: mechanics, E&M, circuits, waves/optics, thermodynamics
- • Chemistry: gases (vdW/ideal), equilibrium, kinetics, thermodynamics, electrochemistry, colligative properties

##### 3. Generate parameterized prompts (per template) requiring the teacher to:

- • Write a textbook-style problem with given numbers and a specified answer unit
- • Provide a step-by-step solution including computations
- • End with: The answer is therefore \boxed{...}. where ... is a decimal with exactly three digits

##### 4. Bulk generation:

• Use api_generate_batch with batch size 80 to produce ∼2200 candidates (buffer for filtering)

##### 5. Post-process and filter:

- • Keep only items whose output contains at least one \boxed{...} and whose last \boxed{} matches a decimal number with 3 digits
- • Ensure output ends with the exact final sentence
- • Ensure fields instruction, input, output are non-empty strings

##### 6. Regenerate if needed:

• If < 2000 valid samples, regenerate only the deficit with stricter formatting reminders

##### 7. Finalize dataset:

• Shuffle, truncate to first 2000, and write to ../submission/submission.json

##### 8. Checkpointing:

• Save checkpoints every 200 valid samples to avoid data loss

Based on self-reflection and environmental feedback, GPT-5.2 proposed the optimization approach shown below based on the original solution:

Improvement Plan of GPT-5.2 in Science Task.

##### 1. Improve solution quality by:

- • Generating more focused, step-by-step solutions without excessive verbosity
- • Ensuring all calculations are complete and accurate
- • Requiring clear final answers in proper boxed format

##### 2. Enhance problem diversity by:

- • Creating problems across broader difficulty ranges
- • Including more applied/real-world scientific scenarios
- • Balancing theoretical and computational problems

##### 3. Better prompt engineering:

- • More specific instructions for concise, accurate solutions
- • Explicit requirements for complete calculations
- • Template-based solution structure to ensure consistency

##### 4. Quality control:

- • Filter out incomplete or malformed solutions
- • Validate that solutions have proper final answers
- • Ensure mathematical notation is correct

Guided by the improvement plan above, the model generated a more complex and robust code version that covers a broader scope and includes more challenging questions. The improved code corresponding to this optimization is shown below.

Listing 1: Correnponding Code to Improve Data Quality

- 1 import os

- 2 import re

- 3 import json

- 4 import random

- 5 import hashlib

- 6 from typing import List, Dict, Tuple

- 7 from services.library import api_generate_batch, get_available_api_calls

- 8

- 9 OUT_PATH = "../submission/submission.json"

- 10 API_BATCH = 80

- 11 TARGET_N = 2000

- 12 MAX_API_CALLS_THIS_ITER = 5000

- 13

- 14 INSTRUCTION = (

- 15 "Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. "

- 16 "The problem will specify the unit of measurement, which should not be included in the answer. "

- 17 "Express the final answer as a decimal number with three digits after the decimal point. "

- 18 "Conclude the answer by stating \"The answer is therefore \\\\boxed{[ANSWER]}.\""

- 19 )

- 20

- 21 BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

- 22 THREE_DEC_RE = re.compile(r"^-?\d+\.\d{3}$")

- 23 UNIT_SENT_RE = re.compile(r"The unit of the answer is\s+[^.]+\.", re.IGNORECASE)

- 24

- 25 def extract_last_boxed(text: str):

- 26 boxes = BOX_RE.findall(text or "")

- 27 return boxes[-1].strip() if boxes else None

- 28

- 29 def required_final_sentence(boxed_value: str) -> str:

- 30 return f"The answer is therefore \\\\boxed{{{boxed_value}}}."

- 31

- 32 def ends_with_required_sentence(text: str) -> bool:

- 33 last = extract_last_boxed(text)

- 34 if last is None:

- 35 return False

- 36 return text.rstrip().endswith(required_final_sentence(last))

- 37

- 38 def is_valid_item(item: Dict[str, str]) -> bool:

- 39 if not isinstance(item, dict):

- 40 return False

- 41 for k in ("instruction", "input", "output"):

- 42 if k not in item or not isinstance(item[k], str) or not item[k].strip():

- 43 return False

- 44

- 45 out = item["output"]

- 46 last = extract_last_boxed(out)

- 47 if last is None or not THREE_DEC_RE.match(last):

- 48 return False

- 49 if not ends_with_required_sentence(out):

- 50 return False

- 51

- 52 # Input should look like a question and include unit statement

- 53 inp = item["input"]

- 54 if "Question:" not in inp:

- 55 return False

- 56 if not UNIT_SENT_RE.search(inp):

- 57 return False

- 58

- 59 # Ensure boxed answer has no units (just numeric already enforced)

- 60 return True

- 61

- 62 def normalize_text(s: str) -> str:

- 63 return re.sub(r"\s+", " ", (s or "").strip())

- 64

- 65 def fingerprint_input(inp: str) -> str:

- 66 return hashlib.md5(normalize_text(inp).encode("utf-8")).hexdigest()

- 67

- 68 def save_json(data: List[Dict[str, str]], path: str):

- 69 os.makedirs(os.path.dirname(path), exist_ok=True)

- 70 with open(path, "w", encoding="utf-8") as f:

- 71 json.dump(data, f, ensure_ascii=False, indent=2)

- 72

- 73 def load_existing(path: str) -> List[Dict[str, str]]:

- 74 if not os.path.exists(path):

- 75 return []

- 76 with open(path, "r", encoding="utf-8") as f:

- 77 obj = json.load(f)

- 78 return obj if isinstance(obj, list) else []

- 79

- 80 # ---------- Improved prompting ----------

- 81 TOPICS = [

- 82 # Mathematics

- 83 ("Mathematics", "Calculus", "definite integral (substitution/parts)"),

- 84 ("Mathematics", "Calculus", "improper integral / convergence"),

- 85 ("Mathematics", "Series", "Taylor/Maclaurin approximation"),

- 86 ("Mathematics", "Multivariable", "double integral / change of variables"),

- 87 ("Mathematics", "Optimization", "Lagrange multipliers"),

- 88 ("Mathematics", "Differential Equations", "first-order linear IVP with application"),

- 89 ("Mathematics", "Differential Equations", "second-order ODE (mass-spring/damping)"),

- 90 ("Mathematics", "Linear Algebra", "eigenvalues/eigenvectors numeric"),

- 91 ("Mathematics", "Probability", "expectation/variance continuous RV"),

- 92 ("Mathematics", "Statistics", "MLE / confidence interval numeric"),

- 93 # Physics

- 94 ("Physics", "Mechanics", "work-energy with friction / incline / spring"),

- 95 ("Physics", "Mechanics", "momentum/impulse collision"),

- 96 ("Physics", "Mechanics", "circular motion/banked curve"),

- 97 ("Physics", "Electricity & Magnetism", "electric field/potential superposition"),

- 98 ("Physics", "Circuits", "RC transient time to reach a voltage"),

- 99 ("Physics", "Circuits", "DC circuit (equivalent resistance/current)"),

- 100 ("Physics", "Waves", "standing waves / beat frequency"),

- 101 ("Physics", "Thermodynamics", "ideal gas process (W,Q, U)"),

- 102 ("Physics", "Optics", "thin lens / mirror imaging"),

- 103 # Chemistry

- 104 ("Chemistry", "Thermodynamics", " G and equilibrium constant"),

- 105 ("Chemistry", "Gases", "van der Waals / compression factor"),

- 106 ("Chemistry", "Equilibrium", "buffer pH / K_a, K_b"),

- 107 ("Chemistry", "Kinetics", "rate laws / half-life"),

- 108 ("Chemistry", "Electrochemistry", "Nernst equation cell potential"),

- 109 ("Chemistry", "Solutions", "colligative properties ( T_f/ T_b)"),

- 110 ]

- 111

- 112 UNITS_BY_DOMAIN = {

- 113 "Mathematics": ["unitless", "s", "m", "kg", "Pa", "J"], # math may be unitless

- 114 "Physics": ["m/s", "m/s^2", "N", "J", "W", "C", "V", "Hz", "K", "Pa"],

- 115 "Chemistry": ["kJ/mol", "J/mol", "mol/L", "atm", "Pa", "K", "V", "s", "g/mol"],

- 116 }

- 117

- 118 def build_generation_prompt(seed: int) -> str:

- 119 rnd = random.Random(seed)

- 120 domain, area, style = rnd.choice(TOPICS)

- 121 difficulty = rnd.choice(["introductory", "intermediate", "advanced"])

- 122 scenario = rnd.choice([

- 123 "applied real-world scenario",

- 124 "textbook-style theoretical scenario",

- 125 "lab measurement scenario with given uncertainties ignored",

- 126 "engineering estimation scenario",

- 127 ])

- 128 unit = rnd.choice(UNITS_BY_DOMAIN[domain])

- 129

- 130 # Encourage concise but complete structure

- 131 return f"""

- 132 Create ONE SciBench-style example.

- 133

- 134 Hard requirements:

- 135 - Domain: {domain}. Area: {area}. Style: {style}. Difficulty: {difficulty}.

- 136 - Scenario flavor: {scenario}.

- 137 - Write the QUESTION first, starting with exactly: "Question: "

- 138 - The question must be self-contained and include all constants needed.

- 139 - The question must include exactly one sentence: "The unit of the answer is {unit}."

- 140 (If {unit} is not appropriate for your question, replace it with an appropriate unit and keep the same sentence pattern.)

- 141 - Then write the SOLUTION with a clear structure:

- 142 1) Given / Find

- 143 2) Key equations

- 144 3) Step-by-step calculation (show intermediate numeric values)

- 145 4) Final line exactly: The answer is therefore \\boxed{{X.XXX}}.

- 146 - Output must be concise: aim for ~12-25 lines; no filler.

- 147 - Final boxed value must be a decimal with exactly three digits; no units in the box.

- 148 - Ensure the last \\boxed{{...}} in the output is the final answer.

- 149

- 150 Return JSON only:

- 151 {{"input": "...", "output": "..."}}

- 152 """.strip()

- 153

- 154 def build_repair_prompt(item: Dict[str, str]) -> str:

- 155 # Ask teacher to keep question, rewrite solution concisely + correct format.

- 156 inp = item.get("input", "")

- 157 out = item.get("output", "")

- 158 return f"""

- 159 You will REPAIR a dataset example for SciBench numeric answering.

- 160

- 161 Keep the QUESTION text exactly as-is.

- 162 Rewrite the SOLUTION to be correct, concise, and fully computed.

- 163

- 164 Hard constraints for the rewritten solution:

- 165 - Use a clear structure (Given/Find, equations, calculations).

- 166 - Do not be overly verbose (target ~12-25 lines).

- 167 - The final answer must be a decimal with exactly three digits after the decimal point.

- 168 - Do NOT include units in the boxed answer.

- 169 - The last line MUST be exactly: The answer is therefore \\boxed{{X.XXX}}.

- 170 - Ensure the last \\boxed{{...}} is the final numeric answer.

- 171

- 172 Return JSON only:

- 173 {{"input": "<same question>", "output": "<new solution>"}}

- 174

- 175 QUESTION:

- 176 {inp}

- 177

- 178 ORIGINAL SOLUTION (may be wrong/malformed):

- 179 {out}

- 180 """.strip()

- 181

- 182 def call_teacher_json(prompts: List[str]) -> List[Dict]:

- 183 # Must call in fixed batch size 80

- 184 assert len(prompts) == API_BATCH, f"api_generate_batch must be called with batch size {API_BATCH} "

- 185 sys_prompt = "You are a meticulous STEM textbook solution writer. Follow constraints exactly and output valid JSON only."

- 186 return api_generate_batch(prompts, sys_prompt=sys_prompt, json=True)

- 187

- 188 def parse_generated(objs: List) -> List[Dict[str, str]]:

- 189 items = []

- 190 for o in objs:

- 191 if isinstance(o, dict) and "input" in o and "output" in o:

- 192 items.append({"instruction": INSTRUCTION, "input": o["input"], "output": o["output"]})

- 193 return items

- 194

- 195 def dedup_keep_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:

- 196 seen = set()

- 197 out = []

- 198 for it in items:

- 199 fp = fingerprint_input(it.get("input", ""))

- 200 if fp in seen:

- 201 continue

- 202 seen.add(fp)

- 203 out.append(it)

- 204 return out

- 205

- 206 def main():

- 207 random.seed(20260208)

- 208

- 209 existing = load_existing(OUT_PATH)

- 210 print(f"Loaded {len(existing)} raw items from {OUT_PATH}")

- 211

- 212 # Separate valid and invalid

- 213 valid = []

- 214 invalid = []

- 215 for it in existing:

- 216 if is_valid_item(it):

- 217 valid.append(it)

- 218 else:

- 219 # coerce to expected keys if partially missing

- 220 if isinstance(it, dict) and "input" in it and "output" in it:

- 221 invalid.append({"instruction": INSTRUCTION, "input": it.get("input", ""), "output": it.get("output", "")})

- 222

- 223 valid = dedup_keep_order(valid)

- 224 print(f"Valid after strict filtering + dedup: {len(valid)}; candidates for repair: {len(invalid)} ")

- 225

- 226 # Save immediate checkpoint of filtered-valid

- 227 save_json(valid[:TARGET_N], OUT_PATH)

- 228

- 229 api_calls_used = 0

- 230

- 231 # ---- Repair pass (single pass over as many invalid as needed) ----

- 232 # We'll repair up to the deficit and not exceed call budget.

- 233 def deficit():

- 234 return max(0, TARGET_N - len(valid))

- 235

- 236 repair_pool = invalid[:] # order as loaded

- 237 repair_idx = 0

- 238 while deficit() > 0 and repair_idx < len(repair_pool):

- 239 if api_calls_used >= MAX_API_CALLS_THIS_ITER:

- 240 break

- 241 if get_available_api_calls() <= 0:

- 242 break

- 243

- 244 batch_items = repair_pool[repair_idx:repair_idx + API_BATCH]

- 245 if len(batch_items) < API_BATCH:

- 246 break # keep strict batching; remaining can be handled by generation

- 247 repair_idx += API_BATCH

- 248

- 249 prompts = [build_repair_prompt(it) for it in batch_items]

- 250 objs = call_teacher_json(prompts)

- 251 api_calls_used += 1

- 252

- 253 repaired = parse_generated(objs)

- 254 repaired_valid = [it for it in repaired if is_valid_item(it)]

- 255 if repaired_valid:

- 256 valid.extend(repaired_valid)

- 257 valid = dedup_keep_order(valid)

- 258

- 259 if len(valid) % 200 < len(repaired_valid):

- 260 save_json(valid[:TARGET_N], OUT_PATH)

- 261 print(f"Checkpoint after repair: {min(len(valid), TARGET_N)}")

- 262

- 263 print(f"Repair call {api_calls_used}: got {len(repaired)} parsed, {len(repaired_valid)} valid ; total valid {len(valid)}")

- 264

- 265 # ---- Top-up generation ----

- 266 gen_seed_base = 900000

- 267 gen_round = 0

- 268 while deficit() > 0:

- 269 if api_calls_used >= MAX_API_CALLS_THIS_ITER:

- 270 break

- 271 if get_available_api_calls() <= 0:

- 272 break

- 273

- 274 prompts = [build_generation_prompt(gen_seed_base + gen_round * API_BATCH + i) for i in range( API_BATCH)]

- 275 objs = call_teacher_json(prompts)

- 276 api_calls_used += 1

- 277 gen_round += 1

- 278

- 279 items = parse_generated(objs)

- 280 items_valid = [it for it in items if is_valid_item(it)]

- 281 if items_valid:

- 282 valid.extend(items_valid)

- 283 valid = dedup_keep_order(valid)

- 284

- 285 if len(valid) % 200 < len(items_valid) or len(valid) >= TARGET_N:

- 286 save_json(valid[:TARGET_N], OUT_PATH)

- 287 print(f"Checkpoint after gen: {min(len(valid), TARGET_N)}")

- 288

- 289 print(f"Gen call {api_calls_used}: parsed {len(items)}, valid {len(items_valid)}; total valid {len(valid)}")

- 290

- 291 # If validity is unexpectedly low, we could tighten further, but keep runtime simple.

- 292

- 293 # Finalize

- 294 random.shuffle(valid)

- 295 final = valid[:TARGET_N]

- 296 save_json(final, OUT_PATH)

- 297 print(f"Final saved: {len(final)} items to {OUT_PATH}. API calls used this iteration: { api_calls_used}")

- 298

- 299 if __name__ == "__main__":

- 300 main()

#### G Code Snippets for Failure Analysis

In this section, we provide the specific code implementations generated by the agents that correspond to the failure cases discussed in Section 4.3.

G.1 Science Task: Partial Distribution Shift. Listing 2 presents the code.py generated during the from-scratch iterative Science task.

Listing 2: Agent Code That Induced Data Distribution Shift

- 1 import os

- 2 import re

- 3 import json

- 4 import math

- 5 import random

- 6 from typing import List, Dict, Any, Optional

- 7 from services.library import api_generate_batch, get_available_api_calls

- 8

- 9 BEST_PATH = "../submission/submission_best.json"

- 10 OUT_PATH = "../submission/submission.json"

- 11

- 12 INSTRUCTION = (

- 13 "Please provide a clear and step-by-step solution for a scientific problem in the categories "

- 14 "of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, "

- 15 "which should not be included in the answer. Express the final answer as a decimal number "

- 16 "with three digits after the decimal point. Conclude the answer by stating "

- 17 "\"The answer is therefore \\\\boxed{[ANSWER]}.\""

- 18 )

- 19

- 20 # Must be 65 prompts per call.

- 21 BATCH_SIZE = 65

- 22

- 23 # Accept one-or-more backslashes before boxed (handles '\boxed' and '\\boxed' in stored strings)

- 24 BOX_RE = re.compile(r"\\+boxed\{([^}]*)\}")

- 25 FINAL_LINE_RE = re.compile(r"The answer is therefore\s*\\+boxed\{([^}]*)\}\.\s*$")

- 26

- 27 NUM_RE = re.compile(

- 28 r"^[\s]*"

- 29 r"(?P<sign>[+-]?)"

- 30 r"(?P<num>(?:\d+(?:\.\d*)?|\.\d+))"

- 31 r"(?:[eE](?P<exp>[+-]?\d+))?"

- 32 r"[\s]*$"

- 33 )

- 34

- 35 UNIT_CUE_RE = re.compile(

- 36 r"(unit\s+of\s+the\s+answer\s+is|units?\s*:|answer\s+should\s+be\s+in|the\s+unit\s+is|\bin\s+\$ ?\\?[a-zA-Z]+)",

- 37 re.IGNORECASE,

- 38 )

- 39

- 40 # Corruption / low-quality heuristics seen in existing samples.

- 41 BAD_TOKEN_RE = re.compile(

- 42 r"(\\nrac|[^\\]rac\{|boxed\{\s*\}|@@|\?\?|nan|inf)", re.IGNORECASE

- 43 )

- 44

- 45 def load_json_list(path: str) -> List[Dict[str, Any]]:

- 46 if not os.path.exists(path):

- 47 return []

- 48 try:

- 49 with open(path, "r", encoding="utf-8") as f:

- 50 data = json.load(f)

- 51 return data if isinstance(data, list) else []

- 52 except Exception:

- 53 return []

- 54

- 55 def save(path: str, data: List[Dict[str, Any]]) -> None:

- 56 os.makedirs(os.path.dirname(path), exist_ok=True)

- 57 with open(path, "w", encoding="utf-8") as f:

- 58 json.dump(data, f, ensure_ascii=False, indent=2)

- 59

- 60 def parse_number(s: str) -> Optional[float]:

- 61 s = s.strip()

- 62 if not NUM_RE.match(s):

- 63 return None

- 64 try:

- 65 x = float(s)

- 66 if math.isnan(x) or math.isinf(x):

- 67 return None

- 68 return x

- 69 except Exception:

- 70 return None

- 71

- 72 def normalize_final_line(output: str) -> Optional[str]:

- 73 out = output.rstrip()

- 74 boxes = BOX_RE.findall(out)

- 75 if not boxes:

- 76 return None

- 77 last_box_raw = boxes[-1].strip()

- 78 val = parse_number(last_box_raw)

- 79 if val is None:

- 80 return None

- 81 val_3 = f"{val:.3f}"

- 82 lines = out.splitlines()

- 83 if not lines:

- 84 return None

- 85 lines[-1] = f"The answer is therefore \\boxed{{{val_3}}}."

- 86 return "\n".join(lines)

- 87

- 88 def validate_entry(inp: Any, out: Any) -> bool:

- 89 if not isinstance(inp, str) or not isinstance(out, str):

- 90 return False

- 91 inp = inp.strip()

- 92 out = out.strip()

- 93

- 94 if not inp.startswith("Question:"):

- 95 return False

- 96 if not UNIT_CUE_RE.search(inp):

- 97 return False

- 98

- 99 if BAD_TOKEN_RE.search(out):

- 100 return False

- 101

- 102 # Ensure final line matches required template

- 103 if not FINAL_LINE_RE.search(out):

- 104 return False

- 105

- 106 # Ensure last boxed number is parseable

- 107 boxes = BOX_RE.findall(out)

- 108 if not boxes:

- 109 return False

- 110 if parse_number(boxes[-1]) is None:

- 111 return False

- 112

- 113 # Ensure solution is not trivial/answer-only: require at least 6 lines

- 114 if len(out.splitlines()) < 6:

- 115 return False

- 116

- 117 return True

- 118

- 119 def sanitize_item(inp: Any, out: Any) -> Optional[Dict[str, str]]:

- 120 if not isinstance(inp, str) or not isinstance(out, str):

- 121 return None

- 122 norm = normalize_final_line(out)

- 123 if norm is None:

- 124 return None

- 125 inp_s = inp.strip()

- 126 out_s = norm.strip()

- 127 if not validate_entry(inp_s, out_s):

- 128 return None

- 129 return {"instruction": INSTRUCTION, "input": inp_s, "output": out_s}

- 130

- 131 SYS_PROMPT_TARGETED = (

- 132 "You are generating instruction-tuning data for solving college-level scientific problems.\n"

- 133 "Return ONLY valid JSON (no markdown, no extra text).\n"

- 134 "Schema: {\"input\": string, \"output\": string}\n"

- 135 "Hard rules:\n"

- 136 "- input MUST start with exactly 'Question:'\n"

- 137 "- input MUST explicitly state the unit of the final answer in a sentence: 'The unit of the answer is ... .'\n"

- 138 "- output MUST be a correct step-by-step solution with unit conversions shown.\n"

- 139 "- output MUST have at least 6 lines.\n"

- 140 "- output MUST end with EXACT last line: The answer is therefore \\\\boxed{NUMBER}.\n"

- 141 "- NUMBER must be numeric only (no units) and prefer decimal with three digits.\n"

- 142 "Focus heavily on these topics (rotate among them):\n"

- 143 "1) Two-level system / Boltzmann population ratios with energies in cm^-1; use hc/kB=1.4388 cm*K

.\n"

- 144 "2) Photoelectric/ionization energy: wavelength in nm to eV; subtract electron KE from v.\n"

- 145 "3) Coriolis deflection for projectile at given latitude; use Earth rotation Omega=7.292e-5 s ^-1.\n"

- 146 "4) Manometer pressure conversions (cm H2O, mmHg) and computing R from PV=nRT; avoid factor-1000 errors.\n"

- 147 "5) Orbital mechanics energy changes between circular orbits; use mu=3.986e14 m^3/s^2, Re=6.371e6 m.\n"

- 148 "Use standard constants when needed: g=9.81, kB=1.381e-23 J/K, h=6.626e-34 J s, c=2.998e8 m/s, "

- 149 "e=1.602e-19 C, 1 eV=1.602e-19 J.\n"

- 150 )

- 151

- 152 SYS_PROMPT_GENERAL = (

- 153 "You are generating instruction-tuning data for solving college-level scientific problems.\n"

- 154 "Return ONLY valid JSON (no markdown, no extra text).\n"

- 155 "Schema: {\"input\": string, \"output\": string}\n"

- 156 "Rules:\n"

- 157 "- input MUST start with exactly 'Question:'\n"

- 158 "- input MUST explicitly state the unit of the final answer in a sentence: 'The unit of the answer is ... .'\n"

- 159 "- output MUST be a correct step-by-step solution.\n"

- 160 "- output MUST have at least 6 lines.\n"

- 161 "- output MUST end with EXACT last line: The answer is therefore \\\\boxed{NUMBER}.\n"

- 162 "- NUMBER must be numeric only and prefer decimal with three digits.\n"

- 163 "Cover a broad mix of undergraduate topics across calculus, ODEs, probability/statistics, mechanics, E&M, circuits, "

- 164 "thermodynamics, equilibrium, kinetics, electrochemistry, optics.\n"

- 165 "Use standard constants when needed: g=9.81, R=8.314, k=8.988e9, h=6.626e-34, c=2.998e8.\n"

- 166 )

- 167

- 168 def build_prompt(seed: int, mode: str) -> str:

- 169 random.seed(seed)

- 170 if mode == "targeted":

- 171 focus = random.choice([

- 172 "two-level Boltzmann population with energy separation in cm^-1",

- 173 "photoelectric/ionization energy from wavelength and electron speed",

- 174 "Coriolis deflection for a projectile fired due north/south at given latitude",

- 175 "manometer pressure conversion and computing gas constant R from measurements",

- 176 "orbital mechanics: energy required to move between circular orbits including synchronous orbit"

- 177 ])

- 178 return (

- 179 "Generate ONE original, solvable, college-level quantitative problem and its correct step

-by-step solution.\n"

- 180 f"Topic MUST be: {focus}.\n"

- 181 "Hard constraints:\n"

- 182 "1) Return exactly one JSON object: {\"input\":..., \"output\":...}.\n"

- 183 "2) input starts with 'Question:' and explicitly states the unit in: 'The unit of the answer is ... .'\n"

- 184 "3) output shows all key equations and unit conversions and has at least 6 lines.\n"

- 185 "4) output ends with EXACT last line: The answer is therefore \\\\boxed{NUMBER}.\n"

- 186 "5) NUMBER is numeric only, no units, and prefer three decimals.\n"

- 187 f"Seed tag: {seed}"

- 188 )

- 189 else:

- 190 return (

- 191 "Generate ONE original, solvable, college-level scientific problem (Math/Physics/ Chemistry) "

- 192 "and its correct step-by-step solution.\n"

- 193 "Ensure multi-step reasoning and intermediate computations; avoid trivial one-liners.\n"

- 194 "Hard constraints:\n"

- 195 "1) Return exactly one JSON object: {\"input\":..., \"output\":...}.\n"

- 196 "2) input starts with 'Question:' and explicitly states the unit in: 'The unit of the answer is ... .'\n"

- 197 "3) output has at least 6 lines.\n"

- 198 "4) output ends with EXACT last line: The answer is therefore \\\\boxed{NUMBER}.\n"

- 199 "5) NUMBER is numeric only, no units, and prefer three decimals.\n"

- 200 f"Seed tag: {seed}"

- 201 )

- 202

- 203 def main(target_total: int = 2000, max_calls_cap: int = 5000) -> None:

- 204 # 1) Start from best, filter hard for quality

- 205 best = load_json_list(BEST_PATH)

- 206 data: List[Dict[str, str]] = []

- 207 seen_inputs = set()

- 208

- 209 for d in best:

- 210 if not isinstance(d, dict):

- 211 continue

- 212 item = sanitize_item(d.get("input"), d.get("output"))

- 213 if item is None:

- 214 continue

- 215 if item["input"] in seen_inputs:

- 216 continue

- 217 data.append(item)

- 218 seen_inputs.add(item["input"])

- 219

- 220 save(OUT_PATH, data)

- 221

- 222 # 2) Regenerate until we reach 2000

- 223 if len(data) >= target_total:

- 224 save(OUT_PATH, data[:target_total])

- 225 return

- 226

- 227 available_calls = get_available_api_calls()

- 228 calls_budget = min(max_calls_cap, available_calls)

- 229

- 230 seed_base = 20260129

- 231 call_idx = 0

- 232

- 233 while len(data) < target_total and call_idx < calls_budget:

- 234 mode = "targeted" if (call_idx % 2 == 0) else "general"

- 235 sys_prompt = SYS_PROMPT_TARGETED if mode == "targeted" else SYS_PROMPT_GENERAL

- 236

- 237 prompts = [build_prompt(seed_base + call_idx * 1000 + i, mode=mode) for i in range(BATCH_SIZE )]

- 238 responses = api_generate_batch(prompts, sys_prompt=sys_prompt, json=True)

- 239

- 240 for resp in responses:

- 241 if not isinstance(resp, dict):

- 242 continue

- 243 item = sanitize_item(resp.get("input"), resp.get("output"))

- 244 if item is None:

- 245 continue

- 246 if item["input"] in seen_inputs:

- 247 continue

- 248 data.append(item)

- 249 seen_inputs.add(item["input"])

- 250 if len(data) >= target_total:

- 251 break

- 252

- 253 save(OUT_PATH, data)

- 254 call_idx += 1

- 255

- 256 save(OUT_PATH, data[:target_total])

- 257

- 258 if __name__ == "__main__":

- 259 main(target_total=2000, max_calls_cap=5000)

G.2 Code Task: Naive Rule-based Augmentation. Listing 3 presents the code.py generated during the from-seed iterative Code task.

Listing 3: Agent Code That Employed Naive Rule-based Augmentation

- 1 import os

- 2 import json

- 3 import re

- 4 import random

- 5 import uuid

- 6 import argparse

- 7 import logging

- 8 import math

- 9 from typing import List, Dict, Any, Optional

- 10 from datetime import datetime

- 11 from collections import Counter

- 12

- 13

- 14 logging.basicConfig(

- 15 level=logging.INFO,

- 16 format='%(asctime)s - [DATA_PIPELINE] - %(levelname)s - %(message)s',

- 17 datefmt='%Y-%m-%d %H:%M:%S'

- 18 )

- 19 logger = logging.getLogger(__name__)

- 20

- 21

- 22 DEFAULT_TARGET_SIZE = 5000

- 23 PERTURBATION_RANGE = (-5, 5)

- 24 MUTATION_PROBABILITY = 0.8

- 25 MIN_VALUE_CLAMP = 1

- 26

- 27 NUMBER_PATTERN = re.compile(r'(?<![a-zA-Z_])\d+(?![a-zA-Z_])')

- 28

- 29 class DataAugmenter:

- 30 def __init__(self, seed_path: str, output_path: str, target_size: int):

- 31 self.seed_path = seed_path

- 32 self.output_path = output_path

- 33 self.target_size = target_size

- 34 self.stats = Counter()

- 35 self.generated_ids = set()

- 36

- 37 def load_seeds(self) -> List[Dict[str, Any]]:

- 38

- 39 if not os.path.exists(self.seed_path):

- 40 logger.error(f"Seed file not found at {self.seed_path}")

- 41 return []

- 42

- 43 try:

- 44 with open(self.seed_path, 'r', encoding='utf-8') as f:

- 45 data = json.load(f)

- 46

- 47 valid_seeds = [

- 48 item for item in data

- 49 if all(k in item for k in ('instruction', 'input', 'output'))

- 50 ]

- 51 logger.info(f"Loaded {len(valid_seeds)} valid seeds from {len(data)} total entries.")

- 52 return valid_seeds

- 53 except json.JSONDecodeError as e:

- 54 logger.critical(f"Failed to parse seed JSON: {e}")

- 55 return []

- 56

- 57 def _stochastic_perturbation(self, text: str) -> str:

- 58 def replace_match(match):

- 59 if random.random() > MUTATION_PROBABILITY:

- 60 return match.group()

- 61

- 62 try:

- 63 original_val = int(match.group())

- 64

- 65 shift = random.randint(*PERTURBATION_RANGE)

- 66

- 67 new_val = max(MIN_VALUE_CLAMP, original_val + shift)

- 68

- 69 return str(new_val)

- 70 except ValueError:

- 71 return match.group()

- 72

- 73 return NUMBER_PATTERN.sub(replace_match, text)

- 74

- 75 def generate_variant(self, seed_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:

- 76 try:

- 77 new_input = self._stochastic_perturbation(seed_item['input'])

- 78

- 79 new_output = self._stochastic_perturbation(seed_item['output'])

- 80

- 81 variant = {

- 82 "instruction": seed_item['instruction'],

- 83 "input": new_input,

- 84 "output": new_output,

- 85 "meta": {

- 86 "origin": "augmented",

- 87 "parent_id": seed_item.get("id", "unknown"),

- 88 "aug_method": "regex_perturbation_v2"

- 89 }

- 90 }

- 91

- 92 if new_input == seed_item['input'] and new_output == seed_item['output']:

- 93 self.stats['skipped_no_change'] += 1

- 94 return None

- 95

- 96 return variant

- 97

- 98 except Exception as e:

- 99 logger.warning(f"Failed to generate variant: {e}")

- 100 self.stats['errors'] += 1

- 101 return None

- 102

- 103 def validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

- 104

- 105 unique_data = []

- 106 seen_hashes = set()

- 107

- 108 for item in dataset:

- 109 content_hash = hash(f"{item['input']}|{item['output']}")

- 110

- 111 if content_hash in seen_hashes:

- 112 self.stats['duplicates_removed'] += 1

- 113 continue

- 114

- 115 seen_hashes.add(content_hash)

- 116 unique_data.append(item)

- 117

- 118 return unique_data

- 119

- 120 def run(self):

- 121 start_time = datetime.now()

- 122 logger.info(f"Starting pipeline at {start_time}")

- 123

- 124 seeds = self.load_seeds()

- 125 if not seeds:

- 126 return

- 127

- 128 dataset = []

- 129 dataset.extend(seeds)

- 130 logger.info(f"Initialized with {len(seeds)} seed samples.")

- 131

- 132 pbar_interval = self.target_size // 10

- 133 while len(dataset) < self.target_size:

- 134 base_seed = random.choice(seeds)

- 135 variant = self.generate_variant(base_seed)

- 136

- 137 if variant:

- 138 dataset.append(variant)

- 139 self.stats['generated'] += 1

- 140

- 141 current_count = len(dataset)

- 142 if current_count % pbar_interval == 0 and current_count > 0:

- 143 logger.info(f"Progress: {current_count}/{self.target_size} samples generated...")

- 144

- 145 logger.info("Running final validation and deduplication...")

- 146 final_dataset = self.validate_dataset(dataset)

- 147

- 148 final_dataset = final_dataset[:self.target_size]

- 149

- 150 self.save_data(final_dataset)

- 151

- 152 duration = datetime.now() - start_time

- 153 logger.info(f"Pipeline completed in {duration}.")

- 154 logger.info(f"Statistics: {dict(self.stats)}")

- 155

- 156 def save_data(self, data: List[Dict[str, Any]]):

- 157 os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

- 158 with open(self.output_path, 'w', encoding='utf-8') as f:

- 159 json.dump(data, f, indent=2, ensure_ascii=False)

- 160 logger.info(f"Successfully saved {len(data)} entries to {self.output_path}")

- 161

- 162 def parse_args():

- 163 parser = argparse.ArgumentParser(description="Rule-based Data Augmentation Tool")

- 164 parser.add_argument('--seed_path', type=str, default='../data/seed/seed.json', help='Path to seed data')

- 165 parser.add_argument('--output_path', type=str, default='../submission/submission.json', help=' Path to save submission')

- 166 parser.add_argument('--target_size', type=int, default=5000, help='Target dataset size')

- 167 return parser.parse_args()

- 168

- 169 if __name__ == "__main__":

- 170 args = parse_args()

- 171

- 172 augmenter = DataAugmenter(

- 173 seed_path=args.seed_path,

- 174 output_path=args.output_path,

- 175 target_size=args.target_size

- 176 )

- 177

- 178 try:

- 179 augmenter.run()

- 180 except KeyboardInterrupt:

- 181 logger.info("Pipeline interrupted by user. Saving partial progress...")

- 182 pass

- 183 except Exception as e:

- 184 logger.exception("Fatal pipeline error")

- 185 exit(1)

#### H Seed Examples

In this section, we present representative seed examples used in our experiments for the Science, Code, and Finance domains. These examples are extracted directly from the seed.json files of the respective datasets.

##### H.1 Science Domain

The seed data for Sci-Bench primarily consists of complex scientific problems involving physics and chemistry calculations.

- 1 [

- 2 {

- 3 "question": "Three identical metal spheres have the same diameter. Spheres 1 and 2 carry equal like charges

- 4 Q, with separation much greater than their diameter, and experience force F. Sphere 3 is uncharged with an

- 5 insulating handle. If sphere 3 touches sphere 1, then touches sphere 2, and is removed, what is the new

- 6 interaction force between spheres 1 and 2?"

- 7 },

- 8 {

- 9 "question": "In an isolated town of 5000 inhabitants, the spread of an epidemic is such that the rate of

- 10 spread is jointly proportional to the number of infected and uninfected people. If 160 people are infected at

- 11 the start and 1200 are infected after one week, how long does it take for 80% of the population (4000 people)

- 12 to become infected?"

- 13 },

- 14 ...

- 15 ]

##### H.2 Code Domain

The seed data for the Code task includes algorithmic problems with problem descriptions, examples, and test inputs.

- 1 [

- 2 {

- 3 "question_content": "Given n, a and d as the number of terms, first term and common difference respectively

- 4 of an Arthimetic Series. Find the sum of the series upto nth term.\n \nExample 1:\nInput: 5 1 3\nOutput:

- 5 35\nExplanation: Series upto 5th term is\n1 4 7 10 13, so sum will be 35.\nExample 2:\nInput: 3 1 2\n

- 6 Output: 9\nExample: Series upto 3rd term is \n1 3 5, so sum will be 9.\n \nYour Task:\nYou don' t need to

- 7 read or print anything. Your task is to complete the function sum_of_ap() which takes n, a and d as input

- 8 parameter and returns the sum of the series.\n \nExpected Time Complexity: O(1)\nExpected Space Complexity:

- 9 O(1)\n \nConstranits:\n1 <= n, a, d <= 100",

- 10 "test_input": "5 1 3"

- 11 },

- 12 {

- 13 "question_content": "Let $f_{x} = c^{2x-6} \\cdot f_{x-1} \\cdot f_{x-2} \\cdot f_{x-3}$ for $x \\ge 4$.

- 14 \n\nYou have given integers $n$, $f_{1}$, $f_{2}$, $f_{3}$, and $c$. Find $f_{n} \\bmod (10^{9}+7)$.

- 15 \n\n\n-----Input-----\n\nThe only line contains five integers $n$, $f_{1}$, $f_{2}$, $f_{3}$, and $c$ ($4

- 16 \\le n \\le 10^{18}$, $1 \\le f_{1}$, $f_{2}$, $f_{3}$, $c \\le 10^{9}$).\n\n\n-----Output

-----\n\nPrint

- 17 $f_{n} \\bmod (10^{9} + 7)$.\n\n\n-----Examples-----\nInput\n5 1 2 5 3\n\nOutput\n72900\n\ nInput\n17 97 41

- 18 37 11\n\nOutput\n317451037\n\n\n\n-----Note-----\n\nIn the first example, $f_{4} = 90$, $f_{5}

= 72900$.

- 19 \n\nIn the second example, $f_{17} \\approx 2.28 \\times 10^{29587}$.",

- 20 "test_input": "5 1 2 5 3"

- 21 },

- 22 ...

- 23 ]

- H.3 Finance Domain The Finance-Reasoning seed data comprises specific financial scenarios (context) and quantitative questions requiring reasoning over that context.

- 1 [

- 2 {

- 3 "question": "What is Alice's new adjusted monthly mortgage payment after the fixed-rate period for the

- 4 remaining 10 years? Answer in dollars, rounded to the nearest cent.",

- 5 "context": "Alice took a 15-year fixed-rate mortgage with a principal amount of $250,000 at an annual

- 6 interest rate of 4.5%. After the fixed-rate period ended, the remaining principal balance was $150,000.

- 7 Her mortgage transitioned to an adjustable-rate with the current index rate at 2% and a bank margin of 1.5%.

- 8 She wants to calculate her new monthly payment for the remaining 10 years of the mortgage under these new

- 9 terms, assuming there are no rate caps."

- 10 },

- 11 {

- 12 "question": "What is the difference in the high and low prices of the common stock in the fourth quarter of

- 13 2019? Answer to two decimal places.",

- 14 "context": "{\"2019: -- Fourth Quarter\": {\"High\": 11.44, \"Low\": 9.47}, \"2019: -- Third Quarter\":

- 15 {\"High\": 14.96, \"Low\": 10.26}, \"2019: -- Second Quarter\": {\"High\": 20.91, \"Low\": 12.61}, \"2019:

- 16 -- First Quarter\": {\"High\": 18.19, \"Low\": 8.87}, \"2018: -- Fourth Quarter\": {\"High\": 12.16,

- 17 \"Low\": 7.43}, \"2018: -- Third Quarter\": {\"High\": 20.6, \"Low\": 10.95}, \"2018: -Second Quarter\":

- 18 {\"High\": 18.3, \"Low\": 6.7}, \"2018: -- First Quarter\": {\"High\": 7.35, \"Low\": 6.0}}"

- 19 },

- 20 ...

- 21 ]

- I Prompt Templates

In this section, we present the core prompt templates used in our framework. To ensure reproducibility, we provide the full content of the system instructions and the specific action prompts used by the agents. Note that placeholders such as {dataset_size} are dynamically filled during runtime.

##### I.1 System Description

The following system prompt is injected into the context of all agents (both One-Shot and Iterative) to define the AutoDataBench task environment, constraints, and evaluation criteria.

##### System Prompt: Task Instruction

You are participating data-driven model specialization, a new challenge designed to test your ability to act as an autonomous AI data synthesis scientist.

##### ===== TASK =====

Your mission is to craft a high-quality, synthetic instruction-tuning dataset for a given task. Your formance is evaluated by how much a standard model’s performance improves after being finetuned on the dataset you create. This task evaluates your core capabilities in understanding, synthesizing and distilling valuable training data from raw resources. Please read the instruc-

tions below, which are specific to this environment. These instructions can be found again in ‘../agent/utils/instructions.py‘.

===== RESOURCES ===== All necessary resources for your data synthesis process are located in ../data/. This includes:

- • ../data/description.md: Describes the target task, including the goal and example data format. [Description Content]
- • ../data/seed/seed.json: Necessary raw material from which you create instruction-tuning pairs, refine and expand this as needed. Provided only if required by the task. [Source Content]
- • ../data/seed/few_shot.json: A small set of high-quality examples to follow for style, complexity, and format. [Few-shot Content]
- • ../data/sample_submission.json: A template demonstrating the required JSON structure for your final submission instruction-tuning file. [Sample Submission]
- • ../data/train_config.yaml: Contains the model architecture and default hyperparameters for fine-tuning. [Train Config]

##### ===== SUBMISSION =====

Your final output is a synthetic training dataset. You MUST produce this dataset at EXACTLY ../submission/submission.json, strictly adhere to the format specified in sample_submission.json. The submission should be around {dataset_size} entries, regardless of the amount of seed data provided (if any). Too little synthetic data severely degrades model training performance.

##### ===== ENVIRONMENT ======

- • Teacher Models: API Access of {teacher_model}.
- • Target Models: Standard models to be fine-tuned and then evaluated: {student_model}.
- • GPU Avaliable: 160GB.
- • Synthetic Data Constraint: We will filter your final submission to retain only entries matching the textttsample_submission.csv format with non-empty ‘output‘ fields, and use the first {dataset_size} entries for training.
- • API Limit: You are limited to a total of {api_limit} calls to the API-based teacher models.
- • Runtime Limit: Your entire process must complete within {max_hours} hours.

##### ===== IMPORTANT NOTES ======

- • You must save your final dataset at exactly the specified path:

../submission/submission.json. Save regularly to prevent data loss.

- • Your only task is to generate instruction-tuning dataset. Do not include any code for model training or evaluation.
- • You are only allowed to generate instruction-tuning data by calling the provided tearcher models. Do not directly enumerate synthetic data in the code.
- • You should synthesize dataset that is diverse, complex, and task-aligned. The given few-shot examples are for reference only in terms of format and quality.

- • You should be mindful to stay within your allocated API call quota. Avoid using loops that only check the number of generated entries while ignoring the API calls limit.
- • You must ALWAYS prioritize calling the helper functions (if provided) directly to perform any relevant task. You are strictly prohibited from re-implementing their logic or creating any similar functions.
- • You are participating in this competition independently. Ensure that the code you generate is DIRECTLY executable, no dummy implementations or placeholders of any kind.

##### I.2 One-Shot Agent

The One-Shot Agent receives a single comprehensive prompt asking for a plan and the execution code. It does not receive feedback from the execution environment unless a retry is triggered by a crash.

One-Shot Agent: Generation Prompt

Your response should include a brief plan for the data synthesis, followed by a single markdown code block that implements this plan and generates the final synthetic data. Conduct a concise analysis of the given information, and then wrap the plan and code separately in Markdown Code Blocks. Example Response: [... Necessary Analysis ...] Here is the plan: [... Brief Plan ...] Here is the code: [... Implemented Code ...]

##### I.3 Iterative Agent

The Iterative Agent operates in a loop. Depending on the state of the previous iteration (Success, Execution Error, Submission Error, or Improvement Opportunity), it receives different prompts.

Iterative Agent: Draft (Initial Generation)

##### ===== CURRENT STATUS =====

- - Remaining Time of All Iterations: {remaining_hours} hours
- - Remaining API Calls of All Iterations: {remaining_calls}
- - API Calls Limit For This Iteration: {SESSION_API_LIMIT} Keep these constraints in mind when planning your next action. Propose a brief plan and implementation code for synthesizing a high-quality dataset. Conduct a concise analysis of the given information, and then wrap the plan and code separately in “‘ blocks.

##### Iterative Agent: Debug (Execution Failure)

The data generation process failed during execution. This is debug attempt {debug_attempts}. Failed Code:

{current_code} Error Message:

{last_error} Conduct a concise analysis of the given information, and then wrap the correction plan and the code in separate “‘ blocks.

Iterative Agent: Repair (Invalid Submission Format) Your previous attempt resulted in an invalid submission file located at

../submission/submission.json. Your task is to resolve the issue. Here is your original plan: {current_plan} Here is your original code: {current_code} Here is the submission error details: {last_error} You can either repair the existing file or regenerate the data. Conduct a concise analysis... [Instructions on format]

##### Iterative Agent: Improve (Optimization on Success)

Your current best solution achieved a metric of {best_metric}. Your task is to improve the dataset. Make full use of the remaining API calls. [Optional: Performance of the base model on this test set: {score}] Here is your original plan: {best_plan} Here is your original code: {best_code} Here are the submission details: [Sample of submission file] The model trained on that data failed on these cases: {bad_case_sample} Analyze these failures to identify model weaknesses and generate more targeted data. You can either improve the existing data quality or regenerate the data.

