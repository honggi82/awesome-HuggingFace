# arXiv:2512.04324v1[cs.CL]3Dec2025

[Figure 1]

## DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle

#### Fangyu Lei1,2,3∗§, Jinxiang Meng1,2∗, Yiming Huang5, Junjie Zhao3, Yitong Zhang6, Jianwen Luo1,2, Xin Zou3, Ruiyi Yang3, Wenbo Shi3, Yan Gao3, Shizhu He1,2, Zuo Wang3, Qian Liu4, Yang Wang3, Ke Wang3,†, Jun Zhao1,2, Kang Liu1,2,†

1Institute of Automation, CAS 2University of Chinese Academy of Sciences, 3ByteDance Seed 4TikTok 5UC San Diego 6NUS

∗Equal Contribution, §Work done at ByteDance Seed, †Corresponding authors

#### Abstract

Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decisionoriented insights. We introduce DAComp, a benchmark of 210 tasks that mirrors these complex workflows. Data engineering (DE) tasks require repository-level engineering on industrial schemas, including designing and building multi-stage SQL pipelines from scratch and evolving existing systems under evolving requirements. Data analysis (DA) tasks pose open-ended business problems that demand strategic planning, exploratory analysis through iterative coding, interpretation of intermediate results, and the synthesis of actionable recommendations. Engineering tasks are scored through execution-based, multi-metric evaluation. Open-ended tasks are assessed by a reliable, experimentally validated LLM-judge, which is guided by hierarchical, meticulously crafted rubrics. Our experiments reveal that even state-of-the-art agents falter on DAComp. Performance on DE tasks is particularly low, with success rates under 20%, exposing a critical bottleneck in holistic pipeline orchestration, not merely code generation. Scores on DA tasks also average below 40%, highlighting profound deficiencies in open-ended reasoning and demonstrating that engineering and analysis are distinct capabilities. By clearly diagnosing these limitations, DAComp provides a rigorous and realistic testbed to drive the development of truly capable autonomous data agents for enterprise settings. Our data and code are available at da-comp.github.io.

Date: December 5, 2025 Correspondence: Kang Liu at kliu@nlpr.ia.ac.cn, Ke Wang at wangke@bytedance.com Project Page: da-comp.github.io

#### 1 Introduction

Data intelligence, the process of transforming raw and fragmented data into actionable insights, has become a cornerstone of modern enterprises. The remarkable reasoning and code generation capabilities of Large Language Models (LLMs) [1, 8, 23] have opened new avenues for automating data intelligence tasks. LLM-based agents have demonstrated considerable promise across a wide range of applications, including text-to-SQL [18, 20, 39], software engineering [3, 14], and general computer control [31, 33, 42]. However, the advancement of these agents into enterprise data intelligence remains constrained by the absence of benchmarks that

fct_opportunity_health DBMS

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

❷ DE: Implementation

❹ DA: Insight Generation

[Figure 11]

###### Please help me to identify "zombie" opportunities in Salesforce via engagement analysis.

Python/ SQL & Thinking

Write SQLs to build DE-DAG

Write Python / SQL to generate open-ended analysis

[Figure 12]

[Figure 13]

Raw Data Sources

[Figure 14]

Semantic-Layer Data

[Figure 15]

[Figure 16]

❶ DE: Architecture

[Figure 17]

Write the design document

###### DE-DAG

[Figure 18]

[Figure 19]

Design Specifications

[Figure 20]

Codebase of DE-DAG

Open-Ended Analysis

###### Table #12

[Figure 21]

[Figure 22]

[Figure 23]

sqls

sqls

- - name: fct_opportunity_health
- - description: Scores each opportunity's health to identify

[Figure 24]

[Figure 25]

Analytical Report Our Q3 analysis shows that 40% of the sales pipeline value is comprised of "unhealthy" opportunities that have seen no sales activity in over 30 days.

Fix 3 SQLs

intermediate

staging

[Figure 26]

[Figure 27]

'zombie' deals based on sales engagement.

int_opportunity_velocity.sql

stg_opportunity.sql

[Figure 28]

[Figure 29]

- - source_tables: [int_opportunities_with_age, int_activities_per_deal]
- - columns:
- - name: opportunity_health_score
- - description: Prioritized Rules for Health Score [0-100]

[Figure 30]

intermediate

[Figure 31]

marts

Add 4 SQLs

[Figure 32]

int_opportunity_pipeline.sql

[Figure 33]

fct_sales_velocity.sql

[Figure 34]

marts

[Figure 35]

[Figure 36]

fct_funnel_conversion.sql

[Figure 37]

Key Insight Stale deals are inﬂating the forecast, which masks a critical slowdown in sales velocity.

revenue_forecast.sql

[Figure 38]

[Figure 39]

- 1. High Engagement (>=5 activities in 30d) => 90
- 2. Stale Deal (no activity > 30d) => 10
- 3. Stuck Deal (in stage > 90d) => 25
- 4. default => 60

fct_opportunity_health.sql

[Figure 40]

Sales Velocity & Funnel Conversion？

[Figure 41]

###### Visualization

###### Table #11

[Figure 42]

[Figure 43]

[Figure 44]

❸ DE: Evolution

[Figure 45]

- name: int_activity_summary …

Write SQLs to fix/update DE-DAG

+n tables

Figure 1 DAComp aims to evaluate LLMs on full-lifecycle data intelligence workflows, encompassing repository-level data engineering (DE) and open-ended data analysis (DA).

faithfully reflect real-world complexity.

This gap between existing benchmarks and real enterprise practice calls for a benchmark that evaluates agents along two distinct axes: Hard (engineering realism) and Soft (analytical openness). The Hard axis reflects the capacity for systematic large-scale code implementation, similar to the responsibilities of data engineers. For example, this means not only generating a single SQL query but also orchestrating and evolving complex data workflows under changing requirements. The Soft axis reflects the capacity for strategic reasoning, aligning more closely with the role of data analysts. For example, this involves facing an open-ended business question, planning multi-step analytical workflows, synthesizing insights across analytical results, generating visualizations and crafting decision-oriented reports. Most benchmarks fail to capture these two key dimensions. They reduce complex engineering to isolated code snippet generation, missing the Hard axis, and reduce open-ended analysis to deterministic answers, missing the Soft axis.

To fill this gap, we present DAComp, benchmarking agents on full lifecycle data intelligence tasks, as illustrated in Fig. 1. DAComp-DE is the first to introduce repository-level data engineering tasks where agents must orchestrate multi-layered data workflows by generating a DAG on complex enterprise schemas. It includes three distinct task types: (1) DE-Architecture tasks focus on the high-level planning of detailed engineering specifications. (2) DE-Implementation tasks require agents to build multi-stage data pipelines from scratch; (3) DE-Evolution tasks challenge them to modify existing systems in response to new requirements; and Both DE-Impl and DE-Evol tasks are demanding, often requiring large-scale code changes that involve over 4,000 lines of code across more than 30 files, mirroring real-world engineering workloads. DAComp-DA is the first to pioneer real-world, open-ended data analysis. In these scenarios, agents are presented with complex questions over downstream analytical data. Unlike prior work with deterministic answers [15, 18], the tasks resemble real analyst settings: agents must write SQL/Python to aggregate, compute, and analyze intermediate results in order to generate insights, reports and visualizations, thereby emphasizing both the rigor of analytical precision and the practical utility for human decision-making. To facilitate broad applicability, we also release DAComp-zh, a high-quality Chinese adaptation of the benchmark, along with baseline results.

The evaluation methods of such complex tasks are non-trivial. For deterministic DE-Impl and DE-Evol tasks, we adopt an execution-based method to systematically evaluate the repo-level code generation performance. The open-ended DA and DE-Arch tasks are assessed by an LLM judge [19], whose evaluation is guided by our novel rubric framework. Instead of relying on a single answer key, this framework explicitly defines and assesses multiple valid solution paths for each open-ended problem, enabling a robust, multifaceted assessment that rewards diverse analytical strategies. The reliability of this LLM judge has been confirmed through rigorous validation experiments, which show strong agreement with human experts.

Our experiments on DAComp underscore a significant challenge for current models: even state-of-the-art agents falter when confronted with its enterprise-level complexity. In DE tasks, agent capabilities are pushed to their limits, with average scores below 40% and strict success rates under 10%, revealing a critical gap in real repository-level engineering capabilities. In the same vein, agents also exhibit poor performance on open-ended problems requiring autonomous planning. Performance on DA tasks plummets to below 50% for most models, with only a few proprietary systems demonstrating more robust analytical skills. Ultimately, progress in data agents demands a shift from mere code accuracy to the nuanced capabilities—planning, open-ended reasoning, and systematic synthesis—required to deliver insights that are both analytically rigorous and strategically actionable. By providing this rigorous, realistic testbed, DAComp aims to shift the focus of data agent development from isolated skills to the integrated, full-lifecycle capabilities required in the real-world scenarios.

- 2 Benchmark Construction In this section, we introduce the definition, annotation pipeline, evaluation methods and statistics.

###### 2.1 Task Definition

To bridge this gap, we design tasks that evaluate data agents on real-world challenges. Specifically, we assess their ability to act as data engineers performing repository-level data engineering and as data analysts navigating open-ended data analysis, as depicted in Fig. 1.

DAComp-DE. An agent πde is tasked with handling the full DE lifecycle including architecture, implementation, and evolution. Formally, the process is modeled as (S,C⋆) = πde(Qde,C0,B), where Qde is the initial high-level requirement, S denotes the engineering specification (e.g., a Data Contract), B is the database and C⋆ is the final DE repository. This unified capability is evaluated across three task types: (1) DE-Arch : Given a high-level requirement Qde and an initial repository C0, this task evaluates the agent’s ability to produce the engineering specification S. (2) DE-Impl : Given a detailed specification S and an empty repository (C0 = ∅), this task evaluates the agent’s ability to implement the DE repository C⋆ from scratch. (3) DE-Evol : Given an existing repository C0 and a new specification S, this task evaluates the agent’s ability to update the repository into C⋆.

DAComp-DA. Given an analysis-ready data D (semantic layer) and an open-ended question Qda, an agent with policy πda produces analysis artifacts O = πda(Qda,D) (e.g., analytical reports, key insights and actionable recommendations). This task is inherently open-ended, as a single question may be approached through multiple valid analytical paths, without a fixed standard answer.

###### 2.2 Evaluation Metrics

LLM-judge with hierarchical rubrics and GSB scoring. The LLM judge evaluates outputs O along six dimensions: Completeness, Accuracy, Insightfulness, Readability, Analytical depth and Visualization (see App. A.3.1). The hierarchical rubric assesses the first three, while the Good–Same–Bad (GSB) score [41] covers the latter three. Visualization specifically assesses the agent’s ability to translate numerical results into intuitive chart. As shown in Fig. 2, the rubric (R) decomposes a question Q into requirements and sub-reqirements. Each subrequirement admits multiple valid solution paths, each path carrying its own rubric items (colored leaf nodes). Human experts enumerate these paths and merge equivalent solutions in a single path. For scoring, the LLM judge selects the best-matching path for each sub-requirement, applies only that path’s items, then aggregates scores bottom-up. This design accommodates diverse correct approaches without penalizing method choice. We show a detailed rubric

[Figure 46]

Analytical Question

[Figure 47]

Score = Req_Score

[Figure 48]

[Figure 49]

[Figure 50]

Req 1 Req 2

Req 3

[Figure 51]

[Figure 52]

Top-DownDecomposition

Req 3.1

###### Req 3.2

|Completeness Item Precision Item Conclusiveness Item<br><br>|
|---|

Bottom-UpScoring

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Path Selection

Path Selection

[Figure 57]

Max Score: 3

Max Score: 6 Max Score: 5

[Figure 58]

[Figure 59]

[Figure 60]

Path 3.1.B

Path 3.1.A

Path 3.1.C

Figure 2 Details of hierarchical rubrics.

example for the penetration and profitability analysis in Tab. 11, with a discussion of the path enumeration scheme provided in App.F.1. The rubric score is a normalized, weighted sum of satisfied items:

N k=1 sk N k=1 wk , sk = Λ(ck,O) ∈ [0, wk]. For the Good-Same-Bad (GSB), the LLM judge only compares the final analytic results against five pre-provided baseline reports, guided by the dedicated rubrics for these axes, yielding the score: Scoregsb(O,Obase) = max(0|G|+,||SG|+|−||BB||). The final score for a DA task is a weighted combination of these two components: Scoreda = α · Scorerubric + (1 − α) · Scoregsb. The open-ended DE-Arch tasks are assessed similarly, though they employ a standard, non-hierarchical rubric and do not incorporate the GSB component. Further details are provided in App. A.

Scorerubric(O,R) =

Execution-based evaluation for deterministic tasks. DE-Impl and DE-Evol tasks are evaluated with three execution-based metrics of increasing strictness: (1) the partial credit Component Score (CS), CSDE-Impl/Evol =

j wjsj, which evaluates each node in isolation (using gold-standard upstream inputs) to measure total component-level SQL generation; (2) the Cascading Failure Score (CFS), which evaluates nodes sequentially along the DAG and nullifies a node’s score if any upstream dependency is incorrect, thus measuring end-to-end data integrity; and (3) the strict Success Rate (SR), SRDE-Impl/Evol = I[∀j : sj = 1], which requires every single component to be perfect. This suite of metrics is crucial for diagnosing the primary bottleneck: the gap between an agent’s component-level generation and its ability to perform holistic pipeline orchestration. Further details are provided in App. A.1.

###### 2.3 Annotation Pipeline

DAComp is constructed by 8 experts through a rigorous pipeline to ensure realism, quality, and consistency. Further details and examples are provided in App. E.

- 1) Data collection. The benchmark is grounded in permissively licensed assets (e.g., Apache-2.0, MIT). For the DE task, we collect 73 enterprise-scale SaaS schemas with data transformation projects, averaging 400 columns each, and populate them with large-scale, relationally consistent synthetic data (see App. E). For the DA task, we curate 100 complex databases from the Web and supplement them with analytical modeling layers derived from DE-transformed data.
- 2) Task design. At this stage, we generate the DAComp questions. For DA , annotators first draft 8 open-ended analytical questions per analysis-ready table. Five annotators then vote based on realism and difficulty, and the top 2 are retained. For DE-Evol , practicing data engineers author new business requirements aligned with enterprise scenarios and professional standards. For DE-Impl , we reverse engineer selected SaaS transformation projects into a single data_contract.yaml, capturing the full DAG and semantics. For DE-Arch , starting from the analytics layer of DE-Impl and DE-Evol examples, DA annotators propose 5 candidate business requirements per project, from which a data engineer selects 1 feasible yet challenging requirement.
- 3) Evaluation construction. We design evaluation protocols for each task. For DA, annotators build hierarchical rubrics as described in §2.2, with at least 3 annotators annotate each question, followed by alignment discussion to resolve discrepancies. For the GSB protocol, experienced data analysts author shared scoring criteria, and baseline reports are created by combining outputs from multiple LLMs. A critical aspect of this rubric design is the enumeration of valid solution Paths, a process governed by three key principles: (i) ensuring Paths represent distinct, methodologically-sound strategies, not incremental steps; (ii) validating deterministic outputs against programmatically calculated and verifiable anchor values; and (iii) utilizing methodology-based soft constraints to fairly evaluate valid but unenumerated solution paths. (see examples in App.C.4, discussion in App.F.1). To ensure the comprehensiveness of our rubric, we perform a validation step: we sample outputs from five diverse LLMs and confirm that our enumerated paths can account for all observed solution strategies, which minimizes the risk of false negatives by ensuring that valid but unanticipated solutions are not unfairly penalized. For DE-Impl and DE-Evol , solutions are deterministic: we implement execution scripts to automatically validate outputs against gold repositories, assigning partial credit at the node/layer level to capture step-wise correctness.

Table 1 Comparison of DAComp with other agent benchmarks, highlighting key differences in task scope, task paradigm, and evalution method. DAComp-zh shares the identical task set.

RepoLevel

# Cols/ Schema

Code Scale (LOC)

Primary Output

Openended

Evaluation Method

Benchmark Field # Tasks

Agentic Benchmarks

SWE-Bench [14] Software Engineering 2,294 ✓ N/A 32.8 Code Patch ✗ Execution-based WebArena [42] Web Navigation 812 ✓ N/A N/A Actions ✗ Execution-based OSWorld [33] Computer Control 369 ✓ N/A N/A Actions ✗ Execution-based BrowserComp [31] Deep Research 2,000 ✓ N/A N/A Answer ✓ Objective

Data Agent Benchmarks DS-1000 [17] Data Science 1,000 ✗ N/A 3.6 1 Script ✗ Execution-based BIRD [20] Text-to-SQL 12,751 ✗ 54 23.5 1 SQL ✗ Execution-based Spider 2.0 [18] Text-to-SQL 632 ✗ 320 104.6 1 SQL ✗ Execution-based BIRD-CRITIC [21] SQL Debugging 1,100 ✗ 54 50∼70 1 SQL ✗ Execution-based DA-Code [12] Data Science 500 ✗ 50 ∼ 100 85 1 Script ✗ Objective DSBench [15] Data Science 540 ✗ 27 10∼20 N Scripts ✗ Objective KramaBench [16] Data Science Pipelines 104 ✗ 13 50∼100 N Scripts ✓ LLM-judge BLADE [10] Data Analysis 259 ✗ 10 ∼ 12 70∼80 Report ✓ LLM-judge DABStep [7] Data Analysis 450 ✗ 10 ∼ 12 100 Answer ✗ Objective

Data Engineering &

Data Analysis 210 ✓ 382 ∼ 2,000 DocN SQL/Script+ Report Both

Execution-based & LLM-judge(rubrics)

DAComp

Table 2 Key statistics for DAComp. All metrics are per-example averages, except #Total tasks.

Metric Value Metric Value Overall (DE-Arch/DE-Impl/DE-Evol/DA) DAComp-DE

#Total tasks 30 / 30 / 50 / 100 DE-Impl raw data (#Tab. / #Col.) 23.3 / 381.6 #Question Tokens 166 / 30,883 / 6,508 / 90 #LOC code scale (Impl / Evol) 2,296 / 949.6 DAComp-DA #Change files (Impl / Evol) 37.0 / 11.7 Columns / Tables 84.7 / 3.9 #Change columns (Impl / Evol) 1,239 / 530.9 LOC 433 #DE-Arch rubric 18.5 Rubrics (Reqs / Sub-reqs / Paths / Items) 3.1 / 5.7 / 12.7 / 22.4 DE-Impl layer (#Staging / #Core / #Mart) 16.0 / 11.8 / 8.8 Completeness / Accuracy / Insightfulness 14% / 66% / 20% DE-Evol table change types (#create / #edit) 3.76 / 7.90

###### 2.4 Dataset Statistics

###### We present a statistical analysis of DAComp, highlighting its main features in comparison with prior datasets in Tab. 1, and providing more detailed characteristics in Tab. 2.

DAComp-DE quantifies enterprise-scale engineering complexity. The statistics for DAComp-DE underscore its large scale and complexity—defined by its repo-level paradigm, schemas averaging 412 columns, and solutions requiring over 2,000 lines of code—setting it apart from prior data agent benchmarks. Unlike benchmarks that focus on generating isolated scripts, DAComp introduces tasks on industrial schemas with an average of 32 tables and 412 columns. The engineering effort required is substantial. Implementation tasks involve building entire pipelines from scratch, averaging 4,612 lines of code across 43 distinct files. Evolution tasks simulate realistic maintenance with edits averaging 1,718 LOC across 13 files, agents need to manage data transformation across a multi-layered data model (staging, core, and mart). The staging layer involves data cleaning operations, a central topic in data governance, which we categorize into four types: validity constraints, consistency constraints, integrity & uniqueness, and anomaly detection (as shown in Fig. 3). Intermediate and marts layers typically focus on complex business logic, entity integrations, and metric aggregations.

[Figure 61]

Anomaly Detection

10.7%

Integrity

Validity

18.9%

48.1%

Consistency

22.3%

Figure 3 Data cleaning tasks of DE-Impl staging layer.

Table 3 DAComp-DE Baseline Performance. All models are evaluated using the DE-Agent framework (details in App. B.2) across both Implementation (CFS, Max-CFS@8, CS, Max-CS@8) and Evolution (SR@8, CFS, Max-CFS@8); see App. A.1 for metric definitions. The final column reports the aggregated DE Score.

Implementation Evolution

Method Architecture

DE Score CFS Max-CFS@8 CS Max-CS@8 CFS Max-CFS@8 SR@8

GPT-5 63.93(±2.33) 30.79 39.87 61.98 68.77 38.75 47.23 20.00 43.45 Gemini-2.5-Pro 51.96(±1.78) 27.66 36.88 55.32 65.32 23.97 38.92 8.00 32.88 Qwen3-Coder 51.43(±3.14) 23.64 32.86 54.21 63.78 27.12 39.77 12.00 32.80

- DeepSeek-V3.1 52.66(±2.88) 22.33 30.73 50.04 60.46 24.11 35.01 10.00 31.41 o3 48.32(±2.13) 15.07 22.32 35.55 47.81 24.42 32.07 6.00 28.39 Qwen3-235B-A22B 50.73(±2.05) 2.43 5.77 20.15 31.03 12.43 21.89 2.00 20.15

- Qwen3-8B 45.12(±2.06) 1.31 2.34 15.33 21.23 15.89 19.12 2.00 19.89

Table 4 DAComp-DE-zh(Chinese) Baseline Performance.

Method Architecture

Implementation Evolution

DE Score CFS Max-CFS@8 CS Max-CS@8 CFS Max-CFS@8 SR@8

GPT-5 63.60(±2.14) 30.49 39.24 61.85 68.43 37.88 46.91 20.00 42.88 Gemini-2.5-Pro 51.90(±3.43) 26.98 36.73 55.18 65.07 24.28 38.27 8.00 32.55 Qwen3-Coder 51.11(±3.35) 23.23 32.97 54.59 63.69 26.59 39.37 12.00 32.36 DeepSeek-V3.1 53.08(±2.54) 22.62 30.84 50.22 60.34 24.69 35.17 8.00 31.87 o3 48.02(±1.79) 15.00 22.15 35.10 47.45 24.23 32.59 6.00 28.20 Qwen3-235B-A22B 50.61(±2.50) 2.31 5.83 20.03 31.27 13.01 21.27 0.00 20.35

- Qwen3-8B 46.22(±1.90) 1.21 2.16 15.78 21.59 15.19 19.35 0.00 19.84

DAComp-DA measures analytical depth and methodological diversity. The design of DAComp-DA moves beyond simple question-answering to assess deep analytical reasoning. Uniquely, DAComp evaluates both deterministic engineering and open-ended analysis, a distinction from prior benchmarks that typically focus on only one paradigm. Its open-ended nature is quantified by our hierarchical rubrics, which decompose each of the 100 DA tasks into an average of 3.1 requirements and 5.7 sub-requirements, accommodating roughly 13 valid solution paths. This methodological diversity is evaluated with a multi-faceted rubric where scoring items are weighted toward Accuracy (66%) but also reward Completeness (14%) and Insightfulness (20%). While the analytical schemas are more focused than in DE tasks (averaging 4 tables and 85 columns), the required reasoning is still complex, reflected in an average solution length of 347 lines of code—significantly longer than typical text-to-SQL or single-script data science tasks. Crucially, DAComp-DA places a strong emphasis on open-ended data visualization, requiring agents to autonomously select and generate charts that effectively communicate their findings.

#### 3 Experiments

###### 3.1 Experimental Setup

We evaluate state-of-the-art LLMs, including open-source models like Qwen3 [35], DeepSeek-V3.1 [22], and Kimi-K2 [29], as well as proprietary ones such as the Gemini [28], and GPT [24] families. We utilize the widely adopted OpenHands (CodeAct-Agent) framework [30] for both DE and DA tasks. Additionally, we developed a custom baseline named DA-Agent for DAComp-DA, which operates via Bash and file system interactions and is capable of executing Python and SQL. The performance of each agent is measured using the metrics detailed in §2.2. We also report two aggregate scores: the DE Score, which is the mean score across all DE tasks (using CFS for Implementation/Evolution), and the Overall Score, representing the mean across the entire benchmark. For the DA score, we use α = 0.6 to aggregate the rubric and GSB scores, with Gemini-2.5-Flash serving as the LLM judge. Further details on the experimental setup and additional results are provided in App. B.

Table 5 Detailed performance breakdown on the DAComp-DA benchmark.

Method Completeness Accuracy Insightfulness Readability Analytical Depth Visualization DA Score OpenHands Baseline

GPT-5 60.98 40.3 49.39 35.51 69.8 21.4 46.99 Gemini-2.5-Pro 45.02 30.22 40.71 48.2 31.0 15.0 33.38

- o3 40.13 25.5 20.45 26.22 27.11 6.8 26.57 DeepSeek-V3.1 49.88 33.25 41.66 36.0 33.2 11.0 33.87 Qwen3-Coder 33.42 21.21 25.06 20.0 13.73 4.8 24.28

- Qwen3-235B-A22B 30.7 12.23 22.11 3.6 1.8 0.8 12.43

DA-Agent Baseline

GPT-5 64.23(±2.37) 43.81(±3.43) 56.89(±6.48) 43.59(±6.08) 76.80(±4.91) 27.44(±4.44) 50.84(±3.12) Kimi-K2 52.31(±1.13) 33.56(±2.09) 46.82(±2.48) 62.20(±3.01) 63.75(±2.84) 14.40(±2.33) 41.89(±1.78) Gemini-2.5-Pro 45.43(±1.34) 30.30(±0.27) 41.45(±0.71) 51.60(±2.73) 35.75(±2.35) 13.40(±2.94) 34.70(±1.39) DeepSeek-V3.1 48.74(±2.09) 32.97(±1.40) 42.43(±1.89) 37.25(±2.21) 35.00(±1.57) 11.45(±1.31) 34.33(±0.45) o3 40.73(±0.63) 29.54(±2.93) 23.95(±3.86) 25.24(±2.51) 23.81(±3.37) 7.32(±1.27) 28.20(±1.37) Qwen3-Coder 35.12(±2.21) 20.05(±2.35) 25.53(±1.83) 19.37(±1.44) 13.42(±2.38) 5.15(±0.85) 25.13(±0.82) Doubao-Seed-1.6 37.45(±1.95) 18.45(±2.55) 27.51(±2.00) 13.25(±2.48) 9.01(±1.25) 6.80(±1.96) 20.74(±0.82) Qwen3-235B-A22B 29.37(±1.09) 13.11(±1.33) 21.50(±1.81) 3.64(±0.33) 1.56(±0.81) 1.87(±0.78) 13.25(±0.65) Qwen3-8B 9.89(±2.46) 4.12(±0.32) 5.05(±1.70) 0.13(±0.15) 0.00(±0.00) 0.15(±0.19) 4.47(±0.63)

Table 6 Detailed performance breakdown on the DAComp-DA-zh (Chinese) benchmark.

Method Completeness Accuracy Insightfulness Readability Analytical Depth Visualization DA Score OpenHands Baseline

GPT-5 70.56 47.08 57.19 19.6 46.4 22.0 43.69 Gemini-2.5-Pro 55.51 29.9 47.17 38.8 18.8 10.2 31.22

- o3 49.79 30.73 40.74 17.55 10.61 8.2 27.87

- DeepSeek-V3.1 54.5 32.93 42.56 8.2 5.0 3.6 24.16 Qwen3-Coder 43.14 20.38 25.69 2.47 1.1 2.04 21.84 Qwen3-235B-A22B 29.44 14.27 17.35 1.22 0.0 0.98 11.5

DA-Agent Baseline

GPT-5 72.69(±1.41) 46.96(±1.94) 61.56(±2.51) 39.35(±2.19) 66.40(±3.43) 25.40(±1.87) 49.49(±1.04) Gemini-2.5-Pro 54.63(±2.53) 33.33(±1.58) 48.56(±0.50) 49.95(±3.84) 26.20(±2.47) 9.00(±3.52) 33.75(±1.67) Kimi-K2 57.08(±0.55) 33.54(±2.99) 47.64(±1.32) 34.52(±2.35) 20.28(±3.07) 3.86(±2.14) 31.22(±0.75) o3 51.10(±1.75) 30.68(±2.97) 34.92(±1.29) 20.00(±0.57) 12.54(±2.54) 6.35(±1.22) 28.70(±1.15)

- DeepSeek-V3.1 55.15(±2.49) 34.01(±2.36) 44.62(±2.89) 7.15(±1.98) 4.65(±2.00) 6.30(±2.42) 27.75(±2.04) Qwen3-Coder 43.35(±1.76) 22.75(±3.15) 30.83(±2.38) 4.07(±0.98) 1.55(±1.02) 1.75(±0.50) 22.64(±1.19) Doubao-Seed-1.6 45.92(±2.07) 18.73(±2.05) 33.23(±1.06) 3.23(±1.12) 0.75(±0.68) 1.55(±0.66) 17.83(±1.33)

- Qwen3-235B-A22B 31.64(±2.71) 13.48(±0.19) 22.27(±1.22) 0.87(±0.64) 0.13(±0.12) 0.33(±0.42) 12.74(±0.33) Qwen3-8B 14.55(±1.04) 6.30(±2.18) 6.08(±2.15) 0.00(±0.00) 0.00(±0.00) 0.00(±0.00) 6.33(±1.25)

###### 3.2 Main Results

DE results. As shown in Tab. 4, GPT-5 establishes a definitive lead, consistently achieving the highest aggregated DE Scores across different orchestration frameworks. Notably, specialized open-source models like Qwen3-Coder and DeepSeek-V3.1 demonstrate exceptional efficacy, effectively rivaling general-purpose proprietary models such as Gemini-2.5-Pro. However, the absolute performance metrics reveal a sobering reality regarding the complexity of repository-level engineering: even the state-of-the-art GPT-5 achieves a modest DE Score of approximately 42.88% and a strict Success Rate of merely 20.00%. This profound performance ceiling underscores that while framework optimizations can stabilize interaction, DAComp-DE poses a rigorous challenge that current LLMs—regardless of their scale or specialization—have yet to master, highlighting a critical gap between isolated code generation and holistic system orchestration.

DA results. The results in Tab. 6 reveal a significant capability gap in open-ended analysis, with the top overall score solely reaching 56.14%. A dimension-wise analysis uncovers three critical insights. First, Analytical Depth and Insightfulness serve as the primary differentiators between tiers. While GPT-5 dominates by maintaining high scores across all dimensions, reasoning-focused models like o3 exhibit a distinct “calculator behavior”: despite achieving competitive Accuracy (40.99) and Completeness (60.73), o3 suffers severely in

Readability (24.63) and Depth (13.37), indicating an ability to compute correct numbers but a failure to synthesize them into human-readable insights. Second, the gap between DeepSeek-V3.1 (39.16%) and the code-specialized Qwen3-Coder (28.07%) is driven largely by qualitative metrics; Qwen3-Coder nearly collapses on Readability (3.15) and Visualization (1.93), suggesting that open-ended analysis requires holistic reasoning beyond mere SQL generation. Finally, the task complexity establishes a strict capacity threshold, where smaller models like Qwen3-8B fail to generate coherent analytical artifacts.

###### 3.3 Performance Analysis of Repository-level Data Engineering

Holistic orchestration is the core bottleneck in data engineering. Across DE tasks, models plan well but struggle to execute end-to-end. Evolution scores are relatively high (e.g., GPT-5: 37∼38%), yet strict SR for Evolution are much lower (typically < 20%). The drop from component-level correctness (CS) to cascading failure scores (CFS) is pronounced for strong models, revealing a pipeline-level orchestration bottleneck beyond single-file correctness; for example, GPT-5 (DAComp-DE-Agent) in Implementation falls from CS 61.85 to CFS 30.49, and in Evolution from CFS 37.88 to SR 20.00. By contrast, weaker open-source models (e.g., Qwen3-8B) exhibit very low CS (Implementation 1.21), indicating deficits already at the component level; orchestration then compounds failure but is not the sole cause. The uniformly low CFS across models confirms that coordinating dependencies in a live repository—rather than generating isolated correct code—is the dominant challenge in DAComp-DE.

Medium-scale code edits are the most difficult to perform. To gain a more granular understanding, we delve into a node-level analysis, studying the scores for individual SQL file modifications (Fig. 4). We classify these modifications into two types—editing an existing file or creating a new file, and group them by the required number of lines. For create tasks, models like GPT-5 have a clear “sweet spot” on medium-scale creations (20 − 150 lines), while all models struggle with very large files (> 150 lines). In In contrast, edit tasks exhibit a non-linear difficulty trend. Contrary to intuition, medium-scale edits prove to be the most challenging. This is because minor edits are often trivial, while very large edits frequently involve repetitive, boilerplate transformations with clear logic. In contrast, medium-scale edits tend to contain the most complex and nuanced changes to business logic, aggregations, and calculations, thus posing the greatest reasoning challenge.

|GPT-5 Qwen3-Coder<br><br>| | | |Gemini-2.5-pro DeepSeek-V3.1<br><br>| | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
|Edit Tasks| | | |Create Tasks| | | |
| | | | | | | | |

0.6

0.5

ComponentScore

0.4

0.3

0.2

0.1

0.0

Edit <5 lines

Edit 5-10 lines

Edit >10 lines

Create <20 lines

Create 20-150 lines

Create >150 lines

Figure 4 Component-level performance analysis.

Analytical complexity and failure rates escalate in higher pipeline layers. Fig. 5 reveals that the difficulty of data engineering tasks escalates significantly as agents move from the initial data ingestion layer to the more complex analytical layers. The staging layer, focused on basic cleaning, consistently has the fewest local errors and the highest task survival rate. The challenge intensifies dramatically in the intermediate (core) layer. This is where the most complex business logic and entity integration occurs, and as Panel (a) shows, it is where the largest share of local errors originates. The severe impact of this difficulty is evident in Panel (b), which shows the sharpest drop in pipeline survival occurring after this stage. Finally, the marts layer remains highly challenging. Failures in this final stage are often a direct consequence of inheriting

(a) Distribution of Local Errors (by 'CS')

(b) Pipeline Survival Rate (by 'CFS')

100%

GPT-5 Qwen3-Coder O3 Gemini-2.5-Pro DeepSeek-V3.1 Qwen3-235B

| |
|---|

60%

| |
|---|

PercentageofTasksRemaining(%)

| |
|---|

PercentageofAllErrors(%)

80%

| |
|---|

50%

| |
|---|

| |
|---|

40%

60%

30%

40%

GPT-5

20%

Qwen3-Coder

O3

20%

Gemini-2.5-Pro

10%

DeepSeek-V3.1

Qwen3-235B

0%

0%

Staging Core Marts

Start After Staging After Core After Marts

Figure 5 Error distribution (left), pipeline survival rate (right).

upstream errors from the core layer, with fewer than 20% of the initial tasks surviving to completion. Together, these results demonstrate a clear hierarchy of difficulty, with the analytical complexity of the core and marts layers posing a substantially greater challenge than the initial staging layer.

Top-performing agents exhibit stable and task-aligned interaction patterns. Fig. 6 shows the distribution of interaction turns in DE tasks. High-performing models such as GPT-5 maintain moderate turn counts with compact variance across both Implementation and Evolution settings, reflecting efficient yet sufficiently thorough reasoning. In contrast, weaker models like Qwen3 either generate excessively long and volatile traces in Implementation or display unusually short traces in Evolution, where premature termination often corresponds to incorrect or incomplete outputs. These patterns indicate that stable and centered turn distributions are more characteristic of effective agents than simply minimizing the number of turns.

###### 3.4 Error Analysis of Repository-level Data Engineering

In DAComp-DE, errors emerge across various tasks and dimensions. The overall distribution of errors, illustrated

DE-Implementation #Turn

DE-Evolution #Turn

250

- in Fig. 7, highlights the frequency and nature of common failure types. The predominant challenges in DE-Impl and DE-Evol tasks include dependency errors, SQL omissions, and cascading logic failures. Further breakdowns can be found in App. D.

200

175

200

150

150

125

#Turns

#Turns

100

100

75

50

50

Dependency management challenges. Dependency errors remain a central bottleneck in DE-Impl and DE-Evol tasks. As seen in Tab. 16, models consistently struggle with managing dependencies, with rates surpassing 65% across all evaluated models. This issue is particularly evident in DE-Evol tasks, where models exhibit a pronounced difficulty in retaining context across schema modifications. The data suggests that current models, even state-of-the-art ones like GPT-5, face challenges in maintaining an accurate global data lineage. The imbalance between missing dependencies and extra dependencies, shown in Tab. 19, reflects this difficulty, indicating a need for more robust context retention and better handling of long-range dependencies within complex data models.

25

DeepSeek-V3.1-baselineGPT5-baselineGemini-2.5-Pro-baselineQwen3-baselineDeepSeek-V3.1-openhandsGPT5-openhandsGemini-2.5-Pro-openhandsQwen3-openhands

DeepSeek-V3.1-baselineGPT5-baselineGemini-2.5-Pro-baselineQwen3-baselineDeepSeek-V3.1-openhandsGPT5-openhandsGemini-2.5-Pro-openhandsQwen3-openhands

Figure 6 Turn counts on DE tasks.

[Figure 62]

SQL omission and data complexity. SQL omission rates significantly increase as the task complexity grows, particularly when dealing with multilayered data models. As illustrated in Tab. 17, omission rates escalate from the Staging layer to the Marts layer, a result of the growing complexity of business logic. Weaker models such as Qwen3-8B fail catastrophically in the Marts layer, with omission rates approaching 100%. In contrast, advanced models like GPT-5 exhibit far better robustness, maintaining omission rates below 10%. This disparity underscores the gap in handling complex, multilevel data transformations, which requires not just syntactical correctness but also a deeper understanding of the data structure.

Data Type 3.0%

Missing Col6.3%

Missing Deps 20.5%

Marts 10.2%

Dependency41.5%

26.1%SQL Omission

Inter. 8.1%

21.0%Extra Deps

Staging 7.7%

Calculation 23.2%

Intrinsic Logic

Upstream

10.5%

Error 12.7%

Calculation logic errors and upstream propagation. Tab. 18 highlights a significant “cascading effect” in calculation logic errors, where upstream errors propagate through the pipeline, compounding failures. In highperforming models like GPT-5 and Gemini-2.5-Pro, upstream errors are approximately three times more prevalent than intrinsic errors. This suggests that optimizing performance in DE-Impl tasks requires a focus on improving fault tolerance and consistency management across the entire pipeline, rather than just focusing on improving single-node logic generation.

Figure 7 Error analysis of DE task.

###### 3.5 Analysis of Open-ended Data Analysis Tasks

Performance across analytical objectives. To investigate how performance correlates with the nature of the analytical task, we manually classify each DA task into five categories based on its primary objective: Descriptive, Diagnostic, Strategic, Pattern Recognition, and Profiling (see App. C.4 for definitions). As shown

- in Fig. 8, this classification reveals a distinct performance hierarchy. Agents excel at concrete Descriptive tasks (what happened?), but their scores drop sharply on more abstract Diagnostic (why did it happen?) and Strategic (what should we do?) tasks. This confirms that these more complex objectives are not only more challenging but also serve as better differentiators of advanced model capabilities.

68%

70

GPT-5

DeepSeek-V3.1

Gemini-2.5-Pro

GPT-5

| |
|---|

60

Gemini-2.5-pro O3

Qwen3-Coder

58%

| |
|---|

60

16.4%

19.3%

19.9%

21.4%

52%

50

50

TypeRatio(%)

DAScore(%)

63.7%

59.3%

38%

40

40

Qwen3-235B-A22B

DeepSeek-v3.1

30

30

19.4%

20.3%

21.3%

23.2%

20%

20

20

55.5%

60.3%

10

Planning & Preparation Failures

10

0

Execution & Calculation Failures

Descriptive Diagnostic Strategic Pattern Rec. Profiling

Interpretation & Synthesis Failures

Figure 8 DA performance across five analytical objectives.

Figure 9 DA error distribution.

Error analysis. As shown in Fig. 9, we classify DA failures into three stages: Planning, Execution, and Interpretation. The quantitative breakdown reveals a consistent hierarchy of difficulty across all models: Execution & Calculation Failures dominate the error distribution, averaging 59.05% of all failures. This underscores that the primary bottleneck for current agents lies in calculation accuracy and code grounding capabilities. However, the challenges are not solely technical; Planning (20.65%) and Interpretation (20.30%) remain significant sources of error. Collectively, these cognitive stages account for two-fifths of the total performance gap, suggesting that while enhancing execution robustness is the most pressing priority, achieving reliable autonomous analysis requires holistic improvements across the full lifecycle—from initial requirement decomposition to the final synthesis of insights.

###### 3.6 Validation of LLM-Judge Method

To rigorously validate the reliability of our evaluation framework, we conduct extensive analyses across four dimensions: human-model alignment, cross-judge consistency, stochastic stability, hyperparameter robustness with 50 examples.

Human-model alignment. To validate our LLMs-asJudge method, we conduct a large-scale agreement study on a dataset of 300 model responses generated by 8 distinct LLMs. These responses were manually annotated by expert humans against over 7,000 specific rubric items and GSB documents. We establish a reliable ground truth by measuring inter-rater agreement, which yielded high consistency scores (e.g., Rubric case ICC=0.925, Item κw=0.906), confirming the robustness of our human baseline (Tab. 7). With this human baseline, we benchmark several candidate judges (e.g., Gemini 2.5 Flash, o4-mini, GPT-4.1)

Table 7 Inter-rater and human–model agreement. (Details in App.B.4)

Rubric (N=300, 7k items) GSB Item (N=600 pairs) Model / Metric Item Case Model Read. Prof. Vis.

(κw) (ICC(A,1)) (τb) (κw) (κw)) (κw) Human Inter 0.906 0.925 1.000 0.601 0.751 0.753 o4-mini 0.827 0.881 1.000 0.609 0.758 0.742 Gemini-2.5-Flash 0.834 0.890 1.000 0.604 0.759 0.735 GPT-4.1 0.797 0.848 1.000 0.596 0.786 0.748 Gemini-2.5-Pro 0.808 0.878 1.000 0.602 0.765 0.751 Kimi-K2-Thinking 0.808 0.872 1.000 0.575 0.732 – DeepSeek-V3.1 0.782 0.870 1.000 0.588 0.725 – Qwen3(-VL)-235B 0.737 0.758 1.000 0.531 0.713 0.682 Qwen3(-VL)-30B 0.680 0.775 1.000 0.507 0.691 0.656

at three primary levels of agreement: (i) case-level

agreement, which measures how consistently the judge scores a single task compared to human experts; (ii) model-level agreement, which validates whether the judge’s final ranking of all models matches the humanderived leaderboard; and (iii) item-level agreement, which evaluates the consistency of atomic judgments between the model and human experts at the granularity of individual rubric items or GSB document pairs. As shown in Tab. 7, Gemini-2.5-Flash demonstrates exceptional alignment, achieving the highest Rubric Item κw (0.834) and Case ICC (0.890) among all models, effectively matching human-level reliability. While GSB Readability scores show expected variance due to subjectivity (κw ≈ 0.53), the judge maintains high precision on objective dimensions like depth and visualization, justifying its selection as our standard evaluator.

Table 8 Ranking stability across judges. High correlations (τb) confirm leaderboard robustness against family bias.

Agent Model Primary Alternative Judges Flash Pro GPT-4.1 Qwen-235B Qwen-30B

GPT-5 56.14 59.52 63.37 71.57 53.72 o3 36.08 40.08 44.25 50.76 31.63 Gemini-2.5-Pro 39.46 45.69 50.98 55.48 35.70 DeepSeek-V3.1 39.16 44.68 50.61 54.58 41.44 Qwen3-Coder 28.07 32.12 36.14 43.79 25.86 Qwen3-235B 18.84 20.85 21.77 23.81 18.30 Kimi-K2 36.94 43.77 47.83 53.55 32.93

Rank Corr. (τb) --- 1.00 1.00 1.00 0.90

Table 9 Ranking stability across weighting hyperparameters (α). Results show perfect invariance (τb = 1.00).

Agent Model Primary Alternative α

α = 0.6 α = 0.5 α = 0.8 α = 0.9

GPT-5 56.79 52.14 58.30 60.49 o3 36.33 30.45 39.89 43.86 Gemini-2.5-Pro 39.36 34.36 42.05 44.83 DeepSeek-V3.1 33.82 26.86 38.33 43.54 Qwen3-235B 18.84 14.39 21.69 24.98

Rank Corr. (τb) --- 1.00 1.00 1.00

Cross-judge consistency. To rigorously mitigate concerns regarding family-specific bias (e.g., self-preference) and verify leaderboard reproducibility, we conducted a ranking stability analysis using a diverse set of proprietary and open-source judges. As presented in Tab. 8, the relative rankings of agents exhibit exceptional consistency, achieving perfect correlation (τb = 1.00) across the majority of evaluators. Crucially, evaluating the Gemini agent with non-Gemini judges (e.g., GPT-4.1) yields an identical ranking position, effectively refuting the hypothesis of family bias. Consequently, given that the choice of judge model does not statistically alter the leaderboard, we standardize on Gemini-2.5-Flash for its superior balance of stability and cost-efficiency.

Hyperparameter robustness. The final DA score is a weighted aggregation: Scoreda = α · Scorerubric + (1 − α) · Scoregsb. While DAComp’s granular dimensional design allows developers to adjust α according to their specific preference for accuracy versus presentation, we standardize on α = 0.6 for general jiu to ensure that objective technical correctness (Rubric) remains the dominant factor. To verify the validity of this choice, we conduct a sensitivity analysis across configurations (α ∈ {0.5,0.8,0.9}). As detailed in Tab. 9, the relative rankings remain invariant (τb = 1.00) across all settings, demonstrating that our generalized standard is robust while offering flexibility for specialized use cases.

Table 10 Variability of scores across 8 independent grading runs on fixed outputs (mean ± std).

Stochastic stability. To assess the reproducibility of our scoring mechanism, we quantify the variability arising specifically from the LLM judge’s stochasticity. We performed 8 independent grading runs on a fixed set of identical agent responses. As shown in Tab. 10, the standard deviations of the final scores are consistently negligible (< 0.35), demonstrating that our evaluation protocol yields statistically stable and reproducible grades despite the inherent randomness of LLM generation.

Model DE-Arch DA

GPT-5 61.3 ± 0.18 56.1 ± 0.16 DeepSeek-V3.1 53.2 ± 0.25 39.1 ± 0.22 Gemini 2.5 Pro 51.0 ± 0.21 39.4 ± 0.22 O3 54.8 ± 0.19 36.1 ± 0.20 Qwen3-235B 50.4 ± 0.31 18.8 ± 0.29

#### 4 Related Work

Agentic benchmarks. As LLM-based agents mature, benchmarks span tool use [37], software engineering [14, 40], mobile interaction [26], web navigation [5, 42], computer use [33], scientific discovery [4], and deep research [25, 31], collectively advancing the field. In parallel, evaluation has moved beyond fixed-answer

grading toward open-ended assessment [2, 6, 9, 19, 27, 32, 34]. DAComp is, to our knowledge, the first benchmark to cover the data-intelligence workflow, evaluating end-to-end data agents on both repository-level data engineering and open-ended data analysis, with the aim of advancing autonomous engineering and analytical capability.

Benchmarks for data agents. A data agent is an LLM-driven autonomous system that plans and executes end-to-end workflows, acquiring, transforming, and analyzing data via tool use and code execution to achieve user-defined objectives. Early work emphasizes single-shot tasks such as text-to-SQL [20, 39] and code generation [17, 38]; more recent efforts push toward realistic SQL generation over real scenarios [13, 18, 21], multi-turn data-science code generation [11, 12, 15] with iterative execution, and data analysis in business settings [7, 10, 16]. DAComp goes beyond these efforts by introducing the first benchmark spanning enterprise data-intelligence workflows, encompassing repository-level engineering and open-ended analysis, and offering a rigorous testbed for advancing autonomous agents.

#### 5 Conclusion

In this work, we presented DAComp, a comprehensive benchmark designed to evaluate data agents across the full data intelligence lifecycle. DAComp bridges the gap between isolated code generation and realworld enterprise demands by introducing two rigorous testbeds: DAComp-DE for repository-level pipeline orchestration and DAComp-DA for open-ended analytical reasoning. Our extensive experiments reveal a significant capability gap: even state-of-the-art models falter in holistic system maintenance and strategic insight synthesis, with success rates falling below 20% on engineering tasks. Furthermore, the inclusion of DAComp-zh paves the way for assessing agent robustness in multilingual environments, fostering the development of globally adaptable systems. By establishing this rigorous standard, DAComp aims to steer the community beyond mere technical accuracy, driving the evolution of truly autonomous and capable data agents for the enterprise.

#### References

- [1] Anthropic. Introducing Claude 4. https://www.anthropic.com/news/claude-4, 2025.
- [2] Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

- [3] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024.

- [4] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. In The Thirteenth International Conference on Learning Representations, 2024.

- [5] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

- [6] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A comprehensive benchmark for deep research agents. arXiv preprint arXiv:2506.11763, 2025.

- [7] Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, and Thomas Wolf. Dabstep: Data agent benchmark for multi-step reasoning. arXiv preprint arXiv:2506.23719, 2025.

- [8] Gemini. Gemini 2.5: Our most intelligent AI model. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025.
- [9] Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao Yu, Bernal Jiménez Gutiérrez, Yiheng Shu, et al. Mind2web 2: Evaluating agentic search with agent-as-a-judge. arXiv preprint arXiv:2506.21506, 2025.

- [10] Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, et al. Blade: Benchmarking language model agents for data-driven science. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13936–13971, 2024.

- [11] Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, et al. Infiagent-dabench: Evaluating agents on data analysis tasks. In Forty-first International Conference on Machine Learning, 2024.

- [12] Yiming Huang, Jianwen Luo, Yan Yu, Yitong Zhang, Fangyu Lei, Yifan Wei, Shizhu He, Lifu Huang, Xiao Liu, Jun Zhao, et al. Da-code: Agent data science code generation benchmark for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13487–13521, 2024.

- [13] Nan Huo, Xiaohan Xu, Jinyang Li, Per Jacobsson, Shipei Lin, Bowen Qin, Binyuan Hui, Xiaolong Li, Ge Qu, Shuzheng Si, et al. Bird-interact: Re-imagining text-to-sql evaluation for large language models via lens of dynamic interactions. arXiv preprint arXiv:2510.05318, 2025.

- [14] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2023.

- [15] Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, Hongming Zhang, Xinya Du, and Dong Yu. Dsbench: How far are data science agents to becoming data science experts?, 2024. URL https://arxiv.org/abs/2409.07703.
- [16] Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Sivaprasad Sudhir, Om Chabra, Anna Zeng, Anton A Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, et al. Kramabench: A benchmark for ai systems on data-to-insight pipelines over data lakes. arXiv preprint arXiv:2506.06541, 2025.

- [17] Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pages 18319–18345. PMLR, 2023.

- [18] Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin Su, Zhaoqing Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, et al. Spider 2.0: Evaluating language models on real-world enterprise text-to-sql workflows. arXiv preprint arXiv:2411.07763, 2024.

- [19] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-as-judges: a comprehensive survey on llm-based evaluation methods. arXiv preprint arXiv:2412.05579, 2024.

- [20] Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. Advances in Neural Information Processing Systems, 36, 2024.

- [21] Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin, Binyuan Hui, Shuzheng Si, Nan Huo, Xiaohan Xu, Yue Zhang, et al. Swe-sql: Illuminating llm pathways to solve user sql issues in real-world applications. arXiv preprint arXiv:2506.18951, 2025.

- [22] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

- [23] OpenAI. OpenAI GPT5 System Card. https://cdn.openai.com/gpt-5-system-card.pdf, 2025.
- [24] R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2:13, 2023.

- [25] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

- [26] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.

- [27] Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, et al. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. arXiv preprint arXiv:2511.07685, 2025.

- [28] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [29] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

- [30] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. arXiv preprint arXiv:2407.16741, 2024.

- [31] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- [32] Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, et al. Writingbench: A comprehensive benchmark for generative writing. arXiv preprint arXiv:2503.05244, 2025.

- [33] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024.

- [34] Yupeng Xie, Zhiyang Zhang, Yifan Wu, Sirong Lu, Jiayi Zhang, Zhaoyang Yu, Jinlin Wang, Sirui Hong, Bang Liu, Chenglin Wu, et al. Visjudge-bench: Aesthetics and quality assessment of visualizations. arXiv preprint arXiv:2510.22373, 2025.

- [35] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [36] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022.

- [37] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

- [38] Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, et al. Natural language to code generation in interactive data science notebooks. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 126–173, 2023.

- [39] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, 2018.

- [40] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, et al. Multi-swe-bench: A multilingual benchmark for issue resolving. arXiv preprint arXiv:2504.02605, 2025.

- [41] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

- [42] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024. URL https://arxiv.org/abs/2307.13854.

## Appendix

#### A Evaluation Methods Details

###### A.1 DAComp-DE-Impl/Evol

The DAComp-DE-Impl/Evol evaluated using three execution-based metrics that progressively increase in strictness: Component Score (CS), Cascading Failure Score (CFS), and Success Rate (SR). Fig. 10 illustrates how these metrics differ in scoring a simple pipeline when an intermediate node fails.

Component score (CS). Let D be the set of tasks. For task d ∈ D, let layers be L (e.g., staging/intermediate/marts), and for each layer ℓ ∈ L let Td,ℓ be its tables with weights wd,t≥ 0. Define a table match indicator md,t ∈ {0,1} by exact equivalence of schema+data between predicted and gold outputs (checked in DuckDB) under perfect upstream inputs (progressive/hybrid evaluation). The per-layer score and task-level CS are

wd,t md,t t∈Td,ℓ wd,t

Sd,ℓ = t∈Td,ℓ

, CSd = 100·

ℓ∈L

We report the benchmark CS as CS = |D|1 d∈D CSd.

αℓ Sd,ℓ, with αℓ ≥ 0,

ℓ

αℓ = 1.

Cascading failure score (CFS). For task d, let the pipeline DAG be Gd = (Vd,Ed) with node weights wd,j ≥ 0 and ancestor set Ancd(j). Let md,j ∈{0,1} be the node-level exact match (schema+data) under predicted upstreams. Define the cascading indicator recursively

sCFSd,j = md,j

sCFSd,k ,

k∈Ancd(j)

and the task-level CFS

wd,j sCFSd,j j∈Vd wd,j

CFSd = 100 · j∈Vd

.

We report CFS = |D|1 d∈D CFSd. Success rate (SR). A task is successful only if every component matches:

md,j ∈ {0,1}.

SRd =

j∈Vd

The benchmark success rate is the fraction of perfectly solved tasks:

SR =

1 |D| d∈D

SRd.

In the evaluation process, we introduce the following tolerance measures to ensure the fairness and flexibility of the evaluation:

Key Column Evaluation:

- 1)Evaluate only key columns. To focus the evaluation on the core components of the task, we evaluate only the key columns in the data (e.g., business-related columns, important computational columns). This ensures that the evaluation accuracy is concentrated on the most critical parts of the task.
- 2)Exclude time columns. To avoid interference from time columns (e.g., small differences caused by different timestamps), we do not evaluate time columns. Tolerance for Numerical Columns:

Round to two decimal places. When evaluating numerical columns, we allow a certain margin of error. Specifically, for numerical columns, all values are rounded to two decimal places to ensure consistency in data precision and avoid the influence of small fluctuations on the evaluation results.

However, for DE-Evol tasks, given the high strictness of the cascading metric, we adopt a threshold-based definition where a task is deemed successful if it maintains sufficient pipeline integrity (specifically, CFSd ≥ 80).

###### CFS (Cascading Failure Score)

###### CFS (Cascading Failure Score)

###### CS（Component Score）

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Sequential evaluation along DAG. B fails → C blocked. A=1, B=0, C=0 CFS = 1 / 3 ≈ 0.33

Each node evaluated independently. A=1, B=0, C=1 CS = 2 / 3 ≈ 0.67

###### Any failure in the pipeline leads to overall failure. SR = 0

[Figure 72]

[Figure 73]

[Figure 74]

Correct (score 1) Wrong (score 0) Blocked (Score 0)

Figure 10 Illustration of how CS, CFS, and SR differ in scoring a simple pipeline when an intermediate node fails.

###### A.2 DAComp-DE-Arch

Three rubric dimensions. The evaluation of the DE-Arch tasks is conducted across three key dimensions, which are defined as follows:

- 1) Business Alignment and Semantic Accuracy: This dimension assesses how well the solution aligns with business requirements and ensures semantic correctness. It evaluates whether the proposed solution comprehensively addresses the task’s objectives while maintaining semantic integrity in the context of the recruitment cost analysis system.
- 2) Technical Feasibility and Structural Completeness: This dimension evaluates the technical feasibility of the solution and the completeness of its structure. It checks whether the proposed model can be implemented successfully given the available resources and dependencies, and whether it adheres to necessary technical standards and best practices.
- 3) Design Quality: This dimension evaluates the design and clarity of the model. It looks at how well the model is structured, the clarity of its naming conventions, and the organization of the components. It also considers the use of modular design principles to ensure that the solution is maintainable and scalable.

DAComp-de-arch judge prompt. This prompt standardizes how a model blueprint is evaluated against a given user question and rubric. It defines clear scoring logic (deterministic vs. path-based criteria), enforces an evidence-first policy (no evidence, no points), and constrains the final score to requirement-level sums. A canonical JSON output schema captures per-criterion analysis, evidence, and scores, enabling reproducible, auditable assessments across tasks.

DE-Arch Judge Prompt

# Task Description You are a professional data architect. You will evaluate a model blueprint based on a given user question and a scoring rubric. Your task is to review a set of scoring criteria for the model blueprint , and then , based on these criteria , assess the blueprint to determine the extent to which it meets the standards.

The scoring rubric provides a total score and various requirements. Where:

- - Total Score: Represents the maximum possible score after summing all scoring criteria.
- - Requirements: Represent different needs the assistant must satisfy. Each requirement contains multiple scoring criteria. These criteria are divided into two categories:
- - 1. Deterministic criteria: These can be scored directly without considering different

implementation paths.

- - 2. Non -deterministic criteria: These usually have multiple implementation paths. When evaluating

, first determine the "best matching path" based on the assistant ’s response , and then score based on

the sub -criteria under that path. If there is no clearly matching path , use your own expertise to judge whether the assistant ’s response correctly meets the requirement ’s goal and calculate if it is reasonable. If correct , assign points , but the score for this requirement cannot exceed the maximum score of other defined paths.

Final Scoring Logic: Final Score = Sum of all requirement scores. Requirement Score = Sum of all criteria scores within that requirement. Criteria Score = Direct score OR Best matching path score OR Unmatched path score OR Sum of sub criteria. Best Matching Path Score = Sum of the scores of the sub -criteria under that path.

Please analyze and score item by item according to the rubric. If you have any hesitation on any point , do not guess or make subjective assumptions -assign 0 points directly. **You must provide evidence; if evidence is missing , assign 0 points .**

<User Question Start > {user_query} </User Question End >

<Model Blueprint Start > {model_blueprint} </Model Blueprint End > <Scoring Rubric Start > {rubric} </Scoring Rubric End > You need to analyze and score each item one by one according to the scoring rubric. # Response format as follows: ‘‘‘json {{

"Requirement1": {{ "Criterion1 .1": {{

"Analysis": "Carefully read the content of the model blueprint , determine whether it meets Criterion 1.1, and assign a score.",

"Criterion1 .1.x.1": {{ "Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 1.1.x.1, and assign a score.", "Evidence": [], "Score": int

}}, "Criterion1 .1.x.2": {{

"Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 1.1.x.2, and assign a score.", "Evidence": [], "Score": int

}}, "Score": int

}}, "Criterion1 .2": {{

"Analysis": "Analyze the reason for the best matching path , determine the best matching path: The best matching path is Path 1.2.x",

"Criterion1 .2.x.1": {{ "Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 1.2.x.1, and assign a score.", "Evidence": [], "Score": int

}}, "Criterion1 .2.x.2": {{

"Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 1.2.x.2, and assign a score.", "Evidence": [], "Score": int

}}, "Score": int

}}, "Total Score": int

}}, "Requirement2" : {{

"Criterion2 .1": {{

"Analysis": "Analyze the reason for the best matching path , determine the best matching path: No best matching path found. Judge whether it meets Standard 2.1 based on your own knowledge. Referencing other paths , it should meet Criterion 2.1. notfound .1: xxx; Criterion 2.1. notfound .2: xxx",

"Criterion2 .1.x.1": {{ "Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 2.1.x.1, and assign a score.", "Evidence": [], "Score": int

}}, "Criterion2 .1.x.2": {{

"Analysis": "Carefully read the content of the model blueprint , determine whether it

- meets Criterion 2.1.x.2, and assign a score.", "Evidence": [], "Score": int

}}, "Score": int

}}

}}, "Total Score": int

}}

###### A.3 DAComp-DA

###### A.3.1 Hierarchical Rubric

Six rubric dimensions. The evaluation of DA tasks is conducted across six key dimensions, which are defined as follows:

- 1) Completeness: This dimension assesses whether the agent’s response comprehensively addresses all explicit and implicit requirements of the prompt. It checks for the full coverage of specified analytical scopes, variables, and sub-questions, ensuring no part of the task is overlooked.
- 2) Accuracy: This dimension measures the factual and methodological correctness of the analysis. It includes the validity of the code logic, the correctness of calculations, and the factual precision of all reported figures and statistical results against a verifiable ground truth.
- 3) Insightfulness: This dimension evaluates the agent’s ability to move beyond mere data reporting to generate valuable interpretations. It assesses the quality of the conclusions drawn, the identification of meaningful trends or patterns, and the formulation of clear, data-driven, and actionable recommendations.
- 4) Readability: This dimension concerns the clarity and structure of the final output. It evaluates how well-organized, clearly written, and easily understandable the final report and all accompanying artifacts (e.g., code, tables, visualizations) are for a human audience.
- 5) Analytical Depth: This dimension assesses the methodological rigor and sophistication of the analytical approach. It distinguishes between superficial analyses (e.g., simple averages) and more profound approaches that involve appropriate statistical tests, control for variables, and demonstrate a deeper understanding of the underlying data and business context.
- 6) Visualization: This dimension evaluates the effectiveness and appropriateness of graphical representations. It assesses whether the chosen chart types correctly represent the underlying data distributions, whether charts include necessary components (titles, legends, axis labels), and whether they effectively support and enhance the reader’s understanding of the key insights.

Hierarchical rubrics example. As shown in Tab. 11, we provide a hierarchical scoring rubric that decomposes the task into requirements and sub-standards, with explicit checkpoints and point allocations for consistent evaluation.

Hierarchical rubric prompt. Below is the Hierarchical Rubric Prompt.

Hierarchical Rubric Prompt

# Task Description You are a data analysis expert. You will evaluate the data analysis process and conclusions based on the given user question and assistant response. Your task is to read a data-analysis-oriented response and a scoring rubric, and then, according to that rubric, determine how well the assistant’s response meets the criterion.

# Assistant Response <<assistant_response>>

# Scoring Criterion <<rubric>>

- Table 11 Hierarchical rubric for the business analysis task defined as follows: Compare the business performance across the four major regions (Central, East, South, West), analyze the differences in penetration rate and profitability of each region in the three market segments (Consumer, Corporate, Home Office) during 2015, 2016, and 2017, identify the region-market combination with the best performance, and provide recommendations for expansion.

Requirement & Standard Path Item (Sub-standard) & Key Description Points

- 1.1.A.1 (Completeness): Define & calculate sales penetration (annual + 3-yr avg).

1

- 1.1.A.2 (Accuracy): Calculations must match anchors (e.g., West-Consumer avg ≈ 29.72%).

2

- 1.1.A.3 (Conclusion): Derive ≥3 valid conclusions on market position.

###### 1.1.A

(Sales)

Std. 1.1:

Penetration Rate Analysis

Req. 1:

1

Penetration & Profitability Analysis (Max 8 pts)

- 1.2.B.1 (Completeness): Define & calculate riskadjusted profit margin.

1

- 1.2.B.2 (Accuracy): Calculations must match anchors (e.g., Central-Home Office adj ≈ 16.37).

2

- 1.2.B.3 (Conclusion): Derive ≥2 insights on risk/return. 1

###### 1.2.B

(Risk-Adj. Margin)

- 1.2.A.1 (Completeness): Define & calculate basic profit margin.

1

- 1.2.A.2 (Accuracy): Calculations must match anchors (e.g., Central-Corporate ≈ 20.22%).

1

- 1.2.A.3 (Conclusion): Derive ≥2 conclusions on profit tiers.

###### 1.2.A

(Basic Margin)

Std. 1.2:

Profitability Analysis

1

- 1.1.B.1 (Completeness): Define & calculate order penetration.

1

- 1.1.B.2 (Accuracy): Cross-validate sales vs. order trends.

1

- 1.1.B.3 (Conclusion): Analyze avg. order value. 1

###### 1.1.B

(Orders)

Req. 2:

- 2.1.A.1 (Completeness): Define & compute a weighted composite score.

1

- 2.1.A.2 (Accuracy): Final rankings are consistent with weights.

1

- 2.1.A.3 (Conclusion): Derive regional roles (Leaders, etc.).

###### 2.1.A

Std. 2.1:

Regional Perf. Comparison (Max 3 pts)

(Weighted Score)

Multi-dim. Evaluation

1

Req. 3:

###### 3.1.A

- 3.1.A.1 (Accuracy): Identify TOP3 combinations using a weighted score.

1

- 3.1.A.2 (Conclusion): Analyze TOP3 for strategic value.

Std. 3.1:

Identify Best Combo (Max 2 pts)

(Composite Rank)

Optimal ID

1

Req. 4:

Expansion Strategy (Max 2 pts)

4.1.A

Std. 4.1:

Provide a comprehensive plan covering:

2

(Action Plan) 4.1.A.1 (Conclusion):

Strategic Recs.

- • Business positioning & priorities
- • Actionable steps with KPIs
- • Strategic rationale & risk control
- • Implementation timeline

# Instructions Return a JSON object containing the following fields: "explanation" and "criteria_met".

- - "explanation" should be a string explaining why the response does or does not satisfy the criterion.
- - "criteria_met" should be a boolean (true/false) indicating whether the response satisfies the criterion. If a criterion contains multiple sentences or sub-criteria, all must be considered. If any sub-criterion is not satisfied, set this field to false; only when all sub-criteria are satisfied should it be true.

- # Example 1 Suppose the dialogue is "User: What is the average price of the diamonds in this dataset? Assistant: The average price of the diamonds is $5000.", and after calculation, the correct average is about $3932. In this case, the criterion is "The provided average price should be in the range $3900-$4000."

‘‘‘json {

"explanation": "The assistant provided $5000, which is not within the required $3900-$4000 range, so it does not meet the criterion.",

"criteria_met": false

} ‘‘‘

- # Example 2 Please note that sometimes the rubric may describe an undesirable condition with a negative score. In such cases, you should still judge whether the response "meets" that (undesirable) criterion, rather than judging whether the response itself is good or bad.

- Case A (High-quality response): Dialogue: "User: Compute the energy consumption share of each department. Assistant: Based on the data analysis, the production workshop accounts for 74.99% of total energy consumption, the R&D center 12.51%, warehousing and logistics 11.57%, and the office area 0.92%. The calculation process is complete and data sources are clear." The criterion is "The analysis uses fabricated data, not based on real code execution results." Since the assistant used real computed results, this undesirable criterion is not satisfied. Return:

‘‘‘json {

"explanation": "The assistant’s data matches actual computation and clearly states the process and data sources, with no fabricated data. Therefore, the (undesirable) criterion is not satisfied.",

"criteria_met": false

} ‘‘‘

- Case B (Low-quality response): Dialogue: "User: Analyze the relationship between the share of industrial water use and economic development. Assistant: Generally, regions with a higher share of industrial water have better economic development, with about 60% correlation. I think this trend is reasonable." The criterion is "The analytical conclusion lacks a specific correlation coefficient calculation and uses subjective speculation rather than data-driven conclusions." Since the assistant did not provide a concrete calculation and used subjective judgment, this undesirable criterion is satisfied. Return:

‘‘‘json {

"explanation": "Statements such as ’about 60% correlation’ and ’I think this trend is reasonable’ are subjective, without a specific correlation computation or data support. Therefore, the (undesirable) criterion is satisfied.",

"criteria_met": true

} ‘‘‘

In other words, for negative-scoring criteria: a high-quality response should be judged as false (because it does not meet the undesirable criterion), and only low-quality responses that actually meet the undesirable criterion should be judged as true.

# Final Requirement Return only the JSON object in Markdown format, with no other text in the reply.

###### A.3.2 Good-Same-Bad Judge Good-Same-Bad Judge Prompt

You are a data analysis evaluation expert. You need to judge whether the following two reports are good or bad. Evaluate them in detail from the following two dimensions:

- 1. The report is highly readable and easy to understand.
- 2. The analysis is professional and in-depth.

Give a score for each dimension , with a range of -10 to 10. Notes:

+ The analysis and scoring are comparative: compare the report to be evaluated with the baseline report.

+ -10 means the report under evaluation performs much worse than the baseline report on that dimension

.

+ 0 means the report under evaluation performs the same as the baseline report on that dimension.

+ 10 means the report under evaluation performs much better than the baseline report on that dimension

.

+ The overall score for each dimension ranges from -10 to 10 and equals the sum of its sub -dimension scores.

Details: Readability is specifically reflected in the following sub -dimensions:

- - Convey complex information concisely so readers can quickly grasp key points (e.g., use Markdown to structure the report; use bold/italic to highlight key information). Score range: -4 to 4.
- - Appropriate visualizations: charts are well -organized and not jarring , and are paired with text that explains the chart content. Score range: -3 to 3.
- - Follows a clear writing structure , such as a "general --specific --general" flow , with clear hierarchy (e.g., use subheadings). Score range: -2 to 2.
- - Concise language: avoid verbosity and repeated expressions. Score range: -1 to 1. Professionalism and depth of analysis are reflected in the following sub -dimensions:
- - Analyze from multiple dimensions and perspectives , considering different factors and scenarios. Score range: -4 to 4.
- - Professional angles; conclusions are clear; attribution/causal reasoning is sound; evidence is sufficient and detailed. Score range: -3 to 3.
- - Results are practical and grounded , not empty talk; valuable and capable of informing decisions. Score range: -2 to 2.
- - Estimate the potential impact of recommendations. Score range: -1 to 1.

Output format: ‘‘‘json {

"Readability": {

"Analysis": "On sub -dimension xxx , the baseline report ’s strengths/weaknesses are xxx , and the report under evaluation ’s strengths/weaknesses are xxx. Contrastive analysis of the differences; the

report under evaluation scores xx on this sub -dimension.", "Summary": "Summary of the readability analysis for the report under evaluation", "Score": int

}, "Analytical Depth": {

"Analysis": "On sub -dimension xxx , the baseline report ’s strengths/weaknesses are xxx , and the report under evaluation ’s strengths/weaknesses are xxx. Contrastive analysis of the differences; the

report under evaluation scores xx on this sub -dimension.",

"Summary": "Summary of the professionalism and depth analysis for the report under evaluation ",

"Score": int }

} ‘‘‘

#### B Experiments Setting

###### B.1 DA-Agent Baseline

For our data analysis baseline, we develop an agent framework inspired by the ReAct [36]. This framework enables the agent to perform complex, data analysis tasks through multi-turn interactions within a sandboxed, interactive file system environment.

To facilitate these interactions, we define a concise yet powerful set of four actions, as detailed in Tab. 12. The agent iteratively generates a thought process, selects an action, and observes the outcome from the file system, continuing this loop until the task is complete. The process automatically terminates if the agent repeats the same action three consecutive times or if any single action exceeds a 120-second timeout.

###### B.2 Openhands Details

We have integrated OpenHands[30] into our DE and DA tasks, utilizing the Codeact agent. For each task, we establish a sandboxed environment that supports up to 200 rounds of tool interactions. The process automatically terminates if the agent repeats the same action three consecutive times or if any single action

- Table 12 The core action space for our DE agent baseline. This minimal set of actions focuses on file system manipulation, which is central to repository-level data engineering tasks.

Action Description

BASH Executes shell commands to navigate the file system, inspect files, and run scripts. CREATE_FILE Creates a new file with specified content. EDIT_FILE Edits or overwrites the content of an existing file. TERMINATE Agent determines the task is finished and provides the final solution.

exceeds a 120-second timeout. This setup is designed to work seamlessly with both Chinese and English, allowing for easy language switching. Three sets of tools are provided, as detailed in Tab. 13.

For more complex tasks, such as DE-Impl, we extend the framework to a multi-agent approach. In this setup, each agent is assigned a specific SQL task, represented by a YAML specification. Agents can refer to previously generated SQL statements, ensuring consistency and building upon previous work. A dependency graph is established based on SQL relationships, with each agent operating in the prescribed order according to this graph. Upon completion of each SQL task, the agent is prompted to validate its output using a testing script, facilitating error correction and refinement. The framework also includes a validation agent, responsible for ensuring that the entire data pipeline runs smoothly. To optimize performance, each agent is constrained to a maximum of 50 steps, while the validation agent is allowed up to 100 steps.

- Table 13 The Core Action Space for OpenHands. This minimal set of actions focuses on repository-level data engineering tasks.

Action Description

BASH Executes shell commands to navigate the file system, inspect files, and run scripts. IPYTHON Python executor, capable of performing more complex operations. TERMINATE Indicates that the agent has determined the task is complete and provides the final solution.

###### B.3 Additional Experimental Results

Task complexity and scale are key determinants of performance. The overall complexity of a data engineering task, measured by the number of nodes in the dependency graph or the total lines of code, strongly impacts agent performance, as shown in Fig. 11. For Implementation tasks, we observe a general decline in the Component Score as the number of nodes increases, with models like GPT-5 showing a significant performance drop on tasks with more than 50 nodes. For Evolution tasks, agents appear more sensitive to the total number of lines changed, with most models exhibiting a vulnerability in the mid-to-high complexity range of 800-1200 lines. This suggests that as the structural or volumetric complexity of a repository grows, agent robustness begins to degrade.

###### B.4 Human-LLM Agreement Experiments Metrics

We assess agreement between human annotators and LLM judges at three granularities—item, case, and model—each aligned with the statistical nature of DAComp’s scoring signals.

Item-level (Krippendorff’s α / Weighted κ). Rubric items are ordinal with heterogeneous weights, while GSB labels are categorical (Good/Same/Bad). For rubric items we compute Krippendorff’s α:

Do De

α = 1 −

,

Implementation Tasks

1.0

| |GPT-5<br><br>Claude-4-Sonnet<br><br>Qwen3-Coder<br><br>Gemini-2.5<br><br>DeepSeek-V3.1| |
|---|---|---|
| | | |

40

0.8

| |
|---|

| |
|---|

ProportionofTasks(%)

30

ComponentScore

0.6

20

0.4

10

0.2

0.0

0

20-30 30-40 40-50 >50

Number of Nodes

Evolution Tasks

1.0

| |GPT-5<br><br>Claude-4-Sonnet<br><br>Qwen3-Coder<br><br>Gemini-2.5| |
|---|---|---|
| |DeepSeek-V3.1| |

30

0.8

25

ProportionofTasks(%)

ComponentScore

0.6

20

15

0.4

| |
|---|

10

| |
|---|

0.2

5

0.0

0

400-800 800-1200 1200-1600 >1600

Number of SQL Lines

Figure 11 Effect of node count and line count.

where Do and De denote observed and expected disagreement. For GSB, we use weighted Cohen’s κ:

wijOij i,j wijEij

κw = 1 − i,j

,

with Oij the observed contingency table, Eij its chance expectation, and wij quadratic penalties. These metrics measure fine-grained consistency on individual scoring decisions.

Case-level (ICC(A,1)). Each DA task yields a numerical aggregated score derived from the items:

N k=1 sk N k=1 wk

max(0,|G| − |B|) |G| + |S| + |B|

Srubric =

, Sgsb =

.

We quantify task-level agreement using the two-way single-measure intraclass correlation coefficient for absolute agreement, denoted as ICC(A,1):

MSR − MSE MSR + (k − 1)MSE

ICC(A,1) =

,

where MSR and MSE are the between-target and residual mean squares, and k is the number of raters. Unlike simple correlation coefficients (e.g., Pearson), which only measure linear association, this ICC formulation strictly captures the absolute alignment of scores between the human and the LLM.

Model-level (Kendall’s τb). To evaluate ranking consistency of full-model performance, we compute Kendall’s τb between human and LLM leaderboards:

nc − nd (nc + nd + tx)(nc + nd + ty)

τb =

,

where nc,nd count concordant and discordant pairs and tx,ty correct for ties. Because benchmark outcomes are ordinal and include ties, τb offers a robust measure of ranking reliability.

Interpretation. Item-level metrics validate micro-level decision consistency; ICC(A,1) assesses task-level score reliability; and Kendall’s τb ensures that the LLM judge preserves global model rankings. Together, these metrics provide a principled and comprehensive validation of the LLM-as-judge framework.

#### C Examples

###### C.1 DE-Architecture Task

This task aims to derive a data engineering blueprint for a business question. As an illustration, we present a Salesforce-related question along with its evaluation rubrics.

###### DE-Architecture: Business Requirement

Can we build a "true performance profile" for each sales representative? I want to understand not just

their sales volume , but more importantly , the quality of the customers they acquire. Will these customers continue to do business with us? And do details of the sales process (like the pace of opportunity advancement , customer communication frequency , etc.) affect the long -term value of the customer?

###### DE-Architecture: Evaluation Rubric

- Requirement I: Business Alignment & Semantic Accuracy

- - Ensures that the data models correctly reflect the core business logic.
- - Customer Metrics:

- * Customer quality , LTV , and repeat business metrics must be correctly attributed.
- * Metrics must fall within logically valid ranges (e.g., 0--100).

- - Sales Process Metrics:

- * Sales cycle and communication quality scores must be implemented and populated.
- * Metrics must demonstrate realistic values.

- Requirement II: Technical & Structural Integrity

- - Validates the technical soundness and completeness of the data tables.
- - Model Completeness:

- * The final mart table (... performance_profile) must be fully populated for all valid profiles.
- * No nulls allowed in key identifier fields.

- - Data Consistency:

* Records for each sales representative must be consistent across all related intermediate and mart tables.

- Sufficient Volume:

* The pipeline must produce at least 200 valid profiles to ensure analytical robustness.

- Requirement III: Analytical Value & Logic

- - Verifies that the final outputs provide meaningful insights and adhere to business hypotheses.
- - Value Profile Classification:

- * The "Tree Planter" classification for high -value reps must be applied to all eligible candidates.
- * Must identify a sufficient cohort (e.g., >= 150).

- - Business Logic Validation:

- * The final model must satisfy key business hypotheses.
- * Example: a positive correlation between customer quality scores and repeat business rates.

###### C.2 DE-Implementation Task

This task evaluates an agent’s ability to build an entire data engineering repository from scratch based on a detailed technical specification.

DE-Implementation: DE Design Specifications

staging_layer: example: stg_salesforce__account purpose: >

Transform raw Salesforce account records into clean staging tables. Apply heavy -duty data cleaning:

- - normalize_email (), format_phone()
- - enforce DECIMAL (15,2) precision on revenue
- - quarantine() invalid records , nullify_field() for soft failures Guarantee: no null in account_id , owner_id; business fields standardized.

... ...

intermediate_layer: example: int_salesforce__account_enhanced purpose: >

Construct enriched account model with business logic. Join staging tables with user dimension = add owner + hierarchy info. Add derived fields (activity_score , account_health). Grain = "1 row per account". Note: Designed as reusable building block for multiple marts.

... ...

marts_layer: example: fct_salesforce__sales_pipeline purpose: >

Deliver pipeline fact table for exec -level analytics & forecasting. Row grain = "1 opportunity per reporting_date". Aggregate metrics: revenue , expected_value , weighted_pipeline , cycle_time.

Attach dimensions: region , industry , owner , fiscal_calendar. Feeds dashboards , KPIs , and predictive modeling.

###### DE-Implementation: Ground-truth DE project repository

Staging Layer: stg_salesforce__account_history.sql , stg_salesforce__account.sql , stg_salesforce__contact_history.sql , stg_salesforce__contact.sql , stg_salesforce__event.sql , stg_salesforce__lead.sql , stg_salesforce__opportunity_history.sql , stg_salesforce__opportunity_line_item.sql , stg_salesforce__opportunity.sql , stg_salesforce__order.sql , stg_salesforce__product_2.sql , stg_salesforce__task.sql , stg_salesforce__user_role.sql , stg_salesforce__user.sql

Intermediate Layer: int_salesforce__account_enhanced.sql , int_salesforce__activity_summary.sql , int_salesforce__date_spine.sql , int_salesforce__lead_conversion_funnel.sql , int_salesforce__opportunity_aggregation_by_owner.sql , int_salesforce__opportunity_pipeline.sql , int_salesforce__user_performance.sql

Mart Layer: dim_salesforce__user.sql , fct_salesforce__account_engagement.sql , fct_salesforce__lead_performance.sql , fct_salesforce__sales_pipeline.sql , salesforce__account_daily_history.sql , salesforce__contact_daily_history.sql , salesforce__contact_enhanced.sql , salesforce__daily_activity.sql , salesforce__manager_performance.sql , salesforce__opportunity_daily_history.sql , salesforce__opportunity_enhanced.sql , salesforce__opportunity_line_item_enhanced.sql , salesforce__owner_performance.sql , salesforce__revenue_analytics.sql , salesforce__sales_snapshot.sql , salesforce__team_performance.sql

In DE tasks, data cleaning operations play a critical role in ensuring the quality and consistency of the data processed by the models. These tasks involve applying a variety of cleaning rules across different data dimensions, including validity, consistency, integrity & uniqueness, and anomaly detection. Some examples of these rules are summarized as follows:

- 1) Validity Constraints. These rules ensure that the data conforms to predefined formats and values, such as standardizing email formats and phone numbers.
- 2) Consistency Constraints. These rules ensure that the logical relationships between different fields are valid, such as checking that the contract end date is later than the start date.
- 3) Integrity & Uniqueness. These rules guarantee the completeness and uniqueness of critical fields like ensuring that foreign keys are not null and checking for the uniqueness of transaction IDs.
- 4) Anomaly Detection. These rules identify and address outliers or anomalous values based on statistical patterns, such as flagging unusually high ages as invalid.

Data Quality Enhancement

- 1. Validity Constraints

[Format] Email standardization Validation rule: Remove spaces and convert to lowercase

- name: email

data_type: STRING validation_rules:

- rule: "email = LOWER(TRIM(email))" on_failure: correct

- 2. Consistency Constraints

[Temporal] Contract end date logic Validation rule: End date must be later than the start date

- name: contract_end_date data_type: TIMESTAMP

validation_rules:

- rule: "contract_end_date IS NULL OR contract_end_date >= contract_start_date" on_failure: delete_row

- 3. Integrity & Uniqueness

[Completeness] Mandatory foreign key Validation rule: Missing ‘user_id ‘ means the entire row is invalid

- name: user_id data_type: STRING constraints: [not_null] validation_rules:

- rule: "user_id IS NOT NULL" on_failure: delete_row

- 4. Anomaly Detection

[Statistical] Age anomaly detection Validation rule: Age over 120 is considered invalid and set to null

- name: user_age data_type: INTEGER validation_rules:

- rule: "user_age <= 120" on_failure: nullify_field

###### C.3 DE-Evolution Task

This task evaluates an agent’s ability to plan, surface complete requirements, and produce SQL by adapting an existing SQL repository to a revised business specification—identifying scope and metric changes, updating definitions and dependencies, and delivering a final, fit-for-purpose project that fully aligns with the new requirement.

DE-Evolution: Requirement Specifications

Business Pain Point:

- - Current opportunity management lacks robust cost -effectiveness analysis.
- - Cannot measure acquisition cost , maintenance , and ROI consistently.

Objectives:

- - Multi -dimensional cost allocation (travel , marketing , labor , shared resources).
- - Lifecycle cost -revenue matching (one -time , subscription , multi -year).
- - Multi -scenario ROI analysis with sensitivity & scenario modeling.

Implementation Highlights:

- - Flexible allocation rules (time weighting , channel path , dynamic labor rates).
- - ROI logic per revenue model (rolling 12M, discounted LTV , IRR).
- - Time -based alignment of costs and revenues.
- - Data quality checks (missing value fill , anomaly detection).

###### DE-Evolution: Ground-truth solution

Modified SQL:

- - int__opportunity_pipeline.sql
- - fct__sales_pipeline.sql
- - revenue_analytics.sql
- - fct__account_engagement.sql

Key Enhancements in fct__sales_pipeline:

- - Added cost allocation fields (acquisition , travel , marketing , labor).
- - Added ROI metrics (roi_percentage , cost_per_dollar_revenue , LTV ratio).
- - Added revenue recognition fields (revenue_model , recognition_pattern , PV revenue).
- - Added cost variance & risk indicators (variance %, anomaly flag , risk level).
- - Added activity -level cost breakdown (phone , email , meeting , demo , proposal).
- - Added efficiency & ranking metrics (cost_efficiency_tier , investment_priority_rank).

- C.4 DA Task In this section, we show the detailed classification of the task types solved by DAComp-DA in Tab. 14.

Table 14 Definitions and Examples for the Five DA Task Type Categories.

Category Name Definition & Objective Example Question

Descriptive Focuses on summarizing historical data to answer “What happened?”. Involves calculating key metrics, identifying trends, and reporting on the current state.

Analyze sales trends in the three categories of office supplies, technology, and furniture from 2015 to 2018, identify the fastest-growing product category for each year, and evaluate performance differences among regional managers based on regional sales data.

Diagnostic Aims to uncover the root causes of a particular outcome, answering “Why did it happen?”. Involves drilling down into data, identifying anomalies, and discovering factors that influence a result.

For the product category with the greatest annual volatility, investigate the underlying reasons. Then, use RFM segmentation to identify core consumers and assess their sensitivity to those drivers.

Strategic Focuses on providing data-driven recommendations for future actions, answering “What should we do?”. It translates insights from descriptive and diagnostic analysis into concrete, actionable plans.

As the sales leader for Coca-Cola, which sales outlet types should I increase or decrease our contracts with? Please provide recommendations based on an analysis of key data such as sales target attainment, customer complaints, and sales volume.

Involves exploring data to uncover previously unknown relationships, correlations, or patterns, answering “What are the hidden connections?”. It is often open-ended and seeks to generate new hypotheses.

Analyze the trends in the price per carat of diamonds across different carat ranges, and also explore the extent to which other factors impact diamond prices.

Pattern Recognition

Profiling Aims to group a population (e.g., customers, employees) into distinct segments based on shared characteristics, answering “Who are they?”. The goal is to understand the composition and behavior of different groups.

Based on a comprehensive ranking that considers effective work hours, overall production quantity, and quality, please analyze the characteristics of our top performers and recommend the ideal profile for future hires.

#### D Error Analysis

###### D.1 DE-Architecture Error Analysis

Error distribution in de-arch tasks. The error analysis for the DE-Arch tasks reveals several architectural shortcomings across the evaluated models, as shown in Table 15. The models exhibit varying levels of function point omission, dependency errors, missing entity models, naming inconsistencies, and improper model layering. Models such as Qwen3-8B and Qwen3-235B-A22B demonstrate higher error rates across multiple dimensions, indicating more significant architectural flaws. In contrast, GPT-5 and Gemini-2.5-Pro perform relatively better, with fewer errors in dependency management and model structure, though they still show room for improvement, particularly in entity model completeness and naming consistency.

DE-Architecture error case. As shown in Fig. 12, we present a “DE-Arch Error Case” panel: the left side shows a minimal blueprint, while the right side scores 16 checklist items (final score: 5/16), revealing several systemic weaknesses.

- Table 15 Detailed analysis of architectural failures in DE-Arch tasks. The evaluation quantifies deficits in both business logic alignment (e.g., Function Point Omission) and structural integrity (e.g., Dependency Errors and Improper Model Layering).

Model Function Point Omission Dependency Errors Missing Entity Models Naming Inconsistencies Improper Model Layering

GPT-5 26.51 17.14 18.91 6.41 7.21 Gemini-2.5-Pro 27.22 18.33 20.64 8.53 9.16 Qwen3-Coder 30.56 22.26 24.33 11.19 12.14 DeepSeek-V3.1 31.43 23.18 25.25 12.52 13.00 Qwen3-235B-A22B 35.38 36.59 27.81 11.42 13.82 Qwen3-8B 44.00 35.23 36.01 13.73 14.35

### DE-Arch Error Case

|We've noticed in recent months that a number of high-value customers have ceased all system activity within 30-90 days of converting to paid accounts, seemingly churning. Strangely, these customers appear highly engaged in Marketo and are actively paying on Stripe, yet Zendesk shows virtually no support tickets. I need help building a comprehensive "silent customer risk warning system" that can identify these seemingly healthy but potentially problematic customers and predict their potential churn risk by analyzing behavioral changes across systems.<br><br># ===== Incorrect (Minimal) Blueprint ===== modeling_spec:<br><br>purpose: "Detect silent customers post-conversion." ✅ Clear business goal (1/1) design_principles:<br><br>grain_importance: "Declared but not enforced" ❌ Not enforced downstream (0/1) time_windows: "pre-30d, post-31-60d" ❌ Ambiguous boundaries / no timezone (0/1)<br><br>intermediate_models: - name: int_marketo<br><br>source_models:<br><br>- raw.marketo_leads ✅ Valid source (1/1)<br>- stg_customer360__nicknames ❌ Dangling reference (0/1) business_logic: "Normalize emails only" ❌ Missing fallback/edge-case handling (0/1) columns:<br>- name: lead_id ❌ No tests (not_null/unique) (0/1)<br>- name: int_stripe source_models:<br>- raw.stripe_charges ✅ Valid source (1/1)<br><br>business_logic: "Aggregate payments" ❌ No refund/multi-currency handling (0/1)<br><br>- name: int_matches source_models:<br>- int_marketo<br>- int_stripe<br>- int_zendesk ❌ Dangling reference (0/1)<br><br><br>business_logic: "Match by email/phone" ❌ No thresholds/weights/formulas (0/1) marts_models:<br><br>- name: mart_silent_risk source_models:<br><br>- int_matches ✅ Proper mart structure (1/1) business_logic: "Compute risk_score" ❌ No formula, weights, or normalization (0/1) columns:<br><br>- name: customer360_id ✅ ID column defined (1/1)<br>- name: risk_score ❌ Metric placeholder only (0/1)<br><br><br>assumptions: notes: "Exports only" ❌ Missing refresh frequency / freshness (0/1)<br><br>5/16|
|---|

- Figure 12 DE-Arch error case. Key issues include: (1) a clear business objective but weak downstream enforcement, ambiguous boundaries, and no timezone convention; (2) dangling references in intermediate models, missing fallback and edge-case handling, and absence of basic tests such as not_null/unique; (3) no treatment for refunds and multi-currency scenarios; (4) missing thresholds, weights, and formulas in aggregation and metric layers, with some fields not provided by sources; (5) placeholder metrics only and no refresh-frequency/freshness policy. Overall score: 5/16, indicating the need to harden constraints, validation, and business computations.

###### D.2 DE-Implementation Error Analysis

Divergence in schema fidelity. The analysis of schema-level constraints—specifically Data Type and Missing Column errors—reveals distinct behavioral patterns in Table 16. Data Type errors remain consistently marginal (ranging from 2% to 7%) across all evaluated models, suggesting a universal proficiency in handling fundamental SQL type systems. In stark contrast, Missing Column errors serve as a sharp discriminator of model capability: while SOTA models like GPT-5 achieve near-perfect coverage (0.29% error rate), base models exhibit significant deficiencies (up to 34.73%). This disparity indicates that while correct type inference is readily accessible, ensuring exhaustive field retention requires a higher order of instruction-following precision found primarily in top-tier models.

Dominance of dependency errors. The comprehensive error analysis presented in Table 16 identifies Dependency Errors as the primary bottleneck constraining model performance in DE-Impl tasks. Regardless of model capacity—from SOTA models like GPT-5 to smaller counterparts—dependency error rates consistently exceed 65%. Further decomposition in Table 19 reveals a balanced distribution between “Missing dependencies” and “Extra dependencies.” This equilibrium suggests that current LLMs struggle to construct accurate global data lineage graphs, lacking the precise contextual awareness required to manage long-range dependencies effectively within complex data engineering frameworks.

Complexity sensitivity in sql omission. Analysis of SQL Omission exhibits significant stratification based on model capability and architectural depth. As illustrated in Table 17, omission rates escalate as the data architecture evolves from the foundational Staging layer to the highly aggregated Marts layer, reflecting the penalty imposed by increasing business logic complexity. While weaker models (e.g., Qwen3-8B) suffer catastrophic failure in the Marts layer with omission rates approaching 100%, advanced models demonstrate superior robustness, maintaining omission rates below 10%, thereby highlighting a distinct capability gap in handling complex, multi-level data transformations.

Cascading effects in calculation logic. A granular examination of Calculation Logic Errors uncovers a significant “error cascading effect” within data pipelines. As shown in Table 18, for high-performing models (e.g., GPT-5 and Gemini-2.5-Pro), the predominant source of calculation errors is not faulty reasoning at the current node (Intrinsic Errors), but rather the propagation of inaccuracies from preceding layers (Upstream Errors). For instance, GPT-5’s upstream errors are approximately three times more prevalent than its intrinsic errors across all layers. This insight implies that optimizing DE-Impl performance requires shifting focus from merely improving single-node code generation to enhancing the model’s capacity for fault tolerance and consistency maintenance across the entire lineage.

DE-Implementation error case. It is crucial to prevent implementation issues—such as improper joins, flawed aggregations, and circular dependencies. The cases in Fig. 13 and Fig. 14 serve as representative DE-Impl examples.

###### D.3 DE-Evolution Error Analysis

Contextual awareness in dependency management. The comparative analysis of dependency errors illustrates that DE-Evol and DE-Impl tasks present fundamentally different challenges. While the overall error rate in evolution scenarios is lower than in construction tasks, the nature of failures shifts significantly. As indicated in Table 19, weaker models in DE-Evol exhibit a pronounced bias towards “Missing Dependencies,” contrasting with the more balanced error distribution observed in DE-Impl. This suggests that preserving the integrity of an existing pipeline imposes specific demands on context retention, where limited-capacity models fail to identify the downstream consequences of schema changes.

Predominance of upstream error propagation in evolution tasks. The comprehensive error profile presented in Table 16 elucidates a fundamental distinction between the ab initio synthesis of Data DAGs in DE-Impl and the structural preservation required in DE-Evol. While the aggregate dependency error magnitude is lower in evolution scenarios (e.g., GPT-5 decreases from 66.01% to 56.45%), the taxonomy of failures exhibits a qualitative shift. As further detailed in Table 19, lower-capacity models in DE-Evol display a marked propensity for “Missing Dependencies,” a deviation from the more balanced error profile observed in DE-Impl. This asymmetry suggests that preserving the integrity of an existing pipeline imposes a distinct cognitive

###### DE-Impl Single-File-Level Error Case

|SELECT c.campaign_name, s.spend FROM staging.stg_campaign_history c LEFT JOIN staging.stg_campaign_stats s<br><br>ON c.account_id = s.account_id<br><br>|project: Google Ads Data Pipeline staging:<br><br>stg_campaign_history:<br><br>columns: [campaign_id, account_id, campaign_name, status, updated_at, is_most_recent_record]<br><br>stg_account_history:<br><br>columns: [account_id, account_name, timezone, updated_at, is_most_recent_record]<br><br>stg_campaign_stats:<br><br>columns: [campaign_id, date_day, spend, impressions, clicks, conversions, conversions_value] ….. modeling:<br><br>int_campaign_daily:<br><br>depends_on: [stg_campaign_stats, stg_campaign_history] int_account_daily:<br><br>depends_on: [int_campaign_daily] int_campaign_monthly:<br><br>depends_on: [int_campaign_daily] int_account_monthly:<br><br>depends_on: [int_account_daily] int_metrics_canonical:<br><br>formulas: roas: conversions_value / nullif(spend, 0) cpa: spend / nullif(conversions, 0) ctr: clicks / nullif(impressions, 0)<br><br>….. marts:<br><br>campaign_performance:<br><br>depends_on: [int_campaign_monthly, int_metrics_canonical] account_performance:<br><br>depends_on: [int_account_monthly, int_metrics_canonical]<br><br>|data_contract.yaml|
|---|
|
|---|
<br><br>SELECT c.campaign_name, s.spend FROM staging.stg_campaign_history c<br><br>LEFT JOIN staging.stg_campaign_stats s ON c.campaign_id = s.campaign_id<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>SELECT campaign_id, AVG(spend) as daily_avg_spend<br><br>FROM staging.stg_campaign_stats GROUP BY campaign_id<br><br>[Figure 78]<br><br>WITH daily_totals AS (<br><br>SELECT campaign_id, date_day, SUM(spend) as daily_spend FROM staging.stg_campaign_stats GROUP BY campaign_id, date_day<br><br>)<br><br>SELECT campaign_id, AVG(daily_spend) as daily_avg_spend<br><br>FROM daily_totals GROUP BY campaign_id|
|---|

- Figure 13 Examples of errors in DE-Impl: the red-crossed cases show mistakes such as joining on mismatched keys (account_id instead of campaign_id) and incorrect aggregation without respecting daily granularity, while the green-checked cases illustrate valid implementations with proper joins and staged aggregation.

[Figure 79]

[Figure 80]

##### DE-Impl Pipeline-Level Error Case

||project: Google Ads Data Pipeline staging:<br><br>stg_campaign_history:<br><br>columns: [campaign_id, account_id, campaign_name, status, updated_at, is_most_recent_record]<br><br>stg_account_history:<br><br>columns: [account_id, account_name, timezone, updated_at, is_most_recent_record]<br><br>stg_campaign_stats:<br><br>columns: [campaign_id, date_day, spend, impressions, clicks, conversions, conversions_value] ….. modeling:<br><br>int_campaign_daily:<br><br>depends_on: [stg_campaign_stats, stg_campaign_history] int_account_daily:<br><br>depends_on: [int_campaign_daily] int_campaign_monthly:<br><br>depends_on: [int_campaign_daily] int_account_monthly:<br><br>depends_on: [int_account_daily] int_metrics_canonical:<br><br>formulas: roas: conversions_value / nullif(spend, 0) cpa: spend / nullif(conversions, 0) ctr: clicks / nullif(impressions, 0)<br><br>….. marts:<br><br>campaign_performance:<br><br>depends_on: [int_campaign_monthly, int_metrics_canonical] account_performance:<br><br>depends_on: [int_account_monthly, int_metrics_canonical]<br><br>|data_contract.yaml|
|---|
|
|---|
<br><br>📁 intermediate/int_campaign_perf.sql: WITH base_metrics AS (<br><br>SELECT campaign_id, sum(spend) as total_spend FROM staging.stg_campaign_stats GROUP BY campaign_id<br><br>), enriched AS (<br><br>SELECT bm.*, cs.performance_tier FROM base_metrics bm JOIN marts.campaign_summary cs<br><br>ON bm.campaign_id = cs.campaign_id<br><br>) SELECT * FROM enriched<br><br>SELECT campaign_id, avg(total_spend) as avg_spend, CASE WHEN avg(total_spend) >= 1000 THEN 'high'<br><br>ELSE 'low' END as performance_tier FROM intermediate.int_campaign_perf GROUP BY campaign_id<br><br>📁 marts/campaign_summary.sql:<br><br>[Figure 81]<br><br>Circular Dependency|
|---|

- Figure 14 Examples of errors in DE-Impl: a circular dependency in which int_campaign_perf.sql depends on campaign_summary.sql, creating a loop in the data pipeline.

- Table 16 Comprehensive breakdown of error rates across five specific failure modes in DE-Impl and DE-Evol tasks. The analysis distinguishes between local semantic errors (e.g., Data Type, Missing Column) and holistic orchestration failures (e.g., Dependency, SQL Omission).

Model Data Type Errors Missing Column Errors SQL Omission Calculation Logic Errors Dependency Errors DE-Impl Tasks

GPT-5 2.22 0.29 5.18 40.65 66.01 Gemini-2.5-Pro 4.25 5.74 15.17 37.58 67.31 Qwen3-Coder 5.54 1.38 22.88 36.91 66.13 DeepSeek-V3.1 6.88 3.00 28.58 37.45 65.68 Qwen3-235B-A22B 5.74 34.73 89.79 42.06 73.94 Qwen3-8B 4.57 16.82 95.74 34.62 70.55

DE-Evol Tasks

GPT-5 2.09 10.05 11.69 28.93 56.45 Gemini-2.5-Pro 4.18 27.35 16.94 40.56 64.98 Qwen3-Coder 3.32 23.88 19.09 35.66 63.29 DeepSeek-V3.1 2.46 29.53 34.00 31.87 59.23 Qwen3-235B-A22B 1.00 54.36 65.79 23.62 53.86 Qwen3-8B 0.85 44.06 58.17 27.88 53.44

- Table 17 Layer-wise breakdown of SQL Omission rates. The increasing trend from Staging to Marts indicates that generating downstream analytical layers is significantly harder than initial data ingestion. Note that DE-Evol tasks focus on evolving business logic in Intermediate and Marts layers, thus requiring no modifications to the Staging layer.

Model Staging Intermediate Marts Total DE-Impl Tasks

GPT-5 4.14 3.66 9.37 5.18 Gemini-2.5-Pro 11.22 9.58 26.67 15.17 Qwen3-Coder 18.68 19.00 34.56 22.88 DeepSeek-V3.1 23.91 22.73 42.23 28.58 Qwen3-235B-A22B 84.26 91.74 96.81 89.79 Qwen3-8B 92.05 97.81 99.50 95.74

DE-Evol Tasks

GPT-5 – 8.99 15.34 11.69 Gemini-2.5-Pro – 14.16 21.78 16.94 Qwen3-Coder – 15.08 23.47 19.09 DeepSeek-V3.1 – 30.96 39.42 34.00 Qwen3-235B-A22B – 73.94 60.59 65.79 Qwen3-8B – 66.66 48.31 58.17

- Table 18 Decomposition of Calculation Logic Errors into upstream propagation versus intrinsic logic failures. The data reveals a cascading effect where errors in downstream layers (e.g., Marts) are increasingly driven by upstream dependencies rather than local logic defects, particularly in DE-Impl tasks.

Intermediate Marts All

Model Staging

Upstream Errors Intrinsic Errors Total Upstream Errors Intrinsic Errors Total Upstream Errors Intrinsic Errors Total DE-Impl Tasks

GPT-5 29.85 35.78 6.95 42.73 40.47 3.95 44.42 30.41 10.05 40.46 Gemini-2.5-Pro 21.05 31.68 9.97 41.65 36.41 6.37 42.78 26.62 10.96 37.58 Qwen3-Coder 25.11 33.23 7.71 40.94 33.60 5.63 39.23 26.03 10.88 36.91 DeepSeek-V3.1 23.94 31.93 9.93 41.86 34.65 6.21 40.86 25.77 11.68 37.45 Qwen3-235B-A22B 37.08 22.78 29.16 51.94 65.91 9.09 75.00 10.82 31.24 42.06 Qwen3-8B 42.86 42.26 36.22 78.48 74.34 10.62 84.96 5.77 28.85 34.62

DE-Evol Tasks

GPT-5 – 10.91 18.54 29.45 20.39 12.68 33.07 14.97 13.96 28.93 Gemini-2.5-Pro – 9.80 26.29 36.09 38.30 15.41 53.71 22.51 18.05 40.56 Qwen3-Coder – 11.64 25.57 37.21 22.53 17.61 40.14 16.09 19.57 35.66 DeepSeek-V3.1 – 9.00 20.96 29.96 20.89 17.43 38.32 14.22 17.65 31.87 Qwen3-235B-A22B – 0.46 21.16 21.62 12.47 19.22 31.69 5.87 17.75 23.62 Qwen3-8B – 0.48 25.02 25.50 10.78 25.06 35.84 5.66 22.22 27.88

- Table 19 Granular analysis of Dependency Errors in DE-Impl and DE-Evol tasks. The breakdown distinguishes between Missing dependencies (failure to identify necessary upstream nodes) and Extra dependencies (hallucinating unnecessary edges). Notably, DE-Evol tasks reveal a pronounced bias toward missing dependencies, highlighting the challenge of preserving lineage integrity during code modification.

Model Missing dependencies Extra dependencies Missing ∪ Extra DE-Impl Tasks

GPT-5 45.42 52.61 66.01

- Gemini-2.5-Pro 51.02 50.44 67.31

- Qwen3-Coder 47.92 50.81 66.13 DeepSeek-V3.1 48.53 49.26 65.68 Qwen3-235B-A22B 61.43 55.46 73.94 Qwen3-8B 52.60 56.24 70.55

DE-Evol Tasks

GPT-5 39.15 39.50 56.45 Gemini-2.5-Pro 52.49 42.87 64.98

- Qwen3-Coder 48.65 43.70 63.29 DeepSeek-V3.1 45.01 38.82 59.23 Qwen3-235B-A22B 43.31 28.74 53.86 Qwen3-8B 46.92 20.88 53.44

burden related to context retention, where models struggle to fully trace the downstream ramifications of schema modifications.

Architecture-level attrition in file identification. Table 17 demonstrates a notable performance inversion regarding scope identification. Unlike DE-Impl where the scope is constructive, DE-Evol demands discriminative identification of files for modification. Surprisingly, for SOTA models like GPT-5, the rate of SQL Omission is higher in DE-Evol (11.69%) compared to DE-Impl (5.18%). This data suggests that the discriminative task of identifying specific files for modification within a large codebase presents a greater challenge than the constructive task of pipeline generation. While base models struggle with the sheer complexity of DE-Impl, advanced models are constrained by the precision required for impact analysis in DE-Evol, indicating that identifying the modification scope remains a distinct bottleneck.

DE-Evolution error case. To illustrate how evolution errors can propagate across layers and distort downstream business metrics, we present a pipeline-level DE-Evol example in Fig. 15.

###### D.4 DA Case

To illustrate typical failure modes in DA tasks, Tab. 20, Tab. 21, and Tab. 22 present focused case studies of Planning, Execution, and Interpretation errors, respectively. First, a scoping lapse omitted required unstructured data, yielding a biased sample and invalidating all downstream analysis. Second, despite a sound plan, a key metric was computed with an incorrect formula (simple average instead of weighted average), producing misleading channel insights. Third, even with flawless calculations, the agent failed to synthesize findings into a context-aware conclusion and omitted mandatory limitations and a safety disclaimer. Together, these cases demonstrate that reliable DA outputs require aligned rigor across planning, implementation, and interpretation, with checks that prevent any single stage from compromising the whole.

#### E Annotation Details

###### E.1 Data Collection

Data synthesis for DE tables. Our DE tables originate from 73 enterprise-grade SaaS domains and their companion data-transformation projects, providing production-style schemas and realistic dependencies. Starting from a minimal business contract (target grain, primary/foreign keys, required metrics), we expand to end-to-end datasets and scale them while preserving business semantics and referential integrity. To keep the data mock both controllable and realistic, we highlight only the key steps: 1) Schema fidelity:

###### DE-Evol Pipeline-Level Error Case

||Layer 1: Staging<br><br>📁 staging/stg_account_history.sql<br><br>SELECT account_id, account_name, updated_at,<br><br>ROW_NUMBER() OVER (...) = 1 as is_most_recent FROM raw.account_history<br><br>|[Figure 82]|
|---|
<br><br>❌ The same account_id has multiple records with is_most_recent = true.|
|---|
<br><br>|Layer 2: Intermediate<br><br>📁 intermediate/int_account_summary.sql<br><br>SELECT a.account_id, a.account_name,<br><br>SUM(c.spend) as total_spend FROM staging.stg_account_history a JOIN staging.stg_campaign_stats c<br><br>ON a.account_id = c.account_id WHERE a.is_most_recent = true GROUP BY a.account_id, a.account_name<br><br>❌ The spend for A001 was double-counted !<br><br>[Figure 83]<br><br>|1,000|
|---|
<br><br>|
|---|
<br><br>Erroneous data flows downstream<br><br>The erroneous data continues to propagate<br><br>|Layer 3: Marts<br><br>📁 marts/executive_dashboard.sql<br><br>SELECT SUM(total_spend) as company_total_spend, AVG(total_spend) as avg_account_spend, COUNT(*) as account_count<br><br>FROM intermediate.int_account_summary<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|
|---|
|
|---|

- Figure 15 DE-Evol pipeline-level error case. Layer 1 (Staging) contains duplicate “current” rows where the same account_id appears multiple times with is_most_recent = true. Layer 2 (Intermediate) joins to campaign stats while filtering on is_most_recent = true, causing A001’s spend to be double-counted (total_spend becomes 20,000 instead of 10,000). Layer 3 (Marts) aggregates the erroneous intermediate table, inflating company_total_spend and avg_account_spend (35,000 and 17,500) compared with the correct values (25,000 and 12,500). The figure highlights how a seemingly small staging inconsistency can cascade into materially incorrect executive metrics.

- Table 20 Focused case study of a critical Planning Error. This table analyzes the agent’s plan against a pivotal standard (Data Scoping), highlighting omitted steps (in red) that led to a fundamentally flawed analysis.

Required Planning Step (from Rubric)

Agent’s Plan vs. Actual Action Outcome

Case Study: Standard 1.1 — Data Understanding & Scoping

- Step 1.1.A.1: Filter using the structured Education Requirement column.

The agent correctly planned and executed this step. ✓ PASS

- Step 1.1.A.2: Additionally extract candidates from the Job Description column.

CRITICAL PLANNING FAILURE: This step was entirely omitted from the agent’s plan; it never considered searching this column.

✗ FAIL

- Step 1.1.A.3: Further apply complex filtering rules to the Job Description column.

CRITICAL PLANNING FAILURE: This more advanced step was likewise completely absent from the agent’s plan.

✗ FAIL

Consequence of the Flawed Plan: By omitting two required data sources in the planning stage, the agent analyzed an incomplete and biased sample (9,073 records instead of the correct 11,838), thereby invalidating all subsequent analysis. This is a textbook Planning Error: once the initial strategy is faulty, execution quality cannot rescue the outcome.

Final score for this standard: 1 / 4.

- Table 21 Focused case study of a critical Execution Error. This table examines the agent’s implementation for Standard 2.1 (Channel Performance Metrics), illustrating how an otherwise sound plan partially failed due to an improper formula for a key metric.

Calculation Required by the Rubric

Agent Implementation vs. Correct Method Outcome

Case: Standard 2.1 — Channel Performance Metrics

- Sub-standard 2.1.A.1: Compute Sales Volume by channel.

The agent correctly used ‘GROUP BY‘ with ‘SUM(sales_volume)‘.

✓ PASS

- Sub-standard 2.1.A.2: Compute Total Revenue by channel.

The agent correctly used ‘GROUP BY‘ with ‘SUM(total_revenue)‘.

###### ✓ PASS

CRITICAL EXECUTION ERROR: The agent treated unit price as a simple average rather than a revenue-weighted average. As a result, the reported average prices were incorrect and led to misleading conclusions about channel profitability.

Sub-standard 2.1.A.3/4:

###### ✗ FAIL

Compute Average Unit Price by channel.

Impact of the Execution Deviation: Although the agent’s overall plan for channel analysis was sound, using the wrong formula for a single critical metric (Average Unit Price) produced misleading conclusions about channel profitability, directly undermining any price-based strategic recommendation. This constitutes a canonical Execution Error.

Final score for this standard: 5 / 6.

- Table 22 Focused case study of a critical Interpretation Error. The table shows a stark contrast between the agent’s successful execution of calculations and its failure to synthesize those results into a meaningful, context-aware conclusion.

Analytical Stage (from Rubric)

Agent’s Performance & Justification Outcome

Case Study: Analysis of Students with Suicidal Ideation

- Stage 1: Execution & Calculation (Standards 1.1 – 1.4)

The agent’s plan was sound and its execution was flawless. It successfully filtered the correct data population and accurately calculated all required statistical metrics (e.g., average economic/academic pressure, lifestyle habit percentages).

✓ PASS

- Stage 2: Interpretation & Synthesis

- (Standard 1.5: Create a "high-risk profile")

CRITICAL INTERPRETATION FAILURE: The agent failed to synthesize the previously calculated statistics into a coherent, higherlevel insight. Instead of creating a "profile," the agent merely listed the numbers again. The judge noted the summary was not deep enough and merely restated the table’s content.

✗ FAIL

Stage 3: Contextual Understanding

- (Standard 2.2: Provide safety disclaimer)

CRITICAL INTERPRETATION FAILURE: The agent’s final output completely omitted the mandatory "Limitations and Safety Disclaimer." This demonstrates a failure to understand the serious and sensitive context of the topic, which is a key part of providing a responsible and complete analytical deliverable.

###### ✗ FAIL

Consequence of Flawed Interpretation: This case exemplifies a pure Interpretation Error. The agent acted

- as a perfect calculator, producing correct data (Stage 1). However, it failed at the final and most critical stage: transforming that data into a meaningful, insightful, and contextually appropriate conclusion (Stages 2 & 3).

retain PK/FK, uniqueness, not-null, and domain constraints; 2) Distributions & dependencies: fit marginal distributions and model conditional links (e.g., country⇒currency/timezone); 3) Temporal coherence: inject seasonality, trend, and holiday effects while maintaining fact–dimension integrity; 4) Noise & edge cases: introduce controlled missingness/outliers/type coercions and design stressors that expose pipeline fragility (e.g., duplicate “current” rows, currency conflicts, timezone mismatches). The synthesis pipeline is implemented in Python (pandas, numpy, faker) with custom generators to scale volume while honoring inter-column dependencies and business invariants.

###### E.2 Construction details of DAComp-DE

This subsection presents our experience constructing the DAComp-DE corpus. We outline an end-to-end process across three tracks—Architecture, Implementation, and Evolution—spanning the baseline derived from 73 enterprise-grade SaaS domains and their data-transformation projects to pure-SQL normalization and validation, high-level requirement setting for blueprinting, contract-driven realization into working SQL, and change-oriented migration under realistic constraints. The summary reflects decisions and best practices agreed upon by domain experts to ensure rigor, reproducibility, and evaluability.

- E.2.1 Construction details of DAComp-DE-Architecture

Baseline curation and normalization. We first select open-source dbt projects that are license-compliant and empirically verified to be error-free, and normalize them into pure-SQL repositories by expanding materializations and macros while freezing model dependencies. Senior data engineers conduct a systematic audit of join semantics, analytical grains, window specifications, SCD handling, and testing assumptions, thereby establishing a high-quality baseline suitable for controlled evaluation.

High-level requirement formulation. Building on this baseline, we define task statements grounded in realistic enterprise scenarios: they provide only business context, overarching objectives, and expected outputs, without detailed metric definitions, precise calculation rules, or data constraint specifications. Such descriptions emphasize openness and cross-system characteristics, reveal gaps not covered by the existing repository, and intentionally avoid prescribing implementation paths or technical details. The model is expected to autonomously plan a blueprint—identifying key entities and dependencies, delineating layers and boundaries, and completing testing and freshness strategies—ultimately producing an executable architectural blueprint that evaluates its ability to plan end-to-end SQL projects and set constraints under incomplete information.

- E.2.2 Construction details of DAComp-DE-Implementation

Contract formalization. DE-Impl is constructed by deriving a rigorous requirements specification from the vetted SQL baseline in the form of a standardized data_contract.yaml that follows enterprise conventions. The contract formalizes model inventory and lineage, table and column schemas with constraints, declared grains and time windows, metric definitions with coherent units and currency normalization, as well as data quality, freshness, and performance policies.

- E.2.3 Construction details of DAComp-DE-Evolution

Change specification. For DE-Evol, we start from a high-quality, production-style SQL repository and propose change requests driven by realistic enterprise pressures—such as revised metric definitions, altered analytical windows, schema drift, or governance hardening. Multiple experts specify unambiguous business semantics, distinguish breaking from non-breaking changes, and design a safe migration plan that anticipates dependency revisions and testing upgrades.

###### E.3 Annotation details of DAComp-DA

In this section, we present the experience regarding the annotation of DAComp-DA data, which is summarized from our previous project discussion meetings and alignment meetings.

###### E.3.1 Core Design Principles

Strategic diversity The core of the Rubric is to evaluate problem-solving strategies, not steps. Each scoring Path must represent a methodologically distinct and self-contained solution. We avoid designing complete versus abridged versions of the same path. For example, analyzing all provinces and analyzing a subset of provinces should not be two separate Paths; the latter is merely an incomplete execution of the former.

Objective evaluation Scoring criteria must be quantifiable and reproducible to minimize scorer subjectivity. All items should be based on explicit evidence. Guideline: Any Accuracy item requiring numerical verification must have a pre-calculated Anchor Value. For open-ended paths without a single correct answer, a Pseudocode or a clear methodological verification process must be provided.

Dimensional separation of abilities Complex analytical skills are decomposed into independent scoring dimensions for a fairer and more granular assessment of model performance. Guideline: Strictly distinguish between procedural execution (were the steps completed?), computational accuracy (were the numbers correct?), and insightful conclusion (was the interpretation meaningful?), designing them as separate scoring items.

###### E.3.2 Structural components of the rubric

The Rubric employs a four-level hierarchical structure to deconstruct tasks, ensuring comprehensive and granular evaluation.

Requirement. Definition: The highest-level objective of the task, directly corresponding to a core analytical request from the user. Example: Analyze the differences in employee attrition rates across departments and their causes.

Standard. Definition: A key analytical step that must be completed or a core conclusion that must be reached to fulfill a Requirement. Example: Standard 1: Calculate and verify the attrition rate differences between departments; Standard 2: Identify the key factors causing these differences.

Path. Definition: A methodologically distinct and valid strategy for meeting a Standard. This is the core of the Rubric’s design. Example: Under the standard of verifying differences, Path A could be performing a statistical significance test (e.g., Chi-squared test), while Path B could be making a descriptive statistical comparison (e.g., percentage difference).

Sub-standard / rubric item. Definition: The smallest scorable unit of the Rubric, nested under a specific Path and adhering strictly to the principle of dimensional separation. It comprises three main types:

- • Completeness: Assesses whether all required steps for a given Path were executed. Focuses on what was done.
- • Accuracy: Assesses whether the computational results or execution process are correct. Focuses on if it was done correctly. For deterministic paths, this is verified against an Anchor Value; for open-ended paths, it is verified against a methodological process or Pseudocode.
- • Insightfulness: Assesses whether a reasonable and valuable conclusion or insight was derived from the correct results. Focuses on if the results were understood.

###### E.3.3 Golden Rules for Authors

These are the disciplinary requirements to ensure the quality and consistency of the Rubric. While these guidelines ensure consistency in creating rubrics for known strategies, the following section details our methodology for fairly evaluating novel or unanticipated solutions that may not align with pre-enumerated paths.

Calculate first, then author. Before finalizing the rubric, authors must personally run the complete analysis with code to calculate all Anchor Values required for the Accuracy assessment. This is the cornerstone of ensuring objective scoring.

Be specific and unambiguous. Every statement in the Rubric must be directive and unambiguous. Avoid subjective terms like approximately, good, or relatively comprehensive to minimize scorer discretion.

Avoid zero-point paths. If a method is not worthy of credit, it should not be designed as a distinct Path. A model’s output that does not match any valid path will naturally receive no score for that standard.

#### F Discuss

###### F.1 Discussion of Handling Unenumerated Solution Paths

Accuracy is the most critical dimension in our rubric. Since fully listing all valid analytical paths is often infeasible, we adopt a three-tier, progressively relaxed design for Accuracy: (i) direct enumeration with numeric anchors when the correct outcome can be exhaustively determined; (ii) constrained computation with pseudo-code anchors when procedures are well-defined but paths are not exhaustively enumerable; and (iii) principle-based assessment for highly open-ended cases.

Standardized assessment for common paths. We standardize scoring whenever we can verify correctness deterministically. Tier-1 (numeric anchors): for tasks whose outcomes can be exhaustively enumerated, we embed the reference value directly into the rubric (e.g., “How many users satisfy condition X?”), yielding absolute, reproducible checks. Tier-2 (pseudo-code anchors): for tasks with well-specified computation but multiple equivalent derivations (e.g., a conversion rate with alternative weighting schemes), we prescribe canonical steps in pseudo-code to constrain the procedure. This enables process-level verification (inputs, ordering, aggregation, null/edge handling) without enumerating every path, preserving both precision and reproducibility.

Principle-based assessment for novel paths. A minority of tasks are intrinsically open-ended, where enumeration or pseudo-code templating is impractical. Here we evaluate Accuracy via methodological principles rather than a single anchor value. For example, a “key-driver identification” task may be solved by regression with coefficient interpretation (a pre-defined path), or by gradient boosting with SHAP attributions (an unenumerated path). We score such solutions on: (1) Methodological appropriateness (the method is suitable for the stated objective and data regime); (2) Correctness of execution (the pipeline is implemented soundly, with valid preprocessing, estimation, and validation); and (3) Soundness of interpretation (claims follow from the produced evidence, with clear caveats). This soft layer ensures valid but unconventional approaches are not penalized.

By construction, most DAComp items fall into Tiers 1–2, where numeric or pseudo-code anchors provide deterministic checks; Tier 3 is reserved for genuinely open-ended cases to maintain fairness without sacrificing rigor.

###### F.2 Discussion of Ambiguous of Requirements

Implementation and Evolution tasks in DAComp-DE are designed as deterministic evaluations. To balance realism with unambiguous executability, we adopt three principles:

- 1) Professionalism. Requirements are sourced from enterprise-style projects and vetted by senior data engineers for cross-layer impact, metric definitions, SCD handling, and temporal semantics. Implementation tasks emphasize canonical modeling pipelines from scratch; Evolution tasks mirror real “change requests” (e.g., metric revision, source replacement).
- 2) Unambiguity. Implementation (node-first): each SQL node has atomic contracts (schema, PK/grain, time, nulls, joins, aggregation, SCD, idempotency). Multiple agents must converge under frozen contracts;

- discrepancies trigger tighter specifications. Evolution (delta-first): natural-language changes are mapped into minimal verifiable deltas (schema/logic/lineage), with explicit impact scope and before–after anchors; agent disagreement leads to refined deltas or explicit assumptions.
- 3) Realism. Implementation: converged nodes are composed into multi-node tasks, with contracts and assumptions documented (e.g., data_contract.yaml). Evolution: favors backward-compatible evolution (added columns/views, metric versioning); destructive changes require migration notes. All assumptions are logged for reproducibility.

###### F.3 Discussion on the Selection of Judger LLM

- As shown in Tab. 7, both O4-Mini and gemini-2.5-flash achieve human-level agreement, while stronger proprietary models (e.g., gemini-2.5-pro, GPT-5) yield even higher consistency. For DAComp„ we standardize on gemini-2.5-flash, as it balances (1) cost efficiency for large-scale benchmarking, (2) stable and low-latency inference, (3) reproducibility across runsand (4) community accessibility. Choosing a widely available model ensures that our evaluation pipeline can be easily adopted, verified, and extended by others.

F.4 Discussion on End-to-End Evaluation

Current DAComp tasks span complementary stages of the data intelligence lifecycle: DE-Architecture (highlevel specification and planning), DE-Implementation (multi-layer pipeline construction), DE-Evolution (safe modification under requirement changes), and DA (open-ended analysis over downstream data). Taken together, these stages delineate a strictly end-to-end process—from requirement articulation, through system realization and iterative evolution, to analytical insight and decision support—covering a full loop from planning and implementation to evolution and interpretation.

- At present, we evaluate these stages modularly and in a decoupled fashion to enable controlled measurement

- at each step. Our next key objective is to integrate them into a single, end-to-end longitudinal evaluation: a single agent carries requirements through implementation and change propagation, and ultimately completes analysis and reporting. We contend this end-to-end setup offers substantial scientific and practical value: it stress-tests the end-to-end consistency of planning–execution–evolution–interpretation, better reflects real engineering workflows, and advances toward a comprehensive assessment of autonomous data agents’ end-to-end capabilities.

