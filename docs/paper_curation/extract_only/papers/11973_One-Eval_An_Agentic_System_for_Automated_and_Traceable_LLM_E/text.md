# arXiv:2603.09821v1[cs.CL]10Mar2026

##### One-Eval: An Agentic System for Automated and Traceable LLM Evaluation

Chengyu Shen1*, Yanheng Hou2*, Minghui Pan3*, Runming He1, Zhen Hao Wong1, Meiyi Qiang1, Zhou Liu1, Hao Liang 1,4, Peichao Lai1, Zeang Sheng1, Wentao Zhang1,4†, 1Peking University, 2Beijing Institute of Technology, 3Beijing University of Posts and Telecommunications, 4Zhongguancun Academy scuuy05@gmail.com

Abstract

Reliable evaluation is essential for developing and deploying large language models, yet in practice it often requires substantial manual effort: practitioners must identify appropriate benchmarks, reproduce heterogeneous evaluation codebases, configure dataset schema mappings, and interpret aggregated metrics. To address these challenges, we present OneEval, an agentic evaluation system that converts natural-language evaluation requests into executable, traceable, and customizable evaluation workflows. One-Eval integrates (i) NL2Bench for intent structuring and personalized benchmark planning, (ii) BenchResolve for benchmark resolution, automatic dataset acquisition, and schema normalization to ensure executability, and (iii) Metrics & Reporting for taskaware metric selection and decision-oriented reporting beyond scalar scores. The system further incorporates human-in-the-loop checkpoints for review, editing, and rollback, while preserving sample evidence trails for debugging and auditability. Experiments show that One-Eval can execute end-to-end evaluations from diverse natural-language requests with minimal user effort, supporting more efficient and reproducible evaluation in industrial settings. Our framework is publicly available at https://github.com/OpenDCAI/One-Eval.

###### 1 Introduction

With the rapid adoption of large language models and multimodal models in industrial systems (Team et al., 2026a; Bai et al., 2025; Liang et al., 2025), model evaluation has become a critical component throughout the model lifecycle, including development, selection, iteration, and pre-deployment validation (Srivastava et al., 2023; Chang et al., 2023). Evaluation results are no longer used solely for reporting benchmark scores, but increasingly serve

* Equal contribution. † Corresponding author.

as decision-making signals for model comparison, deployment readiness, and risk assessment (Yang et al., 2025; DeepSeek-AI et al., 2025). As evaluation objectives grow more diverse and task-specific, existing evaluation workflows struggle to provide sufficient flexibility and usability in practice.

In current mainstream practices, model evaluation typically follows one of two approaches. Users either identify and reproduce task-specific benchmark repositories (Hendrycks et al., 2021a), manually setting up environments and running scripts, or rely on static evaluation frameworks that require explicit configuration of models, datasets, parameters, and metrics (Gao et al., 2024; Contributors, 2023). While these approaches standardize execution to some extent, they still place a heavy burden on users to discover appropriate benchmarks, construct valid configurations, and interpret results. Such workflows are highly experience-dependent, costly to iterate, and difficult to adapt to evolving evaluation needs.

Meanwhile, agent-based systems have gained significant traction in industrial applications (Yang et al., 2024a; Team et al., 2026b). Prior work has shown that agentic systems can reduce engineering overhead by allowing users to express high-level goals rather than low-level procedures (Yao et al., 2023; Luo et al., 2025; Mao et al., 2025). This motivates a rethinking of model evaluation as an agent-driven task, where the core challenge lies not only in executing evaluations, but in transforming abstract evaluation intents into reliable and actionable evaluation pipelines.

However, treating model evaluation as an end-toend agent-driven process remains underexplored. Existing tools primarily focus on execution and score aggregation, while treating benchmarks and metrics as static configurations. They rarely address higher-level stages such as evaluation intent interpretation, personalized benchmark selection, configuration validation, or result analysis tailored

to downstream decisions. As a result, evaluation outputs are often limited to isolated scalar metrics, which are insufficient for supporting real-world industrial decision making.

In this paper, we propose One-Eval, an agentic evaluation framework that transforms natural language evaluation requests into executable, verifiable, and customizable evaluation workflows. OneEval follows an end-to-end design with three main stages. First, NL2Bench interprets natural language requests, decomposes evaluation intents, and retrieves or recommends benchmarks that align with

- user goals, with support for interactive refinement. Second, automated benchmark resolution and settings completion handle dataset acquisition, dependency management, and configuration validation, reducing manual effort and configuration errors. Third, One-Eval performs metric recommendation and task-oriented report generation, producing structured, decision-support evaluation reports rather than single scalar scores. To ensure reliability, One-Eval incorporates a human-in-the-loop mechanism at key decision points, enabling users to review and refine agent decisions while preserving automation efficiency. 2 Related Work

Model Evaluation. Model evaluation has long been a central topic in natural language processing and has gained renewed importance with the rise of large language models. A wide range of benchmarks have been proposed to assess model capabilities across domains, including mathematical reasoning benchmarks such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b), and broad knowledge and reasoning benchmarks such as MMLU (Hendrycks et al., 2021a). In addition, evaluation toolkits such as lm-eval-harness (Gao et al., 2024) and OpenCompass (Contributors, 2023) provide standardized interfaces for running benchmarks and aggregating scores. While these frameworks improve evaluation reproducibility, they largely assume predefined tasks, benchmarks, and metrics, leaving users to manually map evaluation goals to concrete evaluation setups.

Automation and Agent-Based Systems. Agentbased and multi-agent systems have shown strong effectiveness in automating complex, multi-step tasks such as code generation and tool-oriented workflows (Yang et al., 2024b; Wu et al., 2023). By decomposing high-level goals into sequential

decisions, these approaches reduce manual effort and support iterative refinement. From a structural perspective, model evaluation is also a multi-stage process involving intent interpretation, benchmark selection, execution, and result analysis. However, existing work has largely applied automation to isolated components, rather than treating it as an end-to-end, agent-driven decision process, resulting in fragmented automation support in practice.

Personalized Evaluation and Reporting. Most existing evaluation studies present results as single or aggregated metrics (Rein et al., 2023; Zhong et al., 2023), which support standardized comparison but offer limited guidance for practical deployment decisions. Prior work has explored multidimensional evaluation to better characterize model behavior (Liang et al., 2023; Srivastava et al., 2023), yet these approaches typically rely on fixed evaluation dimensions and static reporting formats. As a result, evaluation outputs remain weakly aligned with user-specific goals and task requirements. Motivated by these limitations, our work focuses on evaluation requirement modeling, evaluation workflow automation, and task-oriented report generation, enabling an end-to-end evaluation paradigm driven by user objectives.

###### 3 System Design

###### 3.1 Framework Overview

One-Eval is an agentic evaluation framework designed to transform high-level, natural language evaluation requests into executable and verifiable model evaluation workflows. Instead of requiring users to manually identify benchmarks, configure evaluation settings, and interpret results, One-Eval treats model evaluation as an end-to-end decision process driven by user intent.

- As illustrated in Figure 1, One-Eval follows a

modular, three-stage pipeline. Given a user’s evaluation request expressed in natural language, the framework first interprets the evaluation intent and constructs an appropriate evaluation plan. It then resolves benchmarks and evaluation settings to produce an executable evaluation workflow, and finally generates task-oriented evaluation results and reports that support downstream decision making. A human-in-the-loop mechanism is integrated throughout the pipeline, allowing users to inspect, refine, and validate intermediate decisions when necessary.

- At a high level, One-Eval consists of the follow-

###### Step1: NL2Bench

###### Step2: BenchResolve

###### Step3: Metrics & Reporting

Intent to Benchmark Planning

Resolve, Download, Configure

Run Evaluation & Generate Report

[Figure 1]

Local

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

HuggingFace

Intent Structuring

[Figure 10]

User Evaluation Request

[Figure 11]

###### Task-Oriented Evaluation Report

###### Local-first Resolution

###### HF tools Resolution

###### Metric Recommendation

(domain, constraints, target_model, ...)

###### Benches Retrieval

curated configs

search + structure

task-aware

[Figure 12]

Scores & Statistics distribution

###### Executable BenchInfo

[Figure 13]

[Figure 14]

"Evaluate my model on math reasoning and provide actionable insights."

[Figure 15]

[Figure 16]

Rank in different models

[Figure 17]

[Figure 18]

Planned Benches

[Figure 19]

[Figure 20]

Benchmark Selection

[Figure 21]

Failure data analysis

###### Auto Download

###### Schema Structured

###### DataFlow Evaluator

Heterogeneous datasets

Actionable suggestions

[Figure 22]

datasets & metadata

split/subset & column mapping

[Figure 23]

- id:1, name:GSM8K, domain:math,...
- id:2, name:Truthful-QA, domain:...
- id:3, ...

DataFlow engine for evaluation

Unified interface

Bench Gallery

###### Human-in-the-Loop

- 1. Review & Interrupt
- 2. Edit
- 3. Approve request
- 4. Rollback & Re-run

Figure 1: One-Eval overview. One-Eval converts a natural-language evaluation request into an executable EvalPlan (NL2Bench), resolves and configures benchmarks by automatic dataset download and schema normalization (BenchResolve), and produces task-aware metrics and a decision-oriented evaluation report (Metrics & Reporting), with human-in-the-loop refinement at key steps.

ing components. (1) NL2Bench translates natural language evaluation requirements into structured evaluation intents and recommends suitable benchmarks that align with user goals. (2) Benchmark Resolution and Configuration completes dataset acquisition, configuration construction, and validation to ensure the evaluation workflow is executable and consistent. (3) Metric Recommendation and Reporting selects evaluation metrics based on task requirements and produces structured, task-oriented evaluation reports rather than isolated scalar scores.

By explicitly modeling evaluation intent, workflow construction, and result interpretation as interconnected stages, One-Eval bridges the gap between user goals and executable evaluation pipelines. This design enables flexible customization, reduces manual configuration effort, and provides evaluation outputs that are directly actionable in practical deployment scenarios.

###### 3.2 NL2Bench

NL2Bench is the entry point of One-Eval. Given a natural language evaluation request, it produces an executable benchmark plan: a curated set of benchmarks together with the minimal metadata needed for downstream execution (e.g., canonical identifiers, evaluation splits, and schema hints). The plan can be iteratively refined through lightweight user interaction to ensure that the selected benchmarks truly match the user’s intent.

Intent Structuring. NL2Bench first translates the user request into a structured intent representa-

tion that captures (i) the target evaluation domain and capability focus (e.g., mathematical reasoning, general knowledge, text QA), (ii) any benchmarks explicitly specified by the user, (iii) execution constraints such as language or formatting requirements, and (iv) additional preferences that are difficult to encode as fixed fields. This structured representation serves as the control signal for subsequent retrieval and selection.

Candidate Retrieval. Based on the structured intent, NL2Bench retrieves benchmark candidates from two complementary sources. The first source is a local benchmark gallery of 77 curated benchmarks. We construct this gallery by collecting publicly available evaluation datasets, removing all entries whose data files cannot be successfully loaded or parsed, and retaining only those benchmarks that execute end-to-end without error. Each surviving benchmark is stored together with its canonical metadata (aliases, category tags, tasktype annotations, HuggingFace configuration, and key mappings), forming a self-contained registry of ready-to-run evaluations. To match the user query against this gallery, we provide two interchangeable retrieval backends that share the same API: (i) an embedding-based mode that encodes both the query and benchmark descriptions into dense vectors and ranks candidates by cosine similarity, and (ii) a lightweight TF-IDF mode that tokenizes mixed Chinese–English text and combines cosine similarity with a keyword-overlap bonus, requiring no external service. A relevance threshold τ (set to 0.5 for embedding retrieval and 0.3 for TF-

IDF) partitions the results into quality matches and marginal matches: when the number of quality matches is below the desired count k, the system falls back to a second source—live search over the HuggingFace Hub—to cover long-tail and newly released benchmarks. The threshold is calibrated so that the embedding mode, which produces semantically grounded similarity scores, applies a stricter cutoff to maintain precision, while the TFIDF mode, whose scores are inherently noisier due to surface-level lexical matching, uses a more permissive cutoff to preserve recall. Candidates from both sources are merged with any user-specified benchmarks to form a unified pool for validation and selection.

Resolution and Normalization. To ensure executability, NL2Bench normalizes each candidate into a canonical benchmark identifier and collects essential structural metadata. For external benchmarks, the agent reads dataset metadata (e.g., dataset cards and split/configuration information) and inspects feature fields when necessary, converting heterogeneous representations into a unified internal schema. Resolved benchmarks are presented in a benchmark gallery, which simultaneously provides user-facing explanations (why a benchmark is suggested) and supplies consistent configuration entry points for downstream execution.

Selection Under Constraints. NL2Bench selects a compact subset of benchmarks that best match the user intent while respecting practical constraints such as evaluation cost, redundancy, and executability. In practice, this is implemented by combining intent-alignment scoring with rulebased validation, successful resolution checks, and budget-aware pruning. This design avoids overselecting similar benchmarks and reduces the risk of producing plans that cannot be executed due to missing splits, incompatible schemas, or unavailable resources.

Human-in-the-Loop. Because benchmark selection is inherently open-ended and misalignment can invalidate evaluation results, NL2Bench integrates human-in-the-loop refinement via interrupt points. The system shows the current benchmark plan with concise justifications (e.g., domain match, capability coverage, dataset characteristics) and allows the user to approve, edit the plan, refine the request, or inject a custom local benchmark. If the user modifies the intent, NL2Bench re-runs retrieval and selection until the user confirms a satisfactory plan.

The final output of NL2Bench is a user-approved benchmark plan with normalized identifiers, structural metadata, and configuration entry points, which is directly consumed by the next stage for executable resolution and configuration.

3.3 Benchmark Resolution and Configuration Benchmark Resolution and Configuration, orchestrated by BenchResolveAgent, turns the nominal benchmark plan from NL2Bench (user-specified and recommended) into executable and reproducible configurations. To handle real-world heterogeneity in hosting sources, schemas, task definitions, and split conventions, the agent automatically resolves benchmark identifiers, acquires datasets when needed, and constructs validated configuration objects, enabling downstream evaluation to run without manual setup.

Hierarchical Benchmark Resolution. To balance stability for widely used benchmarks with extensibility to long-tail benchmarks, One-Eval adopts a hierarchical resolution strategy with a local-first, dynamic fallback design. The system maintains a local registry of high-frequency benchmarks, each associated with expert-validated configurations. When a benchmark matches the registry, BenchResolveAgent loads the predefined configuration directly (including verified evaluation splits, column mappings, and task annotations), ensuring stable and reproducible execution across environments.

For benchmarks not found in the local registry, One-Eval falls back to HuggingFace for dynamic resolution: it first tries direct loading via the given name, and otherwise searches for candidates and selects the best match using lightweight heuristics (e.g., suffix cues and semantic similarity). Once resolved, the dataset and metadata are downloaded and integrated automatically, enabling seamless use of previously unseen community benchmarks without manual access or compatibility handling.

Unified Configuration and Heterogeneous Data Adaptation. To decouple evaluation logic from data representations, One-Eval normalizes each resolved benchmark into a unified configuration object (BenchInfo) stored in the system state. BenchInfo records the dataset source (HuggingFace ID or local path), the evaluation subset/split, a column mapping to One-Eval’s standardized input– output interface, and task metadata for downstream metric recommendation. BenchResolve validates

these fields during resolution and persists them as traceable artifacts (e.g., resolved IDs and cache paths), making protocol choices inspectable and reproducible across runs. This abstraction separates evaluation execution from data heterogeneity and enables seamless integration of curated internal benchmarks and community datasets, supporting scalable evaluation workflows in industrial settings.

###### 3.4 Metric Recommendation and Reporting

Following the execution phase, this module serves as the analytical core, transforming the raw model outputs into actionable decision signals. Addressing the static evaluation framework and limited guidance (as highlighted in Sec. 1), One-Eval adopts an agentic pipeline that couples semantic reasoning with rule-based priors to orchestrate metric selection, execution, and root-cause reporting.

Dual-Track Metric Recommendation. To reconcile the flexibility required for unseen agentic tasks with the robustness needed for standard benchmarks, the MetricRecommendAgent implements a prioritized dual-track strategy that eliminates the need for manual configuration: (1) User Override (Static Control): explicit metric configurations provided in benchmark metadata take strict precedence, enabling bespoke evaluation protocols when required. (2) Knowledge-Augmented Reasoning (Dynamic Adaptation): for unconfigured or open-ended tasks, the agent performs semantic reasoning over rich dataset context (e.g., prompt templates, few-shot samples, task descriptors), grounded by dynamic prompt construction that scans the registered metric library at runtime to generate semantic descriptions and decision rules; these are injected into the LLM context to guide metric selection. (3) Registry Fallback: if the LLM fails to produce a valid plan, the system reverts to rule-based suggestions from the MetricDispatcher or a minimal default set to guarantee pipeline continuity.

Decentralized Metric Registration. One-Eval provides an extensible metric ecosystem via a decentralized registration interface. New metrics are integrated by decorating computation functions with semantic metadata, after which the system automatically registers them into the global metric registry. This indexed library serves as the knowledge base for the agent’s recommendations.

Execution Engine. Once metrics are selected, the ScoreCalcAgent invokes the MetricRunner as a unified execution layer. It normalizes hetero-

geneous inputs, aligns predictions with references, supports parallel execution for large-scale datasets, and packages results with scores, priorities, and details when available.

Hierarchical Diagnostic Reporting. To overcome the limitation of isolated scalar metrics, OneEval generates multi-granular diagnostic reports via ReportGenAgent: (1) Macro View (Capability Profiling): aggregates results into radar and sunburst summaries for holistic capability profiling. (2) Diagnostic View (Root Cause Analysis): attributes failure modes (e.g., instruction-following errors vs. hallucinations), performs blind-spot analysis over failed samples, and summarizes length distributions for correct vs. incorrect outputs. (3) Micro View (Case Study): provides case-level inspection tables that link aggregate metrics to specific failure instances.

Specialized Metrics. To support the hierarchical reporting described above, One-Eval incorporates a comprehensive library of custom metrics designed to uncover specific failure modes. Table 3 highlights a representative subset of these featured metrics, selected to demonstrate how the system moves beyond standard accuracy to capture domain-specific nuances (e.g., symbolic equivalence in math) and behavioral patterns (e.g., format compliance). These metrics serve as the building blocks for the diagnostic views in the final report.

###### 4 Experiments

We evaluate One-Eval from an industrial usability and reliability perspective. Rather than targeting leaderboard improvements on a fixed benchmark suite, our experiments focus on whether One-Eval can (i) produce actionable end-to-end evaluation outputs from natural-language requests with minimal user effort, (ii) reliably generate executable evaluation plans and run them through to results without human edits, and (iii) provide practical capabilities beyond existing evaluation frameworks. To this end, we conduct three complementary studies: (1) qualitative case studies that illustrate the full workflow and decision-oriented reporting, (2) a controlled end-to-end success-rate evaluation on a diverse set of evaluation requests to quantify executability and automation reliability, and (3) a feature-level comparison table against representative evaluation frameworks.

###### 4.1 Case Study

We present a real run to illustrate how One-Eval turns a single natural-language request into an executable and auditable evaluation workflow, and how it preserves an evidence trail for diagnosing failures.

Request → plan. The user request is:

“I want to focus on broad general-knowledge coverage, and check whether the model can handle some light reasoning.”

One-Eval structures the request into intent slots (e.g., domain=[text, reasoning]) and

- uses NL2Bench to propose a benchmark plan spanning knowledge and reasoning, including mmlu, truthful_qa, commonsenseqa, and reasoning-oriented sets such as openai/gsm8k and HuggingFaceH4/MATH-500. Non-canonical names are resolved to concrete repositories (e.g., mmlu → cais/mmlu) to ensure executability.

Resolution, configuration, and schema normalization. BenchResolve automatically selects runnable configurations and splits with recorded rationales, and caches resolved datasets with reproducible paths. For example, GSM8K is configured as {config=main, split=test}, MATH500 uses {config=default, split=test} (only available subset), and TruthfulQA falls back to split=validation (no test split). BenchResolve then normalizes heterogeneous schemas via key mappings (e.g., GSM8K maps question to input and answer to target; TruthfulQA maps question to input and correct_answers to targets), enabling a unified evaluation interface across heterogeneous benchmarks.

###### 4.2 End-to-End Success Rate

A central question for any automated evaluation system is whether it can turn a free-form requirement into a ready-to-run evaluation plan without human edits. We collect 100 natural-language evaluation requests that span six broad capability domains (reasoning, mathematics, code, safety, retrieval, and factual QA) and feed each request into the One-Eval pipeline, running it from intent parsing through benchmark retrieval, configuration inference, and dataset preparation—stopping right before actual model evaluation. No manual correction is applied at any stage. We measure three cumulative success metrics along the pipeline and report the average decision time. Table 1 summarizes the results.

- Table 1: End-to-end success rates and efficiency on 100 evaluation requests. Each metric is cumulative: later checkpoints require all earlier ones to have succeeded.

Metric Count Rate Plan Executable Rate 99/100 99% Auto-Complete Rate 85/100 85% Full Plan Rate 84/100 84% Avg. Tokens 10,652

Plan Executable Rate measures whether the system can parse the user intent and retrieve at least one benchmark candidate without interruption; 99 out of 100 requests pass this checkpoint, with the single failure caused by a highly ambiguous query that the intent-structuring agent cannot decompose. Auto-Complete Rate further requires that the inferred subset, split, and key mappings are correct—errors at this stage would cause the downstream runner to crash or produce meaningless results; 85% of requests clear this bar, indicating that the automated schema inference is reliable for the majority of publicly available benchmarks. Full Plan Rate additionally demands successful task-type inference and metric recommendation, reaching 84%. The 8-step pipeline completes in a median wall-clock time of approximately 11.4 minutes per request (mean 13 minutes), demonstrating that One-Eval can deliver an executable evaluation plan from a single sentence of natural language within a practical time budget and without any human intervention.

Table 2: Feature-level comparison.

Framework Custom Automate Rec. Bench Rec. Metric

One-Eval (ours) ✓ ✓ ✓ ✓ lm-eval-harness ✓ ✗ ✗ ✗ OpenCompass ✓ ✗ ✗ ✗ HELM ✗ ✗ ✗ ✗

4.3 Feature-level Comparison

- Table 2 summarizes a feature-level comparison with representative evaluation frameworks. We focus on workflow-critical capabilities that affect practical benchmark evaluation, including customization, end-to-end automation, and intentconditioned recommendations. 5 Conclusion

One-Eval enables natural-language evaluation requests to be executed as traceable end-to-end workflows via NL2Bench, BenchResolve, and taskaware Metrics & Reporting. Experiments demon-

strate reliable execution with minimal manual effort and actionable evidence for auditing and debugging. Future work will broaden coverage to more tasks and modalities and further strengthen support for long-tail benchmarks.

###### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A survey on evaluation of large language models. Preprint, arXiv:2307.03109.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, and 245 others. 2025. Deepseek-v3.2: Pushing the frontier of open large language models. Preprint, arXiv:2512.02556.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. Preprint, arXiv:2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Hao Liang, Xiaochen Ma, Zhou Liu, Zhen Hao Wong, Zhengyang Zhao, Zimo Meng, Runming He, Chengyu Shen, Qifeng Cai, Zhaoyang Han, and 1 others. 2025. Dataflow: An llm-driven framework for unified data preparation and workflow automation in the era of data-centric ai. arXiv preprint arXiv:2512.16676.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, and 31 others. 2023. Holistic evaluation of language models. Preprint, arXiv:2211.09110.

Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, Rongcheng Tu, Xiao Luo, Wei Ju, Zhiping Xiao, Yifan Wang, Meng Xiao, Chenwu Liu, Jingyang Yuan, Shichang Zhang, and 7 others. 2025. Large language model agent: A survey on methodology, applications and challenges. Preprint, arXiv:2503.21460.

Zhenyu Mao, Jacky Keung, Fengji Zhang, Shuo Liu, Yifei Wang, and Jialong Li. 2025. Towards engineering multi-agent llms: A protocol-driven approach. Preprint, arXiv:2510.12120.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. Preprint, arXiv:2311.12022.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, and 432 others. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Preprint, arXiv:2206.04615.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, and 307 others. 2026a. Kimi k2.5: Visual agentic intelligence. Preprint, arXiv:2602.02276.

Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, and 181 others. 2026b. Kimi k2: Open agentic intelligence. Preprint, arXiv:2507.20534.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun

Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation. Preprint, arXiv:2308.08155.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

John Yang, Carlos E. Jimenez, Alexander Wettig, Maik Latzke, and 1 others. 2024a. Swe-agent: Agentcomputer interfaces enable automated software engineering. In International Conference on Learning Representations.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024b. Swe-agent: Agent-computer interfaces enable automated software engineering. Preprint, arXiv:2405.15793.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. Preprint, arXiv:2210.03629.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A humancentric benchmark for evaluating foundation models. Preprint, arXiv:2304.06364.

###### A Alignment with Representative Evaluation Frameworks

This section provides additional details on how we align the criteria in Table 2 with how representative evaluation frameworks are typically used in practice. Our goal is to make the comparison reproducible and interpretable by clarifying what each feature flag captures at the workflow level.

Scope of the comparison. The feature-level comparison is intentionally scoped to capabilities that directly affect evaluation workflows in industrial settings: (i) integrating organization-specific assets (custom benchmarks/metrics), (ii) reducing front-loaded setup work (automation), and (iii) supporting intent-conditioned selection (recommendations) rather than static or manual choices. The table is not intended to rank overall quality, speed, or benchmark coverage, but to summarize whether a framework exposes these capabilities as first-class components.

- Criterion 1: Custom benchmarks and metrics. We consider Custom supported when a framework provides a documented and maintainable mechanism to: (i) register a new benchmark/dataset (or adapter) and (ii) attach custom evaluation logic or metrics, without modifying core runner internals. In practice, this includes extension points such as benchmark registries, task adapters, dataset loaders, and pluggable metric modules. We also treat “custom” as supported when users can incorporate local datasets or private benchmarks through a clear interface (e.g., path-based loading with schema adapters) and keep them as reusable assets in subsequent runs.
- Criterion 2: End-to-end automation. We consider Automate supported when the system can start from a high-level evaluation requirement and produce a runnable evaluation plan with minimal manual specification of: benchmark identifiers, split/config selection, schema mappings, and execution settings. This criterion reflects whether the framework reduces the manual effort of “finding the right benchmark and making it runnable” rather than only providing an execution harness once the benchmark list and configuration are already known. In One-Eval, automation is realized as an agentic workflow that (a) converts intent to a benchmark plan, (b) resolves identifiers, (c) downloads assets, and (d) normalizes schemas into

- executable configurations, while recording intermediate artifacts for traceability.
- Criterion 3: Benchmark recommendation. We consider Rec. Bench supported when the system provides an explicit recommendation component that proposes benchmarks conditioned on user intent and constraints, instead of relying on static suites or manual selection. Concretely, this includes (i) intent structuring (domain/capability/constraints), (ii) candidate retrieval over curated registries and/or external spaces, and (iii) producing a benchmark plan with justifications that can be inspected or refined. One-Eval implements this capability in NL2Bench, which retrieves candidates from both curated registries and open repositories and outputs an EvalPlan that serves as the entry point for downstream execution.
- Criterion 4: Metric recommendation. We consider Rec. Metric supported when the system proposes a metric suite (and corresponding reporting template) conditioned on the task type and evaluation objective, rather than always reporting a fixed scalar (e.g., accuracy) or requiring users to manually choose metrics for each benchmark. This criterion covers not only selecting a metric function, but also packaging metrics into a task-oriented report structure (e.g., breakdowns, slices, and diagnostic signals). In One-Eval, metric recommendation is coupled with report generation to support decisionoriented evaluation outputs.

Interpretation of blanks/✗. Blank/✗ indicates that the capability is not exposed as an intentconditioned, first-class module in the framework and is typically achieved through manual selection/configuration or external scripting around the runner. This does not preclude users from implementing similar behavior with sufficient engineering effort; rather, the table summarizes what is provided natively as part of the framework’s workflow.

Reproducibility and reporting. To keep the comparison reproducible, we apply the above criteria consistently and focus on user-facing workflow primitives (configuration interfaces, registries, and built-in planning/recommendation modules). For One-Eval, the corresponding artifacts (plans, resolved benchmark identifiers, configuration choices, and report outputs) are persisted as part of the traceable evaluation state, allowing read-

ers to map each checked feature to concrete system outputs.

###### B Selected Custom Metrics

Table 3 summarizes a curated set of One-Eval’s custom metrics designed for diagnostic insight. These metrics go beyond generic aggregate scores by targeting specific failure modes (e.g., symbolic equivalence, schema/format compliance, and judge-based error attribution), enabling more actionable analysis and decision-oriented reporting across tasks.

###### C Additional Discussion

This section discusses practical considerations that motivate One-Eval’s design, focusing on deployment-oriented evaluation and how decisionmakers consume evaluation outcomes.

Decision-oriented evaluation outputs. In industrial settings, evaluation is commonly used to support decisions such as model selection, release gating, regression monitoring, and targeted iteration. As a result, practitioners often need more than an aggregate score: they need structured signals that help identify where a model succeeds or fails and why performance changes across versions. One-Eval operationalizes this by producing task-oriented reports that naturally accommodate multiple diagnostic views, such as breakdowns by benchmark/subset, failure-mode summaries, and objective-aligned indicators that can be reused in iterative evaluation cycles.

From scores to actionable signals. A practical report is most useful when it bridges model outputs to actions. One-Eval structures evaluation artifacts so that stakeholders can quickly connect (i) aggregate results to (ii) slices or categories of interest and (iii) representative examples. This structure supports common operational needs: identifying regressions, prioritizing data collection or prompt/template adjustments, and communicating evaluation findings across roles (engineering, product, and quality). By packaging these signals consistently, One-Eval supports evaluation as a repeatable process rather than an ad-hoc analysis step.

Traceability and auditability for iterative evaluation. Industrial evaluation often involves repeated runs over time, comparing model variants, configuration changes, and shifting requirements. One-Eval treats intermediate artifacts as first-class

state (plans, configurations, resolved benchmark identifiers, schema mappings, caches, and persample traces), enabling results to be audited and reproduced. This is particularly important when evaluation outcomes must be explained to stakeholders or used as part of a release decision, where reproducible evidence reduces ambiguity and accelerates review.

Human-in-the-loop as controlled refinement. Evaluation involves intent-sensitive choices (e.g., what constitutes “coverage” for a domain) and configuration-sensitive choices (e.g., how to handle multiple subsets/splits). One-Eval integrates human-in-the-loop interaction as controlled refinement at key checkpoints: users can confirm benchmark coverage, adjust constraints, and approve reporting preferences. In typical usage, these interactions are lightweight and focus on validating highimpact decisions, ensuring that automation remains aligned with the user’s evaluation objective.

Supporting evolving and heterogeneous requirements. Evaluation requirements vary across domains (reasoning, knowledge, safety, retrievalaugmented settings) and evolve rapidly as new benchmarks and practices emerge. One-Eval separates intent understanding, benchmark resolution/configuration, and evaluation/reporting into modular stages. This separation enables the system to incorporate new benchmark sources, add domain-specific report templates, and extend metric suites while keeping a stable end-to-end workflow abstraction for users.

###### D Consistency with Benchmark Cards and Standard Protocols

This section discusses how One-Eval promotes evaluation consistency by grounding configuration and execution in benchmark metadata (e.g., dataset cards) and by recording protocol-relevant decisions as reproducible artifacts.

Protocol grounding via benchmark metadata. Benchmarks in open repositories often provide guidance in dataset cards, including intended task definition, available configurations/subsets, supported splits, label semantics, and field descriptions. One-Eval retrieves and records such metadata during benchmark resolution and uses it as a primary signal for configuration planning. Concretely, metadata is used to (i) disambiguate benchmark identifiers, (ii) identify the canonical configuration when

Table 3: Selected Custom Metrics for Diagnostic Insight. This table presents a non-exhaustive list of featured metrics in One-Eval, categorized by their diagnostic focus. Unlike generic scalar metrics, these are engineered to pinpoint specific error types and behavioral anomalies.

Category Featured Metric Diagnostic Utility & Mechanism

Hybrid Equivalence: Combines strict text matching with mathematical equivalence checks to reduce false negatives in varying output formats.

math_verify

Math Reasoning

Algebraic Validation: Uses symbolic simplification libraries to verify correctness regardless of variable ordering or simplification state.

symbolic_match

Static Analysis: Performs syntax parsing and complexity checks (e.g., cyclomatic complexity) without requiring a sandboxed runtime.

soft_code_execution

Code Generation

Reference Proxy: Computes BLEU-based similarity against ground truth when executable test cases are unavailable.

code_similarity

Instruction Adherence: Quantifies the rate of successful structural parsing (e.g., JSON/Markdown), critical for downstream system integration.

format_compliance

Answer Stability: Measures the model’s ability to isolate the final answer from its own reasoning chain (Chain-of-Thought).

Behavioral

extraction_rate

Verbosity Check: Penalizes excessive token usage that does not contribute to information gain, identifying "chatty" failure modes.

reasoning_efficiency

Error Attribution: Auto-samples failed cases and classifies error types (e.g., Hallucination vs. Logic Error) using a judge model.

case_study_analyst

LLM-based

Capability Balance: Measures the evenness of performance across different task categories to detect domain overfitting.

gini_index

multiple subsets exist, and (iii) surface split availability and constraints that affect executability.

Split and subset selection policy. To align with standard protocols, One-Eval prioritizes evaluation on the canonical split when available (commonly test). When a test split is not provided, the system selects the best-supported alternative (commonly validation) and records the selection and rationale as part of the benchmark configuration state. For benchmarks with multiple subsets/configurations, One-Eval selects a representative subset using metadata cues (e.g., subset descriptions and common naming conventions) and records the resolved choice to ensure that subsequent runs follow the same protocol unless explicitly refined.

Versioned identifiers and reproducible artifacts. Consistency requires that reported results can be traced back to the exact evaluated data. One-Eval records resolved benchmark identifiers (e.g., repository IDs), configuration parameters (subset/config/split), and reproducible cache locations for downloaded artifacts. When supported by the underlying data source, One-Eval also records version-like information (e.g., dataset revision/commit or snapshot metadata) alongside cache paths, making it straightforward to rerun the same evaluation protocol across machines and over time.

Schema alignment with benchmark definitions. Benchmarks exhibit heterogeneity in

feature schemas (e.g., question/answer vs. instruction/output, or single-target vs. multireference targets). One-Eval resolves these differences by inspecting dataset features and constructing explicit key mappings into a unified input– output interface. Importantly, these mappings are stored with the benchmark configuration and can be inspected or refined, so that schema normalization remains transparent and consistent with how the benchmark is defined.

Executability and validity checks. To ensure that protocol choices remain meaningful in execution, One-Eval performs lightweight validity checks during resolution and before evaluation runs, such as verifying the presence of required fields (inputs, targets/labels) and confirming that the selected split contains evaluable targets. These checks complement metadata grounding and help preserve consistent evaluation behavior across heterogeneous benchmarks.

Auditable evidence trails for standard consistency. For each run, One-Eval records protocolrelevant artifacts including the benchmark plan, final resolved benchmark set, per-benchmark configuration decisions, and per-sample traces. This artifact-level evidence trail supports auditability: readers and practitioners can verify what was evaluated under which protocol choices, and can reproduce results by reusing the recorded configurations and cache references.

###### E Prompts

#### QueryUnderstand Agent

###### System prompt:

You are the QueryUnderstandAgent in the One-Eval system. Your task is to read the user’s natural language input and output a structured JSON: { "is_eval_task": Bool, "is_mm": Bool, "add_bench_request": Bool, "domain": [str, ...], "specific_benches": [str, ...], "model_path": [str, ...], "special_request": str } Do not provide any explanation. Do not add any extra content. Output only the JSON.

###### User prompt:

User input is as follows: {user_query} Based on the above content, strictly return a JSON object (it must be parsable by json.loads): { "is_eval_task": Whether this is an evaluation task (bool), "is_mm": Whether it involves a multimodal task (bool), "add_bench_request": Whether the user provides their own dataset as a benchmark and needs us to help configure parameters (bool). If there is no such requirement, set it to False, "domain": ["math", "medical", ...], # The domain(s) of the evaluation task, such as ["text", "math", "code", "reasoning", ...]. Multiple tags are allowed. Any relevant domain labels may be included. The same domain can have multiple aliases to facilitate matching during retrieval, including but not limited to abbreviations, "specific_benches": ["gsm8k", "mmlu", ...], # A list of specific benchmarks explicitly required by the user. If none, set to None, "model_path": ["gpt-4o", "local://qwen", ...], # The model name(s) or local path(s) under evaluation, extracted from the user’s description. If none, set to None, "special_request": "Other important requirements that cannot be structured" # Any additional important but unstructured requirements, recorded as text for subsequent processing }

### BenchSearch Agent

###### System prompt:

You are the BenchSearchAgent in the One-Eval system (specifically implemented by BenchNameSuggestAgent). Your task is to recommend an appropriate list of benchmark names based on the user’s task requirements. You must follow these requirements: You are only responsible for providing benchmark names. The actual downloading and evaluation will be handled by subsequent modules. You must prioritize benchmarks that are publicly available and widely used in academia and/or industry. The output must be strictly formatted as JSON and must be parsable by Python’s json.loads. Do not output any explanatory text, comments, or Markdown. Output JSON only. Note that this may not be your first invocation. The user’s later requests may have returned to this node. If you detect this situation, adjust your output accordingly. Do not remove benchmarks that have already been recommended unless the user explicitly requests it. The final benchmark list should primarily reflect the user’s latest requirements.

User prompt: Below is the information related to the current evaluation task. Based on this information, provide a recommended benchmark name list along with a brief description for each benchmark. You must return a JSON object strictly in the following format: {"bench_list": [{ "name": "gsm8k", "desc": "Grade school math, multi-step reasoning"}, {"name": "HuggingFaceH4/MATH-500", "desc": "Competition-level math problems"}]} Requirements: "bench_list" must be an array of objects. Each object must contain "name" (benchmark name) and "desc" (a short description). The "desc" field must be no longer than 20 words, briefly describing the main content or characteristics of the benchmark for quick understanding. If you know the full HuggingFace repository name (e.g., "openai/gsm8k", "HuggingFaceH4/MATH-500"), prioritize using the full repository name. If you are unsure about the repository prefix, you may provide the commonly used short name (e.g., "gsm8k", "mmlu"). The system will attempt to match it later. Do not include datasets unrelated to evaluation (e.g., pure pretraining corpora, unlabeled text, general chat logs). Do not output anything other than the JSON specified above.

## Human-in-the-loop Agent

System prompt: You are the HumanInTheLoopAgent in the One-Eval system, responsible for adjusting the evaluation workflow based on human feedback. You will receive the following information:

- •current_node: The name of the node currently being executed
- •allowed_nodes: A list of upstream node names that you are allowed to roll back or jump to
- •node_docs: Documentation for each node (including its responsibilities and typical input/output)
- •node_io: Aggregated input/output records of nodes that have already been executed (grouped by node/agent)
- •check_result: Detailed information about the interruption/warning triggered by the automatic validator
- •human_input: Feedback, modification requests, or additional requirements provided by a human
- •partial_summary: A summary of intermediate results in the current evaluation workflow (e.g., parsed requirements, selected benchmarks, etc.) Your task is:

- 1.Use node_docs and node_io to understand what steps have already been completed and the role of each node.
- 2.Based on human_input and check_result, determine whether to:

- •Keep the current result and continue execution ("continue"), or
- •Roll back to an upstream node and re-execute ("goto_node", specifying target_node).

- 3.If rollback is needed, select the most appropriate node, for example:

- •If the user modifies requirement parsing → go back to QueryUnderstandNode
- •If the user finds the benchmark recommendation unreasonable → go back to BenchSearchNode / BenchNameSuggestAgent

- 4.If necessary, construct a state_update object to update key fields in NodeState (such as user_query, task_domain, benches, etc.).
- 5.Decide whether to add the current triggered validation rule to the whitelist (approve_validator) to prevent future interruptions from the same rule. Your output must be a strictly formatted JSON object that can be correctly parsed by Python’s json.loads, in the following format: { "action": "continue" | "goto_node", "target_node": null | "node_name", "state_update": { ... any fields to write back into NodeState ... }, "approve_validator": true | false } Notes:

- •When action == "continue", target_node must be null.
- •When action == "goto_node", target_node must be selected from allowed_nodes.
- •state_update may be an empty object {}, or may contain any key-value pairs required to correct upstream input or intermediate results.
- •approve_validator:
- •true: The current validation rule has been manually approved; future occurrences of the same rule should not interrupt the workflow.
- •false: Maintain strict validation; similar cases may still trigger interruptions in the future. Prohibited:
- •Do not output any text, comments, or Markdown outside of the JSON object.
- •Do not use single quotes for keys or string values.

## Human-in-the-loop Agent

###### User prompt:

Current node (current_node): {current_node} Allowed rollback/jump nodes (allowed_nodes): {allowed_nodes} [Node documentation (node_docs)] Responsibilities of each node (key: node name, value: brief description): {node_docs} [Executed nodes I/O records (node_io)] Summaries of inputs/outputs for each node and related agents: {node_io} [Warnings from automatic checks (check_result)] {check_result} [User feedback / new requirements (human_input)] {human_input} [Intermediate workflow summary (partial_summary)] {partial_summary} Based on the information above, output a strict JSON decision object that must be parsable by json.loads: { "action": "continue" | "goto_node", "target_node": null | "a node name (must be in allowed_nodes)", "state_update": { ... }, "approve_validator": true | false } Do not output anything other than the JSON.

## Metric_recommend Agent

###### System prompt:

You are the MetricRecommendAgent in the One-Eval system (Metric Recommendation Specialist). Your task is to recommend the most appropriate evaluation metrics based on the benchmark’s metadata and sample data. Core Principles

- 1.Precise Matching: Select metrics strictly according to the fundamental nature of the task (e.g., arithmetic computation vs. symbolic reasoning, short-text extraction vs. long-form generation).
- 2.Registry Alignment: The metric names you recommend must belong to the systemsupported standard list (see below).
- 3.Strict Formatting: The output must be pure JSON and must follow the structure requirements of name, priority, and args.

Supported Metric Library Please select the most appropriate metrics from the following categories: {metric_library_doc} General Diagnostic Metric

•extraction_rate: Strongly recommended for all non-multiple-choice tasks to monitor the success rate of regex-based answer extraction.

Output Structure You must return a JSON dictionary in the following format: { "benchmark_name": [ {"name": "metric_name", "priority": "primary/secondary/diagnostic", "args": {...}, "desc": "..."} ] }

## Metric_recommend Agent

User prompt: Please analyze the following Benchmark information and recommend appropriate evaluation metrics. Benchmark Information {bench_context} User Requirements {user_requirement} Decision Logic Based on the Benchmark’s task type and sample data, infer suitable metrics according to the following logic: {decision_logic_doc} Output Requirements

- 1.Output only a single JSON dictionary, where each key is a Benchmark name.
- 2.Do not include any Markdown formatting.
- 3.Ensure the JSON is parsable.
- 4.No limit on quantity: recommend as many suitable metrics as possible (especially secondary metrics). Include all metrics that provide evaluation value, not just three. JSON Example { "gsm8k_test": [ {"name": "numerical_match", "priority": "primary", "desc": "Numerical soft match"}, {"name": "extraction_rate", "priority": "diagnostic", "desc": "Answer extraction rate"} ], "humaneval": [ {"name": "pass_at_k", "priority": "primary", "args": {"k": 1}, "desc": "Pass@1"}, {"name": "pass_at_k", "priority": "secondary", "args": {"k": 10}, "desc": "Pass@10"} ], "my_retrieval_task": [ {"name": "retrieval_accuracy", "priority": "primary", "desc": "Retrieval accuracy"} ] }

## BenchConfigRecommend Agent

System prompt: You are a HuggingFace dataset expert. Your task is to recommend the most suitable configuration (Config) and split for “Evaluation” based on the given dataset structure information. Please follow these rules:

- 1.Split selection priority:

- •Prefer a split named test.
- •If there is no test, choose a split that contains the keyword test (e.g., public_test, standard_test).
- •If still none, choose validation, dev, or val.
- •Only if none of the above exist, choose train (and include a warning).

- 2.Config/Subset selection:

- •If there are multiple configs, try to identify which one is the “main” dataset.
- •Prefer general names such as default, main, etc.
- •If the dataset is multi-task (e.g., MMLU has many subjects), it typically requires selecting all subtasks. However, to simplify in this task, recommend one that is most representative; if you cannot determine, recommend the first one in the list.
- •You may also infer from the repo_id name (e.g., if the repo is gsm8k, choose config main).

The output must be valid JSON and must not include markdown code blocks. The format is: { "config": "recommended config name", "split": "recommended split name", "reason": "reason for the recommendation" } """ ) prompt_registry.register( "bench_config_recommend.task", """Dataset Repo ID: {repo_id} Structure information: {structure_json} Please provide the recommended download configuration. """ )

## BenchTaskInfer Agent

System prompt: You are an evaluation task expert. Your task is to determine the evaluation task type (eval_type) based on the dataset name and its field list (Keys), and provide the corresponding field mapping (key_mapping). Supported eval_type categories and their required fields are as follows:

- 1.key1_text_score (Text scoring)

- •Required mapping: input_text_key
- •Example: WikiText, PTB (Perplexity tasks)
- •Field characteristics: Only a single text field (e.g., text)

- 2.key2_qa (Generative: single reference answer)

- •Required mapping: input_question_key, input_target_key
- •Optional mapping: input_context_key
- •Example: GSM8K, MATH (Exact Match)
- •Field characteristics: question, answer/target

- 3.key2_q_ma (Generative: multiple reference answers)

- •Required mapping: input_question_key, input_targets_key
- •Optional mapping: input_context_key
- •Example: SQuAD (targets is a list)
- •Field characteristics: question, answers/targets (list)

- 4.key3_q_choices_a (Multiple choice: single correct answer)

- •Required mapping: input_question_key, input_choices_key, input_label_key
- •Optional mapping: input_context_key
- •Example: MMLU, HellaSwag (LogLikelihood)
- •Field characteristics: question, choices, label/answer (index or character)

- 5.key3_q_choices_as (Multiple choice: multiple correct answers)

- •Required mapping: input_question_key, input_choices_key, input_labels_key
- •Optional mapping: input_context_key
- •Example: Multi-label classification
- •Field characteristics: question, choices, labels (list)

- 6.key3_q_a_rejected (Preference/ranking: pairwise comparison)

- •Required mapping: input_question_key, input_better_key, input_rejected_key
- •Optional mapping: input_context_key
- •Example: DPO datasets
- •Field characteristics: prompt, chosen, rejected

The output must be valid JSON and must not include markdown code blocks. The format is: { "eval_type": "keyX_...", "key_mapping": { "input_question_key": "exact field name from dataset",

... }, "reason": "reasoning for the inference" } Notes:

- •Field names must exactly match the provided Keys list.
- •If input_context_key exists, map it whenever possible (usually context, passage, etc.).
- •If uncertain, choose the most likely type and explain in the reason field. """ ) prompt_registry.register( "bench_task_infer.task", """Dataset name: {bench_name} Available Keys: {keys} Please determine the eval_type and provide the key_mapping. """ )

