# arXiv:2507.01853v5[cs.CL]5Mar2026

[Figure 1]

## EKA-EVAL : An Evaluation Framework for Low-Resource Multilingual Large Language Models

#### Samridhi Raj Sinha⋆ *, Rajvee Sheth§ , Abhishek Upperwal†, Mayank Singh§ ⋆NMIMS, †Soket AI, §Indian Institute of Technology Gandhinagar, LINGO Research Group

Correspondence: singh.mayank@iitgn.ac.in

### Abstract

The rapid evolution of Large Language Models has underscored the need for evaluation frameworks that are globally applicable, flexible, and modular, and that support a wide range of tasks, model types, and linguistic settings. We introduce EKA-EVAL, a unified, end-to-end framework that combines a zero-code web interface and an interactive CLI to ensure broad accessibility. It integrates 55+ diverse benchmarks across nine evaluation categories, supports local and proprietary models, and provides 11 core capabilities through a modular, plug-and-play architecture. Designed for scalable, multilingual evaluation with support for low-resource multilingual languages, EKAEVAL is, to the best of our knowledge, the first suite to offer comprehensive coverage in a single platform. Comparisons against five existing baselines indicate improvements of at least 2x better on key usability measures, with the highest user satisfaction, faster setup times, and consistent benchmark reproducibility.

Website Eka-Eval Demo bit.ly/Eka-Eval Code github.com/lingo-iitgn/eka-eval

### 1 Introduction

Large Language Models (LLMs) growing capabilities continue to reshape NLP, enabling impressive generalization across diverse tasks, including instruction following, reasoning, summarization, translation, and tool use. With the advent of general-purpose foundation models such as GPT4 (OpenAI, 2023), Claude (Anthropic, 2024), Gemini (Anil et al., 2023), Deepseek (Liu et al., 2024), and Llama-3 (AI at Meta, 2024), the focus of research has increasingly shifted from building taskspecific models to systematically evaluating these powerful systems. Evaluation plays a critical role not only in measuring progress but in identifying

*Work done while interning at IIT Gandhinagar (SRIP).

[Figure 2]

Result Processing

Score computation + comparative stats Histogram & benchmark-wise bar charts. Exports: JSON, CSV

Benchmark Registry

###### Model Inference

[Figure 3]

Evaluation Engine

[Figure 4]

[Figure 5]

55+ Benchmarks (Multilingual + Global)

Distributed & async execution Prompt construction + batching

Supports both local + API-based models Automatic quantization & memory optimization

Categories: QA, MCQ, Reasoning, Code, Long-Context Custom datasets Standardized interface: loading, prompting, metrics

CLI-based interactive model selection

Metric aggregation, logging, reproducibility

| | | | |
|---|---|---|---|
|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>Hugg<br><br>[Figure 10]<br><br>| |ing Face Inferenc<br><br>|e layer|

Open-Source & API models Open source datasets

Distributed Inference

Figure 1: Architecture of EKA-EVAL: A modular framework combining model inference, benchmark registry, evaluation engine, and result processing with support for caching and distributed inference.

capabilities, exposing limitations, and informing deployment strategies.

An Alarming Need: Existing evaluation frameworks for LMs, such as HELM (Liang et al., 2022), lm-eval-harness (Gao et al., 2021), OpenCompass (OpenCompass Contributors, 2023), DeepEval (DeepEval Contributors, 2024), and UltraEval (He et al., 2024) predominantly target high-resource languages and rely on command-line interfaces with limited benchmark coverage (Watts et al., 2024). These frameworks create significant barriers for non-technical users who require graphical interfaces and researchers working with lowresource and multilingual settings. The challenge is especially pronounced in linguistically diverse regions globally, where numerous languages serve large populations; yet, existing evaluation frameworks lack comprehensive multilingual support and culturally grounded benchmarks, limiting rigorous assessment of language model capabilities across diverse linguistic settings (Pomerenke et al., 2025; Xuan et al., 2025; Vayani et al., 2025).

Motivation: Existing evaluation frameworks suffer from several limitations, such as (1) absence of user-friendly interfaces, requiring extensive

Core Flexibility Advanced Capabilities Usability & Specialization Framework

Custom Data

Custom Models

Custom Prompt

Quantization

Long Context

Tool Use

Dist. Infer.

Visual Analysis

Zero-Code UI

Interact. CLI

Low-Resource Multilingual

lm-eval-harness ✓ ✓ ✓ ✓ ✓ ✗ ✓ ✗ ✗ ✗ ✗ OpenCompass ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✗ ✗ HELM ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✗ ✗ ✗ OpenAI Evals ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ DeepEval ✓ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✗ ✓ ✗ FreeEval ✓ ✓ ✓ ✓ ✗ ✗ ✓ ✓ ✗ ✓ ✗ indic-eval ✗ ✓ ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ ✓ UltraEval ✓ ✓ ✓ ✓ ✗ ✗ ✓ ✓ ✗ ✗ ✗ GlotEval ✓ ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓

EKA-EVAL ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

- Table 1: Feature comparison with state-of-the-art frameworks. Takeaway: EKA-EVAL is the only framework to achieve full coverage across all eleven dimensions, uniquely combining a zero-code UI and low-resource multilingual support with advanced capabilities such as long-context and tool-use evaluation.

command-line expertise, broken-pipelines, and manual configuration (Gao et al., 2021; Liang et al., 2022), (2) restricted flexibility in benchmark selection and task customization (Wang et al., 2018; Srivastava et al., 2022), (3) insufficient task diversity, with fragmented support for advanced capabilities like long-context reasoning and tool use (Qin et al., 2023; Zhang et al., 2024), and (4) limited multilingual coverage, particularly for low-resource languages (Ahuja et al., 2023; Luo et al., 2025).

To address this, we propose EKA-EVAL, a uniquely scalable and extensible evaluation framework that unifies multilingual benchmarks under flexible task configurations. It enables seamless model integration and multi-platform accessibility, establishing a new benchmark for low-resource multilingual LLM evaluation frameworks.

Contributions: The key contributions are:

- • We propose EKA-EVAL, a unified and modular evaluation framework that integrates over 55 multilingual benchmarks across nine major evaluation categories.
- • We introduce an open-source LLM evaluation framework that unifies a zero-code UI, an interactive CLI, and built-in support for low-resource multilingual settings, along with visualisations, and leaderboard, making large-scale evaluation accessible to both nontechnical and technical users.
- • Through a comprehensive human evaluation involving eleven participants comparing five existing frameworks, showing that EKAEVAL achieves the highest participant ratings across all six features, performing at least 2×

better than existing frameworks in key usability and effectiveness.

### 2 Relevant Work

General-Purpose Frameworks: Early multi-task suites such as GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019) established foundational standards. More comprehensive frameworks, including HELM (Liang et al., 2022), BIG-Bench (Srivastava et al., 2022), and lm-eval-harness (Gao et al., 2021), introduced extensible evaluation pipelines but typically require substantial technical effort and provide limited multilingual capabilities. Modular ecosystems such as FreeEval (Yu et al., 2024) and OpenCompass (OpenCompass Contributors, 2023) offer advanced features like distributed inference. While OpenCompass covers extensive benchmarks and zero-code UI for limited models, custom evaluations still require CLI interaction or manual YAML configuration.

Specialized Capability Benchmarks: Several domain-specific benchmarks target advanced LLM abilities, such as tool use in ToolBench (Qin et al., 2023), agentic behavior in API-Bank (Li et al., 2023) and InfiniteBench (Zhang et al., 2024), longcontext reasoning in RULER (Hsieh et al., 2024), and mathematical and coding competencies in GSM8K (Cobbe et al., 2021) and HumanEval (Du et al., 2024). While powerful, these benchmarks are fragmented across independent repositories, requiring users to maintain multiple codebases to conduct a thorough evaluation.

Multilingual Evaluation: Multilingual suites such as XTREME (Hu et al., 2020) and MEGA (Ahuja

et al., 2023) cover 30+ languages but often rely on machine-translated content that lacks cultural grounding. Recent efforts of MCEval (Huang et al., 2025) (13 cultures, 39K dynamic instances), GlotEval (Luo et al., 2025) (language-agnostic, culturally adaptive prompts), BLEnD (Myung et al., 2024) (16 cultures, 52K native QA pairs), CulturalBench (Chiu et al., 2024) (45 global regions), and ALM-Bench (Vayani et al., 2025) (100 languages with cultural images) address this gap. However, these benchmarks still face insufficient data for lowresource languages, cultural biases favor dominant dialects, and evaluation metrics struggle to capture cultural appropriateness.

Existing frameworks are often complex and poorly modularized, hindering seamless incorporation into research workflows (He et al., 2024). Furthermore, most toolkits also remain disproportionately focused on English and a handful of high-resource languages, making non-Englishcentric evaluations less practical for many researchers (Luo et al., 2025), and none offer a zerocode UI or native support for low-resource languages, creating a gap for practitioners evaluating models in low-resource settings without dedicated systems expertise.

EKA-EVAL addresses these gaps by unifying 55+ benchmarks across nine evaluation categories within a single modular workflow, with a strong focus on low-resource multilingual evaluation. It is, to our knowledge, the only framework to pair a zero-code web UI along with interactive CLI, easy-to-use framework making rigorous evaluation accessible to non-technical users.

### 3 Capabilities of LLM Evaluation Frameworks

We identify eleven foundational capabilities critical for modern LLM evaluation frameworks, informed by established systems like lm-eval-harness (Gao et al., 2021), OpenCompass (OpenCompass Contributors, 2023), FreeEval (Yu et al., 2024), and DeepEval (DeepEval Contributors, 2024). These capabilities are organized into three categories:

##### A. Core Flexibility:

- (1) Custom Datasets: Support for user-defined evaluation data beyond standard benchmarks.
- (2) Custom Models: Compatibility with local checkpoints and API-hosted endpoints.
- (3) Custom Prompting: Flexible templates sup-

porting zero-shot, few-shot, and chain-of-thought configurations.

- (4) Quantization: Support for memory-efficient quantized formats (8-bit, 4-bit weights).

- B. Advanced Capabilities:

- (5) Long Context: Processing extended inputs exceeding 4,000 tokens.
- (6) Tool Use: Evaluating agent-like behavior including API calls and multi-step reasoning.
- (7) Distributed Inference: Parallelized evaluation across multiple compute nodes.
- (8) Visual Analysis: Generation of interpretable visualizations (bar charts, plots, heatmaps, etc.).

- C. Usability & Specialization:

- (9) Zero-Code UI: Web-based interface enabling end-to-end evaluation workflows without coding.
- (10) Interactive CLI: Command-line interface for configuring datasets, models, and strategies.
- (11) Low-Resource Multilingual: Comprehensive multilingual benchmarks featuring languagespecific tokenization.

Table 1 demonstrates that EKA-EVAL achieves complete technical, linguistic, and usability coverage. While lm-eval-harness and HELM offer strong capabilities, they demand substantial programming expertise. OpenCompass includes a zerocode UI, but restricts it to leaderboard models and pre-configured benchmarks. OpenAI Evals and DeepEval support custom pipelines but lack quantization, long-context, and multilingual capabilities. Similarly, UltraEval and GlotEval provide provide robust customization and visual analysis (with the latter targeting low-resource languages) but but lack tool-use, long-context evaluation, and a zerocode UI. indic-eval is limited to seven benchmarks and offers no general-purpose capabilities. Critically, no existing framework pairs a zero-code UI with native low-resource multilingual support; EKA-EVAL addresses both, lowering barriers to rigorous multilingual assessment.

4 EKA-EVAL

##### 4.1 Design and Implementation

EKA-EVAL is architected as a modular, extensible evaluation framework that balances practical usability with comprehensive benchmark coverage. It provides a zero-code, web-based interface that allows users to run benchmarks, adjust parameters, and visualize results through interactive analytics and leaderboards, while a CLI enables detailed con-

figuration and large-scale evaluations. The framework supports flexible model selection for both local and API-based models and is built around three core principles: modularity for seamless extension and customization; accessibility to support diverse environments, including low-resource language settings; and comprehensiveness to cover a wide range of capabilities and low-resource multilingual benchmarks, with particular attention to underserved areas such as long-context reasoning and tool use.

##### 4.2 System Architecture

The framework has a layered architecture with four main components (as illustrated in Figure 1):

- 4.2.1 Evaluation Engine Manages all evaluation workflows: Task Scheduler: Manages task scheduling, prompt formatting, and result aggregation across distributed inference setups. The scheduler implements intelligent work distribution, as shown in main_orchestrator(), by dynamically assigning evaluation tasks to available workers based on resource constraints and model requirements. Batch Optimizer: Implements intelligent batching strategies and supports various quantization schemes to optimize memory usage and inference speed. As seen in the PIQA evaluation implementation, the optimizer automatically adjusts generation_batch_size parameters to maximize throughput while preventing out-of-memory errors. Distributed Coordinator: Coordinates evaluation across multiple GPUs and workers using Python’s multiprocessing library, launching multiple worker_process instances to execute independent evaluation tasks in parallel across benchmarks and model configurations.
- 4.2.2 Benchmark Registry Provides a unified interface for managing datasets: Dataset Manager: The BenchmarkRegistry class handles diverse dataset formats and sources, abstracting the complexities of different evaluation protocols. The manager supports datasets from HuggingFace Hub, local files, and custom APImodels through standardized interfaces.
- 4.2.3 Model Interface Layer Access to different local and API-based models: Local Model Loader: Initializes transformerbased checkpoints with automatic device allocation and quantization.

API Client Manager: Manages proprietary endpoints through dedicated clients (OpenAIClient, GeminiClient, ClaudeClient) that extend BaseAPIClient, providing unified request handling with rate limiting and authentication.

Interactive Selection Interface: Implements get_model_selection_interface() for dynamic model discovery and selection, supporting local model paths and API configurations(See Appendix C).

Resource Manager: Ensures efficient memory management via cleanup functions, preventing resource leaks during repeated evaluation runs.

##### 4.2.4 Results Processing System

Handles comprehensive output management through three secondary components:

Metrics Calculator: Employs HuggingFace’s industry-standard evaluate library to compute essential metrics such as accuracy, BLEU (Papineni et al., 2002), F1-score (Christen et al., 2023), exact match (Rajpurkar et al., 2016), and Pass@1 (Du et al., 2024), ensuring consistency with widely adopted benchmarks and standard evaluation practices across diverse LLM tasks.

Visualisations analytics: Provides comparative analysis across multiple models and benchmark configurations by generating visualizations such as bar charts, heatmaps, and radar plots (including support for cross-model comparisons).

Export Manager: Handles result export in multiple formats, including JSON and CSV. The manager maintains evaluation metadata including model parameters, benchmark versions, execution timestamps, and system configurations.

##### 4.2.5 Interface and Deployment Layer

Handles complex evaluation logic with ease of use through a streamlined full-stack user interface (See Appendix §B and Figure 4).

Full-Stack Architecture: A decoupled REACT (Fedosejev, 2015) frontend and FASTAPI (Tiangolo, 2018) backend, served through NGINX, provide a stable and scalable deployment layer.

Real-Time Telemetry: WebSocket-based streaming delivers live inference logs and GPU status directly to the UI, eliminating the need for CLI.

LLM-Powered Diagnostics: An LLM (LLaMA3.3 70B (Grattafiori et al., 2024)) interprets evaluation outputs and generates summaries of model performance and failure patterns, revealing additional insights.

###### Framework Setup & Config Navigation UI Result Export Extensibility Multilingual Support

lm-eval-harness 3.73 ± 0.72 3.91 ± 0.66 1.00 ± 0.00 3.64 ± 0.70 4.18 ± 1.00 1.64 ± 0.50 OpenCompass 2.18 ± 0.93 2.36 ± 0.67 3.09 ± 1.36 3.09 ± 0.93 2.80 ± 1.06 2.09 ± 0.71 HELM 1.91 ± 0.71 2.64 ± 1.05 2.36 ± 1.22 2.55 ± 1.42 2.09 ± 0.93 1.36 ± 0.50 indic-eval 2.55 ± 0.88 3.18 ± 0.67 1.00 ± 0.00 2.55 ± 1.00 2.55 ± 0.88 4.55 ± 0.53 FreeEval 2.55 ± 1.24 2.45 ± 0.71 2.73 ± 1.01 2.64 ± 1.24 2.36 ± 0.87 1.45 ± 0.53 EKA-EVAL 4.27 ± 0.93 4.55 ± 0.73 4.64 ± 0.53 4.55 ± 0.50 4.64 ± 0.50 4.73 ± 0.50

- Table 2: Average participant ratings of the evaluation frameworks by eleven participants (mean ± standard deviation; Likert scale: 1–5, 1 = poor and 5 = excellent). Takeaway: EKA-EVAL achieves the highest mean ratings across all six dimensions, with notably stronger scores on Zero-Code UI and Global Low-Resource Language Support.

Framework Time Taken

lm-eval-harness 22 ± 7.58 OpenCompass 36 ± 9.83 HELM 58 ± 5.70 IndicEval 32 ± 4.14 FreeEval 35 ± 7.07 EKA-EVAL 11 ± 3.18

- Table 3: Comparison of time taken for installation and configuration across six frameworks reported across five successful runs on GPU-setup. Takeaway: EKA-EVAL records the lowest setup time and variance.

SenseQA (Talmor et al., 2019), and OpenBookQA (Mihaylov et al., 2018).

- 5. World Knowledge: Factual knowledge is tested through TriviaQA (Joshi et al., 2017) and NaturalQuestions (Kwiatkowski et al., 2019).
- 6. Long-Context Understanding: For extended context, we include ZeroSCROLLS (Shaham et al., 2023), Needle-in-a-Haystack (Wang et al., 2025), and InfiniteBench (Zhang et al., 2024).
- 7. General Reasoning: Foundational capabilities are tested via MMLU (Hendrycks et al., 2020), MMLU-Pro (Wang et al., 2024), IFEval (Zhou et al., 2023), BBH (Suzgun et al., 2023), and AGI-Eval (Zhong et al., 2024).
- 8. Tool Use and API Reasoning: Practical capabilities are assessed through API-Bank (Li et al.,

- 2023) and API-Bench (Patil et al., 2024).

9. Multilingual and Low-Resource Language Support: LLMs are known to underperform on languages spoken across South Asia, Africa, and Southeast Asia, yet no existing evaluation framework provides sufficient coverage to diagnose this gap (Luo et al., 2025; Kakwani et al., 2020). EKA-EVAL features the largest unified Multilingual suite (23 benchmarks), including: (i) Knowledge: IndicMMLU-Pro (Sankalp et al., 2025), MMLU-IN (Hendrycks et al., 2021), TriviaQA-IN (Joshi et al., 2017), MILU (Verma et al., 2025) spanning Kannada, Odia, Malayalam, and Urdu. (ii) Reasoning: HellaSwagIN (Zellers et al., 2019), ARC-C-IN (Clark et al., 2018), IndicCOPA (Kakwani et al., 2020), XCOPA (Ponti et al., 2020) in Quechua, Indonesian, and Swahili, and GSM8K-IN (Cobbe et al., 2021) for reasoning across Indic languages. (iii) Reading & QA: Belebele (Bandarkar et al.,

- 2024) covering 122 languages including Yoruba, Oromo, and Indonesian; BoolQ-IN (Clark et al., 2019), XQuAD-IN (Artetxe et al., 2020) in Hindi and Greek, XorQA-IN (Asai et al., 2021) in Bengali and Telugu, and Indic-QA (Singh

Interactive Dashboard: A “Zero-Code” userinterface with visual comparisons and live leaderboards enables users to run and analyze evaluations without technical expertise.

##### 4.3 Comprehensive Benchmark Coverage

EKA-EVAL covers nine major evaluation categories with comprehensive benchmark support across 55+ benchmarks (See Appendix A). These categories include:

- 1. Code Generation and Programming: These are assessed using HumanEval (Du et al., 2024), MBPP (Austin et al., 2021), HumanEval+ (Liu et al., 2023), PythonSaga (Yadav et al., 2024) and MBPP EvalPlus (Liu et al., 2023) with Pass@1 metrics.
- 2. Mathematics and Logical Reasoning: Mathematical capabilities are evaluated through GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), and ARCChallenge (Clark et al., 2018).
- 3. Reading Comprehension: Text understanding is evaluated using SQuAD (Rajpurkar et al., 2018), QuAC (Choi et al., 2018), and BoolQ (Clark et al., 2019) with F1 and exact match metrics.
- 4. Commonsense Reasoning: We incorporate PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), ARCEasy/Challenge (Clark et al., 2018), WinoGrande (Sakaguchi et al., 2021), Common-

et al., 2025). (iv) Generation (NLG): FloresIN (Goyal et al., 2022) for translation across 200 languages; IndicParaphrase, IndicWikiBio, IndicQuestionGeneration, IndicSentenceSummarization, and IndicHeadlineGeneration (Singh et al., 2024) in Odia, Assamese and Punjabi. (v) NLU: IndicNER and IndicSentiment in 11 languages, IndicGLUE (Kakwani et al., 2020), and XNLI (Conneau et al., 2018) for cross-lingual inference in Arabic, Swahili, and Urdu.

### 5 Experiments

To assess the effectiveness and usability of EKAEVAL, we conducted a comprehensive evaluation combining benchmark coverage analysis and usercentered feedback across six prominent frameworks: lm-eval-harness, OpenCompass, HELM, FreeEval, indic-eval, and EKA-EVAL. These frameworks were selected as representative stateof-the-art baselines spanning across widely adopted general-purpose to specialized multilingual evaluation suites, ensuring a rigorous comparison.

Evaluation Procedure Eleven graduate-level computer science researchers were selected from NLP and software engineering programs. Participants were stratified by prior experience using at least one existing evaluation framework (5 with, 6 without) and evaluated each framework in a singleblind setup. Selection was not tied to knowledge of EKA-EVAL, mitigating demand bias; sample size was constrained by available GPU resources. Participants followed a standardized evaluation protocol in a controlled environment. They installed each framework, integrated HuggingFace models - including gemma-2-2b (Gemma et al., 2024) and sarvam-1B (Sarvam, 2024), and executed the recommended workflows on four diverse benchmarks: WinoGrande, PIQA, ARC-C-IN and MMLU-IN. All ratings were recorded using the standardized protocol (Appendix §D) to ensure direct and consistent comparison across frameworks. Participants assessed each framework across six evaluation criteria and rated them using a Likert scale ranging from 1 to 5:

- 1. Setup and Configuration Time: Time and effort required to install dependencies, configure models, and run an initial benchmark.
- 2. Ease of Navigation: Intuitiveness of navigation, benchmark selection, and configuration, including CLI clarity, documentation quality, and ease of discovering options.

- 3. Zero-Code UI Availability: Presence and usability of GUI-based tools for visualization, benchmark execution, and result analysis, eliminating the need for command-line interaction.
- 4. Result Reporting and Export: Clarity and accessibility of evaluation outputs, with options to export results (e.g., JSON, CSV) and create visualizations such as bar charts or heatmaps.
- 5. Extensibility: Ease of customizing the framework to add new prompt templates, models, benchmarks, or evaluation metrics.
- 6. Multilingual Language Support: Support for multilingual benchmarks such as Belebele, ARC-IN, FLORES, and XNLI.

Table 2 presents average participant ratings across six evaluation criteria using a 1-5 Likert scale (1= poor or absence of functionality, 5= excellent or flexible capability). EKA-EVAL achieved the highest ratings across all criteria, establishing superior usability and capability. Moreover, Table 3 summarizes the average setup times across six frameworks, which varied widely due to dependency issues and unstable codebases in several baselines. Nine of the eleven participants reported that HELM suffered from poor documentation and a broken pipeline, while OpenCompass required manual code edits and indic-eval experienced installation difficulties. EKA-EVAL achieved the shortest setup time. Table 4 (See Appendix 4 for full results) presents reproduced benchmark scores on gemma-2-2b, showing how they align with broader framework differences and highlighting EKA-EVAL’s consistency across global-multilingual benchmarks. HELM was excluded from this comparison, as it neither supports these benchmarks nor produced functional results despite repeated attempts.

### 6 Conclusion and Future Work

We present EKA-EVAL, an open-source unified evaluation framework designed to overcome persistent challenges in accessibility, reproducibility, and low-resource multilingual coverage for language model assessment. Our study with eleven participants demonstrates substantial usability improvements, achieving lowest setup time and the highest satisfaction ratings. EKA-EVAL enables an interactive UI, rigorous, reproducible evaluation without manual coding expertise. The current release is an ongoing work, with 30 stars and 3 forks on GitHub. Future work will expand coverage to multimodal LLMs, benchmarks, and AI diagnostic capabilities.

### 7 Limitations

Multimodal Support. EKA-EVAL currently supports text-only LLMs; multimodal model evaluation is left to future work.

Reproducibility. Evaluation results may be affected by changes in external dataset or model versions; we recommend explicit versioning and caching for fully reproducible runs.

User study scope. The usability study (n=11) constitutes a formative evaluation conducted under GPU resource constraints, providing initial usability insights and motivating broader future evaluations.

Inference Backend. EKA-EVAL currently relies on HuggingFace Transformers for model inference; vLLM backend integration is underway to enable faster, high-throughput evaluation at scale.

Scalability. The current release processes requests sequentially; supporting concurrent multiuser deployments may require additional infrastructure and compute beyond the scope of this work.

### 8 Ethics Statement

This research uses publicly available data without personally identifiable information. All datasets and models comply with their terms of use. The work is intended for academic research. Potential misuse or unintended amplification of biases should be carefully considered before deployment.

### 9 Acknowledgements

The authors express their gratitude to Himanshu Beniwal for helping with writing the manuscript and to Birudugadda Srivibhav, Sailesh Panda, Gautham Bharati, Indrayudh Mandal, Krudant Randai, Sahil Gawande, Tirth Bhatt, Shruti Bhat, Mann agarwal, Himanshu Sharma and Khushbu Bijawat for their contributions in evaluating the framework, reviewing the manuscript, and reporting the results. We also appreciate the valuable suggestions and feedback provided by Aamod Thakur, Prathamesh Shanbhag and Mahavir Patil.

### References

Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023.

MEGA: Multilingual evaluation of generative AI. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232–4267, Singapore. Association for Computational Linguistics.

AI at Meta. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Anthropic. 2024. Claude 3.5 sonnet model card addendum. https://www-cdn.anthropic.com/ fed9cc193a14b84131812372d8d5857f8f304c52/ Model_Card_Claude_3_Addendum.pdf. Addendum to the Claude 3 Model Card.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4623–4637.

Akari Asai, Jungo Kasai, Jonathan H Clark, Kenton Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2021. Xor qa: Cross-lingual open-retrieval question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 547–564.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1 others. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand. Association for Computational Linguistics.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, and 1 others. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439.

Yu Ying Chiu, Liwei Jiang, Bill Yuchen Lin, Chan Young Park, Shuyue Stella Li, Sahithya Ravi, Mehar Bhatia, Maria Antoniak, Yulia Tsvetkov, Vered Shwartz, and 1 others. 2024. Culturalbench: A robust, diverse, and challenging cultural benchmark by human-ai culturalteaming. arXiv preprint arXiv:2410.02677.

Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wentau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018. QuAC: Question answering in context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2174–2184, Brussels, Belgium. Association for Computational Linguistics.

Peter Christen, David J. Hand, and Nishadi Kirielle. 2023. A review of the f-measure: Its history, properties, criticism, and alternatives. ACM Comput. Surv., 56(3).

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. ArXiv, abs/2110.14168.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.

DeepEval Contributors. 2024. DeepEval. https:// github.com/confident-ai/deepeval.

Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen, Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. 2024. Evaluating large language models in class-level code generation. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, ICSE ’24, New York, NY, USA. Association for Computing Machinery.

Artem Fedosejev. 2015. React.js Essentials. Packt Publishing Ltd.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, and 1 others. 2021. A framework for few-shot language model evaluation. Version v0. 0.1. Sept, 10:8– 9.

Gemma, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, PengJen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Chaoqun He, Renjie Luo, Shengding Hu, Ranchi Zhao, Jie Zhou, Hanghao Wu, Jiajie Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2024. UltraEval: A lightweight platform for flexible and comprehensive evaluation for LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 247–257, Bangkok, Thailand. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In International conference on machine learning, pages 4411–4421. PMLR.

Shulin Huang, Linyi Yang, and Yue Zhang. 2025. Mceval: A dynamic framework for fair multilingual cultural evaluation of llms. ArXiv, abs/2507.09701.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Divyanshu Kakwani, Anoop Kunchukuttan, Satish Golla, Gokul NC, Avik Bhattacharyya, Mitesh M Khapra, and Pratyush Kumar. 2020. Indicnlpsuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for indian languages. In Findings of the association for computational linguistics: EMNLP 2020, pages 4948–4961.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. API-bank: A comprehensive benchmark for tool-augmented LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3102–3116, Singapore. Association for Computational Linguistics.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, and 1 others. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558–21572.

Hengyu Luo, Zihao Li, Joseph Attieh, Sawal Devkota, Ona de Gibert, Xu Huang, Shaoxiong Ji, Peiqin Lin, Bhavani Sai Praneeth Varma Mantina, Ananda Sreenidhi, Raúl Vázquez, Mengjie Wang, Samea Yusofi, Fei Yuan, and Jörg Tiedemann. 2025. GlotEval: A test suite for massively multilingual evaluation of large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 602–614, Suzhou, China. Association for Computational Linguistics.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

Junho Myung, Nayeon Lee, Yi Zhou, Jiho Jin, Rifki Putri, Dimosthenis Antypas, Hsuvas Borkakoty, Eunsu

Kim, Carla Perez-Almendros, Abinew Ali Ayele, and 1 others. 2024. Blend: A benchmark for llms on everyday knowledge in diverse cultures and languages. Advances in Neural Information Processing Systems, 37:78104–78146.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

OpenCompass Contributors. 2023. OpenCompass: A Universal Evaluation Platform for Foundation Models. https://github.com/open-compass/ OpenCompass. Accessed: 2025-07-01.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2024. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565.

David Pomerenke, Jonas Nothnagel, and Simon Ostermann. 2025. The ai language proficiency monitor– tracking the progress of llms on multilingual benchmarks. arXiv preprint arXiv:2507.08538.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vuli´c, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362–2376, Online. Association for Computational Linguistics.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia. Association for Computational Linguistics.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.

KJ Sankalp, Ashutosh Kumar, Laxmaan Balaji, Nikunj Kotecha, Vinija Jain, Aman Chadha, and Sreyoshi Bhaduri. 2025. Indicmmlu-pro: Benchmarking indic large language models on multi-task language understanding. arXiv preprint arXiv:2501.15747.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. 2019. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728.

Sarvam. 2024. sarvamai/sarvam-1. https:// huggingface.co/sarvamai/sarvam-1. Accessed: 2025-07-04.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. Zeroscrolls: A zeroshot benchmark for long text understanding. arXiv preprint arXiv:2305.14196.

Abhishek Kumar Singh, Vishwajeet kumar, Rudra Murthy, Jaydeep Sen, Ashish Mittal, and Ganesh Ramakrishnan. 2025. Indic qa benchmark: A multilingual benchmark to evaluate question answering capability of llms for indic languages. Preprint, arXiv:2407.13522.

Harman Singh, Nitish Gupta, Shikhar Bharadwaj, Dinesh Tewari, and Partha Talukdar. 2024. IndicGenBench: A multilingual benchmark to evaluate generation capabilities of LLMs on Indic languages. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11047–11073, Bangkok, Thailand. Association for Computational Linguistics.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, and 1 others. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada. Association for Computational Linguistics.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Sebastian Tiangolo. 2018. Fastapi framework, high performance, easy to learn, fast to code, ready for production.

Ashmal Vayani, Dinura Dissanayake, Hasindri Watawana, Noor Ahsan, Nevasini Sasikumar, Omkar Thawakar, Henok Biadglign Ademtew, Yahya Hmaiti, Amandeep Kumar, Kartik Kukreja, and 1 others. 2025. All languages matter: Evaluating lmms on culturally diverse 100 languages. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19565–19575.

Sshubam Verma, Mohammed Safi Ur Rahman Khan, Vishwajeet Kumar, Rudra Murthy, and Jaydeep Sen. 2025. MILU: A multi-task Indic language understanding benchmark. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10076–10132, Albuquerque, New Mexico. Association for Computational Linguistics.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. 2025. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3221–3241, Albuquerque, New Mexico. Association for Computational Linguistics.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Ishaan Watts, Varun Gumma, Aditya Yadavalli, Vivek Seshadri, Manohar Swaminathan, and Sunayana Sitaram. 2024. Pariksha: A scalable, democratic, transparent evaluation platform for assessing indic large language models. Technical report, Microsoft Research.

Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Aosong Feng, Dairui Liu, Yun Xing, Junjue Wang, Fan Gao, Jinghui Lu, Yuang Jiang, Huitao

Li, Xin Li, Kunyu Yu, Ruihai Dong, Shangding Gu, Yuekang Li, Xiaofei Xie, and 13 others. 2025. MMLU-ProX: A multilingual benchmark for advanced large language model evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1513–1532, Suzhou, China. Association for Computational Linguistics.

Ankit Yadav, Himanshu Beniwal, and Mayank Singh. 2024. PythonSaga: Redefining the benchmark to evaluate code generating LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 17113–17126, Miami, Florida, USA. Association for Computational Linguistics.

Zhuohao Yu, Chang Gao, Wenjin Yao, Yidong Wang, Zhengran Zeng, Wei Ye, Jindong Wang, Yue Zhang, and Shikun Zhang. 2024. FreeEval: A modular framework for trustworthy and efficient evaluation of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 1–13, Miami, Florida, USA. Association for Computational Linguistics.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024. ∞Bench: Extending long context evaluation beyond 100K tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262– 15277, Bangkok, Thailand. Association for Computational Linguistics.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2024. AGIEval: A human-centric benchmark for evaluating foundation models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2299–2314, Mexico City, Mexico. Association for Computational Linguistics.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

### A Appendix

##### A.1 Multilingual benchmark configuration

As demonstrated in Figure 2, a sample configuration used to evaluate the ARC-Challenge-Indic benchmark across 11 Multilingual languages. It illustrates how task parameters, templates, and dataset references are modularly specified in EKAEVAL.

"ARC-Challenge-Indic": { "description": "Zero-shot evaluation

across 11 Indic languages", "evaluation_function":

"indic.arc_c_in.evaluate_arc_c_in", "task_args": {

"dataset_name": "sarvamai/arc-challenge-indic",

"target_languages": ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],

"dataset_split": "validation", "num_few_shot": 0, "max_new_tokens": 10, "generation_batch_size": 8, "prompt_template_name_zeroshot":

"arc_c_in_0shot", "prompt_template_name_fewshot": "arc_c_in_5shot", "prompt_file_benchmark_key": "arc_c_in",

"prompt_file_category": "indic" }

}

Figure 2: ARC-Challenge-Indic benchmark configuration example

##### A.2 Prompt Template System

- A critical component of EKA-EVAL is its sophisticated prompt management system, which handles diverse evaluation paradigms and languages. The framework implements a flexible template system demonstrated through PIQA benchmark prompt 3: The Prompt template system as shown in Figure 3 supports zero-shot, few-shot, and chain-of-thought prompting strategies, ensuring consistency across evaluation modes and languages. Users can customize prompt strategies and easily configure them in the benchmark configuration file, as shown in Figure 2.
- B Zero-code User Interface

To democratize access for LLM evaluation, EKAEVAL includes a “Zero-code” user interface that mirrors the full capabilities of the CLI while adding

###### Framework Winogrande PIQA ARC-IN MMLU-IN

lm-eval-harness 65.8 ± 6.0 74.8 ± 3.5 – – OpenCompass 61.1 ± 5.0 73.3 ± 4.3 – – indic-eval 65.8 ± 1.9 78.7 ± 1.4 28.1 ± 2.3 35.0 ± 2.1 FreeEval 60.0 ± 4.5 54.5 ± 3.2 – – EKA-EVAL 66.5 ± 2.5 77.5 ± 1.5 31.5 ± 3.5 33.5 ± 0.5

Table 4: Benchmark Score Parity. Comparison of benchmark scores (mean ± standard deviation). ‘–’ represents those benchmarks are not supported for the frameworks. Takeaway: EKA-EVAL reproduces verified consistent scores with established frameworks and closely aligns with their actual benchmark results.

{

"piqa_generation": {

"template": "Choose the most appropriate solution (0 or 1) to achieve the goal: \n\nQuestion: {goal}\n0) {sol1} \n1) {sol2}\nAnswer:",

"description": "Generation-based PIQA prompt"

}, "piqa_5shot_generation": {

"template_prefix": "Choose the most "few_shot_example_template":

"Question: {goal}\n0) {sol1} \n1) {sol2}\nAnswer: {answer_label}",

"few_shot_separator": "\n\n", "template_suffix": "Question: {goal}

\n0) {sol1}\n1) {sol2}\nAnswer:", "description": "Few-shot generation prompt"

}, "default_few_shot_examples_piqa": [

{

"goal": "To remove a stain from clothing", "sol1": "Apply cold water immediately...", "sol2": "Set the clothing on fire...", "answer_label": "0"

} ]

}

Figure 3: PIQA prompt templates supporting multiple evaluation paradigms

advanced features for prompt configuration, resource management, and AI-assisted analysis, making rigorous evaluation accessible to both nontechnical and technical users. Figure 4 illustrates the end-to-end architecture, spanning user input, model inference, evaluation execution, and result visualization across EKA-EVAL’s four core components.

The Benchmark Selection Dashboard (Figure 8) allows users to build their own evaluation suite by simply toggling and selecting the benchmarks available and also adding custom benchmarks.

For advanced control over the evaluation process, Configuration Panel (as shown in Figure 9) offers adjustable settings such as temperature, top-p and batch size. It also includes a built-in GPU resource manager, allowing users to allocate specific

[Figure 11]

Figure 4: The overall architectural pipeline of the EKAEVAL framework.

or multi-GPU support for evaluation.

A key advantage of the web user interface is the Prompt Customization Engine (Figure 10). It lets users visually edit system prompts, few-shot templates, and variable fields, GPU allocation, thus removing the need to manually modify JSON configuration files.

The Evaluation Dashboard Figure 11 shows real time progress during inference including a live console and, once finished, it provides a summary of the model’s performance with options to download logs or explore detailed analysis.

AI Diagnosis Dashboard uses LLama 3.3-70b model to analyze the evaluation logs, generating summaries of the model’s observed strengths (e.g., ‘Strong logical coherence’) and weaknesses (e.g., ‘Hallucination in low-resource languages’) as example shown in Figure 12.

Users can also inspect specific failure cases (Figure 13). Selecting a failed benchmark item shows the prompt, the model’s wrong answer, and an AIgenerated explanation of why the error likely occurred, making it easy to spot and fix issues.

Finally, the Model Leaderboard Figure 14, Figure 15 and Figure 16 provides a centralized, dynamic interface for comparative analysis. Unlike static evaluation tables, it aggregates results from all local runs securely. Researchers can filter outputs by model size, language, or specific benchmark, and generate side-by-side multi-dimensional radar charts. To maintain integrity, the leaderboard includes version tracking for datasets and prevents score manipulation by linking directly to the immutable evaluation logs generated by the backend engine.

### C Interactive CLI

##### C.1 CLI Demonstration

The interactive CLI of the EKA-EVAL framework is shown below, which guides users through model selection and evaluation setup. Simplifying benchmarking workflows, it is accessible to both researchers and developers.

As per Figure 17, users are prompted to select highlevel task groups (e.g., Reading Comprehension) during CLI setup. This enables fine-grained benchmarking organization and streamlined selection.

After selecting a task group, users choose specific benchmarks such as SQuAD, BoolQ, or QuAC for

focused evaluation within that domain (see Figure 20). The CLI displays final benchmark scores for each model in tabular format, including per-task and average scores. Results are also exported as CSV (see Figure 22). Figure 21 shows performance breakdown across sub-tasks like BoolQ, SQuAD, and QuAC. It provides intuitive insight into strengths and weaknesses of the model.

### D User Study Protocol

Participants received a standardized evaluation protocol and rating form (Figures 5 and 6) to ensure consistency across all framework evaluations. Each participant was permitted to consult official documentation but was prohibited from seeking external technical assistance beyond publicly available framework resources. Explicit guidelines specified permitted and prohibited actions to minimize evaluation bias.

### E Extensibility and Customization

EKA-EVAL emphasizes extensibility through a lowcode plugin architecture that enables easy integration of new benchmarks and custom metrics with minimal modifications, managed via a hierarchical JSON configuration system that supports complex setups such as parameter sweeps and prompt variations. The framework provides a zero-code UI that enables real-time, interactive adjustment of inference parameters (e.g., temperature, batch size) and evaluation strategies, eliminating the need for users to edit the configuration files directly.

### F Computation Requirement and Budget

For computational infrastructure, experiments were carried out on four NVIDIA Tesla V100 32 GB GPUs, with an estimated cost of $5,431.20 per month based on Google Cloud Platform (GCP) 1 Calculator pricing.

1https://cloud.google.com/products/calculator

[Figure 12]

###### Figure 5: Standardized study protocol specifying permitted and prohibited actions for all participants.

[Figure 13]

###### Figure 6: Participants assessment form.

[Figure 14]

Figure 7: ModelHub selection panel to choose any HuggingFace/Local models for benchmark evaluation.

[Figure 15]

- Figure 8: Benchmark selection dashboard categorized by domain (e.g., Math, Code Generation, and other benchmarks.

[Figure 16]

###### Figure 9: Advanced configuration panel for adjusting inference parameters (temperature, batch size) and managing GPU resources.

[Figure 17]

###### Figure 10: The prompt customization interface, allowing users to modify templates and injection points for specific tasks.

[Figure 18]

###### Figure 11: Post-evaluation summary showing overall progress, specific benchmark scores, and options to export results or view detailed analysis.

[Figure 19]

###### Figure 12: AI diagnosis dashboard providing automated textual insights into model strengths and weaknesses based on performance.

[Figure 20]

###### Figure 13: AI diagnosis showing the wrong result of selected benchmark

[Figure 21]

###### Figure 14: Model Leaderboard through aggregated average scores across multiple benchmarks and through live evaluation

[Figure 22]

###### Figure 15: Model Leaderboard featuring multi-dimensional visualizations (radar and bar charts) for comparative analysis.

[Figure 23]

- Figure 16: Model Key strengths and weaknesses featuring multi-dimensional visualizations along with benchmarks performance analysis for improvement.

[Figure 24]

Figure 17: Available benchmark groups of the EKA-EVAL framework.

[Figure 25]

Figure 18: Model selection in the EKA-EVAL framework.

[Figure 26]

Figure 19: Interactive and user-friendly visualisation setup in EKA-EVAL.

[Figure 27]

Figure 20: Subtask selection within a task group.

[Figure 28]

Figure 21: Bar chart visualisation on Reading Comprehension benchmarks

[Figure 29]

Figure 22: Consolidated evaluation results table.

