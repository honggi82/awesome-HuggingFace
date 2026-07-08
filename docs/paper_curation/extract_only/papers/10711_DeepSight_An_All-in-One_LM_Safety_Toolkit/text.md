# arXiv:2602.12092v1[cs.CL]12Feb2026

[Figure 1]

-:=-.==.. ShanghaiArtificialIntelligenceLaboratory DeepSight

上海人工智能实验室

## DeepSight: An All-in-One LM Safety Toolkit

Bo Zhang∗, Jiaxuan Guo∗, Lijun Li∗, Dongrui Liu∗, Sujin Chen, Guanxu Chen, Zhijie Zheng, Qihao Lin, Lewen Yan, Chen Qian, Yijin Zhou, Yuyao Wu, Shaoxiong Guo,

Tianyi Du, Jingyi Yang, Xuhao Hu, Ziqi Miao, Xiaoya Lu, Jing Shao , Xia Hu

Shanghai AI Laboratory

github.com/AI45Lab/DeepSafe | github.com/AI45Lab/DeepScan

### Abstract

As the development of Large Models (LMs) progresses rapidly, their safety is also a priority. In current Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) safety workflow, evaluation, diagnosis, and alignment are often handled by separate tools. Specifically, safety evaluation can only locate external behavioral risks but cannot figure out internal root causes. Meanwhile, safety diagnosis often drifts from concrete risk scenarios and remains at the explainable level. In this way, safety alignment lack dedicated explanations of changes in internal mechanisms, potentially degrading general capabilities. To systematically address these issues, we propose an open-source project, namely DeepSight, to practice a new safety evaluation-diagnosis integrated paradigm. DeepSight is low-cost, reproducible, efficient, and highly scalable large-scale model safety evaluation project consisting of a evaluation toolkit DeepSafe and a diagnosis toolkit DeepScan. By unifying task and data protocols, we build a connection between the two stages and transform safety evaluation from black-box to white-box insight. Besides, DeepSight is the first open source toolkit that support the frontier AI risk evaluation and joint safety evaluation and diagnosis.

* Co-leads; Corresponding author.

Contents

- 1 Introduction 3
- 2 DeepSight Framework 3

- 2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2.2 DeepSafe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 DeepScan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Experiments 9

- 3.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.2 Content Risk Evaluation and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.3 Frontier AI Risk Evaluation and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 3.4 Joint Safety Evaluation and Diagnosis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 4 Related Work 23

- 4.1 Evaluation Frameworks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 4.2 Diagnosis Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 5 Conclusion and Discussion 24

- 1 Introduction

The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has fundamentally transformed the artificial intelligence landscape, positioning safety as a paramount concern alongside performance capability (Wei et al., 2023). As these models are increasingly integrated into critical applications (Chu et al., 2025; Zhang et al., 2023; Yang et al., 2026b), the industry faces an urgent need to ensure they operate within secure and ethical boundaries.

However, the current landscape of safety evaluation remains fractured between black-box assessment and white-box insight. While the industry has established robust evaluation frameworks, ranging from OpenAI Evals (OpenAI, 2023) and Inspect (UK AI Security Institute, 2024) to comprehensive platforms like OpenCompass (Contributors, 2023) and HELM (Liang et al., 2022), these tools primarily focus on quantifying behavior capabilities instead of safety mechanisms. Conversely, diagnostic research has made significant strides in decoding internal mechanisms, including probing latent geometric boundaries, identifying safety-specific neurons, and analyzing reasoning dynamics via information flow (Qian et al., 2024a; 2025; Burns et al., 2022; Zou et al., 2023). Yet, these diagnostic methods often operate in isolation from standardized benchmarks, treating internal representation analysis as a separate academic pursuit rather than a debugging tool for deployment risks.

To systematically address these issues, we propose DeepSight, an open-source project, that proposes and implements a new safety evaluation-diagnosis integrated paradigm. By unifying task and data protocols, DeepSight connects evaluation and diagnosis. In this way, researchers and developers not only identify what went wrong but also understand how safety concepts are encoded internally, facilitating more reliable and interpretable repairs. DeepSight is composed of two core engines:

- • DeepSafe: A modular, configuration-driven framework for multimodal LLM/MLLM evaluation that integrates over 20 safety benchmarks. More importantly, DeepSafe also supports the evaluation of frontier AI risks, i.e., high-severity risks (Lab et al., 2025). It provides a low-cost, reproducible, efficient, and highly scalable large-scale model security evaluation toolkit for researchers and professional developers.
- • DeepScan: A standardized and extensible framework equipped with a suite of LLM diagnostic tools. It efficiently locates key layers and neurons to probe representation-level structures and objective-level conflicts without modifying model weights.

To the best of our knowledge, DeepSight is the first open source toolkit that support the frontier AI risk evaluation and joint safety evaluation and diagnosis. We conduct a comprehensive analysis of prevailing Large Models (LMs), revealing safety trends in the current LM landscape:

- • The introduction of visual modalities significantly expands the attack surface, causing safety alignment to decline across all model tiers compared to text-only scenarios. Furthermore, this transition amplifies the safety performance disparity between open-source and closed-source models, with the latter maintaining a distinct advantage in cross-modal scenarios.
- • We observe a complex trade-off regarding reasoning capabilities. In multimodal environments, reasoning-enabled models demonstrate superior safety alignment, effectively identifying image-text splitting attacks where non-reasoning models fail.
- • For AI frontier risks, safety advantages prove non-transferable across dimensions. No single model dominates every category, and even top-ranking models can exhibit severe failures in specific risks, for example Kimi-K2-Thinking ranks last in Manipulation, while lower-tier models can surprisingly lead in some dimensions.
- • Diagnosis analysis reveals that effective safety depends on the precise geometric structure of a model’s latent space, where both insufficient and excessive separation between safe and harmful representations prove detrimental to model robustness.

- 2 DeepSight Framework

- 2.1 Overview

While traditional evaluation often isolates evaluation and diagnosis, resulting in misaligned goals where evaluation identifies external risks without addressing internal root causes, DeepSight connects them into

a verifiable engineering loop, effectively transitioning from black-box assessment to white-box insight. The framework is composed of two engines: DeepSafe, a configuration-driven framework that standardizes behavioral assessment across over 20 benchmarks, and DeepScan, a diagnostic tool that probes representation-level structures and objective-level conflicts to explain the mechanisms behind observed failures. By harmonizing task and data protocols, DeepSight enables researchers not only to identify what went wrong but also to understand how safety concepts are encoded internally, facilitating more reliable and interpretable repairs.

- 2.2 DeepSafe

[Figure 2]

- Figure 1: The DeepSafe Architecture. DeepSafe employs a configuration-driven approach where the Registry Hub orchestrates the interaction between Datasets, Models, and Evaluators. This modular design automates the workflow from inference (Runner) to analysis (Summarizer), producing standardized reports.

Safety evaluation for large models currently suffers from a lack of standardized protocols and dedicated assessment tools. We present DeepSafe, an all-in-one framework integrating over 20 safety datasets, such as SALAD-Bench (Li et al., 2024a) and HarmBench (Mazeika et al., 2024), alongside the specialized ProGuard judge model (Yu et al., 2025). DeepSafe supports LLM/MLLM evaluation through a modular, configurationdriven architecture (illustrated in Figure 2) that automates the entire workflow from inference to deep reporting. It is designed to advance LM safety evaluation from outcome-based testing to rigorous analysis, accelerating the development of trustworthy AI evaluation. The core components of the framework are summarized in Table 1.

Component Role Key Implementations Models Unified interface for model loading,

HFModel (Hugging Face), APIModel (e.g., GPT-5), and vLLM integration for accelerated generation.

management, and high-performance inference.

Datasets Standardized data handling supporting various modalities and safety domains.

Multimodal support, Frontier AI Risk datasets, and Preprocess pipelines.

Evaluators Executes safety assessments compatible with diverse benchmark protocols.

Native evaluators, rule-based matching, and model-based judges.

Runner Orchestrates the end-to-end workflow from configuration to execution.

Configuration-driven architecture and automated pipeline management.

Summarizer Aggregates evaluation metrics and generates visualization reports.

BaseSummarizer for statistical analysis and automatic Markdown report generation.

- Table 1: Summary of DeepSafe’s core components. The framework adopts a modular design enabling flexible extension and high-performance execution, supporting native evaluators for various benchmarks.

As listed in Table 1, DeepSafe is built upon five foundational modules, coordinated via a centralized Registry Mechanism. This design decouples task implementation from execution logic, allowing researchers to integrate new datasets or metrics with minimal code changes.

- 2.2.1 Core Modules

Models. The Models component provides a unified interface for model inference, abstracting away differences between local weights and remote APIs. We implement VLLMLocalModel based on the vLLM library to enable high-throughput generation for open-source models. For closed-source models, APIModel encapsulates interactions with commercial APIs. This abstraction allows seamless switching between backends by modifying the configuration.

Datasets. DeepSafe standardizes data handling through the BaseDataset protocol. Regardless of the raw format, all data loaders normalize inputs into a consistent schema containing unique identifiers, prompts, and reference answers. The framework currently integrates loaders for over 20 benchmarks, covering diverse domains from traditional content safety to frontier AI risks. Additionally, JsonlDataset allows users to swiftly plug in custom datasets.

Runners. The Runner serves as the orchestration engine, managing the end-to-end evaluation lifecycle. We provide a robust LocalRunner that handles pipeline automation, including batch processing and result persistence. A key feature is its state-aware execution, which automatically detects and skips generation process, enabling efficient resumption of evaluations.

Evaluators & Metrics. Evaluation logic is decoupled into Evaluators and Metrics. DeepSafe supports a hybrid evaluation paradigm to accommodate diverse benchmarks: (1) Native Evaluators: We integrate official evaluation scripts for specific benchmarks (e.g., HarmBench, BeaverTails) to ensure strict adherence to their original protocols. (2) Rule-based Evaluators: For tasks with deterministic criteria, we provide fast keyword matching and regular expression handlers. (3) Model-based Evaluators: We support LLM-as-a-Judge, leveraging strong models to assess complex responses where rule-based methods fail. Post-evaluation, Metrics modules aggregate raw judgments into statistical scores, ensuring precise measurement.

Summarizers. To facilitate rapid analysis, the Summarizer component aggregates evaluation metrics and renders them into human-readable formats. The StandardSummarizer automatically generates a comprehensive Markdown report containing summary tables and configuration details, along with a structured JSON file for downstream analysis.

- 2.2.2 ProGuard: A Specialized Safety Evaluator

While the framework supports various native and general-purpose evaluators, generic LLMs often lack specific alignment for safety nuances. DeepSafe therefore integrates ProGuard (Yu et al., 2025), a specialized safety evaluator model designed to deliver superior judgment accuracy. ProGuard is fine-tuned on a curated dataset of 87k safety pairs, enabling it to detect subtle risks and adversarial attacks that generic models might miss. Within the framework, ProGuard functions as a plug-and-play judge backend, offering a high-fidelity alternative to the benchmarks’ original evaluators.

- 2.2.3 Unified Evaluation Workflow

DeepSafe streamlines the safety evaluation process into a standardized four-stage workflow, fully driven by a single YAML configuration file:

- 1. Configuration: Users define the target model, dataset, and evaluation parameters in a declarative YAML file.
- 2. Inference: The Runner initializes the model and dataset, performing batch inference to generate responses.
- 3. Assessment: The Evaluator processes the generated responses—using either native protocols, rulebased matching, or ProGuard—to produce judgments.
- 4. Reporting: The Summarizer aggregates judgments into final metrics and exports visualization reports.

This “Config-as-Execution” paradigm significantly lowers the barrier for safety research, ensuring that evaluations are both scalable and reproducible.

- 2.3 DeepScan

Safety evaluation frameworks such as DeepSafe answer the question of whether a model exhibits safe behavior under standardized benchmarks. To support the development of more reliable and interpretable systems, it is equally important to understand how safety-related concepts are encoded in internal representations and whether different safety objectives conflict at the parameter level. Such diagnostic analyses require consistent access to intermediate activations, standardized metrics, and reproducible pipelines across model families and benchmarks. We present DeepScan, a flexible and extensible diagnostic framework for LLMs and MLLMs. DeepScan is designed to complement DeepSafe: together they form a complete evaluation–diagnosis engineering pipeline—DeepSafe assesses behavioral safety outcomes, while DeepScan probes representation-level structure and objective-level conflicts without modifying model weights.

[Figure 3]

[Figure 4]

[Figure 5]

CONFIG REGISTRY HUB

STRUCTURED RESULT

##### Models

##### Summarizers

Evaluators

##### Datasets

Models

LLMs

X-Boundary

Aggregation

[Figure 6]

Local

[Figure 7]

[Figure 8]

Datasets

JSON

[Figure 9]

[Figure 10]

TELLME

MLLMs

X-Boundary

HF Hub

Evaluators

Diagram

[Figure 11]

. . .

. . .

Summarizers

Report

[Figure 12]

##### CONCRETE IMPLEMENTATION

[Figure 13]

[Figure 14]

ORCHESTRATOR

- Figure 2: The DeepScan Architecture. DeepScan follows a configuration-driven paradigm, with a Registry Hub coordinating the interplay between Datasets, Models, and Evaluators. This modular organization streamlines the full workflow—from inference execution (Runner/Orchestrator) to aggregation and analytical summarization (Summarizer)—and produces standardized, machine-readable outputs (e.g., JSON) alongside report-oriented artifacts such as diagrams and written summaries.

DeepScan is built around a “Register → Configure → Execute → Summarize” workflow. A centralized registry system decouples model instantiation, dataset loading, evaluator logic, and report generation, so that new model families, diagnostic benchmarks, or metrics can be integrated with minimal changes to core execution code. The entire pipeline is driven by a single declarative configuration file (YAML or JSON), and can be invoked from the command line without writing code, or programmatically via a unified API. The framework exposes a consistent inference interface across supported model families (Qwen, Llama, Mistral, Gemma, GLM, InternLM, InternVL, etc.) while preserving access to the underlying Hugging Face model and tokenizer for representation extraction. Table 2 summarizes the core components; Table 3 lists the built-in diagnostic evaluators and their associated metrics and data sources.

- 2.3.1 Core Components

Registry system. DeepScan adopts a registry pattern for all pluggable components. The Model Registry stores model factories keyed by family and generation (e.g., qwen3 (Yang et al., 2025), llama3 (Grattafiori

- et al., 2024)); lookup returns a model runner that provides a uniform generate() interface and retains references to the underlying Hugging Face model and tokenizer. The Dataset Registry maps dataset names (e.g., tellme/beaver_tails_filtered, xboundary/diagnostic) to loaders that return a consistent structure (e.g., splits keyed by train/test, or message lists for chat templates). The Evaluator Registry and Summarizer Registry map string identifiers to evaluator and summarizer classes; instantiation is performed

Component Role Key Implementations Registry System Centralized registration and re-

BaseRegistry, ModelRegistry, DatasetRegistry, EvaluatorRegistry, SummarizerRegistry; factorybased instantiation with configurable parameters.

trieval of models, datasets, evaluators, and summarizers.

Configuration Declarative specification of runs via YAML; support for batch runs.

ConfigLoader from file or dict; dot-notation access; perevaluator overrides for dataset and summarizer.

Model Runners Unified inference API and access to raw model/tokenizer for hooks.

BaseModelRunner with generate() and chat-style APIs; model/tokenizer exposed for activation extraction.

Evaluators Encapsulation of diagnostic protocols with a single evaluate(model, dataset,

BaseEvaluator; built-in TELLME (Chen et al., 2025), XBoundary (Lu et al., 2025), MI-Peaks (Qian et al., 2025), SPIN (Qian et al., 2024a);.

...) contract.

Summarizers Aggregation of raw results into structured metrics and humanreadable reports.

BaseSummarizer; benchmark-specific summarizers (e.g., TELLME, X-Boundary, SPIN); combined run-level summarizer.

- Table 2: Summary of DeepScan’s core components. The registry-based design allows new models, datasets, and diagnostic methods to be added without altering the execution engine.

Evaluator Primary metrics Data & methodology X-Boundary separation-score (↑); boundary-ratio

(↓); dist-bound-safe (↓); dist-boundharmful (↑)

Representation diagnostic tool for open-source LLMs focusing on the geometric structure of hidden spaces formed by safe, harmful, and boundary samples. Centroid-based metrics quantify separability and boundary alignment; optional t-SNE per layer.

TELLME rdiff (↑); rsame (↓); eRank; cos_sim (↓); L1, L2, Hausdorff (↑)

Enhances intrinsic transparency by disentangling representations across behaviors. Coding-rate and distance metrics over class-conditioned activations (e.g., filtered BeaverTails); applicable to text/multimodal and Dense/MoE architectures.

SPIN Fairness–Privacy Neurons Coupling Index (↓)

Neuron-level diagnosis for open-source LLMs; quantifies overlap between fairness- and privacy-related neurons. Generalizable to other objective pairs (e.g., safety–utility, alignment–generalization).

MI-Peaks Mutual Information Trajectory (visualization)

Information-theoretic framework to examine reasoning dynamics during generation. Tracks MI between stepwise hidden representations and the correct answer; MI Peaks as structural signature. For Transformer-based autoregressive LLMs/LRMs.

- Table 3: Built-in DeepScan diagnostic evaluators (metric names and directions calibrated). Each implements BaseEvaluator and can be combined in a single run with different datasets.

with configuration dictionaries passed from the run configuration. This design decouples the definition of new models, datasets, or diagnostic protocols from the orchestration logic, so that extensions require only implementing the appropriate interface and registering the component.

Configuration and execution. Run specification is entirely configuration-driven. The ConfigLoader supports loading from YAML or JSON files and from in-memory dictionaries; nested keys are accessible via dot notation, and configurations can be merged. A valid run configuration specifies at least one model block (with generation or name, and optional model_name, device, dtype, etc.), a root-level or per-evaluator dataset block, and either a single evaluator or a list evaluators. Each evaluator entry includes type, optional run_name, and optional dataset and summarizer overrides. The runner validates that all referenced models, datasets, and evaluators are registered, then loads the model once and iterates over evaluators: for each evaluator it loads the corresponding dataset, calls evaluator.evaluate(model, dataset, output_dir=...), and persists results (e.g., results.json) and any per-evaluator summary. A run-level summarizer (e.g.,

type: combined) may aggregate results across evaluators and models. Outputs are organized by run ID, model, and evaluator name to support reproducible comparison and sharing.

Model runners and representation access. Model registry lookups return a runner object that wraps the Hugging Face model and tokenizer. The runner exposes generate() for text and chat-style prompts (with optional GenerationRequest/GenerationResponse structures) so that all evaluators can perform inference through a single API regardless of model family. At the same time, the runner exposes model and tokenizer so that evaluators that need intermediate representations can attach forward hooks, read hidden states at specific layers, or compute gradients (e.g., for SPIN importance scores). This split keeps the execution engine agnostic to the details of each diagnostic method while ensuring that representation-level analyses have the necessary low-level access.

- 2.3.2 Built-in Diagnostic Evaluators

X-Boundary: geometry of safe, harmful, and boundary regions. X-Boundary (Lu et al., 2025) (Table 3) is a training-free representation diagnostic for open-source LLMs that analyzes intermediate-layer hidden spaces induced by safe, harmful, and boundary samples. Beyond reporting centroid-based separability/alignment scores, it is mainly useful for explaining failure modes: whether apparent safety comes from a crisp decision boundary or from collapsing/perturbing boundary-safe representations (a pattern associated with boundary ambiguity and over-refusal). The evaluator can additionally export per-layer t-SNE visualizations to support qualitative inspection of layerwise geometry shifts.

TELLME: disentanglement metrics. TELLME (Chen et al., 2025) (Table 3) measures how well internal representations separate different behaviors, aiming to make behaviors intrinsically monitorable rather than relying on external detectors. In practice, it diagnoses whether mixed concepts are decomposed into compact, more independent subspaces, and can be used to compare pre-/post-training representation migration as evidence of internal-state change. It supports text and multimodal models, including Dense and MoE architectures, and is commonly run on filtered BeaverTails-style splits using last-token hidden states at a configurable layer.

SPIN: conflicts between safety objectives. SPIN (Qian et al., 2024a) (Table 3) provides a neuron-level view of objective trade-offs by quantifying how much two attributes share the same internal neurons (illustrated with fairness vs. privacy). This turns outcome-level trade-off observations into an interpretable structural signal, enabling comparisons across models or checkpoints and guiding multi-objective optimization. The same coupling analysis can be instantiated for other potentially conflicting pairs (e.g., safety–utility, alignment– generalization).

MI-Peaks: information evolution during reasoning. MI-Peaks (Qian et al., 2025) (Table 3) analyzes stepwise reasoning dynamics in autoregressive Transformers by tracking how task-relevant information evolves during generation, moving from final-answer evaluation to a process-level lens. Its key qualitative signature is that the mutual information trajectory often changes via a few sharp surges (MI Peaks) rather than smoothly, highlighting specific steps where representations become disproportionately informative about the correct answer. This provides a reusable, training-free way to compare reasoning traces across LLMs/LRMs.

- 2.3.3 Summarization and reporting

Each evaluator produces a raw results dictionary (metrics, artifact paths, etc.). Benchmark-specific summarizers (e.g., TELLME, X-Boundary, SPIN) consume these results and produce structured summaries: for example, the X-Boundary summarizer selects the best layer by separation score or boundary ratio and records paths to metrics and t-SNE plots. A run-level combined summarizer can aggregate results from multiple evaluators and models into a single summary and Markdown report. Summarizers implement summarize(...) and optionally format_report(..., format="markdown"), with output written to summary.json and summary.md in the appropriate output directory. This enables rapid comparison across runs and models without parsing raw evaluator outputs by hand.

- 2.3.4 Unified diagnosis workflow DeepScan streamlines the full diagnostic pipeline into three stages, all driven by the same configuration file:

- 1. Configuration. The user specifies one or more models (by registry key and optional variant), a default dataset, and one or more evaluators. Each evaluator may override the dataset and attach a dedicated summarizer. Optional run-level summarizer and output directory complete the specification.
- 2. Execution. The runner loads each model once and, for each evaluator, loads the corresponding dataset and invokes evaluator.evaluate(model, dataset, output_dir=...). Results and per-evaluator summaries are written under output_dir/run_id/model_id/evaluator_id/. Multiple evaluators (e.g., TELLME, X-Boundary, SPIN, MI-Peaks) can be run in sequence in a single invocation, with optional CUDA cache cleanup between evaluators to manage memory.
- 3. Summarization and persistence. Per-evaluator summarizers (if configured) produce summary.json and summary.md per evaluator; the run-level summarizer (if configured) produces an aggregate summary and report at the run directory. The full run payload (run ID, timestamp, model and evaluator configs, and paths to all result files) is also written to results.json for traceability and downstream analysis.

This design supports multi-model, multi-evaluator runs (e.g., the same four evaluators over several model families) with a single config and consistent output layout. Together with DeepSafe’s outcome-based safety evaluation, DeepScan provides a unified ecosystem for both assessing and diagnosing large-model safety.

- 3 Experiments

- 3.1 Experimental Setup

DeepSafe setup. Leveraging the DeepSafe framework, we conduct a comprehensive safety evaluation of LLMs and MLLMs to characterize their strengths and vulnerabilities. We employ a diverse suite of benchmarks categorized into two primary domains: content risk and frontier risk. For LLM content risk, we utilize SALADBench (Li et al., 2024a), Flames (Huang et al., 2024), Fake Alignment (Wang et al., 2024), MedHallu (Pandit

- et al., 2025), HaluEval (Li et al., 2023), Do-Not-Answer (Wang et al., 2023b), BeaverTails (Ji et al., 2023), XSTest (Röttger et al., 2024), and HarmBench (Mazeika et al., 2024). MLLM content risk is assessed using SIUO (Wang et al., 2025b), VLSBench (Hu et al., 2025), MMSafetyBench (Liu et al., 2024), MSSBench (Zhou

- et al., 2024), Ch3ef (Shi et al., 2024), and MOSSBench (Li et al., 2024c). Finally, to evaluate frontier risks, we incorporate Evaluation Faking (Fan et al., 2025), Sandbagging (OpenAI, 2023), Manipulation (Lab
- et al., 2025), Mask (Ren et al., 2025), DeceptionBench (Ji et al., 2025), BeHonest (Chern et al., 2024), Reasoning Under Pressure (MacDermott et al., 2025), AIRD (Lab et al., 2025), and WMDP (Li et al.,

- 2024b). For the representative LLMs, we evaluate Kimi-K2-Thinking (Team, 2025), GPT-4o (OpenAI, 2024), GPT-5.2 (Singh et al., 2025), Claude-Sonnet-4-5-20250929 (Anthropic, 2025), Qwen2.5-72B-Instruct (Qwen et al., 2025), Llama-3.3-70B-Instruct (Grattafiori et al., 2024), Mistral-Small-24B-Instruct-2501 (Mistral,
- 2025), Gemini-3-flash (Google, 2025), DeepSeek-R1-250528 (Guo et al., 2025), Gemma-3-27B-IT (Team et al., 2025a), InternLM3-8B-Instruct (Cai et al., 2024), Qwen3-30B-A3B-Thinking-2507 (Yang et al., 2025), DoubaoSeed-1-6-flash-250828 (Bytedance, 2025), GLM-4.5-Air (Zeng et al., 2025). For the MLLMs, we evaluate GLM-4.6V (Team et al., 2026), Qwen3-VL-235B-A22B-Instruct (Bai et al., 2025a), Gemma-3-27B-IT (Team et al., 2025a), GPT-5.2 (Singh et al., 2025), Qwen2.5-VL-72B-Instruct (Bai et al., 2025b), Claude-Sonnet-45-20250929 (Anthropic, 2025), Llama-4-Scout-17B-16E (Meta, 2025), GPT-4o (OpenAI, 2024), InternVL3 5-241B-A28B (Zhu et al., 2025), Gemini-3-flash (Google, 2025), Kimi-VL-A3B-Thinking (Team et al., 2025b), Ministral-3-14B-Instruct-2512 (Mistral, 2025), Doubao-Seed-1-6-flash-250828 (Bytedance, 2025).

DeepScan setup. We evaluate all model checkpoints in our task suite, including Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-72B-Instruct (Qwen et al., 2025); Llama-3.3-70B-Instruct (Grattafiori et al., 2024); Mistral-Small-24B-Instruct-2501 (Mistral, 2025); Ministral-3-14B-Instruct-2512 (Mistral, 2025); Gemma-3-27B-IT (Team et al., 2025a); GLM-4.5-Air (Zeng et al., 2025); InternLM3-8B-Instruct (Cai et al., 2024); and InternVL3.5-14B / InternVL3.5-241B-A28B (Zhu et al., 2025). Across models, we use deterministic decoding with generation defaults aligned with official model recommendations; model loading follows each backbone’s recommended options. The diagnostic setup uses default settings of X-Boundary (Lu et al., 2025), TELLME (Chen et al., 2025), SPIN (Qian et al., 2024a). MI-Peaks (Qian et al., 2025) remains an excluded disabled component in this experiment suite.

- 3.2 Content Risk Evaluation and Analysis

The leaderboards are published on the website∗. Analysis is provided in the following sections.

Our evaluation employs distinct datasets for different analytical objectives. For safety risk analysis across six security dimensions (Sections 3.2.1–3.2.3), we adopt comprehensive safety benchmarks tailored to text and multimodal scenarios. For LLMs, evaluation is conducted on text-based datasets such as Salad-Bench, Fake Alignment, Flames, HaluEval, Do-not-answer, Harmbench, Beavertails, and MedHallu. For MLLMs, we utilize multimodal-specific benchmarks such as VLSBench, MMSafetyBench, SIUO, MSSBench, and Ch3ef. For over-safety analysis (Section 3.2.4), we employ specialized datasets designed to assess model performance in over-safety scenarios: XSText for text-only models and Mossbench for multimodal models.

- 3.2.1 Safety Risk Trends Across Different Model Series

Our safety risk analysis reveals distinct performance characteristics across model architectures: LLMs demonstrate hierarchical distributions in safety capabilities, whereas MLLMs encounter significantly greater challenges attributable to expanded cross-modal attack surfaces. The evaluation framework encompasses six taxonomic dimensions of safety risks: (1) Model Algorithm Security, (2) Data Security, (3) Network System & Information Content Security, (4) Reality & Cognitive Security, (5) Social, Environmental & Ethical Security, and (6) Security Vertical Domain. Through systematic examination of both LLM and MLLM performance across these dimensions, we identify distinct performance tiers and characterize key evolutionary patterns in model safety development.

LLM Safety Risk Trend Analysis. As shown in Figure 3 and Figure 4, LLMs exhibit clear hierarchical distribution in safety performance, with defense capabilities showing significant positive correlation with the foundational cognitive capabilities of the models.

[Figure 15]

Qwen3-30B-A3B-Thinking

0.79

Qwen2.5-72B-Instruct

0.79

Claude-Sonnet-4.5

0.77

GPT-4o

0.76

Doubao-Seed-1.6-flash

0.75

Llama-3.3-70B-Instruct

0.74

InternLM3-8B

0.74

GPT-5.2

0.74

Mistral-Small-24B-Instruct

0.72

Kimi-K2-Thinking

0.71

DeepSeek-R1

0.71

Gemma-3-27B-IT

0.70

Gemini-3-Flash

0.67

GLM-4.5-Air

0.62

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Overall Average Safety Rate

- Figure 3: LLM Safety Risk Ranking. Average safety rate across all datasets for evaluated LLMs, ranked in descending order.

Performance Tier Classification and Risk Quantification Analysis. We classify evaluated LLMs into four performance tiers based on their comprehensive safety rate:

• First Tier (Represented by Qwen3 Series and Claude Series). LLMs in this tier occupy leading positions in comprehensive safety metrics, with overall scores consistently maintained above 0.77. In the “Social and Ethical Safety” dimension, they demonstrate highly precise safety alignment

∗https://ai45.shlab.org.cn/deepsafe

1.2

Models by Tier Qwen3-30B-A3B-Thinking Claude-Sonnet-4.5 GPT-4o

DeepSeek-R1 Kimi-K2-Thinking Gemini-3-Flash

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

Llama-3.3-70B-Instruct

GLM-4.5-Air

1.0

0.88

0.87

0.86

0.86

0.83 0.84

0.81

[Figure 16]

0.82

0.82

0.80

0.79 0.80

0.79

0.79

0.79

0.75

0.74

0.8

0.76

0.76

0.76

AverageSafetyRate

0.74

0.74 0.73

0.73

0.73

0.68

0.72

0.72

0.68

0.71 0.68

0.68 0.69 0.66 0.70

0.67

0.67

0.67

0.66

0.63

0.62 0.60

0.62

0.61

0.60

0.57

0.6

0.49

0.48

0.4

0.2

0.0

ModelAlgorithmSecurity DataSecurityNetwork&InformationSecurityReality&CognitiveSecuritySocial&EthicalSecurity SafetyDomain

Safety Categories

- Figure 4: LLM Safety Rate Comparison Across Safety Categories by Model Tier. Evaluated LLMs are classified into four performance tiers based on their overall safety rates, with rates compared across six safety categories.

capabilities (scores frequently exceeding 0.85). However, in the fundamental defense dimension of “Model Algorithm Safety,” there still exist relatively significant security vulnerabilities, indicating that algorithmic robustness remains a weak point.

- • Second Tier (Represented by GPT-4o and Llama-3 Series). This tier exhibits score distribution within the 0.74–0.76 range, demonstrating high defense stability. However, when processing complex logical adversarial attacks involving “Reality and Cognitive Safety” and “Social Ethics,” their defense consistency shows decline compared to the first tier.
- • Third Tier (Represented by InternLM3 and DeepSeek Series). This tier maintains comprehensive scores within the 0.71–0.74 range. When facing algorithm safety and vertical domain risks, their recognition accuracy demonstrates decline relative to the preceding two tiers.
- • Fourth Tier (Represented by Kimi and Gemini-Flash Series). This tier scores below 0.71, with scores in the “Model Algorithm Safety” dimension generally at lower levels (some as low as 0.48). Constrained by model scale or alignment depth, models in this tier exhibit relatively weak defense capabilities when responding to structured adversarial instructions.

Evolution Trend Analysis. Two key evolutionary trends emerge from the LLM safety landscape. First, the evolution of defense mechanisms exhibits unbalanced development across defense dimensions, having shifted from surface-level content filtering to deep logical alignment. All tiers demonstrate superior performance in the ethical safety dimension compared to the algorithm safety dimension, suggesting future focus should consider transitioning toward enhancing robustness of underlying algorithms. Second, a coupling relationship between cognitive capability and safety performance is observed: data indicates that deep logical analysis capabilities (such as the Thinking series) contribute to improving the accuracy of safety boundaries.

MLLM Safety Risk Trend Analysis. The risk characteristics of MLLMs are more complex than those of text-only models, as the introduction of visual modality significantly increases the attack surface. Figure 5 and Figure 6 illustrate the safety risk ranking and per-model performance comparison for MLLMs.

Performance Tier Classification and Cross-modal Risk Quantification. We similarly classify MLLMs into four tiers, where the overall rate ranges are notably lower than their text-only counterparts:

• First Tier (Represented by Qwen3-VL and GPT-5.2). Safety rates are maintained at 0.65–0.71. This performance level is lower relative to text scenario performance, reflecting the challenges of cross-modal safety alignment. At the “Model Algorithm Safety” level, scores have declined to

[Figure 17]

Qwen3-VL-235B-A22B-Instruct

0.71

GPT-5.2

0.67

GLM-4.6V

0.65

Claude-Sonnet-4.5

0.61

Gemini-3-Flash

0.58

Gemma-3-27B-IT

0.53

GPT-4o

0.53

Qwen2.5-VL-72B-Instruct

0.49

InternVL3_5-241B-A28B

0.48

Doubao-Seed-1.6-flash

0.47

Llama-4-Scout-17B-16E

0.47

Ministral-3-14B-Instruct-2512

0.47

Kimi-VL-A3B-Thinking

0.38

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Overall Average Safety Rate

- Figure 5: MLLM Safety Risk Ranking. Average safety rate across all datasets for evaluated MLLMs, ranked in descending order.

AlgorithmSecurity DataSecurityNetwork&InformationSecurityReality&CognitiveSecuritySocial&EthicalSecurity SafetyDomain

Safety Categories

0.0

0.2

0.4

0.6

0.8

1.0

1.2

AverageSafetyRate

[Figure 18]

0.60

0.73

0.77

0.65 0.66

0.83

0.54

0.76 0.75

0.70

0.49

0.78

0.49

0.59

0.64

0.59

0.57

0.80

0.44

0.59

0.54

0.65

0.49

0.80

0.31

0.48

0.47

0.58

0.54

0.81

0.37

0.43

0.38

0.47

0.45

0.74

0.28

0.51

0.38

0.48

0.42

0.76

0.31

0.26

0.19

0.34

0.45

0.71

Models by Tier Qwen3-VL-235B-A22B-Instruct GPT-5.2 Claude-Sonnet-4.5 Gemini-3-Flash

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

Gemma-3-27B-IT Doubao-Seed-1.6-flash Llama-4-Scout-17B-16E Kimi-VL-A3B-Thinking

| |
|---|

| |
|---|

| |
|---|

- Figure 6: MLLM Safety Rate Comparison Across Safety Categories by Model Tier. Evaluated MLLMs are classified into four performance tiers based on their overall safety rates, with rates compared across six safety categories.

0.54–0.60, indicating that vision-language fusion significantly weakens the existing safety defense foundation.

- • Second Tier (Represented by Claude-Sonnet 4.5 and Gemini-3-Flash). Safety rate range spans 0.58– 0.61. Models demonstrate significant instability when identifying cross-modal harmful instructions (such as image-text semantic consistency induction). Particularly in the “Data Safety” dimension, there is a relatively significant decline compared to the first tier.
- • Third Tier (Represented by Gemma-3 and Doubao-Seed). Safety rate further decline to 0.47–0.53. Average scores in the “Model Algorithm Safety” dimension drop to extremely low levels of 0.31–0.37, with visual modality increasing the possibility of circumventing safety alignment mechanisms to a certain extent.

- • Fourth Tier (Represented by Llama-4 and Kimi-VL). These models demonstrate weak performance in safety defense, with minimum scores of only 0.38. Particularly in the “Network Systems and Information Content Safety” dimension, some scores decline to 0.19, indicating notably insufficient capability in identifying cross-modal adversarial attacks.

Evolution Trend Analysis. Two critical trends characterize the multimodal safety landscape. First, a pronounced cross-modal safety degradation effect is observed: all tiers experience significant decline in defense capabilities following multimodal introduction, reflecting substantial room for improvement in multimodal safety defense. Second, intensified performance differentiation between tiers is evident: the score gap between leading and trailing tiers in multimodal scenarios (0.33) significantly exceeds that in text domains (0.17), revealing marked technical disparities in cross-modal safety alignment.

- 3.2.2 Comparison of Reasoning and Non-Reasoning Models

We investigate the effect of reasoning capabilities on model safety by comparing reasoning-enabled and non-reasoning variants across both text-only and multimodal scenarios. Our analysis reveals a notable modality-dependent divergence: reasoning contributes limited safety improvement in text domains, yet demonstrates measurable defense advantages in multimodal environments.

Analysis of Reasoning Toggle in LLMs. We compare reasoning and non-reasoning LLM variants on text safety tasks. As shown in Figure 7 and Figure 8, the two categories exhibit largely comparable safety profiles in text scenarios.

Network System & Information Content Security

Data Security

Reasoning Models

Non-reasoning Models

- 0.78 0.76

0.63

- 0.79

0.76

0.76

0.61

Reality Security & Cognitive Security

Model Algorithm Security

0.69

0.63

0.68

0.76

Social, Environmental & Ethical Security Risks

Security Vertical Domain

0.81

- Figure 7: Safety Rate Comparison Between Reasoning and Non-Reasoning LLMs Across Six Safety Categories. Average safety rates are computed across all reasoning models for each category, with nonreasoning models averaged similarly.

0.732

Non-Reasoning

0.713

Reasoning

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Average Safety Rate

Figure 8: Average Safety Rate: Reasoning vs. NonReasoning LLMs. Average safety rates computed across all datasets for reasoning and non-reasoning models.

Overall Performance Convergence Phenomenon. Reasoning and non-reasoning LLMs show highly similar performance in text safety tasks, indicating that traditional Instruct LLMs have achieved high levels of text safety alignment. Reasoning capabilities fail to translate into significant advantages, indicating that traditional Instruct models have achieved high levels of text safety alignment, and rapid-response defense strategies prove sufficient for most text attacks.

Asymmetric Characteristics: Coexistence of Advantages and Disadvantages. The radar chart reveals that reasoning LLMs demonstrate slight advantages in Network Systems and Information Content Safety (0.78 vs. 0.76) and Reality Safety and Cognitive Safety (0.63 vs. 0.61) dimensions, indicating their potential in complex semantic understanding tasks. However, their performance is relatively weaker in Model Algorithm Safety (0.63 vs. 0.69) and Vertical Domain Safety (0.68 vs. 0.76) dimensions. Particularly, the disadvantage in algorithm safety dimension indicates that additional computational overhead of reasoning processes actually reduces response efficiency in scenarios requiring rapid interception.

Analysis of Reasoning Toggle in MLLMs. In contrast to text scenarios, reasoning capabilities exhibit markedly different effects in multimodal settings. Figure 9 and Figure 10 present the comparative results for reasoning and non-reasoning MLLMs.

Network System & Information Security

Data Security

Reasoning Models

Non-reasoning Models

0.64

0.62

0.52

0.50

0.52

0.44

Reality Security & Cognitive Security

Algorithm Security

0.55

0.41

0.47

0.51

0.72

Social, Environmental & Ethical Security Risks

0.71

Security Vertical Domain

Figure 9: Safety Rate Comparison Between Reasoning and Non-Reasoning MLLMs Across Six Safety Categories. Average safety rates are computed across all reasoning models for each category, with nonreasoning models averaged similarly.

0.538

Non-Reasoning

0.563

Reasoning

0.0 0.1 0.2 0.3 0.4 0.5 0.6

Average Safety Rate

Figure 10: Average Safety Rate: Reasoning vs. NonReasoning MLLMs. Average safety rates computed across all datasets for reasoning and non-reasoning models.

Emergence of Measurable Safety Advantages. A notable reversal occurs in multimodal scenarios, where reasoning models (0.5633) begin to surpass non-reasoning models (0.5383), indicating that reasoning capabilities demonstrate greater efficacy in complex modal fusion scenarios. This contrasts with the marginal disadvantages observed in text scenarios, with reasoning mechanisms exhibiting systematic defense gains in multimodal environments.

Core Mechanisms of Cross-modal Defense. The data demonstrates that reasoning models achieve improved performance in multimodal scenarios, with significant improvements across most categories. This occurs because adversarial attacks in multimodal environments often employ “image-text splitting” strategies, such as images displaying legitimate content while text contains implicit violation instructions, or triggering malicious code execution through image steganography. Reasoning mechanisms enable models to perform explicit cross-modal consistency verification, simultaneously analyzing logical correlations among visual semantics, textual semantics, and output behaviors, thereby better identifying attack patterns concealed within normal inputs.

- 3.2.3 Comparison of Open-Source and Closed-Source Models

We examine the safety performance differences between open-source and closed-source models across text-only and multimodal scenarios. Our analysis indicates that while the two categories achieve comparable safety performance in text domains, a notable performance gap emerges in multimodal settings, suggesting differing levels of maturity in cross-modal safety alignment.

Performance Analysis of Open-Source vs. Closed-Source LLMs. We compare the safety profiles of opensource and closed-source LLMs in text scenarios. As illustrated in Figure 11 and Figure 12, the two categories demonstrate largely similar overall safety performance.

Overall Performance Convergence. In text safety tasks, closed-source and open-source LLMs show highly similar performance. Closed-source models average 0.7262, while open-source models score 0.7162, with a performance gap of only 1.4%. This convergence indicates that current LLMs have reached relative maturity in text safety domains, with both open-source and closed-source models establishing relatively comprehensive safety defense mechanisms.

Subtle Structural Differences. Despite overall similarity in performance, the two model categories exhibit subtle differences across various dimensions. Closed-source models demonstrate clear advantages in the Vertical Domain Safety dimension, potentially related to targeted optimization for enterprise-level application scenarios.

Performance Analysis of Open-Source vs. Closed-Source MLLMs. In contrast to text scenarios, the introduction of visual modality leads to a more pronounced differentiation between open-source and closedsource models. Figure 13 and Figure 14 present the comparative results in multimodal settings.

Network System & Information Content Security

Data Security

Open-source Models

Closed-source Models

0.78 0.77

0.77

0.76

0.62

0.65

Reality Security & Cognitive Security

Model Algorithm Security

0.63

0.65

0.69

0.74

0.79

Social, Environmental & Ethical Security Risks

Security Vertical Domain

0.80

Figure 11: Safety Rate Comparison Between OpenSource and Closed-Source LLMs Across Six Safety Categories. Average safety rates are computed across all open-source models for each category, with closedsource models averaged similarly.

0.726

Closed-Source

0.716

Open-Source

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Average Safety Rate

Figure 12: Average Safety Rate: Open-Source vs. Closed-Source LLMs. Average safety rates computed across all datasets for open-source and closed-source models.

Network System & Information Security

Data Security

Open-source Models

Closed-source Models

0.65

0.63

0.58

0.50

Reality Security & Cognitive Security

Algorithm Security

0.52

0.49

0.59

0.42

0.50

0.53

0.70

Social, Environmental & Ethical Security Risks

0.76

Security Vertical Domain

Figure 13: Safety Rate Comparison Between OpenSource and Closed-Source MLLMs Across Six Safety Categories. Average safety rates are computed across all open-source models for each category, with closedsource models averaged similarly.

0.600

Closed-Source

0.545

Open-Source

0.0 0.1 0.2 0.3 0.4 0.5 0.6

Average Safety Rate

Figure 14: Average Safety Rate: Open-Source vs. Closed-Source MLLMs. Average safety rates computed across all datasets for open-source and closedsource models.

Expansion of Performance Gap. Upon entering multimodal scenarios, the safety capability gap between open-source and closed-source models is amplified. Closed-source models average 0.6000, while open-source models achieve only 0.5450, representing an expanded performance disparity. The radar chart clearly shows that closed-source models outperform open-source models across all six dimensions. The three dimensions with the most significant gaps are: Algorithm Safety, Network System Security, and Reality and Cognitive Safety. The gap in Data Safety is smallest, reflecting certain accumulation by the open-source community in privacy protection. Even in Vertical Domain Safety, where both sides achieve their highest scores, closed-source models still maintain clear advantages.

Sources of Cross-Dimensional Performance Disparity. Open-source models have disadvantages in all six security dimensions of multimodal scenarios. This may stem from multimodal security alignment requires collaborative optimization across multiple modules such as vision and language, which places high demands on the scale of training data, the richness of test samples, and the investment of engineering resources.

- 3.2.4 Over-Safety Risk Analysis

Over-safety in LLMs manifests when models incorrectly reject legitimate instructions due to overly conservative safety mechanisms. This phenomenon poses a critical challenge to the balance between safety and usability, particularly as models expand from text-only to multimodal scenarios.

#### Over-Safety Characteristics of LLMs. Over-safety in text scenarios primarily manifests as models incorrectly intercepting instructions containing sensitive vocabulary but with legitimate intent.

|1 - Safe Refusal Rate Unsafe Refusal Rate<br><br>| |
|---|
|
|---|

1.00 1.00 0.99 1.00 0.99

0.99 0.99 0.99 0.99 1.00 1.00

0.97

1.0

0.94

0.81

0.80

0.8

0.73

RefusalRate

0.70

0.65

0.63

0.59

0.58

0.6

0.53

0.38

0.36 0.35

0.4

0.34

0.32

0.32

0.2

0.0

Qwen3-30B-A3B-ThinkingQwen2.5-72B-InstructDeepSeek-R1Llama-3.3-70B-InstructGLM-4.5-AirMistral-Small-24B-Instruct GPT-4o GPT-5.2Claude-Sonnet-4.5Gemini-3-FlashInternLM3-8BGemma-3-27B-ITKimi-K2-ThinkingDoubao-Seed-1.6-flash

- Figure 15: Over-Safety Analysis of LLMs. For each model, blue bars represent usability (computed as 1 minus the safe refusal rate on benign queries; higher values indicate better responsiveness to legitimate requests), while pink bars represent safety (unsafe refusal rate on harmful queries; higher values indicate better rejection of malicious content).

Balance Between Safety and Usability. We evaluate the trade-off between safety and usability by measuring models’ rejection rates on both benign and violation instructions. Our analysis reveals two key findings: (1) Cost of Conservative Strategies: The data indicate that some models, while performing well in violation instruction rejection, show relatively low benign instruction acceptance rates. For example, InternLM3-8B achieves a violation rejection rate of 0.81, but its benign acceptance rate is relatively lower compared to other models, suggesting that the model sacrifices some ability to recognize benign requests in pursuit of higher safety. Conversely, some models show extremely high benign acceptance rates but low violation rejection rates. For instance, Doubao-Seek-1.6-flash has a benign acceptance rate of 1.00 but a violation rejection rate of only 0.36, indicating serious security vulnerabilities. (2) Semantic Limitations of Keyword Defense: Models demonstrate excessive sensitivity to neutral vocabulary in specific domains (legal, medical, technical). Some legitimate requests are frequently rejected due to triggered defense mechanisms, indicating that models may have been exposed to excessive "boundary-ambiguous" negative examples during safety alignment training, resulting in overly conservative decision thresholds and insufficient semantic understanding capabilities in complex contexts.

Differentiated Performance Across Model Tiers. First-tier models demonstrate superior boundary judgment precision while maintaining high safety thresholds, effectively distinguishing between violation and benign requests. Conversely, lower-tier models exhibit pronounced “defense overflow,” with usability constrained by excessively strict and mechanical safety mechanisms.

Over-Safety Characteristics of MLLMs. Over-safety risks in multimodal scenarios stem from the unstructured nature of image information and ambiguity of visual implications, with complexity significantly exceeding pure text scenarios.

False Rejection Induced by Visual Semantics. Our benchmark data demonstrate that models frequently associate specific visual elements (e.g., kitchen utensils, medical instruments) with potential dangers, leading to false rejection of legitimate culinary or scientific requests. This “visual stress” response reflects models’ lack of deep scene understanding in visual-text semantic alignment. Furthermore, the introduction of visual modality expands the attack surface, prompting developers to adopt more aggressive filtering strategies that result in declined usability when processing complex image-text dialogues.

Conservative Risk-Mitigation Strategies. Certain models such as GPT-5.2 and Qwen3-VL exhibit an excessive propensity for refusal when processing images with socially sensitive content. Despite such content adhering to public morals and social norms, these models preemptively opt for avoidance as a risk-mitigation strategy.

0.6

0.54

0.50 0.49

0.5

RefusalRate

0.4

0.3

0.26

0.2

0.16 0.15

0.13

0.08 0.08 0.07 0.06

0.1

0.02 0.02

0.0

Qwen3-VL-235B-A22B-InstructGPT-5.2 Gemini-3-FlashClaude-Sonnet-4.5Llama-4-Scout-17B-16E InternVL3_5-241B-A28BGPT-4o Doubao-Seed-1.6-flashMinistral-3-14B-Instruct-2512Qwen2.5-VL-72B-InstructGemma-3-27B-ITKimi-VL-A3B-Thinking GLM-4.6V

- Figure 16: Over-Safety Analysis of MLLMs on Benign Multimodal Inputs. Per-model safe refusal rates on benign multimodal queries, ranked in descending order. Higher values indicate more severe over-safety issues, where models excessively reject legitimate requests.

Consequently, this conservative behavior results in diminished usability in evaluation scenarios involving social, environmental, and ethical risk factors.

- 3.3 Frontier AI Risk Evaluation and Analysis

Beyond content-level safety, we evaluate Frontier AI risks: high-severity risks enabled by frontier generalpurpose models that may rapidly escalate and cause substantial societal harm (Lab et al., 2025). Specifically, we study risks stemming from advanced model capabilities, including strategic misrepresentation, deceptive alignment, and degraded safety performance under high-stakes adversarial conditions. Table 4 shows results for 14 models and 9 risk dimensions. The leaderboard is on our website†.

- 3.3.1 Overall Frontier Risk Landscape

We first establish a holistic view of the frontier risk landscape before examining specific risk factors. Table 4 and Figure 17 jointly present the aggregate rankings and per-dimension profiles of all evaluated models.

Model Overall EvalFaking Sandbagging Manipulation Mask DeceptionBench BeHonest RUP AIRD WMDP

Kimi-K2-Thinking 74.93 98.08 69.25 1.11 63.91 96.67 69.25 87.50 95.33 93.23 GPT-4o 73.64 96.17 79.58 33.33 42.68 75.56 71.71 96.88 100.00 66.89 GPT-5.2 73.14 98.72 67.42 23.33 52.40 95.00 69.90 78.13 99.00 74.38 Claude-Sonnet-4.5 70.86 96.81 80.92 4.44 75.16 78.33 68.50 78.13 99.67 55.82 Qwen2.5-72B-Instruct 71.26 95.85 78.50 31.11 39.69 72.78 69.79 87.50 97.67 68.48 Llama-3.3-70B-Instruct 70.19 99.68 63.17 32.22 46.18 75.56 61.80 93.75 98.00 61.35 Mistral-Small-24B-Instruct 69.51 90.42 79.67 32.22 39.95 71.67 55.04 96.88 98.67 61.12 Gemini-3-Flash 69.20 98.40 83.67 27.78 35.65 76.67 46.97 78.13 94.67 80.87 DeepSeek-R1 68.43 94.57 86.42 2.22 41.34 96.67 60.37 96.88 83.00 54.40 Gemma-3-27B-IT 66.66 88.18 76.17 30.00 40.75 65.56 47.62 96.88 94.00 60.78 InternLM3-8B 66.11 96.49 57.33 28.89 39.26 70.00 61.61 90.63 94.00 56.75 Qwen3-30B-A3B-Thinking 64.57 97.44 61.50 4.44 41.65 58.89 57.93 96.88 99.33 63.10 Doubao-Seed-1.6-flash 62.21 92.65 65.17 1.11 40.16 58.90 69.92 62.50 100.00 69.48 GLM-4.5-Air 62.09 85.94 71.75 11.11 45.02 58.89 68.05 56.25 100.00 61.76

Table 4: Frontier AI Risk Evaluation. Each cell reports the safety rate (%) on the corresponding benchmark, with higher values indicating safer behavior (↑). Models are ranked by the Overall score. Bold denotes the highest score and underline denotes the lowest score in each column.

†https://ai45.shlab.org.cn/deepsafe

[Figure 19]

74.93

Kimi-K2-Thinking

73.64

GPT-4o

73.14

GPT-5.2

71.26

Qwen2.5-72B-Instruct

70.86

Claude-Sonnet-4.5

70.19

Llama-3.3-70B-Instruct

69.51

Mistral-Small-24B-Instruct

69.20

Gemini-3-Flash

68.43

DeepSeek-R1

66.66

Gemma-3-27B-IT

66.11

InternLM3-8B

64.57

Qwen3-30B-A3B-Thinking

62.21

Doubao-Seed-1.6-flash

62.09

GLM-4.5-Air

55 60 65 70 75

Overall Average Score

- Figure 17: Frontier AI Safety Risk Ranking. Overall average safety scores of evaluated LLMs, ranked in descending order.

Aggregate Ranking and Tier Structure. As shown in Table 4, the 14 evaluated models exhibit a clear tiered distribution in frontier safety performance, with an overall spread of approximately 13 percentage points. The first tier (Overall > 73%) comprises Kimi-K2-Thinking (74.93%), GPT-4o (73.64%), and GPT-5.2 (73.14%). The second tier (66–73%) encompasses eight models from Qwen2.5-72B-Instruct (71.26%) to Gemma-3-27BIT (66.66%). The third tier (< 66%) includes Qwen3-30B-A3B-Thinking (64.57%), Doubao-Seed-1.6-flash (62.21%), and GLM-4.5-Air (62.09%).

Dimension-Level Difficulty Spectrum. The nine risk dimensions exhibit dramatically different average difficulty levels. AIRD (mean 96.7%) and EvalFaking (mean 95.0%) are nearly saturated, indicating that current alignment techniques effectively address these risk categories. In the intermediate range, DeceptionBench (mean 75.1%), Sandbagging (mean 72.9%), and WMDP (mean 66.3%) show significant inter-model divergence. At the lower end, Mask (mean 46.0%) and especially Manipulation (mean only 18.8%, with five models falling below 5%) represent critical weak points that demand urgent attention.

Non-Transferability of Safety Advantages Across Dimensions. A critical observation from Table 4 is that no single model dominates across all safety dimensions. As indicated by the bold (highest) and underlined (lowest) entries in each column, safety advantages are clearly non-transferable across benchmarks:

- • Models with top overall scores may still expose critical risks on specific dimensions. For example, Kimi-K2-Thinking achieves the highest Overall score but records the lowest Manipulation score. Similarly, DeepSeek-R1 leads on Sandbagging yet scores the lowest on both AIRD and WMDP.
- • Conversely, lower-ranked models can excel on individual dimensions. GLM-4.5-Air ranks last in Overall but achieves the top score on AIRD. Doubao-Seed-1.6-flash ranks near the bottom overall yet also attains the top score on AIRD and performs competitively on BeHonest.
- • Notably, no single model holds the top position on more than two dimensions, and the bold entries in Table 4 are distributed across 8 different models, confirming that safety performance is highly dimension-specific, with strength on one benchmark providing no guarantee of robustness on others.

Dimension-level leaders are distributed across different model families, with no single model ranking first on more than two dimensions. This raises a natural question: what factors drive this cross-dimensional

divergence? We analyze this from three perspectives: model reasoning capabilities, temporal evolution of safety, and the trade-off between model efficiency and alignment.

- 3.3.2 Impact of Reasoning Capabilities on Frontier Safety

Our analysis reveals that reasoning-enabled models exhibit notable defensive deficiencies on two key safety metrics: “Manipulation” and “Sandbagging.” As shown in Figure 18, statistical results indicate that:

- • Reasoning-enabled models (e.g., Qwen3-30B-A3B-Thinking, DeepSeek-R1, Kimi-K2-Thinking) exhibit an extremely low “Manipulation” score distribution (mean = 11.6%, with lows down to 1.11%).
- • In contrast, non-reasoning models achieve markedly higher scores, with averages around 30%.

This marked negative correlation suggests a trade-off: while Chain-of-Thought (CoT) mechanisms enhance complex problem-solving, they concurrently provide the computational capacity for constructing deeper deceptive strategies. The lower safety scores imply that these models are not only more prone to circumventing safety audits but also exhibit a stronger tendency to conceal their capabilities.

Notably, this disadvantage is not universal. On EvalFaking, all three reasoning models exceed 94%, and on DeceptionBench, Kimi-K2-Thinking and DeepSeek-R1 tie for first at 96.67% (though Qwen3-30B-A3BThinking scores only 58.89%), suggesting that the safety risks introduced by reasoning capabilities may concentrate on specific attack modalities rather than uniformly undermining all dimensions.

Kimi-K2-ThinkingDoubao-Seed-1.6-flashDeepSeek-R1Claude-Sonnet-4.5Qwen3-30B-A3B-ThinkingGLM-4.5-Air GPT-5.2Gemini-3-FlashInternLM3-8BGemma-3-27B-ITQwen2.5-72B-InstructMistral-Small-24B-InstructLlama-3.3-70B-Instruct GPT-4o

0

5

10

15

20

25

30

35

40

ManipulationRiskScore(%)

Mean = 11.6%

Mean = 31.8%

| |
|---|

Reasoning Mode: ON Reasoning Mode: OFF

| |
|---|

Figure 18: Comparison of Manipulation scores between reasoning-enabled and non-reasoning models. Reasoning models exhibit systematically lower resistance to manipulation.

- 3.3.3 Temporal Evolution of Frontier Safety

The preceding section identifies the lower manipulation resistance of reasoning models, and 2025 coincides with the rapid proliferation of such architectures. We therefore examine whether the overall frontier safety landscape has deteriorated over time by tracing the temporal trajectory of manipulation resistance across model release dates.

- As shown in Figure 19, resistance to “Manipulation” declines significantly over time, revealing a clear trajectory of safety regression:

2024 to early 2025: relatively stable manipulation resistance. Models maintained high resistance to manipulation. GPT-4o (May 2024), Llama-3.3-70B-Instruct (Dec 2024), and Mistral-Small-24B-Instruct (Jan 2025) stabilize within the 30%–33% range, indicating that mainstream models during this period were difficult to induce into manipulative behaviors.

Mid to late 2025: precipitous drop in manipulation resistance. Newly released models experience a sharp decline, generally falling into the 1%–5% range (e.g., Kimi-K2-Thinking at 1.11%, DeepSeek-R1 at 2.22%). This period coincides with the widespread adoption of reasoning-enabled architectures, corroborating the trade-off identified in the preceding section.

Late-stage partial recovery (Dec 2025). Despite the overall downward trend, GPT-5.2 (23.33%) and Gemini3-Flash (27.78%) break the pattern of monotonic decline. Their scores are significantly higher than most

ManipulationResistanceScore(%)

Open Source

40

Closed Source

Trend

GPT-4o

Llama-3.3-70B-Instruct

30

Gemini-3-Flash GPT-5.2

20

10

DeepSeek-R1

High Manipulation Risk Zone (<10%)

Kimi-K2-Thinking

0

2024-H2 2025-H1 2025-H2 2026-H1

Release Date

Figure 19: Manipulation resistance over release dates.

contemporaneous 2025 models but remain below 2024 levels, suggesting that top laboratories may have reintroduced targeted safety interventions in later releases without fully restoring prior-generation manipulation resistance.

- 3.3.4 Efficiency–Alignment Trade-off in Honesty and Trustworthiness

The preceding two sections focus on manipulation-related risks. Here we turn to another category of frontier risk: honesty and trustworthiness. Specifically, we examine the relationship between model efficiency and alignment on honesty-related benchmarks.

- As shown in Figure 20, smaller open-source models (≤30B parameters) and Flash variants of closed-source models consistently demonstrate higher risks of deception and untruthful behavior on “MASK”, “BeHonest”, and “DeceptionBench”:

- • Open-source small models (≤30B). Models such as Mistral-Small-24B-Instruct, InternLM3-8B, Gemma-3-27B-IT, and Qwen3-30B-A3B-Thinking achieve average safety rates of 0.40 (MASK), 0.67 (DeceptionBench), and 0.56 (BeHonest), respectively—substantially lower than larger open-source models (>30B), which score 0.47, 0.80, and 0.66 on the same benchmarks.
- • Closed-source Flash variants. The degradation is even more pronounced among closed-source models. Flash versions record safety rates of 0.38 (MASK), 0.68 (DeceptionBench), and 0.58 (BeHonest), compared to 0.57, 0.83, and 0.70 for their non-Flash counterparts. The largest observed gap reaches 0.19 on the MASK benchmark.

Open 30B Open > 30B Closed w/ Flash

0.9

0.83

0.80

| |
|---|

0.8

Closed w/o Flash

0.70

0.7

0.68

SafetyRate

0.67

0.66

0.58 0.57

0.6

0.56

0.5

0.47

0.40

0.4

0.38

0.3

Mask DeceptionBench BeHonest

- Figure 20: Honesty and trustworthiness safety rates on MASK, DeceptionBench, and BeHonest, comparing small open-source models (≤30B) vs. larger models and Flash vs. non-Flash closed-source variants.

These results indicate that, across both open-source and closed-source ecosystems, models that are optimized for lightweight deployment and high-throughput inference tend to exhibit weaker performance on honestyrelated evaluations. In other words, there is a pronounced trade-off between prioritizing computational efficiency and maintaining robust honesty-related behavior.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 21: Separation Score between safe and harmful representations and MedHallu accuracy for different models in DeepScan X-Boundary analysis.

- 3.4 Joint Safety Evaluation and Diagnosis

- 3.4.1 Extreme Representation Separation Undermines Boundary Reasoning

DeepScan diagnostics reveal that excessively large geometric separation between safe and harmful representations can undermine semantic continuity in the latent space, ultimately degrading the model’s ability to make accurate boundary-level safety judgments. In DeepScan diagnostics of models such as Gemma-327B-IT, significant numerical anomalies are observed in the X-Boundary analysis. Figure 21 shows that the centroid distance between safe and harmful representations (Separation Score) reaches 2998.57, and the measured Euclidean distance between boundary samples and the safe centroid is also extremely large (3893.43). While large inter-class distances are generally assumed to benefit classification, the combination of these geometric extremes with the model’s relatively low performance on tasks requiring fine-grained semantic understanding—such as DeepSafe SALAD-Bench (71.93%) and DeepSafe MedHallu (39.87%)—suggests a different underlying issue.

Specifically, such extreme geometric separation indicates that the model does not construct a smooth decision boundary in latent space. Instead, it maximizes inter-class distance at the cost of sparsifying the representation space. This structure makes it difficult for the model to accurately judge complex instructions located near the boundary between safe and harmful content, due to insufficient local semantic resolution. The results indicate that an excessive emphasis on representation separation in DeepScan’s X-Boundary metric can disrupt semantic continuity in latent space, thereby negatively affecting fine-grained safety discrimination performance in DeepSafe.

- 3.4.2 Discrepancy Between Latent Disentanglement and Surface-Level Safety

Results from the SPIN coupling diagnostic indicate a non-trivial relationship between neuronal functional decoupling and end-to-end safety alignment performance.

As illustrated in Figure 22, GLM-4.5-Air achieves an superior coupling index for fairness and privacy neurons (-16.51), outperforming models like Llama-3.3-70B and Qwen2.5-72B, which have a higher overall DeepSafe safety score but a weaker DeepScan coupling index around –14.95. However, GLM-4.5-Air attains only 66.44% in DeepSafe’s overall text safety score, revealing a discrepancy between internal mechanism quality and external evaluation performance. This discrepancy suggests that while GLM-4.5-Air has successfully learned to encode fairness and privacy as more distinct and orthogonal representations in its latent space, this structural clarity has not yet been fully translated into behavioral constraints during the alignment stage.

By contrast, some models with high coupling index mask internal feature entanglement through strong supervised fine-tuning, achieving higher short-term safety evaluation scores. However, we caution that such entanglement may pose a latent risk of objective interference, where future adjustments to one safety attribute could inadvertently affect others due to shared neuronal pathways. From a long-term perspective,

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 22: Comparison of SPIN coupling index and overall DeepSafe safety performance for different models.

[Figure 35]

[Figure 36]

[Figure 37]

- Figure 23: Relationship between representation gap (R Gap) from DeepScan TELLME analysis and HarmBench score from DeepSafe, with a linear fit and 95% confidence interval.

we hypothesize that neuron-level functional decoupling might offer a more favorable optimization landscape, potentially reducing conflicts between competing safety objectives.

- 3.4.3 Orthogonal Subspace Encoding Enables Robust Defense

Subspace encoding rate and representational orthogonality emerge as key determinants of robustness against adversarial safety attacks. As shown in Figure 23, analysis based on the TELLME composite encoding rate in DeepScan shows that high-performing models such as Qwen2.5-72B-Instruct exhibit pronounced subspace structural advantages. Its composite encoding rate reaches 951.76, indicating that the model can compress safe and unsafe behaviors into distinctly separable low-dimensional subspaces with minimal inter-group interference noise.

This highly orthogonal subspace structure provides a stable basis for discrimination, enabling the model to maintain strong robustness under high-intensity attack evaluations such as DeepSafe HarmBench (86.97%). In contrast, InternLM3-8B achieves a much lower composite encoding rate (285.37) in DeepScan’s TELLME analysis and exhibits a higher effective rank (ERank), indicating a more dispersed internal representation distribution with overlapping behavior modes in latent space.

Such representational redundancy and entanglement reduce the model’s resistance to interference in complex contexts, leading to instability in safety boundaries under multi-turn dialogues or inducement-based attacks.

- 3.4.4 Low Separability Leads to Systematic Defense Failure

Insufficient separability between safe and harmful representation centroids directly contributes to systematic failures in safety defense. Figure 24 illustrates that Mistral-Small-24B-Instruct exhibits an extremely low safe–harmful separation score of 1.89 in DeepScan’s X-Boundary diagnostic. This geometric characteristic directly corresponds to its poor performance in DeepSafe evaluations with high attack success rates, such as Flames (26.74%).

A separation score this low implies that, in the model’s high-dimensional hidden space, safe and harmful samples have highly overlapping feature distributions, lacking support from either linear or nonlinear decision boundaries. Under such conditions, the model cannot reliably distinguish malicious prompts from benign instructions based on representational features alone.

The analysis suggests that, during training, such models fail to learn sufficiently discriminative risk-related features, leading to confusion between safe and harmful concepts in vector space. For representation structures diagnosed by DeepScan in this manner, simply adjusting inference-time thresholds or prompts cannot address the root cause. Instead, methods such as contrastive learning are required to reshape the representation space, increase inter-class distances, and optimize feature distributions.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

- Figure 24: Comparison of representation separation and Flames attack success rate across models in DeepScan and DeepSafe evaluations.

- 4 Related Work

- 4.1 Evaluation Frameworks

The landscape of LLM and MLLM evaluation has evolved from static benchmarks to programmable frameworks that support multi-turn interaction, tool usage, and flexible scoring. OpenAI Evals and Inspect exemplify lightweight evaluation harnesses with reusable templates (including model-graded judging), enabling rapid construction of task-specific tests while keeping data schemas and scoring protocols consistent across runs (OpenAI, 2023; UK AI Security Institute, 2024). Similarly, LightEval provides a lightweight, backend-agnostic pipeline that is commonly adopted in leaderboard-style evaluations (Habib et al., 2023). At larger scale, platforms such as OpenCompass, HELM, and the lm-evaluation-harness ecosystem emphasize standardized model/task interfaces, reproducibility, and reporting pipelines over broad benchmark collections (Contributors, 2023; Liang et al., 2022; Biderman et al., 2024). For multimodal models, VLMEvalKit extends these abstractions to MLLMs for multimodal benchmarks (Duan et al., 2024). Complementing these general-purpose frameworks, safety-focused suites such as Safety-Eval narrow evaluation to deploymentrelevant risk taxonomies (Allen Institute for AI, 2024). It pairs generative safety evaluations, including jailbreak and toxicity prompts derived from in-the-wild jailbreak corpora such as WildJailbreak and toxicity datasets such as ToxiGen (Jiang et al., 2024; Hartvigsen et al., 2022), with systematic assessment of safety classifiers and refusal detection, such as WildGuard (Han et al., 2024). Adversarial robustness is further evaluated with red-teaming oriented frameworks such as HarmBench (Mazeika et al., 2024).

- 4.2 Diagnosis Methods

Geometric representation analysis. Diagnostic research increasingly focuses on the geometry of the latent space to decode internal model states (Park et al., 2023; Marks & Tegmark, 2023). Early approaches utilize linear probing techniques to distinguish truthfulness and safety directions within activation spaces (Burns et al., 2022; Zou et al., 2023). Later, Gurnee & Tegmark (2023) reveal that models spontaneously acquire structured linear representations of space and time. To understand the temporal evolution of such qualities, Qian et al. (2024b) trace the dynamics of trustworthiness throughout the pre-training stage, whereas Zhang

- et al. (2024) utilize representation to uniquely identify and analyze model lineages.

Identifying task-related regions in (M)LLMs. Beyond representation analysis, significant efforts are dedicated to locating specific model regions responsible for distinct capabilities (Dang et al., 2024). Dai et al. (2022) pioneer the identification of knowledge neurons that store specific factual assertions, Wang et al. (2022) extend this concept to skill neurons that govern specific downstream tasks. At the attention level, Zheng et al. (2024) provide a comprehensive survey categorizing the diverse functional roles of attention heads across LLMs. In the context of safety, Wei et al. (2024) reveal the fragility of alignment by demonstrating that pruning or applying low-rank modifications to specific regions can easily compromise model safeguards. Zhao

- et al. (2025) pinpoint safety-specific neurons that explicitly manage refusal mechanisms. Furthermore, Yang
- et al. (2026a) demonstrate that reasoning capabilities are located in low-gradient-magnitude related weights.

Reasoning and generation dynamics analysis. This line of work mainly focuses on analyzing and understanding the underlying mechanisms and phenomena behind LLMs’ generation and reasoning processes (Hu et al., 2026; Gan et al., 2026). To demystify the internal mechanism of in-context learning, Wang et al. (2023a) characterize label words as anchors that aggregate semantic information from demonstrations to guide final predictions. Specific to reinforcement learning for reasoning, recent works analyze entropy-related phenomena during training and design optimized algorithms based on these insights (Wang et al., 2025a; Cui et al., 2025). Moving to agentic systems, Qian et al. (2026) propose a framework for agentic attribution to unveil the internal drivers behind actions, whereas Tang et al. (2026) utilizes Shapley value analysis to attribute and interpret extreme events arising from complex multi-agent systems.

- 5 Conclusion and Discussion

In this work, we introduce DeepSight, an open-source toolkit designed to bridge the critical gap between safety evaluation and model diagnosis. By integrating DeepSafe and DeepScan, we establish a closed-loop engineering paradigm that moves beyond black-box testing toward white-box insight. Through the extensive evaluation of various models across content and frontier risk domains, several critical insights have emerged regarding the current state of LM safety. By open-sourcing DeepSight, we aim to accelerate the community’s transition from reactive safety patching to proactive, verifiable safety engineering, ensuring that the development of Artificial General Intelligence remains beneficial and trustworthy.

References

Allen Institute for AI. Safety-eval: Ai2 safety tool (evaluation suite), 2024. URL https://github.com/ allenai/safety-eval. Software.

Anthropic. Claude sonnet 4.5 system card, 2025. URL https://www.anthropic.com/news/ claude-sonnet-4-5. Released September 29, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-vl technical report, 2025a. URL https://arxiv.org/ abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, et al. Lessons from the trenches on reproducible evaluation of language models. arXiv preprint arXiv:2405.14782, 2024.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models

without supervision. arXiv preprint arXiv:2212.03827, 2022. Bytedance. Seed1.6, 2025. URL https://seed.bytedance.com/en/seed1_6. Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi

Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Guanxu Chen, Dongrui Liu, Tao Luo, Lijie Hu, and Jing Shao. TELLME: Enhancing intrinsic transparency of LLMs by disentangling representations across behaviors. arXiv preprint arXiv:2502.05242, 2025. URL https://arxiv.org/abs/2502.05242.

Steffi Chern, Zhulin Hu, Yuqing Yang, Ethan Chern, Yuan Guo, Jiahe Jin, Binjie Wang, and Pengfei Liu. Behonest: Benchmarking honesty in large language models. arXiv preprint arXiv:2406.13261, 2024.

Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, et al. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733, 2025.

OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass, 2023.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, 2022.

Yunkai Dang, Kaichen Huang, Jiahao Huo, Yibo Yan, Sirui Huang, Dongrui Liu, Mengxi Gao, Jie Zhang, Chen Qian, Kun Wang, et al. Explainable and interpretable multimodal large language models: A comprehensive survey. arXiv preprint arXiv:2412.02104, 2024.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pp. 11198–11201, 2024.

Yihe Fan, Wenqi Zhang, Xudong Pan, and Min Yang. Evaluation faking: Unveiling observer effects in safety evaluation of frontier ai systems. arXiv preprint arXiv:2505.17815, 2025.

Zeyu Gan, Ruifeng Ren, Wei Yao, Xiaolin Hu, Gengze Xu, Chen Qian, Huayi Tang, Zixuan Gong, Xinhao Yao, Pengwei Tang, et al. Beyond the black box: Theory and mechanism of large language models. arXiv preprint arXiv:2601.02907, 2026.

Google. A new era of intelligence with gemini 3, 2025. URL https://blog.google/ products-and-platforms/products/gemini/gemini-3.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Wes Gurnee and Max Tegmark. Language models represent space and time. arXiv preprint arXiv:2310.02207, 2023.

Nathan Habib, Clémentine Fourrier, Hynek Kydlíček, Thomas Wolf, and Lewis Tunstall. Lighteval: A lightweight framework for llm evaluation, 2023. URL https://github.com/huggingface/lighteval.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. Advances in Neural Information Processing Systems, 37:8093–8131, 2024.

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. arXiv preprint arXiv:2203.09509, 2022.

Xuhao Hu, Dongrui Liu, Hao Li, Xuan-Jing Huang, and Jing Shao. Vlsbench: Unveiling visual leakage in multimodal safety. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8285–8316, 2025.

Yi Hu, Jiaqi Gu, Ruxin Wang, Zijun Yao, Hao Peng, Xiaobao Wu, Jianhui Chen, Muhan Zhang, and Liangming Pan. Towards a mechanistic understanding of large reasoning models: A survey of training, inference, and failures. arXiv preprint arXiv:2601.19928, 2026.

Kexin Huang, Xiangyang Liu, Qianyu Guo, Tianxiang Sun, Jiawei Sun, Yaru Wang, Zeyang Zhou, Yixu Wang, Yan Teng, Xipeng Qiu, et al. Flames: Benchmarking value alignment of llms in chinese. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4551–4591, 2024.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. Advances in Neural Information Processing Systems, 36:24678–24704, 2023.

Jiaming Ji, Wenqi Chen, Kaile Wang, Donghai Hong, Sitong Fang, Boyuan Chen, Jiayi Zhou, Juntao Dai, Sirui Han, Yike Guo, et al. Mitigating deceptive alignment via self-monitoring. arXiv preprint arXiv:2505.18807, 2025.

Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, et al. Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models. Advances in Neural Information Processing Systems, 37:47094–47165, 2024.

Shanghai AI Lab, Xiaoyang Chen, Yunhao Chen, Zeren Chen, Zhiyun Chen, Hanyun Cui, Yawen Duan, Jiaxuan Guo, Qi Guo, Xuhao Hu, et al. Frontier ai risk management framework in practice: A risk analysis technical report. arXiv preprint arXiv:2507.16534, 2025.

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. arXiv preprint arXiv:2305.11747, 2023.

Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. arXiv preprint arXiv:2402.05044, 2024a.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, AnnKathrin Dombrowski, Shashwat Goel, Long Phan, et al. The wmdp benchmark: Measuring and reducing malicious use with unlearning. arXiv preprint arXiv:2403.03218, 2024b.

Xirui Li, Hengguang Zhou, Ruochen Wang, Tianyi Zhou, Minhao Cheng, and Cho-Jui Hsieh. Mossbench: Is your multimodal language model oversensitive to safe queries? arXiv preprint arXiv:2406.17806, 2024c.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. In European Conference on Computer Vision, pp. 386–403. Springer, 2024.

Xiaoya Lu, Dongrui Liu, Yi Yu, Luxin Xu, and Jing Shao. X-boundary: Establishing exact safety boundary to shield LLMs from jailbreak attacks without compromising usability. arXiv preprint arXiv:2502.09990,

2025. URL https://arxiv.org/abs/2502.09990. Matt MacDermott, Qiyao Wei, Rada Djoneva, and Francis Rhys Ward. Reasoning under pressure: How do training incentives influence chain-of-thought monitorability? arXiv preprint arXiv:2512.00218, 2025. Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, 2025. URL

https://ai.meta.com/blog/llama-4-multimodal-intelligence/. Mistral. Mistral small 3, 2025. URL https://mistral.ai/news/mistral-small-3/. OpenAI. Evals: A framework for evaluating LLMs and LLM systems, 2023. URL https://github.com/

openai/evals. Software. OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. Shrey Pandit, Jiawei Xu, Junyuan Hong, Zhangyang Wang, Tianlong Chen, Kaidi Xu, and Ying Ding.

Medhallu: A comprehensive benchmark for detecting medical hallucinations in large language models. arXiv preprint arXiv:2502.14302, 2025.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023.

Chen Qian, Dongrui Liu, Jie Zhang, Yong Liu, and Jing Shao. The tug of war within: Mitigating the fairness-privacy conflicts in large language models. arXiv preprint arXiv:2410.16672, 2024a. URL https: //arxiv.org/abs/2410.16672.

Chen Qian, Jie Zhang, Wei Yao, Dongrui Liu, Zhenfei Yin, Yu Qiao, Yong Liu, and Jing Shao. Towards tracing trustworthiness dynamics: Revisiting pre-training period of large language models. arXiv preprint arXiv:2402.19465, 2024b.

Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. Demystifying reasoning dynamics with mutual information: Thinking tokens are information peaks in LLM reasoning. arXiv preprint arXiv:2506.02867, 2025. URL https://arxiv.org/abs/2506.02867.

Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, et al. The why behind the action: Unveiling internal drivers via agentic attribution. arXiv preprint arXiv:2601.15075, 2026.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, et al. The mask benchmark: Disentangling honesty from accuracy in ai systems. arXiv preprint arXiv:2503.03750, 2025.

Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5377–5400, 2024.

Zhelun Shi, Zhipin Wang, Hongxing Fan, Zaibin Zhang, Lijun Li, Yongting Zhang, Zhenfei Yin, Lu Sheng, Yu Qiao, and Jing Shao. Assessment of multimodal large language models in alignment with human values. arXiv preprint arXiv:2403.17830, 2024.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Ling Tang, Jilin Mei, Dongrui Liu, Chen Qian, Dawei Cheng, Jing Shao, and Xia Hu. Interpreting emergent extreme events in multi-agent systems. arXiv preprint arXiv:2601.20538, 2026.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025a.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang,

Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025b. Moonshot AI Team. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. V Team, Wenyi Hong, Wenmeng Yu, et al. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal

reasoning with scalable reinforcement learning, 2026. URL https://arxiv.org/abs/2507.01006. UK AI Security Institute. Inspect AI: Framework for Large Language Model Evaluations, May 2024. URL https://github.com/UKGovernmentBEIS/inspect_ai.

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. arXiv preprint arXiv:2305.14160, 2023a.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025a.

Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, and XuanJing Huang. Safe inputs but unsafe output: Benchmarking cross-modality safety alignment of large vision-language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 3563–3605, 2025b.

Xiaozhi Wang, Kaiyue Wen, Zhengyan Zhang, Lei Hou, Zhiyuan Liu, and Juanzi Li. Finding skill neurons in pre-trained transformer-based language models. arXiv preprint arXiv:2211.07349, 2022.

Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, and Yingchun Wang. Fake alignment: Are llms really aligned well? In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4696–4712, 2024.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. Do-not-answer: A dataset for evaluating safeguards in llms. arXiv preprint arXiv:2308.13387, 2023b.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36:80079–80110, 2023.

Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, and Peter Henderson. Assessing the brittleness of safety alignment via pruning and low-rank modifications. arXiv preprint arXiv:2402.05162, 2024.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/ abs/2505.09388.

Junyao Yang, Chen Qian, Dongrui Liu, Wen Shen, Yong Liu, and Jing Shao. Reasonany: Incorporating reasoning capability to any model via simple and effective model merging. arXiv preprint arXiv:2601.05560,

- 2026a.

Xiaofang Yang, Lijun Li, Heng Zhou, Tong Zhu, Xiaoye Qu, Yuchen Fan, Qianshan Wei, Rui Ye, Li Kang, Yiran Qin, et al. Toward efficient agents: Memory, tool learning, and planning. arXiv preprint arXiv:2601.14192,

- 2026b.

Shaohan Yu, Lijun Li, Chenyang Si, Lu Sheng, and Jing Shao. Proguard: Towards proactive multimodal safeguard. arXiv preprint arXiv:2512.23573, 2025.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Jie Zhang, Dongrui Liu, Chen Qian, Linfeng Zhang, Yong Liu, Yu Qiao, and Jing Shao. Reef: Representation encoding fingerprints for large language models. arXiv preprint arXiv:2410.14273, 2024.

Jintian Zhang, Xin Xu, Ningyu Zhang, Ruibo Liu, Bryan Hooi, and Shumin Deng. Exploring collaboration mechanisms for llm agents: A social psychology view. arXiv preprint arXiv:2310.02124, 2023.

Yiran Zhao, Wenxuan Zhang, Yuxi Xie, Anirudh Goyal, Kenji Kawaguchi, and Michael Shieh. Understanding and enhancing safety mechanisms of llms via safety-specific neuron. In The Thirteenth International Conference on Learning Representations, 2025.

Zifan Zheng, Yezhaohui Wang, Yuxin Huang, Shichao Song, Mingchuan Yang, Bo Tang, Feiyu Xiong, and Zhiyu Li. Attention heads of large language models: A survey. arXiv preprint arXiv:2409.03752, 2024.

Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Anderson Compalas, Dawn Song, and Xin Eric Wang. Multimodal situational safety. arXiv preprint arXiv:2410.06172, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. URL https://arxiv.org/abs/2504.10479.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

