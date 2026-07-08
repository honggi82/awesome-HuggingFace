# arXiv:2603.24649v2[cs.CV]13May2026

## MEDOPENCLAW and MEDFLOW-BENCH: Auditing Medical Agents in Full-Study Workflows

Weixiang Shen1,2,3 Chengzhi Shen1,2 Yanzhu Hu1,2 Che Liu4 Junde Wu5 Jiayuan Zhu5 Xiao Han3 Zongyue Li3,9 Jingpei Wu3,9 Min Xu6 Daguang Xu7 Yueming Jin8 Benedikt Wiestler1,2,9,† Daniel Rueckert1,2,4,9,† Jiazhen Pan1,2,9,†

1Technical University of Munich (TUM) 2TUM University Hospital 3LMU Munich 4Imperial College London 5University of Oxford 6Carnegie Mellon University 7NVIDIA 8National University of Singapore 9Munich Center for Machine Learning †Corresponding authors

### Abstract

Medical imaging benchmarks often evaluate VLMs on pre-selected 2D images, slices, crops, or patches, making evaluation closer to visual recognition. Real clinical workflows impose a different burden: readers must search through complete studies, operate imaging software, navigate across slices and magnifications, and document visual evidence that can be audited. We argue that this evidenceproducing workflow is a critical missing evaluation axis for medical imaging agents. To study it, we introduce MedFlowBench, a full-study benchmark for VLM agents, together with MedOpenClaw, a controlled and replayable runtime in which agents operate medical imaging viewers such as 3D Slicer and QuPath. In each episode, an agent inspects a complete radiology study or whole-slide pathology image, returns a task answer, and submits structured evidence, including key slices, coordinates, regions of interest, or lesion-state fields. This evidence is automatically checked against withheld masks, annotations, and labels. Across evaluated models, final answer-only scoring gives an overly optimistic picture: when answers must also be supported by correct evidence, performance drops substantially on complex workflows. We further find that adding image-analysis tools does not by itself solve the problem. Tools help when they make a complex procedure simple and reliable, but agents still struggle when they must choose inputs, manage viewer state, and verify intermediate outputs over multiple steps. MedFlowBench exposes whether medical imaging agents can produce auditable evidence from complete studies, rather than plausible answers from selected images.

### 1 Introduction

Medical vision-language models have progressed rapidly, from early medical VQA benchmarks [1, 2] to more recent medical multimodal systems and expert-level evaluation sets [3–5]. Yet much of their medical imaging evaluation still relies on a simplified proxy setting: the model is given one or a few pre-selected diagnostically relevant 2D images and asked to answer a localized question. This setup is useful for testing recognition on selected inputs, but it removes the central difficulty of medical imaging, especially radiology, where findings must often be found across a complete volumetric study. A parallel abstraction appears in pathology when whole-slide images are reduced to selected patches or slide-level labels. It also keeps the decision process largely opaque: the model returns an answer, but not a replayable account of where it looked, what evidence it gathered, or how the final conclusion was reached.

This creates a substantial gap to real clinical workflow [6–8]. In practice, medical imaging analysis is a study-level and software-mediated process. A reader must inspect a full 3D examination, choose relevant series or modalities, navigate across many slices, adjust display settings such as windowing or fusion, compare evidence across views and timepoints, and often perform measurements, registration, segmentation, or specialized analysis before committing to an interpretation. In whole-slide pathology,

the corresponding workflow requires opening a gigapixel slide, panning and zooming across tissue regions, and recording evidence at appropriate coordinates and magnifications. Many clinically relevant findings are not visible in a single image. They emerge only across adjacent slices, across sequences, across longitudinal studies, or after the relevant region has been localized [9–11]. A meaningful evaluation setting should therefore test not only whether an agent can produce the correct answer, but whether it can search complete studies, maintain the relevant viewer state, and leave evidence that is transparent, replayable, and auditable.

To evaluate this setting, we introduce MEDOPENCLAW and MEDFLOW-BENCH. MEDOPENCLAW is an auditable runtime that lets a backbone VLM operate standard medical imaging viewers, including 3D Slicer [12] for volumetric radiology and QuPath [13] for whole-slide pathology. It restricts the agent to predefined software actions for viewer navigation, evidence capture, and selected analysis operations, and records each step as a replayable trace. Built on this runtime, MEDFLOW-BENCH evaluates full-study medical imaging episodes rather than pre-selected inputs. It covers five task families across radiology and pathology: brain tumor subtype assessment on pre-operative multisequence MRI, longitudinal treatment-response assessment on MRI, lung PET/CT analysis, breast histology classification, and lymph-node metastasis assessment on whole-slide images. These tasks span clinically meaningful workflows that require lesion localization, integration of multimodal or lesion-status labels for longitudinal MRI tasks, gigapixel slide navigation, and structured evidencegrounded reasoning.

In MEDFLOW-BENCH, each agent must inspect a complete study and return both a task answer and task-specific structured evidence. This evidence is not a free-text rationale; it consists of fields that tie the answer back to image content, such as key-slice positions, lesion or slide coordinates, regions of interest, or lesion-status labels for longitudinal MRI tasks. The benchmark checks these fields with deterministic rules using withheld ground-truth masks, annotations, or labels. This design evaluates whether an agent can not only produce the correct answer, but also identify and document the image evidence needed to support it.

Our results show that full-study evaluation is feasible for VLM agents: models can navigate medical viewers, inspect complete volumetric or whole-slide cases, and solve nontrivial study-level tasks without pre-selected inputs. However, final-answer-only scoring substantially overestimates workflow competence in realistic imaging workflows. When the answer must also be supported by correct evidence, scores drop markedly on complex tasks. These drops motivate evidence-grounded evaluation as a crucial benchmark requirement: medical imaging agents should be evaluated not only on task accuracy, but also on whether they provide correct, checkable evidence.

We further test whether specialized image-analysis operations inside the viewer close this evidencegrounding gap. They do not by themselves. Operations that collapse a complex procedure into a single well-scoped call can help; for example, MONAI-based VISTA3D [14, 15] provides automated volumetric segmentation and can improve localization in some PET/CT cases. In contrast, workflows that require the agent to compose several dependent software steps, such as interactive segmentation and registration, remain brittle. The resulting errors are software-workflow failures rather than only visual-recognition failures: agents may lose track of image space, bind outputs to the wrong viewer state, choose inappropriate parameters, or fail to verify the generated artifact. These failures motivate evidence-grounded full-study benchmarks.

Our contributions are:

- • We introduce MEDOPENCLAW, an auditable runtime that enables VLM agents to operate professional medical software, including 3D Slicer and QuPath, on full studies rather than pre-selected diagnostically relevant inputs.
- • We introduce MEDFLOW-BENCH, a full-study benchmark for medical imaging reasoning across radiology and pathology. MEDFLOW-BENCH evaluates not only final answers but also checkable evidence, including key slices, coordinates, regions, and longitudinal evidence.
- • Using MEDFLOW-BENCH, we show that current VLM agents can operate medical imaging software and solve some full-study radiology and pathology tasks without a pre-selection process. However, evaluating only the final answer overestimates their workflow competence. When evaluation also requires supporting evidence, performance drops markedly on complex tasks. We further show that access to specialized image-analysis tools is not sufficient for workflow competence. Automation tools can help when they reduce intermediate decision-making, but multi-step segmentation and registration workflows remain unreliable.

[Figure 1]

Figure 1: Overview of MEDOPENCLAW and MEDFLOW-BENCH. Left: Conventional medical VQA benchmarks reduce evaluation to pre-selected 2D images and black-box final answers. Right: MEDOPENCLAW enables backbone models to operate professional medical software through bounded software actions, producing visible traces, auditable evidence, and grounded answers. MEDFLOW-BENCH evaluates this full-study interaction loop across radiology and whole-slide pathology workflows, including viewer-native navigation and specialized segmentation or registration functions.

### 2 Related Work

Static medical VQA and medical VLM benchmarks. Early medical VQA datasets established language-conditioned evaluation on medical images [1, 16, 17, 2, 18]. Subsequent medical QA and multimodal benchmarks broadened task scope, reasoning difficulty, and clinical coverage [3–5, 19– 22]. Related lines such as medical report generation and image-grounded medical question answering also largely assume fixed image inputs rather than study-level interaction [23–26]. Together, these benchmarks measure visual recognition, medical knowledge use, and language-conditioned reasoning on selected inputs, but they typically start from one or a few pre-selected diagnostic views rather than a full imaging study. Patch- or slide-level pathology evaluations similarly reduce the search problem before the agent must navigate the whole slide. We target this gap by evaluating auditable study-level reasoning over complete radiology and pathology studies.

Medical agents. Another line of work studies medical agents that perform multi-step reasoning [27– 30] and evidence gathering [31–33]. Some focus on radiology-oriented agents, especially for chest X-ray analysis or reporting [23, 34], while others study broader multimodal medical agents that select among specialized tools or APIs across tasks [35, 36]. These systems make the reasoning loop more explicit than standard static benchmarks, but many still operate on fixed images, isolated APIs, or abstracted tool interfaces rather than continuous interaction with a viewer over a full tomographic exam.

Full-study and interactive medical imaging systems. More closely related work begins to address study-level or interactive radiology settings. One line studies volumetric or 3D medical image reasoning with language-guided analysis over image volumes [37–39]. Another line evaluates agents in simulated or simplified radiology environments [40, 41]. A third line explores natural-language assistance or copilot-style interaction inside existing imaging software such as 3D Slicer [42, 43]. These directions move closer to clinical imaging workflow, but differ in whether the main emphasis is on volumetric reasoning, environment design, or software-integrated interaction. Our work sits at the intersection of these threads by focusing on study-level evaluation in a real viewer with preserved execution traces.

General-purpose agent runtimes. At the systems level, MEDOPENCLAW is related to realenvironment agent benchmarks for software engineering, desktop use, and tabular tasks [44–46], as well as general-purpose runtimes such as OpenClaw [47–50]. Unlike open-ended agent runtimes, MEDOPENCLAW builds bounded control into the medical-imaging interface itself: agents are re-

stricted to software-native actions and vetted advanced operations under a standardized protocol. This makes traces safer, replayable, and aligned with clinical evidence requirements.

### 3 MEDOPENCLAW Runtime

MEDOPENCLAW defines the software-native interaction contract used by MEDFLOW-BENCH. It sits between a backbone VLM agent and standard medical imaging environments, including 3D Slicer [12] for volumetric radiology studies and QuPath [13] for whole-slide pathology images. We use these environments because full-study evidence is naturally represented through software state: selected series, slice index, viewport, windowing or fusion settings, Right-Anterior-Superior (RAS) coordinates, WSI coordinates, masks, measurements, segmentations, registrations, and saved artifacts. These forms of evidence are difficult to represent or evaluate from static selected images alone. MEDOPENCLAW runs externally without modifying the viewer source code and exposes a fixed interface for study inspection, evidence capture, and analysis. Through this interface, the agent can select series, scroll through slices, adjust display settings, bookmark views, take measurements, pan and zoom whole-slide images, convert viewer points into image coordinates, and export evidence.

- Figure 1 summarizes this contract.

The exposed action space is grouped into three roles: Viewer Control, Evidence Capture, and Analysis Operators. Viewer Control actions manipulate the visible study state, such as series selection, slice scrolling, panning, zooming, windowing, and basic foreground/background fusion of already loaded or prealigned volumes. Evidence Capture actions save checkable evidence from the current state, including bookmarked views, key-slice or coordinate locations, drawn regions or masks, and measurement logs. Analysis Operators invoke functions that create derived study state, such as segmentation, registration, resampling, quantitative analysis, or automated expert-model outputs. MONAI-based operation packs [14, 51] and 3D VISTA [15] are exposed as Analysis Operators through 3D Slicer. Section 4 describes how MEDFLOW-BENCH maps these actions to task protocols and scoring rules.

Implementation details of the REST interface, named bridge handlers, QuPath operation registry, and safety restrictions are provided in Appendix B. The bounded design makes auditability operational rather than implicit. The runtime logs each action invocation together with its arguments, the resulting viewer-state snapshot, and any generated artifacts. These records allow the execution to be reconstructed after the fact, including which views were accessed, which operations were executed, and what evidence was available when the final answer was produced.

Although MEDFLOW-BENCH uses MEDOPENCLAW for autonomous agent evaluation, the same traceable interaction contract can also support human-in-the-loop interfaces. We implement MEDCOPILOT as an example application built on top of MEDOPENCLAW, where generated actions and evidence artifacts remain visible for clinician inspection. We treat this as a demonstration of traceable interaction rather than evidence of clinical deployment or workflow-efficiency gains.

### 4 MEDFLOW-BENCH: Study-Level Evaluation

Building on the runtime contract above, MEDFLOW-BENCH defines the evaluation distribution: the clinical task families, allowed tracks, canonical answer schemas, and evidence-based scoring rules. MEDFLOW-BENCH evaluates study-level and whole-slide reasoning rather than perception from pre-selected 2D images. It is designed around full-study interactive access, cross-modality and longitudinal reasoning, active evidence acquisition, differential diagnosis, and agentic execution in professional medical software. A benchmark episode is defined at the study or slide level rather than the image-patch level. Each episode specifies (i) a case package containing the full volumetric exam or whole-slide image and metadata, (ii) the software environment used by the agent, (iii) a task prompt that asks for a case-level or slide-level decision, (iv) an allowed action space determined by the evaluation track, and (v) a canonical answer and evidence schema used for scoring.

Example benchmark episode. As a concrete example, Figure 2 illustrates abbreviated, decisionrelevant execution traces from the Longitudinal MRI and Breast WSI modules. In the longitudinal MRI scenario, the agent loads the BASELINE and FOLLOW-UP studies, registers the follow-up CT1 volume to the baseline CT1 volume, inspects fused aligned views across timepoints, and reports a RANO response label with FOLLOW-UP-study evidence fields. In the breast WSI scenario, the agent

[Figure 2]

- Figure 2: Representative auditable execution traces from MEDOPENCLAW in brain MRI and breast WSI workflows. Top: For longitudinal brain MRI, the agent loads baseline and followup studies, registers the week-053 scan to week-045, inspects fused aligned volumes, and reports RANO [52] progressive disease. Bottom: For breast H&E WSI, the agent screens the slide, reviews the dominant epithelial lesion at higher magnification, maps the diagnostic field to a whole-slide coordinate, and reports atypical ductal hyperplasia. The traces are compressed from longer runtime logs; full records retain operation calls, arguments, viewer states, image evidence, locations, and final outputs for auditability. screens the slide, zooms into a diagnostic epithelial region, converts the viewed field to whole-slide coordinates, and returns the slide-level diagnosis with coordinate-grounded evidence. Across both examples, the query, operation calls, visual outputs, and response remain externally inspectable instead of collapsing into an opaque intermediate state.

Task families and evidence. MEDFLOW-BENCH covers five challenging diagnostic imaging tasks across radiology and pathology: brain tumor subtype assessment on pre-operative MRI, longitudinal treatment-response assessment on brain MRI, lung PET/CT analysis, breast histology classification, and lymph-node metastasis WSI assessment. These tasks were chosen to span representative, clinically meaningful workflows in which diagnosis requires localization, integration of multimodal or longitudinal evidence, and structured clinical reasoning. In radiology, pre-operative brain tumor MRI emphasizes intracranial lesion localization, cross-sequence integration, anatomical reasoning,

and tumor subtype assessment [53], while longitudinal response assessment requires RANO-style comparison of baseline and follow-up studies, including tracking target and non-target lesions over time [54]. Lung PET/CT broadens the radiology setting to thoracic oncology, requiring joint interpretation of metabolic (PET) and anatomical (CT) findings for tumor localization, stagingrelevant assessment, histology, and grade [55]. In pathology, the WSI tasks evaluate gigapixel slide analysis with localized diagnostic evidence: breast WSI covers benign, atypical, and malignant breast lesion patterns [56], while lymph-node WSI focuses on metastasis detection and nodal-status assessment within TNM staging, a clinically important component of cancer staging [57]. Across all tasks, agents must operate professional medical imaging software and provide structured, verifiable evidence whenever required by the task definition.

The Three-Track Design. To support both model evaluation and systems work, MEDFLOWBENCH separates the solution space into three tracks rather than mixing all methods on one leaderboard. All tracks use identical cases, task formulations, and metrics. Conceptually, Track C is a diagnostic compatibility track, whereas Track B is the main software-use intervention.

- • Track A: Viewer-Native. A test of viewer-native study inspection without specialized analysis functions. Aligning with the first layer of our runtime architecture, methods use MEDOPENCLAW to drive the relevant medical viewer using only primitive viewer-native actions, such as series selection, scrolling, windowing, and basic prealigned foreground/background fusion in 3D Slicer, or slide loading, panning, zooming, and viewport capture in QuPath. By excluding specialized analysis functions, this track focuses on visual search, slice-to-slice or region-to-region synthesis, and evidence acquisition rather than configuring specialized functions.
- • Track B: Advanced Operations. The main systems track, allowing access to advanced operations via MEDOPENCLAW where such operations are defined. Here, advanced operations refer to specialized functions beyond viewer-native inspection that require the agent to invoke, configure, and verify algorithm- or model-backed functionality inside professional medical imaging software, including segmentation operators, registration workflows, and automated expert-model workflows such as 3D VISTA. This track tests whether a model can decide when such an operation is needed, set parameters, preserve the returned artifacts in software state, and integrate them back into the diagnostic trajectory.
- • Track C: Runtime-agnostic. Methods are not required to use MEDOPENCLAW and may use any alternative pipeline that consumes the raw cases and outputs the canonical answer and evidence schema. We include this track to ensure the benchmark remains a universal standard rather than just a test of our specific runtime. This leaves room for future full-study paradigms.

Scoring metrics. We report three metric types. Task is the module primary answer score. For LUMIERE, UCSF-PDGM, and BRACS, Task is single-label accuracy. For CAMELYON17, Task is exact-match accuracy over tumor presence and largest-metastasis category. For NSCLC, Task is the partial score #correct/5 over tumor location, pathological T stage, pathological N stage, histology, and grade. Strict is the evidence-constrained version of Task: if the module-specific evidence gate passes, Strict equals Task; otherwise Strict is 0. Evidence gates are deterministic and may use key-slice offsets, source-series provenance, lesion-status fields, RAS points, or WSI coordinates depending on the module. Loc. is reported separately when spatial ground truth is available and measures coordinate localization under the module-specific denominator.

### 5 Experiments

All methods are evaluated under the task-specific MCQ or structured-answer protocol and the evidence requirements defined in Section 4. Unless otherwise noted, the three tracks use identical cases, task formulations, and metrics.

Viewer-Native Radiology Baselines. Table 1 reports the main 3D Slicer viewer-native results, with lesion-state and subtask-level details in Appendix Table 5. Overall, the table shows that current models still have limited ability to operate full-study radiology software with reliable evidence. Brain MRI is the most favorable setting: because the target tumor is relatively spatially salient, frontier proprietary models already achieve high localization and evidence-grounded performance, with Gemini 3.1 Pro Preview reaching 0.73 Task, 0.73 Strict, and 0.96 Loc. However, performance drops sharply in workflows that require longitudinal comparison or precise localization of smaller

Longitudinal MRI Brain MRI Lung PET/CT Model Avg. Task Strict Task Strict Loc. Task Strict Loc. Task Strict Loc.

Model

###### General-purpose models

- GPT-5.4 mini [58] 0.23 0.05 0.38 0.18 0.37 0.31 0.00 0.00 0.31 0.08 0.19

- GPT-5.5 [59] 0.33 0.04 0.71 0.66 0.91 0.55 0.03 0.07 0.53 0.24 0.49 Gemini 3 Flash Preview [60] 0.30 0.07 0.67 0.59 0.87 0.45 0.13 0.26 0.47 0.26 0.57 Gemini 3.1 Pro Preview [61] 0.37 0.08 0.73 0.73 0.96 0.50 0.14 0.21 0.53 0.32 0.59 Claude Sonnet 4.6 [62] 0.33 0.17 0.38 0.18 0.36 0.26 0.00 0.00 0.32 0.12 0.18 Claude Opus 4.7 [63] 0.28 0.18 0.37 0.11 0.68 0.43 0.01 0.01 0.36 0.10 0.35 Qwen3.5-35B-A3B [64] 0.33 0.18 0.53 0.15 0.11 0.35 0.01 0.01 0.40 0.11 0.06 Qwen3.5-27B [64] 0.21 0.09 0.53 0.27 0.36 0.37 0.00 0.00 0.37 0.12 0.18 Qwen3.5-9B [64] 0.17 0.08 0.35 0.05 0.06 0.16 0.01 0.01 0.23 0.05 0.04 Qwen3.5-4B [64] 0.22 0.11 0.32 0.06 0.03 0.20 0.00 0.00 0.25 0.06 0.02 Gemma 3 27B IT [65] 0.32 0.01 0.33 0.11 0.07 0.35 0.00 0.00 0.33 0.04 0.03 Mean over reported 0.28 0.10 0.48 0.28 0.43 0.36 0.03 0.05 0.37 0.14 0.25 Medical-domain models MedGemma 1.5 4B IT [66] 0.16 0.02 0.45 0.09 0.00 0.16 0.00 0.00 0.26 0.04 0.00 MedGemma 27B IT [66] 0.11 0.00 0.30 0.00 0.03 0.16 0.00 0.00 0.19 0.00 0.02 Lingshu-32B [67] 0.16 0.00 0.36 0.08 0.02 0.31 0.00 0.00 0.27 0.03 0.01 Mean over reported 0.14 0.01 0.37 0.06 0.02 0.21 0.00 0.00 0.24 0.02 0.01

Table 1: Main viewer-native radiology results. All rows use 3D Slicer viewer-native access. Yellow columns report Strict, the answer-plus-evidence score defined in the scoring protocol: credit is given only when the task answer is correct and the required evidence fields pass the task-specific deterministic checks; blue columns report localization where available. Model Avg. reports unweighted averages over task-family metrics: Task averages Longitudinal Task, Brain Task, and Lung PET/CT Task; Strict averages the three Strict columns; Loc. averages Brain MRI and Lung PET/CT localization. Detailed lesion-status and subtask-level metrics are reported in Appendix Table 5. Rows are grouped by model category, distinguishing general-purpose and medical-domain models.

thoracic lesions. On Longitudinal MRI, even the best Strict score is only 0.18, indicating that correct response assessment with valid evidence remains difficult. On Lung PET/CT, the best Strict score is only 0.14, and the best localization score is 0.26, showing that models often fail to provide the precise RAS evidence needed for small-lesion thoracic oncology cases. Medical-domain models and smaller open-weight models are weaker still: their final-task scores are modest, and their Strict and localization scores are often near zero. Thus, viewer-native access enables some full-study reasoning, especially for salient Brain MRI findings, but reliable software-state navigation, evidence acquisition, and localization remain unresolved in clinically harder radiology workflows.

Viewer-Native Pathology Baselines. Table 2 reports viewer-native WSI results under the same auditable-evidence requirement used throughout the benchmark. In pathology, the relevant software-native workflow is whole-slide search in QuPath, where the agent must identify a diagnostic region rather than only produce a slide-level label. The results follow the same high-level pattern as radiology: models can produce plausible case-level answers, but evidence and coordinate localization remain weak. This supports the generality of software-native evaluation while showing that verifiable diagnostic-region grounding in WSI remains unresolved. Scoring definitions are provided in Appendix C.

Breast WSI

Lymph-node WSI

Model

Task Strict. Loc. Task Strict. Loc.

GPT-5.4 mini 0.20 0.04 0.03 0.16 0.05 0.39 GPT-5.4 0.16 0.10 0.07 0.35 0.11 0.19 Gemini 3 Flash Preview 0.20 0.16 0.13 0.18 0.05 0.07 Gemini 3.1 Pro Preview 0.34 0.15 0.19 0.26 0.12 0.38

Table 2: Viewer-native pathology. Coordinategrounded QuPath WSI scoring; Strict denotes answer-plus-evidence score and Loc. denotes coordinate localization where required.

Track C: Runtime-agnostic. Track C asks whether the viewer-native findings are merely an artifact of using MEDOPENCLAW rather than a more general property of full-study evaluation. The full comparison is reported in Appendix D, Table 6. The 2D slice-montage baseline compresses each volume into a static tiled overview, as illustrated in Appendix Figure 3, while native 3D VLMs such as M3D and RadFM consume volumetric inputs directly [68, 69]. The results change some final-answer scores but do not yield a consistently stronger pipeline. In particular, 2D slice montages are poorly suited to certain tasks, such as longitudinal comparison, because they remove interactive cross-timepoint navigation and fusion; they also do not scale naturally to more advanced imaging

Segmentation Registration 3D VISTA (MONAI) Longitudinal MRI Brain MRI Lung PET/CT Longitudinal MRI Lung PET/CT

Backbone Metric

###### Gemini 3 Flash Preview

Task 0.28 (↓ 0.02) 0.65 (↓ 0.02) 0.47 (↑ 0.02) 0.28 (↓ 0.05) 0.50 (↑ 0.05) Strict 0.05 (↓ 0.02) 0.56 (↓ 0.03) 0.09 (↓ 0.04) 0.07 (↔ 0.00) 0.13 (↔ 0.00) Loc. — 0.85 (↓ 0.02) 0.24 (↓ 0.02) — 0.50 (↑ 0.24)

###### Gemini 3.1 Pro Preview

Task 0.34 (↓ 0.03) 0.79 (↑ 0.06) 0.56 (↑ 0.06) 0.35 (↓ 0.01) 0.61 (↑ 0.11) Strict 0.09 (↑ 0.01) 0.66 (↓ 0.07) 0.20 (↑ 0.06) 0.06 (↔ 0.00) 0.22 (↑ 0.08) Loc. — 0.87 (↓ 0.09) 0.36 (↑ 0.15) — 0.48 (↑ 0.27)

###### Qwen3.5-35B-A3B

Task 0.23 (↓ 0.10) 0.65 (↑ 0.12) 0.41 (↑ 0.06) 0.30 (↑ 0.03) 0.44 (↑ 0.09) Strict 0.12 (↓ 0.06) 0.21 (↑ 0.06) 0.00 (↓ 0.01) 0.21 (↑ 0.08) 0.01 (↔ 0.00) Loc. — 0.13 (↑ 0.02) 0.00 (↓ 0.01) — 0.36 (↑ 0.35)

- Table 3: Track B: Advanced Operations. Each cell reports the score after enabling the specified advanced operation and its change relative to the Table 1 viewer-native score for the same backbone, except Registration on Longitudinal MRI, which is compared with an additional unregistered LUMIERE control. Green, red, and gray cells indicate improvement, degradation, and no change, respectively. Task denotes final task accuracy, Strict denotes task-plus-evidence correctness, and Loc. denotes localization where available.

analyses such as segmentation or registration. At the aggregate level, frontier models such as GPT-5.5 and Gemini 3.1 Pro Preview operating 3D Slicer substantially outperform native medical 3D VLM baselines such as M3D and RadFM on model-averaged Task, Strict, and localization metrics. Thus, Track C supports a narrow but important conclusion: MEDFLOW-BENCH is not tied to one runtime interface, and the evidence-grounding bottleneck persists across runtime-agnostic alternatives. We report the full table in the appendix.

Track B: Advanced Operations. Having established that bypassing the runtime does not remove the evidence-grounding bottleneck, Track B tests the more consequential systems question: whether richer software operations help agents execute evidence-grounded imaging workflows. Table 3 separates clinical utility from agent executability. Segmentation and registration are valuable operations in real imaging workflows, but exposing them to current agents does not reliably improve benchmark performance: their gains are local and inconsistent, and the Task, Strict, and localization scores often remain flat or slightly decline. This points to a software-use bottleneck rather than a limitation of the operations themselves. These workflows require correct source-volume selection, spatial input, parameterization, state tracking, and output verification; current agents often fail at these steps, as detailed in Section 6. By contrast, MONAI-based 3D VISTA [14, 15] is highly automated, deep-learning-based, and easier to invoke. On Lung PET/CT, agents use it more effectively, yielding consistent localization gains and model-dependent Strict improvements. Thus, advanced operations are useful when they reduce workflow burden, but complex professional operations remain constrained by the agent’s software-control competence.

### 6 Failure Case Analysis

Beyond pre-selected image interpretation, medical imaging agents are expected to execute a clinically coherent software workflow: choose the right operation, apply it in the right image space, preserve the resulting state, and verify that the artifact supports the final answer. The incorrect advanced-operation episodes expose four recurring failure modes, plus a separate run-to-run instability pattern.

Workflow intent management. In a longitudinal MRI case (Appendix G.1), the agent made plausible registration calls but appeared to drift from cross-timepoint response assessment into a mixture of longitudinal CT1 registration, CT1-derived FLAIR resampling, and intra-timepoint alignment. The operations were locally reasonable, but the composed workflow was no longer well aligned with the clinical question.

Spatial and parametric action grounding. In another longitudinal case (Appendix G.2), the agent converted weak 2D screen boxes into RAS coordinates and then reported millimeter-scale lesion diameters and SPD change. This is not merely noisy measurement; it is invalid elevation of poorly grounded screen evidence into quantitative clinical evidence without validated boundaries, registration, or a measurement procedure.

Stateful artifact management. In a registered MRI fusion case (Appendix G.3), transform objects, transformed nodes, viewer layers, and final rationale became inconsistent: a CT1 transform was requested, the executed trace resolved to a FLAIR transform, and the displayed fusion did not match

the cited CT1 evidence. The failure is software-state misbinding, not only imperfect registration quality.

Operation-result calibration and verification. In a lung PET/CT case (Appendix G.4), an early VISTA3D label-prompt segmentation was close to the reference lesion, but the agent discarded it and later used an incorrect manual point to prompt a self-confirming segmentation. The segmentation output was not uniformly bad; the agent failed to arbitrate between operation outputs and anatomical plausibility.

Procedural stability of advanced operations. Repeated runs of the same registration and segmentation cases show an additional failure mode: the agent does not reliably compose advanced operations into a stable workflow. In Appendix G.5, only 3 of 10 registration runs complete the expected BRAINSFit → resample/apply-transform → registered-fusion sequence, and segmentation runs vary from plausible masks to seed-only or no-mask outputs (Figures 7–9).

Together, these cases show that medical VLM agents lack more than visual diagnostic skill. They lack reliable software-operation competence: objective preservation, spatial grounding, state tracking, operation-output calibration, and procedural control. These errors are hard to diagnose from final answers alone but are exposed by auditable traces of operation arguments, viewer states, generated artifacts, and evidence chains.

### 7 Discussion and Conclusion

Discussion. The introduction of MEDFLOW-BENCH and MEDOPENCLAW shifts the evaluation paradigm of medical vision-language models from isolated, static-image recognition to dynamic reasoning inside professional medical software. Our findings highlight several critical implications for the future of medical AI.

- • Evidence-constrained evaluation exposes a stricter failure mode: A slide-level or study-level answer can be correct while the supporting evidence is wrong, missing, or ungrounded. The Strict metrics make this distinction explicit by requiring the agent to provide checkable slices, coordinates, regions, or longitudinal evidence in addition to the task answer. This is the core reason the benchmark evaluates auditability rather than only label accuracy.
- • Advanced-operation access is not workflow competence: Providing current VLMs with advanced segmentation operators, registration workflows, or MONAI-based expert-model operations does not automatically improve diagnostic accuracy. Some operations improve intermediate localization or evidence scores, but robust advanced-operation execution still requires fine-grained spatial grounding, correct parameterization, software-state tracking, and verification of returned artifacts.
- • Professional software operation is a distinct capability: Operating 3D Slicer or QuPath is not equivalent to answering a question about a rendered screenshot. The agent must preserve the clinical objective while manipulating persistent viewer state, selecting regions, converting coordinates, and recording evidence. These behaviors are directly visible in the runtime trace.
- • Connecting evaluation to traceable interaction: MEDFLOW-BENCH evaluates agents in a fullstudy, software-mediated setting where actions, viewer state, and evidence artifacts are explicitly recorded. This interaction contract is closer to the substrate needed for human-in-the-loop imaging systems than pre-selected static 2D benchmarks, and can support exploratory applications such as MEDCOPILOT in which agent actions and evidence remain inspectable by clinicians.

Conclusion. Medical agents must do more than answer localized questions about pre-selected images. They must actively operate professional medical software, navigate full volumetric studies or whole-slide images, gather comprehensive information, and leave an auditable trail of evidence for their decisions. MEDOPENCLAW provides a secure, bounded and auditable runtime for this complex interaction, and MEDFLOW-BENCH offers a rigorous, controlled environment to evaluate it. Our current results suggest that software-native full-study reasoning is already a tangible reality for frontier VLM agents, even as reliable, evidence-grounded and advanced-operation execution remains a compelling challenge for the community to solve.

### References

- [1] Jason J. Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data, 5: 180251, 2018. doi: 10.1038/sdata.2018.251.
- [2] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semanticallylabeled knowledge-enhanced dataset for medical visual question answering. In international symposium on biomedical imaging (ISBI), pages 1650–1654, 2021.
- [3] Jiahao Chen et al. Huatuogpt-vision: Towards injecting medical visual knowledge into multimodal LLMs at scale, 2024. arXiv:2406.19280.
- [4] Yuxin Zuo, Shang Qu, Yifei Li, Zhang-Ren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 80961–80990, 2025.
- [5] TsinghuaC3I. Medxpertqa dataset card. Hugging Face Datasets, 2025. https://huggingface. co/datasets/TsinghuaC3I/MedXpertQA (accessed March 2026).
- [6] Enrico Coiera. The last mile: Where artificial intelligence meets reality. Journal of Medical Internet Research, 21(11):e16323, 2019. doi: 10.2196/16323.
- [7] Davy van de Sande, Eline Fung Fen Chung, Jacobien Oosterhoff, Jasper van Bommel, Diederik Gommers, and Michel E. van Genderen. To warrant clinical adoption AI models require a multi-faceted implementation evaluation. npj Digital Medicine, 7:58, 2024. doi: 10.1038/ s41746-024-01064-1.
- [8] Panagiotis Korfiatis, Timothy Kline, et al. Implementing artificial intelligence algorithms in the radiology workflow: Challenges and considerations. Mayo Clinic Proceedings: Digital Health, 3(1):100188, 2025. doi: 10.1016/j.mcpdig.2024.100188.
- [9] Arthur A. De Smet, Michael J. Tuite, and Mark A. Norris. Use of the “two-slice-touch” rule for the MRI diagnosis of meniscal tears. American Journal of Roentgenology, 187(4):911–914,

2006. doi: 10.2214/AJR.05.1354.

- [10] Nagmi R. Qureshi, Andrew Shah, Rosemary J. Eaton, Ken Miles, and Fiona J. Gilbert. Dynamic contrast enhanced CT in nodule characterization: How we review and report. Cancer Imaging, 16(1):16, 2016. doi: 10.1186/s40644-016-0074-4.
- [11] Yanyu Li, Lu Lin, Jian Wang, Likun Cao, Yajing Liu, Jianing Pang, Jing An, Zhengyu Jin, and Yining Wang. Cardiac cine with compressed sensing real-time imaging and retrospective motion correction for free-breathing assessment of left ventricular function and strain in clinical practice. Quantitative Imaging in Medicine and Surgery, 13(4):2262–2277, 2023. doi: 10. 21037/qims-22-596.
- [12] Andriy Fedorov, Steve Pieper, Ron Kikinis, et al. 3d slicer as an image computing platform for the quantitative imaging network. Magnetic Resonance Imaging, 30(9):1323–1341, 2012. doi: 10.1016/j.mri.2012.05.001.
- [13] Peter Bankhead, Maurice B. Loughrey, José A. Fernández, Yvonne Dombrowski, Darragh G. McArt, Philip D. Dunne, Stephen McQuaid, Ronan T. Gray, Liam J. Murray, Helen G. Coleman, Jacqueline A. James, Manuel Salto-Tellez, and Peter W. Hamilton. QuPath: Open source software for digital pathology image analysis. Scientific Reports, 7(1):16878, 2017. doi: 10.1038/s41598-017-17204-5.
- [14] M. Jorge Cardoso, Wenqi Li, Richard Brown, et al. Monai: An open-source framework for deep learning in healthcare. arXiv preprint, 2022. arXiv:2211.02701.
- [15] Yufan He, Pengfei Guo, Yucheng Tang, Andriy Myronenko, Vishwesh Nath, Ziyue Xu, Dong Yang, Can Zhao, Benjamin Simon, Mason Belue, Stephanie Harmon, Baris Turkbey, Daguang Xu, and Wenqi Li. VISTA3D: A unified segmentation foundation model for 3d medical imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20863–20873, 2025. doi: 10.1109/CVPR52734.2025.01943.

- [16] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.
- [17] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Conference on Computer Vision and Pattern Recognition, pages 22170–22183, 2024.
- [18] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415, 2023.
- [19] Shansong Wang, Mingzhe Hu, Qiang Li, Mojtaba Safari, and Xiaofeng Yang. Capabilities of GPT-5 on multimodal medical reasoning, 2025. arXiv:2508.08224.
- [20] Su Hwan Kim, Severin Schramm, Lisa C Adams, Rickmer Braren, Keno K Bressem, Matthias Keicher, Paul-Sören Platzek, Karolin Johanna Paprottka, Claus Zimmer, Dennis M Hedderich, et al. Benchmarking the diagnostic performance of open source llms in 1933 eurorad case reports. NPJ digital medicine, 8(1):97, 2025.
- [21] Zonghai Yao, Benlu Wang, Yifan Zhang, Junda Wang, Iris Xia, Zhipeng Tang, Shuo Han, Feiyun Ouyang, Zhichao Yang, Arman Cohan, et al. Medical thinking with multiple images. In The Fourteenth International Conference on Learning Representations, 2026.
- [22] Johannes Moll, Markus Graf, Tristan Lemke, Nicolas Lenhart, Daniel Truhn, Jean-Benoit Delbrouck, Jiazhen Pan, Daniel Rueckert, Lisa C Adams, and Keno K Bressem. Evaluating reasoning faithfulness in medical vision-language models using multimodal perturbations. arXiv preprint arXiv:2510.11196, 2025.
- [23] Adibvafa Fallahpour, Jun Ma, Alif Munim, Hongwei Lyu, and Bo Wang. Medrax: Medical reasoning agent for chest x-ray, 2025. arXiv:2502.02673.
- [24] Cosmin I Bercea, Jun Li, Philipp Raffler, Evamaria O Riedel, Lena Schmitzer, Angela Kurz, Felix Bitzer, Paula Roßmüller, Julian Canisius, Mirjam L Beyrle, et al. Nova: A benchmark for anomaly localization and clinical reasoning in brain mri. arXiv preprint arXiv:2505.14064, 2025.
- [25] Ziye Deng, Ruihan He, Jiaxiang Liu, Yuan Wang, Zijie Meng, Songtao Jiang, Yong Xie, and Zuozhu Liu. Med-glip: Advancing medical language-image pre-training with large-scale grounded dataset. arXiv preprint arXiv:2508.10528, 2025.
- [26] Xiao Wang, Fuling Wang, Yuehang Li, Qingchuan Ma, Shiao Wang, Bo Jiang, and Jin Tang. Cxpmrg-bench: Pre-training and benchmarking for x-ray medical report generation on chexpert plus dataset. In Proceedings of the computer vision and pattern recognition conference, pages 5123–5133, 2025.
- [27] Xiaobin Hu, Yunhang Qian, Jiaquan Yu, Jingjing Liu, Peng Tang, Xiaozhong Ji, Chengming Xu, Jiawei Liu, Xiaoxiao Yan, Xinlei Yu, et al. The landscape of medical agents: A survey. Authorea Preprints, 2025.
- [28] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik S Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae W Park. Mdagents: An adaptive collaboration of llms for medical decision-making. Advances in Neural Information Processing Systems, 37:79410–79452, 2024.
- [29] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 337–347. Springer, 2025.
- [30] Kevin Wu, Eric Wu, Rahul Thapa, Kevin Wei, Angela Zhang, Arvind Suresh, Jacqueline J Tao, Min Woo Sun, Alejandro Lozano, and James Zou. Medcasereasoning: Evaluating and learning diagnostic reasoning from clinical case reports. arXiv preprint arXiv:2505.11733, 2025.

- [31] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente Grau. Medical graph rag: Towards safe medical large language model via graph retrievalaugmented generation. arXiv preprint arXiv:2408.04187, 2024.
- [32] Jiayuan Zhu, Jiazhen Pan, Yuyuan Liu, Fenglin Liu, and Junde Wu. Ask patients with patience: Enabling llms for human-centric medical dialogue with grounded reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2846–2857, 2025.
- [33] Weike Zhao, Chaoyi Wu, Yanjie Fan, Pengcheng Qiu, Xiaoman Zhang, Yuze Sun, Xiao Zhou, Shuju Zhang, Yu Peng, Yanfeng Wang, et al. An agentic system for rare disease diagnosis with traceable reasoning. Nature, 651:775–784, 2026. doi: 10.1038/s41586-025-10097-9.
- [34] Naman Sharma. Cxr-agent: Vision-language models for chest x-ray interpretation with uncertainty aware radiology reporting, 2024. arXiv:2407.08811.
- [35] Binxu Li, Tiankai Yan, Yuanting Pan, Jie Luo, Ruiyang Ji, Jiayuan Ding, Zhe Xu, Shilong Liu, Haoyu Dong, Zihao Lin, and Yixin Wang. Mmedagent: Learning to use medical tools with multi-modal agent. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8745–8760, 2024. doi: 10.18653/v1/2024.findings-emnlp.510.
- [36] Ziyue Wang, Junde Wu, Linghan Cai, Chang Han Low, Xihong Yang, Qiaxuan Li, and Yueming Jin. Medagent-pro: Towards evidence-based multi-modal medical diagnosis via reasoning agentic workflow. arXiv preprint arXiv:2503.18968, 2025.
- [37] Mohammed Baharoon, Luyang Luo, Michael Moritz, Abhinav Kumar, Sung Eun Kim, Xiaoman Zhang, Miao Zhu, Mahmoud Hussain Alabbad, Maha Sbayel Alhazmi, Neel P Mistry, et al. Rexgroundingct: A 3d chest ct dataset for segmentation of findings from free-text reports. arXiv preprint arXiv:2507.22030, 2025.
- [38] Andrew Hoopes, Neel Dey, Victor Ion Butoi, John V. Guttag, and Adrian V. Dalca. Voxelprompt: A vision agent for end-to-end medical image analysis, 2024. arXiv:2410.08397.
- [39] Yuren Mao, Wenyi Xu, Yuyang Qin, and Yunjun Gao. Ct-agent: A multimodal-LLM agent for 3d CT radiology question answering, 2025. arXiv:2505.16229.
- [40] Qiaoyu Zheng, Chaoyi Wu, Pengcheng Qiu, Lisong Dai, Ya Zhang, Yanfeng Wang, and Weidi Xie. How well can modern LLMs act as agent cores in radiology environments?, 2024. URL https://arxiv.org/abs/2412.09529. arXiv:2412.09529.
- [41] Yixing Jiang, Kameron C Black, Gloria Geng, Danny Park, James Zou, Andrew Y Ng, and Jonathan H Chen. Medagentbench: A virtual ehr environment to benchmark medical llm agents. NEJM AI, page AIdbp2500144, 2025.
- [42] Colton Barr. Slicerchat: Building a local chatbot for 3d slicer, 2024. arXiv:2407.11987.
- [43] NA-MIC Project Week. 3d and 2d radiology copilot integration in 3d slicer. Project Week page, 2025. https://projectweek.na-mic.org/PW42_2025_GranCanaria/Projects/ 3Dand2DRadiologyCopilotIntegrationin3DSlicer/ (accessed March 2026).
- [44] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. Also available as arXiv:2310.06770.
- [45] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. arXiv:2404.07972.
- [46] Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzm"uller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. Tabarena: A living benchmark for machine learning on tabular data, 2025. arXiv:2506.16791.

- [47] OpenClaw Team. OpenClaw: Personal ai assistant. GitHub repository, 2026. https:// github.com/openclaw/openclaw (accessed March 24, 2026).
- [48] OpenClaw Team. OpenClaw documentation: Tools and plugins. Official documentation, 2026. https://docs.openclaw.ai/tools (accessed March 24, 2026).
- [49] OpenClaw Team. OpenClaw documentation: Sandbox vs tool policy vs elevated. Official documentation, 2026. https://docs.openclaw.ai/gateway/ sandbox-vs-tool-policy-vs-elevated (accessed March 24, 2026).
- [50] OpenClaw Team. OpenClaw documentation: Sandboxing. Official documentation, 2026. https://docs.openclaw.ai/gateway/sandboxing (accessed March 24, 2026).
- [51] Andres Diaz-Pinto, Sachidanand Alle, Alvin Ihsani, et al. Monai label: A framework for AI-assisted interactive labeling of 3d medical images. Medical Image Analysis, 95:103207,

2024. doi: 10.1016/j.media.2024.103207.

- [52] Patrick Y. Wen, David R. Macdonald, David A. Reardon, Timothy F. Cloughesy, A. Gregory Sorensen, Evanthia Galanis, John F. Degroot, Wolfgang Wick, Mark R. Gilbert, Andrew B. Lassman, Christina Tsien, Tom Mikkelsen, Eric T. Wong, Marc C. Chamberlain, Roger Stupp, Kathleen R. Lamborn, Michael A. Vogelbaum, Martin J. van den Bent, and Susan M. Chang. Updated response assessment criteria for high-grade gliomas: Response assessment in neurooncology working group. Journal of Clinical Oncology, 28(11):1963–1972, 2010. doi: 10. 1200/JCO.2009.26.3541.
- [53] Evan Calabrese, Javier E. Villanueva-Meyer, Jeffrey D. Rudie, Andreas M. Rauschecker, Ujjwal Baid, Spyridon Bakas, Soonmee Cha, John T. Mongan, and Christopher P. Hess. The university of california san francisco preoperative diffuse glioma MRI dataset. Radiology: Artificial Intelligence, 4(6):e220058, 2022. doi: 10.1148/ryai.220058.
- [54] Yannick Suter, Urspeter Knecht, Waldo Valenzuela, Michelle Notter, Ekkehard Hewer, Philippe Schucht, Roland Wiest, and Mauricio Reyes. The LUMIERE dataset: Longitudinal glioblastoma MRI with expert RANO evaluation. Scientific Data, 9(1):768, 2022. doi: 10.1038/s41597-022-01881-7.
- [55] Shaimaa Bakr, Olivier Gevaert, Sergio Echegaray, et al. A radiogenomic dataset of non-small cell lung cancer. Scientific Data, 5:180202, 2018. doi: 10.1038/sdata.2018.202.
- [56] Nadia Brancati, Anna Maria Anniciello, Pushpak Pati, Daniel Riccio, Giosuè Scognamiglio, Guillaume Jaume, Giuseppe De Pietro, Maurizio Di Bonito, Antonio Foncubierta, Gerardo Botti, Maria Gabrani, Florinda Feroce, and Maria Frucci. BRACS: A dataset for BReAst carcinoma subtyping in H&E histology images. Database, 2022:baac093, 2022. doi: 10.1093/ database/baac093.
- [57] Peter Bandi, Oscar Geessink, Quirine F. Manson, Marcory van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, Quanzheng Li, Farhad Ghazvinian Zanjani, Svitlana Zinger, Keisuke Fukuta, Daisuke Komura, Vlado Ovtcharov, Shenghua Cheng, Shaoqun Zeng, Jeppe Thagaard, Anders Bjørholm Dahl, Huangjing Lin, Hao Chen, Ludwig Jacobsson, Martin Hedlund, Melih Çetin, Eren Halici, Hunter Jackson, Richard Chen, Fabian Both, Jörg K. H. Franke, Heidi Küsters-Vandevelde, Wendy Vreuls, Peter Bult, Bram van Ginneken, Jeroen van der Laak, and Geert Litjens. From detection of individual metastases to classification of lymph node status at the patient level: The CAMELYON17 challenge. IEEE Transactions on Medical Imaging, 38(2):550–560, 2019. doi: 10.1109/TMI.2018.2867350.
- [58] OpenAI. Introducing GPT-5.4 mini and nano. OpenAI Blog, 2026. https://openai.com/ index/introducing-gpt-5-4-mini-and-nano/ (accessed May 2026).
- [59] OpenAI. GPT-5.5 system card. Deployment Safety Hub, 2026. https://deploymentsafety. openai.com/gpt-5-5 (accessed May 2026).
- [60] Google DeepMind. Gemini 3 Flash model card. Google DeepMind Model Card, 2025. https: //deepmind.google/models/model-cards/gemini-3-flash/ (accessed May 2026).

- [61] Google DeepMind. Gemini 3.1 Pro model card. Google DeepMind Model Card, 2026. https: //deepmind.google/models/model-cards/gemini-3-1-pro/ (accessed May 2026).
- [62] Anthropic. Introducing Claude Sonnet 4.6. Anthropic News, 2026. https://www.anthropic. com/news/claude-sonnet-4-6 (accessed May 2026).
- [63] Anthropic. Introducing Claude Opus 4.7. Anthropic News, 2026. https://www.anthropic. com/news/claude-opus-4-7 (accessed May 2026).
- [64] Qwen Team. Qwen3.5 model collection. Hugging Face model collection, 2026. https: //huggingface.co/collections/Qwen/qwen35 (accessed March 2026).
- [65] Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.
- [66] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Susanna Maria Baby, Jeremy Lai, Samuel Schmidgall, Lu Yang, Kejia Chen, Per Bjornsson, Shashir Reddy, Ryan Brush, Kenneth Philbrick, Mercy Asiedu, Ines Mezerreg, Howard Hu, Howard Yang, Richa Tiwari, Sunny Jansen, Preeti Singh, Yun Liu, Shekoofeh Azizi, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Riviere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-Bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Elena Buchatskaya, Jean-Baptiste Alayrac, Dmitry Lepikhin, Vlad Feinberg, Sebastian Borgeaud, Alek Andreev, Cassidy Hardin, Robert Dadashi, Léonard Hussenot, Armand Joulin, Olivier Bachem, Yossi Matias, Katherine Chou, Avinatan Hassidim, Kavi Goel, Clement Farabet, Joelle Barral, Tris Warkentin, Jonathon Shlens, David Fleet, Victor Cotruta, Omar Sanseviero, Gus Martins, Phoebe Kirk, Anand Rao, Shravya Shetty, David F. Steiner, Can Kirmizibayrak, Rory Pilgrim, Daniel Golden, and Lin Yang. MedGemma technical report. arXiv preprint arXiv:2507.05201, 2025.
- [67] LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, and Yu Rong. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044, 2025.
- [68] Fan Bai, Yuxin Du, Tiejun Huang, Max Q.-H. Meng, and Bo Zhao. M3D: Advancing 3d medical image analysis with multi-modal large language models. arXiv preprint arXiv:2404.00578, 2024.
- [69] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards generalist foundation model for radiology by leveraging web-scale 2d and 3d medical data. Nature Communications, 16, 2025. doi: 10.1038/s41467-025-62385-7.
- [70] 3D Slicer Developers. 3D Slicer documentation: WebServer module. Online documentation, 2026. https://slicer.readthedocs.io/en/latest/user_guide/modules/ webserver.html (accessed March 2026).
- [71] Geert Litjens, Peter Bandi, Babak Ehteshami Bejnordi, Oscar Geessink, Maschenka Balkenhol, Peter Bult, Altuna Halilovic, Meyke Hermsen, Rob van de Loo, Rob Vogels, Quirine F. Manson, Nikolas Stathonikos, Alexi Baidoshvili, Paul van Diest, Carla Wauters, Marcory van Dijk, and Jeroen van der Laak. 1399 H&E-stained sentinel lymph node sections of breast cancer patients: The CAMELYON dataset. GigaScience, 7(6):giy065, 2018. doi: 10.1093/gigascience/giy065.
- [72] Hans J. Johnson, Gregory Harris, and Kent Williams. BRAINSFit: Mutual information rigid registrations of whole-brain 3d images, using the insight toolkit. The Insight Journal, 57(1): 1–10, 2007. doi: 10.54294/hmb052.

### Appendix

- A Limitations and Roadmap 16
- B Runtime Interface and Safety Constraints 16
- C Benchmark Modules, Dataset Sources, and Scoring 16

- C.1 Module construction and evaluation protocols . . . . . . . . . . . . . . . . . . . . 17
- C.2 2D slice-montage input for Track C . . . . . . . . . . . . . . . . . . . . . . . . . 18

- D Runtime-Agnostic Track Results 18
- E Qualitative Workflow Illustrations 19
- F Operation-Budget Study 21
- G Additional Failure Cases 21

- G.1 Case F1: Registration objective drift in longitudinal MRI . . . . . . . . . . . . . . 22
- G.2 Case F2: Pseudo-quantification from ungrounded coordinates . . . . . . . . . . . . 23
- G.3 Case F3: Transform-layer misbinding in registered MRI fusion . . . . . . . . . . . 23
- G.4 Case F4: Self-confirming segmentation after discarding a correct candidate . . . . 24
- G.5 Instability in Agentic Use of Medical Imaging Software . . . . . . . . . . . . . . . 25

- H Prompt and Tool Interface Details 27

- H.1 3D Slicer Radiology Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . 27
- H.2 QuPath Pathology Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . 30
- H.3 Model-Visible Tools . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- H.4 Dynamic Attachments and Logging . . . . . . . . . . . . . . . . . . . . . . . . . 33

- I Licenses and Terms of Use 33

### A Limitations and Roadmap

The benchmark is designed for controlled evaluation rather than clinical deployment. The pathology experiments currently emphasize viewer-native QuPath navigation and coordinate-grounded evidence acquisition; richer pathology-specific advanced operations and larger whole-slide cohorts remain important extensions. The runtime-agnostic track is useful for comparing alternative full-study pipelines, but such methods do not necessarily produce the same auditable software trace. The current scoring protocols also use public dataset labels and hidden annotations as reference standards, which are necessary for reproducibility but remain proxies for full clinical adjudication. Future extensions should broaden the modality coverage, increase pathology and longitudinal cohorts, add multi-turn clinician-agent interaction, and integrate additional vetted operations from the MONAI and medical imaging ecosystems.

### B Runtime Interface and Safety Constraints

To maintain a bounded and legible interface, the callable surface remains strictly explicit. Functions already supported by 3D Slicer are wrapped via documented WebServer REST endpoints [70], which process HTTP requests and responses to allow external control. Operations that are not cleanly covered by this REST interface, such as DICOM import, quantitative measurement, registration, segmentation, and DICOM SEG export, are exposed through named bridge handlers. The QuPath registry similarly exposes named operations such as loading a case or project, loading an image, setting image type, zooming, panning, focusing the viewer, capturing the current view, converting viewer points to whole-slide coordinates, recording the current view as evidence, sleeping, and closing QuPath. Crucially, this runtime is deliberately restrictive. While 3D Slicer includes an embedded Python console, allowing an agent to generate and execute arbitrary code would enlarge the attack surface, weaken auditability, and complicate deployment. Therefore, MEDOPENCLAW exposes predefined operations only: it prohibits raw Python execution in the Slicer interface and does not expose direct Groovy or arbitrary-script execution in QuPath.

### C Benchmark Modules, Dataset Sources, and Scoring

This appendix gives the public dataset sources and benchmark-specific scoring rules behind the task-family names used in the main text; Table 4 summarizes the benchmark modules, input units, case counts, primary tasks, and evidence requirements.

Domain Source Input unit #Cases Primary task Evidence / Strict scoring Localization scoring Radiology LUMIERE [54] BASELINE/FOLLOWUP

139 RANO response category [52] with lesion-state fields

Strict: RANO label is correct and all scorable FOLLOWUPstudy evidence fields are correct: non-measurable lesion, T2 progression, current/new target lesion, and current/new non-target lesion.

Not required.

brain MRI

Radiology UCSF-PDGM [53] Multi-sequence brain MRI 495 Case-level tumor diagnosis Strict: diagnosis is correct and KEY_SLICE_AXIAL_MM, KEY_SLICE_SAGITTAL_MM, and KEY_SLICE_CORONAL_MM fall within the hidden tumor-mask RAS axis ranges.

Parsed RAS point lies inside the hidden tumor mask after RAS-to-voxel conversion.

Radiology NSCLC [55] Paired PET/CT study 162 Five structured labels: tumor location, pathological T stage, pathological N stage, histology, and grade

In NSCLC, the evidence block contains the same three keyslice offsets and RAS point. The evidence gate is stricter than slice-only evidence: all three key slices must be correct and the RAS point must fall inside the hidden lesion mask. The Strict score is not an all-or-nothing five-label exact match; if this evidence gate is satisfied, the case receives the fivequestion partial score #correct/5, and otherwise it receives 0.

Parsed RAS point lies inside the hidden lesion mask after RAS-to-voxel conversion.

Pathology BRACS [56] Breast whole-slide image 113 Seven-class slide diagnosis: N, PB, UDH, FEA, ADH, DCIS, or IC

Strict: the slide-level BRACS label is correct and, for non-N evidence-scored cases, at least one returned WSI coordinate falls inside a QuPath annotation ROI whose normalized label matches the ground-truth class. For N cases, localizations must be empty or are not required, depending on the protocol.

On ground-truth positive slides, at least one output coordinate hits a correct ground-truth ROI.

Pathology CAMELYON17 [57, 71]

Lymph-node whole-slide image

550 Tumor presence and largest metastasis category

Strict: the tumor-presence and largest-metastasis outputs are correct and, for positive slides with a ground-truth evidence source, at least one returned coordinate hits the tumor mask or XML polygon and avoids exclusion regions. For negative slides, no tumor coordinate is required and localizations should be empty.

On ground-truth positive slides, at least one tumor coordinate hits the groundtruth tumor mask or polygon.

- Table 4: Benchmark modules and scoring. MEDFLOW-BENCH evaluates agents that operate professional medical software and return both task answers and checkable evidence. The reported case counts correspond to the benchmark-eligible units used for evaluation. For NSCLC, the benchmark uses 162 eligible cases after excluding AMC cases by default; including AMC would yield 211 cases. For LUMIERE, the benchmark starts from 360 longitudinal questions and retains 139 after applying the evidence-availability and label-frequency filters used by the current evaluation protocol. For radiology, Strict denotes answer correctness under benchmark-specific evidence constraints. For pathology, evidence and localization are scored by deterministic point lookup against QuPath [13] annotations, masks, or XML polygons rather than by human or LLM judging.

#### C.1 Module construction and evaluation protocols

The benchmark spans radiology and pathology tasks evaluated under deterministic answer and evidence protocols. The brain MRI setting includes LUMIERE [54] longitudinal MRI for lesionevolution reasoning under RANO response criteria [52] and UCSF-PDGM [53], a preoperative multi-sequence brain tumor MRI dataset, for case-level diagnosis over a fixed label set. The lung PET/CT setting uses the NSCLC radiogenomics dataset [55], a paired non-small cell lung cancer PET/CT cohort with pathology annotations, for five structured prediction tasks: tumor location, pathological T stage, pathological N stage, histology, and histopathological grade. The pathology setting uses QuPath to evaluate whole-slide image navigation and evidence acquisition: BRACS [56] requires seven-class slide-level diagnosis and one or more evidence coordinates for non-N cases, while CAMELYON17 [57, 71] requires slide-level tumor presence, largest metastasis category, and tumor coordinates for positive predictions. All experiments in this paper use the MCQ or structured-answer protocol, which provides a canonical output schema and enables deterministic scoring.

For radiology, the agent is required to output both the final task answer and an evidence block. Strict is the evidence-constrained score: a case is credited only when the final task answer and the required evidence are both correct, with benchmark-specific definitions. In UCSF-PDGM, the evidence block contains KEY_SLICE_AXIAL_MM, KEY_SLICE_SAGITTAL_MM, KEY_SLICE_CORONAL_MM, and RAS: [R,A,S]. Strict requires the diagnostic class to be correct and the three key-slice offsets to fall within the hidden tumor-mask RAS axis ranges; the RAS point is not included in Strict and is scored separately as localization. In NSCLC, the evidence block contains the same three key-slice offsets and RAS point, but the evidence condition is stricter: all three key slices must be correct and the RAS point must fall inside the hidden lesion mask. The Strict score is the five-question partial score #correct/5 only if this exact evidence condition is satisfied; otherwise the case receives 0. In LUMIERE, there is no key-slice or RAS evidence block. Instead, evidence consists of CURRENTstudy YES/NO fields for non-measurable lesion, T2 progression, current target lesion, new target lesion, current non-target lesion, and new non-target lesion. Strict requires the RANO category to be correct and all scorable evidence fields to be correct; localization is not reported for LUMIERE.

For pathology, the agent outputs a slide-level answer and, when relevant, one or more whole-slide evidence coordinates. BRACS scores evidence on evidence-scored cases by checking whether an output coordinate falls inside a QuPath annotation ROI whose normalized label matches the groundtruth class; localization is computed on ground-truth positive slides by whether at least one coordinate hits the correct ground-truth ROI. CAMELYON17 scores evidence by checking whether the output coordinate falls inside a ground-truth tumor mask or XML polygon and outside exclusion regions;

[Figure 3]

- Figure 3: 2D slice-montage baseline for Track C. A volumetric MRI or CT study is rendered as a static tiled overview of representative slices, analogous to a compact film sheet. The baseline lets non-viewer methods consume the raw case through a single image input, but it does not preserve interactive viewer state, key-slice navigation, coordinate-grounded evidence, or longitudinal/fusion operations.

localization is computed on ground-truth positive slides by the same hit criterion. These pathology evidence scores are deterministic and do not use human review or LLM judging.

#### C.2 2D slice-montage input for Track C

The 2D slice-montage baseline is the simplest runtime-agnostic alternative to interactive viewer control. It compresses a volumetric study into a static grid of representative slices, allowing a general VLM to inspect the case without calling 3D Slicer [12]. This format is useful for testing whether visible slice-level cues survive compression, but it deliberately removes important elements of clinical workflow: dynamic scrolling, cross-plane confirmation, series switching, fusion state, measurements, coordinate conversion, and replayable evidence acquisition.

### D Runtime-Agnostic Track Results

Table 6 reports the full Track C comparison between 3D Slicer viewer-native runs and runtimeagnostic alternatives under the same answer and evidence schema. For runtime-agnostic baselines, the model returns the same canonical answer and evidence fields directly from the supplied rendered or volumetric input; these runs are scored by the same parsers but do not produce a replayable viewer-state trace. This appendix table is a diagnostic companion to the main viewer-native and advanced-operation results: the differences are task-dependent, and the table mainly documents coverage of pipelines that do not use MEDOPENCLAW rather than introducing a separate main claim.

Longitudinal MRI Brain MRI Lung PET/CT Evidence fields Answer + evidence Answer + evidence Grounding Answer + subtasks Evidence + grounding

Model

T2 Prog.

Cur. TL

New TL

Cur. NTL

New NTL

Tumor Loc.

T Stage

Path. N

NML

Task Strict Task Strict Loc. Task

Hist. Grade Strict Loc.

###### Closed-source models

- GPT-5.4 mini 0.58 0.20 0.86 0.00 0.43 0.00 0.23 0.05 0.38 0.18 0.37 0.31 0.23 0.32 0.42 0.37 0.20 0.00 0.00

- GPT-5.5 0.22 0.39 0.73 0.25 0.95 0.02 0.33 0.04 0.71 0.66 0.91 0.55 0.50 0.50 0.72 0.62 0.43 0.03 0.07 Gemini 3 Flash Preview 0.16 0.61 0.71 0.40 0.88 0.30 0.30 0.07 0.67 0.59 0.87 0.45 0.39 0.25 0.46 0.69 0.50 0.13 0.26

- Gemini 3.1 Pro Preview 0.21 0.46 0.85 0.24 0.72 0.10 0.37 0.08 0.73 0.73 0.96 0.50 0.50 0.36 0.62 0.68 0.35 0.14 0.21 Claude Sonnet 4.6 0.58 0.33 0.56 0.16 0.73 0.17 0.33 0.17 0.38 0.18 0.36 0.26 0.18 0.09 0.16 0.46 0.36 0.00 0.00 Claude Opus 4.7 0.53 0.75 0.67 0.14 0.80 0.13 0.28 0.18 0.37 0.11 0.68 0.43 0.38 0.22 0.40 0.73 0.38 0.01 0.01 Mean over reported 0.38 0.46 0.73 0.20 0.75 0.12 0.31 0.10 0.54 0.41 0.69 0.42 0.36 0.29 0.46 0.59 0.37 0.05 0.09 Open-weight / open-source models Qwen3.5-35B-A3B 0.88 0.23 0.57 0.00 0.00 0.00 0.33 0.18 0.53 0.15 0.11 0.35 0.22 0.25 0.46 0.54 0.30 0.01 0.01 Qwen3.5-27B 0.84 0.46 0.76 0.00 0.11 0.00 0.21 0.09 0.53 0.27 0.36 0.37 0.22 0.22 0.54 0.54 0.33 0.00 0.00 Qwen3.5-9B 0.71 0.23 0.65 0.13 0.26 0.08 0.17 0.08 0.35 0.05 0.06 0.16 0.08 0.17 0.22 0.21 0.13 0.01 0.01 Qwen3.5-4B 0.72 0.29 0.86 0.23 0.28 0.16 0.22 0.11 0.32 0.06 0.03 0.20 0.17 0.10 0.21 0.33 0.19 0.00 0.00 MedGemma 1.5 4B IT 0.64 0.45 0.75 0.06 0.17 0.02 0.16 0.02 0.45 0.09 0.00 0.16 0.11 0.05 0.17 0.27 0.20 0.00 0.00 MedGemma 27B IT 0.75 0.00 0.96 0.00 0.10 0.02 0.11 0.00 0.30 0.00 0.03 0.16 0.28 0.13 0.27 0.10 0.02 0.00 0.00 Lingshu-32B 0.82 0.02 0.98 0.00 0.02 0.00 0.16 0.00 0.36 0.08 0.02 0.31 0.15 0.20 0.38 0.57 0.23 0.00 0.00 Mean over reported 0.76 0.24 0.79 0.06 0.13 0.04 0.19 0.07 0.41 0.10 0.09 0.24 0.18 0.16 0.32 0.37 0.20 0.00 0.00

Table 5: Detailed viewer-native radiology metrics. This appendix table expands Table 1 with lesion-state evidence fields for Longitudinal MRI and subtask-level metrics for Lung PET/CT. NML denotes non-measurable-lesion evidence-field accuracy. Gray columns are fine-grained evidence or subtask metrics, yellow columns are Strict evidence-constrained scores, and blue columns are localization metrics. Bold numbers indicate the best reported score within each model group and metric column; summary rows report means over available entries. Model groups in this appendix are organized by access type rather than the scope grouping used in Table 1.

Backbone

Longitudinal MRI Brain MRI Lung PET/CT Task Strict Task Strict Loc. Task Strict Loc.

- 2D slice-montage General-purpose backbones

- GPT-5.4 mini 0.17 (↓ 0.06) 0.00 (↓ 0.05) 0.41 (↑ 0.03) 0.25 (↑ 0.07) 0.00 (↓ 0.37) 0.40 (↑ 0.09) 0.01 (↑ 0.01) 0.00 (↔ 0.00)

- GPT-5.5 0.01 (↓ 0.32) 0.00 (↓ 0.04) 0.84 (↑ 0.13) 0.79 (↑ 0.13) 0.97 (↑ 0.06) 0.54 (↓ 0.01) 0.13 (↑ 0.10) 0.23 (↑ 0.16)

- Gemini 3.1 Pro Preview 0.22 (↓ 0.15) 0.00 (↓ 0.08) 0.76 (↑ 0.03) 0.71 (↓ 0.02) 0.81 (↓ 0.15) 0.49 (↓ 0.01) 0.11 (↓ 0.03) 0.13 (↓ 0.08) Gemini 3 Flash Preview 0.13 (↓ 0.17) 0.00 (↓ 0.07) 0.52 (↓ 0.15) 0.43 (↓ 0.16) 0.77 (↓ 0.10) 0.51 (↑ 0.06) 0.07 (↓ 0.06) 0.08 (↓ 0.18) Claude Sonnet 4.6 0.12 (↓ 0.21) 0.00 (↓ 0.17) 0.52 (↑ 0.14) 0.19 (↑ 0.01) 0.38 (↑ 0.02) 0.45 (↑ 0.19) 0.06 (↑ 0.06) 0.02 (↑ 0.02) Claude Opus 4.7 0.20 (↓ 0.08) 0.03 (↓ 0.15) 0.58 (↑ 0.21) 0.08 (↓ 0.03) 0.05 (↓ 0.63) 0.47 (↑ 0.04) 0.01 (↔ 0.00) 0.00 (↓ 0.01) Qwen3.5-35B-A3B 0.19 (↓ 0.14) 0.12 (↓ 0.06) 0.56 (↑ 0.03) 0.38 (↑ 0.23) 0.54 (↑ 0.43) 0.25 (↓ 0.10) 0.02 (↑ 0.01) 0.00 (↓ 0.01) Medical-domain backbones MedGemma 1.5 4B IT 0.12 (↓ 0.04) 0.05 (↑ 0.03) 0.35 (↓ 0.10) 0.00 (↓ 0.09) 0.00 (↔ 0.00) 0.09 (↓ 0.07) 0.00 (↔ 0.00) 0.00 (↔ 0.00) MedGemma 27B IT 0.29 (↑ 0.18) 0.00 (↔ 0.00) 0.35 (↑ 0.05) 0.00 (↔ 0.00) 0.00 (↓ 0.03) 0.11 (↓ 0.05) 0.00 (↔ 0.00) 0.00 (↔ 0.00) Lingshu-32B 0.24 (↑ 0.08) 0.02 (↑ 0.02) 0.38 (↑ 0.02) 0.06 (↓ 0.02) 0.00 (↓ 0.02) 0.34 (↑ 0.03) 0.00 (↔ 0.00) 0.00 (↔ 0.00)

###### Native 3D VLM, Alt. only

M3D 0.21 0.10 0.40 0.10 0.36 0.18 0.07 0.33 RadFM 0.25 0.00 0.51 0.10 0.36 0.25 0.00 0.00

- Table 6: Runtime-agnostic track. Each paired cell reports the alternative runtime-agnostic score and its change relative to the matched 3D Slicer viewer-native run for the same backbone. Green, red, and gray cells indicate improvement, degradation, and no change, respectively. Task denotes final task accuracy, Strict denotes task-plus-evidence correctness, and Loc. denotes localization where available. The 2D slice-montage baseline renders volumetric studies as static tiled overview figures, while native 3D VLM rows report Alt.-only volumetric baselines without matched same-backbone Slicer controls.

### E Qualitative Workflow Illustrations

This section provides concrete examples of the two interaction regimes used throughout the benchmark. Figure 4 shows a viewer-native Brain MRI episode that uses only primitive 3D Slicer actions, while Figures 5 and 6 illustrate advanced-operation workflows, including MONAI VISTA3D-based segmentation. These figures are not additional results; they clarify what an auditable episode looks like, what evidence is recorded, and where advanced operations enter the runtime.

[Figure 4]

- Figure 4: Viewer-native Brain MRI example in UCSF-PDGM [53]. The agent opens the full multi-sequence case, selects T1c and FLAIR, uses fusion and axial scrolling to localize the dominant enhancing lesion, converts the selected image point to a RAS coordinate, and returns both tumor location and likely histologic subtype. No segmentation, registration, or MONAI expert-model operation is used; the example demonstrates that primitive viewer operations can support a replayable localization-and-diagnosis trace when the agent maintains the correct series and slice state.

[Figure 5]

- Figure 5: Segmentation-assisted Brain MRI example. After inspecting the UCSF-PDGM study and selecting FLAIR/T2-dominant evidence, the agent uses the fused viewer state to seed a RAS point and invokes the segmentation operation to generate a tumor mask. The final answer is grounded by the segment center, orthogonal views, and a reported RAS coordinate. The example illustrates how an advanced operation can convert an inspected lesion candidate into structured evidence, while still requiring the agent to choose the correct source volume, seed location, and interpretation.

[Figure 6]

- Figure 6: MONAI 3D VISTA [14, 15] advanced-operation example for Lung PET/CT. The agent first searches the diagnostic CT volume, invokes the MONAI 3D VISTA workflow with the lung-tumor label to obtain a lesion-centered segmentation, then checks the candidate against PET/CT fusion before returning tumor location, T/N stage, histology, grade, key slices, and RAS evidence. The figure illustrates the intended role of 3D VISTA in MEDOPENCLAW: it is an auditable expert-model operation inside the Slicer workflow, not a replacement for source-series selection, fusion verification, or final clinical reasoning.

### F Operation-Budget Study

The following diagnostic study varies the maximum number of explicit operation calls available to the agent. Table 7 reports this budget control.

Max ops.

Lung PET/CT NML

Longitudinal MRI

Brain MRI

Backbone

T2 Prog.

Cur. TL

New TL

Cur. NTL

New NTL

Tumor Loc.

T Stage

Path. N

Task Strict Task Strict Loc. Task

Hist. Grade Strict Loc.

Qwen3.5-35B-A3B

Qwen3.5-35B-A3B 1 0.88 0.15 0.65 0.00 0.14 0.00 0.21 0.13 0.41 0.11 0.00 0.32 0.15 0.22 0.43 0.47 0.35 0.00 0.00 Qwen3.5-35B-A3B 3 0.82 0.21 0.55 0.00 0.11 0.04 0.21 0.10 0.50 0.11 0.13 0.30 0.17 0.25 0.47 0.42 0.20 0.00 0.00 Qwen3.5-35B-A3B 7 0.86 0.42 0.65 0.13 0.14 0.04 0.27 0.13 0.50 0.15 0.13 0.28 0.18 0.18 0.33 0.40 0.30 0.00 0.00

- Qwen3.5-35B-A3B 9 0.86 0.48 0.74 0.38 0.14 0.00 0.25 0.17 0.52 0.11 0.23 0.36 0.25 0.30 0.50 0.47 0.28 0.00 0.02

- Qwen3.5-35B-A3B 10 0.83 0.30 0.74 0.25 0.14 0.00 0.26 0.16 0.46 0.13 0.25 0.32 0.23 0.20 0.48 0.42 0.25 0.00 0.02 Qwen3.5-35B-A3B 15 0.87 0.36 0.61 0.13 0.14 0.04 0.24 0.12 0.52 0.16 0.23 0.37 0.23 0.30 0.62 0.42 0.28 0.00 0.00 Qwen3.5-35B-A3B 20 0.87 0.45 0.74 0.00 0.14 0.00 0.24 0.16 0.46 0.11 0.17 0.31 0.12 0.27 0.48 0.43 0.25 0.00 0.00

Gemini 3.1 Pro Preview

Gemini 3.1 Pro Preview 1 0.44 0.39 0.87 0.13 0.41 0.04 0.23 0.13 0.76 0.35 0.47 0.55 0.57 0.27 0.75 0.65 0.52 0.00 0.02 Gemini 3.1 Pro Preview 5 0.86 0.06 0.87 0.00 0.14 0.04 0.18 0.10 0.42 0.14 0.64 0.56 0.53 0.27 0.78 0.70 0.50 0.08 0.14 Gemini 3.1 Pro Preview 10 0.57 0.21 0.84 0.38 0.30 0.04 0.29 0.13 0.71 0.60 0.93 0.54 0.58 0.33 0.72 0.65 0.42 0.11 0.28 Gemini 3.1 Pro Preview 15 0.70 0.12 0.74 0.38 0.22 0.04 0.27 0.07 0.75 0.71 0.91 0.52 0.57 0.30 0.68 0.70 0.33 0.10 0.30

Table 7: Operation-budget study. Only the explicit budget-control blocks for Qwen3.5-35B-A3B and Gemini 3.1 Pro Preview are shown. NML denotes non-measurable-lesion evidence-field accuracy.

### G Additional Failure Cases

The following cases expand the qualitative failure analysis in Section 6. Each case is presented as a compact auditable case study: a case card states the clinical objective and observed failure signal, a trace excerpt shows the minimal logged software calls needed to diagnose the error, and a mechanism box links the trace to the missing agent capability.

#### G.1 Case F1: Registration objective drift in longitudinal MRI

F1 summary: workflow intent management Clinical objective Longitudinal response assessment between week-015 and week-024 for Patient-

009.

Ground truth / prediction

RANO complete response (CR) / predicted partial response (PR).

Trace signal The agent created a plausible cross-timepoint CT1 transform, but then mixed longitudinal CT1 registration with CT1-derived FLAIR resampling and intratimepoint FLAIR-to-CT1 alignment.

Why it matters The final workflow no longer supported the intended longitudinal comparison,

despite containing locally plausible registration calls.

Minimal auditable trace

# Intended longitudinal CT1 registration. brains_fit:

moving = FOLLOWUP__week-024__CT1 fixed = BASELINE__week-015__CT1 output = FOLLOWUP_to_BASELINE_CT1_transform

brains_resample: input = FOLLOWUP__week-024__CT1 reference = BASELINE__week-015__CT1 transform = FOLLOWUP_to_BASELINE_CT1_transform output = FOLLOWUP__week-024__CT1__aligned

# Drift: the CT1-derived longitudinal transform is applied to FLAIR. brains_resample:

input = FOLLOWUP__week-024__FLAIR reference = BASELINE__week-015__CT1 transform = FOLLOWUP_to_BASELINE_CT1_transform output = FOLLOWUP__week-024__FLAIR__aligned

# Additional intra-timepoint registration changes the comparison objective. brains_fit:

moving = BASELINE__week-015__FLAIR fixed = BASELINE__week-015__CT1 output = BASELINE_FLAIR_to_CT1_transform

brains_resample: input = BASELINE__week-015__FLAIR reference = BASELINE__week-015__CT1 transform = BASELINE_FLAIR_to_CT1_transform output = BASELINE__week-015__FLAIR__aligned

# Final displayed comparison is between two derived FLAIR volumes. set_fusion:

background = BASELINE__week-015__FLAIR__aligned foreground = FOLLOWUP__week-024__FLAIR__aligned opacity = 0.5

Mechanism. The failure is not the absence of registration calls. The agent repeatedly invoked registration operations and produced plausible intermediate artifacts, but the workflow objective became unstable. The clinical task required a longitudinal response assessment across timepoints, while the executed chain mixed longitudinal CT1 registration, CT1-derived FLAIR resampling, and intra-timepoint BASELINE FLAIR-to-CT1 alignment.

Benchmark implication. This exposes a workflow-intent failure: the agent can execute locally plausible software operations without preserving the semantic purpose of the clinical workflow. The auditable trace reveals the mismatch between the claimed comparison and the actual composed registration chain.

#### G.2 Case F2: Pseudo-quantification from ungrounded coordinates

F2 summary: spatial and parametric action grounding Clinical objective Longitudinal response assessment between week-040 and week-055 for Patient-

059.

Ground truth / prediction

RANO partial response (PR) / predicted progressive disease (PD).

Trace signal The agent converted weak screen-space boxes into RAS coordinates and then

reported millimeter-scale lesion diameters and SPD change.

Why it matters The reported quantitative evidence was not grounded in validated lesion boundaries, a segmentation mask, a registered longitudinal frame, or a measurement procedure.

Minimal auditable trace

# The agent selected two axial offsets from different timepoints. FOLLOWUP CT1 axial offset = 58.159 mm BASELINE CT1 axial offset = 44.529 mm

# The same screen-space box was converted into RAS coordinates. image_coords_to_ras on FOLLOWUP CT1:

bbox_1000 = [640, 320, 860, 540] center_ras = [-25.55, 11.55, 58.159]

image_coords_to_ras on BASELINE CT1: bbox_1000 = [640, 320, 860, 540] center_ras = [-16.64, -15.54, 44.529]

# These derived coordinates were elevated into quantitative evidence. BASELINE lesion:

reported ~= 13.5 mm x 12.3 mm SPD ~= 166 mm^2

FOLLOWUP lesion: reported ~= 18.3 mm x 19.9 mm SPD ~= 364 mm^2

final rationale: "approximately 119% SPD increase" -> progressive disease

Mechanism. The measurements were not produced by a validated measurement workflow. They were derived from weakly grounded screen-space boxes, converted into RAS coordinates, and interpreted as lesion diameters without checking the lesion boundary, slice plane, or longitudinal frame.

Benchmark implication. This exposes a spatial-grounding failure: uncertain visual localization was transformed into precise millimeter-scale clinical evidence. The failure is not merely an inaccurate size estimate; it is the invalid elevation of weak evidence into quantitative RANO-style reasoning.

#### G.3 Case F3: Transform-layer misbinding in registered MRI fusion

F3 summary: stateful artifact management Clinical objective Longitudinal response assessment between week-112 and week-152 for Patient-

029.

Ground truth / prediction

RANO complete response (CR) / predicted stable disease (SD).

Trace signal The agent created separate CT1 and FLAIR transforms, but the executed trace resolved the CT1 transform application to the FLAIR transform and attached both FOLLOWUP volumes to the wrong transform layer.

Why it matters The final viewer state displayed FLAIR fusion, while the final rationale cited

registered CT1 evidence.

Minimal auditable trace

# Two transform objects are created. brains_fit:

moving = FOLLOWUP__week-152__CT1 fixed = BASELINE__week-112__CT1 output = FOLLOWUP_CT1_registered_to_BASELINE_CT1

brains_fit:

moving = FOLLOWUP__week-152__FLAIR fixed = BASELINE__week-112__FLAIR output = FOLLOWUP_FLAIR_registered_to_BASELINE_FLAIR

# Raw model call attempts to apply the CT1 transform to CT1. tool_call_start:

apply_transform( node_ref = FOLLOWUP__week-152__CT1, transform_ref = FOLLOWUP_CT1_registered_to_BASELINE_CT1

)

# Executed trace resolves the operation to the FLAIR transform. tool_trace:

resolved transform = FOLLOWUP_FLAIR_registered_to_BASELINE_FLAIR transformed nodes = [

FOLLOW_UP__week-152__CT1, BASELINE__week-152__FLAIR

]

# list_transforms exposes the mismatch. FOLLOWUP_CT1_registered_to_BASELINE_CT1

- -> [] FOLLOW_UP__FLAIR_registered_to_BASELINE_FLAIR
- -> [

FOLLOW_UP__week-152__CT1, FOLLOW_UP__week-152__FLAIR

]

# Viewer state and final rationale disagree. viewer fusion:

background = BASELINE__week-112__FLAIR foreground = FOLLOW_UP__week-152__FLAIR opacity = 0.5

final rationale: "Axial and coronal views of the registered CT1 volumes show..."

Mechanism. The agent failed to maintain a consistent binding between transform objects, image nodes, viewer layers, and textual evidence. The CT1 transform was left without the expected CT1 child node, while the FLAIR transform contained both the FOLLOWUP CT1 and FOLLOWUP FLAIR nodes.

Benchmark implication. This exposes a software-state failure. The registration algorithm produced plausible objects, but the agent did not verify whether the displayed layers matched the evidence later cited in the final rationale.

#### G.4 Case F4: Self-confirming segmentation after discarding a correct candidate

###### F4 summary: operation-result calibration and verification

Clinical objective Lung PET/CT structured prediction for case R01-004. Ground truth / prediction

1D2A3C4B5B / predicted 1C2B3A4A5B. Only the histopathological grade answer was correct.

Trace signal An early VISTA3D label-prompt segmentation [15] was close to the reference tumor, but the agent discarded it and later reinforced an incorrect manually selected candidate point.

Why it matters The operation sequence converted a mistaken localization into a new segmentation

artifact and then into final coordinate evidence.

Minimal auditable trace

# Reference evidence location. GT center_ras ~= [-65.78, 58.50, -132.25] GT representative_point_ras ~= [-65.38, 58.80, -131.75]

# Early VISTA3D output is close to the reference. vista3d_segment:

label_prompt = [23] segmentation = VISTA3D 23

center_on_segment: center_ras = [-65.38, 58.80, -130.50]

# The agent discards the useful candidate and selects a different point. set_slice_offset:

axial offset = -265.86

image_coords_to_ras: point_1000 = [250, 450] point_ras = [100.60, 46.88, -265.86]

# The incorrect point is used to prompt another segmentation. vista3d_segment:

label_prompt = [23] points_ras = [[100.598, 46.881, -265.86]]

center_on_segment: center_ras = [102.73, 44.72, -265.50]

# Final answer adopts the later, incorrect localization. KEY_SLICE_AXIAL_MM: -265.5 KEY_SLICE_SAGITTAL_MM: 102.7 KEY_SLICE_CORONAL_MM: 44.7 RAS: [102.7, 44.7, -265.5]

Mechanism. The segmentation operation was not uniformly unreliable. The early label-prompt segmentation was close to the ground truth, but the agent failed to preserve it as primary evidence. A later manual point selection produced an incorrect candidate, and the subsequent point-prompt segmentation reinforced that wrong hypothesis.

Benchmark implication. This exposes an operation-calibration failure: the agent did not arbitrate between competing operation outputs or verify anatomical plausibility before using an operation artifact as final coordinate evidence.

#### G.5 Instability in Agentic Use of Medical Imaging Software

The preceding failure cases analyze individual traces. Here we test a different property: whether the same agent can repeat a professional imaging workflow consistently under identical case settings. The answer is currently no. These runs do not primarily evaluate the underlying registration or segmentation algorithms; they evaluate whether the agent can compose them into a stable, clinically meaningful procedure.

For registration, the nominal workflow is BRAINSFit registration (BF) [72], followed by BRAINSResample or apply-transform (BR), followed by registered follow-up versus baseline fusion. Figure 7 shows that only 3 of 10 runs complete this functional sequence. Several trajectories execute BF and BR but never verify with fusion; some call fusion on an unregistered follow-up volume; others repeat selection or inspection calls without closing the loop. This is a procedural-control failure: the agent can invoke the correct named operations but does not reliably maintain the required operation order, artifact binding, and verification step.

The visual consequence is shown in Figure 8. Even with identical inputs, the final axial, coronal, and sagittal views differ markedly across runs. Some outputs appear to show plausible registered fusion, while others display inconsistent contrast, missing overlay behavior, or different effective source layers. Thus, workflow instability propagates into the visual evidence that the final answer would cite.

Segmentation shows the same problem in a different form. Figure 9 shows that repeated runs produce substantially different masks: some cover broad non-lesion brain tissue, some produce local tumorlike overlays, some return seed-only evidence, and two runs return no mask. The issue is not just mask quality; it is unstable agentic use of the segmentation operation, including different prompt types, seed choices, thresholds, and failure recovery behavior.

[Figure 7]

- Figure 7: Procedural inconsistency in agent-driven registration across 10 repeated runs. The nominal milestone order is BRAINSFit registration (BF), resample/apply transform (BR), and registered fusion verification. Green status marks runs that complete a functionally valid registeredfusion workflow; red status marks missing, incomplete, or unregistered fusion. Only 3 of 10 runs complete the expected sequence, showing that the agent’s operation calls are not reliably organized into a clinically valid registration workflow.

[Figure 8]

- Figure 8: Final registration visualizations from 10 repeated runs of the same case. Each column shows the final axial, coronal, and sagittal views produced by one run. The views differ despite identical inputs and prompts, indicating that procedural instability affects not only the call trace but also the displayed image evidence. This undermines the assumption that access to a registration operation automatically yields a stable, auditable comparison.

[Figure 9]

- Figure 9: Final segmentation outputs from 10 repeated runs of the same case. Green overlays show the masks generated through the segmentation operation; yellow markers and red boxes show selected seed or candidate regions. Across identical runs, outputs range from broad over-segmentation to local masks, seed-only evidence, and no-mask failures. This illustrates unstable agentic control of segmentation parameters, source state, and failure recovery.

### H Prompt and Tool Interface Details

This appendix documents the prompt templates and model-visible tools used by MEDFLOW-BENCH. Placeholders such as <CASE_ID>, <SCENE_INVENTORY>, and <CURRENT_VIEWER_SCREENSHOT> indicate text or images filled separately for each case before the model sees the prompt. The reported benchmark scores do not use human or LLM judging; answers are scored by deterministic parsing and hidden-reference evidence checks. Full case-level prompts, screenshots, tool-call traces, tool outputs, viewer states, and final responses are released with the code artifact.

Disclosure granularity. The appendix includes the system-level task instructions, answer formats, evidence formats, coordinate conventions, tool budget, and callable tool names exposed to the model. It does not enumerate every case-instantiated prompt in the PDF because those prompts repeat the same templates and differ mainly in case identifiers, loaded image names, scene inventories, and attached screenshots.

#### H.1 3D Slicer Radiology Prompt Templates

For radiology episodes, the system preloads the relevant study or studies and attaches a scene inventory plus baseline axial, sagittal, and coronal screenshots after the text prompt. LUMIERE, UCSF-PDGM, and NSCLC use the same viewer-native 3D Slicer tool surface and the same 20-round budget. Basic foreground/background fusion of already loaded or prealigned volumes is part of this shared surface.

Longitudinal MRI response assessment. The template below is used for LUMIERE longitudinal response-assessment episodes. The agent compares matched modalities across timepoints and reports evidence fields for the CURRENT study only.

System/task instruction: You are an AI radiology assistant using a remote MRI viewer (3D Slicer). The scene already contains

two MRI studies from the same patient: OLD and CURRENT. Do not call open_case unless the viewer is clearly empty or broken. If recovery is needed, use the exact paths listed in the user message so both OLD and CURRENT studies are loaded together. The scene inventory is provided in the prompt. Use the viewer-native 3D Slicer tools exposed in this run. Focus on the CURRENT study; OLD is only the reference for interval comparison. Compare matching modalities across OLD

and CURRENT. CT1 is usually the most important sequence for RANO response assessment, with

FLAIR/T2/T1 as supporting context. Choose exactly one of the four RANO response classes. Also output the requested YES/NO evidence fields for the CURRENT study only. Do not output UNKNOWN and do not output JSON. End with the structured block below, with ANSWER as the final line.

Tool-use rule: When calling image_coords_to_ras, send exactly one of bbox_1000 or point_1000, never both. Use

bbox_1000 for a box center and point_1000 for a clicked point.

Tool budget: At most 20 tool-calling rounds. If the budget is exhausted, immediately provide the best final answer

using the evidence already in context.

Case brief: Case brief for <PATIENT_ID>__week-OLD__to__week-CURRENT:

- - Dataset: LUMIERE

- - Image space: DeepBraTumIA atlas-space skull-stripped MRI (MNI)

- - Patient: <PATIENT_ID>

- - OLD reference study: week-OLD

- - CURRENT study to classify: week-CURRENT

- - Loaded modalities for both studies: CT1, T1, T2, FLAIR

- - Tumor segmentation masks are intentionally hidden.

- - Evidence fields describe the CURRENT study only.

Task: Compare the CURRENT study against the OLD reference study and choose the CURRENT study’s RANO response

label. Answer options:

- A) PD

- B) SD

- C) PR

- D) CR

After a short rationale, end with exactly this block: NON_MEASURABLE_LESION: <YES|NO> T2_PROGRESSION: <YES|NO> CURRENT_TARGET_LESION: <YES|NO>

Task family Software Template included below Final output Budget Longitudinal MRI (LUMIERE)

3D Slicer Radiology system instruction, OLD/CURRENT study template, RANO options, CURRENT-study evidence fields

RANO option plus six CURRENTstudy YES/NO evidence fields

20 rounds

Brain MRI (UCSFPDGM)

3D Slicer Multi-sequence MRI instruction, diagnostic options, tumor-evidence template when evidence localization is evaluated

Diagnostic option; key slices and RAS point when evidence localization is evaluated

20 rounds

Lung CT/PET (NSCLC) 3D Slicer Thoracic CT/PET instruction, series-catalog placeholder, five-question MCQ schema, lesion-evidence template when evidence localization is evaluated

Compact five-question answer string; key slices, RAS point, and source-series provenance when evidence localization is evaluated

20 rounds

20 rounds

Breast WSI (BRACS) QuPath QuPath system instruction, slidelevel task prompt, per-round observation template, JSON output schema

JSON with BRACS label, confidence, rationale, and ROI coordinates for non-normal predictions

Lymph-node WSI (CAMELYON17)

QuPath QuPath system instruction, slidelevel tumor task, per-round observation template, JSON output schema

JSON with tumor presence, largest metastasis category, confidence, rationale, and tumor coordinates for positive predictions

20 rounds

- Table 8: Prompt templates and answer formats disclosed for each benchmark module. Case-specific text and images are represented by placeholders and filled before inference.

NEW_TARGET_LESION: <YES|NO> CURRENT_NON_TARGET_LESION: <YES|NO> NEW_NON_TARGET_LESION: <YES|NO> ANSWER: <A|B|C|D>

Multi-sequence brain MRI diagnosis. The UCSF-PDGM template uses the same viewer-native

- 3D Slicer tool surface. When tumor localization is evaluated, the model localizes the lesion in shared RAS space using an informative loaded anatomical MRI sequence.

System/task instruction: You are an AI radiology assistant using a remote MRI viewer (3D Slicer). The case has already been

opened by the system before you start. Do not call open_case unless the viewer is clearly empty or broken. If recovery is needed, use the exact absolute path provided in the user message. The case is opened with a core multi-sequence MRI subset, and the current scene inventory is provided in the prompt. Use the viewer-native 3D Slicer tools exposed in this run. If KEY_SLICE_* _MM evidence is requested, report canonical RAS-axis millimeters from the image corner overlay: axial=S, sagittal=R, coronal=A. Do not use slice indices. When calling image_coords_to_ras, send

exactly one of bbox_1000 or point_1000. Do not output JSON. If evidence is requested, include it exactly once before the final answer line. Output must contain a final line exactly like: ANSWER: <A|B|C|D|E|F>.

Tool budget: At most 20 tool-calling rounds. If the budget is exhausted, immediately provide the best final answer

using the evidence already in context.

Case brief: Case brief for <CASE_ID>:

- - CSV ID: <CSV_ID>

- - Loaded core NIfTI files: <comma-separated core T1/T1c/T2/FLAIR-style files>

- - The current scene inventory is shown after open_case.

- - Treat the loaded T1/T1c/T2/FLAIR-style sequences as the main evidence sources.

- - If tumor localization is evaluated, evidence may be measured on any loaded anatomical MRI sequence where the tumor is clearly visible.

Task: Classify the case into one option:

- A) Glioblastoma

- B) Oligodendroglioma / Astrocytoma

- C) No tumor

- D) Dysembryoplastic neuroepithelial tumor (DNET) / Ganglioglioma

- E) Pilocytic Astrocytoma

- F) Central Neurocytoma / Ependymoma

Suggested steps:

- 1) Review baseline views.

- 2) Use the prompt-provided scene inventory to choose among loaded sequences; use select_volume to switch sequences.

- 3) Use observe/scroll/zoom/wl as needed. Prefer scroll output=’keyframes’ unless video is necessary.

If evidence is requested, output exactly once before the answer line: EVIDENCE: KEY_SLICE_AXIAL_MM: <float> KEY_SLICE_SAGITTAL_MM: <float> KEY_SLICE_CORONAL_MM: <float> RAS: [<R>, <A>, <S>] Definitions:

- - KEY_SLICE_*_MM values use canonical RAS-axis millimeters from the image corner overlay: axial=S, sagittal=R, coronal=A.

- - RAS must be a single point strictly inside the visible tumor mask.

- - Use one loaded MRI sequence consistently while localizing the tumor and producing the evidence block

.

- - If image_coords_to_ras is used, pass exactly one of bbox_1000 or point_1000.

Final line: ANSWER: <A|B|C|D|E|F>

Lung CT/PET structured prediction. The NSCLC prompt asks for five structured predictions. When lesion localization is evaluated, the evidence block must be produced on the specified loaded source series.

System/task instruction: You are a clinically grounded multimodal thoracic imaging assistant using a remote 3D Slicer viewer.

All resolved imaging series across available timepoints are already loaded, and the current scene inventory is provided. Use CT morphology first. Use PET as adjunctive evidence for metabolic activity and nodal survey. For chest CT inspection, use window_preset with ct_lung and

ct_mediastinal as needed. If KEY_SLICE_*_MM evidence is requested, report canonical RAS-axis millimeters from the image corner overlay: axial=S, sagittal=R, coronal=A. Do not use the native

viewer-header sign convention if it differs from the corner overlay. Use the viewer-native 3D Slicer tools exposed in this run. When calling image_coords_to_ras, send exactly one of bbox_1000 or point_1000. Do not output JSON. Write a brief English rationale. If evidence is requested, include the evidence block exactly as specified before the final compact answer string. End with a single compact answer string exactly like 1C2D3B4A5B, and do not repeat it.

Tool budget: At most 20 tool-calling rounds. If the budget is exhausted, immediately provide the best final answer

using the evidence already in context.

Case brief: Case brief for <CASE_ID>:

- - All resolved CT/PET series across available timepoints were preloaded at start.

- - Preferred initial viewing heuristic: <DEFAULT_ALIAS, usually diagnostic_ct>

- - Dynamic CT/PET series catalog, default aliases, loaded series names, and curated grouping information are inserted here.

- - If lesion localization is evaluated, the evidence source series is <SOURCE_VOLUME_NAME> (< SOURCE_MODALITY>). The model must switch/select this loaded series before measuring KEY_SLICE_* _MM or RAS and must keep it active while calling image_coords_to_ras.

Task: Answer the following five multiple-choice questions based on the imaging only. Return the final answer

in one compact string, with no spaces, exactly like 1C2D3B4A5B.

- Q1. Tumor location

- 1A Right upper lobe; 1B Right middle lobe; 1C Right lower lobe; 1D Left upper lobe; 1E Left lower lobe ; 1F Lingula; 1G No tumor

Q2. Pathological T stage

- 2A T1; 2B T2; 2C T3; 2D T4; 2E Not available / not collected; 2F No tumor

Q3. Pathological N stage

- 3A N0; 3B N1; 3C N2; 3D Not available / not collected; 3E No tumor

Q4. Histology

- 4A Adenocarcinoma; 4B Squamous cell carcinoma; 4C Small cell carcinoma; 4D Other / indeterminate NSCLC ; 4E No tumor

Q5. Histopathological grade

- 5A G1 Well differentiated; 5B G2 Moderately differentiated; 5C G3 Poorly differentiated; 5D Other Type I: well to moderately differentiated; 5E Other Type II: moderately to poorly differentiated; 5F No tumor

If evidence is requested, output exactly once before the compact answer string: EVIDENCE: KEY_SLICE_AXIAL_MM: <float> KEY_SLICE_SAGITTAL_MM: <float> KEY_SLICE_CORONAL_MM: <float> RAS: [<R>, <A>, <S>] Definitions:

- - The evidence must be reported on the loaded source series <SOURCE_VOLUME_NAME> (<SOURCE_MODALITY>) only.

- - Each KEY_SLICE_*_MM value uses canonical RAS-axis millimeters from the image corner overlay on the active source series: axial=S, sagittal=R, coronal=A.

- - RAS must be a single point strictly inside the visible primary tumor on that same source series.

- - If image_coords_to_ras is used, pass exactly one of bbox_1000 or point_1000.

- - If needed, explicitly call select_volume(volume_name=...) on the source series before the evidence block.

#### H.2 QuPath Pathology Prompt Templates

The pathology setting operates in QuPath through a direct tool interface. The model receives a task prompt, a direct QuPath state summary, a rule-based GUI/control summary, and the current viewer screenshot at each step. The screenshot is the primary image evidence; the state summaries are software-state evidence, not pathology evidence. Pathology episodes use a 20-round budget, with at most one tool call per round.

#### Common QuPath system instruction.

|You are the host model being evaluated. You operate QuPath through an explicit, audited tool surface. Core rules:<br><br>- You are the primary agent. Do not delegate the core reasoning task.<br><br>- The attached viewer screenshot is the primary evidence for pathology interpretation.<br><br>- The direct QuPath state summary reports structured software/viewer state from QuPath itself.<br><br>- The GUI state summary is rule-based and is not pathology evidence.<br><br>- Prefer direct tools whenever they can perform the needed viewer action.<br><br>- Work inside QuPath only. Do not assume access to the full computer.<br><br>- Use at most one tool call per step.<br><br>- Use at most 20 tool-calling rounds.<br><br>- If the task is complete, answer directly without calling a tool.<br><br>- The case is usually already loaded before the first step. Do not call load_case unless recovery or reset is actually needed.<br><br>- Never invent GUI state, direct QuPath state, or pathology findings that are not grounded in the current screenshot or tool outputs.<br><br>- Never claim a requested viewer magnification was reached unless the current direct QuPath state summary confirms it.<br><br><br>|
|---|

#### BRACS slide-level prompt.

Case summary: Case alias: <CASE_ALIAS> A blinded breast H&E whole-slide image is already loaded in QuPath. Ground-truth annotations are hidden from you. You must determine the single best slide-level BRACS label and, when non-negative, support it with one

or more WSI coordinate localizations.

Task: Choose the single best slide-level BRACS label:

- A. N

- B. PB

- C. UDH

- D. FEA

- E. ADH

- F. DCIS

- G. IC Rules:

- - The image is already loaded in QuPath and ground-truth annotations are hidden.

- - If the final slide-level label is N, localizations must be an empty list.

- - If the final label is not N, output one or more ROI coordinates in full-resolution WSI image pixels.

- - To obtain a coordinate, call viewer_point_to_wsi_coordinate with x and y in 0..999 on the current viewer screenshot, measured from the top-left corner.

- - Use the x and y returned by that tool in the final answer.

- - Choose points on the main tissue image, not on the overview thumbnail, scale bar, or text overlays.

- - wsi_label must be exactly one of: N, PB, UDH, FEA, ADH, DCIS, IC.

- - If uncertain, state uncertainty plainly; do not invent findings.

Return JSON only, with exactly these keys: {

"selected_option": "A|B|C|D|E|F|G",

"wsi_label": "N|PB|UDH|FEA|ADH|DCIS|IC", "confidence": 0.0, "brief_rationale": "...", "localizations": [{"x": 0.0, "y": 0.0}]

}

CAMELYON17 slide-level prompt.

|Case summary: Case alias: <CASE_ALIAS> A blinded CAMELYON17 lymph node H&E whole-slide image is already loaded in QuPath. Ground-truth masks and annotations are hidden from you. This run is slide-level only. Determine whether tumor is present, select the slide-level metastasis<br><br>category, and output tumor coordinates when positive. Task:<br><br>1) Tumor presence?<br><br>A. no_tumor<br><br>B. tumor_present<br><br><br>2) Largest metastasis category for this slide?<br><br><br>A. negative<br><br>B. itc<br><br>C. micro<br><br>D. macro Rules:<br><br><br>- The image is already loaded in QuPath; ground-truth masks and annotations are hidden.<br><br>- This run is slide-level only. Do not answer patient-level staging questions.<br><br>- If tumor is absent and the slide-level label is negative, localizations must be an empty list.<br><br>- If tumor is present, output one or more tumor coordinates in full-resolution WSI image pixels.<br><br>- To obtain a coordinate, call viewer_point_to_wsi_coordinate with x and y in 0..999 on the current viewer screenshot, measured from the top-left corner.<br><br>- Use the x and y returned by that tool in the final answer.<br><br>- Choose points on the main tissue image, not on the overview thumbnail, scale bar, or text overlays.<br><br>- Prefer relative zooming and visible tissue context over trusting nominal magnification metadata alone.<br><br>- If a zoomed-in view lands on blank background or off-tissue area, zoom back out or pan until informative tissue is visible again.<br><br>- If uncertain, state uncertainty plainly; do not invent findings.<br><br><br>Return JSON only, with exactly these keys: {<br><br>"tumor_presence_option": "A|B", "tumor_presence": "no_tumor|tumor_present", "largest_metastasis_option": "A|B|C|D", "largest_metastasis": "negative|itc|micro|macro", "confidence": 0.0, "brief_rationale": "...", "localizations": [{"x": 0.0, "y": 0.0}]<br><br>}<br><br>|
|---|

Per-round QuPath observation template.

|Task: <BENCHMARK_TASK_PROMPT><br><br>Case summary: <CASE_SUMMARY_TEXT><br><br>Current QuPath direct state summary (software/viewer state from QuPath itself): <CURRENT_QUPATH_DIRECT_STATE_SUMMARY><br><br>Current QuPath GUI state summary (rule-based GUI/control state, not pathology evidence): <CURRENT_QUPATH_RULE_BASED_GUI_STATE_SUMMARY><br><br>Current viewer evidence:<br><br>- The current QuPath viewer screenshot is attached in this message.<br><br>- Use the attached screenshot itself for pathology or image-content reasoning. Decide the next best action:<br><br>- either call exactly one tool<br><br>- or provide the final answer<br><br><br>|
|---|

Tool Arguments Purpose and returned information

ping – Checks whether the Slicer WebServer is responding. open_case dicom_dir; spacing; slice_planes;

Opens or recovers a case path with the standard viewer preset; returns baseline views and scene inventory.

view_layout; size

select_volume volume_id; volume_name; view; size Selects an already loaded image volume and returns recovered standard views; used for switching MRI sequences or CT/PET series.

observe view; size; include_metadata Returns the current axial, sagittal, and coronal views,

or a selected view. scroll view; direction; num_slices; step; output;

Scrolls through slices and returns sampled keyframes or a cine.

size

zoom view; factor; center_1000; size Zooms a slice view relative to its current field of view

and returns the updated view.

fit view; size Fits the selected slice view to the full volume and

returns the updated view.

set_slice_offset view; offset; size Sets an axial, sagittal, or coronal slice position directly

in millimeters and returns the updated view.

get_slice_offset_range view Returns the valid minimum and maximum offset range

for the selected view.

recover_standard_views volume_id; volume_name; size Restores axial, sagittal, and coronal views around a

volume center. image_coords_to_ras view; point_1000; bbox_1000; volume_id; volume_name

Converts normalized screen coordinates to physical RAS millimeters for the active loaded volume.

get_viewer_state include_volumes; include_fusion; include_slice_positions

Returns active volume, slice offsets, fusion state, and related viewer state.

wl view; window; level; auto; volume_id; volume_name; size

Sets window/level and returns the updated view.

window_preset preset; view; volume_id; volume_name; size Applies modality presets such as ct_lung, ct_mediastinal, ct_bone, or pet_auto; returns the updated view.

jump ras; view; size Jumps linked views to a known RAS point and returns

the updated views.

set_layout layout; size Sets a standard Slicer layout and returns updated

views.

fusion background_volume_ref; foreground_volume_ref; foreground_opacity; mode; view; size

Sets basic foreground/background slice fusion for already loaded or prealigned volumes, e.g., CT+PET, and returns the updated fused view.

- Table 9: Model-visible viewer-native 3D Slicer tools used by all radiology benchmark modules. The released schemas contain the exact JSON definitions used at runtime.

Tool Arguments Purpose and returned information zoom direction; amount; unit; x; y Changes the QuPath viewer zoom by absolute magnifica-

tion, relative factor, or internal steps; can recenter on a normalized screenshot point.

pan direction; amount; unit; x; y Pans the QuPath viewer or recenters the view; returns the

updated viewer state.

focus_qupath – Brings QuPath to the foreground and returns the current

viewer state.

capture_viewer label Captures the current viewer and stores a labeled screenshot

artifact.

viewer_point_to _wsi_coordinate x; y Converts a normalized point on the current QuPath screen-

shot to full-resolution WSI pixel coordinates.

- Table 10: Model-visible QuPath direct tools used for BRACS and CAMELYON17. The released schemas contain the exact JSON definitions used at runtime.

#### H.3 Model-Visible Tools

The runtime exposes named functions only. Raw Python execution in 3D Slicer and arbitrary script execution in QuPath are not provided to the evaluated agent. Every call is logged with its arguments, returned state, screenshots or artifacts, and final answer text.

Coordinate conventions. For image_coords_to_ras, screen coordinates are normalized to the range 0–999 from the top-left corner of the displayed slice. The call uses either a point or a bounding box center, and the returned value is an RAS coordinate in millimeters for the active loaded volume. For viewer_point_to_wsi_coordinate, x and y are normalized to 0–999 on the current QuPath screenshot from the top-left corner, and the returned coordinate is in full-resolution WSI pixels.

#### H.4 Dynamic Attachments and Logging

Radiology prompts are followed by a scene-inventory text block and baseline axial, sagittal, and coronal image attachments from 3D Slicer. Pathology prompts are followed at each step by a QuPath direct state summary, a rule-based GUI/control summary, and the current viewer screenshot. These attachments are filled separately for each case and therefore are not enumerated in the manuscript. The runtime stores the instantiated prompt, image attachments, tool-call arguments, tool outputs, viewer states, screenshots, generated artifacts, and final answer for each episode, making the execution reconstructable without relying on model-hidden reasoning.

### I Licenses and Terms of Use

MEDFLOW-BENCH uses existing open-source software, public research datasets, and public baseline models only under their published licenses and terms of use. We cite the original asset papers and source pages, retain required copyright and license notices, and follow dataset-specific restrictions. In particular, TCIA-hosted datasets are used under the applicable Creative Commons licenses together with the TCIA Data Usage Policy and Restrictions, including the prohibition on attempting to identify or contact data subjects. BRACS and the MONAI/vista3d model weights are used only for noncommercial research/evaluation purposes. Table 11 summarizes the main third-party assets used in the benchmark and the corresponding licenses or usage terms.

Table 11: Existing assets, licenses, and terms of use.

Asset Type License / terms of use 3D Slicer Software BSD-style license; no restriction on software use. Not

FDA approved; no clinical-use claim is made. QuPath Software GNU GPL v3.0. Not intended for clinical, diagnostic, or

therapeutic use. MONAI Software Apache License 2.0. VISTA3D / MONAI VISTA

Model / operation MONAI VISTA source code is Apache-2.0. The MONAI/vista3d model weights are distributed under the NVIDIA model-weight license for non-commercial research/evaluation use.

LUMIERE Dataset CC0 public-domain dedication on Figshare; used as deidentified public research data. UCSF-PDGM Dataset CC BY 4.0 via TCIA; subject to TCIA Data Usage Policy and Restrictions. NSCLC-Radiogenomics Dataset CC BY 3.0 via TCIA; subject to TCIA Data Usage Policy and Restrictions.

BRACS Dataset Creative Commons Attribution-NonCommercial 4.0 International; used for non-commercial research evaluation with citation.

CAMELYON17 Dataset CC0 public-domain dedication / open-access challenge data. M3D / RadFM Baseline models M3D code is MIT and the evaluated M3D-LaMed checkpoint is listed as Apache-2.0; RadFM code is MIT. Public checkpoints are used only as evaluation baselines under their published project/Hugging Face terms.

