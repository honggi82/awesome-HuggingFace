# arXiv:2511.04307v2[cs.AI]10Nov2025

## GUI-360◦: A COMPREHENSIVE DATASET AND BENCHMARK FOR COMPUTER-USING AGENTS

Jian Mu1, Chaoyun Zhang2,∗Chiming Ni3, Lu Wang2, Bo Qiao2, Kartik Mathur2,

Qianhui Wu2, Yuhang Xie4, Xiaojun Ma2, Mengyu Zhou2, Si Qin2, Liqun Li2, Yu Kang2, Minghua Ma2, Qingwei Lin2, Saravan Rajmohan2, Dongmei Zhang2 1Nanjing University 2Microsoft 3ZJU-UIUC 4Peking University

ABSTRACT

We introduce GUI-360◦, a large-scale, comprehensive dataset and benchmark suite designed to advance computer-using agents (CUAs). CUAs present unique challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multimodal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction.

GUI-360◦ addresses these gaps with an LLM-augmented, largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking stateof-the-art vision–language models on GUI-360◦ reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning and reinforcement learning yield significant gains but do not close the gap to human-level reliability. We release GUI-360◦ and accompanying code to facilitate reproducible research and accelerate progress on robust desktop CUAs.

The full dataset has been made public on https://huggingface.co/ datasets/vyokky/GUI-360.

1 INTRODUCTION

Recent advances in vision–language and large language models have sparked rapid progress toward intelligent agents that automate tasks inside digital environments Zhang et al. (2024a). Such agents interpret natural-language requests, perceive screen content via pixels and/or accessibility (a11y) metadata, plan sequences of operations, and then either navigate the GUI or invoke APIs to complete tasks on a user’s behalf Zhang et al.. They can dramatically reduce user effort for routine productivity tasks and enable novel human–computer workflows. However, realizing this potential requires two tightly coupled capabilities: reliable screen understanding (element grounding or screen parsing) Cheng et al. (2024); Lu et al. (2024); Zheng et al. (2025b) and robust action planning (stepwise action prediction and execution) Zhang et al. (2024b). Both capabilities in turn depend critically on large, diverse, and high-quality datasets grounded in realistic execution contexts Wang et al. (2024c).

We focus on a concrete, under-served class of agents called computer-using agents (CUAs) OpenAI (2025a): agents whose primary operating domain is the desktop computer environment. Desktop

∗Corresponding author.

###### GUI Grounding

###### Screen Parsing

###### Action Prediction

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(85, 66)

###### click(...)

[Figure 8]

[Figure 9]

[Figure 10]

Prediction: [

Instruction: Save the document as my_doc.doc Prediction:

Instruction: Open the Paragraph menu Prediction: (85, 66)

{"name": "Search Query", "bbox": [2, 3, 200, 25]},

{"name": "File Menu", "bbox": [8, 8, 60, 28]},

click(x=42, y=51, button='left', double=False)

{"name": "Font", "bbox": [26, 28, 120, 55]},

click(x=300, y=400, button='left', double=False) type(key="my_doc.doc")

... ]

...

Figure 1: An illustration of GUI Grounding, Screen Parsing and Action Prediction tasks included in our GUI-360◦.

CUAs differ from web Zheng et al. (2024) or mobile Wang et al. (2024b) agents in several important ways. Desktop applications present very high-resolution mixed-content screens, heterogeneous widgets and document formats, arbitrary window layouts (multi-window and multi-monitor settings), and frequently lack standardized accessibility metadata Zhang et al. (2025). Tasks on desktop systems are also often longer-horizon and more compositionally structured (e.g., find a table in a document, transform cells in Excel, then copy results into PowerPoint). These characteristics make desktop CUAs substantially more challenging to train and evaluate than their web or mobile counterparts Zhang et al. (2024a).

Despite growing interest, progress on desktop CUAs is hampered by three persistent gaps. First, there is a scarcity of real-world task collections: existing datasets are often handcrafted or LLMsynthesized Sun et al. (2024), which limits their ability to capture the frequency and diversity of authentic user intents. Second, automated pipelines for data collection and annotation are largely missing Nayak et al. (2025). Manual execution and labeling of desktop interactions is expensive, error-prone, and difficult to scale, making it impractical to generate multi-modal execution trajectories at scale. Third, no unified, large-scale benchmark exists that supports the breadth of tasks needed for comprehensive evaluation Nayak et al. (2025); Li et al. (2025). Prior datasets typically focus on a single aspect—such as element detection, a single modality, or a narrow application subset, rather than jointly enabling GUI grounding, screen parsing, and action prediction with execution traces and failure cases.

To fill these gaps, we introduce GUI-360◦, a comprehensive dataset and benchmark suite for desktop computer-using agents. GUI-360◦ is built around three design goals: realism, scalability, and task breadth. Concretely, the dataset and benchmark provide the following properties:

- 1. Real-world, high-frequency queries. Task intents are harvested from authentic sources (search logs, community forums, and in-app help content) to reflect common user needs and highfrequency workflows.
- 2. Automated collection and annotation pipeline. We present an LLM-augmented, largely automated pipeline for template construction, task instantiation, batched execution, and LLM-driven quality filtering, minimizing human intervention while preserving execution realism.
- 3. Comprehensive multi-modal annotations. Each example contains full-resolution screenshots, accessibility metadata, natural-language goals, intermediate agent “thoughts,” and stepwise action trajectories (including successful and failed runs). These assets jointly support three canonical tasks: (a) GUI Grounding: given a step-level plan, predict the screen coordinates or UI element to interact with; (b) Screen Parsing: given a screenshot, enumerate the set of interactable UI elements and their properties; and (c) Action Prediction: given the current state and user intent, predict the next action (click/type/select/API call). We show an illustration of included three tasks in Figure 1.
- 4. Large scale and practical coverage. GUI-360◦ contains over 1.2M executed action steps spanning thousands of trajectories across widely used Windows office applications (Word, Excel,

Table 1: Comparison of GUI datasets across dimensions. A checkmark (✓) indicates support, while a cross (✗) indicates not supported.

Query Source

Data Collection

Dataset

Task Samples

Action Reasoning A11y Info.

Fail Case

Grounding Parsing Action Pred. GUI API ScreenSpot

Humandesigned

✓ ✗ ✗ 1,200+ Human N/A N/A ✗ ✗ ✗

Humandesigned

✓ ✗ ✗ 1,581 Human N/A N/A N/A N/A N/A DeskVision Online ✓ ✗ ✗ 54,855 Auto. N/A N/A ✗ ✗ ✗ UI-Vision

ScreenSpot-Pro

Humandesigned

✓ ✗ ✓ 8,227 Human ✓ ✗ ✗ ✗ ✗

GUI-360◦ In-App/Online/ Search

✓ ✓ ✓ 1,225,177 Auto. ✓ ✓ ✓ ✓ ✓

PowerPoint), and is designed so that the template set and pipeline can be extended to other desktop applications.

- 5. Hybrid GUI+API action space. To reflect modern CUA architectures Zhang et al. (2025); Zhang et al., our action space mixes direct GUI operations with higher-level API calls where available, enabling evaluation of both perception-driven and API-assisted strategies.

Table 1 compares GUI-360◦ with existing GUI datasets across key dimensions. Unlike prior efforts that focus narrowly on grounding or small-scale scripted tasks, GUI-360◦ provides full coverage of grounding, parsing, and action prediction with a large-scale, automatically collected corpus. Notably, it is the first dataset to include accessibility information, reasoning supervision, and both GUIand API-level actions, making it a uniquely comprehensive benchmark for CUA research.

To assess GUI-360◦ ’s utility, we benchmark state-of-the-art vision–language models, asking two questions: (i) how well do existing models generalize to realistic desktop CUAs without adaptation, and (ii) how much can fine-tuning on GUI-360◦ bridge the gap? Our results reveal consistent patterns: off-the-shelf models struggle with grounding in heterogeneous layouts and often fail in stepwise action prediction, leading to cascading errors. Training on GUI-360◦ yields significant gains. These findings underscore both the limitations of current models and the value of GUI-360◦ as a scalable, challenging benchmark for driving progress in CUAs.

- 2 RELATED WORK

GUI and Computer-Using Agents LLMs have enabled agents Zhang et al. (2024a) that automate tasks across web Zheng et al. (2024; 2025a), mobile Wang et al. (2024b;a; 2025), and desktop platforms Zhang et al. (2024b; 2025); Qin et al. (2025). Desktop environments are particularly challenging for the CUAs due to high-resolution displays and complex layouts. Such agent typically rely on accessibility metadata or visual screen understanding Gur et al. (2023); Xie et al. (2023). When accessibility information is missing, they resort to screen parsing and visual grounding, as in SeeClick Cheng et al. (2024), OmniParser Lu et al. (2024), and GUI-Actor Wu et al. (2025). UFO Zhang et al. (2024b; 2025) pioneered hybrid approaches that combine accessibility with screen parsing for more reliable control detection, while recent efforts further integrate APIs for efficiency Zhang et al..

Ultimately, progress depends on large-scale data to support robust screen understanding and tool use. GUI-360◦ aims to fill this gap by providing a comprehensive dataset and benchmark for GUI grounding, action prediction, and screen parsing.

Data and Benchmarks for CUA A key obstacle in building effective CUAS lies in obtaining high-quality training data and reliable benchmarks. While such resources are increasingly available for web Deng et al. (2023); Zhou et al. (2023) and mobile domains Rawles et al. (2023; 2024), the desktop setting remains comparatively underexplored despite its greater complexity. Several efforts have emerged in this space. UI-Vision Nayak et al. (2025) provides a human-annotated desktop benchmark with bounding boxes, UI labels, and action trajectories. DeskVision Xu et al. (2025)

[Figure 11]

###### Automatic Trajectory Collection

[Figure 12]

###### Query Acquisition

[Figure 13]

Concrete Task: Change the font size of all column

[Figure 14]

titles to 20.

Prototypical Query Sourcing

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Evaluation and Post-processing

Step 1: Select all column titles.

Step N: Select the "20" font size.

[Figure 21]

Thought: I need drag on … Action: drag(X0=100, y0=120, ...)

Thought: To complete the task, ... Action: Click(234, 134)

[Figure 22]

[Figure 23]

[Figure 24]

Collected Trajetories

###### A11y information

A11y information

Change the theme on PPT. Reorder the spreadsheet.

{Botton – Font 20, ...} {Botton – Font 35, ...} ...

{Botton - Insert, } {Botton - Chart, ...}

[Figure 25]

[Figure 26]

[Figure 27]

How to set font size?

Trajectory Validation

[Figure 28]

[Figure 29]

[Figure 30]

Task

[Figure 31]

[Figure 32]

...

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Instantiation

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Data Sanitization

[Figure 41]

[Figure 42]

[Figure 43]

Quality Filtering

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Data

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

...

Structuring

[Figure 56]

[Figure 57]

Environment Template Construction

Screenshot Screenshot

Figure 2: The data collection pipeline for GUI-360◦.

introduces a cross-OS dataset for desktop region captioning. OfficeBench Wang et al. (2024d) offers a live testing environment for office applications through handcrafted cases and oracle evaluation. However, these datasets and benchmarks either require costly human annotation of trajectories and GUI states, or cover only narrow subsets of tasks, limiting their scalability and diversity.

Our work, GUI-360◦, addresses this gap by introducing an automated data collection pipeline that generates large-scale resources for GUI grounding, screen parsing, and action prediction. This design makes GUI-360◦ the most comprehensive and scalable dataset–benchmark suite to date for training and evaluating GUI agents.

- 3 THE COLLECTION OF GUI-360◦

The construction of GUI-360◦ follows a three-stage pipeline designed to maximize scalability while minimizing human effort, as shown in Figure 2.

- 1. Query Acquisition. We begin by collecting real-world user queries from reliable sources and augmenting them with synthetic variants to ensure coverage of diverse and realistic task intents.
- 2. Automatic Trajectory Collection. We design a specialized CUA named TrajAgent to execute tasks in an automatic, consistent and high-quality manner. The TrajAgent generates and collect detailed trajectories that jointly support GUI grounding, screen parsing, and action prediction, enabling multi-task supervision from a single execution.
- 3. Evaluation and Post-processing. Finally, we apply automated evaluators and systematic postprocessing to verify correctness, filter noise, and enhance overall data quality.

Together, these stages yield a scalable and comprehensive pipeline for constructing GUI-360◦. By integrating real-world task diversity, automated trajectory collection, and rigorous quality control, our approach provides a unified dataset and benchmark that supports multiple core capabilities of GUI agents.

3.1 QUERY ACQUISITION

High-quality and actionable user queries are the foundation of a reliable dataset, as they determine both task realism and execution fidelity Xu et al. (2025). To capture queries that faithfully reflect real-world user needs, we design a dedicated four-stage acquisition pipeline (Figure 1): (i) Prototypical Query Sourcing. We gather prototypical, high-frequency queries about software usage from diverse and trustworthy real-world sources, ensuring coverage of practical user intents. (ii) Environment Template Construction. Based on the collected queries, we design template software environments that can realistically support their execution. These templates encode the minimal yet sufficient context needed for task instantiation. (iii) Task Instantiation. Each query is grounded into a suitable environment by matching it to the best-fitting template. We then rephrase the query into a concrete, executable form tailored to the selected environment. (iv) Quality Filtering. Finally, we apply post-processing to discard low-quality or ambiguous queries, retaining only those

Table 2: Raw query statistics across applications and sources. Source Word Excel PowerPoint In-App 274 159 316 Online 1,914 3,393 1,701 Search 25,715 25,000 19,681 Total 27,903 28,552 21,698

that are actionable and fully grounded in executable environments. This four-stage pipeline ensures that the resulting queries are both realistic and executable, providing a strong basis for collecting high-quality trajectories in later stages of the GUI-360◦ pipeline.

Prototypical Query Sourcing. Most existing CUA datasets rely heavily on human-crafted instructions or LLM-generated queries, which often fail to reflect real-world usage patterns or capture high-frequency tasks. To better ground our dataset in authentic user behavior, we source prototypical task descriptions from three complementary channels:

- 1. In-App Help Content (In-APP): We mine built-in help documentation and tutorials of software applications, which provide standardized, well-structured task descriptions covering core functionalities.
- 2. Online Websites (Online): We crawl forums, Q&A platforms, and community-driven websites to obtain diverse and authentic task descriptions contributed by real users.
- 3. Search Queries (Search): We extract queries from search engines that mention or relate to the target software. These queries capture pressing, high-frequency user needs and common practical challenges.

We show the queries distribution from different source in Table 2. By combining these sources, we ensure that the collected queries balance authenticity (from search and community data) with completeness (from in-app documentation), resulting in a broad and realistic coverage of user intents.

Environment Template Construction. Each user query must be grounded in a suitable environment that provides the necessary context for agent execution. For example, a task such as “make the first line of text bold” in Word requires a document containing editable text; without such a setup, the query is not actionable. Naively creating a bespoke environment for every query would be prohibitively expensive and redundant, since many queries share common contextual requirements.

To address this, we introduce an environment template construction process that systematically amortizes environment setup across queries. Specifically, we leverage GPT to analyze each query and extract its underlying requirements (e.g., the presence of text, a table, or an image). Queries with similar requirements are clustered and abstracted into environment template descriptions, which specify the minimal context needed for execution. We then manually instantiate a curated set of high-frequency templates from these descriptions.

In practice, we design 30 templates for Word, 30 for Excel, and 6 for PowerPoint. Despite this relatively small set of 66 templates, the coverage is substantial: a single template can accommodate diverse content and scenarios, enabling the collection to support approximately 95% of prototypical queries. This template-driven strategy dramatically reduces human effort by avoiding per-query environment creation, and ensure consistent and reproducible environments, enabling robust execution and trajectory collection at scale.

Task Instantiation. Prototypical queries collected from real-world sources are often vague and underspecified (e.g., “How to make text bold?”). To be executable by an agent, each query must be grounded in a specific environment and reformulated into an actionable instruction (e.g., “Make the phrase ‘Hello World’ bold in the Word document”). We refer to this process as task instantiation. To systematically achieve this, we design an automated two-stage pipeline:

- 1. Template Matching. For each query, we retrieve candidate environment templates. Each template is defined by a set of contextual constraints (e.g., a Word document containing text, or an

- Excel sheet with numerical entries). Using an LLM, we provide the query together with textual descriptions and screenshots of all available templates. The LLM identifies the most suitable template by reasoning over query requirements and environment affordances. Queries that fail to match any template are discarded.
- 2. Query Concretization. Once a template is selected, the LLM rephrases the vague query into a fully instantiated one, grounded in the chosen environment. For example, the generic request “make text bold” is concretized as “make the text ‘Hello World’ bold in the current document”. This ensures the query is unambiguous, executable, and directly tied to a well-defined environment.

This pipeline is fully automated and enforces that all instantiated tasks are (i) actionable, by guaranteeing an associated environment exists, (ii) concrete, by eliminating ambiguity in the query, and (iii) scalable, as the process can be applied to thousands of queries with minimal human intervention. In practice, this approach yields a large collection of executable tasks while maintaining high fidelity to real-world user intent.

Quality Filtering. Although task instantiation produces a large set of grounded queries, not all of them are suitable for reliable agent training. Many instantiated tasks may suffer from contextual mismatches, external dependencies, or inherent ambiguities. To ensure robustness, we design a task filtering pipeline that employs an LLM as an automatic quality gate. Concretely, the LLM-based judge evaluates each candidate task against a set of well-defined constraints. Tasks are discarded if they:

- 1. Non-Executable (NONEXEC): Inputs that do not describe a concrete action are removed. This includes subjective statements, vague preferences, or general inquiries that cannot be directly executed. For instance, instructions containing words such as “custom,” “you want,” or undefined operations like “edit text” without specifying the target element fall under this category.
- 2. Cross-Application Dependency (CROSSAPP): Tasks that require interaction with applications beyond {app} are excluded. This includes operations that involve opening or manipulating content in external software (e.g., Excel, Edge, File Explorer, or system settings). Representative examples include merging files across applications, printing documents (which requires printer integration), or exporting data to a third-party tool.
- 3. Version Management (VERCTRL): Tasks that involve checking, updating, downgrading, or modifying the version of {app} are discarded. Since version management depends heavily on system environment and external factors, these tasks are not considered executable within the scope of our benchmark.
- 4. Template Dependency (TPLMISS): Tasks that rely on specific document or workspace templates that are absent from the provided context are excluded. For example, instructions that assume the existence of a predefined table, chart, or object not available in the given file are categorized as TPLMISS. Note that application-wide settings (e.g., enabling dark mode) do not fall under this restriction, as they do not depend on document templates.
- 5. Irrelevant or Invalid (INVALID): Remaining cases that do not fit into the above categories, or are otherwise infeasible due to irrelevance, ambiguity, or context mismatch, are marked as invalid. For example, a task unrelated to {app} or lacking sufficient contextual information for execution would be discarded under this category.

- Table 3 summarizes the outcome of our LLM-based task filtering across Word, Excel, and PowerPoint. Overall, we retain 59,553 out of 79,075 candidate tasks (75.3%) as NORMAL (self-contained, concretely specified, and template-feasible); the remaining 24.7% are filtered for various reasons.

Summary. The complete query acquisition pipeline transforms raw, vague user queries into a curated collection of high-quality, executable tasks. Starting from diverse sources of authentic user requests, we progressively ground them into concrete environments through template construction, instantiate them into actionable forms, and finally enforce strict quality filtering. This systematic process yields a dataset that is both scalable and faithful to real-world user intent, while maintaining the rigor necessary for robust agent training.

Table 3: Distribution of task categories after filtering, shown as percentages of total candidate tasks for each application.

Category Word Excel PowerPoint CROSSAPP 15.61% 13.27% 12.13% TPLMISS 2.93% 4.18% 7.51% NONEXEC 1.56% 1.68% 3.65% VERCTRL 0.17% 0.13% 0.27% INVALID 2.70% 6.98% 1.46% NORMAL 77.02% 73.76% 74.97%

##### MAgent

[Figure 58]

[Figure 59]

Concrete Task

Perception

[Figure 60]

Assigned Subtasks

Collected Trajetories

[Figure 61]

Screenshots

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Recorder

[Figure 71]

[Figure 72]

##### EAgent

Action

Environment

Executor

Accessibility Information

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

MCP Severs

{"name": "File", "type": "Menu","bbox": [...]}, {"name": "Insert", "type":

[Figure 79]

GUI Actions App APIs

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

"Menu", "bbox": [...]},

[Figure 86]

[Figure 87]

……

Figure 3: The overall architecture of the TrajAgent.

- 3.2 AUTOMATIC TRAJECTORY COLLECTION

Once queries have been instantiated and grounded, the next step is to generate corresponding action plans, execution trajectories, and outcomes. Traditionally, this stage relies on human operators to perform tasks and log interactions, which is costly and difficult to scale. To overcome this limitation, we develop a specialized execution framework, TrajAgent, that automatically completes queries in batch (2) and records detailed execution data trajectories. The workflow consists of three main phases:

- 1. Environment Preparation. For each task, TrajAgent initializes the corresponding environment template and ensures all preconditions for execution are satisfied.
- 2. Task Execution and Data Logging. The agent performs the query step-by-step, generating a complete action trajectory while recording the full GUI state at each step. Additional relevant data, such as intermediate screenshots, control properties, and execution metadata, are also captured.
- 3. Environment Reset. After execution, TrajAgent closes or resets the environment to a clean state, preparing it for the next task in the batch.

This fully automated process ensures that every task is executed reliably, trajectories are captured consistently, and human intervention is entirely eliminated. By integrating environment preparation, execution, and logging into a single pipeline, TrajAgent enables large-scale, high-fidelity trajectory collection suitable for training and benchmarking CUAs.

- 3.2.1 TRAJAGENT DESIGN

The core requirement for large-scale trajectory collection is twofold: the executor must (i) complete instantiated queries with a high success rate to maximize data efficiency, and (ii) record every element required for dataset construction (screen snapshots, accessibility metadata, action logs,

Table 4: Success rate of the two-stage execution strategy across applications. Word Excel PowerPoint Total

- Round 1 (GPT-4o) 16.65% 9.27% 8.00% 11.63%
- Round 2 (GPT-4.1) 16.03% 18.20% 14.91% 16.38% Overall 30.50% 25.78% 21.71% 26.09%

pre/post states, etc) with strict fidelity. To meet these requirements we design TrajAgent, an orchestrated, multi-agent execution framework that reliably completes concrete tasks and produces high-fidelity execution traces suitable for downstream training and evaluation.

Architecture overview. TrajAgent follows an orchestration pattern Zhang et al. (2024b) composed of a MasterAgent (MAgent), a pool of ExecutionAgents (EAgent), with a set of auxiliary services (Perception, Action Executor, and Recorder), as shown in Figure 3. The MAgent receives an concrete task and decomposes it into a sequence of manageable subtasks (planning). Each subtask is dispatched to an available EAgent for execution. EAgents operate as lightweight workers that (a) perceive the current GUI state, (b) select or synthesize the next low-level action (click/type/select/API call), (c) execute the action via UI automation or API invocation, and (d) return observations and status back to the MAgent. The Recorder persistently logs the full GUI state and metadata before and after each action, ensuring a temporally-aligned dataset.

Perception. Within each EAgent, the Perception service captures a full-resolution screenshot at every decision step and queries the Windows accessibility API (UI Automation) Haverty (2005) to extract a list of actionable controls (name, type and exact bounding box). The accessibility-derived control metadata is rendered on the screenshot as a Set-of-Mark (SoM) Yang et al. (2023). The EAgent uses both the raw screenshot and the SoM/controls for decision-making: screenshots provide visual context while accessibility metadata supplies precise semantic and locational information, reducing visual reasoning overhead.

Action Executor. The Action Executor utilizes app-specific MCP servers Hou et al. (2025) to provide an extensible set of tools for the EAgent. In addition to conventional GUI actions (e.g., mouse clicks, keyboard input), our design incorporates app-level API actions, reflecting modern CUA practices. These API actions improve task efficiency and serve as reliable fallbacks when GUI interactions are prone to failure. At each step, the EAgent selects the most appropriate action based on its current observation, internal reasoning, and task plan, following the classical ReAct paradigm Yao et al. (2023). This iterative perception–reasoning–action loop continues until the task is fully completed, ensuring both robustness and fidelity in trajectory collection.

Recorder. The Recorder is responsible for collecting all multi-modal information at each execution step to construct the dataset. This includes screenshots, accessibility metadata, and agent outputs. Importantly, a single execution of the agent produces data for all downstream tasks, significantly improving data efficiency. Table 5 summarizes the input and output for each task type. Accessibility-derived data ensures precise element locations and properties, while screenshots provide full visual context.

Two-Stage Execution. Leveraging the components described above, queries are executed in batches within a Windows Sandbox, with the EAgent automatically collecting all required data. To reduce dependency on the capabilities of a single model and improve coverage, we adopt a twostage execution strategy. In the first stage, GPT-4o serves as the base model to complete the queries. Any queries that fail in this stage are then re-executed using a stronger model GPT-4.1. As shown in

- Table 4, the two-stage execution strategy substantially improves success rates compared to relying on a single model. In the first round, GPT-4o captures a non-trivial portion of tasks, particularly in Word. In the second round, GPT-4.1 recovers many of the failures from GPT-4o, with notable gains in Excel and PowerPoint. Together, this staged approach boosts overall completion to 26.09%, showing that cascading models of complementary strengths increases both success rate and dataset versatility, while avoiding over-reliance on a single model.

Table 5: Task input-output specifications for dataset collection. Task Input Output GUI Grounding Application screenshot,

Operation coordinates of the target element, obtained via accessibility APIs

Agent’s thought at the current step

Screen Parsing Application screenshot List of all actionable controls on screen with name and bounding box, e.g., {"name": "Open Menu", "bbox": [12,34,56,78]}

Action Prediction User query, Application screenshot, Accessibility information (optional)

Action call, with optional metadata such as agent’s thought and plan

- 3.3 EVALUATION AND POST-PROCESSING

To ensure the high quality and usability of the automatically collected trajectories, we perform a three-stage post-processing procedure: (i) Trajectory Validation. Each trajectory is automatically evaluated to retain only successful and executable task executions, ensuring that downstream training and evaluation are based on realistic completions. (ii) Data Sanitization. Low-quality steps, incomplete records, or any data that fail to meet predefined quality criteria are removed. This step eliminates noise and increases the overall reliability of the dataset. (iii) Data Structuring. The cleaned trajectories are reformatted and normalized into the required structure for the 3 downstream tasks. This includes standardizing screenshot metadata, accessibility information, action calls, and annotations to create a consistent, machine-readable dataset.

Together, these stages guarantee that the final dataset is both high-quality and fully compatible with diverse CUA training and benchmarking pipelines.

Trajectory Validation. To ensure the reliability of collected data, each trajectory undergoes automatic validation. We design an evaluation agent, EvaAgent, which leverages GPT-4.1 in an LLMas-a-judge paradigm Gu et al. (2024). EvaAgent inspects the trajectory step by step, including the screenshot, accessibility information, executed actions, intermediate thoughts, and the final application state. Following prior work Wang et al. (2024c), it employs a chain-of-thought style reasoning process to decompose the query into several fine-grained evaluation criteria. A trajectory is marked as successful only if all criteria are satisfied, thereby enforcing a stricter notion of task completion.

To assess its reliability, we conducted a small-scale study on 100 randomly sampled trajectories. EvaAgent’s judgments achieved 86% agreement with human annotators, demonstrating that it provides sufficiently accurate and scalable validation. Compared to hard-coded scripts or brittle oracle rules, this approach offers greater flexibility across diverse applications and enables rapid filtering of high-quality, successful trajectories at scale.

Data Sanitization. After validation, we perform a final cleaning step to ensure completeness and consistency of the collected trajectories. This involves removing any step that lacks an executed action, a screenshot, or essential metadata required for downstream tasks. Such sanitization further improves the overall data quality and ensures that only fully executable, well-documented steps are retained for model training and evaluation.

Data Structuring. Finally, we transform the sanitized data into a standardized JSON format tailored for model consumption, following the input–output specifications summarized in Table 5. For the Action Prediction task, we provide two input modalities: visual-only and visual+a11y:

- • Visual-only: The model receives raw screenshots as input. Interaction-related arguments (e.g., click positions) are represented as the absolute coordinates of the center of the corresponding bounding box.

Table 6: Training and test dataset statistics across domains (Word, Excel, PowerPoint). GUI-360◦-Train

Word Excel PowerPoint Total

Total Trajectories 5,633 4,348 3,769 13,750 Total Steps 41,742 29,363 34,263 105,368 Average Steps per Trajectory 7.41 6.75 9.09 7.66 Steps for Grounding Tasks 30,695 22,319 26,473 79,487 Steps for Screen Parsing 41,742 29,363 34,263 105,368 Steps for Action Prediction 41,742 29,363 34,263 105,368 Total Elements 3,270,104 12,211,852 2,186,738 17,668,694 Total Images 83,484 58,726 68,526 210,736

- Average Elements per Image 78.34 415.89 63.82 167.69

- GUI Action Rate (%) 76.1 80.7 87.4 81.0 API Action Rate (%) 23.9 19.3 12.6 19.0

GUI-360◦-Bench

Word Excel PowerPoint Total

Total Trajectories 1,409 1,087 943 3,439 Total Steps 10,597 7,175 8,512 26,284 Average Steps per Trajectory 7.52 6.60 9.03 7.64 Steps for Grounding Tasks 7,784 5,444 6,552 19,780 Steps for Screen Parsing 10,597 7,175 8,512 26,284 Steps for Action Prediction 10,597 7,175 8,512 26,284 Total Elements 839,273 2,940,016 545,328 4,324,617 Total Images 21,194 14,350 17,024 52,568 Average Elements per Image 79.20 409.76 64.07 164.53

- GUI Action Rate (%) 76.2 80.2 87.5 81.0 API Action Rate (%) 23.8 19.8 12.5 19.0

- • Visual+a11y: The model additionally receives the list of actionable elements from the accessibility API, which are also annotated on the screenshot using the Set-of-Mark (SoM) representation. Interaction arguments are expressed as element ID and name, chosen from the provided element list. This reduces the need for explicit coordinate prediction and lowers the visual grounding overhead.

For evaluation, we partition the trajectories into training (80%) and benchmark (GUI-360-Bench, 20%) splits. All three tasks—GUI Grounding, Screen Parsing, and Action Prediction—share the same data partition to maintain consistency across evaluations.

- 3.4 GUI-360◦ STATISTICS

Following the pipeline described above, we construct a comprehensive dataset, GUI-360◦, for training across three core GUI tasks, and a companion benchmark, GUI-360◦-Bench, for systematic evaluation.

Scale. Table 6 summarizes the statistics of GUI-360◦-Train (80%) and GUI-360◦-Bench (20%). In total, GUI-360◦ contains 13,750 trajectories with over 105k steps, averaging 7.66 steps per trajectory. The dataset also provides 210k screenshots paired with 17.7M annotated UI elements, yielding rich multimodal supervision at both the visual and accessibility levels. For evaluation, GUI360◦-Bench adds a further 3,439 trajectories and 26k steps, maintaining a similar distribution of average step length and GUI/API action rates. In addition, we include 62,170 trajectories comprising 1,093,525 steps for failure cases, which can serve as valuable signals for reinforcement learning, similar to the approach in Wang et al. (2024c). These failure cases capture challenging or error-prone situations that models often struggle with, providing rich supervision for improving robustness and reliability.

This unprecedented scale, both in interaction traces and annotated elements, makes GUI-360◦ one of the largest and most comprehensive resources for GUI learning—sufficiently large to train highcapacity models and to rigorously benchmark their generalization in realistic desktop environments.

Diversity. Beyond scale, GUI-360◦ emphasizes breadth and functional diversity. Using GPT-4o, we classified user queries into finegrained categories, as shown in Figure 4. The corpus spans Word (41.0%), Excel (31.6%), and PowerPoint (27.4%), each with rich internal coverage. Word tasks range from text formatting to layout and review, Excel covers data entry, formulas, and visualization, while PowerPoint emphasizes content editing, design, and transitions. This balance ensures exposure to both frequent operations and rarer, long-tail behaviors. As a result, models trained on GUI360◦ are encouraged to generalize across routine workflows while remaining robust to less common but practically important tasks, making it a comprehensive and challenging benchmark for desktop CUA research.

[Figure 88]

- 4 EXPERIMENT

Figure 4: Dataset Composition. For each app, tasks are divided into six categories according to their core operational intent.

Our experimental evaluation proceeds in two stages. First, we perform an out-of-the-box evaluation of several state-of-the-art vision– language and agent models on GUI-360◦Bench to measure their zero-/few-shot capabilities and to diagnose common failure modes. Second, we investigate how targeted training on GUI360◦ (supervised fine-tuning and policy optimization) improves model performance and robustness. All experiments use the same data partitioning: the training split described in Section 3.3 (80%) and the held-out benchmark GUI-360◦-Bench (20%).

- 4.1 GUI GROUNDING

We begin with the GUI grounding task: given a natural-language task description and the current GUI state, the model must predict the screen location for the next interaction (represented as a 2D coordinate). Predicted coordinates are evaluated against the accessibility-derived bounding box of the target element.

Baselines. We evaluate a mix of general-purpose and domain-specialized models. The generalpurpose VLM/LLM baselines are GPT-4o Hurst et al. (2024), GPT-4.1 OpenAI (2025b), o3 OpenAI (2025d), and GPT-5 OpenAI (2025c). We also include several open-source and grounding-focused models: Qwen-VL-2.5 (7B) Bai et al. (2025), UGround-7B Gou et al. (2024), Aguvis-7B Xu et al. (2024), UI-TARS-1.5 (7B) Qin et al. (2025), and GUI-Actor (7B) Wu et al. (2025). Finally, we report results for supervised fine-tuned variants (Qwen-2.5 7B-SFT, UI-TARS-1.5 7B-SFT) that are trained on the GUI-360◦ training split.

Performance Metrics. The primary evaluation metric for GUI grounding is accuracy, defined as the proportion of predictions where the predicted coordinate cˆi lies within the bounding box of the corresponding ground-truth target element bi. Formally,

N

1 N

Acc =

⊮{cˆi ∈ bi},

i=1

where N is the total number of test cases and ⊮{·} is the indicator function.

Table 7: Performance of different models on the GUI grounding task across applications. Model Word Excel PowerPoint Overall

- GPT-4o 15.22% 5.08% 5.41% 9.38% GPT-4.1 17.30% 7.48% 7.01% 11.41% o3 36.62% 19.44% 31.06% 29.96%
- GPT-5 34.52% 20.31% 17.36% 25.34% Qwen-2.5-VL-7B 38.09% 26.76% 41.55% 35.78% UGround-7B 57.44% 43.09% 59.53% 53.85% Aguvis-7B 53.14% 37.69% 59.57% 50.50% UI-TARS-1.5 7B 63.50% 58.61% 64.21% 62.27% GUI-Actor 7B 54.84% 45.84% 62.68% 54.50% Qwen-2.5-VL-7B-SFT 84.11% 79.20% 82.84% 82.30% UI-TARS-1.5 7B-SFT 84.73% 79.84% 81.98% 82.49%

We report accuracy separately for each application domain (Word, Excel, and PowerPoint) as well as an overall aggregated score across all benchmark examples. To highlight the effect of task-specific adaptation, we evaluate two settings: (i) the zero-shot performance of each baseline model directly on GUI-360◦-Bench, and (ii) the performance after supervised fine-tuning (SFT) on the GUI-360◦ training set.

- Performance Comparison. Table 7 reports the evaluation results of different models on GUI360◦-bench for the GUI grounding task. We observe that general-purpose GPT models (e.g., GPT-4o and GPT-4.1) achieve only modest performance, with overall accuracy below 12%. More advanced general models such as GPT-o3 and GPT-5 show improvements (20–30%), yet still struggle with precise GUI grounding. Domain-specific pretraining brings substantial gains: models like UGround-7B and GUI-Actor 7B surpass 50%, demonstrating the effectiveness of grounding-oriented pretraining. Finally, supervised fine-tuning (SFT) on GUI-360◦ yields the largest performance leap, with Qwen2.5 7B-SFT and UI-TARS-1.5 7B-SFT achieving over 82% accuracy across applications. This progression clearly highlights the value of GUI-360◦ for both training and benchmarking: it not only reveals the limitations of general-purpose models on GUI tasks but also provides high-quality training data that enables fine-tuned models to achieve state-of-the-art performance.

- 4.2 SCREEN PARSING

The screen parsing task requires a model to take a clean screenshot as input and output the complete set of interactable UI elements on the screen. Each predicted element consists of a semantic name (e.g., “Start Menu”) and a bounding box. This task is challenging due to heterogeneous widgets, dense layouts, occlusions, and mixed content (text, icons, images).

Baselines. We evaluate both general-purpose VLMs and specialized screen-parsing models. General models include GPT-4o Hurst et al. (2024), GPT-4.1 OpenAI (2025b), GPT-o3 OpenAI (2025d), GPT-5 OpenAI (2025c), and Qwen-VL-2.5 (7B) Bai et al. (2025). Specialized baselines include OmniParser and OmniParser-v2 Lu et al. (2024).

Evaluation metrics. We measure parsing quality along three complementary axes: (i) element detection accuracy (precision / recall / F1), (ii) localization quality (mean IoU on matched pairs), and (iii) semantic name accuracy (average text embedding similarity on matched pairs). All metrics are computed per image and then averaged across the benchmark (macro-average).

Let G be the ground-truth set of elements for an image and P the predicted set. We obtain a one-toone matched set M ⊆ P × G by performing greedy bipartite matching sorted by descending IoU, and keeping only pairs with IoU > 0.5. For a predicted box p and ground-truth box g we use the standard intersection-over-union:

IoU(p,g) =

area(p ∩ g) area(p ∪ g)

.

For each image i define:

2 · Precisioni · Recalli Precisioni + Recalli

Precisioni = |Mi| |Pi|

, Recalli = |Mi| |Gi|

, F1i =

### ,

where | · | denotes set cardinality. The reported precision, recall, and F1 are the macro-averages across images:

Precision =

N

1 N

Precisioni, Recall =

i=1

N

1 N

Recalli, F1 =

i=1

N

1 N

F1i.

i=1

To quantify localization quality, we compute the mean IoU for each image i over the matched pairs Mi. If an image has no matched pairs (|Mi| = 0), we define its IoU as 0. The overall mean IoU is then the macro-average across all images:

IoU =

N

1 N

i=1

1

|Mi| (p,g)∈Mi IoU(p,g), |Mi| > 0 0, |Mi| = 0

.

Similarly, for semantic-name accuracy, we embed predicted and ground-truth names using a sentence encoder ϕ(·) (e.g., a sentence-transformer) and compute the cosine similarity for each matched pair. If an image has no matches, we assign a similarity of 0 for that image. The macro-average over all images is:

Sim =

N

1 N

i=1

⟨ϕ(namep),ϕ(nameg)⟩

1 |Mi| (p,g)∈Mi

∥ϕ(namep)∥ ∥ϕ(nameg)∥, |Mi| > 0 0, |Mi| = 0

### .

Together, these metrics separate whether elements are detected (precision/recall/F1), how precisely they are localized (mean IoU), and how well their semantic roles are recovered (mean embedding similarity).

- Performance Comparison. Table 8 presents a detailed comparison of general-purpose VLMs and specialized screen parsing models across Word, Excel, and PowerPoint. Overall, general-purpose models such as GPT-4o, GPT-4.1, GPT-o3, and GPT-5 struggle with both element detection and localization, achieving low F1 scores (0.019–0.128) and moderate mean IoU values (0.229–0.578). Notably, GPT-o3 exhibits the highest overall F1 among the general models (0.128) and maintains relatively strong localization (IoU 0.578), but its recall remains limited, indicating many missed elements. GPT-4.1 and GPT-5 show uneven performance across applications: GPT-5 performs best on Excel (F1 0.126) but poorly on PowerPoint (F1 0.059), suggesting sensitivity to layout complexity and domain-specific content.

In contrast, specialized parsers significantly outperform general-purpose models in all metrics. OmniParser and OmniParser-v2 achieve overall F1 scores above 0.40 and mean IoU above 0.73, with strong text similarity (0.565–0.568), demonstrating robust detection, accurate localization, and reliable semantic recovery. The incremental improvement from OmniParser to OmniParser-v2 is modest but consistent, reflecting refinement in handling dense layouts and occlusions. Application-wise, Word and PowerPoint benefit most from these specialized models due to dense interactive regions, while Excel remains challenging because of its compact grid structure, though performance still exceeds general-purpose VLMs.

These results reveal two key insights: (i) general-purpose VLMs are limited in screen parsing due to the need for fine-grained spatial reasoning and UI semantics, and (ii) task-specific training, as in OmniParser, provides substantial gains in both element coverage and semantic correctness, highlighting the necessity of specialized architectures for accurate and reliable screen understanding.

- 4.3 ACTION PREDICTION

The action prediction task bridges the gap between a user’s natural language command and the executable action calls required by the agent. This represents the ultimate goal of contextual user

- Table 8: Comparison of different models across domains (Precision, Recall, F1, Text Similarity, Avg IOU Accuracy).

Model Domain Precision Recall F1 Text Sim. Avg IOU

Word 0.040 0.017 0.024 0.170 0.252 Excel 0.020 0.002 0.004 0.085 0.133 PowerPoint 0.037 0.021 0.026 0.171 0.282 Overall 0.034 0.014 0.019 0.147 0.229

GPT-4o

- GPT-4.1

Word 0.101 0.065 0.077 0.307 0.518 Excel 0.102 0.026 0.039 0.278 0.514 PowerPoint 0.091 0.073 0.080 0.330 0.480 Overall 0.098 0.057 0.067 0.306 0.505

o3

Word 0.173 0.128 0.144 0.481 0.631 Excel 0.178 0.099 0.118 0.335 0.523 PowerPoint 0.129 0.109 0.115 0.526 0.559 Overall 0.160 0.114 0.128 0.456 0.578

- GPT-5

Word 0.106 0.079 0.088 0.315 0.615 Excel 0.172 0.109 0.126 0.274 0.574 PowerPoint 0.065 0.056 0.059 0.314 0.508 Overall 0.111 0.080 0.089 0.304 0.569

Word 0.384 0.014 0.023 0.137 0.358 Excel 0.041 0.002 0.003 0.082 0.094 PowerPoint 0.047 0.011 0.014 0.111 0.128 Overall 0.181 0.010 0.015 0.113 0.211

Qwen2.5-VL-7B

Word 0.392 0.520 0.440 0.619 0.730 Excel 0.431 0.217 0.270 0.450 0.748 PowerPoint 0.417 0.588 0.479 0.594 0.718 Overall 0.411 0.459 0.406 0.565 0.731

OmniParser

Word 0.396 0.525 0.444 0.625 0.738 Excel 0.431 0.217 0.270 0.450 0.748 PowerPoint 0.418 0.590 0.481 0.596 0.721 Overall 0.413 0.462 0.408 0.568 0.735

OmniParser v2

automation (CUA): translating abstract intent into precise, structured interactions with the GUI. As discussed in Section 3.3, we consider two evaluation settings: visual-only (where the agent has access solely to screenshots) and visual+a11y (where accessibility metadata is also provided). The latter setting is designed to test how much accessibility information improves grounding and execution.

Baselines. We evaluate a set of general-purpose VLMs, including GPT-4o Hurst et al. (2024), GPT-4.1 OpenAI (2025b), GPT-o3 OpenAI (2025d), GPT-5 OpenAI (2025c), and Qwen-VL-2.5 (7B) Bai et al. (2025). In addition, we fine-tune Qwen-VL-2.5 (7B) via supervised fine-tuning (SFT) and reinforcement learning (RL) on GUI-360◦ to examine post-training improvements.

Performance Metrics. The evaluation of action prediction is more nuanced than grounding, since each action step is composed of a function, a set of arguments, and a status flag (continue or finish). We therefore report three component accuracies and one aggregated metric:

• Function accuracy (Accfunc): the proportion of predictions where the predicted function fˆi exactly matches the ground-truth function fi.

1 N

Accfunc =

N

⊮{fˆi = fi}.

i=1

#### Table 9: Model performance comparison with screen Visual-only (left) and with screen Visual+A11y (right). Values are reported as percentages.

Visual-only Visual+A11y

Model Word Excel PowerPoint Total Word Excel PowerPoint Total

- GPT-4o 3.61 1.96 3.35 3.12 61.36 29.15 48.11 36.71 GPT-4.1 3.60 1.88 2.55 2.82 35.46 33.13 50.98 39.19 GPT-o3 16.85 13.06 24.42 17.92 34.00 28.76 48.84 36.72
- GPT-5 9.05 6.21 10.26 8.59 31.68 26.39 48.23 34.86 Qwen-2.5 7B 15.70 12.75 25.09 17.52 15.64 3.56 22.51 14.18 Qwen-2.5 7B-SFT 49.10 45.12 56.53 50.08 31.68 7.44 34.99 25.78

#### Table 10: Comparison of Qwen and SFT models with/without A11Y across domains. Each cell shows “Qwen / SFT”, with bold indicating the better value.

w/o A11Y w/ A11Y Metric Excel Word PPT Overall Excel Word PPT Overall

Function Match 61.67 / 81.07 62.97 / 81.18 88.50 / 91.02 69.82 / 83.93 50.61 / 80.64 66.62 / 83.19 90.82 / 91.53 68.95 / 84.83 Args Match 12.84 / 45.43 15.80 / 49.38 25.16 / 56.72 17.61 / 50.34 3.60 / 7.47 15.74 / 31.82 22.60 / 35.16 14.25 / 25.91 Status Match 95.96 / 94.13 96.37 / 93.85 97.44 / 96.17 96.56 / 94.59 85.74 / 95.27 98.23 / 95.77 99.00 / 96.34 94.93 / 95.79 Args Mismatch Err. 87.17 / 54.56 84.21 / 50.62 74.84 / 43.28 82.39 / 49.66 96.40 / 92.53 84.26 / 68.18 77.40 / 64.84 85.74 / 74.10 Coord. OOB 74.94 / 63.97 66.63 / 61.39 96.15 / 82.83 76.68 / 67.47 74.91 / 89.77 83.58 / 76.72 98.78 / 90.54 84.71 / 84.73

- • Argument accuracy (Accargs): evaluated conditionally on the predicted function. If the function is a spatial action such as click, correctness requires that the predicted coordinate (ˆxi,yˆi) falls inside the ground-truth bounding box bi. For symbolic arguments (e.g., menu item name, keystroke, value,), correctness requires exact match between predicted and ground-truth arguments. Formally,

Accargs =

1 N

N

i=1

⊮{aˆi ≡ ai | fi},

where the equivalence relation ≡ depends on the function type fi.

- • Status accuracy (Accstatus): whether the predicted status flag sˆi matches the ground-truth status si.
- • Step success rate (Accstep): a step is considered correct only if all three components (function, arguments, status) are correct simultaneously.

1 N

Accstep =

N

⊮{fˆi = fi ∧ aˆi ≡ ai ∧ sˆi = si}.

i=1

- Performance Comparison. Table 9 reports the results of action prediction under the visual-only and visual+a11y settings. When relying on screenshots alone, all models perform poorly, with accuracy below 20% in most cases. This highlights the intrinsic difficulty of inferring precise action arguments purely from pixel-level cues, even for state-of-the-art proprietary VLMs such as GPT4.1 and GPT-5. By contrast, providing accessibility information dramatically boosts performance. For example, GPT-4o improves from 3.12% to 36.71%, and GPT-4.1 nearly triples its performance from 2.82% to 39.19%. This demonstrates that structured element annotations effectively reduce the burden of visual grounding, enabling models to focus on action semantics.

Furthermore, supervised fine-tuning (SFT) on GUI-360◦ delivers substantial gains. Qwen-2.5 7B improves from 17.52% to 50.08% after SFT in the visual-only setting, a nearly threefold improvement, showing the strong training signal provided by GUI-360◦. However, when a11y information is introduced, the benefits of SFT diminish, suggesting that a11y annotations already encode much of the structural alignment that SFT otherwise learns. Overall, these results highlight both the challenge and opportunity presented by GUI-360◦: action prediction is extremely difficult without explicit structural information, but with accessibility-enhanced input and dataset-driven post-training, models can achieve substantial improvements.

Result Analysis. To obtain a fine-grained understanding of action prediction, we decompose evaluation into three aspects: (i) function match, which tests whether the predicted function type is correct; (ii) argument match, which checks whether the predicted arguments (e.g., coordinates or values) align with ground truth; and (iii) status match, which verifies whether the model correctly predicts task continuation or completion. In addition, we track two error sources: Args Mismatch Error, capturing the proportion of incorrect arguments, and Coord. OOB, which reflects out-ofbounds coordinate predictions in the visual-only setting or incorrect element selections when a11y information is available.

From Table 10, we observe that supervised fine-tuning (SFT) brings large gains in function match and especially in argument match, reducing errors by more than half across domains. The dominant source of failure remains Args Mismatch Error, which suggests that grounding actions to the correct interface element is the most challenging aspect. Notably, Coord. OOB contributes significantly to these mismatches, highlighting the model’s difficulty in spatial grounding from raw screenshots.

Comparing the two evaluation settings, we find that introducing a11y metadata substantially reduces Coord. OOB errors, showing that structured semantic cues provide more reliable grounding than relying on visual information alone. However, even with a11y support, argument prediction accuracy lags far behind function prediction, indicating that grounding arguments—either through spatial reasoning or element identification—remains the key bottleneck for reliable GUI action prediction. Overall, these findings highlight that while models can correctly identify the intended operation, achieving precise grounding is still an open challenge, and incorporating structured UI metadata such as a11y is a promising direction.

- 5 CONCLUSION

In this work, we introduced GUI-360◦, a large-scale dataset and benchmark suite for advancing research on desktop computer-using agents. GUI-360◦ fills three critical gaps in the field: the lack of realistic task collections, the absence of scalable data collection pipelines, and the shortage of unified benchmarks spanning GUI grounding, screen parsing, and action prediction. Through an automated, LLM-augmented pipeline, we curated over 1.2M steps across thousands of trajectories in widely used Windows applications, paired with rich multimodal annotations including screenshots, accessibility metadata, reasoning traces, and both successful and failed executions.

Our empirical evaluation highlights both the difficulty of the domain and the promise of GUI-360◦. State-of-the-art vision–language models exhibit significant limitations when applied out-of-the-box, particularly in grounding and action prediction. Yet, fine-tuning and reinforcement learning on GUI360◦ deliver consistent improvements, underscoring the dataset’s utility as a training and evaluation resource. Importantly, results remain far from human-level reliability, establishing GUI-360◦ as a challenging but necessary foundation for future progress.

We release GUI-360◦, the accompanying benchmark GUI-360◦-Bench, and the full data collection framework to the research community. We hope these resources catalyze systematic advances in screen understanding, multimodal reasoning, and robust action planning, ultimately bringing practical and reliable computer-using agents closer to reality.

REFERENCES

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.

Rob Haverty. New accessibility model for microsoft windows and cross platform development. ACM SIGACCESS Accessibility and Computing, (82):11–17, 2005.

Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context protocol (mcp): Landscape, security threats, and future research directions. arXiv preprint arXiv:2503.23278, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.

Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203, 2024.

Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M Tamer Ozsu,¨ Aishwarya Agrawal, David Vazquez, et al. Uivision: A desktop-centric gui benchmark for visual perception and interaction. arXiv preprint arXiv:2503.15661, 2025.

OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025a. URL https://openai.com/index/computer-using-agent.

OpenAI. Introducing gpt-4.1 in the api. April 2025b. URL https://openai.com/index/

gpt-4-1/. Accessed: 2025-09-18. OpenAI. Gpt-5 is here. https://openai.com/gpt-5/, 2025c. Accessed: 2025-09-18. OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/

introducing-o3-and-o4-mini/, April 2025d. Accessed: 2025-09-18.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708–59728, 2023.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723, 2024.

Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. Advances in Neural Information Processing Systems, 37:2686–2710, 2024a.

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024b.

Lu Wang, Fangkai Yang, Chaoyun Zhang, Junting Lu, Jiaxu Qian, Shilin He, Pu Zhao, Bo Qiao, Ray Huang, Si Qin, et al. Large action models: From inception to implementation. arXiv preprint arXiv:2412.10047, 2024c.

Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-agent-e: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733, 2025.

Zilong Wang, Yuedong Cui, Li Zhong, Zimin Zhang, Da Yin, Bill Yuchen Lin, and Jingbo Shang. Officebench: Benchmarking language agents across multiple applications for office automation. arXiv preprint arXiv:2407.19056, 2024d.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, et al. Openagents: An open platform for language agents in the wild. arXiv preprint arXiv:2310.10634, 2023.

Yibin Xu, Liang Yang, Hao Chen, Hua Wang, Zhi Chen, and Yaohua Tang. Deskvision: Large scale desktop region captioning for advanced gui agents. arXiv preprint arXiv:2503.11170, 2025.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Api agents vs. gui agents: Divergence and convergence. In ICML 2025 Workshop on Computer Use Agents.

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024a.

Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, et al. Ufo: A ui-focused agent for windows os interaction. arXiv preprint arXiv:2402.07939, 2024b.

Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, et al. Ufo2: The desktop agentos. arXiv preprint arXiv:2504.14603, 2025.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, et al. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079, 2025a.

Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Vem: Environment-free exploration for training gui agent with value environment model. arXiv preprint arXiv:2502.18906, 2025b.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

- A ACTION SET

To support diverse interaction across desktop applications, we design a unified action set that combines general-purpose GUI operations with application-specific APIs using MCP servers. The action set is deliberately lightweight yet expressive, enabling agents to cover the full spectrum of common productivity tasks while remaining tractable for model training.

The GUI actions (click, type, drag, wheel mouse input) form the foundation of interaction, as they are applicable to any graphical user interface. These actions abstract low-level mouse and keyboard events into structured calls, supporting variants such as absolute and normalized coordinates, modifier keys, and multi-step operations.

On top of this universal layer, we extend the action set with fine-grained APIs for Word, Excel, and PowerPoint. These APIs expose high-level document, spreadsheet, and presentation semantics, such as inserting tables, modifying cell values, reordering columns, adjusting font properties, or setting slide backgrounds. By combining GUI-agnostic operations with domain-specific APIs, the action set achieves both generality and efficiency: agents can rely on GUI actions for arbitrary interfaces while exploiting APIs for structured tasks where precise semantics matter.

Table 11 summarizes the full action set. This unified design allows agents trained on GUI-360◦ to operate seamlessly across heterogeneous applications, balancing robustness with expressivity.

- B GUI-360◦ SCHEMA

Each execution step in GUI-360◦ is stored as a structured JSON object following a unified schema. This schema ensures consistency across tasks and provides rich multimodal supervision for grounding, parsing, and action prediction. Table 12 summarizes the key fields.

Discussion. The schema integrates three complementary perspectives: (i) Visual context through multi-view screenshots, (ii) Structural context via accessibility metadata and hierarchical UI trees, and (iii) Cognitive traces through observations, reasoning, actions, and evaluations. This rich structure allows GUI-360◦ to jointly support grounding, parsing, and action prediction, while enabling both supervised training and fine-grained evaluation. By standardizing every execution step, the schema provides a scalable foundation for reproducibility and extensibility across applications.

- C BASELINE DETAILS

We summarize the baselines evaluated on GUI-360◦ across the three core tasks: GUI grounding, screen parsing, and action prediction. These baselines include both general-purpose vision–language models and domain-specific approaches designed for GUI reasoning. Below we briefly introduce each group of models.

- C.1 GUI GROUNDING

- • GPT-4o Hurst et al. (2024): proprietary multimodal VLM used off-the-shelf for grounding.
- • GPT-4.1 OpenAI (2025b): proprietary VLM emphasizing instruction-following and tool use.

- Table 11: Supported actions across Word, Excel, and PowerPoint, grouped by shared GUI actions and application-specific APIs.

Action Description Type click Click at a given position (absolute or normalized), support-

GUI

ing left/right/middle/x button, single/double click, and optional modifier key.

type Type text or hotkeys at a position, with options for clearing text or focusing on control. Supports special keys like {VK CONTROL}c.

GUI

drag Drag from a start to an end position with configurable mouse button, duration, and optional key hold (e.g., shift, control).

GUI wheel mouse input Scroll at a given position with positive (up) or negative (down)

GUI

wheel distance.

insert table Insert a table with a specified number of rows and columns. Word API select text Select exact text in the document. Word API select table Select a table by its index number. Word API select paragraph Select paragraphs by start and end indices, with option to re-

Word API save as Save the document with specified directory, file name, and ex-

strict to non-empty ones.

Word API

tension (default: PDF).

set font Change font family and/or size. Word API table2markdown Extract the contents of a worksheet table into Markdown for-

Excel API

mat.

insert excel table Insert a table (list of lists) into a sheet at a specified starting cell. Excel API select table range Select a range of cells by coordinates in a sheet. Excel API set cell value Set the value (or formula) of a specific cell. Excel API auto fill Autofill values in a specified cell range. Excel API reorder columns Reorder columns in a sheet according to a given list of column

Excel API

names.

set background color Change the slide background color using a hex RGB value, for

PowerPoint API

selected or all slides.

save as Save the presentation with specified directory, file name, and

PowerPoint API

extension (default: PowerPointX). Optionally save slides as images.

- • o3 OpenAI (2025d): OpenAI “reasoning” model family; evaluated zero-shot for grounding.
- • GPT-5 OpenAI (2025c): latest OpenAI flagship; strong general reasoning baseline.
- • Qwen2.5-VL-7B Bai et al. (2025): open-source 7B multimodal baseline.
- • UGround-7B Gou et al. (2024): GUI visual grounding model (Qwen2-VL backbone) trained with large-scale GUI data.
- • Aguvis-7B Xu et al. (2024): vision-centric GUI agent with a unified cross-platform action space.
- • UI-TARS-1.5 (7B) Qin et al. (2025): multimodal agent optimized for GUI reasoning and interactive tasks.
- • GUI-Actor (7B) Wu et al. (2025): coordinate-free grounding model with an attention-based action head.
- • SFT variants: Qwen2.5-VL-7B fine-tuned on GUI-360◦ for task-adapted grounding.

- C.2 SCREEN PARSING

- • GPT-4o / GPT-4.1 / o3 / GPT-5 Hurst et al. (2024); OpenAI (2025b;d;c): general-purpose VLMs used off-the-shelf for detection/localization.
- • Qwen2.5-VL-7B Bai et al. (2025): open-source multimodal baseline.
- • OmniParser / OmniParser-v2 Lu et al. (2024): screen-parsing tools that produce element sets (names + bounding boxes) from raw screenshots.

- Table 12: Execution Step Schema for GUI-360◦. Each entry records metadata, screenshots, accessibility data, reasoning traces, and actions.

Field Description execution id Unique identifier for the execution instance (e.g., word 1 1). app domain Application domain (e.g., Word, Excel, PowerPoint). request Natural-language task description provided to the agent. template Environment template file used for instantiation. step id / total steps Current step index and total number of steps in the trajectory. evaluation Automatic assessment of the step, including reasoning, evi-

dence, sub-scores, and a final completeness label. step.screenshots Multiple synchronized screenshots: clean view, full desktop, an-

notated version, and selected-controls view. step.ui tree Hierarchical UI structure with element IDs, names, control

types, bounding boxes, and children. step.control infos Metadata from accessibility APIs and merged control sources,

providing bounding boxes, labels, and semantic text. step.observation Agent’s textual observation of the current state. step.thought Agent’s intermediate reasoning for the next action. step.action Executed action, including function type (e.g., click), argu-

ments, target coordinates, and status flag. status Overall status of the step (CONTINUE or FINISH).

- C.3 ACTION PREDICTION

- • GPT-4o / GPT-4.1 / o3 / GPT-5 Hurst et al. (2024); OpenAI (2025b;d;c): proprietary VLMs evaluated in both visual-only and visual+a11y settings.
- • Qwen2.5-VL-7B Bai et al. (2025): open-source baseline for structured action generation.
- • Qwen2.5-VL-7B-SFT: supervised fine-tuning on GUI-360◦ for step-wise function/argument/status prediction.

Summary. Together, these baselines span a spectrum from general-purpose VLMs to specialized GUI-focused agents. This diversity allows us to systematically evaluate the unique challenges posed by GUI-360◦ across grounding, parsing, and action prediction, and to measure how far current models remain from robust, human-level computer-using agents.

- D IMPLEMENTATION DETAILS

Data Collection. To construct GUI-360◦, we deployed a cluster of 15 Windows 11 virtual machines, each provisioned with 4 CPU cores. These VMs executed tasks in parallel, enabling efficient large-scale trajectory collection. Task execution followed a two-phase strategy: in Phase 1, GPT-4o was used as the agent; in Phase 2, all failed tasks were re-executed with GPT-4.1 for recovery (see Section 3). Both models were queried with the temperature fixed at 0.0 to ensure deterministic outputs and reproducibility.

Model Access. For evaluation on GUI-360◦-Bench, we used multiple OpenAI models, including GPT-4o, GPT-4.1, o3, and GPT-5, all accessed via the Azure OpenAI Service. Unless otherwise

stated, the decoding temperature was set to 0.0 across all experiments to minimize variance and ensure consistent evaluation.

Fine-tuning and Training. All supervised fine-tuning (SFT) and reinforcement learning (RL) experiments were conducted on a compute cluster equipped with NVIDIA A100 GPUs (40GB memory per GPU). Specifically, each run was distributed across 4 A100 GPUs using mixed-precision training (FP16) for efficiency. We adopted standard optimization settings following prior work on multimodal fine-tuning, with learning rates tuned over {1e-5,5e-6,1e-6}. Checkpointing and gradient accumulation were applied to ensure stable training for long trajectories.

ETHICS STATEMENT

This work presents GUI-360◦, a large-scale dataset and benchmark for GUI agents on Windows applications. We carefully considered ethical implications throughout the data collection, processing, and release pipeline. First, no human subjects were directly involved in the data collection process, and thus no personally identifiable information (PII) or sensitive user data is included. All queries and trajectories were generated and executed within controlled sandbox environments to ensure both privacy and security. Second, we adhered to software license terms and platform usage guidelines, ensuring that the collection process does not violate proprietary restrictions or legal compliance requirements. Third, we performed multiple post-processing stages, including trajectory validation, data sanitization, and structuring, to filter out incomplete, low-quality, or potentially misleading samples, thereby reducing the risk of harmful insights or erroneous model behaviors.

We acknowledge the potential downstream misuse of GUI automation technologies, such as unauthorized system manipulation or exploitation of accessibility features. To mitigate this, we restrict dataset release to non-sensitive application contexts (Word, Excel, PowerPoint) and exclude scenarios that could pose privacy, security, or safety risks. The dataset is intended solely for academic research on improving robustness, generalization, and evaluation of GUI agents. All models and baselines are evaluated under responsible-use guidelines, and we encourage future researchers to follow the same principles.

REPRODUCIBILITY STATEMENT

We place strong emphasis on reproducibility. To this end, we will release all data collection code, execution framework, and templates used to instantiate user queries. The full dataset GUI-360◦, along with the benchmark split GUI-360◦-Bench, will also be publicly available. Detailed descriptions of the pipeline are provided in Section 3, with task definitions summarized in Table 5, filtering statistics in Table 3, and execution details in Appendix D. Additional implementation details, hyperparameters, and evaluation protocols are included in the supplementary material. Together, these resources ensure that both our dataset creation and experimental results can be fully reproduced, verified, and extended by the research community.

LLM USAGE STATEMENT

In preparing this work, we used large language models (LLMs) strictly as an assistive tool for text polishing and minor language refinement. All research ideas, technical designs, analyses, and conclusions were conceived and carried out entirely by the authors.

