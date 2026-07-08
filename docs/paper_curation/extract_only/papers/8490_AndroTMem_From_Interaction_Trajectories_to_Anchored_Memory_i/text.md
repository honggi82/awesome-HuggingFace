## arXiv:2603.18429v1[cs.CV]19Mar2026

2026-03-20

# AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents

Yibo Shi1,∗, Jungang Li2,3,∗,†, Linghao Zhang4,∗, Zihao Dongfang2,3,∗, Biao Wu5, Sicheng Tao2, Yibo Yan2,3, Chenxi Qin6, Weiting Liu7, Zhixin Lin8, Hanqian Li2, Yu Huang2, Song Dai2,3, Yonghua Hei2,3, Yue Ding9, Xiang Li2,3, Shikang Wang4, Chengdong Xu10, Jingqi Liu1, Xueying Ma1, Zhiwen Zheng1, Xiaofei Zhang1, Bincheng Wang11, Nichen Yang1, Jie Wu10, Lihua Tian1,‡, Chen Li1,‡, Xuming Hu2,3,‡

1 XJTU 2 HKUST(GZ) 3 HKUST 4 CityU 5 UTS 6 TJU 7 FDU 8 SDU 9 CASIA 10 SYSU 11 NWPU

∗ Core Contributors. † Project Lead. ‡ Corresponding Authors. Contact: yiboshi86@gmail.com, ljungang.02@gmail.com, lhtian@xjtu.edu.cn, xuminghu@hkust-gz.edu.cn

Abstract | Long-horizon GUI agents are a key step toward real-world deployment, yet effective interaction memory under prevailing paradigms remains under-explored. Replaying full interaction sequences is redundant and amplifies noise, while summaries often erase dependency-critical information and traceability. We present AndroTMem, a diagnostic framework for anchored memory in long-horizon Android GUI agents. Its core benchmark, AndroTMem-Bench, comprises 1,069 tasks with 34,473 interaction steps (avg. 32.1 per task, max. 65). We evaluate agents with TCR (Task Complete Rate), focusing on tasks whose completion requires carrying forward critical intermediate state; AndroTMem-Bench is designed to enforce strong step-to-step causal dependencies, making sparse yet essential intermediate states decisive for downstream actions and centering interaction memory in evaluation. Across open- and closed-source GUI agents, we observe a consistent pattern: as interaction sequences grow longer, performance drops are driven mainly by within-task memory failures, not isolated perception errors or local action mistakes. Guided by this diagnosis, we propose Anchored State Memory (ASM), which represents interaction sequences as a compact set of causally linked intermediate-state anchors to enable subgoal-targeted retrieval and attribution-aware decision making. Across multiple settings and 12 evaluated GUI agents, ASM consistently outperforms full-sequence replay and summary-based baselines, improving TCR by 5%–30.16% and AMS by 4.93%–24.66%, indicating that anchored, structured memory effectively mitigates the interaction-memory bottleneck in long-horizon GUI tasks. The code, benchmark, and related resources are publicly available at https://github.com/CVC2233/AndroTMem.

#### 1. Introduction

Multimodal large language models (MLLMs) have rapidly progressed from static image understanding Dang et al. [2025], Ding et al. [2026], Wang et al. [2024b], Zheng et al. [2026] toward richer forms of perception and interaction, including stronger visual reasoning Dong et al. [2025], video understanding Hu et al. [2025], Liu et al. [2025c], Pan et al. [2026], Tang et al. [2025], Tao et al. [2025], and increasingly responsive real-time multimodal interfaces Bai et al. [2025], Comanici et al. [2025], Hurst et al. [2024], Xun et al. [2025]. This evolution is reshaping human–computer collaboration: systems are moving beyond tools that execute isolated commands toward agents that can perceive, reason, and act continuously in real-world environments Li and Huang [2025], Zhang et al. [2024a]. Along this trajectory, GUI agents have emerged as one of the most practical and impactful instantiations, aiming to complete everyday and professional tasks from natural-language instructions across heterogeneous device settings such as mobile, PC, and web Huang et al. [2025], Liu et al. [2025b], Rawles et al. [2023, 2024], Xie et al. [2024], Zheng et al. [2024].

ii) Task Case

i) Dataset Construction

###### (B) Intent-driven Type (C) Template&Slot Filling

(A) APP Taxonomy

Task: Compare prices of Airpords Pro2 across apps in JD and Taobao.Add the cheaper item to the cart and share it with Alice on WeChat, sending "Found a better deal!".

App groups:

Template Slot pool Instantiatedtask

[Figure 1]

[Figure 2]

Lookup Config Communicate

...

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Shopping

Phase B Decision & Cart (step 12 - step 16)

Phase A Price extraction (step 1- step 11)

l Compare {item} across

Phase C Share & Confirm (step 17 - step 21)

... ...

Decide Purchase Create Content

{#group:shopping} l Add the cheaper one to cart l Share link to {contact} via

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Music

Booking Share

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Social

...

...

{#group:social}

...

Same app × Different Intent ⇒Different Tasks

[Figure 15]

[Figure 16]

[Figure 17]

Entertainment

S12 Comparison done S12 Cheaper_Item S14 Add to cart

[Figure 18]

- S1 Message to send

- S2 Splash Ad (skip) S7 Price_JD

[Figure 19]

- S20 Contact found

- S21 Link sent to Alice

Slots: {item} | {#contact} | {#group}

50 APPs

[Figure 20]

Task Characteristics

[Figure 21]

[Figure 22]

S11 Price_Taobao

[Figure 23]

Long-horizon Steps: Avg 32.1, Max 65 Cross-app Workflows: 2–4 apps/task

###### Anchor States: 6 types Causal Dependency Chains

[Figure 24]

Subgoal Dependency State Change

Context Info Finish Exception

[Figure 25]

[Figure 26]

[Figure 27]

Causal State Anchor

[Figure 28]

[Figure 29]

[Figure 30]

###### iii) Diagnostic Evaluation Suite

###### iV) Anchored State Memory (ASM)

Agents: Open 5 | Closed 5 | Multi 2

[Figure 31]

###### Causal Link

[Figure 32]

[Figure 33]

[S7] Price_JD

Anchor Memory Bank

###### Evaluation Scope

Raw Summary

0.6

Value: $xxx.xx Evidence:screenshot/bbox/

0.5

[Figure 34]

Step-Level

[Figure 35]

[Figure 36]

###### [S11] Price_Taobal

###### [S12] Cheaper_Item

0.4

Value: $xxx.xx Evidence: ...

Value: (app, min_price, ...) From: price_JD, price_Taobao

TCR

0.3

Dependency for Retrieval

Task-Level

0.2

UI state User Unstruction Next Action

[Figure 37]

[Figure 38]

[Figure 39]

###### MLLM

0.1

Efficiency

0.0

+0.16 +0.28

0.6

10-19 20-29 30-39 40-49 50-59 60-69

Raw Summary ASM (Ours)

Step Range

0.4

+0.03 +0.06

TCR

[Figure 40]

As horizons grow, performance degrades sharplyrevealing memory as the key bottleneck.

+0.08

0.2

0.0

Qwen-2.5-VL Gemini-2.5-Flash InfoGUI-r1-3b Gemini-3-Flash Gemini-2.5-Pro

- Figure 1 | Overview of AndroTMem. AndroTMem comprises: (i) AndroTMem-Bench, a long-horizon Android GUI benchmark constructed with intent-driven, cross-app, causally dependent workflows;

- (ii) representative task cases where sparse intermediate states determine downstream decisions;
- (iii) a diagnostic evaluation suite showing that performance drops with horizon length are primarily caused by memory failures; and (iv) Anchored State Memory (ASM), which stores causally linked intermediate-state anchors for targeted retrieval and improves long-horizon GUI-agent performance.

Despite steady progress in UI grounding and single-step action reasoning Cheng et al. [2024], Gou

- et al. [2024], Li et al. [2025], Lin et al. [2025a], Qin et al. [2025], Wu et al. [2024], GUI agents remain brittle when interaction trajectories extend to dozens of steps. Real-world mobile tasks are rarely just long chains of loosely connected actions; instead, they consist of indispensable intermediate steps whose outcomes must be preserved and reused later. Users often need to extract values across pages, verify prerequisite conditions, handle exception branches, and bring these earlier results back when they become relevant several steps later. As a result, the central challenge in long-horizon execution is no longer only perceiving the current interface or selecting the next action, but maintaining, retrieving, and operationalizing critical intermediate state over time—that is, anchored memory.

Recent studies have begun to explore how memory mechanisms can improve planning and reasoning in agentic systems Packer et al. [2023], Park et al. [2023], Shinn et al. [2023]. However, mainstream GUI-agent benchmarks still focus primarily on short-horizon routines or weakly coupled multi-step tasks, and thus lack dedicated evaluation of agents’ memory ability in long-horizon tasks Chai et al. [2025], Chen et al. [2024b], Rawles et al. [2023], Wang et al. [2024a], Xu et al. [2025a]. In such settings, later decisions may succeed even without faithfully reusing earlier information, which obscures a more fundamental issue in real workflows: many intermediate results that determine downstream success—such as extracted values, prerequisite completions, or exceptionhandling outcomes—may exert their causal effect only several steps later. Existing evaluations therefore expose a critical blind spot: they can measure whether an agent completes a workflow, but not whether it truly preserves and correctly reuses task-critical state across time Liu et al. [2025d], Wang et al. [2025b].

In practice, this blind spot manifests as three tightly coupled challenges. First, existing datasets are not designed to evaluate long-horizon memory explicitly: their length often comes from chaining loosely related steps rather than enforcing strong step-to-step causal dependencies. Second, longhorizon failure is poorly diagnosed: end-to-end success conflates perception errors, local action mistakes, and memory breakdowns, offering little visibility into where degradation begins and what drives it as trajectories lengthen. Third, history modeling remains at an impasse: replaying full interaction sequences is redundant and amplifies noise, while coarse summaries often erase exactly the fine-grained states and dependencies required downstream. Together, these gaps obscure genuine progress in long-horizon GUI agents and make memory bottlenecks difficult to measure and improve reliably Wang et al. [2024b], Zhang et al. [2024a].

To address this, we present AndroTMem, a diagnostic framework for interaction memory in longhorizon Android GUI agents. At its core is AndroTMem-Bench, comprising 1,069 realistic tasks and 34,473 interaction steps across 50 applications (avg. 32.1 steps per task, max. 65). The benchmark is explicitly constructed to enforce strong step-to-step causal dependencies, making sparse but essential intermediate states decisive for downstream decisions. We evaluate agents with TCR (Task Complete Rate), focusing on key tasks whose completion hinges on carrying forward critical intermediate state, thereby placing long-horizon memory ability at the center of evaluation. Systematic studies across diverse open- and closed-source GUI agents reveal a consistent pattern: as interaction sequences lengthen, performance drops are driven primarily by within-task memory failures, rather than isolated perception errors or local action mistakes.

Our main contributions are summarized as follows: ❶ We introduce AndroTMem, a diagnostic framework for memory in long-horizon GUI agents, together with AndroTMem-Bench, a benchmark that evaluates memory via TCR on dependency-critical long-horizon tasks. ❷ Using this benchmark, we show that long-horizon degradation is dominated by within-task memory failures, not isolated perception errors or local action mistakes. ❸ We propose Anchored State Memory (ASM), which organizes history into causally linked intermediate-state anchors for targeted retrieval and attribution.

Across multiple settings and 12 GUI agents, ASM consistently outperforms full-sequence replay and summary-based baselines, improving TCR by 5%–30.16% and AMS by 4.93%–24.66%, underscoring anchored memory as a core capability for reliable and scalable long-horizon GUI agents.

#### 2. Related Work

Long-Horizon Task Execution and Memory in GUI Agents. GUI agents aim to complete user instructions by iteratively perceiving the current interface and issuing low-level actions such as taps, swipes, and text input. Recent MLLMs have substantially improved UI understanding and instruction following, enabling stronger page perception, UI grounding, and step-level action prediction Chen

- et al. [2025], Wang et al. [2024b], Zhang et al. [2024a]. Prior work has therefore progressed from improving perception and element grounding Cheng et al. [2024], Gou et al. [2024], Hong et al.

- [2024], Lin et al. [2025b] to studying multi-step GUI interaction through end-to-end agents Lin et al.
- [2025a], Wu et al. [2024], modular systems Liu et al. [2025a], Wang et al. [2025a], and diverse training strategies including supervised learning, reinforcement learning, and self-improvement Li and Huang [2025], Lu et al. [2025b].

Despite these advances, reliably executing long-horizon GUI tasks remains challenging. In realistic workflows, interaction trajectories often span dozens of steps across multiple applications, where later decisions depend on intermediate results obtained several steps earlier, such as extracted values, completed subgoals, or environment changes. This makes effective history utilization and intermediatestate management essential for successful task execution Gu et al. [2025b], Liu et al. [2025b], Xi

et al. [2025]. However, existing GUI-agent frameworks incorporate history through raw interaction traces Lin et al. [2025a], Lu et al. [2025a], Wu et al. [2024], compressed summaries Liu et al. [2025a], Lu et al. [2025b], Xu et al. [2025b], or other generic context aggregation strategies. While these mechanisms help incorporate past information, they are not explicitly designed to preserve and retrieve sparse but causally critical intermediate states in long-horizon cross-app workflows. This limitation motivates memory mechanisms that explicitly model and retrieve decision-critical intermediate states during long-horizon GUI interaction.

Benchmarks and Datasets for Mobile GUI Agents. Benchmarks for GUI agents have evolved from single-screen or few-step operations Chen et al. [2024a], Xie et al. [2024] to longer multi-step tasks emphasizing end-to-end completion Liu et al. [2025d], Wang et al. [2025b], with evaluation environments becoming increasingly realistic and covering more diverse web and mobile applications Kong et al. [2025], Xu et al. [2025b]. In parallel, agent observations have shifted from structured UI metadata toward multimodal and pure-vision settings, improving generality and reducing dependence on platform-specific annotations Li et al. [2025], Lu et al. [2025a]. However, existing benchmarks still provide limited support for studying long-horizon interaction memory. Under such benchmark settings, some tasks remain relatively short or weakly coupled, such that later decisions may rely primarily on local perception and action prediction rather than on faithfully preserving earlier intermediate outcomes. Consequently, they offer limited visibility into whether agents can maintain and correctly reuse task-critical state across time. This motivates benchmarks that explicitly enforce strong step-to-step causal dependencies and make intermediate state management a first-class target of evaluation.

In contrast to prior work, AndroTMem focuses on explicitly diagnosing interaction memory in long-horizon GUI agents. Our benchmark enforces strong cross-step causal dependencies, enabling systematic evaluation of how agents preserve and reuse intermediate states. Furthermore, we propose ASM, a structured history representation that organizes interaction trajectories around causally linked state anchors.

#### 3. Dataset Construction and Statistics

###### 3.1. Long-Horizon Task Formulation

Crucially, unlike short-horizon or loosely coupled GUI tasks, tasks in AndroTMem are deliberately constructed and annotated to exhibit substantial cross-step causal dependencies. Later actions often depend on intermediate outcomes produced by earlier steps, which we refer to as intermediate task states. These intermediate states are neither directly observable from the initial instruction nor recoverable solely from the current GUI state. As a result, executing tasks in AndroTMem requires agents to correctly establish and maintain task-relevant intermediate states over extended interaction horizons, therefore making the effective utilization of interaction history and intermediate states a prerequisite for task success, rather than an optional enhancement. To make such intermediate task states explicit and amenable to analysis, we annotate sparse State Anchors along the trajectory, each summarizing a task-relevant state change or intermediate outcome that constrains subsequent steps.

Task Types. We observe that in multi-app environments, tasks cannot be adequately characterized using coarse scenario labels such as shopping or travel alone. Even when involving similar combinations of applications, tasks may exhibit substantially different interaction patterns, step dependencies, and behavior distributions, largely due to differences in user intent.

To capture this distinction, we define primary intent as the dominant objective that drives task execution. Following real-world app usage, where user interactions are typically organized around a

Table 1 | Comparison of AndroTMem-Bench with other mobile GUI agent benchmarks. #Traj: number of trajectories (tasks); #Steps: average interaction steps per trajectory; #App: number of applications. GT: availability of ground-truth trajectories; Low-Level Instr.: low-level instructions. History Modeling: supported within-task history representations.

History Modeling

Dataset #Traj #Steps #App GT Low-Level Instr.

Summary ASM

AMEX Chai et al. [2025] 2,946 12.8 110 ✓ ✗ ✗ ✗ AITW Rawles et al. [2023] 30,378 6.5 357 ✓ ✗ ✗ ✗ AITZ Zhang et al. [2024b] 2,504 7.5 70 ✓ ✓ ✓ ✗

Android Control Li et al. [2024] 15,283 5.5 833 ✓ ✓ ✓ ✗

Android World Rawles et al. [2024]

116 – 20 ✗ ✗ ✗ ✗ Mobile-R1 Gu et al. [2025a] 4,635 5.3 28 ✓ ✓ ✗ ✗

Mobile-Bench-v2 Xu et al. [2025a]

12,856 7.28 49 ✗ ✗ ✗ ✗

GUIOdyssey Lu et al. [2025a] 8,334 15.3 212 ✗ ✓ ✓ ✗ SPA-Bench Chen et al. [2024b] 201 8.2 21 ✗ ✗ ✗ ✗

LearnGUI Liu et al. [2025a] 2,353 13.2 73 ✓ ✓ ✓ ✗

LongHorizonUI Kang et al. [2026]

354 22.1 28 ✓ ✓ ✓ ✗ AndroTMem-Bench 1,069 32.1 50 ✓ ✓ ✓ ✓

single goal, we assign each task in AndroTMem exactly one primary intent. Accordingly, we group tasks into 8 intent classes: (1) Lookup; (2) Compare & Decide; (3) Purchase / Order; (4) Booking / Reservation; (5) Communicate / Coordinate; (6) Share / Recommend; (7) Create Content; and (8) Configure / Authorize. See Appendix A.2 for details.

Action Space. The action space of AndroTMem includes 11 action types: open_app, tap, long_press, swipe, input_text, swipe_two_points, wait, capture_screen, home, back, finish. swipe represents sliding based on the starting point and direction, while swipe_two_points represents sliding based on the starting point and ending point. During the evaluation, swipe and swipe_two_points operations were simplified to whether the sliding directions matched. See Appendix A.3 for details.

###### 3.2. Data Pipeline

An overview of the entire data pipeline is illustrated in Figure 2, which consists of task instruction construction (Step 1–2) and trajectory annotation (Step 3).

Motivation for a Semi-Automatic Pipeline. Long Horizon Task requires long-horizon execution with explicit cross-step causal dependencies and fine-grained intermediate states, which we record as sparse State Anchors that must be correct and consistent across the entire trajectory.

While fully automatic pipelines can scale data collection, they typically optimize for obtaining raw interaction traces, and thus provide limited control over (i) intent-faithful task design, (ii) the correctness of intermediate causal states, and (iii) the consistency of dependency links across steps and apps. In our preliminary trials, LLM-generated tasks and auto-collected trajectories frequently exhibit mismatches between the intended goal and the executed behavior, and recovering the required intermediate states afterward would still require processing the full trajectory—either by extensive human re-annotation or by automation that is not yet reliable under long causal chains.

We therefore adopt a semi-automatic pipeline that places human expertise only where it is indispensable: experienced GUI experts specify intent-driven tasks, while the platform automates

###### Step 1: Functional App Group Construction

###### Step 3: Trajectory Annotation

Social Music Shopping ... ... ... ...

[Figure 41]

[Figure 42]

[Figure 43]

Annotation Platform

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Group by core functionalities

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Android

Collect popular APPs across domains

[Figure 56]

Annotators

[Figure 57]

[Figure 58]

Entertainment

Device / Emulator

...

[Figure 59]

Annotated trajectories

###### Step 2: Task Instruction Construction

[Figure 60]

[Add to Cart]

(m1: cheaper_item) (m2: item_link)

Trajectory Quality Check

[Figure 61]

###### Causal Dependency Design

[Figure 62]

[Compare] [Share]

AndroTrace Dataset

[Figure 63]

• App-group slot: instantiated with an app from a functional group

Template: Compare the prices of {item} across {#group: shopping}/{#group: shopping}->Add the cheaper one to cart>share the link to {#contact} in {#group: social} # instantiation from a predefined set of options.

- • Choice slot: instantiated with a fixed option
- • Content slot: instantiated with free-form text

[Figure 64]

[Figure 65]

[Figure 66]

- • Screenshot-action pairs
- • UI hierarchy (Accessibility Tree)
- • Causal Dependencies
- • Reasoning
- • Summary

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Template Instantiation

Instruction: Compare the prices of AirPods Pro across JD/Taobao->Add the cheaper one to cart->share the link to Alice in WeChat

[Figure 71]

[Figure 72]

[Figure 73]

Instruction Quality Check

Instruction Paraphrasing

- Figure 2 | Overview of the AndroTMem-Bench dataset construction pipeline. (1) Collect popular mobile apps and group them by function. (2) Generate long-horizon cross-app task instructions with step-to-step causal dependencies using dependency-aware templates. (3) Execute and annotate tasks on Android devices or emulators, producing quality-checked trajectories and dataset outputs.

closed-loop action execution with real-time visual feedback, synchronized collection of UI states, and the management of configurable structured annotation fields. Compared to purely automatic data collection, this design yields annotations that are (a) causally grounded, (b) fine-grained and directly usable for evaluation, and (c) significantly cheaper to produce than full post-hoc re-annotation of long trajectories.

Task Instruction Construction. We first collect about 50 commonly used mobile applications and organize them into distinct application groups(Functional App Group. See Appendix A.1 for details.) based on their core functionalities, where apps within the same group share similar interaction patterns and functional semantics. Based on these groups, we design over 70 cross-app task templates that are deliberately constructed to induce strong step-to-step dependencies across applications. Each task template contains slots for application function groups and task parameters (e.g., search queries, target contacts, or message contents). These parameters are instantiated only at the template level and do not fully specify the intermediate results required by subsequent steps. For functionalities that are unique to a specific application, the corresponding app is explicitly specified to preserve realism. After instantiation of the template, we further employ GPT-4o to rewrite the generated instructions, transforming them into more natural, human-like expressions while preserving the essential task information and constraints.

Data Annotation. In the annotation process, the annotator first acquires the screenshot of the current step through the platform, with the accessibility tree synchronized simultaneously. The platform allows annotators to inspect the coordinates and other properties of target UI elements in real time and to annotate the corresponding action accordingly. The annotator then records the State Anchors, reasoning analysis, and operation summary for the current step, and executes the annotated action via the platform to transition the device to the next state. This process is repeated until the task is completed, which is marked by a finish action.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 3 | Overview statistics of AndroTMem-Bench. The first row reports the top app combinations, step length distribution by task type, and overall trajectory length distribution (with comparison to prior benchmarks). The second row shows app usage frequency, overall action-type proportions, and action diversity per task.

Data Quality Control. Our data quality control consists of two components: instruction validation and trajectory validation. Instruction validation is performed before annotation begins: designated reviewers audit each task instruction to identify unclear or ambiguous descriptions, and then edits, revises, or filters out problematic instructions when necessary. Trajectory validation is applied to the annotated interaction traces, with criteria including: (i) whether each step is associated with a corresponding screenshot, (ii) whether the executed trajectory matches the task instruction, and (iii) whether extra annotations (e.g., Summary and Reasoning) are aligned with the corresponding steps. After this quality checking and consolidation process, we obtain the dataset AndroTMem.

- 3.3. Data Statics

Our dataset consists of 1,069 high-quality GUI tasks with substantial step-to-step causal dependencies, comprising 34,473 interaction steps in total, with an average of 32.1 steps per task and a maximum of 65 steps. The dataset spans 50 widely used mobile applications, covering diverse functionalities and cross-app interaction scenarios. In addition to raw trajectories, AndroTMem provides step-aligned auxiliary annotations, including reasoning traces, summaries, and sparse state anchors that mark taskrelevant intermediate states for analysis and evaluation. Figure 3 presents key statistical properties of our dataset.

- 4. AndroTMem-Bench

- 4.1. Experiment Setup

Setup. We evaluate existing GUI agents on AndroTMem-Bench to characterize the intrinsic challenges posed by long-horizon, causally dependent GUI tasks. This evaluation measures the out-of-the-box capability of current GUI agents to solve long-horizon, cross-app tasks with strong step-to-step dependencies. Additional implementation and evaluation details are provided in Appendix C.

Metrics. For step-level evaluation, we adopt Action Matching Score (AMS) following GUI-Odyssey and

AITW. An action is considered correct if its type matches the ground truth. For tap and long_press, predictions are correct when the normalized screen distance to the annotated point is within 14% or the predicted point falls inside the target UI element segmented by SAM2. For swipe actions, correctness is determined by direction, and for text, we use the Average Normalized Levenshtein Similarity (ANLS).

To measure long-horizon interaction memory capability, we introduce Task Completion Rate (TCR) based on state anchors. A task is considered successful if the agent reaches the final anchor while satisfying the causal dependencies among preceding anchors, reflecting whether task-critical intermediate states are correctly preserved and reused.

We further report two efficiency metrics: Avg. Tokens, the average number of tokens consumed per interaction step, and Avg. Time, the average execution time per step. These metrics characterize the computational cost of different history utilization strategies during long-horizon execution.

###### 4.2. Benchmark Results and Diagnosis

Overall Benchmark Performance. Table 2 reports the performance of all evaluated agents on AndroTMemBench, including both closed-source and open-source systems as well as multi-agent frameworks. Overall, all agents exhibit relatively low absolute performance, indicating that long-horizon Android GUI tasks remain highly challenging for current approaches. Among closed-source models, Gemini-3Flash achieves the strongest performance with an AMS of 46.14% and a TCR of 55.21%. However, most other models remain far below this level. For instance, GPT-4o and GPT-5 achieve AMS scores of only 14.24% and 12.37% respectively, with similarly low TCR values. A similar pattern is observed among open-source agents, where UI-TARS-1.5-7B performs best but still achieves only 35.62% AMS and 34.55% TCR. These results suggest that even the strongest existing GUI agents are far from reliably solving long-horizon cross-app workflows. Maintaining consistent task execution across extended interaction trajectories remains an open challenge.

Performance across Task Types. To better understand where current agents struggle, we further analyze performance across different user intent categories. Table 3 reports the performance of GUI agents across seven representative intent categories. We observe substantial performance variation across intents. In particular, intents that involve non-local state reuse or coordination across apps are often among the most challenging for many agents. For example, tasks under Compare & Decide often require agents to gather and reconcile information from multiple sources, while Purchase/Order tasks frequently depend on carrying intermediate state across multiple steps toward successful completion. These workflows involve long dependency chains where later actions critically depend on intermediate results obtained several steps earlier. In practice, such dependency-heavy intents are often more challenging for many agents than more localized interaction patterns, although the exact difficulty varies across models. This suggests that a key challenge in these scenarios lies

Table 2 | Overall performance of evaluated GUI agents on AndroTMem-Bench. AMS measures step-level action accuracy, while TCR evaluates task completion based on anchor-defined intermediate states.

Model AMS TCR Closed-Source

- GPT-4o Hurst et al. [2024] 14.24 11.75
- GPT-5 Singh et al. [2025] 12.37 11.46 Gemini-2.5-Flash Comanici et al. [2025] 15.62 14.51 Gemini-2.5-Pro Comanici et al. [2025] 32.71 41.11 Gemini-3-Flash 46.14 55.21

###### Open-Source

Qwen2.5-VL-7B Bai et al. [2025] 20.71 16.04 UI-TARS-1.5-7B Qin et al. [2025] 35.62 34.55 AgentCPM-GUI-8B Zhang et al. [2025] 14.06 10.49 UI-Venus-Navi-7B Gu et al. [2025b] 14.08 9.64 InfiGUI-R1 Liu et al. [2025e] 14.12 9.20

###### Multi-Agent

Mobile-Agent-E Wang et al. [2025a] 15.77 11.71 COLA Zhao et al. [2025] 17.22 11.85

- Table 3 | Per-intent AMS and TCR of evaluated agents on seven user intents in AndroTMem-Bench. Intent categories follow the taxonomy in Table 6. Abbreviations denote task types (e.g., Lkp: Lookup, CpD: Compare & Decide, Pur: Purchase / Order).

Model CC Cre CpD SeR Lkp Bok Pur

AMS TCR AMS TCR AMS TCR AMS TCR AMS TCR AMS TCR AMS TCR

###### Closed-Source

- GPT-4o 18.72 18.08 22.07 18.95 20.40 18.64 19.10 13.45 21.23 11.90 20.75 10.21 18.18 9.35
- GPT-5 12.35 13.44 12.15 11.66 10.95 11.40 16.05 14.62 10.61 7.14 24.53 20.15 11.36 9.38 Gemini-2.5-Flash 25.21 33.20 27.93 35.86 29.86 32.68 29.05 34.50 17.88 16.67 35.85 30.00 34.09 22.22 Gemini-2.5-Pro 30.81 39.67 33.99 47.80 32.09 36.67 43.79 43.75 37.50 47.06 35.15 12.50 33.45 41.07 Gemini-3-Flash 46.21 56.13 48.12 64.72 42.63 46.05 50.24 56.73 43.58 52.38 43.40 50.00 40.91 44.44

###### Open-Source

Qwen2.5-VL-7B 41.71 46.11 41.84 53.80 40.20 43.13 34.18 41.08 28.76 36.81 32.34 28.57 43.18 44.44 UI-Venus-Navi-7B 16.88 16.11 20.36 14.58 21.45 12.72 21.35 11.70 23.46 9.52 16.98 10.76 27.27 33.33 UI-TARS-1.5-7B 46.94 44.94 40.90 47.93 47.86 48.23 50.73 46.45 44.07 38.71 43.23 53.06 52.27 55.56 AgentCPM-GUI-8B 25.50 25.15 35.41 40.87 30.44 23.70 26.97 25.85 30.13 32.26 19.80 26.53 27.27 11.11 InfiGUI-R1 12.50 8.26 15.83 10.42 14.22 8.89 13.33 8.33 21.40 14.28 18.66 33.33 22.80 50.00

###### Multi-Agent

COLA 15.42 10.92 17.26 12.45 18.05 12.01 16.88 14.28 16.43 9.85 15.82 7.92 12.58 12.38 Mobile-Agent-E 16.50 10.50 16.00 12.00 17.46 13.50 21.47 14.50 18.21 8.61 24.79 15.00 22.23 18.54

not only in individual action prediction, but also in the ability to preserve and reuse task-critical intermediate information across the interaction trajectory.

Diagnosis: Memory as the Primary Bottleneck. We further analyze agent performance with respect to interaction horizon length. As shown in Figure 1, performance consistently degrades as the number of steps increases across all evaluated models. This decline cannot be explained solely by local perception or action prediction errors. Instead, many failures occur when agents must recall and reuse information obtained earlier in the interaction, such as retrieved prices, selected items, or identified contacts. This pattern aligns with the design of AndroTMem-Bench, where sparse but causally critical intermediate states determine downstream decisions. However, existing history representations—whether raw trajectories or coarse summaries—often fail to preserve these key states as trajectories grow longer. These results indicate that the primary bottleneck in long-horizon GUI interaction lies in representing and retrieving task-critical intermediate states across the interaction trajectory.

Implication for Interaction Memory Design. The above diagnosis indicates that effective long-horizon GUI interaction requires more than accurate perception or action prediction. Agents must be able to explicitly represent and reuse task-critical intermediate states that form the causal backbone of the interaction trajectory. Motivated by this observation, we introduce ASM, a causally structured interaction memory mechanism that organizes past interactions around state anchors and their causal dependency relations. By explicitly modeling intermediate states that determine downstream decisions, ASM enables agents to retrieve the most relevant task information without relying on excessively long raw histories. In the following section, we describe the design of ASM and evaluate how structured anchor-based memory affects agent performance on AndroTMem-Bench.

#### 5. Anchored State Memory

###### 5.1. Motivation from Benchmark Diagnosis

Long-horizon mobile GUI tasks require agents to maintain and utilize interaction history across many steps. In AndroTMem-Bench, tasks often span multiple applications and contain dependency chains among intermediate states (e.g., extracted information, completed subgoals, or environment changes) that determine downstream decisions. Existing GUI agents typically rely on two forms of history representation. Raw interaction traces preserve full screenshot–action trajectories but grow rapidly with task horizon, causing irrelevant UI transitions to dilute attention over task-relevant states. Compressed summaries reduce context length but often omit fine-grained intermediate states needed for later reasoning. Our benchmark diagnosis shows that both approaches degrade significantly as interaction horizons grow (Fig. 4), indicating that history representation is a key bottleneck for long-horizon GUI agents. These findings motivate a history representation that explicitly captures task-relevant intermediate states and their causal roles in the workflow.

###### 5.2. Anchored State Memory Definition

To address this limitation, we propose Anchored State Memory (ASM), a structured memory representation that organizes interaction history into sparse but decision-critical intermediate states, rather than replaying the full trajectory or compressing it into a free-form summary. ASM explicitly represents not only what intermediate states should be remembered, but also how they are causally related and reused across steps.

Given an interaction trajectory

𝜏 = {(𝑠𝑡, 𝑎𝑡)}𝑇𝑡=0, (1) where 𝑠𝑡 denotes the UI state and 𝑎𝑡 the executed action, ASM extracts a set of state anchors

𝐴 = {𝑚𝑘}𝑘𝐾=1. (2)

Each anchor corresponds to a task-relevant intermediate state that may be required by future decisions, such as an extracted value, an identified entity, a completed subgoal, a persistent state change, or an exception that must later be handled. Each anchor is represented as

𝑚𝑘 = ⟨type𝑘, content𝑘, evidence𝑘, links𝑘⟩. (3)

Specifically, type𝑘 indicates the functional role of the state in the workflow(see Appendix B for the full definition of anchor types), content𝑘 describes the semantic information carried by the state, evidence𝑘 records the supporting UI observations or interaction steps from which the anchor is grounded, and links𝑘 represent causal or dependency relations with other anchors.

The key distinction between ASM and conventional history representations is that ASM treats memory as a set of reusable intermediate states, rather than as a verbatim sequence of observations and actions. This design is particularly suitable for long-horizon GUI tasks, where information obtained in one phase or app must often be reused only after many intervening steps in a later phase. In addition to individual anchors, ASM explicitly models the dependency structure among them. A link in links𝑘 indicates that an anchor is derived from, depends on, or should be retrieved together with another anchor. For example, in a cross-app comparison task, the anchor Cheaper_Item may depend on two previously extracted anchors, Price_JD and Price_Taobao; the resulting dependency relation makes

0.6

UI-TARS-1.5-7B GPT-4o AgentCPM-GUI-8B

| |
|---|

| |
|---|

0.5

AMS(ActionMatchingScore)

Qwen2.5-VL-7B UI-Venus-Navi-7B Gemini-2.5-flash

| |
|---|

| |
|---|

0.4

0.3

0.2

0.1

0.0

10-19 20-29 30-39 40-49 50-59 60-69 Step Range

(a) Raw History

0.6

UI-TARS-1.5-7B GPT-4o AgentCPM-GUI-8B

| |
|---|

| |
|---|

0.5

AMS(ActionMatchingScore)

Qwen2.5-VL-7B UI-Venus-Navi-7B Gemini-2.5-flash

| |
|---|

| |
|---|

0.4

0.3

0.2

0.1

0.0

10-19 20-29 30-39 40-49 50-59 60-69 Step Range

(b) Coarse Summary

0.6

UI-TARS-1.5-7B GPT-4o AgentCPM-GUI-8B

| |
|---|

| |
|---|

0.5

AMS(ActionMatchingScore)

Qwen2.5-VL-7B UI-Venus-Navi-7B Gemini-2.5-flash

| |
|---|

| |
|---|

0.4

0.3

0.2

0.1

0.0

10-19 20-29 30-39 40-49 50-59 60-69 Step Range

(c) ASM (Ours)

- Figure 4 | Agent performance across different interaction step ranges under three history utilization strategies: (a) Raw History, (b) Coarse Summary, and (c) Anchored State Memory (ASM).

CC

Cre

CpD

LkP SeR

Bok

Pur

UI-TARS-1.5-7B

GPT-4o

AgentCPM-GUI-8B

Qwen2.5-VL-7B

UI-Venus-Navi-7B

Gemini-2.5-flash

(a) Raw History

CC

Cre

CpD

LkP SeR

Bok

Pur

UI-TARS-1.5-7B

GPT-4o

AgentCPM-GUI-8B

Qwen2.5-VL-7B

UI-Venus-Navi-7B

Gemini-2.5-flash

(b) Coarse Summary

CC

Cre

CpD

LkP SeR

Bok

Pur

UI-TARS-1.5-7B

GPT-4o

AgentCPM-GUI-8B

Qwen2.5-VL-7B

UI-Venus-Navi-7B

Gemini-2.5-flash

(c) ASM(ours)

- Figure 5 | Model performance across different task categories under three history utilization strategies.

this non-local reasoning step explicit and facilitates later retrieval when the agent needs to decide which item to add to the cart.

During task execution, the agent maintains a memory bank of anchors and interacts with ASM in a retrieve–reason–update manner. At each step, the agent retrieves anchors conditioned on the current UI state and user instruction, enabling it to reference the most relevant historical states without relying on extremely long raw trajectories. The retrieved anchors are then used jointly with the current UI context to predict the next action. After that, the agent analyzes the current step to determine whether new anchors should be created, whether existing anchors should be updated or invalidated, and whether new dependency links should be added to the memory bank.

Formally, at step 𝑡, given the current UI state 𝑠𝑡 and the current memory bank 𝐴𝑡−1, ASM produces a retrieved subset 𝐴ˆ𝑡 ⊆ 𝐴𝑡−1 for decision making, predicts the next action 𝑎𝑡, and then updates the memory bank:

𝐴ˆ𝑡 = Retrieve(𝑠𝑡, 𝐴𝑡−1), (4) 𝑎𝑡 = Act(𝑠𝑡, 𝐴ˆ𝑡), (5) 𝐴𝑡 = Update(𝐴𝑡−1, 𝑠𝑡, 𝑎𝑡). (6)

Compared with raw trajectory replay, ASM removes locally irrelevant observations and reduces distraction from long histories. Compared with free-form summaries, it preserves dependency-critical intermediate states in a grounded, structured, and updateable form. This makes ASM particularly effective for long-horizon GUI tasks that require non-local state reuse and cross-app coordination.

- Table 4 | History modeling ablation on AndroTMem-Bench. Columns report AMS and TCR (higher is better, ↑). For each model, best is in bold and second-best is underlined across Raw/Summary/Anchored State Memory.

Model History AMS↑ TCR↑ Avg Token Avg Time↓ Closed-Source

- GPT-4o Hurst et al. [2024]

+ Raw 14.24 11.75 2671.1 7.019 + Summary 15.49 13.02 993.4 4.521 + Anchored State Memory 19.83 17.72 1265.2 5.611

- GPT-5 Singh et al. [2025]

+ Raw 12.37 11.46 2676.1 10.465

+ Summary 18.87 17.08 911.3 6.210 + Anchored State Memory 21.63 23.30 1511.9 7.320

+ Raw 15.62 14.51 2893.6 14.917

Gemini-2.5-Flash Comanici et al. [2025]

+ Summary 17.14 16.59 1348.6 9.538 + Anchored State Memory 26.90 33.24 1783.7 8.694

+ Raw 32.71 41.11 2713.6 15.511 + Summary 36.39 40.83 1363.7 10.617 + Anchored State Memory 57.37 63.40 1810.1 8.732

Gemini-2.5-Pro Comanici et al. [2025]

+ Raw 46.14 55.21 6239.1 8.108 + Summary 50.05 56.44 2021.3 5.427 + Anchored State Memory 59.03 65.05 2503.6 8.345

Gemini-3-Flash

###### Open-Source

+ Raw 20.71 16.04 13012.8 8.102 + Summary 35.09 42.39 5691.3 2.922 + Anchored State Memory 40.36 46.19 6065.7 2.511

Qwen2.5-VL-7B Bai et al. [2025]

+ Raw 35.62 34.55 4762.7 3.109 + Summary 44.49 43.84 1343.5 2.023 + Anchored State Memory 46.07 46.21 1926.7 1.925

UI-TARS-1.5-7B Qin et al. [2025]

+ Raw 14.06 10.49 3561.4 0.880 + Summary 21.40 22.96 1785.8 0.844 + Anchored State Memory 28.63 27.89 1902.3 0.837

AgentCPM-GUI-8B Zhang et al. [2025]

+ Raw 14.08 9.64 4083.8 3.151 + Summary 16.07 13.12 1807.1 2.578 + Anchored State Memory 19.01 16.64 1186.9 2.343

UI-Venus-Navi-7B Gu et al. [2025b]

+ Raw 14.12 9.20 15987.3 5.303 + Summary 19.84 14.36 5563.0 2.485 + Anchored State Memory 25.46 18.92 6115.9 3.424

InfiGUI-R1 Liu et al. [2025e]

###### Multi-Agent

COLA Zhao et al. [2025] - 15.77 11.71 2167.1 14.348 Mobile-Agent-E Wang et al. [2025a] - 17.22 11.85 11099.83 33.960

###### 5.3. History Utilization Ablation: Does ASM Help?

Setup (History ablation). To study how interaction history representation affects long-horizon decision making, we evaluate three history utilization mechanisms that differ only in how historical information is represented:

- (i) Raw History, where the agent receives the sequence of past screenshot–action pairs;
- (ii) Coarse Summary, where the agent is provided with a compressed textual summary of prior

interactions;

- (iii) Anchored State Memory, where task-relevant intermediate states and their causal relations

are represented as structured milestone anchors.

Importantly, we do not provide manually crafted summaries or anchors. Instead, both summaries and anchors are generated automatically by each agent from the raw interaction history using the

###### Main History-Related Failure Modes under Long-Horizon GUI Tasks

Current UI Current UI

###### FM1: State Loss FM2: State Mis-binding

[Figure 80]

[Figure 81]

GPT-4O Qwen-2.5-vl

[Figure 82]

[Figure 83]

[Figure 84]

History: Action: TAP(E1) → add-to-cart

History: raw trace is long/noisy Action: TEXT("") → missing T1

Raw Trace

Raw Trace

[Figure 85]

[Figure 86]

T1

History: “… get the cheaper item by …” Action: TAP(E1) → add-to-cart

History: “… first video title: T1_partial …” Action: TEXT(T1_partial) → truncated History: [ctx] title=T1; [ctx] msg="Here is the title." Action: TEXT(T1) → then send msg

Coarse Summary

Coarse Summary

[Figure 87]

[Figure 88]

History: [dep] price_jd=1249; [dep] price_tb=1329;[sub] cheaper=JD Action: OPEN_APP(JD)

[Figure 89]

[Figure 90]

ASM

ASM

E1

WeChat chat with Alice (input field)

Taobao item detail page

Task: Compare prices of AirPods Pro 2 on JD and Taobao → add the cheaper one to cart → share the item.

Task: Open Bilibili → get the title of the first homepage video → send it to Alice on WeChat → send “Here is the title.”

[Figure 91]

Current UI Current UI

###### FM3: Context Drift FM4: Unverified Progress

[Figure 92]

Gemini-2.5 GPT-5

[Figure 93]

[Figure 94]

E4

History: phase not stabilized Action: TAP(E3) → comment again

Raw Trace

History: milestone implicit Action: TAP(E5) → redundant add

Raw Trace

[Figure 95]

[Figure 96]

- E2

- E3

History: “Comment posted.” Action: LONG_PRESS(E2) → copy

Coarse Summary

History: “… has been added to cart …” Action: TAP(E4) → share the cheaper one

Coarse Summary

[Figure 97]

[Figure 98]

History: [state] comment_posted; [ctx] my_comment Action: LONG_PRESS(E2) → copy

History: [sub] added_to_cart=true Action: TAP(E4) → share the cheaper one

[Figure 99]

[Figure 100]

ASM

ASM

E5

Comment posted (should copy)

JD item detail page (item in the cart)

Task: Compare prices of AirPods Pro 2 on JD and Taobao → add the cheaper one to cart → share the item.

Task: Play “Sunny Day” on NetEase Music → post a comment → copy the posted comment → paste into Notes

- Figure 6 | History-related failure modes in long-horizon GUI tasks. Raw traces and coarse summaries often lose task-relevant states, leading to errors such as state loss, mis-binding, context drift, and unverified progress, while ASM preserves structured intermediate states for correct decisions.

same set of rules. This design avoids biases introduced by artificially controlling the information strength and ensures that differences across mechanisms primarily arise from how interaction history is represented and utilized.

Overall Performance. Table 4 reports the results across multiple GUI agents. Across both closed-source and open-source models, ASM consistently achieves the best performance in terms of AMS and TCR. These improvements demonstrate that explicitly modeling intermediate states substantially enhances agents’ ability to perform long-horizon reasoning.

Robustness and Task-Level Analysis. We analyze how history representations scale with task length and generalize across task categories. As shown in Fig. 4, both raw traces and summaries exhibit noticeable performance degradation as the step range increases, while ASM maintains significantly higher AMS across long horizons, indicating stronger robustness to history accumulation. Across task categories (Fig. 5), ASM consistently improves performance, with particularly notable gains in tasks requiring cross-step reasoning or cross-app coordination.

Efficiency Analysis. Table 4 also reports execution cost. ASM consistently improves AMS and TCR over summary-based and raw-history baselines, while keeping token usage and inference time comparable to summary-based history and far below raw trajectory replay, suggesting a better efficiency–effectiveness trade-off.

###### 5.4. Failure Modes: What goes wrong in long-horizon history?

Beyond aggregate performance metrics, we analyze common failure modes observed in long-horizon interactions. These failure modes help explain why raw traces and coarse summaries degrade more rapidly than ASM as horizons grow. As illustrated in Fig. 6, across models and tasks, failures frequently

arise from difficulties in maintaining and reusing task-critical intermediate states. We identify several recurring patterns:

- (i) State Loss, where task-relevant information extracted earlier in the trajectory becomes un-

available when needed later;

- (ii) State Mis-binding, where intermediate states are associated with incorrect UI elements or

objects;

- (iii) Context Drift, where agents misinterpret the current interaction context or application state;
- (iv) Unverified Progress, where agents proceed without confirming that a subgoal has been

successfully completed;

- (v) Interruption Handling Failure, where agents fail to recover from pop-ups, permission dialogs,

or unexpected UI interruptions.

These failure modes frequently occur in long interaction trajectories and highlight the importance of explicitly modeling intermediate states and their dependencies. ASMmitigates these issues by organizing task-relevant intermediate states into structured anchors with causal relations, enabling more stable decision making across extended interaction sequences.

#### 6. Limitations

While AndroTMem isolates and diagnoses within-task interaction memory under long-horizon Android trajectories, it does not yet fully capture cross-session long-term tasks where critical state must persist across interruptions, days-long gaps, or multi-episode workflows. In addition, our benchmark is built on curated trajectories with fixed task goals and relatively stable app states; future progress will benefit from more dynamic benchmarks that expose UI drift, content updates, and stochastic outcomes, as well as environment-oriented evaluations that more tightly couple agent behavior with evolving external conditions and feedbacks.

#### 7. Conclusion

In this paper, we introduce AndroTMem-Bench, a benchmark for studying long-horizon mobile GUI tasks with cross-app workflows and complex dependency structures. Our diagnosis shows that existing GUI agents degrade significantly as task horizons increase, revealing history modeling as a key bottleneck. To address this issue, we propose Anchored State Memory (ASM), a structured history representation that organizes interaction trajectories into sparse intermediate-state anchors and their causal relations. Experiments across diverse GUI agents show that ASM consistently improves both action accuracy and task completion, while enhancing robustness in long-horizon scenarios. Our analysis further identifies common history-related failure modes and demonstrates how structured intermediate-state memory mitigates them. We hope AndroTMem and ASM facilitate future research on memory mechanisms for reliable long-horizon GUI agents.

#### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Guozhi Wang, Dingyu Zhang, Shuai Ren, and Hongsheng Li. Amex: Android multi-annotation expo dataset for mobile gui agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2138–2156, 2025.

Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, et al. Gui-world: A dataset for gui-oriented multimodal llm-based agents. arXiv e-prints, pages arXiv–2406, 2024a.

Jingxuan Chen, Derek Yuen, Bin Xie, Yuhao Yang, Gongwei Chen, Zhihao Wu, Li Yixing, Xurui Zhou, Weiwen Liu, Shuai Wang, et al. Spa-bench: A comprehensive benchmark for smartphone agent evaluation. In NeurIPS 2024 Workshop on Open-World Agents, 2024b.

Junhao Chen, Yu Huang, Siyuan Li, Rui Yao, Hanqian Li, Hanyu Zhang, Jungang Li, Jian Chen, Bowen Wang, and Xuming Hu. Knowmt-bench: Benchmarking knowledge-intensive long-form question answering in multi-turn dialogues. arXiv preprint arXiv:2509.21856, 2025.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yunkai Dang, Mengxi Gao, Yibo Yan, Xin Zou, Yanggan Gu, Jungang Li, Jingyu Wang, Peijie Jiang, Aiwei Liu, Jia Liu, and Xuming Hu. Exploring response uncertainty in MLLMs: An empirical evaluation under misleading scenarios. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18132–18173, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.916. URL https://aclanthology.org/2025.emnlp-main.916/.

Yue Ding, Yiyan Ji, Jungang Li, Xuyang Liu, Xinlong Chen, Junfei Wu, Bozhou Li, Bohan Zeng, Yang Shi, Yushuo Guan, et al. Omnisift: Modality-asymmetric token compression for efficient omni-modal large language models. arXiv preprint arXiv:2602.04804, 2026.

Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9062–9072, 2025.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.

Jihao Gu, Qihang Ai, Yingyao Wang, Pi Bu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Ziming Wang, Yingxiu Zhao, Ming-Liang Zhang, et al. Mobile-r1: Towards interactive reinforcement learning for vlm-based mobile agent via task-level rewards. arXiv preprint arXiv:2506.20332, 2025a.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833, 2025b.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281– 14290, 2024.

Xuming Hu, Hanqian Li, Jungang Li, Yu Huang, Shuliang Liu, Qi Zheng, Junhao Chen, and Aiwei Liu. Videomark: A distortion-free robust watermarking framework for video diffusion models. arXiv preprint arXiv:2504.16359, 2025.

Sirui Huang, Hanqian Li, Yanggan Gu, Xuming Hu, Qing Li, and Guandong Xu. Hyperg: Hypergraphenhanced llms for structured knowledge. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1218–1228, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Bin Kang, Shaoguo Wen, Yifei Bi, Shunlong Wu, Xinbin Yuan, Rui Shao, Junle Wang, and Zhuotao Tian. LonghorizonUI: A unified framework for robust long-horizon task automation of GUI agent. In The Fourteenth International Conference on Learning Representations, 2026. URL https://open review.net/forum?id=BK7Mk5d4WE.

Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, et al. Mobileworld: Benchmarking autonomous mobile agents in agent-user interactive, and mcp-augmented environments. arXiv preprint arXiv:2512.19432, 2025.

Jiahao Li and Kaer Huang. A survey on gui agents with foundation models enhanced by reinforcement learning. arXiv preprint arXiv:2504.20464, 2025.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 8778–8786, 2025.

Wei Li, William E Bishop, Alice Li, Christopher Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on ui control agents. Advances in Neural Information Processing Systems, 37:92130–92154, 2024.

Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Stan Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19498–19508,

- 2025a.

Zhixin Lin, Jungang Li, Shidong Pan, Yibo Shi, Yue Yao, and Dongliang Xu. Mind the third eye! benchmarking privacy awareness in mllm-powered smartphone agents. arXiv preprint arXiv:2508.19493,

- 2025b.

Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, and Wenchao Meng. Learnact: Few-shot mobile gui agent with a unified demonstration benchmark. arXiv preprint arXiv:2504.13805, 2025a.

Haowei Liu, Xi Zhang, Haiyang Xu, Yuyang Wanyan, Junyang Wang, Ming Yan, Ji Zhang, Chunfeng Yuan, Changsheng Xu, Weiming Hu, et al. Pc-agent: A hierarchical multi-agent collaboration framework for complex task automation on pc. arXiv preprint arXiv:2502.14282, 2025b.

Kai Liu, Jungang Li, Yuchong Sun, Shengqiong Wu, Jianzhang Gao, Daoan Zhang, Wei Zhang, Sheng Jin, Sicheng Yu, Geng Zhan, Jiayi Ji, Fan Zhou, Liang Zheng, Shuicheng YAN, Hao Fei, and Tat-Seng Chua. Javisgpt: A unified multi-modal llm for sounding-video comprehension and generation. In Conference on Neural Information Processing Systems [Spotlight], November 2025c.

Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Wendong Fan, Ge Zhang, Jiajun Shi, Weihao Xuan, et al. Verigui: Verifiable long-chain gui dataset. arXiv preprint arXiv:2508.04026, 2025d.

Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239, 2025e.

Quanfeng Lu, Wenqi Shao, Zitao Liu, Lingxiao Du, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, and Ping Luo. Guiodyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22404–22414, 2025a.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025b.

Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Ye Pan, Chi Kit Wong, Yuanhuiyi Lyu, Hanqian Li, Jiahao Huo, Jiacheng Chen, Lutao Jiang, Xu Zheng, and Xuming Hu. Egointent: An egocentric step-level benchmark for understanding what, why, and next. arXiv preprint arXiv:2603.12147, 2026.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708–59728, 2023.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366, 2023.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025.

Sicheng Tao, Jungang Li, Yibo Yan, Junyan Zhang, Yubo Gao, Hanqian Li, ShuHang Xun, Yuxuan Fan, Hong Chen, Jianxiang He, et al. Moss-chatv: Reinforcement learning with process reasoning reward for video temporal reasoning. arXiv preprint arXiv:2509.21113, 2025.

Junyang Wang, Haiyang Xu, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v: Learning mobile device operation through video-guided multi-agent collaboration. arXiv preprint arXiv:2502.17110, 2025a.

Luyuan Wang, Yongyu Deng, Yiwei Zha, Guodong Mao, Qinmin Wang, Tianchen Min, Wei Chen, and Shoufa Chen. Mobileagentbench: An efficient and user-friendly benchmark for mobile llm agents. arXiv preprint arXiv:2406.08184, 2024a.

Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, et al. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024b.

Xuehui Wang, Zhenyu Wu, JingJing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, et al. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478, 2025b.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.

Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, et al. Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning. arXiv preprint arXiv:2509.08755, 2025.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Pengzhi Gao, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, and Bo An. Mobile-bench-v2: A more realistic and comprehensive benchmark for vlm-based mobile agents. arXiv preprint arXiv:2505.11891, 2025a.

Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. Androidlab: Training and systematic benchmarking of android autonomous agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2144–2166, 2025b.

Shuhang Xun, Sicheng Tao, Jungang Li, Yibo Shi, Zhixin Lin, Zhanhui Zhu, Yibo Yan, Hanqian Li, Linghao Zhang, Shikang Wang, et al. Rtv-bench: Benchmarking mllm continuous perception, understanding and reasoning through real-time video. arXiv preprint arXiv:2505.02064, 2025.

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024a.

Jiwen Zhang, Jihao Wu, Teng Yihua, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12016–12031, 2024b.

Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, et al. Agentcpm-gui: Building mobile-use agents with reinforcement fine-tuning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 155–180, 2025.

Di Zhao, Longhui Ma, Siwei Wang, Miao Wang, and Zhao Lv. Cola: A scalable multi-agent framework for windows ui task automation. arXiv preprint arXiv:2503.09263, 2025.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

Qi Zheng, Shuliang Liu, Yu Huang, Sihang Jia, Jungang Li, Lyuhao Chen, Junhao Chen, Hanqian Li, Aiwei Liu, Yibo Yan, et al. A visual semantic adaptive watermark grounded by prefix-tuning for large vision-language model. arXiv preprint arXiv:2601.07291, 2026.

### Appendix

#### A. Details of AndroTMem-Bench

###### A.1. App Categories

To cover a broad spectrum of mobile interaction scenarios, the applications in our study are organized into 16 task-oriented functional domains. These categories represent the most frequent and essential user activities within the Android ecosystem, encompassing a broad spectrum of UI structures and interaction logic—from simple utility functions to complex social and productivity workflows. The detailed classification, representative applications, and their corresponding functional descriptions are systematically documented in Table 5.

Table 5 | Functional app grouping based on predefined task-oriented categories (Part I).

Category Applications Description Shopping JD, Taobao, Pinduoduo, Suning Product browsing, price comparison, order placement, and online

payment for consumer goods. Travel Ctrip, Didi Chuxing, Fliggy Trip planning, transportation booking, ride-hailing services, and real-time travel management. Map Amap, Baidu Maps, Tencent Maps Location search, route planning, navigation, and real-time traffic information. Mail Gmail, Outlook Email communication, message management, information exchange, and productivity-oriented correspondence handling. Cloud Storage Baidu Netdisk, Google Drive Cloud-based file storage, synchronization, backup, and cross-device

file sharing. Meeting DingTalk, Tencent Meeting, WeCom,

Online meetings, video conferencing, real-time communication, and collaborative work coordination.

Zoom

Note-taking & Document Tools

Feishu (Lark), WPS Office, DingTalk, Notes, Youdao Notes

Document editing, note-taking, information organization, and collaborative content creation.

Social WeChat, QQ, Weibo, Xiaohongshu Social networking, instant messaging, content sharing, and online

community interaction. Category Applications Description Entertainment Bilibili, Douyin, Tencent Video, iQIYI,

Consumption of short-form and long-form video content, live streaming, and entertainment media browsing.

YouTube

News & Information Toutiao, NetEase News, Sina News News aggregation, personalized information feeds, and real-time information consumption.

System Tool File Manager, Gallery, Settings, Browser, Recorder, Calculator, Clock, Calendar, Chrome

Device management, system configuration, file organization, and basic utility functions.

Knowledge Zhihu Knowledge sharing, question answering, expert content consump-

tion, and community-based discussions. Music App NetEase Cloud Music, QQ Music, Kugou

Music streaming, playlist management, audio search, and personalized music recommendation.

Music, Kuwo Music

Takeout Meituan, Ele.me Food delivery services, restaurant browsing, order placement, and logistics tracking. Payment Alipay Mobile payment, financial transactions, digital wallet management, and online-offline payment integration. Health Keep Fitness training, exercise tracking, health monitoring, and personalized workout guidance.

###### A.2. Task Types

In AndroTMem-Bench, long-chain tasks are categorized according to the user’s primary intent rather than application domains. This approach acknowledges that even within the same application, user goals vary significantly and dictate different interaction patterns. In AndroTMem-Bench, each task is mapped to a single primary intent that represents the core objective of the sequence. Table 6 details the classification of simulated daily operations, providing definitions and representative examples for each category.

A parameterized action space A is defined to support precise GUI interaction and trajectory annotation in AndroTMem-Bench. The action space provides a unified interface that covers both low-level touch-based interactions and high-level system commands on Android devices. Each action 𝑎 ∈ A is specified by an operation type and associated arguments. Table 7 summarizes the action space used in AndroTMem-Bench, detailing the supported action types together with their required arguments and functional descriptions.We use a unified coordinate system and consistent argument schema so that the same action definitions apply across devices and screen resolutions. All actions are logged with their parameters to support reproducible replay and fine-grained analysis.

- Table 6 | Classification of tasks in AndroTMem based on primary user intent.

Task Type Primary Intent Example Scenarios Lookup Information retrieval/veri-

Checking shop hours, searching for news, or finding answers to specific questions.

fication

Compare & Decide Decision making via comparison

Comparing menus of three restaurants or checking product parameters across different platforms.

Purchase / Order Transaction completion Placing food delivery orders, clearing a shopping cart, or

purchasing memberships. Booking Resource/Time reserva-

Booking hotel rooms, reserving train tickets, or scheduling a doctor’s appointment.

tion

Communicate Coordination with others Confirming meeting details with a contact or initiating a group discussion.

Share / Recommend

Content distribution Sending a store link to a friend or sharing news and music playlists to social media.

Create Content Generative output Writing a note, editing a document, or posting a review/comment on a platform.

Configure System/App adjustment Granting location permissions, enabling notifications, or modifying system settings.

Payment Financial processing Paying for an existing order, applying for a refund, or processing utility bills.

Navigation Trip execution Planning a route between locations or hailing a ride and sharing the trip status.

- A.3. Action Space
- A.4. Processing of Private Information

All personally identifiable information (PII) is strictly anonymized during the annotation process to ensure privacy protection. This was achieved by utilizing temporary accounts and systematically modifying or replacing sensitive data fields. Specifically, any information that could potentially identify an individual—including residential addresses, account usernames, phone numbers, contact lists, and financial payment details—was entirely removed or substituted with generic placeholders. Consequently, the traces within AndroTMem-Bench contain no traceable real-world identity markers.

- A.5. Task Template Examples

To ensure structural consistency while maintaining data diversity, task instructions in AndroTMemBench are generated using a templated approach. Placeholders prefixed with # indicate slots filled from predefined sets: #group refers to specific application functional categories (as detailed in

- Table 7 | The argument and functionality of different actions in AndroTMem. ’pos1’ and ’pos2’ denote the position (𝑥, 𝑦).

Action Argument Functionality

TAP [pos1] click the on-screen position INPUT_TEXT [text] input the text into the currently focused text field using the system

keyboard

LONG_PRESS [pos1] press the screen for a long time to copy texts or download images SWIPE [pos1] swipe the screen in a given direction (up, down, left, or right) to

scroll the page

SWIPE_TWO_POINT [pos1,pos2] swipe from position pos1 to position pos2, enabling precise scrolling

or drag-and-drop operations WAIT – wait for a specified duration or until the UI state changes FINISH – the sign that the instruction has been completed OPEN_APP – launch the specified application CAPTURE_SCREEN – capture the current screen for visual perception or state analysis HOME – go to the home screen BACK – go to the previous screen

Table 5), and #contract represents a selection from a curated list of virtual contacts. An example of a complex, cross-app task template and its instantiated version is shown below:

Template: Compare the prices of the best-selling {product} on #group:shopping_app and #group:shopping_app. Add the product with the highest price to your shopping cart and share it with your contact #contract on #group:social.

Example Instance: Compare the prices of the best-selling Apple AirPods Pro 2 on JD and Taobao. Add the product with the highest price to your shopping cart and share it with your contact Alice on WeChat.

###### A.6. Annotation Platform

We develop a semi-automatic annotation platform based on the Android Debug Bridge (ADB) to support efficient collection and fine-grained labeling of long-horizon GUI trajectories. The platform can connect to either real Android devices or Android emulators, capture the current screen in real time, and allow annotators to label interaction actions such as tap, long_press, swipe based on the screenshot. The annotated action commands are synchronously executed on the connected device, and the resulting device UI state is immediately reflected back to the annotator to support subsequent interactions and annotations, forming a closed-loop human-in-the-loop workflow.

In addition to recording the screenshot and corresponding action information at each step, the platform also synchronously captures the XML representation of the current UI screen (derived from the Android accessibility tree), which contains the page layout structure and UI element attributes. More importantly, the platform supports configurable auxiliary annotation fields, covering a wide range of data types, including free-form text, boolean values, JSON objects, object arrays, singlechoice groups, and multiple-choice groups. Leveraging this functionality, we annotate for each step a reasoning trace, a short summary, and State Anchors (object arrays) that capture task-relevant intermediate states and their causal roles.

#### B. Anchored State Memory

###### B.1. State Anchor Categories

Anchored State Memory (ASM) organizes interaction history into sparse intermediate-state anchors together with their causal dependency links. Based on their semantic roles and how they affect subsequent decisions, we categorize state anchors into six types. We summarize the taxonomy in

- Table 8 and further elaborate below.

- Table 8 | Causal state anchors representing key task progress and execution states in GUI agent interactions.

Type Name Recommended Values Description

Completion of intermediate tasks, such as successful login, retrieval of search results, or adding an item to the shopping cart

Key Subgoal Achievement

SUBGOAL

Changes in the environment or application mode, such as entering the payment interface or opening a permission request page

State Transition and Mode Switching

STATE_CHANGE

Causally Dependent Steps

Steps whose execution depends on prior data or states, such as copying text or filling in a form

DEPENDENCY

Actions taken to address interruptions or errors, such as closing pop-up advertisements or retrying a network request

Exception Handling EXCEPTION

Essential contextual information that constrains task execution, such as selecting dates or setting filter conditions

Global Constraint Information

CONTEXT_INFO

Task Completion Indicator

FINISH The sign that the instruction has been completed

Subgoal completion (subgoal). These anchors mark the successful completion of intermediate objectives (i.e., meaningful progress toward the overall task), such as “successfully logged in”, “search results displayed”, or “item added to cart”. Subgoal anchors indicate that the agent has reached a verifiable intermediate milestone that may be required by later steps.

State transition and mode switching (state_change). These anchors capture significant changes in application state or operating mode that alter the execution context for subsequent steps. Examples include entering a new module, switching accounts, granting permissions, or enabling a critical setting. Such transitions often change what actions are valid or what information is visible.

Causal dependency steps (dependency). These anchors denote steps whose outcomes are explicitly required by later actions. This is especially common in cross-app workflows, e.g., copying content in App A that will be pasted in App B, or setting a filter that directly affects subsequent results. Dependency anchors make these prerequisites explicit and support downstream validation.

Exception handling (exception). These anchors record the handling of disruptive events that must be resolved to proceed, such as closing pop-ups, dismissing ads, handling error dialogs, or retrying after network failures. Exception anchors help agents recognize abnormal conditions and apply recovery strategies instead of derailing the main workflow.

Global contextual information (context_info). These anchors store task-critical context that may not correspond to a single atomic action but is essential for subsequent reasoning, such as key

user-provided parameters (dates, locations, names) or important system feedback (warnings, error messages). They function as persistent constraints or references throughout the task.

Task completion (finish). These anchors indicate that the overall task has been successfully completed, e.g., “order submitted” or “message sent”. Finish anchors define terminal success conditions and provide a clear endpoint for evaluation.

###### B.2. Generation Protocol for Summary and ASM

To ensure fairness and reproducibility in the comparison between Raw History, Summary, and ASM, all historical representations are generated automatically from the same interaction trajectory using a unified rule-based protocol. Importantly, the generation process is model-agnostic and does not involve model-specific tuning.

Unified Prompting Protocol. Both summaries and anchors are generated using a fixed prompt template that specifies the available action space, the normalized coordinate system, the output schema, and the generation guidelines. The prompt structure is identical across all evaluated agents. The complete prompt templates used in our experiments are provided in Table 9 and Table 10. These templates define the allowed actions, field requirements for action parameters, structured output formats, and semantic guidelines for summarization and anchor extraction. As a result, all models operate under the same instruction format when generating summaries or anchors.

Trigger Conditions for Anchor Generation. Anchors are generated incrementally during task execution. During evaluation, anchors are generated online by the agent from the observed interaction history rather than taken from the annotated ground truth. At each interaction step, the model receives the current UI observation, previously generated anchors, and the interaction history. Based on the generation guidelines defined in the prompt, the model decides whether a new anchor should be produced. Anchors are generated only when the interaction produces a semantically meaningful task state, such as subgoal completion, state transitions, dependency establishment, exception handling, or contextual information updates. This design ensures that anchors represent task-relevant milestones rather than trivial UI transitions.

Causal Link Generation. In addition to individual anchors, ASM explicitly models the causal dependencies between anchors. Each anchor may optionally include a causal link that connects it to a previously generated anchor, forming a lightweight dependency structure over the interaction trajectory. These links capture task-relevant relations such as prerequisite conditions, enabling states, or results of previous actions. The goal of causal links is not to construct a full execution graph, but to expose task-critical dependencies that guide memory retrieval during long-horizon interaction. During generation, the model is instructed to produce causal links only when a semantically meaningful dependency exists between two anchors. This design avoids constructing dense or trivial connections and instead focuses on decision-critical dependencies. By organizing anchors together with their causal links, ASM forms a structured interaction memory rather than a flat list of events. This allows the agent to retrieve only causally relevant anchors instead of processing the entire interaction history. During action prediction, the agent can retrieve anchors that are causally relevant to the current subgoal or context, enabling more targeted use of historical information.

Robustness to Model Output Style. Different models may exhibit varying abilities in structured text generation. To mitigate this effect, we apply three normalization steps: (1) unified prompting across all models, (2) strict schema enforcement for generated outputs, and (3) an automatic retry mechanism for invalid outputs. These steps ensure that the anchor generation process primarily reflects the model’s interpretation of the interaction state rather than stylistic differences in output formatting.

Why Improvements Are Not Due to Anchor Writing Ability. Although anchors are generated by the evaluated models themselves, the improvements observed with ASM cannot be attributed solely to differences in anchor writing ability. First, all history representations (raw history, summary, and anchors) originate from the same interaction trajectory. Second, the same model generates both summaries and anchors using the same prompting framework. Third, consistent improvements are observed across both open-source and closed-source models, suggesting that the gains primarily arise from the structured memory representation introduced by ASM rather than model-specific generation behavior.

###### B.3. Fairness of History Representation

A potential concern when comparing different history representations is whether one representation implicitly provides stronger supervision or more information to the agent. To ensure a fair comparison between Raw History, Summary, and ASM, we enforce the following principles.

Same Information Source. All history representations are derived from the same interaction trajectory. Specifically, the underlying information consists of the sequence of past UI observations and executed actions. Neither summaries nor anchors introduce additional external annotations or ground-truth signals. Instead, they are automatically generated by the agent itself based solely on the observed interaction history.

No Additional Evidence Signals. ASM does not introduce additional perceptual evidence beyond what is already available in the raw interaction history. The evidence fields in ASM simply reference observations present in the trajectory (e.g., screenshots, UI elements, or extracted values) rather than providing new information. Consequently, ASM reorganizes existing observations into a structured representation instead of injecting extra supervision. Both summaries and ASM serve as compact abstractions of the same trajectory and are substantially shorter than the raw interaction history.

Consistent Observation Inputs. At each interaction step, the agent receives the same primary observation inputs regardless of the history representation strategy: the current UI state and the user instruction. The only difference lies in how past interactions are represented in the context (raw trajectory, textual summary, or structured anchors).

Representation vs. Information Strength. The goal of ASM is not to increase the amount of information available to the agent, but to reorganize previously observed information into semantically meaningful intermediate states and their causal dependencies. Therefore, the comparison isolates the effect of history representation structure rather than differences in information strength.

#### C. Experiment Details

###### C.1. Prompts for Summary and Anchored State Memory Generation

Structured prompts are used for both high-level history summarization and the generation of Anchored State Memory to support effective reasoning over long-duration tasks. These prompts define the action space, normalized coordinate system, and the core decision principles used during interaction. While the history summary prompt focuses on maintaining a compact abstraction of the current task state, the Anchored State Memory prompt is designed to identify and record semantically significant transitions—such as subgoal achievements or task dependencies—that influence long-term planning. The specific structures, field requirements, and generation guidelines for these prompts are detailed in Table 9 and Table 10.

###### C.2. Evaluation Protocol and Success Criteria

This section clarifies how task success is determined in AndroTMem-Bench and how the evaluation protocol operationalizes long-horizon task completion. The goal is to ensure reproducibility and consistent evaluation across different agent implementations. Rather than relying solely on final UI states, the benchmark evaluates whether agents correctly establish and reuse dependency-critical intermediate states throughout the interaction trajectory.

Task Success Definition. Each task in AndroTMem-Bench is annotated with a sparse sequence of state anchors representing task-relevant intermediate states and a final anchor indicating successful completion. A task is considered successful if the agent reaches the final anchor while preserving the required causal dependencies among preceding anchors. This definition reflects the nature of long-horizon GUI workflows, where later decisions often depend on intermediate results obtained several steps earlier.

Dependency Satisfaction. The causal dependencies between anchors specify how earlier task states constrain later actions. During evaluation, a dependency is considered satisfied when the downstream behavior of the agent is consistent with the intermediate state required by the task. In practice, this is determined using the semantic content of the annotated anchors together with the observed action trajectory and UI state. For example, if a later step requires reusing a previously extracted value, copied text, selected item, or identified contact, the evaluation verifies that the agent’s subsequent actions correctly reflect that earlier state. The evaluation relies on the predefined anchor annotations of each task rather than post-hoc free-form judgment.

Matching and Tolerance. Anchor contents may include categorical states, entity references, or structured values. Evaluation follows task-dependent matching rules that preserve the correctness of task-critical information. Exact semantic matching is required for discrete states such as page transitions, subgoal completion, or item selection. For anchors that contain key fields (e.g., prices, contact names, or message recipients), correctness is determined based on whether the extracted value or entity matches the annotated ground-truth state. Limited tolerance is applied only to superficial surface variations that do not alter the underlying semantic value, such as formatting differences in numerical values or equivalent textual forms referring to the same entity. Any mismatch in task-critical fields is treated as an incorrect state. This design ensures that evaluation focuses on whether the correct intermediate state is preserved, rather than on character-level formatting differences.

Handling UI Drift and Interruptions. The evaluation in AndroTMem-Bench is performed on prerecorded interaction trajectories with fixed screenshots and annotated actions. These trajectories may contain interruption-related steps, such as pop-up advertisements, permission dialogs, temporary loading screens, or other unexpected UI states that arise in realistic mobile interaction. Such cases are not treated as ignorable noise in evaluation. Instead, when an interruption introduces a task-relevant state transition or requires explicit recovery, it is evaluated as part of the trajectory semantics and may be annotated as an exception anchor or as a dependency-critical intermediate state. In these cases, the agent is required to correctly recognize and handle the interruption; otherwise, the corresponding anchor is considered unsatisfied. At the same time, the evaluation does not require exact step-by-step trajectory replication. What matters is whether the required semantic state is correctly established and whether the annotated dependency structure is preserved. Therefore, interruption handling is evaluated based on state correctness rather than low-level action identity, but it is not exempted from the success criteria.

Task-Level Metrics. Within this evaluation framework, Task Completion Rate (TCR) serves as the primary task-level metric. TCR measures the proportion of tasks in which the agent successfully reaches the final anchor while satisfying the dependency constraints induced by earlier anchors. By

incorporating intermediate state consistency into the success definition, TCR captures whether an agent can correctly preserve and reuse task-critical information across long interaction horizons rather than merely reaching the final interface state.

###### C.3. Agent Implementation and Model Behavior

Table 2 and Table 4 report relatively low absolute performance for several general-purpose multimodal models, including GPT-4o and GPT-5. Since these results may appear counterintuitive given the strong capabilities of such models in other benchmarks, we clarify the evaluation setup and discuss the main factors affecting performance.

Unified Agent Framework. All evaluated models are deployed within the same GUI-agent execution framework. At each step, the model receives the current UI observation (screenshot and UI metadata when available), the user instruction, and the chosen history representation (raw trajectory, summary, or ASM). The model is then required to produce a single structured action following the predefined action schema of AndroTMem-Bench. This design ensures that differences in performance primarily reflect the model’s ability to interpret the UI state and maintain task-relevant information across steps rather than differences in system implementation.

Action-Schema Alignment. The action space in AndroTMem-Bench is defined using a normalized coordinate system and a fixed set of action types. To ensure compatibility across models, the output format is strictly constrained using structured JSON schemas together with automatic retry mechanisms when formatting errors occur. As a result, the evaluation pipeline minimizes failures caused by output formatting mismatches or tool-interface inconsistencies.

Long-Horizon Interaction Difficulty. Despite their strong reasoning abilities, general-purpose multimodal models are not specifically optimized for long-horizon GUI interaction. Tasks in AndroTMemBench often involve 30–60 interaction steps with cross-app dependencies and intermediate state reuse. In our experiments, many failures occur when models must recall or reuse information extracted several steps earlier (e.g., previously observed values, selected entities, or copied content). These errors accumulate over long trajectories and significantly reduce overall task completion rates.

Failure Patterns Observed in Large Models. Qualitative inspection of trajectories reveals several common failure modes for large general-purpose models, including losing track of previously extracted values, misbinding entities across applications, or drifting from the intended subgoal after multiple steps. These behaviors are consistent with the memory-related failure modes discussed in Section 5.4. Importantly, these failures occur even when individual perception or reasoning steps appear correct, suggesting that the primary difficulty arises from maintaining consistent intermediate state over long interaction horizons.

Interpretation of Absolute Performance. Therefore, the relatively low absolute performance of certain large models should not be interpreted as a limitation of the models themselves, but rather as an indication of the difficulty of long-horizon GUI interaction under strict action-execution constraints. In this sense, AndroTMem-Bench exposes a gap between strong multimodal reasoning capabilities and reliable long-horizon interaction memory, which is precisely the phenomenon the benchmark is designed to diagnose and measure.

###### Table 9 | Detailed Prompt Structures for History Prompt Details Mobile Assistant.

- I. Action Space

Allowed actions: [{"label":"Tap","value":"tap"},{"label":"Input Text","value":"text"}, {"label":"Need Feedback","value":"need_feedback"},{"label":"Long

Press","value":"long_press"},{"label":"Swipe","value":"swipe"},{"label":"Swipe (Two Points)","value":"swipe_two_points"},{"label":"Wait","value":"wait"},

{"label":"Finish","value":"FINISH"},{"label":"Open App","value":"open_app"}, {"label":"Capture Screen","value":"capture_screen"},{"label":"Home","value":"home"}, {"label":"Back","value":"back"}]

- II. Field Requirements

- 1. action.action: String. Must be in Action Space.
- 2. action.x,y,x_end,y_end: Use normalized coords 0-1000. (0,0)=Top-Left. (1000,1000)=Bottom-Right. For "tap"/"long_press" use (x,y). For "swipe_two_points" use start(x,y) & end(x_end,y_end). Unused=0.
- 3. action.value: String. Input text, feedback request, or app name.
- 4. action.direction: String. "up"|"down"|"left"|"right" (for "swipe").
- 5. action.distance: String. "long"|"medium"|"short" (for "swipe").

- III. Decision Principles

Visual Evidence: Only act on what you see. If loading, use "wait". Precision: Target UI element center. Use 0-1000 scale carefully. Step-by-Step: Output ONE action per response.

- IV. Summary Generation Guidelines

The summary_en field represents a compact, high-level abstraction of the current task progress and interface state. It serves as persistent context for subsequent decision steps.

When writing summary_en:

- 1. Describe the current goal-relevant state of the task, not low-level UI details.
- 2. Capture what has been accomplished so far and what remains unresolved.
- 3. Include critical constraints, user inputs, or system feedback that affect future actions.
- 4. Avoid step-by-step narration or coordinate-level descriptions.
- 5. Keep the summary concise, factual, and stable across steps unless the task state meaningfully changes. The summary should enable future steps to reason about progress without access to full action history.

- V. Output Format

Return a single JSON object:

{ "action": { "action": "...", "x": 0, "y": 0, "value": "", "x_end": 0, "y_end":

- 0 }, "summary_en": "..." }

###### Table 10 | Detailed Prompt Structures for Anchored State Memory

###### Prompt Details

Mobile Assistant.

##### I. Action Space

Allowed actions: [{"label":"Tap","value":"tap"},{"label":"Input Text","value":"text"}, {"label":"Need Feedback","value":"need_feedback"}, {"label":"LongPress","value":"long_press"},{"label":"Swipe","value":"swipe"}, {"label":"Swipe (Two Points)","value":"swipe_two_points"}, {"label":"Wait","value":"wait"},{"label":"Finish","value":"FINISH"},{"label":"Open App","value":"open_app"},{"label":"Capture Screen","value":"capture_screen"}, {"label":"Home","value":"home"},{"label":"Back","value":"back"}]

##### II. Field Requirements

- 1. action.action: String. Must be in Action Space.
- 2. action.x,y,x_end,y_end: Use normalized coords 0-1000. (0,0)=Top-Left. (1000,1000)=Bottom-Right. For "tap"/"long_press" use (x,y). For "swipe_two_points" use start(x,y) & end(x_end,y_end). Unused=0. 3. action.value: String. Input text, feedback request, or app name.

- 4. action.direction: String. "up"|"down"|"left"|"right" (for "swipe").
- 5. action.distance: String. "long"|"medium"|"short" (for "swipe").

##### III. Decision Principles

Visual Evidence: Only act on what you see. If loading, use "wait". Precision: Target UI element center. Use 0-1000 scale carefully. Step-by-Step: Output ONE action per response.

##### IV. Anchor and Causal Structure Generation Guidelines

State anchors represent semantically meaningful task events that influence long-term planning and decision-making. Each anchor summarizes a key transition, dependency, or achievement. Beyond recording important states, the model should explicitly identify causal links between anchors so that the interaction history is organized as a structured dependency graph rather than a flat list of events.

Anchor categories: [subgoal] Achievement of an intermediate objective. [state_change] Entry into a new screen, mode, or functional state. [dependency] Completion of a prerequisite required for future steps. [exception] Errors, failures, or unexpected states requiring handling. [context_info] Important parameters, settings, or user-provided information. [finish] Final goal successfully completed.

Causal link types: [prerequisite] A previous anchor must hold before the current one can occur. [enables] A previous anchor creates the condition for a future action or subgoal. [result_of] The current anchor is the direct result of a previous anchor or action outcome. [blocks] An exception or state prevents progress until resolved.

When generating anchors and links:

- 1. content_en should be a concise, category-tagged semantic statement describing the current anchor.
- 2. description_en should explain why this anchor matters for subsequent reasoning and execution.
- 3. If the current anchor has a direct causal dependency on a previous anchor, generate a causal link identifying the source anchor and relation type.
- 4. Only create causal links for decision-critical dependencies, not for trivial temporal succession.
- 5. Only generate anchors for durable and task-relevant events; avoid trivial UI transitions.
- 6. Do not repeat previous anchors unless the task state fundamentally changes.
- 7. Preserve both key intermediate states and the causal structure connecting them.

##### V. Output Format

###### Prompt Details (continued)

Return a single JSON object:

{ "action": { "action": "...", "x": 0, "y": 0, "value": "", "x_end": 0, "y_end": 0 }, "content_en": "...", "description_en": "..." ,"causal_link": { "source": "...", "relation": "..." }}

|[Figure 101]<br><br>1|
|---|

|[Figure 102]<br><br>2|
|---|

|[Figure 103]<br><br>INPUT<br><br>3|
|---|

|[Figure 104]<br><br>4|
|---|

|[Figure 105]<br><br>5|
|---|

|[Figure 106]<br><br>6|
|---|

Open(JD)

Click(690, 302)

Type("Apple AirPods Pro 2")

Click(938, 350)

Wait

Click(664, 1664)

|[Figure 107]<br><br>7|
|---|

|[Figure 108]<br><br>8|
|---|

|[Figure 109]<br><br>INPUT<br><br>9|
|---|

|[Figure 110]<br><br>10|
|---|

|[Figure 111]<br><br>11|
|---|

|[Figure 112]<br><br>12|
|---|

Open(Taobao)

Click(646, 295)

Type("Apple AirPods Pro 2")

Click(963, 324)

Click(711, 1821)

Open(JD)

###### Figure 7 | A complete trajectory example from AndroTMem.

|[Figure 113]<br><br>13|
|---|

|[Figure 114]<br><br>14|
|---|

|[Figure 115]<br><br>15|
|---|

|[Figure 116]<br><br>16|
|---|

|[Figure 117]<br><br>17|
|---|

|[Figure 118]<br><br>INPUT<br><br>18|
|---|

Type("cvc2233")

Click(558, 2225)

Click(544, 2196)

Click(905, 197)

Click(146, 1773)

Click(558, 317)

|[Figure 119]<br><br>19|
|---|

|[Figure 120]<br><br>20|
|---|

|[Figure 121]<br><br>21|
|---|

Click(296, 379)

Click(730, 2079)

Finish

###### Figure 8 | A complete trajectory example from AndroTMem.

