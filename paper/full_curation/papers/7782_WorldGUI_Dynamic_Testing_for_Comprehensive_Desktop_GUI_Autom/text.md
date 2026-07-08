## WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point

Henry Hengyuan Zhao1 Kaiming Yang1 Wendi Yu1 Difei Gao1 Mike Zheng Shou1†

# arXiv:2502.08047v5[cs.AI]23May2026

##### Abstract

Recent progress in GUI agents has substantially improved visual grounding, yet robust planning remains challenging, particularly when the environment deviates from a canonical initial state. In real applications, users often invoke assistance mid-workflow, where software may be partially configured, steps may have been executed in different orders, or the interface may differ from its default setup. Such task-state variability is pervasive but insufficiently evaluated in existing GUI benchmarks. To address this gap, we introduce WorldGUI, a benchmark covering ten widely used desktop and web applications with tasks instantiated under diverse, systematically constructed initial states. These variations capture realistic human–computer interaction settings and enable diagnostic evaluation of an agent’s ability to recover, adapt plans, and handle non-default contexts. We further present WorldGUI-Agent, a simple and model-agnostic framework that organizes planning and execution around three critique stages, improving reliability in dynamic environments. Experiments demonstrate that state-of-the-art GUI agents exhibit substantial performance degradation under non-default initial conditions, revealing limited robustness and fragile planning behaviors. Our benchmark and framework provide a foundation for developing more adaptable and reliable GUI agents. The code and data are available at https://github.com/showlab/WorldGUI.

##### 1. Introduction

Graphical User Interface (GUI) automation has emerged as a promising direction for productivity-oriented AI systems. With the rapid advancement of Multimodal Large

1Show Lab, National University of Singapore. Correspondence to: Mike Zheng Shou <mike.zheng.shou@gmail.com>. Preprint. May 26, 2026.

Language Models (MLLMs) such as GPT-5 (Singh et al., 2025) and Claude-4-Sonnet (Anthropic, 2025), GUI agents have demonstrated the potential to assist users in software usage, file management, office design, coding, and web browsing, reducing repetitive work and improving workflow efficiency.

Unlike traditional computer vision tasks such as image recognition (He et al., 2016) or visual question answering (Antol et al., 2015; Goyal et al., 2017), GUI automation operates in a highly dynamic environment where the interface state evolves as a result of user actions, system configurations, or partial task progress. However, existing online GUI benchmarks, including WebArena (Zhou et al.), WebVoyager (He et al., 2024), OSWorld (Xie et al., 2024), AssistGUI (Gao et al., 2024), and WindowsAgentArena (Bonatti et al., 2024), primarily evaluate agents from a single, canonical initial state and measure success only at the end of the trajectory. Such static evaluation overlooks several common situations in real-world usage: (1) software interfaces are often not in their default configurations, (2) users frequently request assistance from intermediate states of partially completed tasks, and (3) agents with the same low success rate (e.g., 20%) may differ substantially in their robustness, planning stability, and self-correction behavior differences that cannot be revealed under a single start-state protocol.

In parallel, robustness-oriented GUI benchmarks such as GUI-Robust (Yang et al., 2025) examine how agents respond to interface-level perturbations, including layout shifts, appearance variations, and transient pop-ups. These benchmarks vary the visual presentation while keeping the task goal and initial state fixed, providing valuable insights into perception robustness. In contrast, WorldGUI does not alter UI skin, resolution, or layout; instead, it targets taskprogress–level variation by modifying the executed steps themselves (Add-, Trim-, and Adjust-step). This form of task-state variability, such as missing, redundant, or reordered steps, poses challenges fundamentally different from interface perturbations, making the two robustness dimensions complementary and largely orthogonal.

To complement existing benchmarks, we focus on robustness to intermediate starting states, a ubiquitous but under-explored aspect of GUI interactions. We introduce

Table 1. Comparison with other online GUI benchmarks. WorldGUI is a unique benchmark that embraces diverse initial states and better reflects the authentic interactions in GUI scenarios. Env?: Indicates whether an environment is required to be deployed.

Task State variability?

Benchmark Software Task Platform Env? Inst. Video? GT Plan

WebArena (Zhou et al.) 6 812 Web Yes ✗ ✗ ✗ VisualWebArena (Koh et al., 2024) 3 910 Web Yes ✗ ✗ ✗ WebVoyager (He et al., 2024) 15 643 Web Yes ✗ ✗ ✗ AutoDroid (Wen et al., 2024) 13 158 Android OS Yes ✗ ✗ ✗ AndroidWorld (Rawles et al., 2024) 20 116 Android OS Yes ✗ ✗ ✗ AgentStudio (Zheng et al., 2025) 9 205 Desktop + Web Yes ✗ ✗ ✗ Mobile-Eval (Wang et al., 2024) 10 30 Android OS Yes ✗ ✗ ✗ APPAgent (Zhang et al., 2023) 10 50 Android OS Yes ✗ ✗ ✗ OSWorld (Xie et al., 2024) 10 369 Desktop Yes ✗ ✗ ✗

AssistGUI (Gao et al., 2024) 9 100 Windows No ✓ ✗ ✗ WindowAgentArena (Bonatti et al., 2024) 11 154 Windows Yes ✗ ✗ ✗ WorldGUI 10 611 Win. + Web No ✓ ✓ ✓

WorldGUI, a benchmark that systematically constructs diverse initial states for each task by applying controlled preactions. These pre-actions modify task progress itself (e.g., steps partially completed, over-executed, or executed in a different order), allowing tasks with the same goal to begin from multiple realistic contexts. This design mirrors how real users summon assistance in the middle of workflows and enables diagnostic evaluation of an agent’s ability to recover, adapt plans, and handle contextual variability. WorldGUI covers 10 widely used desktop applications and provides 611 tasks, each paired with a user query, instructional video, and project file. To construct meaningful state variations, trained annotators first demonstrate ground-truth (GT) plans, after which we generate augmented initial states via pre-actions.

In addition, we propose WorldGUI-Agent, a simple and general GUI agent framework grounded in critical-thinking principles, an aspect that has received relatively limited attention in prior GUI agents (Hong et al., 2024; Cheng et al., 2024; Lai et al., 2024; Agashe et al., 2025a; Wu et al., 2024). Dynamic GUI environments often deviate from expected states due to user behavior or system configurations, making it essential for agents to detect mismatches and adapt their plans. Through an analysis of real-world GUI scenarios, we distill three design principles that we argue are fundamental for GUI automation: (1) Post-Planning Critique, (2) PreAction Validation, and (3) Post-Action Evaluation. These components enable WorldGUI-Agent to refine plans, verify step prerequisites, and evaluate action outcomes, thereby improving reliability in dynamic settings.

Experiments reveal three key observations: (1) a substantial gap remains between humans and current agents, with models such as UI-Tars and GPT-5.1 still struggling on our curated desktop GUI tasks; (2) performance degrades sharply under non-default initial states, highlighting limited robustness and fragile planning capabilities; and (3) WorldGUIAgent significantly improves robustness and overall task success across WorldGUI and WindowsAgentArena, even

when built upon weaker base models and without prompt tuning, demonstrating the importance of multi-level critique for dynamic GUI automation (see Section 5).

##### 2. Related Work

###### 2.1. GUI Benchmarks

GUI benchmarks provide the foundation for evaluating computer-use agents across web, desktop, and mobile environments. Early interactive benchmarks such as WebShop (Yao et al., 2022), WebArena (Zhou et al.), and WebVoyager (He et al., 2024) center on browser tasks with long-horizon goals. OSWorld (Xie et al., 2024) and mobile benchmarks, including MobileAgent (Wang et al., 2024) and AppAgent (Zhang et al., 2023) broaden coverage to multi-application desktop and mobile ecosystems. AssistGUI (Gao et al., 2024) and WindowsAgentArena (Bonatti et al., 2024) provide online execution environments for Windows-based tasks.

Recent benchmarks emphasize scalability and online evaluation. UIExplorer (Nica et al., 2025) offers a large-scale environment for GUI exploration in both structured-DOM and pure screen modes. GUI-World (Chen et al., 2025) provides multi-application desktop tasks with online execution and standardized initial states. Robustness-oriented suites such as GUI-Robust (Yang et al., 2025) examine agents under interface-level perturbations (e.g., layout shifts, visual noise, pop-ups), focusing on perception and action robustness. While these benchmarks significantly advance GUI automation, they overwhelmingly assume a single canonical initial state for each task. They do not explicitly evaluate task-state variability, such as partially completed workflows, missing or redundant steps, or alternative intermediate contexts common in real user workflows. Our benchmark, WorldGUI, complements existing online environments by systematically constructing diverse initial states for the same task goal via controlled pre-actions, enabling diagnostic evaluation of robustness, recoverability, and planning stability in

[Figure 1]

Initial State

###### User Query Final State

###### WorldGUI Benchmark

###### Interactions

Pre-Actions

[Figure 2]

[Figure 3]

[Figure 4]

Go to the pexels website and download a photo about sky.

Software

None

Interface

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Instructional Video

Observe Act

[Figure 16]

Click(933, 78) …

Agent (e.g., 4o)

Evaluation

Platform

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

write('pexels.co m', 0.1)...

[Figure 25]

Successful Rate

PyAutoGUI

Figure 1. WorldGUI. Left: WorldGUI creates pre-actions for each meta task, leading to different initial states. It successfully reflects the real-world human-computer interaction process. Right: components in WorldGUI.

##### 3. WorldGUI Benchmark

dynamic GUI environments.

Following prior work (Xie et al., 2024; Zhou et al.), we model GUI automation as a partially observable Markov decision process (POMDP). A complete specification of the POMDP components and the execution environment is provided in Appendix A.

###### 2.2. GUI Agents

Recent progress in GUI agents spans perception, grounding, planning, and full computer-use control. CogAgent (Hong et al., 2024) improves GUI perception, while SeeClick (Cheng et al., 2024) and SeeAct (Zheng et al.,

- 2024) focus on grounding UI elements for more reliable action prediction. MobileAgent (Wang et al., 2024) and AppAgent (Zhang et al., 2023) extend these capabilities to mobile environments. On desktop and OS-level tasks, AssistGUI (Gao et al., 2024) adopts a Plan-Act pipeline with a single post-action checker. Agent-S (Agashe et al.,
- 2025a) introduces experience-augmented hierarchical planning with a Manager-Worker architecture, a self-evaluator, and an Agent-Computer Interface (ACI). Agent-S2 (Agashe et al., 2025b) further incorporates specialist grounding (Mixture-of-Grounding) and Proactive Hierarchical Planning, achieving strong performance across OSWorld, WindowsAgentArena, and AndroidWorld. Beyond inferencetime frameworks, several works train GUI agents or their components. Aguvis (Xu et al., 2024) and UGround (Gou et al., 2025) train pure-vision or universal grounding models at scale. AgentTrek (Xu et al., 2025) synthesizes multi-step trajectories from web tutorials for supervised training, and SE-GUI (Yuan et al., 2025) applies reinforcement learning to strengthen grounding robustness. While these approaches enhance perception, grounding, or hierarchical planning, they typically employ only a single verification stage (e.g., self-evaluation or post-action checking). In contrast, our framework WorldGUI-Agent focuses on structuring the verification process itself for dynamic GUI environments. WorldGUI-Agent introduces three complementary critique stages: Post-Planning Critique, Pre-Action Validation, and Post-Action Evaluation, enabling agents to detect state mismatches, assess step readiness, and repair execution failures. This multi-level critique design is orthogonal to hierarchical planners and training-based pipelines, and can be combined with them without additional model training. A component-wise comparison is provided in Table 3.

###### 3.1. Data Source

WorldGUI consists of a broad spectrum of widely-used desktop applications, which can be categorized into five main groups: (i) Office, includes PowerPoint, Word, Excel, and Adobe Acrobat; (ii) Windows Usage, includes System Settings and File Management; (iii) Web Usage, includes the configuration of Youtube and website operations; (iv) Coding, focus on the customization, configuration and editing of Visual Studio Code (VSCode); (v) Media, operating VLC player for video editing and creation.

###### 3.2. Pipeline of Data Construction

We engaged four annotators and developed the necessary scripts to structure and format the data. To reduce annotatorstyle variance, all annotators followed a standardized annotation protocol. Additionally, to facilitate ground truth (GT) plan generation and pre-action generation, we implemented simple agent-based methods to collect the relevant data. The overall data construction pipeline comprises six steps, as detailed below.

Raw Video Collection. We collect raw videos from the YouTube website as there are a lot of high-quality tutorials for desktop applications with high views. For each software, we ask the annotators to watch the videos first and download them via the diversity of software usage.

Instruction Video Preparation. After obtaining the raw videos, we write the script codes to cut the lengthy and noisy videos into the sub-clips (30 seconds to 3 minutes) that serve as the instructional video.

User Query Generation. After obtaining the instructional videos, annotators are asked to manually write user queries corresponding to each video. For example, a user query for

Table 2. Task category, task activities, and project file of the desktop applications in WorldGUI.

Category Applications All Task Avg. Length Task Activities Project File Type

Office PowerPoint 64 7.1 Change the content style and layout; Design new effects project.pptx Office Word 63 4.1 Formatting the content style and layout project.docx Office Excel 70 5.1 Table formatting; Data management and processing project.xlsx Office Adobe Acrobat 66 5.1 Automatic add electric signature; Document management project.pdf Coding VSCode 56 4.4 Code editing; Software configuration vscode.exe Windows Usage Settings 69 7.1 Advanced personalized and safety settings; ms-Settings Windows Usage File Explorer 44 5.1 File management: Add, delete, rename, and move files explorer.exe Web Usage Web Browser 59 7.8 Web operation web browser + URL Web Usage Youtube (Online) 61 4.8 Video and account configurations web browser + URL Media VLC Player 59 10.8 Video editing and creation project.mp4 Total (Average) – 611 6.0

Average Meta-Task Length per Software

11% 10%PowerPoint

10

AverageMeta-TaskLength

Acrobat

8

Word 10%

6

Excel 11%

Office43%

4

Youtube 10% WebStock

Web Browsing 20%

Coding 9%

VSCode 9%

2

s

Media

w

%

- n

d

- o

10 %

8

10%

10%VLCPlayer

1

Wi

0

AdobeAcrobat ExcelPowerPoint WordVSCodeFileExplorerSettingsWebBrowserYoutubeVLCPlayer

File

Setting 11%

M a n a g e

7 %

m e

Software

nt

Figure 2. The left shows software taxonomy in WorldGUI. The right shows the distribution of task length.

a task involving File Explorer might be: “Please compress the project.mp4 into an MPEG-4 file optimized in 1080p.”

Project File Preparation. Following the AssistGUI (Gao et al., 2024), we create the project file for each task to ensure reproducibility without relying on resource-intensive virtual machines (Xie et al., 2024) or Docker environments (Bonatti et al., 2024). This approach guarantees that the testing process begins from a consistent state. When combined with pre-actions, it enables augmentation of the same task with various initial states.

GT Plan Generation. Given a user query q and instructional video v, we automatically generate an initial draft plan using a GPT-4o–based agent. Annotators then watch the video and execute the task while reviewing the draft plan, correcting any inaccurate or missing steps to obtain the final GT plan. GT plans are used solely for constructing pre-actions and are never provided to or used by any evaluated agent.

Pre-Actions Generation. To vary the task, we propose introducing pre-actions before the task begins. These preactions are created by annotators and involve corresponding scripts and agents. They are written in Python code, for example: from pyautogui import click, rightClick\n rightClick(800,400). The preactions primarily serve two purposes: 1) Simulating Intermediate Task States: Pre-actions can complete specific steps of a task, creating a starting point from an intermediate state. This approach addresses scenarios where users

may invoke GUI assistant at any time. For example, if the task involves opening a dropdown menu, the pre-action may pre-open the menu. If the agent fails to recognize this precondition and follows its plan to click the menu again, it might inadvertently close the menu, causing task failure. 2) Introducing Diverse Initial Context States: Pre-actions can also introduce variations in the initial state, such as opening random tabs or settings. This ensures that the starting state is unconventional, challenging the agent to adapt by modifying its plan or adding necessary new steps. See example in Figure 5. Each pre-action was manually verified to ensure the task remained solvable.

###### 3.3. Data Statistics

Overall. We present the detailed task taxonomy and lengths about WorldGUI in Figure 2 and Table 2. A total of 111 meta tasks were collected from these applications, with each task being augmented 5 times based on the task’s functionality, resulting in 500 augmented tasks. In total, WorldGUI comprises 611 tasks, and every task has almost 6 variation instances, which is capable of reflecting the realworld interactions of the GUI environment. All data are available for reference.1

Augmentation tasks type analysis. As we summarize, the real GUI scenarios include: (1) The software interface is not in its default state. (2) The human-computer interactions

1https://huggingface.co/datasets/ anonymousABC/WorldGUI-Bench

Augmentation types by Software

[Figure 26]

###### User Query “Set the document to Two Page View from the View menu.”

Aug: Add-step Aug: Trim-step Aug: Adjust-step

30

[Figure 27]

[Figure 28]

[Figure 29]

###### GT Plans Initial State Pre-Action

25

Task 1: Change the page display to two-page view in Adobe Acrobat

[Figure 30]

20

TaskCount

- Subtask 1: Click 'View' in the top menu.
- Subtask 2: Hover over 'Page Display' from the dropdown menu.
- Subtask 3: Click 'Two Page View' to display the document in a two-page layout.

[Figure 31]

Original Data

15

10

[Figure 32]

Task 1: Change the page display to two-page view in Adobe Acrobat

5

- Subtask 1: Hover over 'Page Display' from the dropdown menu.
- Subtask 2: Click 'Two Page View' to display the document in a two-page layout.

[Figure 33]

0

AdobeAcrobat Excel PowerPoint Word VSCodeFileExplorer SettingsWebBrowser YouTube VLCPlayer

Augmented Data

Software

Task 1: Change the page display to two-page view in Adobe Acrobat

[Figure 34]

Figure 3. The distribution of different augmentation types.

- Subtask 1: Ignore the Tool Page and Click 'View' in the top menu.
- Subtask 2: Hover over 'Page Display' from the dropdown menu.
- Subtask 3: Click 'Two Page View' to display the document in a two-page layout.

[Figure 35]

Task difficulty of Software

Simple

40

Medium

Hard

35

Figure 5. An example of augmenting one GUI task with manually aug the initial state and then using the execution scripts and corresponding agents to obtain the pre-action for each augmented case.

30

TaskCount

25

20

15

task difficulty of our created data is diverse across different applications.

10

5

0

AdobeAcrobat Excel PowerPoint Word VSCodeFileExplorer SettingsWebBrowser YouTube VLCPlayer

- 4. WorldGUI-Agent Table 3. Comparison with closely related GUI agents.

Method Post-Plan Pre-Action Post-Action

Mobile-Agent / V2 ✗ ✗ ✓ AssistGUI ✗ ✗ ✓ Agent-S / S2 ✗ ✗ ✓ Mobile-Agent-E ✗ ✗ ✓ WorldGUI-Agent (ours) ✓ ✓ ✓

To study robustness under the dynamically perturbed initial states of WorldGUI, we adopt a lightweight, non-training agent framework. As summarized in Table 3, most existing agents rely on a single post-action evaluation module. Rather than proposing a new algorithm, our aim is simply to structure the standard perception–planning–execution loop into clearer verification stages.WorldGUI-Agent introduces three small modules at different abstraction levels: PlannerCritic refines the draft plan; Step-Check uses the screenshot and metadata to decide whether a plan step should be executed, skipped, or rewritten; Actor-Critic validates each action using the current observation and recent action history. Compared to prior self-reflection approaches (e.g., Reflexion, CRITIC), which apply a single global critique, our checks are lightweight and distributed across plan, step, and action levels to better handle task-progress variation. Full module specifications and I/O formats are included in Appendix B.

- 5. Experimental Results

Software

Figure 4. The distribution of different task difficulty.

may start from the intermediate state of a specific task. Our augmentations lie in two main groups: (1) simulating the intermediate states and (2) introducing diverse initial states. We divide the two groups into three types: Add-step, Trimstep, and Adjust-step. For Add-step, it represents various unrelated state augmentations to simulate the scenario that we may start the task in another unrelated task or interfaces, the agent should replan the task to add necessary steps. For Trim-step, it represents that we finish several steps of a long task and make the task in an intermediate state. For Adjust-step, it is usually a small modification of the existing state, such as changing the interface by clicking another Tab or clicking a button to open an unrelated dropdown menu. Most of the time, it would not require new steps to return to the target task procedure. This augmentation may mislead the agent in state understanding, making them jump or miss the key steps. As shown in Figure 3, the manually created augmentations mainly belong to the add-step. Adjust-step could be the second-largest part. Due to the low complexity of the interfaces of File Explorer, we cannot create many augmentations for adjust-step.

Task difficulty analysis. Figure 4 shows the distribution of the task difficulty across desktop applications. We annotated the task difficulty level based the subjective software usage experience. The results indicate that the tasks in Adobe Acrobat and VLC player are more challenging. The tasks in Excel, PowerPoint, and Word are more at the medium and simple levels. By considering the Success Rate and task length on these tasks, one can know that the tasks are easy for humans, but hard for current GUI agents. Overall, the

Implementation Details. We use PyAutoGUI2 to extract basic metadata from GUI screenshots, including bounding boxes of buttons, icons, and text regions. We then imple-

2https://pyautogui.readthedocs.io

Table 4. Success rate (%) of different agents on WorldGUI. Human∗ denotes the average performance of four expert participants who have watched the instructional video only once, similar to the model. Meta represents the meta task, while Aug. represents the augmented task.

Office Win. Usage Web Coding Media

Method

Overall

Meta Aug. Meta Aug. Meta Aug. Meta Aug. Meta Aug.

Plan-Act w/ Gemini-2.0 8.9 3.2 8.3 3.4 28.6 16.2 18.2 2.2 10.0 2.0 6.9 Plan-Act w/ GPT-4o 13.3 10.1 8.3 2.3 23.8 11.1 9.1 2.2 10.0 2.0 8.5 AssistGUI w/ GPT-4o 26.7 16.1 29.2 7.9 33.3 20.2 27.3 11.1 10.0 8.2 16.5 CCU w/ Claude-3.5-Sonnet 28.9 19.3 29.2 14.6 71.4 32.3 54.6 22.2 30.0 6.1 23.6 UI-TARS-1.5 28.9 16.1 12.5 2.2 28.6 9.1 36.7 6.7 0.0 0.0 12.3 Agent S2 33.3 16.5 70.8 59.6 52.4 45.5 45.5 37.8 20.0 16.3 34.2

###### WorldGUI-Agent (Ours)

w / Gemini-2.0 31.1 17.0 20.8 9.0 38.1 29.3 36.4 11.1 20.0 10.2 19.1 w / GPT-4o 42.2 24.3 41.7 11.2 47.6 35.4 45.5 15.6 40.0 12.2 26.0 w / Claude-3.5-Sonnet 57.8 32.6 50.0 19.1 76.2 46.5 54.6 26.7 50.0 18.4 36.0 w / GPT-5.1 64.4 37.6 62.5 40.4 85.7 53.5 63.6 46.6 60.0 26.5 45.8

Human∗ 88.9 83.5 100.0 89.9 95.2 80.8 81.8 77.8 90.0 85.7 85.3

Table 5. Success rate (%) of our WorldGUI-Agent (w/ GPT-4o) with the ablation of different critical modules.

Office Win. Usage Web Coding Media

Method

Overall

Meta Aug. Meta Aug. Meta Aug. Meta Aug. Meta Aug. Full Model 42.2 24.3 41.7 11.2 47.6 35.4 45.5 15.6 40.0 12.2 26.0

- – w/o Planner-Critic 31.1 17.0 20.8 9.0 38.1 25.3 36.4 11.1 20.0 10.2 18.5

- – w/o Step-Check 31.1 19.3 20.8 9.0 33.3 28.3 45.5 13.3 20.0 8.2 19.8

- – w/o Actor-Critic 15.6 7.8 4.2 3.4 28.6 17.2 0.0 8.9 10.0 6.1 9.7

Table 6. Success rate (%) of our WorldGUI-Agent with the ablation of Instructional Video.

Method PPT Word Excel Acrobat VSCode Overall

Full Model 45.5 36.4 50.0 36.4 45.5 42.9 w/o Inst. Video 45.5 27.3 25.3 18.2 27.3 28.6

Table 7. Running time comparison.

Method Executed Steps Time (seconds)

Agent-S 10 131.98 Agent-S2 9 108.65 WorldGUI-Agent 24 129.55

ment software-specific GUI parsers3 that further refine element grounding by incorporating OCR results obtained through the Google OCR API. Details of our baseline agent WorldGUI-Agent are provided in Appendix B. All experiments are conducted under a fixed screenshot resolution of 1920×1080 and a display scale of 125%. For WorldGUIAgent, we limit the Actor–Critic correction attempts to three steps to control interaction cost. The total number of trials allowed per task is set to 4N + 1, where N is chosen empirically based on the task category.

Evaluation. Given that our WorldGUI includes 611 GUI tasks, we engaged four participants with strong coding and software backgrounds to test all tasks and document their evaluation results. Metric. Following the previous works, we use Success Rate (SR) as the metric.

Baselines. We implement the baseline approach called PlanAct with different MLLMs as the base model. It focuses on investigating the basic capabilities of task planning and

3https://anonymous.4open.science/r/ WorldGUI-7A5C/agent/gui_parser/

Table 8. Task counts and success rates (SR, %) across different augmentation types for WorldGUI-Agent (GPT-5.1).

Augmentation Type #Tasks SR (%) #Success

Add-step 255 34.1 87 Trim-step 119 52.1 62 Adjust-step 130 43.1 56

Total 500 41.0 205

action prediction. Additionally, we compare our WorldGUIAgent with two agentic frameworks and two SOTA GUI models: AssistGUI (Gao et al., 2024), Agent-S2 (Agashe et al., 2025b), Computer Use (Claud-3.5-Sonnet) (Anthropic, 2024), and UI-TARS-1.5 (Qin et al., 2025). AssistGUI and Agent-S2 are two prominent agentic frameworks designed for Desktop GUI Automation, which can plan the task and then execute the task step by step by following the query. We increase the base model to GPT-4o for AssistGUI and Claude-Sonnet-4 of Agent-S2 for fair performance. Claude Computer Use (CCU) is the leading proprietary model specially designed for computer use. We use the open-source implementation OOTB (Hu et al., 2024) as the codebase and then add the subtitle of instructional videos into the input prompt for a fair comparison. We also implement our WorldGUI-Agent with four different MLLMs to illustrate the effectiveness of our proposed universal agent framework.

###### 5.1. Main Results on WorldGUI

Overall. Table 4 reports the success rates (SR) of different agents and human experts on our WorldGUI benchmark, broken down by task type (Meta vs. Aug.) across five categories: Office, Win. Usage, Web, Coding, and Media.

From these results we draw the following main conclusions. A large gap remains between agents and humans. The best-performing agent (WorldGUI-Agent with GPT-5.1) achieves an overall SR of only 45.8%, which is less than half of the 85.3% attained by human experts. This stark contrast underscores the difficulty of our tasks and the need for further advances in desktop GUI automation.

Agents generalize poorly to augmented tasks. Across all methods, performance on Augmentation tasks (which introduce interface or context variations) is substantially lower than on their corresponding Meta tasks. For example, Claude-3.5-Sonnet in the Win. Usage category attains 50.0% on Meta tasks but drops to just 19.1% on Aug. tasks. This highlights the importance to evaluate the agents on various initial conditions.

Desktop GUI tasks pose a greater challenge than web tasks. Every agent scores higher on Web tasks than on desktop application tasks. WorldGUI-Agent with GPT-5.1, for instance, jumps from 85.7% on Web Meta to only 62.5% on Win. Usage Meta, and the gap widens on their Augmentation counterparts. Thus, desktop GUI tasks remain challenges than widely-studied web GUI tasks (Cheng et al., 2024; He et al., 2024).

WorldGUI-Agent consistently outperforms a naive PlanAct counterpart. By incorporating our three critical modules into the planning and execution loop, WorldGUI-Agent substantially improves success rates over the basic PlanAct counterpart. Relative to Plan-Act, WorldGUI-Agent raises overall SR by +12.2% with Gemini-2.0, +17.5% with GPT-4o, and +12.4% with Claude-3.5-Sonnet, demonstrating the effectiveness of our critical modules proposed for WorldGUI-Agent.

Specialized GUI models transfer poorly to WorldGUI. Despite achieving strong results on several existing GUI benchmarks, specialized models do not perform well in our setting. For example, UI-TARS-1.5 attains only 12.3% overall SR on WorldGUI, which is notably lower than generalpurpose MLLMs when used within our framework (e.g., 36.0% for WorldGUI-Agent with Claude-3.5-Sonnet and 45.8% with GPT-5.1), and even below the CCU baseline built on Claude-3.5-Sonnet (23.6%). Similarly, Agent-S2 reaches 34.2% overall SR, still far from human performance and trailing behind WorldGUI-Agent with GPT-5.1. These results indicate that training pipelines and architectures tuned for single-start or less diverse benchmarks do not automatically transfer to the multi-start, dynamically perturbed settings of WorldGUI. In our evaluation, specialized GUI models exhibit strong grounding but substantially weaker planning, which leads to low success rates even when provided with additional signals such as instructional-video subtitles.

###### 5.2. Ablation Study

Impact of different critical modules. Table 5 reports an ablation study of WorldGUI-Agent across five application categories. The full model achieves an overall success rate (SR) of 26.0%. Removing any core component leads to notable degradation. Excluding the Planner-Critic reduces SR to 18.5% (–7.5%), highlighting its role in plan refinement. Removing Step-Check lowers SR to 19.8% (–6.2%), mainly affecting multi-step interaction scenarios, suggesting its effectiveness in intermediate error correction. The most severe impact comes from removing the Actor-Critic, which collapses SR to 9.7% (–16.3%), nearly eliminating performance on Coding and Win. Usage tasks, underscoring the necessity of action-level reward feedback. Overall, the three modules provide complementary benefits, enabling robust agent performance.

Impact of Instructional Video. In Table 6, we study the impact of removing the instructional video (subtitles) by modifying the prompt to include only the user query for generating the initial plan. In the Excel applications, we observe a significant performance decline, as their tasks are complex and difficult, and rely more heavily on additional domain knowledge for successful planning. In contrast, the agents performs relatively well on Win. Usage tasks, such as Settings and File Management, are where it has more inherent pretrained prior. These findings underscore the necessity of instructional videos for complex tasks like visual effect design, mirroring how users learning to build a slide often rely on tutorial videos. It is note that it is a setup proposed in our benchmark; one can also use our data without instructional videos.

###### 5.3. Results by Augmentation Type

Across the 500 augmented tasks, WorldGUI-Agent (GPT5.1) solves 205 cases. As shown in Table 8, Add-step tasks are the most challenging (34.1% SR), since inserted steps create larger state mismatches. Adjust-step tasks show moderate difficulty (43.1%), reflecting the need for fine-grained manipulations. Trim-step tasks are relatively easier (52.1%), as removing redundant steps keeps the interface closer to the meta configuration. These trends highlight how different forms of task-progress variation distinctively impact agent robustness.

###### 5.4. Failure Analysis and Robustness Dimensions

To better understand the limitations of current GUI agents under the dynamic initial states of WorldGUI, we analyze representative failure modes drawn from our qualitative study (Appendix I). We observe three major robustness dimensions that frequently lead to failures.

###### (1) State-mismatch robustness. When pre-actions intro-

Table 9. Experimental results on WindowsAgentArena. The reported results are from the (Bonatti et al., 2024) and (Agashe et al., 2025b).

Method Office Web Win. System Coding Media Win. Utils Overall

Phi3-V (Bonatti et al., 2024) 0.0 6.9 8.3 0.0 6.2 0.0 3.5 GPT-4o-mini (Bonatti et al., 2024) 0.0 14.9 8.3 0.0 0.0 0.0 4.2 GPT-4o (Bonatti et al., 2024) 0.0 13.7 29.2 0.0 10.3 0.0 8.6 NAVI (Bonatti et al., 2024) 0.0 27.3 33.3 27.3 30.3 8.3 19.5 Agent S (Agashe et al., 2025a) w/ GPT-4o 0.0 13.3 45.8 29.2 19.1 22.2 18.2 Agent S2 (Agashe et al., 2025b) w/ Claude-3.7-Sonnet 7.0 16.4 54.2 62.5 28.6 33.3 29.8 WorldGUI-Agent w/ Claude-3.5-Sonnet 7.0 53.3 45.8 33.3 28.6 33.3 31.2

duce interface configurations different from the default view (e.g., an expanded settings dropdown or partially filled input fields), agents often misinterpret the visible state or fail to recognize that previous steps have already been executed. As illustrated in Figure 24 (left), once a submenu overlays other elements, the agent struggles to recover the intended target (e.g., the “System” button), leading to incorrect plan continuation or premature termination.

- (2) Fine-grained manipulation robustness. Tasks requiring precise adjustments—such as dragging sliders or manipulating bars to reach a specific value—remain challenging. Even when grounding is correct, the agent may overshoot, undershoot, or repeatedly click without achieving the required adjustment, as shown in Figure 24 (right). This reflects a broader difficulty in performing continuous or multi-step manipulations that depend on spatial granularity.
- (3) Ambiguous-visual-context robustness. In the absence of clear text cues, agents exhibit uncertainty in selecting visually similar elements. Figure 25 (left) illustrates a case where the model must choose a centered icon from a symmetric layout, yet struggles due to limited visual discrimination without textual anchors. Similarly, when multiple text elements appear close together, GUI parsers may surface misleading anchors (e.g., the label “Replace with”), causing the agent to click an irrelevant surface-level text region rather than the actual input box (Figure 25, right).

Overall, these failure modes echo our quantitative findings: while grounding on static interfaces is often strong, robustness to state changes, fine-grained manipulations, and textsparse visual contexts remains limited. These observations highlight the need for future progress in both perception and planning components, particularly under the multi-start, dynamically perturbed conditions introduced by WorldGUI.

###### 5.5. Computational Costs Analysis

We select two SoTA non-training agent frameworks Agent-S (Agashe et al., 2025a) and Agent-S2 (Agashe et al., 2025b) for comparing the computational costs. Take a Windows setting task as an example, we tested on the same desktop PC with an AMD Ryzen 7 5800H CPU. As shown in Table 7, our WorldGUI-Agent shows a competitive running time of 129.55s, as compared with Agent-S (131.98s) and Agent-S2 (108.64s). The main computational costs of

our designed modules are largely affected by calling base model. Since the main problem of desktop GUI automation is still the suboptimal performance as shown in Table 4, such computational costs are currently acceptable. There remains a clear tradeoff between performance and time costs in GUI automation, and this challenge is shared across the community.

###### 5.6. Results on WindowsAgentArena

Settings. The official WindowsAgentArena (WAA) evaluation runs inside a Linux-based VM using a pre-built image and a JSON configuration file. Since the essential logic of WAA lies in its JSON specification of the environment state and task instruction, we reconstruct the same environment on a Windows machine by following the provided configuration (e.g., opening specified tabs or launching required applications). We then use the instruction field from the JSON file as the prompt for our WorldGUI-Agent, which executes the corresponding GUI actions on Windows. After completion, we follow WAA’s evaluation protocol to determine task success. When an official evaluation script is available, we apply it directly; otherwise, we perform manual judgment based on the given instructions. We do not optimize the prompts, just apply our agent directly on its tasks. Main results. Table 9 compares WorldGUI-Agent against six agents on the WAA benchmark. WorldGUIAgent achieves a 31.2% overall SR surpass two SOTA apicalling agents, Agent-S and Agent-S2, even with a weaker base mode,l Claude-3.5-Sonnet. These results underscore the effectiveness of our designed critical modules.

##### Conclusion

We introduced WorldGUI, a benchmark that evaluates GUI agents under diverse, dynamically perturbed initial states, an aspect largely absent from existing evaluations. By constructing task-progress variations via pre-actions, WorldGUI enables systematic assessment of robustness and recovery in realistic GUI scenarios. We also presented WorldGUIAgent, a lightweight, universal framework that inserts three verification stages into the standard planning loop to improve stability in dynamic environments. Our experiments show that current agents, including strong commercial models, still struggle under non-default states, while WorldGUIAgent offers consistent gains.

##### Impact Statement

This paper introduces a benchmark and agent framework intended to advance research on dynamic GUI automation. Our goal is to improve the evaluation and reliability of GUI agents in controlled environments, which may support future progress in productivity tools, accessibility assistance, and educational software. As with many advances in machine learning and automation, there are potential risks, including misuse of automated GUI agents for large-scale or unauthorized interactions with software systems. Our benchmark does not involve personal or sensitive data and is designed strictly for research use. We encourage practitioners to incorporate appropriate safeguards, such as permission checks, sandboxing, and oversight, when deploying GUI agents in real-world applications. Overall, we believe the broader societal impacts of this work align with those commonly associated with progress in machine learning, and we do not identify specific concerns beyond these general considerations.

Cheng, K., Sun, Q., Chu, Y., Xu, F., Li, Y., Zhang, J., and Wu, Z. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X.,

- Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen,

- Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen,
- R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye,
- S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen,

- X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang,

- X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang,

- X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang,
- Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y.,

Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang,

- Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

##### References

Agashe, S., Han, J., Gan, S., Yang, J., Li, A., and Wang, X. E. Agent s: An open agentic framework that uses computers like a human. In The Thirteenth International Conference on Learning Representations, 2025a. URL https:// openreview.net/forum?id=lIVRgt4nLv.

Agashe, S., Wong, K., Tu, V., Yang, J., Li, A., and Wang, X. E. Agent s2: A compositional generalistspecialist framework for computer use agents, 2025b. URL https://arxiv.org/abs/2504.00906.

Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku, 2024. URL https://www.anthropic.com/news/ 3-5-models-and-computer-use.

Anthropic. Introducing claude 4, 2025. URL https: //www.anthropic.com/news/claude-4.

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., and Parikh, D. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pp. 2425–2433, 2015.

Bonatti, R., Zhao, D., Bonacci, F., Dupont, D., Abdali, S., Li, Y., Lu, Y., Wagle, J., Koishida, K., Bucker, A., Jang, L., and Hui, Z. Windows agent arena: Evaluating multi-modal os agents at scale, 2024. URL https:

//arxiv.org/abs/2409.08264.

Chen, D. et al. Gui-world: A dataset for gui-oriented multimodal llm-based evaluation. https://gui-world. github.io, 2025.

Gao, D., Ji, L., Bai, Z., Ouyang, M., Li, P., Mao, D., Wu, Q., Zhang, W., Wang, P., Guo, X., Wang, H., Zhou, L., and Shou, M. Z. Assistgui: Task-oriented pc graphical user interface automation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13289–13298, June 2024.

Gou, B., Wang, R., Zheng, B., Xie, Y., Chang, C., Shu, Y., Sun, H., and Su, Y. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=kxnoqaisCT. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and

Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

He, H., Yao, W., Ma, K., Yu, W., Dai, Y., Zhang, H., Lan, Z., and Yu, D. WebVoyager: Building an end-to-end web agent with large multimodal models. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6864–6890, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 371. URL https://aclanthology.org/2024.

acl-long.371/.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14281–14290, 2024.

Hu, S., Ouyang, M., Gao, D., and Shou, M. Z. The dawn of gui agent: A preliminary case study with claude 3.5 computer use, 2024. URL https://arxiv.org/ abs/2411.10323.

Koh, J. Y., Lo, R., Jang, L., Duvvur, V., Lim, M. C., Huang, P.-Y., Neubig, G., Zhou, S., Salakhutdinov, R., and Fried, D. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024.

Lai, H., Liu, X., Iong, I. L., Yao, S., Chen, Y., Shen, P., Yu, H., Zhang, H., Zhang, X., Dong, Y., and Tang, J. Autowebglm: A large language model-based web navigating agent, 2024. URL https://arxiv.org/abs/ 2404.03648.

Lin, K. Q., Li, L., Gao, D., Yang, Z., Wu, S., Bai, Z., Lei, W., Wang, L., and Shou, M. Z. Showui: One vision-language-action model for gui visual agent. arXiv preprint arXiv:2411.17465, 2024.

Nica, A. C., Kudlu Shanbhogue, A. V., Shah, H., Cambray,

- A., Berariu, T., Maystre, L., and Barber, D. Toward autonomous ui exploration: The uiexplorer benchmark. arXiv preprint arXiv:2506.17779, 2025.

Qin, Y., Ye, Y., Fang, J., Wang, H., Liang, S., Tian, S., Zhang, J., Li, J., Li, Y., Huang, S., Zhong, W., Li, K., Yang, J., Miao, Y., Lin, W., Liu, L., Jiang, X., Ma, Q., Li, J., Xiao, X., Cai, K., Li, C., Zheng, Y., Jin, C., Li, C.,

Zhou, X., Wang, M., Chen, H., Li, Z., Yang, H., Liu, H., Lin, F., Peng, T., Liu, X., and Shi, G. Ui-tars: Pioneering automated gui interaction with native agents, 2025. URL https://arxiv.org/abs/2501.12326.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Robust speech recognition via largescale weak supervision. In International conference on machine learning, pp. 28492–28518. PMLR, 2023.

Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., Toyama, D., Berry, R., Tyamagundlu, D., Lillicrap, T., and Riva, O. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024. URL https:

//arxiv.org/abs/2405.14573.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., Nathan, A., Luo, A., Helyar, A., Madry, A., Efremov, A., Spyra, A., Baker-Whitcomb, A., Beutel, A., Karpenko, A., Makelov, A., Neitz, A., Wei, A., Barr, A., Kirchmeyer, A., Ivanov, A., Christakis, A., Gillespie, A., Tam, A., Bennett, A., Wan, A., Huang, A., Sandjideh, A. M., Yang, A., Kumar, A., Saraiva, A., Vallone, A., Gheorghe, A., Garcia, A. G., Braunstein, A., Liu, A., Schmidt, A., Mereskin, A., Mishchenko, A., Applebaum, A., Rogerson, A., Rajan, A., Wei, A., Kotha, A., Srivastava, A., Agrawal, A., Vijayvergiya, A., Tyra, A., Nair,

- A., Nayak, A., Eggers, B., Ji, B., Hoover, B., Chen, B., Chen, B., Barak, B., Minaiev, B., Hao, B., Baker, B., Lightcap, B., McKinzie, B., Wang, B., Quinn, B., Fioca,
- B., Hsu, B., Yang, B., Yu, B., Zhang, B., Brenner, B., Zetino, C. R., Raymond, C., Lugaresi, C., Paz, C., Hudson, C., Whitney, C., Li, C., Chen, C., Cole, C., Voss,
- C., Ding, C., Shen, C., Huang, C., Colby, C., Hallacy, C., Koch, C., Lu, C., Kaplan, C., Kim, C., Minott-Henriques, C., Frey, C., Yu, C., Czarnecki, C., Reid, C., Wei, C., Decareaux, C., Scheau, C., Zhang, C., Forbes, C., Tang,
- D., Goldberg, D., Roberts, D., Palmie, D., Kappler, D., Levine, D., Wright, D., Leo, D., Lin, D., Robinson, D., Grabb, D., Chen, D., Lim, D., Salama, D., Bhattacharjee,

- D., Tsipras, D., Li, D., Yu, D., Strouse, D., Williams, D., Hunn, D., Bayes, E., Arbus, E., Akyurek, E., Le, E. Y., Widmann, E., Yani, E., Proehl, E., Sert, E., Cheung, E., Schwartz, E., Han, E., Jiang, E., Mitchell, E., Sigler, E., Wallace, E., Ritter, E., Kavanaugh, E., Mays, E., Nikishin,
- E., Li, F., Such, F. P., de Avila Belbute Peres, F., Raso,
- F., Bekerman, F., Tsimpourlas, F., Chantzis, F., Song, F., Zhang, F., Raila, G., McGrath, G., Briggs, G., Yang, G., Parascandolo, G., Chabot, G., Kim, G., Zhao, G., Valiant,
- G., Leclerc, G., Salman, H., Wang, H., Sheng, H., Jiang,
- H., Wang, H., Jin, H., Sikchi, H., Schmidt, H., Aspegren,

- H., Chen, H., Qiu, H., Lightman, H., Covert, I., Kivlichan,
- I., Silber, I., Sohl, I., Hammoud, I., Clavera, I., Lan, I.,

Akkaya, I., Kostrikov, I., Kofman, I., Etinger, I., Singal,

- I., Hehir, J., Huh, J., Pan, J., Wilczynski, J., Pachocki, J., Lee, J., Quinn, J., Kiros, J., Kalra, J., Samaroo, J., Wang,
- J., Wolfe, J., Chen, J., Wang, J., Harb, J., Han, J., Wang,

- J., Zhao, J., Chen, J., Yang, J., Tworek, J., Chand, J., Landon, J., Liang, J., Lin, J., Liu, J., Wang, J., Tang, J., Yin,

- J., Jang, J., Morris, J., Flynn, J., Ferstad, J., Heidecke, J., Fishbein, J., Hallman, J., Grant, J., Chien, J., Gordon, J., Park, J., Liss, J., Kraaijeveld, J., Guay, J., Mo, J., Lawson,

- J., McGrath, J., Vendrow, J., Jiao, J., Lee, J., Steele, J., Wang, J., Mao, J., Chen, K., Hayashi, K., Xiao, K., Salahi,
- K., Wu, K., Sekhri, K., Sharma, K., Singhal, K., Li, K., Nguyen, K., Gu-Lemberg, K., King, K., Liu, K., Stone,

K., Yu, K., Ying, K., Georgiev, K., Lim, K., Tirumala, K., Miller, K., Ahmad, L., Lv, L., Clare, L., Fauconnet, L., Itow, L., Yang, L., Romaniuk, L., Anise, L., Byron, L., Pathak, L., Maksin, L., Lo, L., Ho, L., Jing, L., Wu,

- L., Xiong, L., Mamitsuka, L., Yang, L., McCallum, L., Held, L., Bourgeois, L., Engstrom, L., Kuhn, L., Feuvrier, L., Zhang, L., Switzer, L., Kondraciuk, L., Kaiser, L., Joglekar, M., Singh, M., Shah, M., Stratta, M., Williams,
- M., Chen, M., Sun, M., Cayton, M., Li, M., Zhang, M., Aljubeh, M., Nichols, M., Haines, M., Schwarzer, M., Gupta, M., Shah, M., Huang, M., Dong, M., Wang, M., Glaese, M., Carroll, M., Lampe, M., Malek, M., Sharman, M., Zhang, M., Wang, M., Pokrass, M., Florian,

- M., Pavlov, M., Wang, M., Chen, M., Wang, M., Feng,

- M., Bavarian, M., Lin, M., Abdool, M., Rohaninejad, M., Soto, N., Staudacher, N., LaFontaine, N., Marwell, N., Liu, N., Preston, N., Turley, N., Ansman, N., Blades, N., Pancha, N., Mikhaylin, N., Felix, N., Handa, N., Rai, N., Keskar, N., Brown, N., Nachum, O., Boiko, O., Murk, O., Watkins, O., Gleeson, O., Mishkin, P., Lesiewicz, P., Baltescu, P., Belov, P., Zhokhov, P., Pronin, P., Guo, P., Thacker, P., Liu, Q., Yuan, Q., Liu, Q., Dias, R., Puckett,

- R., Arora, R., Mullapudi, R. T., Gaon, R., Miyara, R., Song, R., Aggarwal, R., Marsan, R., Yemiru, R., Xiong,

- R., Kshirsagar, R., Nuttall, R., Tsiupa, R., Eldan, R., Wang, R., James, R., Ziv, R., Shu, R., Nigmatullin, R., Jain, S., Talaie, S., Altman, S., Arnesen, S., Toizer, S., Toyer, S., Miserendino, S., Agarwal, S., Yoo, S., Heon, S., Ethersmith, S., Grove, S., Taylor, S., Bubeck, S., Banesiu,
- S., Amdo, S., Zhao, S., Wu, S., Santurkar, S., Zhao, S., Chaudhuri, S. R., Krishnaswamy, S., Shuaiqi, Xia, Cheng,

- S., Anadkat, S., Fishman, S. P., Tobin, S., Fu, S., Jain,

- S., Mei, S., Egoian, S., Kim, S., Golden, S., Mah, S., Lin, S., Imm, S., Sharpe, S., Yadlowsky, S., Choudhry,

- S., Eum, S., Sanjeev, S., Khan, T., Stramer, T., Wang,
- T., Xin, T., Gogineni, T., Christianson, T., Sanders, T., Patwardhan, T., Degry, T., Shadwell, T., Fu, T., Gao, T., Garipov, T., Sriskandarajah, T., Sherbakov, T., Kaftan,

- T., Hiratsuka, T., Wang, T., Song, T., Zhao, T., Peterson, T., Kharitonov, V., Chernova, V., Kosaraju, V., Kuo, V., Pong, V., Verma, V., Petrov, V., Jiang, W., Zhang,

W., Zhou, W., Xie, W., Zhan, W., McCabe, W., DePue, W., Ellsworth, W., Bain, W., Thompson, W., Chen, X., Qi, X., Xiang, X., Shi, X., Dubois, Y., Yu, Y., Khakbaz, Y., Wu, Y., Qian, Y., Lee, Y. T., Chen, Y., Zhang, Y., Xiong, Y., Tian, Y., Cha, Y., Bai, Y., Yang, Y., Yuan, Y., Li, Y., Zhang, Y., Yang, Y., Jin, Y., Jiang, Y., Wang, Y., Wang, Y., Liu, Y., Stubenvoll, Z., Dou, Z., Wu, Z., and Wang, Z. Openai gpt-5 system card, 2025. URL https://arxiv.org/abs/2601.03267.

Wang, J., Xu, H., Ye, J., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wen, H., Li, Y., Liu, G., Zhao, S., Yu, T., Li, T. J.-J., Jiang, S., Liu, Y., Zhang, Y., and Liu, Y. Autodroid: Llmpowered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, ACM MobiCom ’24, pp. 543–557, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704895. doi: 10. 1145/3636534.3649379. URL https://doi.org/ 10.1145/3636534.3649379.

Wu, Z., Wu, Z., Xu, F., Wang, Y., Sun, Q., Jia, C., Cheng, K., Ding, Z., Chen, L., Liang, P. P., and Qiao, Y. Os-atlas: A foundation action model for generalist gui agents, 2024. URL https://arxiv.org/abs/2410.23218.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., Liu, Y., Xu, Y., Zhou, S., Savarese, S., Xiong, C., Zhong, V., and Yu, T. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024.

Xu, Y., Wang, Z., Wang, J., Lu, D., Xie, T., Saha, A., Sahoo, D., Yu, T., and Xiong, C. Aguvis: Unified pure vision agents for autonomous gui interaction. 2024. URL https://arxiv.org/abs/2412.04454.

Xu, Y., Lu, D., Shen, Z., Wang, J., Wang, Z., Mao, Y., Xiong, C., and Yu, T. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=EEgYUccwsV.

Yang, J., Song, Z., Chen, J., Song, M., Zhou, S., Sun, L., Ouyang, X., Chen, C., and Wang, C. Gui-robust: A comprehensive dataset for testing gui agent robustness in

real-world anomalies. In NeurIPS Datasets and Benchmarks Track, 2025. URL https://openreview. net/forum?id=22gw3kITCd.

Yao, S., Chen, H., Yang, J., and Narasimhan, K. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in neural information processing systems, 2022.

Yuan, X., Zhang, J., Li, K., Cai, Z., Yao, L., Chen, J., Wang, E., Hou, Q., Chen, J., Jiang, P.-T., and Li, B. SEGUI: Enhancing visual grounding for GUI agents via selfevolutionary reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/ forum?id=IbzDaIDyt6.

Zhang, C., Yang, Z., Liu, J., Han, Y., Chen, X., Huang, Z., Fu, B., and Yu, G. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771,

- 2023.

Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y. Gpt4v(ision) is a generalist web agent, if grounded. In Fortyfirst International Conference on Machine Learning,

- 2024. URL https://openreview.net/forum? id=piecKJ2DlB.

Zheng, L., Huang, Z., Xue, Z., Wang, X., An, B., and YAN, S. Agentstudio: A toolkit for building general virtual agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=axUf8BOjnH.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations.

##### A. Task Formulation

GUI Automation. Following prior work (Xie et al., 2024; Zhou et al.), we model GUI automation as a partially observable Markov decision process (POMDP) (S,O,A,T ,R). Given a natural-language query q and optional instructional video v, the agent receives an observation ot of the current state st and outputs an action at ∈ A, which updates the environment to st+1. The episode terminates when the task succeeds or fails, and R returns a binary completion signal.

WorldGUI Tasks. As illustrated in Figure 1, each task in WorldGUI is instantiated under multiple initial states that all converge to the same goal. We create these variants through pre-actions—short executable sequences that shift task progress without altering the UI skin, resolution, or layout. This design enables controlled task-progress perturbations (Add/Trim/Adjust-step) and complements existing online GUI benchmarks; key differences are summarized in Table 1.

Observation Space. The agent observes (i) application metadata mt (e.g., panel structure and parsed UI elements), and (ii) a screenshot Vt of the current GUI state, following (Gao et al., 2024). Metadata provides structured grounding anchors, while screenshots offer holistic visual context for planning and action generation.

Action Space. Actions include raw mouse and keyboard operations (click, drag, type, shortcuts), with mouse actions parameterized by pixel coordinates in Vt. We use PyAutoGUI4 for execution, and represent each action in the form action type(arguments) (Table 10).

Table 10. The action types and examples.

Action Type Example Mouse Movement moveTo(120, 200) Mouse Clicks click(200, 300) Keyboard Type write(’classes’) Hotkey hotkey(’ctrl’, ’a’) Scrolling scroll(-100) Drag dragTo(120, 220, 2) Mouse Down and Up mouseDown(); mouseUp() Press Keys press(’delete’) Key Down and Up keyDown(’shift’)

##### B. WorldGUI-Agent: Thinking before Doing

In this section, we introduce an universal GUI framework WorldGUI-Agent with a core and essential designing principle: critical thinking, which is vital for designing GUI agents capable of handling dynamic environments that have been overlooked in prior GUI agents (Hong et al., 2024; Cheng et al., 2024; Lin et al., 2024; Zhang et al., 2023; Agashe et al., 2025a). The WorldGUI-Agent includes the five fundamental but essential components as in Figure 6

4https://pyautogui.readthedocs.io

and an Interaction reasoning loop detailed in Algorithm 1. We summarize our critical designs in the following:

- • Post-Planning Critique: After the planning phase, a critique module verifies and, if necessary, self-corrects the generated plans to ensure their accuracy.
- • Pre-Action Validation: Before executing each subtask, a validation module determines whether the subtask should be executed. This step is crucial, as the current GUI environment may indicate that the subtask is unnecessary or requires modification to align with the current state.
- • Post-Action Evaluation: After each action execution, a mechanism evaluates whether the action was successfully completed before proceeding to the next subtask.

These critique designs ensure the reliability and adaptability of WorldGUI-Agent in complex GUI environments.

###### B.1. State-Aware Planner

The State-Aware Planner processes the instructional video v and the user query q generates an initial plan as shown in the left of Figure 7. We use the speech recognition model Whisper (Radford et al., 2023) to translate the video v into the subtitle and then send it to the MLLM for task planning. The task plan is hierarchically structured as p = [p1,p2,...,pN] where pi is a text string describing the i-th milestone of the task. Under each pi, there is a list of subtasks [S1i,S2i,SNi ], where Sji is the j-th subtask in the i-th milestone. To ensure the produced plans fit the GUI environment, we propose incorporating an initial screenshot V0 to represent the current state. This additional context allows the agent to output plans that align with the actual state. For example, if the instructional video suggests clicking on the “Layout” tab in the Word application, but the current state (as indicated by the screenshot) shows that the “Layout” tab is already selected, there is no need to perform this action again. By utilizing the visual information from the screenshot, the State-Aware Planner can modify the plans accordingly, rather than strictly following the guidance in the instructional video or the existing knowledge from backbone MLLMs. It also avoids the occlusion issue when not seeing the screenshot.

###### B.2. Planner-Critic

Post-Planning Critique. The goal of the Planner-Critic is to assess the correctness of the initial plans generated by the State-Aware Planner and provide corrections if needed. This module is designed to ensure the accuracy of the plans while leveraging the self-reflection capabilities of MLLMs. As illustrated in Figure 7, for each Initial Plan, the output consists of four components:

- (1) <Flag>: Indicates whether the Initial Plan is correct.
- (2) <Feedback>: Identifies the error type, categorized

###### 1. Planner

[Figure 36]

[Figure 37]

Generate the plans based on current state

User Query + Instructional Video

Observe

[Figure 38]

[Figure 39]

Task 1: Adjust text wrapping around a Word table

[Figure 40]

- Subtask 1: Click within the table to select it.
- Subtask 2: Click on the 'Layout' tab in the toolbar.
- Subtask 3: Click 'Properties' from the options available. …

[Figure 41]

Observe

###### Desktop Environment 5.AssessActor-Criticthe success of the last action

If wrong, correct the action

Observe

###### 2. Planner-Critic

Act

Assess the correctness of plans generated by Planner

Observe

Feedback

Task 1: Adjust text wrapping around a Word table

###### 3. Step-Check

###### 4. Actor

- Subtask 1: Click on the 'Table Layout' tab in the toolbar.

- Subtask 2: Click within the table to select it.

- Subtask 3: Click 'Properties' from the options available. …

Check the task completion and redundancy

Generate action represented by code Correct the action with critic feedback

- Figure 6. WorldGUI-Agent. The Planner module receives the user query and an instructional video as input and generates an initial plan. This plan is then refined and executed step by step. Before each step is passed to the Actor module, it undergoes a Step-Check. After the Actor produces an action, the Actor-Critic module iteratively verifies the completion of the action and makes corrections.

State-Aware Planner

Planner Critic

[Figure 42]

User Query: Make the text around the table

Instructional Video

Task 1: Adjust text wrapping around a Word table

- Subtask 1: Click within the table to select it.
- Subtask 2: Click on the 'Layout' tab in the toolbar.
- Subtask 3: Click 'Properties' from the options available.
- Subtask 4: In the 'Table Properties' dialog, go to the 'Table' tab.
- Subtask 5: Click 'Around' button.
- Subtask 6: Confirm the changes to apply the text wrapping around the table.

Initial Plan

Corrected Plan

You need to verify whether the provided plans can fulfill the user query. If not, please revise the plans. <Flag>: should be set to either true or false. If the plans are correct, selecting true, else selecting false. <Feedback>: If the plan is correct, please explain why. If the plan is incorrect selecting one of the following error types: 'Wrong steps', 'Missing steps', or 'Redundant steps'. <Correction>: If the plans are correct or the task is finished, output 'None', else output the corrected plans. <Reason>: Please give your reasons.

[Figure 43]

Task 1: Adjust text wrapping around a Word table

- Subtask 1: Click on the 'Table Layout' tab in the toolbar.
- Subtask 2: Click within the table to select it.
- Subtask 3: Click 'Properties' from the options available.
- Subtask 4: In the 'Table Properties' dialog, go to the 'Table' tab.
- Subtask 5: Click 'Around' under 'Text Wrapping'.
- Subtask 6: Confirm the changes to apply text wrapping around the table.

Initial Screenshot

Instructional Video

[Figure 44]

- Figure 7. State-Aware Planner and Planner-Critic. The Planner generates an initial plan. Then, the Planner-Critic provides necessary corrections.

###### 1. Check the Step Completion.

You need to verify, based on the screenshot, whether the current task has been completed or requires modification. The output should be one of the following states. <Modify>: If require modification, please either add more plans or modify current step. <Pass>: If you think current task is unnecessary. <Continue>: No change <Finished>: Already Finished. If you think current screenshot is not give all information to check the current task completion, please output '#Cannot confirm'.

Current Task: Subtask 1: Click on the 'Table Layout' tab in the toolbar.

###### MLLM

#Cannot confirm

###### Retry

[Figure 45]

| |
|---|

[Figure 46]

Crop

Region Search

Current Screenshot

New Screenshot

###### 2. Subtask Navigation If #Cannot confirm

###### If <Modify> or <Continue>

If <Pass> or <Finished>

Next Subtask

Execute current subtask, go to next module (Actor)

Invoke Region Search Retry

Go to next Subtask

Figure 8. Step-Check. This module first checks the step completion status via an MLLM and then navigates to the current task processing.

into one of three options: “Wrong Steps,” “Missing Steps,” or “Redundant Steps.”

extensive investigation, we discovered that during GUI task testing, perfect execution plans are rarely feasible due to the unpredictable nature of real application environments. Most software retains user preferences (e.g., remember the last configuration of user), meaning that when executing a specific task, the plan p generated by the Planner might not align with the actual state of the software. Therefore, the model must determine whether to proceed with a subtask St based on the current state (screenshot: Vt, metadata: Mt).

- (3) <Correction>: Provide the corrected plans if the Flag indicates that the Initial Plan is incorrect.
- (4) <Reason>: In addition to giving the corrected plans, we force the model to give the reasons. As previous studies (e.g., CoT (Wei et al., 2022), Deepseek-R1 (DeepSeek-AI et al., 2025)) demonstrate that generating reasoning steps along with the answer would enhance the performance.

###### B.3. Step-Check

As illustrated in Figure 8, we employ an MLLM to determine whether the current task has been completed or requires modification. We systematically categorize the possible outcomes into:

Pre-Action Validation. After the plan assessment, a navigation mechanism is crucial before sending each subtask St = Sji at the time step t to the Actor module. To address this, we designed a new module called Step-Check. Through

(1) <Modify>: Indicates that the subtask should be modi-

fied or additional subtasks should be added.

Step1: Verify the Action

Based on the screenshots before and after the action, task description, software name, please check the task completion status. Current Task: <Subtask St-1> Screenshots: <Screenshot Vt-1> <Screenshot Vt> Output: <Success> bool (Current task completion status) </Success> <Reason> str (Analysis of possible mistakes if action is wrong) </Reason>

- (2) <Pass>: Indicates that the current subtask is unnecessary and can be skipped.
- (3) <Continue>: Indicates that the subtask is valid and should be executed as planned.
- (4) <Finished>: Indicates that the subtask has already been completed and requires no further action.

Current Task: Subtask 3: Click 'Properties' from the options available.

|<Success>|
|---|

MLLM

###### Step2: Iteratively Action Correction

In cases where the screenshot does not provide sufficient visual information for the MLLM to determine the output, the model outputs “#Cannot confirm”. When this occurs, we design a Region Search module implemented by an LLM. This module takes the corresponding GUI information extracted by the GUI parser and the task description of the current subtask to identify the relevant region. It then crops the region using the generated bounding box as the center coordinate, with the maximum width and height set to half of the original screenshot dimensions (ensure the region is smaller than the original screenshot). The cropped screenshot is subsequently sent to the Step-Check module to regenerate the decision.

If <Success> and t < max trials

is false, State =<Critic>

is true, State =<Next>

Next Subtask

Actor-Critic

Locate GUI Elements

Properties... [161, 127];

Act&Observe

Actor Correction

Click(161, 127)

Figure 9. Actor-Critic. This module includes two parts: task verification and task correction. The design follows the verify-thencorrect mechanism.

###### B.4. Actor

The goal of the Actor is to translate natural language subtask St into executable code Ct. Using an MLLM as the backbone model, the Actor processes metadata mt and screenshot Vt as GUI context to generate precise executable actions, such as click(100, 200). Additionally, it leverages the history of previous actions as memory to aid in generating subsequent actions. The generated actions will be executed in the environment, and then the new screenshot Vt+1 and metadata mt+1 will be captured for the next processing.

data for the continued Actor-Critic iteration. The process repeats until the <Success> flag is true or the maximum number of trials is reached.

##### C. Additional Experiments

As shown in Table 11, Agent-S2 (Agashe et al., 2025b) shows competitive results as compared with our WorldGUIAgent. We also test on two representative office software to compare the effectiveness of our proposed agentic framework by replacing the base model with UI-TARS-1.5 (Qin et al., 2025) in Table 12. It is noted that to improve the performance of UI-TARS-1.5, we use the GPT-4o to task planning, as we found that UI-TARS struggles with understanding complex desktop software layout and cannot capture the dynamic initial condition changes. We use GPT-4o for better implementation.

###### B.5. Actor-Critic

Post-Action Evaluation. After generating an action, the Actor-Critic module evaluates subtask St−1 completion and makes corrections if necessary. As illustrated in Figure 9, in the first step, the module implemented by an MLLM compares screenshots Vt−1 (before action execution) and Vt (after execution) while processing each subtask St to determine the action correctness. The model outputs a <Success> flag to indicate task completion. If the <Success> flag is true, the current state st = <Next>. If the <Success> flag is false (set st = <Critic>) and the number of trial steps is below the maximum limit, the Actor-Critic module activates the Locate GUI Elements and Actor Correction processes. We introduce the module Locate GUI Elements to identify the relevant GUI elements and regenerate actions using the Actor Correction module. The corrected actions are then executed in the environment, generating updated observations (Ot) that include new screenshots and meta-

##### D. Data

###### D.1. Annotators

In this work, we have four annotators: A, B, C, and D. The team comprises one PhD student, one Master’s student, and two undergraduate students. Prior to annotation, all annotators receive training on using the applications in WorldGUI to ensure high-quality annotations. For the 10 desktop applications, we divide the software into four parts, assigning each part to a different annotator. For the human

Table 11. Performance comparison between WorldGUI-Agent (with Claude-3.5-Sonnet and Claude-Sonnet-4) and Agent-S2 (ClaudeSonnet-4). Results are reported across five representative applications.

PPT VSCode Acrobat VLC File Explorer

Method

Meta Aug. Meta Aug. Meta Aug. Meta Aug. Meta Aug.

Agent-S2 w/ Claude-Sonnet-4 45.5 18.9 45.5 37.8 18.2 14.5 20.0 16.3 60.0 64.7 WorldGUI-Agent w/ Claude-3.5-Sonnet 54.5 39.6 54.5 26.7 63.6 20.0 50.0 18.4 50.0 17.6 WorldGUI-Agent w/ Claude-Sonnet-4 63.6 52.8 54.5 28.9 54.5 30.9 40.0 28.6 70.0 26.5

Table 12. Performance comparison between WorldGUI-Agent (with UI-TARS-1.5) and UI-TARS-1.5.

PPT Acrobat

Method

Meta Aug. Meta Aug.

WorldGUI-Agent w/ UI-TARS-1.5 36.6 18.9 36.4 9.1 UI-TARS-1.5 27.3 17.0 9.1 1.8

tests presented in Table 4, the annotators demonstrate tasks on software that they did not annotate. As shown in Table 1, each annotator is responsible for different software during both the annotation and human testing phases to make the soundness of the Human Test results.

###### D.2. Creating Augmented Tasks

In our study, to simulate dynamic testing processing in real GUI interactions, we propose to design GUI tasks with various initial tasks. Specifically, we propose pre-actions before executing the task. The pre-actions primarily serve two purposes: 1) Simulating Intermediate Task States: Pre-actions can complete specific steps of a task, creating a starting point from an intermediate state. This approach addresses scenarios where users may seek AI assistance because they are unable to complete a task. For example, if the task involves opening a dropdown menu, the pre-action may pre-open the menu. If the agent fails to recognize this precondition and follows its plan to click the menu again, it might inadvertently close the menu, causing task failure.

###### D.3. Introducing Diverse Initial Context States

Pre-actions can also introduce variations in the initial state, such as opening random tabs or settings. This ensures that the starting state is unconventional, challenging the agent to adapt by modifying its plan or adding new steps. We illustrate one example in Figure 5. Here, the meta task and augmented task, have the same user query and instructional video and it will ideally have the same final state. We additionally provide more examples about augmenting the meta task in Figure 11.

##### E. Detailed Experimental Results

Table 14 shows the detailed results of WorldGUI-Agent across individual applications in WindowsAgentArena (Bon-

atti et al., 2024) benchmark. The results of this related Windows-centric interactive GUI benchmark indicate that current the desktop GUI tasks are more challenging than web tasks. As we complete 11 out of 17 tasks in Web Browsing, a similar phenomenon is also discovered in Table 4.

##### F. Computational Costs Discussion

The average number of execution steps and tokens consumed are shown below Table 15. The execution steps are calculated based on our experimental log files, while the token costs are sampled from representative tasks in each category by taking Actor module as an example.

Take a Windows Setting task as an example, we provide detailed time costs across different modules tested on a desktop PC with AMD Ryzen 7 5800H CPU. Task length: 6 (generated by Planner+Planner-Critic). To facilitate a fair comparison, we additionally selected two of the latest SOTA agents, Agent-S (2024-10-08) and Agent-S2 (2025-04-01), and measured their run times on the same successful task under identical hardware and the same base MLLM (ClaudeSonnet-4). The results are shown in Tables 16, 17, 18. To summarize, our WorldGUI-Agent shows a competitive running time of 129.55s, as compared with Agent-S (131.98s) and Agent-S2 (108.64s). The main computational costs of our designed modules are largely affected by the underlying large multimodal model, leaving room for acceleration optimization.

Since desktop GUI automation is still in its early stages, such computational costs are currently unavoidable. For reference, even OpenAI’s Deep Research reportedly takes over 10 minutes in daily usage. According to OpenAI Operator’s report, achieving 38.1% on OS-World requires over 100 steps, which is similarly costly. In summary, there remains a clear tradeoff between performance and time costs in GUI automation, and this challenge is shared across the

Table 13. The annotation arrangement during the annotation and human testing phases by different annotators.

Annotators Annotation Phase Human Test Phase

- A PowerPoint, Word, Excel VSCode, VLC Player, Web

- B Adobe Acrobat, VLC Player Excel, Settings

- C Settings, Web PowerPoint, File Explorer, Youtube

- D VSCode, File Explorer Word, Adobe Acrobat

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Inst. Video Preparation: Human checks to cut the Video Clips (30s to maximum of 3 mins) as Inst. Video.

###### Project File Preparation:

###### Pre-actions Generation:

[Figure 53]

Human Code Agent

Create the Project File for each task. It ensures the reproducibility.

Augment each task and generate Pre-actions.

[Figure 54]

[Figure 55]

02

04

06

01

03

05

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

###### Raw Video Collection:

###### User Query Generation:

###### GT-Plan Generation:

Find potential Computer Use Full Videos from Youtube (PPT, Word, Excel) or official software website.

Manually write a User Query for each instructional video.

Use the Agent to generate Initial Plans and then ask the annotators to verify and correct the error plans.

Figure 10. Pipeline of Data Construction. Human: Represents the annotators. Code: Refers to the scripts (e.g., Python Code) utilized to achieve the goal. Agent: We design an agent built upon the MLLMs to achieve the goal.

community.

inherited knowledge or the instructinal videos. In Figure 15, we show an example of changing the interface by clicking the Data tab to hide the Merge & Center button under the Home tab. In Figure 16, we complete the first step about selecting A1 to K1, which requires the agent to jump this step to reduce the time costs.

##### G. Examples of Augmentations

In this section, we present several augmentation examples in Figures 12, 13, 14, 15, 16, 17, 18. It is noted that our augmentations are not only making the first step changing but also require the agent add new step in its second step. For instance, in Figure 12, our augmentation is about click on Data tab in the ribbon, in the default software state, the Merge & Center button exhibit in the Home tab, there is no need to click on Home tab, after our augmentations, the agent should add a new task “Click on Home Tab” before it click on the Merge & Center button. Similarly, in Figure 17, the text editing buttons are under the Home Tab, if we augment the initial state with other Tab like Animation Tab, after the first step “Select the text ’US SUBMARINE DAY’ ”, the agent should add a new step like “Click on Home Tab” back to the default state for task execution. Except for adding new steps, we also present an example about adjust the step in Figure 13, as the target is about merging cells A1 to K1, we augment the initial state by selecting A2 to K2. Such a slight difference may mislead the agent to perceive such a minor difference, and the agent may jump the first step about selecting the correct cells lead to finally unsucess. In Figure 14 and Figure 18, we show two examples of introducing pop-up window in the initial state which require the agents accurately identify the pop-up windows and correctly close it by replanning the task based on the visual screenshot not only strictly planning based on

##### H. WorldGUI-Agent Reasoning Loop Algorithm

In this section, we provide the details of our reasoning loop algorithm in Algorithm 1.

##### I. Qualitative Results

(1) In Figure 19, we present a successful prediction example, demonstrating that the WorldGUI can effectively plan each step for a task, accurately perceive specific elements in the GUI, and convert them into the correct action code. Additionally, we display the parsed GUI elements, which can accurately identify most content, including small icons and dense text elements. (2) We provide the visualization results of using Planner-Critic, Step-Check, and Actor-Critic in Figure 21, Figure 22, and Figure 23. These qualitative results demonstrate the effectiveness of these critical modules in GUI automation. (3) We also highlight some common errors encountered. 1) The model has difficulty obtaining the desired information when we augment the task by invoking the dropdown menu of the Settings application. As shown on the left of Figure 24, when we click on the ’System’

- Query 1: Add a note to the second ppt, the content of the note is 123

- Query 2: Make the text around the table

[Figure 60]

[Figure 61]

[Figure 62]

Meta Aug 1 Aug 2

Pre-actions: Click on the “Replace” button in the Home ribbon. It augments the context of performing the task. The agent should understand the current state and then close the pop-up window to continue the process.

Pre-actions: None Pre-actions: Click on slide 3. It may mislead the planner to omit adding the step 'click on slide 2' because of the concept of slide 1, 2, 3 is not easy to obtain only from screenshot.

[Figure 63]

[Figure 64]

[Figure 65]

Meta Aug 1 Aug 2

Pre-actions: None Pre-actions: Click on “Design” Tab. It make the context changed. The agent should not mislead by current state.

Pre-actions: Click on “Table Layout” to stimulate the intermediate initial state.

Figure 11. We present the examples of conducting the augmentations from the meta task.

button on the left, it is challenging for our model to extract the button’s position as it is hidden. Such cases require the model to have a higher level of ability to delete the content in the input box or click on the blank area. 2) As shown in the right of Figure 24, the model has difficulty dragging a bar to achieve the desired value. 3) The model struggles with the visual choice when there is no text information in the screenshot, as shown on the left of Figure 25. The subtask aims to select the center button, but the current model makes it hard to detect the center choice only from the screenshot. 4) The model cannot successfully locate the position of the input box, as the GUI parser will easily locate the text location ’Replace with’, it always outputs the action like clicking on the ’Replace with’, which will destroy the whole task’s success.

Table 14. Detailed experimental results of WorldGUI-Agent across individual applications in WindowsAgentArena (Bonatti et al., 2024).

###### Domain Application #Tasks #Successes SR (%)

Web Browsing chrome 17 11 64.71 Windows Utilities clock 4 2 50.00 Windows System file explorer 19 7 36.84 Office libreoffice calc 24 1 4.17 Office libreoffice writer 19 2 10.53 Windows Utilities microsoft paint 3 1 33.33 Web Browsing msedge 13 5 38.46 Windows Utilities notepad 2 1 50.00 Windows System settings 5 4 80.00 Media & Video vlc 21 6 28.57 Coding vs code 24 8 33.33 Windows Utilities windows calc 3 0 0.00

###### Overall 154 48 31.17

Table 15. Average execution steps and token costs on different software.

Application category Average execution steps Input tokens per step (Actor) per task Output tokens per step (Actor) per task

Office ∼23 2350 212 Win. Usage ∼20 1929 108 Web ∼17 1637 84

Table 16. Running time with WorldGUI-Agent (ours).

Subtask Index Executed Modules Time (seconds)

- 0 Planner 4.48

- 0 Planner-Critic 11.47
- 1 Parser 2.03

1 Step-Check 4.95

- 1 Actor 7.47

- 1 Parser 2.07

- 1 Actor-Critic 7.38
- 2 Parser 2.03

2 Step-Check 6.71 2 Actor 6.05 2 Parser 2.07 2 Actor-Critic 7.97 3 Parser 2.12 3 Step-Check 6.35 3 Actor 6.71 3 Parser 2.22

- 3 Actor-Critic 10.06
- 4 Parser 2.01

- 4 Step-Check 6.19

- 4 Actor 8.54

- 4 Actor-Critic 9.55
- 5 Parser 2.22

- 4 Parser 2.40
- 5 Step-Check 6.50

Total - 129.55 Average (per action) - 22.72

Table 17. Running time with Agent-S.

Step Executed Modules Time (seconds)

- 0 Manager 2.02
- 1 Manager 10.11
- 2 Worker 16.05
- 3 Worker 12.38
- 4 Worker 17.30
- 5 Worker 12.26
- 6 Worker 15.17
- 7 Worker 15.46
- 8 Worker 11.83
- 9 Worker 19.40

Total - 131.98 Average (per worker) - 14.98

Table 18. Running time with Agent-S2.

Step Executed Modules Time (seconds)

- 0 Manager 11.87
- 1 Worker 7.27
- 2 Worker 13.21
- 3 Worker 13.59
- 4 Worker 14.04
- 5 Manager 10.61
- 6 Worker 9.87
- 7 Manager 7.53
- 8 Worker 20.65

Total - 108.64 Average (per worker) - 13.22

UserQuery:MergeA1:K1 1

[Figure 66]

[Figure 67]

| |
|---|

| |
|---|

[Figure 68]

[Figure 69]

Default State Augmented State

###### GT Plan:

- Task 1: Select A1:K1 Subtask 1: Select the cells A1 through K1.
- Task 2: Merge cells

- Subtask 1: Click the Merge & Center command.
- Subtask 2: Click 'OK' to confirm the change.

Figure 12. Augmented example of an Excel Task.

### 2

User Query: Merge A1:K1

[Figure 70]

[Figure 71]

| |
|---|

| |
|---|

[Figure 72]

[Figure 73]

Augmented State

Default State

Figure 13. Augmented example of an Excel Task.

### 3

User Query: Merge A1:K1

[Figure 74]

[Figure 75]

| |
|---|

[Figure 76]

Default State

Augmented State

Figure 14. Augmented example of an Excel Task.

### 4

User Query: Merge A1:K1

[Figure 77]

| |
|---|

| |
|---|

[Figure 78]

[Figure 79]

[Figure 80]

Augmented State

Default State

Figure 15. Augmented example of an Excel Task.

### 5

User Query: Merge A1:K1

[Figure 81]

[Figure 82]

| |
|---|

Default State Augmented State

[Figure 83]

Figure 16. Augmented example of an Excel Task.

User Query: Set US SUBMARINE DAY in the first ppt to fontsize 40

[Figure 84]

[Figure 85]

| |
|---|

| |
|---|

[Figure 86]

[Figure 87]

Default State Augmented State

Figure 17. Augmented example of a PowerPoint Task.

User Query: Select all text and apply numbered list for them. Use '1, 2, 3' symbol of numbered list.

[Figure 88]

[Figure 89]

| |
|---|

| |
|---|

[Figure 90]

[Figure 91]

Default State Augmented State

Figure 18. Augmented example of a PowerPoint Task.

Subtask 1: Open the 'Settings' application on your PC. Subtask 2: Click on 'System' in the left sidebar.

Subtask 3: Click on 'Notifications' from the available options.

[Figure 92]

[Figure 93]

[Figure 94]

|[Figure 95]|
|---|

|[Figure 96]|
|---|

Shortcut: Win+I

Subtask 6: Toggle the switch next to 'Battery saver' to the 'Off' position to disable notifications.

Subtask 4: Scroll down to the 'Notifications from apps and other senders' section.

Subtask 5: Locate the 'Battery saver' item in the list.

[Figure 97]

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

Scroll Down

|[Figure 101]|
|---|

Figure 19. We show one successful prediction of our WorldGUI-Agent.

Algorithm 1 WorldGUI-Agent Reasoning Loop Algorithm Input: State s, Action Code C, Screenshot V , Metadata m, Current subtask S, Critic count z Generate task plan p with Planner and Planner-Critic Initial current subtask St=0 = S11, where S10 is the 1-th subtask in the 1-th milestone of p. Initial s0 = <Continue> while St is not end and t < max trials do

Observe metadata mt and Screenshot Vt from Env. Obtain state st by running Step-Check. if st = <Next> then

Go to the next task St+1 = next(St) end if Check potential modification of subtask St Obtain action code Ct by running Actor; Execute the action code Ct in the Env.; Observe metadata mt and Screenshot Vt from Env. Set Ct = None; t = t + 1; Set state st = <Critic> (For each subtask, the first step is finished, then execute the actor-critic process) Observe metadata mt and Screenshot Vt from Env. Running Actor-Critic and obtain the state st if st = <Next> then

Go to the next task St+1 = next(St). end if while st = <Critic> and z < max critique trials do

Running Actor-Critic and obtain the state st and corrected action code Ct if st = <Next> then

Go to the next task St+1 = next(St). end if Execute the action code Ct in the Env.; Observe metadata mt and Screenshot Vt from Env. Set Ct = None; z = z + 1

end while Go to the next task St+1 = next(St) t = t + 1

###### end while

[Figure 102]

[Figure 103]

Figure 20. We show two examples of using GUI Parser to obtain the element position information.

###### Initial Plan

Screenshot

[Figure 104]

Task 1: Format the whole document into two columns

- Subtask 1: Click on the Layout tab.

- Subtask 2: Click the Columns command.

- Subtask 3: Click on Two to format the document into two columns.

#### Planner Critic

<Feedback>Missing steps. The current plan does not include selecting all the text in the document before applying the column format, which is necessary to ensure the entire document is formatted into two columns.</Feedback> <Flag>false</Flag> <Correction> Task 1: Format the whole document into two columns

###### Output

- Subtask 1: Click on the Home tab.

- Subtask 2: Click anywhere in the text area and press Ctrl + A to select all the text in the document.

- Subtask 3: Click on the Layout tab.

- Subtask 4: Click the Columns command.

- Subtask 5: Click on Two to format the document into two columns. </Correction> <Reason>The user query is to change the whole document from one column to two columns. The current plan does not include selecting all the text in the document, which is necessary to apply the column format to the entire document. The correction includes selecting all text before applying the column format, ensuring the entire document is affected.</Reason>

Figure 21. An example of using Planner-Critic to correct the plan.

[Figure 105]

###### Screenshot

Previous Finished Tasks: Subtask 1: Click on the text "US SUBMARINE DAY" within the main Slide Pane to select it. Current Task: Subtask 2: Go to the "Font" group on the toolbar. Next Task (for reference, you should consider whether current task is necessary when we complete next task ): Subtask 3: Type "40" in the font size box to set the font size to No. 40.

Input

<Finished> <Reason>The "Font" group on the toolbar is already visible in the screenshot, indicating that the current task of going to the "Font" group has been completed.</Reason>

Output

[Figure 106]

Previous Finished Tasks: …Subtask 2: Click the Insert tab in the PowerPoint toolbar. Subtask 3: Click the Table command. Current Task: Current Task: Subtask 4: Mouse over the grid to select a 3x3 table. Next Task (for reference, you should consider whether current task is necessary when we complete next task ): Subtask 5: Click to insert the table with the selected number of rows and columns.

Screenshot

Input

Output: <Continue>

Output

Figure 22. Two examples of using Step-Check to check the subtask status.

[Figure 107]

###### Submission and Formatting Instructions for ICML 2026

[Figure 108]

Screenshot Before Action Screenshot After Action (Crop for best view)

Input

Based on the screenshots before and after the action, subtask description, software name Excel, please check the action completion status. Current Task: Subtask 1: Click on cell A18 to select it current action: from pyautogui import click\nclick(17, 675) Parsed GUI Screenshot Info: [Note that: element format is "name [its position]", separate with comma] Excel:

elements:

Status Bar [960, 1006]; Status Bar [960, 1006]; Status Bar [960, 1006]; Cell Mode Ready [33, 1007]; Macro Recording Not Recording [85, 1007]; Accessibility Checker Accessibility: Good to go [198, 1007]; Average 152364686.9 [1126, 1007]; …

Output

<Success> True </Success> <Reason> The action to click on cell A18 was successfully completed. The second screenshot shows that cell A18 is selected, as indicated by the highlighted cell and the cell reference "A18" in the Name Box. </Reason>

Figure 23. An example of using Actor-Critic to correct the actions.

###### Query 1: Turn on Storage Sense

###### Query 2: Set the brightness under Dynamic Lighting with 70%

Subtask: Navigate to the 'System' section.

Subtask: Drag to set brightness to 70%

[Figure 109]

[Figure 110]

|[Figure 111]|
|---|

|[Figure 112]|
|---|

Error: The dropdown menu hides the “System” button. Thus the GUI Parser and MLLM cannot find the “System” in this screenshot.

###### Error: The Actor cannot provide exact bbox to achieve the goal of dragging to 70%.

Figure 24. We display some common errors.

###### Query 2: Replace all the 'Reading' in the text with 'RA'

###### Query 1: Format the slide background with gradient fill

Subtask: Enter 'RA' in the field behind the 'Replace with' text.

Subtask: Select 'Center'

[Figure 113]

[Figure 114]

|[Figure 115]|
|---|

[Figure 116]

Error: The Actor Model cannot identify the “center” button in this dropdown menu

Error: The Actor cannot click on the input field as the parsed bbox is the position of text “Replace with”

Figure 25. We display some common errors

