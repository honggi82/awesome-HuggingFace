# arXiv:2511.06251v1[cs.SE]9Nov2025

### WebVIA: A Web-based Vision-Language Agentic Framework for Interactive and Verifiable UI-to-Code Generation

Mingde Xu1,3*, Zhen Yang2,3*†, Wenyi Hong2,3, Lihang Pan3, Xinyue Fan3, Yan Wang3, Xiaotao Gu3, Bin Xu2, Jie Tang2† 1Faculty of Mathematics, University of Waterloo 2The Knowledge Engineering Group (KEG), Tsinghua University 3Zhipu AI

m339xu@uwaterloo.ca, yang-zhen@mail.tsinghua.edu.cn, jietang@mail.tsinghua.edu.cn

###### Abstract

User interface (UI) development requires translating design mockups into functional code, a process that remains repetitive and laborintensive. While recent Vision–Language Models (VLMs) automate UI-to-Code generation, they generate only static HTML/CSS/JavaScript layouts lacking interactivity. To address this, we propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation. The framework comprises three components: 1) an exploration agent to capture multi-state UI screenshots; 2) a UI2Code model that generates executable interactive code; 3) a validation module that verifies the interactivity. Experiments demonstrate that WebVIA-Agent achieves more stable and accurate UI exploration than generalpurpose agents (e.g., Gemini-2.5-Pro). In addition, our fine-tuned WebVIA-UI2Code models exhibit substantial improvements in generating executable and interactive HTML/CSS/JavaScript code, outperforming their base counterparts across both interactive and static UI2Code benchmarks. Our code and models are available at https://webvia.github.io.

###### 1 Introduction

User interface (UI) development is a core step in modern software engineering, yet translating design mockups into functional code remains a repetitive and labor-intensive process. Automated UI-toCode (UI2Code) has therefore emerged as a promising direction, aiming to automatically transform UI screenshots into structured front-end code. Recent advances in Vision-Language Models (VLMs) (Liu et al., 2023; Wang et al., 2024a,b; Bai et al., 2025; Hong et al., 2025; Zhu et al., 2025) have created new opportunities for UI2Code that jointly interprets visual layouts and textual semantics, thus

* Core contributors. † Corresponding author.

moving beyond shallow pattern recognition (Beltramelli, 2018; A¸sıro˘glu et al., 2019; Chen et al., 2022) toward more robust and visually grounded code generation (Wu et al., 2025; Jiang et al., 2025).

Despite these advances, current VLM-powered UI2Code approaches remain limited in functionality. As demonstrated in Figure 1, their outputs are typically restricted to static HTML/CSS/JavaScript code, which reproduce the visual appearance of interfaces but lack support for GUI interactions such as clicking, selecting, or entering text. The generated interfaces cannot correctly respond to user actions and therefore cannot be integrated into real-world UI development workflows. These limitations highlights the necessity for a new framework capable of producing executable and truly interactive user interfaces.

To address this gap, we propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation. The framework is composed of three components: (1) an exploration agent that interacts with the HTML environment to capture multiple UI screenshots across different interface states; (2) a UI2Code model that leverages these screenshots to generate executable HTML/CSS code for an interactive GUI; (3) a validation module that assesses the feasibility and interactivity of the generated GUI.

We train two core models to ensure the high performance of WebVIA. The first is an exploration agent, WebVIA-Agent, that traverses the HTML environment to collect diverse interface states. We construct a large-scale GUI interaction dataset, and experimental results show that WebVIA-Agent outperforms general-purpose models such as Gemini2.5-Pro (Comanici et al., 2025) in both stability and accuracy. The second is a UI2Code model that generates executable and interactive interfaces. Based on Qwen-2.5-VL-7B and GLM-4.1V-9B, we train WebVIA-UI2Code-Qwen and WebVIA-

[Figure 1]

Interactive UI2Code Generation

[Figure 2]

Interactive UI2Code Generation for Qwen2.5-VL

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Start Click Flights Click Insurance Click Hotels

[Figure 10]

Start Click Flights Click Hotels

Rendering

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Rendered UI without interactive operations

Start Click Flights Click Insurance Click Hotels

Interactive UI2Code Generation for GLM-4.1V-9B

Generalization

[Figure 16]

[Figure 17]

[Figure 18]

Static UI2Code Generation

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Start Click Flights Click Hotels

[Figure 25]

Rendered UI without interactive operations

Reference UI Ours (UI2Code)

GLM-4.1V-9B Qwen2.5-VL Results on Design2Code

Figure 1: Motivating example illustrating the gap between static and interactive code generation.

UI2Code-GLM using paired multi-state UI screenshots and executable interactive HTML. Compared with their respective base models, the two variants achieve improvements of 5.2 and 4.7 points on the Design2Code benchmark, and on UIFlow2Code they reach performance levels (75.9 and 84.9, respectively) close to that of the large-parameter stateof-the-art Claude-Sonnet-4 .

ods. Before VLMs, works like Pix2Code (Beltramelli, 2018) used CNN–RNN architectures for static UI generation. With VLMs, methods such as DeclarUI (Zhou et al., 2024) integrate vision and language models for better component grounding, while others enhance performance through largescale datasets (Laurençon et al., 2024; Yun et al., 2024; Gui et al., 2024b,a), multi-model collaboration (Jiang et al., 2025; Liang et al., 2024), or finegrained interface decomposition (Wu et al., 2025; Chen et al., 2025; Wan et al., 2024). However, these approaches remain limited to static HTML/CSS generation, lacking interactivity or functional validation. In contrast, WebVIA introduces an agentic, interaction-aware framework that enables verifiable and executable UI2Code generation.

Our contributions can be summarized as follows:

- • Framework: We propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation, which bridges the gap between static UI rendering and executable, interactive front-end development.
- • Model: Under the guidance of WebVIA, we train two dedicated models: an exploration agent for HTML environment interaction and state collection, and a UI2Code model capable of generating executable code that supports real user interactions.
- • Experiments: In our experiments, the WebVIA-Agent achieves higher stability and accuracy than general-purpose models such as Gemini-2.5-Pro, while the WebVIA-UI2Code model surpasses static layout reconstruction to produce robust and verifiable interactive web code.

###### 2.2 Interactive Web Agents

Recent research has moved beyond static UI translation to explore interactive web environments, where agents perform multi-step actions on real webpages. Early systems such as World of Bits (Shi et al., 2017), WebGPT (Nakano et al., 2021), and BrowserGym (Chezelles et al., 2024) integrate language models with browser environments for reasoning and decision-making over dynamic content. Later benchmarks like WebArena (Zhou et al., 2023) and Mind2Web (Deng et al., 2023) enable grounded web interaction by allowing agents to perceive, plan, and execute GUI actions from both DOM structures and visual inputs. While these frameworks highlight the value of multimodal reasoning and feedback, existing agents remain optimized for single, predefined tasks, lacking the ability to systematically explore or verify interactive components. In contrast, our approach extends

###### 2 Related Work

###### 2.1 UI-to-Code Generation

The UI-to-Code (UI2Code) task aims to translate user interfaces into executable code, evolving from early deep learning approaches to recent vision–language model (VLM)-based meth-

this paradigm toward agent-based UI synthesis and verification, where the agent actively traverses and tests webpage components to ensure functional and behavioral correctness.

###### 2.3 UI2Code Benchmark

Several benchmarks have advanced VLM-based UI2Code research. Design2Code (Si et al., 2024) introduces 484 real-world webpages for visualto-code evaluation, while Web2Code (Yun et al., 2024) and Flame-React (Ge et al., 2025) refine data pipelines via LLM-assisted layout–code synthesis, though still relying on synthetic HTML. FullFront (Sun et al., 2025) broadens evaluation to the full front-end workflow—design, perception, and code generation. More recently, Interaction2Code (Xiao et al., 2024) extends benchmarking to interactive UI2Code, assessing VLMs’ ability to reproduce functional behaviors such as event handling and state transitions.

This manual dependency hinders large-scale random webpage generation and limits coverage of diverse interaction types. Moreover, the benchmark’s evaluation paradigm—requiring explicit tagging of interactive elements and pre-specified commands—diverges from real user behaviors, emphasizing instruction following rather than true interaction reasoning. To overcome these issues, we propose UIFlow2Code, a scalable, flow-based benchmark that evaluates models’ ability to understand and reproduce multi-step interactions directly from visual and structural webpage states, eliminating the need for handcrafted task annotations and enabling behavior-grounded assessment.

###### 3 WebVIA Framework

We propose WebVIA, the first agentic framework for interactive UI-to-Code generation and validation, designed to move beyond static rendering toward executable and verifiable front-end development. As illustrated in Figure 2, WebVIA integrates three core components: an exploration agent that systematically interacts with HTML environments to uncover hidden states and produce validated UI screenshots, a UI2Code model that leverages these screenshots to generate executable code with both layout fidelity and interactivity, and a validation module that executes the generated code, verifies the support for intended GUI behaviors and functionalities. These components form a pipeline that bridges the gap between static UI rendering

and robust, interactive front-end development.

Problem Formulation. We formalize interactive UI-to-Code generation as a sequential decisionmaking problem over a structured environment. The exploration agent interacts with a webpage environment E. At each step t, the agent observes a multimodal state st = (It,Dt), where It is the rendered screenshot and Dt is the DOM snapshot. The agent then selects and executes an action at ∈ A , and the environment transitions to a new state st+1. The action space A consists of standard web interactions, including clicks, text inputs, selections, and navigations. The agent progressively uncovers hidden states and constructs an interaction graph G = (S,T ), where S is the set of discovered states and T the set of verified transitions. Based on this interaction graph G, the UI2Code model generates executable front-end code Cˆ that faithfully reproduces the interactive behaviors of the original environment. Finally, the validation module executes the generated code and assesses the interactivity of the rendered GUI by verifying whether it can still support the transitions T defined in the interaction graph G.

Environment. The environment serves as the foundation of performing WebVIA framework. Inspired by prior work such as WebArena (Zhou et al., 2023), we implement a dedicated web environment, denoted as WebEnv, which renders a given HTML document within an isolated browser instance. The implementation builds on GymAPI for standardized interaction and Playwright for browser automation. WebEnv supports three core capabilities: (1) rendering and capturing full-page screenshots, (2) extracting DOM trees with annotated interactive elements and the corresponding XPaths via JavaScript instrumentation, and (3) executing user interactions within the browser. These functionalities enable systematic grounding of both the visual and structural aspects of the webpage, providing a reliable basis for agent-driven exploration and evaluation.

Real-world webpages are noisy and unstable due to advertisements, asynchronous loading, and external dependencies, making them unsuitable for controlled training and reproducible evaluation. To address this, we construct a large-scale synthetic environment in which webpages are automatically generated from templates and textual specifications. This design provides diversity and systematic coverage of common interaction patterns, ensures full

###### WebVIA Framework

###### (c) Validation Module

###### (a) WebVIA-Agent

Cycle

###### Tasks

###### Action Generation Interaction Verification

|Task1: Action Seq 1|
|---|

###### .......

Environment Environment

###### WEBVIA-Agent

Action Seq 1 Screenshots

| | |
|---|---|
| | |

[Figure 26]

[Figure 27]

Generated Action Seq 1 WEBVIA-Agent

Fail to execute

|TaskN: Action Seq N|
|---|

[Figure 28]

Click[1]

Execute

Web

[Figure 29]

[Figure 30]

...... ......

[Figure 31]

Succeed but no new elements

###### Env

[Figure 32]

[Figure 33]

[Figure 34]

Images Web

Action Seq N Screenshots

Verify

Generated Action Seq N

Screenshots DOM

Web

[Figure 35]

[Figure 36]

Enter[3][Hello World] Click[1]

[Figure 37]

[Figure 38]

Succeed and expose new elements

DOM

Execute

###### Validation Model (GPT-5)

Generated Action Seq 1 Click[2]

###### (b) WebVIA-UI2Code

Input Samples

[Figure 39]

Samples

Prompt Instruction

###### .......

Reconstruct

"Please generate an interactive HTML webpage ......"

| |[Figure 40]<br><br>......<br><br>Action Seq 1 Screenshots<br><br>Screenshots| |
|---|---|---|

Generated Action Seq N Select[6][Hi!] Click[7]

| | |
|---|---|
| | |

[Figure 41]

[Figure 42]

[Figure 43]

Interaction Description

"The first action seq is .... The second action seq is ...."

Interactive HTML

[Figure 44]

UI2Code Model

Verify Results

| | |
|---|---|
| | |

Filter

###### Validation Model (GPT-5)

- Action Seq1: Succeed
- Action Seq2: Failed

Screenshots

Verify

........ Action SeqN: succeed

[Figure 45]

[Figure 46]

[Figure 47]

......

......

Success! Failed

Figure 2: Overview of the WebVIA framework, which comprises three components: (a) an exploration agent to capture multi-state UI screenshots; (b) a UI2Code model to generate interactive code; (c) a validation module to verify the interactivity.

controllability, and eliminates the unpredictability of real-world websites. The detailed synthesis pipeline is described in Appendix A.1.

ment to generate new candidate states for further exploration.

Interaction Verification To ensure efficiency and correctness, the agent verifies each executed action sequence by comparing the resulting state against the initial state according to the following criteria: (1) whether the action sequence successfully executed, and (2) whether new interactive elements appeared on the page. The comparison result falls into three categories: sequences that fail to execute (non-interactive), sequences that succeed but reveal no new elements (usable but not explored further), and sequences that succeed and expose new elements (usable and retained for subsequent exploration). This interaction verification mechanism prunes redundant sequences and incorporates new interactive states into the exploration process.

- 3.1 Part 1: Exploration Agent for Interactive UI Discovery

The exploration agent is the core driver of WebVIA’s interactive UI discovery process. It uncovers interactive elements within an HTML-based environment and constructs a reliable interaction graph of the user interface. The agent has two key capabilities: action generation and interaction verification, and explores the GUI according to a perception–action–verification strategy: it grounds its understanding in rendered screenshots and DOM trees, proposes candidate interaction sequences, executes them in the environment, and verifies whether meaningful changes occur. This iterative process ensures comprehensive coverage of interactions and robustness against ineffective operations or redundant screenshots.

Exploration Strategy. We propose a hybrid exploration strategy that integrates breadthfirst and depth-first search within a perception–action–verification loop. At each iteration, the agent generates and executes candidate actions in parallel (breadth-first) to maximize coverage, while promising states are further expanded through depth-first exploration to uncover longhorizon workflows. As illustrated in Figure 2 (a), this process incrementally expands an interaction graph, where each validated state becomes the root of subsequent exploration. By balancing breadth and depth, WebVIA achieves both comprehensive

Action Generation. The exploration agent begins by proposing candidate interaction sequences that systematically cover potential user operations on a webpage. The agent, conditioned on its historical trajectory, observes the GUI state (i.e., the rendered screenshot and the DOM tree) and outputs action sequences. Each sequence can be a primitive operation (e.g., a single click) or a composite workflow (e.g., text entry followed by a button press). These sequences are executed within the environ-

coverage of interactive elements and efficient discovery of dynamic states, thereby constructing a reliable interaction graph for downstream code generation.

- 3.2 Part 2: UI2Code Model for Interactive Front-End Code Generation

The ultimate objective of WebVIA is to move beyond static UI reconstruction and generate executable front-end code that faithfully captures the interactive functionality of the original interface. As illustrated in Figure 2(b), we introduce a dedicated UI2Code model that is conditioned on the interaction graph derived from multiple screenshots produced by the exploration agent. Unlike prior approaches that rely on a single static screenshot, our model benefits from diverse interface states captured during exploration, enabling it to synthesize functionally coherent GUI components that support essential interactive behaviors.

Multimodal Inputs. Unlike prior approaches that rely on a single static screenshot, our UI2Code model is conditioned on multiple screenshots collected during exploration, along with their verified interaction relationships. This structured input captures both the visual layouts of diverse interface states and the causal transitions induced by user actions. Leveraging this enriched representation, the model learns to reason about dynamic workflows rather than treating the UI as a static rendering.

Code Generation. Conditioned on these multimodal inputs, the UI2Code model generates executable front-end HTML/CSS/JavaScript code. In contrast to prior approaches that only reconstruct static layouts, our model explicitly preserves interactive functionality, ensuring that the generated code is both visually faithful and behaviorally reliable.

- 3.3 Part 3: Validation Module for Interactive Code

Existing evaluation methods for UI2Code models primarily focus on the visual fidelity of the generated interfaces, with little attention paid to verifying their interactivity. To address this limitation, the final component of WebVIA is the validation module, which ensures that the generated frontend code is functionally interactive. As shown in Figure 2(c), we adopt a task-oriented validation procedure that directly evaluates the usability of the generated code. A set of tasks (e.g., filling out

a form, submitting a query, or navigating to a target page) is pre-defined based on the interaction graph of the original GUI and executed on the synthesized interface. The synthesized interface passes validation if all tasks can be completed as intended. This evaluation goes beyond mere state–transition matching by providing a direct measure of whether the generated code supports coherent end-to-end user workflows.

###### 4 Training Methodology

In this section, we describe how the two core models of WebVIA—WebVIA-Agent and WebVIAUI2Code—are trained.

###### 4.1 WebVIA-Agent Training

The exploration agent is trained on multimodal webpage states, consisting of rendered screenshots and filtered DOM trees. Its training objectives are threefold: (1) to ensure stability across diverse webpage layouts, (2) to comprehensively detect and interact with standard interactive elements, and (3) to jointly support the two core functions of action generation and interaction verification.

To support these objectives, we construct two complementary datasets. The Action Generation Dataset contains pairs of webpage states and annotated interaction sequences. Each entry includes a screenshot, the corresponding DOM tree, and one or more ground-truth action sequences with their historical trajectories.The Interaction Verification Dataset stores the screenshots before and after executing operation sequences, together with annotations indicating whether meaningful changes (such as new elements or layout updates) have occurred. Details of the dataset construction process are provided in Appendix A.2.

The curated datasets are used to supervise the agent through supervised fine-tuning (SFT) on GLM-4.1V-9B-base. The model learns to (i) predict valid action sequences from the action generation dataset, and (ii) classify meaningful transitions from the interaction verification dataset. Both tasks are optimized using cross-entropy loss. This joint training paradigm equips the agent with the ability to autonomously generate feasible interactions while reliably filtering out non-productive actions, providing a robust foundation for the WebVIA exploration pipeline.

###### 4.2 WebVIA-UI2Code Model Training

The UI2Code model is trained to translate multiple UI screenshots into executable front-end code that preserves both layout fidelity and interactive functionality. Unlike conventional UI-to-code systems that rely solely on static screenshots, our training paradigm exploits the structured interaction graph collected by WebVIA, ensuring that the model learns to generate interactive HTML/CSS/JavaScript code.

We construct the WebView dataset with 11k synthesized webpages, each paired with its groundtruth HTML/CSS/JavaScript code. For every webpage, the exploration agent systematically discovers states and transitions, producing an interaction graph G that contains rendered screenshots and validated action sequences. Instead of directly using the template-level HTML as supervision, we feed the exploration traces (multiple screenshots and their interaction graph) into Claude and generates the corresponding executable HTML/CSS/JavaScript code. The resulting (interaction graph, generated code) pairs form the core training data for our UI2Code model, ensuring that the supervision is aligned with the observed multimodal states and their verified interactions.

For fine-tuning, we adopt a structured prompt–response format, which is organized as: <think>···</think><answer>···</answer>. This formatting explicitly separates the reasoning context from the expected code output, enabling the model to learn stable mappings from multimodal observations to structured, executable code. The detailed data construction and data cases are provided in Appendix A.3. We perform supervised fine-tuning (SFT) on GLM-4.1V-9B-base and Qwen2.5-VL-7B-Instruct on these formatted pairs.

###### 5 Experiments

In this section, we conduct comprehensive experiments to validate the effectiveness of the proposed WebVIA framework. Our evaluation is organized around its two trainable components: the exploration agent and the UI2Code model. For the exploration agent, we examine both its intrinsic ability to generate and verify UI actions, as well as its performance in the full pipeline of interaction screenshot collection. For the UI2Code model, we evaluate its capability to generate interactive HTML/CSS/JavaScript code from multiple screenshots,

focusing on structural fidelity and functional correctness.

###### 5.1 Evaluation Setup

Benchmark. Since no public benchmark exists for evaluating our proposed WebVIA framework, we construct two dedicated benchmarks. UIExplore-Bench evaluates the exploration agent’s ability to navigate complex webpages and collect interaction screenshots, while UIFlow2Code-Bench assesses the ability of UI2Code model to reconstruct webpages with both structural fidelity and functional correctness. All samples are carefully annotated to ensure accuracy and consistency. These benchmarks provide the first standardized protocol for this task and are designed to facilitate future research. Additional construction details of both UIExploreBench and UIFlow2Code-Bench are provided in Appendix A.4 and Appendix A.5, respectively.

Baselines. The WebVIA framework is designed to be model-agnostic, allowing both the exploration agent and the UI2Code model to be instantiated with any vision-language models. To systematically evaluate the advantages of our trained exploration agent and UI2Code model, we compare against a suite of state-of-the-art VLMs, including Claude-Sonnet-4, Claude-Sonnet-3.7, GPT-5, o4mini, GPT-4o and Gemini-2.5-pro. These models can be seamlessly integrated into the WEBVIA framework via API calls, without requiring any task-specific adaptation. The versions and API endpoints of baselines are provided in Appendix A.6.

###### 5.2 Single-Step Agent Evaluation

To assess the exploration agent’s performance independent of the full pipeline, we conduct singlestep experiments on two fundamental tasks: action generation and interaction verification. The full prompt templates used for Action Generation and Interaction Verification are provided in Appendix A.7.

Action generation. In this task, the agent predicts a list of valid interactive elements from a given UI screenshot and DOM tree. We report Precision, which measures the proportion of correctly predicted actions among the agent’s selected actions, Recall, which measures the proportion of correctly predicted actions with respect to the ground-truth actions, and F1, which captures the harmonic mean of Precision and Recall.

As shown in Table 1, the WebVIA-agent outperforms all baselines except Gemini-2.5-Pro. Its advantage stems from SFT training, which enables the agent to capture subtle structural patterns and focus on truly actionable elements. Different from Gemini-2.5-pro that tends toward an overly aggressive strategy of selecting nearly all visible elements, WebVIA-agent demonstrates a more balanced and reliable behavior. Furthermore, WebVIA-agent’s predictions maintain high protocol fidelity, rarely producing mismatched DOM identifiers or invalid actions. By contrast, GPT-4o performs poorly, largely due to its inability to adhere to the prescribed interaction format, which undermines its applicability within the pipeline. The prompts

Table 1: Comparison of action generation performance (Precision, Recall, and F1) on UIExplore-Bench with 87 action samples.

Model Precision (%) Recall (%) F1 (%)

Gemini-2.5-pro 74.01 95.94 81.70 GPT-5 81.66 88.41 81.85 o4-mini 79.01 91.80 83.16 GPT-4o 4.77 5.43 4.85 Claude-Sonnet-3.7 75.29 95.18 81.72 Claude-Sonnet-4 81.16 89.67 83.38

WebVIA-Agent 82.37 92.61 85.30

Interaction verification. Given a set of images representing executed actions, the agent produces two Boolean outputs: "pass", indicating whether the sequence executes correctly, and "terminate", indicating whether the interaction introduces any new elements. We compute accuracy separately for both dimensions and use their average as the overall score. As shown in Table 2, the WebVIA-agent achieves the best performance across all three metrics, demonstrating its superior verification ability. The superior terminate accuracy of WebVIA-agent underscores the effectiveness of SFT training in enhancing visual understanding and distinguishing genuinely new interactive elements. For example, when an interaction contains repeat elements of the same type (e.g., multiple “delete item” buttons), the agent successfully identifies the redundancy and terminates the exploration branch, thereby preventing unnecessary actions.

###### 5.3 Pipeline-Level Agent Evaluation

While single-step experiments isolate the agent’s capabilities on action generation and interaction verification, they fail to provide a comprehensive assessment of its effectiveness in realistic, end-to-

Table 2: Comparison of verification performance across baseline VLMs and the WebVIA-agent on UIExploreBench with 53 verification samples.

Model Pass Acc Terminate Acc Overall Acc

Gemini-2.5-pro 94.34 81.13 87.74 GPT-5 96.23 84.91 90.57

- o4-mini 94.34 83.02 88.68 GPT-4o 33.96 62.26 48.11 Claude-Sonnet-3.7 94.34 77.36 85.85 Claude-Sonnet-4 94.34 77.36 85.85 WebVIA-Agent 98.11 86.79 91.51

end scenarios. We further evaluate the exploration agent within the WebVIA framework, where it autonomously explores webpages, generates interaction traces, and collects representative screenshots.

Evaluation metrics. To assess the effectiveness

- of the exploration agent throughout the entire collection of interaction screenshots, we adopt three complementary metrics: Completeness measures the coverage of action generation, i.e., the proportion of distinct UI elements successfully explored. Correctness quantifies the correctness of verification results, reflecting the agent’s ability to determine whether an executed action achieves its intended effect. Deduplication Rate quantifies the prevalence of redundant or repeated actions within the generated traces, serving as an indicator of exploration efficiency. We compute an overall score as a weighted combination of the three metrics:

Overall = 0.40 · Comp + 0.35 · Correct + 0.25 · Dedup (1)

, where the weights are empirically determined to balance coverage, correctness, and efficiency.

Evaluation results. As reported in Table 3, WebVIA-Agent achieves the best overall score of 89.8%, surpassing all baseline models. It achieves the best Completeness (93.1%) and Correctness (97.7%), confirming its ability to both discover diverse actionable elements and reliably validate their outcomes. These improvements arise from supervised fine-tuning, which strengthens structural understanding and encourages a balanced exploration strategy. For instance, when encountering redundant screenshots with overlapping content, WebVIA-Agent prioritizes unexplored regions and reducing duplication. By contrast, models such as o4-mini and Gemini-2.5-pro often re-trigger identical actions (e.g., repeatedly clicking the same button), yielding inefficiencies despite high nominal coverage.

Table 3: Pipeline-level performance comparison across baseline VLMs and the proposed WebVIA-Agent on UIExplore-Bench with 56 webpages.

Model Completeness (%) Correctness(%) Deduplication Rate (%) Overall Score (%)

Gemini-2.5-pro 92.61 95.39 5.60 71.83 GPT-5 76.66 90.19 93.82 85.69 o4-mini 91.73 94.07 52.73 82.80 GPT-4o 16.46 62.63 97.45 52.87 Claude-Sonnet-3.7 75.86 94.06 72.36 81.35 Claude-Sonnet-4 86.26 95.07 80.36 87.87

WebVIA-Agent 93.12 97.71 72.73 89.63

Table 4: Performance comparison on static (Design2Code) and interactive UI2Code (UIFlow2Code).

###### 5.4 Interactive Code Generation Evaluation

To assess the capability of UI2Code model in generating fully interactive HTML/CSS/JavaScript code, we extend the evaluation beyond conventional static UI2Code task to interactive code generation.

Model Design2Code UIFlow2Code Gemini-2.5-pro 89.5 90.2

GPT-5 89.7 69.9 o4-mini 63.8 69.4 GPT-4o 35.3 59.3

Claude-Sonnet-3.7 77.7 81.5 Claude-Sonnet-4 81.2 82.1

Evaluation metrics. For each generated HTML page, a set of tasks is defined based on the corresponding input images. Each task undergoes the validation process and is labeled as either pass or fail. The final evaluation metric is calculated as the ratio of the number of pass tasks to the total number of tasks.

Qwen2.5-VL-7B-Instruct 29.1 / WebVIA-UI2Code-Qwen 34.3 75.9

GLM-4.1-V-9B-Base 58.3 / WebVIA-UI2Code-GLM 63.0 84.9

[Figure 48]

Evaluation results. As reported in Table 4, supervised fine-tuning on the WebView dataset substantially improves the ability of both Qwen-2.5VL-7B and GLM-4.1V-9B to generate executable interactive HTML/CSS/JavaScript code, whereas their base counterparts fail to produce valid outputs. This contrast highlights that interactive training data are indispensable for enabling interaction capabilities. Interestingly, although our supervised fine-tuning is conducted exclusively on interactive UI2Code data, we also observe consistent improvements on static UI2Code benchmarks. This suggests that interactive training data provide richer structural and functional supervision than conventional single-state screenshots, thereby enhancing the model’s ability to capture layout fidelity and semantic alignment even in static scenarios. Details of the evaluation prompt for interactive code generation are provided in Appendix A.8.

Figure 3: Correlation between the mean interaction trace length and the overall exploration score across our WebVIA-Agent and various VLMs.

indicate more intelligent action planning, but it may also reflect premature termination that fails to capture deeper interactive elements. As shown in Figure 3, WebVIA-Agent achieves a balanced mean trace length together with the highest overall performance. This combination indicates WebVIAAgent is not only efficient but also consistently effective, indicating that it strikes a strong balance between quality and speed.

###### 5.5 Average Trace Length of WebVIA-Agent

In our ablation study, we analyze the relationship between average trace length and overall performance as reported in Table 3. Here, trace length refers to the number of interaction steps executed by the pipeline-level WebVIA-Agent during a full webpage exploration. A lower trace length may

###### 6 Conclusion

In this work, we present WebVIA, an agentic framework for interactive UI-to-code generation and validation. Unlike prior methods limited to static HTML/CSS reconstruction, WebVIA introduces an ex-

ploration–generation–validation pipeline that enables interaction-aware, behavior-preserving code synthesis. Built upon large-scale GUI interaction and WebView data, two specialized agents—an exploration agent and a UI2Code generator—jointly produce executable and verifiable web interfaces.

###### Limitations

Although WebVIA establishes a new paradigm for interactive UI-to-Code generation, there remain two limitations in scalability and generalization that need to be addressed before achieving broader applicability. (1) In the exploration stage of WebVIA pipeline, the action types are restricted to Click, Enter, and Select. Executing broader action types such as Drag and Draw requires precise pixel coordinates, which defers from our current approach of ID based DOM to XPath execution. (2) Training the agent primarily on synthetic Webpages may limit its ability to handle certain specialized interaction tasks in real-world settings. For example, WebVIA-Agent struggles with domains such as calculators or function-plotting interfaces, where interaction patterns deviate substantially from the structures observed in the training environment. These constraints delineate the current scope of WebVIA and point to concrete directions for extending its applicability in future research.

###### References

Batuhan A¸sıro˘glu, Büs¸ta Rümeysa Mete, Eyyüp Yıldız, Ya˘gız Nalçakan, Alper Sezen, Mustafa Da˘gtekin, and Tolga Ensari. 2019. Automatic html code generation from mock-up images using machine learning techniques. In 2019 Scientific meeting on electricalelectronics & biomedical engineering and computer science (EBBT), pages 1–4. Ieee.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Tony Beltramelli. 2018. pix2code: Generating code from a graphical user interface screenshot. In Proceedings of the ACM SIGCHI symposium on engineering interactive computing systems, pages 1–6.

Wen-Yin Chen, Pavol Podstreleny, Wen-Huang Cheng, Yung-Yao Chen, and Kai-Lung Hua. 2022. Code generation from a graphical user interface via attentionbased encoder–decoder model. Multimedia Systems, 28(1):121–130.

Yunnong Chen, Shixian Ding, YingYing Zhang, and 1 others. 2025. Designcoder: Hierarchy-aware and

self-correcting ui code generation with large language models. arXiv preprint arXiv:2506.13663.

De Chezelles, Thibault Le Sellier, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F Xu, Siva Reddy, Quentin Cappart, and 1 others. 2024. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114.

Tong Ge, Yashu Liu, Jieping Ye, Tianyi Li, and Chao Wang. 2025. Advancing vision-language models in front-end development via data synthesis. arXiv preprint arXiv:2503.01619.

Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Yi Su, Shaoling Dong, Xing Zhou, and Wenbin Jiang. 2024a. Vision2ui: A real-world dataset with layout for code generation from ui designs. arXiv preprint arXiv:2404.06369.

Yi Gui, Zhen Li, Yao Wan, and 1 others. 2024b. Webcode2m: A real-world dataset for code generation from webpage designs. arXiv preprint arXiv:2404.06369.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, and 1 others. 2025. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507.

Yilei Jiang, Yaozhi Zheng, Yuxuan Wan, Jiaming Han, Qunzhong Wang, Michael R Lyu, and Xiangyu Yue. 2025. Screencoder: Advancing visual-to-code generation for front-end automation via modular multimodal agents. arXiv preprint arXiv:2507.22827.

Hugo Laurençon, Léo Tronchon, and Victor Sanh. 2024. Unlocking the conversion of web screenshots into html code with the websight dataset. arXiv preprint arXiv:2403.09029.

Shanchao Liang, Nan Jiang, Shangshu Qian, and 1 others. 2024. Waffle: Finetuning multi-modal model for automated front-end development. arXiv preprint arXiv:2410.18362.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, and 1 others. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In International Conference on Learning Representations (ICLR).

Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. 2024. Design2code: Benchmarking multimodal code generation for automated front-end engineering. arXiv preprint arXiv:2403.03163.

Haoyu Sun, Huichen Will Wang, Jiawei Gu, Linjie Li, and Yu Cheng. 2025. Fullfront: Benchmarking mllms across the full front-end engineering workflow. arXiv preprint arXiv:2505.17399.

Yuxuan Wan, Chaozheng Wang, Yi Dong, and 1 others. 2024. Automatically generating ui code from screenshot: A divide-and-conquer-based approach. arXiv preprint arXiv:2406.16386.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, and 1 others. 2024b. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499.

Fan Wu, Cuiyun Gao, Shuqing Li, Xin-Cheng Wen, and Qing Liao. 2025. Mllm-based ui2code automation guided by ui layout information. Proceedings of the ACM on Software Engineering, 2(ISSTA):1123– 1145.

Jingyu Xiao, Yuxuan Wan, Yintong Huo, and 1 others. 2024. Interaction2code: Benchmarking mllm-based interactive webpage code generation from interactive prototyping. arXiv preprint arXiv:2411.03292.

Sukmin Yun, Haokun Lin, Rusiru Thushara, and 1 others. 2024. Web2code: A large-scale webpage-tocode dataset and evaluation framework for multimodal llms. arXiv preprint arXiv:2406.20098.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, and 1 others. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.

Ting Zhou, Yanjie Zhao, Xinyi Hou, Xiaoyu Sun, Kai Chen, and Haoyu Wang. 2024. Bridging design and development with automated declarative ui code generation. arXiv preprint arXiv:2409.11667.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, and 1 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

###### A Appendix

###### A.1 Environment HTML Synthesis

To enable scalable data generation for agent training, we construct a simulated HTML synthesis pipeline (See Figure 4) that automatically produces diverse and interactive webpage environments. The pipeline begins with a theme list (e.g., online shopping, news websites, online maps), from which a general instruction template is provided to generate task-specific prompts. Using o4-mini, we expand each general instruction into detailed naturallanguage prompts that specify the structure, layout, and interaction requirements for webpages. These detailed prompts are then fed into Claude-Sonnet-4, which generates executable HTML/CSS/JavaScript documents formulated within the React framework (hereafter referred to as HTML documents/HTML codes), enabling the construction of interactive elements such as buttons, input forms, and navigation menus. This two-stage generation process ensures semantic diversity (via high-level theme variation) and functional richness (via prompt-guided interaction synthesis). The resulting webpages form a large-scale synthetic environment for the WebVIAAgent, supporting consistent and reproducible training across varied interface types and behaviors.

###### Pipeline for HTML Document Synthesis

###### General Instruction for HTML

You are a web development expert highly sensitive to details and interaction experience, proficient in React and Tailwind CSS. Please generate a ......"

HTML Documents

General Instruction for prompt

[Figure 49]

"Please write a detailed prompt for me, which will be used to instruct a LLM to generate an interactive webpage, here are the instructions: ................ "

[Figure 50]

[Figure 51]

[Figure 52]

###### Detailed Prompts Claude-sonnet-4

Theme List O4mini

""Online Shopping", "News website", "Online maps", .............

"Please use HTML, CSS, and JavaScript to create an interactive webpage with the theme: Online Shopping ................."

........

"Please generate an interactive webpage with the theme: News Website ................."

........

Figure 4: Overview of the webpage synthesis process in the WebVIA framework.

Webpage Design Instruction Template. To guide the generation of diverse interactive webpages, we design a general instruction template that can be automatically adapted to different themes in the synthesis pipeline. Each theme (e.g., Online Shopping, News Website, Online Maps, Portfolio Page) is inserted into the template to form a specific prompt. The following template illustrates the general structure used to generate detailed webpage instructions, as shown in Figure 5.

Code Generation Prompt. To ensure functional completeness and visual consistency, we design a dedicated code generation prompt that explicitly instructs the model to generate self-contained and interaction-ready HTML code. Each generated

###### Webpage Design Instruction Template

Please write a detailed prompt that will be used to instruct a text-to-image model to generate an interactive webpage HTML code with the theme “<INSERT THEME FROM THEME LIST>”.

###### Specific Requirements:

- 1. The webpage content should revolve around the specified theme and include a wide variety of theme-related modules.
- 2. The webpage must contain multiple interactive elements, limited to buttons, input fields, and dropdown selectors. Each interactive component should cause corresponding and reasonable changes on the webpage.
- 3. The webpage content should be rich, detailed, and contextually diverse.
- 4. The output should only contain the final prompt for the AI to generate the webpage—without explanations, metadata, or additional commentary.

Figure 5: Template used to construct webpage design prompts for generating interactive webpage HTML code.

page is automatically executed and rendered in a browser environment using Playwright to verify both visual correctness and interactive functionality. Only webpages that successfully render and execute without errors are retained, ensuring that the synthesized dataset is composed of valid, executable, and behaviorally rich webpages. The following prompt defines the instruction used for generating executable HTML code, as illustrated in Figure 6.

###### A.2 Training Dataset for Exploration Agent

To train the WebVIA-Agent for robust and generalizable UI exploration, we construct a large-scale GUI interaction dataset derived from the synthetic HTML environments described in Section A.1. Each webpage instance provides a structured environment where the agent can perceive the DOM tree, rendered screenshot, and interaction history, enabling the model to learn both visual and semantic representations of interactive elements.

Automated Data Construction. To efficiently construct the training data for the WebVIA-Agent, we design an automated data generation pipeline that produces two complementary datasets: (1) Action Generation and (2) Interaction Verification. Both are generated using the o4-mini model within

###### Code Generation Prompt

You are a web development expert highly sensitive to details and interaction experience, proficient in React and Tailwind CSS. Please generate a highly interactive single-page application with reasonable layout and rich content for the specified theme according to the following requirements.

###### Basic Requirements:

- 1. Generate a complete interactive single-page website rendered using React (v18) and Tailwind CSS (v3+).
- 2. Return only the full source code wrapped within <html>...</html> tags. Do not include markdown wrappers, explanations, or code comments.
- 3. Must include the following dependencies:

<script src="https://cdn.jsdelivr.net/npm/react@18 .0.0/umd/react. development.js"></script > <script src="https://cdn.jsdelivr.net/npm/react -dom@18.0.0/umd/react -dom. development.js"></script > <script src="https://cdn.jsdelivr.net/npm/@babel/standalone/babel.js"></

script > <script src="https://cdn.tailwindcss.com"></script > <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font -

awesome /5.15.3/css/all.min.css"></link >

###### Interactivity and Functional Areas:

- 1. All interactive components (input, button, select) must trigger meaningful updates to the rendered page.
- 2. For editable content, use modals, dropdowns, or input forms with complete validation.
- 3. Use real pictures from https://picsum.photos/. Each image must have a fixed URL and remain constant across reloads.

###### Page Structure and Layout:

- 1. Include logical partitions (navigation, sidebar, main content, etc.) referencing modern app layouts.
- 2. Ensure all sections are populated; empty placeholders are not allowed.
- 3. The visual style must match the assigned theme (e.g., business, minimalism, tech, lifestyle).

###### Notes:

- • Do not output explanations or text outside the code.
- • Ensure all theme-related UI logic is complete and intuitive.

Webpage Description: <INSERT DETAILED PROMPT FROM STAGE 1>

Figure 6: Code Generation Prompt used for large-scale HTML synthesis.

the WebEnv environment, which supports both synthetic and real webpages.

For the Action Generation dataset, o4-mini serves as a general-agent and is executed once across the entire WebVIA environment, with its exploration trajectories recorded and subsequently reconstructed. The reconstruction format preserves both the historical context and the input (a paired visual–structural state, consisting of the rendered UI screenshot and its associated DOM hierarchy), while the output is represented as sequences of operations (e.g., boxed{click[1]}, boxed{enter[2][Hello World!], click[5]}).

For the Interaction Verification dataset, the pipeline executes these action sequences. Each sequence at:t+k may consist of multiple actions, where each action produces an intermediate state st+1,st+2,...,st+k. The resulting set of screenshots across these successive states is jointly considered as the post-action evidence. Formally, each interaction tuple is represented as

(st,at:t+k,{st+1,st+2,...,st+k},rt),

where rt ∈ {0,1} indicates the correctness of the overall outcome. This dual-branch data generation process yields approximately 180K verified interac-

tion samples across 20K webpages, encompassing a wide range of UI components, event bindings, and layout hierarchies.

Human-in-the-Loop Verification. To ensure data reliability, we incorporate a semi-automated verification stage, where annotators rely on rulebased checks to assess both the correctness of generated action sequences and the corresponding success labels. For action sequences, automated filtering is applied to remove cases with overly short selections or high redundancy, while inconsistent examples are corrected when necessary. Verification is handled through a mixed procedure: annotators first sample and manually inspect a subset of instances, and their findings are used to identify recurring types of outcomes that diverge from human judgment. These patterns are then formalized into rules, which guide the selective removal or adjustment of the affected cases.

Discussion. The resulting dataset unifies both action generation and interaction verification supervision, encouraging WebVIA-Agent to reason not only about what actions to take but also about their functional outcomes. Despite being primarily trained on synthetic webpages, the agent exhibits strong generalization to real-world sites, successfully handling unseen layouts, interaction patterns, and DOM structures.

A.3 Training Dataset for Interactive UI2Code Model

To enable the WebVIA-UI2Code model to generate executable and interactive HTML code, we construct the WebView dataset, which aligns multi-state UI screenshots with their corresponding groundtruth interactive webpages. Each data instance captures both the static visual layout and the dynamic behavioral transitions of a webpage, providing comprehensive supervision for learning interactionaware code generation.

Data Generation Pipeline. As illustrated in Figure 7, the construction of the WebView dataset follows a three-stage pipeline: (1) Webpage Construction. Using the HTML synthesis pipeline described in Appendix A.1, we generate a large number of interactive webpages with diverse themes (e.g., shopping, news, portfolio, dashboard). Each page contains multiple interactive elements such as buttons, input fields, dropdowns, and forms, all bound with interactive behaviors. (2) State Explo-

###### WebView Dataset Pipeline

Input WebView Data

[Figure 53]

Raw Dataset

###### Input Data

Text

###### WebVIA

| |......<br><br>Action Seq 1 Screenshots<br><br>Images<br><br>[Figure 54]<br><br>[Figure 55]<br><br>Action Seq N Screenshots<br><br>[Figure 56]<br><br>[Figure 57]| |
|---|---|---|

###### Prompt Instruction

[Figure 58]

"Please generate an interactive HTML webpage with the given UI screenshots ......"

Reorganize

Text Images

[Figure 59]

###### Action

Output

[Figure 60]

[Figure 61]

Interactions

Reasoning: User want me to generate an interactive HTML webpage with ...... Answer: <html> .... </html>

"The first action seq is ...., starting images: image 1, ending image: image 2; The second action seq is ....; ......... The N action seq is ......"

[Figure 62]

Filter

HTML

Claude-Sonnet-4

Web

###### Ground Truth

Verify Results

[Figure 63]

Action Seq1 (Click[2]): Succeed Seq2: Failed ........ SeqN: Succeed

<think>User want me to generate an interactive HTML webpage with ....</think> <answer> <html> .... </html> </answer>

[Figure 64]

Images

Verification

[Figure 65]

[Figure 66]

[Figure 67]

Reorganize

......

Figure 7: Pipeline for constructing the WebView dataset.

ration. The WebVIA-Agent interacts with each synthesized page to traverse all reachable states. During this process, it captures multi-state screenshots {I1,I2,...,In} together with corresponding DOM snapshots and event logs, reflecting visual and structural transitions triggered by user interactions. (3) Interactive Code Generation. After obtaining multi-state UI screenshots and corresponding interaction traces from the exploration stage, we adopt a multimodal instruction–response formulation to transform these visual observations into executable code. Specifically, we prompt the Claude-Sonnet-4 model to generate code under a structured reasoning format that explicitly separates the thought process and the final output using the tags <think> and <answer>. Within the <think> block, the model is encouraged to analyze the provided screenshots and interaction logs, infer component hierarchies, and reason about event dependencies. The final HTML implementation is then produced in the <answer> block, ensuring a clear delineation between reasoning and generation. This design allows the model to perform interpretable, step-by-step reasoning about webpage structure and interactivity before emitting executable code, leading to more functionally correct, visually coherent, and behavior-consistent outputs.

Prompt for Interactive Code Generation. To ensure consistent reasoning and interpretable generation during interactive code synthesis, we design a multimodal instruction prompt tailored for CLAUDE-SONNET-4, as illustrated in Figure 8.

Quality Assessment. To ensure that the synthesized interactive code is functionally executable and visually coherent, each generated HTML file is automatically rendered in a browser environment powered by Playwright. The automatic check focuses on whether the page can be successfully loaded and rendered without runtime errors. For interactive components and state transitions, we rely on sample-based human verification: annotators review a subset of interactions (e.g., clicking, text input, and selection), and their inspection con-

###### Interactive Code Generation Prompt

You are highly skilled at building interactive webpages with React and Tailwind, and can precisely reconstruct a complete HTML interactive webpage based on multiple webpage screenshots provided by the user.

###### Initial Interface Requirements:

- 1. Build the page strictly according to the first webpage screenshot provided by the user. It must be exactly the same as the first screenshot you receive.
- 2. Do not miss any details. Background colors, fonts, font sizes, spacing, borders, icons, and text must strictly match the screenshot.
- 3. Every line of text in the screenshot must be presented verbatim.
- 4. For images, please use real pictures from the https://picsum.photos/ library, with URLs like https://picsum. photos/id/.../.../.... Each image must explicitly list its URL. Do not use reusable image components. Each webpage component’s image URL must be fixed and must not be randomly regenerated each time.

###### Task Requirements:

- 1. The user will send you multiple images. Each image represents a screenshot of the webpage after a single interactive operation, and all images together represent the screenshots resulting from all operations performed on this page.
- 2. The user will send you a detailed operation-sequence list. Each item in the list represents one operation sequence and will tell you which image (by index in the images you received) is the starting image (the page before the operation), and which images correspond to the sequence of screenshots after each step in the operation. Locate these images yourself. An operation sequence may include multiple operations, i.e., it may span multiple images. Some operation sequences have many intermediate steps, but among the images only the first and the last are provided—identify the specific operation content yourself.
- 3. After locating these images, read the operation description for that item. There are three types of operations: “input,” “click,” and “select.” Correctly identify which interactive component in the screenshots corresponds to each operation, and implement them correctly in the generated HTML webpage.
- 4. All interactive operations given to you must be perfectly replicated in the generated HTML, meaning they must be fully functional, and once completed, the page must match the corresponding screenshots.

Library Requirements: <script src="https://cdn.jsdelivr.net/npm/react@18 .0.0/umd/react.development.js

"></script > <script src="https://cdn.jsdelivr.net/npm/react -dom@18.0.0/umd/react -dom.

development.js"></script > <script src="https://cdn.jsdelivr.net/npm/@babel/standalone/babel.js"></script > <script src="https://cdn.tailwindcss.com"></script > <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font -awesome

/5.15.3/css/all.min.css"></link > You may use Google Fonts. Code Output Format:

- 1. Output only the code within the complete <html></html> tags.
- 2. Do not add markdown quotes or “html” before or after the code.

Figure 8: The instruction prompt used for interactive code generation within the WEBVIA framework.

firms that the vast majority of components behave as expected. This combined strategy allows us to filter out pages that fail to render while providing evidence that the generated interactions remain largely reliable, ensuring high-quality supervision for training the WebVIA-UI2Code model.

###### A.4 UIExplore-Bench Construction

To systematically evaluate the performance of the WebVIA-Agent, we construct UIExplore-Bench, a benchmark specifically designed for assessing both fine-grained interaction reasoning and end-toend exploration performance. Unlike previous web agent datasets that primarily target task completion, UIExplore-Bench focuses on measuring the

agent’s capability to recognize UI components, execute valid interactions, and verify their functional correctness.

Benchmark Composition. UIExplore-Bench comprises three complementary subsets:

- • Action Generation Set. This subset contains 87 annotated interaction samples, each represented as paired (image, DOM tree) inputs. Each sample specifies a target action in natural language (e.g., “click the Search button” or “enter text in the input field”), enabling quantitative evaluation of the agent’s action prediction accuracy given a specific webpage state.
- • Interaction Verification Set. This subset includes 53 verification cases, each consisting of pre- and post-interaction screenshots along with DOM snapshots. The task requires the agent to determine whether the executed action produces a functionally valid change (e.g., modal opening, content update, navigation), thereby assessing its ability to reason about dynamic webpage transitions.
- • Pipeline-Level Evaluation Set. We build a larger-scale evaluation suite containing 56 complete webpages. For each webpage, the agent must autonomously explore all interactive components, generate valid action traces, and collect representative screenshots across multiple UI states. This subset evaluates the agent’s full exploration pipeline—from perception and action generation to interaction validation.

Construction Pipeline. UIExplore-Bench comprises three subsets: an Action Generation Set, an Interaction Verification Set, and a Pipeline Evaluation Set. Each subset targets a distinct dimension of GUI reasoning and follows a dedicated data construction pipeline to ensure reliability and coverage.

• Action Generation Set. This subset focuses on evaluating the agent’s ability to generate valid interaction actions from a single webpage state. We first select 100 webpages across diverse themes and employ the WebVIA-Agent to collect all possible interaction screenshots within each page. For every

UI < screenshot,DOM > pair, the leading models such as Claude-Sonnet-4, GPT-5, and Gemini-2.5-Pro +re are prompted to generate candidate interaction sequences. From each webpage, we retain the UI screenshot associated with the richest interaction set and aggregate all model-generated actions. Subsequently, human annotators manually verify the combined sequences to remove nonfunctional, ambiguous, or nonexistent interactions, resulting in a high-quality ground-truth action set.

- • Interaction Verification Set. This subset targets the evaluation of an agent’s ability to verify whether an interaction has been successfully executed. We reselect 100 webpages spanning diverse application domains and employ the WebVIA-Agent solely to collect paired pre- and post-interaction states, including screenshots and action logs. Subsequently, human annotators manually inspect and label these pairs to determine whether the executed interaction leads to a valid and functionally consistent state transition. After filtering out ambiguous or redundant cases, 53 high-quality samples are retained as groundtruth verification data, each providing a reliable reference for assessing interaction correctness.
- • Pipeline Evaluation Set. This subset is designed to evaluate the agent’s end-to-end exploration capability within complete webpage environments. We re-select 100 webpages across diverse domains and conduct autonomous exploration using the leading models such as Claude-Sonnet-4, GPT-5, and Gemini-2.5-Pro within the WebVIA framework. For each webpage, we merge all screenshots from different models and perform manual annotation to filter out invalid, overly long, or non-existent interaction elements. After this refinement process, 56 webpages are retained as high-quality ground-truth cases.

###### A.5 UIFlow2Code-Bench Construction

To systematically evaluate interactive UI-to-Code generation, we construct UIFlow2Code-Bench, a benchmark designed to assess a model’s ability to generate executable, behavior-preserving HTML code from multi-state user interface (UI) observations. Unlike existing UI2Code benchmarks such

as Design2Code (Si et al., 2024) and FullFront (Sun et al., 2025), which focus solely on static layout reconstruction, UIFlow2Code-Bench explicitly incorporates state transitions and interaction traces, enabling fine-grained evaluation of interaction-aware code synthesis.

Benchmark Composition. UIFlow2CodeBench contains 50 synthesized webpages covering diverse domains including e-commerce, news, dashboards, and portfolio sites. Each sample is composed of (1) a sequence of multi-state UI screenshots captured during interaction, and (2) the corresponding executable ground-truth HTML implementation. On average, each webpage includes 4–6 interaction states and 8–12 functional components such as buttons, modals, forms, and dropdown menus. These paired multi-view samples enable fine-grained evaluation of interaction-aware code synthesis.

Construction Pipeline. We select 100 webpages across diverse domains and employ the WebVIA framework to conduct systematic exploration using the WebVIA-Agent. For each webpage, the agent traverses all available interactive components and records corresponding multi-state UI screenshots. Human annotators then manually select 6 representative screenshots per webpage, each associated with 2 to 5 interaction actions, covering diverse visual layouts and interaction task types such as clicking, text input, and selection. These selected UI states collectively define the interaction trajectories that the model is expected to reproduce. In the evaluation phase, the generated HTML code is considered correct if it can faithfully execute the annotated actions and reproduce the corresponding state transitions, rather than merely replicating static visual appearance. This design ensures that UIFlow2Code-Bench emphasizes interaction consistency and functional correctness over superficial layout matching.

###### A.6 Baselines Versions and API Endpoints

Table 5 summarizes the specific model versions and API endpoints used for each vision-language model evaluated within the WEBVIA framework.

###### A.7 Prompts for Baseline Models

To ensure a fair comparison across all baselines, we designed unified prompt templates for the two main subtasks in the WEBVIA framework: (1) Action Generation and (2) UI2Code Translation.

Each baseline model (e.g., Claude-Sonnet, GPT-5, Gemini-2.5) was queried using the same textual instructions, with minimal format adaptation to comply with their API requirements. Temperature was fixed to 0.0 for deterministic outputs unless otherwise noted.

Action Generation Prompt This task evaluates a model’s ability to identify and describe actionable interactive elements given a static webpage representation. Specifically, the model is provided with a webpage screenshot and its corresponding DOM tree and is required to generate a set of valid user actions (e.g., clicks, text inputs, or selections) that can be performed on the interface. The complete instruction template used for this task is illustrated

- in Figure 9.

Interaction Verification Prompt. To evaluate whether a predicted interaction leads to a functionally correct state transition, the WebVIA-Agent employs a specialized verification prompt. Given a sequence of webpage screenshots before and after user actions, the model is required to determine whether the visual and structural changes align with the expected interaction outcome. This prompt guides the agent to reason about the consistency between DOM transitions and visual differences, distinguishing successful interactions (e.g., modals opening, content updates) from failed or redundant ones. The complete verification prompt is shown

- in Figure 10.

A.8 Interactive Code Generation Evaluation Prompts

To rigorously evaluate the functionality and interactivity of the generated code, we design a three-stage prompting protocol that aligns with the validation module described in Section 3.3. Each stage corresponds to a distinct phase of task-oriented execution and enables consistent benchmarking of action reasoning, process tracking, and outcome verification.

- (1) Initial Action Selection Prompt. At the beginning of each evaluation episode, the model receives a predefined task description (e.g., “search for an item,” “fill out and submit a form,” or “navigate to the contact page”) along with the initial webpage screenshot and DOM tree. As shown in Figure 11, the following prompt is used to request the model’s first interaction decision.
- (2) Process-State Action Prompt. After execut-

Table 5: Versions and official API endpoints for each evaluated vision-language model.

Model Version API Endpoint / URL Claude-Sonnet-4 claude-sonnet-4-20250514-thinking https://www.anthropic.com/api Claude-Opus-4 claude-opus-4-20250514-thinking https://www.anthropic.com/api Claude-Sonnet-3.7 claude-3-7-sonnet-20250219-thinking https://www.anthropic.com/api GPT-5 gpt-5-2025-08-07 https://platform.openai.com/docs/models o4-mini o4-mini-2025-04-16 https://platform.openai.com/docs/models GPT-4o gpt-4o-2024-11-20 https://platform.openai.com/docs/models Gemini-2.5-pro gemini-2.5-pro-preview-06-05 https://ai.google.dev/gemini-api/docs/models Gemini-2.5-flash gemini-2.5-flash-preview-05-20 https://ai.google.dev/gemini-api/docs/models

###### Action Generation Prompt

You are an interactive web assistant. I now want to check whether all interactive buttons on this webpage work properly. For example, if there is a search box on the page, please search with a reasonable query and click confirm, expecting the page to change. Note that if multiple interactive components are almost identical, please select only one of them. For example, if the page has multiple similar items each with an "Edit" button, please choose only once.

The current page state is part of the detection process. I will send you which components have already been clicked. If you find that an image was clicked before, please focus on what is different in the image I send you this time compared with the previous one. For example, if a new window has popped up, please make sure to only select interactive components in the new part. If you find that the image I send you this time is almost identical to one of the historical ones (for example, all buttons are the same, with only minor text differences), then directly reply with: “All operations on this page are completed”. Note: If two interactive buttons are not sequentially related (for example, two separate click buttons on the same page), please include only one in each boxed response, separating them. If they are sequentially related (for example, entering multiple values and then clicking confirm), please put them together in the same boxed response. Wrap your answers in LaTeX using \boxed{}.

###### Action Format:

- • click[id] = click
- • enter[id][text] = input text
- • select[id][text] = select option

Separate each action with a comma. DOM elements clicked previously: {history_info_prompt} Important: Please return only the answer! Do not include anything extra! Page Information: {domtree}

Figure 9: Prompt used for action generation in WebVIA.

ing the initial interaction, the webpage transitions into a new state. For each task branch (typically five per task), the model observes the updated screenshot and DOM tree corresponding to the current state and must decide the subsequent action. If the task is incomplete, the following process prompt is used (See Figure 12).

(3) Task Completion Verification Prompt. Once a task branch reaches termination, we verify whether the task goal has been successfully accomplished. The model receives the full textual task description and the sequence of screenshots collected during its execution. As shown in Figure 13, the following prompt is used for task-level verification.

###### A.9 Demo Cases for Exploration

To demonstrate the versatility and robustness of the WebVIA exploration process, we present qualitative examples of the agent’s behavior in both real-world and synthetic webpage environments. These cases highlight how WebVIA effectively handles complex UI layouts, multi-step operations, and dynamic visual feedback during autonomous exploration.

Synthetic Webpage Exploration. Figures 14–17 demonstrate the exploration trajectory of WebVIA-Agent on the synthetic webpages. Each figure corresponds to a distinct interaction scenario generated within our procedural environment. The agent autonomously identifies visible

###### Interaction Verification Prompt

You will receive multiple webpage images as part of the verification process for interaction history. The multiple webpage images are arranged in chronological order: the last image represents the completion of the interaction, and the first image is the starting image. A screenshot is taken after each operation until the final image completes the operation. For example, if there are only two images, then only one operation was performed: the pre-operation screenshot is the first image, and the post-operation screenshot is the second image. If there are four images, then three operations were performed: the pre-operation screenshot is the first image, after the first operation is the second image, after the second operation is the third image, and so on. Tasks:

- 1. Interaction Consistency Check: Determine whether the webpage shows changes consistent with the described interactive components after this interaction sequence. For example, clicking “Edit” should open an editing window; entering values and clicking “Save” should persist changes; clicking “Cancel” or “Close” should not. Carefully compare the final and initial images to infer whether the expected modification occurred. Reply “Yes” if changes are functionally correct, or “No” if the images remain largely unchanged. Extract your final answer and place it inside LaTeX \boxed{}, followed by your reasoning.
- 2. Continuation Check: Compare the last image with the starting image to determine whether any new significant part has appeared on the webpage. If new interactive content appears and further checking is needed, wrap “Continue” with \terminate{Continue}. If no new meaningful change is observed (e.g., no new section or trivial modifications), wrap “Complete” with \terminate{Complete}. Both boxed answers and termination tags must be output, with detailed reasoning provided afterward.

Interactive Action / Component Name: <interact_element_names>

Figure 10: Prompt used for interaction verification during the WebVIA-Agent.

interactive components such as buttons, forms, and dropdowns, and performs multi-step actions to manipulate the webpage state. Across these examples, WebVIA-Agent demonstrates its ability to perceive layout hierarchies, maintain consistency between visual and structural states, and accurately capture interaction outcomes. These exploration traces form the foundation for downstream UI2Code synthesis and interaction verification.

Real-World Webpage Exploration. Figures 18–22 present qualitative demonstrations of WebVIA-Agent exploring real-world webpages collected from open-access sites. Notably, although the agent is trained exclusively within our synthetic environment, it generalizes effectively to complex real webpages without any additional fine-tuning. It can accurately identify functional UI components—such as navigation bars, search boxes, and modal dialogs—and execute multistep interactions involving both visual reasoning and structural understanding. During exploration, the agent maintains alignment between rendered screenshots and DOM hierarchies, correctly detecting dynamic transitions. These results demonstrate WebVIA-Agent’s strong zero-shot generalization ability from procedurally generated environments to real web interfaces, validating the robustness and transferability of its visual–structural reason-

ing process. A.10 Demo Cases for Interactive Code

Generation

To further illustrate the capabilities of WebVIAUI2Code, we showcase qualitative results of interactive UI-to-Code generation. Figures 23–26 demonstrate WebVIA-UI2Code-GLM performing interactive code generation in procedurally synthesized environments. Each synthetic webpage is automatically composed of diverse UI components such as navigation bars, cards, modals, and dropdowns, each associated with predefined interaction logic. The model observes sequential webpage states during user–interface interactions and generates executable React + Tailwind code that faithfully reproduces both the visual layout and dynamic behaviors observed in the interaction flow. Furthermore, we conduct a comparative study across multiple models, where each model receives the same sequence of multiple UI screenshots as input and is tasked with generating interactive HTML code. As shown in Figures 30–29, WebVIA-UI2CodeGLM consistently produces structurally complete and functionally executable webpages, while baseline models often fail to maintain state consistency or omit interaction logic.

###### Action Selection Prompt

You are an interactive web assistant. I now want to check whether certain interactive buttons on this webpage are working properly. I will give you several tasks. You need to read the current page and select an action sequence for each task. If you think the current page content is insufficient to complete a task, please only select the interactive components on the page that can accomplish part of the task.

Wrap each of your interactive components in LaTeX \boxed{}. Action format: click[id] = click, enter[id][text] = input, select[id][text] = select option. Separate each action with a comma. Please note, id refers to the identifier of this component in the DOM tree.

At the same time, before each \boxed{}, write \task{<task name>}, and after each \boxed{}, write \state{Complete} or \state{Continue}.

Task list: {str(tasks)} Page information: {domtree}

- Figure 11: Prompt used in the Validation Module of the WebVIA framework to guide task-specific action selection.

Process-State Action Prompt

You are an interactive web assistant. I now want you to complete a task. You are currently in the detection process. Please read which buttons have already been clicked on the historical pages. Only select the buttons on the page that can actually be clicked. You should focus only on completing one task. Read the current page and select an ongoing action sequence for the current task. If you think the current page content is insufficient to complete the task, please only select the interactive components on the page that can accomplish part of the task.

Wrap your interactive components with a LaTeX \boxed{}. Action format: click[id] = click, enter[id][text] = input, select[id][text] = select option. Separate each action with a comma. At the same time, before each \boxed{}, write \task{<task name>}, and after each \boxed{}, write \state{Complete} or \state{Continue}.

Task content: {task_text} Page information: {domtree}

- Figure 12: Action execution prompt used in the Validation Module of the WebVIA framework for determining actionable components and task progress on interactive webpages.

Task Completion Verification Prompt

Provide all the screenshots along this path (in chronological order). Task: {task_text}

Please determine whether the expected webpage changes for this task have been completed.

Important: Each task name corresponds to a single interactive button or operation. For example, “New” only represents opening the new page, not saving. Thus, you only need to verify whether the new page can be opened. Only if the task explicitly specifies “New - Input ... - Save” do you need to confirm the saving step. Similarly, “Delete” only refers to opening the delete dialog, while “Delete - Confirm Delete” represents two operations—only in the latter case should you check whether the deletion was actually completed.

If the operation has been successfully completed, respond with \boxed{Yes}; if not completed, respond with \boxed{No}. Afterward, briefly explain your reasoning.

- Figure 13: Prompt used by Validation Module in the WebVIA framework to determine whether a task has been successfully completed based on sequential webpage screenshots.

Click "Cancel"

[Figure 68]

[Figure 69]

Start

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Click "Register"

Enter "Password"

Click "Sign Up"

Enter "newuser" Enter "Email"

[Figure 75]

[Figure 76]

Click "Cancel"

Click "Login"

[Figure 77]

[Figure 78]

[Figure 79]

Enter "Password"

Click "Login"

Enter "testuser"

[Figure 80]

Click "Today's Ranking"

[Figure 81]

Click "Yesterday's Ranking"

[Figure 82]

[Figure 83]

[Figure 84]

Enter "Text"

Select "Jokes" Click "Post"

[Figure 85]

Click "Reset Comment"

[Figure 86]

Click "This Week's Ranking"

[Figure 87]

Select "1150 Likes"

###### Figure 14: Exploration results of the WebVIA-Agent on synthesized web environments.

[Figure 88]

[Figure 89]

Click "Left/Right"

Start

[Figure 90]

[Figure 91]

Click "Check User Info"

Enter "User ID"

[Figure 92]

Click "Check Electricity Usage"

[Figure 93]

Select "January"

Click "Confirm Payment"

[Figure 94]

[Figure 95]

[Figure 96]

Click "Pay Now"

Enter "50"

[Figure 97]

Click "Cancel"

[Figure 98]

[Figure 99]

[Figure 100]

Click "Check Water Usage"

Select "January"

Select “Year 2024”

[Figure 101]

Select "Residential Tariff"

Click "Estimate Cost"

[Figure 102]

[Figure 103]

[Figure 104]

Enter “10"

Enter “100"

###### Figure 15: Exploration results of the WebVIA-Agent on synthesized web environments.

[Figure 105]

[Figure 106]

Select "Flavor Type"

Start

[Figure 107]

Select "Latest Share"

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Click "Submit Share" Click “Twitter”

Select "Salty"

Enter "Test Ingredient" Enter "Description"

[Figure 113]

Click "Left"

[Figure 114]

Click "Like"

[Figure 115]

Click“Facebook”

###### Figure 16: Exploration results of the WebVIA-Agent on synthesized web environments.

[Figure 116]

[Figure 117]

Click “Front Page"

Start

[Figure 118]

Click "Local Bookstalls"

[Figure 119]

Enter "Dream of the Red Chamber"

Click "Confirm Filter"

[Figure 120]

[Figure 121]

[Figure 122]

Click "All Categories"

Select "Beijing"

[Figure 123]

[Figure 124]

Click "View Details"

Click "My Favorites"

[Figure 125]

[Figure 126]

[Figure 127]

Click "Add to Favorites"

Click “Remove from Favorites"

Click "View Details"

[Figure 128]

Click "Load More Books"

[Figure 129]

Click "Reset Filter"

###### Figure 17: Exploration results of the WebVIA-Agent on synthesized web environments.

[Figure 130]

Click “Free and Open Source”

Click “Staging Area”

[Figure 131]

[Figure 132]

Start

[Figure 133]

Click “Windows Build”

Click “Pro Git book”

[Figure 134]

[Figure 135]

Click “Software Freedom Conservancy”

Click “Logos”

[Figure 136]

[Figure 137]

Click “Download for Mac”

[Figure 138]

Click “Macports”

[Figure 139]

Input “Git”

[Figure 140]

Click “Tarballs”

Figure 18: Exploration results of the WebVIA-Agent on real-world web environments.

[Figure 141]

[Figure 142]

Click “Political Geography”

Start

[Figure 143]

[Figure 144]

Click “MATH_INPUT”

Click “Set”

[Figure 145]

Click “More”

[Figure 146]

Click “Upgrade to Pro”

[Figure 147]

[Figure 148]

Click “delete"

Input "√2"

[Figure 149]

Click“Sign in”

Figure 19: Exploration results of the WebVIA-Agent on real-world web environments.

[Figure 150]

[Figure 151]

[Figure 152]

Click “SQL_Syntax”

Click “ATACH DATABASE”

Start

[Figure 153]

Click “small”

[Figure 154]

Click “SQLite Consortium”

[Figure 155]

[Figure 156]

Click “Transactions”

Click “Features”

[Figure 157]

Click“SQL Functions”

[Figure 158]

Click“Download”

[Figure 159]

Click “Source Code”

[Figure 160]

Click “TCL Interface”

###### Figure 20: Exploration results of the WebVIA-Agent on real-world web environments.

[Figure 161]

[Figure 162]

Click “Rad”

Start

[Figure 163]

[Figure 164]

Click “Search”

Input “test”

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Click “Sign in”

Input “Email” Input “Password” Click “Sign In”

[Figure 169]

Click “Forget password”

[Figure 170]

Click “About us”

[Figure 171]

Click “Create a free account”

[Figure 172]

[Figure 173]

Click “Financial calculator”

Input “test”

[Figure 174]

Click “Mortgage calculator”

[Figure 175]

Click “Sitemap”

###### Figure 21: Exploration results of the WebVIA-Agent on real-world web environments.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Click “Sign Up” Input “Email”

Click “Next”

Start

[Figure 180]

[Figure 181]

[Figure 182]

Click “Log In” Input “Email” Click “Next”

[Figure 183]

[Figure 184]

Click “Files”

Click“Examples”

[Figure 185]

[Figure 186]

Input “0”

Click“Learn More”

[Figure 187]

Click “Questions”

[Figure 188]

Click“Languages”

[Figure 189]

Click “Apps”

[Figure 190]

Click “Add”

[Figure 191]

Click “Share”

###### Figure 22: Exploration results of the WebVIA-Agent on real-world web environments.

## UI2Code Inputs

## Interactive HTML Code Rendered Results

Start

Code Thumbnail

Start

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Click “Editing Area”

Click “Editing Area”

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Input “Testing”

Input “Testing”

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Input Text

Input Text

[Figure 205]

[Figure 206]

Figure 23: Rendered UI2Code demo for WebVIA-UI2Code-GLM

###### UI2Code Inputs

###### Interactive HTML Code Rendered Results

Start

Start

Code Thumbnail

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Click “Add Recipe”

Click “Add Recipe”

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Modify Name

Modify Name

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Choose “Staple Food”

Choose “Staple Food”

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Modify Data

Modify Data

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Modify Remarks

Modify Remarks

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Save

Save

[Figure 232]

[Figure 233]

Figure 24: Rendered UI2Code demo for WebVIA-UI2Code-GLM

##### UI2Code Inputs

##### Interactive HTML Code Rendered Results

Start

Code Thumbnail Start

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Delete

Delete

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Add Achievements

Add Achievements

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Input Text

Input Text

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Save

Save

[Figure 251]

[Figure 252]

Figure 25: Rendered UI2Code demo for WebVIA-UI2Code-GLM

#### UI2Code Inputs

#### Interactive HTML Code Rendered Results

Start

Code Thumbnail Start

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Edit Planning

Edit Planning

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Data Statistics

Data Statistics

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Help Center

Help Center

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Add Training Plan

Add Training Plan

[Figure 270]

[Figure 271]

Figure 26: Rendered UI2Code demo for WebVIA-UI2Code-GLM

UI2Code Inputs

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

WebVIA-UI2Code Render

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Gemini2.5 Pro Render

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

O4-mini Render

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

- Figure 27: Comparison of WebVIA-UI2Code and baseline renders on the same interaction trace.

UI2Code Inputs

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

WebVIA-UI2Code-GLM Render

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

WebVIA-UI2Code-Qwen Render

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Claude-Sonnet-4 Render

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

- Figure 28: Comparison of WebVIA-UI2Code and baseline renders on the same interaction trace.

UI2Code Inputs

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

WebVIA-UI2Code-GLM Render

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

WebVIA-UI2Code-Qwen Render

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Gemini2.5 Flash Render

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

GPT-5 Render

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Figure 29: Comparison of WebVIA-UI2Code and baseline renders on the same interaction trace.

UI2Code Inputs

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

WebVIA-UI2Code-GLM Render

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

WebVIA-UI2Code-Qwen Render

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Gemini2.5 Pro Render

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

o4-mini Render

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Figure 30: Comparison of WebVIA-UI2Code and baseline renders on the same interaction trace.

