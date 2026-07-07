# arXiv:2602.09007v2[cs.AI]10Feb2026

[Figure 1]

## GEBench: Benchmarking Image Generation Models as GUI Environments

Haodong Li1,2, Jingwei Wu1, Quan Sun1,†, Guopeng Li1, Juanxi Tian7, Huanyu Zhang5, Yanlin Lai1,4, Ruichuan An3, Hongbo Peng1, Yuhong Dai1, Chenxi Li6, Chunmei Qing2,∗, Jia Wang1, Ziyang Meng1, Zheng Ge1,∗, Xiangyu Zhang1, Daxin Jiang1 1 StepFun 2 South China University of Technology 3 Peking University 4 Tsinghua University 5 Institute of Automation, Chinese Academy of Sciences 6 The University of Chicago 7 Nanyang Technological University † Project Leader ∗ Corresponding Author

### Abstract

Recent advancements in image generation models enable the prediction of future Graphical User Interface (GUI) states based on user instructions. However, existing benchmarks primarily focus on general domain visual fidelity, leaving evaluation of state transitions and temporal coherence in GUI-specific contexts underexplored. To address this gap, we introduce GEBench, a comprehensive benchmark for evaluating dynamic interaction and temporal coherence in GUIs generation. GEBench comprises 700 carefully curated samples spanning five task categories, covering both single-step interactions and multi-step trajectories across real-world and fictional scenarios, as well as grounding point localization. To support systematic evaluation, we propose GE-Score, a five-dimensional metric that assesses Goal Achievement, Interaction Logic, Content Consistency, UI Plausibility, and Visual Quality. Extensive evaluation indicates that current models perform well on single-step transitions but struggle with temporal coherence and spatial grounding over longer interaction sequences. Moreover, our findings identify icon interpretation, text rendering, and localization precision as key bottlenecks, and suggest promising directions for future research toward high-fidelity generative GUI environments. The code is available at: https://github.com/stepfun-ai/GEBench

##### 1. Introduction

Recent advancements in image generation models (Comanici et al., 2025; Google, 2025b; Hurst

- et al., 2024; Labs, 2025; Seedream et al., 2025; Wan et al., 2025) enable the prediction of future Graphical User Interface (GUI) states based on specific user instructions and current visual contexts. This capability positions image generation models as powerful GUI Environments (Garg
- et al., 2025; Luo et al., 2025; Wei et al., 2023; Xie et al., 2025b; Yan et al., 2025; Zhang et al., 2025a), capable of simulating dynamic interaction sequences to facilitate the scalable training of autonomous agents. Distinguished from conventional simulators (Bonatti et al., 2024; Cobbe et al., 2020; Xie et al., 2024) tethered to physical hardware or fixed software stacks (Zhang et al., 2025a), these generative models offer a flexible, low-cost alternative for creating diverse interaction trajectories across countless applications (Liu et al., 2025b; Zhao et al., 2021).

However, the potential of image generation models as reliable GUI environments remains largely unverified, as traditional visual benchmarks (Ghosh et al., 2023; Hu et al., 2024; Huang et al., 2023, 2024; Niu et al., 2025; Sun et al., 2025; Zhao et al., 2025; Zhuang et al., 2025) prioritize general-domain visual fidelity (for images) and continuous state transitions (for videos), leaving

[Figure 2]

[Figure 3]

[Figure 4]

GUI Environment Benchmark (Ours)

Video Generation Benchmarks

Traditional T2I Benchmarks

###### Single Image Multi Images

###### Video Frames

State 1 State 2 State 3

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

˂ PhoneDetail

[Figure 10]

Settings

[Figure 11]

Search

[Figure 12]

Phone Name

Version 1.6 Memory 12.0 GB

[Figure 13]

Wi-Fi

Battery capacity

4610mAH

[Figure 14]

Bluetooth

Resolution

[Figure 15]

2670*1200

Notification

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

˂

Factory reset

App Store Safari Camera Settings

[Figure 21]

[Figure 22]

Detail

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Image 1 Image 2 Image 3

User Action

User Action

[Figure 28]

[Figure 29]

[Figure 30]

General-domain Visual Fidelity

###### Discrete State Transitions

Continuous State Transitions

[Figure 31]

[Figure 32]

Main Challenges

[Figure 33]

Semantic Interpretation of Icons Refining text rendering

[Figure 34]

Improving grounding point localization

- Figure 1 | Comparison of evaluation paradigms across different benchmark types. Existing image generation benchmarks prioritize general-domain visual fidelity and video generation benchmarks evaluate continuous state transitions. GEBench uniquely evaluates discrete state

transitions induced by user actions, capturing the essence of GUI interactions.

a critical gap in evaluating the functional logic and state-transition consistency inherent to GUI interactions (Xie et al., 2025a; Yan et al., 2025). As shown in Figure 1, when acting as GUI environments (Zhang et al., 2025a), generation models must seamlessly navigate discrete, actiontriggered interface jumps. Such transitions necessitate precise coordinate grounding (Cheng

- et al., 2024; Zhao et al., 2021), icon recognition (Liu et al., 2025b; Xie et al., 2025a), and high-fidelity text rendering (Chen et al., 2024), compelling the models to maintain logical continuity even when visual elements do not persist (Li et al., 2025a). Such demands strain existing architectures and call for a new evaluation approach to verify if generated GUIs respond felicitously to user instructions.

To bridge this gap, we present GEBench (Benchmarking image generation models as GUI Environments), a benchmark designed to evaluate how effectively image generation models can serve as GUI environments. GEBench comprises 700 high-quality samples, where each entry aligns a sequence of GUI images with corresponding user instructions. These samples span five distinct tasks, allowing for a multifaceted assessment of the model’s ability. To provide a concrete measure of generative quality, we propose GE-Score, a multi-dimensional metric derived from Vision Language Model (VLM)-guided (Bai et al., 2025; Google, 2025a; Hurst

- et al., 2024) evaluations across five specialized rubrics. GE-Score systematically validates intent fulfillment and interaction logic while verifying UI content consistency and structural integrity. By ensuring high visual fidelity and logical coherence, GE-Score confirms the practical utility of these synthetic environments.

Our systematic evaluation of state-of-the-art image generation models (Deng et al., 2025;

Google, 2025b; Labs, 2025; Li et al., 2025b; OpenAI, 2025; Seed, 2025; Seedream et al., 2025; Team

- et al., 2025a; Wan et al., 2025; Wu et al., 2025) identifies promising pathways for their evolution into reliable GUI environments. While current architectures demonstrate robust proficiency in executing localized, single-step state transitions, they offer significant opportunities for advancement in long-term interaction consistency and precise spatial grounding. In particular, deficiencies in icon interpretation, Chinese text rendering, and grounding point localization lead to layout drift and logical inconsistencies. These observations delineate critical bottlenecks and outline clear directions for future research toward high-fidelity, temporally coherent generative GUI systems.

Our primary contributions are as follows:

- 1. We introduce GEBench, a systematic benchmark with 700 samples across five task categories to evaluate image generation models as dynamic GUI environments.
- 2. We propose GE-Score, a five-dimensional metric that emphasizes the quality assessment of image sequences by accounting for the unique visual properties of GUIs.
- 3. Our evaluation reveals critical deficiencies in current image generation models, underscoring significant room for improvement in high-fidelity GUI generation.

##### 2. Related Work

###### 2.1. Automated GUIs Generation

The evolution of GUIs generation reflects a significant paradigm shift from heuristic-based structural mapping to data-driven synthesis powered by Multimodal Large Language Models (MLLMs) (Chen et al., 2018; Kolthoff et al., 2024, 2025; Li et al., 2020; Mozaffari et al., 2022; Sandhaus et al., 2011; Sobolevsky et al., 2023; Yang et al., 2016; Zhang et al., 2025b; Zhao et al., 2021). Early methodologies relied on traditional rule-based algorithms to perform layout reconstruction (Huang et al., 2016; Sandhaus et al., 2011), yet these approaches frequently failed to capture the semantic depth of complex hierarchies. Subsequent frameworks simplified this process using model-based approaches to translate visual features directly into code sequences (Chen et al., 2018). Contemporary research leverages Transformer-based architectures to bridge the gap between visual design abstractions and executable source code (Kolthoff et al., 2025; Sobolevsky et al., 2023). Furthermore, the rapid advancement of generative AI suggests that direct utilization of image generation models for GUI synthesis is becoming increasingly viable (Li et al., 2020; Mozaffari et al., 2022; Zhang et al., 2025c; Zhao et al., 2021). These models offer the potential to produce high-fidelity GUIs directly from user instructions.

###### 2.2. Advanced Image Generation Models

Recent progress in image generation exhibits a rapid evolution from text-to-image synthesis to sophisticated reference-based frameworks (Comanici et al., 2025; Deng et al., 2025; Google, 2025b; Hurst et al., 2024; Li et al., 2025b; Liu et al., 2025a; Seedream et al., 2025; Team et al.,

- 2025a,b; Wan et al., 2025). Ongoing advancements in text-to-image synthesis have empowered models to produce aesthetically superior visuals with precise semantic alignment to the provided instructions (Chen et al., 2020; Fan et al., 2024; Han et al., 2025; Ho et al., 2020; Labs, 2025; Lin et al., 2025; Ramesh et al., 2022). Building on these foundations, reference-based techniques integrate visual priors with textual prompts to enhance generative control (An et al., 2025; Google, 2025b; Seedream et al., 2025; Team et al., 2025a; Wan et al., 2025). These methods incorporate style or structural references to ensure spatial precision and identity consistency (Deng et al., 2025; Liu

- et al., 2025a). These advances enable image generation models to function as interactive GUI environments.

###### 2.3. Sequential Generation Benchmarks

Standard image generation benchmarks focus on visual fidelity and text-alignment for a single image, using metrics like FID and CLIP score to measure visual quality (Ghosh et al., 2023; Heusel et al., 2017; Huang et al., 2023; Radford et al., 2021). While these metrics provide a robust baseline for aesthetic realism (Heusel et al., 2017), they set the stage for incorporating logical coherence to achieve a more holistic evaluation. Recent efforts in benchmarking sequential image generation explore temporal consistency and reasoning (Guo et al., 2025; Hu et al., 2024; Huang et al., 2023; Niu et al., 2025; Zhang et al., 2026; Zhao et al., 2025; Zhuang et al., 2025), yet these frameworks typically target natural scenes with continuous movement, simple spatial relationships or characters identity (Liu et al., 2025a). GUI environments differ fundamentally because they involve discrete state jumps where a single action replaces the entire visual layout (Yan et al., 2025; Zhang et al., 2025a). Furthermore, the stringent text-rendering requirements of GUIs push current generative architectures to their limits (Chen et al., 2024). This necessitates a new evaluation approach to bridge the significant gap in assessing whether image generation models can maintain the strict semantic and structural integrity required for multi-step GUI trajectories generation.

##### 3. GEBench

A GUI environment functions as an interactive medium that translates user instructions or agent actions into corresponding visual feedback through software logic (Zhang et al., 2025a). This mechanism allows a system to simulate the evolution of a digital interface in response to user interventions. The highly structured and rule-based composition of GUIs, governed by precise functional logic, distinguishes these environments from the patterns of natural scenes.

GEBench establishes an evaluation framework that treats image generation models as interactive GUI environments and benchmarking their performances. Under this paradigm, the model receives visual observations of the current GUI state along with specific user instructions to synthesize the subsequent state. In the following sections, we use terms “GUI state" and “GUIs" to refer to image-based inputs and outputs of image generation models.

###### 3.1. Benchmark Design and Task Suites

GEBench comprises 700 high-quality interaction sequences curated under strict consistency and fidelity constraints. As shown in Figure 2, by organizing samples into five task categories, GEBench enables a fine-grained evaluation of model capabilities across multiple dimensions of GUIs generation:

- 1. Single-step Visual Transition (single-step) The model receives an initial GUI state as reference image and a detailed action specification to generate the subsequent GUI state. This task evaluates fine-grained instruction following of the models.
- 2. Multi-step Planning (multi-step) Starting from an initial GUI state as reference and a high-level user objective such as “Order a coffee", the model must generate a five-step GUIs trajectory. This task assesses long-horizon planning, temporal coherence, and the consistency of UI structure across multiple steps.

200 Samples 200 Samples

###### Multi-Step

###### Single-Step

Phone Phone Computer

Computer

|[Figure 35]<br><br>1|
|---|

|[Figure 36]|
|---|

| |[Figure 37]<br><br>5 St|[Figure 38]<br><br>[Figure 39]<br><br>ep|[Figure 40]<br><br>s| |
|---|---|---|---|---|

[Figure 41]

|--q1` 1 Step<br><br>[Figure 42]|
|---|

|[Figure 43]|
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

5 Steps

Step

p

--q1`

###### User instruction：

###### User instruction：

The first screenshot shows a cluttered Windows desktop. In the second, the Zoom client window is open with options like Join, Start, and Schedule, indicating the app was launched.

“The initial April 1 view shows no data. Navigating back reveals the dashboard with zero activity and a timeline graph.”

User instruction: "How do I create an account?"

User instruction: "What's on the menu at Papa Murphy's?"

100 Samples 100 Samples

Fictional App Real App

###### Phone Computer

Phone Computer

|[Figure 49]<br><br>[Figure 50]|[Figure 51]<br><br>5|Steps|[Figure 52]|
|---|---|---|---|

| |[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>5 Ste|p|[Figure 56]<br><br>s| |
|---|---|---|---|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

5 Steps

5 Steps

--q1`

--q1`

User instruction： "Create a recurring bill reminder with notifications."

User instruction： "Book an appointment and add it to the calendar." User instruction：

User instruction： "Download an area for offline use."

"Share your location for one hour."

100 Samples

Grounding

Phone Phone Computer Computer

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]<br><br>1|
|---|

|[Figure 68]|
|---|

|[Figure 69]<br><br>[Figure 70]<br><br>1 St|[Figure 71]<br><br>ep|
|---|---|

|--q1`<br><br>[Figure 72]<br><br>1|
|---|

|[Figure 73]|
|---|

[Figure 74]

[Figure 75]

[Figure 76]

Step 1 Step Step

User instruction： “Given a grounding point [938, 61], the model predicts the subsequent frame transition after a click is performed at that point."

User instruction： "Given a grounding point [866, 80], the model predicts the subsequent frame transition after a click is performed at that point."

User instruction： "Given a grounding point [940, 40], the model predicts the subsequent frame transition after a click is performed at that point."

User instruction： "Given a grounding point [119, 78], the model predicts the subsequent frame transition after a click is performed at that point."

- Figure 2 | Examples of the five task types in GEBench, which are designed to comprehensively evaluate the capabilities of image generation models as GUI environments. GEBench provides image generation models with user instructions and reference GUI state (no reference provided

for the Fiction App task) and evaluates the generated GUIs.

- 3. Zero-shot Virtual GUI (fiction-app) This task evaluates out-of-distribution generalization of models, testing whether they can generate unseen layouts by relying on detailed instructions without external reference.
- 4. Rare Trajectory Synthesis (real-app) This task evaluates the model’s ability to generate long-tail interaction trajectories by prioritizing logical reasoning over pattern imitation, particularly in data-scarce scenarios.
- 5. Grounding-based Generation (grounding) The model generates the next GUI state based on a normalized relative coordinates within the range of [0,1000]. This task assesses spatial awareness and the ability to render changes at precise pixel locations.

Unless otherwise indicated, these five task types are subsequently referred to by their respective parenthetical abbreviations for brevity.

###### 3.2. Evaluation Dimension and Scoring Rubric

GEBench adopts a multi-dimensional scoring rubric designed to assess model abilities in GUIs generation setting. Rather than relying on a single correctness signal, we decompose model performance into five complementary dimensions that jointly capture functional accuracy, interaction realism, and visual fidelity. This design enables fine-grained and interpretable

[Figure 77]

[Figure 78]

Frame 0 → Frame 1 → ...

###### 5 task types

[Figure 79]

Rule-based Pre-processing

[Figure 80]

Click Settings icon

[Figure 81]

[Figure 82]

Recording

[Figure 83]

Single-Step

[Figure 84]

Expert-driven Verification

or

200 Samples

[Figure 85]

[Figure 86]

[Figure 87]

###### Multi-Step

Scroll down

Recording

200 Samples

[Figure 88]

Statistical Calibration

###### Fictional App

100 Samples

[Figure 89]

[Figure 90]

###### Real App

[Figure 91]

[Figure 92]

100 Samples

7 0 0

###### Data Collection Grounding

100 Samples

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Raw Data Capture

Task Annotation

Dataset Construction

Quality Control

- Figure 3 | GEBench data construction pipeline. The process involves raw data capture through recording user interactions, task annotation of actions, quality control via preprocessing and

verification, and data construction across five task categories: Single-Step, Multi-Step, Grounding, Real App, and Fictional App, totaling 700 samples.

comparisons across all 5 task types. Specifically, model generated GUI states are evaluated along the following five dimensions:

- • Goal Achievement (GOAL) assesses whether the generated GUI state satisfies the specified action or global objective, focusing on the correctness and completeness of the intended outcome.
- • Interaction Logic (LOGIC) evaluates the plausibility and coherence of state transitions with respect to realistic GUI interaction patterns, ensuring that visual changes can be explained by valid user actions.
- • Consistency (CONS) measures the preservation of unaffected regions within a single image or the stability of UI elements across multiple GUIs, reflecting resistance to unintended visual drift.
- • UI Plausibility (UI) examines whether generated GUI state components are structurally coherent, native-looking, and free from hallucinated or physically impossible elements.
- • Visual Quality (QUAL) evaluates the perceptual fidelity of the generated GUIs, including text readability, icon clarity, and the absence of rendering artifacts.

All dimensions are scored on a discrete ordinal scale from 0 to 5, where higher scores indicate stronger alignment with expected GUIs behavior and visual realism. Detailed scoring criteria for each dimension are provided in the appendix.

We synthesize these multi-dimensional assessments into GE-Score, a holistic metric reflecting the aggregate performance. For each i-th sample evaluated on the d-th dimension, let 𝑟𝑖,𝑑 ∈ {0, . . . ,5} represent the discrete fidelity score. The GE-Score is formally defined as

∑︁

∑︁𝑁

###### ∑︁5

###### ∑︁𝑁

4

1 |D| · 𝑁

𝑟𝑖,𝑑 (1)

F (𝑟𝑖,𝑑) =

𝐺𝐸 𝑠𝑐𝑜𝑟𝑒 =

𝑁

𝑖=1

𝑑=1

𝑖=1

𝑑∈D

Table 1 | Main evaluation results on GEBench across Chinese and English Subsets. The table presents a performance comparison across five core dimensions involving 8 commercial models and 4 open-source models. Orange and Champagne cells indicate the Top 1 and Top 2 performers respectively.

Chinese Subset English Subset

Model

single multi fiction real ground GE single multi fiction real ground GE

-step -step -app -app -ing Score -step -step -app -app -ing Score

Nano Banana pro (Google, 2025b) 84.50 68.65 65.75 64.35 64.83 69.62 84.32 69.51 46.33 47.20 58.64 61.20 Nano Banana (Google, 2025b) 64.36 34.16 64.82 65.89 54.48 56.74 64.80 50.75 48.88 47.12 49.04 52.12 GPT-image-1.5 (OpenAI, 2025) 83.79 56.97 60.11 55.65 53.33 63.22 80.80 58.87 63.68 58.93 49.23 63.16 GPT-image-1.0 (OpenAI, 2025) 64.72 49.20 57.31 59.04 31.68 52.39 60.92 64.33 58.94 56.16 37.84 55.64 Seedream 4.5 (Seed, 2025) 63.64 53.11 56.48 53.44 52.90 55.91 49.49 45.30 53.81 51.80 49.63 50.01 Seedream 4.0 (Seedream et al., 2025) 62.04 48.64 49.28 50.93 53.53 52.88 53.28 37.57 47.92 49.36 44.17 46.46 Wan 2.6 (Wan et al., 2025) 64.20 50.11 52.72 50.40 59.58 55.40 60.17 44.36 49.55 44.80 53.36 50.45 Flux-2-pro (Labs, 2025) 68.83 55.07 58.13 55.41 50.24 57.54 61.00 52.17 49.92 47.16 45.67 51.18

Bagel (Deng et al., 2025) 34.84 13.45 27.36 33.52 35.10 28.85 32.91 8.61 26.08 35.12 37.30 28.00 UniWorld-V2 (Li et al., 2025b) 55.33 24.95 32.03 21.39 49.60 36.66 42.68 14.14 30.08 26.83 47.04 32.15 Qwen-Image-Edit (Wu et al., 2025) 41.12 26.79 23.78 26.10 50.80 33.72 40.12 18.61 25.80 25.95 54.55 33.01 Longcat-Image (Team et al., 2025a) 48.76 12.75 30.03 30.00 51.02 34.51 36.69 8.44 37.30 36.83 47.12 33.28

where F (𝑟) = 20 × 𝑟 represents the linear normalization transform into percentage domain [0,100]. This formulation effectively captures the mean semantic-structural alignment of generation models across the entire benchmark distribution.

- 3.3. Data Construction Pipeline

The construction of GEBench data follows a structured pipeline designed to transform raw screen recordings into high-quality trajectories for GUI-based interaction. As Figure 3 illustrates, the process begins with the collection of raw interaction data through screen recordings on both mobile and desktop platforms. During the task annotation phase, annotators define specific actions, such as clicking icons or scrolling through interfaces, and convert these sequences into structured JSON metadata.

To further improve data quality, we incorporate a three-stage quality control mechanism. First, a rule-based preprocessing step automatically filters out inconsistent or noisy samples. Second, human experts manually verify the remaining sequences to ensure that the annotated actions accurately match the visual state transitions. Finally, a statistical calibration process adjusts the data distribution to mitigate potential biases, resulting in a final collection of 700 refined samples, which are categorized into five types of tasks, as mentioned in Section 3.1.

- 4. Evaluation

- 4.1. Evaluation Setup

Evaluated Models. Our experimental evaluation encompasses 12 image generation models, grouped into two categories based on accessibility: 8 commercial models and 4 open-source models.

The commercial model group includes Nano Banana pro (Google, 2025b), Nano Banana (Google,

- 2025b), GPT-image-1.5 (OpenAI, 2025), GPT-image-1 (OpenAI, 2025), Seedream 4.5 (Seed, 2025), Seedream 4.0 (Seedream et al., 2025), Wan2.6 (Wan et al., 2025), Flux-2-pro (Labs, 2025).

[Figure 97]

- Figure 4 | Performance of models across GEBench task suites. The radar chart illustrates the performance of 12 prominent image generation models, including commercial models (solid

line) and open-sourced models (dashed line). The reported results represent the average scores on Chinese and English subsets.

The open-source model group includes Bagel (Deng et al., 2025), UniWorld-V2 (Li et al., 2025b), Qwen-Image-Edit (Wu et al., 2025), LongCat-Image (Team et al., 2025a). VLM-based Judges. To ensure the objectivity and robustness of GEBench, we deploy 3 state-ofthe-art VLMs as independent cross-evaluators: 2 commercial models Gemini-3-Flash-Native (Google, 2025a), GPT-4o (Hurst et al., 2024) and 1 open-source model Qwen3-vl-235b-a22b-thinking (Bai et al., 2025). By utilizing these evaluators, we mitigate potential bias inherent in a single judge model. To ensure fair and reproducible comparisons, we use official default configurations for evaluated models and perform evaluation three times for each generated GUIs trajectory.

###### 4.2. Evaluation Results

Overall Performance and Model Comparison. Experimental results in Table 1 show that Nano Banana Pro (Google, 2025b) delivers the most robust performance, particularly on Chinese subset with a top-ranking GE-Score of 69.62. GPT-image-1.5 (OpenAI, 2025) follows closely, excelling on English subset and securing the first position with a score of 63.16. The radar chart, as shown in Figure 4, further illustrates that commercial models, led by Nano Banana Pro (Google, 2025b), exhibit a balanced and “full" pentagonal profile. In contrast, open-source

[Figure 98]

- Figure 5 | Comparison of GOAL score on grounding task. The universally low scores across all models highlight a critical deficiency in current generative models’ ability to perceive and align

with precise spatial grounding points.

models (e.g., UniWorld-V2 (Li et al., 2025b), Bagel (Deng et al., 2025)) show performance curves that significantly shrink inward, revealing a substantial gap in handling complex tasks.

The Performance Gap in Multi-step Planning. The evaluation results in Table 1 highlights a critical bottleneck: while most models excel in Single-step transitions, with both Nano Banana Pro (Google, 2025b) and GPT-image-1.5 (OpenAI, 2025) exceeding 80 points, their scores plummet in Multi-step Planning scenarios, generally dropping below 60 or even 10 points. The radar chart, as shown in Figure 4, clearly identifies the Multi-step axis as a general weakness across nearly all models.

This phenomenon underscores current limitations in long-horizon logical planning:

- • Imbalance between Perception and Planning: Models possess strong instruction-following capabilities for single action mapping but struggle to maintain logical consistency across long sequences.
- • Error Accumulation: During multi-step transitions, minor visual deviations in intermediate steps snowball over time. This accumulated error eventually causes the generated trajectory to diverge entirely from the intended goal.
- • Deficiency in Visual-level Reasoning: Despite their ability to reason within complex textual spaces, models fail to logically grasp inter-step dependencies in interconnected visual tasks, making it difficult to predict the impact of current actions on subsequent visual states.

Challenges in Spatial Grounding. Cross-metric analysis reveals a general deficit in Groundingbased Generation tasks. As shown in Figure 5, the GOAL (Goal Achievement) score is remarkably low: even top-performing Nano Banana Pro (Google, 2025b) achieves only 23.9%, while most other models (e.g., Qwen-Image-Edit (Wu et al., 2025)) fall below 10%. The pronounced “dent" on the Grounding axis of the radar chart, as in Figure 4, further validates this bottleneck: The dismal GOAL scores indicate a logical disconnect; models identify what to generate but cannot translate this into where to place it on a precise [0,1000] coordinate grid. This suggests models lack a fundamental understanding of the mapping between abstract coordinates and image pixels.

[Figure 99]

Figure 6 | Pearson correlation analysis between human expert scores and VLM-based evaluation. Results for Nano Banana Pro (Google, 2025b) and GPT-Image-1 (OpenAI, 2025) demonstrate a strong alignment between the VLM-as-a-Judge framework and human judgment across different models.

###### 4.3. Validity of VLM-as-a-Judge

To validate the reliability of using VLM as evaluators, we analyze the correlation between VLM-based assessments and human expert judgments. Specifically, we randomly sample results from two representative models, Nano Banana Pro (Google, 2025b) and GPT-Image-1 (OpenAI, 2025). For each model, we select 10 edited samples from each of the 10 tasks, resulting in 100 evaluated samples per model. Four human experts independently assess all selected samples using the same evaluation criteria and metrics as the VLM-based judges. Human scores are obtained by averaging all scores across four experts. We then compute the Pearson Correlation Coefficient between the human-annotated scores and the scores produced by the VLM-based judges.

As shown in Figure 6, our VLM-based evaluations exhibit strong correlation with human judgments. The overall Pearson correlation coefficient across all samples reaches r = 0.9892. When analyzed by model, the correlation remains consistently high, with r = 0.9926 for Nano Banana Pro (Google, 2025b) and r = 0.9833 for GPT-Image-1 (OpenAI, 2025). These results indicate a high level of agreement between the VLM evaluator and human experts across different tasks and models. Together, this analysis demonstrates that the proposed VLM-as-aJudge framework provides reliable and human-aligned evaluations across diverse tasks and models.

###### Weakness

###### Instruction Input Image Nano Banana Pro Qwen Image Bagel

|[Figure 100]<br><br>| |
|---|
|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

[Figure 103]

|[Figure 104]|
|---|

[Figure 105]

[Figure 106]

User instruction: “After confirming the departure time, proceed to the next step to continue the booking workflow, navigate through the interface, and enter the passenger selection stage.”

Text Rendering

###### Weakness

Instruction Input Image Nano Banana Pro Seedream 4.5 Flux2 Pro

|[Figure 107]|
|---|

|[Figure 108]<br><br>| |
|---|
|
|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

User instruction: “Type the icon ‘W’ to trigger the interface update, allow the interface to respond and update accordingly, and predict the next frame that follows from this updated screen state.”

Icon Interpretation

Weakness

###### Instruction Input Image Nano Banana Pro GPT Image Seedream 4

User instruction: “Given a grounding point [100, 80], the model predicts the subsequent frame transition after a click is performed at that point.”

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

[Figure 117]

|[Figure 118]|
|---|

[Figure 119]

[Figure 120]

Localization Precision

Grounding point explanation: “Click the search button to enter the search interface.”

Figure 7 | Qualitative results of the three primary weaknesses identified in image generation models acting as GUI environments. The comparison highlights significant deficiencies in text rendering accuracy, icon interpretation for state transitions, and localization precision regarding coordinate-based grounding.

##### 5. Discussion and Analysis

###### 5.1. Hierarchy of Task Difficulty: From Local Mimicry to Global Reasoning Failure

The experimental results, as shown in Table 1 and Figure 4, reveal a pronounced inverse correlation between task complexity and model performance, highlighting a fundamental deficiency in current models’ deep understanding of GUI mechanics. In single-step transition scenarios, leading models (Google, 2025b; OpenAI, 2025) achieve robust scores exceeding 80% through powerful visual synthesis. However, this “illusory prosperity" is largely driven by a shortcut mapping from instructions to visual patterns; rather than mastering the underlying interactive logic, models primarily leverage statistical distribution fitting to match expected GUI states. This structural weakness becomes evident as tasks extend to multi-step trajectories. The significant inward “shrinkage" of the radar chart demonstrates an inability to maintain temporal coherence, where the lack of explicit state-space logic leads to a severe logical disconnect when processing high-complexity interaction flows.

###### 5.2. In-depth Bottleneck Analysis: Qualitative Insights from Failure Cases

By synthesizing qualitative evidence, as shown in Figure 7, we identify three primary technical bottlenecks that hinder reliable GUIs generation. The first is the failure of text rendering accuracy. Qualitative cases reveal that while models like Nano Banana Pro (Google, 2025b) encounter deformations in complex layouts, open-source models frequently exhibit severe character overlapping and semantic corruption. This suggests that models treat text as a local texture rather than a symbolic unit with rigid structural information, leading to unreadable

characters in layout-dense environments without hard topological constraints.

Secondly, icon interpretation and consistency remain major barriers. Models exhibit significant difficulty in the semanticization of visual symbols, often failing to recognize correct interactive boundaries even when instructions target specific icons. This instability results in “functional distortion" during state transitions, where a specific trigger may degenerate into a meaningless geometric shape. Such a severance of interaction intent and affordance renders interaction entry points unrecognizable for downstream interactions, causing an irreversible break in the task chain.

Finally, a lack of localization precision leads to a critical logical disconnect. Even with explicit coordinate points, generated response elements like pop-up menus exhibit significant spatial jitter, often offsetting by dozens of pixels from their intended locations. This confirms a decoupling of perception and execution, as evidenced by GOAL scores generally falling below 20%. This “blindness" to abstract spatial instructions remains the most formidable obstacle to achieving functionally valid generative GUI environments.

###### 5.3. The Paradox of Visual Fidelity vs. Functional Plausibility

Multi-dimensional analysis through GE-Score reveals a critical paradox: visual fidelity does not equate to functional viability. Models such as GPT-image-1.5 (OpenAI, 2025) generate GUIs with exceptional composition and clarity, earning high QUAL scores. However, granular functional inspection reveals that this “visual over-optimism" is often deceptive, as these aesthetically pleasing images frequently contain hallucinated widgets or illogical layouts. This reinforces that benchmarking image generation models as GUI environments must be predicated upon a assessment of temporal coherence and interactive logic, which take precedence over generaldomain visual fidelity.

##### 6. Conclusion

We introduces GEBench, the first systematic benchmark designed to evaluate image generation models as GUI environments. By shifting the focus from general-domain visual fidelity to GUI interaction logic, we provide a comprehensive testbed for assessing the potential of generative models as GUI simulators. Existing image generation models often struggle with the precise structural requirements of interactive GUI generation, a gap that GEBench is uniquely positioned to measure. Through the proposed GE-Score and a VLM-based evaluation pipeline, we identify critical barriers to high-fidelity GUI simulation. Our analysis highlights that while image generation models show promise in predicting basic state transitions, they suffer from icon hallucinations, coordinate drift, and text rendering limitations that hinder their application as robust GUI generators. These findings underscore the need for future research to prioritize finegrained structural control and semantic persistence over simple visual realism. We believe that GEBench establishes a necessary foundation for developing the next generation of generative GUI simulators capable of supporting the large-scale training of autonomous GUI agents.

##### References

- R. An, S. Yang, R. Zhang, Z. Shen, M. Lu, G. Dai, H. Liang, Z. Guo, S. Yan, Y. Luo, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025.
- S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren,

- X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie,
- Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang,

- B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

- R. Bonatti, D. Zhao, F. Bonacci, D. Dupont, S. Abdali, Y. Li, Y. Lu, J. Wagle, K. Koishida, A. Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024.

- C. Chen, T. Su, G. Meng, Z. Xing, and Y. Liu. From ui design image to gui skeleton: a neural machine translator to bootstrap mobile gui implementation. In Proceedings of the 40th International Conference on Software Engineering, pages 665–676, 2018.

- J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pages 386–402. Springer, 2024.

M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.

- K. Cheng, Q. Sun, Y. Chu, F. Xu, L. YanTao, J. Zhang, and Z. Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In ACL 2024 (Volume 1: Long Papers), pages 9313–9332, 2024.

- K. Cobbe, C. Hesse, J. Hilton, and J. Schulman. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning, pages 2048–2056. PMLR, 2020.

- G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein,

- O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- L. Fan, T. Li, S. Qin, Y. Li, C. Sun, M. Rubinstein, D. Sun, K. He, and Y. Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

A. Garg, Y. Jiang, and A. Oulasvirta. Controllable gui exploration. arXiv preprint arXiv:2502.03330, 2025.

- D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Google. A new era of intelligence with Gemini 3, 2025a. URL https://blog.google/prod ucts/gemini/gemini-3.

Google. Introducing nano banana pro, 2025b. URL https://blog.google/innovation-a nd-ai/products/nano-banana-pro/.

- Z. Guo, X. Chen, R. Zhang, R. An, Y. Qi, D. Jiang, X. Li, M. Zhang, H. Li, and P.-A. Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025.

- J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15733–15744, 2025.

M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

- J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.

- R. Huang, Y. Long, and X. Chen. Automaticly generating web page from a mockup. In SEKE, pages 589–594, 2016.

Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda,

- A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

K. Kolthoff, F. Kretzer, L. Fiebig, C. Bartelt, A. Maedche, and S. P. Ponzetto. Zero-shot prompting approaches for llm-based graphical user interface generation. arXiv preprint arXiv:2412.11328,

- 2024.

K. Kolthoff, F. Kretzer, C. Bartelt, A. Maedche, and S. P. Ponzetto. Guide: Llm-driven gui generation decomposition for automated prototyping. In 2025 IEEE/ACM 47th International Conference on Software Engineering: Companion Proceedings (ICSE-Companion), pages 1–4. IEEE,

- 2025.

- B. F. Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

- J. Li, J. Yang, A. Hertzmann, J. Zhang, and T. Xu. Layoutgan: Synthesizing graphic layouts with vector-wireframe adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(7):2388–2399, 2020.

- S. Li, K. Kallidromitis, A. Gokul, Y. Kato, K. Kozuka, and A. Grover. Mobileworldbench: Towards semantic world modeling for mobile agents. arXiv preprint arXiv:2512.14014, 2025a.

Z. Li, Z. Liu, Q. Zhang, B. Lin, F. Wu, S. Yuan, Z. Yan, Y. Ye, W. Yu, Y. Niu, et al. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025b.

- W. Lin, X. Wei, R. An, T. Ren, T. Chen, R. Zhang, Z. Guo, W. Zhang, L. Zhang, and H. Li. Perceive anything: Recognize, explain, caption, and segment anything in images and videos, 2025. URL https://arxiv.org/abs/2506.05302.

S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025a.

- X. Liu, X. Zhang, Z. Zhang, and Y. Lu. Ui-e2i-synth: Advancing gui grounding with large-scale instruction synthesis. arXiv preprint arXiv:2504.11257, 2025b.

D. Luo, B. Tang, K. Li, G. Papoudakis, J. Song, S. Gong, J. Hao, J. Wang, and K. Shao. Vimo: A generative visual gui world model for app agents. arXiv preprint arXiv:2504.13936, 2025.

M. A. Mozaffari, X. Zhang, J. Cheng, and J. L. Guo. Ganspiration: balancing targeted and serendipitous inspiration in user interface design with style-based generative adversarial network. In Proceedings of the 2022 CHI conference on human factors in computing systems, pages 1–15, 2022.

- Y. Niu, M. Ning, M. Zheng, W. Jin, B. Lin, P. Jin, J. Liao, C. Feng, K. Ning, B. Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

###### OpenAI. The new chatgpt images is here, 2025. URL https://openai.com/index/new-cha tgpt-images-is-here/.

A. Radford et al. Learning transferable visual models from natural language supervision. ICML, 2021.

- A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

P. Sandhaus, M. Rabbath, and S. Boll. Employing aesthetic principles for automatic photo book layout. In International Conference on Multimedia Modeling, pages 84–95. Springer, 2011.

- B. Seed. Seedream 4.5, 2025. URL https://seed.bytedance.com/en/seedream4_5.

- T. Seedream, Y. Chen, Y. Gao, L. Gong, M. Guo, Q. Guo, Z. Guo, X. Hou, W. Huang, Y. Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- A. Sobolevsky, G.-A. Bilodeau, J. Cheng, and J. L. Guo. Guilget: Gui layout generation with transformer. arXiv preprint arXiv:2304.09012, 2023.

- K. Sun, K. Huang, X. Liu, Y. Wu, Z. Xu, Z. Li, and X. Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.

- M. L. Team, H. Ma, H. Tan, J. Huang, J. Wu, J.-Y. He, L. Gao, S. Xiao, X. Wei, X. Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025a.

- N. Team, C. Han, G. Li, J. Wu, Q. Sun, Y. Cai, Y. Peng, Z. Ge, D. Zhou, H. Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025b.

- T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang,

- X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan,
- Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z.-F. Wu, and Z. Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

J. Wei, A.-L. Courbis, T. Lambolais, B. Xu, P. L. Bernard, and G. Dray. Boosting gui prototyping with diffusion models. In 2023 IEEE 31st International Requirements Engineering Conference (RE), pages 275–280. IEEE, 2023.

- C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- B. Xie, R. Shao, G. Chen, K. Zhou, Y. Li, J. Liu, M. Zhang, and L. Nie. Gui-explorer: Autonomous exploration and mining of transition-aware knowledge for GUI agent. In ACL 2025 (Volume 1: Long Papers), pages 5650–5667. Association for Computational Linguistics, 2025a.

T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Y. Xie, Z. Li, R. Shao, G. Chen, K. Zhou, Y. Li, D. Jiang, and L. Nie. Mirage-1: Augmenting and updating gui agent with hierarchical multimodal skills. arXiv preprint arXiv:2506.10387, 2025b.

H. Yan, Y. Shen, X. Huang, J. Wang, K. Tan, Z. Liang, H. Li, Z. Ge, O. Yoshie, S. Li, et al. Gui exploration lab: Enhancing screen navigation in agents via multi-turn reinforcement learning. arXiv preprint arXiv:2512.02423, 2025.

X. Yang, T. Mei, Y.-Q. Xu, Y. Rui, and S. Li. Automatic generation of visual-textual presentation layout. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 12(2):1–22, 2016.

- C. Zhang, S. He, J. Qian, B. Li, L. Li, S. Qin, Y. Kang, M. Ma, G. Liu, Q. Lin, S. Rajmohan,
- D. Zhang, and Q. Zhang. Large language model-brained GUI agents: A survey. Trans. Mach. Learn. Res., 2025, 2025a.

H. Zhang, C. Li, W. Wu, S. Mao, Y. Zhang, H. Tian, I. Vuli´c, Z. Zhang, L. Wang, T. Tan, et al. Scaling and beyond: Advancing spatial reasoning in mllms requires new recipes. arXiv preprint arXiv:2504.15037, 2025b.

- H. Zhang, W. Wu, C. Li, N. Shang, Y. Xia, Y. Huang, Y. Zhang, L. Dong, Z. Zhang, L. Wang, et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514, 2025c.
- H. Zhang, X. Bai, C. Li, C. Liang, H. Tian, H. Li, R. An, Y. Zhang, A. Korhonen, Z. Zhang, L. Wang, and T. Tan. How well do models follow visual instructions? vibe: A systematic benchmark for visual instruction-driven image editing, 2026. URL https://arxiv.org/ab s/2602.01851.

T. Zhao, C. Chen, Y. Liu, and X. Zhu. Guigan: Learning to generate gui designs using generative adversarial networks. In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pages 748–760. IEEE, 2021.

X. Zhao, P. Zhang, K. Tang, X. Zhu, H. Li, W. Chai, Z. Zhang, R. Xia, G. Zhai, J. Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

- C. Zhuang, A. Huang, Y. Hu, J. Wu, W. Cheng, J. Liao, H. Wang, X. Liao, W. Cai, H. Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. arXiv preprint arXiv:2505.24862, 2025.

##### Appendix

##### A. Evaluation Framework

In this section, we detail the proposed evaluation framework. The evaluation framework operates through a systematic three-stage pipeline designed to rigorously benchmark image generation models as GUI environments. The process initiates with Image Generation, where image generation models are tasked with generating visual outputs across five distinct task categories: Single-step Visual Transition (single-step), Multi-step Planning (multi-step), Zeroshot Virtual GUI (fiction-app), Rare Trajectory Synthesis (real-app), and Grounding-based Generation (grounding). Subsequently, the generated samples undergo a VLM-as-a-Judge process employing a evaluator strategy. We leverage VLMs, specifically GPT-4o (Hurst et al., 2024), Gemini-3-Pro-Native (Google, 2025a) and Qwen3-vl-235b-a22b-thinking (Bai et al., 2025), to assess the results along five critical dimensions: Goal Achievement (GOAL), Interaction Logic (LOGIC), Consistency (CONS), UI Plausibility (UI), and Visual Quality (QUAL). Finally, the framework concludes with Metrics Analysis, where the calculated scores are validated through statistical verification, pattern analysis, and human relevance alignment to ensure robust and meaningful benchmarking results.

[Figure 121]

[Figure 122]

[Figure 123]

Image Generation

VLM as Judge

Metrics Analysis

LMM Model Evaluation

Statistical-Analysis & Verification

Generate Models as GUI Data Engine

Model Types

Evaluator

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

GPTImage Seedream

Model Comparison

Human relevance

Flux. 2

GPT Gemini

5 Task Types

[Figure 133]

Evaluation Analysis

5 Evaluation Dimensions

- • Single-Step Task
- • Multi-Step Task
- • Fictional Apps Task
- • Real Apps Task
- • Grounding Task

Generated GUI Sequences

Score Calculation

Figure A1 | GEBench Evaluation Framework Overview. This diagram outlines the comprehensive evaluation process of the GEBench framework, which assesses the performance of image generation models in generating GUI sequences.

##### B. Detailed Performance On GEBench Using Different Judges

In this section, we provide a granular breakdown of the experimental results, evaluated by three distinct state-of-the-art Visual Language Models (VLMs) acting as autonomous judges. To ensure the robustness and objectivity of our benchmarking, we report the full performance profiles of all 12 evaluated models across three separate evaluation runs:

- • Table A1: Detailed scores assigned by Gemini-3-Pro-Native (Google, 2025a) as the judge.
- • Table A2: Detailed scores assigned by GPT-4o (Hurst et al., 2024) as the judge.
- • Table A3: Detailed scores assigned by Qwen3-vl-235b-a22b-thinking (Bai et al., 2025) as the judge.

Table A1 | Detailed Performance on GEBench using Gemini-3-Pro-Native (Google, 2025a) as Judge

Nano Banana Pro

Nano Banana

GPT-Image -1.5

GPT-Image -1.0

Seedream 4.5

Seedream 4.0

Qwen -Image-Edit

Longcat -Image

Wan 2.6 Flux-2-Pro Bagel UniWorld

Task Metric

GOAL 95.04 69.60 87.65 86.50 56.60 54.58 58.94 65.08 24.35 41.00 25.60 36.07 LOGIC 82.04 58.50 86.50 78.80 49.25 45.29 52.17 53.60 29.12 43.33 27.93 35.47 CONS 75.24 69.70 72.65 58.10 66.23 65.68 81.81 77.46 54.76 58.77 54.93 53.73 UI 79.50 51.60 84.76 58.00 50.19 47.87 53.50 48.43 28.44 47.93 40.07 42.60 QUAL 90.22 73.50 79.97 32.00 61.89 68.39 64.65 79.70 30.95 54.00 52.00 45.73 Overall 84.41 64.58 82.31 62.68 56.83 56.36 62.21 64.85 33.52 49.01 40.11 42.72

single-step

GOAL 86.63 47.61 79.13 77.45 68.25 60.14 46.89 67.18 3.85 2.29 1.98 1.78 LOGIC 56.75 30.46 36.94 35.31 36.63 38.59 25.54 52.71 8.87 3.73 6.70 2.96 CONS 56.95 45.69 36.39 34.08 38.66 38.70 56.43 46.46 15.46 31.86 43.32 19.89 UI 54.96 26.80 50.93 50.41 35.15 33.44 37.56 29.79 12.10 20.17 29.17 16.08 QUAL 90.08 62.03 86.23 85.82 66.80 48.04 64.88 63.18 13.54 40.41 31.60 10.41 Overall 69.07 42.52 57.92 56.61 49.10 43.78 46.26 51.86 10.76 19.69 22.55 10.22

multi-step

GOAL 40.51 40.40 46.20 38.78 40.00 39.33 36.49 45.93 15.80 4.00 0.40 17.01 LOGIC 51.03 48.28 43.13 40.61 39.73 39.80 42.54 48.60 29.80 4.60 0.81 20.90 CONS 57.95 60.00 53.20 53.88 61.00 64.68 61.17 50.67 41.20 88.07 62.36 85.00 UI 50.26 45.45 68.72 61.63 42.60 39.40 41.24 39.00 27.60 28.03 27.95 25.42 QUAL 89.49 89.70 98.20 95.71 92.40 59.87 74.16 85.93 19.20 30.47 32.39 20.90 Overall 57.85 56.77 61.89 58.12 55.15 48.62 51.12 54.03 26.72 31.03 24.78 33.85

fiction-app

GOAL 28.73 32.80 26.96 34.75 40.00 43.80 33.16 34.15 18.20 2.06 0.40 17.07 LOGIC 44.40 45.20 36.36 39.80 37.20 43.27 41.10 47.24 33.80 2.06 0.40 20.00 CONS 67.13 65.33 56.83 56.77 55.00 67.87 59.07 50.71 63.60 72.23 67.67 82.79 UI 48.00 46.80 70.24 58.79 41.20 40.60 39.24 39.25 34.20 22.54 27.60 25.99 QUAL 90.60 92.40 96.16 97.84 89.80 55.20 68.78 84.25 21.80 21.79 32.80 20.88 Overall 55.77 56.51 57.31 57.59 52.64 50.15 48.27 51.12 34.32 24.14 25.77 33.35

real-app

GOAL 23.88 21.80 15.19 15.51 16.12 15.42 11.74 11.29 5.86 4.00 4.95 7.47 LOGIC 36.94 41.00 37.39 37.55 51.84 44.58 35.65 36.13 17.17 18.20 14.43 15.35 CONS 77.96 71.20 51.20 45.31 64.29 67.71 83.26 69.68 62.42 69.80 80.21 74.95 UI 77.35 52.00 66.90 61.77 48.57 48.96 68.04 50.32 41.62 70.40 80.41 74.34 QUAL 92.24 72.80 85.57 81.50 75.51 67.08 83.26 79.35 53.94 79.20 83.09 73.13 Overall 61.67 51.76 51.25 48.33 51.27 48.75 56.39 49.35 36.20 48.32 52.62 49.05

grounding

Each table provides a comprehensive matrix of scores across all five GE-Score dimensions (GOAL, LOGIC, CONS, UI, and QUAL) for every task category in GEBench. For consistent comparison and to mitigate the variance in internal scoring scales across different VLM judgers, all raw evaluation outputs have been linearly normalized to a standard range of [0,100]. This multi-judger approach allows for a cross-validation of model capabilities and highlights the consistency of our GE-Score framework.

##### C. Detailed Rubric on five tasks

To ensure a rigorous and standardized evaluation, we developed a series of fine-grained scoring rubrics tailored to the specific requirements of different GUI generation tasks. These rubrics serve as the foundational logic for our VLM-as-a-judge framework.

Each rubric decomposes the five GE-Score dimensions (GOAL, LOGIC, CONS, UI, and QUAL) into explicit, linguistic descriptions across multiple performance tiers. This structured approach minimizes the subjective bias of the VLM judges by providing concrete visual and functional benchmarks for each score level.

##### D. Detailed Rubric on five tasks

To ensure a rigorous and standardized evaluation, we developed a series of fine-grained scoring rubrics tailored to the specific requirements of different GUI generation tasks. These rubrics, detailed in Figures A2, A3, A4, and A5, serve as the foundational logic for our VLM-as-a-judge framework.

Table A2 | Detailed Performance on GEBench using GPT-4o (Hurst et al., 2024) as Judge

Nano Banana Pro

Nano Banana

GPT-Image -1.5

GPT-Image -1.0

Seedream 4.5

Seedream 4.0

Qwen -Image-Edit

Longcat -Image

Wan 2.6 Flux-2-Pro Bagel UniWorld

Task Metric

GOAL 90.73 74.57 88.37 80.41 63.37 60.56 71.38 73.10 31.40 47.30 37.40 40.28 LOGIC 94.25 81.10 92.04 84.39 68.91 66.92 77.18 78.07 38.00 52.20 42.40 44.68 CONS 84.52 79.49 78.27 68.88 70.40 70.32 84.34 83.25 55.90 62.20 64.60 61.44 UI 91.69 78.09 88.57 82.96 66.00 65.55 75.84 78.48 42.20 55.40 55.60 51.00 QUAL 78.95 69.54 78.98 77.14 55.37 58.71 69.68 77.06 37.80 53.20 56.90 48.56 Overall 88.03 76.56 85.25 78.76 64.81 64.41 75.68 77.99 41.06 54.06 51.38 49.19

single-step

GOAL 89.83 72.30 95.44 95.31 88.92 76.96 77.21 83.74 18.31 26.21 15.63 14.01 LOGIC 84.99 72.80 87.16 87.45 80.43 70.14 73.03 78.38 26.83 36.56 32.28 29.53 CONS 83.12 77.70 82.77 82.04 77.70 68.34 81.89 80.51 31.47 56.11 68.36 44.75 UI 89.80 76.60 89.16 89.80 84.39 71.55 80.88 79.49 27.27 52.60 55.82 33.87 QUAL 90.68 78.40 92.32 93.98 85.83 64.68 79.02 76.97 25.37 56.11 42.07 21.31 Overall 87.68 75.56 89.37 89.72 83.45 70.33 78.41 79.82 25.85 45.52 42.83 28.69

multi-step

GOAL 40.36 42.07 51.60 49.21 48.21 40.47 40.14 47.61 23.00 14.53 6.33 25.21 LOGIC 62.28 61.48 70.07 65.15 62.13 55.46 60.89 64.53 43.40 24.87 17.07 46.94 CONS 79.77 76.14 85.33 79.17 79.87 71.20 76.91 72.20 52.60 77.67 68.57 69.58 UI 83.26 79.26 91.28 87.83 81.46 67.72 72.38 72.01 46.87 57.47 45.80 54.20 QUAL 78.10 73.07 87.87 92.32 82.46 52.00 66.80 65.26 35.47 43.40 27.83 40.00 Overall 68.75 66.40 77.23 74.74 70.83 57.37 63.42 64.32 40.27 43.59 33.12 47.19

fiction-app

GOAL 33.40 39.24 43.73 43.83 43.73 41.34 41.42 41.26 22.80 9.11 4.27 25.00 LOGIC 55.80 59.32 61.72 59.06 57.20 55.47 50.25 62.87 50.67 19.42 17.53 43.80 CONS 79.00 80.48 81.66 81.22 76.33 69.67 76.75 76.28 64.13 71.10 69.27 70.80 UI 81.60 81.56 88.40 87.35 78.00 64.93 71.75 76.54 50.73 46.77 42.33 58.80 QUAL 78.40 78.77 88.15 92.40 80.33 48.99 61.92 69.67 34.53 32.85 22.47 41.80 Overall 65.64 67.87 72.73 72.77 67.12 56.08 60.42 65.32 44.57 35.85 31.17 48.04

real-app

GOAL 35.60 26.07 20.40 16.33 21.55 17.73 15.85 17.40 8.47 7.89 6.73 5.82 LOGIC 67.00 66.13 53.80 55.80 58.38 58.14 57.21 58.87 54.73 61.00 63.87 62.07 CONS 78.20 85.00 75.82 59.33 75.35 68.04 84.69 79.80 67.07 86.45 91.53 89.03 UI 81.40 77.67 76.03 67.73 69.23 62.89 77.41 71.93 55.13 85.45 91.47 88.23 QUAL 80.40 76.53 80.74 75.93 75.76 63.30 82.18 77.53 55.27 87.17 90.33 84.01 Overall 68.52 66.28 61.36 55.02 60.05 54.02 63.47 61.11 48.13 65.59 68.79 65.83

grounding

Table A3 | Detailed Performance on GEBench using Qwen3-vl-235b-a22b-thinking (Bai et al., 2025) as Judge

Nano Banana Pro

Nano Banana

GPT-Image -1.5

GPT-Image -1.0

Seedream 4.5

Seedream 4.0

Qwen -Image-Edit

Longcat -Image

Wan 2.6 Flux-2-Pro Bagel UniWorld

Task Metric

GOAL 93.64 70.54 84.70 75.53 67.24 65.85 67.65 66.84 27.14 38.72 33.23 35.05 LOGIC 92.45 72.86 81.27 71.20 66.02 64.41 69.13 68.27 30.44 38.32 35.32 34.78 CONS 83.28 66.40 64.60 55.43 59.66 57.68 70.31 69.97 37.58 36.50 44.07 34.88 UI 94.25 65.62 83.98 70.14 62.41 61.16 65.02 63.57 31.85 40.17 44.78 35.59 QUAL 98.04 80.51 94.85 86.80 72.59 76.03 77.37 80.58 41.21 55.19 60.30 48.48 Overall 92.33 71.19 81.88 71.82 65.58 65.03 69.90 69.85 33.64 41.78 43.54 37.76

single-step

GOAL 92.35 60.97 93.61 96.22 89.39 83.45 72.70 83.80 4.62 6.44 4.22 3.70 LOGIC 88.53 60.44 84.43 89.46 77.59 75.09 65.92 76.15 17.65 23.95 19.70 16.69 CONS 84.04 71.32 75.50 79.22 73.23 71.26 76.32 72.29 19.69 42.06 57.16 30.29 UI 95.96 60.77 90.68 92.31 75.00 73.23 74.48 64.74 16.12 33.32 51.10 21.16 QUAL 97.65 71.89 96.34 98.20 86.87 80.17 87.13 77.29 17.65 48.06 53.59 21.16 Overall 91.71 65.08 88.11 91.08 80.42 76.64 75.31 74.85 15.15 30.77 37.15 18.60

multi-step

GOAL 28.13 34.48 43.33 34.68 39.19 34.27 27.50 34.13 15.69 9.73 8.07 18.40 LOGIC 41.93 49.90 58.33 46.87 48.26 45.67 45.28 48.70 30.24 19.60 13.93 31.39 CONS 49.47 58.05 52.87 43.30 49.26 43.73 49.51 43.33 25.86 53.07 20.53 36.04 UI 53.47 55.69 70.20 63.91 55.64 45.27 47.64 40.67 22.36 34.00 19.47 29.93 QUAL 76.53 72.73 97.00 94.75 82.21 60.47 73.12 63.00 20.88 34.00 24.47 29.03 Overall 49.91 54.17 64.35 56.70 54.91 45.88 48.61 45.97 23.01 30.08 17.29 28.96

fiction-app

GOAL 30.47 30.47 34.73 37.33 36.67 39.00 27.95 35.08 17.58 8.40 4.33 21.73 LOGIC 43.47 47.32 50.73 49.40 45.87 46.27 47.03 48.56 33.60 14.95 12.80 31.33 CONS 58.33 62.21 53.13 49.93 48.73 42.80 51.30 47.81 34.75 37.75 17.60 43.96 UI 59.80 59.53 73.07 65.47 50.47 41.53 50.88 48.42 26.06 22.39 15.13 34.93 QUAL 75.60 78.52 97.93 93.67 80.13 56.00 67.03 66.94 20.88 22.73 16.60 34.53 Overall 53.53 55.61 61.92 59.16 52.37 45.12 48.84 49.36 26.57 21.24 13.29 33.30

real-app

GOAL 24.38 20.89 15.68 18.72 19.31 14.41 16.05 16.58 12.44 12.50 6.69 9.08 LOGIC 36.50 36.56 32.26 30.54 37.66 32.21 34.45 31.81 22.34 22.85 15.00 18.16 CONS 46.20 42.89 33.77 25.10 40.96 38.76 48.21 46.44 35.79 40.83 42.04 42.13 UI 69.49 58.11 56.85 44.30 56.22 52.55 67.22 56.98 40.07 59.03 68.66 61.35 QUAL 89.12 79.38 81.10 70.40 76.29 72.76 86.08 73.56 53.11 81.39 84.23 80.21 Overall 53.14 47.57 43.93 37.81 46.09 42.14 50.40 45.07 32.75 43.32 43.32 42.19

grounding

###### Each rubric decomposes the five GE-Score dimensions (GOAL, LOGIC, CONS, UI, and QUAL) into explicit, linguistic descriptions across multiple performance tiers (e.g., from “In-

complete" to “Exceptional"). This structured approach minimizes the subjective bias of the VLM judges by providing concrete visual and functional benchmarks for each score level. Specifically:

- • Figure A2 outlines the criteria for Single-Step Transition, focusing on the immediate visual mapping of user instructions.
- • Figure A3 details the Multi-Step Planning rubrics, emphasizing the accumulation of errors and temporal coherence across 5 step trajectories.
- • Figure A4 provides the standards for Zero-shot Virtual GUI generation, where the judge assesses the imaginative plausibility and structural integrity of non-existent applications.
- • Figure A5 defines the benchmarks for Grounded Generation, specifically evaluating the pixel-level alignment between generated content and coordinate-based prompts.

By employing these detailed rubrics, we bridge the gap between qualitative visual inspection and quantitative performance metrics.

[Figure 134]

#### JUDGE RUBRICS -- Type1: Single-Step

GOAL (Goal Achievement): Does the generated state achieve the caption-described change?

5: Caption change is achieved completely and unambiguously. All key elements mentioned in the caption are present and correctly transformed. No ambiguity about whether the change matches the description.

4: Goal is achieved with only minor, non-critical omissions or formatting differences. The core change is clearly visible and matches the caption, but there might be slight differences in layout, text formatting, or minor elements not central to the goal.

3: Goal is mostly achieved but key details are ambiguous/partially missing. The main intent is recognizable, but important elements are unclear, partially implemented, or require inference to confirm they match the caption.

2: Partial achievement; only a small portion of the intended change is visible. Most of the expected transformation is missing or incorrect, though some elements

loosely relate to the caption. 1: Barely related; the change does not match caption semantics. There might be a superficial similarity, but the core change described in the caption is not present. 0: Complete failure; no relevant change or totally wrong change. The generated image shows either no change from the initial state or a change completely unrelated

to the caption.

LOGIC (Interaction/State Logic): Is the change consistent with plausible GUI interaction and state transitions?

5: Transitions are fully plausible and consistent with standard UI behavior. The change follows natural GUI interaction patterns, state transitions are logical, and there are no impossible jumps or discontinuities. The transformation looks like a real UI would behave.

4: Mostly plausible; minor implausibility but still credible. The overall transition makes sense, but there might be slight inconsistencies that don't break the believability of the interaction (e.g., slightly accelerated animation, minor state inconsistency).

3: Some implausible elements; still partially coherent. The change shows recognizable UI logic in parts, but contains elements that don't follow standard interaction patterns or have logical gaps that require assuming missing intermediate steps.

2: Largely implausible; broken state transition or inconsistent UI behavior. The change shows significant logical problems, such as abrupt "teleportation" between unrelated states, contradictory UI element behaviors, or changes that would require impossible user actions.

1: Almost entirely illogical; UI changes contradict basic interaction patterns. The transformation violates fundamental UI principles, shows elements appearing/disappearing without cause, or demonstrates behaviors that no real GUI would exhibit.

0: Impossible; severe "teleportation" or nonsensical transformation. The change is completely disconnected from any plausible UI interaction, showing random or meaningless state changes that cannot be explained by any reasonable user action sequence.

CONS (Consistency/Preservation): Are unrelated regions preserved from the Initial Image?

5: Unaffected regions are preserved nearly perfectly. Every pixel outside the directly affected area is identical between Initial and Generated images. No drift, no color changes, no subtle alterations in preserved elements.

4: Minor drift (small shifts, slight color/blur changes) but clearly preserved. The preserved regions remain recognizable and functionally identical, though there might be nearly imperceptible changes due to compression or anti-aliasing artifacts.

3: Noticeable drift in multiple areas, but the overall UI identity is maintained. Elements outside the target region may shift slightly, show minor color variations, or have subtle blur, but the general layout and all UI components remain identifiable and in approximately correct positions.

2: Significant unintended changes outside the target region. Background elements show obvious movement, text changes in supposedly unchanged areas, or layout modifications that weren't part of the intended change. The preservation is clearly imperfect.

1: Widespread unintended changes; most of the UI is altered. Large sections of the UI that should remain unchanged show visible differences, with elements moving, changing appearance, or being corrupted across the screen.

0: Entire screen is corrupted or replaced. No meaningful preservation of unchanged regions; the entire UI appears transformed, broken, or replaced with unrelated content.

UI (UI Plausibility/Integrity): Are UI elements plausible (no hallucinations, broken layout, impossible widgets/states)?

5: UI components are coherent, correctly structured, and look native. All elements follow platform conventions, proper layering is maintained, states are correct, no visual anomalies, and text is perfectly rendered. The interface looks like it could be a real, professional GUI.

4: Mostly coherent; minor layout/element issues. The UI generally follows platform conventions with only subtle anomalies. There might be minor layering issues, slight misalignment, or small state inconsistencies, but overall it resembles a usable interface.

3: Several element/layout issues but still resembles a usable UI. Some UI elements appear slightly off (wrong sizes, minor misalignment, occasional text artifacts) but remain recognizable and the interface still appears functional despite the issues.

2: Many hallucinations or severe layout problems. Multiple UI components show significant problems: incorrect states, major layering errors, impossible layouts, or generated elements that don't exist in real UIs. The interface integrity is compromised.

1: UI is mostly broken; elements are nonsensical. Most UI components are incorrect, showing impossible states, major layering violations, or hallucinated controls that make no sense in the context. Very few elements appear correct or usable.

0: UI is unusable or completely hallucinated. The interface shows no recognizable UI elements, contains completely impossible layouts, hallucinated controls that violate all design principles, or appears as random visual noise rather than a coherent interface.

QUAL (Visual Quality): Visual readability (text/icon clarity, artifacts).

5: Crisp, readable, no obvious artifacts. Text and icons have sharp edges, perfect color accuracy, no compression artifacts, and professional rendering quality throughout the image.

4: Very readable; minor blur/artifacts. Text and icons remain clear and legible, with only slight compression artifacts or minimal edge blurring that doesn't affect readability.

3: Readable but noticeable blur/artifacts in important areas. Text is mostly readable but shows some artifacts, icons may be slightly blurry, and there might be minor color distortions in critical regions.

2: Significant quality degradation; text/icons hard to read. Obvious artifacts throughout the image, significant blurring of important elements, color issues, or compression noise that interferes with reading text or recognizing icons.

1: Severe artifacts; most text is unreadable. Major quality problems throughout, extensive blurring, strong color distortions, or heavy compression artifacts make most text illegible and icons unrecognizable.

0: Completely unusable image quality. Extreme artifacts, total blur, meaningless color shifts, or severe corruption make the image completely unreadable and unusable for any UI evaluation purpose.

Figure A2 | Evaluation Rubrics for Single-Step Transition Generation

[Figure 135]

#### JUDGE RUBRICS -- Type2: Multi-Steps

###### GOAL (Final Goal Achievement): Does Frame 5 achieve the global goal?

5: Perfect achievement - Frame 5 shows EXACTLY what Global Goal describes, all objectives fully met with no ambiguity. The final state precisely matches the intended outcome described in the goal.

4: Strong achievement - Final goal achieved with only minor omissions or slight deviations that don't affect core objectives. Frame 5 accurately reflects the goal with only non-critical details missing or slightly different.

3: Partial achievement - Some aspects of final goal met, but important elements are missing or incorrectly implemented. Frame 5 shows progress toward the goal but key components are absent, wrong, or ambiguous.

2: Weak achievement - Only minimal progress toward final goal, most key objectives not met. Frame 5 barely resembles the intended outcome, with most critical elements missing or wrong.

1: Minimal achievement - Almost no progress toward final goal, only superficial similarity. Frame 5 might share some basic UI structure but shows virtually no meaningful progress toward the stated objective.

0: No achievement - Frame 5 completely unrelated to final goal or shows wrong outcome. The trajectory ends in a state that contradicts or has no meaningful connection to the global goal.

LOGIC (Interaction Logic Coherence): Are Frame0→5 transitions plausible and consistent with GUI interaction?

5: Perfect logic - Every interaction step is natural and reasonable, with coherent step-by-step progression and no jumps. Each frame logically follows from the

previous one, showing realistic UI state transitions without teleportation or impossible changes. The entire trajectory demonstrates a believable user interaction flow.

4: Strong logic - Mostly natural progression with minor logical gaps, but overall interaction makes sense. The trajectory is mostly coherent, though there might be one or two slightly abrupt transitions that are still plausible within GUI interaction patterns.

3: Partial logic - Some steps follow expected flow, but several show logical jumps or unclear causation. The trajectory has noticeable gaps where it's unclear how one frame transitions to the next, requiring assumptions about missing intermediate states or actions.

2: Weak logic - Frequent logical inconsistencies across frames, progression is confusing. Multiple frames appear disconnected, showing "teleportation" between unrelated UI states, or transitions that contradict normal GUI behavior patterns.

1: Minimal logic - Actions barely follow expected flow, most steps are disconnected. The trajectory shows little coherent progression, with frames that seem randomly ordered or only loosely related to each other, making the interaction pattern unclear.

0: No logic - Frame-to-frame changes are completely disconnected and meaningless. The trajectory appears as a random sequence of unrelated UI states with no discernible interaction logic or coherent progression.

###### CONS (Visual Consistency Across Frames): Are UI elements stable across all 5 frames?

5: Perfect consistency - ALL UI elements (system bars, layout, style, colors) remain completely stable across all frames, zero drift. Every unchanged component maintains identical position, appearance, and state throughout the entire trajectory. No flicker, no subtle shifts, perfect stability.

4: Strong consistency - Minor variations in some elements (slight color shifts, subtle positioning) but core UI remains stable. Background elements might show nearly imperceptible changes, but all major components stay consistent and recognizable across frames.

3: Partial consistency - Noticeable drift in some UI elements between frames, but major components remain recognizable. Some elements shift position, change color slightly, or show minor flicker, but the overall UI structure and identity is maintained.

2: Weak consistency - Significant inconsistencies across frames: layout shifts, style changes, unstable UI elements. Background components move noticeably, colors change substantially, or there's clear flicker/corruption in multiple frames affecting UI stability.

1: Minimal consistency - UI changes substantially across frames, few elements remain stable, major visual drift. Most background elements show significant movement or change, with only a few components maintaining their position/appearance throughout the trajectory.

0: No consistency - Completely different UI across frames, no recognizable stability or coherent interface. Each frame appears to show a different UI layout, style, or completely unrelated interface elements with no continuity.

UI (Element Integrity): Are UI components in each frame native-looking and coherent?

5: Perfect integrity - All UI components follow platform conventions, proper layering, correct states, no visual anomalies, text perfectly rendered in EVERY frame. Every frame shows professional-quality UI that looks indistinguishable from a real native application interface.

4: Strong integrity - UI mostly follows conventions with minor layering issues or subtle anomalies, text mostly clear across frames. There might be occasional minor issues like slight misalignment or subtle rendering problems, but overall UI quality remains high in all frames.

3: Partial integrity - Some UI elements appear slightly off (wrong sizes, minor misalignment, occasional text artifacts) but still plausible in most frames. The UI remains recognizable and generally functional despite several noticeable imperfections across the trajectory.

2: Weak integrity - Noticeable UI problems in multiple frames: incorrect states, layering errors, text clarity issues, minor hallucinations. Multiple frames show significant UI defects that affect the believability and usability of the interface.

1: Poor integrity - Significant UI breakdown across frames: impossible states, major layering violations, unreadable text, clear hallucinations. Most frames contain serious UI problems that make the interface appear broken or nonsensical.

0: No integrity - UI completely nonsensical across frames, with hallucinated controls, impossible layouts, or no recognizable interface elements. The trajectory shows no coherent UI structure, appearing as random visual elements rather than a functional interface.

QUAL (Visual Quality): Are text/icons readable and artifacts minimal across all frames?

5: Excellent quality - Crystal clear text/icons in ALL frames, sharp edges, no compression artifacts, perfect color accuracy. Every frame shows professional rendering quality with crisp, easily readable text and sharp iconography throughout.

4: Good quality - Text/icons clear and readable across frames, minor compression artifacts or slight blurring at edges only. Slight quality degradation may be visible but doesn't affect readability or overall visual quality.

3: Fair quality - Text mostly readable but shows some artifacts in multiple frames, icons slightly blurry, minor color distortions. Quality issues are noticeable and affect some regions, but important text and icons remain legible.

2: Poor quality - Text becoming difficult to read in several frames, noticeable artifacts, significant blurring, color issues. Quality degradation is substantial and affects readability of important UI elements in multiple frames.

1: Very poor quality - Text barely legible across frames, severe artifacts, major blurring throughout, strong color distortions. Most text is extremely difficult or impossible to read, with major quality problems affecting the entire image.

0: Unusable quality - Text completely unreadable in most frames, extreme artifacts, totally blurred, colors meaningless. The visual quality is so poor that the UI cannot be evaluated or used for any practical purpose.

###### Figure A3 | Evaluation Rubrics for Multi-Step Planning Generation

[Figure 136]

#### JUDGE RUBRICS -- Type3/4: Fiction-App/Real-App

###### GOAL (Final Goal Achievement): Does Frame5 achieve the Final Goal?

5: Perfect achievement - Frame5 shows EXACTLY what Final Goal describes, all objectives fully met with no ambiguity. The final state precisely matches the intended outcome, with all key elements correctly implemented and clearly visible.

4: Strong achievement - Final Goal achieved with only minor omissions or slight deviations that don't affect core objectives. Frame5 accurately reflects the goal with only non-critical details missing or slightly different from the exact description.

3: Partial achievement - Some aspects of Final Goal met, but significant elements are missing or incorrectly implemented. Frame5 shows progress toward the goal but key components are absent, wrong, or ambiguous, requiring inference to see the connection.

2: Weak achievement - Only minimal progress toward Final Goal, most key objectives not met. Frame5 barely resembles the intended outcome, with most critical elements missing or wrong, showing only surface-level similarity.

1: Minimal achievement - Almost no progress toward Final Goal, only superficial similarity. Frame5 might share some basic UI structure with the goal but shows virtually no meaningful progress toward the stated objective.

0: No achievement - Frame5 completely unrelated to Final Goal or shows wrong outcome. The trajectory ends in a state that contradicts or has no meaningful connection to the desired final state.

###### LOGIC (Action-Visual Alignment & Step Coherence): Do actions logically lead to visual changes described?

5: Perfect logic - Every action naturally and plausibly causes the visual changes described in the Visual Description, with coherent step-by-step progression. Each action-result pair is believable and follows natural GUI interaction patterns without requiring unexplained assumptions.

4: Strong logic - Actions mostly lead to expected visual changes, minor logical gaps but overall progression makes sense. The action-visual alignment is mostly correct, though one or two steps might have slight inconsistencies that don't break the overall coherence.

3: Partial logic - Some actions align with visual changes, but several steps show logical jumps or unclear causation. The progression requires making assumptions about missing intermediate actions, and some steps don't clearly follow from their described actions.

2: Weak logic - Actions and visual changes poorly connected, frequent logical inconsistencies across steps. Multiple actions don't realistically cause the described visual changes, and the progression shows significant gaps in causality that make the trajectory confusing.

1: Minimal logic - Actions barely relate to visual changes, progression is confusing or nonsensical. Most actions are disconnected from the resulting UI states, requiring major leaps of logic to understand how one step leads to the next.

0: No logic - Actions completely disconnected from visual changes, no coherent progression through steps. The action descriptions and visual results appear unrelated, showing no meaningful causal relationship throughout the trajectory.

###### CONS (Visual Consistency Across Frames): Are UI elements stable across all 5 frames?

5: Perfect consistency - ALL UI elements (system bars, layout, style, colors) remain completely stable across all frames, zero drift. Every unchanged component maintains identical position, appearance, and state throughout the entire trajectory with perfect preservation.

4: Strong consistency - Minor variations in some elements (slight color shifts, subtle positioning) but core UI remains stable. Background elements might show nearly imperceptible changes, but all major components stay consistent and recognizable across frames.

3: Partial consistency - Noticeable drift in some UI elements between frames, but major components remain recognizable. Some elements shift position, change color slightly, or show minor flicker, but the general layout and all UI components remain identifiable.

2: Weak consistency - Significant inconsistencies across frames: layout shifts, style changes, unstable UI elements. Background components move noticeably, colors change substantially, or there's clear flicker/corruption in multiple frames affecting UI stability.

1: Minimal consistency - UI changes substantially across frames, few elements remain stable, major visual drift. Most background elements show significant movement or change, with only a few components maintaining their position/appearance throughout.

0: No consistency - Completely different UI across frames, no recognizable stability or coherent interface. Each frame appears to show a different UI layout, style, or completely unrelated interface elements with no continuity.

UI (UI Plausibility/Integrity): Are UI components in each frame native-looking and coherent?

5: Perfect integrity - All UI components follow platform conventions, proper layering, correct states, no visual anomalies, text perfectly rendered in EVERY frame. Every frame shows professional-quality UI that looks indistinguishable from a real native application interface.

4: Strong integrity - UI mostly follows conventions with minor layering issues or subtle anomalies, text mostly clear across frames. There might be occasional minor issues like slight misalignment or subtle rendering problems, but overall UI quality remains high.

3: Partial integrity - Some UI elements appear slightly off (wrong sizes, minor misalignment, occasional text artifacts) but still plausible in most frames. The UI remains recognizable and generally functional despite several noticeable imperfections.

2: Weak integrity - Noticeable UI problems in multiple frames: incorrect states, layering errors, text clarity issues, minor hallucinations. Multiple frames show significant UI defects that affect the believability and usability of the interface.

1: Poor integrity - Significant UI breakdown across frames: impossible states, major layering violations, unreadable text, clear hallucinations. Most frames contain serious UI problems that make the interface appear broken or nonsensical.

0: No integrity - UI completely nonsensical across frames, with hallucinated controls, impossible layouts, or no recognizable interface elements. The trajectory shows no coherent UI structure, appearing as random visual elements.

QUAL (Visual Quality): Are text/icons readable and artifacts minimal across all frames?

5: Excellent quality - Crystal clear text/icons in ALL frames, sharp edges, no compression artifacts, perfect color accuracy. Every frame shows professional rendering quality with crisp, easily readable text and sharp iconography.

4: Good quality - Text/icons clear and readable across frames, minor compression artifacts or slight blurring at edges only. Slight quality degradation may be visible but doesn't affect readability or overall visual quality.

3: Fair quality - Text mostly readable but shows some artifacts in multiple frames, icons slightly blurry, minor color distortions. Quality issues are noticeable and affect some regions, but important text and icons remain legible.

2: Poor quality - Text becoming difficult to read in several frames, noticeable artifacts, significant blurring, color issues. Quality degradation is substantial and affects readability of important UI elements in multiple frames.

1: Very poor quality - Text barely legible across frames, severe artifacts, major blurring throughout, strong color distortions. Most text is extremely difficult or impossible to read, with major quality problems affecting the entire image.

0: Unusable quality - Text completely unreadable in most frames, extreme artifacts, totally blurred, colors meaningless. The visual quality is so poor that the UI cannot be evaluated or used for any practical purpose.

###### Figure A4 | Evaluation Rubrics for Zero-shot Virtual GUI Generation and Rare Trajectory Synthesis

[Figure 137]

#### JUDGE RUBRICS -- Type5: Grounding

GOAL (Effect Alignment): Does Generated Image match Expected Effect?

5: Perfect match - Effect achieved exactly as described, all mentioned elements present and correctly transformed with no ambiguity. 4: Strong match - Effect achieved with only minor omissions or slight deviations that don't affect core semantics. 3: Partial match - Some aspects match Expected Effect, but significant elements are missing or incorrectly implemented. 2: Weak match - Change vaguely relates to Expected Effect but misses most key elements or introduces wrong transformations. 1: Minimal match - Almost no correspondence between change and Expected Effect, only superficial similarity.

- 0: No match - Change completely contradicts or is unrelated to Expected Effect.

Goal checklist (use to justify scoring):

- - Correct UI outcome happened (e.g., dialog opened, dropdown expanded, tab switched, checkbox toggled, page navigated).
- - Direction is correct (open vs close, enable vs disable, selected vs unselected).
- - If Expected Effect mentions specific text labels/selected items, they must be visible and correct for 4-5.
- - If Expected Effect implies a very specific screen/state, wrong screen/state caps goal at 2.

LOGIC (Trigger Localization): Is change plausibly triggered by tapping given point/region?

5: Perfect localization - Change originates PRECISELY from tapped element/region with appropriate ripple effects and spatial reasoning. 4: Strong localization - Change originates from area very close to tap point, minor spatial offset but clearly connected and plausible. 3: Partial localization - Change appears in vicinity of tap but connection is ambiguous or spans multiple unrelated UI elements. 2: Weak localization - Change appears in entirely different region from tap point, showing poor spatial understanding.

- 1: Minimal localization - Only trace evidence that change might relate to tap location, mostly disconnected.

- 0: No localization - Change completely unrelated to tap location or affects random/global areas without spatial coherence.

Logic checklist:

- - Change should be anchored to tapped widget/region or its plausible dependent area.
- - Plausible: button→modal/dialog, tab→content panel, menu item→navigation, toggle→state change.
- - Implausible: unrelated far-away changes, global theme swaps, random text/icon rewrites, unexplained page jumps.

CONS (Background Consistency): Are non-affected regions preserved relative to Initial Image?

5: Perfect consistency - EVERY pixel outside directly affected area is IDENTICAL between Initial and Generated images. 4: Strong consistency - Minor variations in non-affected regions (subtle color shifts, compression artifacts) but layout/content unchanged. 3: Partial consistency - Some background elements shift or change slightly, but major layout remains stable. 2: Weak consistency - Noticeable drift in background elements, text changes, or layout modifications in supposedly unchanged areas.

- 1: Minimal consistency - Background shows significant changes but retains some similar elements or overall structure.

- 0: No consistency - Background completely different, showing no preservation of unchanged regions.

Consistency guidance:

- - A tap should not rewrite large unrelated regions. If large areas change without clear reason, cons should be ≤2.
- - Minor compression/resampling artifacts are acceptable for 4; new/changed text in unrelated areas is not.

UI (UI Plausibility/Integrity): Are UI components coherent (native-looking, correct layering, no hallucinations)?

5: Perfect integrity - All UI components follow platform conventions, proper layering, correct states, no visual anomalies, text perfectly rendered and readable. 4: Strong integrity - UI mostly follows conventions with minor layering issues or subtle anomalies, text mostly clear with minimal artifacts. 3: Partial integrity - Some UI elements appear slightly off (wrong sizes, minor misalignment, occasional text artifacts) but still recognizable. 2: Weak integrity - Noticeable UI problems: incorrect component states, layering errors, text clarity issues, minor hallucinations or impossible layouts.

- 1: Poor integrity - Significant UI breakdown: impossible states, major layering violations, unreadable text, clear hallucinations of non-existent controls.

- 0: No integrity - UI completely nonsensical with hallucinated controls, impossible layouts, or no recognizable interface elements.

UI checklist:

- - Correct component hierarchy (e.g., modal overlays the page; dropdown anchored to its control).
- - Correct affordances and states (pressed/selected/disabled look plausible).
- - No duplicated/smeared labels; consistent fonts and spacing.

QUAL (Visual Quality): Are text/icons readable and artifacts minimal?

5: Excellent quality - Crystal clear text and icons, sharp edges, no compression artifacts, perfect color accuracy, professional rendering. 4: Good quality - Text/icons clear and readable, minor compression artifacts or slight blurring at edges only. 3: Fair quality - Text mostly readable but shows some artifacts, icons slightly blurry, minor color distortions present. 2: Poor quality - Text becoming difficult to read, noticeable artifacts throughout, significant blurring, color issues.

- 1: Very poor quality - Text barely legible, severe artifacts, major blurring throughout, strong color distortions. 0: Unusable quality - Text completely unreadable, extreme artifacts, totally blurred, colors meaningless.

Figure A5 | Evaluation Rubrics for Grounding-based Generation

