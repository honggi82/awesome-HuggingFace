## CutVerse: A Compositional GUI Agents Benchmark for Media Post-Production Editing

Haobo Hu∗1, Xiangwu Guo∗2, Zhiheng Chen2, Difei Gao3, Haotian Liu1,Libiao Jin1,Qi Mao1†

1MIPG, Communication University of China, 2National University of Singapore,3USEIT AI https://github.com/CUC-MIPG/CutVerse

# arXiv:2605.19484v1[cs.CV]19May2026

|[Figure 1]<br><br>Step 1: Generate multiple short video segments using video generation models.<br>Step 2: Edit and integrate these segments into a complete long-form video using professional video editing software.<br><br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>…… ~10s Video clips<br><br>[Figure 7]<br><br>[Figure 8]<br><br>I wanna create a viral AI video<br><br>like this<br><br>[Figure 9]<br><br>Too difficult and time-consuming!<br><br>[Figure 10]|
|---|

|Let’s try Vibe Cutting !! NoProblem!<br><br>[Figure 11]<br><br>0:20/03:32<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Use Keling to generate a video of Drog battling Godzilla<br><br>Video color grading with DaVinci Resolve<br><br>Use Premiere Pro to add continuous horizontal flipping<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>Help me create a 3-minute video<br><br>about ‘Drog VS Godzilla’.|
|---|

Figure 1 CutVerse: A benchmark for evaluating GUI agents in media post-production. Top: Existing AI video creation pipelines require manual composition of generated clips within professional editing software. Bottom: CutVerse evaluates GUI agents on realistic post-production tasks across diverse professional tools, covering complete workflows such as timeline editing, visual effects, audio alignment, and content composition through real software interaction.

#### Abstract

While GUI agents have made significant progress in web navigation and basic operating system tasks, their capabilities in professional creative workflows remain largely underexplored. To bridge this gap, we introduce CutVerse, a benchmark designed to systematically evaluate autonomous GUI agents in realistic media postproduction environments. We curate expert demonstrations across 7 professional applications (e.g., Premiere Pro, Photoshop), covering 186 complex, long-horizon tasks grounded in authentic editing workflows, involving dense multimodal interfaces and tightly coupled interaction sequences. To support scalable evaluation, we develop a lightweight parser that transforms raw screen recordings and low-level interaction logs into structured, compositional GUI action trajectories with precise grounding. Extensive evaluations reveal that existing agents achieve only 36.0% task success on realistic media editing tasks, underscoring the challenges posed by complex, long-horizon media post-production workflows in our benchmark.While current models demonstrate promising spatial grounding, multimodal alignment, and coordinated action execution, they remain limited in long-horizon reliability and domain-specific planning.

∗Equal contribution. †Corresponding authors.

#### Contents

- 1 Introduction 3
- 2 Related Work 4

- 2.1 AIGC Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 GUI Agents and Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Media Creative Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3 The CutVerse Benchmark 5

- 3.1 Task Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Scalable Evaluation Infrastructure and Capability Decomposition . . . . . . . . . . . . . . . . . 6
- 3.3 Dataset Construction and Statistical Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.4 Online Execution and Automated Milestone Assessment . . . . . . . . . . . . . . . . . . . . . . 8

- 4 Baseline 9

- 4.1 Baselines Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 5 Analysis 10

- 5.1 Milestone-Task Consistency Gap . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.2 Media Applications Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.3 Long-Horizon Multimodal Task Difficulty . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.4 Missing Compositional Action Space . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.5 Qualitative Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 6 Conclusion 13

Appendix 14

- A Details for Benchmark 14
- B Additional Experimental Results 23

#### 1 Introduction

The development of computer-use agents (CUA)[13, 14] emerges as a promising direction for bridging natural language instructions with executable actions in software environments. By leveraging vision-language models[5, 11], these agents can perceive screen content[32, 54] and generate coherent interaction sequences[51], enabling automation across a wide range of web and desktop applications. Recent advances demonstrate strong capabilities in structured tasks, including web navigation [49, 62], official software operation [2, 48], and basic system-level interactions [19, 44], marking an important step toward general-purpose computer-use agents[47]. As agents master these general-purpose domains, their capability boundaries remain fundamentally underexplored when confronted with the intricate, unstructured demands of highly professional real-world workflows.

A representative yet underexplored domain is media post-production. Compared to existing scenarios, professional creative software presents substantially higher interface density[59], more fine-grained and intricate interaction patterns, and significantly longer execution horizons. Users must orchestrate a sequence of tightly coupled operations, including timeline manipulation, layer composition, parameter tuning, and cross-modal alignment between audio and visual signals. Such workflows impose strong requirements on spatial precision, temporal consistency, and coordinated multi-modal control, posing fundamental challenges that are not captured by current evaluation settings.

However, evaluating CUA agents in media post-production further introduces substantial system-level and infrastructural challenges. Unlike conventional benchmarks, which operate in lightweight and relatively stable environments, media editing workflows involve significantly higher memory footprints, complex and continuously evolving system states, and substantially more diverse and longer action trajectories. These characteristics place strict demands on environment reproducibility, state management, and execution stability. Existing benchmarks and datasets are not designed to support such high-fidelity, resource-intensive scenarios, making it difficult to reliably instantiate and evaluate agent behavior in realistic media production settings.

These limitations highlight the need for an evaluation framework that captures the complexity of real-world creative workflows, including continuous GUI interaction, multimodal perception, and long-horizon execution. To address these challenges, we introduce CutVerse, a benchmark designed to systematically evaluate CUA agents in realistic media post-production environments. We further build a robust infrastructure that includes (i) a lightweight parser that transforms raw multimodal interaction logs into structured GUI trajectories with grounding annotations, and (ii) a Windows-based virtual environment that enables agents to execute actions directly within software to support scalable and reproducible evaluation.

In parallel, AIGC-based pipelines primarily target high-level semantic alignment and visual consistency [10, 15, 24], while code-driven approaches are often limited to simple operations such as direct video stitching.

Both paradigms struggle to support fine-grained editing under fixed source content, including layer-wise color grading, geometric transformations, and precise transition effects that are fundamental to professional post-production. To bridge this gap, our benchmark is grounded in complete, real-world media post-production workflows, comprising 186 well-designed tasks across 7 professional software platforms, each paired with a specific virtual machine checkpoint and manually recorded interaction trajectories to faithfully capture authentic editing processes for realistic agent evaluation.

Asset import

Timeline editing

[Figure 32]

and management

and arrangement

24

28

[Figure 33]

[Figure 34]

Preview, check and validation

Audio and rhythm editing

14

23

[Figure 35]

[Figure 36]

Masking, matting, and tracking

10

CutVerse

24

[Figure 37]

[Figure 38]

9

Launch and setup

[Figure 39]

[Figure 40]

8

Generative workflow

51

Effects and visual tuning

29

Export and delivery

Extensive experiments reveal a substantial performance gap. Even the strongest models struggle with sustained execution in complex workflows, exhibiting failures in spatial grounding, temporal coordination, and compositional interaction. These results suggest that current agents, while effective in simplified domains, remain far

Figure 2 CutVerse task and software ecosystem. The inner circle displays integrated postproduction applications. The outer ring categorizes 186 human-verified tasks across nine domains.

from reliable deployment in professional creative environments. Beyond benchmarking, our findings point toward a broader paradigm for AI-assisted media production, which we term Vibe Cutting, where generation provides multimodal assets and agents transform them into structured outputs through real software interaction, as illustrated in Fig. 1. As a broader vision, we anticipate that CutVerse will provide a practical foundation for advancing end-to-end multimedia production.

Our contributions are summarized as follows:

- • We introduce CutVerse, a comprehensive dataset comprising 186 complex, long-horizon tasks across 7 professional applications, specifically targeting realistic media post-production workflows.
- • We build an end-to-end pipeline consisting of a infrastructure parser that converts raw multimodal logs into structured GUI trajectories, and a Windows VM-based evaluation environment for authentic agent execution.
- • We design fine-grained evaluation metrics that move beyond traditional Success Rates (SR) to strictly reflect the fine-grained operations and specific characteristics of creative applications.
- • Extensive evaluations of state-of-the-art VLMs reveal a striking performance gap, exposing critical bottlenecks in handling spatially dense layouts and compositional GUI actions.

#### 2 Related Work

##### 2.1 AIGC Agents

Recent AIGC agents leverage planner-executor paradigms [46, 56] and tool augmentation [39] to automate multimodal content generation [10, 15, 24, 25, 42, 43, 45, 58, 60]. However, these frameworks predominantly target coarse-grained semantic alignment and high-level visual consistency. When confronted with the rigorous demands of professional multimedia post-production, including fine-grained video effects (VFX), precise timeline manipulations, and complex transition editing, existing AIGC architectures prove fundamentally inadequate. They currently lack the execution granularity required to navigate the intricate, trivial operational workflows essential for professional-grade media post-production.

##### 2.2 GUI Agents and Benchmarks

While recent VLM-based GUI agents [3, 9, 12, 23, 28, 36, 40, 50, 52, 57] exhibit strong interactive capabilities across general-purpose domains [7, 21, 31, 34, 37] like web navigation [6, 19, 20, 49, 62] and operating systems [2, 27, 30, 33, 38, 41, 44, 48, 55], they aim to bridge natural language instructions and executable actions within interactive software environments. However, the specialized domain of media post-production remains severely underexplored. Professional editing environments present unique challenges characterized by exceptionally dense interface layouts and long-horizon operational sequences. Because existing GUI benchmarks are largely constrained to simplified and short-step interactions, they are incapable of effectively evaluating the complex, multi-step execution trajectories inherent to real-world editing workflows.

##### 2.3 Media Creative Benchmarks

Existing media creative benchmarks [4, 16–18, 26, 29, 61, 63] have driven significant advancements in assessing the high-dimensional perceptual quality and semantic fidelity of generated multimodal content. Nevertheless, these evaluations remain fundamentally output-oriented. There is a critical absence of standardized protocols capable of comprehensively evaluating the interaction density of professional creative tools, specifically the precise cutting actions and dynamic effect tuning executed during the creation process. To address this gap, CutVerse introduces a rigorous evaluation standard that shifts the focus from static output assessment to the dynamic, trajectory-based verification of professional media manipulation.

###### Recording Workflows

###### Parsing Recordings into Trajectories Evaluation Using Parsed Structured Trajectories

[Figure 41]

[Figure 42]

###### Trajectory

###### Milestone QA Evaluator

[Figure 43]

Recorder

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Real-world Post-production Live Environments

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Action Caption Visual Observation Spatial Grounding

Evaluating Task 1 ……

[Figure 52]

[Figure 53]

“Before & After” “Think” “Action” “Expectation”

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

…

[Figure 60]

Is the Paint Bucket Tool icon replaced by the Gradient Tool icon in the top-left toolbar?

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

…

[Figure 65]

Atomic Actions Single-application Cross-application

[Figure 66]

[Figure 67]

[Figure 68]

- Milestone 1 Question

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- Milestone 2 Question

[Figure 73]

Is the tool selection popup visible in the top-left corner with the Gradient Tool highlighted?

[Figure 74]

[Figure 75]

###### Hierarchy Milestone 1

[Figure 76]

[Figure 77]

MKV Video Interaction Event Timestamp

[Figure 78]

Milestone 2

Milestone 3

Delayed Feedback Error Accumulation

[Figure 79]

[Figure 80]

[Figure 81]

- Step 0
- Step 1

Step 2 Step 3 Step 5 Step 7

[Figure 82]

VM Checkpoints

[Figure 83]

Step 16

[Figure 84]

[Figure 85]

[Figure 86]

— Task 1 Failed.

[Figure 87]

Atomic

Task

Figure 3 The CutVerse data and evaluation pipeline. (1) Recording: Capturing synchronized expert workflows across professional applications. (2) Parsing: Structuring raw data into milestone-driven trajectories with rich spatiotemporal grounding. (3) Evaluation: Assessing full agent trajectories in live environments via a post-hoc Milestone QA Evaluator. Although the entire task is executed, intermediate milestone failures still dictate overall task failure, authentically exposing error accumulation in long-horizon editing.

- Table 1 Comparison of GUI agent benchmarks across platforms and workflow complexity. Media Tier denotes the professional level of supported multimedia creation environments, categorized into Basic (lightweight/consumer-grade tools) and Pro (paid professional post-production software, e.g., Adobe Premiere Pro and After Effects); E2E Workflow indicates the inclusion of compositional, long-horizon editing workflows that culminate in a final exported video product, as opposed to isolated atomic kills; AIGC indicates the integration of generative AI tools and pipelines; Human refers to human-curated data; Env. indicates the availability of live, executable environments for interactive evaluation.

Media Tier

E2E Workflow

Benchmark Platform Tasks

AIGC Human Env.

Web & Mobile

Mind2Web [6] Web 2,350 ✗ ✗ ✗ ✓ ✗ AgentTrek [49] Web 10,398 ✗ ✗ ✗ ✗ ✗ AITW [37] Mobile 715K ✗ ✗ ✗ Mix. ✗ GUI-Odyssey [31] Mobile 8,334 ✗ ✗ ✗ Mix. ✗

###### Desktop & General

OmniACT [19] Desktop + Web 9,802 ✗ ✗ ✗ ✓ Partial OpenCUA [44] Desktop 22,625 Basic ✗ ✗ ✓ ✗ VideoGUI [27] Desktop 178 Pro ✗ ✗ Mix. ✗ ScaleCUA [30] Cross-platform ∼19K Basic ✗ ✗ Mix. ✗ Window Agent Arena [2] Desktop 154 ✗ ✗ ✗ Mix. ✓ OSWorld [48] Desktop 369 ✗ ✗ ✗ ✓ ✓ macOSWorld [55] Desktop 230 ✗ ✗ ✗ ✓ ✓

CutVerse (Ours) Desktop 186 Pro ✓ ✓ ✓ ✓

#### 3 The CutVerse Benchmark

To systematically evaluate GUI agents in realistic creative workflows, we introduce CutVerse, a comprehensive benchmark engineered to bridge the critical “last mile” between isolated artificial intelligence-generated content (AIGC) and production-ready media. To encapsulate real-world complexities into a highly scalable evaluation

- Table 2 Task-centric analysis of CutVerse. We reorganize workflows by task type, combining distribution statistics with interaction complexity and functional coverage.

Functional Coverage Task Type Primary Software Count Ratio Avg.

Avg. Steps

Complexity

Duration

Edit Audio VFX Motion Color AIGC Asset

Effects and visual tuning After Effects / Photoshop 51 27.4% 52.81 20.27 Extreme ✓ ✓ ✓ ✓ Export and delivery All Platforms 29 15.6% 48.98 15.41 High ✓ ✓ Asset import and management All Platforms 24 12.9% 43.01 20.83 High ✓ ✓ Audio and rhythm editing Premiere Pro / JianYing 23 12.4% 45.94 26.00 High ✓ ✓ Timeline editing and arrangement Premiere Pro / DaVinci 18 9.7% 46.48 23.67 High ✓ ✓ ✓ Preview, check, and validation All Platforms 14 7.5% 22.01 5.50 Medium ✓ ✓ Masking, matting, and tracking After Effects / Photoshop 10 5.4% 72.98 25.40 Extreme ✓ ✓ ✓ ✓ Launch and setup All Platforms 9 4.8% 31.18 7.56 Low ✓ Generative workflow ComfyUI / Keling 8 4.3% 35.45 10.00 Medium ✓ ✓

infrastructure, CutVerse formulates an end-to-end pipeline comprising high-fidelity data recording, structural multimodal parsing, and systematic dual-mode evaluation.

##### 3.1 Task Formulation

Unlike conventional benchmarks targeting static webpages, agents in CutVerse are immersed in dynamic workspaces characterized by severe multimodal information overload and high resource complexity. Professional editing necessitates managing extensive asset libraries, continuous audio waveforms, and dense parameter panels simultaneously. Crucially, successful navigation in this environment demands a dual-tiered visual perception capability: beyond basic UI widget localization, agents must exhibit profound media content comprehension. They are required to interpret the semantic, aesthetic, and temporal nuances of the underlying visual and auditory streams to formulate contextually appropriate editing decisions.

Furthermore, CutVerse redefines the benchmark task as a holistic objective requiring rigorous cross-modal alignment and spatiotemporal synchronization—such as precisely aligning an audio effect to a specific dynamic event within a video frame. To mirror modern creative paradigms, our tasks mandate seamless cross-application workflows, dictating that agents fluently orchestrate operations from generating raw visual assets via AIGC node-based interfaces (e.g., ComfyUI) to refining and composing them within traditional nonlinear editing platforms (e.g., Adobe Premiere Pro).

To faithfully reflect these steep technical barriers, CutVerse enforces an anthropomorphic action space entirely grounded in vision-only perception. Rather than executing privileged software APIs, agents are compelled to embody the human motor-cognitive loop. They interact with the software exclusively through continuous mouse drag-and-drops, precise coordinate-based clicks, and complex keyboard shortcut combinations. In stark contrast to web tasks driven by structured HTML DOM trees, media post-production relies heavily on unstructured multi-track timelines. Executing tasks within this space necessitates rigorous pixel-level grounding, compelling agents to translate high-level creative intentions into concrete physical operations precisely as a human creator would.

##### 3.2 Scalable Evaluation Infrastructure and Capability Decomposition

To rigorously support the dynamic and human-centric nature of media post-production, CutVerse is instantiated upon a robust, scalable evaluation infrastructure powered by a custom Windows virtualization engine. Crucially, to genuinely evaluate agents in authentic scenarios, the environment enforces a strict human-aligned execution paradigm. Rather than relying on privileged backend APIs—which are largely nonexistent in professional creative suites—the engine isolates each task within a resettable virtual machine, restricting agent interactions entirely to simulated, low-level mouse and keyboard events driven by live visual feedback. This architectural design forces autonomous agents to operate under the exact systemic and cognitive constraints as human professionals. Furthermore, precise state checkpointing guarantees systemic reproducibility and visual consistency across large-scale evaluations, ensuring a reliable testbed without compromising the live, interactive nature of the host operating system.

Operating synergistically with this execution engine is a dedicated multimodal parsing pipeline, designed to transform unstructured human demonstrations into evaluable, structured formats. Specifically, the parser meticulously synchronizes high-framerate screen recordings with low-level I/O event logs, extracting spatiotemporally aligned action sequences. Through this rigorous alignment, continuous expert workflows—comprising natural language instructions, raw video frames, and complex keystrokes—are translated into structured multimodal trajectories. Each discrete step is firmly grounded in its corresponding visual state and semantic context, thereby effectively bridging the semantic gap between continuous pixel arrays and actionable agent representations.

Beyond simple trajectory extraction, this parsing infrastructure introduces a profound paradigm shift by decomposing long-horizon, monolithic workflows into hierarchical semantic milestones, which we subsequently map to transferable atomic capabilities. While specific post-production tasks exhibit massive combinatorial diversity (e.g., color grading a cinematic shot versus synthesizing a dynamic transition), their underlying milestones rely on a finite set of foundational, cross-domain skills. These include temporal navigation across multi-track timelines, granular parameter fine-tuning, and cross-modal asset retrieval. By decoupling complex tasks into these atomic capabilities, CutVerse achieves an unprecedented level of fine-grained diagnostic resolution. It transcends binary task success rates, empowering researchers to precisely quantify whether an agent has acquired generalizable editing skills that can seamlessly transfer across disparate generative tools and traditional software ecosystems.

##### 3.3 Dataset Construction and Statistical Complexity

Constructed atop our robust virtualization infrastructure, the CutVerse dataset encapsulates the authentic complexity of modern media workflows through 2.43 hours of high-fidelity recording. As illustrated in Figure 2 and detailed in Table 2, it yields 186 human-verified tasks and 3,484 atomic GUI interactions (averaging 23.8 interactions per minute) spanning nine functional domains. Beyond sheer scale, CutVerse covers the entire production pipeline—from procedural asset management to complex visual tuning. Crucially, we push the benchmark beyond traditional industry-standard software (e.g., Adobe Premiere Pro, After Effects) by incorporating interactions with emerging generative platforms like Keling, Jimeng, and ComfyUI. This hybrid composition effectively mirrors contemporary creative paradigms, where users fluidly orchestrate end-to-end workflows by synthesizing raw AIGC materials and subsequently refining them within conventional editing ecosystems.

A defining characteristic of CutVerse is its pronounced long-horizon complexity, which stringently evaluates an agent’s capacity for sustained planning and continuous multimodal context maintenance. The dataset exhibits a severe long-tail distribution, averaging 18.73 steps per trajectory—substantially surpassing standard web-navigation benchmarks—with peak execution horizons reaching 239 steps. To systematically quantify this, we stratify workflows along a complexity spectrum from Low to Extreme (see Table 2). Notably, tasks demanding dense cross-modal alignment and fine-grained spatiotemporal precision—such as audio rhythm editing (averaging 26.00 steps) or masking and tracking (25.40 steps)—exhibit exceptionally high interaction density. This challenge is further exacerbated across software boundaries, where cross-application workflows escalate to an average of 21.20 steps compared to 17.56 for isolated applications. Consequently, mastering CutVerse necessitates reliably executing prolonged sequences of compositional operations without losing the overarching semantic goal.

Finally, our statistical analysis exposes the exceptional visual parsing threshold inherent to professional creative environments. A breakdown of interacting UI elements reveals that timelines completely dominate the visual focus, accounting for 46.07% of all operations, followed closely by complex layer and track controls at 25.32%. In stark contrast to standard web DOMs populated by discrete HTML buttons, timelines operate as unstructured, spatiotemporal interfaces demanding granular spatial adjustment and continuous coordination (e.g., continuous drag-and-drop, precise multi-key combos). The overwhelming prevalence of these elements definitively shifts the evaluation bottleneck from simple point-and-click navigation to maintaining pixel-level audio-visual grounding amidst severe multimodal information overload.

- Table 3 Unified task and milestone success rates across operation categories. Models demonstrate strong capabilities in procedural setup and basic file management, such as generative workflows, software launching, and exporting. However, performance degrades significantly when executing core media editing tasks. The stark contrast between local milestone success and overall task success highlights a fundamental weakness in complex content manipulation, audio coordination, and precise visual tuning.

Task Success Rate Milestone Success Rate Qwen3

Task Category

UI-TARS

Claude

Gemini3

EvoCUA

Qwen3

UI-TARS

Claude

Gemini3

EvoCUA

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

-32B-T [53]

-1.5-7B [36]

###### -32B [52] Procedural Setup and File Management

-Opus-4.6 [1]

-flash [8]

-32B [52]

-32B-T [53]

-1.5-7B [36]

-Opus-4.6 [1]

-flash [8]

Generative Workflow (GW) 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 Export and Delivery (ED) 0.750 0.917 1.000 0.917 0.917 0.767 0.833 0.967 1.000 0.900 Launch and Setup (LS) 0.900 0.900 0.933 0.967 0.767 0.803 0.752 0.897 0.872 0.786 Preview, Check, and Validation (PCV) 0.800 0.600 0.800 0.800 0.800 0.737 0.579 0.947 0.842 0.737 Asset Import and Management (AIM) 0.421 0.333 0.719 0.667 0.456 0.605 0.542 0.814 0.757 0.588 Average (Procedural) 0.774 0.750 0.890 0.870 0.788 0.782 0.741 0.925 0.894 0.802

###### Core Media Editing and Processing

Timeline Editing and Arrangement (TEA) 0.550 0.350 0.600 0.650 0.550 0.359 0.333 0.577 0.538 0.295 Effects and Visual Tuning (EVT) 0.207 0.276 0.586 0.483 0.310 0.232 0.183 0.537 0.488 0.415 Audio and Rhythm Editing (ARE) 0.167 0.167 0.333 0.500 0.333 0.643 0.429 0.929 0.786 0.643 Masking, Matting, and Tracking (MMT) 0.143 0.095 0.286 0.381 0.238 0.368 0.439 0.649 0.605 0.395 Average (Core Editing) 0.267 0.222 0.451 0.504 0.358 0.400 0.346 0.673 0.604 0.437

Overall Performance Average (Overall) 0.484 0.441 0.683 0.672 0.516 0.532 0.502 0.748 0.704 0.552

##### 3.4 Online Execution and Automated Milestone Assessment

Given the open-ended and temporally extended nature of professional media post-production, CutVerse mandates online execution as its foundational evaluation paradigm. As Fig. 7 illustrates, core editing workflows exhibit substantially longer execution horizons and higher interaction density than procedural tasks, rendering static action prediction fundamentally insufficient. Instead, agents are deployed within live, resettable Windows virtual machines. In this closed-loop environment, agents continuously perceive high-density

workspaces—coupling dynamic video canvases, multi-track timelines, and audio waveforms—to iteratively issue low-level actions. This rigorously tests an agent’s capacity to resolve visual ambiguity, maintain temporal synchronization, and autonomously recover from cascading errors over prolonged trajectories.

[Figure 98]

- Milestone 1

Is the Paint Bucket Tool icon replaced by the Gradient Tool icon in the top-left toolbar?

Is there a faint circular outline observed at the center of the lime image on the canvas?

- Milestone 2

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Is the white-to-transparent radial gradient effect visible on the lime image in the lower right of the canvas?

Is the gradient effect enhanced and smoothly blended with the green background on the lime image?

[Figure 110]

[Figure 111]

[Figure 112]

While imperative for authentic benchmarking, online execution introduces a formidable bottleneck: evaluating open-ended, multimodal outcomes lacking deterministic programmatic verification. Unlike software engineering benchmarks governed by unit tests, operations in creative suites — such as continuous timeline dragging or non-linear spatial adjustments—produce highly contextual, non-symbolic outcomes that completely defy hard-coded heuristics. To alleviate this, we formulate a Milestonedriven Automated Evaluation Protocol. We decom-

Reason: While the agent successfully selected the Gradient

[Figure 113]

Tool, the trajectory became confused from step 17 onward

[Figure 114]

Success? False

when a ‘Create Agent from a Demonstration’ overlay appeared. The agent spent the remaining 30+ steps clicking repetitively on coordinates without clear progress. There is no clear evidence that the gradient was properly, …

Figure 4 Core media editing tasks require longer horizons. Left: average duration. Right: average steps. Orange bars are core media editing and processing tasks, which are generally higher than procedural tasks in both duration and step count.

pose monolithic task trajectories into a hierarchical sequence of semantically meaningful milestones, each encapsulating a verifiable audio-visual state transition. We then orchestrate a scalable VLM-as-a-Judge pipeline, evaluating agent progress via grounded question-answer (QA) pairs aligned with intermediate editing states. This transforms ill-posed trajectory comparison into interpretable, fine-grained verification steps.

To mitigate evaluator hallucination and architectural bias, we instantiate this protocol across distinct frontier vision-language models (i.e., GPT-5.4 [35] and Claude-4.6-Opus [1]). This multi-model grounding ensures

- Table 4 Task execution accuracy by software across models, augmented with benchmark complexity statistics. Software AvgSteps AvgDur(s) Claude Gemini Evo Qwen UI-TARS

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Keling 8.31 26.56 0.815 0.852 0.704 0.852 0.593 ComfyUI 10.33 33.30 0.667 0.833 0.500 0.500 0.500 JianYing 22.33 63.32 0.754 0.725 0.493 0.522 0.493 DaVinci 16.60 46.68 0.750 0.700 0.550 0.450 0.450 Premiere Pro 12.98 26.35 0.642 0.660 0.604 0.491 0.396 Photoshop 42.61 91.20 0.576 0.576 0.455 0.424 0.455 After Effects 14.81 47.44 0.577 0.500 0.269 0.269 0.346

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

that milestone verification relies on robust, model-agnostic visual comprehension rather than evaluator-specific leniency, establishing a principled foundation for assessing complex audio-visual transformations.

To empirically validate this automated infrastructure, we conducted a comprehensive human-alignment study across 300 agent-executed trajectories. Parallel assessments by professional creators and our QA-grounded VLM evaluators demonstrated exceptional concordance: a 98.3% human-agreement rate using GPT-5.4 [35], and 99% with Claude-4.6-Opus [1]. These findings unequivocally prove that our milestone protocol empowers automated models to match expert-level judgment. Consequently, CutVerse successfully relegates human experts to the role of ultimate ground-truth curators, establishing a fully scalable, reproducible, and scientifically rigorous evaluation pipeline for multimodal agentic systems.

#### 4 Baseline

##### 4.1 Baselines Setup

We benchmark a diverse set of current large-scale vision-language models in CutVerse under a unified online execution framework. Specifically, we evaluate state-of-the-art proprietary models, namely Claude-Opus-4.6 [1] and Gemini-3-flash [8], accessed via their official APIs. Concurrently, we locally deploy leading open-source models, including Qwen3-32B [53], UI-TARS-1.5-7B [36], and EvoCUA-32B [52], on a hardware cluster equipped with four NVIDIA RTX 5090 GPUs. All models are prompted to generate structured GUI actions and tool calls based on task descriptions and visual observations. Evaluations are conducted exclusively in an online setting, where agents interact with a live environment and perceive real-time screenshots and state changes. To ensure rigorous and fair comparisons, all models execute the identical set of tasks within completely standardized Windows 11 Pro virtual machines powered by Hyper-V. Each testing episode is strictly initialized with the exact same system states, source files, input formats, and software configurations.

Evaluation Setting. We evaluate under a task-level action execution setting that reflects realistic agent deployment. At each step, the model receives the overall high-level task instruction, the historical context consisting of the last k=5 actual execution screenshots alongside their natural language descriptions and pyautogui code, and the current keyframe screenshot. Crucially, we move beyond passive prediction. Our framework requires the agent to actually execute the inferred pyautogui operations directly within the live virtual machine. The model must autonomously determine the next action based solely on the task goal and multimodal history without relying on step-level instructions. This closed-loop setup accurately mirrors how autonomous agents operate within practical post-production workflows.

##### 4.2 Results

High Proficiency in Procedural Operations. As presented in Table 3, all evaluated models exhibit robust capabilities within the procedural setup and file management category. Notably, every model achieves a perfect success rate of 1.000 at both the task and milestone levels for generative workflows. Performance remains highly competitive in operations such as export and delivery, as well as launch and setup. In these areas, Claude-Opus-4.6 [1] and Gemini-3-flash [8] consistently outpace the other models. Although performance slightly decreases in asset import and management, the overall success rates in this upper section of the table indicate a strong baseline for basic software navigation.

[Figure 127]

[Figure 128]

Task Instruction Use the RotoBrush tool to segment the main character from AE_original.mp4.

| |
|---|

| |
|---|

[Figure 129]

Thought To use the Roto Brush tool, I first need to select it from the toolbar. The Roto Brush tool icon is the 13th icon from the left (excluding the Home icon). Once selected, I will need to doubleclick the layer in the Composition panel to open it in the Layer panel, as Roto Brush operations must be performed there.

[Figure 130]

[Figure 131]

Action Click (678, 63)

| |
|---|

| |
|---|

Actual Point

Expected Point

(a) Failure Case of Component Misrecognition

[Figure 132]

[Figure 133]

Task Instruction Import background music BGM_JMPR.mp3 on the desktop, and align the audio length with the video timeline.

[Figure 134]

Thought With the Razor Tool active, I will click on the audio clip at the exact point where the video

clip ends to split the audio. This will allow me to

delete the trailing portion and align the lengths.

[Figure 135]

[Figure 136]

Action Click (428,529)

| |
|---|

| |
|---|

| |
|---|

Actual Point

Expected Point

(b) Failure Case of Inaccurate Grounding

- Figure 5 Typical visual perception failures. (a) Component Misrecognition: Agents struggle to identify unlabelled tools in condensed layouts. (b) Inaccurate Grounding: Lack of pixel-level precision prevents delicate timeline operations.

Severe Degradation in Core Media Editing. Conversely, the data reveals a drastic performance drop across all models when transitioning to core media editing and processing tasks. The task success rates plummet in domains requiring precise content manipulation. For instance, in masking, matting, and tracking tasks, the task success rate drops to 0.095 for UI-TARS [36]. Even the top-performing models struggle significantly in this category, with Claude-Opus-4.6 [1] and Gemini-3-flash [8] scoring only 0.286 and 0.381, respectively. Similarly, effects and visual tuning tasks yield extremely low task success rates, bottoming out at 0.207 for Qwen3-32B [53], which clearly illustrates the complexity of these operations.

Quantitative Gap Between Milestones and Tasks. Furthermore, the dual-metric structure of the table explicitly exposes a substantial numerical gap between local milestone success rates and overall task success rates, particularly within the complex editing categories. A prominent example is observed in the audio and rhythm editing operations. While Claude achieves a remarkably high milestone success rate of 0.929 in this specific category, its overall task success rate falls sharply to 0.333. Gemini similarly drops from a 0.786 milestone success rate to a 0.500 task success rate. This consistent statistical discrepancy across EvoCUA32B [52], Qwen3[53], and UI-TARS-1.5-7B [36] confirms that achieving high accuracy on intermediate procedural steps does not quantitatively translate to the successful completion of the entire multi-step editing task.

#### 5 Analysis

In this section, we analyze model behavior in multimodal media settings and identify key bottlenecks of GUI agents.

| |
|---|

| |
|---|

LeftClick

KeyPress

Drag Avg Success

1.00

Top-3ActionRatioSum

1.00

AverageSuccess

0.75

0.75

0.540

0.50

0.50

0.372

0.300

0.229

0.25

0.25

0.00

0.00

Masking / Matting / Tracking

Effects and Visual Tuning

Audio and Rhythm Editing

Timeline Editing and Arrangement

- Figure 6 Action distribution vs. success rate in core media editing tasks. Core editing tasks exhibit a higher reliance on compositional interactions, reflected by the more balanced distributions of top-3 action types.

Procedural setup/file Core media editing

(a) Avg Duration

(b) Avg Steps

25

60

20

Seconds

Steps

15

72.98

26.00

25.40

40

23.67

20.83

20.27

52.81

48.98

10

46.48

45.94

15.41

43.01

35.45

20

31.18

10.00

22.01

5

7.56

5.50

0

0

GWEDLSPCVAIMTEAEVTAREMMT

GWEDLSPCVAIMTEAEVTAREMMT

Figure 7 Core media editing tasks require longer horizons. Left: average duration. Right: average steps. Orange bars are core media editing tasks, which are generally higher than green procedural tasks in both duration and step count.

[Figure 137]

- Figure 8 Infinite loops from static feedback. Lacking obvious visual alterations ("Vision No Change"), the agent fails to register state transitions. This perceptual blind spot traps the model in repetitive cycles of redundant clicks, halting progress. Red/green boxes denote actual/expected targets.

##### 5.1 Milestone-Task Consistency Gap

As indicated in Table 3,milestone-level performance consistently exceeds end-to-end task success across all models, indicating that current GUI agents already exhibit basic planning and execution capabilities in media editing workflows. However, this local competence fails to translate into global success. While agents reliably complete procedural milestones such as Launch, Export, and Preview, they struggle on critical stages in core editing workflows, including Effects and Visual Tuning, Audio and Rhythm Editing, etc.

This starkly demonstrates that existing GUI agents fundamentally lack the requisite planning and execution capabilities within the specialized vertical of professional media editing. We present a more comprehensive analysis regarding these domain-specific deficiencies in the subsequent sections.

##### 5.2 Media Applications Complexity

As shown in Table 4, models achieve high accuracy on structured generation tools (e.g., ComfyUI) but degrade substantially on professional editing software (e.g., After Effects). These professional environments impose dense visual layouts and demand sustained, long-horizon interactive operations. This disparity exposes a critical bottleneck: current agents fail to maintain robust cross-modal alignment and audio-visual grounding under multimodal information overload. Consequently, software complexity serves as a direct proxy for multimodal

reasoning difficulty.

##### 5.3 Long-Horizon Multimodal Task Difficulty

As illustrated in Fig. 7, task structural complexity directly dictates agent performance. Core media editing operations demand significantly longer execution horizons and higher step densities than procedural tasks. For instance, masking and tracking operations average nearly 73 seconds across 25 atomic steps, whereas procedural previewing requires merely 22 seconds and fewer than 6 steps.

This extended temporal horizon fundamentally exacerbates multimodal complexity. Throughout lengthy sequences, agents must continuously align evolving visual layouts, audio signals, and latent editing intents across dozens of interface states. Maintaining this cross-modal consistency proves highly challenging, as minor perceptual or planning errors compound irreversibly over time. This error accumulation ultimately drives the systemic failures and high incomplete ratios inherent to core media editing tasks.

##### 5.4 Missing Compositional Action Space

As shown in Fig. 6, core media editing tasks exhibit more balanced and diverse action distributions across LeftClick, KeyPress, and Drag, yet their success rates remain consistently low. This mismatch indicates that the challenge lies not in action availability, but in action coordination. Such tasks inherently require tightly coupled, compositional interactions (e.g., key–mouse combinations and temporally synchronized operations), which cannot be decomposed into independent atomic steps. As a result, current action space of GUI Agent fundamentally limit executability in complex editing workflows.

##### 5.5 Qualitative Evaluation

To elucidate the specific failure modes of current vision-language models, we conduct a qualitative analysis of their interactive execution trajectories which reveals four critical behavioral deficiencies that restrict their deployment in professional editing workflows.

Component Misrecognition and Blind Spots. Professional media software features highly condensed interfaces populated with a massive array of specialized functional components. However, evaluated agents predominantly recognize universally common icons or buttons accompanied by explicit text labels.As illustrated in Fig. 5 (a) They frequently fail to identify domain-specific tools, unlabelled toolbars, or subtle interface elements, severely limiting their ability to utilize advanced software features.

Inaccurate Fine-Grained Grounding. Current graphical user interface agents struggle significantly with precise spatial localization. As illustrated in Fig. 5 (b), when tasks require pinpoint accuracy on a video timeline or exact coordinate selection for specific canvas elements, models frequently miss the intended target. This lack of pixel-level visual grounding prevents agents from performing delicate temporal trimmings or precise spatial adjustments.

Lack of Global Perception. Current agents lack proactive visual exploration, relying on localized observations rather than verifying the global workspace state. This behavior is intrinsically linked to the Missing Compositional Action Space. Because operations like global zooming require complex key-mouse coordination, agents are mechanically restricted from acquiring macro-level context, inevitably triggering erroneous operations based on incomplete information.

Repetitive Action Loops Triggered by Static Visual Feedback. Agents rely heavily on immediate visual confirmation to verify state transitions. When an executed action produces no obvious visual alteration in the subsequent interface screenshot, the model fails to register the system state change. Consequently, the agent repeatedly issues the exact same historical action commands, ultimately trapping the execution process in an infinite operational loop, as shown in Fig. 8.

[Figure 138]

- Figure 9 Global state neglect. Confined to a localized, zoomed-in timeline view, the agent misses the macrolevel context. This myopic perception falsely suggests missing clips, triggering redundant drag operations that erroneously duplicate existing assets.

#### 6 Conclusion

This work presents CutVerse, the first systematic benchmark and scalable infrastructure for systematically evaluating computer-use agents within real-world media post-production workflows. Our findings expose a profound gap between current agent capabilities and professional creative demands. While agents handle structured procedural operations, they systematically fail during the sustained execution of complex editing tasks that necessitate precise spatial grounding, temporal coordination, and compositional control. These limitations strictly underscore the necessity of authentic, high-fidelity evaluation environments. Ultimately, it is our vision that CutVerse will serve as a practical foundation for advancing end-to-end multimedia production.

## Appendix

#### A Details for Benchmark

- Table 5 Unified atomic action space combining standard operating system actions and real user interaction traces. The vocabulary is strictly limited to low-level mouse and keyboard operations to enforce pure visual grounding.

Function Description moveTo(𝑥, 𝑦) Moves the mouse cursor to the specified screen coordinates. click(𝑥, 𝑦) Performs a left mouse click at the given coordinates. dragTo(𝑥, 𝑦) Drags the mouse cursor to the target position while holding the mouse button. scroll(Δ) Scrolls the interface vertically by a given offset. write(text) Types the specified text at the current cursor location. keyDown(𝑘) Presses and holds a keyboard key (such as Ctrl). keyUp(𝑘) Releases a previously pressed keyboard key. keyPress(𝑘) Presses and releases a keyboard key as a single atomic action. hotkey(𝑘1, 𝑘2) Executes a keyboard shortcut combination such as Ctrl+C. WAIT Agent pauses execution and waits for observable environment changes. DONE Agent declares that the task has been successfully completed. FAIL Agent determines that the task is infeasible or cannot be completed.

To ensure complete transparency and facilitate future research, this section provides a comprehensive breakdown of the CutVerse benchmark. We detail the rigorous human annotation protocol, formally define our multimodal compositional action space, and present the granular specifications for the 186 evaluation tasks across the 7 professional software platforms.

##### A.1 Human Annotation Protocol

Ensuring high-fidelity expert trajectories is critical for rigorously evaluating computer-use agents in professional workflows. To construct our dataset, we recruited a dedicated cohort of 10 professional creators possessing extensive expertise in both traditional post-production software and AIGC workflows. These experts meticulously authored the foundational data for all 186 tasks, a comprehensive procedure encompassing formal task definition, the recording of ground-truth execution videos guided by authentic task instructions, and the instantiation of dedicated Virtual Machine (VM) checkpoints for environment standardization. Following the raw data collection, the recorded execution videos were systematically processed through our proposed multimodal parsing pipeline infrastructure. This pipeline autonomously parsed the continuous interaction traces to extract high-level task milestones alongside the granular operational content of each individual step. Furthermore, we leveraged Large Language Models (LLMs) in conjunction with pre-action and post-action screenshots to generate contextually rich multimodal Question-Answer (QA) pairs. Finally, to guarantee utmost data integrity, the pipeline concluded with a rigorous human-in-the-loop refinement phase: the original expert recorders manually evaluated the quality of the extracted milestones and generated QA pairs, iteratively adjusting the textual details to ensure absolute semantic precision and alignment with the visual trajectories. Each standalone benchmark run is launched using a registry preset passed via the command-line flag –config. The preset has the form:

##### A.2 Action Space Definition

We formulate the continuous GUI interaction as a multimodal Partially Observable Markov Decision Process (POMDP). At each time step 𝑡, the observation 𝑂𝑡 = (𝐼𝑡, 𝐻𝑡) strictly encapsulates the raw, high-resolution visual interface 𝐼𝑡 ∈ ℝ𝐻×𝑊×3 and the action history 𝐻𝑡. By deliberately isolating the agent from structured

underlying metadata (e.g., accessibility trees or DOMs), we systematically enforce pure visual grounding. Consequently, this formalization authentically mirrors the inherent complexity of professional multimedia production, compelling the agent to navigate pervasive multimodal information overload exclusively via manual, compositional key-mouse operations without algorithmic shortcuts.

To robustly support this paradigm, the action space A is meticulously constrained to low-level GUI executions. As detailed in Table 5, these operations encompass precise pixel-level point-and-click mechanisms (click), sustained spatial controls strictly required for audio-visual temporal synchronization (dragTo), and semantic keyboard inputs for generative cross-modal prompting (write, hotkey). Furthermore, the vocabulary natively integrates macro-level viewport navigation (scroll), a vital capability for agents to actively alleviate spatial blind spots during complex editing. Ultimately, this scalable infrastructure allows us to rigorously benchmark whether an agent can autonomously orchestrate the long-horizon, multimodal coordination mandated by professional creative workflows.

Consequently, the action space A consists exclusively of low-level GUI operations. As detailed in Table 5, these actions encompass basic point-and-click operations requiring precise pixel-level coordinates (such as click(𝑥, 𝑦)), continuous spatial controls heavily utilized for timeline adjustments (such as dragTo(𝑥, 𝑦)), and keyboard inputs for shortcuts and parameter tuning (such as write(text) and hotkey(𝑘1, 𝑘2)). Furthermore, the vocabulary includes viewport navigation operations like scrolling, which are mandatory for agents to actively explore the macro-level workspace, alongside system control states (WAIT, DONE, FAIL). This precise formulation allows us to evaluate not merely whether an agent triggered a button, but whether it can autonomously sustain the long-horizon key-mouse coordination required by sophisticated professional software.

##### A.3 Detailed Task Specifications

Anchored in authentic multimedia post-production scenarios, our benchmark comprises 186 meticulously curated tasks designed to rigorously evaluate computer-use agents. To ensure comprehensive coverage of realworld creative demands, our expert recorders systematically aggregated a complete taxonomy of fundamental editing typologies, subsequently tailoring specific task instantiations across diverse software environments. These tasks encapsulate the full spectrum of multimodal workflows, ranging from generative cross-modal asset creation to precise audio-visual timeline synchronization and intricate visual tuning. Table 11 presents our formalized nine-category taxonomy, illustrating the distinct complexities with concrete, domain-specific examples. The complete, exhaustive list of all 186 tasks, alongside their corresponding initial states and multimodal evaluation criteria, is open-sourced and available in our project repository.

Milestone-Driven visualization of CutVerse.. In professional media post-production, execution workflows are inherently continuous and long-horizon, rendering binary success metrics insufficient for robust evaluation. To address this challenge, we introduce a granular, milestone-driven evaluation framework within CutVerse, as illustrated in Figure [Insert Figure Number]. Instead of assessing a complex task merely by its final output, our parsing infrastructure systematically decomposes the continuous execution trajectory into discrete, semantic milestones. For instance, the instruction to apply and configure a transition effect is parsed into sequential milestones, such as locating the specific effects panel, dragging the "Cross Dissolve" transition onto the targeted timeline intersection, and precisely adjusting the "Edge Feather" parameter to a specific numerical value.

Crucially, to automate and rigorously assess the completion of each intermediate step, we formulate a multimodal Question-Answer (QA) verification mechanism. For every defined milestone, context-specific QA pairs are established to interrogate the visual interface before and after the agent’s designated action. These QA pairs target precise spatial-temporal state transitions rather than superficial clicks. As demonstrated in the evaluation protocol, the system visually verifies granular interface alterations, such as confirming the appearance of a specialized ’fx’ badge on the video clip or validating that the parameter slider in the Effect Controls panel has successfully shifted to the value of 71. By anchoring the evaluation to these visually grounded QA pairs, CutVerse provides an interpretable, step-by-step diagnostic of the agent’s multimodal perception and reasoning capabilities, effectively preventing false positives and exposing the exact points of failure within sophisticated creative workflows.

[Figure 139]

- Figure 10 Milestone parsing and multimodal QA verification in CutVerse. The continuous execution of a complex editing workflow is systematically decomposed into sequential visual milestones. To rigorously assess agent performance, each milestone is coupled with specific multimodal Question-Answer (QA) pairs. These QA pairs visually interrogate the interface to verify precise spatial-temporal state transitions, ensuring interpretable and highly reliable task evaluation.

Task-Centric visualization of CutVerse. To rigorously evaluate the sustained operational capabilities of computer-use agents, CutVerse systematically deconstructs long-horizon media production pipelines into discrete, categorical tasks. As illustrated in Figure [Insert Figure Number], a complex creative objective—such as integrating a dynamically tracked asset into a moving sequence—is not evaluated as a monolithic black box. Instead, the continuous execution is meticulously mapped onto our formalized taxonomy of fundamental editing typologies. This approach isolates specific multimodal challenges, tracing the agent’s progression from initial Asset import and management to advanced Masking, matting, and tracking, and culminating in Export and delivery. By isolating these atomic components, our framework can pinpoint precisely where an agent’s spatial reasoning, temporal synchronization, or functional understanding fails during a sustained editing session. Furthermore, this task-level granularity ensures that the benchmark goes beyond assessing rote memorization of procedural interface clicks; it rigorously evaluates the agent’s capacity to autonomously orchestrate diverse, cross-modal operations. Ultimately, this methodology provides a high-resolution diagnostic instrument, exposing the strict boundaries of current multimodal foundation models when confronted with the compositional complexity inherent to professional software environments.

[Figure 140]

###### Figure 11 Task-centric decomposition in CutVerse. Continuous post-production pipelines are systematically deconstructed into standardized, discrete tasks. Mapping high-level creative intents to quantifiable actions enables the granular evaluation of an agent’s compositional execution capabilities.

Table 6 Summary of agent configurations in the CutVerse benchmark.

Agent Size Backend Prompt Image Size Coord. Output History Think

Claude Opus 4.6 [1] Closed API OSWorld 1280 × 720 Abs. (px) JSON 10 imgs – Qwen3-VL-32B-T[53] 32B vLLM OSWorld Smart-resize Rel. (0–999) XML Tool Call 4 turns ✓ UITars 1.5[36] – vLLM OSWorld 1920 × 1080 Norm. (0–1000) Python-style 5 imgs ✓ EvoCUA[52] 32B vLLM OSWorld Smart-resize Rel. (0–999) CoT+Code / XML 4 turns – Gemini 3 Flash[8] Closed API CutVerse ≤1280 (long side) Norm. (0–1000) JSON 5 imgs Optional

##### A.4 Agent Implementation Details

To ensure a rigorous and reproducible evaluation of GUI agents across the CutVerse benchmark, we integrate five state-of-the-art multimodal foundation models, each instantiated as an autonomous agent with a carefully designed system prompt that governs its perception–reasoning–action loop. All agents share a unified execution pipeline: at each step, the agent receives a screenshot of the current desktop state (resized to a model-specific resolution), along with the task instruction and, where applicable, a sliding window of multi-turn interaction history. The agent then produces a structured action prediction that is parsed and executed by our centralized action engine.

Prompt Design Rationale. For four of the five agents—Claude Opus 4.6 [1]1, Qwen3-VL [53]2, UITars 1.5 [36]3, and EvoCUA [52]4—we adopt the prompt paradigm established by the OSWorld benchmark [48]5, which has become the de facto standard for desktop GUI agent evaluation. The OSWorld-style prompt defines a computer_use tool schema that enumerates a canonical action vocabulary (click, type, scroll, key, drag, wait, terminate, etc.), specifies a structured output format (either XML tool-call blocks, JSON objects, or PyAutoGUI code), and provides descriptive guidelines for coordinate usage, screenshot consultation, and cursor alignment. We preserve these conventions to ensure a fair, apples-to-apples comparison: every agent is evaluated under the same prompt contract that governs perception, reasoning, and action emission, differing only in model-specific adaptations (e.g., coordinate resolution, image preprocessing, and output parser). In contrast, the Gemini 3 Flash [8]6 agent employs a distinct unified planner JSON schema developed specifically for the CutVerse autonomous pipeline, which introduces a normalised 0–1000 coordinate space, compound multi-action arrays, and milestone-completion metadata not present in the OSWorld template.

Below, we provide a comprehensive account of each agent’s prompt design, coordinate convention, and output schema, enabling full reproducibility of our benchmark results.

###### A.4.1 Claude Opus 4.6 (Anthropic)

We employ Claude Opus 4.6 through the native Anthropic Messages API (proprietary, closed-weight). Following the OSWorld prompt paradigm [48], we define the standard computer_use action vocabulary but adapt the output format to a JSON-only mode: the system prompt explicitly instructs the model to respond with a single, valid JSON object containing structured fields for observation, chain-of-thought reasoning, action type, coordinates, and a milestone-completion flag. This design eliminates the need for fragile free-text parsing and enforces a deterministic output schema that is directly consumable by the downstream action executor.

Coordinate Convention. Screenshots are resized to a fixed 1280 × 720 pixel canvas before being sent to the model. Coordinates in the JSON response are specified in this pixel space and subsequently rescaled to the actual screen resolution via linear interpolation: 𝑥screen = ⌊𝑥 · 𝑤screen/1280⌋, 𝑦screen = ⌊𝑦 · ℎscreen/720⌋.

Multi-turn History. Up to 10 most recent screenshots are retained in the conversation context; older turns are replaced with a textual placeholder ([previous screenshot]) to manage context length while preserving

- 1https://www.anthropic.com/claude/opus
- 2https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking
- 3https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- 4https://huggingface.co/meituan/EvoCUA-32B-20260105
- 5https://github.com/xlang-ai/OSWorld
- 6https://aistudio.google.com/models/gemini-3

the full reasoning trajectory. System Prompt. The complete system prompt is presented in Listing 1.

Listing 1 Claude Opus 4.6 system prompt.

|1 You are a computer automation agent. You will be shown a screenshot of a<br><br>2 computer screen together with a task description. Your job is to decide the<br><br>3 single best next action to take toward completing the task.<br><br>4<br><br>5 You MUST respond with ONLY a valid JSON object -- no markdown , no code fences ,<br><br>6 no surrounding text of any kind.<br><br>7<br><br>8 JSON format:<br><br>9 {<br><br>10 "Observation": "Brief description of the current screen state",<br><br>11 "Reasoning": "Step -by-step reasoning that leads to the chosen action",<br><br>12 "Action": "<action_name >",<br><br>13 "Coordinate": [x, y],<br><br>14 "Text": "text or key combination",<br><br>15 "ScrollDirection": "up|down|left|right",<br><br>16 "ScrollAmount": 3,<br><br>17 "StartCoordinate": [x, y],<br><br>18 "MilestoneCompleted ": false<br><br>19 }<br><br>20<br><br>21 Valid actions and their required fields:<br><br>22 left_click -- Coordinate=[x,y]<br><br>23 right_click -- Coordinate=[x,y]<br><br>24 double_click -- Coordinate=[x,y]<br><br>25 drag -- StartCoordinate =[x,y], Coordinate=[x,y] (end position)<br><br>26 type -- Text="string to type"<br><br>27 key -- Text="key or combo" (e.g. "enter", "ctrl+c")<br><br>28 scroll -- Coordinate=[x,y], ScrollDirection , ScrollAmount<br><br>29 wait -- (no extra fields needed)<br><br>30 done -- set MilestoneCompleted=true (ONLY when the full task is complete)<br><br>31 fail -- task cannot be completed<br><br>32<br><br>33 Coordinates are in a 1280 x720 pixel space.<br><br>34 Omit fields that are not needed for the chosen action.<br><br>35 Respond with ONLY the JSON object.<br><br><br>|
|---|

###### A.4.2 Qwen3-VL (Alibaba)

We deploy the Qwen3-VL-32B-Thinking variant (32B parameters, open-weight) via a self-hosted vLLM inference endpoint. Adhering closely to the OSWorld prompt template [48], the agent employs a tool-call XML output format: the system prompt defines the canonical computer_use function inside <tools> XML tags, including the standard environment description and action vocabulary, and the model is instructed to produce each action as a JSON object within <tool_call>...</tool_call> delimiters.

Coordinate Convention. The model supports both relative (0–999 normalised grid) and absolute (processed image pixel) coordinate modes. In relative mode, the system prompt advertises a 1000×1000 virtual resolution; coordinates are converted via 𝑥screen = ⌊𝑥 · 𝑤screen/999⌋. In absolute mode, the model outputs coordinates in the processed (smart-resized) image space, which are then rescaled to the screen resolution.

Image Processing. Screenshots are smart-resized using a factor-of-32 rounding scheme with a maximum pixel budget of 16 × 16 × 4 × 12800, preserving the aspect ratio while fitting within the model’s vision encoder constraints.

System Prompt. The full system prompt, including the embedded tool definition and response format specification, is presented in Listing 2.

Listing 2 Qwen3-VL system prompt (tool-call XML format).

- 1 # Tools

- 2

- 3 You may call one or more functions to assist with the user query.

- 4

- 5 You are provided with function signatures within <tools ></tools > XML tags:

- 6 <tools >

- 7 {"type": "function", "function": {"name": "computer_use",

- 8 "description": "Use a mouse and keyboard to interact with a computer , and

- 9 take screenshots. [...] The screen ’s resolution is 1000 x1000. [...]",

- 10 "parameters": {"properties": {

- 11 "action": {"enum": ["key","type","mouse_move","left_click",

- 12 "left_click_drag","right_click","middle_click","double_click",

- 13 "scroll","wait","terminate"], "type": "string"},

- 14 "keys": {"type": "array"}, "text": {"type": "string"},

- 15 "coordinate": {"type": "array"}, "pixels": {"type": "number"},

- 16 "time": {"type": "number"},

- 17 "status": {"type": "string", "enum": ["success","failure "]}

- 18 }, "required": ["action"], "type": "object "}}}

- 19 </tools >

- 20

- 21 For each function call , return a json object with function name and arguments

- 22 within <tool_call ></tool_call > XML tags:

- 23 <tool_call >

- 24 {"name": <function -name >, "arguments": <args -json -object >}

- 25 </tool_call >

- 26

- 27 # Response format

- 28 1) Action: a short imperative describing what to do in the UI.

- 29 2) A single <tool_call >...</tool_call > block.

- 30

- 31 Rules:

- 32 - Output exactly in the order: Action , <tool_call >.

- 33 - Be brief: one sentence for Action.

- 34 - If finishing , use action=terminate in the tool call.

###### A.4.3 UITars 1.5 (ByteDance Seed)

UITars 1.5 is a GUI-native vision-language model served via a self-hosted vLLM endpoint. Its prompt design inherits the OSWorld action vocabulary [48] (click, drag, scroll, type, hotkey, wait, finished) but re-expresses it through a Python-style function-call syntax that is native to the UITars model family. The agent supports two inference modes: a thinking mode in which the model produces an explicit chain-of-thought enclosed in <think>...</think> tags before the action, and a non-thinking mode that outputs the Thought–Action pair directly.

Coordinate Convention. UITars outputs coordinates in a 0–1000 normalised space via <point>x y</point> tokens. These are converted to relative 0–1 values by the parser and subsequently multiplied by the screen resolution to yield final pixel coordinates.

Action Space. The action vocabulary is specified through a Python-style function-call syntax (e.g., click(point=’<point>x1 y1</po int>’)), which is natively understood by the UITars model family. This contrasts with the JSON- or XML-based schemas used by other agents.

System Prompt (Thinking Mode). Listing 3 presents the system prompt used when thinking mode is enabled (COMPUTER_USE_DOUBAO).

Listing 3 UITars 1.5 system prompt (thinking mode).

- 1 You are a GUI agent. You are given a task and your action history , with

- 2 screenshots. You need to perform the next action to complete the task.

- 3

- 4 ## Output Format

- 5 You should first think about the reasoning process in the mind and then

- 6 provide the user with the answer.

- 7 The reasoning process is enclosed within <think > </think > tags.

- 8 After the <think > tags , you should place the final answer , which concludes

- 9 your summarized thought and your action.

- 10

- 11 For example ,

- 12 <think >detailed reasoning content here </think >

- 13 Thought: a small plan and finally summarize your next action in one sentence

- 14 Action: ...

- 15

- 16 ## Action Space

- 17 click(point=’<point >x1 y1 </point >’)

- 18 left_double(point=’<point >x1 y1 </point >’)

- 19 right_single(point=’<point >x1 y1 </point >’)

- 20 drag(start_point=’<point >x1 y1 </point >’, end_point=’<point >x2 y2 </point >’)

- 21 hotkey(key=’ctrl c’)

- 22 type(content=’xxx ’)

- 23 scroll(point=’<point >x1 y1 </point >’, direction=’down|up|right|left ’)

- 24 wait()

- 25 finished(content=’xxx ’)

- 26

- 27 ## Note

- 28 - Write a small plan and summarize your next action in one sentence in Thought.

- 29 - If repeated actions have no effect , try a modified action.

- 30

- 31 ## User Instruction

- 32 {instruction}

###### A.4.4 EvoCUA

EvoCUA (32B parameters, open-weight) is a native multimodal GUI agent model deployed via a self-hosted vLLM inference endpoint. Its architectural design is grounded in the OSWorld evaluation protocol [48] and formulated around a Tool-Call paradigm that directly instantiates the XML-based computer_use tool-calling schema. Crucially, EvoCUA extends the canonical action vocabulary with stateful key_down and key_up primitives, thereby enabling fine-grained modifier-held interactions—such as Shift-constrained dragging or Alt-held scrubbing—that are indispensable for professional multimedia post-production workflows requiring precise cross-modal alignment between visual feedback and keyboard state.

Prompt Architecture. The prompt structure is aligned with the tool-call XML paradigm: a computer_-

use function is formally defined within <tools> tags, and the model orchestrates each interaction step by emitting an Action: line articulating the high-level intent, followed by a structured <tool_call> block that encapsulates the executable operation. Specifically, the action vocabulary encompasses standard GUI primitives (left_click, right_click, double_click, type, key, scroll, mouse_move, left_click_-

drag) alongside the aforementioned stateful keyboard actions (key_down, key_up) and a triple_click convenience action. This enriched action space is specifically designed to alleviate the expressiveness gap encountered when agents must manipulate timeline-centric, audio-visual editing interfaces where temporal synchronization between held modifier keys and spatial cursor trajectories is critical.

Coordinate Convention. EvoCUA defaults to a 1000 × 1000 relative coordinate grid (configurable to absolute mode), which normalizes heterogeneous screen resolutions into a unified spatial representation. Screenshots are smart-resized with a factor-of-32 scheme to maintain compatibility with the vision encoder’s patch-alignment requirements, ensuring robust cross-modal grounding between the visual observation and the agent’s spatial reasoning.

System Prompt. The system prompt follows the same tool-call XML structure as Qwen3-VL (Listing 2), with the addition of key_down, key_up, and triple_click to the action enum. The full system prompt is presented

- in Listing 4. Listing 4 EvoCUA system prompt (Tool-Call mode, abbreviated).

|1 # Tools<br><br>2 You are provided with function signatures within <tools ></tools > XML tags:<br><br>3 <tools > {computer_use tool definition} </tools ><br><br>4<br><br>5 For each function call , return a JSON object within <tool_call ></tool_call >:<br><br>6 <tool_call ><br><br>7 {"name": "computer_use", "arguments": {...}}<br><br>8 </tool_call ><br><br>9<br><br>10 # Response Format<br><br>11 1) Action: a short imperative describing what to do in the UI.<br><br>12 2) A single <tool_call >...</tool_call > block.<br><br>13 Rules: Output exactly in the order Action , <tool_call >. Be brief.<br><br>14<br><br>15 # Action Enum<br><br>16 key , key_down , key_up , type , mouse_move , left_click , left_click_drag ,<br><br>17 right_click , middle_click , double_click , triple_click , scroll , wait ,<br><br>18 terminate<br><br><br>|
|---|

###### A.4.5 Gemini 3 Flash (Google)

In contrast to the four agents described above, Gemini 3 Flash (proprietary, closed-weight; accessed via the Google API) does not follow the OSWorld prompt template. Instead, it employs the unified planner JSON schema developed specifically for the CutVerse autonomous pipeline. This schema outputs a structured JSON object that includes observation, reasoning, a milestone-completion flag, and an actions array supporting compound multi-action steps—capabilities that go beyond the single-action-per-turn paradigm of the OSWorld template.

Coordinate Convention. The model operates in a 0–1000 normalised coordinate space where (0, 0) denotes the top-left corner and (1000, 1000) the bottom-right. Screenshots are resized so that the longest side does not exceed 1280 pixels (preserving aspect ratio). The normalised coordinates are converted to screen pixels by the framework’s convert_action_coords_to_screen() utility.

Multi-turn History. The agent maintains a sliding window of up to 5 screenshots in the conversation context. For turns within the image window, both the screenshot and a brief task reminder are included; for older turns, only a textual placeholder is retained.

System Prompt. The system prompt defines the full action vocabulary (mouse clicks, press/release, drag, scroll, keyboard, and system actions) and specifies the JSON output schema. The complete prompt is presented

- in Listing 5. Listing 5 Gemini 3 Flash system prompt.

|1 You are an autonomous GUI automation agent for computer -use tasks on a<br><br>2 {os_name} device. You are operating in AUTONOMOUS MODE - you must plan and<br><br>3 act independently.<br><br>4<br><br>5 You must perform TWO tasks in ONE response:<br><br>6 1. Plan: Analyze the screenshot and decide what action to take next<br><br>7 2. Act: Generate the precise action(s) with exact coordinates/parameters<br><br>8<br><br>9 ## Coordinate System<br><br>10 Output coordinates in normalized 0-1000 range:<br><br>11 - (0, 0) = top -left corner<br><br>12 - (1000, 1000) = bottom -right corner<br><br>13<br><br>14 ## Action Types<br><br>15 ### Mouse - Click: click , right_click , middle_click , double_click<br><br>16 ### Mouse - Press/Release: mouse_down , mouse_up<br><br>17 ### Mouse - Move/Drag/Scroll: move , drag , scroll<br><br>18 ### Keyboard: type , key , hotkey , key_down , key_up<br><br>19 ### System: wait , stop<br><br>20<br><br>21 ## Output Format (JSON)<br><br>22 {<br><br>23 "Observation": str ,<br><br>24 "Reasoning": str ,<br><br>25 "Action": str | null ,<br><br>26 "MilestoneCompleted ": bool ,<br><br>27 "actions": [<br><br>28 {<br><br>29 "action_type": str ,<br><br>30 "x": int | null , // 0-1000 normalized<br><br>31 "y": int | null , // 0-1000 normalized<br><br>32 "text": str | null ,<br><br>33 "key": str | null ,<br><br>34 "keys": [str] | null ,<br><br>35 "scroll_x": int | null ,<br><br>36 "scroll_y": int | null ,<br><br>37 "end_x": int | null ,<br><br>38 "end_y": int | null<br><br>39 }<br><br>40 ]<br><br>41 }<br><br>42<br><br>43 ## CRITICAL RULES<br><br>44 1. MilestoneCompleted=true AND actions =[{" action_type ":"stop "}]: Goal achieved<br><br>45 2. MilestoneCompleted=false AND actions=[action]: Action needed<br><br>46 3. Consider history to avoid repeating failed attempts<br><br>47 4. Always output actions as an array<br><br><br>|
|---|

###### A.4.6 Summary of Agent Configurations

- Table 6 summarises the key configuration parameters across all five agents. Table 7 details the exact vLLM [22]7 deployment parameters utilized for the evaluated multimodal models. To ensure strict reproducibility and standardized inference across our benchmark, we explicitly outline the hardware allocation (Tensor Parallelism) and structural constraints (maximum context length and multimodal prompt limits) applied during testing. By universally deploying models through this transparent and controlled infrastructure, CutVerse guarantees that all baseline evaluations are fair, verifiable, and readily reproducible by the research community.
- Table 7 Detailed vLLM deployment configurations, software environment, and empirical memory footprint.

Global Infrastructure & Software Environment

Hardware Node 4 × NVIDIA RTX 5090 Software Stack CUDA 12.8.61, PyTorch 2.8.0+cu128, vLLM 0.11.0, Transformers 4.57.1 Multimodal Limits Max 5 Images, 0 Videos (–limit-mm-per-prompt)

Model-Specific Execution Parameters Served Model GPUs TP Size Max Context GPU Util. Peak VRAM

UI-TARS-1.5-7B [36] 2 2 98,304 0.92 63,170 MB Qwen3-VL-32B-T [53] 4 4 28,488 Default 126,688 MB EvoCUA-32B [52] 4 4 34,768 Default 127,508 MB

- Table 8 Overall failure statistics and completion-to-execution consistency gaps. Task Consistency Gap is Task Completion Rate minus Task Execution Accuracy. Milestone Consistency Gap is Milestone Completion Rate minus Milestone Execution Accuracy.

Incomplete Tasks

Incomplete Task Ratio

Incomplete Milestones

Incomplete Milestone Ratio

Task Consistency Gap

Milestone Consistency Gap

Model

Claude Opus 4.6 [1] 59/186 0.317 159/631 0.252 0.116 0.016 Gemini 3 Flash [8] 61/186 0.328 187/631 0.296 0.091 0.015 EvoCUA-32B [52] 90/186 0.484 283/631 0.448 0.116 0.012 Qwen3-VL-32B-T [53] 96/186 0.516 295/631 0.468 0.148 0.015 UITars-1.5-7B [36] 104/186 0.559 314/631 0.498 0.123 0.014

#### B Additional Experimental Results

##### B.1 Additional Data Statistics

Task Distribution Heterogeneity. Part A of Table 10 details the distribution of dominant tasks, exposing the profound heterogeneity of media workflows. While Effects and visual tuning predominantly leads in visually intensive tools such as After Effects (61.5%) and DaVinci Resolve (50.0%), Export and delivery serves as a universally mandatory concluding milestone across nearly all platforms. Furthermore, the distribution captures specific functional biases, highlighting ComfyUI’s reliance on Generative workflows (50.0%) and Premiere Pro’s emphasis on Audio editing (25.9%). Crucially, varying degrees of task concentration directly dictate execution complexity. Professional environments such as After Effects and ComfyUI exhibit highly concentrated distributions (Top-1 task ≥ 50%), demanding deep, long-horizon specialized operations within a single domain. Conversely, comprehensive editing platforms such as Premiere Pro and Photoshop display a flattened distribution

7https://github.com/vllm-project/vllm

Table 9 Task-type action distribution profile.

Task Type Top-1 Top-2 Top-3 Masking, Matting, and Tracking Left click (48.0%) Drag (24.8%) Key press (17.3%) Effects and Visual Tuning Left click (59.4%) Drag (19.7%) Key press (6.7%) Audio and Rhythm Editing Left click (44.3%) Key press (30.6%) Drag (11.4%) Timeline Editing and Arrangement Key press (36.2%) Left click (35.9%) Drag (18.5%) Asset Import and Management Left click (44.0%) Key press (30.8%) Drag (18.2%) Preview, Check, and Validation Left click (71.4%) Drag (14.3%) Wait (9.1%) Export and Delivery Left click (53.9%) Key press (21.7%) Drag (8.1%) Launch and Setup Left click (69.1%) Key press (8.8%) Wait (7.4%) Generative Workflow Left click (55.0%) Drag (17.5%) Key press (11.2%)

- Table 10 Detailed breakdown of dominant task types and interaction modalities across different software environments.

Software Top-1 Top-2 Top-3

Part A: Dominant Task Types After Effects Effects and visual tuning

Export and delivery (19.2%) Masking and tracking (7.7%)

(61.5%)

ComfyUI Generative workflow (50.0%) Export and delivery (33.3%) Asset import and management

(16.7%) DaVinci Resolve Effects and visual tuning

Export and delivery (20.0%) Timeline editing (15.0%)

(50.0%)

JianYing Effects and visual tuning (18.8%)

Export and delivery (18.8%) Asset management (15.9%)

Jimeng Effects and visual tuning (20.0%)

Audio editing (16.0%) Export and delivery (16.0%)

Keling Asset management (26.9%) Preview and validation (15.4%) Export and delivery (11.5%) Premiere Pro Audio editing (25.9%) Effects tuning (20.4%) Asset management (13.0%) Photoshop Effects and visual tuning

Export and delivery (18.2%) Asset management (12.1%)

(24.2%)

###### Part B: Dominant Interaction Modalities

After Effects Left click (62.1%) Drag (24.4%) Scroll (4.7%) ComfyUI Left click (53.2%) Drag (21.0%) Scroll (6.5%) DaVinci Resolve Left click (48.5%) Drag (24.4%) Key press (7.8%) JianYing Left click (58.9%) Drag (17.1%) Key press (11.3%) Jimeng Left click (58.6%) Key press (22.4%) Drag (8.9%) Keling Left click (71.3%) Drag (18.1%) Type text (3.2%) Premiere Pro Left click (43.7%) Key press (31.8%) Drag (12.1%) Photoshop Left click (46.0%) Key press (31.2%) Drag (11.3%)

(Top-1 hovering around 20-25%), necessitating rapid context-switching between diverse modalities, thereby imposing stricter demands on sustained cross-modal reasoning.

Interaction Modality Complexity. Part B of Table 10 delineates the execution modalities, exposing the mechanical bottlenecks of professional environments. While Left click predictably serves as the foundational operation across all platforms, peaking at 71.3% in web-centric tools such as Keling, the true complexity emerges in secondary interactions. Professional software, particularly Premiere Pro and Photoshop, exhibits a substantial reliance on Key press operations (31.8% and 31.2%, respectively). This underscores the absolute necessity of keyboard shortcuts for rapid tool switching and global viewport navigation. Furthermore, continuous Drag operations manifest as a universal requirement for precise spatial and temporal adjustments, notably in After Effects (24.4%) and DaVinci Resolve (24.4%). These distributions firmly establish that basic point-and-click capabilities are fundamentally inadequate; mastering tightly coupled, compositional key-mouse coordination and sustained dragging is strictly mandatory for autonomous agents in creative workflows.

Task-Driven Action Mapping. Table 9 delineates the intrinsic correlation between post-production task typologies and their underlying execution actions, revealing that an agent’s required interaction modality is strictly dictated by the specific task type. While foundational Left click actions predictably dominate procedurally driven tasks such as Preview, check, and validation (71.4%) and Launch and setup (69.1%), structurally complex tasks demand entirely distinct action mappings. Specifically, Timeline editing and arrangement exhibits a fundamental shift towards Key press operations (36.2%), highlighting a direct relationship between structural video assembly and the absolute necessity of keyboard shortcuts for rapid tool switching. Furthermore, tasks requiring precise spatial-temporal grounding, such as Masking, matting, and tracking and Effects and visual tuning, establish a strong dependency on continuous Drag actions (24.8% and 19.7%, respectively) for manipulating bezier paths and parameter sliders. Conversely, organizational tasks such as Asset import and management tightly couple with Key press inputs (30.8%) to facilitate semantic search and file routing. Ultimately, this distribution confirms that the execution action space is fundamentally task-dependent; succeeding in diverse multimedia workflows requires agents to dynamically map specific task semantics to tightly coupled, compositional key-mouse action combinations.

##### B.2 Additional Evaluation Analysis

Failure Analytics and Execution Consistency. Table 8 details the overall failure statistics and consistency metrics across the evaluated multimodal agents, exposing a profound gap between current capabilities and professional production standards. Even industry-leading models struggle significantly; Claude and Gemini exhibit incomplete task ratios of 31.7% and 32.8%, respectively. Meanwhile, open-weight and UI-centric models, specifically Qwen and UI-TARS, fail to complete over half of the assigned tasks (51.6% and 55.9%).

Beyond absolute failure rates, the data reveals a critical vulnerability in standard agent evaluation paradigms through the "Consistency Gap"—the mathematical difference between the perceived completion rate and the strictly verified execution accuracy. At the macro-task level, agents demonstrate a severe task consistency gap, peaking at 0.148 for Qwen and 0.116 for Claude. This significant discrepancy indicates that agents frequently suffer from execution hallucinations; they declare task completion despite failing to achieve the precise spatialtemporal requirements of the creative workflow. Conversely, the milestone consistency gap remains exceptionally tight across all models, strictly bounded between 0.012 and 0.016. This striking contrast definitively proves the necessity of our granular evaluation infrastructure. By decomposing continuous workflows into visually grounded, intermediate milestones, CutVerse effectively eliminates false positives, ensuring that an agent’s perceived progress perfectly aligns with its authentic execution accuracy in complex multimodal environments.

- Table 11 The formalized nine-category taxonomy of media post-production tasks in CutVerse. Each category is grounded in authentic task instructions and evaluated via granular multimodal QA pairs to rigorously verify spatial-temporal state transitions.

Task Category Authentic Task Instruction Multimodal QA Evaluation Example Effects and visual tuning Applying and Adjusting Glow Effect in

Q: In the right panel’s ’Glow’ effect settings, is the ’Input Alpha’ slider value changed from 0.543 to 0.612, as observed in the slider display?

DaVinci Resolve

A: True Export and delivery Exporting and Recording Workflow in

Q: Is the blue Export button clicked at the bottom right of the export settings screen with a progress dialog appearing in the center showing export progress for ’hello world’?

Adobe Premiere Pro

A: True

Import and Manage Video Clip in Premiere Pro

Asset import and management

Q: Is ’AntiguaArchTL’ video thumbnail bordered in blue and marked with a checkmark in the central footage grid panel after selection?

A: True

Audio and rhythm editing Audio Editing in Adobe Premiere Pro Q: Does the right edge of the first audio clip extend visually to fill the gap within the bottom timeline panel after the second clip’s audio is deleted?

A: True

Trimming Video Clips in Adobe Premiere Pro

Timeline editing and arrangement

Q: Is the second video clip shortened on the timeline after clicking and dragging its right edge, with a tooltip displaying the new duration above the timeline?

A: True

Video Editing and Documentation Process Q: Is the playhead advancing from the position 00:00:01:00 in the bottom timeline panel, with the audio meters below the preview window showing activity?

Preview, check, and validation

A: True

Mask Animation Creation in Adobe After Effects

Masking, matting, and tracking

Q: Is the colorful image fully revealed in the center composition panel, with the mask path expanded to cover the entire area?

A: True Launch and setup Transitioning Between Applications and

Q: Has the Adobe Premiere Pro interface switched to the main editing workspace after selecting the ’task1’ project, displaying the media browser, preview monitor, and timeline?

Opening Projects

A: True Generative workflow Semantic Guidance Input for Video Pro-

Q: Is the ’positive_prompt’ input field in the ’WanVideo T5 text encoder’ node filled with Chinese text?

cessing Workflow

A: True

#### References

- [1] Anthropic. Claude 4.6 model card. Technical report, 2026.
- [2] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Keunho Jang, and Zheng Hui. Windows agent arena: Evaluating multi-modal OS agents at scale. In Forty-second International Conference on Machine Learning, 2025.

- [3] Liangyu Chen, Hanzhang Zhou, Chenglin Cai, Jianan Zhang, Panrong Tong, Quyu Kong, Xu Zhang, Chen Liu, Yuqi Liu, Wenxuan Wang, Yue Wang, Qin Jin, and Steven Hoi. Ui-ins: Enhancing gui grounding with multi-perspective instruction-as-reasoning, 2025.
- [4] Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin Hu, and Shuicheng Yan. Ivebench: Modern benchmark suite for instruction-guided video editing assessment. In The Fourteenth International Conference on Learning Representations, 2026.

- [5] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents, 2024. URL https://arxiv.org/abs/2401.10935.
- [6] Xiang Deng, Yu Gu, Boyu Zheng, Shuyan Zhou, Samuel Stevens, et al. Mind2web: Towards a generalist agent for the web. 2023.
- [7] Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, Hengxu Wang, Luowei Zhou, and Mike Zheng Shou. Assistgui: Task-oriented desktop graphical user interface automation, 2024.
- [8] Gemini Team, Google DeepMind. Gemini 3 technical report. Technical report, 2026.
- [9] Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao, Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. Ui-venus technical report: Building high-performance ui agents with rft, 2025.
- [10] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. PAMI, 2025.

- [11] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents, 2024.
- [12] Wenyi Hong, Weihan Wang, Qinkai Zheng, Jiawei Liu, and Jianguo Zhu. Cogagent: A visual language model for gui agents. In CVPR, 2024.

- [13] Siyuan Hu, Mingyu Ouyang, Difei Gao, and Mike Zheng Shou. The dawn of gui agent: A preliminary case study with claude 3.5 computer use, 2024.
- [14] Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, Yuhuai Li, Shengze Xu, Shenzhi Wang, Xinchen Xu, Shuofei Qiao, Zhaokai Wang, Kun Kuang, Tieyong Zeng, Liang Wang, Jiwei Li, Yuchen Eleanor Jiang, Wangchunshu Zhou, Guoyin Wang, Keting Yin, Zhou Zhao, Hongxia Yang, Fan Wu, Shengyu Zhang, and Fei Wu. Os agents: A survey on mllm-based agents for general computing devices use, 2025.
- [15] Kaiyi Huang, Yukun Huang, Xintao Wang, Zinan Lin, Xuefei Ning, Pengfei Wan, Di Zhang, Yu Wang, and Xihui Liu. Filmaster: Bridging cinematic principles and generative ai for automated film generation, 2025. URL https: //arxiv.org/abs/2506.18899.
- [16] Y. Huang et al. Comfybench: Benchmarking llm-based agents in comfyui for autonomously designing collaborative ai systems. arXiv preprint arXiv:2409.01392, 2024.

- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- [18] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. doi: 10.1109/TPAMI.2025.3633890.

- [19] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem AlShikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In European Conference on Computer Vision, pages 161–178. Springer, 2024.

- [20] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. In LunWei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 881–905, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.50. URL https://aclanthology.org/2024.ac l-long.50/.

- [21] Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, Zhidan Liu, Steven Hoi, and Yue Wang. Mobileworld: Benchmarking autonomous mobile agents in agent-user interactive, and mcp-augmented environments, 2025.
- [22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023. URL https://arxiv.org/abs/2309.06180.
- [23] Kaixin Li, Meng Ziyang, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: GUI grounding for professional high-resolution computer use. In Workshop on Reasoning and Planning for Large Language Models, 2025.

- [24] Mingcheng Li, Xiaolu Hou, Ziyang Liu, Dingkang Yang, Ziyun Qian, Jiawei Chen, Jinjie Wei, Yue Jiang, Qingyao Xu, and Lihua Zhang. MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation . In CVPR, 2025.

- [25] Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. Animdirector: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

- [26] G. Liang et al. Editval: Benchmarking diffusion based text-guided image editing methods. In ICCV, 2023.

- [27] Kevin Qinghong Lin, Linjie Li, Difei Gao, Qinchen WU, Mingyi Yan, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. VideoGUI: A benchmark for GUI automation from instructional videos. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

- [28] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Stan Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19498–19508, 2025.

- [29] Hongbo Liu, Jingwen He, Yi Jinn, Dian Zheng, Yuhao Dong, Fan Zhang, Ziqi Huang, Yinan He, Weichao Chen, Yu Qiao, Wanli Ouyang, Shengjie Zhao, and Ziwei Liu. Shotbench: Expert-level cinematic understanding in vision-language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- [30] Zhaoyang Liu, JingJing Xie, Zichen Ding, Zehao Li, Bowen Yang, Zhenyu Wu, Xuehui Wang, Qiushi Sun, Shi Liu, Weiyun Wang, Shenglong Ye, Qingyun Li, Zeyue Tian, Gen Luo, Xiangyu Yue, Biqing Qi, Kai Chen, Bowen Zhou, Yu Qiao, Qifeng Chen, and Wenhai Wang. ScaleCUA: Scaling open-source computer use agents with cross-platform data. In The Fourteenth International Conference on Learning Representations, 2026.

- [31] Quanfeng Lu, Wenqi Shao, Zitao Liu, Lingxiao Du, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, and Ping Luo. Guiodyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22404–22414, 2025.

- [32] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024.
- [33] Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Nicolas Chapados, M. Tamer Özsu, Aishwarya Agrawal, David Vazquez, Christopher Pal, Perouz Taslakian, Spandana Gella, and Sai Rajeswar. UI-vision: A desktop-centric GUI benchmark for visual perception and interaction. In Forty-second International Conference on Machine Learning, 2025.

- [34] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Jihyung Kil, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernoncourt. GUI agents: A survey. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 22522–22538, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1158.

- [35] OpenAI. Gpt-5 series models. https://platform.openai.com, 2025. Accessed: 2026.
- [36] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

- [37] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Android in the wild: A large-scale dataset for android device control, 2023.
- [38] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William E Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Kenji Toyama, Robert James Berry, Divya Tyamagundlu, Timothy P Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations, 2025.

- [39] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, et al. Toolformer: Language models can teach themselves to use tools. In ICLR, 2023.

- [40] ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025.
- [41] Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, and Kristina Toutanova. From pixels to ui actions: learning to follow instructions via graphical user interfaces. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

- [42] Haoyuan Shi, Yunxin Li, Xinyu Chen, Longyue Wang, Baotian Hu, and Min Zhang. Animaker: Multi-agent animated storytelling with mcts-driven clip generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.

- [43] Bryan Wang, Yuliang Li, Zhaoyang Lv, Haijun Xia, Yan Xu, and Raj Sodhi. Lave: Llm-powered agent assistance and language augmentation for video editing. Proceedings of the 29th International Conference on Intelligent User Interfaces, 2024.

- [44] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Zheng Boyuan, LI PEIHANG, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Hu Jiarui, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Y.Charles, Zhilin Yang, and Tao Yu. OpenCUA: Open foundations for computer-use agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- [45] Zhenyu Wang, Aoxue Li, Zhenguo Li, and Xihui Liu. Genartist: Multimodal llm as an agent for unified image generation and editing. volume 37, pages 128374–128395, 2024.
- [46] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

- [47] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents, 2024.
- [48] Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Weng Lam Tam, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. 2024.
- [49] Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao Yu. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials. In The Thirteenth International Conference on Learning Representations, 2025.

- [50] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous GUI interaction. In Forty-second International Conference on Machine Learning, 2025.

- [51] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction, 2025. URL https://arxiv.org/abs/2412.0 4454.
- [52] Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, Jinrui Ding, Xiandi Ma, Yuchen Xie, Peng Pei, Xunliang Cai, and Xipeng Qiu. Evocua: Evolving computer use agents via learning from scalable synthetic experience, 2026.
- [53] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [54] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v, 2023.
- [55] Pei Yang, Hai Ci, and Mike Zheng Shou. macOSWorld: A multilingual interactive benchmark for GUI agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- [56] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICCV, 2022.

- [57] Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, Song-Chun Zhu, and Qing Li. Tongui: Internet-scale trajectories from multimodal web tutorials for generalized gui agents, 2025.
- [58] Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. Stage: Storyboard-anchored generation for cinematic multi-shot narrative, 2026.
- [59] Henry Hengyuan Zhao, Kaiming Yang, Wendi Yu, Difei Gao, and Mike Zheng Shou. Worldgui: An interactive benchmark for desktop gui automation from any starting point, 2026.
- [60] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: Step-by-step generating multi-shot video with minimal manual intervention. arXiv preprint arXiv:2412.02259, 2024.

- [61] Mingzhe Zheng, Dingjie Song, et al. Cml-bench: A framework for evaluating and enhancing llm-powered movie scripts generation. arXiv preprint arXiv:2510.06231, 2025.

- [62] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024.
- [63] Cailin Zhuang, Ailin Huang, Yaoqi Hu, Jingwei Wu, Wei Cheng, Jiaqi Liao, Hongyuan Wang, Xinyao Liao, Weiwei Cai, Hengyuan Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. arXiv preprint arXiv:2505.24862, 2025.

