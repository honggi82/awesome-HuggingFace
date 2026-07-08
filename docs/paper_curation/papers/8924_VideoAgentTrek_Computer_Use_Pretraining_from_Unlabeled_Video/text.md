# arXiv:2510.19488v1[cs.CL]22Oct2025

[Figure 1]

2025-10-23

## VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos

Dunjie Lu1,2*, Yiheng Xu1,2*, Junli Wang2*, Haoyuan Wu1, Xinyuan Wang1, Zekun Wang2, Junlin Yang1, Hongjin Su1, Jixuan Chen1, Junda Chen1, Yuchen Mao1, Jingren Zhou2, Junyang Lin2, Binyuan Hui2†, Tao Yu1† 1The University of Hong Kong 2Qwen Team, Alibaba Group

Project Page: https://videoagenttrek.github.io/

### Abstract

Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VIDEOAGENTTREK, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos at web scale, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop VIDEO2ACTION, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries and context, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text with high fidelity. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps automatically. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation.

### 1 Introduction

Teaching machines to use computers like humans do (clicking buttons, typing text, navigating interfaces) represents a fundamental challenge in AI. While recent advances in vision-language models have made computer-use agents increasingly feasible (Bai et al., 2025; Qin et al., 2025; Team et al., 2025; Wang et al., 2025b), their development remains bottlenecked by data availability. Training these agents requires extensive trajectories that precisely document GUI interactions: screenshots paired with exact action parameters like click coordinates (x, y) and typed strings. However, creating such datasets through manual annotation is extraordinarily expensive, making it impractical to achieve the scale necessary for robust generalization across diverse applications and operating systems.

Meanwhile, the internet hosts millions of screen-recorded tutorials where humans demonstrate computer use, from Excel tutorials to software walkthroughs. These videos implicitly contain the supervision we need: they show where users click, what they type, and how interfaces respond. Yet this resource remains untapped because videos lack the structured action labels required for training. The cursor movements are visible but not tracked; the typed text appears but isn’t extracted; the timing of actions is implicit but not annotated. We can learn to automatically extract structured action trajectories from raw videos by training specialized models to detect when actions occur and infer what their parameters are, effectively converting passive recordings into active training data.

We introduce VIDEOAGENTTREK, a scalable pipeline that mines computer-use trajectories from publicly available unlabeled videos without manual annotation. Our approach employs VIDEO2ACTION, an inverse dynamics module (IDM) with two stages: First, an action event detection model performs dense event detection, identifying action types and their precise temporal boundaries (e.g., click at [1.5,2.0]s, type at [3.5,5.5]s). Second, the action parameterization model, an action-content recognizer, analyzes these localized segments to extract structured parameters (pointer coordinates for clicks, literal text for typing), yielding complete (screenshot, action, parameters) trajectories suitable for training.

VIDEOAGENTTREK enables large-scale computer-use pretraining with unlabeled web videos. From 39,000 YouTube videos, we automatically extract 1.52 million interaction steps. This represents not just more data, but more diverse data: the trajectories span hundreds of applications across Windows, macOS, and web platforms, capturing interaction patterns that would be infeasible to annotate manually.

We validate VIDEOAGENTTREK with a two-stage training recipe: continued pretraining on the mined trajectories followed by supervised fine-tuning on a curated dataset. This combination leverages the broad coverage from videos to learn fundamental GUI interaction patterns, while supervised fine-tuning sharpens task-specific performance.

*Equal contribution. † Corresponding authors.

[Figure 2]

[Figure 3]

[Figure 4]

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

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 1: Overview of VIDEOAGENTTREK. (1) Video Collection: crawl screen-recorded tutorials and filter GUI footage with SCREENFILTER. (2) Video2Action: an inverse dynamics module that first performs dense action-event detection to localize clips and assign action types, then action parameterization (e.g., click coordinates, typed text) to yield structured (screenshot,action,parameters) trajectories. (3) Agent Training: use the mined trajectories for continued pretraining and supervised finetuning of computer-use agents.

Our models achieve 15.8% task success on OSWorld-Verified compared to 9.3% for baselines, a 70% relative improvement. The gains are particularly pronounced in online environments where robustness to visual variation matters most. We summarize our main contributions and findings below:

- • We propose VIDEOAGENTTREK, an unsupervised approach to training computer-use agents that automatically converts screen-recorded videos into structured training data through learned inverse dynamics, thereby eliminating the need for manual annotation.
- • Our VIDEO2ACTION module implements inverse dynamics, combining action event detection with millisecondprecision temporal localization and action parameter extraction. It enables accurate reconstruction of GUI interactions (clicks, typing,...) from raw video without ground-truth labels.
- • Experiments demonstrate that our approach achieves 15.8% task success on OSWorld-Verified compared to 9.3% for SFT-only baselines (70% relative improvement), and improves step accuracy on AgentNetBench from 64.1% to 69.3%, validating that passive internet videos can provide effective supervision at scale.
- • We provide a reproducible pipeline and training methodology that enables researchers to leverage publicly available screen recordings for computer-use agent training. To facilitate future research, we release SCREENFILTER for efficient GUI filtering and VIDEO2ACTION for action extraction as open-source tools.

### 2 VideoAgentTrek

We introduce VIDEOAGENTTREK, a video-driven pipeline that turns web tutorials into training supervision for computer-use agents. Each trajectory is a sequence R = {(Ik,rk, ak, πk)}kK=1 following Yao et al. (2023), where Ik is a representative screenshot, rk is a brief inner monologue, ak ∈ A is the action type (e.g. click, type), and πk is the action content (e.g., pointer (x, y) or typed text). The pipeline has three parts:

- • Video collection and preprocessing. We crawl tutorial videos with seeded queries and tag expansion, apply human-in-the-loop screening, and use cursor-based filtering to retain screen segments with GUI interactions (Section 2.1).
- • VIDEO2ACTION. From raw video, we recover stepwise supervision without manual labels: (i) dense event detection produces typed segments with tight start/end times; (ii) action identification infers parameters πk (e.g.,

click coordinates, typed strings); and (iii) a short inner monologue rk makes the intent explicit. Assembling these per-clip steps yields ReAct tuples for training (Section 2.2).

- • Agent training. We combine large-scale agentic data produced by the method with human demonstrations and targeted GUI grounding pairs, and train an end-to-end agent in two stages: interleaved video–text pretraining followed by instruction-style finetuning (Section 2.3).

This structure scales supervision to web-scale while preserving the stepwise semantics needed for robust computeruse policies.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

- Figure 2: Video candidate auto-discovery. From seed keywords and tags, we search and evaluate videos, expand to related videos and high-quality channels (≥80% pass), and iteratively collect GUI-containing videos for VAT.

- 2.1 Video collection and preprocessing

- 2.1.1 Video Candidate Auto-Discovery

We employ a scalable pipeline for video collection that leverages channel coherence—the observation that YouTube channels typically maintain consistent content types and quality. Starting from seed keywords such as “Excel tutorial” and “How to use Windows”, we validate initial results and extend to entire channels when sampling indicates high quality (i.e., when ≥ 80% of samples meet our criteria). This channel-based expansion enables efficient scaling: validated channels become trusted sources, while their tags and metadata enable iterative discovery.

When we identify high-quality channels through seed validation, we include all their videos as candidates rather than individually vetting each one. This approach deliberately optimizes recall over precision, as the subsequent SCREENFILTER stage ensures final data quality. The channel coherence property—where content creators typically focus on consistent topics—makes this expansion particularly effective.

Through iterative rounds of keyword search, channel expansion, and tag extraction, we transform a small set of manually validated seeds into 55,000 candidate videos (∼10,000 hours). This process requires minimal human oversight: initial quality validation on seed videos and periodic verification of expansion effectiveness. The resulting candidate pool intentionally includes some non-GUI content (presentations, tutorials with mixed content), which our filtering stage handles efficiently.

- 2.1.2 Video Preprocessing with SCREENFILTER

Although keyword-based searches typically retrieve relevant computer operations, they also include non-interactive segments, such as explanatory sections where the presenter uses PowerPoint or other presentation tools. Additionally, some of the videos retrieved through this method may not meet the standards for GUI interaction content.

To address this, we developed SCREENFILTER, a lightweight cursor detection model upon YOLOv8x (Reis et al.,

- 2024) to efficiently extract video segments that focus exclusively on GUI interactions. Using the detection results, we retain video segments where at least 80% of the frames contain a cursor for 6 seconds or more, with a 2-second merge gap for temporal smoothing. When applied to our corpus, SCREENFILTER successfully extracts 7,377 hours of verified GUI interactions from 10,000 hours of raw video. SCREENFILTER’s details are in Appendix B.

- 2.1.3 VideoAgentTrek Data Analysis

Quality and relevance. We collected 55k screen-capture videos (about 10,000 hours) from 50+ channels. The corpus is predominantly clear (about 97% are 720p or higher) and most clips are minutes long, yielding sustained, readable interactions suitable for our pipeline (Table 6). A lightweight title/description audit groups videos into tutorials, background pieces, tech talks, and unrelated; tutorials dominate (69.6%), with the remainder used mainly for tag mining or removed during filtering (Table 7). Together, these checks indicate that the collected data are both visually clean and topically aligned with computer-use supervision.

Data classification. We label each video as daily, office, workflow, professional, operating-system (OS), or other using a lightweight GPT-4.1 pass over the title and a short transcript snippet. The distribution (Figure 3) is skewed toward OS-level operations (∼36%), followed by professional (∼19%), daily (∼18%), and office (∼16%); workflow is smaller (∼7%) with a small remainder labeled as other (∼4%). This indicates broad coverage with a bias toward system and professional use cases.

Figure 3: Domain distribution.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 4: Overview of VIDEO2ACTION: Given a screen-capture video (with optional subtitles), the module (1) detects GUI action events and segments clips, (2) parameterizes each action (type and arguments), and (3) generates step-level thoughts, yielding training-ready sequences of {action clip, action, thought}

#### 2.2 VIDEO2ACTION: Inverse Dynamics Module

We develop VIDEO2ACTION, an inverse dynamics module (IDM) that extracts structured action supervision from unlabeled GUI videos. Following insights from robotics where inverse dynamics recovers actions from observations (Nguyen-Tuong & Peters, 2010), VIDEO2ACTION detects GUI events (clicks, drags, scrolls, typing) and infers their parameters directly from pixel changes. This yields training-ready (screenshot, action) pairs without manual annotation, forming the second core component of our VideoAgentTrek toolkit.

#### 2.2.1 Action Event Detection

Task. Given an unlabeled screen-capture video v (length T), perform prompt-free, dense event detection: predict a set of typed GUI interactions with tight temporal bounds,

fθ(v) → S = {(ak, tsk, tek)}kK=1, ak ∈ A, 0 ≤ tsk < tek ≤ T.

Unlike query-based setups, our input contains only v; the output is a multi-event set with both action types and start/end timestamps.

Approach. We equip a VLM with video grounding so that, given a clip, it emits a sequence of (ak, tsk, tek) for all GUI actions, reframing keyframe detection as multi-class temporal event detection with tight bounds. (1)

Training data: We utilize the annotation tool provided by OpenCUA (Wang et al., 2025b) to obtain synchronized screen videos and timestamped GUI interactions (mouse/keyboard events). These raw demonstration logs are then used to create temporal-grounding supervision, allowing precise event detection without manual annotation. (2) Model training: We leverage Qwen2.5-VL (Bai et al., 2025) as the base model, benefiting from its multimodal understanding and fine-grained spatiotemporal capabilities. We perform full-parameter supervised fine-tuning on the Qwen2.5-VL-7B-Instruct model to enable it to generate ordered, typed event spans directly from raw video clips. (3) Evaluation: We evaluate the detector in two phases. First, we check its performance on a small curated subset from the source corpus, ensuring tight boundaries and full recovery of relevant GUI actions. Second, we apply the model to unseen web tutorials and conduct blinded manual review to assess its robustness and real-world usability. Details are provided in Appendix C.

#### 2.2.2 Action Parameterization

Task. Given a detected action segment vk = v[tsk : tek] with type ak ∈ A, predict the action content (parameters) πk: hϕ(vk) → (aˆk, πk).

For example, a click segment yields hϕ(vk)→(click, (x, y)), while a typing segment yields hϕ(vk)→(type, ⟨content⟩).

Approach. We build a recognizer hϕ that, for each detected segment vk, predicts both the action type and its parameters (aˆk, πk). (1) Training data: We start from the OpenCUA raw demonstration logs, which pair screencapture video with timestamped mouse and keyboard events. Each event is converted into type-specific parameter labels and temporally aligned to its clip, yielding prompt-free supervision that captures the exact content of the interaction. (2) Model training: Using Qwen2.5-VL (7B Instruct) as the base, we perform full-parameter supervised fine-tuning so the model maps vk directly to (aˆk, πk); when available, we optionally condition on the detector’s ak to stabilize type predictions. (3) Evaluation: Because ground-truth object boxes are unavailable, we evaluate only on unseen web tutorials via blinded manual review, assessing whether the predicted action type and parameters are correct and practically actionable. Details are provided in Appendix D.

#### 2.2.3 Inner Monologue Generation

Dense event detection and action identification recover what happened on screen but omit the stepwise rationale. We therefore generate a brief inner monologue rk before each action to make explicit the intent, the local plan, and the expected state change (e.g., “type query into the search box to reveal results,” “scroll to bring the ‘Settings’ button into view”). Explicit rationales provide structured supervision for planning and credit assignment, tie cursor–target grounding to goals and affordances, and improve robustness on long-horizon tasks via better error detection and recovery. Recent GUI-agent work that injects step-level “thoughts” or System-2 reasoning reports notable gains in perception, grounding, and task execution, motivating our inclusion of rk in ReAct-style trajectories (Xu et al., 2025b; Qin et al., 2025; Wang et al., 2025b).

We cast inner-monologue generation as conditional paraphrasing with GPT-5 Medium. For each step k, we build a structured prompt that includes: (i) the detected action type ak; (ii) its parameters πk (e.g., typed text, cursor coordinates); (iii) the screen state immediately before and after the action (keyframes or thumbnails); and (iv) short ASR transcripts spanning a 1-minute window before the action, the during span [tsk, tek], and a 1-minute window after. Conditioned on these inputs, the model outputs a concise rationale rk that states the intent, the local plan, and the expected state change (grounded to visible UI). Additional prompt templates and representative inner-monologue examples are provided in Appendix E.

#### 2.3 Computer Use Model Pretraining

We demonstrate VIDEOAGENTTREK’s effectiveness by training an end-to-end computer-use agent with our video-driven data and a high-quality supervised finetuning set. On this strong finetuning basis, VIDEOAGENTTREK improves performance on online and offline agent evaluations.

#### 2.3.1 Agentic Data Collection

VideoAgentTrek Data. We apply VIDEO2ACTION to the collected tutorial videos and convert them into agentic supervision. For each processed clip, we (i) run dense event detection to obtain typed, tightly bounded segments, (ii) infer action parameters with the action-identification recognizer, and (iii) generate a brief inner monologue for intent and expected state change. We then assemble the resulting steps (Ik,rk, ak, πk) into trajectories and serialize them for downstream training. In total, we processed 39,000 videos; each video produces on average 39 steps, yielding approximately 1.52 million ReAct steps overall, and about 26 billion training tokens. Detailed data statistics and examples will be provided in the Appendix H.

Human demonstrations Data We sample human-annotated trajectories from OpenCUA (Wang et al., 2025b) and AGUVIS (Xu et al., 2025b), harmonizing formats and labels into a single schema. The corpus spans Windows, macOS, and Android, contributing about 8B tokens to training.

GUI Grounding Data. We include a focused subset of GUI grounding pairs from the OSWorld-G dataset (Xie et al., 2025a) to strengthen pointer–target alignment and layout-aware perception. This contributes roughly 1B tokens to training.

#### 2.3.2 Training strategy.

Automatically mined trajectories, while large-scale, inevitably contain residual noise. Motivated by prior findings that decoupling perception/grounding from policy learning improves robustness (Xu et al., 2025b; Wang et al.,

- 2025b), we adopt a two-stage schedule that first stabilizes grounding on broad but imperfect supervision and then consolidates policy on a clean subset.

Foundation Qwen2.5-VL-7B (Bai et al., 2025) is a general vision-language model with superior vision understanding capability, but it is not sufficiently pretrained on computer-use tasks with an end-to-end success rate of 4.5% on OSWorld (Xie et al., 2024), which makes it a proper starting point (base) for evaluating the data generated by VIDEOAGENTTREK.

- Stage 1 training. We train for one epoch over 26B tokens drawn from the VideoAgentTrek trajectories, augmented with a small number of GUI grounding pairs. Trajectories are formatted as interleaved vision–text sequences: frames (or frame-equivalent images) appear inline with the stepwise textual outputs, preserving temporal order across the entire clip. Loss is masked to the textual portions only; images are conditioning context and are not predicted. Please refer to Appendix F for representative formatting examples and complete training configurations including hardware, batch sizes, and optimization details.
- Stage 2 training. We continue training for 8B tokens on a curated set of clean, human-annotated trajectories. Here we reformat the data into a chat template with user prompts and assistant responses that describe or execute the next action. We apply standard supervised finetuning with loss computed only on the assistant turns, leaving user turns

- as pure conditioning. Representative formatting examples and training details are provided in the Appendix F.

- Figure 5: Experimental Results on OSWorld-Verified (Xie et al., 2025b) and AgentNetBench (Wang et al., 2025b). VideoAgentTrek demonstrates significant improvements over baseline models, with test-time scaling providing additional performance gains

### 3 Experiments

#### 3.1 Computer Use Agent Performance

- 3.1.1 Experiment Setup.

We evaluate the performance of our model on two computer-use agent benchmarks: OSWorld-Verified (Xie et al., 2025b; 2024) for online settings and AgentNetBench (Wang et al., 2025b) for offline settings. Further protocol, metrics, and computational details are provided in Appendix G.

- 1. OSWorld-Verified. OSWorld (Xie et al., 2024) is an online computer-use agent evaluation benchmark that includes 369 human-crafted Ubuntu desktop tasks. OSWorld-Verified (Xie et al., 2025b) is a more stable version, with updated evaluation scripts, environments, and clarified instructions, designed to measure CUA’s task-solving capabilities in dynamic, real-world environments.
- 2. AgentNetBench. AgentNetBench is an offline benchmark is based on 100 representative tasks from the AgentNet dataset, covering a wide range of applications and websites on Windows and macOS. The tasks are manually refined and offer multiple valid action options for each step to reflect the variety of correct interactions.
- 3.1.2 Main Results.

Video pretraining enhances performance on offline benchmarks. On AgentNetBench, incorporating VideoAgentTrek pretraining achieves a step success rate of 69.3%, representing a 5.2 percentage point improvement over the SFT-only baseline (64.1%) and a substantial 30.8 percentage point gain over the base model (38.5%). This consistent improvement demonstrates that video pretraining effectively transfers knowledge to structured offline evaluation scenarios.

Video pretraining delivers greater improvements on online benchmarks. On OSWorld-Verified, our complete approach achieves a task success rate of 14.13%, demonstrating a 4.83 percentage point improvement (+52% relative) over SFT-only training (9.3%) and more than tripling the performance of the base model (4.5%).

Video pretraining enables effective test-time scaling for computer-use agents. As shown in Figure 5, the performance of model trained with stage 1 and stage 2 improved from 14.13% to 15.78 % when the action step budget increases from 20 to 50 steps on OSWorld-Verified. This 1.65 percentage point improvement demonstrates the model’s ability to effectively utilize additional exploration opportunities. This test-time scaling benefit emerges specifically from our video pretraining: models trained on longer video trajectories learn to effectively utilize extended planning horizons, while the SFT-only baseline shows no improvement with additional steps (see Section 4.1).

3.2 VIDEO2ACTION Performance

- 3.2.1 Action Event Detection

We assess VIDEO2ACTION with a two-part protocol: a held-out, annotated test set and an in-the-wild manual validation:

Action Preds GT Precision Recall F1 Click 12,222 14,247 0.88 0.76 0.82 Drag 971 1,462 0.78 0.52 0.62 Press 177 842 0.40 0.08 0.14 Scroll 1,448 1,691 0.93 0.80 0.86 Type 1,480 2,040 0.89 0.64 0.75 Total 17,298 20,282 0.88 0.70 0.78

Action type Samples Accuracy Click 324 0.713 Drag 22 0.366 Press 47 0.362 Scroll 34 0.735 Type 73 0.671 Overall 500 0.658

Table 2: Action parameterization evaluation: manual in-the-wild assessment.

Table 1: Action-event detector evaluation: held-out testset results by action type.

Held-out test set. We hold out 23 hours of screen-capture videos with 20,282 annotated GUI events. Each event is a tuple (type, ts, te). A prediction counts as a hit iff its type matches and its interval has any temporal overlap with a ground-truth event; unmatched predictions are false positives and unmatched ground truths are false negatives. We report per-type Precision/Recall/F1 and micro/macro aggregates.

Manual validation (in-the-wild). On 10 unseen YouTube tutorials, we apply the same overlap criterion and estimate recovery rates by human review to assess robustness outside the curated set.

Results. As the results shown in Table 1, Overall precision is high (0.88) with solid recall (0.71). Pointer-centric actions (click, scroll) are reliably localized; keystroke-only actions show lower recall/precision due to subtle visual evidence. In the manual study, the detector recovers ∼70% of actions under the same criterion, consistent with in-house results.

#### 3.2.2 Action Identification

Evaluating action identification automatically is difficult because target-element boxes are unavailable. We therefore apply the identifier to in-the-wild videos and perform a blinded manual assessment. An action is judged proper if, when executed, it would plausibly produce the observed on-screen transition (for example, the clicked control changes state, typed text appears in the focused field, or the page scrolls). We evaluate 500 predictions sampled across action types.

Results. Annotators review pre/post frames and verify whether predicted parameters explain the observed changes, with disagreements resolved through second-pass review. Performance varies by action type: pointer-based actions (click, scroll) achieve highest accuracy, typing shows moderate accuracy despite OCR noise, while drag/press actions struggle with subtle visual cues. Despite these challenges, the predicted parameters are accurate enough for trajectory construction and downstream training; detailed counts and validation rates appear in Table 2.

4 Analysis

#### 4.1 Effectiveness of Data Scaling

To assess the impact of Stage-1 data scale, we train models using 0%, 50%, and 100% of the video tokens, then apply identical Stage-2 SFT to each variant. With increasing tokens, performance scales consistently across both benchmarks. AgentNetBench step success rates increase from 64.1% to 68.1% and 69.3%, while OSWorld-Verified task SR@50 grow from 9.3% to 13.3% and 15.7% (Figure 6). These findings establish a clear relationship between pretraining data size and computer-use agent performance, demonstrating the benefits of scaling video pretraining data.

#### 4.2 Improving Long Horizon planning

Figure 6: Performance Scaling

VIDEOAGENTTREK provides substantially longer trajectories than previous CUA corpora. As illustrated in Figure 10, 42.1% of trajectories exceed 20 steps, while 14.5% contain 50 or more steps, yielding an average trajectory length of 39.25 steps. Cross-dataset comparisons (Table 9) reveal that this average substantially exceeds those of established benchmarks, demonstrating that VIDEOAGENTTREKcorpus emphasizes supervision of complex, multi-step workflows rather than brief, single-interaction sequences.

The benefits of long-horizon supervision become evident when evaluating planning capabilities under varying step budgets. On OSWorld-Verified, we observe a striking difference in how models respond to increased action

budgets. The Stage2-only model shows no performance improvement when the step budget expands from 20 to 50 steps, remaining flat at 9.3% task success, indicating it cannot effectively plan beyond its training horizon or recover from early mistakes. In contrast, after Stage-1 pretraining on VideoAgentTrek’s long video trajectories, the agent demonstrates true test-time scaling: task success rate increases from 14.13% at 20 steps to 15.78%

- at 50 steps, a +1.65 point absolute improvement (+11.7% relative gain, Figure 5). This differential reveals that exposure to extended video demonstrations during pretraining teaches the model to decompose complex tasks into subgoals, persist through intermediate failures, and leverage additional computational budget for exploration and error correction, capabilities that supervised fine-tuning on shorter trajectories fails to instill.

### 5 Related Work

Generating agent trajectories. Computer-use trajectories have been obtained through human annotation, programmatic synthesis in instrumented environments, and web-scale mining of public resources. Human annotation, often aided by instrumentation to log pointer coordinates and keystrokes, yields precise labels but is costly and narrow in coverage (Qin et al., 2025; Wang et al., 2025a;b). Programmatic synthesis inside headless browsers or scripted desktop flows can generate large volumes with exact parameters, yet coverage is constrained by simulator APIs and may diverge from real-world UI variability (Su et al., 2025; Sun et al., 2025). Web-scale mining taps tutorials, RPA logs, and screen recordings to obtain diverse trajectories, but typically lacks precise temporal boundaries or action parameters (Xu et al., 2025a; Jang et al., 2025).

Precise Event Grounding in Video. Temporal grounding approaches such as temporal action localization, moment retrieval, keyframe detection, and dense video captioning seek to determine when events take place and provide corresponding descriptions (Lin et al., 2019; Zhuang et al., 2025; Wasim et al., 2024). Meanwhile, recent multimodal systems (e.g., Qwen2.5-VL (Bai et al., 2025), Gemini 2.5 Pro (Comanici et al., 2025)) have advanced the field by enabling more detailed spatiotemporal understanding and long-horizon video reasoning. Nonetheless, most general-purpose grounding frameworks focus primarily on semantic interpretation, rather than achieving the millisecond-level precision and parameter extraction required to faithfully reconstruct GUI interactions.

Learning from unlabeled video to act in environments. VPT demonstrated that large-scale unlabeled videos can be converted into effective training signals (e.g., via inverse-dynamics auto-labeling followed by behavior cloning), substantially improving an agent’s ability to act (Baker et al., 2022).Building on this idea, subsequent work leverages internet-scale human videos to distill human policy priors that transfer to interactive environments, including learning action-centric latent spaces without action labels (Ye et al., 2025) and scaling to humanoid control (Mao et al., 2024).

### 6 Conclusion

We presented VideoAgentTrek, a scalable pipeline that transforms publicly available screen recordings into structured supervision for computer-use agents without manual annotation. By developing an inverse dynamics module that accurately detects GUI events and extracts action parameters from raw video, we demonstrate that the implicit supervision in tutorial videos can be effectively harvested at scale. Our experiments on 39,000 YouTube videos yielded 1.52 million interaction steps, enabling continued pretraining that improved task success rates by 70% on OSWorld-Verified (9.3% to 15.8%) and increased step accuracy on AgentNetBench from 64.1% to 69.3%. These results establish that unlabeled internet videos, when processed through learned inverse dynamics, provide a viable and cost-effective alternative to expensive manual annotation for training robust computer-use agents. The open-source release of our ScreenFilter and Video2Action tools enables the community to leverage this abundant resource for advancing GUI automation research.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos, 2022. URL https://arxiv.org/abs/2206.11795.

Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, Yuan Yao, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Guicourse: From general vision language models to versatile gui agents, 2025. URL https://arxiv.org/abs/2406.11317.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, and Ice Pasupat.etc. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web, 2023. URL https://arxiv.org/abs/2306.06070.

Yunseok Jang, Yeda Song, Sungryull Sohn, Lajanugen Logeswaran, Tiange Luo, Dong-Ki Kim, Kyunghoon Bae, and Honglak Lee. Scalable video-to-dataset generation for cross-platform mobile agents, 2025. URL https://arxiv.org/abs/2505.12632.

Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on ui control agents, 2024. URL https://arxiv.org/abs/2406.03679.

Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. Bmn: Boundary-matching network for temporal action proposal generation, 2019. URL https://arxiv.org/abs/1907.09702.

Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, and Yue Wang. Learning from massive human videos for universal humanoid pose control, 2024. URL https://arxiv.org/abs/2412.14172.

Duy Nguyen-Tuong and Jan Peters. Using model knowledge for learning inverse dynamics. In 2010 IEEE international conference on robotics and automation, pp. 2677–2682. IEEE, 2010.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. Ui-tars: Pioneering automated gui interaction with native agents, 2025. URL https://arxiv.org/abs/2501.

12326. Dillon Reis, Jordan Kupec, Jacqueline Hong, and Ahmad Daoudi. Real-time flying object detection with yolov8,

2024. URL https://arxiv.org/abs/2305.09972.

Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, and Sercan O.¨ Arık. Learn-by-interact: A datacentric framework for self-adaptive agents in realistic environments, 2025. URL https://arxiv.org/ abs/2501.10893.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, and Zhiyong Wu. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis, 2025. URL https://arxiv.org/ abs/2412.19723.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu, Sihan Cao, Tao Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinhao Li, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yuhao Dong, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, Ziwei Chen, and Zongyu Lin. Kimi-vl technical report, 2025. URL https://arxiv.org/abs/2504.07491.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, ..., Qinghao Zhao, and Guang Shi. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning, 2025a. URL https://arxiv.org/abs/2509.02544.

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents, 2025b. URL https://arxiv.org/abs/2508.09123.

Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Videogroundingdino: Towards open-vocabulary spatio-temporal video grounding, 2024. URL https://arxiv. org/abs/2401.00901.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. URL https://arxiv.org/abs/2404.07972.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025a. URL https://arxiv.org/abs/2505. 13227.

Tianbao Xie, Mengqi Yuan, Danyang Zhang, Xinzhuang Xiong, Zhennan Shen, Zilong Zhou, Xinyuan Wang, Yanxu Chen, Jiaqi Deng, Junda Chen, Bowen Wang, Haoyuan Wu, Jixuan Chen, Junli Wang, Dunjie Lu, Hao Hu, and Tao Yu. Introducing osworld-verified. xlang.ai, July 2025b. URL https://xlang.ai/blog/ osworld-verified.

Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao Yu. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials, 2025a. URL https://arxiv. org/abs/2412.09605.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction, 2025b. URL https://arxiv. org/abs/2412.04454.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos, 2025. URL https://arxiv.org/abs/2410.11758.

Weijun Zhuang, Qizhang Li, Xin Li, Ming Liu, Xiaopeng Hong, Feng Gao, Fan Yang, and Wangmeng Zuo. Grounding-md: Grounded video-language pre-training for open-world moment detection, 2025. URL https: //arxiv.org/abs/2504.14553.

### A YouTube Video Quality Standards

To ensure consistency and usability in selecting high-quality instructional videos from YouTube for research purposes, the following standards must be met:

- 1. Minimal Overlays. If overlays, such as face cams or titles, are present, they must occupy no more than 101 of the screen area to avoid obstructing the primary content.

- 2. Primary Focus on Screen Recording. The video should predominantly feature clean screen recordings. Brief transitions to other scenes, such as PowerPoint slides or face capture, are permissible but should be limited to introductory or concluding segments.
- 3. Screen Recording Method. The video must consist of direct screen recordings rather than footage captured by an external camera.
- 4. Language Requirement. The video must be in English to facilitate monolingual captioning in subsequent processing steps.
- 5. Stable Visual Presentation. Frequent zooming in or out should be avoided. The entire screen or application window must be visible for the majority of the video duration.
- 6. Caption Availability. The video must include captions, indicated by the availability of the closed caption (CC) icon in the bottom right corner of the player. Captions may be auto-generated or manually annotated.
- 7. Orientation. The video must be recorded in a horizontal format, as vertical videos often fail to capture complete desktop screens, limiting their utility.
- 8. Recency. Videos must be no older than five years to ensure that the user interfaces depicted remain relevant and applicable.

These criteria ensure that selected videos are suitable for detailed analysis and processing in research contexts.

### B SCREENFILTER Details

SCREENFILTER is trained on 15,000 synthetic images generated by compositing cursor sprites onto GUI screenshots from the GUIEnv (Chen et al., 2025) dataset. To enhance its generalization across different platforms, we incorporate various cursor patterns from both Windows and macOS. On the held-out test set, SCREENFILTER achieves an F1 score of 89.58%, with 90.64% precision and 88.54% recall, demonstrating its effectiveness in accurately separating computer-use content from unrelated material.

For video processing, SCREENFILTER operates at 1-2 frames per second to balance both accuracy and efficiency. The model retains segments where at least 80% of the frames contain a cursor for a minimum of 6 seconds, with a 2-second temporal smoothing gap to merge frames. This design allows SCREENFILTER to process approximately 840 hours of video per GPU-day, facilitating large-scale filtering.

### C Dense Event Detection

Our dense detector is trained on 154 hours of screen-capture video paired with raw interaction logs from OpenCUA (Wang et al., 2025b). The logs contain complete demonstrations with precisely timestamped GUI interactions (mouse and keyboard). We convert these logs into prompt-free temporal grounding supervision by mapping lowlevel events to our action taxonomy, merging short consecutive micro-events into typed spans with start and end timestamps, and discarding segments without actionable GUI operations.

For training-set preparation, we downsample videos to 4fps, segment them into non-overlapping 10s clips, and align the interaction logs within each clip to obtain typed spans with start and end timestamps. We adopt a temporal patch size of 2 frames for modeling efficiency. Label names are normalized to our action taxonomy. We visualize the data sample in Figure 7

Model training. We perform full-parameter supervised fine-tuning of Qwen2.5-VL-7B-Instruct. The training configuration and loss curve are shown side-by-side in Figure 8.

### D Action Identification

#### Training data.

Our dense detector is trained on 512,000 screen-capture clips paired with raw interaction logs from OpenCUA (Wang et al., 2025b). To preprocess action segments, we adopt a dynamic frame-rate policy that caps frames per

Table 3: Event distribution in the training data (154 hours).

Action type Count click 410,101 key 138,660 write 80,749 scroll 46,597 moveTo 32,840 dragTo 32,840 doubleClick 14,241 rightClick 7,451 hscroll 3,411 hotkey 2,570 tripleClick 2,428 middleClick 57 Total 771,945

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Figure 7: Example training sample for the dense event detector.

clip at 20 while preserving short, fast interactions. For a segment of duration ∆t (seconds), we set

f = min{30, max{4, ⌊20/∆t⌋}},

then sample frames uniformly within [tsk, tek]. This yields, for example, f=30 for brief clicks/scrolls (∆t≈0.5s, ≈ 15 frames), f≈20 for ∆t≈1.0s (20 frames), and f=4 for extended typing segments (∆t≈5s, 20 frames). We

visualize the data sample in Figure

### E Inner Monologue Generation Prompt Content.

Inputs: Action type: ak Parameters: πk Before/after keyframes: Ikpre, Ikpost ASR windows: [−60s,0], [tsk, tek], [0,60s] Instruction (to model): You are generating inner-monologue annotations for a dataset of GUI agent trajectories built from in-the-wild screen recordings. End-to-end setting.

- • Source: real GUI screen recordings from the wild.
- • Extraction: each GUI interaction (an action) is automatically detected from video/audio.
- • For every detected action, you receive three kinds of evidence:

– Action details: {action type} and {action content}.

action content may contain: coordinates (absolute or normalized) and/or a bbox; typed text;

[Figure 97]

Framework Megatron-LM Hardware 32× H100 GPUs Tensor parallelism TP = 4 Pipeline parallelism PP = 1 Global batch size 256 Training iterations 2000 LR decay iterations 2000 Wall-clock time ∼15 h

Figure 8: Dense event detector: training loss (left) and training configuration (right).

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Figure 9: Example training sample for the action parametrization model.

pressed keys; scroll amount/direction; drag start/end; and similar specifics.

- – Keyframes: a start screenshot and, if available, an end screenshot right after the action executes.
- – Surrounding transcripts: short snippets of narration or speech immediately before, during, and after the action.
- – Action validation (optional): a brief validator description summarizing what occurred.

Your task. For each action, output exactly one JSON object with two fields: action description and thought. Field definitions (strict).

- • action description: a concise natural-language description of what I do in the UI at this step. Name the target UI element if inferable (button, menu, tab, field); otherwise describe by role/label/relative position. Mention the immediate visible outcome only if it is clearly observable. Forbidden: raw coordinates, code, function/method names, automation tokens, key–value argument lists.

- • thought: my first-person inner monologue (4–8 sentences) as the demonstrator (use “I”, “me”, “my”). Provide substantive reasoning. Include: (a) what I aim to accomplish and why now; (b) how the speech context informs my intent (weave naturally); (c) a brief summary of what likely changes from start to end if both frames exist; (d) a short breakdown of the atomic actions in this step (e.g., type

+ press) and why each is needed; (e) what I expect to verify or do next. Prefer present tense when natural.

#### General rules.

[Figure 103]

[Figure 104]

(a) Stage-1 training

[Figure 105]

(b) Stage-2 training

Figure 10: Computer Use Agent Training Data (a) Stage-1 training, (b) Stage-2 training.

- • The thought must be in first person; never switch to third person.
- • Evidence priority: prefer visual evidence from start/end keyframes; treat speech as a weak hint for why. If they conflict, prefer visuals.
- • Weave evidence naturally without naming “transcripts” or “frames.”
- • For coordinate-based actions, a red hollow circle may mark the interaction point; do not mention the marker, describe the target element instead.
- • If only a start keyframe is available, focus on intent; if an end keyframe exists, you may include the immediate visible result.
- • When a step bundles multiple atomic actions, reason across them as one coherent operation.
- • Keep action description concise; let thought carry the details; avoid hedging and boilerplate.

- • Output format: exactly one valid JSON object with only action description and thought; no extra keys or commentary.

Output: rk: inner-monologue JSON with fields action description and thought.

- F Computer Use Agent Training Training Data. we visualize the training data samples in stage-1 and stage-2 training in Figure 10.

- Stage-1 training. We perform full-parameter continue-pretraining Qwen2.5-VL-7B-Instruct. The training configuration and loss curve are shown side-by-side in Figure 11.
- Stage-2 training. We continue a full-parameter supervised fine-tuning on the stage-1-trained checkpoint. The training configuration and loss curve are shown side-by-side in Figure 12.

- G Computer Use Agent Evaluation

Evaluation Setting. We follow the OSWorld-Verified protocol: the agent interacts with a live desktop given a natural-language instruction and the full history of prior states and actions. At each step, the policy conditions on the instruction and a bounded visual context of up to five recent screenshots (FIFO window) together with the action/rationale history, then emits the next action. For the 20-step budget, we conduct three independent runs per model and report the average Task SR. For the 50-step budget, we perform a single run. All models use identical

[Figure 106]

Framework Megatron-LM Hardware 32× H100 GPUs Tensor parallelism TP = 4 Pipeline parallelism PP = 1 Global batch size 512 Training iterations 6500 LR decay iterations 6500 Wall-clock time ∼60 h

- Figure 11: CUA Stage-1 Training: training loss (left) and training configuration (right).

[Figure 107]

Framework Megatron-LM Hardware 64× H100 GPUs Tensor parallelism TP = 4 Pipeline parallelism PP = 1 Global batch size 512 Training iterations 3000 LR decay iterations 3000 Wall-clock time ∼16 h

- Figure 12: CUA Stage-2 Training: training loss (left) and training configuration (right).

inference settings and action executors; no manual interventions are allowed during evaluation.

AgentNet Bench summary. Overall step SR rises from 0.385 (base) to 0.641 with SFT-only and to 0.693 with Stage 1+Stage 2. These trends suggest that video pretraining notably strengthens grounding and multi-action control, especially for less frequent or harder motor primitives.

OSWorld-Verified summary. Table 4 reports task success across turns and step budgets. With SFT-only (stage2 only), Task SR hovers around 9.1–9.4% at 20 steps and shows no improvement at 50 steps (9.27%), indicating limited ability to leverage longer budgets. Adding VideoAgentTrek pretraining (stage1 + stage2) raises Task SR to 13.6–14.7% at 20 steps and further to 15.78% at 50 steps. Per-domain counts improve most for chrome/46 (up to 15 solved) and workflow/92 (up to 8 solved), with steady gains in os/24 and authoring apps (writer, impress). Across three 20-step runs, variance is modest, suggesting stable benefits from Stage 1. Overall, the results show that large-scale video pretraining yields higher step quality and makes the agent budget-sensitive—able to convert extra steps into additional task completions.

###### Model Eval Task SR (%) calc/46 chrome/46 gimp/26 vscode/23 writer/23 tbird/15 os/24 impress/47 workflow/92 vlc/17

- turn1 (20) 9.42 1 8 3 6 2 3 2 3 5 1
- turn2 (20) 9.13 2 8 1 6 2 2 2 3 5 2
- turn3 (20) 9.42 1 8 2 6 2 2 2 4 5 2
- turn4 (50) 9.27 1 12 2 3 2 2 2 4 4 1

stage2 only

- turn1 (20) 14.68 2 13 2 6 5 6 4 6 7 2
- turn2 (20) 13.57 2 13 2 5 5 6 3 6 5 2
- turn3 (20) 14.13 2 12 2 7 7 6 3 5 5 2
- turn4 (50) 15.78 1 15 2 6 6 6 4 6 8 3

stage1 + stage2

- Table 4: OSWorld-Verified full results. Counts indicate solved tasks per application bucket (denominators shown in headers).

Model Step SR click write press scroll moveTo dragTo hotkey dbClick rClick terminate

base 0.385 0.402 0.605 0.286 0.615 0.189 0.000 0.250 0.000 0.000 0.188 stage2 only 0.641 0.671 0.719 — 0.500 0.300 0.145 0.484 0.526 0.214 0.588 stage1 + 2 0.693 0.767 0.733 0.441 0.600 0.502 0.264 0.562 0.650 0.417 0.237

- Table 5: AgentNet Bench: step success rate (overall and per action type). “—” indicates the metric was not applicable/recorded.

### H VideoAgentTrek Data Analysis

Resolution and scale. We downloaded 55,603 screen-capture videos (about 10,000 hours) from 50+ channels. The corpus is predominantly clear: 97% are 720p or higher (Table 6). Most videos are minutes long, providing sustained interactions suitable for dense detection and action identification.

Resolution bucket Count High (1080p+) 2,322 Standard (720p–1080p) 49,589 Low (<720p) 1,464

Table 6: Resolution distribution of downloaded videos.

Title/description-based content classification. To quickly audit topical relevance at scale, we apply a lightweight classifier to each video’s title and brief description.

#### • Labels.

- – A tutorial: hands-on screen tutorials.

- * Include: step-by-step demonstrations, cursor-driven walkthroughs, “how to ...” tasks; frequent UI focus changes; imperative phrasing in titles (“Create...”, “Install...”, “Fix...”).
- * Exclude: talk-style narrations with few concrete on-screen actions; marketing teasers without real steps.
- * Signals: verbs tied to UI operations (open, click, type), timestamps/chapters per step, tool/app names plus action verbs.

- – B background: expository background with incidental screen use.

- * Include: market share reports, product overviews, concept explainers where the desktop appears only as a backdrop.
- * Exclude: segments that actually show multi-step operations (move to A tutorial).

- * Signals: nouns like “overview, history, comparison, review,” charts/stats in title/description, little or no cursor interaction.

- – C tech talk: talks or presentations with slides.

- * Include: conference talks, webinars, lectures; slide navigation with limited live demos.
- * Exclude: talks that transition into substantial live step-by-step demos (then split or relabel A tutorial).

- * Signals: “keynote, webinar, seminar, lecture,” speaker names/affiliations, slide thumbnails.

- – D unrelated: off-topic for computer-use learning.

- * Include: content where a screen appears but no actionable computer-use task is taught (e.g., pure entertainment, face-cam only).
- * Exclude: any clip with consistent stepwise UI operations (move to A tutorial).

- * Signals: lifestyle/vlog tags, gameplay without UI instruction, no app/task keywords.

- • Procedure. Single-pass GPT-4.1 classification with a short instruction to choose exactly one of the four labels given the title and short description; no transcript or frames are used. We use the result only for corpus auditing, tag mining, and optional down-weighting in later filters, not as a hard accept/reject gate.
- • Limitations. Metadata-only classification can mislabel borderline cases (e.g., talks that include substantial demos). Final training sets are still screened by cursor gating, license/PII checks, and downstream detectors.

Distribution. Class counts and shares are shown in Table 7. The majority are tutorials, indicating strong alignment with our target use case.

Label Count Share

- A tutorial 38,700 69.6%

- B background 12,900 23.2%

- C tech talk 2,391 4.3%

- D unrelated 1,612 2.9% Total 55,603 100%

Table 7: distribution from title/description classification.

Action type Count Share (%) left click 1,037,617 67.1 type 214,816 13.9 key 145,860 9.4 scroll 111,203 7.2 right click 24,111 1.6 double click 11,848 0.8 mouse move 8,441 0.1 drag 6,372 0.1 hscroll 196 0.0 Total 1,547,092 100.0

Table 8: Action distribution in the VideoAgentTrek agentic dataset.

Action distribution. Table 8 summarizes action counts in the VideoAgentTrek agentic data. Cross-dataset comparison. We summarize reported average step counts (and task counts when available) for common CUA datasets and include our corpus for context.

Dataset Tasks Avg. Step AndroidControl (Li et al., 2024) 15,283 5.5 OS-Genesis (Sun et al., 2025) 2,451 6.4 AgentTrek (Xu et al., 2025a) 10,398 12.1 Mind2Web (Deng et al., 2023) 2,350 7.3 AgentNet (Wang et al., 2025b) 22,625 18.6 VideoAgentTrek 39,000+ 39.25†

Table 9: Average steps across datasets (as reported in their papers). †Estimated from a 5,416-trajectory sample in our corpus. Table 10: VideoAgentTrek data distribution of

step number

