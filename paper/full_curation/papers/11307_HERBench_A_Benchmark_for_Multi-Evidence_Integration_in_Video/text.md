# arXiv:2512.14870v2[cs.CV]2Apr2026

HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering

Dan Ben-Ami1,∗ Gabriele Serussi1,∗ Kobi Cohen2 Chaim Baskin1

1INSIGHT Lab, Ben-Gurion University of the Negev, Israel 2Ben-Gurion University of the Negev, Israel ∗Equal contribution.

###### Abstract

Video Large Language Models (Video-LLMs) are improving rapidly, yet current Video Question Answering (VideoQA) benchmarks often admit single-cue shortcuts, under-testing reasoning that must integrate evidence across time. We introduce HERBench, a benchmark designed to make multi-evidence integration unavoidable: each question requires at least three non-overlapping cues drawn from distinct video segments. HERBench contains 26,806 five-way multiple-choice questions across 12 compositional tasks. To make evidential demand measurable, we introduce the Minimum Required Frame-Set (MRFS), the smallest number of frames a model must fuse to answer correctly, and show that HERBench imposes higher evidential demand than prior benchmarks. Evaluating 13 state-of-the-art Video-LLMs yields only 31–42% accuracy, only modestly above the 20% random-guess baseline. We disentangle this failure into two critical bottlenecks: (1) a retrieval deficit, where frame selectors overlook key evidence, and (2) a fusion deficit, where models fail to integrate information even when all necessary evidence is provided. HERBench thus provides a principled benchmark for studying robust multi-evidence video understanding.

[Figure 1]

#### 1 Introduction

As Video Large Language Models [2–4] achieve strong scores on established VideoQA benchmarks [5–11], their video understanding capabilities appear to be rapidly emerging. However, recent audits reveal these high scores often stem from language priors or single-cue shortcuts rather than grounded temporal reasoning [12, 13], causing models to fail tasks that explicitly require multi-hop inference [14–16]. In contrast, tasks like Referring Video Object Segmentation (RVOS) demonstrate that robust, multi-frame aggregation is achievable, as models successfully link instances across occlusions and appearance changes [17–20].

We advocate centering evaluation on evidential requirement, because single-cue questions fail to measure multievidence integration. We define the Evidential Requirement (ER) as the minimum number of distinct, non-redundant visual evidence needed for an answer. High-ER items make compositional reasoning, such as temporal binding and clue combination, unavoidable [7, 15, 21]. Controlling ER therefore distinguishes models that integrate information from those that rely on isolated cues [12]. This approach makes aggregation measurable, aligns VideoQA with real-world reasoning, and offers a principled path for progress beyond single-cue success [7, 15, 21].

We introduce HERBench (High Evidential Requirement Benchmark), where questions across twelve compositional subtasks (e.g., entity binding, temporal ordering) are constructed to structurally enforce k ≥ 3 distinct pieces of evidence, as presented in Figure 1. To measure this, we present the Minimum Required Frame-Set (MRFS) metric, defined as the minimum number of frames needed for a correct answer. Cross-benchmark comparison under a canonical MRFS protocol confirms our high-ER design: HERBench attains the highest mean MRFS among the benchmarks considered and enables principled ER-focused diagnostics.

Our evaluation of state-of-the-art Video-LLMs exposes two critical bottlenecks. Finding 1: Frame selection is a major bottleneck. While adaptive selectors [22, 23] outperform uniform sampling, they still lag significantly behind ground-truth keyframes. Finding 2: Multi-evidence reasoning is also a bottleneck. Even with ground-truth frames, models achieve only modest accuracy because they fail to assign proper importance to all critical frames and struggle to integrate them. Progress therefore requires advances in both frame selection and multi-evidence reasoning.

## MVBench

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

What direction is the blue sphere moving in within the video?

A The object is stationary. B Down and to the left. D Down and to the right.

C Up and to the right.

## HERBench (Ours)

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

The following 4 shots take place in the video:

A 1->2->3->4 B 4->1->3->2 C 1->4->3->2

1->4->2->3

###### QUESTION

- 1. <Shot description>
- 2. <Shot description>
- 3. <Shot description>
- 4. <Shot description> Select the option that correctly re ects the order in which these shots occur in the video:

What is the person doing in the video?

E At least two descriptions do not accurately re ect any shot from the video

- Figure 1: From Single-Cue to Multi-Evidence Integration. While existing benchmarks like MVBench [1] (top) often focus on short-term attributes solvable via single salient frames or language priors, HERBench (bottom) enforces a high Evidential Requirement (ER). In this Temporal Shot Ordering example, the model must identify and temporally bind four distinct, non-overlapping visual evidence dispersed across the video to reconstruct the correct sequence. This design ensures that successful answering requires genuine multi-evidence integration rather than reliance on static shortcuts. Our main contributions are summarized as follows.

- • We introduce HERBench: a benchmark with 26,806 questions that are constructed to structurally enforce k ≥ 3 distinct, non-redundant visual cues.
- • We propose the Minimum Required Frame-Set (MRFS) metric: a measure of the smallest number of frames a model must aggregate to answer a question correctly, thereby enabling apples-to-apples comparison across benchmarks and powering ER-focused diagnostics.
- • We identify two critical bottlenecks in current Video-LLMs: By disentangling frame selection from multi-evidence reasoning, we reveal two systemic failures. (i) Frame selection: Adaptive selectors, though an improvement over uniform sampling, still overlook key evidence and do not yet match the performance of oracle key-frames. (ii) Multievidence reasoning: Even with oracle frames, models fail to integrate complementary information and systematically underweight necessary evidence. Progress requires advances in both selection and reasoning.

#### 2 Related Work

Video Large Language Models. Video-LLMs architectures have evolved from simple feature pooling [24, 25] to sophisticated systems employing advanced alignment modules (e.g., Q-Formers) and large-scale instruction tuning [26–28] to bridge the modality gap. Despite massive token capacities in proprietary models like Gemini 2.5 [29] and GPT-4o [30], recent audits [12, 31] reveal a persistent failure in robust temporal aggregation. Instead of performing multi-hop inference, these models frequently default to language priors or single-frame shortcuts to solve tasks.

Video Question Answering Benchmarks. VideoQA benchmarks have evolved from short-clip recognition [32, 33] toward longer-form evaluation, but they often probe different aspects of video understanding. MVBench [1] broadened the task space with diverse temporal questions, yet its short clips limit the assessment of long-horizon assessment. More recent benchmarks, including EgoSchema [9], LongVideoBench [11], and Video-MME [10], expand temporal scope, while MINERVA [34] emphasizes complex multi-step reasoning and reasoning-trace auditing. HERBench targets a different axis: not only how long the context is, but how many visual pieces of evidence must be combined. While phenomena such as temporal ordering and counting appear in prior benchmarks, they are typically not formulated so that solving them requires aggregating multiple temporally separated cues. HERBench instead explicitly controls the Evidential Requirement (ER): each question is constructed to require at least three non-overlapping cues drawn from distinct moments in the video, and the benchmark’s oracle-frame design enables retrieval failures to be disentangled from fusion failures.

#### 3 HERBench: High Evidential Requirement Benchmark

##### 3.1 Task Taxonomy

To evaluate whether models truly integrate evidence rather than rely on a salient cue, we organize 12 tasks into four reasoning families (Figure 2). These families recast familiar VideoQA settings under structural k ≥ 3 constraints over long videos, while also introducing tasks for appearance-grounded identity binding, set-level identity maintenance, and region-conditioned aggregation.

Temporal Reasoning & Chronology [TR&C]. These tasks require understanding event order, co-occurrence, and durations, compiling distributed cues into a linear chronology. The three tasks are: 1) [TSO] Temporal Shot Ordering: Arrange four shot descriptions from a trailer into the correct chronological order, using only content cues. 2) [MPDR] Multi-Person Duration Reasoning: Compare interval statistics for appearance-described people (e.g., who stayed in view the longest, or who entered/exited first), focusing on fine-grained time-span contrasts across individuals. 3) [ASII] Action Sequence Integrity & Identification: Select the correct ordering of five narrated actions among plausible permutations, stressing micro-level task sequencing rather than scene-level ordering. The ER is driven by ordering and interval comparisons across at least three temporally separated observations, but each task probes a distinct temporal structure.

Referring & Tracking [R&T]. This family tests binding a uniquely appearance-described target across time to reason about trajectory-dependent properties. Models must maintain a stable reference as the target interacts with the scene. The tasks are: 1) [AGBI] Appearance-Grounded Behavior Interactions: Identify who accompanies or interacts with the target during traversal, emphasizing social and relational cues. 2) [AGAR] Appearance-Grounded Attribute Recognition: Track the target to read out attributes anchored to their immediate local context (e.g., a passerby’s jacket color), focusing on moment-specific attribute extraction. 3) [AGLT] Appearance-Grounded Localization Trajectory: Recover path endpoints and coarse trajectory (e.g., exit method), highlighting global, path-level motion reasoning. This enforces k ≥ 3 through identity maintenance across separated glimpses, as the target description is composed of cues scattered across distinct moments that must be jointly resolved, with each task centering on a different aspect of target evolution.

Global Consistency & Verification [GC&V]. Next, we test exhaustive video-wide verification and absence detection, sweeps that must confirm what occurred and surface plausible but missing elements. The three tasks are: 1) [FAM] False Action Memory: Among several plausible actions, select the one that never occurs while verifying the others

(1) Temporal Reasoning & Chronology

|[TSO] Temporal Shot Ordering<br><br>The following 4 shots take place in the video: <Shot description 1>, <Shot description 2>, <Shot description 3>, <Shot description 4>. Select the option that correctly reflects the order in which these shots occur in the video:<br><br>Answer: D) 1->4->3->2<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]|
|---|

|[ASII] Action Sequence Integrity & Identi cation<br><br>What is the correct temporal order of the 5 narrated events?<br><br>Answer: C) 1. slide co ee capsule -> 2. close lid of food recycling bin -> 3. turn o  food processor -> 4. place orange -> 5. put down sponge<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]|
|---|

[MPDR] Multi-Person Duration Reasoning

These people were in the video: <Appearance description 1>, <Appearance description 2>, <Appearance description 3>. Who stayed in the frame FOV for the longest time?

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Answer: D) Person 2

###### (2) Referring & Tracking

[AGBI] Appearance-Grounded Behavior Interactions [AGAR] Appearance-Grounded Attribute Recognition

[AGLT] Appearance-Grounded Localization Trajectory

In the video there is exactly one individual that fits the following description. <Appearance description>. How does the person exit the frame at the end of their path?

In the video there is exactly one individual that fits the following description. <Appearance description>. Who is accompanying the person they walk across the frame in the video?

In the video there is exactly one individual that fits the following description. <Appearance description>. In the video, what color is the jacket worn by the individual who remains seated as the main subject walks past?

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

Answer: E) Another male wearing a light blue shirt and dark jacket

Answer: B) Blue

Answer: A) Through the right edge

###### (3) Global Consistency & Veri cation [FAM] False Action Memory

[SVA] Scene Veri cation Arrangement

[FOM] False Object Memory

Which of the following actions did NOT occur in the video?

From the correctly described shots, which is the one that appears first in the video?

Which object did the camera wearer NOT interact with?

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

Answer:

C) A person wearing a dark hoodie and jeans stands under a spotlight in a dimly lit room, with a focused expression.

Answer: B) open up fridge

Answer: D) Garlic presser

(4) Multi-Entity Aggregation & Numeracy

|[RLPC] Region-Localized People Counting<br><br>How many people entered the frame of the video through the top edge? Select the range that includes the correct count.<br><br>Answer: D) 4-13<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|
|---|

[MEGL] Multi-Entities Grounding & Localization

[AC] Action Counting

How many times does the action-object pair 'close tap' occur?

Which of the following people appeared in the video (the person description must match exactly and accurately): <Appearance description 1>, <Appearance description 2>, <Appearance description 3>

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

Answer: A) only 1 and 3

Answer: B) 5

- Figure 2: Task taxonomy of HERBench. We organize 12 fine-grained compositional tasks into four essential reasoning families: (1) Temporal Reasoning & Chronology, (2) Referring & Tracking, (3) Global Consistency & Verification, and (4) Multi-Entity Aggregation & Numeracy. Unlike existing benchmarks that may allow for single-frame shortcuts, every task in HERBench is constructed to enforce a High Evidential Requirement, requiring models to aggregate at least three distinct, temporally separated visual cues (k ≥ 3) to derive the correct answer.

do, requiring action-level absence detection. 2) [SVA] Scene Verification Arrangement: Given 2-4 shot descriptions where some may be fabricated, first identify the faithful ones, then arrange the correct shots in temporal order, or return a calibrated abstention when too many descriptions are false; this combines shot-level fidelity checking with chronology. 3) [FOM] False Object Memory: Among plausible objects, identify the one the camera wearer does not interact with while verifying the rest, stressing object-level absence tied to first-person interactions. Here k ≥ 3 arises from multi-moment sweeps needed to validate presence and detect absence across the video.

Multi-Entity Aggregation & Numeracy [MEA&N]. Finally, this family stresses many-way binding, spatial partitioning, and precise counting across multiple people or events. Models must deduplicate identities across time and fuse evidence spread over the video. The three tasks are: 1) [MEGL] Multi-Entities Grounding & Localization: Given 2-3 detailed appearance descriptions, decide which individuals actually appear in the video (exact-match verification among plausible distractors), focusing on set membership and identity deduplication. 2) [AC] Action Counting: Count the occurrences of a specified action-object pair distributed across the timeline, emphasizing event-accumulation across dispersed moments. 3) [RLPC] Region-Localized People Counting: Count unique individuals subject to spatial constraints (e.g., entries through the top edge), with answers reported as binned ranges, requiring region-conditioned identity aggregation. Here k ≥ 3 is enforced by set-level aggregation and cardinality constraints over multiple moments,

[Figure 98]

- Figure 3: HERBench construction pipeline. Videos are processed through three streams—object tracking and trajectory analysis, shot segmentation, and ground-truth integration (refining human verified raw event logs) —whose outputs are compiled by oriented task programming and filtered by manual review and text-only bias checks. with each task stressing a complementary aggregation mode.

##### 3.2 Benchmark Construction

We construct HERBench through the tripartite data construction pipeline shown in Figure 3. The core of this process is the creation of a rich spatiotemporal scaffold by processing each video through three complementary streams. The first stream, Object Tracking & Trajectory Analysis, focuses on continuous, micro-level object dynamics. Complementing this, the second stream, Shot Segmentation, provides a macroscopic view by discretizing the video into semantic units. Finally, the Ground Truth Integration stream anchors the analysis in human-verified facts. Together, these streams produce a diverse set of refined data (such as A/B cards, scene cards, and event labels).

- Pipeline I: Object Tracking & Trajectory Analysis. This first stream anchors tasks in continuous object dynamics. We employ RF-DETR [35] and DeepSORT [36] to obtain entity tracks, retaining top 20% entities via a TrackRank score, a composite score favoring appearance rarity, trajectory length and frame coverage (see Supplementary). For each track, we generate strictly non-overlapping A-cards (appearance) and B-cards (behavior/trajectory). This decorrelation intentionally separates the identifying appearance from the queried behavior, often placing them in temporally distant frames, and enables a new class of appearance-grounded identity-binding tasks that cannot be solved by local attribute lookup alone. Full error quantification and noise control details are provided in the Supplementary.

This scaffold supports tasks requiring fine-grained interaction and motion analysis:

- • [AGBI], [AGAR], [AGLT]: We generate questions strictly from B-cards while referring to entities via A-cards, separating appearance from behavior. [AGBI] queries behavioral interactions with other entities; [AGAR] queries attributes; and [AGLT] queries path integration and motion topology.
- • [MPDR]: We compute per-entity visible-time intervals to generate queries comparing durations (e.g., longest presence) or checking for temporal overlaps. The correct answer is, by definition, a property of the relationship between multiple, ordered cues.
- • [RLPC]: We execute spatial programs to count unique track IDs traversing predefined regions of interest or entry/exit gates, testing spatiotemporal aggregation capabilities.
- • [MEGL]: We form sets of appearance descriptors and inject plausible distractors, forcing models to verify the exact set of present individuals throughout the video.

- Pipeline II: Shot Segmentation. Where the first pipeline focuses on continuous entity-level detail, this second stream discretizes the video into larger semantic units. It uses shot boundary detection, employing an MLLM to summarize each segment into a concise scene card. This macroscopic view supports tasks dependent on global temporal coherence:

[Figure 99]

[Figure 100]

[Figure 101]

- Figure 4: Left: Wordcloud of frequent terms in HERBench queries. Center: Distribution of samples across source datasets. Right: Number of questions per task category.

- • [TSO]: We query the chronological arrangement of the generated scene cards, requiring the model to reorder shuffled scenes.
- • [SVA]: We mix faithful scene cards with plausibly perturbed variants, altering 2-5 atomic details (e.g. actions, attributes), to test resistance to gist cues or partially correct descriptions.

- Pipeline III: Ground Truth Integration. Finally, this stream moves beyond automated analysis to leverage human verified narrated events [37]:

- • [FAM], [FOM]: We introduce corpus-plausible distractors, entities or actions common in similar videos but verified as absent, requiring multi-timestamp scanning rather than single-frame spot checks.
- • [ASII]: We establish ground-truth chronology from narrated events and present proposed sequences (faithful vs. perturbed) for careful verification.
- • [AC]: Ground-truth counts are derived directly from verified event logs to test long-horizon aggregation.

Synthesis & Quality Control. The refined representations are compiled by oriented task programming into multiplechoice questions, then filtered through several safeguards. We enforce A/B-card disentanglement via token-level similarity checks and manual leakage review, discard items solved by ≥ 3 of 4 blind LLMs, and perform expert verification on a stratified 15% sample to confirm minimum frame-set (k ≥ 3) compliance and answer uniqueness. Human validation further confirmed both answerability and oracle-evidence quality: annotators achieved 88.8% accuracy with full-video access and 95.7% in an oracle-frame setting; details are deferred to the Supplementary Material. As HERBench is partially instantiated through an oriented task-programming pipeline, some residual systematic artifacts may nevertheless remain.

##### 3.3 Benchmark Statistics

HERBench contains 26,806 five-way multiple-choice questions from 336 unique videos spanning 12 tasks and four reasoning families. The videos are long-form (395 s on average; range 60–2100 s), diverse in source and viewpoint, and chosen to ensure temporal dispersion of evidence; Figure 4 summarizes the resulting vocabulary, source distribution, and per-task question counts.

##### 3.4 Evidential Requirement & the MRFS Metric

Motivation. VideoQA questions may require fusing distributed evidence or be solvable via single salient frames. To quantify this, we introduce the Minimum Required Frame-Set (MRFS): the smallest number of frames a model must fuse to answer correctly. A higher mean MRFS confirms that questions resist single-cue shortcuts and genuinely demand multi-moment integration. Although prior metrics capture important temporal dynamics, they measure fundamentally different properties. Temporal Indispensability (1-frame vs. N-frame performance) [38] tests for static shortcuts, while Certificate Length [9, 10] measures the human-annotated temporal span needed for verification. In contrast, MRFS is an automated, model-centric metric that isolates the exact number of frames a model must fuse, directly quantifying the multi-evidence aggregation challenge without relying on human annotation.

Lang. debias

Enforced fusion

Absence check

Oracle frames

Benchmark # Videos # Questions MRFS↑

TemporalBench 2,179 9,867 2.21 ✗ ✗ ✗ ✗ MMBench-Video 609 1,998 4.41 ✗ ✗ ✗ ✗ Video-MME 900 2,700 5.31 ✗ ✗ ✗ ✗ AGQA (balanced) 9,600 3.9 M 3.42 ✗ ✗ ✗ ✗ CVRR-ES 217 2,400 2.77 ✗ ✗ ✓ ✗ LongVideoBench 3,763 6,678 4.07 ✗ ✗ ✗ ✗ NExT-QA 5,440 99,736 2.61 ✗ ✗ ✗ ✗ MINERVA 223 1,515 5.14 ✓ ✗ ✗ ✗ MVBench 4,000 4,000 3.52 ✗ ✗ ✗ ✗

HERBench 336 26,806 5.49 ✓ ✓ ✓ ✓

- Table 1: Benchmark comparison under the canonical MRFS protocol. MRFS is reported with f = Qwen2.5-VL, r = AKS, and x = 16. Enforced fusion indicates whether correct answering structurally requires combining multiple temporally separated visual cues.

Definition. Let v denote the video, q the question, and y the ground-truth answer. Let f be a fixed MLLM, r a question-conditioned frame selector, and x a frame budget. The selector produces a ranking π = r(v,q) over frames and we denote Fk = {π1,...,πk} as the top-k subset. With evaluator E(ˆy,y) = 1{yˆ = y}, we define MRFSx(q;f,r) =

min k ∈ {1,...,x} : E f(q,Fk), y = 1 , (1)

subject to the precondition E(f(q,∅),y) = 0 so that text-only solvable items are excluded from MRFS computation. Intuitively, MRFSx is the least amount of visual evidence (in frames) that suffices for f to be correct when frames are supplied in an r-determined, question-aware order.

Computation. We search for the smallest success index using an adaptive bisection over k ∈ [1,x], requiring O(log x) model calls per item. Each question is categorized as: (i) text-only (correct with no frames, f(q,∅)), (ii) visual-required (correct for some 1 ≤ k ≤ x), or (iii) undefined (incorrect even at k = x).

Cross-benchmark MRFS comparison. To enable cross-benchmark comparison, we fix a canonical MRFS protocol: Qwen2.5-VL as backbone, AKS as frame selector, and x = 16 frames. Table 1 compares HERBench to prior benchmarks under this shared setting, together with complementary design properties. HERBench achieves the highest mean MRFS (5.49) in the comparison and is uniquely defined by the combination of language debiasing, structurally required temporal fusion, absence checking, and oracle-frame availability. Among prior benchmarks, MINERVA [34] is the closest in spirit, pairing language debiasing with a relatively high MRFS (5.14), but focusing on multi-step reasoning and reasoning-trace auditing rather than enforced multi-frame evidence aggregation. The benchmark ordering remains stable across alternative backbones and selectors (see Supplementary), and the comparison with LongVideoBench [11] suggests that HERBench derives its difficulty from evidential density rather than duration alone.

#### 4 Experiments

To validate the challenges posed by HERBench, we conduct a comprehensive evaluation of current state-of-the-art Multimodal Large Language Models. Our experiments are designed to quantify their performance on tasks explicitly requiring the integration of multiple, temporally dispersed visual cues.

Setup. We evaluate 13 prominent MLLMs, including both closed- and open-source systems (Table 2). To isolate the effects of evidence aggregation from frame selection, all models receive the same budget of 16 uniformly sampled frames from each video.

TR&C R&T GC&V ME&N

Model

Overall Avg. TSO MPDR ASII Avg. AGBI AGAR AGLT Avg. FAM SVA FOM Avg. MEGL AC RLPC Avg.

GPT-4.1 [39] 18.9 29.7 27.7 25.4 78.0 59.1 61.0 66.0 30.4 38.9 41.9 37.1 25.5 24.3 37.3 29.0 39.4 Gemini-2.5-Flash [29] 28.6 35.8 24.8 29.7 75.2 71.4 63.1 69.9 29.2 31.3 44.2 34.9 22.6 26.6 31.2 26.8 40.3 Qwen2.5-VL-72B [40] 10.4 42.6 27.8 26.9 74.4 76.1 62.2 70.9 25.6 50.6 33.5 36.6 18.1 23.0 32.0 24.4 39.7 Gemma-3-27B [41] 38.4 42.0 15.7 32.0 69.0 50.5 55.6 58.4 21.8 14.3 28.4 21.5 15.7 29.0 25.7 23.5 33.8 LLaMA-4-Scout-17B [42] 6.2 30.0 20.1 18.8 64.7 51.6 55.6 57.3 19.3 36.5 20.7 25.5 17.2 26.1 29.4 24.2 31.4 InternVL3.5-14B [43] 43.9 38.8 30.3 37.7 75.9 69.4 62.6 69.3 26.8 22.8 43.8 31.1 25.3 20.8 37.3 27.8 41.5 Ovis-2.5-9B [4] 0.1 30.6 26.0 18.9 79.7 76.2 64.7 73.5 33.6 57.2 49.6 46.8 27.5 23.4 36.7 29.2 42.1 InternVL3.5-8B [43] 41.3 31.3 28.1 33.6 77.6 71.6 61.4 70.2 26.3 21.2 41.5 29.7 33.1 21.2 38.1 30.8 41.1 LLaVA-OneVision1.5-8B [3] 26.6 28.7 23.0 26.1 76.8 67.5 58.8 67.7 29.8 33.9 37.1 33.6 25.2 17.6 31.9 24.9 38.1 Qwen3-VL-8B [2] 2.2 28.7 26.0 19.0 74.6 69.6 61.9 68.7 30.0 51.3 40.4 40.6 18.8 21.8 34.9 25.2 38.3 MiniCPM-V4.5-8B [44] 19.1 26.3 26.0 23.8 77.9 72.3 63.2 71.1 30.2 43.7 45.2 39.7 24.1 22.9 27.9 24.9 39.9 Qwen2.5-VL-7B [40] 14.6 28.0 22.9 21.8 69.7 59.3 52.9 60.6 33.0 36.0 47.1 38.7 21.1 20.3 26.3 22.6 35.9 LLaVA-OneVision-7B [28] 33.3 24.9 23.7 27.3 67.1 58.0 52.3 59.1 28.9 22.4 38.9 30.1 22.8 22.4 32.8 26.0 35.6 Avg. 23.1 31.9 25.5 26.8 74.5 66.3 59.7 66.8 28.4 35.2 40.9 34.8 23.2 23.0 32.7 26.3 38.2

- Table 2: HERBench results. We report per-task accuracy (%) for 13 leading MLLMs. The highest performance in each task column is marked in bold. The 4 largest-size models (first rows) were run on a representative 10% subset (∼2.6K questions), while the remaining models were evaluated on the full benchmark.

Results. As shown in Table 2, performance is systematically poor. The mean accuracy across all 13 state-of-the-art models is 38.2%, with the best model (Ovis-2.5-9B [4]) reaching only 42.1% and the lowest (LLaMA-4-Scout-17B [42]) at 31.4%. This narrow performance band, just 11–22 percentage above the 20% random baseline, reveals that failure to integrate dispersed evidence is a pervasive limitation across all current architectures.

The performance breakdown by task is telling. Models show relative competence on single-entity tracking tasks like [AGBI] and [AGAR] (Ovis-2.5-9B: 79.7%, 76.2%). This suggests they can track a single described entity. However, performance collapses on all tasks strictly requiring multi-cue aggregation. For [AC] and [MEGL], mean accuracies are 23.0% and 23.2% respectively, barely above chance. Similarly, models fail at temporal ordering ([TSO]), with scores as low as 0.1%, demonstrating a clear inability to compose dispersed information.

In summary, these results demonstrate that while state-of-the-art MLLMs can track single entities, they fundamentally fail at the core challenge of multi-evidence compositional reasoning. Our controlled-frame evaluation confirms this deficit stems from a failure to integrate information, not merely a failure to access it.

#### 5 Analysis

This section analyzes the two major challenges highlighted by HERBench: (Q1) how frame selection strategies affect performance through evidence retrieval, and (Q2) whether models can effectively aggregate evidence across the correct frames once retrieval uncertainty is removed.

##### 5.1 Isolating the Evidence Retrieval Bottleneck

Frame Selection methods. To address (Q1), i.e. the impact of evidence retrieval, we compare five strategies (all operating in the same BLIP [45] embedding space for fairness): AKS [22] learns a keyframe policy that balances relevance and temporal coverage; BOLT-ITS [23] using inverse transform sampling to select query-relevant frames; Uniform takes evenly spaced frames; Vanilla-BLIP retrieves frames with highest cosine similarity to the question; Oracle Frames (OF) use frame indices gathered from our benchmark’s construction pipeline along with a few nonevidence frames to match the fixed frame budget (they are meant to represent a best-case retrieval setting rather than the oracle-only fusion setting studied in Sec. 5.2). They are applied only in tasks where relevant evidence is scarce or confined to very short portions of the video (notably [TSO], [FAM], [SVA]).

Performance across frame selection strategies. Table 3 presents the accuracy, averaged across all tasks, for each selection method applied to three representative models. Learned selectors such as AKS and BOLT-ITS outperform simple uniform sampling on many tasks, yet still trail behind the Oracle Frames (OF) configuration, a performance gap

###### Frame Selection InternVL3.5-14B Qwen3-VL-8B Ovis-2.5-9B

Uniform 42.7 37.7 43.1 Vanilla-BLIP 42.1 37.9 41.6 BOLT-ITS 41.1 38.4 42.1 AKS 42.7 36.2 42.6 Oracle Frames (OF) 47.8 41.0 47.9

- Table 3: Mean accuracy by frame selection method and model. Rows list frame selection methods. Each cell shows the mean accuracy over 1200 questions (100 from each task).

[Figure 102]

- Figure 5: Top-1 frame share under oracle-only frames. We plot the maximum normalized frame-importance share for three models, separated by correct and incorrect predictions. Correct answers distribute importance more evenly, whereas errors concentrate it on a single frame, indicating weak multi-evidence fusion even when evidence-bearing frames are provided.

that is even more pronounced in the per-task breakdown (see Supplementary), reinforcing the fact that evidence retrieval remains a major performance bottleneck. More importantly, even when the model is provided with the correct evidence frames (using OF), performance gains are limited, with accuracy remaining below 50%, indicating that access to the right information alone is insufficient for successful multi-evidence reasoning. This finding aligns with our broader observation that current models underweight or fail to integrate critical cues, even when ground-truth evidence is fully available.

##### 5.2 Evidence Aggregation with Oracle-Only Frames

Having established that evidence retrieval is a significant bottleneck (Sec. 5.1), we now turn to (Q2): can models effectively aggregate evidence even when retrieval uncertainty is removed? To isolate the fusion capability from the retrieval challenge, we conduct a targeted study on a subset of HERBench supplying models with only the manually curated ground-truth frames (the ”oracle” set). In a parallel human study under the same oracle presentation format, annotators achieved 95.7% accuracy, indicating that the curated oracle frame-set is generally sufficient for resolving the question.

Measuring Frame-Level Contribution. For each item, we compute per-frame deltas and shares that quantify how much each frame contributes to the model’s own predicted option:

- 1. Full prediction. Run the model on all oracle frames and compute log pfull, where p is the post-softmax probability of the chosen letter token (A–E), with the softmax taken only over these candidate tokens.
- 2. Leave-one-out re-run. For each frame i, re-run with that frame excluded (the context contains the remaining n − 1 frames) to obtain log pminus[i].
- 3. Delta. ∆i = log pfull − log pminus[i]; positive ∆i means frame i supports the model’s chosen option.
- 4. Share. si = ∆+i / j ∆+j , yielding a normalized importance distribution across frames.

Diagnosing Fusion Failures via Importance Distribution. We analyze per-frame importance shares to understand why models succeed or fail under oracle-only inputs, summarizing each item using the Top-1 Share (maxi si) as

shown in Figure 5. This statistic captures how strongly a model concentrates its decision on a single frame. The distributions reveal a consistent pattern: correct predictions (green) exhibit substantially more balanced allocations, with mean Top-1 shares near 0.5. Incorrect predictions (red), in contrast, show pronounced over-concentration, with Top-1 shares frequently approaching 0.8. This separation indicates that errors arise not merely from insufficient signal, but from misallocation of attention-models place disproportionate weight on one frame while failing to assign sufficient importance to the multiple, distributed evidential cues present across the oracle set. Because HERBench questions structurally require multi-frame reasoning, this behavior shows that the fusion module itself-independent of retrieval—remains a primary source of failure.

#### 6 Conclusion

We introduced HERBench, a VideoQA benchmark comprising 26,806 questions across 12 tasks, each designed to structurally enforce the aggregation of k ≥ 3 distinct, temporally separated visual cues. To quantify evidential demand, we proposed the Minimum Required Frame-Set (MRFS) metric; under a canonical evaluation protocol, HERBench attains the highest mean MRFS among the benchmarks considered. Experiments on 13 state-of-the-art MLLMs reveal a narrow accuracy range (31.4–42.1%), only modestly above the 20% chance baseline, exposing persistent limitations in multi-evidence reasoning. We trace these failures to two main bottlenecks: a retrieval deficit, where frame selectors fail to recover all necessary cues, and a fusion deficit, where models fail to combine them even when available, often collapsing onto a single frame. HERBench therefore establishes a principled benchmark for studying high-evidential-requirement video understanding, exposes substantial headroom beyond single-cue success, and points toward future improvements in retrieval-aware querying and distributed evidence fusion.

#### References

- [1] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo, L. Wang, and Y. Qiao, “MVBench: A comprehensive multi-modal video understanding benchmark,” in CVPR, 2024. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/papers/Li MVBench A Comprehensive Multi-modal Video Understanding Benchmark CVPR 2024 paper.pdf

- [2] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
- [3] X. An et al., “Llava-onevision-1.5: Fully open framework for democratized multimodal training,” arXiv preprint arXiv:2509.23661, 2025. [Online]. Available: https://arxiv.org/abs/2509.23661
- [4] S. Lu et al., “Ovis2.5 technical report,” arXiv preprint arXiv:2508.11737, 2025. [Online]. Available: https: //arxiv.org/abs/2508.11737
- [5] J. Lei, L. Yu, M. Bansal, and T. L. Berg, “Tvqa: Localized, compositional video question answering,” in EMNLP, 2018, arXiv:1809.01696.
- [6] Y. Jang, Y. Song, Y. Yu, Y. Kim, and G. Kim, “Tgif-qa: Toward spatio-temporal reasoning in visual question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2758–2766.
- [7] J. Xiao, X. Shang, A. Yao, and T. Chua, “NExT-QA: Next phase of question-answering to explaining temporal actions,” in CVPR, 2021. [Online]. Available: https://arxiv.org/abs/2105.08276
- [8] M. Tapaswi, Y. Zhu, R. Stiefelhagen, A. Torralba, R. Urtasun, and S. Fidler, “Movieqa: Understanding stories in movies through question-answering,” in CVPR, 2016. [Online]. Available: https://openaccess.thecvf.com/content cvpr 2016/papers/ Tapaswi MovieQA Understanding Stories CVPR 2016 paper.pdf

- [9] K. Mangalam, R. Akshulakov, and J. Malik, “Egoschema: A diagnostic benchmark for very long-form video language understanding,” Advances in Neural Information Processing Systems, vol. 36, pp. 46212–46244, 2023.
- [10] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, P. Chen, Y. Li, S. Lin, S. Zhao, K. Li, T. Xu, X. Zheng, E. Chen, C. Shan, R. He, and X. Sun, “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 24108–24118.

- [11] H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A benchmark for long-context interleaved video-language understanding,” in NeurIPS, 2024, arXiv:2407.15754. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2024/hash/ 329ad516cf7a6ac306f29882e9c77558-Abstract-Datasets and Benchmarks Track.html

- [12] Y. Liu, S. Li, Y. Liu, Y. Wang, S. Ren, L. Li, S. Chen, X. Sun, and L. Hou, “Tempcompass: Do video llms really understand videos?” in Findings of ACL, 2024, arXiv:2403.00476. [Online]. Available: https://aclanthology.org/2024.findings-acl.517/
- [13] J. Xiao, A. Yao, Y. Li, and T. Chua, “Can i trust your answer? visually grounded video question answering,” in CVPR,

2024. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao Can I Trust Your Answer Visually Grounded Video Question Answering CVPR 2024 paper.pdf

- [14] R. Girdhar and D. Ramanan, “Cater: A diagnostic dataset for compositional actions & temporal reasoning,” in ICLR, 2020. [Online]. Available: https://arxiv.org/abs/1910.04744
- [15] K. Yi, C. Gan, Y. Li, P. Kohli, J. Wu, A. Torralba, and J. B. Tenenbaum, “Clevrer: Collision events for video representation and reasoning,” in ICLR, 2020. [Online]. Available: https://arxiv.org/abs/1910.01442
- [16] M. Grunde-McLaughlin, R. Krishna, and M. Agrawala, “Agqa: A benchmark for compositional spatiotemporal reasoning,” in CVPR, 2021. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2021/papers/ Grunde-McLaughlin AGQA A Benchmark for Compositional Spatio-Temporal Reasoning CVPR 2021 paper.pdf

- [17] K. Gavrilyuk, A. Ghodrati, Z. Li, and C. G. M. Snoek, “Actor and action video segmentation from a sentence,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [18] S. Seo, J.-Y. Lee, and B. Han, “Urvos: Unified referring video object segmentation network with a large-scale benchmark,” in European Conference on Computer Vision (ECCV), 2020.
- [19] A. Botach, E. Zheltonozhskii, and C. Baskin, “End-to-end referring video object segmentation with multimodal transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [20] J. Wu, Y. Jiang, P. Sun, Z. Yuan, and P. Luo, “Language as queries for referring video object segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [21] B. Wu, S. Yu, Z. Chen, J. B. Tenenbaum, and C. Gan, “STAR: A benchmark for situated reasoning in real-world videos,” in NeurIPS, 2021, datasets & Benchmarks Track. [Online]. Available: https://arxiv.org/abs/2405.09711
- [22] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, “Adaptive keyframe sampling for long video understanding,” in CVPR, 2025, arXiv:2502.21271.
- [23] S. Liu, C. Zhao, T. Xu, and B. Ghanem, “Bolt: Boost large vision-language model without training for long-form video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 3318–3327.
- [24] H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audio-visual language model for video understanding,” arXiv preprint arXiv:2306.02858, 2023. [Online]. Available: https://arxiv.org/abs/2306.02858
- [25] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024.
- [26] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.
- [27] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi, “Instructblip: towards general-purpose vision-language models with instruction tuning,” in Proceedings of the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY, USA: Curran Associates Inc., 2023.
- [28] B. Li et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024. [Online]. Available: https://arxiv.org/abs/2408.03326
- [29] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

- [30] OpenAI, “Gpt-4o,” 2024, model card and system card.
- [31] B. Feng, Z. Lai, S. Li, Z. Wang, S. Wang, P. Huang, and M. Cao, “Breaking down video llm benchmarks: Knowledge, spatial perception, or true temporal understanding?” in NeurIPS Workshop, 2025. [Online]. Available: https://arxiv.org/abs/2505.14321
- [32] D. Xu, Z. Zhao, J. Xiao, F. Wu, H. Zhang, X. He, and Y. Zhuang, “Video question answering via gradually refined attention over appearance and motion,” in ACM Multimedia, 2017.
- [33] J. Xu, T. Mei, T. Yao, and Y. Rui, “Msr-vtt: A large video description dataset for bridging video and language,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 5288–5296.
- [34] A. Nagrani, S. Menon, A. Iscen, S. Buch, R. Mehran, N. Jha, A. Hauth, Y. Zhu, C. Vondrick, M. Sirotenko, C. Schmid, and T. Weyand, “Minerva: Evaluating complex video reasoning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 23968–23978.
- [35] I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, and N. Peri, “Rf-detr: neural architecture search for real-time detection transformers,” arXiv preprint arXiv:2511.09554, 2025.
- [36] N. Wojke, A. Bewley, and D. Paulus, “Simple online and realtime tracking with a deep association metric,” in 2017 IEEE International Conference on Image Processing (ICIP), 2017.
- [37] T. Perrett, A. Darkhalil, S. Sinha, O. Emara, S. Pollard, K. K. Parida, K. Liu, P. Gatti, S. Bansal, K. Flanagan, J. Chalk, Z. Zhu, R. Guerrier, F. Abdelazim, B. Zhu, D. Moltisanti, M. Wray, H. Doughty, and D. Damen, “Hd-epic: A highly-detailed egocentric video dataset,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 23901–23913.
- [38] X. Fang, K. Mao, H. Duan, X. Zhao, Y. Li, D. Lin, and K. Chen, “MMBench-Video: A long-form multi-shot benchmark for holistic video understanding,” in NeurIPS, 2024, datasets and Benchmarks Track, arXiv:2406.14515.
- [39] OpenAI, “Introducing gpt-4.1 in the api,” https://openai.com/index/gpt-4-1/, 2025.
- [40] S. Bai et al., “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025. [Online]. Available: https://arxiv.org/abs/2502.13923
- [41] Gemma Team, “Gemma 3 technical report,” arXiv preprint arXiv:2503.19786, 2025. [Online]. Available: https: //arxiv.org/abs/2503.19786
- [42] Meta AI, “Llama 4 model card (scout models),” https://github.com/meta-llama/llama-models/blob/main/models/llama4/ MODEL CARD.md, 2025, model card; no public technical report for Llama 4 Scout as of Nov 2025.

- [43] W. Wang et al., “Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025. [Online]. Available: https://arxiv.org/abs/2508.18265
- [44] T. Yu et al., “Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe,” arXiv preprint arXiv:2509.18154, 2025. [Online]. Available: https://arxiv.org/abs/2509.18154
- [45] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International conference on machine learning. PMLR, 2022, pp. 12888–12900.
- [46] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024. [Online]. Available: https://github.com/ultralytics/ultralytics
- [47] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in European Conference on Computer Vision (ECCV). Springer, 2024, pp. 38–56.
- [48] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, “ByteTrack: Multi-object tracking by associating every detection box,” in European Conference on Computer Vision (ECCV). Springer, 2022, pp. 1–21.
- [49] P. Sun, J. Cao, Y. Jiang, Z. Yuan, S. Bai, K. Kitani, and P. Luo, “DanceTrack: Multi-object tracking in uniform appearance and diverse motion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20993–21002.
- [50] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in European Conference on Computer Vision (ECCV). Springer, 2014, pp. 740–755.

- [51] T. Souˇcek and J. Lokoˇc, “TransNet V2: An effective deep network architecture for fast shot transition detection,” arXiv preprint arXiv:2008.04838, 2020.
- [52] Qwen Team, “Qwen2: A family of open large language models,” 2024, alibaba Cloud.
- [53] ——, “Qwen2.5 technical report,” 2024, alibaba Cloud.
- [54] Meta AI, “The llama 3 herd of models,” 2024, model release report.
- [55] W.-L. Chiang, Z. Li et al., “Vicuna v1.5: An open-source chatbot,” 2023, fastChat project report.
- [56] T. Chavdarova, P. Baqu´e, S. Bouquet, A. Maksai, C. Jose, T. Bagautdinov, L. Lettry, P. Fua, L. Van Gool, and F. Fleuret, “Wildtrack: A multi-camera hd dataset for dense unscripted pedestrian detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 5030–5039.
- [57] B. Shuai, A. Bergamo, U. Buechler, A. Berneshawi, A. Boden, and J. Tighe, “Large scale real-world multi-person tracking,” in European Conference on Computer Vision. Springer, 2022, pp. 504–521.
- [58] J. Ye, Z. Wang, H. Sun, K. Chandrasegaran, Z. Durante, C. Eyzaguirre, Y. Bisk, J. C. Niebles, E. Adeli, L. Fei-Fei et al., “Re-thinking temporal search for long-form video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 8579–8591.

### HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering

Supplementary Material

- 7 Implementation Details

This section provides comprehensive implementation details for the HERBench construction pipeline, which employs a tripartite structure to enforce high evidential requirements (ER). We detail the algorithms, mathematical formulations, thresholds, and quality control procedures used to transform raw videos into the final dataset.

##### 7.1 Track Ranking and Selection

Tracking and Trajectory Refinement. To ensure robust performance in our benchmark’s highly dynamic environments—

characterized by dense crowds and significant depth variations—we conducted a rigorous empirical evaluation of state-of-the-art detection and tracking pipelines. Specifically, we benchmarked three leading detectors (RF-DETRL [35], YOLO-v11-x [46], and Grounding-DINO [47]) in combination with two widely adopted multi-object trackers (DeepSORT [36] and ByteTrack [48]). This evaluation was performed on a curated subset of the DanceTrack [49] benchmark, explicitly selected to mirror the crowd density, complex motion dynamics, and high inter-subject appearance diversity typical of our video corpus.

As detailed in Table 4, the combination of RF-DETR-L [35] and DeepSORT [36] yielded the highest tracking fidelity, achieving a HOTA of 47.1% and an IDF1 of 53.2%. Consequently, we adopted this configuration as the foundation of our pipeline. We utilize this detection-tracking stack by applying a high-recall RF-DETR [35] detector with a confidence threshold of 0.3 and a per-frame cap of 300 detections. Association within DeepSORT [36] uses a two-stage IoU matching: high-confidence detections (score > 0.5) are matched with an IoU threshold of 0.7, followed by lower-confidence detections with a relaxed IoU threshold of 0.35.

HOTA / IDF1 (%) RF-DETR-L YOLO-v11-x Grounding-DINO

DeepSORT 47.1 / 53.2 43.3 / 51.1 42.9 / 50.6 ByteTracker 46.9 / 52.7 43.3 / 51.0 42.6 / 50.5

- Table 4: Tracking pipeline benchmarking on a curated DanceTrack [49] subset mirroring HERBench’s dynamic characteristics. The combination of RF-DETR-L and DeepSORT achieved the highest scores and was selected for our data generation pipeline.

To enforce physical plausibility, we apply an outlier removal step that explicitly discards per-frame boxes implying implausible motion (velocity > 50 pixels/frame) to eliminate spurious detections. To ensure continuity, we apply gap interpolation for missing detections up to 30 frames (1s at 30 fps) and trajectory smoothing via Gaussian filtering (window size 5). We specifically address track fragmentation by detecting merge candidates (Ti,Tj) that are temporally ordered with a gap ≤ 30 frames and spatially compatible. We minimize the following merge cost:

Cmerge = ∆tgap + ∥cilast − cjfirst∥2

(2)

IoU(boxilast,boxjfirst)

where c denotes the bounding box centroid. The overall tracking, post-processing, and ranking pipeline is visualized in Figure 6.

TrackRank scoring function. To select the top m ∈ [6,10] salient entities per video, we compute a composite TrackRank score Si that aggregates metrics for each track i (all computed per video and normalized by the maximum over tracks). Unlike simple duration-based ranking, we use the following weighted formulation:

wk · Mi,k k wk

Si = k

(3)

[Figure 103]

- Figure 6: Tracking, post-processing, and ranking pipeline. RF-DETR [35] detections are linked with DeepSORT [36] into raw person tracks, followed by outlier removal, gap interpolation, and Gaussian smoothing. A TrackRanker then scores and selects salient trajectories, which are passed to an MLLM descriptor module to generate temporally decoupled appearance (A) and behavior (B) cards that serve as the scaffold for downstream HERBench tasks. The specific components and their empirically tuned weights are:

- • Duration (w = 2.0) & Size (w = 1.0): Favors tracks with sustained presence and higher average bounding box area.
- • Associated Objects (w = 2.0): Normalized count of distinct non-person object classes overlapping the person’s box (IoU > 0.2).
- • Center Distance (w = 2.4) & Motion (w = 1.0): Euclidean distance between first and last centroids, favoring traversals over stationary behavior.
- • Appearance Exceptionality (w = 2.2): We quantify rarity as the normalized L1 distance from the dataset’s average appearance in feature space (HSV and LBP histograms).
- • Scene Coverage (w = 1.5): Area of the Convex Hull enclosing the track’s boxes.
- • Quality Metrics: Aggregates Average Confidence (w = 0.8, mean detection score), Smoothness (w = 0.7, computed as 1 minus normalized acceleration magnitude to penalize jitter), and Aspect-Ratio Stability (w = 0.5, defined as 1 minus the standard deviation of width/height ratios to penalize shape fluctuations).

Hard Filter Cascade. Prior to ranking, we enforce a hard filter: we keep only the COCO [50] “person” class, require length ≥ 20 frames, average area ≥ 5,500 pixels, and require the track center to fall within the central safe region (frame cropped by 10% margins) in at least 5 frames.

Diversity Sampling Strategy. To ensure diversity among the selected tracks, we employ a round-robin selection across rankings generated from multiple perturbed weight configurations (γ ∼ U(0.5,1.5)). This prevents redundancy (e.g., selecting visually identical pedestrians) and ensures a broad coverage of high-quality entities, which are subsequently manually validated to exclude phantom detections or identity switches.

Track Selection as Noise Control. Per video, we select the top 6–10 tracks according to the TrackRank composite score (Eq. 3). Post-hoc analysis of the selected tracks’ average detection confidence shows that they consistently fall within the top ∼20% most confident tracks per video, confirming that the ranking procedure implicitly filters out low-confidence, noise-prone trajectories before they can propagate errors into the question generation pipeline. The TrackRank selection process thus doubles as a noise control mechanism: by funneling only high-confidence, well-resolved trajectories into downstream task programming, it substantially mitigates the risk of identity switches, fragmented detections, or phantom tracks propagating into the ground-truth labels.

##### 7.2 Decoupled Descriptor Generation

A-card and B-card generation. For each selected track, we generate disentangled descriptions using GPT-4o [30]. We sample 10-11 crops, reserving the first and last 20% of the trajectory for Appearance (A-cards) and the middle 60% for Behavior (B-cards). This ensures a temporal gap of at least 30 frames between appearance and behavior cues. An example of the resulting disentangled A- and B-cards for a single track is shown in Figure 7. We use the following prompt structure:

[Figure 104]

- Figure 7: Example of disentangled A- and B-cards. For a single tracked individual (highlighted trajectory in the top-left strip), we show the sampled frames and the corresponding appearance (A-card) and behavior (B-card) descriptions. The A-card captures only static visual attributes (clothing, colors, accessories, physique), while the B-card describes the person’s path, timing, and interactions over time without repeating appearance cues, enforcing the “Look & Separate” principle.

|System prompt. For the following tasks, use only your vision capabilities. When referring to directions, use the camera’s point of view.<br><br>1. Person Description. All images depict the same individual. In 2–4 sentences, describe their appearance in detail: clothing types and colors, accessories, hair, body build, and any distinctive features that make them easy to pick out. Do not mention position in the frame or any actions.<br>2. Path Description. In 3–7 sentences, describe the person’s path and behavior over time. Mention the overall path shape, entry and exit edges, stops, and interactions. Do not repeat any appearance details from the first description.<br>|
|---|

To visualize the output of this pipeline, Figure 7 presents qualitative examples of the generated Appearance (A) and Behavior (B) cards alongside their corresponding tracked image crops. These examples highlights the effectiveness of the temporal split: the tracked visual crops from the start and end of the trajectory inform the static attribute descriptions in the A-card, while the central frames drive the dynamic action summaries in the B-card. This separation ensures that the descriptors remain disentangled.

Leakage prevention. To strictly enforce the “Look & Separate” principle, we calculate the token-level Jaccard similarity between the generated A-card and B-card. We set the Jaccard threshold to 0.15 based on manual inspection: above this, descriptors often share explicit appearance/behavior leakage.

##### 7.3 Spatial Operations and Region Definitions

Entry/exit edge labeling. For tasks like Region-Localized People Counting (RLPC), we define entry and exit edges based on the position of a track’s centroid in its first and last frames. Let ct = (xt,yt) be the centroid at frame t of a track with start frame tstart and end frame tend, and let W,H denote the frame width and height. We say that a track enters through edge e if ct

lies in the band of e′. The top edge band is defined as y < 0.3H, the bottom as y > 0.85H, and the left/right edges as the outer 15% of the width (x < 0.15W and x > 0.85W, respectively).

lies in the corresponding edge band, and exits through edge e′ if ct

start

end

Region-of-interest (ROI) membership. For [RLPC], we also define rectangular ROIs (e.g., frame halves or specific zones). A track is counted as visiting an ROI if, at any frame, at least 50% of its bounding box area lies within the region (Intersection-Over-Box ≥ 0.5). We count the unique track IDs that satisfy this predicate to derive people counts under spatial constraints. To absorb residual tracking noise (missed detections, fragmented tracks), multiple-choice options are reported as binned count ranges rather than exact integers. The bins are constructed so that the correct range

[Figure 105]

- Figure 8: Faithful and perturbed scene cards for SVA. The top card provides a faithful one-sentence description of a shot, mentioning the main actor, appearance, background, and motion. The bottom card is a perturbed variant where 2-5 atomic details (e.g., clothing pattern, background appearance, additional objects) are modified or added while remaining globally plausible. These pairs form positive and negative options in the Scene Verification & Arrangement task, probing fine-grained scene-level sensitivity to small but visually significant details.

spans approximately ±40% around the true count on average, ensuring that minor tracking errors do not invalidate the ground truth while still requiring models to perform meaningful spatial counting, see Figure 21 for example.

Duration computation ([MPDR]). We compute visible-time intervals (tstart,tend) for every track. Using interval algebra, we determine ground truth for questions such as “Who stayed longest?” or “Who entered first?” by comparing duration scalars (tend − tstart) and timestamps.

##### 7.4 Scene Card Perturbations

Shot Segmentation and Description. We use TransNetV2 [51] for shot boundary detection. To calibrate its reliability on our video corpus, we manually reviewed shot boundaries on 30 videos used for [TSO] and [SVA] tasks (34% of videos contributing to these tasks). Comparing TransNetV2 [51] predictions against manual annotations yields F1 =

- 0.97, confirming that shot segmentation noise is negligible. For the [SVA] task, faithful scene cards are generated via an MLLM using the following prompt: “

|“Describe concisely the scene in one sentence without reference to the ‘scene’, refer (if relevant) to the entities, genders and appearance (type and colors of hair/clothing/accessories) of each entity, occurrence, actions, background, and location.”|
|---|

Perturbation Engine. To generate negative samples for [SVA], we prompt the model to modify faithful descriptions by altering 2-5 atomic details. The prompt constraints ensure:

- • Modifications: Change existing details (color, count, attributes).
- • Additions: Insert plausible but absent elements (extra objects, background items).
- • Plausibility: Changes must be false but highly plausible within the context of the video. An example of a faithful scene card and its perturbed counterpart used for the [SVA] task is shown in Figure 8.

###### 7.5 Corpus-Plausible Foil Generation Ground Truth Integration. For tasks requiring verification of absence, we leverage human-verified event logs.

- • False Action Memory ([FAM]): We sample a “false” action by pairing an object present in the video with an action from the corpus that does not occur in the current video.
- • False Object Memory ([FOM]): We select an absent object from the corpus-wide index that is compatible with actions present in the video (e.g., if “cutting” occurs, “carrot” is a valid distractor if absent).
- • Action Counting ([AC]): Distractor counts are generated such that the correct count’s rank varies uniformly across options.

Task Full video (%) Oracle frames (%)

[TSO] 91.7 93.8 [MPDR] 88.3 97.0 [ASII] 86.7 95.8 [AGBI] 95.8 98.1 [AGAR] 95.0 98.3 [AGLT] 92.5 97.4 [FAM] 84.2 92.8 [SVA] 84.2 94.8 [FOM] 87.5 96.3 [MEGL] 85.8 95.4 [AC] 84.2 97.7 [RLPC] 90.0 90.9

Overall 88.8 95.7

- Table 5: Per-task human accuracy. Accuracy of human annotators in the full-video and oracle-frame settings across all HERBench tasks. In the full-video setting, annotators answer with unrestricted video access and free scrubbing. In the oracle-frame setting, annotators answer using only the curated oracle frame-set, without access to the source video.

• Action Sequence Integrity ([ASII]): We sample a 5-event ground-truth timeline. Distractors are generated using two perturbation functions: swap mid (swapping two non-adjacent events) and rotate (shifting the sequence). Crucially, we verify against the event log that the perturbed timeline does not accidentally exist in the video.

##### 7.6 Text-Only Bias Filtering Details

Filtering procedure. To suppress language priors, we apply a rigorous Text-Only Filtering stage. We discard any question correctly answered by ≥ 3 of 4 blind LLMs (Qwen2-7B [52], Qwen2.5-7B [53], Llama-3-8B [54], and Vicuna-7B v1.5 [55]). This step rejects approximately 10% of candidates (e.g., questions answerable via object-color co-occurrence priors).

##### 7.7 Human Verification Protocol

Experts conduct verification on a stratified 15% sample of instantiated questions to audit whether the construction pipeline preserves the intended evidential constraints. The audit focuses on three properties: (i) minimum frame-set compliance, i.e. confirming that the item requires at least three distinct frames; (ii) uniqueness of the answer, i.e. confirming the existence of a single objective ground-truth answer; and (iii) descriptor disentanglement, i.e. verifying that A-cards and B-cards do not leak information from one another. Items that violate any of these conditions are rejected. This process resulted in a 17.8% rejection rate.

##### 7.8 Human Validation and Oracle-Frame Study

To complement the construction-time audit above, we conducted two human studies that assess HERBench from two complementary perspectives: overall question answerability and oracle-evidence sufficiency. Each study used a separate group of 6 annotators.

Study design. In the full-video setting, annotators answered a shared set of 240 questions spanning all 12 tasks (20 questions per task) with unrestricted video access and free scrubbing. In the oracle-frame setting, annotators answered 2,160 questions using only the curated oracle frame-set provided by the benchmark construction pipeline, without access to the source video. Each oracle item was presented in the same format used by the oracle-based analysis in the main paper, namely the curated evidence frames together with distractor frames.

Results. Table 5 reports the per-task accuracies. In the full-video setting, annotators achieved 88.8% accuracy overall, with substantial inter-annotator agreement (Fleiss’ κ = 0.74), indicating that the benchmark remains highly answerable for humans despite its high evidential demand. In the oracle-frame setting, annotators achieved 95.7% accuracy overall,

showing that the curated oracle frame-set is generally sufficient to resolve the question without access to the full temporal context. The largest improvements appear in tasks such as [AC], [SVA], and [MEGL], where the answer depends on sparse or temporally localized evidence.

Benchmark cleanup. After each oracle-frame response, annotators were shown the ground-truth answer and invited to flag problematic items. This process surfaced three types of issues: mis-indexed evidence frames, incorrect groundtruth labels, and genuinely ambiguous items. Specifically, annotators flagged 42/2160 items (1.9%) for evidence mis-indexing, 18/2160 (0.8%) for incorrect ground truth, and 58/2160 (2.7%) as ambiguous. All flagged items were subsequently corrected or removed from the final release.

##### 7.9 Dataset Statistics

Scale and Video Characteristics. HERBench comprises 26,806 questions derived from 336 unique videos. The videos feature substantial duration (avg. 395s, range 60-2100s) to ensure temporal dispersion of evidence. Sources include HD-EPIC [37], WildTrack [56], PersonPath22 [57], and movie trailers.

Question Properties. The average question length is 65.5 tokens with a vocabulary of ∼7.3k unique word types. Questions are strictly balanced across 5 multiple-choice options. The mean temporal span of evidence required per question is 101.1 seconds.

#### 8 Extended Experimental Results & Analysis

We provide a deeper quantitative analysis of the challenges posed by HERBench, expanding on the MRFS metrics and frame selection ablation.

##### 8.1 Extended MRFS Analysis

Per-Task MRFS. Table 6 details the Minimum Required Frame-Set statistics. We observe a distinct correlation between the reasoning scope of a task and its evidential requirement. Tasks requiring global chronology and the integration of multiple semantic units, specifically [TSO] (Temporal Shot Ordering, MRFS 9.05), [FAM] (False Action Memory, MRFS 6.77), and [SVA] (Scene Verification, MRFS 6.74), naturally exhibit the highest MRFS. To answer these questions correctly, a model must aggregate evidence from widely dispersed video segments or perform an exhaustive search to verify absence, effectively precluding single-frame shortcuts.

In contrast, tasks focused on local attributes or spatially constrained counting, such as [RLPC] (Region-Localized People Counting, MRFS 3.11) and [AGAR] (Attribute Recognition, MRFS 3.85), require fewer distinct frames. However, even these “lower” MRFS values demonstrate that reliance on a single frame is insufficient, confirming that HERBench successfully enforces multi-evidence integration even for localized tasks. The overall weighted mean MRFS of 5.49 validates the benchmark’s design goal: forcing models to look at multiple snapshots to derive correct answers.

MRFS vs Accuracy As illustrated in Figure 9, there is a pronounced inverse relationship between the evidential demand of a benchmark—quantified by the Mean MRFS—and the performance of state-of-the-art Video-LLMs. Benchmarks with low evidential requirements, such as TemporalBench (2.21 MRFS, 87.5%) and NeXT-QA [7] (2.61 MRFS, 76.3%), allow Qwen 2.5 VL 7B [40] to achieve relatively high accuracy, possibly due to the feasibility of single-frame shortcuts or language priors. Mid-range benchmarks—CVRR-ES, MVBench, LongVideoBench, and MMBench-Video—cluster between 2.8–4.4 MRFS with accuracies that progressively decline from 70.0% to 50.8%, tracing the fitted trend closely. At the high-demand end, MINERVA (5.14 MRFS, 37.9%), Video-MME (5.31 MRFS, 50.6%), and HERBench (5.49 MRFS, 35.9%) impose substantially greater evidential burdens, coinciding with markedly lower accuracies. Notably, AGQA (3.42 MRFS, 44.0%) falls below the trend line, suggesting that factors beyond evidential density—such as compositional question complexity—can independently depress performance. The broadened comparison across ten benchmarks strengthens the evidence for a fusion deficit in current architectures: while models may be effective at retrieving isolated frames, their capacity for compositional reasoning degrades consistently as the number of required evidence pieces grows. HERBench, positioned at the extreme of this spectrum, is specifically designed to stress-test this bottleneck by requiring the integration of non-redundant, temporally dispersed cues.

Task Total Mean MRFS [TSO] 2123 9.05 [MPDR] 2717 4.30 [ASII] 2127 6.00 [AGBI] 1226 3.81 [AGAR] 876 3.85 [AGLT] 2362 4.45 [FAM] 1962 6.77 [SVA] 4569 6.74 [FOM] 2022 5.14 [MEGL] 3061 6.33 [AC] 1623 5.26 [RLPC] 2138 3.11 Total / Weighted Mean 26,806 5.49

Table 6: Per-task MRFS statistics Computed with x = 16 using Qwen2.5-VL [40] and AKS [22].

0.9

TemporalBench

0.8

NeXT-QA

Full-contextAccuracy(k=16)

CVRR-ES

0.7

MVBench LongVideoBench

0.6

MMBench-Video

Video-MME

0.5

AGQA

0.4

HERBench (ours)

MINERVA

0.3

2 3 4 5 6

Mean Minimum Required Frame-Set (MRFS)

- Figure 9: Impact of Evidential Requirement on Model Accuracy. We plot the Mean Minimum Required FrameSet (MRFS) against Full-context Accuracy (k = 16), measured using Qwen 2.5 VL 7B [40], across ten video QA benchmarks. The dashed line indicates the fitted trend. A clear inverse correlation emerges: as the evidential burden increases (higher MRFS), model performance tends to decrease (R2 = 0.82). HERBench (green star) occupies the high-demand end of the spectrum (MRFS 5.49), highlighting the challenges current Video-LLMs face in multi-evidence integration relative to lower-requirement benchmarks such as TemporalBench and NeXT-QA.

##### 8.2 Full Frame-Selection Ablation

To more precisely disentangle the role of evidence retrieval from that of multi-evidence fusion, we perform an extensive ablation over five frame selection strategies—Uniform, Vanilla-BLIP [45], BOLT-ITS [23], AKS [22], and Oracle Frames (OF)—and evaluate their effect across all twelve HERBench tasks (Table 7). Overall, learned strategies such as

- Table 7: Frame Selection Ablation. Accuracy (%) on a random subsample of questions using InternVL3.5 [43], Qwen3-VL [2], and Ovis-2.5 [4] with Uniform, Vanilla-BLIP [45], BOLT-ITS [23], AKS [22], and GT Frames (OF) selectors. GT Frames represents the upper bound with manually curated evidence.

Model Selector AC AGAR AGBI AGLT ASII FAM FOM MEGL MPDR RLPC SVA TSO Mean

Uniform 23.0 75.0 77.0 70.0 32.0 30.0 30.0 34.0 48.0 27.0 23.0 41.0 42.7 Vanilla-BLIP 26.0 74.0 76.0 71.0 27.0 27.0 29.0 28.0 46.0 33.0 41.0 43.0 42.1 BOLT-ITS 22.0 72.0 74.0 71.0 20.0 27.0 33.0 30.0 48.0 33.0 27.0 36.0 41.1 AKS 27.0 66.0 77.0 74.0 36.0 29.0 30.0 35.0 54.0 33.0 33.0 17.0 42.7 GT Frames 24.0 81.0 81.0 79.0 20.0 50.0 39.0 27.0 52.0 32.0 37.0 53.0 47.8

InternVL3.5

Uniform 26.0 67.0 78.0 68.0 34.0 30.0 24.0 16.0 36.0 23.0 50.0 0.0 37.7 Vanilla-BLIP 27.0 71.0 76.0 66.0 26.0 24.0 23.0 30.0 37.0 19.0 56.0 0.0 37.9 BOLT-ITS 25.0 68.0 75.0 66.0 27.0 21.0 33.0 30.0 38.0 21.0 57.0 0.0 38.4 AKS 24.0 65.0 73.0 69.0 29.0 22.0 27.0 20.0 35.0 22.0 49.0 0.0 36.2 GT Frames 24.0 69.0 73.0 71.0 35.0 50.0 25.0 24.0 36.0 21.0 61.0 3.0 41.0

Qwen3-VL

Uniform 25.0 79.0 81.0 71.0 34.0 35.0 34.0 35.0 38.0 21.0 65.0 0.0 43.1 Vanilla-BLIP 32.0 77.0 83.0 69.0 17.0 29.0 33.0 37.0 44.0 19.0 58.0 0.0 41.6 BOLT-ITS 35.0 78.0 82.0 70.0 17.0 28.0 33.0 38.0 46.0 18.0 60.0 0.0 42.1 AKS 25.0 76.0 80.0 74.0 31.0 39.0 39.0 30.0 49.0 17.0 51.0 0.0 42.6 GT Frames 30.0 85.0 84.0 80.0 23.0 60.0 39.0 40.0 41.0 21.0 68.0 4.0 47.9

Ovis-2.5

BOLT-ITS and AKS provide moderate gains over Uniform sampling, reflecting their ability to prioritize query-relevant frames while maintaining broader temporal coverage. However, their improvements are uneven across tasks: both methods show the largest benefits in sparse-evidence settings such as [TSO] and [FAM], where the critical evidence may appear only briefly within long videos. The oracle-based setting establishes an upper bound by supplying the manually curated evidence frames used during dataset construction. As shown in the rightmost column of Table 7, all three representative models experience non-trivial but still limited performance improvements in the OF regime (typically +3-6 absolute accuracy points relative to the best learned selector).

Importantly, the OF results highlight two key phenomena. First, even perfect access to the relevant frames does not resolve the majority of model failures: fusion-bound tasks such as [AC], [RLPC], and [MEGL] remain bottlenecks with accuracies barely above chance, indicating that retrieval is not the sole limiting factor. Second, improvements under OF are disproportionately large for temporally global tasks such as [TSO] and [SVA], where correct reasoning requires coordinating multiple distant, non-overlapping visual clues. Here retrieval quality is a dominant factor, and learned selectors struggle to consistently surface all required frames. However, the inability of models to capitalize fully on oracle-quality evidence emphasizes that multi-frame integration itself remains a major unresolved challenge. Taken together, these results reinforce a two-stage deficit: (i) an evidence retrieval bottleneck, where existing selectors fail to reliably surface all critical cues, and (ii) a more fundamental fusion bottleneck, where models fail to combine available cues even when retrieval uncertainty is eliminated. HERBench’s high evidential density and stringent cue separation make both deficits sharply visible, underscoring the need for future MLLMs to improve not only frame selection but also the downstream mechanisms for multi-cue aggregation.

##### 8.3 MRFS robustness across backbones and frame selectors

To assess whether MRFS-based benchmark comparison is sensitive to the choice of backbone or selector, we evaluate MRFS under a range of configurations: four backbone models (Qwen2.5-VL [40], Gemini 2.5 Flash [29], LLaVA-OV-

- 1.5 [3], and GPT-4o [30]) with AKS [22], and three selectors (AKS, BOLT [23], and T⋆ [58]) with Qwen2.5-VL (see Tab. 8). Here, T⋆ serves as a non-CLIP baseline. All results are computed on 50% stratified random samples from each benchmark. Across all tested settings, the benchmark ordering remains stable, with HERBench consistently yielding the highest MRFS, followed by LongVideoBench, MVBench, and NExT-QA. These results support MRFS as a robust measure of dataset-level evidential requirement.

- Table 8: MRFS robustness across models and keyframe selectors. Benchmarks: NExT-QA [7], MVBench [1], LongVideoBench [11], and HERBench.

Benchmark Model on AKS (MRFS / Acc.) Selector on Qwen2.5-VL (MRFS / Acc.) Qwen2.5-VL Gemini2.5F LLaVA-OV-1.5 GPT-4o AKS BOLT T∗

NExT-QA 2.61 / 65.79 3.81 / 76.29 3.45 / 68.77 3.85 / 70.81 2.61 / 65.79 3.64 / 75.33 2.78 / 64.56 MVBench 3.52 / 56.71 3.92 / 57.41 3.69 / 54.24 4.09 / 54.42 3.52 / 56.71 3.73 / 56.28 3.53 / 55.53 LongVideoBench 4.07 / 41.38 4.57 / 64.59 4.50 / 61.49 4.92 / 59.01 4.07 / 41.38 4.89 / 49.14 4.11 / 48.59 HERBench 5.49 / 35.91 5.68 / 38.74 5.67 / 36.50 5.81 / 38.65 5.49 / 35.91 5.72 / 42.08 5.43 / 43.41

- 9 Illustrative Examples for All Tasks

This section provides qualitative examples for all twelve HERBench tasks, each figure displays one representative structured question for the corresponding task. However, each task in HERBench contains many distinct question structures and evidential templates, and the examples below illustrate only a single instance of the broader variability present in the dataset.

Temporal Reasoning & Chronology. Figure 10 presents an example of the Temporal Shot Ordering ([TSO]) task, which requires reconstructing the chronological order of four non-overlapping shots. Figure 11 shows the Multi-Person Duration Reasoning ([MPDR]) task, where models must compare visible-time intervals across multiple individuals. Figure 12 illustrates the Action Sequence Integrity & Identification ([ASII]) task, requiring identification of the correct sequence among plausible permutations of narrated events.

Referring & Tracking. Figure 13 shows the Appearance-Grounded Behavior Interactions ([AGBI]) task, where models must track a target described only by appearance and determine who interacts with them. Figure 14 provides an example of the Appearance-Grounded Attribute Recognition ([AGAR]) task, requiring attribute extraction anchored to the tracked target. Figure 15 illustrates the Appearance-Grounded Localization Trajectory ([AGLT]) task, where the model must infer how the target enters or exits the scene.

Global Consistency & Verification. Figure 16 presents the False Action Memory ([FAM]) task, requiring verification of which plausible action did not occur in the video. Figure 17 shows the Scene Verification & Arrangement ([SVA]) task, combining faithful and perturbed shot descriptions to assess fine-grained scene-level verification and ordering. Figure 18 depicts the False Object Memory ([FOM]) task, requiring identification of a plausible but absent object interaction.

Multi-Entity Aggregation & Numeracy. Figure 19 provides an example of the Multi-Entities Grounding & Localization ([MEGL]) task, where models must verify which appearance-described individuals actually appear in the video. Figure 20 illustrates the Action Counting ([AC]) task, requiring enumeration of all instances of a specified action–object pair across the entire video. Finally, Figure 21 shows the Region-Localized People Counting ([RLPC]) task, where the model must count unique individuals entering through specific spatial regions.

[Figure 106]

###### Figure 10

[Figure 107]

###### Figure 11

[Figure 108]

###### Figure 12

[Figure 109]

###### Figure 13

[Figure 110]

###### Figure 14

[Figure 111]

###### Figure 15

[Figure 112]

###### Figure 16

[Figure 113]

###### Figure 17

[Figure 114]

###### Figure 18

[Figure 115]

###### Figure 19

[Figure 116]

###### Figure 20

[Figure 117]

###### Figure 21

