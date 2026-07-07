# arXiv:2511.04670v1[cs.CV]6Nov2025

### Cambrian-S: Towards Spatial Supersensing in Video

Shusheng Yang1∗ Jihan Yang1∗ Pinzhi Huang1† Ellis Brown1† Zihao Yang1 Yue Yu1 Shengbang Tong1 Zihan Zheng1 Yifan Xu1 Muhan Wang1 Daohan Lu1 Rob Fergus1 Yann LeCun1 Li Fei-Fei2 Saining Xie1

1 New York University 2 Stanford University

#### Abstract

We argue that progress in true multimodal intelligence calls for a shift from reactive, taskdriven systems and brute-force long context towards a broader paradigm of supersensing. We frame spatial supersensing as four stages beyond linguistic-only understanding: semantic perception (naming what is seen), streaming event cognition (maintaining memory across continuous experiences), implicit 3D spatial cognition (inferring the world behind pixels), and predictive world modeling (creating internal models that filter and organize information). Current benchmarks largely test only the early stages, offering narrow coverage of spatial cognition and rarely challenging models in ways that require true world modeling. To drive progress in spatial supersensing, we present VSI-SUPER, a two-part benchmark: VSR (longhorizon visual spatial recall) and VSC (continual visual spatial counting). These tasks require arbitrarily long video inputs yet are resistant to brute-force context expansion. We then test data scaling limits by curating VSI-590K and training Cambrian-S, achieving +30% absolute improvement on VSI-Bench without sacrificing general capabilities. Yet performance on VSISUPER remains limited, indicating that scale alone is insufficient for spatial supersensing. We propose predictive sensing as a path forward, presenting a proof-of-concept in which a selfsupervised next-latent-frame predictor leverages surprise (prediction error) to drive memory and event segmentation. On VSI-SUPER, this approach substantially outperforms leading proprietary baselines, showing that spatial supersensing requires models that not only see but also anticipate, select, and organize experience.

Website https://cambrian-mllm.github.io

Code https://github.com/cambrian-mllm/cambrian-s

Cambrian-S Models https://hf.co/collections/nyu-visionx/cambrian-s

VSI-590K https://hf.co/datasets/nyu-visionx/vsi-590k

VSI-SUPER https://hf.co/collections/nyu-visionx/vsi-super

*SY led the project, JY and SY contributed equally. †Core contributor.

##### Contents

- 1 Introduction 3
- 2 Benchmarking Spatial Supersensing 4

- 2.1 Deconstructing Existing Video Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.2 VSI-SUPER: Towards Benchmarking Spatial Supersensing in Multimodal LLMs . . . . . 6

- 3 Spatial Sensing Under the Current Paradigm 10

- 3.1 Base Model Training: Upgraded Cambrian-1 . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.2 Spatial Video Data Curation: VSI-590K . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.3 Post-Training Recipe for Spatial Sensing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 3.4 Cambrian-S: Spatially-Grounded MLLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.5 Empirical Results: Improved Spatial Cognition . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4 Predictive Sensing as a New Paradigm 16

- 4.1 Predictive Sensing via Latent Frame Prediction . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 4.2 Case Study I: Surprise-driven Memory Management System for VSI-SUPER Recall. . . . 17
- 4.3 Case Study II: Surprise-driven continual video segment for VSI-SUPER Count. . . . . . . 19

- 5 Related Work 21
- 6 Conclusion 22 References 23

- A Benchmark Diagnostic Test Results 33
- B VSI-SUPER Benchmark 33

- B.1 VSI-SUPER Recall . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- B.2 VSI-SUPER Count . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

- C VSI-590K Dataset 34

- C.1 Details of Question Type Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.2 Detailed QA-Pair Construction Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.3 Additional Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- C.4 Examples of VSI-590K . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- D Cambrian-S Implementation Details 36

- D.1 Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.2 Training Data Mixture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- D.3 Training Recipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- D.4 Infrastructure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- E Cambrian-S Additional Results 39

- E.1 Detailed Evaluation Setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- E.2 Detailed Performance on Image and Video Benchmarks . . . . . . . . . . . . . . . . . . . . 39
- E.3 Contributions from Image-based and Video-based Instruction Tuning . . . . . . . . . . . . 39
- E.4 On the Trade-off between Spatial Sensing and General Video Understanding . . . . . . . 41

- F Predictive Sensing 42

- F.1 Latent Frame Prediction Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . 42
- F.2 Memory Framework Design for VSI-SUPER Recall . . . . . . . . . . . . . . . . . . . . . . 42
- F.3 Agentic Framework Design for VSI-SUPER Count . . . . . . . . . . . . . . . . . . . . . . 43
- F.4 Comparisons with Existing Long-video Methods . . . . . . . . . . . . . . . . . . . . . . . . 44

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Spatial Cognition

Linguistic-Only Understanding

Semantic Perception

Streaming Event Cognition

Predictive World Modeling

Naming and describing things for user prompts

Always-on sensing for open-ended streams; memory across time; proactive answering

Seeing the world behind the video; implicit 3D

Unconscious inference; Predictive, selective, and self-updating world model

Knowledge recall; no sensory modeling

TASKDRIVEN

WORLD MODELING

VIDEO AS A MEDIUM FOR SPATIAL PREDICTIVE SENSING

[Figure 6]

[Figure 7]

TIME

- Figure 1 | From pixels to predictive mind. We look beyond linguistic-only understanding to envision multimodal intelligence that sees, remembers, and reasons as part of a continuous, lived world. It begins with semantic perception: naming and describing what is seen. Streaming event cognition goes further, enabling always-on sensing across continuous input streams, integrating memory, and supporting proactive responses. Spatial cognition captures the implicit 3D structure of video, enabling reasoning about objects, configurations, and metrics. Finally, a predictive world model emerges, one that learns passively from experience, updates through prediction and surprise, and retains information for future use. Lower illustration: Video serves as the ideal experimental domain. Models must advance from frame-level Q&A to constructing implicit world models that enable deeper spatial reasoning, scale to unbounded horizons, and achieve supersensing that rivals, and ultimately surpasses, human visual intelligence.

###### 1. Introduction

A video is not just a sequence of frames in isolation. It is a continual, high-bandwidth projection of a hidden, evolving 3D world onto pixels [46, 90]. Although multimodal large language models (MLLMs) have advanced rapidly by pairing strong image encoders with language models [1, 122, 3, 78, 124], most video extensions [137, 65, 9] remain fundamentally constrained. They still treat video as sparse frames, underrepresent spatial structure and dynamics [148], and lean heavily on textual recall [168], thus overlooking what makes the video modality uniquely powerful.

In this paper, we argue that advancing toward true multimodal intelligence requires a shift from languagecentric perception toward spatial supersensing: the capacity not only to see, but also to construct, update and predict with an implicit model of the 3D world from continual sensory experience. We do not claim to realize supersensing here; rather, we take an initial step toward it by articulating the developmental path that could lead in this direction and by demonstrating early prototypes along that path:

- 0. (Linguistic-only understanding): no sensory capabilities; reasoning confined to text and symbols. Current MLLMs have progressed beyond this stage, yet still retain traces of its bias.
- 1. Semantic perception: parsing pixels into objects, attributes, and relations. This corresponds to the strong multimodal “show and tell” capabilities present in MLLMs.
- 2. Streaming event cognition: processing live, unbounded streams while proactively interpreting and responding to ongoing events. This aligns with efforts to make MLLMs real-time assistants.
- 3. Implicit 3D spatial cognition: understanding video as projections of a 3D world. Agents must know what is present, where, how things relate, and how configurations change over time. Today’s video models remain limited here.
- 4. Predictive world modeling: the brain makes unconscious inferences [130] by predicting latent world states based on prior expectations. When these predictions are violated, surprise guides attention, memory, and learning [41, 120, 60]. However, current multimodal systems lack an internal model that anticipates future states and uses surprise to organize perception for memory and decision making.

Our paper unfolds in three parts. First (§ 2), we re-examine existing benchmarks through the lens of our supersensing hierarchy. We find that most benchmarks map to the first few stages, while some, such as VSI-Bench [148], begin to probe spatial reasoning. However, none sufficiently address the final crucial stage of predictive world modeling. To make this gap concrete and motivate a shift in approach, we introduce VSI-SUPER (VSI stands for visual-spatial intelligence), a two-part benchmark for spatial supersensing: VSI-SUPER Recall (VSR) targets long-horizon spatial observation and recall, while VSI-SUPER Count (VSC) tests continual counting across changing viewpoints and scenes. Built from arbitrarily long spatiotemporal videos, these tasks are deliberately resistant to the predominant multimodal recipe; they require perception to be selective and structured rather than indiscriminately accumulated. We show that even the best long-context commercial models struggle on VSI-SUPER.

Second (§ 3), we investigate whether spatial supersensing is simply a data problem. We curate VSI-590K, a spatially focused instruction-tuning corpus over images and videos, which we use to train Cambrian-S, a family of spatially-grounded video MLLMs. Under the current paradigm, careful data design and training push Cambrian-S to state-of-the-art spatial cognition on VSI-BENCH (>30% absolute gain) without sacrificing general capabilities. Nevertheless, Cambrian-S still falls short on VSI-SUPER, indicating that while scale lays crucial groundwork, it alone is not sufficient for spatial supersensing.

This motivates the third and final part (§ 4), where we propose predictive sensing as a first step toward a new paradigm. We present a proof-of-concept solution built upon self-supervised next-latent-frame prediction. Here, we leverage the model’s prediction error, or “surprise,” for two key functions: (1) managing memory by allocating resources to unexpected events, and (2) event segmentation, breaking unbounded streams into meaningful chunks. We demonstrate that this approach, though simple, significantly outperforms strong long-context baselines such as Gemini-2.5 on our two new tasks. Although not a final solution, this result provides compelling evidence that the path to true supersensing requires models that not only see but actively predict and learn from the world.

Our work makes the following contributions. (1) We define a hierarchy for spatial supersensing and introduce VSI-SUPER, a supersensing benchmark that reveals the limitations of the current paradigm. (2) We develop Cambrian-S, a state-of-the-art model that pushes the limits of spatial cognition. Cambrian-S serves as a powerful new baseline, and, by delimiting the boundaries of current methods on our new benchmark, paves the path for a new paradigm. (3) We propose predictive sensing as a promising new direction for MLLMs, showing that leveraging model surprise is more effective for long-horizon spatial reasoning than passive context expansion.

###### 2. Benchmarking Spatial Supersensing

To ground our pursuit of spatial supersensing, we first establish how to measure it. This section undertakes a two-part investigation into benchmarking this capability. We begin by auditing a suite of popular video MLLM benchmarks, where our analysis (Fig. 3) reveals that they overwhelmingly focus on linguistic understanding and semantic perception while neglecting the more advanced spatial and temporal reasoning required for supersensing (Sec. 2.1). To address this critical gap, we then introduce VSI-SUPER, a new benchmark specifically designed to probe these harder, continual aspects of spatial

(d) diff(Multiple Frames, Blind)

(g) diff(Multiple Frames, Chance Acc.)

(j) diff(Multiple Frames, Captions)

(a) Multiple Frames

30

30

50

10

20

40

5

20

30

10

0

20

10

0

Multiple Frames better

Above chance Below chance

5

10

Captions better

10

0

0

MME

ES

VM

LV

TM

MV

PT

VSI

HV

VSR

VSC

VSC

VSR

VM

HV

VSI

LV

TM

PT

ES

MME

MV

VSI

TM

VSR

HV

VM

PT

MV

LV

ES

MME

VM

ES

MME

VSC

PT

LV

TM

HV

MV

VSI

VSR

(e) diff(Single Frame, Blind)

(h) diff(Single Frame, Chance Acc.)

(b) Single Frame

50

25

20

20

40

|MME: VideoMME [42] ES: EgoSchema [87] VM: VideoMMMU [53] LV: LongVideoBench [140] TM: Tomato [116] MV: MVBench [71] PT: Perception Test [103] HV: HourVideo [20] VSI: VSIBench [148] VSR: VSI-SUPER Rec. VSC: VSI-SUPER Cnt.|
|---|

10

15

30

10

0

20

5

Above chance Below chance

10

10

0

0

MME

ES

VM

LV

TM

MV

PT

VSI

HV

VSR

VSC

VSR

VSC

VSI

HV

VM

LV

TM

MME

PT

ES

MV

VSI

TM

VSR

HV

VM

MME

MV

PT

LV

ES

(f) diff(Captions, Blind)

(i) diff(Captions, Chance Acc.)

(c) Captions

30

30

50

20

20

40

10

10

30

20

0

0

Captions better

Above chance Below chance

10

10

Blind better

10

0

MME

ES

VM

LV

TM

MV

PT

VSI

HV

VSR

VSC

VSR

VSC

VSI

HV

TM

LV

PT

VM

ES

MME

MV

VSR

VSI

TM

HV

MV

PT

VM

LV

MME

ES

- Figure 2 | Benchmark diagnostic results reveal varying dependence on visual input. We evaluate model under distinct input conditions: (a) multiple (32) uniformly sampled frames, (b) a single (middle) frame, and (c) frame captions, benchmarked against chance-level and blind test results (visual input ignored). Panels (a–c) show absolute accuracies; panels (d–j) show performance differences between conditions. Visual inputs are substantially more critical for VSI-Bench [148], Tomato [116], and HourVideo [20], while their impact is less pronounced for VideoMME [42], MVBench [71], and VideoMMMU [53]. VSR and VSC are new supersensing benchmarks introduced in Sec. 2.2.

intelligence in arbitrarily long streaming scenarios (Sec. 2.2). We use this benchmark to test the limits of the current MLLM paradigm throughout the rest of the paper.

###### 2.1. Deconstructing Existing Video Benchmarks

Recent advances in MLLMs have led to a surge of Video-QA benchmarks. However, a critical question remains: to what extent do existing video benchmarks truly examine visual sensing capabilities rather than simply testing language priors? Our diagnostic tests disentangle the model’s reliance on visual sensing versus linguistic priors by varying the richness of visual input and the informativeness of textual cues. Benchmarks solvable with text-only inputs (e.g., captions or a blind MLLM) are skewed towards examining linguistic understanding. In contrast, benchmark questions that can only be answered with multi-frame inputs require genuine visual sensing. We use an image-based multimodal large language model Cambrian-1 [124] for evaluation, which allows us to probe the underlying task demands without conflating them with the capabilities of video-specific architectures and post-training recipes.

We establish several experimental conditions for feeding video input to a Cambrian-1 [124] model:

- • Multiple Frames: The model processes 32 frames uniformly sampled from the video clip. This is the standard method for representing video input in the literature [65].

- • Single Frame: The model processes only the middle frame of a given video clip. This condition tests the reliance on minimal, contextually-central visual information.

- • Frame Captions: Instead of video frames, the model receives captions corresponding to the same 32 uniformly-sampled frames. This condition is designed to reveal how solvable a task is without low-level perceptual grounding. We use the Gemini-2.0-Flash API to re-caption video frames.

To contextualize the performance under these conditions, we introduce two other baselines:

- • Blind Test: The model attempts the task using solely the task’s question. All visual input is ignored, no visual captions are used. This baseline measures the model’s performance based on its pre-existing knowledge, language priors, and any potential biases in the benchmark questions.

- • Chance Acc: This represents the accuracy achievable by randomly guessing for the specific task format (e.g., multiple-choice questions), serving as a floor for performance.

We conduct a fine-grained analysis of each benchmark’s characteristics by comparing performance across these conditions and baselines. We focus on the following key comparisons (diff(A,B) = A-B):

- • diff(x, Blind), x ∈ Multiple, Single, Captions to quantify the uplift provided by different input modalities over the blind baseline;

- • diff(x, Chance), x ∈ Multiple, Single, Captions to measure performance gains over chance;

- • diff(Multiple, Captions) to understand the performance gap between the current mainstream practice and a strong language-only baseline

Results presented in Fig. 2 (a-c) demonstrate that Cambrian-1 [124], an image-based MLLM without any video post-training, can attain reasonable performance across many benchmarks, in some instances surpassing chance-level accuracy by 10-30% (see Fig. 2-g,h). This suggests that much of the knowledge these benchmarks target is accessible via standard single-image instruction-tuning pipelines. Nevertheless, on two existing datasets, VSI-Bench [148] and Tomato [116], the model’s performance falls below chance-level. For VSI-Bench, this is largely because its spatial understanding questions require true video sensing and targeted data curation and training. For Tomato, this underperformance is expected: the benchmark demands understanding of fine-grained details from higher frame-rate video, rendering the largely temporally-subsampled single-frame and 32-frame inputs inadequate.

Employing textual captions in place of visual inputs also yields notable performance improvements, surpassing chance accuracy by more than 20% on benchmarks such as EgoSchema [87], VideoMME [42], LongVideoBench [140], VideoMMMU [53], Perception Test [103], and MVBench [71] (Fig. 2-i). Similar conclusions can be drawn when comparing benchmark performance against blind test results (Fig. 2d,f). Such performance implies that these benchmarks primarily probe abilities inferable from textual summaries of video content. Interpreting the performance difference between using “multiple frames” and “frame captions” (Fig. 2-j), a significantly positive margin (in favor of multi-frame inputs) signifies a benchmark’s demand for nuanced visual sensing. Conversely, a small or negative margin (more in favor “frame captions”) suggests a more language-centric nature. Our analysis places VideoMMMU, EgoSchema, VideoMME, Perception Test, and LongVideoBench in this latter category, indicating their potential reliance on linguistic understanding rather than visual cues. A notable exception is VSC, which is so challenging for current MLLMs that all three input conditions yield near-zero performance, precluding any meaningful comparison between them.

Existing benchmarks overwhelmingly focus on linguistic understanding and semantic perception while neglecting the more advanced spatial and temporal reasoning required for supersensing.

We hope to emphasize the inherent challenges in benchmarking and the impracticality of creating a single, all-encompassing benchmark to evaluate every capability. For example, reliance on language priors should not be viewed merely as a drawback, as access to rich world knowledge and its effective retrieval is undoubtedly beneficial in many scenarios. We argue that video benchmarks should not be treated as measuring a single, uniform notion of “video understanding.” Instead, their design and evaluation should be grounded in the specific capabilities they aim to assess. The preceding analyses are therefore intended to guide the development of tasks that more effectively drive progress towards spatial supersensing, which will be the central focus of the rest of the paper.

###### 2.2. VSI-SUPER: Towards Benchmarking Spatial Supersensing in Multimodal LLMs

Referring to Fig. 1, spatial supersensing requires MLLMs to have four key capabilities: semantic perception, streaming event cognition, implicit 3D spatial cognition, and predictive world modeling. However, as outlined by

[Figure 8]

[Figure 9]

VideoMME

VSI-Bench

[Figure 10]

[Figure 11]

Which feature of the astronaut's equipment indicates they can move independently in space?

If I am standing by the refrigerator and facing the washer, is the stove to my left, right, or back?

Why are the objects flying?

How many chair(s) are in this room?

- Figure 3 | Illustrations of how spatial sensing is conceptualized in current video benchmarks. The left panel features examples from the “spatial reasoning” subcategory of VideoMME [42], including a question regarding gravity from Shutter Authority’s “What if the Moon Crashed into the Earth?” and a question regarding astronaut gear from NASA’s “Astronaut Bruce McCandless II Floats Free in Space.” In contrast, the right panel shows samples from VSI-Bench [148], which highlight visualspatial reasoning tasks such as object counting, identifying relative directions, route planning, and more.

our analysis in Fig. 2, most existing video QA benchmarks mainly evaluate the linguistic understanding and semantic perception aspects, which are more reactive and driven by specific tasks [42, 87, 53]. While recent research has begun to address streaming event cognition through continual sensing, memory architectures, and proactive answering [24, 104, 97, 139, 119, 159], this capability is often engineered at test time rather than being a native model skill. Furthermore, although spatial reasoning occasionally appears as a category in existing benchmarks, these tasks seldom reach the level of true spatial cognition, and are far from probing the world-modeling capacity that defines supersensing (Fig. 3). Although VSI-Bench [148] takes an initial step toward examining spatial cognition, its videos remain short-form and single-scene, and it neither formalizes the problem nor evaluates the essential capability of predictive modeling of the world.

To illuminate the gap between current MLLMs and spatial supersensing, we introduce VSI-SUPER, a two-part benchmark for continual spatial sensing. The tasks are intuitive and generally easy for humans, where one simply watches and keeps track of what happens, but they remain surprisingly challenging for machines. They demand selective filtering and structured accumulation of visual information across unbounded spatial videos to maintain coherent understanding and answer questions. Importantly, they are resistant to brute-force context expansion, exposing the need for true spatial reasoning. We detail the two components below.

Frame Editing

[Figure 12]

[Figure 13]

Which of the following correctly represents the order in which the Teddy Bear appeared in the video? A. Toilet, Bathtub, Sink, Floor B. Bathtub, Toilet, Sink, Floor C. Toilet, Sink, Floor, Bathtub D. Floor, Toilet, Bathtub, Sink

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

[Figure 31]

Random Video Concatenating

- Figure 4 | Illustration of the VSR benchmark’s construction process and format. We use generative models to edit videos by inserting surprising or out-of-place objects into the space. The core task then challenges models to recall the spatial placements of these objects in the correct order of their appearance across arbitrarily long videos.

VSI-SUPER Recall: Long-horizon spatial observation and recall. The VSR benchmark requires MLLMs to observe long-horizon spatiotemporal videos, and sequentially recall the locations of an unusual object. As shown in Fig. 4, to construct this benchmark, human annotators use an image editing

model (i.e., Gemini [30]) to insert surprising or out-of-place objects (e.g., a Teddy Bear) into four distinct frames (and spatial location) of a video capturing a walkthrough of an indoor environment [33, 153, 12]. This edited video is then concatenated with other similar room-tour videos to create an arbitrarily long and continuous visual stream. This task parallels the needle-in-a-haystack (NIAH) test commonly used in the language domain to stress test the long-context capabilities of LLMs [79]. Similar NIAH setups have also been proposed for long-video evaluation [162, 138, 54]. However, unlike benchmarks that insert unrelated text segments or frames, VSR preserves the realism of the “needle” through in-frame editing. It further extends the challenge by requiring sequential recall, effectively a multi-hop reasoning task, and remains arbitrarily scalable in video length. To thoroughly evaluate model performance across different time scales, the benchmark is provided in five durations: 10, 30, 60, 120, and 240 minutes. Further details on the VSR benchmark construction are provided in Sec. B.

VSI-SUPER Count: Continual counting under changing viewpoints and scenes. Here we test the capacity of MLLMs to continuously accumulate information in long-form spatial videos. To build VSC, we concatenate multiple room-tour video clips from VSI-Bench [148] and task models with counting the total number of target objects across all rooms (see Fig. 5). This setting is challenging because the model must handle viewpoint shifts, repeat sightings, and scene transitions, all while maintaining a consistent cumulative count. For humans, counting is an intuitive and generalizable process. Once the concept of “one” is understood, extending it to larger quantities is natural. In contrast, as we later demonstrate, current MLLMs lack true spatial cognition and depend excessively on learned statistical patterns.

In addition to standard evaluations (i.e., ask question at the end of video), we query the model at multiple timestamps to assess its performance in streaming settings, where the correct answer in VSC evolves dynamically over time. To examine long-term consistency, VSC includes four video durations: 10, 30, 60, and 120 minutes. For this quantitative task, we report results using the mean relative accuracy (MRA) metric, consistent with the VSI-Bench evaluation protocol [148].

Num. of Chairs: 3 1 16

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

Streaming Questions:

Q: How many different chair(s) are there in this video? A: 2 A: 3 A: 3 A: 4 A: 20

- Figure 5 | Overview of the VSC benchmark. The benchmark evaluates counting capabilities on long-horizon, multiroom videos composed of concatenated scenes. Queries are posed at various time points to simulate a streaming question-answering setting.

State-of-the-art models struggle on VSI-SUPER. To test whether VSI-SUPER poses a real challenge for frontier MLLMs, we evaluate the latest Gemini-2.5-Flash [122]. As shown in Tab. 1, the model reaches its context limit when handling two-hour videos, despite a context length of 1,048,576 tokens. This highlights the open-ended nature of video understanding, where continuous streams effectively require an “infinite-in, infinite-out” context and can grow arbitrarily long, suggesting that simply scaling up tokens, context length, or model size may not suffice. Though synthetic, our benchmark reflects a real challenge in spatial supersensing: humans effortlessly integrate and retain information from ongoing sensory experiences that unfold over hours or years, yet current models lack comparable mechanisms for sustained perception and memory. Gemini-2.5-Flash demonstrates strong performance on semantic-perception and linguisticunderstanding-focused video benchmarks such as VideoMME [42] and VideoMMMU [53], achieving around 80% accuracy. However, even for 60-minute videos in VSI-SUPER that fall well within its context window, performance on VSR and VSC remains limited—only 41.5 and 10.9, respectively. As shown in Fig. 6, the model’s predicted object counts fail to scale with video length or the true number of objects, instead saturating at a small constant value, suggesting a lack of generalization in counting ability and a reliance on training distribution priors.

|Model<br><br>|VideoMME[42] VideoMMMU[53] VSI-Bench[148]<br><br>VSR VSC 60 min 120 min 60 min 120 min|
|---|---|
|Gemini-2.5-Flash|81.5 79.2 45.7 41.5 Out of Ctx. 10.9 Out of Ctx.<br><br>|

- Table 1 | Gemini-2.5-Flash results. As a state-of-the-art video understanding model with long-context capabilities, Gemini demonstrates strong performance on general video benchmarks but shows clear limitations towards spatial supersensing.

Gemini-2.5-Flash

Line of Perfect Prediction

| |
|---|

10 mins

30 mins

60 mins

250

150

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

75

200

100

50

Prediction

Prediction

Prediction

150

100

50

25

50

25 50 75 Ground Truth

50 100 150 Ground Truth

50 100 150 200 250 Ground Truth

- Figure 6 | Visualization of Gemini-2.5-Flash’s predictions v.s. ground truth on VSC. The model’s predicted object counts saturate at small constant values and fail to scale with video length or true object counts, indicating limited generalization in counting and reliance on training distribution priors.

How VSI-SUPER challenges the current paradigm. Although the task setup is simple, the challenge posed by VSI-SUPER goes beyond just spatial reasoning and reveals fundamental limitations of the current MLLM paradigm.

VSI-SUPER tasks challenge the belief that scaling alone guarantees progress.

By allowing arbitrarily long video inputs that emulate the dynamics of streaming cognition, VSISUPER is intentionally constructed to exceed any fixed context window. This design suggests that frame-by-frame tokenization and processing are unlikely to be computationally viable as a long-term solution. Humans address such problems efficiently and adaptively by selectively attending to and retaining only a small fraction of sensory input1, often unconsciously [40, 130]. This predictive and selective mechanism, core to human cognition, remains absent in current MLLMs but is fundamental to a predictive world model.

VSI-SUPER tasks demand generalization to new temporal and spatial scales at test time.

For example, VSC requires counting in arbitrarily long videos, similar to how humans, who understand the concept of counting, can extend it to any number. The key is not maintaining an extremely long context window, humans do not retain every visual detail from extended visual experiences, but rather learning the process of counting itself. Predictive sensing facilitates this by segmenting continuous visual streams into coherent events, using moments of “surprise” to impose temporal structure. This segmentation acts as a divide-and-conquer mechanism that allows the model to decide when to start, continue, or reset behaviors in dynamically changing scenes.

Together, these challenges, which span computational efficiency, generalization, and cognitive mechanisms such as unconscious inference and predictive sensing, call for a paradigm shift. Rather than relying solely on scaling data, parameters, or context length, future models should learn internal world models capable of perceiving and predicting within an endlessly unfolding visual world across space and time.

1Each eye’s 6 million cone photoreceptors can send about 1.6 Gbits/s, yet the brain uses only 10 bits/s to guide behavior [62, 163].

To further motivate this paradigm shift, the next section investigates the extent to which progress remains possible within the current paradigm through improved engineering and targeted data curation. We assess whether the existing MLLM framework can be adapted to address the challenges posed by VSI-SUPER. These efforts, while operating within the limits of the present framework, are indispensable for building the data and empirical foundations of the next generation of spatial supersensing models.

###### 3. Spatial Sensing Under the Current Paradigm

As demonstrated in the previous section, Gemini-2.5-Flash exhibits subpar performance on spatial sensing tasks (see Tab. 1). This observation raises a key question: Is limited spatial sensing simply a data issue? It is a valid question to ask, as current video MLLMs do not explicitly prioritize spatial-focused videos during training, and it remains s whether existing pre-training and post-training designs are well-suited for our target tasks. We begin by enhancing Cambrian-1 [124] with a series of architectural and training improvements to establish a stronger image MLLM as our base model (Sec. 3.1). We proceed to construct a large-scale, spatial-focused instruction-tuning dataset, VSI-590K (Sec. 3.2). The dataset is curated from diverse sources and carefully annotated. As such data does not currently exist publicly, VSI-590K is intended to provide a strong data foundation for spatial sensing. Finally, with a refined training recipe (Sec. 3.3), we introduce the spatially-grounded Cambrian-S model family (Sec. 3.4).

The Cambrian-S model family demonstrates strong performance on established spatial reasoning benchmarks such as VSI-Bench [148] and offers valuable insights into base model design, data curation, and training strategies for spatial supersensing. However, despite these advances, this approach does not directly address the continual sensing challenges of VSI-SUPER (Sec. 3.5); instead, it provides a crucial foundation that motivates the new paradigm introduced in (Sec. 4).

###### 3.1. Base Model Training: Upgraded Cambrian-1

We begin by developing an image-based MLLM base model, as robust semantic perception forms the foundation for higher-level spatial cognition. We follow the two-stage training pipeline of Cambrian-1 [124]. We upgrade the visual encoder to SigLIP2-SO400m [128] and the language model to the instruction-tuned Qwen2.5 [145]. For the vision-language connector, we adopt a simple two-layer MLP primarily for its computational efficiency. Other training components from Cambrian-1, including hyperparameters and the data recipe, remain unchanged. Full implementation details are provided in Sec. D.

###### 3.2. Spatial Video Data Curation: VSI-590K

Extracted Semantics

Question Templates

|[Figure 45]<br><br>[Figure 46]<br><br>Front View<br><br>Top View|
|---|

[Figure 47]

| |Scene Name Object Counts<br><br>Object Boxes<br><br>Room Size<br><br>...|
|---|---|
| | |

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

|§How far apart are the {obj_1} and<br><br>the {obj_2} measured in {unit}? §...|
|---|

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

|§How far apart are the trash bin and the backpack, measured in centimeters?<br><br>§...|
|---|

Selected Frames

3D Lifting 3D Semantics

Scene Description

3D Annotation Videos & images

Unannotated Real Videos

SFT Data Generation

Annotated Videos (Sim and Real)

- Figure 7 | VSI-590K data curation pipeline. We collect data from 3D-annotated real and simulated video sources, as well as from pseudo-annotated frames extracted from web videos. We then use diverse templates to automatically generate question–answer pairs for instruction tuning.

It is well recognized that data quality and diversity play a critical role in the training of MLLMs [124, 93]. We hypothesize that the performance gap on VSI-Bench [148] comes mainly from the lack of highquality, spatially grounded data in current instruction-tuning datasets [161, 32]. To fill this gap, we build VSI-590K, a large-scale instruction-tuning dataset designed to improve visual-spatial understanding.

Data curation and processing. We construct VSI-590K from a diverse span of data sources and types (i.e., simulated and real). See Tab. 2 for the data sources and for dataset statistics on the number of videos,

- Table 2 | Data statistics for VSI-590K. We collect data from 10 sources with different video types and annotations to improve diversity.

Dataset # Videos # Images # QA Pairs Annotated Real Videos

S3DIS [4] 199 - 5,187 Aria Digital Twin [102] 183 - 60,207 ScanNet [33] 1,201 - 92,145 ScanNet++ V2 [153] 856 - 138,701 ARKitScenes [12] 2,899 - 57,816 Simulated Data

ProcTHOR [36] 625 - 20,092 Hypersim [113] - 5,113 176,774 Unannotated Real Videos

YouTube Room Tour - 20,100 20,100 Open X-Embodiment [100] - 14,801 14,801 AgiBot-World [16] - 4,844 4,844

Total 5,963 44,858 590,667

images, and QA pairs from each dataset. We find that this yields a dataset substantially more robust than one of comparable size derived from a single source. Below, we detail the data processing procedure.

- • Annotated real videos. Multimodal visual–spatial reasoning relies on a solid understanding of 3D geometry and spatial relationships. Following VSI-Bench, we repurpose the training splits of existing indoor scan and first-person video datasets that provide 3D instance-level annotations, including S3DIS [4], ScanNet [33], ScanNet++ V2 [153], ARKitScenes [12], and ADT [102]. For each dataset, annotations are consolidated into a meta-information file capturing scene-level attributes such as object counts by category, object bounding boxes, room dimensions, and related metadata. Question templates are then automatically instantiated to generate corresponding questions.
- • Simulated data. Due to the limited availability of 3D-annotated data, constructing a large-scale and diverse 3D-annotated SFT dataset solely from real annotated videos is challenging. Following SIMSV [13], we utilize embodied simulators to procedurally generate spatially grounded video trajectories and QA pairs, rendering 625 video traversals within ProcTHOR [36] scenes featuring diverse layouts, object configurations, and visual appearances. We apply the same methodology to Hypersim [113], sampling 5,113 images from 461 indoor scenes. Using instance-level bounding boxes, we generate question-answer pairs consistent with our annotated real-video setup.
- • Unannotated real videos. Although web-sourced videos lack explicit annotations, they offer rich diversity in indoor environment types, geographical regions, and spatial layouts. We collected approximately 19K room tour videos from YouTube and additionally incorporated videos from robotic learning datasets, including Open-X-Embodiment [100] and AgiBot-World [16]. Since these videos do not contain the 3D annotations required for constructing spatial instruction-tuning data, we develop a pseudo-annotation pipeline. As illustrated in Fig. 7, we subsample and filter video frames, applying object detection [80], segmentation model [109], and 3D reconstruction model [133] to generate pseudo-annotated images following the approach of SpatialVLM [21]. We choose to generate annotations at the image level rather than across full videos, as full-video pseudo-annotations derived from recognition and reconstruction models tend to be too noisy for training.

Question type definition and template augmentation. We define 12 question types within a spatiotemporal taxonomy to construct a comprehensive and diverse set of questions for instruction tuning. We define five main question types—size, direction, count, distance, and appearance orderbroadly categorized as measuring configuration, measurement, or spatiotemporal capabilities following [148]. Except for the appearance order type, each question category includes both relative and absolute variants, reflecting the importance of these complementary forms of reasoning in visual–spatial understanding [148]. For example, for size, we ask for both size comparison between two objects (relative) and the metric dimensions of an object (absolute). To enhance diversity, we vary the perspective used in formulating direction and distance questions. For instance, a distance question may ask which of two objects is closer to the camera or which object is closer to a third reference

- Table 3 | Contributions of Different Data Sources in the VSI-590K Mixture. This table illustrates the impact of different data sources on VSI-Bench performance. The combined dataset, VSI-590K Full Mix, achieves the best overall results. Among individual sources, annotated real video datasets contribute the most significant improvements, followed by simulated videos, and then pseudo-annotated images.

|VSI Data Mixture|Image<br><br>MMVP<br><br>3DSR<br><br>CV-B<br><br>|VSI-Bench (Video)<br><br>Avg<br><br>ObjCt<br><br>AbsDst<br><br>ObjSz<br><br>RmSz<br><br>RelDst<br><br>RelDir<br><br>RtePln<br><br>ApOrd|
|---|---|---|
|Baseline|52.7 54.5 73.5<br><br>|28.5 18.1 20.0 36.0 22.2 42.9 31.3 24.6 33.0<br><br>|
|Real Videos<br><br>+ S3DIS<br><br>+ ADT<br><br>+ ARKitScenes<br><br>+ ScanNet<br><br>+ ScanNet++ V2<br><br>|54.0 54.9 75.3 50.6 56.5 77.5 50.0 56.7 77.3 54.7 57.7 77.5 52.7 57.3 77.5|41.6 63.8 21.0 44.9 37.0 43.8 47.4 34.0 41.1 41.0 51.0 29.8 52.5 40.2 42.3 38.8 34.0 39.8 51.0 70.2 32.7 64.5 60.0 55.1 45.2 37.1 43.5 56.3 70.9 37.9 67.5 59.3 57.0 46.7 35.1 76.1 56.3 72.5 40.7 65.7 56.9 59.7 47.1 31.4 76.2<br><br>|
|Simulated Videos<br><br>+ ProcThor<br><br>+ HyperSim|53.3 55.7 74.9 52.0 56.0 79.7<br><br>|36.4 21.0 29.7 49.3 3.8 52.3 45.7 30.4 58.7 45.6 67.8 32.0 59.3 36.4 53.2 47.0 32.5 36.6|
|Pseudo-Annotated Images<br><br>+ YTB RoomTour + OXE & AGIBot<br><br>|55.3 52.6 75.0<br>56.0 54.4 72.5<br>|32.5 43.4 25.8 24.2 27.3 38.7 31.4 28.4 40.9 30.6 40.3 23.1 27.9 26.6 38.0 22.8 32.0 33.8<br><br>|
|Full Mix|54.7 54.0 77.9<br><br>|63.2 73.5 49.4 71.4 70.1 66.9 61.5 36.6 76.6<br><br>|

AbsDst

ApOrd

MMVP

RelDst

RtePln

RelDir

ObjCt

ObjSz

RmSz

3DSR

CV-B

Avg

object. We also diversify the dataset through variations in question wording and in measurement units (e.g., meters versus feet). Additional details of the dataset are provided in Sec. C.

VSI-590K data source ablation. To evaluate the effectiveness of our proposed VSI-590K dataset, we perform an ablation study by finetuning the improved Cambrian-1 MLLM described in Sec. 3.1 with part of the video instruction tuning samples from LLaVA-Video-178K [161]. This model serves as the baseline in Tab. 3. The contribution of each data source is evaluated by fine-tuning the model on individual datasets as well as their combination. The VSI-590K Full Mix achieves the highest overall performance on video spatial reasoning tasks, outperforming both the baseline and all single-source counterparts. All data sources contribute positively after fine-tuning, though their effectiveness varies.

Data effectiveness ranks as: annotated real videos > simulated data > pseudo-annotated images.

This indicates that videos are inherently more informative than static images for spatial reasoning, as training exclusively on video data yields superior performance on both video- and image-based spatial reasoning benchmarks. These findings support the intuition that the temporal continuity and multi-view diversity of videos are key to developing robust spatial representations.

###### 3.3. Post-Training Recipe for Spatial Sensing

We further analyze and ablate our video instruction-tuning pipeline, focusing on the roles of the pretrained base video model and the instruction-tuning dataset mixture. As shown in Tab. 4, we begin with four base models that represent a progressive increase in video understanding capability:

- • A1 is trained only with image-text alignment on Cambrian-1 alignment data. The language model is identical to base QwenLM as it is frozen during training.
- • A2 is finetuned with image instruction tuning on top of A1, essentially our improved Cambrian-1.
- • A3 is initialized from A2 and finetuned on 429K video instruction tuning data.
- • A4 is initialized from A2 and finetuned on 3M video instruction tuning data.

We then finetune these models using two different data recipes: (1) VSI-590K only, and (2) VSI-590K mixed with a similar amount of general video instruction tuning data.

- Table 4 | Post-training exploration for spatial sensing. We examine four base models with progressively increasing exposure to visual data, from image-only training to extensive video training, and analyze their distinct trends during spatial sensing tuning under two different data recipes. A1: only the connector is trained for image–language alignment; A2: A1 w/. Cambrian-7M image instruction-tuning data; A3: A2 further finetuned on 429K video instruction-tuning samples; A4: A2 further finetuned on 3M video instruction-tuning samples. From A1 to A4, the models show a monotonic improvement in video understanding ability. I-IT and V-IT denote instruction finetuning on image and video data, respectively. Finally, we show that stronger base models yield better SFT performance on spatial sensing tasks.

Model VSI-Bench VideoMME EgoSchema Perception Test Different Base Models

- A1 (w/o. I-IT, i.e. QwenLM) 21.4 44.2 42.9 44.5
- A2 (A1 + I-IT, i.e. Cambrian-1) 25.8 53.7 48.1 55.4
- A3 (A2 + V-IT, 429K data) 28.9 61.2 50.3 66.3
- A4 (A2 + V-IT, 3M data) 35.7 62.6 77.0 70.9 SFT w/. VSI-590K

- from A1 57.2 40.3 38.7 52.3
- from A2 66.8 46.7 47.2 52.3
- from A3 68.8 52.3 48.4 55.8
- from A4 69.2 54.1 55.2 59.2

SFT w/. VSI-590K & general V-IT data mixture

- from A1 61.3 60.5 52.8 65.0
- from A2 63.2 62.6 52.9 65.6
- from A3 64.0 61.0 54.9 66.8
- from A4 65.1 61.9 77.3 71.2

A stronger base model with greater exposure to general video data leads to improved spatial sensing after SFT.

As shown in Tab. 4, SFT with a stronger base model, one that performs well on general video benchmarks such as VideoMME [42] and EgoSchema [87], leads to enhanced spatial understanding. This highlights the importance of broad exposure to general video data during base model training.

Mixing general video data prevents the generalization loss caused by in-domain SFT.

Furthermore, while in-domain SFT solely on VSI-590K achieves the highest performance on VSIBench, it results in a noticeable decline on general video benchmarks. However, this performance drop can be effectively mitigated by training on a data mix that includes general videos.

###### 3.4. Cambrian-S: Spatially-Grounded MLLMs

Building on all the previous insights, we develop Cambrian-S, a family of spatially-grounded models with varying LLM scales: 0.5B, 1.5B, 3B, and 7B parameters. These models are built through a four-stage training pipeline specifically designed to first establish general semantic perception and then develop specialized spatial sensing skills, as illustrated in Fig. 8.

The first two stages adhere to the Cambrian-1 framework to develop strong image understanding capabilities. In stage 3, we extend the models to video by conducting general video instruction tuning on CambrianS-3M, a curated dataset composed of 3 million samples (see detailed composition in Fig. 16). This stage establishes a solid foundation for general video understanding prior to introducing specialized skills. In the final and crucial stage 4, the models are trained for spatial sensing. Here, we finetune the models on a blended corpus combining our specialized VSI-590K with a proportional subset of the general video data used in stage 3, following the setup described in Tab. 4. Complete training details are provided in Sec. D.3.

Cambrian-1 Training Cambrian-S Training

[Figure 71]

Vision-Language Alignment

Image Instruction Tuning

General Video Instruction Tuning

Spatial Video Instruction Tuning

Stage 1 Stage 3 Stage 4

Stage 2

- Figure 8 | Overall Cambrian-S training pipeline. Stages 1 and 2 enhance image understanding, stage 3 improves general video understanding, and stage 4 strengthens spatial sensing capability.

- Table 5 | Comparison of Cambrian-S with other leading MLLMs. Cambrian-S outperforms both proprietary and open-source models across a range of image and video visual–spatial benchmarks and model sizes. For video evaluation, we uniformly sample 128 frames as input. Detailed evaluation settings are provided in Sec. E.

Video Image

|Model|Base LM<br><br>|VSI-Bench<br><br>DebiasedVSI-Bench<br><br>Tomato<br><br>HourVideo<br><br>MMEVideo†<br><br>EgoSchema†<br><br>MMMUVideo†<br><br>LongVBench†<br><br>MVBench†<br><br>Percept.Test†|MMVP<br><br>3DSR<br><br>CV-Bench|
|---|---|---|---|
|Proprietary Models Claude-3.5-sonnet GPT-4o<br><br>Gemini-1.5-Pro<br><br>Gemini-2.5 Pro<br><br><br>|UNK. UNK. UNK. UNK.<br><br>|- - 27.8 - 62.9 - 65.8 - - 34.0 - 37.7 37.2 71.9 - 61.2 66.7 - 45.4 40.1 36.1 37.3 75.0 72.2 53.9 64.0 - 51.5 49.1 - - - - 83.6 67.4 - -|- 48.2 66.0 44.2 - - 51.3 - -<br><br>|
|Open-Source Models LLaVA-Video-7B LLaVA-One-Vision-7B Qwen-VL-2.5-7B InternVL2.5-8B InternVL3.5-8B Cambrian-S-7B<br><br>|Qwen2-7B<br><br>Qwen2-7B<br><br>Qwen2.5-7B<br><br>InternLM2.5-7B<br><br>Qwen3-8B Qwen2.5-7B<br><br><br>|35.6 30.7 22.5 28.6 63.3 57.3 36.1 58.2 58.6 67.9<br><br>32.4 28.5 25.5 28.3 58.2 60.1 33.9 56.4 56.7 57.1<br><br>33.5 29.6 - - 65.1 65.0 47.4 56.0 69.6 -<br><br>34.6 24.9 - - 64.2 50.6 - 60.0 72.0 56.3 49.7 - - 66.0 61.2 49.0 62.1 72.1 67.5 59.9 27.0 36.5 63.4 76.8 38.6 59.4 64.5 69.9<br><br><br>|- - 75.7<br><br>54.7 - 74.3 56.7 48.4 -<br>55.3 50.9 -<br>56.0 - 60.0 54.8 76.9<br><br><br>|
|VILA1.5-3B Qwen2.5-VL-3B Cambrian-S-3B<br><br>|Sheared-LLaMA-2.7B<br><br>Qwen2.5-3B Qwen2.5-3B<br><br>|- - - - 42.2 - - 42.9 - 49.1 26.8 22.7 - - 61.5 - - 54.2 - 66.9 57.3 49.7 25.4 36.8 60.2 73.5 25.2 52.3 60.2 65.9<br><br>|- - 39.3 - 50.0 50.9 75.2<br><br>|
|SmolVLM2-2.2B InternVL2.5-2B InternVL3.5-2B Cambrian-S-1.5B<br><br>|SmolLM2-1.7B InternLM2.5-1.8B Qwen3-1.7B Qwen2.5-1.5B<br><br>|27.0 22.3 - - - 34.1 - - 48.7 51.1<br><br>25.8 20.7 - - 51.9 47.4 - 52.0 68.8 51.5 46.1 - - 58.4 50.8 - 57.4 65.9 54.8 47.5 22.5 31.4 55.6 68.8 24.9 50.0 58.1 63.2<br><br>|- - 45.3 - 44.0 - 42.7 51.9 69.6<br><br>|
|SmolVLM2-0.5B LLaVA-One-Vision-0.5B InternVL2.5-1B InternVL3.5-1B Cambrian-S-0.5B<br><br>|SmolLM2-360M<br><br>Qwen2-0.5B<br><br>Qwen2.5-0.5B<br><br>Qwen3-0.6B Qwen2.5-0.5B<br><br><br>|26.1 23.1 - - - 20.3 - - 43.7 44.8 28.5 20.6 - - 44.0 26.8 - 45.8 45.5 49.2 22.5 17.5 - - 50.3 39.8 - 47.9 64.3 -<br><br>49.9 41.8 - - 51.0 41.5 33.0 53.0 61.0 -<br><br>50.6 42.2 23.4 27.9 44.0 62.4 15.7 44.0 51.8 56.0<br><br><br>|- - 28.7 - 55.5<br><br>33.3 - 32.0 - 26.0 48.5 59.8<br><br>|

DebiasedVSI-Bench

LongVBench†

Percept.Test†

MMMUVideo†

EgoSchema†

MMEVideo†

MVBench†

HourVideo

VSI-Bench

CV-Bench

Tomato

MMVP

3DSR

###### 3.5. Empirical Results: Improved Spatial Cognition

We next evaluate the Cambrian-S multimodal models to assess both the strengths and limitations of our data-driven approach.

Improved spatial cognition. As shown in Tab. 5, our models achieve state-of-the-art performance in visual-spatial understanding in video. Cambrian-S-7B achieves 67.5% on VSI-Bench, significantly outperforming all open-source models and surpassing the proprietary Gemini-2.5-Pro by over 16 absolute points. Since our work in this section can be viewed as a data scaling effort, a natural question is: are the performance improvements simply due to broader data coverage (including more diverse visual configurations and question–answer pairs), or has the model actually developed stronger spatial cognition? First, we emphasize that there is no data overlap between VSI-590K and the benchmark datasets. Although some datasets originate from the same sources (e.g.from ScanNet), we only use the training split, while the benchmarks use validation and test splits. Moreover, we observe clear signs of generalization in spatial reasoning. For example, in the challenging “Route Planning” subtask, whose question types are absent from VSI-590K because of the high annotation cost, Cambrian-S-7B still performs strongly, showing pronounced scaling behavior with increasing model size too (see Tab. 6).

Furthermore, our training approach proves highly effective even with smaller model sizes: our

- Table 6 | VSI-Bench sub-task breakdown. Best results are bolded. Notably, even without any route planning data in training, Cambrian-S-7B outperforms Gemini-1.5-Pro on this task.

|Methods<br><br>|Avg.|Obj. Count Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir.<br><br>Route Plan Appr. Order<br><br>Numerical Answer Multiple-Choice Answer<br><br>|
|---|---|---|
|Statistics Chance Level (Random) Chance Level (Frequency)|34.0<br><br>|- - - - 25.0 36.1 28.3 25.0 62.1 32.0 29.9 33.1 25.1 47.9 28.4 25.2<br><br>|
|Proprietary Models (API) GPT-4o Gemini-1.5 Flash<br><br>Gemini-1.5 Pro<br><br>Gemini-2.5 Pro<br><br><br>|34.0 42.1 45.4 51.5<br><br>|46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 49.8 30.8 53.5 54.4 37.7 41.0 31.5 37.8 56.2 30.9 64.1 43.6 51.3 46.3 36.0 34.6 43.8 34.9 64.3 42.8 61.1 47.8 45.9 71.3|
|Open-source Models Cambrian-S-7B Cambrian-S-3B Cambrian-S-1.5B Cambrian-S-0.5B<br><br>|67.5 57.3 54.8 50.6|73.2 50.5 74.9 72.2 71.1 76.2 41.8 80.1 70.7 40.6 68.0 46.3 64.8 61.9 27.3 78.8 68.4 40.0 61.5 50.1 62.4 48.9 29.9 77.5 67.9 35.4 52.2 52.5 52.3 46.5 25.8 72.2<br><br>|

smallest 0.5B model achieves performance comparable to Gemini-1.5 Pro on VSI-Bench. Importantly, this emphasis on spatial reasoning does not come at the expense of general capabilities: Cambrian-S continues to deliver competitive results on standard video benchmarks such as Perception Test [103] and EgoSchema [87] (see Tab. 14 for complete results).

Cambrian-S achieves state-of-the-art spatial sensing performance with robust generalization to unseen spatial question types, while staying competitive in general video understanding.

Robust spatial reasoning on VSI-Bench-Debiased. A recent study [14] reveals that models can rely on strong language priors for spatial reasoning tasks. For instance, when asked to estimate a table’s length, a model might leverage natural world knowledge about typical table sizes (e.g., 120–180 cm) rather than analyzing the visual evidence. To investigate whether Cambrian-S learns to reason visually, we evaluate it on VSI-Bench-Debiased [14], a benchmark specifically designed to eliminate language shortcuts through debiasing. As shown in Tab. 5, although performance decreases by about 8% compared to standard VSI-Bench, our models still outperform proprietary counterparts, demonstrating robust visual-spatial reasoning capabilities and confirming that our training extends beyond language-based learning.

Results on VSI-Super: limitations in continual spatial sensing. Despite its strong performance on spatial reasoning tasks in short, pre-segmented videos from VSI-Bench, Cambrian-S isn’t well-equipped for continual spatial sensing. This limitation is evident in two ways. First, its performance deteriorates significantly on long videos. As shown in Tab. 7, when evaluated on VSI-SUPER with 1 FPS sampling in a streaming-style setup, scores drop steadily from 38.3% to 6.0% as video length increases from 10 to 60 minutes, and the model fails completely on videos longer than 60 minutes. Second, the model has difficulty generalizing to new test scenarios. Although trained on multi-room house tour videos, it fails to handle unseen examples with just a few additional rooms. This issue isn’t simply about context length: performance drops even on short 10-minute videos that fit comfortably within model’s context window. These results highlight that a purely data-driven approach within the current MLLM framework, no matter how much data or engineering effort is invested, faces fundamental limits. Addressing these limitations calls for a paradigm shift toward AI systems that can actively model and anticipate the world while organizing their experiences more efficiently, which we explore next.

Scaling data and models is essential, but alone it cannot unlock true spatial supersensing.

- Table 7 | Cambrian-S-7B results on VSI-SUPER. Despite strong performance on VSI-Bench, accuracy on VSR drops sharply from 38.3% (10 min) to 0.0% (>60 min), and VSC completely fails. Note that VSI-SUPER focuses on continual, streaming evaluation, where uniform sampling 128 frames across the entire video does not align with the online setting; results shown in gray are provided for reference only.

|Eval Setup|VSR 10 min 30 min 60 min 120 min 240 min<br><br>|VSC 10 mins 30 min 60 min 120 min|
|---|---|---|
|Uni. Sampling, 128F FPS Sampling, 1FPS<br><br>|26.7 21.7 23.3 30.0 28.2 38.3 35.0 6.0 0.0 0.0<br><br>|16.0 0.0 0.0 0.0 0.6 0.0 0.0 0.0|

###### 4. Predictive Sensing as a New Paradigm

Performance of both Gemini-2.5-Flash (Tab. 1) and Cambrian-S (Tab. 7) drops sharply on VSI-SUPER, revealing a fundamental paradigm gap: scaling data and context alone is insufficient for supersensing. We propose predictive sensing as a path forward, where models learn to anticipate their sensory input and construct internal world models to handle unbounded visual streams. This design is inspired by theories of human cognition. Unlike current video multimodal models that tokenize and process entire data streams, human perception (and memory) is highly selective, retaining only a fraction of sensory input [130, 95, 52, 108]. The brain continuously updates internal models to predict incoming stimuli, compressing or discarding predictable inputs that contribute no novel information [29, 41]. In contrast, unexpected sensory information that violates predictions generates “surprise” and drives increased attention and memory encoding [115, 45, 60]. We prototype this concept via a self-supervised next-latentframe prediction approach (Sec. 4.1). The resulting prediction error serves as a control signal for two key capabilities: memory management to selectively retain important information (Sec. 4.2), and event segmentation to partition unbounded streams into meaningful chunks (Sec. 4.3). We demonstrate through two case studies on VSI-SUPER that this approach substantially outperforms strong long-context and streaming video model baselines.

###### 4.1. Predictive Sensing via Latent Frame Prediction

We implement our predictive sensing paradigm through a lightweight, self-supervised module called the Latent Frame Prediction (LFP) head, which is trained jointly with the primary instruction-tuning objective. This is achieved by modifying the stage 4 training recipe as follows:

- • Latent frame prediction head. We introduce an LFP Head, a two-layer MLP that operates in parallel with the language head, to predict the latent representation of the subsequent video frame. This architecture is illustrated in the top left of Fig. 9.
- • Learning objectives. To optimize the LFP head, we introduce two auxiliary losses, mean squared error (MSE) and cosine distance, which measure the discrepancy between the predicted latent feature and the ground truth feature of the next frame. A weighting coefficient balances the LFP loss against the primary instruction-tuning next token prediction objective.
- • Data for LFP training. We augment stage 4 data with a 290K video subset from VSI-590K used exclusively for the LFP objective. Unlike instruction tuning, these videos are sampled at a constant rate of 1 FPS to ensure uniform temporal spacing for latent frame prediction.

During this modified stage 4 finetuning, we train the connectors, language model, and both the language and LFP heads jointly in an end-to-end manner, while keeping the SigLIP vision encoder frozen. All other training settings remain consistent with the original stage 4 configuration. For brevity, we still denote the model jointly optimized with the LFP objective as Cambrian-S in subsequent experiments.

Inference: Estimating surprise via prediction error. During inference, we leverage the trained LFP head to evaluate the “surprise” for every incoming visual sensory input. In psychology, this framework is often described as the Violation-of-Expectation (VoE) paradigm [17]. Specifically, during inference, video frames are fed into Cambrian-S at a constant sampling rate. Unless otherwise noted, the videos in the following experiments are sampled at 1 FPS before being input into the model. As the model receives incoming video frames, it continuously predicts the latent features of the next frame. We then measure the cosine distance between the model’s prediction and the actual ground truth feature of that incoming

frame. This distance serves as a quantitative measure of surprise: a larger value indicating a greater deviation from the model’s learned expectations. This surprise score acts as a powerful, self-supervised guidance signal for the downstream tasks explored next.

| | |
|---|---|
| | |

Answer: 1

Teacher Forcing

LFP Head LM Head

TrainingInference

LLM Main Model (Transformer Blocks)

Vision Enc

Emb. Layer

[Figure 72]

How many desks are there in this room?

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]| |
|---|---|
| | |

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

Feature Extraction 𝐹

Feature Extraction 𝐹

Prediction Error (Surprise)

Next (Latent) Frame Prediction

Surprise

room change

0.5

novel object

0.4

Timestamp

- Figure 9 | Training and inference pipeline for the latent frame prediction (LFP) approach. Our model employs a Latent Frame Prediction (LFP) head to predict the next frame in latent space. During training, the LFP head predicts the latent representation of the subsequent video frame. During inference, the model measures surprise by computing the cosine distance between the LFP head’s prediction and the actual latent features of the subsequent frame. The surprise signal exhibits distinct spikes for events such as the sudden appearance of unusual objects and abrupt scene changes. Our predictive-sensing prototype allows Cambrian-S to generalize to longer videos on VSI-SUPER, outperforming frontier models (e.g., Gemini-2.5-Flash) that rely solely on context length expansion.

###### 4.2. Case Study I: Surprise-driven Memory Management System for VSI-SUPER Recall.

Most current MLLMs treat all video frames equally, storing every frame without selective compression or forgetting, which limits efficiency and scalability. In this case study, we explore augmenting MLLMs with a surprise–driven memory management framework to support continual spatial-sensing question answering over long-duration videos. We show that through the surprise-guided compression, CambrianS maintains consistent accuracy and stable GPU memory footprints, independent of video length.

Surprise-driven memory management system. Our memory management system dynamically compresses and consolidates visual streams based on the estimate of “surprise”. As shown in Fig. 10-a, we encode incoming frames using sliding window attention with fixed window size. The latent frame prediction module then measures a “surprise level” and assigns it to each frame’s KV caches. Frames with a surprise level below a predefined threshold undergo 2× compression before being pushed into long-term memory. To maintain a stable GPU memory footprint, this long-term memory is constrained to a fixed size by a consolidation function that, once again, operates based on surprise: dropping or merging frames according to their surprise scores (see Fig. 10-b). Finally, upon receiving a user query, the system retrieves the top-𝐾 most relevant frames from the long-term memory by calculating the cosine similarity between the query and the stored frame features (see Fig. 10-c). See Sec. F.2 for more design details. While prior works have explored memory system designs for long videos [119, 159], our focus is on exploring prediction errors (i.e., surprise) as guiding signals.

###### (b) Long-term Memory

(a) Visual Streams Encoding

|Window Attention|
|---|

... ...

|MLLM|
|---|

KVCaches

Calculate Surprise Level

Store

Compress

...

(c) Memory Retrieval

Retrieve

User Query

|LLM Decoder|
|---|

Consolidate

- Figure 10 | Surprise-driven memory management framework design. The proposed memory system (a) encodes incoming visual streams, compressing frames with low surprise; (b) performs consolidation when memory is full by dropping or merging the least surprising frames; and (c) retrieves relevant frames during query answering. Color shading (dark→light) reflects the degree of surprise, with hatched boxes denoting compressed frames and solid boxes representing uncompressed ones.

1MCtxForGemini

10 30 60 120 240 Video Duration (in Mins.)

0

20

40

60

80

Accuracy(%)

Gemini-2.5-Flash Gemini-2.0-Flash Cambrian-S (w/o Mem.)

Cambrian-S (w/ Mem.)

(a) VSR results

10 30 60 120 240 Video Duration (in Mins.)

20

40

60

| |Cambrian-S (w/ Mem.)| |
|---|---|---|
| |Cambrian-S (w/o Mem.)| |

80

100

120

140

MemoryUsage(GB)

OOM

Memory Limitation

(b) GPU memory usage

10 30 60 120 240 VSR Duration (in Mins.)

34

36

38

40

42

| |Pred. Error as Surprise| |
|---|---|---|
| |Adj. Frame Vision Feature Diff. as Surprise| |

44

Accuracy(%)

(c) Surprise comparison

- Figure 11 | Performance analysis of surprise-driven memory on VSR. (a) Surprise-driven memory allows Cambrian-S to maintain strong performance as video length increases. (b) Surprise-driven memory maintains a stable GPU memory footprint as video length increases. (c) Ablation: Using LFP prediction error as the surprise signal is more robust and consistently outperforms using adjacent-frame similarity.

Results. We compare Cambrian-S with and without the surprise-based memory system against two advanced proprietary models, Gemini-1.5-Flash [122] and Gemini-2.5-Flash [30], on the VSR benchmark. As shown in Fig. 11a, Cambrian-S (w/ Mem.) outperforms both Gemini-1.5-Flash and Cambrian-S (w/o. Mem.) at all video lengths, demonstrating consistent spatial sensing performance across video durations. Although Gemini-2.5-Flash yields strong results for videos within an hour, it fails to process longer inputs. In addition to maintaining high accuracy, Cambrian-S (w/ Mem.) also maintains stable GPU memory usage across different video lengths (Fig. 11b). This demonstrates that surprise-based memory effectively compresses redundant data without losing critical information. We include two long-video baselines, MovieChat [119] and Flash-VStream[159], for comparison in Tab. 17.

Ablation on surprise measurement. Central to our surprise-based memory system is the mechanism for measuring surprise, which dictates how frames are compressed or consolidated in a passive sensing manner—without assuming any prior knowledge of future queries. Here, we compare our design, prediction error as surprise, to another straightforward baseline: adjacent-frame visual-feature similarity. Specifically, we use SigLIP2 as the vision encoder and directly compare the frame feature difference (cosine distance) between two adjacent frames. If the difference exceeds a threshold, we treat the later frame as a surprise frame. We compare these two methods across all VSR variants. For each VSR duration, we keep the experimental setup identical except for the surprise threshold, which we tune for

both methods. As shown in Fig. 11c, using prediction error as the surprise measurement consistently outperforms adjacent-frame similarity across different video durations.

Predictive sensing provides a more principled approach to modeling the spatiotemporal dynamics of video data than static similarity measures based on per-frame features.

While our current system employs a simple predictive head as an initial prototype, future integration of a more capable world model could produce richer and more reliable surprise signals, ultimately enabling broader advances in spatial supersensing.

###### 4.3. Case Study II: Surprise-driven continual video segment for VSI-SUPER Count.

While VSR focuses on evaluating the long-term observation and recall abilities of MLLMs, a more challenging test of supersensing would involve testing a model’s capacity to interpret its sensory input, navigate across varied environments, and perform cumulative, multihop reasoning. For example, the model might need to complete a task in one environment, move to another, and ultimately integrate information from all experiences to reach a final decision.

Continual Visual Streams

## ... ...

Calculate surprise

Event Buffer

Examine surprise level

store

| | |
|---|---|
| | |

|LLM Decoder| |
|---|---|
| | |

load

Query

...

clear

Per-segment answers

Final answer

For low surprise frames For high surprise frames

- Figure 12 | Illustration of our surprise-driven event segmentation framework for VSC. The model continuously accumulates frame features in an event buffer. When a high-surprise frame is detected, the buffered features are summarized to produce a segment-level answer, and the buffer is cleared to start a new segment. This process repeats until the end of the video, after which all segment answers are aggregated to form the final output. Color shading (dark→light) reflects the degree of surprise.

Surprise-driven event segmentation. An event can be understood as a spatiotemporally coherent segment of experience [64]. In the context of spatial supersensing, an event corresponds to a continuous experience of being situated within a specific space and sensing its environment. This definition emphasizes that real sensory experience is typically organized into locally coherent segments—episodes where perceptual, spatial, and temporal features remain relatively stable or consistent. Event segmentation, then, is the process of parsing a continuous stream of sensory input into discrete, meaningful units based on changes in this coherence. Such segmentation is essential for reasoning and behavior [37]: it allows an agent (biological or artificial) to form structured representations of experience, detect boundaries where significant change occurs, and update predictions about the environment accordingly. Recent studies highlight that prediction error and changes in working memory/context are two possible mechanisms driving segmentation [98, 118].

In the VSI-SUPER Count (VSC) benchmark, we examine a simple setting where surprise is used to segment continuous visual input, identifying scene changes as natural breakpoints that divide the

video stream into spatially coherent segments. This approach also parallels human problem-solving: when counting objects across a large area, people typically focus on one section at a time before combining the results. This behavior is also related to the “doorway effect” [106], in which passing through a doorway or entering a new room creates a natural boundary in memory. As illustrated in Fig. 12, the model continuously accumulates frame features in an event buffer. When a high-surprise frame is detected, the buffered features are summarized to produce a segment-level answer, and the buffer is cleared to start a new segment. This cycle repeats until the end of the video, after which all segment answers are aggregated to form the final output.

40

| | |1M Ctx For Gemini<br><br>| |
|---|---|---|---|
| | | | |
| | | | |
| |Gem|ini-2.5-Flash-Live| |
| |GPT Cam<br><br>|-Realtime brian-S (w/ Surprise Seg.)| |
| | | | |
| | | | |
| | | | |
| | | | |

42

Gemini-2.5-Flash Gemini-1.5-Flash

Cambrian-S (w/ GT Seg.)

Cambrian-S (w/ Surprise Seg.)

35

40

60

30

1M Ctx For Gemini

38

50

25

36

40

MRA

MRA

20

MRA

34

30

15

32

20

10

10

30

5

Adj. Frame Vision Feature Diff. as Surprise

Pred. Error as Surprise

0

0

28

10 30 60 120 Video Duration (in Mins.)

10 30 60 120 Video Duration (in Mins.)

10 30 60 120 Video Duration (in Mins.)

(a) VSC results

(b) Different surprise measurement

(c) Streaming evaluation

- Figure 13 | Performance analysis on VSC. (a) Cambrian-S with surprise-driven event segmentation achieves consistently higher and more stable performance across all video lengths compared to Gemini2.5-Flash; (b) Ablation: prediction error as surprise outperforms adjacent-frame similarity; (c) Streaming evaluation: Although GPT-Realtime and Gemini-Live are marketed as “live assistants”, they achieve less than 15% MRA and their performance drops to near zero on long videos, while our method maintains substantially higher performance.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

25 50 75 Ground Truth

25

50

75

Prediction

10 mins

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

50 100 150 Ground Truth

50

100

150

Prediction

30 mins

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

50 100 150 200 250 Ground Truth

50

100

150

200

250

Prediction

60 mins

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

100 200 300 Ground Truth

100

200

300

Prediction

120 mins

Cambrian-S

| |
|---|

Gemini-2.5-Flash

| |
|---|

Line of Perfect Prediction

- Figure 14 | Cambrian-S scales to higher ground truth object counts whereas Gemini saturates. Predicted counts are plotted against ground-truth counts for videos of different lengths (10, 30, 60, and 120 minutes). Using surprise-driven segmentation, Cambrian-S’s predicted counts grow approximately linearly with the ground-truth, tracking the 𝑦 = 𝑥 perfect-count line (gray dashed), whereas Gemini-2.5-Flash’s predicted counts remain clustered near small values and fail to increase with ground-truth count, indicating early saturation and poor extrapolation to larger counts.

Results. Gemini-1.5-Flash attains near-zero performance on VSC (Fig. 13a), showing the task’s difficulty. Although Gemini-2.5-Flash yields much better results on 10-minute videos, its performance declines rapidly on longer videos. In contrast, the surprise-driven event segmentation approach used by CambrianS (w/ Surprise Seg.) achieves higher and more stable performance across all video lengths. When the video is segmented using ground-truth scene transitions (i.e., Cambrian-S w/ GT Seg.), performance improves further, representing an approximate upper bound. A deeper analysis in Fig. 14 reveals that Gemini-2.5-Flash’s predictions are confined to a limited range and do not scale as more objects appear in the video. In contrast, Cambrian-S (w/ Surprise Seg.) produces counts that, while not yet fully accurate, exhibit a stronger correlation with the true object numbers, indicating better generalization.

Ablation on surprise measurement. We compare our surprise-driven approach with a baseline using adjacent-frame feature similarity (Fig. 13b). For both methods, we report the best results after hyperparameter tuning. Consistent with our observations in VSR, using prediction error as a measure of surprise consistently outperforms appearance similarity across all video durations by a notable margin.

Evaluation in streaming setup. As the correct answer in VSC evolves throughout the video, we create a streaming QA setup where the same question is asked at 10 different timestamps. The final performance is averaged across all queries. We benchmark against commercial MLLMs marketed for live visual input. As shown in Fig. 13c, although Gemini-Live and GPT-Realtime are intended for streaming scenarios, they achieve under 15% MRA on 10-minute videos and their performance declines to near zero on 120-minute streams. Cambrian-S, however, shows stronger performance, reaching 38% MRA on 10-minute streams and maintaining around 28% at 120 minutes.

Summary. Across both VSR recall and VSC counting tasks, predictive sensing through surprise-driven memory and event segmentation enables Cambrian-S to overcome the fixed-context limitations described in Sec. 3. Although this remains an early prototype, it highlights the potential for building AI systems that not only see but also anticipate, select, and organize experience. Such systems move beyond frame-level Q&A toward constructing implicit world models that support deeper spatial reasoning, scale across unbounded temporal horizons, and achieve supersensing that rivals and ultimately surpasses human visual intelligence.

###### 5. Related Work

Video Multimodal Large Language Models The strong linguistic understanding capabilities of pretrained LLMs [15, 126, 7, 127], combined with the representational power of vision foundation models used as feature extractors [105, 157, 128, 50, 39], have driven significant advances in extending these models beyond text to achieve semantic perception of visual content, primarily in the image domain [56, 78, 65, 8, 124, 121, 27, 134, 68]. This momentum has spurred growing research into video-based MLLMs [74, 65, 161, 119, 9, 167, 158, 69, 168, 89], which are seen as a key step toward connecting multimodal intelligence with real-world applications such as embodied agents [61, 147]. As emphasized throughout this paper, developing a truly capable supersensing system requires rethinking several core aspects, including how progress is benchmarked, what constitutes the right data, which architectural designs are most effective, and what modeling objectives best align with the system’s goals.

Streaming Video Understanding Video is a continuous and potentially infinite stream of visual signals. While humans process it effortlessly, its unbounded nature challenges video MLLMs because token lengths increase with duration, causing rising computational and storage costs. Recent work has explored several approaches to address this problem: Efficient architectural design. The quadratic cost of self-attention makes it hard to handle long videos. Recent methods [70, 112] use simpler, faster architectures [135, 48, 58] that reduce computation and work better with longer inputs. Context window expansion. The fixed context length in pre-trained LLMs limits their understanding of long-term content. Recent work [26, 160, 25] extends this window by careful system design, enabling models to handle and reason over longer video sequences. Retrieval-augmented video understanding. To process long videos, some approaches retrieve only the most relevant segments from a larger collection [63, 101, 136] and use them as context for further analysis.Visual token reduction or compression. Other methods shorten the input by reducing visual tokens across or within frames [117, 73, 57, 72, 19], making it easier to handle long video sequences. While these methods improve performance, they largely treat continuous videos as standard sequence modeling problems, similar to text. We believe future MLLMs should build internal predictive models to efficiently process continuous visual streams, as humans do.

Visual Spatial Intelligence Understanding spatial relationships from visual inputs is crucial for perceiving and interacting with the physical world. As multimodal models become more physically grounded, interest in spatial intelligence has surged, leading to new benchmarks [148, 107, 154, 86, 152, 75, 142, 123] and research focused on enhancing models’ spatial reasoning capabilities [151, 84, 99, 38, 21, 28, 18, 76, 67, 166, 110]. In this paper, we study visual spatial intelligence through the concept of spatial supersensing in videos and explore ways to strengthen MLLMs’ spatial reasoning by refining data curation, optimizing training strategies, and introducing new paradigms.

Predictive Modeling A learned internal predictive model [31, 49] allows an intelligent agent to represent and simulate aspects of its environment, enabling more effective planning and decision-making. Model predictive control (MPC) [43] applies similar principles in control theory, leveraging internal forward models to anticipate future trajectories and select optimal actions in real time. This concept draws inspiration from how humans form mental models of the world [108, 52, 41] and how these internal representations influence behavior (e.g., unconscious inference [130]), serving as simplified abstractions of reality that enable prediction and efficient action. A growing body of work has explored the idea of predictive modeling through self-supervised representation learning [5, 6], and text- or action-conditioned video generation [164, 150, 11, 22, 10, 44]. In this paper, motivated by how humans leverage internal world models to process unbounded sensory input efficiently and effectively, we investigate how to equip MLLMs with a similar predictive sensing capability.

###### 6. Conclusion

We highlight the importance of and propose a hierarchy for spatial supersensing capabilities in videos, arguing that achieving superintelligence requires AI systems to move beyond text-based knowledge and semantic perception, the current focus of most MLLMs, to also develop spatial cognition and predictive world models. To measure progress, we introduce VSI-SUPER and find that current MLLMs struggle with it. To test whether current progress is limited by data, we curate VSI-590K and train our spatially grounded MLLM, Cambrian-S, on it. Although Cambrian-S performs well on standard benchmarks, its results on VSI-SUPER reveal the limitations of the current MLLM paradigm. We prototype predictive sensing, using latent frame prediction and surprise estimation to handle unbounded visual streams. It improves Cambrian-S performance on VSI-SUPER and marks an early step toward spatial supersensing.

Limitations. Our goal is to present a conceptual framework that encourages the community to reconsider the importance of developing spatial supersensing. As a long-term research direction, our current benchmark, dataset, and model design remain limited in quality, scale, and generalizability, and the prototype serves only as a proof of concept. Future work should explore more diverse and embodied scenarios and build stronger connections with recent advances in vision, language, and world modeling.

###### Acknowledgments

We are grateful to Cambrian-1 [124] for the excellent codebase, which served as the launching point for our research. Thanks to the TorchXLA team for helpful discussions on TPU, TorchXLA, and JAX distributed training infrastructure. We also thank Anjali Gupta, Sihyun Yu, Oscar Michel, Boyang Zheng, Xichen Pan, Weiyang Jin, and Arijit Ray for reviewing this manuscript and providing constructive feedback. This work was primarily supported by the Google TPU Research Cloud (TRC) program and the Google Cloud Research Credits program (GCP19980904). E.B. is supported by the DoD NDSEG Fellowship Program. S.X. acknowledges support from the MSIT IITP grant (RS-2024-00457882) and the NSF award IIS-2443404.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, and Lorenzo Torresani. Ht-step: Aligning instructional articles with how-to videos. In NeurIPS, 2023.
- [3] Anthropic. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5

-sonnet, 2024.

- [4] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In CVPR, 2016.
- [5] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023.
- [6] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.
- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [9] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [10] Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, and Jitendra Malik. Whole-body conditioned egocentric video prediction. arXiv preprint arXiv:2506.21552, 2025.
- [11] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In CVPR, 2025.
- [12] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In NeurIPS, 2021.
- [13] Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, and Saining Xie. SIMS-V: Simulated instruction-tuning for spatial video understanding. arXiv preprint, 2025.
- [14] Ellis Brown, Jihan Yang, Shusheng Yang, Rob Fergus, and Saining Xie. Benchmark designers should “train on the test set” to expose exploitable non-visual shortcuts. arXiv preprint, 2025.
- [15] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.
- [16] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Xindong He, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In IROS, 2025.
- [17] Judee K Burgoon and Jerold L Hale. Nonverbal expectancy violations: Model elaboration and application to immediacy behaviors. Communications Monographs, 55(1):58–79, 1988.
- [18] Wenxiao Cai, Yaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In ICRA, 2025.

- [19] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. In ICLR, 2025.
- [20] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Cristóbal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. Hourvideo: 1-hour videolanguage understanding. In NeurIPS, 2024.
- [21] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024.
- [22] Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, and Sungjin Ahn. Simple hierarchical planning with diffusion. In ICLR, 2024.
- [23] Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, et al. Gui-world: A video benchmark and dataset for multimodal gui-oriented understanding. In ICLR, 2025.
- [24] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In CVPR, 2024.
- [25] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. In NeurIPS, 2025.
- [26] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In ICLR, 2025.
- [27] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024.
- [28] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024.
- [29] Andy Clark. Whatever next? predictive brains, situated agents, and the future of cognitive science. Behavioral and brain sciences, 2013.
- [30] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [31] Kenneth James Williams Craik. The nature of explanation. CUP Archive, 1967.
- [32] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024.
- [33] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.
- [34] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, 2022.
- [35] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In ICML, 2017.
- [36] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural generation. In NeurIPS, 2022.

- [37] Peter Ford Dominey. Narrative event segmentation in the cortical reservoir. PLOS Computational Biology, 17(10):e1008993, 2021.
- [38] Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In ACL, 2024.
- [39] David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. In ICCV, 2025.
- [40] Li Fei-Fei, Asha Iyer, Christof Koch, and Pietro Perona. What do we perceive in a glance of a real-world scene? Journal of vision, 2007.
- [41] Karl Friston. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 2010.
- [42] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025.
- [43] Carlos E Garcia, David M Prett, and Manfred Morari. Model predictive control: Theory and practice—a survey. Automatica, 25(3):335–348, 1989.
- [44] Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025.
- [45] Samuel J Gershman, Marie-H Monfils, Kenneth A Norman, and Yael Niv. The computational nature of memory modification. Elife, 2017.
- [46] James J Gibson. The ecological approach to visual perception: classic edition. Psychology press, 2014.
- [47] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022.
- [48] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In COLM, 2024.
- [49] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [50] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.
- [51] D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [52] Jakob Hohwy. The predictive mind. OUP Oxford, 2013.
- [53] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-MMMU: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [54] Zi-Yuan Hu, Shuo Liang, Duo Zheng, Yanyang Li, Yeyao Tao, Shijia Huang, Wei Feng, Jia Qin, Jianguang Yu, Jing Huang, et al. Nemo: Needle in a montage for video-language understanding. arXiv preprint arXiv:2509.24563, 2025.
- [55] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.
- [56] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [57] Jindong Jiang, Xiuyu Li, Zhijian Liu, Muyang Li, Guo Chen, Zhiqi Li, De-An Huang, Guilin Liu, Zhiding Yu, Kurt Keutzer, et al. Token-efficient long video understanding for multimodal llms. arXiv preprint arXiv:2503.04130, 2025.
- [58] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020.
- [59] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.
- [60] Nicholas GW Kennedy, Jessica C Lee, Simon Killcross, R Fred Westbrook, and Nathan M Holmes. Prediction error determines how memories are organized in the brain. Elife, 2024.
- [61] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.
- [62] Kristin Koch, Judith McLean, Ronen Segev, Michael A Freed, Michael J Berry, Vijay Balasubramanian, and Peter Sterling. How much the eye tells the brain. Current biology, 2006.
- [63] Bruno Korbar, Yongqin Xian, Alessio Tonioni, Andrew Zisserman, and Federico Tombari. Textconditioned resampler for long form video understanding. In ECCV, 2024.
- [64] Christopher A Kurby and Jeffrey M Zacks. Segmentation in the perception and memory of events. Trends in cognitive sciences, 12(2):72–79, 2008.
- [65] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. TMLR, 2025.
- [66] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In CVPR, 2024.
- [67] Chengzu Li, Caiqi Zhang, Han Zhou, Nigel Collier, Anna Korhonen, and Ivan Vuli´c. Topviewrs: Vision-language models as top-view spatial reasoners. In EMNLP, 2024.
- [68] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.
- [69] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [70] Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, and Yu Qiao. Videomamba: State space model for efficient video understanding. In ECCV, 2024.
- [71] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. MVbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024.
- [72] Wei Li, Bing Hu, Rui Shao, Leyang Shen, and Liqiang Nie. Lion-fs: Fast & slow video-language thinker as online video assistant. In CVPR, 2025.
- [73] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.
- [74] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In ECCV, 2024.
- [75] Yun Li, Yiming Zhang, Tao Lin, XiangRui Liu, Wenxiao Cai, Zheng Liu, and Bo Zhao. Sti-bench: Are mllms ready for precise spatial-temporal world understanding? In ICCV, 2025.

- [76] Benlin Liu, Yuhao Dong, Yiqin Wang, Zixian Ma, Yansong Tang, Luming Tang, Yongming Rao, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondences boost spatial-temporal reasoning in multimodal language model. In CVPR, 2025.
- [77] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.
- [78] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [79] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. In ACL, 2024.
- [80] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.
- [81] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024.
- [82] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. SCIS, 2024.
- [83] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.
- [84] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. In NeurIPS, 2025.
- [85] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024.
- [86] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In CVPR, 2024.
- [87] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In NeurIPS, 2023.
- [88] Kevis-Kokitsi Maninis, Kaifeng Chen, Soham Ghosh, Arjun Karpur, Koert Chen, Ye Xia, Bingyi Cao, Daniel Salz, Guangxing Han, Jan Dlabal, et al. Tips: Text-image pretraining with spatial awareness. In ICLR, 2024.
- [89] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.
- [90] David Marr. Vision: A computational investigation into the human representation and processing of visual information. MIT press, 2010.
- [91] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022.
- [92] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.

- [93] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Anton Belyi, et al. Mm1: methods, analysis and insights from multimodal llm pre-training. In ECCV, 2024.
- [94] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV, 2019.
- [95] Beren Millidge, Tommaso Salvatori, Yuhang Song, Rafal Bogacz, and Thomas Lukasiewicz. Predictive coding: Towards a future of deep learning beyond backpropagation? In IJCAI, 2022.
- [96] Muhammad Ferjad Naeem, Yongqin Xian, Xiaohua Zhai, Lukas Hoyer, Luc Van Gool, and Federico Tombari. Silc: Improving vision language pretraining with self-distillation. In ECCV, 2024.
- [97] Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In CVPR, 2025.
- [98] Sophie Nolden, Gözem Turan, Berna Güler, and Eren Günseli. Prediction error and event segmentation in episodic memory. Neuroscience & Biobehavioral Reviews, 157:105533, 2024.
- [99] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.
- [100] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In ICRA, 2024.
- [101] Junwen Pan, Rui Zhang, Xin Wan, Yuan Zhang, Ming Lu, and Qi She. Timesearch: Hierarchical video search with spotlight and reflection for human-like long video understanding. arXiv preprint arXiv:2504.01407, 2025.
- [102] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In ICCV, 2023.
- [103] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. In NeurIPS, 2023.
- [104] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In CVPR, 2025.
- [105] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [106] Gabriel A Radvansky, Sabine A Krawietz, and Andrea K Tamplin. Walking through doorways causes forgetting: Further explorations. Quarterly journal of experimental psychology, 2011.
- [107] Santhosh Kumar Ramakrishnan, Erik Wijmans, Philipp Kraehenbuehl, and Vladlen Koltun. Does spatial cognition emerge in frontier models? In ICLR, 2025.
- [108] Rajesh PN Rao and Dana H Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 1999.
- [109] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025.

- [110] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. SAT: Spatial Aptitude Training for Multimodal Language Models. In COLM, 2025.
- [111] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In CVPR, 2024.
- [112] Weiming Ren, Wentao Ma, Huan Yang, Cong Wei, Ge Zhang, and Wenhu Chen. Vamba: Understanding hour-long videos with hybrid mamba-transformers. In ICCV, 2025.
- [113] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021.
- [114] Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. Scienceqa: A novel resource for question answering on scholarly articles. IJDL, 2022.
- [115] Wolfram Schultz and Anthony Dickinson. Neuronal coding of prediction errors. Annual review of neuroscience, 2000.
- [116] Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. In ICLR, 2025.
- [117] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. In ICML, 2025.
- [118] Sunjae Shim, Franck B Mugisho, Lila Davachi, and Christopher Baldassano. Generating event boundaries in memory without prediction error. PsyArXiv Preprints, 2024.
- [119] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, 2024.
- [120] Aimee E Stahl and Lisa Feigenson. Observing the unexpected enhances infants’ learning and exploration. Science, 2015.
- [121] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [122] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [123] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.
- [124] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In NeurIPS, 2024.
- [125] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, 2024.
- [126] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [127] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [128] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [129] Paul Voigtlaender, Soravit Changpinyo, Jordi Pont-Tuset, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with video localized narratives. In CVPR, 2023.
- [130] Hermann Von Helmholtz. Handbuch der physiologischen Optik. L. Voss, 1867.
- [131] Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim M Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, and Xiaohua Zhai. Locca: Visual pretraining with location-aware captioners. In NeurIPS, 2024.
- [132] Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Cosmo: Contrastive streamlined multimodal model with interleaved pre-training. arXiv preprint arXiv:2401.00849, 2024.
- [133] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025.
- [134] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [135] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
- [136] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In ECCV, 2024.
- [137] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024.
- [138] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, et al. Videorope: What makes for good video rotary position embedding? In ICML, 2025.
- [139] Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Vivian Chen, and Hung-yi Lee. Streambench: Towards benchmarking continuous improvement of language agents. In NeurIPS, 2024.
- [140] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. In NeurIPS, 2024.
- [141] xAI. Grok-1.5 Vision Preview. https://x.ai/blog/grok-1-5v, April 2024. RealworldQA, Blog post, Announced on April 12, 2024.
- [142] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025.
- [143] Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi, Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, et al. Gspmd: general and scalable parallelization for ml computation graphs. arXiv preprint arXiv:2105.04663, 2021.
- [144] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In CVPR, 2022.

- [145] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [146] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words. In NeurIPS, 2024.
- [147] Jihan Yang, Runyu Ding, Ellis Brown, Xiaojuan Qi, and Saining Xie. V-IRL: Grounding virtual intelligence in real life. In ECCV, 2024.
- [148] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. In CVPR, 2024.
- [149] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. Egolife: Towards egocentric life assistant. In CVPR, 2025.
- [150] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In ICLR, 2024.
- [151] Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, and Chuang Gan. Mindjourney: Test-time scaling with world models for spatial reasoning. arXiv preprint arXiv:2507.12508, 2025.
- [152] Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280, 2025.
- [153] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In ICCV, 2023.
- [154] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025.
- [155] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 2024.
- [156] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.
- [157] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.
- [158] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, 2023.
- [159] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flashvstream: Memory-based real-time understanding for long video streams. In ICCV, 2025.
- [160] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [161] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. TMLR, 2025.
- [162] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue, Longteng Guo, Bingning Wang, Weipeng Chen, and Jing Liu. Needle in a video haystack: A scalable synthetic evaluator for video mllms. arXiv preprint arXiv:2406.09367, 2024.

- [163] Jieyu Zheng and Markus Meister. The unbearable slowness of being: Why do we live at 10 bits/s? Neuron, 2025.
- [164] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. In ICML, 2025.
- [165] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In AAAI, 2018.
- [166] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. In ICCV, 2025.
- [167] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [168] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, et al. Apollo: An exploration of video understanding in large multimodal models. In CVPR, 2025.

###### Appendix

This appendix provides comprehensive implementation details, experimental results, and supplementary analyses supporting the main paper:

- • § A presents detailed diagnostic test results for video MLLM benchmarks under different evaluation setups.
- • § B describes the VSI-SUPER benchmark, including implementation details, visualizations, and streaming setups for both Recall and Count tasks.
- • § C provides comprehensive documentation of the VSI-590K dataset, including question type taxonomy, QA-pair construction pipeline, ablation studies, and qualitative examples.
- • § D details the Cambrian-S model architecture, training data mixture, training recipe across all four stages, and infrastructure setup.
- • § E presents additional experimental results including detailed evaluation setups, performance on image and video benchmarks across all model scales, ablations on image-video data contributions, and analysis of the trade-off between spatial sensing and general video understanding.
- • § F describes predictive sensing components, including latent frame prediction implementation details, memory framework design for VSI-SUPER Recall, agentic framework design for VSI-SUPER Count, and comparisons with existing long-video methods.

###### A. Benchmark Diagnostic Test Results We provide detailed results of Fig. 2 in Tab. 8.

- Table 8 | Detailed results of our improved Cambrian-1-7B on video MLLM benchmarks under different evaluation setups.

|Evaluation Setups<br><br>|VideoMME<br><br>EgoSchema<br><br>VideoMMMU<br><br>LongVideoBench<br><br>Tomato<br><br>MVBench<br><br>PerceptionTest<br><br>HourVideo<br><br>VSI-Bench<br><br>VSI-SRecallUPER<br><br>VSI-SCountUPER|
|---|---|
|Chance-Level|25.0 20.0 14.0 25.0 22.0 27.3 33.3 20.0 34.0 25.0 0.0|
|Cambrian-1-7B (Our upgraded) Blind Test Single Frame Multiple (32) Frames<br><br>(32) Frame Captions|31.2 31.9 25.0 42.5 7.8 19.6 40.7 24.3 17.4 20.0 0.0 41.6 44.0 29.0 46.9 15.8 46.1 52.1 27.7 20.4 19.7 0.0 53.7 48.1 31.9 51.4 18.9 55.4 55.6 31.6 25.8 22.7 0.0 55.3 52.4 40.1 52.2 16.8 47.7 55.6 29.5 21.8 9.6 0.5<br><br>|

###### B. VSI-SUPER Benchmark

###### B.1. VSI-SUPER Recall

Implementation details. To construct this benchmark, we begin with videos from the VSI-Bench collection [148]. Annotators select videos and manually insert an unusual object from a curated pool into four distinct frames using Gemini-2.0-Flash, focusing on placing the objects in plausible locations. For each insertion, the annotators record the object’s location and its order of appearance. We then combine these edited clips with randomly sampled unedited videos to produce final videos with lengths of 10, 30, 60, 120, and 240 minutes. For each duration, we create 60 videos, each with one corresponding question. We downsample videos to 1 frame per second to ensure the model can always see the edited frames during inference.

Visualization. We present qualitative examples of edited frames of our VSR video dataset in Fig. 18. The inserted objects appear visually plausible at their locations, which is a direct result of our high-quality annotations.

###### B.2. VSI-SUPER Count

Implementation Details. To build VSI-SUPER Count, we concatenate videos from VSI-Bench [148] and sum their object counts to create a new ground truth. This process requires two additional normalization steps. First, we unify the object category labels from the different source datasets (i.e., ScanNet [33], ScanNet++ [153], and ARKitScenes [12]). Second, we address a data bias towards small object quantities by rebalancing the question-answer pairs to create a more uniform distribution of counts. The final benchmark includes videos with lengths of 10, 30, 60, and 120 minutes, each accompanied by 50 corresponding questions. Different from VSR, all videos in VSC are downsampled to 24 FPS.

Streaming setups. For the streaming setup, we repeatedly query the total number of objects in a video at 10 distinct timestamps. To construct the ground truth at these query timestamps, we need to determine the first appearance time of each unique object in the video. To find these appearance times, we use the method proposed by the VSI-Bench [148]. This allows for the direct calculation of the ground truth object count at any given timestamp.

###### C. VSI-590K Dataset

In this section, we provide more details for our VSI-590K dataset, including the question type definition, question-answer pair construction pipeline, and some examples for each data source.

###### C.1. Details of Question Type Definition

Taxonomy. When curating visual-spatial intelligence supervised fine-tuning datasets, an important perspective is how to define the question type. Inspired by VSI-Bench [148], we expand its task definition in a more systematic manner. As shown in Tab. 9, we distinguish these question types in four perspectives:

- • Spatial-temporal attributes: We categorize questions into five distinct spatial-temporal attribute types: size (comparing or measuring object/space dimensions), direction (orientation in space), count (enumeration of objects), distance (proximity between objects), and appearance order (temporal sequence of objects appearing in videos).
- • Relative versus absolute: Questions are classified as relative when they involve comparison between multiple objects (e.g., “which is larger?”), or absolute when they require specific measurements or quantities (e.g., “what is the height in meters?”). This distinction applies across most attribute types.
- • Perspective taking: This dimension captures the viewpoint from which spatial relationships are evaluated. Questions may be posed from the camera’s perspective (e.g., “from the camera’s perspective, is the object on the left or right?”) or from the perspective of specific objects in the scene (e.g., “facing the object1 from object2...”)
- • Modality: Questions are categorized based on whether they can be answered using static images only, or require dynamic video information. Some attribute types, like appearance order, are only applicable to videos, while others like size can be questioned in either modality.

Additionally, following VSI-Bench, we also categorize our question types into three different groups (i.e., Configuration, Measurement, or Spatiotemporal) according to their different spatiotemporal characteristics.

###### C.2. Detailed QA-Pair Construction Pipeline We introduce the concrete pipeline used for curating VSI-590K here.

3D-annotated real videos. For the 3D-annotated real videos, we follow the practice established by Thinking in Space [148]. We begin by researching all publicly available datasets containing both 3D instance-level annotations and video or panorama images. From these datasets, we extract key information including object counts, object bounding boxes, and room size measurements, which we then standardize into a unified format. Afterward, this structured information is incorporated into augmented question templates to create paired question-answer sets.

- Table 9 | Taxonomy of spatiotemporal question types in VSI-590K. Questions are stratified along five axes: attribute type, relative vs. absolute (Rel./Abs.), perspective, modality (V: video, I: image), and group. An example question template is provided for each type.

Types Rel./Abs. Perspective Modality Group Example template

Rel. — V & I Configuration “Between {object1} and {object2}, which is larger?” Abs. — V & I Measurement “What is the height of the {object} in {unit}?” Abs. — V & I Measurement “What is the room’s size in {unit}?”

Size

Rel. Camera I Configuration “From the camera’s perspective, is the {object} on the left or the right?” Rel. Object V & I Configuration “Facing the {object1} from the {object2}, would the {object3} be placed left, right, or back?” Abs. Object V & I Measurement “Standing at {object1}, facing toward {object2}, how far

Direction

clockwise do I rotate (in degrees) to see the {object3}?” Count

Rel. — V & I Configuration “Are there fewer {object1} than {object2} ?” Abs. — V & I Measurement “How many {object} are present?”

Rel. Camera I Configuration “Which object is closer to the camera, the {object_1} or the {object_2}?” Rel. Object V & I Configuration “Which is nearer to the {object_3}, the {object_1} or the {object_2}?” Abs. Object V & I Measurement “What is the distance between the {object_1} and the {object_2} in {unit}?”

Distance

“Determine how {object_1}, {object_2}, {object_3}, and {object_4} are ordered by their initial appearances in the video”

Appr. Order — — V Spatiotemporal

Appearance Order

Count

Direction (43.1%)

| |
|---|

7.4%

- - Relative Direction Object (30.0%)

| |
|---|

- - Absolute Direction Object (7.3%)

| |
|---|

- - Relative Direction Camera (5.8%)

| |
|---|

Size 16.4%

Distance (30.4%)

| |
|---|

- - Relative Distance Object (20.2%)

| |
|---|

- - Absolute Distance Object (10.0%)

| |
|---|

- - Relative Distance Camera (0.2%)

| |
|---|

Direction

43.1%

VSI-590K

Size (16.4%)

| |
|---|

- - Relative Size Object (8.8%)

| |
|---|

- - Absolute Size Object (6.5%)

| |
|---|

- - Absolute Size Room (1.1%)

| |
|---|

Count (7.4%)

| |
|---|

- - Absolute Count (4.6%)

| |
|---|

- - Relative Count (2.8%)

| |
|---|

30.4%

Appearance Order (2.7%)

| |
|---|

- Appearance Order (2.7%)

| |
|---|

Distance

Spatiotemporal

Configuration (72.3%)

| |
|---|

- - Relative Direction Object (30.0%)

| |
|---|

- - Relative Distance Object (20.2%)

| |
|---|

- - Relative Size Object (8.8%)

| |
|---|

- - Relative Direction Camera (5.8%)

| |
|---|

- - Absolute Count (4.6%)

| |
|---|

- - Relative Count (2.8%)

| |
|---|

- - Relative Distance Camera (0.2%)

Measurement 24.9%

| |
|---|

VSI-590K

Measurement (24.9%)

| |
|---|

- - Absolute Distance Object (10.0%)

| |
|---|

- - Absolute Direction Object (7.3%)

| |
|---|

- - Absolute Size Object (6.5%)

| |
|---|

- - Absolute Size Room (1.1%)

| |
|---|

72.3%

Spatiotemporal (2.7%)

| |
|---|

Configuration

- Appearance Order (2.7%)

| |
|---|

- Figure 15 | VSI-590K dataset statistics. QAs are grouped by: question types (left) and task groups (right).

3D-annotated simulated videos and images. For simulated data, which inherently contains rich annotations, we followed a procedure similar to that used for 3D-annotated real videos. As for ProcTHOR [36], our primary effort is generating 3D scenes with randomly placed agents to render traverse videos. For Hypersim [113], which provides image-level rather than scene-level 3D annotations, we utilize individual images with their corresponding 3D annotations. In both cases, we extract the necessary information, convert it to our designed unified format, and incorporate it into augmented question templates, following the same approach used for 3D-annotated real videos.

Unannotated web-crawled real videos. For unannotated web-crawled real videos, as shown in Algorithm 1, we implement a multi-stage processing pipeline. We begin by sampling frames at regular intervals and filtering out blurry images. For each valid frame, we employ the open-vocabulary object detector Grounding-DINO [80] with predefined categories of interest. When a frame contains sufficient valid objects, we use SAM2 [109] to extract instance-wise semantic masks. Besides, to transform 2D image content into 3D representations, we employ VGGT [133] to extract 3D point sets for each image and integrate them with the previously generated instance masks. Notably, we apply an erosion algorithm to refine the instance masks, which mitigates inaccurate point cloud estimations at object boundaries. This pipeline has enabled us to create pseudo-annotations from approximately 19,000 room tour videos from

YouTube and robotic learning datasets, yielding diverse spatial question-answer pairs across various room types and layouts without manual 3D annotations. By processing individual frames rather than complete videos, our pipeline ensures higher quality semantic extraction and more reliable reconstruction results, avoiding the noise and inconsistent issues typically encountered when applying reconstruction and semantic extraction techniques to entire video sequences.

###### C.3. Additional Ablation Study

Table 10 | Ablation study on VSI-590K task groups. We study models’ performance change when one certain task group are omitted from the training data.

|VSI-590K Mixture<br><br>|VSI-Bench<br><br>Avg<br><br>ObjCt<br><br>AbsDst<br><br>ObjSz<br><br>RmSz<br><br>RelDst<br><br>RelDir<br><br>RtePln<br><br>ApOrd|
|---|---|
|All<br><br>|63.2 73.5 49.4 71.4 70.1 66.9 61.5 36.6 76.4<br><br>|
|w/o. Configuration w/o. Measurement w/o. Spatiotemporal|51.9 46.2 43.0 70.4 66.0 48.0 36.8 27.3 77.3 49.7 74.5 19.1 31.1 38.5 63.9 55.6 35.1 79.5 58.1 73.7 47.7 70.9 65.2 68.3 58.9 32.5 47.6<br><br>|

AbsDst

ApOrd

RelDst

RtePln

RelDir

ObjCt

ObjSz

RmSz

Avg

Tab. 10 presents an ablation study on how different task groups affect the model’s spatial sensing capability. Our results show that all three task groups—configuration, measurement, and spatiotemporal—are integral, as removing any one of them degrades performance. We further assess spatial reasoning using the held-out Route Plan subtask and find that the configuration group is the most influential, whereas the measurement group is the least. We attribute this outcome to the fact that route planning requires a holistic understanding of the spatial layout, which is more explicitly provided by configuration QA pairs compared to measurement and spatiotemporal tasks.

###### C.4. Examples of VSI-590K

To better illustrate VSI-590K, we provide qualitative visualization results in Figs. 19 to 25. These visualizations demonstrate that VSI-590K delivers great diversity and quality for spatial question-answering supervised fine-tuning.

- D. Cambrian-S Implementation Details In this section, we provide holistic training details of our Cambrian-S models.

###### D.1. Model Architecture

Following the original Cambrian-1 [124] and common practices in most MLLMs [78, 65], our model (both our upgraded Cambrian-1 and Cambrian-S) integrates a pre-trained vision encoder, a pre-trained language model as the decoder, and a vision-language connector to bridge these two modalities. Specifically, we employ SigLIP2-So400M [128] as the vision encoder. This encoder was trained using a combination of losses: text next-token-prediction (LocCa [131]), image-text contrastive (or sigmoid [105, 157]), and masked self-prediction (SILC [96]/TIPS [88]). For the language model, we utilize the instruction-tuned Qwen2.5 LLMs [145]. Unlike Cambrian-1, which used SVA for a deeper vision-language fusion, we employ a simpler GELU-activated [35] two-layer MLP as the vision-language connector to maintain a balance between performance and efficiency.

###### D.2. Training Data Mixture

As mentioned in Sec. 3.4, our Cambrian-S models are trained with four training stages (See Fig. 8). For the first two stages (i.e., vision-language alignment stage and image instruction tuning stage), we refer readers to Cambrian-1 [124] for the detailed training data mixture. In the third stage, we finetune the

###### Algorithm 1: QA generation pipeline for unannotated web-scrawled videos

Input: Video sequence 𝑉, valid category list Cvalid, invalid category list Cinvalid, sampling interval Δ𝑡, blur

threshold 𝜏blur, minimum object count 𝜃min, minimum 3D point count 𝜃3D, erosion kernel 𝐾erosion Output: Selected frame set F, Question-answer pairs Q

- 1 Initialize F ← ∅, Q ← ∅;
- 2 S ← SampleFrames(𝑉,Δ𝑡) ; // Sample frames at interval Δ𝑡
- 3 foreach frame 𝑓 ∈ S do

- 4 if BlurDetection( 𝑓) > 𝜏blur then

- 5 continue;

- 6 O ← GroundingDINO( 𝑓, Cvalid ∪ Cinvalid) ; // Detect objects from both category lists
- 7 if ∃𝑜 ∈ O : category(𝑜) ∈ Cinvalid then

- 8 continue;

- 9 Ovalid ← {𝑜 ∈ O : category(𝑜) ∈ Cvalid};
- 10 if |Ovalid| < 𝜃min then

- 11 continue;

- 12 M ← ∅ ; // Initialize mask set
- 13 foreach object 𝑜 ∈ Ovalid do

- 14 𝑏 ← GetBoundingBox(𝑜);
- 15 𝑚 ← SAM2( 𝑓,𝑏) ; // Generate mask using SAM2
- 16 𝑚′ ← Erode(𝑚, 𝐾erosion) ; // Apply erosion on the masks
- 17 M ← M ∪ {𝑚′};

- 18 Pmap ← VGGT( 𝑓) ; // Generate 3D point map using VGGT
- 19 P ← ∅ ; // Initialize 3D point set
- 20 foreach mask 𝑚 ∈ M do

- 21 𝑃 ← ExtractMaskedPoints(𝑚, Pmap) ; // Extract 3D points covered by mask
- 22 if |𝑃valid| ≥ 𝜃3D then

- 23 P ← P ∪ {𝑃};

- 24 if |P| > 0 then

- 25 𝑞 ← QAGenerator(P) ; // Generate QA pairs from 3D geometry
- 26 Q ← Q ∪ {𝑞};
- 27 F ← F ∪ { 𝑓};

- 28 Return F, Q;

image instruction-tuned models CambrianS-3M, and during the last stage, we conduct spatial video instruction tuning by finetuning the model on VSI-590K. CambrianS-3M is our curated video instruction tuning dataset with around 3M video QA samples, built upon a set of open-sourced video datasets (e.g., LLaVA-Video [161], ShareGPT4o [32],VideoChat2 [71], MovieChat [119], EgoIT [149], Perception Test [103], Vript [146],VideoChatGPT-Plus [85], Ego4D [47], HowTo100M [94],HD-VILA [144], HTStep [2], TimeIT [111], HowToInterlink7M [132], GUI-World [23], Video-Localized-Narratives [129], and etc.). We detail its composition in Fig. 16.

###### D.3. Training Recipe

- Stage 1: Vision-language alignment. We freeze most of the model’s parameters and train only the vision-language connector on the Cambrian-Alignment-2.5M dataset [124]. Input images are padded to a fixed resolution of 384 × 384, and the maximum sequence length is set to 2048.
- Stage 2: Image instruction tuning. We unfreeze both the vision-language connector and the LLM decoder, while keeping the vision encoder frozen. The model is then fine-tuned on the Cambrian-7M image instruction tuning dataset. Compared to Cambrian-1 [124], we adopt the AnyRes strategy [77] to enhance the model’s image understanding capabilities. Specifically, input images are resized while preserving aspect ratio, then divided into multiple 384 × 384 sub-images. This enables the model to handle images with higher and more flexible resolutions. To accommodate the increased number of visual tokens introduced by the AnyRes strategy, we extend the sequence length to 8192. Detailed training

General Video QA / Caption (81.3%)

- - WebVideoQA (0.8%)

| |
|---|

- - PerceptionTest (0.3%)

| |
|---|

- - YouCook2 (0.3%)

| |
|---|

- - FAVD (0.1%)

| |
|---|

- - ShareGPT4o (0.1%)

| |
|---|

- - MoiveChat (0.1%)

- Ego4D-HCap (0.1%)

| |
|---|

| |
|---|

| |
|---|

Egocentric QA (3.2%)

- - LLaVA-Video-178K (44.2%)

| |
|---|

- - LLaVA-Hound (8.4%)

| |
|---|

- - ShareGemini (7.3%)

| |
|---|

- - TVQA (4.0%)

| |
|---|

- - Vript (3.6%)

| |
|---|

- - NeXT-QA (3.4%)

| |
|---|

- - TGIF (2.8%)

| |
|---|

- - LSMD (2.5%)

| |
|---|

- - STAR (1.5%)

| |
|---|

- - ActivityNet (1.1%)

| |
|---|

- - ActivityNet (1.1%)

| |
|---|

- - NTURGBD (0.9%)

| |
|---|

| |
|---|

- - Ego-IT-99K (2.9%)

| |
|---|

- - EgoQA (0.3%)

| |
|---|

GUI Understanding QA (3.1%)

| |
|---|

- GUIWorld (3.1%)

| |
|---|

Fine-grained Video QA / Caption (7.7%)

CS3M

Video Reasoning (2.7%)

| |
|---|

| |
|---|

- - TimeIT (2.4%)

| |
|---|

- - Didemo (2.2%)

| |
|---|

- - Video Localized Narrative (1.5%)

| |
|---|

- - ActivityNet-RTL (1.1%)

| |
|---|

- - ActivityNet-HL (0.3%)

| |
|---|

- - HTStep (0.1%)

| |
|---|

- CLEVRER (2.7%)

| |
|---|

Text Recognition (1.3%)

| |
|---|

- TextVR (1.3%)

| |
|---|

Video Classification (0.7%)

| |
|---|

- - K700 (0.3%)

| |
|---|

- - SSV2 (0.3%)

| |
|---|

- Figure 16 | General video instruction tuning datasets of CambrianS-3M, used in Cambrian-S stage 3 &

- 4 training.

- Table 11 | Training configuration for stage 1 and stage 2.

| |Stage 1 (Vision-Language Alignment) Stage 2 (Image Instruction Tuning)|
|---|---|
|Model Vision Encoder Language Decoder VL-Connector|SigLIP2-So400M Qwen2.5-0.5B, 1.5B, 3B, 7B-Instruct 2×MLP-GELU<br><br>|
|Data Recipe Data Image Resolution # of Tokens per Image<br><br>|Cambrian-Alignment-2.5M Cambrian-7M Pad (384×384) AnyRes (Up to 9 sub-images) 729 Up to 7,290|
|Training Recipe Max Sequence Length Trainable Module Learning Rate Batch Size Warmup Ratio<br><br>|2,048 8,192 VL-Connector VL-Connector & LLM<br><br>1 × 10−3 1 × 10−5 512 256 0.06 0.03|

configurations for stage 1 and 2 are provided in Tab. 11.

- Table 12 | Training recipe for Cambrian-S stage 3 and stage 4.

| |Stage 3 (General Video Instruction Tuning) Stage 4 (Spatial Video Instruction Tuning)|
|---|---|
|Model Vision Encoder Language Decoder VL-Connector|SigLIP2-So400M Qwen2.5-0.5B, 1.5B, 3B, 7B-Instruct 2×MLP-GELU<br><br>|
|Data Recipe Data Source Video Frame Resolution Frame Sampling Strategy # Frames per Video # Tokens per Video Frame<br><br>|CambrianS-3M VSI-590K + 590K general Video IT data (sampled from CambrianS-3M)<br><br>Pad (384×384) Pad (384×384) Uniform Uniform 64 128 64 64|
|Training Recipe Max Sequence Length Trainable Modules Learning Rate Global Batch Size Warmup Ratio|8,192 16,384<br><br>VL-Connector and LLM 1 × 10−5 256 0.03<br><br>|

- Stage 3: General video instruction tuning. To equip the model with general video understanding capabilities, we perform video instruction tuning on a mixture of curated CambrianS-3M video data and sampled image instruction data from Cambrian-7M. As in previous stages, the vision encoder remains frozen, and the remaining modules are fine-tuned. For image data, we reuse the sampling strategy from stage 2. For video data, we uniformly sample 64 frames per video, resize them to 384 × 384, and further downsample their feature maps to 8 × 8, i.e., 64 tokens per frame.
- Stage 4: Spatial video instruction tuning. The final stage focuses on enhancing the model’s spatial reasoning capabilities by fine-tuning on our proposed VSI-590K. To preserve general video and image understanding, we mixed 590K video samples from CambrianS-3M and 120K image samples from

Cambrian-7M. Training settings are mostly consistent with stage 3, except for two key changes: (1) we increase the number of frames per video to 128, and (2) we extend the sequence length to 16,384, both to support richer temporal modeling. Detailed configurations for stage 3 and 4 are listed in Tab. 12.

- D.4. Infrastructure

All models in this paper are trained using TPU v4 Pods with the TorchXLA framework. To support large-scale video instruction tuning—where long sequence lengths introduce prohibitive computational and memory costs—we leverage GSPMD [143] and FlashAttention [34] implemented by Pallas.

GSPMD is an automatic parallelization system designed for flexible and user-friendly large-scale distributed training. It allows users to write training code as if for a single device, and then scale effortlessly across hundreds of devices with minimal changes. Our training framework is based on TorchXLA and GSPMD to shard data, model parameters, activations, and optimizer states across multiple devices. This reduces the peak memory usage and improves training throughput.

To accommodate long sequences, we integrate FlashAttention backed by Pallas, which significantly reduces TPU HBM (V-Mem) usage under long-context inputs. This enables us to scale the input sequence length up to 16,384 tokens for the 7B model on a TPU v4-512 Pod.

- E. Cambrian-S Additional Results

- E.1. Detailed Evaluation Setups

We describe the evaluation settings used for most image and video benchmarks, excluding VSI-SUPER. For image inputs, following the any-resolution design adopted in our training pipeline, each image is resized while preserving its aspect ratio, and its resolution is maximized so that it can be partitioned into at most nine 384×384 sub-images. For video inputs, we apply uniform frame sampling with a fixed number of frames. Specifically, checkpoints from stage 1 and stage 2 are evaluated with 32 uniformly sampled frames, while those from stage 3 and stage 4 use 64 and 128 frames, respectively.

###### E.2. Detailed Performance on Image and Video Benchmarks

Tab. 13 and Tab. 14 detail the performance of all our checkpoints (from stage 1 to stage 4 and from 0.5B to 7B) on image-based and video-based MLLM benchmarks, respectively. For image benchmarks, we report the results on MME [155], MMBench [81], SeedBench [66], GQA [55], ScienceQA [114], MMMU [156], MathVista [83], AI2D [59], ChartQA [91], OCRBench [82], TextVQA [165], DocVQA [92], MMVP [125], RealworldQA [141], and CVBench [124], following Cambrian-1’s grouping strategy.

###### E.3. Contributions from Image-based and Video-based Instruction Tuning

To elaborate on the respective contributions of image-based and video-based instruction tuning to a model’s final video understanding capabilities, we conducted a series of experiments. These experiments employed varying proportions of image and video data during the finetuning stages, and we observed the resulting performance trends across diverse video benchmarks.

More specifically, for the initial image MLLM training, we randomly sampled 1M, 4M, and 7M image question-answering (QA) pairs from Cambrian-7M to train distinct models. Subsequently, for videospecific finetuning, we randomly sampled 25%, 50%, 75%, and 100% of video QA pairs from LLaVAVideo-178K (∼1.6M data samples in total) to perform video-only finetuning on each of these pretrained image MLLMs. The hyperparameters for image instruction tuning and video finetuning were maintained as detailed in Table 11 and Table 12, respectively. The experimental results, presented in Table 15, yield the following observations:

• Models trained with more image data do not inherently outperform those trained with less when evaluated on video benchmarks without finetuning. As indicated in the table, direct evaluation on video benchmarks

###### Table 13 | Detailed results of Cambrian-S checkpoints on image MLLM benchmarks.

General Knowledge OCR & Chart Vision-Centric

RealworldQA

2DCV-Bench

3DCV-Bench

MMathVista

OCRBench

VMMMU

TextVQA

ChartQA

DocVQA

MMVP

PMME

ISEED

MMB

AI2D

ISQA

GQA

Avg

Avg

Avg

Avg

Method

Open-source Models

Mini-Gemini-HD-8B 72.7 1606.0 72.7 73.2 64.5 55.7 75.1 37.3 37.0 73.5 62.9 59.1 47.7 70.2 74.6 51.5 18.7 62.1 62.2 63.0 LLaVA-NeXT-8B 72.5 1603.7 72.1 72.7 65.2 55.6 72.8 41.7 36.3 71.6 63.9 69.5 49.0 64.6 72.6 56.6 38.7 60.1 62.2 65.3 Cambrian-1-8B 73.1 1,547.1 75.9 74.7 64.6 61.3 80.4 42.7 49.0 73.0 71.3 73.3 62.4 71.7 77.8 65.0 51.3 64.2 72.3 72.0 Cambrian-S-7B

- Stage 1 11.5 209.9 29.6 5.6 0.1 2.5 3.1 2.2 2.9 1.7 7.4 0.9 27.6 0.9 0.1 0.9 0.0 2.7 0.8 0.0

- Stage 2 74.9 1604.6 79.0 76.3 64.0 63.9 83.7 48.7 45.3 78.1 79.1 78.9 67.6 79.2 90.6 66.3 53.3 67.7 70.0 74.0

- Stage 3 74.4 1583.9 79.7 76.4 62.4 60.4 82.2 46.2 36.1 77.0 75.5 75.3 64.0 77.1 85.6 67.0 58.0 66.1 71.8 72.3

- Stage 4 74.8 1598.4 80.4 77.0 61.8 64.6 82.7 48.0 50.6 76.9 75.2 74.7 64.8 76.6 84.8 70.5 60.0 64.8 74.3 83.0 Cambrian-S-3B

- Stage 1 8.3 9.3 31.7 1.0 0.0 0.9 0.9 0.9 0.1 1.7 7.0 0.0 28.1 0.0 0.1 0.7 0.0 2.1 0.7 0.0

- Stage 2 71.9 1524.6 74.8 74.2 62.1 55.5 78.7 42.8 27.8 72.7 72.0 69.8 63.9 71.5 82.7 59.0 37.3 62.4 65.6 70.7

- Stage 3 72.1 1495.7 76.5 75.1 61.8 58.5 79.4 42.2 41.3 71.2 69.6 68.0 61.3 69.6 79.4 62.5 46.0 61.2 70.6 72.4

- Stage 4 71.5 1485.6 76.0 75.1 60.8 58.7 78.7 42.1 43.0 70.9 69.6 70.0 60.5 68.7 79.1 65.6 50.0 60.1 76.1 76.3 Cambrian-S-1.5B

- Stage 1 11.7 282.1 28.6 0.8 3.2 3.8 6.9 4.2 1.4 2.6 7.9 1.0 27.8 1.4 1.5 0.7 0.0 0.0 2.9 0.0

- Stage 2 68.5 1417.3 71.3 71.2 60.6 50.9 75.5 41.1 20.8 66.1 68.0 64.8 59.9 68.8 78.6 54.4 39.3 59.7 60.3 58.3

- Stage 3 68.1 1423.2 70.5 72.1 58.7 52.6 72.4 40.8 32.3 64.8 64.2 59.5 57.6 66.7 72.9 54.6 40.0 59.9 60.7 57.8

- Stage 4 68.0 1394.4 70.1 73.5 58.7 54.7 72.3 42.0 39.7 64.7 65.6 63.1 58.0 66.6 74.8 59.2 43.3 54.5 62.6 76.3 Cambrian-S-0.5B

- Stage 1 10.1 379.6 10.7 9.0 1.8 6.2 8.4 8.9 1.9 5.5 3.0 0.2 7.9 2.0 1.9 10.9 0.7 10.6 20.1 12.3

- Stage 2 57.7 1124.3 56.6 61.7 56.1 38.6 61.5 31.0 10.5 51.5 56.0 51.1 51.0 58.7 63.1 41.2 23.3 51.8 45.6 44.1

- Stage 3 58.6 1200.0 55.8 63.5 55.3 41.2 62.7 32.6 18.0 51.4 52.1 46.6 46.8 56.0 59.1 45.5 22.0 52.8 52.2 54.9

- Stage 4 60.0 1190.8 60.7 66.4 53.5 44.0 63.4 34.0 28.6 50.1 52.6 48.0 47.1 56.6 58.6 48.7 26.0 51.1 51.6 66.2

###### Table 14 | Detailed results of Cambrian-S checkpoints on video MLLM benchmarks.

|Model<br><br>|Base LLM|VSI-Bench<br><br>Tomato<br><br>HourVideo<br><br>MMEVideo<br><br>EgoSchema<br><br>MMMUVideo<br><br>LongVBench<br><br>MVBench<br><br>Percept.Test|
|---|---|---|
|Cambrian-S-7B<br><br>Stage 1<br>Stage 2<br>Stage 3<br>Stage 4<br>|Qwen2.5-7B<br><br>|21.4 21.0 27.5 44.3 42.9 11.3 32.3 43.9 44.4 24.6 20.1 31.3 52.3 47.5 28.1 51.1 49.2 53.5 35.7 30.3 38.9 62.8 76.9 38.3 56.7 66.3 70.8 67.5 27.9 36.5 63.3 76.3 38.3 59.4 64.8 69.8|
|Cambrian-S-3B<br><br>Stage 1<br>Stage 2<br>Stage 3<br>Stage 4<br>|Qwen2.5-3B<br><br>|0.7 16.5 0.7 15.9 19.5 8.4 23.8 30.6 18.6<br><br>22.3 21.3 31.7 49.4 42.2 26.0 48.7 44.5 47.0<br>23.3 26.3 35.9 58.9 73.4 27.1 52.0 61.0 65.7 57.3 26.0 36.8 60.1 73.6 26.3 52.3 60.2 65.9<br>|
|Cambrian-S-1.5B<br><br>Stage 1<br>Stage 2<br>Stage 3<br>Stage 4<br>|Qwen2.5-1.5B<br><br>|21.1 23.5 26.2 40.1 33.0 18.7 38.5 40.8 45.2<br>22.6 24.6 34.4 47.8 38.2 20.7 46.9 45.3 49.8<br>23.4 23.1 33.2 56.1 67.8 28.6 49.4 58.2 63.6 54.8 22.2 31.2 56.4 69.0 25.0 50.2 57.1 63.2<br>|
|Cambrian-S-0.5B<br><br>Stage 1<br>Stage 2<br>Stage 3<br>Stage 4<br>|Qwen2.5-0.5B<br><br>|16.7 23.6 23.4 26.4 21.5 13.1 25.0 34.3 37.0 19.6 20.0 27.9 37.4 29.7 17.3 39.0 40.2 46.3 18.8 23.9 29.5 41.8 63.8 16.7 44.9 50.7 56.1 50.4 23.8 28.1 44.0 62.4 15.9 43.8 51.8 56.0|

reveals comparable performance across all three models, which were initially trained on 1M, 4M, and 7M image datasets, respectively.

- • Finetuning on video data can be generally beneficial for models pretrained with larger image datasets, though not universally. When all models were finetuned on 100% video data, the model initially trained on 7M images outperformed the other two on 5 out of 9 video benchmarks (specifically, HourVideo, VideoMME, EgoSchema, LongVideoBench, and Perception Test).
- • Incorporating video data into the training process consistently benefits performance across all video benchmarks. We observed that finetuning an image-based MLLM with video data, even a small portion such as 25%, improved its performance on all evaluated video benchmarks.

###### Table 15 | Video MLLM performance trained with different proportions of image and video data.

|Image data|Video data<br><br>|VSI-Bench<br><br>Tomato<br><br>HourVideo<br><br>MMEVideo<br><br>EgoSchema<br><br>MMMUVideo<br><br>LongVBench<br><br>MVBench<br><br>Percept.Test|
|---|---|---|
|Chance-Level|-<br><br>|34.0 22.0 20.0 25.0 20.0 14.0 25.0 27.3 33.3<br><br>|
|1M<br><br>|0% 25% 50% 75% 100%|26.0 20.2 32.5 52.1 46.9 32.0 51.4 50.5 54.2<br><br>32.4 25.4 36.2 60.4 47.0 40.1 53.5 57.0 61.9<br><br>33.3 27.2 36.2 61.7 47.1 40.1 53.2 59.2 64.3<br><br>32.7 28.8 34.4 60.7 48.7 37.7 53.3 59.5 66.3<br><br>34.4 28.4 35.1 61.3 48.9 39.6 53.0 60.1 67.5<br>|
|4M<br><br>|0% 25% 50% 75% 100%|26.7 20.5 31.8 53.1 44.8 32.0 52.1 51.5 54.9<br><br>32.3 26.7 37.0 61.3 45.0 38.6 53.1 57.6 61.9<br><br>31.9 27.4 37.2 61.9 45.7 38.1 54.2 59.5 65.2<br><br>33.8 27.9 36.2 61.1 47.3 40.9 53.1 60.1 67.0<br><br><br>33.8 28.0 35.5 60.5 50.2 40.2 52.2 60.5 67.7|
|7M<br><br>|0% 25% 50% 75% 100%|25.8 18.9 31.6 53.7 48.1 31.9 52.5 51.4 55.4 31.5 24.6 36.7 61.3 48.8 37.7 54.7 58.3 62.3 31.4 27.6 36.6 61.0 49.0 37.9 53.6 59.7 65.6<br><br>31.8 27.0 35.7 61.8 50.7 38.0 53.0 60.2 67.9<br><br>32.6 27.7 37.3 62.1 52.4 39.4 54.3 60.6 68.8<br>|

LongVBench

Percept.Test

MMMUVideo

EgoSchema

HourVideo

VSI-Bench

MMEVideo

MVBench

Tomato

• Increasing the amount of video data used for finetuning does not guarantee consistent performance improvements across all benchmarks. While video finetuning is generally advantageous, some benchmarks (e.g., VideoMME, VSI-Bench, Tomato) do not show further gains with more video data. For instance, models finetuned with 100% video data exhibited performance on par with those finetuned with only 25% video data on the VideoMME benchmark. Only EgoSchema, MVBench, and Perception Test demonstrated consistent benefits from increased video data, a phenomenon we hypothesize is related to the underlying video distribution of the training videos.

###### E.4. On the Trade-off between Spatial Sensing and General Video Understanding

65

80

70.0

67.5

60

70

65.0

EgoSchemaPerf.(%)

VideoMMEPerf.(%)

VSI-BenchPerf.(%)

55

62.5

60

60.0

50

57.5

50

45

55.0

| |VSI-Only FT| |
|---|---|---|
| |Mixed FT| |

52.5

40

40

50.0

0.5 1.5 3.0 7.0 Model Size (B)

0.5 1.5 3.0 7.0 Model Size (B)

0.5 1.5 3.0 7.0 Model Size (B)

###### Figure 17 | On the trade-off between spatial-sensing and general video understanding.

In Sec. 3.3, we compare model performance when fine-tuned either on VSI-590K alone or on a mixture of VSI-590K and general video data. We observe that fine-tuning on VSI-590K alone consistently yields higher performance on spatial sensing tasks, whereas mixed-data fine-tuning offers a better balance between spatial sensing and general video understanding. To further explore this trade-off across model scales, we conduct fine-tuning after stage 3 using either VSI-590K alone or the mixed dataset, under four different model sizes: 0.5B, 1B, 3B, and 7B parameters. We then evaluate these models on both general video understanding and spatial sensing benchmarks, as shown in Fig. 17.

The results confirm that the previous conclusion holds across all scales: VSI-590K-only fine-tuning excels at spatial sensing, while mixed-data fine-tuning provides a better overall balance. Notably, however,

the performance gap on VSI-Bench narrows as model size increases. We attribute this to the greater capacity of larger models to learn and retain diverse capabilities. This trend suggests that scaling to even larger models may further mitigate the spatial sensing performance drop typically observed when fine-tuning with mixed data.

###### F. Predictive Sensing

###### F.1. Latent Frame Prediction Implementation Details

Latent frame prediction head. As shown in Algorithm 2, our next-frame prediction head is a simple two-layer MLP with GELU activation [51], running in parallel with the MLLM’s original language model head. The output dimension is set to 1152, matching the output dimension of our vision encoder (i.e., siglip2-so400m-patch14-384).

###### Algorithm 2: Latent frame prediction (LFP) head architecture (in PyTorch style).

LFPHead( Sequential(

- (0): Linear(in_features=3584, out_features=3584, bias=True)

- (1): GELU(approximate=none)

- (2): Linear(in_features=3584, out_features=1152, bias=True) )

)

On the balance between LFP and instruction tuning losses. As mentioned in Sec. 4.1, to build the model’s internal world model, we slightly modify our stage 4, introducing two auxiliary losses (i.e., cosine distance and mean-squared error) to optimize the next frame prediction objectiveness. A coefficient is applied to balance the LFP loss against the instruction tuning loss, which we ablate in Tab. 16.

- Table 16 | Evaluation results across different benchmarks with varying LFP loss weights. Our default setup (0.1 loss coefficient) is highlighted in gray .

|LFP loss coeffcient<br><br>|VSI-Bench VideoMME EgoSchema Perception Test|
|---|---|
|0.0 (i.e., No LFP Loss)|67.5 63.4 76.8 69.9<br><br>|
|0.1 0.5 1.0<br><br>|66.1 63.9 76.9 69.7 60.8 63.6 77.2 66.4 56.6 61.0 72.9 65.1<br><br>|

- F.2. Memory Framework Design for VSI-SUPER Recall

As introduced in main paper (and shown in Algorithm 3), our predictive memory mechanism comprises three distinct memory levels (𝑀𝑠, 𝑀𝑙, 𝑀𝑤) and four key transition functions governing their interaction: Sensory Streaming, Memory Compression, Memory Consolidation, and Retrieval. This section details the implementation of these functions.

Basic memory units. For our implementation, we utilize the encoded key-value pairs from each Large Language Model (LLM) layer as the basic memory units. This choice, rather than using output latent features from a vision encoder or vision-language connector, allows us to fully leverage the LLM’s internal capabilities for memory construction without requiring external modules. This design decision will be elaborated upon in subsequent sections.

Streaming sensing. Each incoming frame is initially processed independently by the vision encoder and the vision-language connector with a window size of 𝑊𝑠. Subsequently, it is further encoded by the LLM, referencing selected previous frames. The key-value pairs from these preceding frames, cached in the Sensory memory buffer (𝑀𝑠), provide the necessary context for this encoding step.

Surprise-based memory compression. In the meantime of encoding a single frame, we assess its “surprise” level. This is achieved by calculating the difference between the model’s prediction for the current frame and the actual ground truth observation (both in the latent feature space). When a frame of timestamp 𝑡 is moved from the sensory memory buffer 𝑀𝑠 to the long-term memory 𝑀𝑙, if it is deemed non-surprising (i.e., its surprise score is below a predefined threshold 𝑇𝑠), we will downsample its’ key-value pairs by a factor of 2 along the spatial (𝐻 ×𝑊) dimension. This surprise-based compression mitigates redundancy in the information stored within 𝑀𝑙.

Surprise-based memory consolidation. Long-term memory 𝑀𝑙 is initialized with a predefined budget size 𝐵𝑙𝑜𝑛𝑔 (e.g., 32,768 tokens). When the volume of memory tokens surpasses this budget, we apply a surprise-based consolidation function to 𝑀𝑙 to ensure it remains within the allocated limit. Our consolidation function is straightforward yet effective: we identify the surprise score associated with each frame in 𝑀𝑙. Then, the frame with the lowest surprise score is removed (or “forgotten”). Then, we merge or drop some of these frames according to their surprise scores (we tried three different strategies here: 1. forget the oldest memory, 2. forget the least surprise memory, and 3. forget the least surprise memory while merging adjacent surprise memories if any adjacent surprise memories exist). This process is iterated until the total size of 𝑀𝑙 falls below the budget.

Retrieval. Upon receiving a user query 𝑞, we first retrieve the most relevant frames from the long-term memory (𝑀𝑙) to construct the working memory (𝑀𝑤). This 𝑀𝑤 then serves as the context for answering the user’s query. To perform this retrieval efficiently without resorting to external modules, we utilize the inherent similarity measurement capabilities of the LLM’s attention mechanism. Specifically, for each transformer layer, the user query 𝑞 is transformed into the attention mechanism’s query feature space. We then compute the similarity between this query feature and the key features of each frame stored in 𝑀𝑙. Similarity is measured using cosine distance, and for simplicity, multi-head features are treated as a single feature. The 𝑘 frames with the highest similarity scores have their key-value pairs selected and utilized by the attention mechanism to further encode the user query.

###### Algorithm 3: Memory framework design for VSI-SUPER Recall.

Input: Frames { 𝑓1, . . . , 𝑓𝑇}, User query 𝑞 Input: Encoder E, Decoder D, Surprise Estimator S, Surprise threshold 𝜏 Input: Compression function C, Consolidation function G, Retrieval function R Input: Sensory memory M𝑠 ← ∅ with budget 𝐵𝑠, Long-term memory M𝑙 ← ∅ with budget 𝐵𝑙, Working

memory M𝑤 ← ∅

- 1 for 𝑡 ← 1 to 𝑇 do

- 2 𝑧𝑡 ← E( 𝑓𝑡,M𝑠);
- 3 M𝑠 ← M𝑠 ∪ {𝑧𝑡} ; // Streaming sensing
- 4 𝑠𝑡 ← S( 𝑓𝑡,M𝑠) ; // Surprise estimation
- 5 while |M𝑠| > 𝐵𝑠 do

- 6 Dequeue 𝑧old from M𝑠;
- 7 𝑚 ← 1[𝑠𝑡 ≥ 𝜏] · 𝑧old + 1[𝑠𝑡 < 𝜏] · C(𝑧old) ; // Selective compression
- 8 M𝑙 ← M𝑙 ∪ {𝑚};
- 9 if |M𝑙| > 𝐵𝑙 then

- 10 M𝑙 ← G(M𝑙) ; // Memory consolidation

- 11 M𝑤 ← R(𝑞,M𝑙) ; // Retrieve working memory
- 12 𝑎ˆ ← D(𝑞,M𝑤) ; // Answering query with M𝑤
- 13 return 𝑎ˆ

F.3. Agentic Framework Design for VSI-SUPER Count

###### Algorithm 4 presents our agentic framework for the VSI-SUPER Count task. Similar to the memory design in Algorithm 3, we encode sensory frames using a sliding window approach with a window size

of 𝑊𝑠. The latent frame prediction module continuously estimates the expected next frame and computes the prediction error to quantify how "surprise" the actual next frame is. As new frame arrivs, the oldest frames that exceed the sensory memory window are dequeued and stored in the long-term memory. If a dequeued frame is deemed “surprising” (i.e., its prediction error exceeds a predefined threshold 𝜏),

which may indicate a scene or spatial boundary, we trigger a query response using the accumulated long-term memory and reset it afterward. The generated response is then stored in the answer memory bank. The final answer is computed as the aggregation of all intermediate answers stored in this bank.

###### Algorithm 4: Agentic framework design for VSI-SUPER Count task.

Input: Frames { 𝑓1, . . . , 𝑓𝑇}, user query 𝑞 Input: Encoder E, Decoder D, Surprise Estimator S, threshold 𝜏 Input: Sensory memory M𝑠 ← ∅ with budget 𝐵𝑠 Input: Long-term memory M𝑙 ← ∅, Answer memory bank MAns ← ∅

- 1 for 𝑡 ← 1 to 𝑇 do

- 2 𝑧𝑡 ← E( 𝑓𝑡,M𝑠);
- 3 M𝑠 ← M𝑠 ∪ {𝑧𝑡} ; // Streaming sensing
- 4 𝑠𝑡 ← S( 𝑓𝑡,M𝑠) ; // Surprise estimation
- 5 if |M𝑠| > 𝐵𝑠 then

- 6 Remove oldest 𝑧old from M𝑠;
- 7 M𝑙 ← M𝑙 ∪ {𝑧old} ; // Store to long-term memory

- 8 if 𝑠𝑡 ≥ 𝜏 then

- 9 𝑎ˆ ← D(𝑞,M𝑙) ; // Answer query using long-term memory
- 10 MAns ← MAns ∪ {𝑎ˆ};
- 11 M𝑙 ← ∅ ; // Reset long-term memory

- 12 return Sum(MAns)

###### F.4. Comparisons with Existing Long-video Methods

We compare our method (both surprise-driven memory and agentic framework) with existing methods designed for long-video understanding, in Tab. 17. Specifically, all experiments here are conducted with our LFP-finetuned Cambrian-S-7B, with a different strategy to handle the ever-expanding visual sensory input. For MovieChat, we follow the official implementation in [119], maintain a fixed-size long-term memory bank, and set the long-term and short-term memory budgets to 64 and 16, respectively. For Flash-VStream [159], as its abstract memory module introduces additional parameters and requires a dedicated training process, we only implement the three remaining memory components (i.e., spatial memory, temporal memory, and retrieved memory), and keeping all other hyperparameters aligned with the default setup.

###### Table 17 | Compare our framework with existing long-video methods on VSI-SUPER.

|Eval Setups<br><br>|VSR (Duration in Mins.) 10 30 60 120 240<br><br>|VSC (Duration in Mins.) 10 30 60 120|
|---|---|---|
|MovieChat Flash-VStream<br><br>|18.3 21.7 16.7 26.7 25.6 28.3 33.3 23.3 28.3 31.7<br><br>|0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0|
|Ours<br><br>|45.0 41.7 40.0 40.0 40.0|40.6 42.0 35.0 34.0|

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Which of the following correctly represents the order in which the Stitch appeared in the video? A. Stove, Trash bin, Refrigerator, Counter B. Trash bin, Refrigerator, Counter, Stove C. Stove, Counter, Refrigerator, Trash bin D. Trash bin, Stove, Counter, Refrigerator

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Which of the following correctly represents the order in which the Hello Kitty appeared in the video? A. Nightstand, Bed, Crib, Blue bench B. Blue bench, Crib, Nightstand, Bed C. Bed, Nightstand, Blue bench, Crib D. Blue bench, Bed, Crib, Nightstand

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Which of the following correctly represents the order in which the Golden Retriever appeared in the video? A. Bed, Table, Chest of drawers, Floor B. Table, Chest of drawers, Bed, Floor C. Chest of drawers, Floor, Table, Bed D. Floor, Bed, Chest of drawers, Table

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Which of the following correctly represents the order in which the white Ragdoll cat appeared in the video? A. Ground, Trash bin, Bench, Table B. Table, Bench, Ground, Trash bin C. Ground, Trash bin, Table, Bench D. Trash bin, Bench, Table, Ground

- Figure 18 | More examples of our VSI-SUPER Recall benchmark. Note that only edited frames are visualized.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Absolute Direction (Object)

Standing by the backpack, looking toward the table, how far counterclockwise in degrees must I turn to see the trash bin? Answer: 334.09

Absolute Distance

Measuring from the closest points of each, how far apart are the chair and the door in meters? Answer: 2.32

Absolute Distance Considering the chair and the door, which object’s longest edge is the shorter?

- A. Door
- B. Chair

Figure 19 | Examples of VSI-590K (Annotated Real Video).

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Object Appearance Order

Determine the initial appearance order of these categories in the video: door, chair, lamp, refrigerator. A. refrigerator, door, lamp, chair B. refrigerator, chair, door, lamp

- C. refrigerator, chair, lamp, door D. door, chair, lamp, refrigerator

Absolute Size

Provide the longest side’s length for the door in inches. Answer: 72.00

Room Size

Indicate the room’s dimensions in square feet. If there’s more than one room, estimate their total size. Answer: 232.76

- Figure 20 | Examples of VSI-590K (Annotated Real Video).

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Relative Direction (Object Perspective)

Facing the door while standing near the window, in which of the following positions is the board relative to me: front-left, front-right, back-left, or back-right? Use Cartesian quadrants, with me at the origin looking toward positive y-axis

- A. Back-right
- B. Front-right
- C. Front-left
- D. Back-left Relative Distance (Object Perspective)

Identify the object among (bookcase, chair, board, door) that is closest to the window based on the shortest distance between their closest points. Choose the nearest instance if several exist.

- A. Bookcase
- B. Chair
- C. Board
- D. Door

- Figure 21 | Examples of VSI-590K (Annotated Real Video).

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Relative Distance (Object Perspective)

If I am standing by the dresser and facing the chair, is the closet to my left, right, or back? An object is to my back if I would have to turn at least 135 degrees in order to face it.

- A. Left
- B. Right
- C. Back

- Figure 22 | Examples of VSI-590K (Annotated Simulated Video).

[Figure 127]

Relative Direction (Object Perspective)

With the toilet beside me and facing the cabinet, is the lamp positioned front-left, front-right, back-left, or back-right relative to me, based on Cartesian plane quadrants?

- A. Back-right
- B. Front-right
- C. Front-left
- D. Back-left Relative Distance (Object Perspective)

Identify the object among (bookcase, chair, board, door) that is closest to the window based on the shortest distance between their closest points. Choose the nearest instance if several exist.

- A. Bookcase
- B. Chair
- C. Board
- D. Door

- Figure 23 | Examples of VSI-590K (Annotated Simulated Video (Frame)).

[Figure 128]

Object Counting (Relative) If counted, would chairs be fewer than, more than, or equal in number to tables?

- A. Fewer
- B. More
- C. Equal

Relative Direction (Camera Perspective) Through the camera’s lens, is the sink captured on the left or right part of the scene?

- A. Right
- B. Left

- Figure 24 | Examples of VSI-590K (Unannotated Real Video (Frame)).

[Figure 129]

Object Counting (Absolute)

What would be the count if you tallied all the chairs? Answer: 6

Relative Distance (Camera Perspective) In terms of proximity to the camera, which is closer: a table or a sofa?

- A. Table
- B. Sofa

- Figure 25 | Examples of VSI-590K (Unannotated Real Video (Frame)).

