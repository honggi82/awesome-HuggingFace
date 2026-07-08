## VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos

### Hanoona Rasheed1, Abdelrahman Shaker1, Anqi Tang1, Muhammad Maaz1 Ming-Hsuan Yang2,3, Salman Khan1,4, Fahad Shahbaz Khan1,5

1MBZUAI 2University of California Merced 3Google Research 4Australian National University 5Linköping University https://mbzuai-oryx.github.io/VideoMathQA

arXiv:2506.05349v2[cs.CV]24Jun2025

### Abstract

Mathematical reasoning in real-world video settings presents a fundamentally different challenge than in static images or text. It requires interpreting fine-grained visual information, accurately reading handwritten or digital text, and integrating spoken cues, often dispersed non-linearly over time. In such multimodal contexts, success hinges not just on perception, but on selectively identifying and integrating the right contextual details from a rich and noisy stream of content. To this end, we introduce VIDEOMATHQA, a benchmark designed to evaluate whether models can perform such temporally extended cross-modal reasoning on videos. The benchmark spans 10 diverse mathematical domains, covering videos ranging from 10 seconds to over 1 hour. It requires models to interpret structured visual content, understand instructional narratives, and jointly ground concepts across visual, audio, and textual modalities. We employ graduate-level experts to ensure high quality, totaling over 920 man-hours of annotation. To reflect real-world scenarios, questions are designed around three core reasoning challenges: direct problem solving, where answers are grounded in the presented question; conceptual transfer, which requires applying learned methods to new problems; and deep instructional comprehension, involving multi-step reasoning over extended explanations and partially worked-out solutions. Each question includes multi-step reasoning annotations, enabling fine-grained diagnosis of model capabilities. Through this benchmark, we highlight the limitations of existing approaches and establish a systematic evaluation framework for models that must reason, rather than merely perceive, across temporally extended and modality-rich mathematical problem settings. Our benchmark and evaluation code are available at: VideoMathQA.

Reasoning Types

Mathematic Concepts

Duration & Difficulty

[Figure 1]

[Figure 2]

Chart

[Figure 3]

Problem Focused

Arithmetic

Puzzle

Video presents question, apply world knowledge

10

Ranging from 10 sec to over an hour Short-term and long-range reasoning

[Figure 4]

Geo-Area Geo-Angle

Concepts

[Figure 5]

[Figure 6]

Concept Transfer

VideoMathQA

Learn from video, reason and then apply to answer

Difficulty levels for varying reasoning depths

Leaderboard

[Figure 7]

Counting Statistics

GPT-04-mini Qwen-2.5VL-72B InternVL-3-78B

[Figure 8]

Easy: 13% Medim: 30% Hard: 57%

[Figure 9]

[Figure 10]

Deep Comprehension

[Figure 11]

[Figure 12]

Geo-Length

Topology

Trace explanation in long video, reason and answer

Graph Theory

[Figure 13]

Needle-in-a-multimodal-haystack in Cross-Modal Reasoning Across Time: Visual, Text and Audio

Figure 1: The foundation of our benchmark is the “needle-in-a-multimodal-haystack” challenge, capturing the core difficulty of cross-modal reasoning across time from visual, textual, and audio streams. Built on this, VideoMathQA categorizes each question along four key dimensions: reasoning type, mathematical concept, video duration, and difficulty.

### 1 Introduction

As opposed to mathematical reasoning benchmarks developed for static-image, video settings presents a set of unique challenges. In educational and instructional videos, key information is conveyed through evolving diagrams, handwritten or digital notations, and spoken explanations that unfold non-linearly across time. A reasoning model must therefore sift through high-resolution frames, align visual representations with subtitles or voice-over, and integrate disparate cues into a coherent problem-solving pipeline. This

‘needle-in-a-multimodal-haystack’ problem requires not only accurate perception (e.g., frame-aware OCR of equations) but also precision in symbolic manipulation and multi-step inference, where missing a single visual or verbal cue can lead to incorrect conclusions.

Problem Focused Concept Transfer Deep Comprehension

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(5 minutes) (22 minutes)

(12 minutes)

Question: Based on the lecture, can you complete the rest of the solution?

Question: Determine the number of possible triangles in the final example.

Question: Using data from the video on the cost of various healthcare procedures, if someone in the U.S. underwent all four treatments shown, what would be the total cost, and how much more would that be compared to having the same procedures in Spain? Select the closest pair in USD, in the order: average U.S. cost and difference with Spain?

[Context: Requires understanding the context, identify what is being asked, localize and interpret the partially worked out solution, use these to complete the solution.]

[Context: The demonstrated method or solution approach must be applied to a similar but newly posed problem. Locating the correct figure is vital to answering the question correctly.]

Category: Arithmetic and Calculus Options: A) B) C) D) E)

Category: Counting Options: A) 29 B) 28 C) 24 D) 30 E) 27 Steps:

Category: Chart Reading

Options:

A) 77065, 60029 B) 40000, 30000 C) 77065, 59629 D) 76465, 51893 E) 92900, 67728

###### Steps:

- 1) The problem is to calculate the value of the double integral of the function f(x, y) = x²y³ over the rectangular region defined by 1 ≤ x ≤ 2 and 1 ≤ y ≤ 3.
- 2) For a double integral over a rectangular domain, the problem can be solved by computing an iterated integral. The order of integration is a matter of choice and does not affect the final result.
- 3) First, we integrate with respect to 𝑥, treating 𝑦 as a constant. The integral of 𝑥 with respect to 𝑥 from 1 to 2 is computed.
- 4) The integral of 𝑥 with respect to 𝑥 is . Evaluating this from 1 to 2 gives − = .

- 5) This result is multiplied by 𝑦 , giving 𝑦 .

We now integrate this expression with respect to 𝑦 from 1 to 3.

- 6) The integral of 𝑦 with respect to 𝑦 is . Evaluating from 1 to 3 gives − = 20.

- 7) Multiplying 20 by gives .

- 8) Thus, the value of the double integral over the given region based on the lecture is C) .

Steps:

- 1) The problem involves determining the number of triangles in a figure composed of multiple square boxes.
- 2) Each square box individually contains 8 triangles, determined by writing numbers 1 to 4 and multiplying the maximum number by 2.
- 3) For three vertically connected boxes, each box contributes 8 triangles, totaling 24 triangles.
- 4) When two boxes are joined, an additional 2 triangles are formed at the joining area. With three boxes, there are two such joining areas, contributing 4 additional triangles.
- 5) Thus, the total from the three boxes and their connections is 24 (from individual boxes) + 4 (from connections) = 28 triangles.
- 6) Additionally, when three boxes are connected, a large triangle is formed across all three boxes, contributing 1 more triangle.
- 7) Therefore, the final count is 28 (from previous steps) + 1 (large triangle) = 29 triangles in total.

- 1) Early in the video, the focus is on healthcare spending as a share of GDP and the roles of public vs. private funding. Toward the middle, the emphasis shifts to the cost of individual treatments, which is key to answering this question.
- 2) Video shows a series of bar charts comparing average cost of 4 treatments in US to other countries.
- 3) U.S. treatment costs are as follows: Angioplasty

= $31,600, Hip Replacement = $16,100, C-section

= $16,100, OxyContin (60 pills, 20mg) = $265.

- 4) Total healthcare cost in the U.S. for these treatments: $31,600 + $29,100 + $16,100 + $265 = $77,065.
- 5) Spain’s treatment costs area: Angioplasty = $7,800, Hip Replacement = $6,800, C-section = $2,400, OxyContin = $36.
- 6) Total healthcare cost in Spain for the same treatments: $7,800 + $6,800 + $2,400 + $36 = $17,036.
- 7) Difference in cost between U.S. and Spain: $77,065 - $17,036 = $60,029.
- 8) Final answer: $77,065 in the U.S., and the cost difference is $60,029.

Answer: A) 29

Answer: C)

Answer: A) 77065 and 60029

- Figure 2: Example questions from the VIDEOMATHQA benchmark illustrating the three reasoning types: Problem Focused, Concept Transfer, and Deep Comprehension. The benchmark includes evolving dynamics in a video, complex text prompts, five multiple-choice options, the expert-annotated step-by-step reasoning to solve the given problem, and the final correct answer as shown above.

Existing benchmarks for mathematical reasoning (e.g., MathQA [2], ChartQA [22], and MathVista [19]) have driven substantial progress in image-based and text-based settings. However, they typically evaluate models on static diagrams, printed formulas, or single-turn queries and lack support for temporally extended contexts, dynamically evolving visual content, and do not offer true integration of vision, audio, text and background subject knowledge. Some recent efforts explore video question answering in general domains [8, 21, 15], but they do not target the precise, multi-modal, multi-step inference demanded by mathematical problem solving. Moreover, existing video benchmarks often rely on synthetic [33] or narrowly scoped tasks [24] without detailed reasoning annotations, making it difficult to diagnose whether a model’s solution stems from genuine logical inference or simple pattern matching.

To bridge these gaps, we introduce VIDEOMATHQA, a comprehensive benchmark designed explicitly to evaluate deep mathematical reasoning in videos. It comprises 420 manually annotated real-world video–question

pairs drawn from various educational resources. VIDEOMATHQA tests three core reasoning scenarios: a) Direct Problem Solving, where answers are grounded entirely in the presented content; b) Conceptual Transfer, requiring agents to apply learned methods to novel problems; and c) Deep Instructional Comprehension, involving multi-step reasoning over extended, partially worked-out explanations. Each instance includes high-resolution frames, aligned subtitles, and audio narration, and is annotated with fine-grained reasoning traces that capture every inference step to enable assessment of both intermediate steps and final outcomes. Videos span ten mathematical domains (e.g., geometry, calculus, statistics, graph theory, chart) and range from 10s clips to hour-long lectures, ensuring both short-term perception and long-range dependency evaluation (see Fig. 2 for illustrative examples of the reasoning scenarios). Our key contributions are:

- • A multimodal video reasoning benchmark that demands precise integration of vision (high-resolution), text, audio and subject knowledge in solving complex math problems.
- • Our benchmark offers three real-world task categories: problem solving, concept transfer, and deep comprehension, which reflect the spectrum of instructional scenarios in videos. It covers ten mathematical domains and various environment types (e.g., whiteboard scribbles, digital slides, animated charts), sourced from diverse sources.
- • Fine-grained reasoning annotations and metrics for measuring both intermediate inference fidelity and final answer correctness, with explicit mechanisms to detect confabulations.
- • A comprehensive evaluation across 30 proprietary and open-source multimodal models. We also conduct a comprehensive analysis of model performance trends and failure modes.

#### Takeaways: Comprehensive Evaluation

A comprehensive evaluation across 30 models reveals that success relies not just on visual perception, but on sustained attention to subtle cues dispersed across time, modalities, and context. Models often fail when key frames, symbols, or spoken details are missed, revealing limited ability to integrate long-range multimodal information. While performance generally scales with size, architecture and training quality are often more decisive; newer, smaller models frequently outperform older, larger ones. Notably, the gap between proprietary and open-source systems is narrowing, as recent open models now match or exceed proprietary models.

### 2 Related Works

Video Multimodal Understanding Benchmarks. Recent advances in multimodal understanding [1, 17, 41] have led to the development of multiple video benchmarks. Early datasets like ActivityNet-QA [34] and NextQA [30] focus on short clips and temporal action reasoning, while EgoSchema [21] and MovieChat [25] extend to long-form narrative comprehension. Recent efforts such as WorldQA[40], Next-GQA[31], VCGBench[20], MVBench [15] and Video-MME [8] broaden the scope with diverse reasoning tasks. While these benchmarks have advanced multimodal understanding in everyday contexts, they fall short of the deeper complexity posed by mathematical reasoning, where models must navigate non-linear content and demonstrate tightly integrated understanding across vision, audio, text, and background subject knowledge.

Multimodal Mathematical Benchmarks. Recent image-based mathematical benchmarks have significantly advanced the evaluation of multimodal models. MathVista [19] and Math-V [27] include visual questions drawn from textbooks and competitions, while MMMU [35] and MMMU-Pro [36] introduce subject-specific questions with CoT prompts and OCR inputs. DynaMath [43] evaluates the robustness of mathematical reasoning through visual perturbations. Although these benchmarks span a diverse range of mathematical topics, they are fundamentally limited to assessing reasoning over static images, and the temporal dimension intrinsic to video, where mathematical information may unfold through lectures, stepwise derivations, or interactive explanations, is not captured.

Video-MMMU [9] begins to explore video-based academic QA, but includes only a small subset of mathematical questions and focuses on general comprehension. It does not target the depth of reasoning or modality integration that mathematical problem solving demands. In contrast, VIDEOMATHQA is designed to evaluate deep mathematical reasoning by challenging models to interpret high-resolution visuals, follow non-linear spoken explanations, and integrate vision, language, and domain knowledge over time, while providing detailed step-by-step reasoning annotations to enable comprehensive analysis beyond just final answer accuracy.

[Figure 20]

[Figure 21]

7min – 1hr

0-20s 20s – 32s 32s – 45s 45s - 2min 2min - 7min

12%

11% 20% 18% 20% 15% 16%

4%

[Figure 22]

18%

Short Medium Long

72

(b) Distribution of video length duration

54

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

YouTube Data API

36

9%

18%

18

Video Selection Rules ü Question cannot be answered

QA Annotation Rules ü Solving requires a thorough engagement with the video. ü Multi-step and multi-modal

Steps Annotation Rules ü Step-by-step CoT reasoning is

only with static frames or audio. ü Rich Temporal Progression. ü Matches 1 of 3 reasoning types.

required to arrive at answer. ü Each question is assigned 4 to 10 CoT reasoning steps.

reasoning is required.

13%

12%

Stage # 3 Step-wise Reasoning

Stage # 2 QA Annotation

Stage # 1 Video Selection

VideoMathQA

17%

4%

Quality Assurance üQuality assurance follows each annotation stage for high quality. üAnnotations are performed by expert science graduates.

5%

(a) Math Concept-wise Distribution and Model Performance

(c) Annotation Pipeline used to create VideoMathQA Benchmark

- Figure 3: The figure illustrates a) Distribution of questions and model performance across ten mathematical concepts in the VIDEOMATHQA. The consistently low performance across all concepts reveals a significant gap in the ability of the current multimodal models to perform mathematical reasoning over videos. b) Distribution of video durations in VIDEOMATHQA, highlighting a diverse range from short clips of 10s to long-videos up to 1hr. c) The three-stage annotation pipeline for VIDEOMATHQA was performed by expert science graduates, who annotated detailed step-by-step reasoning trails, with each stage governed by strict quality assessment.

- 3 VIDEOMATHQA

#### 3.1 Overview

Despite the significant progress in image-based mathematical reasoning benchmarks, existing datasets fail to capture the unique challenges posed by video-based math problems. In designing VIDEOMATHQA, we asked: How can we reliably evaluate mathematical reasoning in video-based settings, and what core challenges a benchmark must capture to reflect real-world multimodal understanding? Our answers shaped the core principles of the benchmark. We identified three critical challenges: the need to interpret dynamic visual representations, such as diagrams constructed or modified over time, often involving handwritten or digital content requiring robust, frame-aware OCR; the need for temporal reasoning over long, non-linear contexts, where concepts unfold gradually or are revisited across time; and the need for joint grounding across visuals, text, and audio, as all three modalities often contribute distinct pieces of essential information. These interconnected challenges define the complexity of video-based mathematical tasks and directly inform the design of our benchmark.

We introduce VIDEOMATHQA, a dataset of 420 carefully curated video-question pairs drawn from diverse mathematical instructional content, including structured problem walkthroughs, concept demonstrations with follow-up questions, full-length whiteboard or digital lectures, and animated documentaries involving chartbased reasoning. Each question includes multi-step reasoning steps, with a total of 2,945 expert-annotated steps across the dataset. Each question is characterized along four dimensions: (i) Mathematical concept, covering 10 domains such as geometry, arithmetic and calculus, statistics and probability, counting, graph theory, puzzles, topology, and chart reading, (See Fig. 3a); (ii) Reasoning type, categorized as problem focused, concept transfer, or deep comprehension; (iii) Video duration, ranging from 10 seconds to over an hour and grouped as short, medium, or long, supporting evaluation of both short-term and long-range reasoning (See Fig. 3b); and (iv) Difficulty level, categorized as easy, medium, or hard (See Fig. 5b).

The annotation process consists of three stages: video selection, question-answer annotation, and step-wise reasoning (see Fig. 3c), requiring considerable expert effort. On average, 30 mins to identify a suitable video, 40 mins to craft a high-quality question-answer pair, and 1 hour to compose detailed step-by-step reasoning, resulting in approximately 2 to 2.5 hours per sample. Across the dataset, this amounts to roughly 115 person-days of annotation by science graduates. Different annotators handled each stage of a sample to ensure independent verification. Annotators could consult academic resources or LLMs for clarification.

#### 3.2 Video Selection

We curate video sources that capture core challenges, including dynamic visual representations, long-range temporal dependencies, and the need for joint grounding across visual, textual, and audio modalities. We

do this by selecting videos where the associated questions cannot be reliably answered using static frames or audio transcripts alone, instead focusing on content that exhibits rich temporal progression, such as stepby-step diagram construction and incremental equation derivation. Static slide presentations and videos with minimal visual transitions are excluded. All videos are sourced using the YouTube API, manually reviewed, and trimmed to retain only segments relevant to the annotated question. In addition, selection is guided by conceptual difficulty, ensuring coverage across a broad spectrum, from high-school topics to advanced university-level and Olympiad-style problems.

To construct the candidate video pool, we begin with a manually curated seed set and expand it using YouTube’s recommendation algorithm. After assembling a large set of video IDs, we apply automatic filtering based on metadata such as titles and descriptions. In the final stage, expert annotators manually review the videos review the videos and select those that meet our quality criteria and exhibit the desired reasoning types. For chartbased questions, we include animated charts from documentary-style videos and public news articles, covering various chart types such as bar plots (vertical and horizontal), pie charts, line graphs, histograms and stacked bar charts. We specifically select videos that present multiple charts whose interpretations are temporally and conceptually linked. An example is shown in Fig. 2, where the first problem involves interpreting multiple related charts and reasoning across them to derive the answer. For the remaining nine mathematical categories, we collect a diverse mix of lecture videos, screen recordings, and digital whiteboard tutorials that reflect instructional formats commonly used in educational content.

#### 3.3 Question-Answer Annotation

Each video in VIDEOMATHQAis paired with a carefully constructed multiple-choice question designed to evaluate a model’s ability to reason over visual and temporal mathematical content. Questions are written to ensure that solving them requires meaningful engagement with the video, rather than relying on surface-level cues from transcripts or isolated frames. This stage also serves as quality assurance for video selection, with approximately 70% of selected videos retained, and the rest are discarded. We categorize questions into three reasoning types: (i) Problem Focused, where the question is explicitly stated and can be solved through direct observation and reasoning; (ii) Concept Transfer, where a demonstrated method or solution approach must be applied to a similar but newly posed problem; and (iii) Deep Instructional Comprehension, which involves following long instructional content, such as a lecture or tutorial, to understand the context, identify what is being asked, interpret the partially worked-out solution, and complete the solution. Fig. 2 illustrates one example of each reasoning type, shown from left to right. All types require multi-step reasoning, with increasing levels of contextual and temporal integration across categories.

#### 3.4 Step-wise Reasoning and Quality Assessment

Following question-answer annotation, each sample undergoes a second stage of annotation, where a separate annotator writes a step-by-step explanation detailing the reasoning required to arrive at the final answer. Each question is assigned four to ten steps, with each step reflecting a meaningful semantic progression, capturing a distinct, essential part of the overall solution. This process results in a total of 2,945 high-quality expert-annotated reasoning steps. The steps are constructed in a logical sequential order, enabling step-level evaluation of how far a model progresses toward the correct answer; model-generated chain-of-thought responses are compared against these steps during evaluation. This stage also serves as quality assurance for the question-answer pairs, allowing the annotator to verify their correctness and fix any errors or improve clarity. Approximately 30% of the questions were refined during this stage. Additionally, each question is tagged with a set of likely error categories to support structured error analysis during the step evaluation.

#### 3.5 Comparison with Existing Video Benchmarks

While recent video-language benchmarks offer broad coverage across domains, they primarily focus on perceptual or narrative understanding, often lacking the depth, structure, and multi-modal precision required for mathematical reasoning. In contrast, VIDEOMATHQA targets the ‘needle-in-a-multimodal-haystack’ challenge by demanding tightly aligned interpretation across high-resolution visuals, spoken explanations, and textual content, characteristics underrepresented in prior works. Unlike benchmarks that emphasize short-form clips or general comprehension, our dataset centers on domain-specific, step-intensive reasoning with annotated chain-of-thought traces, enabling fine-grained evaluation of both intermediate and final responses. We present a comparison of VIDEOMATHQA with recent benchmarks in terms of domain, duration, STEM focus, and step-wise CoT annotation in the Appendix Tab. 3. By focusing on mathematical problem-solving in long instructional videos, it reveals challenges still underexplored in current video-language benchmarks.

### 4 Experiments

We evaluate a diverse set of models on VIDEOMATHQA, including 5 proprietary multimodal models: Claude-3.7-sonnet, GPT-4o, GPT-o4-mini, Gemini 2.0 Flash, and Gemini 1.5 Flash, and 25 open-source models selected for their strong capabilities in video reasoning. We specifically focus on state-of-the-art video models, spanning 4 model size categories: 5B, 9B, 40B, and 80B parameters. The pool includes recent models with advanced reasoning abilities, including those supporting chain-of-thought prompting (Qwen2.5-VL [3], VideoChat-R1 [16], Video-R1 [7]). To contextualize the performance of video-specialized models, we include two baselines, one vision-blind LLM (Qwen-2.5), and one state-of-the-art image model (Qwen2.5-VL) evaluated on a single key frame.

#### 4.1 Evaluation Strategies

Inference Protocol: Given that the benchmark emphasizes the importance of multimodal grounding and temporal reasoning across modalities, we choose a setting that ensures fairness by adapting input configurations to align with the strengths and operational capabilities of each model. Specifically, each model samples frames according to its optimal setting (LLaVA-OneVision [13] samples 32 frames, Qwen2.5-VL samples up to 768 frames, and Gemini has access to the full video). Correspondingly, subtitles are precisely aligned with the sampled frames, providing richer audio context to models capable of handling longer temporal sequences. This approach inherently rewards models equipped to reason over extended temporal context and to effectively integrate information across modalities. LMMs-Eval [12, 37] was used for our experiments and now includes official support for evaluating VideoMathQA.

We implement four evaluation strategies to comprehensively assess model performance: (i) Multiple-Choice Evaluation (MCQ): Each question provides five answer options (one correct and four distractors). This direct evaluation format offers clear reproducibility without reliance on LLM-based scoring. However, it can inflate weak model scores, especially in smaller models (e.g., ≤ 9B). (ii) Multi-Binary Evaluation (MBin): To better distinguish performance in such cases, we construct binary-choice variants by pairing the correct answer against each distractor independently. A model must select the correct option across all pairs to be marked correct, significantly reducing randomness and more accurately revealing true model capabilities, particularly critical when evaluating across a wide range of model scales. (iii) Chain-of-Thought (CoT) vs. Direct Answering: In direct evaluation, strict instruction-following is crucial since models are instructed to respond with only the final correct option, allowing immediate, format-based extraction without post-processing. In contrast, CoT evaluations encourage models to articulate detailed reasoning steps prior to answering, reflecting human-like problem-solving approaches, and providing leniency regarding response formatting. Here, we use a lightweight state-of-the-art open-source LLM (Qwen-3-4B) in non-thinking mode to extract the final answer. This setup allows us to analyze whether reasoning enhances performance. (iv) Step-wise Reasoning Evaluation: For CoT responses, we further evaluate the quality of reasoning by comparing model-generated rationales with annotated solution steps (typically 4-10 per question), each representing a distinct semantic progression. We use the Qwen-3-4B model in its thinking mode for this task, leveraging its strong mathematical reasoning ability to ensure fair and accurate evaluation. The model assigns each response a score between 0 and 10, along with a rationale. We further use the rationales to enable error analysis against predefined error categories, providing deep, actionable insights that highlights model limitations and reasoning gaps.

Implementation Details: For all models, we run direct answering for both MCQ (420 QA pairs) and MBin (1680 QA pairs). For models capable of reasoning, we additionally perform CoT evaluations for both MCQ and MBin. We use a light-weight LLM based post-processing to extract the option choice from the CoT responses to compute accuracy. Step-wise evaluation is conducted only for MCQ CoT responses. Using ground-truth steps, model response, and critique from step evaluation, we perform error analysis for selected models. We further ablate the effect of subtitles across all settings to analyze model sensitivity to multimodal input. All prompts used, including those for CoT prompting, postprocessing, step-wise evaluation, error analysis, and subtitle handling are detailed in Appendix C.1 to C.2.

#### 4.2 Quantitative Analysis

- Tab. 1 presents the direct answer evaluation, and Tab. 2 shows the chain-of-thought (CoT) evaluation. Both tables cover the MCQ and MBin evaluation, with and without subtitles, providing a comprehensive view of model capabilities. Tab. 2 additionally reports CoT step evaluation scores based on alignment with annotated reasoning traces. Proprietary models tend to benefit significantly from CoT prompting, whereas

MCQ MBin Mathematic Concepts Duration

Models Size

V +Sub V +Sub GAng GAre GLen Chart Stat Arth Topo Grph Cntg Pzle Short Med Long Random - 17.4 17.4 7.9 7.9 8.7 7.0 7.8 10.7 8.7 3.9 6.7 11.1 2.6 11.1 9.0 7.8 6.9

Proprietary Models

Claude-3.7-sonnet - 26.2 27.1 8.6 9.5 17.4 9.9 5.9 8.0 17.4 11.5 13.3 5.6 5.3 9.3 8.2 11.0 9.1 GPT-4o - 20.2 24.5 12.6 13.6 13.0 12.7 15.7 12.0 4.4 17.3 20.0 5.6 7.9 20.4 14.2 15.6 10.6

- Gemini-1.5-Flash - 20.5 23.1 12.6 17.6 26.1 15.5 19.6 9.3 17.4 23.1 6.7 22.2 15.8 24.1 17.9 22.1 12.1

- Gemini-2.0-Flash - 28.6 31.7 14.1 20.5 30.4 23.9 27.5 13.3 8.7 19.2 13.3 16.7 7.9 33.3 25.4 24.0 11.4 Open-source Models (< 5B)

Qwen2.5-VL [3] 3B 26.9 27.6 19.3 19.6 26.1 23.9 23.5 21.3 34.8 17.3 26.7 11.1 15.8 20.4 25.4 23.4 15.9 InternVL2.5 [4] 2B 24.3 20.7 14.3 14.5 21.7 9.9 27.5 10.7 4.4 15.4 20.0 0.0 15.8 16.7 17.9 16.9 8.3 PLM-LLaMA [5] 3B 22.9 22.1 13.6 15.0 17.4 16.9 25.5 8.0 26.1 9.6 20.0 11.1 13.2 13.0 16.4 18.8 9.1 InternVL3 [42] 2B 22.4 23.3 18.8 16.4 21.7 16.9 17.7 17.3 30.4 15.4 20.0 22.2 13.2 5.6 18.7 14.9 15.9

Open-source Models (< 9B)

PLM-LLaMA [5] 8B 22.1 23.1 16.7 14.5 13.0 11.3 17.7 13.3 17.4 17.3 20.0 11.1 10.5 16.7 16.4 14.9 12.1 Oryx-1.5 [18] 7B 22.6 22.6 16.9 17.4 13.0 23.9 23.5 9.3 21.7 23.1 20.0 5.6 18.4 11.1 20.2 20.8 10.6 LLaVA-OV [13] 7B 20.7 21.2 14.8 15.5 8.7 15.5 17.7 16.0 30.4 17.3 13.3 5.6 15.8 11.1 16.4 18.8 10.6 LongVA-DPO [38] 7B 21.4 21.7 16.2 14.1 8.7 15.5 17.7 12.0 30.4 9.6 6.7 5.6 10.5 18.5 14.9 11.7 15.9 Video-R1 [7] 7B 21.4 17.4 16.0 16.2 8.7 22.5 25.5 16.0 26.1 13.5 6.7 5.6 13.2 9.3 16.4 16.9 15.2 InternVL2.5 [4] 8B 24.3 24.8 18.6 18.6 26.1 19.7 17.7 17.3 21.7 19.2 26.7 11.1 10.5 20.4 17.9 22.7 14.4 LLaVA-Video [39] 7B 26.9 26.4 20.0 19.3 13.0 21.1 31.4 17.3 17.4 15.4 26.7 5.6 18.4 18.5 23.9 20.8 12.9 InternVideo2.5 [28] 8B 25.2 28.6 19.1 19.1 34.8 22.5 15.7 14.7 21.7 19.2 20.0 27.8 10.5 18.5 18.7 22.1 15.9 Qwen2.5-VL [3] 7B 26.7 27.9 19.8 19.1 8.7 25.4 25.5 18.7 13.0 23.1 13.3 5.6 15.8 16.7 22.4 19.5 15.2 InternVL3 [42] 8B 29.1 27.9 20.0 20.7 13.0 29.6 27.5 13.3 13.0 28.9 20.0 22.2 15.8 14.8 25.4 24.0 12.1 VideoChat-R1 [16] 7B 27.6 29.1 21.2 21.2 8.7 22.5 31.4 21.3 17.4 30.8 6.7 11.1 15.8 18.5 26.9 20.1 16.7

Open-source Models (< 40B)

Aria [14] 34B 23.8 26.4 17.4 19.1 8.7 25.4 19.6 22.7 17.4 19.2 20.0 11.1 21.1 11.1 21.6 16.9 18.9 Oryx-1.5 [18] 32B 30.5 33.1 22.9 24.1 30.4 39.4 31.4 10.7 17.4 21.2 6.7 11.1 15.8 33.3 27.6 29.9 13.6 Qwen2.5-VL [3] 32B 32.4 32.6 25.7 24.8 43.5 31.0 25.5 14.7 26.1 26.9 6.7 27.8 10.5 33.3 28.4 30.5 14.4 InternVL2.5 [4] 38B 31.0 33.6 24.1 26.0 43.5 38.0 39.2 8.0 13.0 32.7 6.7 11.1 18.4 29.6 34.3 31.8 10.6 InternVL3 [42] 38B 31.7 35.7 25.2 29.5 34.8 42.3 37.3 13.3 17.4 25.0 13.3 33.3 26.3 40.7 35.8 38.3 12.9

Open-source Models (< 80B)

LLaVA-Video [39] 72B 28.3 30.0 20.2 24.3 8.7 32.4 25.5 20.0 13.0 36.5 13.3 22.2 21.1 24.1 27.6 27.3 17.4 LLaVA-OV [13] 72B 25.5 28.3 21.0 24.8 17.4 31.0 23.5 12.0 21.7 38.5 20.0 27.8 18.4 31.5 30.6 28.6 14.4 InternVL2.5 [4] 78B 33.3 31.7 28.3 27.9 39.1 36.6 31.4 18.7 26.1 32.7 26.7 27.8 13.2 27.8 33.6 35.1 13.6 Qwen2.5-VL [3] 72B 36.9 37.6 26.0 27.9 26.1 36.6 31.4 17.3 30.4 38.5 20.0 16.7 18.4 29.6 34.3 29.2 19.7 InternVL3 [42] 78B 33.3 31.7 28.3 27.9 39.1 36.6 31.4 18.7 26.1 32.7 26.7 27.8 13.2 27.8 33.6 35.1 13.6

- Table 1: Performance on VIDEOMATHQA using direct answer prompting in both MCQ and Multi-Binary (MBin) settings. We report results with video-only (V) and video+subtitle (+Sub) inputs to highlight the effect of subtitles on multimodal grounding. MBin performance is further broken down across ten mathematical concepts and three video duration categories. The table covers open-source models across multiple size tiers. The mathematic concepts are geometric angle (GAng), geometric area (GAre), geometric length (GLen), chart reading (Chart), statistics and probability (Stat), arithmetic and calculus (Arth), topology (Topo), graph theory (Grph), counting (Cntg) and puzzles (Pzle).

open-source models show mixed gains. In step evaluation, GPT-o4-mini achieves the highest score of 6.9, with Qwen2.5-VL-72B leading among open models with a score of 5.0.

How does model size influence performance? Across both MCQ and MBin settings, we observe that model performance improves with scale. We observe this trend in both evaluations with CoT prompting in

- Tab. 2 and direct answering in Tab. 1. For instance, in the CoT MBin evaluation with subtitles, InternVL-
- 3 [42] shows consistent improvement across model scale: 20.0% (8B) to 25.0% (38B) and 27.9% (72B). Comparable trends are observed for other video-MLLM models (LLaVA-Video [39], LLaVA-OneVision [13], Qwen2.5-VL [3]), indicating that larger models are better at retaining temporal context, focusing on key visual details, and grounding information across modalities, all crucial for video-based mathematical reasoning. However, scale alone does not determine performance. Smaller, newer models often outperform older, larger ones. For example, InternVL-3-38B surpasses multiple 72B models (LLaVA-Video-72B, LLaVA-OneVision72B) in both CoT and direct answers. Newer models benefit from stronger architectures, improved visual understanding, and better reasoning, enabling them to outperform larger, previously SoTA models.

Takeaways: Model Size and Performance Larger models typically perform better, but newer, smaller models often outperform older, larger ones, highlighting that architecture and training quality matter as much as scale.

MCQ MBin Mathematic Concepts Duration CoT

Models Size

V +Sub V +Sub GAng GAre GLen Chart Stat Arth Topo Grph Cntg Pzle Short Med Long Eval Random - 17.4 17.4 7.9 7.9 8.7 7.0 7.8 10.7 8.7 3.9 6.7 11.1 2.6 11.1 9.0 7.8 6.9 Open-source (< 9B)

Video-R1 [7] 7B 23.8 27.6 18.1 20.0 13.0 26.8 23.5 9.3 13.0 34.6 20.0 16.7 18.4 16.7 21.6 26.0 11.4 3.9 LLaVA-Video [39] 7B 26.4 23.6 20.0 16.0 4.4 15.5 23.5 16.0 21.7 7.7 26.7 0.0 21.1 18.5 16.4 16.9 14.4 2.7 Qwen2.5-VL [3] 7B 25.2 29.5 17.6 18.3 13.0 15.5 11.8 20.0 21.7 36.5 13.3 16.7 10.5 16.7 16.4 20.1 18.2 3.7 InternVL3 [42] 8B 28.8 26.9 17.9 20.0 17.4 22.5 27.5 13.3 4.4 17.3 13.3 16.7 7.9 24.1 19.4 23.4 9.9 3.4 InternVideo2.5 [28] 8B 24.3 27.6 18.3 19.8 26.1 25.4 13.7 14.7 26.1 21.2 13.3 27.8 18.4 18.5 17.9 25.3 15.2 3.0 VideoChat-R1 [16] 7B 22.4 28.3 21.4 19.8 13.0 29.6 15.7 10.7 17.4 26.9 6.7 22.2 18.4 24.1 20.9 25.3 12.1 3.6

###### Open-source (< 40B)

Oryx-1.5 [18] 32B 29.1 33.6 21.7 25.2 34.8 35.2 43.1 13.3 17.4 21.2 20.0 22.2 18.4 22.2 30.6 29.2 15.2 3.7 InternVL3 [42] 38B 30.0 31.4 21.7 25.0 43.5 31.0 25.5 16.0 17.4 32.7 20.0 11.1 13.2 31.5 26.9 31.2 15.9 4.1 Qwen2.5-VL [3] 32B 31.4 36.9 22.6 27.1 47.8 29.6 33.3 16.0 26.1 32.7 13.3 16.7 23.7 29.6 29.1 32.5 18.9 4.9

###### Open-source (< 80B)

LLaVA-Video [39] 72B 23.6 29.3 14.8 18.6 8.7 22.5 17.7 14.7 8.7 21.2 26.7 11.1 26.3 20.4 17.2 21.4 16.7 3.1 LLaVA-OV [13] 72B 23.3 26.9 14.3 18.1 8.7 14.1 19.6 13.3 21.7 26.9 20.0 22.2 10.5 25.9 15.7 23.4 14.4 3.2 Qwen2.5-VL [3] 72B 37.4 36.9 24.5 28.6 30.4 31.0 31.4 24.0 21.7 50.0 13.3 22.2 15.8 25.9 27.6 34.4 22.7 5.0 InternVL3 [42] 78B 34.1 37.1 25.2 27.9 39.1 39.4 33.3 13.3 26.1 23.1 33.3 22.2 10.5 40.7 28.4 36.4 17.4 4.9

###### Proprietary Models

Claude-3.7-sonnet - 24.8 29.5 12.1 19.3 34.8 29.6 19.6 4.0 26.1 13.5 20.0 16.7 21.1 22.2 23.1 26.0 7.6 4.2 GPT-4o - 27.1 34.3 18.6 22.9 26.1 22.5 17.7 17.3 30.4 32.7 20.0 33.3 13.2 25.9 19.4 29.9 18.2 4.9 Gemini-2.0-Flash - 35.2 38.8 19.5 24.8 34.8 21.1 27.5 18.7 21.7 28.9 13.3 33.3 18.4 33.3 27.6 27.9 18.2 4.7 GPT-o4-mini - 49.8 61.4 42.1 44.8 43.5 49.3 45.1 40.0 65.2 63.5 20.0 72.2 23.7 31.5 45.5 44.8 42.4 6.9

- Table 2: Performance on VIDEOMATHQA using chain-of-thoughts prompting in MCQ and Multi-Binary (MBin) settings. We report results with video-only (V) and video+subtitle (+Sub) inputs to highlight the effect of subtitles on multimodal grounding. MBin performance is further broken down across ten mathematical concepts and three video duration categories. The table covers open-source models across multiple size tiers.

How do proprietary models compare to open-source models? We evaluate five proprietary models with CoT prompting and find that Gemini-2.0-Flash and GPT-o4-mini deliver the best performance. GPT-o4-mini achieves the highest overall accuracy, with 44.8% in CoT MBin with subtitles. It performs particularly well in complex reasoning categories such as arithmetic-calculus (63.5%), statistics (65.2%), and geometric area (49.3%), with performance significantly higher than the average of other proprietary and open-source models. These results suggest that its strong performance stems from a better ability to integrate understanding across vision, audio, text, and background subject knowledge, enabling more coherent mathematical reasoning. While proprietary models continue to lead, our results show that the gap between proprietary and open-source models is narrowing. Optimized open-source models such as Qwen2.5-VL-72B and InternVL-3-78B outperform several proprietary counterparts, including Claude-3.7-Sonnet, Gemini-2.0-Flash, and GPT-4o.

Takeaways: Proprietary vs. Open-source Models The gap between proprietary and open-source models is narrowing, as optimized open-source architectures and enhanced multimodal integration now closely match or surpass proprietary models.

How do subtitles influence model performance? Subtitles consistently enhance model performance across both CoT evaluation (Tab. 2) and direct answering (Tab. 1), especially for larger open-source and proprietary models. However, the impact of subtitles is not uniform: smaller models (<5B and <9B) often show minimal or inconsistent gains. In contrast, reasoning-capable models like GPT-o4-mini improve from 42.1% (video-only) to 44.8% (video+sub), and Qwen2.5-VL improves from 24.5% to 28.6% in the CoT MBin setting. As shown in Fig. 4b, models with stronger reasoning capabilities benefit more from subtitles, while smaller models struggle to extract meaningful gains from them. These improvements reflect the ability to integrate fine-grained audio cues with visual frames, a ‘needle-in-a-multimodal-haystack’ challenge, where critical information is distributed across modalities, and stronger reasoning models are better equipped to ground these disparate cues into coherent solutions, while others may overlook small but essential verbal cues.

#### Takeaways: Subtitles and Multimodal Reasoning

Stronger reasoning models gain more from subtitles by anchoring verbal cues to visual frames. This reflects a ‘needle-in-a-multimodal-haystack’ challenge, where key signals are scattered across modalities. In contrast, smaller models often fail to integrate these cues, missing critical information for reasoning.

How does video length and frame sampling affect model performance? We evaluate model performance across short (<30s), medium (30s–2min), and long (2min–1hr) video categories and observe two distinct trends

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

(a) Math Concept-wise Distribution and Model Performance (b) Effect of subtitles in CoT evaluation (c) Effect of frame sampling

- Figure 4: The figure shows VIDEOMATHQA performance a) Across video duration categories using the CoT MBin +Sub setting; b) Impact of subtitles under the CoT MBin setting; and c) Effect of varying the number of input frames under CoT MCQ setting. Overall, models perform best on medium-length videos, and overall accuracy improves with the inclusion of subtitles and more frames during evaluation.

(see Fig. 4a). First, while most models perform reasonably well on short videos, accuracy typically improves on medium-length videos and declines for longer durations. These trends align with the three reasoning challenges targeted by the benchmark. Short videos often correspond to problem focused questions, requiring the model to derive a solution, where success hinges on general mathematical competence and the ability to extract key visual or verbal cues. Medium-length videos commonly involve concept transfer questions, where the model is first shown a solution or method, then asked to adapt it to a related problem, favoring models that can effectively comprehend the instructions. In contrast, long videos correspond to deep instructional comprehension questions, which require following extended, often non-linear instructional sequences to interpret the context. Here, the informational load is higher, and cues essential for solving the problem may be distributed sparsely across modalities and time. This setting closely aligns with the central challenge, where overlooking even a few critical details can derail the entire reasoning process. Second, we study how frame sampling influences performance by evaluating Qwen2.5-VL with 16, 64, 256, and 768-frame settings (see Fig. 4c). We find that increasing frame count provides consistent improvements, particularly for longer videos: a 5-point gain for short videos and up to 8 points in long videos, highlighting that models capable of handling extended frame sequences and maintaining long-range temporal coherence are better equipped for video-based mathematical reasoning. Further, in Fig. 5a, we compare video models with vision-blind text only and single-image models, highligting that in-depth temporal reasoning is required to perform well on our VIDEOMATHQA benchmark.

#### Takeaways: Impact of Video Length and Reasoning Type

Models perform relatively well on applying learned concepts, moderately on direct problem solving, and struggle most with following detailed instructions in longer videos. Performance declines on tasks requiring alignment of sparse, cross-modal cues over time, highlighting the challenge of sustaining attention and inference across complex multimodal sequences.

How well do models perform across different mathematical concepts? We analyze model performances across the ten mathematical categories covered in the benchmark and observe notable variation in their ability to comprehend and solve different mathematical concepts (see Fig. 3a). Current models tend to perform better on questions involving arithmetic and calculus, with average accuracy around 32% and GPT-o4-mini achieving the best performance of 63.5% with CoT evaluation. Most models show moderate performance on categories such as geometric reasoning and puzzles, with average performance ranging between 24% and 30%. In contrast, chart reading, topology, graph theory, and statistics and probability are more challenging for all models. Average accuracy across these categories typically falls between 16% and 21%, with GPT-o4-mini scoring only 20% in topology and graph theory, and a maximum of 40% in chart reading.

How does question difficulty affect model performance? Model accuracy varies notably with question difficulty, while most models solve a moderate proportion of easy questions, they struggle with harder ones (Fig. 5b). GPT-4o answers 96% of easy questions correctly, yet handles only 46% of the hard ones. InternVL3-78B solves 60% of easy examples, but manages just 8% at the hard level. Other models follow the same

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Problem MisUnderstand

Computation Error

Concept Application Error

Strategy Selection Error

Information Retrieval Error Visual Interpretation Error Recall/ Memory Error

(a) Vision-blind vs Image-only vs Video-MLLM (b) Impact of Question Difficulty (c) Chain-of-thoughts error analysis

- Figure 5: The figure shows a) Comparison among vision-blind, image-only, and video models, highlighting the need for video-level understanding to perform well in VIDEOMATHQA. b) Distribution of questions in VIDEOMATHQAacross three difficulty levels for varying reasoning depths, and the relationship between performance and question difficulty across top-performing models. c) Error analysis based on CoT step evaluation. Most model errors stem from misunderstanding the question, where models misinterpret what the question asks or overlook critical multimodal cues.

trend, with limited success on more complex questions, exposing a key limitation in current models, their ability to generalize weakens under higher cognitive load.

Where do models fail in the reasoning process? Fig. 5c presents a breakdown of model errors in the VIDEOMATHQAbased on CoT step evaluation across seven error categories: misunderstanding the problem, failing to retrieve relevant information, misinterpreting visuals, incorrect application of concepts, choosing the wrong strategy or formula, forgetting previous context, and making calculation errors. Among these, the most common failure mode is problem misunderstanding, where models misinterpret what the question asks or overlook critical multimodal cues in the video. This reflects the core challenge of our benchmark, where missing even a small verbal or visual detail can derail reasoning entirely. Proprietary models like GPT-o4-mini and Gemini-2.0-Flash show fewer errors in concept application and strategy selection (12% and

- 6% respectively), suggesting stronger domain grounding and better problem-solving execution. In contrast, open-source models like InternVL-3 exhibit more broadly distributed errors, with concept application and strategy selection together accounting for 23% of total errors, alongside notable mistakes in computation. Meanwhile, GPT-o4-mini shows a higher share of visual interpretation errors, indicating difficulty with fine-grained visual cues such as charts and diagrams. See Tab. C.2 for details on the prompts used for error annotation and definitions of each error category.

#### Takeaways: Common Failure Modes in Reasoning

A key challenge for all models is extracting and integrating sparse information spread across modalities. Proprietary models show better integration of background subject knowledge, making fewer errors in concept application and strategy selection. Open-source models struggle more across reasoning steps, often applying the right idea incorrectly or choosing an ineffective path to the solution.

### 5 Conclusion

In this work, we introduce VideoMathQA, a comprehensive benchmark designed to evaluate mathematical reasoning in real-world educational video settings. By introducing carefully curated video-question pairs with detailed step-by-step reasoning trails, spanning diverse mathematical domains and instructional formats, our benchmark addresses the critical gap in the assessment of temporally extended, cross-modal reasoning capabilities in open-source and proprietary multimodal models. Unlike existing mathematical benchmarks, which mainly rely on static images and limited context, VideoMathQA requires models to jointly interpret dynamic visual, textual, and auditory information and to tackle real-world educational scenarios through problem solving, concept transfer, and deep instructional comprehension tasks.

# Appendix

### A Comparison with Existing Benchmarks

Benchmark Domain Dur. (s) STEM CoT Annotation

CLEVRER [32] Physics (Syn.) 5 ✓ ✗ Auto MovieChat-1K [25] Movie 500 ✗ ✗ Human EgoSchema [21] Egocentric 180 ✗ ✗ Auto+Human Video-MME [8] General 1018 ✗ ✗ Human PerceptionTest [23] Perceptual 23 ✗ ✗ Human MMBench-Video [6] General ∼100 ✗ ✗ Human LongVideoBench [29] General 473 ✗ ✗ Human Video-MMMU [10] Scientific 506 ✓ ✗ Human

VIDEOMATHQA(Ours) Math 241 ✓ ✓ Human†

Table 3: Comparison of VIDEOMATHQA with recent video-language benchmarks in terms of domain, duration, STEM focus, step-wise CoT annotation, and source. † Annotations for our VIDEOMATHQA are provided by graduate-level human experts.

### B Additional Implementation Details

We use 8 A100 − 80GB GPUs for all evaluations. For models with ≤ 8B parameters, we use data parallelism across 8 workers. For larger models, we utilize tensor parallelism with TP = 8. All MLLM evaluations are conducted using lmms_eval [12], while evaluations for language-only models are performed using VLLM [11]. We provide the implementation of VIDEOMATHQA˜in lmms_eval, along with scripts to run both MLLM evaluations as part of the submission.

For all evaluated models, we use the recommended number of input frames following the official code base. Specifically, we use 128 frames for Aria [14], 512 frames for InternVideo-2.5 [28], 64 frames for LLaVA-Video [39], 128 frames for LongVA [38], 32 frames for LLaVA-OneVision [13] and 768 frames for Qwen2.5-VL [3]. We use greedy decoding with a temperature of 0 for all MLLM evaluations. For chain of thought (CoT) evaluation, we post-process the responses to extract the final answer option using Qwen3-4B [26] LLM. For the CoT step evaluation, we also use the Qwen3-4B model as the LLM judge and report the average over three runs. Below we provide all the LLM prompts we use in this work including CoT prompting, postprocessing, step-wise evaluation, error analysis and subtitle handling.

### C Limitations and Future Work

VIDEOMATHQA is an initial effort to select and annotate question-answer pairs along with step-by-step chain-of-thought reasoning in temporally rich videos, where the answer cannot be inferred merely from a few static frames or the audio transcript. The selection and annotation of these samples require a significant amount of time. For example, on average, it took graduate-level experts with at least a Master’s degree in Science approximately 30 minutes to find a suitable video, 40 minutes to write a good question and answer, and 1 hour to compose detailed step-by-step reasoning. In total, annotating one sample took around 2 to 2.5 human hours, amounting to approximately 115 working days for 420 samples. This effort is substantial and makes scaling the dataset size difficult. We identify this as a limitation of our work and plan to explore semi-automatic annotation pipelines in the future to significantly reduce the annotation effort while maintaining high annotation quality.

LLM Evaluation Prompt for Chain of thoughts Step Evaluation You are a intelligent assistant for grading math question solutions. You will be given:

- • A mathematical question (question) with multiple-choice options (options).
- • A list of numbered ground truth steps (gt_steps) showing the correct reasoning to solve a math problem.
- • A answer (answer) that is the correct final solution to the question.
- • A model prediction (prediction) that includes the steps the model followed and possibly the final answer.

TASK: Compare the prediction to gt_steps and assign a score out of 10 using the rubric below. You must reward both matching logic and valid alternative reasoning. Avoid overly strict step-by-step comparison, instead focus if the model follows a coherent and plausible mathematical approach.

Scoring Rubric:

- 1. Relative Step Matching (Main Criterion)

- • Count the total number of ground truth steps: N.
- • Evaluate how many predicted steps correctly align with gt_steps in terms of mathematical logic, reasoning, or computations.
- • Score = (matching steps / N) × 10, rounded to nearest whole number.
- • A step MATCHES if it serves the same mathematical purpose, even if phrased or ordered differently.

- 2. Correct Final Answer via Different Reasoning

- • If the model’s final answer is correct, and the steps are logically valid (even if they differ from gt_steps, assign a full score of 10.
- • Ignore number of matching steps in this case unless the reasoning is clearly flawed or incoherent.
- • Reduce the score proportionally if reasoning contradicts parts of the ground truth.

- 3. Implicit or Inferred Steps

- • Do NOT penalize if early steps are skipped, but later logic clearly depends on them.
- • If a model does not state "identify the chart," but proceeds to use correct values, assume it did this implicitly.
- • ALWAYS check for implied steps before reducing the score.

- 4. Ignore Superficial Differences

- • Do NOT deduct score for formatting, different notation or variable names, or additional clarifications.
- • FOCUS on the mathematical meaning, not the literal step match.

Output Format SCORE_CARD: {"matched_steps": "X/N", "final_answer_correct": <0 or 1>, "critique": "<2-3 sentence summary>", "score": <0-10>}

Be strict when awarding credit. Do NOT be lenient. Carefully evaluate how far the model’s reasoning aligns with the ground truth steps before assigning a score.

LLM Evaluation Prompt for Error Analysis using CoT Step Evaluation You are an intelligent assistant for analyzing model-generated math solutions. You will be given:

• A mathematical question (question) with multiple-choice options (options).

- • A list of numbered ground truth steps (gt_steps) showing the correct reasoning and a correct final solution (answer).
- • A model prediction (prediction) that includes the steps the model followed.
- • A critique (critique), which is a short rationale describing the quality of the model’s reasoning and its alignment with the ground truth.

TASK: Your job is to carefully read and analyze the model’s prediction. Compare it to the ground truth steps and determine whether the prediction contains any of the following 7 types of reasoning errors. Use the critique and reason about where and why the model diverges from correct logic. Then, assign all relevant error category labels from the list below. The error types are not specific to each step; instead, identify a set of error types that apply to the overall reasoning.

- • TYPE1: Question Misunderstanding Error: The model fails to understand what the question is asking or misunderstands its demand. It cannot correctly interpret which quantity, relationship, or part of the video is referenced, often mixing up figures, charts, or examples.
- • TYPE2: Information Retrieval Failure: The model cannot locate or recognize the needed data in the video, including charts. It overlooks presented numbers, labels, angles, side lengths, diagrams, or text overlays, so no raw facts are available for further processing.
- • TYPE3: Visual Interpretation Error: Although the model attends to the correct visual element, it reads it wrongly, misinterpreting axis scales, bar heights, marker positions, legends, or estimating distances and angles improperly.
- • TYPE4: Concept Application Error The model knows which principle or method applies but is not able to execute it properly on the given question. It may recall the right concept yet misalign variables, swap parameters, or break the logical steps needed for that specific problem.
- • TYPE5: Strategy & Formula Selection Error: The model picks an entirely inappropriate approach, choosing the wrong theorem, formula, or problem-solving strategy for the task.
- • TYPE6: Recall & Memory Error: The model forgets or ignores earlier information that are essential. It also covers cases when it contradicts key information from earlier in the question or in its own reasoning. This includes dropping previously used values, repeating steps unnecessarily, or breaking logical flow by not following through on earlier steps.
- • TYPE7: Computational Error: The model has the correct inputs and method but makes calculation mistakes, incorrect addition, subtraction, multiplication, division, rounding, or unit-conversion errors.

#### RULES:

- • Multiple error types may apply to a single prediction. Errors may be global (relevant to all steps) or local (relevant to one or a few steps).
- • Important Note: If the model uses different but valid reasoning and arrives at the correct answer, assign: none, even if the steps do not match the ground truth.
- • Do not assign an error unless you are confident it reflects a real mistake.
- • If a clear mistake exists, but it does not match any of the listed error types, assign: uncategorized.
- • Important Note: If you are not confident about which error applies, do not guess. Use uncategorized instead of forcing a type.

OUTPUT FORMAT: Return only a comma-separated list of error type labels. Examples:

- • TYPE2, TYPE4
- • none
- • uncategorized Do not include any explanations or extra text.

#### Prompt for LLM Based Post Processing Multiple-Choice Chain-of-Thought Responses

Given the original multiple-choice options and a model-generated answer containing reasoning and a final answer, identify the option that best matches the final answer and return only the corresponding letter (A, B, C, D, or E).

The options are: {Options} The model response is: {Response} Only return the letter A, B, C, D, or E. If none is found, return None. Prompt for Multiple-Choice Evaluation with Direct Answering

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.

Answer with the option’s letter (A or B) from the given choices directly.

#### Prompt for Multiple-Choice Evaluation with Chain-of-Thought

Select the best answer to the following multiple-choice question based on the video. Respond with the letter (A, B, C, D or E) of the correct option.

First, please perform reasoning, and think step-by-step to provide the best answer to the following question with the option’s letter (A, B, C, D or E) from the given choices.

Prompt for Multiple-Choice Evaluation with Subtitles The subtitles of the video are listed below: {Subtitles} Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option. Prompt for Multi-Binary Evaluation with Direct Answering

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A or B) of the correct option.

Answer with the option’s letter (A or B) from the given choices directly.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Aida Amini et al. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [5] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025.
- [6] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. In Advances in Neural Information Processing Systems, 2024.
- [7] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [8] Chaoyou Fu et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025.
- [9] Kairui Hu, Jindong Liu, Yeyu Duan, Song Li, Haoyu Zhang, Can Xu, Xiaodan Liang, and Dahua Lin. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [10] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [11] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [12] Bo Li, Peiyuan Zhang, Kaichen Zhang, Fanyi Pu, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Accelerating the development of large multimodal models, 2024. URL https://github.com/EvolvingLMMs-Lab/lmms-eval.
- [13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [14] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024.
- [15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22195–22206, 2024.

- [16] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.
- [17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [18] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024.
- [19] Pan Lu, Hritik Bansal, et al. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.
- [20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424,

- 2023.

[21] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems,

- 2024.

- [22] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, 2022.
- [23] Viorica Patraucean et al. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023.
- [24] Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266, 2024.
- [25] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18221–18232, 2024.
- [26] Qwen Team. Qwen3, April 2025. URL https://qwenlm.github.io/blog/qwen3/.
- [27] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with the math-vision dataset. In Advances in Neural Information Processing Systems, 2024.
- [28] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025.
- [29] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. In Advances in Neural Information Processing Systems, 2024.
- [30] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.
- [31] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13204–13214, 2024.
- [32] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.
- [33] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.

- [34] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 9127–9134, 2019.
- [35] Xiang Yue et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In CVPR, 2024.
- [36] Xiang Yue et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [37] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. URL https://arxiv.org/abs/2407.12772.
- [38] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [39] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.
- [40] Yuanhan Zhang, Kaichen Zhang, Bo Li, Fanyi Pu, Christopher Arif Setiadharma, Jingkang Yang, and Ziwei Liu. Worldqa: Multimodal world knowledge in videos through long-chain reasoning. arXiv preprint arXiv:2405.03272, 2024.
- [41] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [42] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [43] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In ICLR, 2025.

