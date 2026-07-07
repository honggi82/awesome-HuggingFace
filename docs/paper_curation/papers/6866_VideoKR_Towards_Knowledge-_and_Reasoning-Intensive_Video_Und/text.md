## VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding

##### Lin Fu1* Zheyuan Yang2* Yang Wang1 Tingyu Song3 Arman Cohan4 Yilun Zhao4

VideoKR VideoKR

#### Abstract

Knowledge Distribution

Healthcare (3.2%)

###### Engineering (18.8%)

We introduce VideoKR, the first large-scale training corpus specifically designed to strengthen knowledge- and reasoning-intensive video understanding. It comprises 315K video reasoning examples over 145K newly collected, CClicensed, expert-domain videos. We develop a human-in-the-loop, skill-oriented example generation pipeline that targets progressively deeper video reasoning capabilities while ensuring the difficulty, diversity, and reliability of both the examples and their CoT rationales. We also curate VideoKR-Eval, a new expert-annotated benchmark where questions require genuine video understanding and knowledge-intensive reasoning rather than textual shortcuts. Our experiments show that, under a standard SFT→GRPO pipeline, models post-trained on VideoKR outperform prior post-training approaches on knowledge-intensive video reasoning while remaining competitive on general video reasoning, highlighting data design as a key driver of progress in video reasoning. We further conduct comprehensive ablations to isolate the contributions of VideoKR, providing actionable insights for future work.

Software Engineering

Medical Laboratory Science

Biomedical Informatics Public Health Radiography Nursing

# arXiv:2606.05259v1[cs.CV]3Jun2026

Mechatronics & Robotics

Mechanical Engineering Electrical Engineering Civil Engineering

VideoKR-SFT-260K VideoKR-RL-170K

Others

Others

### Core Skills

###### Social Science (40.1%)

###### Natural Science (37.9%)

Chemistry Biochemistry Atmospheric Science Mathematics Geology

Finance Education Political Science Psychology Sociology

Others

Others

Core Skills

###### Example of Collected CC-licensed Video

Basic Video Perception

[Figure 1]

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

Knowledge Reasoning 00:35 00:38 00:44 01:09

QA Examples via Skill-Oriented Example Generation

Q: What is the overall trend in thermometer reading?

Answer: Rising Reasoning: (...abbreviated...) The liquid column between 1:06 – 1:12 is rising (...abbreviated...)

[Figure 19]

[Figure 20]

Q: What is the role of the black solid in the video? (A) Homogeneous catalyst. (B) Heterogeneous catalyst (C) Oxidant (D) Dehydrating agent

Q: What is the role of the black solid?

A. Homogeneous catalyst B. Oxidant C. Heterogeneous catalyst D. Dehydrating agent Answer: C Reasoning: (...abbreviated...) ;<#!(solid) catalyzes "!#! (liquid), therefore it is a heterogeneous catalysis.

[Figure 22]

[Figure 23]

A: B Reasoning (abbreviated): !"#!(solid) catalyzes $!#! (liquid), Q: Using the same solution as in the video (35.0(), 18.0%(./.), density 1.081 ⋅ therefore it is a heterogeneous catalysis. ()!") and excess black powder, determine the volume of dry gas produced at STP.

Answer: 2.24L Reasoning: From 00:30 to 00:52 in the video, we (...abbreviated...), therefore, we identify the demo as the decomposition of hydrogen peroxide to water and

oxygen ( 2"!#! → 2"!# + #! ). (...abbreviated...) For 35.0*+ of an 18.0%(0/0) solution(3 = 1.085/*+), the solution mass is 35.0×1.08 = 37.805 ; therefore, * "!#! =

(...abbreviated...) the dry oxygen volume is 0.1×22.414 ≈ 2.24+.

[Figure 24]

[Figure 25]

[Figure 26]

Figure 1. An overview of the VideoKR training corpus. All videos are newly collected and CC licensed, and span a wide range of professional domains. We develop a skill oriented QA synthesis pipeline in which every example is grounded in one of three core skills essential for advanced video reasoning, and examples in the CoT subset are further paired with a high quality reasoning trace.

#### 1. Introduction

Multimodal foundation models for video understanding have achieved rapid progress in recent years, driven by architectural advances (Shu et al., 2025; Zohar et al., 2025; Ren et al., 2025; Li et al., 2024d; 2026), large-scale pretraining (Zhang et al., 2025a; Wang et al., 2025c; Bai et al.,

2025; Chen et al., 2025a), and sophisticated post-training strategies (Feng et al., 2025a; Meng et al., 2025; Wang et al.,

*Equal contribution 1Zhejiang University, Hangzhou, Zhejiang, China 2Tongji University, Shanghai, China 3University of Chinese Academy of Sciences, Beijing, China 4Yale University, New Haven, CT, USA. Correspondence to: Yilun Zhao <yilun.zhao@yale.edu>.

- 2025a; Li et al., 2025a;b). However, current models still face significant limitations when transitioning from surfacelevel video perception to video reasoning tasks that demand domain knowledge and multi-step inference (Zhao et al.,
- 2025b; Hu et al., 2025; Deng et al., 2025; Song et al., 2025). A key bottleneck lies in the nature of the training corpora

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

- Table 1. Comparison of VideoKR with prior post-training corpora for video understanding. %Video denotes the fraction of video understanding examples, CC indicates whether all videos are Creative Commons (CC) licensed. †: data has not been open-sourced.

Video Source %Video # Video Avg Duration CC # Example Example Source Expert-domain Example/CoT Generator

LLaVA-Video (Zhang et al., 2025c) Existing Dataset 100% 178K 36.9 seconds ✗ 1156K Newly Generated ✗ GPT-4o VideoEspresso (Han et al., 2025) Existing Dataset 100% 259K 47.7 seconds ✗ 202K Newly Generated ✗ GPT-4o Video-R1 (Feng et al., 2025a) Existing Dataset 52% 61K 36.9 seconds ✗ 260K Existing Dataset ✗ Qwen2.5-VL-72B-Instruct VideoRFT (Wang et al., 2025a) Existing Dataset 56% 127K 24.7 seconds ✗ 310K Existing Dataset ✗ GPT-4o-mini + DeepSeek-R1 †Video-CoT (Zhang et al., 2025b) Existing Dataset 100% Unknown Unknown ✗ 192K Existing Dataset ✗ Qwen2.5-VL-72B-Instruct OneThinker (Feng et al., 2025b) Existing Dataset 42% 158K 90.9 seconds ✗ 600K Existing Dataset ✗ Seed1.5-VL VideoAuto-R1 (Liu et al., 2026a) Existing Dataset 59% 35K 63.8 seconds ✗ 83K Existing Dataset ✗ –

Expert-validated selection from a pool of 7 frontier models

VideoKR (Ours) Newly Collected 100% 145K 344.1 seconds ✓ 315K Newly Generated ✓

used to develop these models.

Existing large-scale video datasets are predominantly constructed for perceptual objectives such as action recognition, event localization, and short-range temporal understanding (Feng et al., 2025a; Wang et al., 2025a; Meng et al., 2025; Chen et al., 2025b; 2026). Their content is heavily skewed toward everyday activities, with limited coverage of specialized domains and little support for knowledgeand reasoning-intensive video understanding. Consequently, models trained on current corpora often struggle with tasks requiring multi-hop inference, scientifically grounded explanations, or interpretation of events governed by nonobservable principles, limiting their reliability in real-world applications that demand accurate, domain-aware reasoning.

To bridge this gap, we open-source VideoKR, the first large-scale training corpus targeted for knowledge- and reasoning-intensive video understanding. We collect 145K CC-licensed videos across 82 professional subjects using a knowledge-driven collection protocol that targets realworld manifestations of domain knowledge. To transform these raw videos into effective video reasoning training data, we design a skill-oriented QA generation framework that decomposes knowledge- and reasoning-intensive video understanding into three complementary capabilities: basic video reasoning, knowledge-enhanced video perception, and knowledge-intensive video reasoning. For each video, the framework generates challenging QA examples tailored to each skill category, each paired with a high-quality CoT rationale. We apply rigorous quality control with humanexpert involvement, yielding a high-quality supervised finetuning corpus, VideoKR-SFT-201K, and a reinforcement learning corpus, VideoKR-RL-114K. In addition, through a manual audit of existing knowledge-intensive video reasoning benchmarks, we find that many examples are solvable with little video understanding. To address these issues, we construct a new evaluation benchmark, VideoKR-Eval.

We adopt a standard SFT→GRPO pipeline to isolate data design as the primary bottleneck and attribute performance gains more cleanly to VideoKR. We further establish a standardized evaluation framework to enable fair, reproducible

model comparisons. Experiments show that even without sophisticated post-training algorithmic design, base models (i.e., Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct) post-trained on VideoKR already outperform prior posttraining approaches. To inform how VideoKR can advance future work in video reasoning, we conduct comprehensive ablations that disentangle its key contributors, including the effect of CoT supervision, the impact of skill-based data composition, and controlled SFT and RL studies that compare VideoKR against prior post-training corpora.

We summarize our main contributions as follows:

- • We open-source VideoKR, the first large-scale training corpus designed for knowledge- and reasoning-intensive video understanding. We apply rigorous quality control to ensure consistently high-quality training data (§3).
- • We construct VideoKR-Eval, a new evaluation benchmark that mitigates single-frame answerability in prior benchmarks through multi-model single-frame probing and expert re-annotation of filtered videos (§4).
- • We establish a standardized evaluation framework to ensure fair and reproducible model comparisons (§5.2).
- • Models post-trained on VideoKR achieve the best knowledge-intensive performance among similar-sized models, while remaining competitive on general video benchmarks (§6).
- • We conduct comprehensive ablations to isolate the contributions of VideoKR, including the effectiveness of CoT supervision, the impact of skill-based data composition, comparisons against prior post-training corpora, yielding actionable insights for future work (§6.4 to §6.5).

#### 2. Related Work

Video Understanding Datasets. Recent video understanding benchmarks have expanded their scope to evaluate a broader range of multimodal and reasoning capabilities (Wu et al., 2021; Zhou et al., 2025; Fu et al., 2025; Zhao et al., 2025b; Deng et al., 2025; Li et al., 2024c; Liu et al., 2026b; Xu et al., 2025). General-purpose benchmarks such as Video-MME (Fu et al., 2025), MVBench (Li

[Figure 27]

###### VideoKR : Towards Knowledge- and Reasoning-Intensive Video Understanding

Feature SFT-201K RL-114K # Examples 201,156 114,381

Domain Knowledge Bank Construction (§3.1)

###### Knowledge-Driven Video Collection (§3.2)

[Figure 28]

Subject Course Lecture

[Figure 29]

[Figure 30]

[Figure 31]

1-3 Scenarios

| | |
|---|---|
| | |

[Figure 32]

# Multi-choice 99,843 54,461 Question Length 30.6 31.7 # Open-ended 101,313 59,920 Question Length 11.6 13.1

Generate search Search

[Figure 33]

[Figure 34]

[Figure 35]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 36]

keywords Construct knowledge bank

Corresponding Search Keywords

[Figure 37]

[Figure 38]

Videos with CC license

Name: Redox reaction Definition: A chemical process in which one substance loses electrons (...abbreviated…)

[Figure 39]

\ Knowledge Point

[Figure 40]

Judge video relevance to targeted scenarios

82 subjects across 4 disciplines

% VIDR 43.48% 35.66% % KNOWVID 33.02% 33.38% % KNOWVIDR 23.49% 30.96%

63,745 Knowledge Points

Skill-Oriented Example Generation (§3.3) Data Quality Control (§3.4)

# Knowledge Points 20,372 12,446 # Videos 85,934 59,625

Random Split:

Collected Video

Core Skills

Human-Validated Model Selection for Each Pipeline Step

[Figure 41]

Knowledge Point

Basic Video Perception

[Figure 42]

VideoKR-SFT-201K VideoKR-RL-114K

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Avg Length (second) 339.0 351.6 25s - 5min 57.61% 55.16% 5min - 10min 25.35% 26.61% 10min - 30min 17.04% 18.23%

[Figure 53]

[Figure 54]

[Figure 55]

Knowledge Reasoning

For each step, sample 100 examples per model, have them validated by experts, select models that meet quality criteria for that step

[Figure 56]

[Figure 57]

Generate

Expert annotate seed examples

Quality Check

[Figure 58]

[Figure 59]

Data Contamination Mitigation

[Figure 60]

[Figure 61]

- 1. Self-consistency Check
- 2. Video Dependency Filter
- 3. CoT Rationale Validation

- 1. YouTube-ID Filtering
- 2. Near-Duplicate Video Filtering via frame hashing then sequence matching

CoT Rationale Length 196.9 –

150 examples ✕ 3 skills ✕ 4 disciplines

New Examples

- Figure 2. (Left) Overview of data construction pipeline. (Right) Statistics of VideoKR-SFT-201K and VideoKR-RL-114K training corpus.

- et al., 2024b), VSI-Bench (Yang et al., 2025b), and VideoVista (Li et al., 2024c) assess perceptual skills, spatiotemporal comprehension, and cross-modal reasoning, providing a solid foundation for evaluating video understanding. Building on this trend, a growing set of knowledge-, science-, and reasoning-intensive evaluation benchmarks focuses on deeper, domain-aware reasoning that goes beyond surface-level video perception (Xu et al., 2025; Liu et al., 2026b). For instance, MMVU (Zhao et al., 2025b) requires models to reason over specialized-domain videos and apply relevant domain knowledge; VideoMMMU (Hu
- et al., 2025) and Video-MMLU (Song et al., 2025) target expert-level understanding of subject-specific lecture videos; and SciVideoBench (Deng et al., 2025) evaluates advanced reasoning over scientific videos.

basic perception: as shown in Table 1, existing open-source corpora largely rely on short videos from datasets released years ago, and synthesis-based efforts (Zhang et al., 2025c; Han et al., 2025) typically depend on a single model, which can introduce systematic biases. We address this need by open-sourcing VideoKR: every video is newly collected, depicts expert-domain scenarios, and is released under CC licenses, and we adopt a human-in-the-loop, skill-oriented example generation framework to ensure the difficulty, diversity, and reliability of the data. Our experiments show that, under a standard SFT→GRPO pipeline, models posttrained on VideoKR already outperform prior post-training approaches, suggesting that better data remains crucial.

#### 3. VideoKR Training Corpus Construction

This section describes the VideoKR data construction process, with an overview of the pipeline shown in Figure 2. Because our goal is large-scale corpus synthesis, exhaustive manual construction is infeasible. However, model based generation can introduce systematic artifacts. We therefore adopt a quality-controlled, semi-automated pipeline: whenever a step involves model outputs, it is audited and validated by human experts (detailed in Section 3.4). We engage 34 domain experts, each with graduate-level background in the relevant discipline (see Appendix A.2 for annotator biographies), to enforce quality criteria throughout the process.

Post-training for Video Understanding. Current reasoning models are typically trained through a two-stage posttraining pipeline that combines SFT and RL (Guo et al., 2025; Team et al., 2025; Yang et al., 2025a; Li et al., 2025c). To enhance video reasoning capabilities, the SFT stage is usually initialized on video reasoning datasets that include explicit chain-of-thought annotations, temporal cues, and spatial grounding signals, helping the model form more structured and interpretable reasoning patterns (Munasinghe et al., 2025; Zohar et al., 2024; Wu et al., 2025; Feng et al., 2025b; Wang et al., 2025b). In the RL phase, recent work has concentrated on adapting reinforcement learning with verifiable rewards (RLVR) to video reasoning, exploring complex reward engineering that emphasizes spatial understanding (Ouyang et al., 2025; Tang et al., 2025a), temporal dynamics (Wang et al., 2025d; Chen et al., 2025b; Feng et al., 2025a; Zhao et al., 2025a), or integrated spatiotemporal relationships (Meng et al., 2025; Tang et al., 2025b). Despite these advances, most post-training approaches still build on repurposed video understanding datasets that target

##### 3.1. Domain Knowledge Bank Construction

To achieve comprehensive coverage of domain-related videos, we begin by constructing a Domain Knowledge Bank, where each entry represents a knowledge point consisting of a term and its corresponding definition. The authors manually reviewed undergraduate curricula from top universities worldwide and identified 82 representative subjects distributed across four major disciplines: Natural Sci-

ences, Healthcare, Humanities and Social Sciences, and Engineering. The complete subject list is provided in Appendix A.1. To ensure systematic and fine-grained representation of domain knowledge within each discipline, we adopt a hierarchical knowledge organization framework with four layers: Subject → Course → Lecture → Knowledge Point. Specifically, for each subject, we ask expert annotators to provide a list of 4 to 8 core undergraduate courses consistent with standard academic programs. For each selected course, annotators compile a structured syllabus based on well-established curricula from top universities, outlining major lecture topics and learning objectives. Then for each lecture, we prompt LLMs (see Section 3.4 for our expertinvolved procedure to select and validate the models used here and in subsequent steps) to identify several key knowledge points, each paired with a term and a paragraph-length definition. We collect a total of 63,745 knowledge points.

##### 3.2. Knowledge-Driven Video Collection

Building upon the constructed domain knowledge bank, we then collect a large-scale video corpus, outlined below.

Knowledge-based Video Scenario Generation. In practice, directly using knowledge point terms (e.g., “Newton’s Second Law”) as search queries often yields lecture recordings or purely instructional materials. While informative, such videos often lack the diversity and situational richness found in real-world contexts where the knowledge is implicitly applied. To obtain more authentic and engaging content, we first ask LLMs to generate 1–3 short scenarios that describe realistic situations involving each knowledge point. For example, instead of searching “Newton’s Second Law”, a generated scenario might be “a rocket launching into the sky”, which inherently reflects Newton’s Second Law. These scenarios are then transformed into semantically relevant search keywords, helping retrieve videos that embody the knowledge rather than merely explain it.

Scenario-Guided Video Search and Filtering. For each generated search keyword, we employ the YouTube Data API1 to retrieve metadata (e.g., titles, descriptions, and durations) for the top-10 candidate videos. We restrict the search to videos released under CC licenses to ensure legal reusability, a critical aspect that has been ambiguous in prior training corpora. For each candidate video, we instruct models to evaluate its relevance to the expected knowledge point and scenario based on the textual metadata. Videos exceeding 30 minutes are excluded, as long-context video understanding falls beyond the scope of this work. For the remaining candidates, we download the videos and prompt MLLMs to perform a secondary relevance assessment using visual content to confirm alignment with the intended knowledge context. To remove potentially harmful or sensitive

1https://developers.google.com/youtube/v3/docs.

content, we randomly sample four frames from each video and run Azure AI’s image moderation APIs2 to filter out unsafe videos. We collect 146,567 CC-licensed videos.

##### 3.3. Skill-Oriented Example Generation

For each video, we generate multiple QA examples. Following recent video post-training work (Feng et al., 2025a; Meng et al., 2025), we adopt multi-choice and open-ended QA formats, as they offer verifiable supervision suitable for RLVR. To enable scalable and high-quality data creation, we design a skill-based example generation pipeline:

Core Skill Categorization. As illustrated in Figure 1, three complementary dimensions, i.e., perception, knowledge, and reasoning, are essential for knowledge- and reasoning-intensive video understanding. Accordingly, we define three core skills that guide the example generation process: (1) Basic Video Reasoning (VIDR), which involves direct comprehension of events observable from the visual sequence, such as tracking actions, spatial relations, or temporal order, without relying on external domain knowledge. (2) Knowledge-enhanced Video Perception (KNOWVID), where visual perception is enriched by explicit domain knowledge. The model must align observed visual cues with relevant concepts across both spatial and temporal dimensions, for example, recognizing laboratory apparatus such as a “burette” or “condenser” and understanding their roles in a sequence of chemical procedures. (3) Knowledge-Intensive Video Reasoning (KNOWVIDR), which focuses on integrating visual understanding with domain knowledge to perform sophisticated, multi-hop inference, e.g., estimating the quantity of chemical product formed from observed reactant amounts, or inferring a patient’s likely diagnosis by interpreting symptoms and medical procedures depicted in a clinical video.

Seed Examples Curation By Human Experts. To ensure the quality and domain accuracy of the generated examples, we engage expert annotators to curate a seed set of examples for every core skill defined above. For each core skill, the annotators select representative knowledge points and their corresponding collected videos, then construct question–answer pairs accompanied by detailed, step-by-step reasoning processes that clearly articulate how visual evidence, domain knowledge, and logical inference jointly lead to the final answer. Each annotated example then undergoes a manual review by the authors to verify the accuracy of the QA content and reasoning process. In total, 150 examples are created for each skill within every discipline, resulting in 1,800 high-quality, expert-curated seed examples. To improve reliability, each example is independently reviewed by a second annotator; 74 examples are revised in this stage.

2https://learn.microsoft.com/en-us/azure/ai-services/contentsafety/quickstart-image.

Example Generation. Building on the expert-curated seed set, we use frontier MLLMs to scale up example creation in a controlled, skill-aware manner. For each video, the model generates two examples per skill, producing six examples in total through six independent generation rounds (one per example). During each round, the model is provided with (1) video frames uniformly sampled at 0.2 fps with timestamps, (2) three randomly sampled human-curated examples from the same discipline and skill category, and (3) the knowledge point and associated subjects when targeting the KNOWVID or KNOWVIDR skills. The model is instructed to emulate the seed examples in QA formulation and reasoning process generation, while maintaining fidelity to the video’s unique visual content and knowledge context.

Example Validation and Filtering. To mitigate generation errors and reasoning bias, we adopt three complementary strategies: (1) Self-Consistency Verification: The model is re-prompted with the generated question and corresponding video frames to produce a detailed, step-by-step answer. An example is retained only if the re-derived answer matches the original, and the reasoning process from this verification step is used as the final reasoning trace.

- (2) Video Dependency Filtering: To ensure that each example in VideoKR genuinely requires visual understanding rather than relying on textual cues or shortcut reasoning, we instruct InternVL3.5-38B and Qwen3-VL-32B-Instruct to answer the question using only the text and four randomly sampled video frames. If both models successfully predict the correct answer under this limited setting, the example is removed from the dataset. Notably, this filter is stricter than what existing evaluation benchmarks use, which commonly rely on text-only (Fu et al., 2025; Di & Xie, 2024) or singleframe (Saravanan et al., 2025; Plizzari et al., 2025) settings.

- (3) CoT Rationale Validation: To mitigate systematic generator bias in the reasoning traces, we use an independent strong MLLM as a verifier: given the question, reasoning trace, and videos, it checks that each key step is supported by observable evidence or standard domain knowledge, and that the reasoning decisively distinguishes the chosen answer from plausible alternatives. We discard examples with critical unsupported steps.

##### 3.4. VideoKR Data Quality Control

Beyond the automated example validation and filtering described above, we further strengthen VideoKR through pipeline-level quality control and contamination mitigation.

Human-Validated Model Selection for Each Pipeline Step. As summarized in Table 1, prior work on video reasoning corpus construction typically relies on a single model throughout the pipeline. This design can introduce model specific artifacts. To improve synthesis diversity and avoid overcommitting to any single model’s biases, we use a pool of seven frontier models (i.e., GPT-5.2, GPT-5-

mini, Claude-4.5-Sonnet, Gemini-3-Flash, DeepSeek-V3.2, Qwen3-VL-235B-A22B, and GLM-4.6V). However, our pipeline is difficulty stratified: lightweight steps such as metadata relevance screening can be handled reliably by multiple models, whereas demanding stages such as QA example generation and verification are reliable only for a subset of models. We therefore introduce a human validated model selection protocol to determine stage eligibility: For each candidate model and each pipeline step, we sample 100 instances from that step’s real inputs and ask domain expert annotators to assess the model outputs and label errors. A model is eligible for a step only if its error rate falls below a predefined threshold. We provide details of this process and the human-validated models for each step in Appendix A.4. During large-scale synthesis, for each instance at a given step, we randomly select one qualified model from the pool.

Data Contamination Mitigation. To prevent evaluation leakage, we traverse all video benchmarks supported by LMMs-Eval (Li et al., 2024a) and apply a two-stage decontamination protocol over the videos in VideoKR: (1) YouTube-ID Filtering: When a benchmark (e.g., MMVU) provides YouTube video IDs, we directly filter out any training video whose YouTube ID matches an evaluation video ID, resulting in 131 videos removed. (2) Near-Duplicate Video Filtering: We also perform duplicate detection using frame level perceptual hashing and windowed sequence matching, resulting in 877 videos removed. We detail the process in Appendix A.5.

Manual Quality Assessment. To further assess the endto-end quality of the finalized corpus and quantify residual noise, we randomly sample 800 examples from VideoKRSFT-201K and ask ten expert annotators, who previously participated in seed-example curation, to evaluate them end to end. Of the sampled items, 52 questions are flagged as potentially non visual solvable. For reasoning traces, annotators identify 32 errors. Among these, 17 cases change the final answer, while 15 cases preserve the correct answer but rely on unsupported domain claims or fail to ground key steps in the relevant video evidence. These error rates are comparable to the error levels observed during human expert seed example curation (§3.3) and are therefore acceptable.

##### 3.5. VideoKR-SFT-201K & VideoKR-RL-114K

We then randomly partition the generated 315,537 examples into two subsets while preserving video-level grouping, resulting in VideoKR-SFT-201K for supervised fine-tuning and VideoKR-RL-114K for RLVR training. For VideoKRSFT-201K, each example retains its validated CoT rationale as the supervision target, whereas VideoKR-RL-114K keeps only the question and verifiable answer, since RLVR optimizes against the verifiable answer while the policy model generates its own reasoning during training. Figure 2 presents the key data statistics. Randomly-sampled

- Table 2. Single-frame answerability rates across existing benchmarks. A QA example is classified as single-frame-solvable for a model only if the model answers it correctly in all three independent trials using only the question, answer options, and one randomly sampled video frame.

VidMMMU MMVU SciVidBench VideoKR-Eval

Model

(900) (1,000) (1,000) (ours) (2,000)

Claude-4.5-Sonnet 35.3 41.3 21.8 9.5 Qwen3-VL-235B-A22B 39.3 45.2 13.2 10.1 GPT-5.2 38.3 49.7 23.0 10.7

examples from VideoKR are shown in Appendix A.3.

- 4. VideoKR-Eval Evaluation Benchmark We next discuss the motivation and build of VideoKR-Eval.

##### 4.1. Limitations of Existing Benchmarks

We observe that existing knowledge-intensive video reasoning benchmarks (e.g., VideoMMMU, MMVU, and SciVideoBench) contain a substantial fraction of examples that can be answered without continuous video understanding. We quantify this issue through single-frame probing: each model is given only the question, answer options, and one randomly sampled frame from the video, and each example is evaluated in three independent trials. As shown in Table 2, frontier models achieve surprisingly high single-frame answerability rates on existing benchmarks (e.g., >35% on MMVU and VideoMMMU).

##### 4.2. VideoKR-Eval Benchmark Construction

We construct VideoKR-Eval from VideoMMMU, MMVU, and SciVideoBench by retaining 1,254 original examples that require continuous video understanding under multimodel single-frame probing, and augmenting them with 746 expert-reannotated examples from filtered videos.

Multi-Model Single-Frame Filtering. For each example in VideoMMMU, MMVU, and SciVideoBench, we run single-frame probing with three frontier models: Qwen3VL-235B-A22B, Claude-4.5-Sonnet, and GPT-5.2. Each model receives only the question, answer options, and one randomly sampled frame from the video, and is evaluated with three independent trials. For each model, an example is considered single-frame-solvable if the model answers it correctly in all three trials; otherwise, it is treated as requiring continuous video understanding for that model. We retain only the intersection of examples judged as requiring continuous video understanding by all three models, yielding 1,254 original examples.

Expert Re-annotation of Filtered Videos. For examples outside this intersection, we discard the original QA pairs and ask domain experts to re-annotate new QA examples

using the corresponding videos. Annotators are required to write questions grounded in clearly observable video evidence, requiring relevant domain knowledge, and paired with uniquely determined ground truth answers. This process yields 746 expert-reannotated examples. Together with the 1,254 retained original examples, VideoKR-Eval contains 2,000 examples. Detailed statistics for VideoKR-Eval are provided in Appendix B.1.

#### 5. Experiment Setup

In this section, we discuss the experiment setup for posttraining on VideoKR and the subsequent model evaluation.

##### 5.1. Post-Training on VideoKR

Recent post-training work for video reasoning emphasizes sophisticated RL variants and reward engineering. In contrast, we aim to isolate a different, and arguably more fundamental, bottleneck: whether data design is the primary limiting factor for knowledge and reasoning intensive video understanding. Accordingly, we deliberately adopt a standard, widely used SFT→GRPO pipeline as a controlled scaffold, ensuring that algorithmic complexity does not become a confounder and that observed gains can be attributed more cleanly to the training data.

Specifically, we use Qwen2.5-VL-7B-Instruct and Qwen3VL-8B-Instruct as base models to assess whether VideoKR yields consistent gains under various architectural designs and pretraining priors. For SFT, we fine-tune each base model on VideoKR-SFT-201K for one epoch. Starting from the resulting SFT checkpoint, we then run GRPO on VideoKR-RL-114K for one epoch. For Qwen3-VL-8BInstruct, we also conduct Zero-RL training by directly running GRPO on VideoKR-RL-114K for one epoch. We set batch size as 32 for both SFT and GRPO. For the GRPO accuracy reward, we follow prior video reasoning work (Feng et al., 2025a; Wang et al., 2025a) and use ROUGE for openended QAs and Exact Match for multiple-choice QAs. The maximum video token number is 4,096, and the maximum number of frames is 128. The training hyperparameters and details are provided in Appendix C.1.

##### 5.2. Evaluation Setup

We next describe our evaluation benchmarks and the standardized protocol for fair, reproducible model comparisons.

Evaluation Benchmarks. We evaluate models on seven benchmarks grouped into two categories: (1) General Video Reasoning, including Video-MME (Fu et al., 2025), MVBench (Li et al., 2024b) and LongVideoBench (Wu et al., 2024), which measure broad video understanding; and (2) Knowledge-intensive Video Reasoning, including VideoMMMU (Hu et al., 2025), MMVU (Zhao et al., 2025b),

- Table 3. Benchmark results across general and knowledge-intensive video reasoning. Models are grouped into (i) Other Models and (ii) methods built on Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct (with the indicated input Frames). Within each group for (ii), the best score is bold and the second-best is underlined.

General Video Reasoning Knowledge-Intensive Video Reasoning

Video-MMEMVBench LongVideoBench

VideoMMMUMMVU SciVideoBenchVideoKR-Eval

Average

Average

Model Release Frames

Closed-source models

GPT-5.4 2026-03 64 86.0 78.3 76.7 80.3 87.1 82.0 52.9 63.2 71.3 Gemini 3 Pro 2025-11 64 87.7 74.1 77.4 79.7 87.6 77.5 50.4 60.3 69.0 Claude Opus 4.5 2025-11 64 81.4 67.2 67.2 71.9 84.4 77.3 48.6 56.3 66.7

###### Other Models

Qwen3-VL-32B-Thinking 2025-10 128 72.9 71.7 63.8 69.5 69.6 67.8 42.1 50.2 57.4 Qwen3-VL-32B-Instruct 2025-10 128 74.4 72.6 65.5 70.8 72.0 68.2 39.7 45.0 56.2 Qwen2.5-VL-72B 2025-02 128 72.8 67.4 63.2 67.8 67.0 65.1 38.9 42.6 53.4 InternVL3.5-8B 2025-08 128 65.5 73.3 61.0 66.6 57.2 54.0 24.5 35.4 42.8 LLaVA-OneVision-7B 2024-08 32 59.0 58.1 56.5 57.9 36.2 43.1 16.2 23.5 29.8 LLaVA-NeXT-Video-34B 2024-07 32 51.0 48.0 49.9 49.6 19.1 39.4 16.0 18.8 23.3 LLaVA-NeXT-Video-7B 2024-07 32 32.0 38.1 38.9 36.3 21.8 28.1 10.9 14.7 18.9

###### Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct as Base Models

Qwen2.5-VL-7B-Instruct 2025-02 16 57.1 65.0 55.2 59.1 48.4 52.5 23.1 31.3 38.8

Video-R1 2025-03 16 59.7 65.5 55.3 60.2+1.1 51.1 53.3 26.6 28.9 40.0+1.2 VideoRFT 2025-05 16 57.6 61.7 53.6 57.6−1.5 51.1 53.6 26.3 29.8 40.2+1.4 VideoKR (SFT + RL) 2026-05 16 56.6 66.6 57.0 60.1+1.0 52.6 59.2 27.3 37.7 44.2+5.4

Qwen2.5-VL-7B-Instruct 2025-02 128 65.1 66.3 60.9 64.1 51.1 55.7 28.1 32.7 41.9 VideoAuto-R1 2026-01 128 66.8 70.2 59.7 65.6+1.5 52.1 55.7 32.7 36.5 44.3+2.4

- VideoKR (SFT + RL) 2026-05 128 66.4 68.9 61.3 65.5+1.4 52.2 60.5 32.5 41.2 46.6+4.7

Qwen3-VL-8B-Instruct 2025-10 128 68.2 67.9 61.6 65.9 61.8 59.6 33.4 39.0 48.5

OneThinker 2025-12 128 65.8 69.3 61.4 65.5−0.4 62.9 61.6 33.8 38.3 49.2+0.7 VideoAuto-R1 2026-01 128 68.7 68.8 58.8 65.4−0.5 63.1 59.6 32.7 43.8 49.8+1.3 Qwen3-VL-8B-Thinking 2025-10 128 67.6 68.0 60.0 65.2−0.7 64.9 60.5 33.0 41.5 50.0+1.5 VideoKR (SFT) 2026-05 128 64.8 63.6 58.5 62.3−3.6 61.7 63.0 28.3 43.6 49.2+0.7 VideoKR (zero RL) 2026-05 128 67.4 65.5 60.0 64.3−1.6 61.9 63.5 32.5 44.6 50.6+2.1

- VideoKR (SFT + RL) 2026-05 128 67.8 67.0 61.5 65.4−0.5 63.0 64.8 32.8 45.3 51.5+3.0

SciVideoBench (Deng et al., 2025), and VideoKR-Eval, which focus on domain-specific, expert-level reasoning.

Reproducibility Challenges in Prior Post-Training Work. We observe substantial cross-paper inconsistencies in reported results, particularly for base models used as posttraining starting points. Based on careful follow-up experiments, we attribute these discrepancies primarily to prompt misalignment: base models are sometimes evaluated under prompt conditions that are misaligned with their intended inference mode. For example, Qwen2.5-VL-Instruct is not a “reasoning” model, yet some papers evaluate it with elaborate self-reflection and forced reasoning-trace instructions designed for post-trained ‘reasoning” variants.

Standardizing Evaluation for Fair Model Comparisons. To ensure fair and reproducible comparisons, for each model, we use the official prompt released by the original paper whenever available; otherwise, we adopt the default prompt templates from LMMs-Eval (Li et al., 2024a). For input frames, we follow the model-recommended inference

configuration whenever it is specified by the model release. We run each model three times with independent sampling and report the mean. All evaluations are performed using the LMMs-Eval framework (Li et al., 2024a). Full evaluation details (e.g., model prompts and inference parameters) are provided in Appendix C.2.

#### 6. Experiment Results and Analysis

We next discuss our main findings and ablation analysis.

##### 6.1. Main Results

Table 3 presents the main experimental results. Posttraining on VideoKR consistently improves Qwen2.5VL-7B-Instruct across all evaluated benchmarks, and yields clear gains for Qwen3-VL-8B-Instruct especially on knowledge-intensive video reasoning benchmarks. The gains are most pronounced on knowledge-intensive tasks: SFT followed by RL on VideoKR raises the knowledge-

ing all other inference settings fixed. As shown in Figure 3, increasing the number of input frames consistently improves performance for both the base model and our post-trained model. For example, on general video reasoning benchmarks, Qwen2.5-VL-7B (SFT+RL) improves from 60.1% at 16 frames to 65.5% at 128 frames. On knowledge-intensive video reasoning benchmarks, it further improves from 44.2% to 46.6%. These results suggest that our model benefits from richer visual and temporal evidence at inference time, and that the gains from VideoKR remain consistent across different frame budgets.

70

55

Qwen2.5-VL-7B

Qwen2.5-VL-7B

Qwen2.5-VL-7B (SFT+RL)

Qwen2.5-VL-7B (SFT+RL)

68

50

65.5

66

45.8 46.6

63.8

44.2 44.3

64

45

64.1 60.1

62.2

41.9

40.5 41.0

62.8

62

38.8

61.8

40

60

59.1

58

35

56

30

16 32 64 128

16 32 64 128

Input Frames

Input Frames

(a) General Reasoning

(b) Knowledge-intensive Reasoning

In the following subsections, we conduct controlled SFT and RL ablations on VideoKR to provide insights for future work. Unless otherwise specified, all experiments use Qwen2.5-VL-7B-Instruct as the base model and are trained and evaluated with 128 input frames.

- Figure 3. Inference-time frame scaling results on general and knowledge-intensive video reasoning benchmarks. The figure shows category-wise average accuracies for Qwen2.5-VL-7BInstruct and its VideoKR post-trained variant (SFT+RL) under different input frame budgets. Appendix D.1 provides the full per-benchmark results for post-trained Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct models.

##### 6.4. Ablations on VideoKR-SFT-201K

intensive average of Qwen2.5-VL-7B from 41.9 to 46.6 (+4.7) and of Qwen3-VL-8B from 48.5 to 51.5 (+3.0), with the largest per-dataset improvements on MMVU and VideoKR-Eval (e.g., +4.8 and +8.5 points for Qwen2.5-VL7B). Notably, the post-trained Qwen3-VL-8B attains the best knowledge-intensive average among 7/8B-scale models (51.5, versus 50.0 for the strongest competing model, Qwen3-VL-8B-Thinking). These results underscore the value of our training data, which tightly integrates domain knowledge, visual grounding, and structured reasoning. Starting from an SFT-initialized model, subsequent RL training consistently delivers higher performance than the SFTonly baseline, highlighting that combining SFT with RL is important for fully leveraging the strengths of VideoKR data. Moreover, the RL-only variant also generally outperforms the SFT-only model, indicating that RL alone can induce stronger generalizable reasoning abilities.

##### 6.2. Case Study

To better understand how post-training on VideoKR changes model behavior, we randomly sample 100 examples from VideoKR-Eval and compare outputs from different models. Our case study shows that the post-trained Qwen3-VL-8B model can integrate visual evidence with relevant domain knowledge to perform complex reasoning. It also exhibits “aha-moment” reasoning patterns indicative of deeper understanding. Examples are shown in Appendix D.3.

##### 6.3. Analysis of Inference-Time Frame Scaling

To analyze how input frame count affects performance, we evaluate our post-trained model, which was trained with 128 frames, under varying numbers of frames at inference. Specifically, we test 16, 32, 64, and 128 frames while keep-

Using VideoKR-SFT-201K, we ablate two design choices, the skill composition of the training examples and the use of CoT supervision, to quantify their individual contributions.

Skill-Oriented Data Composition. To assess the contribution of each skill component, we fine-tune models on cumulative subsets of our skill-oriented data. Specifically, we construct three 80K-example variants from VideoKRSFT-201K: (1) VIDR only; (2) a balanced mixture of VIDR and KNOWVID (1:1); and (3) a randomly sampled subset from the full VideoKR-SFT-201K (i.e., VIDR +KNOWVID +KNOWVIDR, we reuse examples in the previous ablation). We fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. We observe that incorporating all three skill components yields the best knowledge-intensive performance: training on VIDR alone obtains 41.4% on knowledge-intensive benchmarks. Adding Knowledge-Enhanced Perception (VIDR +KNOWVID) gives 41.3%, while incorporating KnowledgeIntensive Reasoning (VIDR +KNOWVID +KNOWVIDR) further improves performance to 42.4%. The same trend holds on VideoKR-Eval, where accuracy rises monotonically from 35.3 (VIDR) to 35.9 (VIDR +KNOWVID) to 36.8 (VIDR +KNOWVID +KNOWVIDR). These results indicate that combining domain knowledge and complex reasoning supervision is crucial.

CoT vs. Direct Output. To validate the necessity of explicit CoT rationales in SFT, we randomly sample 80K examples from VideoKR-SFT-201K and create two variants: one with CoT rationales and one without. We fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. As illustrated in Table 4, the CoT-trained model improves over the Direct Output baseline on knowledge-intensive reasoning, lifting the average from 39.4% to 42.4% (a 3.0-point gain), underscoring the

- Table 4. Ablation studies on post-training data. All experiments use Qwen2.5-VL-7B-Instruct as the base model, with 128 input frames. The complete results are provided in Appendix D.2.

General Knowledge-Intensive Average VideoKR-Eval Average

Ablation Setting

Qwen2.5-VL-7B-Instruct 64.1 32.7 41.9 Skill-Oriented Data Composition (SFT, 80K examples, one epoch)

VIDR 58.0−6.1 35.3+2.6 41.4−0.5 VIDR + KNOWVID 58.4−5.7 35.9+3.2 41.3−0.6 VIDR + KNOWVID + KNOWVIDR 58.3−5.8 36.8+4.1 42.4+0.5 CoT Supervision Format (SFT, 80K examples, one epoch)

Direct Output 61.4−2.7 35.9+3.2 39.4−2.5 Chain-of-Thought 58.3−5.8 36.8+4.1 42.4+0.5

###### Comparison with Other SFT Corpora (SFT, 80K examples, one epoch)

Video-R1-CoT-165k 57.3−6.8 27.5−5.2 36.2−5.7 OneThinker-SFT-340k 60.5−3.6 29.7−3.0 38.3−3.6 VideoRFT-CoT-102K 59.6−4.5 32.1−0.6 38.4−3.5 VideoKR-SFT-201K (Ours) 58.3−5.8 36.8+4.1 42.4+0.5 Comparison with Other RL Corpora (GRPO, 50K examples, one epoch)

Video-R1-260k 63.7−0.4 33.1+0.4 41.6−0.3 OneThinker-600k 60.3−3.8 33.2+0.5 42.3+0.4 VideoRFT-RL-310K 63.8−0.3 33.5+0.8 42.3+0.4 VideoAuto-R1-83K 62.8−1.3 33.3+0.6 42.7+0.8 VideoKR-RL-114K (Ours) 61.7−2.4 34.5+1.8 43.0+1.1

importance of high-quality CoT supervision for advanced knowledge-intensive video reasoning.

##### 6.5. VideoKR vs Prior Post-Training Corpus

We next conduct a comprehensive comparison of VideoKR with prior open-source post-training corpora, under both SFT and zero-RL settings. For SFT, we randomly sample 80K examples from each SFT corpus and fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. For zero-RL, we randomly sample 50K QA examples from each RL corpus and train Qwen2.5VL-7B-Instruct on each variant for one epoch with a batch size of 16; we exclude non-QA tasks since they require task-specific reward functions that cannot be unified under our standard GRPO training pipeline.

Main Findings. As illustrated in Table 4, under SFT, the model trained on the VideoKR-SFT subset reaches a 42.4 knowledge-intensive average and is the only corpus to surpass the base model (41.9), whereas prior corpora such as Video-R1 and VideoRFT lower it to 36.2 and 38.4, respectively. Under zero-RL, VideoKR with VideoKR-RL-114K achieves the strongest gain (43.0, +1.1 over the base model, ahead of the next-best VideoAuto-R1 at 42.7), indicating that high-quality data is key to maximizing post-training benefits for advanced video reasoning.

Training-Data Difficulty Analysis. To diagnose why prior corpora provide limited improvements relative to VideoKR, we analyze training-data difficulty with respect to the base models. Concretely, we randomly sample 3,000

Table 5. Accuracy of Qwen2.5/3-VL models on 3,000 randomly sampled QA examples from various post-training corpora.

Model Video-R1 VideoRFT OneThinker VidAuto-R1 VideoKR

2.5-VL-7B-Inst. 55.3 47.8 45.8 57.1 39.2 3-VL-8B-Inst. 57.1 51.1 49.1 54.5 42.3 3-VL-8B-Think 59.0 52.3 49.3 54.3 43.5

video QA examples from each corpus and measure the zeroshot accuracy of Qwen2.5-VL-7B and Qwen3-VL-8B in the 128-frame setting. As shown in Table 5, all evaluated models attain high accuracy on prior corpora (e.g., Qwen3VL-8B scores between 49.1% and 57.1%), suggesting these datasets are effectively saturated for current frontier base models and thus offer weak learning signals. In contrast, accuracy on VideoKR remains lower (42.3% for the same model), indicating a more challenging distribution that better supports continued capability gains during post-training.

#### 7. Conclusion

This work offers a corpus-centric perspective on posttraining foundation models for advanced video reasoning. Instead of viewing visual perception, domain knowledge, and advanced reasoning as loosely linked elements, we show that integrating structured domain concepts with visually grounded examples yields stronger reasoning performance, without relying on sophisticated RL reward engineering. Extensive experiments and analyses confirm that post-training on the VideoKR dataset produces strong improvements, especially on knowledge-intensive video reasoning.

#### Impact Statement

All videos in VideoKR are licensed under CC license, which enables reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. By restricting our dataset to CC-licensed videos, we ensure legal reusability and clear provenance, a critical aspect that has been ambiguous in prior training corpora. For human expert annotation and validation, we compensated annotators at an average rate of $13 USD per hour, which exceeds the prevailing hourly rates for comparable local work. All expert annotators provided informed consent to participate and explicitly authorized the public release and redistribution of their annotations as part of the resulting dataset and accompanying materials. The VideoKR and VideoKR-Eval data construction incurs approximately 70.4K US dollars in model inference costs.

#### Acknowledgments

We thank Dr. Yu Rong for valuable suggestions on the posttraining experiment design and paper writeup improvement.

#### References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,

- Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
- Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Chen, G., Li, Z., Wang, S., Jiang, J., Liu, Y., Lu, L., Huang, D.-A., Byeon, W., Le, M., Rintamaki, T., Poon, T., Ehrlich, M., Rintamaki, T., Poon, T., Lu, T., Wang, L., Catanzaro, B., Kautz, J., Tao, A., Yu, Z., and Liu, G. Eagle 2.5: Boosting long-context post-training for frontier vision-language models, 2025a. URL https:

//arxiv.org/abs/2504.15271.

Chen, R., Fan, Z., Luo, T., Zou, H., Feng, Z., Xie, G., Zhang, H., Wang, Z., Liu, Z., and Zhang, H. Datasets and recipes for video temporal grounding via reinforcement learning, 2025b. URL https://arxiv.org/abs/ 2507.18100.

Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., QI, X., Liu, S., Yin, H., Lu, Y., and Han, S. Scaling RL to long videos. 2026. URL https://openreview.net/forum? id=TxedB8hI5O.

Deng, A., Yang, T., Yu, S., Spencer, L., Bansal, M., Chen, C., Yeung-Levy, S., and Wang, X. Scivideobench: Benchmarking scientific video reasoning in large multimodal models, 2025. URL https://arxiv.org/abs/ 2510.08559.

Di, S. and Xie, W. Grounded question-answering in long egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12934–12943, 2024.

Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., and Yue, X. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025a.

Feng, K., Zhang, M., Li, H., Fan, K., Chen, S., Jiang, Y., Zheng, D., Sun, P., Zhang, Y., Sun, H., et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025b.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Han, S., Huang, W., Shi, H., Zhuo, L., Su, X., Zhang, S., Zhou, X., Qi, X., Liao, Y., and Liu, S. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26181–26191, 2025.

Hu, K., Wu, P., Pu, F., Xiao, W., Zhang, Y., Yue, X., Li, B., and Liu, Z. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. URL https://arxiv.org/abs/2501.13826.

Li, B., Zhang, P., Zhang, K., Pu, F., Du, X., Dong, Y., Liu, H., Zhang, Y., Zhang, G., Li, C., and Liu, Z. Lmmseval: Accelerating the development of large multimodal models, March 2024a. URL https://github.com/ EvolvingLMMs-Lab/lmms-eval.

Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., Wang, L., and Qiao, Y. Mvbench: A comprehensive multi-modal video understanding benchmark, 2024b. URL https://arxiv.

org/abs/2311.17005.

- Li, X., Yan, Z., Meng, D., Dong, L., Zeng, X., He, Y., Wang, Y., Qiao, Y., Wang, Y., and Wang, L. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025a.
- Li, Y., Chen, X., Hu, B., Wang, L., Shi, H., and Zhang, M. Videovista: A versatile benchmark for video understanding and reasoning, 2024c. URL https://arxiv. org/abs/2406.11303.

Li, Y., Jiang, S., Hu, B., Wang, L., Zhong, W., Luo, W., Ma, L., and Zhang, M. Uni-moe: Scaling unified multimodal llms with mixture of experts, 2024d. URL https:// arxiv.org/abs/2405.11273.

Li, Y., Chen, X., Li, Z., Liu, Z., Wang, L., Luo, W., Hu, B., and Zhang, M. Veripo: Cultivating long reasoning in video-llms via verifier-gudied iterative policy optimization, 2025b. URL https://arxiv.org/abs/ 2505.19000.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., Chen, P., Li, Y., Lin, S., Zhao, S., Li, K., Xu, T., Zheng, X., Chen, E., Shan, C., He, R., and Sun, X. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2025. URL https:

//arxiv.org/abs/2405.21075.

Li, Y., Liu, Z., Li, Z., Zhang, X., Xu, Z., Chen, X., Shi, H., Jiang, S., Wang, X., Wang, J., Huang, S., Zhao, X., Jiang, B., Hong, L., Wang, L., Tian, Z., Huai, B., Luo, W., Luo, W., Zhang, Z., Hu, B., and Zhang, M. Perception, reason, think, and plan: A survey on large multimodal reasoning models, 2025c. URL https://arxiv.org/abs/ 2505.04921.

Li, Y., Liu, Z., Hu, B., Wang, W., Ding, Y., Cao, X., and Zhang, M. Vision enhancing llms: Empowering multimodal knowledge storage and sharing in llms. IEEE Transactions on Image Processing, 35:858–871, 2026. doi: 10.1109/TIP.2025.3649356.

Liu, S., Zhuge, M., Zhao, C., Chen, J., Wu, L., Liu, Z., Zhu, C., Cai, Z., Zhou, C., Liu, H., Chang, E., Suri, S., Xu, H., Qian, Q., Wen, W., Varadarajan, B., Liu, Z., Xu, H., Bordes, F., Krishnamoorthi, R., Ghanem, B., Chandra, V., and Xiong, Y. Videoauto-r1: Video auto reasoning via thinking once, answering twice. 2026a. URL https://arxiv.org/abs/2601.05175.

Liu, Y., Ouyang, K., Wu, H., Liu, Y., Sui, L., Li, X., Zhong, Y., Charles, Y., Zhou, X., and Sun, X. Videoreasonbench: Can mllms perform vision-centric complex video reasoning?, 2026b. URL https://arxiv.org/abs/ 2505.23359.

Meng, J., Li, X., Wang, H., Tan, Y., Zhang, T., Kong, L., Tong, Y., Wang, A., Teng, Z., Wang, Y., et al. Open-o3video: Grounded video reasoning with explicit spatiotemporal evidence. arXiv preprint arXiv:2510.20579, 2025.

Munasinghe, S., Gani, H., Zhu, W., Cao, J., Xing, E., Khan, F. S., and Khan, S. Videoglamm: A large multimodal model for pixel-level visual grounding in videos, 2025. URL https://arxiv.org/abs/2411.04923.

Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., and Sun, X. Spacer: Reinforcing mllms in video spatial reasoning, 2025. URL https://arxiv. org/abs/2504.01805.

Plizzari, C., Tonioni, A., Xian, Y., Kulshrestha, A., and Tombari, F. Omnia de egotempo: Benchmarking temporal understanding of multi-modal llms in egocentric videos. 2025. URL https://arxiv.org/abs/ 2503.13646.

Ren, W., Ma, W., Yang, H., Wei, C., Zhang, G., and Chen, W. Vamba: Understanding hour-long videos with hybrid mamba-transformers. arXiv preprint arXiv:2503.11579, 2025.

Saravanan, D., Gupta, V., Singh, D., Khan, Z., Gandhi, V., and Tapaswi, M. Velociti: Benchmarking video-language compositional reasoning with strict entailment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18914–18924, 2025.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shu, Y., Liu, Z., Zhang, P., Qin, M., Zhou, J., Liang, Z., Huang, T., and Zhao, B. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26160–26169, 2025.

Song, E., Chai, W., Xu, W., Xie, J., Liu, Y., and Wang, G. Video-mmlu: A massive multi-discipline lecture understanding benchmark. arXiv preprint arXiv:2504.14693, 2025.

Tang, H., Cao, M., Liu, R., Liang, X., Li, L., Li, G., and Liang, X. Video spatial reasoning with object-centric 3d rollout. 2025a. URL https://arxiv.org/abs/ 2511.13190.

Tang, Y. Y., Shimada, D., Hua, H., Huang, C., Bi, J., Feris, R., and Xu, C. Video-r4: Reinforcing text-rich video reasoning with visual rumination. 2025b. URL https: //arxiv.org/abs/2511.17490.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Wang, Q., Yu, Y., Yuan, Y., Mao, R., and Zhou, T. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025a.

Wang, S., Jin, J., Wang, X., Song, L., Fu, R., Wang, H., Ge, Z., Lu, Y., and Cheng, X. Video-thinker: Sparking” thinking with videos” via reinforcement learning. arXiv preprint arXiv:2510.23473, 2025b.

Wang, Y., Li, X., Yan, Z., He, Y., Yu, J., Zeng, X., Wang, C., Ma, C., Huang, H., Gao, J., Dou, M., Chen, K., Wang, W., Qiao, Y., Wang, Y., and Wang, L. Internvideo2.5: Empowering video mllms with long and rich context modeling, 2025c. URL https://arxiv.org/abs/ 2501.12386.

Wang, Y., Wang, Z., Xu, B., Du, Y., Lin, K., Xiao, Z., Yue, Z., Ju, J., Zhang, L., Yang, D., et al. Time-r1: Posttraining large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025d.

Wu, B., Yu, S., Chen, Z., Tenenbaum, J. B., and Gan, C. STAR: A benchmark for situated reasoning in realworld videos. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.

net/forum?id=EfgNF5-ZAjM.

Wu, H., Li, D., Chen, B., and Li, J. Longvideobench: A benchmark for long-context interleaved video-language understanding. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.

net/forum?id=3G1ZDXOI4f.

Wu, R., Ma, X., Ci, H., Fan, Y., Wang, Y., Zhao, H., Li, Q., and Wang, Y. Longvitu: Instruction tuning for long-form video understanding. arXiv preprint arXiv:2501.05037, 2025.

Xu, Y., Wu, Y., Yu, J., Yan, Z., Jiang, T., He, Y., Zhao, Q., Chen, K., Qiao, Y., Wang, L., Okumura, M., and Wang, Y. Expvid: A benchmark for experiment video understanding & reasoning, 2025. URL https://arxiv.org/ abs/2510.11606.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces, 2025b. URL https://arxiv.org/abs/2412.14171.

Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., Jin, P., Zhang, W., Wang, F., Bing, L., and Zhao, D. Videollama 3: Frontier multimodal foundation models for image and video understanding. 2025a. URL https://arxiv.

org/abs/2501.13106.

Zhang, S., Hao, X., Tang, Y., Zhang, L., Wang, P., Wang, Z., Ma, H., and Zhang, S. Video-cot: A comprehensive dataset for spatiotemporal understanding of videos based on chain-of-thought, 2025b. URL https://arxiv.

org/abs/2506.08817.

Zhang, Y., Wu, J., Li, W., Li, B., MA, Z., Liu, Z., and Li, C. LLaVA-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025c. ISSN 2835-8856. URL https:// openreview.net/forum?id=EElFGvt39K.

Zhao, F., Zhang, L., Shi, D., Gao, Y., Ye, C., Cai, Y., Gao, J., and Yan, D. Videoperceiver: Enhancing fine-grained temporal perception in video multimodal large language models. arXiv preprint arXiv:2511.18823, 2025a.

Zhao, Y., Xie, L., Zhang, H., Gan, G., Long, Y., Hu, Z., Hu, T., Chen, W., Li, C., Song, J., Xu, Z., Wang, C., Pan, W., Shangguan, Z., Tang, X., Liang, Z., Liu, Y., Zhao, C., and Cohan, A. Mmvu: Measuring expertlevel multi-discipline video understanding, 2025b. URL https://arxiv.org/abs/2501.12380.

Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., and Ma, Y. Llamafactory: Unified efficient fine-tuning of 100+ language models, 2024. URL https://arxiv. org/abs/2403.13372.

Zhou, J., Shu, Y., Zhao, B., Wu, B., Liang, Z., Xiao, S., Qin, M., Yang, X., Xiong, Y., Zhang, B., Huang, T., and Liu, Z. Mlvu: Benchmarking multi-task long video understanding. 2025. URL https://arxiv.org/ abs/2406.04264.

Zohar, O., Wang, X., Bitton, Y., Szpektor, I., and YeungLevy, S. Video-star: Self-training enables video instruction tuning with any supervision, 2024. URL https: //arxiv.org/abs/2407.06189.

Zohar, O., Wang, X., Dubois, Y., Mehta, N., Xiao, T., Hansen-Estruch, P., Yu, L., Wang, X., Juefei-Xu, F., Zhang, N., et al. Apollo: An exploration of video understanding in large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18891–18901, 2025.

#### A. VideoKR Data Construction

- A.1. Domain Knowledge Bank Construction Table 6. Complete subject list by major disciplines. Columns list subfields under each of the four major disciplines.

Natural Sciences (20) Engineering (20) Healthcare (18) Humanities & Social Sciences (24)

Chemistry Electrical Engineering Medicine History Physics Computer Science Public Health Psychology Biology Materials Science & Engineering Pharmacy Arts

Astrophysics & Astronomy Mechanical Engineering Dentistry Education Earth Science Civil & Environmental Engineering Nursing Sociology Mathematics Chemical Engineering Biomedical Sciences Philosophy

Statistics & Data Science Biomedical Engineering Medical Laboratory Science Linguistics

Environmental Science Aerospace Engineering Nutrition & Dietetics Anthropology Ecology & Evolutionary Biology Industrial & Operations Research Physiotherapy Archaeology

Microbiology & Immunology Systems Engineering Occupational Therapy Political Science Biochemistry & Molecular Biology Nuclear Engineering Speech & Language Therapy International Relations

Biophysics Energy Engineering Radiography / Imaging Sciences Economics Neuroscience Mechatronics & Robotics Health / Biomedical Informatics Law Genetics & Genomics Software Engineering Epidemiology & Biostatistics Geography

Cell & Developmental Biology AI Engineering Global Health Communication & Media Studies

Marine Science Computer Engineering Veterinary Medicine Literature & Comparative Literature Atmospheric Science Communications Engineering Optometry / Vision Science Modern Languages & Cultures

Geology & Geophysics Control & Automation Health Policy & Management Theology & Religious Studies

Paleontology Structural Engineering Business Administration Scientific Computing Geotechnical Engineering Finance

Accounting Architecture & Urban Planning Public Policy & Administration Gender & Sexuality Studies

Based on a manual review of undergraduate curricula from leading universities worldwide, we identified 82 representative subjects spanning four major disciplines. Table 6 organizes these subjects across Natural Sciences, Healthcare, Humanities and Social Sciences, and Engineering, forming the top-level index of our four-layer knowledge base of subject, course, lecture, and knowledge point, and enabling broad cross-domain coverage and balanced sampling.

##### A.2. Annotator Information

- Table 7. Biographies of 34 annotators involved in the VideoKR construction pipeline. The table details their participation in: Know. Bank (Domain Knowledge Bank Construction), Seed Ex. (Seed Example Curation), Model Val. (Human-Validated Model Selection), Quality (Manual Quality Assessment), and Eval Bench. (VideoKR-Eval Construction).

Annotation Tasks Know. Bank Seed Ex. Model Val. Quality Eval Bench.

ID Year Major Assigned Discipline

- 1 3rd yr PhD Electrical Eng. Engineering ✓ ✓ ✓ ✓ ✓

- 2 3rd yr PhD Mechanical Eng. Engineering ✓ ✓ ✓ ✓ ✓

- 3 1st yr PhD Computer Science Engineering ✓ ✓ ✓ ✓

- 4 2nd yr PhD Civil Engineering Engineering ✓ ✓ ✓

- 5 2nd yr Master Chemical Eng. Engineering ✓ ✓

- 6 1st yr Master Materials Science Engineering ✓ ✓ ✓

- 7 1st yr Master Aerospace Eng. Engineering ✓ ✓

- 8 2nd yr Master Biomedical Eng. Engineering ✓ ✓ ✓

- 9 2nd yr Master Software Eng. Engineering ✓

- 10 1st yr Master Electronic Eng. Engineering ✓ ✓

- 11 1st yr Master Industrial Eng. Engineering ✓ ✓

- 12 3rd yr PhD Physics Natural Sciences ✓ ✓ ✓ ✓ ✓

- 13 2nd yr PhD Chemistry Natural Sciences ✓ ✓ ✓ ✓ ✓

- 14 2nd yr PhD Biology Natural Sciences ✓ ✓ ✓ ✓

- 15 1st yr PhD Mathematics Natural Sciences ✓ ✓ ✓

- 16 2nd yr Master Statistics Natural Sciences ✓ ✓ ✓

- 17 1st yr Master Earth Science Natural Sciences ✓ ✓

- 18 1st yr Master Astrophysics Natural Sciences ✓ ✓

- 19 1st yr Master Environmental Sci. Natural Sciences ✓ ✓

- 20 2nd yr Master Geology Natural Sciences ✓ ✓

- 21 1st yr Master Ecology Natural Sciences ✓ ✓

- 22 2nd yr PhD Economics Humanities & Social Sciences ✓ ✓ ✓ ✓ ✓

- 23 2nd yr PhD Psychology Humanities & Social Sciences ✓ ✓ ✓ ✓ ✓

- 24 3rd yr PhD Sociology Humanities & Social Sciences ✓ ✓

- 25 1st yr Master Political Science Humanities & Social Sciences ✓ ✓

- 26 1st yr Master Philosophy Humanities & Social Sciences ✓ ✓ ✓ ✓

- 27 1st yr Master History Humanities & Social Sciences ✓ ✓ ✓

- 28 1st yr Master Law Humanities & Social Sciences ✓ ✓ ✓

- 29 2nd yr Master Linguistics Humanities & Social Sciences ✓ ✓

- 30 1st yr Master Education Humanities & Social Sciences ✓ ✓

- 31 4th yr PhD Public Health Healthcare ✓ ✓ ✓ ✓ ✓

- 32 2nd yr PhD Clinical Medicine Healthcare ✓ ✓ ✓ ✓ ✓

- 33 2nd yr PhD Dentistry Healthcare ✓

- 34 2nd yr Master Pharmacy Healthcare ✓ ✓ ✓

##### A.3. VideoKR Data Example

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Figure 62]

Q: Which tested sample shows formation of an insoluble solid during measurement?

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

1 2 3 4

[Figure 67]

Answer: magnesium hydroxide sample Reasoning: Let me analyze the video content around the relevant measurement mark. Looking at the sequence: At around 01:53, the text "PH of Saturated Ca(OH)₂" appears,(...abbreviated...) The question asks about which sample forms an insoluble solid during measurement. However, I need to consider the chemistry involved:(...abbreviated...) The video is demonstrating the difference in solubility between these two hydroxides, and the formation of insoluble solid would be most notable with the magnesium hydroxide sample due to its much lower Ksp value.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

5 6 7 8

[Figure 72]

Q: A saturated solution of Mg(OH)₂ exhibits a pH value measured in the experiment. Calculate the Ksp of Mg(OH)₂ using the observed pH.

[Figure 73]

[Figure 74]

[Figure 75]

Q: What value is shown on the display at around 01:38?

[Figure 76]

A. 1.0 × 10⁻¹² B. 5.6 × 10⁻¹² C. 1.8 × 10⁻¹¹ D. 4.0 × 10⁻¹²

Answer: −6.419 Reasoning: The user is asking about what value is displayed at approximately 1 minute and 38 seconds (98 seconds) into the video. Looking at the timestamp at around 98 seconds, I can see a Texas Instruments TI-83 Plus calculator display. The screen shows statistical calculation results, including: - "INTERCEPT= -6.419" - "SLOPE=11.824" The question asks for "the value" shown at this time, and given the correct answer provided is -6.419, this corresponds to the intercept value displayed on the calculator screen.

Answer: A Reasoning: To calculate the Ksp of Mg(OH)₂, first extract the saturated solution's pH = 10.10 from the video(...abbreviated...). Calculate pOH = 14.00 - 10.10 = 3.90, then [OH⁻] = 10⁻³.⁹⁰ ≈ 1.2589 × 10⁻⁴ M(...abbreviated...). From Mg(OH)₂ ⇌ Mg²⁺ + 2OH⁻, [Mg²⁺] = [OH⁻]/2 ≈ 6.2945 × 10⁻⁵ M(...abbreviated...). Substitute into Ksp = [Mg²⁺][OH⁻]²: Ksp = (6.2945 × 10⁻⁵)(1.2589 × 10⁻⁴)² ≈ 9.98 × 10⁻¹³, rounding to 1.0 × 10⁻¹².

- Figure 4. A VideoKR-SFT-201K example from the natural science domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

[Figure 77]

[Figure 78]

Q: What device is physically connected to the phone before measurements begin?

Answer: External speakers Reasoning: I need to examine the video carefully to identify what device is physically connected to the phone before the measurement phase begins.(...abbreviated...)Then at around 194.6 seconds onwards, I can observe a table setup showing a phone positioned between two black external speakers. The speakers appear to be wired and connected to the phone with visible cables. This setup is shown clearly before the person begins walking around the room taking measurements (which occurs around 225.7 seconds).(...abbreviated...)The external speakers are part of the controlled setup for the noise mapping exercise - they likely serve as the sound source that the participants are trying to locate and map using the sound measuring app on their phone.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

1 2 3 4

5 6 7 8

[Figure 87]

[Figure 88]

Q: At the moment the handheld display shows its highest reading, which category applies to that spot on the diagram?

Answer: Red Reasoning: Looking at the video timeline, I need to identify when the handheld SPL meter shows its highest reading and then determine which category that corresponds to on the noise map.(...abbreviated...)The highest reading on the handheld SPL meter is 79.1 dBA, recorded at around 101 and 109 seconds.(...abbreviated...)This peak reading corresponds to the red category on the noise map’s Green-Yellow-Orange-Red color legend, which is reserved for the loudest measured noise levels in the room.

[Figure 89]

[Figure 90]

[Figure 91]

Q: During the SPL meter demonstration, at the moment when the current SPL value is displayed as 77.2 dBA, calculate the ratio of (maximum value - average value) to (average value - minimum value). Round to two decimal places.

Answer: B Reasoning: Extract the three key values of the SPL meter shown in the video when the current SPL

value is 77.2 dBA(...abbreviated...), the maximum value is 79.5 dBA, the minimum value is 76.5 dBA and the average value is 77.49 dBA. First calculate the numerator: 79.5 - 77.49 = 2.01, then calculate the denominator: 77.49 - 76.5 = 0.99(...abbreviated...). Divide the two results to get the ratio: 2.01 ÷ 0.99 ≈ 2.0303, and round it to two decimal places to get 2.03.

A. 1.85 B. 2.03 C. 2.15 D. 3.03

- Figure 5. A VideoKR-SFT-201K example from the healthcare domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Q: Which specific field keeps the selected objects reachable and prevents reclamation?

1

2 3 4

[Figure 97]

Answer: leakMap field Reasoning: I analyze the video’s content about object reachability and garbage collection prevention in a Java profiling tool(...abbreviated...). The tool displays a hierarchical tree of object references for specific instances, and a dedicated internal field is used to maintain references to selected objects during analysis(...abbreviated...). The leakMap field is the specific one that keeps these objects reachable, stopping the garbage collector from reclaiming them during profiling.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

5

6 7 8

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 102]

Q: In the heap analysis of new objects since the last mark heap action, what percentage of these objects are instances of the class java.awt.geom.GeneralPath? Round your answer to two decimal places.

[Figure 103]

Q: Which view tab is selected immediately after creating the object set from the dialog?

[Figure 104]

[Figure 105]

[Figure 106]

A. 16.89% B. 17.29% C. 17.50% D. 18.00%

A. References B. Classes C. Allocations D. Time

Answer: A Reasoning: I check the video’s sequence of creating an object set via the dialog and the subsequent active view tab(...abbreviated...). After the object set creation dialog is closed and the new object set is generated, the interface displays content with "Incoming references" and a corresponding tree structure(...abbreviated...). This visual confirms the References tab is the active view immediately after the action, so the answer is A.

Answer: B Reasoning: Extract two key values from the video’s heap analysis interface(...abbreviated...): the total number of new objects since the last mark heap action is 3384, and the instance count of java.awt.geom.GeneralPath is 585. Calculate the percentage with the formula (target class instances / total new objects) × 100(...abbreviated...). Substitute the values: (585 ÷ 3384) × 100 ≈ 17.2872%, round to two decimal places to get 17.29%, which corresponds to option B.

- Figure 6. A VideoKR-SFT-201K example from the engineering domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

[Figure 107]

[Figure 108]

Q: In the illustration with a warehouse, how many helmeted figures stand in a row?

Answer: D Reasoning: I examine the warehouse illustration with helmeted stick figures in the video(...abbreviated...). This illustration appears under the "Step 1: Define Problem" section, with a business/observer figure on the left and a row of helmeted figures on the right(...abbreviated...). Counting the helmeted figures in this row from the visual shows exactly five, so the answer is D.

[Figure 109]

[Figure 110]

Q: Why are two separate models shown when predicting different future horizons?

Answer: different target labels Reasoning: I analyze the content in the video that addresses using separate models for predicting different future horizons(...abbreviated...). The video explains that predicting values for two distinct future time points (e.g., 3 days and 10 days out) requires individual models, each trained to forecast a specific time horizon(...abbreviated...). Each model corresponds to a unique prediction objective, meaning they are trained on different target labels—one for the nearer future horizon and one for the farther one, which is why two separate models are needed.

- A. Four B. Three C. Six D. Five

[Figure 111]

[Figure 112]

[Figure 113]

Q: Based on the table shown in the video under “Step 3: What is the data saying?”, which lists the number of laptop inbound volumes for five consecutive dates, what is the average number of laptops received per day across the entire period?

Answer: 164 Reasoning: To determine the average number of laptops received per day, I first extract the five daily inbound laptop volumes presented in the table in the video: 90 (2020-11-15), 213 (2020-1116), 178 (2020-11-17), 187 (2020-11-18), and 152 (2020-11-19). I then calculate the total inbound volume by summing these values: 90 + 213 + 178 + 187 + 152 = 820. Finally, I divide the total by the number of days (5) to find the daily average: 820 ÷ 5 = 164. This calculation confirms the average number of laptops received per day across the period is 164.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

1 2 3 4

5 6 7 8

- Figure 7. A VideoKR-SFT-201K example from the engineering domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

[Figure 122]

[Figure 123]

Q: At around 03:32, which item is shown immediately after the text sequence ends?

Answer: C Reasoning: Let me work through the timeline to understand what happens at the timestamp mentioned in the question.(...abbreviated...)Examining the sequence around that time shows a window image at 208.0 seconds, with faces and prison bar visuals following shortly after.(...abbreviated...)Given the timeline around 3:32 (212 seconds) and the sequence of visuals, the window image at 208.0 seconds would be what appears after a text sequence in that general timeframe.

[Figure 124]

[Figure 125]

Q: "What form of collective action is illustrated by the confrontations with authorities?

Answer: militant protest Reasoning: This video is about the women's suffrage movement in 1909, specifically focusing on "Deeds" as stated at around 00:18.(...abbreviated...)The title specifically uses "Deeds" and the historical context of 1909 points to the tactics of the Women's Social and Political Union (WSPU), which advocated for direct action rather than peaceful petition.(...abbreviated...)all indicate this is illustrating militant protest tactics - involving civil disobedience, direct confrontation with authorities, and willingness to face arrest and imprisonment for the cause.

- A. A sign image B. A portrait image C. A window image D. A crowd image

[Figure 126]

[Figure 127]

[Figure 128]

Q: Based on the protesters’ tactics shown, what type of political strategy is illustrated?

Answer: militant direct action Reasoning: To answer this question, I need to examine the protest tactics and imagery shown throughout the video. Key observations:(...abbreviated...)The phrase "DEEDS NOT WORDS" is particularly telling - it explicitly contrasts action-oriented protest with mere verbal advocacy. The combination of street protests, apparent physical confrontations with authorities, and subsequent imprisonment indicates these suffragettes went beyond traditional peaceful lobbying. This matches the historical record of the Women's Social and Political Union (WSPU), which adopted increasingly confrontational tactics including civil disobedience, property damage, and direct confrontation with authorities - tactics that deliberately provoked arrest to draw attention to their cause.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

1 2 3 4

5 6 7 8

- Figure 8. A VideoKR-SFT-201K example from the humanities and social science domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

##### A.4. Human-Validated Model Selection Protocol

To ensure that the VideoKR corpus is constructed with the highest possible quality while maximizing data diversity, we implemented a rigorous, human-in-the-loop qualification protocol for all foundation models used in our pipeline. Instead of relying on a single model (which risks imprinting specific model biases), we maintain a dynamic pool of eligible models for each synthesis stage. This section details the qualification methodology and the resulting model assignments.

We evaluate seven frontier models for potential inclusion in each pipeline step: GPT-5.2, GPT-5-mini, Claude-4.5-Sonnet, Gemini-3-Flash, DeepSeek-V3.2, Qwen3-VL-235B-A22B, and GLM-4.6V. For every pipeline stage defined in §3.1–§3.4, we conducted a controlled pilot study by sampling 100 representative input instances. Domain experts evaluated the model outputs against strict criteria, distinguishing between hard compliance failures (e.g., JSON format violations) and soft content failures (e.g., hallucinations or weak reasoning). A model was deemed eligible for a specific stage only if its total error rate ≤ 3%. Table 8 shows the human-validated models for each pipeline step.

- Table 8. The eligible models at each VideoKR construction pipeline step. “-” indicates the model is not optimized for the modality required at that step; “✗” indicates the model fails human validation for that step.

Pipeline Stage Modality GPT-5.2 GPT-5-mini Claude-4.5-Sonnet Gemini-3-Flash DeepSeek-V3.2 Qwen3-VL-235B-A22B GLM-4.6V

- §3.1 Domain Knowledge Bank Construction Lecture → Knowledge Point Text ✓ ✓ ✓ ✓ ✓ – –

- §3.2 Knowledge-Driven Video Collection Scenario Generation Text ✓ ✓ ✓ ✓ ✗ – – Search Keyword Generation Text ✓ ✓ ✓ ✓ ✓ – – Metadata Relevance Judge Text ✓ ✗ ✓ ✗ ✗ – – Visual Relevance Judge Vision ✓ ✗ ✗ ✓ – ✗ ✗

- §3.3 Skill-Oriented Example Generation Example Generation Vision ✓ ✗ ✓ ✓ – ✗ ✗ CoT Rationale Validation Vision ✓ ✗ ✓ ✓ – ✗ ✗

A.5. Data Contamination Mitigation

For Near-Duplicate Video Filtering, we uniformly sample both benchmark and VideoKR videos at 1 fps using ffmpeg and compute 64-bit perceptual hashes per frame. We partition each training video into overlapping 20-second windows with a 1-second stride and build an index over window-level hash sequences to enable scalable retrieval. For each benchmark video, we retrieve the top-10 candidate training windows and verify them by aligned-frame Hamming distance; we flag an overlap when the best 20-second window has at least 70% of frames with distance ≤ 30, and remove any matched training video.

- B. VideoKR-Eval Benchmark

- B.1. Detailed Statistics of VideoKR-Eval

As shown in Table 9, we construct VideoKR-Eval from three source benchmarks: VideoMMMU, MMVU, and SciVideoBench. We first perform multi-model single-frame probing and retain only original examples that are judged to require continuous video understanding by all three probing models. This yields 1,254 retained original examples. For the remaining filtered videos, domain experts re-annotate new visually grounded QA examples, contributing 746 additional examples. The final benchmark comprises 2,000 high-quality examples designed to require genuine video-level understanding and knowledge-intensive reasoning.

- Table 9. Detailed statistics for the VideoKR-Eval benchmark construction. We retain original examples that are judged to require continuous video understanding by all three single-frame probing models, and add expert-reannotated examples for the filtered videos.

VideoKR-Eval Composition

Source Benchmark Candidate Count

Final Count Filtered Retained Original Expert-Reannotated

MMVU 1,000 639 361 398 759 VideoMMMU 900 560 340 241 581 SciVideoBench 1,000 447 553 107 660

###### Total 2,900 1,646 1,254 746 2,000

#### C. Experiment Setup

##### C.1. Post-Training Details

GRPO Reward Design. We employ GRPO (Shao et al., 2024) as our reinforcement learning algorithm. Following the standard RLVR-style reward formulation, the total reward is defined as R = 0.1 · Rf + 0.9 · Ra, where Rf and Ra denote the format and accuracy rewards, respectively. Specifically, Rf is set to 1.0 if the model output strictly satisfies the required format: <think>...</think><answer>...</answer>. For the accuracy reward Ra, we adopt the ROUGE metric for open-ended QA, while employing Exact Match (EM) for multiple-choice tasks.

Training Details. We train all models on up to 8 NVIDIA A800 GPUs (80 GB). For SFT, we use a learning rate of 1 × 10−5, while for RL we use a learning rate of 5 × 10−6. Both stages are optimized with AdamW, and the maximum response length is set to 2,048 tokens. For GRPO rollout generation, we set the rollout size G to 8 and use a temperature of 1.0 to encourage exploration. The KL penalty coefficient β is set to 0.01. Supervised fine-tuning is implemented with LLaMA-Factory (Zheng et al., 2024), while reinforcement learning is implemented with verl (Sheng et al., 2024).

##### C.2. Evaluation Setup

To ensure fair and reproducible comparisons, we standardize the inference configuration for all evaluations by setting the temperature to 0.1. The maximum response token is set to 8,192 tokens.

All Benchmarks – For VideoKR and OneThinker {Question} Please answer this question based on the visual content. Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. At the end, you must output the final answer in the format: <answer><your answer here></answer>

All Benchmarks – For VideoAuto-R1 Video-MME Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option {question} Put your final answer in \\boxed{}.

Other benchmarks {question} Put your final answer in \\boxed{}.

All Benchmarks – For Video-R1 and VideoRFT {Question} Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as ’let me think’, ’wait’, ’Hmm’, ’oh, I see’, ’let’s break it down’, etc, or other natural language thought expressions. It’s encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags. For multiple-choice: Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags. For open-ended: Please provide your text answer within the <answer> </answer> tags.

Video-MME | MVBench – For Qwen series instruct models

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: {question} Possible answer choices: {options} The best answer is:

Video-MME | MVBench – For Qwen series thinking models

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: {question} Possible answer choices: {options} Please reason step-by-step, identify relevant visual content, analyze key timestamps and clues, and then provide the final answer.

Video-MME – For other models

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: {question} Possible answer choices: {options} Answer with the option’s letter from the given choices directly.

MVBench – For other models

{question} {options} Only give the best option.

LongVideoBench – For other models

{question} {options} Answer with the option’s letter from the given choices directly.

VideoMMMU – For other models Perception & Comprehension: {question} {options} Please ignore the Quiz question in last frame of the video.

Adaptation-multiple-choice(open-ended): You should watch and learn the video content. Then apply what you learned to answer the following multi-choice(open-ended) question. The image for this question is at the end of the video. {question} {options}

MMVU – For Qwen series models and InternVL3.5-8B multiple-choice: {question} {options} Visual Information: processed video Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: $LETTER” (without quotes), where $LETTER is one of the options. Think step by step before answering. open-ended: {question} Visual Information: processed video Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: $ANSWER” (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering.

MMVU – For other models multiple-choice: {question} {options} Visual Information: processed video Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.

open-ended: {question} Visual Information: processed video Do not generate any intermediate reasoning process. Directly output the final answer.

SciVideoBench – For Qwen series thinking models, GPT-5.4, Gemini 3 Pro, Claude Opus 4.5 and InternVL3.5-8B

{question} {options} Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: $ANSWER” (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering.

SciVideoBench – For Qwen series instruct models and other models

{question} {options} Answer with the option’s letter from the given choices directly.

VideoKR-Eval– For Qwen series thinking models, GPT-5.4, Gemini 3 Pro, Claude Opus 4.5 and InternVL3.5-8B multiple-choice: {question} {options}

Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: $LETTER” (without quotes), where $LETTER is one of the options. Think step by step before answering.

open-ended: {question} Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: $ANSWER” (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering.

VideoKR-Eval– For Qwen series instruct models and other models multiple-choice: {question} {options} Answer with the option’s letter from the given choices directly. open-ended: {question} Please answer the question using a single word or phrase.

#### D. Experiment Results and Analysis

##### D.1. Performance with Different Frames

- Table 10. Detailed accuracy on general and knowledge-intensive video reasoning benchmarks for post-trained models across different input frames.

General Video Reasoning Knowledge-Intensive Video Reasoning

Video-MMEMVBench LongVideoBench

VideoMMMUMMVU SciVideoBenchVideoKR-Eval

Model Release Frames

Average

Average

Post-trained Qwen2.5-VL-7B-Instruct

VideoKR (SFT+RL) 2026-05 16 56.6 66.6 57.0 60.1 52.6 59.2 27.3 37.7 44.2 VideoKR (SFT+RL) 2026-05 32 60.1 68.2 58.2 62.2 51.2 58.9 27.4 39.8 44.3 VideoKR (SFT+RL) 2026-05 64 64.0 68.6 58.9 63.8 52.6 60.6 29.8 40.0 45.8

- VideoKR (SFT+RL) 2026-05 128 66.4 68.9 61.3 65.5 52.2 60.5 32.5 41.2 46.6

Post-trained Qwen3-VL-8B-Instruct

VideoKR (SFT+RL) 2026-05 16 57.2 65.5 54.7 59.1 60.2 63.8 29.3 40.5 48.5 VideoKR (SFT+RL) 2026-05 32 61.7 65.7 56.6 61.3 60.7 64.7 30.7 42.3 49.6 VideoKR (SFT+RL) 2026-05 64 63.9 66.2 57.9 62.7 62.1 63.8 31.5 44.6 50.5

- VideoKR (SFT+RL) 2026-05 128 67.8 67.0 61.5 65.4 63.0 64.8 32.8 45.3 51.5

##### D.2. Ablation Studies

- Table 11. Ablation studies on post-training data. All experiments use Qwen2.5-VL-7B-Instruct as the base model, with 128 input frames. General Video Reasoning Knowledge-Intensive Video Reasoning

Video-MMEMVBench LongVideoBench

VideoMMMUMMVU SciVideoBenchVideoKR-Eval

Average

Average

Ablation Setting

Qwen2.5-VL-7B-Instruct 65.1 66.3 60.9 64.1 51.1 55.7 28.1 32.7 41.9 Skill-Oriented Data Composition

VR (Basic Reasoning) 61.0 60.8 52.3 58.0−6.1 51.3 51.7 27.3 35.3 41.4−0.5 VR + KV (Perception) 61.5 60.0 53.7 58.4−5.7 51.0 52.3 26.1 35.9 41.3−0.6 VR + KV + KVR (Full) 60.6 60.9 53.4 58.3−5.8 50.6 51.6 30.7 36.8 42.4+0.5

###### Supervision Format

Direct Output 65.6 62.8 55.8 61.4−2.7 45.8 51.8 24.1 35.9 39.4−2.5 Chain-of-Thought (CoT) 60.6 60.9 53.4 58.3−5.8 50.6 51.6 30.7 36.8 42.4+0.5

###### Comparison with Other SFT Corpora (SFT-only)

Video-R1-CoT-165k 56.9 63.2 51.9 57.3−6.8 45.3 50.4 21.4 27.5 36.2−5.7 OneThinker-SFT-340k 59.1 66.7 55.8 60.5−3.6 45.7 52.2 25.5 29.7 38.3−3.6 VideoRFT-CoT-102K 61.8 62.9 54.2 59.6−4.5 47.3 50.3 24.0 32.1 38.4−3.5 VideoKR-SFT-201K (Ours) 60.6 60.9 53.4 58.3−5.8 50.6 51.6 30.7 36.8 42.4+0.5

###### Comparison with Other RL Corpora (RL-only)

Video-R1-260k 64.8 67.5 58.7 63.7−0.4 51.3 54.9 26.9 33.1 41.6−0.3 OneThinker-600k 63.7 64.5 52.7 60.3−3.8 53.7 54.5 27.6 33.2 42.3+0.4 VideoRFT-RL-310K 65.2 67.9 58.3 63.8−0.3 50.9 55.7 29.0 33.5 42.3+0.4 VideoAuto-R1-83K 65.0 66.9 56.6 62.8−1.3 52.4 55.9 29.0 33.3 42.7+0.8 VideoKR-RL-114K (Ours) 64.2 65.7 55.3 61.7−2.4 51.8 56.2 29.6 34.5 43.0+1.1

##### D.3. Case Study

| | | | | |
|---|---|---|---|---|

OneThinker: <think> Video context: a supply-chain flowchart (sourcing → manufacturing → distribution → transactional management) indicates production and logistics stages. Option check: A) Diversification—portfolio risk spreading; not depicted. B) Vertical integration —end-to-end control; no evidence. C) Horizontal integration—mergers; absent. D) Outsourcingcontracting external suppliers; truck/boxed goods suggest third-party handling, letting the firm focus on core tasks. E) Speculation—risky bets; irrelevant here. Thus, the risk-handling method most consistent with the video is outsourcing. Final: D. </think><answer>D</answer>

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

1 2 3 4

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

5 6 7 8

| | | | | |
|---|---|---|---|---|

Question: For the company shown in the video, what is one of the methods to handle associated risks? Options: (A) diversification (B) vertical integration (C) horizontal integration (D) outsourcing (E) speculation Explanation: A coffee brand and its upstream chain are shown; to hedge green bean price risk, the firm can vertically integrate—own/establish plantations to secure supply and stabilize costs.

Error Reason: Misread the scene as generic logistics and inferred outsourcing, ignoring that the bubbles indicate coffee beans and the focus is upstream cost volatility. For a coffee company, the correct risk mitigation depicted is vertical integration (own plantations), not outsourcing.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video depicts a truck departing from a coffee plantation adorned with coffee bean icons and the Folgers logo, heading towards a roasting/processing plant, and then to a retail store. The process traces the supply chain from SOURCING → MANUFACTURING

Qwen3-VL-8B-Thinking: <think> The video showcases a supply chain process: procurement, manufacturing, distribution, and trading. Among the various options, there is no indication of diversification or speculation; nor is there any vertical or horizontal integration. Phased handover and third-party transportation imply outsourcing some processes to mitigate non-core risks. Therefore, option D (outsourcing) is the most suitable. </think><answer>D </answer>

→ DISTRIBUTION. This highlights exposure to upstream green-coffee price and supply volatility. A fitting mitigation is vertical integration —the brand establishes/owns plantations to secure supply, stabilize input costs, and tighten quality control. Therefore, the answer is B, vertical integration. </think><answer> B</answer>

Error Reason: Misread generic supply-chain stages as proof of outsourcing,

Note: From coffee-bean icons and the SOURCING → MANUFACTURING

ignoring cues (coffee beans, upstream cost volatility) that point to vertical integration as the risk-mitigation strategy. The correct inference is owning upstream production (e.g., plantations), not contracting it out.

→ DISTRIBUTION flow, it identifies upstream risk and correctly maps the mitigation to vertical integration (owning plantations), not outsourcing.

Figure 9. Comparison of model responses on a knowledge-intensive video reasoning sample.

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

1 2 3 4 5 6 7 8

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

OneThinker: <think> The video demonstrates an overnight magnetic stirring step. In laboratory practice, prolonged gentle stirring is used to maintain the homogeneity of the dispersion, prevent precipitation/aggregation, and improve mass transfer, thereby ensuring a uniform distribution of the suspension (in this case, quantum dots) in the next step. It is not used to change solvent concentration (B), cool (E), control temperature (H), concentrate/evaporate (I, J), or increase droplet size (F). Enhanced sonication (A) is also not applicable—stirring is a low-shear process, typically performed **after** sonication, and its role is to maintain droplet integrity, not to further break down the droplets. Preventing precipitation (D) and any accidental interactions (G) may occur, but they are secondary. Therefore, the purpose is to mix to ensure a uniform distribution of the quantum dots in the solution—option C. </think><answer>C </answer>

Question: What is the main purpose of the overnight magnetic stirring step starting

at 02:31? Options:

- (A) Enhancing ultrasonication effects for smaller droplet size
- (B) Increasing the solvent concentration for better dispersion
- (C) Mixing to evenly distribute quantum dots in solution
- (D) Preventing sedimentation of formed nanobeads
- (E) Cooling the solution to induce polymer crystallization
- (F) Breaking the emulsion into larger droplets
- (G) Promoting chemical reaction between polymer and quantum dots
- (H) Reducing temperature fluctuations during particle formation
- (I) Removing excess water to concentrate the emulsion
- (J) Complete evaporation of solvent leading to solidification

Error Reason: Misread overnight stirring as simple homogenization (C), ignoring cues of a volatile-solvent emulsion and later dry beads. In this workflow, stirring primarily drives complete solvent evaporation and solidification (J), not just preventing settling.

Explanation: This step is central to the emulsion–solvent evaporation route: ultrasonication (02:23) forms a chloroform emulsion, and overnight stirring renews the solvent–air interface to drive evaporation. As chloroform leaves, non-volatile PSMA and encapsulated QDs precipitate and harden into stable nanobeads; without it, solidification fails—thus the answer is J.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video at 02:31 demonstrates the overnight magnetic stirring step in the nanobead preparation process. Overnight magnetic stirring is a routine homogenization step in the quantum dot-nanobead preparation process, and is therefore likely for maintaining uniform dispersion (C) and inhibiting precipitation (D). Comparing other options, many (B, E, F, H, I) do not conform to the typical function of stirring; D seems best suited for the overnight step. Waitrewatching the segment around 02:23, we can see ultrasonic treatment forming an emulsion in a volatile solvent (chloroform), followed by footage of dried, solid nanobeads.

Qwen3-VL-8B-Thinking: <think> At 02:31 in the video, the procedure introduces an overnight magnetic stirring step during the preparation of QD-nanobead LFIA. In fact, prolonged gentle stirring helps homogenization, and more importantly, it keeps the particles suspended, preventing precipitation. The following options were evaluated: A (enhanced sonication), B (increased solvent concentration), E (cooling/crystallization), F (increased droplet size), G (promoted reaction), H (temperature control), and I/J (concentration/evaporation). None of these align with the typical purpose of stirring. C (uniform distribution of QD), while feasible, is less important thanmaintaining suspension —the core objective. Therefore, the primary purpose of this step is D: preventing the precipitation of the formed nanobeads. </think><answer>D</answer>

During the emulsion-solvent evaporation process, the prolonged gentle stirring primarily serves to renew the solvent-air interface between droplets, thereby promoting complete solvent evaporation, allowing the polymer + quantum dots to precipitate and harden.

Therefore, preventing precipitation is secondary; the primary goal is complete evaporation, ultimately leading to solidification—J. </think><answer>J </answer>

Note: Initial reasoning favored homogenization/anti-settling (C/D), but the “aha moment” came when linking the ultrasonication-created chloroform emulsion (02:23) to the later dry beads—revealing that overnight stirring chiefly drives complete solvent evaporation and solidification (J).

Error Reason: Misread the step as anti-sedimentation (D). The sequence shows a volatile-solvent emulsion and later dry beads, indicating overnight stirring primarily drives complete solvent evaporation and solidification (J).

Figure 10. Comparison of model responses on a knowledge-intensive video reasoning sample.

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

1 2 3 4 5 6 7 8

OneThinker: <think> Based on the pelvis plain film and the listed options, the abnormality most plausibly involves the hip joint. Typical signs of a bone cyst, tumor, osteoporosis, Paget’s disease, osteoarthritis, pubic symphysis resection, hip dislocation, or bone infection are not clearly demonstrated, and an acute hip fracture is not convincingly visible. With the absence of definitive features for these alternatives, the most specific and common explanation in this context is a hip-related abnormality consistent with surgical hardware. Therefore, the best answer is D. Surgical hardware. </think><answer>D </answer>

Question: Can you identify the abnormality on this plain film of the pelvis? <image 1> Options:

- (A) Bone cyst
- (B) Acute hip fracture
- (C) Osteoarthritis
- (D) Surgical hardware
- (E) Resection of the pubic symphysis
- (F) Bone tumor
- (G) Hip dislocation
- (H) Osteoporosis
- (I) Paget's disease
- (J) Bone infection

Error Reason: Although no metallic density was observed on the imaging, the model incorrectly inferred the presence of a "surgical implant" and overlooked the smooth pubic symphysis and cortical absence—characteristic features of previous resections. Furthermore, the diagnosis incorrectly described the abnormality as related to the hip joint, rather than located at the pubic symphysis.

Explanation: The key finding is a smooth cortical bone defect at the pubic symphysis, suggesting a previous pubic symphysis resection rather than acute separation or trauma. The sacroiliac and hip joints are well-matched, with no fracture lines or metal implants, and no signs of osteoarthritis, Paget's disease, osteoporosis, infection, or tumor.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video shows significant destruction at the pubic symphysis, with a marked gap or separation, which is uncommon in a normal pelvis. Therefore, there is a significant abnormality in the pubic symphysis region. The interpubic cortical space is smooth, and the surrounding bone structure appears intact, but this is consistent with previous surgical resection rather than acute trauma. The hip joint is well-aligned, without displacement or dislocation; no cortical fracture or trabecular bone destruction is observed, nor is there an acute hip fracture; no focal osteolytic/sclerotic lesions, bone cysts, tumors, Paget's disease, or infection are seen. Diffuse demineralization suggests osteoporosis, but is insufficient for a definitive diagnosis; typical features of osteoarthritis such as joint space narrowing or osteophyte formation are also absent. No metallic feel or surgical implants are observed. Therefore, the pathological findings are most consistent with a diagnosis following pubic symphysis resection. </think><answer>E </answer>

Qwen3-VL-8B-Thinking: <think> The video shows a diffuse radiolucent area in the pelvis, accompanied by widespread cortical thinning and loss of normal trabecular detail. The hip joint is symmetrical, without displacement, and no focal osteolytic or sclerotic lesions, fracture lines, periosteal reaction, or metallic density shadows are observed. The joint space is intact, with no osteophyte formation, suggesting the absence of osteoarthritis, and lacks the typical features of Paget's disease (bone expansion, cortical thickening) or infection (invasive destruction, soft tissue swelling). The lesion is diffuse rather than focal, therefore osteoporosis is the most plausible explanation.</think><answer>H</answer>

Error Reason: The model incorrectly overinterpreted diffuse radiolucent areas as osteoporosis, neglecting a crucial finding—smooth cortical spaces at the pubic symphysis, consistent with surgical resection. The apparent "demineralization" may be related to bone protrusion/exposure, and there is no corroborating evidence of osteopenia prevalent in other sites.

Note: VideoKR focuses on the pubic symphysis and utilizes radiological prior information (depicting the pelvic ring and checking joint consistency) to avoid the biases of other models that are hip-centric and diffuse osteoporosis.

Figure 11. Comparison of model responses on a knowledge-intensive video reasoning sample.

