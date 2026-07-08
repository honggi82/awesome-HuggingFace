# arXiv:2603.14145v2[cs.CL]20Jun2026

[Figure 1]

MMOU: A Massive Multi-Task Omni

Understanding and Reasoning Benchmark for Long and Complex Real-World Videos

Arushi Goel1†* Sreyan Ghosh1,2†* Vatsal Agarwal2* Nishit Anand2* Kaousheik Jayakumar2‡ Lasha Koroshinadze2‡ Yao Xu1 Katie Lyons1 James Case1 Karan Sapra1 Kevin J. Shih1 Siddharth Gururani1 Abhinav Shrivastava2 Ramani Duraiswami2 Dinesh Manocha2 Andrew Tao1 Bryan Catanzaro1 Mohammad Shoeybi1 Wei Ping1

1 NVIDIA, USA 2 University of Maryland, College Park, USA

## Abstract

Multimodal Large Language Models (MLLMs) have shown strong performance in visual and audio understanding when evaluated in isolation. However, their ability to jointly reason over omni-modal (visual, audio, and textual) signals in long and complex videos remains largely unexplored. We introduce MMOU, a new benchmark designed to systematically evaluate multimodal understanding and reasoning under these challenging, real-world conditions. MMOU consists of 20,000 carefully curated questions paired with 11877 web-collected videos of varying length, spanning diverse domains and exhibiting rich, tightly coupled audio-visual content. The benchmark covers 13 fundamental skill categories, all of which require integrating evidence across modalities and time. All questions are manually annotated across multiple turns by professional annotators, ensuring high quality and reasoning fidelity. We evaluate 20+ state-of-the-art open-source and proprietary multimodal models on MMOU. The results expose substantial performance gaps: the best closed-source model achieves only 64.2% accuracy, while the strongest open-source model reaches just 46.8%. Our results highlight the challenges of long-form omni-modal understanding, revealing that current models frequently fail to apply even fundamental skills in long videos. Through detailed analysis, we further identify systematic failure modes and provide insights into where and why current models break. Project: https://mmou.pages.dev/

## 1 Introduction

The pursuit of Artificial General Intelligence (AGI) has driven rapid progress in Large Language Models (LLMs), particularly through the emergence of Multimodal Large Language Models (MLLMs) that process information across multiple modalities such as text, images, audio, and video [Ye et al., 2025, Xu et al., 2025b, Hurst et al., 2024, Comanici et al., 2025, Caffagni et al., 2024]. These models have enabled compelling applications, allowing LLMs to see through vision [Dai et al., 2024, Liu et al., 2025, 2023b] and listen through audio [Goel et al., 2025, Ghosh et al., 2025a, Chu et al., 2024a, Tang et al., 2024, Tian et al., 2025]. Recent MLLMs demonstrate strong capabilities

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### OmniVinci: 24.7 % Qwen3-Omni: 19.4 % Gemini-2.5 Pro: 64.2 %

Figure 1: Overview of MMOU, a benchmark for evaluating omni-modal understanding in long, complex realworld videos, showing that both open and closed multimodal models struggle even with basic understanding.

across audio tasks (e.g., automatic speech recognition, sound classification, and audio captioning) and visual tasks (e.g., OCR, visual question answering, and video grounding), often surpassing prior benchmarks by a large margin.

Despite this progress, existing MLLMs exhibit notable limitations. Most models are optimized for single-modality reasoning [Bai et al., 2025a, Goel et al., 2025], such as vision-only or audio-only understanding, and often fail to jointly perceive and reason across modalities analogous to human cognition. This limitation is partly due to the imbalance in available training data and benchmarks: single-modality datasets are more abundant, higher quality, and cover a wider range of tasks [Liu et al., 2023a, Hurst et al., 2024, Google, 2023] than their multi-modal counterparts. As a result, current models rarely learn to integrate audio and visual cues in a unified and consistent manner.

Benchmarking has long played a central role in advancing AI by providing structured, diagnostic evaluation frameworks [Hendrycks et al., 2021, Sakshi et al., 2024b, Kumar et al., 2025, Fu et al., 2025, Hu et al., 2025]. While evaluation of LLMs has matured substantially, covering domains such

- as mathematics, code generation, reasoning, and instruction following, holistic evaluation of MLLMs remains underdeveloped. Although numerous image and video benchmarks have emerged in recent years, benchmarks that rigorously evaluate audio-visual reasoning are scarce. In particular, most video benchmarks either ignore audio entirely or treat it as auxiliary, and predominantly focus on short clips that fail to capture long-term temporal dependencies [Li et al., 2024b]. Consequently, existing evaluations do not adequately reflect the challenges posed by long and complex real-world videos, where meaningful understanding requires tightly coupled reasoning over audio and visual streams across extended time horizons.

Main Contributions. We present MMOU, a Massive, Multi-task Omni-modal Understanding and Reasoning. Our benchmark is designed to evaluate joint audio-visual understanding and reasoning on long and complex real-world videos under realistic conditions (see Fig. 1). Specifically, (i) each question requires simultaneous integration of audio and visual information, such that removing either modality leads to failure; (ii) the questions require models to demonstrate proficiency in 13 distinct and fundamental skills (as shown in Fig. 2); (iii) the benchmark is large-scale, comprising 20,000 multiple-choice QA pairs sourced from 11877 long-form real-world videos spanning 10 domains and 35 fine-grained subcategories, with each video exhibiting strong temporal and semantic alignment between audio and visual streams; and (iv) all questions are annotated by a group of 11 professionally trained human experts and each is optionally paired with 10 carefully constructed answer options that include hard distractors. To summarize, our main contributions are:

- • We introduce MMOU, a comprehensive benchmark for evaluating advanced omni-modal (audio-visual) perception and reasoning in MLLMs on long and complex real-world videos. MMOU spans 13 skill categories and includes 20,000 expertly annotated multiple-choice questions, covering both breadth and depth in multimodal understanding.
- • We evaluate 20+ open-source and proprietary MLLMs on MMOU and show that even the most advanced models struggle with tasks that humans find intuitive. The best closed-source model achieves only 64.2% accuracy, with open-source models performing substantially worse (46.8%), revealing significant gaps in current multimodal reasoning capabilities.
- • We conduct an in-depth analysis of model predictions, uncovering systematic failure modes.

## 2 Related Work

Multimodal Large Language Models. Recent years have seen rapid progress in multimodal large language models (MLLMs), which extend the capabilities of text-only LLMs [Hurst et al., 2024, Meta, 2024, Yang et al., 2025] to visual, audio, and audio–visual inputs [Xu et al., 2025b, Goel et al., 2025, Dai et al., 2024, Bai et al., 2025b, Cheng et al., 2024, Xu et al., 2025a]. These models typically integrate modality-specific encoders [Xu et al., 2024, Radford et al., 2021, Ghosh et al., 2025b, Radford et al., 2023] with a shared language model backbone [Chu et al., 2024b, Meta, 2024, Hurst et al., 2024], and are trained using large-scale multimodal instruction-tuning data [Li et al., 2024a, Zhang et al., 2024, Goel et al., 2025, Xu et al., 2025b]. As a result, state-of-the-art models demonstrate strong performance on a wide range of established benchmarks, including image–text, video–text, and audio–text understanding tasks [Fu et al., 2024, Sakshi et al., 2024a, Yue et al., 2024].

Despite these advances, existing evaluation protocols remain largely unimodal, with most benchmarks isolating a single modality or task. Such narrowly defined settings fail to capture the complexity

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 2: Illustrative examples from MMOU, demonstrating the different skill types evaluated in the benchmark.

of real-world multimodal reasoning. Consequently, strong results on individual benchmarks do not necessarily translate to robust omni-modal understanding, which requires joint reasoning across modalities, tasks, and temporal context [Li et al., 2024c].

Multimodal Benchmarks. A wide range of benchmarks have been proposed to evaluate multimodal models, including visual question answering [Antol et al., 2015], video understanding [Fu et al., 2024, Hu et al., 2025], general image understanding [Yue et al., 2024, Masry et al., 2022, Sidorov et al., 2020], and audio reasoning [Ma et al., 2025, Sakshi et al., 2024a, Kumar et al., 2025]. While these benchmarks have driven substantial progress, they predominantly evaluate isolated modalities or single-task settings, resulting in an incomplete evaluation of multimodal capabilities. Several audiovisual datasets such as VALOR [Chen et al., 2023], AVQA [Yang et al., 2022], MusicAVQA [Li et al., 2022], AV-Odyssey [Gong et al., 2024], AVHBench [Sung-Bin et al., 2024], AVCaps [Sudarsanam et al., 2025] have been proposed for joint evaluation of multimodal models. More recent benchmarks such as WorldSense Hong et al. [2025], DailyOmni Zhou et al. [2025], OmniBench Li et al. [2024d], OmniVideoBench et al. [2025], and UNO-Bench Chen et al. [2025] move towards more complex joint audio–visual evaluation, but remain constrained in critical ways. They often limit questions to a single dominant modality [Hong et al., 2025, Yang et al., 2022, Li et al., 2022, 2024d], focus on short-duration videos [Zhou et al., 2025, Benchekroun et al., 2023], or operate at a small scale with limited task diversity and category coverage [Chen et al., 2025, Li et al., 2025a], preventing rigorous evaluation of long-context reasoning and joint cross-modal inference.

## 3 MMOU

In this section, we first provide detailed statistics of MMOU in Section 3.1 and compare it with previous benchmarks in Section 3.2. This is followed by a description of the data collection and annotation processes in Section 3.3.

- 3.1 Dataset Statistics Table 1: Detailed statistics of MMOU.

- Table 1 summarizes the key statistics of MMOU. The benchmark consists of 20,000 multiplechoice QA pairs, divided across test (15K) and test-mini (5K) subsets, collected from 11877 long-form real-world videos sourced from the web. Our videos are long, with an average duration of 522.9/754.8 seconds, a minimum of 7.0/13.0 seconds, and a maximum of 7255/3586 seconds for test/test-mini. All videos are sampled at 720p and hand-curated to promote content and domain diversity.

Video Statistics Annotation Statistics Test:Test-Mini Test:Test-Mini

#Videos 9038:2841 #QAs 15000:5000 Major Categories 10:10 Skill Types 13:11 Subcategories 35:30 Avg. Skills per QA 2.71:3.75 Avg. Duration (s) 522.9:754.8 Avg. Question Len. (w) 26.22:34.78 Min. Duration (s) 7.0:13.0 Avg. Answer Len. (w) 25.53:69.54 Max. Duration (s) 7255:3586 Avg. Answer Position (s) 302.36:357.37

[Figure 9]

- Figure 3: Distribution of MMOU. (a) Video category distribution in MMOU, covering 10 major domains and 35 fine-grained subdomains. (b) Co-occurrence matrix of QA task types, illustrating how multiple reasoning skills are jointly required within individual questions. (c) Distribution of the relative temporal positions (average of start–end time-stamps) of answer evidence within videos, showing that answers are spread across the entire video timeline. (d) Distribution of QA instances across the 13 skill/task types in MMOU. (e) Video duration distribution, highlighting the prevalence of long and complex real-world videos.

The videos span 10 major categories and 35 fine-grained subcategories, covering diverse domains such as academic lectures, sports, and other real-world scenarios (see Fig. 3). Each question in MMOU is annotated with one or more of 13 skill types, with an average of 3 skills per question. A detailed breakdown of skill-wise question distribution is provided in Fig. 3. All questions are initially annotated in an open-ended format. We subsequently convert them into a multiple-choice setting by constructing 9 hard distractors per question, resulting in 10 answer options per QA, as described in Section 3.3. The distribution of correct answer options is approximately uniform across all choices (A–J), as summarized in Table 7, 8.

To avoid positional biases, where models may exploit answers appearing near the beginning or end of the video [Liu et al., 2024, Yuan et al., 2025], we deliberately frame QAs with answer-relevant evidence at diverse temporal locations during annotation. As shown in Table 1, the average answer position is 302.36 seconds, with its distribution relative to video length illustrated in Fig. 3.

- 3.2 Dataset Comparison

- Table 2 compares MMOU with existing multimodal benchmarks. Benchmarks such as AV-Odyssey and OmniBench primarily focus on single images paired with audio, whereas MMOU targets realworld videos with synchronized audio, requiring joint audio-visual understanding. Compared to other omni-modal benchmarks, including DailyOmni, WorldSense, and OmniVideoBench, MMOU features substantially longer and more complex videos, spanning durations from a few seconds to several hours, far exceeding the temporal scope of prior benchmarks.

To further validate the necessity of cross-modal reasoning, we randomly sample 20% of MMOU and manually evaluate the instances. We find that this subset satisfies 100% answer correctness and 100% strict audio-visual dependency, substantially exceeding the cross-modal rigor of existing benchmarks reported in Chen et al. [2025]. Additionally, we highlight that modality-specific models perform poorly on MMOU. As shown in Table 3, the vision-only Qwen3-VL-32B achieves only 44% accuracy, while the audio-only Qwen3-Omni attains 35.6%, confirming that unimodal reasoning is insufficient. Overall, MMOU poses a significantly greater challenge than prior omni-modal benchmarks: even the widely used Qwen3-Omni-30B-A3B-Thinking model reaches only 19.4% accuracy, markedly lower than its performance on existing benchmarks.

- 3.3 Data Collection, Curation & Annotation

- Figure 4 illustrates the data construction pipeline for MMOU. We follow a structured, expert-driven process to ensure that all QAs require joint audio-visual understanding and reasoning over long, complex real-world videos.

##### 1. Skill and Task Curation. First, we define a taxonomy of 13 fundamental audio-visual reasoning skills to capture the diverse challenges posed by long-form, real-world videos. These skills are

Table 2: Comparison of MMOU with image (I), audio (A), video (V), and omni-modal QA benchmarks, highlighting MMOU’s scale, long-form videos and question type (Multiple Choice / Open Ended) in addition to strong audio-visual correspondence. ∗ denotes that only the number of available videos is reported.

Benchmarks Modality #Videos/Audios Avg. Len. #QA Pairs #Skills QA Type Open domain

MMMU I N/A N/A 11500 32 MC/Open ✓ MMVU V 1529 51.4 3000 27 MC/Open ✗ MMAU A 9000 10.1 10000 27 MC ✓ MMAU-Pro A 5787 123.8 5305 49 MC/Open ✓ Video-MME V 900 1017.9 2700 30 MC ✓ VideoMMMU V 300 506.2 900 30 MC ✗ LongVideoBench V 3763 473.0 6678 17 MC ✗

OmniBench A+I 1142 9.17 1142 8 MC ✗ AV-Odyssey A+V+I 620∗ 15.58 4555 26 MC ✗ UNO-Bench A+V+I 384∗ 27.1 1250 44 MC/Open ✓ DailyOmni A+V 684 43.7 1197 6 MC ✓ WorldSense A+V 1662 141.1 3172 26 MC ✓ OmniVideoBench A+V 628 384.2 1000 13 MC ✓

MMOU (Ours) A+V 11877 578.4 20000 13 MC/Open ✓

designed to require explicit integration of audio and visual information and reflect the annotation ontology followed by expert annotators.

Temporal understanding and event sequencing assess a model’s ability to reason about the order, progression, and temporal dependencies of audio-visual events across a video. Sub-scene understanding focuses on identifying and interpreting semantically important segments within long videos, often requiring contextual understanding of surrounding events. Holistic video reasoning evaluates global comprehension of the video’s main activity, objective, or theme, requiring integration of information across the entire timeline. Inference and context understanding require models to deduce unstated intentions, causes, or situational context from multiple audio-visual cues. Needle-in-the-haystack reasoning tests the ability to localize and reason about specific moments in long videos, while referential grounding evaluates linking between audio references and visual entities (or vice versa). Counting and comparative reasoning assess quantitative and relational reasoning over repeated or distinct audio-visual events. Object interaction reasoning examines the understanding of actions performed on objects and their resulting transformations over time. Audio-visual stitching evaluates reasoning over edited or stitched segments, requiring understanding of narrative continuity and editing intent. Finally, tracking spurious correlations captures cases where correct answers rely on surprising or unintuitive audio-visual evidence that cannot be inferred from language priors alone. All questions are additionally tagged with audio-visual understanding, ensuring that every instance requires joint reasoning over both modalities; questions solvable from a single modality are explicitly excluded. We provide examples in Table 9 and 10.

- 2. Video Domain Selection. Guided by our curated skill taxonomy, we then systematically select a set of video domains to ensure broad coverage of real-world audio-visual understanding and reasoning scenarios. Specifically, we define 10 major video categories and 35 fine-grained subcategories, each chosen to exercise distinct combinations of the targeted skills. For each category and subcategory, we carefully curate videos to balance coverage across domains while maintaining sufficient diversity in content, temporal structure, and audio-visual dynamics. This domain-driven selection strategy ensures that MMOU spans a wide range of real-world contexts and supports comprehensive evaluation.
- 3. Source Video Collection. We collect a total of 11877 real-world videos from publicly available online platforms (e.g., YouTube), with durations ranging from 7 seconds to 121 minutes. Videos are selected to align with the curated skill taxonomy, ensuring that each video supports the construction of

at least one high-quality question. We prioritize naturally occurring content over scripted or synthetic data, resulting in realistic audio conditions, diverse visual scenes, and authentic temporal structure suitable for evaluating long-horizon audio-visual reasoning.

- 4. Expert Question Generation. Eleven expert annotators follow a standardized annotation protocol. For each video, annotators first watch the video in its entirety. They then generate openended question–answer pairs that require joint audio and visual understanding, explicitly avoiding yes/no questions or questions answerable from text alone. More detailed guidelines are present in Appendix C. Annotators are required to annotate the earliest and latest timestamps at which the supporting evidence for the answer appears, and are encouraged to diversify the same. Each question

[Figure 10]

Figure 4: Overview of the dataset-construction pipeline for MMOU.

is tagged with one or more skill categories from our predefined taxonomy. We encourage annotators to generate multiple diverse questions per video, which are then filtered.

- 5. Distractor Generation. All questions are initially authored in an open-ended format. We then convert them into a multiple-choice setting by generating nine hard distractors per question, resulting in ten answer options. Distractors are generated using GPT-5.2 OpenAI [2025], conditioned on the question and additional video-level metadata; the full prompt is provided in Fig. 7. To increase difficulty, half of the distractors are designed to be semantically plausible and grounded in the video context, while the remaining half are intentionally out-of-context. This balanced construction prevents elimination via superficial cues and encourages genuine audio-visual reasoning. To further increase question difficulty and following prior work [Tam et al., 2025], we replace the correct answer with

“None of the above” in 13% (2000) of the test QAs, and one of the incorrect options in a further 13%

(2000). For test-mini, 250 QAs have the correct answer replaced with “None of the above”, and 250 QAs have an incorrect option replaced with “None of the above”.

- 6. Quality Control and Filtering. A separate group of expert reviewers conducts rigorous quality control, removing ambiguous, redundant, or overly trivial questions, as well as instances with misaligned timestamps or weak audio-visual grounding. Only questions that strictly require joint audio-visual reasoning and adhere to the annotation guidelines are retained, resulting in a final set of 20,000 QA pairs. The resulting annotations show strong consistency, achieving a Fleiss’ kappa score of 0.86. We also provide an empirical analysis of distractor quality in Appendix H.2.
- 7. MMOU Finalization. The final MMOU benchmark consists of 20,000 carefully curated and reviewed QA instances.

## 4 Experimental Setup

4.1 Baselines We evaluate MMOU on a diverse set of baselines spanning omni-modal, audio-only, vision-only, and text-only models.

Audio-Visual Multimodal Large Language Models. We evaluate SOTA large omni-modal models that are explicitly designed to jointly process audio and visual inputs. These models integrate modality-specific encoders with a shared language backbone and are trained using large-scale multimodal instruction-tuning data. We include both closed-source and open-source omnimodal models. Specifically, the closed-source baselines include Gemini 2.5 Flash and Pro [Comanici et al., 2025]. The open-source omni-modal models evaluated are Qwen 2.5-Omni [Xu et al., 2025a], Qwen 3-Omni-Instruct, Qwen 3-Omni-Think [Xu et al., 2025b], Phi-4 Multimodal [Abouelenin et al., 2025], Gemma 3n [Team et al., 2025], MiniCPM [OpenBMB, 2025], Video-LLaMA 2 [Cheng et al., 2024], OmniVinci [Ye et al., 2025], and Baichuan-Omni [Li et al., 2025b].

Audio-only and Vision-Only MLLMs. To isolate the contributions of visual and audio cues, we additionally evaluate MMOU using modality-restricted models. For vision-only large vision–language models, we consider Qwen3-VL-32B-Instruct and Qwen3-VL-8B-Instruct [Bai et al., 2025a] and

Table 3: Performance breakdown across video domains and video durations for closed-source, open access, and open-source audio-visual MLLMs, video-only, audio-only MLLMs, and text-only LLMs.

Video Domains Video Durations (min) Overall

Methods Sports Travel Video Games Daily Life Academic Film Pranks Music Animation News < 5 5–10 10–20 20–30 > 30 Any Random 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 Human 86.3 85.7 82.7 85.1 83.5 85.0 83.9 86.1 82.0 90.0 87.2 85.6 84.0 83.0 81.5 84.3

Closed-Source Audio-Visual MLLMs

Gemini 2.5 Pro 61.2 67.3 60.9 68.1 71.4 66.5 71.0 59.7 58.2 61.8 62.2 66.2 66.2 59.0 58.5 64.2 Gemini 2.5 Flash 56.2 59.1 46.1 60.2 57.5 61.1 54.3 52.1 49.5 52.9 55.9 57.4 57.6 49.8 45.6 55.8

Open-Source Audio-Visual MLLMs

Qwen2.5-Omni-7B 35.8 29.0 18.5 36.0 26.4 26.2 20.4 28.3 20.5 30.0 35.4 32.6 29.9 25.6 22.6 31.3 Qwen3-Omni-30B-A3B-Instruct 50.3 39.5 28.3 51.6 40.3 39.8 27.4 41.3 37.0 47.1 48.2 47.9 44.9 38.0 43.6 46.0 Qwen3-Omni-30B-A3B-Thinking 20.3 19.8 14.6 20.3 23.9 22.8 11.8 13.8 21.2 18.9 20.4 20.1 18.9 16.5 18.2 19.4 Phi-4 Multimodal 34.9 28.9 23.2 33.3 27.1 24.3 24.5 33.2 23.6 31.4 33.6 32.0 30.2 29.5 27.9 31.4 Gemma 3n 36.6 23.5 19.4 35.7 23.4 24.9 26.3 29.0 24.6 28.6 33.8 31.3 29.3 27.0 27.5 30.7 Minicpm-o 4.5 50.7 39.3 30.4 50.8 43.6 36.0 35.1 43.3 29.2 46.3 48.1 49.8 39.2 33.3 9.1 46.8 Video-LLaMA 2 27.1 24.4 18.5 27.7 24.7 22.1 23.3 25.1 22.5 22.5 26.7 25.9 23.4 22.8 22.7 24.8 OmniVinci 27.9 26.1 16.1 26.6 27.6 24.2 19.1 23.4 6.3 24.7 28.4 26.1 24.7 21.7 9.9 24.7 Baichuan-Omni-1.5 27.9 24.5 19.5 25.4 21.9 19.9 16.9 23.8 17.2 23.3 28.9 25.2 20.0 14.6 8.5 23.2

Video-Only Multimodal MLLMs

Qwen3-VL-32B-Instruct 47.8 39.9 31.9 48.4 37.2 40.4 41.9 45.5 44.1 42.2 44.5 45.3 43.3 40.4 44.1 44.0 Qwen3-VL-8B-Instruct 38.9 33.3 26.3 40.6 31.5 34.9 32.2 36.4 39.7 33.8 36.4 36.8 36.0 33.1 35.6 36.1 Qwen2.5-VL-7B-Instruct 34.2 28.1 21.5 34.9 24.8 23.7 24.6 31.0 27.4 27.6 32.0 31.4 29.0 26.2 27.6 30.2

Audio-Only Multimodal MLLMs

Audio Flamingo 3 18.7 15.4 13.0 19.4 15.6 13.7 11.1 17.8 12.3 18.8 18.7 19.1 16.9 16.2 13.9 17.7 Qwen3-Omni-30B-A3B 35.9 37.3 28.0 36.4 38.9 36.7 33.1 44.4 36.4 50.0 40.6 36.7 33.7 35.1 34.5 35.6

Cascaded Models

Qwen3-(VL+O-A) + Qwen3-235B 34.5 37.3 26.4 37.3 39.0 31.4 20.8 24.7 27.2 30.9 32.8 33.4 33.7 31.9 30.8 33.1 Qwen3-(VL+O-A) + GPT-5.2 28.8 29.0 21.6 30.3 36.8 30.8 20.7 21.1 27.7 26.5 27.7 28.6 29.4 25.0 25.4 28.1

Text-Only LLMs

Qwen3-235B 40.8 32.5 24.2 38.7 29.7 34.4 28.4 39.2 34.2 39.6 40.5 37.6 36.4 32.8 36.4 37.5 GPT-4o mini 22.9 28.3 16.8 22.2 26.8 23.6 22.2 25.5 28.1 20.0 26.3 23.7 23.6 22.0 23.2 23.8 GPT-4.1 37.4 30.7 19.6 38.3 28.6 27.7 22.2 34.0 27.2 34.0 35.5 35.0 33.1 28.8 33.3 33.9

Qwen2.5-VL-7B-Instruct [Bai et al., 2025b]. For audio-only evaluation, we include Audio Flamingo 3 [Goel et al., 2025] and Qwen3-Omni-Instruct [Xu et al., 2025b] operating in audio-only mode.

Text-Only Large Language Models & Cascaded Models. Finally, we evaluate text-only large language models and text-centric reasoning baselines. We employ Qwen3-235B, GPT variants, and only pass the question and options without any audio or visual inputs. In addition, we consider two cascaded caption-based baselines. For this setup, we first generate audio and visual captions of the video separately using Qwen3-Omni-30B-A3B and Qwen3-VL-235B-A22B-Instruct, respectively. The generated captions are then fused into a single coherent audio-visual description of the video, which is then provided to a text-only LLM to answer the question. This design evaluates whether text descriptions alone are sufficient for solving MMOU in the absence of multimodal perception.

Evaluation We evaluate our models using micro-averaged accuracy. For each question, models are shown a set of answer options and instructed to select exactly one. Next, we apply robust regularexpression–based parsing to extract the predicted option and match it via string comparison. To reduce option-order bias, we randomize the option order five times and take the majority-selected answer. We further evaluate each model under multiple prompt variants and report the best-performing prompt configuration for all MLLMs.

## 5 Results and Discussion

In Table 3, we present results on the MMOU test-subset on 20+ open- and closed-source audio-visual MLLMs, LVLMs, LALMs, and text-only LLMs. In Table 4, we present results on test-mini-subset. Proprietary closed-source Gemini 2.5 Pro [Google, 2023] establishes itself as the strongest baseline with an overall accuracy of 64.2% across diverse video domains (sports, news, travel, etc.) and durations (short, medium, and long). Compared to the performance of other open-source audiovisual multimodal models (e.g., Qwen3Omni and OmniVinci), which experience a relative drop in performance of more than 24.7%, we hypothesize the relatively strong performance of Gemini to pre-training on YouTube videos. Even the state-of-the-art models fall well short of human-level performance of 84.3% posing fundamental challenges to joint audio-visual perception and reasoning.

Cross-modal understanding is critical in MMOU. To evaluate the importance of cross-modal reasoning, we benchmark several video-only baselines from the Qwen-VL series. Despite being the state-of-the-art model in complex vision tasks, Qwen3-VL-32B achieves a low performance of 44%, necessitating the need for strong audio-visual integration. Similarly, state-of-the-art audio-only language models fail to answer most of the questions with audio modality alone, seeing a significant drop in performance of 17.7% with Audio Flamingo 3 and 35.6% with Qwen3-Omni-30B-A3B (audio-only). This confirms that MMOU requires both audio and visual modalities to answer the questions. Additionally, we also perform several unimodal ablations on omni models in Section H.1 and show that inference on a single modality almost always underperforms inference on both, further strengthening our argument.

Table 4: Performance of models on test-mini subset (overall accuracy).

Methods Overall

Human 82.60 Gemini 2.5 Pro 78.58 Gemini 2.5 Flash 66.21 Qwen3-Omni-30B-Inst. 54.10 Qwen3-Omni-30B-Think 35.76 Phi-4 Multimodal 36.40 MiniCPM-V-4.6 32.90 Gemma-3n-E4B-it 32.06 VideoLLaMA2 28.40 OmniVinci 27.75 Qwen3-VL-32B-Inst. 52.19 Qwen3-VL-8B-Inst. 41.90 Qwen2.5-VL-7B-Inst. 34.58 Audio Flamingo 3 16.81 Qwen3-Omni-30B-Audio 36.06 GPT4-mini 41.64 Qwen3-235B (Cascaded) 47.91

Text-only Large Language Models & Cascaded Models. Furthermore, we present an evaluation on SOTA text-only LLMs, Qwen3-235B [Yang et al., 2025] and GPT-4 [OpenAI, 2025], confirming minimal textual biases in the question and answer choices. Without audio-visual inputs, we cannot achieve state-of-the-art performance using commonsense knowledge and language biases alone. This effectively validates the dataset’s design and the need for true, temporally grounded audio-visual perception and reasoning. Moreover, we benchmark cascaded models by fusing video and audio captions with the question as context to the LLM. Providing a rich contextual audio-visual summary is not sufficient and indicates the need for joint end-to-end cross-modal perception.

## 6 Results Analysis

Skill-wise Performance Analysis. Figure 5a presents a skill-level breakdown of model performance on MMOU. While closed models consistently outperform open models across most skills, all models exhibit substantial weaknesses in basic and essential skills such as temporal understanding, and counting (this is consistent in test-mini shown in Table 13).

Temporal Position Sensitivity. Figure 5b analyzes model accuracy as a function of the temporal position of answer evidence within videos. Performance degrades steadily as relevant evidence appears later in the video, with a sharp drop for evidence located toward the end of long sequences. This trend is consistent across open and closed models and highlights a fundamental limitation in long-horizon temporal reasoning and context retention, even for state-of-the-art multimodal systems.

OmniVinci Qwen3Omni-Think Gemini 2.5 Pro

A-V Stitching

80

Temporal

Comparative

60

Spurious Corr

Context

40

20

Counting

Subscene

0

Sequential

Hol. Reason.

Ref. Ground.

Inference Obj. Int. Needle

(a) Skill-wise performance comparison of various models on MMOU Test set.

60

Gemini 2.5 Pro

OmniVinci

50

Qwen3-Omni

Accuracy(%)

40

30

20

10

0 10 20 30 40 50

Start Time (minutes)

(b) Model accuracy vs. evidence position in long videos.

Figure 5: Comparison of model performance across skills and temporal evidence positioning on MMOU Test. Open-Ended Evaluation. To complement our MCQ evaluation, we conduct an open-ended evaluation where models generate free-form answers without access to predefined options, similar to real-world usage of MLLMs. This helps us understand whether models possess underlying knowledge but struggle with articulation, or if their MCQ performance relies primarily on recognition and elimination strategies. Evaluation Protocol: We evaluate multiple models by prompting them to generate open-ended responses without answer options. We use GPT-5 as an LLM judge (prompt

in Fig. 8) using a four-dimensional rubric on a 1-5 scale: Correctness measures factual alignment with ground-truth answers; Completeness measures coverage of all key points; Faithfulness measures whether responses introduce unsupported claims or hallucinations; and Clarity measures whether answers are understandable, concise, and directly address the question. The weighted overall

score is computed as: 0.5 × Correctness + 03.5(Completeness + Faithfulness + Clarity). Overall Performance: Table 5 reports open-ended evaluation scores across eight models on the MMOU

Test set. Gemini 2.5 Pro leads with an overall score of 3.90, outperforming Qwen3-Omni-30BInstruct (2.86) and other models such as OmniVinci (2.64) and Qwen3-Omni-30B-Thinking (2.66).

Other models score notably lower: Gemma 3n, Audio Flamingo 3, Qwen2.5VL-7B, and Qwen3-VL-8B range from

Table 5: Open-ended evaluation scores on Test set across different dimensions with weighted overall score. Best values are in bold, and second-best are underlined.

- 1.76 to 2.37 overall. This spread indicates that open-ended evaluation clearly separates model capability: weaker models fail to perform well when required to produce free-form, grounded answers rather than selecting from options. Among the top models, Gemini 2.5 Pro and Qwen3-Omni-30B both achieve strong Faithfulness (3.80 and 3.36) and Clarity (4.62 and 4.62), suggesting they articulate responses clearly and avoid egregious hallucinations. However, Correctness (3.71 vs. 2.27) and Completeness (3.86 vs.
- 2.34) remain considerably lower even for these models, indicating ongoing challenges in accurately comprehending and fully addressing open-ended questions.

Model Correct. Complete. Faithful. Clarity Overall

Audio Flamingo 3 1.77 1.86 2.99 4.03 2.37 Qwen2.5-VL-7B-Instruct 1.53 1.64 2.63 3.83 2.12 Qwen3-VL-8B-Instruct 1.30 1.41 2.06 3.18 1.76 Gemma 3n 1.71 1.92 2.48 4.15 2.28 Qwen3-Omni-30B-Instruct 2.27 2.34 3.36 4.62 2.86 Qwen3-Omni-30B-Think 2.31 2.55 2.45 4.05 2.66 OmniVinci 2.06 2.17 3.06 4.40 2.64 Gemini 2.5 Pro 3.71 3.86 3.80 4.62 3.90

Skill-Specific Analysis. Figure 6 shows performance variation on open-ended evaluation across skill categories for Gemini 2.5 Pro. Holistic Reasoning achieves the highest scores on Correctness (4.24), Completeness (4.34), and Faithfulness (4.18), whereas Counting is the most challenging category, with the lowest scores on all three dimensions (Correctness 3.01, Completeness 3.27, Faithfulness 3.50). Completeness is generally at or above Correctness across categories. Faithfulness is strong for Holistic Reasoning (4.18) and Object Interaction (3.97), but relatively lower for Counting (3.50) and Spurious Correlations (3.75). Clarity remains consistently high across all skill types (4.50–4.70), indicating that responses are well-articulated even when overall scores vary.

[Figure 11]

4.6

- 3.93 3.67 3.73 3.01 4.24 4.00 3.63 3.88 3.64 3.70 3.77 3.75 3.68
- 4.06 3.85 3.85 3.27 4.34 4.12 3.78 3.96 3.79 3.86 3.89 3.91 3.84

Correct.

4.4

4.2

Complete.

4.0

Dimensions

3.8

- 3.96 3.73 3.80 3.50 4.18 3.95 3.76 3.97 3.77 3.76 3.75 3.87 3.76
- 4.57 4.54 4.61 4.60 4.60 4.61 4.63 4.50 4.63 4.60 4.70 4.62 4.60

Faithful.

3.6

3.4

Clarity

3.2

AVStitchingComparative ContextCountingHol.Reason.Inference Needle Obj.Int.Ref.Ground.SequentialSpuriousCorrSubsceneTemporal

Skills

Figure 6: Dimensions vs Skill Type on open-ended evaluation of Gemini-2.5 Pro outputs.

Open-Ended vs Multiple-Choice. We analyzed cases where models scored poorly on open-ended correctness (< 2 out of 5) and computed what fraction of those questions it answered correctly in MCQ format. Among questions with poor open-ended performance, Gemini 2.5 Pro answered 21.1% correctly in MCQ; Qwen3-Omni-Think 13.5%; and Omnivinci 12.9%. The discrepancy varies by skill: for Gemini 2.5 Pro, Subscene shows the highest such MCQ-correct rate (29.1%) and General Holistic Reasoning the lowest (10.5%); for Qwen3-Omni, Inference is highest (15.4%) and Object Interaction Reasoning lowest (10.5%); for Omnivinci, General Holistic Reasoning is highest (23.4%) and Counting lowest (10.8%).

This reveals three insights: (1) Open-ended evaluation is inherently harder, requiring generation rather than recognition; (2) MCQ format provides scaffolding that helps constrain the search space; and (3) Models may “know” the answer but struggle to articulate it in an open-ended format. These findings highlight that MCQ performance may overestimate true understanding and that current models exhibit asymmetric competencies across evaluation paradigms.

## 7 Conclusion, Limitations and Future Work

We introduce MMOU, a large-scale benchmark for evaluating omni-modal understanding and reasoning in long and complex real-world audio-visual videos. MMOU emphasizes joint audio–visual perception across a diverse set of reasoning skills that are central to real-world understanding. Exten-

sive evaluations show that current multimodal models struggle even with basic audio-visual reasoning over long real-world videos, revealing a substantial gap between models and humans.

MMOU also has limitations. Our benchmark is derived from publicly available web videos, which may introduce content biases and potential train–test leakage in closed and open-weight models. In addition, the multiple-choice evaluation setting, while robust, does not fully capture real-world open-ended reasoning. Future work includes (i) developing more robust evaluation protocols for openended audio-visual QA, (ii) continuously expanding the benchmark to incorporate emerging concepts and scenarios, and (iii) extending coverage beyond curated online content to include unstructured real-world videos, such as egocentric or driving scenarios.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv, 2025.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In ICCV, 2015.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, et al. Qwen3-vl technical report,

- 2025a. URL https://arxiv.org/abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

- 2025b.

Youssef Benchekroun, Megi Dervishi, Mark Ibrahim, Jean-Baptiste Gaya, Xavier Martinet, Grégoire Mialon, Thomas Scialom, Emmanuel Dupoux, Dieuwke Hupkes, and Pascal Vincent. Worldsense: A synthetic benchmark for grounded reasoning in large language models. arXiv, 2023.

Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, and Rita Cucchiara. The revolution of multimodal large language models: a survey. arXiv preprint arXiv:2402.12451, 2024.

Chen Chen, ZeYang Hu, Fengjiao Chen, Liya Ma, Jiaxing Liu, Xiaoyu Li, Ziwen Wang, Xuezhi Cao, and Xunliang Cai. Uno-bench: A unified benchmark for exploring the compositional law between uni-modal and omni-modal in omni models, 2025. URL https://arxiv.org/abs/2510.189 15.

Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audio-language omni-perception pretraining model and dataset. arXiv, 2023.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Lim, Luowei Yang, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv, 2024.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report, 2024a. URL https://arxiv.org/abs/2407.10759.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv, 2024b.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuoling Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. NVLM: Open Frontier-Class Multimodal LLMs. arXiv:2409.11402, 2024.

Caorui Li et al. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms,

#### 2025. URL https://arxiv.org/abs/2510.10689.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-Modal LLMs in Video Analysis. arXiv:2405.21075, 2024.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2025. URL https://arxiv.org/abs/2405.21075.

Sreyan Ghosh, Arushi Goel, Lasha Koroshinadze, Sang-gil Lee, Zhifeng Kong, Joao Felipe Santos, Ramani Duraiswami, Dinesh Manocha, Wei Ping, Mohammad Shoeybi, et al. Music flamingo: Scaling music understanding in audio language models. arXiv preprint arXiv:2511.10289, 2025a.

Sreyan Ghosh, Zhifeng Kong, Sonal Kumar, S Sakshi, Jaehyeon Kim, Wei Ping, Rafael Valle, Dinesh Manocha, and Bryan Catanzaro. Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities. arXiv, 2025b.

Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, ChaoHan Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, et al. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. arXiv preprint arXiv:2507.08128, 2025.

Kaixiong Gong, Kaituo Feng, Bohao Li, Yibing Wang, Mofan Cheng, Shijia Yang, Jiaming Han, Benyou Wang, Yutong Bai, Zhuoran Yang, et al. Av-odyssey bench: Can your multimodal llms really understand audio-visual information? arXiv preprint arXiv:2412.02611, 2024.

Google. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805, 2023. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob

Steinhardt. Measuring massive multitask language understanding, 2021. URL https://arxiv. org/abs/2009.03300.

Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. https://arxiv.org/abs/2502.04326, 2025. URL https://arxiv.org/abs/2502.04326.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. URL https://arxiv.org/abs/2501.13826.

Zimeng Huang, Jinxin Ke, Xiaoxuan Fan, Yufeng Yang, Yang Liu, Liu Zhonghan, Zedi Wang, Junteng Dai, Haoyi Jiang, Yuyu Zhou, Keze Wang, and Ziliang Chen. MM-OPERA: Benchmarking open-ended association reasoning for large vision-language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum?id=6BpKATZQd8.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Sonal Kumar, Šimon Sedláˇcek, Vaibhavi Lokegaonkar, Fernando López, Wenyi Yu, Nishit Anand, Hyeonggon Ryu, Lichang Chen, Maxim Pliˇcka, Miroslav Hlaváˇcek, William Fineas Ellingwood, Sathvik Udupa, Siyuan Hou, Allison Ferner, Sara Barahona, Cecilia Bolaños, Satish Rahi, Laura Herrera-Alarcón, Satvik Dixit, Siddhi Patil, Soham Deshmukh, Lasha Koroshinadze, Yao Liu, Leibny Paola Garcia Perera, Eleni Zanou, Themos Stafylakis, Joon Son Chung, David Harwath, Chao Zhang, Dinesh Manocha, Alicia Lozano-Diez, Santosh Kesiraju, Sreyan Ghosh, and Ramani Duraiswami. Mmau-pro: A challenging and comprehensive benchmark for holistic evaluation of audio general intelligence, 2025. URL https://arxiv.org/abs/2508.13992.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326, 2024a.

Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Jiafu Tang, Zhenghao Song, Dingling Zhang, et al. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. arXiv, 2025a.

Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In CVPR, 2022.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark, 2024b. URL https://arxiv.org/abs/2311.17005.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024c.

Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv, 2025b.

Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Zhaoxiang Zhang, Zachary Liu, Emmanouil Benetos, Wenhao Huang, and Chenghua Lin. Omnibench: Towards the future of universal omni-language models, 2024d. URL https:

#### //arxiv.org/abs/2409.15272.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. URL

#### https://arxiv.org/abs/2304.08485.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.

Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, Xiuyu Li, Yunhao Fang, Yukang Chen, Cheng-Yu Hsieh, De-An Huang, An-Chieh Cheng, Vishwesh Nath, Jinyi Hu, Sifei Liu, Ranjay Krishna, Daguang Xu, Xiaolong Wang, Pavlo Molchanov, Jan Kautz, Hongxu Yin, Song Han, and Yao Lu. Nvila: Efficient frontier visual language models, 2025. URL https://arxiv.org/abs/2412.04468.

Ziyang Ma, Yinghao Ma, Yanqiao Zhu, Chen Yang, Yi-Wen Chao, Ruiyang Xu, Wenxi Chen, Yuanzhe Chen, Zhuo Chen, Jian Cong, et al. Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix. arXiv, 2025.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark

for Question Answering about Charts with Visual and Logical Reasoning. In ACL, 2022. Meta. Llama 3, 2024. OpenAI. GPT-4o, 2024. OpenAI. Gpt-5.2 system card: Gpt-5.2. Technical report, OpenAI, December 2025. URL https:

#### //cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system

-card.pdf. Accessed: 2025-12-11.

OpenBMB. Minicpm-o 2.6: A gpt-4o level mllm for vision, speech, and multimodal live streaming on your phone, Jan 2025. Accessed: 2026-01-27.

Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang, Zhiyuan Zhang, Zetong Zhou, Shuang Gong, Yi Gui, Yao Wan, and Philip S. Yu. Judge anything: Mllm as a judge across any modality, 2025. URL https://arxiv.org/abs/2503.17489.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In ICML, 2023.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv, 2024a.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark, 2024b. URL https://arxiv.org/abs/2410.19168.

Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. TextCaps: A Dataset for Image Captioning with Reading Comprehension. In ECCV, 2020.

Parthasaarathy Sudarsanam, Irene Martín-Morató, Aapo Hakala, and Tuomas Virtanen. Avcaps: An audio-visual dataset with modality-specific captions. IEEE Open Journal of Signal Processing, 2025.

Kim Sung-Bin, Oh Hyun-Bin, JungMok Lee, Arda Senocak, Joon Son Chung, and Tae-Hyun Oh. Avhbench: A cross-modal hallucination benchmark for audio-visual large language models. arXiv preprint arXiv:2410.18325, 2024.

Zhi Rui Tam, Cheng-Kuang Wu, Chieh-Yen Lin, and Yun-Nung Chen. None of the above, less of the right parallel patterns in human and llm performance on multi-choice questions answering. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20112–20134, 2025.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models, 2024. URL https://arxiv.org/abs/2310.13289.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, et al. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786.

Jinchuan Tian, Sang gil Lee, Zhifeng Kong, Sreyan Ghosh, Arushi Goel, Chao-Han Huck Yang, Wenliang Dai, Zihan Liu, Hanrong Ye, Shinji Watanabe, Mohammad Shoeybi, Bryan Catanzaro, Rafael Valle, and Wei Ping. Ualm: Unified audio language model for understanding, generation and reasoning, 2025. URL https://arxiv.org/abs/2510.12000.

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying CLIP Data. In ICLR, 2024.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv, 2025a.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report, 2025b. URL https://arxiv.org/abs/2509.17765.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv, 2025.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pages 3480–3491, 2022.

Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, Yuming Lou, Dong Yang, Zhijian Liu, Yukang Chen, Ambrish Dantrey, Ehsan Jahangiri, Sreyan Ghosh, Daguang Xu, Ehsan Hosseini-Asl, Danial Mohseni Taheri, Vidya Murali, Sifei Liu, Yao Lu, Oluwatobi Olabiyi, Yu-Chiang Frank Wang, Rafael Valle, Bryan Catanzaro, Andrew Tao, Song Han, Jan Kautz, Hongxu Yin, and Pavlo Molchanov. Omnivinci: Enhancing architecture and data for omni-modal understanding llm, 2025. URL https://arxiv.org/abs/2510.15870.

Ting Yuan, Wenrui Zhang, Dong Chen, and Jie Wang. Cg-bench: Can language models assist call graph construction in the real world? In Proceedings of the 1st ACM SIGPLAN International Workshop on Language Models and Programming Languages, pages 12–20, 2025.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In CVPR, 2024.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video Instruction Tuning With Synthetic Data. arXiv:2410.02713, 2024.

Ziwei Zhou, Rui Wang, and Zuxuan Wu. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities, 2025. URL https://arxiv.org/abs/2505.17862.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 51257–51296, 2025. URL https: //proceedings.iclr.cc/paper_files/paper/2025/file/7f8f73134e253845a8f829 83219a8452-Paper-Conference.pdf.

## A Additional Dataset Statistics

Table 6 presents the distribution of videos across major categories and their respective subcategories, providing a quantitative overview of the dataset’s content diversity.

In Tables 7 and 8, we show the distribution of correct answers among the 10 answer options for the MMOU test and test-mini subsets respectively. We ensure that the correct answer is distributed approximately uniformly among all 10 option categories in both subsets.

## B Annotator Details

Our institution’s Institutional Review Board (IRB) has granted approval for all forms of human studies and annotations presented in the paper.

For the construction of this benchmark, we recruited annotators with strong backgrounds in creative and technical writing, linguistics, journalism, and analytically rigorous STEM disciplines, ensuring both linguistic sophistication and precise reasoning. Annotators were selected for their demonstrated critical thinking, creative problem-solving ability, and exceptional attention to detail, all of which are essential for producing high-quality, unambiguous question–answer pairs. Their educational backgrounds span bachelor’s and master’s degrees in English, English Literature, Creative Writing (including MFA training), Linguistics and Communication, as well as technically oriented degrees such as Audio Engineering and Acoustics, Applied and Computational Mathematics, Biochemistry with Computer Science, and Computational Applied Mathematics. This diverse yet complementary expertise enabled annotators to effectively integrate fine-grained visual and audio cues from video clips with nuanced language understanding, resulting in complex, carefully reasoned questions and answers that rigorously test multimodal comprehension.

Table 6: Distribution of Questions in MMOU Test set

Category Subcategory Questions

STEM 701 Medicine 133 Law 60 Business 50

Academic Lectures

Comedy & Kids 139 Action & Fantasy 45

Animation

DIY 1941 Crafts 1011 Gaming 233 Home Repair 52

Daily Life

Plot Recaps 249 Character & Theme Focus 126 Trailers & Promos 90

Film

Live Performance 620 Themed Narrative & Parody 10

Music

News World News 4221

Physical Mishap 214 Stunt Pranks 152 Reveal Pranks 68 Animal Pranks 46 Jump Scare 38

Pranks

Training & Preparation 2770 Analytics 288 Combat Events 239 Racket Sports 99 Press Conference 88

Sports

Destination Guides 212 Vlogs 195 Travel Tips & Etiquette 186 Cultural Insights 108

Travel

Esports Tournaments 255 Competitive Match Commentary 118 Production & Overlays 101 Battle & Mechanics 87 Player Reactions 55

Video Games

Total 15000

Table 7: Answer Option Distribution (Test)

Option Count %

- A 1,325 8.83%
- B 1,329 8.86%
- C 1,380 9.20%
- D 1,308 8.72%
- E 1,337 8.91%
- F 1,255 8.37%
- G 1,384 9.23%
- H 1,352 9.01%
- I 1,390 9.27%
- J 2,940 19.60%

Table 8: Answer Option Distribution (Test-Mini)

Option Count %

- A 502 10.04%
- B 483 9.66%
- C 469 9.38%
- D 467 9.34%
- E 484 9.68%
- F 475 9.50%
- G 439 8.78%
- H 453 9.06%
- I 506 10.12%
- J 722 14.44%

All annotators were compensated above the applicable minimum wage for their region, in recognition of the specialized expertise and sustained effort required by this task.

## C Annotation Instructions

The annotators were provided with the set of instructions below along with the skill/task QA types (Table 9 and Table 10).

- • Step 1: Watch the video in full length.

- • Step 2: Create a Q&A pair about the video.

- – All questions should be open-ended (no multiple choice or yes/no questions).
- – All questions should assess both video and audio understanding simultaneously.

- • Step 3: Annotate the timestamps of the video segment where the answer can be located.

– If the answer can be found in several places, annotate only the first occurrence.

- • Step 4: Select the task type of the question as listed in the reference table. Select all that apply.
- • Step 5: Repeat steps 1–4 if you can come up with more questions. General recommendations:

- – 2–3 questions for short videos (< 5 minutes)
- – 3–5 questions for medium videos (5 − 10 minutes)
- – More than 5 questions for long videos (> 10 minutes)
- – We encourage diverse and creative questions.

After the first round of question-answer annotations, a separate group of 10 annotators audit 20% of the QA pairs to verify benchmark quality along four axes: (1) whether the question is relevant to the video, (2) whether the question is grammatically correct, (3) whether the assigned task type is accurate, and (4) whether the provided answer is correct.

## D Human Evaluation on MMOU

Table 3 reports human evaluation results on MMOU. We recruited five graduate students, none of whom are authors of this paper, each holding at least a master’s degree, to answer the benchmark questions. Annotators were allowed to pause and rewind the videos as many times as needed, but were not permitted to revisit a previous question once they had moved on. The reported scores are averaged across all annotators.

## E Baselines

This appendix provides additional details on all models evaluated in our experiments. A complete list of models and their quantitative results is reported in Table 3.

- E.1 Closed-Source Audio-Visual MLLMs We evaluate two state-of-the-art proprietary omni-modal models:

- • Gemini 2.5 Pro is a large-scale closed-source audio-visual language model with longcontext support and advanced multimodal reasoning capabilities.
- • Gemini 2.5 Flash is a lightweight variant optimized for efficiency while retaining strong multimodal understanding.

- E.2 Open-Source Audio-Visual MLLMs

We benchmark a diverse set of open-source omni-modal models that jointly process audio and visual inputs:

- • Qwen2.5-Omni-7B, a compact open-source omni-modal model.
- • Qwen3-Omni-30B-A3B-Instruct and Qwen3-Omni-30B-A3B-Thinking, large-scale instruction-tuned and reasoning-enhanced variants, respectively.
- • Phi-4 Multimodal, a mixture-of-LoRA-based multimodal model.
- • Gemma 3n, an open multimodal extension of the Gemma family.
- • MiniCPM, a lightweight multimodal model designed for efficient deployment.
- • Video-LLaMA 2, a video-centric multimodal language model with audio understanding.

- • OmniVinci, a unified model for omni-modal perception and reasoning.
- • Baichuan-Omni-1.5, a recent open-source omni-modal model with integrated audio-visual encoders.

### E.3 Video-Only Multimodal Models

To isolate the contribution of visual information, we evaluate vision-only large vision–language models:

- • Qwen3-VL-32B-Instruct, a large vision-language model with strong spatial-temporal reasoning.
- • Qwen3-VL-8B-Instruct, a smaller variant with reduced capacity.
- • Qwen2.5-VL-7B-Instruct, an earlier-generation vision-language model.

##### E.4 Audio-Only Multimodal Models We include audio-only baselines to assess unimodal audio reasoning:

- • Audio Flamingo 3, a large audio-language model designed for long-form audio understanding.
- • Qwen3-Omni-30B-A3B operated in audio-only mode.

##### E.5 Cascaded Models We evaluate cascaded approaches that decouple perception and reasoning:

- • Qwen3-(VL+O-A) + Qwen3-235B, where audio and visual captions are generated separately and fused before being passed to a text-only LLM.
- • Qwen3-(VL+O-A) + GPT-5.2, replacing the text-only backbone with GPT-5.2.

##### E.6 Text-Only Language Models Finally, we benchmark text-only large language models using only the question and answer options:

- • Qwen3-235B, a large open-source language model.
- • GPT-4o-mini, a lightweight reasoning based text-only baseline.
- • GPT-4.1, a proprietary text-only LLM baseline.

All models are evaluated using identical question sets and evaluation protocols to ensure fair comparison across modalities.

## F Skill/Task QA Types

In Table 9 and Table 10, we show the detailed definition of the skill types in the MMOU benchmark and an example QA pair from each category.

## G Open-Ended Evaluation: Protocol, Rubric, and Judge Models

This section provides a complete description of the open-ended evaluation pipeline for MMOU, including the evaluation criteria, scoring scheme, and the two judge setups we use: a proprietary LLM judge (GPT-5) and a custom-trained judge model (Qwen 3.5 0.8B).

###### Task Type Description Question Answer

Object Interaction Reasoning

Object Interaction Reasoning asks questions about the effects of actions performed on objects, as well as track their transformations across contexts. It involves recognizing how interactions (lifting, applying, mixing, transforming) lead to different outcomes.

When a red saw is running, how and why does the pitch of the noise it makes change?

It shifts to a higher pitch because it starts shaving off pieces of wood.

Comparative Comparative questions ask about key differences or similarities between two distinct audio-visual segments or presentations.

How does the character with the sword’s motion change when he says, "I’m always three steps ahead"?

He shifts from moving into a pose to being held in a freeze frame.

Needle Needle-in-the-haystack questions are questions about a particular instance in a long video (including corresponding audio).

How does the orange character sound when reading off the jokes on a phone?

The orange character reads the jokes in a confused tone, with an uncertain, questioning inflection in his voice.

Counting Counting questions ask about the count of a particular type of audiovisual event across the video – where one requires watching the entire video and understanding its content.

At the skate park we hear a horn start blaring. How many shadows can be seen on the the skate rink at that time?

There are 7 shadows visible on the skate rink when the horn starts blaring.

Audio-Visual Stitching

Audio-Visual Stitching asks questions about how separate clips or segments are combined to create a cohesive narrative or convey a specific message. It involves reasoning about editing choices, transitions, and thematic continuity across multiple sources.

How does splicing a clip of a yellow ride cart going through water relate to the voiceover?

The woman explains that you shouldn’t wear white clothing to amusement parks because water rides can make it see-through.

Tackling Spurious Correlations

These questions ask about surprising, unexpected, unnatural, or unintuitive details that a text-only LLM would not naturally guess by statistically predicting.

What unexpected event occurs when "Jump Around" begins to play?

A woman walking down the street runs straight into a ladder, and her collision hits exactly on the beat when the song starts.

General Holistic Reasoning

These questions require a complete understanding of all audio and visual events throughout the video and deep thinking.

What is the purpose of overlaying eerie music over clips of this movie?

To give the audience a sense of unease for the man who was turned into a werewolf because he was in the wrong place at the wrong time, building tension and suspense and setting the tone for the following horror-mystery movie.

### G.1 Motivation and Process

Open-ended evaluation complements the multiple-choice (MCQ) evaluation by requiring models to generate free-form answers without access to predefined options. This setting is closer to real-world deployment and helps determine whether strong MCQ performance reflects genuine understanding or reliance on recognition and option-elimination. Recent benchmark and evaluation work has widely adopted LLM-as-a-Judge and rubric-based evaluation for open-ended multimodal outputs: Judge Anything uses multimodal LLMs as judges with scoring and pairwise comparison aligned to human ratings Pu et al. [2025], JudgeLM and related work show that LLMs fine-tuned on judgment data can approximate strong API-based judges Zhu et al. [2025], and MM-OPERA applies LLM-as-judge and multi-dimensional rubric scoring to open-ended reasoning and creative outputs Huang et al. [2025]. We follow this established paradigm and hypothesize that (i) open-ended scores will reveal gaps not

###### Task Type Description Question/Answer Answer

Subscene Questions that would ask the model to caption a relevant and important part of a long video (including audio) that is preceded and succeeded by a particular set of events – or ask a question that requires specifically understanding the context of that part.

What is strange about the scene when the narrator says "They begin to employ their abilities to do something edgy"?

Potato chips are floating through the air directly into the guy’s mouth.

Sequential Event-sequence questions ask the model about the order in which key audio-visual events occur across the timeline.

What is the order of events in this video? (A.) The commentator says, "once again does get that ball to shape away." (B.) Player 36, wearing a yellow uniform is pitching the ball, which goes in the air after the player hit it and crosses the white border line. (C.) The commentator says, "once again, back of the length played away, was in the air." (D.) Player 31, wearing a yellow uniform is pitching the ball, which goes in the air after the player hits it and lands past the white fence.

(A.) (D.) (B.) (C.)

Temporal Understanding

Temporal understanding questions ask about the order of particular audio-visual events in the video (including audio) or require an understanding of the order of audio-visual events.

What does the man in the blue shirt and orange pants do right after a person on the radio is heard saying, "so, my wife and I..."?

The man in the blue shirt and orange pants points his finger directly at a car right after the person on the radio says, "so, my wife and I..."

Referential Grounding

Referential grounding questions ask about the visuals referring to a particular event in the audio or viceversa.

What purpose does the man in the white shirt and gold tie serve? Why isn’t he in every shot?

He is the translator for two of the men giving speeches and only appears when he needs to translate their words.

Context Context-understanding questions ask about broader setting, background elements, or situational context that emerge only by integrating audio and visuals.

What do we hear in the background as the woman in the red dress smiles before the credits begin to roll? Why do we hear this?

We hear the audience applauding because she has just finished a very intense and beautiful concert with the orchestra, and they are applauding her performance.

Inference Inference questions ask about unstated purposes, intentions, or outcomes that must be deduced from multiple audio-visual clues.

Why does the blonde man shocked in the scene when he says "it’s undone now"?

He thinks the room is haunted by the dead actress’s ghost, which he believes moved the cloth and used lipstick to write "Jack" on the silver tray.

visible in MCQ accuracy (e.g., models that “know” the answer but fail to articulate it), and (ii) a four-criterion rubric (correctness, completeness, faithfulness, clarity) with a weighted overall score will provide a reliable and interpretable signal for comparing models.

Process and role of the caption. The pipeline consists of: (1) prompting MLLMs to produce open-ended responses given only the question, and video/audio as per model input. Model responses are generated under the same conditions as in deployment. (2) Scoring each response along four dimensions using an LLM judge. (3) Aggregating dimension scores into a weighted overall score for analysis. The audio-visual caption is used only during the evaluation step: both the GPT-5 judge and our custom-trained judge receive the question, ground truth answer, caption, and model response when assigning scores. The caption gives the judge additional context to verify claims and detect hallucinations, without giving that information to the model under evaluation.

[Figure 12]

- Figure 7: Prompt used for generating distractor options for questions in the MMOU benchmark.

### G.2 Evaluation Rubric: Four Criteria (1–5 Scale)

We evaluate each open-ended response on four criteria, each scored from 1 to 5. The criteria and score anchors are defined as follows.

- 1. Correctness (ground-truth consistency). Measures factual alignment between the model response and the ground-truth answer.

- • 5: Fully correct; matches the ground truth with no errors.

- • 4: Mostly correct; minor inaccuracies that do not change meaning.
- • 3: Partially correct; some correct points, some incorrect.
- • 2: Largely incorrect.
- • 1: Completely incorrect or contradictory.

Partial answers that omit key elements are penalized under correctness as well as completeness.

- 2. Completeness (ground-truth coverage). Measures how thoroughly the response covers all key points in the ground truth.

- • 5: Covers all key points.
- • 4: Misses one minor point.
- • 3: Covers about half of the key points.
- • 2: Covers very few key points.
- • 1: Essentially incomplete.

- 3. Faithfulness (hallucination control). Measures whether the response introduces information not supported by the ground truth answer or the audio-visual caption.

- • 5: No unsupported claims.
- • 4: Minor unsupported additions.
- • 3: Noticeable but limited hallucinations.
- • 2: Significant hallucinations.
- • 1: Dominated by unsupported or fabricated content.

- 4. Clarity & directness. Measures whether the answer is understandable, concise, and directly addresses the question.

- • 5: Clear, direct, and easy to understand.
- • 4: Mostly clear.
- • 3: Somewhat vague or verbose.
- • 2: Hard to follow.
- • 1: Unclear or off-topic.

### G.3 Weighted Overall Score

We combine the four dimension scores into a single overall score to rank and compare models. Correctness is weighted more heavily to reflect its importance for task success. The remaining three dimensions share the rest of the weight equally. The formula is:

Overall = 0.5 × Correctness

(1)

0.5 3

Completeness + Faithfulness + Clarity

+

All dimension scores are on the 1–5 scale, so the overall score lies in [1,5].

### G.4 GPT-5 as LLM Judge

For the main open-ended evaluation reported in the paper, we use GPT-5 as the LLM judge. The judge receives:

- • the question about the video;
- • the ground truth answer (reference);
- • the model response to evaluate;
- • a detailed audio-visual caption describing what can be perceived in the video (visuals, speech, sound, and music).

The caption provides additional context to verify claims, detect hallucinations, and reason about temporal or visual details. The judge is instructed to (i) compare the response primarily against the ground truth answer; (ii) use the caption to check support for claims and identify unsupported content; (iii) be objective and not penalize minor paraphrasing when meaning is preserved; and (iv) return structured JSON with a score and brief reason for each of the four criteria, plus an optional short overall assessment. The exact prompt used for the GPT-5 judge is provided in Figure 8.

### G.5 Custom-Trained Judge: Qwen 3.5 0.8B

In addition to the GPT-5 judge, we train a compact Qwen-3.5-0.8B model to act as an LLM judge, enabling scalable and reproducible open-ended scoring without relying on a proprietary API. The goal is to test whether a small model, fine-tuned on our rubric and data, can approximate the behavior of a strong LLM judge for the same four criteria. Our custom judge model will be released soon.

Training data. We construct a supervised dataset from MMOU-style items: each example includes the question, ground truth answer, audio-visual caption, and a synthetic model response with GPTannotated ratings for correctness, completeness, faithfulness, and clarity. The data is converted to a unified format (ShareGPT-style) for instruction tuning. Synthetic responses are generated by prompting an LLM with the question, reference answer, and audio-visual caption; the full prompt is shown in Figures 9–11. Example synthetic responses and their rubric ratings are shown in Figure 12.

Prompt for the custom judge. The model is prompted to act as an expert LLM judge. At evaluation time (when scoring a model response), the judge receives the question, ground truth answer, audiovisual caption, and model response: the same inputs as the GPT-5 judge. The prompt defines the same four criteria (correctness, completeness, faithfulness, clarity) with the same 1–5 score anchors and instructs the model to output only valid JSON with a score and brief reason per dimension. The caption is the primary grounding source for faithfulness (to detect hallucinations). The reference answer is used for correctness and completeness. The prompt used for training our custom judge is shown in Figure 13.

Training setup. We fine-tune Qwen 3.5 0.8B with LoRA (low-rank adaptation) for supervised fine-tuning (SFT). Hyperparameters are chosen to preserve general capability while adapting the model to the judge task. This yields a lightweight judge that can be run locally and used to score open-ended responses consistently with our rubric.

## H Ablation Studies

### H.1 Unimodal Ablation Study

To further validate that strong performance on MMOU requires joint audio-visual reasoning, we evaluate the top-performing models under unimodal (audio-only and video-only) settings alongside their full audio-visual performance.

- Table 11: MCQ accuracy (%) under unimodal and omni-modal input settings. The substantial gap between unimodal and audio-visual performance confirms that strong benchmark performance requires integrating both modalities.

Model Audio-only Video-only Audio-Visual

Gemini 2.5 Pro 35.1 39.9 57.5 Gemini 2.5 Flash 37.5 36.1 55.2 Qwen3-Omni-30B-A3B-Instruct 35.6 36.3 36.3

- Table 12: Open-ended evaluation scores under unimodal and omni-modal input settings. Without multiple-choice options to enable guessing or elimination, the unimodal-to-multimodal gap widens further, reinforcing the necessity of joint audio-visual perception.

Model Audio-only Video-only Audio-Visual

Gemini 2.5 Pro 1.99 2.20 3.74 Qwen3-Omni-30B 1.28 1.62 2.81

### H.2 Distractor Quality Analysis

To empirically validate distractor quality, we analyse the type of wrong option selected by three models evaluated on MMOU. Each distractor is labelled at generation time as either in-video (grounded in real

Table 13: Skill-wise accuracy (%) on MMOU test-mini.

Skill Gemini 2.5 Flash Qwen3-Omni-30B-Instruct Phi-4 Multimodal

Audio-Visual Stitching 67.0 50.9 34.1 Comparative 66.4 53.0 36.4 Context Understanding 65.0 51.6 33.8 Counting 55.5 46.3 31.5 General Holistic Reasoning 67.1 55.1 40.9 Inference 67.6 53.9 36.0 Object Interaction Reasoning 66.8 55.0 37.7 Sequential 63.0 50.2 31.4 Subscene 67.2 52.3 33.4 Tackling Spurious Correlations 62.8 53.4 30.1 Temporal Understanding 64.1 50.7 32.5

audio-visual events from the video; options generated to confuse models that partially understood the content) or out-of-video (plausible-sounding but entirely fabricated; not present in the video). Of the 9 distractor options per question, 4 are in-video and 5 are out-of-video. As shown in Table 14, across all three models 77–91% of wrong answers select an in-video distractor — far exceeding the 44.4% expected under random selection. This confirms that our distractors require precise audio-visual understanding to reject: models demonstrably watch the video and identify relevant content, yet are systematically confused by wrong-but-grounded options. The small fraction of out-of-video errors (9–23%) further shows that fabricated distractors are largely filtered out by all models, ruling out superficial elimination strategies.

- Table 14: Distractor type breakdown of model errors on MMOU. In-Video distractors reference real audio-visual content from the video; Out-of-Video distractors are plausible but fabricated. Unknown and unparseable predictions are split equally between the two wrong categories. Accuracy figures match the official results in Table 3. The random baseline assumes uniform selection over all 10 options; “among wrong” percentages are conditional on selecting a wrong answer (4 in-video and 5 out-of-video wrong options out of 9 total wrong options).

Category Qwen2.5-Omni-7B Qwen3-Omni-Inst. OmniVinci Random Baseline Prediction breakdown

Correct 4,695 (31.3%) 6,900 (46.0%) 3,705 (24.7%) 10.0% Wrong 10,305 (68.7%) 8,100 (54.0%) 11,295 (75.3%) 90.0%

Among wrong answers — distractor type selected

In-Video wrong 8,673 (84.2%) 7,338 (90.6%) 8,742 (77.4%) 44.4% Out-of-Video wrong 1,632 (15.8%) 762 (9.4%) 2,553 (22.6%) 55.6%

## I Compute Resources

All model evaluations were conducted on servers equipped with NVIDIA A6000 and A100 GPUs. Closed-source models (Gemini 2.5 Pro and Gemini 2.5 Flash) were accessed via their respective APIs, as were text-only and cascaded LLM backbones (GPT-4o mini, GPT-4.1, and GPT-5.2). Open-source models were run locally on 8 GPUs per model. For models with 30B+ parameters, evaluation on the full test set of 15,000 questions required approximately 5–6 wall-clock hours (44 GPU-hours per model). For smaller models in the 7B–8B range, evaluation required approximately 3–4 wall-clock hours (28 GPU-hours per model). In total, open-source model evaluations amounted to approximately 530 GPU-hours across locally-run models spanning audio-visual, vision-only, and audio-only baselines.

## J Broader Impacts

MMOU is a benchmark designed to evaluate omni-modal understanding in multimodal large language models, and as such carries both positive and negative societal implications. On the positive side,

MMOU enables rigorous, structured evaluation of joint audio-visual reasoning — a capability central to real-world AI applications such as accessibility tools (e.g., automated audio description for the visually impaired), educational video analysis, and human-computer interaction. By exposing systematic failure modes in current models, MMOU can guide the development of more robust and equitable multimodal systems. On the negative side, advances in omni-modal understanding enabled by benchmarks like MMOU could accelerate development of systems capable of automated video surveillance, large-scale media monitoring, or generation of convincing multimodal disinformation. We note that our dataset is released solely for non-commercial research purposes.

## K Dataset Quality Control and Safeguards

To ensure benchmark quality and content safety, we implemented a multi-stage quality control process. After the initial annotation round, a separate group of reviewers audited 20% of QA pairs across four axes: (1) relevance of the question to the video, (2) grammatical correctness, (3) accuracy of the assigned skill/task type, and (4) correctness of the provided answer. Questions failing any criterion were revised or discarded, with feedback relayed iteratively between reviewers and annotators until all retained questions met the quality bar.

For content safety, all collected videos were manually reviewed by annotators prior to question generation. Videos containing harmful, offensive, or sensitive content were excluded from the dataset. Videos were sourced from publicly available platforms (e.g., YouTube) and are used in accordance with their Terms of Service for non-commercial research purposes.

## L Baseline Model Licenses

- Table 15 lists all baseline models evaluated in this paper along with their associated licenses and references. Closed-source models and API-only LLMs were accessed under their respective terms of service.

Table 15: Licenses for all baseline models used in MMOU evaluations. AV = Audio-Visual, V = Vision-only, A = Audio-only.

Model Type License Reference Closed-Source Audio-Visual MLLMs

Gemini 2.5 Pro AV Proprietary (API Terms of Service) Comanici et al. [2025] Gemini 2.5 Flash AV Proprietary (API Terms of Service) Comanici et al. [2025]

Open-Source Audio-Visual MLLMs Qwen2.5-Omni-7B AV Apache 2.0 Xu et al. [2025a] Qwen3-Omni-30B-Instruct AV Apache 2.0 Xu et al. [2025b] Qwen3-Omni-30B-Thinking AV Apache 2.0 Xu et al. [2025b] Phi-4 Multimodal AV MIT License Abouelenin et al. [2025] Gemma 3n AV Gemma Terms of Use Team et al. [2025] MiniCPM-o 4.5 AV Apache 2.0 OpenBMB [2025] Video-LLaMA 2 AV Apache 2.0 Cheng et al. [2024] OmniVinci AV NVIDIA OneWay Noncommercial License Ye et al. [2025] Baichuan-Omni-1.5 AV Apache 2.0 Li et al. [2025b]

Vision-Only MLLMs

Qwen3-VL-32B-Instruct V Apache 2.0 Bai et al. [2025a] Qwen3-VL-8B-Instruct V Apache 2.0 Bai et al. [2025a] Qwen2.5-VL-7B-Instruct V Apache 2.0 Bai et al. [2025b]

Audio-Only MLLMs Audio Flamingo 3 A NVIDIA OneWay Noncommercial License Goel et al. [2025] Qwen3-Omni-30B (audio-only) A Apache 2.0 Xu et al. [2025b]

Text-Only LLMs & Cascaded Models Qwen3-235B Text Apache 2.0 Yang et al. [2025] GPT-4o mini Text Proprietary (API Terms of Service) OpenAI [2024] GPT-4.1 Text Proprietary (API Terms of Service) OpenAI [2025]

[Figure 13]

##### Figure 8: Prompt used for open-ended evaluation with the GPT-5 LLM judge. The judge receives the question, ground truth answer, model response, and detailed audio-visual caption, and returns scores and reasons for the four criteria.

[Figure 14]

##### Figure 9: Synthetic Response Generation Prompt: Part (a)

[Figure 15]

##### Figure 10: Synthetic Response Generation Prompt: Part (b)

[Figure 16]

Figure 11: Synthetic Response Generation Prompt: Part (c) — Prompt used to generate synthetic model responses for our custom LLM judge training (shown in three parts, (a), (b) and (c)). The prompt provides the question, reference answer, and audio-visual caption and asks to produce exactly 12 responses with controlled error types and ratings on correctness, completeness, faithfulness, and clarity.

[Figure 17]

##### Figure 12: Two examples of synthetic model responses and their four-criterion rubric ratings (correctness, completeness, faithfulness, clarity), illustrating a faithfulness corruption and a combined failure.

[Figure 18]

##### Figure 13: Prompt used for training our custom LLM Judge.

