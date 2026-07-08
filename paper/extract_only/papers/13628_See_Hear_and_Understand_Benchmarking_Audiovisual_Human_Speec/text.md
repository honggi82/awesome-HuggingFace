# arXiv:2512.02231v2[cs.CV]10Apr2026

## See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models

Le Thien Phuc Nguyen1* Zhuoran Yu1* Samuel Low Yu Hang2 Subin An2 Jeongkik Lee2 Yohan Ban2 SeungEun Chung2 Thanh-Huy Nguyen3 JuWan Maeng 2 Soochahn Lee2 Yong Jae Lee1

1University of Wisconsin-Madison 2Kookmin University 3Carnegie Mellon University https://plnguyen2908.github.io/AV-SpeakerBench-project-page/

#### Abstract

Multimodal large language models (MLLMs) are expected to jointly interpret vision, audio, and language, yet existing video benchmarks rarely assess fine-grained reasoning about human speech. Many tasks remain visually solvable or only coarsely evaluate speech, offering limited insight into whether models can align who speaks, what is said, and when it occurs. We introduce AV-SpeakerBench, a curated benchmark of 3,212 multiple-choice questions focused on speaker-centric audiovisual reasoning in realworld videos. It features: (1) a speaker-centered formulation that treats speakers—not scenes—as the core reasoning unit; (2) fusion-grounded question design embedding audiovisual dependencies into question semantics; and (3) expert-curated annotations ensuring temporal precision and cross-modal validity. Comprehensive evaluations show that the Gemini family consistently outperforms opensource systems, with Gemini 2.5 Pro achieving the best results. Among open models, Qwen3-Omni-30B approaches Gemini 2.0 Flash but remains far behind Gemini 2.5 Pro, primarily due to weaker audiovisual fusion rather than visual perception. We believe AV-SpeakerBench establishes a rigorous foundation for advancing fine-grained audiovisual reasoning in future multimodal systems.

#### 1. Introduction

Multimodal large language models (MLLMs) have rapidly evolved in recent years, extending language models into image [9, 32, 51, 65], video [31, 51, 62], and audio understanding [17, 25, 61]. As this evolution continues, recent efforts have moved beyond pairwise modality fusion toward building omni-models that aim to jointly process vision, audio, and language in a unified architecture [21, 47, 56, 57]. Such capability is essential for real-world applications like

*Equal Contribution

[Figure 1]

Figure 1. Motivation of AV-SpeakerBench. Many existing video benchmarks contain visually solvable questions, allowing strong multimodal models to answer correctly even without audio (left; e.g., Video-MME [14]). By contrast, AV-SpeakerBench (right) requires true audiovisual fusion, where answering depends on who speaks, when they speak, and how speech unfolds over time.

video dialog agents, meeting transcription systems, and human–AI interaction platforms, where the model must see, hear, and reason over multiple signals simultaneously. However, evaluating whether current models can truly integrate multiple modalities–rather than treat one as the dominant source–remains an open challenge.

In particular, audiovisual speaker perception has been a long-standing research problem [23, 24, 42, 43, 45, 52], yet modern MLLMs are rarely evaluated on this ability. This gap arises for two reasons. First, existing speaker datasets are built around closed-set labels or framelevel supervision, making them incompatible with openended, language-based evaluation. Second, current video QA benchmarks seldom focus on speaker-level reasoning: many questions can be solved using a single modality [14, 48], while others emphasize coarse audio–visual matching or non-speech acoustic events [27, 58, 66]. As a result, current benchmarks do not systematically evaluate whether multimodal models can jointly determine who is speaking, what is being said, and in what visual context.

To address this gap, we introduce AV-SpeakerBench, a new benchmark for evaluating fine-grained audiovisual

reasoning centered on human speakers in real-world video. AV-SpeakerBench is explicitly designed to test whether multimodal models can jointly interpret visual, auditory, and linguistic information at the speaker level—a capability not captured by existing datasets. Below, we outline the benchmark’s key design principles.

Speaker-centric task formulation. AV-SpeakerBench frames every question around the speaker as the fundamental reasoning unit, shifting evaluation from scene-level understanding to human-centered audiovisual grounding. Each video includes multiple visible individuals, enabling identity-based questions that require models to determine who speaks, when, and under which visual context. By spanning diverse conversational settings and speaker configurations—factors known to increase difficulty in audiovisual perception [23, 42, 43, 45, 52]—the benchmark tests whether models can reliably resolve speaker behavior amid visually complex and varied conversational dynamics.

Fusion-driven question design. AV-SpeakerBench uses a four-choice MCQ format in which auditory and visual cues are jointly encoded in both questions and answer options. This design ensures that solving each item requires cross-modal fusion—for example, associating spoken phrases with visible speakers, interpreting speech in relation to visual events, or resolving multiple voices in a shared scene. Together, these constructions reflect how audiovisual understanding naturally relies on coordinating what is heard with what is seen.

High-quality human annotation. All clips are manually selected and temporally segmented to isolate moments where speech-driven interaction occurs. Annotators then identify segments that genuinely require audiovisual reasoning—such as aligning an utterance with the correct visible speaker—and compose MCQs grounded in these spans. Each clip and question undergoes multi-stage expert review to ensure temporal precision and cross-modal validity.

Dataset summary and evaluation scope. Our IRBapproved AV-SpeakerBench contains 3,212 curated question–answer pairs across 12 task types, all centered on speakers as the core reasoning unit. These tasks span temporal localization, speaker identification, speech-content reasoning, utterance counting, paralinguistic attribute comparison and so on, collectively evaluating a model’s ability to integrate recognition, grounding, and temporal reasoning across audio and vision rather than depend on static or unimodal cues.

Experimental findings and implications. Comprehensive evaluation across open-source and proprietary models reveals a clear performance gap between current MLLMs and human accuracy. Among open-source models,

only recent omni-directional approaches—such as Qwen3Omni [57]—show substantial progress, with the 30B variant reaching parity with Gemini 2.0 Flash [8]. However, Gemini 2.5 Pro [8] remains the strongest overall, outperforming all others across nearly all tasks. Our analysis (Section 4.3) shows that this advantage comes primarily from stronger audiovisual fusion: Gemini 2.5 Pro consistently gains 10–20% from audio inputs, whereas Qwen3-Omni’s improvements are modest or sometimes negative. A systematic error-pattern study further reveals that weaknesses in audio perception and temporal grounding are the dominant sources of failure. Together, these findings show that speaker-centered multimodal reasoning remains a core barrier to human-level audiovisual understanding.

#### 2. Related Work

Multimodal Understanding Benchmarks. Multimodal evaluation has expanded from image benchmarks—such as VQA [3, 19], chart/document understanding [39, 40, 46, 53], and general capability suites [13, 26, 34]—to video benchmarks that introduce temporal structure and richer scene context. Different video datasets emphasize different abilities: long-form narrative reasoning [38], temporal ordering [4, 35], procedural understanding [49, 54], egocentric perception [10, 20], world modeling [22], and broad capability aggregation [14, 29]. While existing video benchmarks broaden multimodal evaluation, they rarely require fine-grained integration between visual signals and human speech, nor do they target speaker-centric reasoning. AVSpeakerBench fills this gap by providing a dedicated benchmark for speaker-level audiovisual understanding.

Audiovisual Understanding Benchmarks. Audiovisual speech understanding has been studied through both nonspeech and speech-focused datasets. Non-speech benchmarks such as AudioSet [16] and VGGSound [5] target event-level acoustic classification, while speech datasets cover specific facets of communication: VoxCeleb [7, 41] for speaker identity, LRS/LRS3 [2] for transcription, and AVA-ActiveSpeaker / ASW [24, 45] for frame-level speaking and audibility labels. However, these resources rely on closed-set labels or low-level annotations and thus do not evaluate the open-ended multimodal reasoning abilities expected of modern MLLMs.

Recent work has begun exploring audiovisual reasoning for multimodal models, but mainly through task formulations that differ fundamentally from speaker-centric reasoning. VGGSounder [66] reannotates VGGSound [5] for multi-label audio event classification; AV-Odyssey [18] focuses on audio–visual matching tasks—such as pairing images with audio clips or selecting which video aligns with a given sound—rather than reasoning about spoken content. WorldSense [22] similarly frames audiovisual understanding as matching between visual scenes and acoustic cues

Audiovisual Reasoning Speaker-centric Reasoning

Benchmarks Modalities #Videos #QA Anno.

AV Temporal Localization

Speech–Speaker Attribution

Speech temporal localization

Speech Grounding

AV Attribution

MSRVTT-QA [55] V 2,990 72,821 A ✗ ✗ ✗ ✗ ✗ ActivityNet-QA [59] V 800 8,000 M ✗ ✗ ✗ ✗ ✗ MVBench [29] V 3,641 4,000 A ✗ ✗ ✗ ✗ ✗ Video-Bench [44] V 5,917 17,036 A&M ✗ ✗ ✗ ✗ ✗ EgoSchema [38] V 5,063 5,063 A&M ✗ ✗ ✗ ✗ ✗ Video-MME [14] V 900 2,700 M ✗ ✗ ✗ ✗ ✗ MMBench-Video [11] V 609 1,998 M ✗ ✗ ✗ ✗ ✗

AVQA [58] A+V 57,000 57,335 M ✓ ✗ ✗ ✗ ✗ OmniBench [30] A+I ✗ 1,142 M ✓ ✗ ✗ ✗ ✗ AV-Odyssey [18] A+I 220 4,555 M ✓ ✗ ✗ ✗ ✗ WorldSense [22] A+V 1,662 3,172 M ✓ ✓ ✗ ✓ ✗ Daily-Omni [64] A+V 648 1,197 A&M ✓ ✓ ✗ ✗ ✗

AV-SpeakerBench (Ours) A+V 2,051 3,212 M ✓ ✓ ✓ ✓ ✓

- Table 1. Comparison of multimodal video QA benchmarks. “A” means audio, “V” means video, “I” means image, “S” means subtitle. “Anno.” is the annotation type (A: automatic, M: manual).

(e.g., music style or sound summary), and its tasks span a wide range of audio types without centering human speech; OmniBench pairs static images with narrated audio for temporal reconstruction; and Daily-Omni targets high-level scene understanding rather than modeling speaker identity or spoken utterances. We compare AV-SpeakerBench with other benchmarks in Table 1.

Across these benchmarks, audio is used to characterize scenes, events, or background context, but not to bind speech to visible speakers. AV-SpeakerBench instead adopts a speaker-centric formulation that requires aligning spoken content with the people who produce it.

Multimodal Large Language Models. Multimodal LLMs have progressed from image–text models such as BLIP-2 [28], InstructBLIP [9], and LLaVA [32, 33] to video-oriented systems like Video-LLaMA [6, 62], VITA [12], and PandaGPT [47], which introduce temporal encoding for multi-frame reasoning. Recent omni-modal models—including Gemini [50], Qwen-Omni [57], Phi-4 Multimodal [1], StreamOmni [63], Unified-IO 2 [37], OLA / OneLLM [21, 36], and AnyGPT [60]—aim to unify audio, video, and language through instruction tuning and shared cross-modal alignment. However, existing evaluations largely emphasize visual QA or coarse audio–visual matching, providing limited insight into whether these models can perform fine-grained, speaker-centric audiovisual reasoning. AV-SpeakerBench directly targets this gap by assessing whether models can jointly resolve who is speaking, what is said, and when it occurs within natural multi-speaker video.

#### 3. AV-SpeakerBench

Recent multimodal video benchmarks such as VideoMME [14] have advanced the evaluation of large multi-

modal models on a wide range of video understanding tasks. However, many existing questions remain largely visionsolvable, such as “What is the total number of people in the video?” or “What clothes is the singer wearing?”, where the correct answer can be inferred from visual frames alone without attending to the audio stream. Moreover, only a few benchmarks include human speech in their audiovisual evaluation [27, 58, 66], and those that do remain at a coarse categorical level. For example, VGGSounder [66] extends VGGSound with multi-label audio–visual classification, where human speech is represented by broad categories such as male speech, female speech, or singing. Such tasks capture the presence of speech but not its linguistic or speakerspecific content, leaving open whether models can reason about who speaks and what is said in natural audiovisual scenes.

Our goal in constructing AV-SpeakerBench is to fill this gap by focusing on speaker-centric audiovisual reasoning. Each question in our benchmark is designed to require joint interpretation of visual and auditory cues in natural human speech scenes. In the following sections, we describe the benchmark’s design principles, data curation and annotation pipeline, and dataset statistics.

##### 3.1. Fusion-Driven Question Design

The central goal of AV-SpeakerBench is to evaluate audiovisual fusion along the temporal dimension—that is, whether a model can coherently align what is seen and heard as events unfold. All questions adopt a unified multiple-choice (MCQ) format to standardize evaluation and to encode auditory and visual cues compactly within the question and its answer options. Each item is written so that the correct response requires the model to integrate auditory and visual evidence to reason over temporally ordered actions and utterances.

Examples of Audiovisual Reasoning

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

Q: Among the people who speak, who speaks the most quietly overall?

Q: Who speaks immediately after someone says, "So can I put forward a bit of a pitch?"

Q: Does the man with glasses speak after he receives the candies?

- A. The woman in the gray sweater

- B. The man in the dark brown shirt

- C. The man in the striped shirt

- D. The woman in the brown jacket

- A. The woman in the blue sweater

- B. The man in the patterned jacket

- C. The man in the light gray sweater

- D. The woman in the white shirt

- A. No, he just gives an awkward smile

- B. Yes, he gives in after refusing at first

- C. Yes, he replies in an unfriendly tone

- D. No, he only looks at the other man

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

Q: What does the woman in pink scrubs say just before she puts the metal tray down? A. "I am sorry, I am sorry" B. "Okay, that helps" C. "I am a little nervous" D. "Well, don't be"

Q: After the man in the grey shirt wiggles his fingers, until the end of the video, how many times is "red line" mentioned by all speakers?

Q: After the woman in a black jacket says, "This is very datable," until the end of the video, how many unique people are visible, even partially?

A. Zero B. One C. Two D. Three

A. Three B. Four C. Five D. Six

Visual Reference Audio Reference

Dataset Statistics

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Figure 2. Top: Examples of audiovisual reasoning questions in AV-SpeakerBench. Each question illustrates a distinct way in which audiovisual dependency is enforced—through spoken-phrase grounding, visual event conditioning, cross-modal temporal localization, or multi-speaker coordination—ensuring that the correct answer cannot be inferred from a single modality. Bottom: Dataset Distribution. We present the distribution of videos by duration, task category, and visual complexity (measured by the number of unique visible people). Together, these statistics highlight the diversity of conversational scenes and reasoning types represented in AV-SpeakerBench.

In our formulation, audiovisual dependency is embedded directly in the textual construction of the question and its options. The fusion of modalities appears in several complementary ways, including—but not limited to— (1) linking a spoken phrase to a visible identity (e.g., When does the man wearing a black T-shirt say ‘What’s going on?’); (2) cross-conditioning one modality on temporal cues from the other—for example, using visual events to localize speech (e.g, What does the woman sitting on the porch swing say just before she drinks from her glass?, or conversely, using auditory events to anchor visual reasoning; At the moment the man with the black headband says, ”We are not cool,” how many people are visible?); and (3) combining auditory and visual cues in multi-speaker settings (e.g., After the man in the grey shirt wiggles his fingers, until the end of the video, how many times is “red line” mentioned by all speakers?). Through this formulation, each question enforces temporally grounded multimodal reasoning, ensuring that successful models must align voices, appearances, and

contextual events rather than rely on static visual or textual shortcuts. Representative examples of different formulation types are illustrated in Figure 2-top, and a complete set of per-task examples is provided in the Appendix D.

##### 3.2. Dataset Construction

Video sources. We collect all video data from YouTube, prioritizing content that naturally features human speech and multi-party interaction. The majority of clips are sampled from movie clips, as these provide rich audiovisual dynamics such as turn-taking, facial gestures, and speaker transitions within short temporal windows. To diversify speaking style, visual layout, and recording conditions, we additionally include material from game shows, street interviews, sport interviews, group interviews, podcasts, and vlogs. All videos are taken from official channels where individuals are clearly and knowingly recorded on camera.

Clip selection and task-oriented sampling. Unlike datasets where annotators label pre-defined clips, our pro-

cess begins with segment selection. Annotators watch full videos and identify 5–30s clips that satisfy task requirements and contain meaningful audiovisual variation. Choosing the right moment is crucial: segments with too few speakers (e.g, monologue) or unchanged conversational dynamics make temporal or speaker-grounded questions nearly deterministic (e.g., if only two speakers are visible, Who speaks after the woman says ‘How are you?’ offers little diagnostic value). Similarly, an anchor condition that does not shift who speaks fails to introduce real temporal grounding. This filtering ensures that all selected clips support genuine temporal localization and cross-modal reasoning rather than superficial pattern matching.

Annotation and refinement. For each selected clip, annotators compose the question and four answer options following detailed task guidelines specifying the required modality pairings. Distractors are drawn from entities, actions, or speech events within the same clip; when insufficient, they are formed by recombining visible or auditory attributes to maintain contextual realism. All annotators are experienced researchers rather than crowdsourced workers, ensuring close adherence to multimodal reasoning criteria. Each annotation undergoes a multi-stage refinement pipeline: (1) an initial review by a separate researcher for clarity and multimodal validity; (2) linguistic and structural polishing using a language model; and (3) a final verification round by at least two additional researchers other than the original annotator. During this process, ambiguous, inconsistent, or trivially solvable examples are filtered out. This rigorous curation pipeline ensures that all remaining questions exhibit temporal sensitivity, speaker grounding, and robust audiovisual reasoning, providing a balanced and reliable basis for evaluation. Annotation details can be found in Appendix A.

##### 3.3. Dataset Statistics

After refinement, AV-SpeakerBench comprises 3,212 MCQ pairs across 12 task types. All tasks are speaker-centric but probe different aspects of audiovisual understanding, including temporal localization, speaker identification, speech-content retrieval, utterance counting, paralinguistic attribute comparison, and so on. Together, they introduce diverse compositional and temporal challenges that require coordinated multimodal reasoning rather than reliance on static or unimodal cues. Appendix D provides illustrative examples for each task type.

We visualize dataset-wide statistics in Figure 2-bottom, including the distributions of question types, video durations, and speaker counts. The dataset spans a wide range of question formulations and task complexities, reflecting the natural diversity of audiovisual reasoning scenarios. We ensure at least 200 validated examples per task to maintain balanced coverage across reasoning types. Speaker-

related tasks contain more examples, since temporal reasoning about who speaks and when can draw on cues from either modality—for instance, determining whether someone speaks after an audio or visual event. In contrast, tasks centered purely on visual or auditory recognition must rely on the opposite modality to remain genuinely multimodal, which limits the diversity of question formulations rather than their total count. Although speaker-related tasks therefore include more examples, they also encompass a wider variety of reasoning patterns, and we report per-task accuracy throughout all evaluations to avoid bias toward these categories.

We further analyze the dataset by measuring its visual complexity, quantified by the number of unique visible speakers per clip—a standard indicator of difficulty in human-centric audiovisual benchmarks [24, 42, 45]. AVSpeakerBench covers a wide range of interaction densities, from short dyadic exchanges to multi-party conversations, providing a balanced yet challenging evaluation environment for multimodal large language models.

##### 3.4. Interpretation and Scope of Evaluation

Although all question types in AV-SpeakerBench are designed to require audiovisual reasoning in principle, we recognize that strong models may occasionally infer answers from visual cues alone—such as mouth movement or gestures—without explicitly relying on the audio stream. We regard this as a natural and desirable behavior rather than a flaw: humans similarly infer speech activity from silent visual context. Our benchmark therefore evaluates whether a model can integrate both modalities when available, rather than penalizing instances where one modality alone happens to suffice. In practice, audio information consistently simplifies reasoning and boosts accuracy, but models achieving correct answers through robust visual understanding remain within the intended evaluation scope.

#### 4. Experiments 4.1. Experiment Setup

Model Selection. We evaluate multimodal LLMs that natively support audio–video inputs. Among proprietary systems available to us, Gemini [50] is the only model offering true A+V processing; we report both the thinking and nonthinking versions of Gemini 2.5 Flash, and only the thinking version of Gemini 2.5 Pro (the only mode provided). For open-source models, we include all publicly available A+Vcapable LLMs across omni-modal and video-first architectures: Qwen2.5-Omni [56], Qwen3-Omni [57], VideoLLaMA [62], Video-LLaMA2 [6], Unified-IO 2 [37], OLA [36], OneLLM [21], VITA [12], VITA-1.5 [15], PandaGPT [47], Phi-4 Multimodal-Instruct [1], and AnyGPT [60]. For Qwen3-Omni-30B, we report only the non-thinking variant, as the thinking mode takes about 5

###### Speaker-centric Visual-centric Audio-centric

Activity

Attribute Recognition

Task Type Detection Recognition Counting

Recognition Counting Recognition Duration Pitch Rate Intensity Counting Overall Human Performance – 96.02 93.13 94.28 93.14 93.20 94.15 96.52 90.68 93.20 91.39 94.17 93.40 93.74

###### Proprietary Models

Gemini 2.5 Pro Thinking [8] – 81.73 74.15 74.13 72.55 73.30 62.93 77.11 78.81 67.48 69.86 71.84 63.89 73.04 Gemini 2.5 Flash Thinking [8] – 74.71 70.62 60.95 65.59 70.39 65.85 78.61 69.92 66.5 67.46 65.05 58.33 67.84 Gemini 2.5 Flash [8] – 69.79 68.24 50.75 61.76 68.45 51.22 71.64 58.47 60.68 60.29 59.71 40.97 60.27 Gemini 2.5 Flash-Lite [8] – 45.90 52.44 39.10 48.53 51.94 43.41 55.72 51.69 51.46 46.89 54.37 36.11 47.23 Gemini 2.0 Flash [8] – 60.19 63.51 47.51 54.9 63.59 45.85 71.14 56.78 62.62 55.98 55.83 41.67 53.21 Gemini 2.0 Flash-Lite [8] – 56.91 57.35 38.81 49.02 55.83 45.85 67.66 52.12 52.43 51.12 56.31 33.33 51.43

###### Open-sourced Audiovisual Models

Video-LLaMA [62] 7B 29.51 26.07 29.35 27.23 32.52 25.37 31.84 28.81 31.55 28.23 27.67 21.53 28.21 Video-LLaMA [62] 13B 30.91 29.86 25.37 27.23 37.38 29.76 27.86 23.31 27.67 33.49 30.58 27.08 29.11 Video-LLaMA2 [6] 7B 34.19 36.02 31.25 35.29 37.38 41.46 29.85 40.25 49.03 44.02 45.15 31.25 37.67 PandaGPT [47] 7B 28.34 20.61 27.36 16.18 20.87 32.20 33.83 19.07 24.27 17.70 14.56 15.63 22.88 PandaGPT [47] 13B 20.61 16.82 28.36 0.98 1.94 25.37 5.97 22.46 22.82 25.84 23.30 15.63 18.37

###### Open-sourced Omni Models

Unified-IO 2 large [37] 1B 24.36 27.49 23.88 24.51 23.79 22.93 21.89 22.03 25.24 31.58 31.55 34.38 26.15 Unified-IO 2 xl [37] 3B 28.81 28.44 31.25 30.39 26.21 23.41 24.88 30.93 22.82 26.32 30.58 31.25 27.52 Unified-IO 2 xxl [37] 7B 26.70 27.25 24.63 30.39 22.82 29.76 23.28 35.59 25.24 33.49 37.38 28.8 24.97 Phi-4 Multimodal [1] 5.6B 37.70 41.71 28.02 46.95 33.50 38.05 45.77 38.56 49.03 42.58 37.38 26.04 38.45 OneLLM [21] 7B 30.44 23.70 20.65 30.88 25.24 27.8 25.37 1.69 24.27 28.23 25.73 34.72 24.97 AnyGPT [21] 7B 24.12 4.27 24.88 0.98 19.90 19.51 15.42 0.00 0.00 0.48 2.43 22.92 12.67 VITA [12] 7B 32.08 34.60 29.85 37.25 36.41 28.78 32.84 37.29 38.83 35.89 37.86 28.13 33.66 VITA-1.5 [15] 7B 32.08 36.73 32.59 38.24 35.44 35.12 32.34 45.34 43.20 44.98 36.89 29.51 36.27 Qwen2.5-Omni [56] 3B 46.37 45.26 36.57 48.04 47.57 41.95 52.74 54.24 44.17 39.23 39.23 28.12 43.52 Qwen2.5-Omni [56] 7B 51.76 44.31 36.57 50.49 50.49 44.88 58.71 51.27 57.28 47.85 49.51 29.51 46.64 Qwen3-Omni [57] 30B 61.83 54.74 46.77 56.86 58.74 40.49 68.16 59.32 58.74 55.98 66.02 34.72 54.14

- Table 2. Evaluation Results of AV-SpeakerBench. Our evaluation covers proprietary models, open-sourced audiovisual and omni models. Highest number per model type in each column is bolded, and second-highest number per model type in each column is underlined.

minutes per query and would place a full benchmark run beyond our budget. All models are evaluated with their native A+V interfaces using identical inputs and prompts; further details are provided in Appendix B.

Frame Sampling. We follow each model’s default temporal sampling policy. Concretely, we sample 8 frames for Video-LLaMA [62], Video-LLaMA2 [6], Phi-4 Multimodal-Instruct [1], Unified-IO 2 [37], and AnyGPT [60]; 5 frames for PandaGPT [47] and OneLLM [21]; 1 fps capped at 100 frames for VITA [12] and VITA-1.5 [15]; and 1 fps for Gemini [50], Qwen2.5Omni [56], and Qwen3-Omni [57]. Unless otherwise noted, frames are sampled uniformly within each clip. We ablate the performance of these models using different sampling strategies in Appendix F.

##### 4.2. Results

The full evaluation results are presented in Table 2; all results are reported in MCQ accuracy. Here, we summarize the main takeaways.

Existing models remain far from human performance. Human evaluation reaches an overall accuracy of 93.74%, confirming that the questions are clear and naturally solvable through audiovisual reasoning. In comparison, the strongest model—Gemini 2.5 Pro—achieves only 73.04%, leaving a gap of over 20 percentage points. This indi-

cates that current multimodal LLMs still have substantial room for improvement on fine-grained, temporally grounded speaker-centric reasoning.

Gemini models demonstrate stronger performance than open-source models. Gemini 2.5 Pro (thinking) delivers the best results on AV-SpeakerBench, outperforming all other models on 11 of the 12 tasks and achieving an overall accuracy of 73.04%. Gemini 2.5 Flash (thinking) improves upon the non-thinking Flash variant by 7.57%, yet still falls notably short of 2.5 Pro. Since both variants employ the thinking process, this performance gap primarily reflects differences in the underlying model strength, with possible additional gains from how the thinking mechanism interacts with Pro’s larger capacity.

Only recent Omni-series models achieve non-trivial performance on AV-SpeakerBench. Earlier opensource multimodal models—such as Video-LLaMA2 [6], PandaGPT [47], OneLLM [21], Unified-IO [37], and AnyGPT [60]—perform poorly on AV-SpeakerBench, often approaching random-guessing accuracy despite supporting both audio and video inputs. Recent Omni-series models, including Phi-4-Multimodal [1], VITA/VITA-1.5 [12, 15], Qwen2.5-Omni [56], Qwen3-Omni [57], and OLa [36], demonstrate substantial improvements, marking the first generation of open-source models capable of handling finegrained audiovisual reasoning. Among them, Qwen3-

100

Gemini 2.5 Pro Visual Only

Qwen3-Omni 30B Visual Only

Gemini 2.5 Pro Gain w/ Audio Gemini 2.5 Pro Loss w/ Audio

Qwen3-Omni Gain w/ Audio Qwen3-Omni Loss w/ Audio

| |
|---|

| |
|---|

+26.5

+23.7

+47.3

80

+26.5

+17.2

+17.5

+27.4

+13.6

16.7% Temporal Localization

+21.1

+38.8

+11.2

+9.7

+29.9

+6.3

+13.4

-2.4

31.7% Audio Perception

+3.0

+6.3

+6.4

+6.2

60

+8.1

Score(%)

+4.7

-3.9

25.0% Temporal Grounding

40

+1.7

13.3% Visual

20

13.3% Perception Cross-modal Attribution

0

SpeakerDetectionSpeakerRecognitionActivityRecognitionSpeechRecognitionAttributeRecognitionSpeakerCountingSpeechCountingVisualCountingSpeechIntensitySpeechDurationSpeechPitchSpeechRate

(a) Modality ablation across task types. Gemini 2.5 Pro demonstrates consistent multimodal gains across most tasks, whereas Qwen3-Omni 30B exhibits limited or even negative audio contributions in certain tasks.

(b) Error type distribution across benchmark categories. Most errors occur in audio perception and temporal reasoning tasks.

Figure 3. Multimodal ablation and error analysis.

Omni-30B is the strongest, exceeding all other open-source models and even slightly surpassing Gemini 2.0 Flash. However, it still remains far behind the Gemini 2.5 family, underscoring the difficulty of speaker-centric audiovisual reasoning even for the most advanced open-source LLMs.

##### 4.3. Analysis

Performance gaps reflect differences in audiovisual fusion capability. From Table 2, we observe that Gemini 2.5 Pro consistently outperforms Qwen3-Omni-30B across all categories on AV-SpeakerBench. We investigate this difference by comparing each model’s performance under vision-only and audiovisual input settings (Figure 3a). Gemini 2.5 Pro exhibits consistent gains of roughly 10–20 percentage points across all tasks when both modalities are available, indicating stable and effective fusion. In contrast, Qwen3-Omni-30B achieves much smaller gains—and in some tasks, even negative differences—suggesting that audio input does not reliably improve its reasoning. These results indicate that Gemini’s advantage primarily arises from stronger temporal alignment and cross-modal integration, whereas future progress in speaker-centric audiovisual reasoning will depend on improving fusion robustness rather than merely scaling unimodal perception.

Why advanced models may answer some audiovisual questions using only vision. Although AV-SpeakerBench is designed to require audiovisual reasoning, we observe that advanced models can occasionally answer certain questions using only visual cues. Human speech naturally produces observable signals—such as mouth motion, gaze shifts, and conversational gestures—that can provide partial evidence about who is speaking or when speech occurs. As a result, strong models may exploit these cues even in the absence of audio. More information can be found in Appendix G.

Number of Unique People ≤2 3 4 ≥ 5 Gemini 2.5 Pro Thinking [8] 74.8 74.1 74.1 70.9 Gemini 2.5 Flash Thinking [8] 71.8 68.4 66.8 65.1 Qwen2.5-Omni 7B [56] 52.0 47.8 45.3 42.7 Qwen3-Omni 30B [57] 58.3 52.9 52.0 54.4

Table 3. Accuracy by the number of visible people. Models generally decrease in accuracy as visual complexity increases.

- Figure 4a shows an example where Gemini 2.5 Pro cor-

rectly identifies the most active speaker under vision-only input by tracking sustained mouth movement and gesturing patterns. However, these cues are not always reliable. In

- Figure 4b, the same vision-only setting leads the model to a wrong answer because the visible gestures mislead it about the speaker’s rate of speech. When audio is provided (Figure 4c), the model correctly resolves the ambiguity, demonstrating that the task genuinely requires multimodal fusion.

Importantly, these visual cues reflect natural properties of human communication rather than dataset bias. AVSpeakerBench does not explicitly penalize unimodal solutions—if a model can reliably infer speech activity from vision alone, this reflects genuine capability (e.g., implicit lip-reading or motion-level reasoning). However, as Figures 4b–c illustrate, fusion offers a clear advantage: audio resolves ambiguities that visual cues alone cannot, and robust performance across the benchmark consistently requires integrating both modalities. Our goal is therefore not to suppress unimodal cues, but to design tasks where multimodal fusion provides the most reliable and general path to correct reasoning.

Failure cases for Gemini 2.5 Pro. To better understand the remaining challenges for Gemini 2.5 Pro on AVSpeakerBench, we qualitatively examined five failure cases per task and categorized them into four major types: (1)

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

From the start of the video until just before a person holding a tray is shown, who speaks the most among those who speak?

###### Of those who speak, who has the lowest rate of speech?

... Then Yellow Shirt joins the scene.From 0:04 onwards, Yellow Shirt takes center stage. Her mouth is moving constantly, she's gesturing, and it's obvious she's the one driving the conversation.

… There's a guy with a shaved head in the beginning. He seems to be taking his time, making deliberate gestures, and not rushing his words, which is giving me a strong indication his speech rate will be on the slower side. … Based on that quick visual assessment, the shaved-head guy (C) seems to be the clear winner for the lowest speech rate. ❌

[Figure 48]

[Figure 49]

... Therefore, the person in the light yellow T-shirt (Yellow Shirt) clearly speaks the most. The final answer, as requested, is "C". ✅

(a) Vision-only: Correct ✔

(b) Vision-only: Incorrect ✖

[Figure 50]

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

After the man shows the drawing to the woman, until the end of the video, how many unique people speak, out of all the people visible on screen, even partially?

Of those who speak, who has the lowest rate of speech?

… Aha! At 00:05, he fully reveals it to the two women. That's the start of our timeframe. At 00:07, the woman on the right says \"Okay\". Great, that's one. The man laughs too and then, at 00:08, he asks \"Is it better or worse?\" That's speaker number two. The woman on the left, still no words. … Therefore, two out of three people spoke during that period. D is the answer. ❌

… The first guy, shaved head, rattles off a few sentences pretty quickly. The second guy, sunglasses, just says "Absolutely," Comparing the two, there's a huge difference. Shaved head is probably doing hundreds of words per minute; sunglasses is one word, maybe a second long. … Hmm, it seems A is the most obvious correct choice..The correct answer is A. ✅

[Figure 62]

[Figure 63]

(d) Audiovisual: Incorrect ✖

(c) Audiovisual: Correct ✔

- Figure 4. Qualitative examples of Gemini 2.5 Pro reasoning traces on AV-SpeakerBench. Green and red highlight colors indicate the model’s correct and incorrect reasoning, respectively. (a) Vision-only example answered correctly: the model identifies the correct speaker by tracking the duration and consistency of mouth movement and conversational gestures, which serve as natural visual cues for inferring who is speaking. (b) Vision-only example answered incorrectly: the model incorrectly associates slower gestures with slower speech, leading to a wrong prediction. (c) The same example as (b) but with audio input: the model correctly identifies the faster speaker once speech-rate evidence becomes available, confirming that the question requires true audiovisual fusion. (d) Vision + audio example answered incorrectly: the model predicts that only one woman speaks while both women say “Okay” after the event. Eventually, all three speakers talk after the event, showing residual difficulty in temporal alignment and speaker disambiguation.

Visual/Audio perception error—misperceiving information within a single modality, such as misidentifying a gesture or mishearing a spoken phrase; (2) Cross-modal attribution error—recognizing unimodal content correctly but mismatching it across modalities, e.g., attributing an utterance to the wrong speaker; (3) Temporal grounding error—identifying the correct event but applying an incorrect temporal relation, such as reasoning over the after segment when the question asks for before; and (4) Temporal localization error—selecting the wrong segment in audio or video, leading to misaligned reasoning. As summarized in Figure 3b, most failures arise from audio perception and temporal grounding, indicating that even advanced models struggle to parse overlapping speech and accurately anchor reasoning in time. Figure 4d further illustrates an audio perception error where the model misses one of two speakers uttering the same word in close succession. Additional examples for each error type are provided in Appendix C.

Evaluation by visual complexity. To examine how visual complexity impacts model performance, we regroup ques-

tions by the number of visible faces, as shown in Table 3. All models show lower accuracy as the number of people increases, indicating that multi-person scenes create substantial challenges for speaker identification, temporal association, and audiovisual grounding.

#### 5. Conclusion

We introduced AV-SpeakerBench, a benchmark for evaluating fine-grained, speaker-centric audiovisual reasoning in real-world video. Unlike prior multimodal datasets that focus on scene- or event-level understanding, AVSpeakerBench centers on human speakers and explicitly requires integrating auditory and visual cues. Our experiments show that while Gemini 2.5 Pro exhibits strong fusion and temporal grounding, other models still face clear limitations in multimodal integration, underscoring the need for more robust fusion mechanisms in future multimodal systems. We hope this benchmark provides a foundation for advancing multimodal models toward genuine audiovisual understanding—where language, vision, and speech are jointly reasoned within the same conversational context.

#### 6. Acknowledgement

This work was supported in part by NSF IIS2404180, IBM, and Institute of Information & communications Technology Planning& Evaluation (IITP) grants funded by the Korea government (MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), (No. RS-2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training), (No. RS-2025-2543949. EnvironmentAware and Domain-Adaptive Multimodal Embodied AI for Real-World Interaction), and (No.RS-202502219317, AI Star Fellowship (Kookmin University)).

#### References

- [1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025. 3, 5, 6
- [2] Triantafyllos Afouras, Joon Son Chung, and Andrew Zisserman. Lrs3-ted: a large-scale dataset for visual speech recognition. arXiv preprint arXiv:1809.00496, 2018. 2
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 2
- [4] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking finegrained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 2
- [5] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725. IEEE, 2020. 2
- [6] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 3, 5, 6
- [7] J. S. Chung, A. Nagrani, and A. Zisserman. Voxceleb2: Deep speaker recognition. In INTERSPEECH, 2018. 2
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2, 6, 7
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in neural

- information processing systems, 36:49250–49267, 2023. 1, 3
- [10] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European conference on computer vision (ECCV), pages 720–736, 2018. 2
- [11] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124, 2024. 3
- [12] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, et al. Vita: Towards opensource interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 3, 5, 6
- [13] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirtyninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 2
- [14] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 1, 2, 3
- [15] Chaoyou Fu, Haojia Lin, Xiong Wang, YiFan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long MA, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, and Ran He. VITA-1.5: Towards GPT4o level real-time vision and speech interaction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 5, 6
- [16] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and humanlabeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017. 2
- [17] Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6288–6313, 2024. 1
- [18] Kaixiong Gong, Kaituo Feng, Bohao Li, Yibing Wang, Mofan Cheng, Shijia Yang, Jiaming Han, Benyou Wang, Yutong Bai, Zhuoran Yang, et al. Av-odyssey bench: Can your multimodal llms really understand audio-visual information? arXiv preprint arXiv:2412.02611, 2024. 2, 3
- [19] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating

- the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 2
- [20] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2
- [21] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26584– 26595, 2024. 1, 3, 5, 6
- [22] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal LLMs. In The Fourteenth International Conference on Learning Representations, 2026. 2, 3
- [23] Chaeyoung Jung, Suyeon Lee, Kihyun Nam, Kyeongha Rho, You Jin Kim, Youngjoon Jang, and Joon Son Chung. Talknce: Improving active speaker detection with talk-aware contrastive learning. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8391–8395. IEEE, 2024. 1, 2
- [24] You Jin Kim, Hee-Soo Heo, Soyeon Choe, Soo-Whan Chung, Yoohwan Kwon, Bong-Jin Lee, Youngki Kwon, and Joon Son Chung. Look who’s talking: Active speaker detection in the wild. In Proc. Interspeech 2021, pages 3675– 3679, 2021. 1, 2, 5
- [25] Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: a novel audio language model with few-shot learning and dialogue abilities. In Proceedings of the 41st International Conference on Machine Learning, pages 25125–25148, 2024. 1
- [26] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 2
- [27] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, JiRong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19108–19118, 2022. 1, 3
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 3

- [29] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206, 2024. 2, 3

- [30] Yizhi LI, Ge Zhang, Yinghao Ma, Ruibin Yuan, King Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Moore Wang, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, Yidan WEN, Yanghai Wang, Shihao Li, Zhaoxiang Zhang, Ruibo Liu, Emmanouil Benetos, Wenhao Huang, and Chenghua Lin. Omnibench: Towards the future of universal omni-language models. In The Thirtyninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 3
- [31] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984, 2024. 1
- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 3
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024. 3
- [34] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 2
- [35] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024. 2
- [36] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025. 3, 5, 6
- [37] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024. 3, 5, 6
- [38] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 2, 3
- [39] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022. 2
- [40] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 2
- [41] A. Nagrani, J. S. Chung, and A. Zisserman. Voxceleb: a large-scale speaker identification dataset. In INTERSPEECH, 2017. 2

- [42] Le Thien Phuc Nguyen, Zhuoran Yu, Khoa Quang Nhat Cao, Yuwei Guo, Tu Ho Manh Pham, Tuan Tai Nguyen, Toan Ngo Duc Vo, Lucas Poon, Soochahn Lee, and Yong Jae Lee. Unitalk: Towards universal active speaker detection in real world scenarios. arXiv preprint arXiv:2505.21954, 2025. 1, 2, 5
- [43] Le Thien Phuc Nguyen, Zhuoran Yu, and Yong Jae Lee. Laser: Lip landmark assisted speaker detection for robustness. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7291–7300, 2026. 1,

- 2

[44] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. Computational Visual Media, 2025.

- 3

- [45] Joseph Roth, Sourish Chaudhuri, Ondrej Klejch, Radhika Marvin, Andrew Gallagher, Liat Kaver, Sharadh Ramaswamy, Arkadiusz Stopczynski, Cordelia Schmid, Zhonghua Xi, et al. Ava active speaker: An audio-visual dataset for active speaker detection. In ICASSP 2020-2020 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 4492–4496. IEEE, 2020. 1, 2, 5
- [46] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 2
- [47] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. In Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the era of Interactive Assistants!, pages 11–23, 2023. 1, 3, 5, 6
- [48] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: speech-enhanced audio-visual large language models. In Proceedings of the 41st International Conference on Machine Learning, pages 47198– 47217, 2024. 1
- [49] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207– 1216, 2019. 2
- [50] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 3, 5, 6
- [51] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1
- [52] Xizi Wang, Feng Cheng, and Gedas Bertasius. Loconet: Long-short context network for active speaker detection. In

- Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18462–18472, 2024. 1, 2
- [53] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024. 2
- [54] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 2
- [55] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM International Conference on Multimedia, page 1645–1653, New York, NY, USA,

2017. Association for Computing Machinery. 3

- [56] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 1, 5, 6, 7
- [57] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. 1, 2, 3, 5, 6, 7
- [58] Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM International Conference on Multimedia, page 3480–3491, New York, NY, USA, 2022. Association for Computing Machinery. 1, 3
- [59] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):9127–9134, 2019. 3
- [60] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9637–9662, 2024. 3, 5, 6
- [61] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15757–15773,

2023. 1

- [62] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations, pages 543–553, 2023. 1, 3, 5, 6
- [63] Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, and Yang Feng. Stream-omni: Simultaneous multimodal in-

- teractions with large language-vision-speech model. arXiv preprint arXiv:2506.13642, 2025. 3
- [64] Ziwei Zhou, Rui Wang, and Zuxuan Wu. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025. 3
- [65] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. 1
- [66] Daniil Zverev, Thadd¨aus Wiedemer, Ameya Prabhu, Matthias Bethge, Wieland Brendel, and A Koepke. Vggsounder: Audio-visual evaluations for foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1027–1037, 2025. 1, 2, 3

## See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models

### Supplementary Material

#### A. Annotation Details

This section outlines the details of our annotation process. We first describe the interface used to collect audio–visual annotations, followed by the task guidelines provided to annotators to ensure consistency across examples. We then outline the criteria used to select experienced annotators for this work. Finally, we include a trivially solvable case to illustrate the intended task clarity and establish a lower bound on annotation difficulty.

##### A.1. Annotation Interface

Figure 5 shows the web-based interface used to collect multiple-choice audio–visual annotations. Each example is presented as a self-contained card with a video player at the top and all annotation fields displayed below. The card includes immutable metadata (e.g., video id, video type, category, sub category, task id) and an approved flag, followed by the natural-language question and four answer options (A–D). Annotators watch the clip, specify the temporal span used for reasoning via the start time and end time fields, select the correct answer, and provide a short explanation in the “reason for the answer” field, typically referencing visual or auditory evidence. The researchers responsible for review will examine the reason together with the question and answer choices to determine whether a question should be refined or approved.

##### A.2. Annotation Guideline

###### A.2.1. General Guidelines

We provide annotators with a set of general guidelines to ensure consistent and high–quality audio–visual question creation. Annotators work from a pool of English-language YouTube videos and follow a standardized workflow: each clip is added to the interface, a 5-to-30-second segment is selected, and a single multiple-choice question is authored according to the assigned task type. Annotators record the answer, and supply a brief justification.

Video Selection and Clip Duration. We provide the following constraints to maintain data quality:

- • Captions. Select videos without persistent on-screen subtitles or captions that would trivially reveal the spoken content.
- • Content safety. Exclude clips containing extreme or explicit violence or gore. Mild or non-gratuitous physical

- conflict is allowed, but highly graphic or disturbing content should be avoided.
- • Language. All selected videos must be in English.
- • Identity visibility. Ensure that the queried identity is clearly visible in the foreground and is visually distinguishable from other people in the scene.
- • Discriminative attributes. When designing questions about a specific person, choose attributes (e.g., clothing, position, actions) that clearly distinguish that person from others in the clip.
- • Contextual distractors. Construct all answer options from entities, actions, or speech events within the same clip. When there are not enough distinct candidates, form distractors by recombining visible or auditory attributes (e.g., clothes, positions, or phrases) so that each option remains contextually plausible and balanced.
- • Clip length. Localize each question to a contiguous 5– 30s segment, depending on the task’s requirements, trimming longer spans to the minimal window that still supports correct reasoning.

###### A.2.2. Task-specific Guideline

We provide task-specific requirements in Table 6. Annotators apply these templates when constructing multiplechoice questions. The annotators will add a few questions for us to provide feedback before they scale up the data annotation.

##### A.3. Annotator Selection

To ensure high-quality labels for tasks requiring finegrained audio–visual reasoning, we restrict annotation to experienced researchers rather than crowd workers. Our annotator team has prior research experience in video understanding, speech processing, or multimodal large language models, enabling them to identify and select clips that are meaningful and informative for the tasks. This careful selection process is crucial for producing labels of sufficient quality and reliability, which would be difficult to achieve with crowd-sourced annotators.

In addition, researchers are trained on the full task taxonomy (see Section A.2) before annotation. This ensures that annotators understand the distinction between trivially solvable cases and those requiring joint audio–visual inference, and could apply task definitions precisely.

##### A.4. Eliminating Trivially Solvable Questions

A key step in our quality-control pipeline is identifying and removing trivially solvable questions—items whose an-

[Figure 64]

- Figure 5. Annotation interface for rate–comparison tasks. The interface presents annotators with the video clip, metadata (video ID, category, task type), the question, all answer choices, and the selected response. Annotators also specify the temporal window used for judgment and provide a brief justification. The examples shown correspond to (left) lowest rate of speech, (middle) highest rate of speech, and (right) lowest rate of speech for a different time span within the same video. These examples illustrate how annotators validate temporal reasoning by explicitly grounding answers in the video timeline.

swers can be inferred without reference to specific temporal or spatial cues in the clip. These questions arise when the structure of the video or the phrasing of the question makes the correct answer globally obvious, regardless of any localized event. Typical examples include questions where the target event occurs throughout the video or is otherwise easily predictable, providing little meaningful challenge for the fine-grained audio–visual reasoning our tasks are designed to assess.

- • Persistent speaker patterns: e.g., one person speaks continuously throughout the clip, making questions such as “Does the man speak when event X happens?” solvable without checking the requested moment.
- • Constant scene composition: e.g., the number of visible people remains unchanged across the entire segment, allowing “How many people are visible when event X occurs?” to be answered without performing momentspecific counting.
- • Globally obvious answers: cases where the question refers to a specific moment or segment, but the answer

can be determined from a property that holds across the entire clip rather than from the localized event.

• Always-on video captions: some videos contain burnedin subtitles throughout the entire clip. In such cases, questions involving spoken content (e.g., “What does the man say after event X?”) become answerable by simply reading the captions rather than performing speech recognition or audio–visual alignment. Therefore, these questions fail to evaluate the intended modality.

Figure 6 shows two representative examples. In the first, only a single person is visible across the entire segment, so the answer to a moment-specific visibility question is globally obvious. In the second, the spoken line appears verbatim as burned-in captions on the screen, allowing the correct answer to be selected without listening to or timing the utterance. Such items do not evaluate the intended audio– visual capabilities and are therefore removed.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Q: At the moment the man in black suit says, "someone is making a ton of money", how many people are visible, even partially?

[Figure 72]

Instruction

- A. One B. Two C. Three D. Four

[Figure 73]

[Figure 74]

[Figure 75]

Q: What does the man in grey sweater says before the man in black sweaters points to him?

- A. “Steven Vegali?”
- B. “Steven Vegetabli”
- C. “What would the plumber say to the singer?”
- D. “If Steven Seagal removed all animal products from diet, what would we call him?”

Figure 6. Examples of trivially solvable questions removed during filtering. (Top) A moment-specific visibility question becomes trivial because only one person is visible throughout the entire clip, making the answer recoverable without grounding to the referenced utterance. (Bottom) A speech-content question becomes trivial because the spoken line appears as burned-in captions, allowing the answer to be selected without performing audio-based reasoning.

- B. Detailed Evaluation

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.

[Figure 76]

Question and Answers

Among the people who speak, who speaks the most quietly overall?

- A. The woman in the gray sweater
- B. The man in the dark brown shirt
- C. The man in the striped shirt
- D. The woman in the brown jacket The best answer is:

Figure 7. Evaluation prompt used for multimodal LLMs. For each example we show a short video (represented here by keyframes and the waveform), a fixed natural-language instruction, and a multiple-choice question with four options (A–D). Models must answer by outputting only the letter of the correct option.

format helps create deterministic results through standardized instructions across all sub-benchmarks, following prior work [14].

##### B.2. Human Performance

This section provides additional details on our evaluation protocol beyond what is reported in the main paper. We first specify the exact prompts and answer format used when querying multimodal language models on our multiple–choice audio–visual questions (Sec B.1). We then describe our procedure for measuring human performance, including annotator recruitment, instructions, and aggregation of responses (Sec B.2), which serves as a reference ceiling for model accuracy on our benchmark.

To obtain a reliable human upper bound, we conduct a separate human evaluation that is fully decoupled from the dataset annotation workflow. None of the human evaluators participated in authoring or reviewing the dataset questions. The evaluation team consists of ten undergraduate and master’s students, all with strong English proficiency. Each evaluator answered roughly 300 questions, providing broad, non-overlapping coverage of the test set. This design follows common practice in recent multimodal benchmarks and ensures that the reported human accuracy reflects genuine task difficulty rather than annotator familiarity.

##### B.1. Evaluation Prompt

At test time, we query models with a single unified prompt for all tasks. Figure 7 illustrates the format. Each instance consists of (i) the video input (uniformly sampled frames plus the full audio track), (ii) a fixed instruction, and (iii) a multiple-choice question with four options labeled A–D.

Evaluation Interface and Protocol. Figure 8 shows the interface used for human evaluation. The interface presents the video clip, accompanied by playback controls (Play, Pause, Replay), followed by the multiple-choice question exactly as posed to the model. Evaluators select their answer from four options (A–D) without access to transcripts, subtitles, or any auxiliary textual resources. To keep track on the number of speakers in each question, the interface also includes a control prompt asking evaluators to report the total number of people visible in the video.

The instruction reads:

“Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.”

The question and answer block then specifies the task (e.g., speech intensity, speaker counting) and lists four candidate answers in natural language. We parse the model’s prediction by mapping it to the corresponding option among A, B, C, or D. Any response that does not contain a valid option is counted as incorrect. This strict, letter-only

Each evaluator completes their assigned questions through the interface in random order. After submission, the interface immediately advances to the next item, preventing revisiting or revision. Evaluators do not have access

[Figure 77]

- Figure 8. Human evaluation interface. Evaluators watch the video clip, then answer the corresponding multiple-choice question (A–D). No transcript or subtitle is provided. The interface also includes an optional refinement tag and a control question asking for the total number of people visible in the video. This setup ensures that human performance is independent of annotation and directly comparable to model outputs.

to ground-truth labels or the responses of other participants. The interface additionally provides an optional refinement tag (e.g., trivial error) that allows evaluators to flag questions with clear issues, such as mismatches between the question and answer options, or between the video and the question. These tags are used for data sanity checks and are not shown to models or used in computing human accuracy. In practice, no refinement tags were submitted during the human evaluation process.

This protocol ensures that human performance is (i) independent of the annotation process, (ii) based solely on audio–visual information, and (iii) directly comparable to model predictions under the same multiple-choice setting.

C. Qualitative Analysis by Error Pattern

- Figure 9 presents qualitative examples of Gemini 2.5 Pro’s reasoning traces on AV-SpeakerBench, organized by four representative error patterns. In all cases, the model has access to the full video and audio segment, yet its thinking reveals where the audiovisual reasoning breaks down.

- (a) Cross-modality attribution. In Figure 9a, the question asks how many unique people speak after the host waves his hand toward the screen. The video shows that only the host and a woman respond verbally. However, the model attributes the utterance to the man standing on the left, and further hallucinates an additional off-screen male speaker. This error reflects a failure to align the acoustic evidence (voice timbre and timing) with the correct visual identity, leading to incorrect speaker attribution and overcounting of speakers.
- (b) Audio and visual perception. The example in Figure 9b targets the person with the lowest voice pitch among several women. The model first misperceives the visual scene, describing a non-existent “woman in a dark green off-the-shoulder top” instead of the actual woman in a blue dress. It then claims that the woman in a khaki sweater has the lowest pitch, even though her pitch is not the lowest in the clip. This case jointly exposes incorrect visual (wrong clothing description) and incorrect audio perception (wrong comparative pitch), showing that errors in one modality can reinforce errors in the other.
- (c) Temporal grounding. The example in Figure 9c probes whether the model can restrict its reasoning to a short temporal window. The question asks how many people are visible, even partially, when the man with a cup in his hand speaks at the specific frame that the man starts saying the phrase. The model correctly identifies this time span as the anchor, but then counts people who appear before (a person walking in the background) and after (a person farther back near the cars). The answer is therefore based on people outside the annotated window, illustrating a temporal grounding error: the anchor is found, but the visual evidence is aggregated over a broader interval than the question specifies.
- (d) Temporal localization. In Figure 9d, the example focuses on the number of times “I remember it” is mentioned by all speakers from the beginning of the clip until a specific action: the man in the blue sweater putting on the headphones. In the video, the relevant action occurs at approximately 00:08, and the correct count is determined by utterances up to that moment. The model, however, localizes the anchor around 00:10 in its reasoning trace, shifting the cut-off point later in time. This mis-localization changes which utterances are counted and leads to an incorrect answer, highlighting that even small temporal offsets can cause large reasoning errors in speech-counting tasks.

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

After the man in the middle waves his hand toward the screen, until the end of the video, how many unique people speak, out of all the people visible on screen, even partially? … First, I need to pinpoint the start time, which is after the host waves his hand toward the screen. I found that moment around the 0:08 mark. I've identified: **Speaker 1:** the host, is clearly speaking. **Speaker 2:** A man on the left says something celebratory. **Speaker 3:** Another man on the left also speaks. … This aligns perfectly with option D, which is \"3 out of 6\". I am now confident in my selection. ❌

[Figure 91]

[Figure 92]

Among those who speak, who has the lowest pitch?

… The task is to identify the woman with the lowest voice pitch. The first is a woman in a dark green, off-the-shoulder top her voice is in the mid-range. Then, a woman in a khaki or beige sweater speaks, and her voice is noticeably lower. There are other women in the video but they're not speaking. … Only the woman in the khaki sweater, actually speaks with the lower pitch. Option D is clearly the answer. ❌

[Figure 93]

[Figure 94]

(a) Cross-modality Attribution

(b) Audio and Visual Perception

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

From the start of the video until just before the man in a blue sweatshirt is seen wearing the headset, how many times is "I remember it" mentioned by all speakers?

At the moment the man with a cup in his hand starts saying, "Yay, I'm a great student," how many people are visible, even partially?

[Figure 105]

[Figure 106]

… Let's see, the timestamp is around 00:12 - 00:13. Right, got it. We've got the obvious ones: the guy filming, pretty clear. Then the guy with the cup who's talking, definitely visible. Now, let's look closer. I spot someone walking in the background, and there's another person further back near the cars … The answer is obviously B. Four. Confirmed. ❌

… The question states \"from the beginning\" to \"just before\" the headphones are fully on. I'll need to scrutinize the dialogue between 0:00 and roughly 0:10. Right before he's putting on the headphones at 00:10, he says, \"Yeah, I remember it. I remember it.\" Yes, at 00:09, he definitely says, \"I remember it,\" *twice*! … The answer is, without a doubt, A. Two. ❌

[Figure 107]

[Figure 108]

(c) Temporal Grounding (d) Temporal Localization

Figure 9. Qualitative examples of Gemini 2.5 Pro reasoning traces on AV-SpeakerBench. Green and red highlight colors indicate the model’s correct and incorrect reasoning, respectively. The figure above contains representative failure cases spanning four key error patterns: (a) cross-modality attribution, (b) audio and visual perception, (c) temporal grounding, and (d) temporal localization. Detailed analyses are provided in the subsection.

#### D. Question Examples

To make the task definitions concrete, we visualize representative multiple–choice question–answer pairs from AVSpeakerBench. Each example is shown as a panel containing (from top to bottom) a strip of video frames, the corresponding audio waveform, a natural–language question, and four answer options. Across all tasks, the model is evaluated by its accuracy in selecting the correct option, given the paired audio–visual clip.

Figure 11 groups the speaker-centric tasks: Speaker Detection, Speaker Recognition, and Speaker Counting. In these tasks, the evaluation focuses on whether the model can correctly associate speech segments with the corresponding visible speaker(s), distinguishing who is speaking, when they speak, and how many distinct speakers are active.

- Figure 12 illustrates speaker–visual tasks such as Ac-

tivity Recognition and visually grounded counting. Here, the model must reason jointly about a speaking person and their on-screen actions or surrounding context (e.g., who performs an action while or around speaking, or how often a visually specified person speaks), coupling appearance, motion, and speech.

- Figure 13 showcases speech reasoning tasks: Speech

Recognition, Speech Counting, and Speech Duration. The evaluation in these tasks requires recovering short spoken phrases, counting phrases, and compare the duration of speech.

Finally, Figure 14 presents speech attribute tasks, including and comparative speech questions on rate, pitch, and intensity. These examples emphasize fine-grained acoustic reasoning: the model must use the audio waveform to com-

Speaker-centric Visual-centric Audio-centric Task Type Detection Recognition Counting Attribute Recognition Activity Recognition Counting Recognition Duration Pitch Rate Intensity Counting Overall

Human Performance – 96.02 93.13 94.28 93.14 93.20 94.15 96.52 90.68 93.20 91.39 94.17 93.40 93.74 Gemini 2.5 Pro Thinking [8] – 81.73 74.15 74.13 72.55 73.30 62.93 77.11 78.81 67.48 69.86 71.84 63.89 73.04 Gemini 3 Pro Thinking – 85.95 79.86 73.13 80.39 79.13 71.71 87.06 75.85 70.87 78.47 75.24 70.14 77.62

Table 4. Evaluation Results of Gemini 3 Pro (Thinking) on AV-SpeakerBench.

Model (Size) 8f 16f 1fps Gemini 2.5P (-) 70.4 72.0 73.0 Phi4-MM (5.6B) 38.5 39.2 39.5 VITA 1.5 (7B) 36.5 36.8 36.3 Qwen2.5-O (7B) 41.0 41.3 42.3 Qwen3-O (30B) 51.3 53.1 54.1

Table 5. Performance on different sampling strategies.

pare how different people speak (e.g., how fast, how high or loud).

For compactness, each figure arranges three examples per row and one row per task group, while maintaining a consistent interface across all panels. Together, these four figures highlight the diversity of natural-language formulations and reasoning skills targeted by AV-SpeakerBench, spanning identity, actions, counting, appearance, speech, and attributes within a unified audio–visual QA framework.

#### E. Performance of Gemini 3 Pro on AVSpeakerBench

We evaluate the recently released Gemini 3 Pro (Thinking)1 and compare it with Gemini 2.5 Pro on AV-SpeakerBench.

As shown in Table 4, Gemini 3 Pro attains an average accuracy of 77.62%, outperforming Gemini 2.5 Pro (73.04%) by +4.6 points. The improvements are broad and most pronounced in tasks requiring fine-grained audio–visual grounding, including speaker-centric understanding, visual reasoning, speech recognition, and prosody-related perception. However, several categories—most notably counting and speech-attribute reasoning—remain far from solved, and a substantial gap persists relative to human performance (93.74%).

Overall, Gemini 3 Pro represents meaningful progress but still falls short of demonstrating robust and generalizable audiovisual reasoning.

#### F. Evaluation beyond default frame sampling.

We conducted additional experiments with models achieving non-trivial accuracy on AV-SpeakerBench under different frame sampling densities. Notably, the strongest models

1Released after the main paper deadline.

[Figure 109]

Figure 10. Performance under visual-only (V) and audiovisual (A+V) settings.

already adopt a similar default policy (1 fps with a frame cap). We report results with 8 and 16 frames since these 2 numbers are widely adopted in audiovisual large language model. Results show that these fixed number of frames consistently leads to slightly lower accuracy, while preserving the same model ranking (Table 5).

#### G. Audio necessity vs. vision-only solutions.

AV-SpeakerBench enforces fusion through question design, grounding speech to visible identities and temporal events. Human evaluation shows that the benchmark is largely not solvable from vision alone, while audiovisual input yields near-ceiling performance (Figure 10). In fact, we observe the same trend across all 12 tasks. Some items may still be inferred from natural visual conversational cues (e.g., expressions or gestures), but overall audiovisual fusion provides the most reliable and general solution. We further label questions that humans cannot answer under visiononly as an audio-required subset; Gemini 2.5 Pro achieves 72.7% on this subset versus 73.04% overall.

#### H. Ethical Statement

We define our dataset as a collection of publicly available YouTube videos paired with precise timestamps and human-written annotations. The benchmark is used solely for evaluation, not model training, and thus poses min-

imal risk of bias amplification or unintended memorization. The dataset will be released under the CC BY-NCSA 4.0 license, which restricts use to non-commercial research and prohibits applications involving facial recognition, surveillance, or biometric identification. Individuals featured in referenced videos may request removal, and we will promptly withdraw the corresponding segments from future releases.

Speaker Detection

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Q: Does the man wearing glasses speak before the man in the blue suit starts saying "Good, huh?"

Q: Does the man in the khaki coat speak after the man in the dark green sweater stands up?

Q: Does the woman with blonde hair speak in the video?

- A. No, he only listens to the argument between the other two men

- B. Yes, he refutes the request from the man in the dark green sweater

- C. Yes, he argues with the man in the red clothes

- D. No, he only yells at the other men

- A. No, she stands up and walks away

- B. No, she only nodes and smiles

- C. Yes, she is listing a few things she knows

- D. Yes, she acknowledges the man's point

- A. No, he only shows an awkward smile

- B. Yes, he gives in after refusing at first

- C. Yes, he chews and replies in an unfriendly tone

- D. No, he only looks at the other man

Speaker Recognition

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Q: Who speaks immediately before the woman in the black shirt says, "Tarzan. Well, you say because he just can't compete with me"?

Q: Who speaks right after the man in the red shirt and the man in the gray T-shirt do a fist bump?

Q: Who says, "Oh, I see what's going on"?

- A. The man in the blue shirt

- B. The man in the striped jacket

- C. The man in the white sweater

- D. The man in the black suit

- A. The man in a gray shirt says "great to meet you"

- B. The man wearing glasses says "what's your name?"

- C. The man wearing a backpack says "what's your name?"

- D. The man in red shirt says "It's never too late"

- A. The woman with blonde hair

- B. The man with blonde hair

- C. The man with black hair

- D. The woman with black hair

Speaker Counting

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Q: After the man in the red patterned shirt finishes saying "No actually, I think JD would be the one", and until the end of the video, how many unique people speak, out of all the people visibly present on screen, even partially?

Q: After the man without a beard closes the door, until the end of the video, how many unique people speak, out of all the people visible on screen, even partially?

Q: From the start of the video until just before the man in the checkered shirt starts saying “oh yeah wait”, how many unique people speak, out of all the people visibly present on screen, even partially?

- A. 2 out of 3

- B. 2 out of 2

- C. 3 out of 3

- D. 3 out of 4

- A. 5 out of 7

- B. 3 out of 5

- C. 4 out of 6

- D. 5 out of 6

- A. 2 out of 4
- B. 2 out of 6
- C. 3 out of 5
- D. 1 out of 4

Visual Reference Audio Reference

Figure 11. Visualization of speaker tasks. Top: Speaker Detection. Middle: Speaker Recognition. Bottom: Speaker Counting.

Activity Recognition

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Q: When does the woman in the black skirt say, "That would be so much fun"?

Q: What does the man with the mustache do right after he says, "The lord hates a coward"?

Q: What does the man in the green shirt do right before the man in the white t-shirt says, "Long time, no see"?

- A. After the man in the black shirt walks to the window

- B. After the woman in the black choker sits on the couch

- C. As the man in the black shirt points at the window

- D. As she walks across the room

- A. Brings his hands together in prayer

- B. Shakes the hand of the man next to him

- C. Taps the man next to him on his arm

- D. Puts a locket away in his pocket

- A. Touches the lips of the man in the white t-shirt

- B. Claps the hand of the man in the white t-shirt

- C. Pats the back of the man in the white t-shirt

- D. Hugs the man in the white t-shirt

Visual Counting

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Q: At the moment the man in a black suit sitting in a chair finishes saying, "This is what I meant," how many unique people are visible, even partially?

Q: After the woman with blonde hair says, "I'm a history major," until the end of the video, how many unique people are visible, even partially?

Q: From the start of the video until just before the man driving the car starts saying, "Uh, what's reverse?", how many unique people are visible, even partially?

- A. One

- B. Two

- C. Three

- D. Zero

- A. One

- B. Two

- C. Three

- D. Four

- A. Five
- B. Three
- C. One
- D. Four

Attribute Recognition

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Q: What does the person who speaks immediately before the person who says, "that was in the car" wear?

Q: What is the person wearing who speaks immediately after the line "I thought this was the bathroom" is spoken?

Q: What does the person who speaks immediately before the person who says, "that was in the car" wear?

- A. Blue tank top

- B. Gray turtleneck

- C. Black crop top

- D. Silver earrings

- A. Green dress

- B. Burgundy jacket

- C. Black suit

- D. Blue dress

- A. Green dress

- B. Burgundy jacket

- C. Black suit

- D. Blue dress

Visual Reference Audio Reference

Figure 12. Visualization of speaker-visual tasks. Top: Activity Recognition. Middle: Visual Counting. Bottom: Attribute Recognition.

Speech Counting

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Q: After the woman with white earrings turns away, until the end of the video, how many times is "never that name" mentioned?

Q: How many times the "15th anniversary" is mentioned by the man wearing blue button up?

Q: From the start of the video until just before the man with the grey scarf receives the money, how many times is "Britney Spears" mentioned?

- A. Zero
- B. One
- C. Two
- D. Three

- A. One

- B. Two

- C. Zero

- D. Three

- A. Zero

- B. Three

- C. Five

- D. Seven

Speech Duration

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Q: From the start of the video until just before a person holding a tray appears, who speaks the least among those who speak?

Q: After the smartphone screen appears, until the end of the video, who speaks the most among those who speak?

Q: After the person enters the door, until the end of the video, who speaks the most among those who speak?

- A. The person in a dark green T-shirt

- B. The person in a dark red T-shirt

- C. The person in a light yellow T-shirt

- D. The person in a light blue T-shirt

- A. The man with black hair

- B. The man with blonde hair

- C. The woman with black hair

- D. The woman with blonde hair

- A. The man in the green uniform

- B. The man in the white shirt

- C. The man in the black vest

- D. The man in the purple shirt

Speech Recognition

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Q: What does the man in the black suit say just after he stands up?

Q: What does the woman sitting on the porch swing say just before she drinks from her glass?

Q: What does the man in the white shirt say while pointing at the girl?

- A. "What do you have in your mouth?"

- B. "Qualification?"

- C. "Wait a moment."

- D. "Nothing, mayhem."

- A. "The voice just now."

- B. "We don't hear anything."

- C. "What is it?"

- D. "What is that?"

- A. "That's how it is."

- B. "Do you mean Jacquelin?"

- C. "You all want to play."

- D. "I have my eyes on you."

Visual Reference Audio Reference

Figure 13. Visualization of speech tasks. Top: Speech Counting. Middle: Speech Duration. Bottom: Speech Recognition.

Speech Intensity

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Q: Of those who speak, whose voice gets the loudest at any moment in the video?

Q: Among the people who speak, who speaks the most quietly overall?

Q: Among the people who speak, who speaks the most quietly overall?

- A. The man with gray shirt

- B. The woman with purple blazer

- C. The woman with pink dress

- D. The man with navy shirt

- A. The woman in the gray sweater

- B. The man in the dark brown shirt

- C. The man in the striped shirt

- D. The woman in the brown jacket

- A. The woman in green shirt

- B. The woman in purple sweater

- C. The man in blue gilet

- D. The man in black gilet

Speech Pitch

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

Q: Among those who speak, who has the highest pitch?

Q: Among those who speak, who has the lowest pitch?

Q: Among those who speak, who has the lowest pitch?

- A. The man in glasses

- B. The woman with gray hair

- C. The woman with blonde hair

- D. The man with gray hair

- A. The man with no hair

- B. The man with pink hair

- C. The woman in brown shoes

- D. The woman in black shoes

- A. The man with a mustache

- B. The man with gray hair

- C. The man in a green t-shirt

- D. The man in a green sweater

Speech Rate

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Q: Of those who speak, who has the lowest rate of speech?

Q: Of those who speak, who has the lowest rate of speech?

Q: Of those who speak, who has the highest rate of speech?

- A. The woman in the black top

- B. The woman in the pink top

- C. The woman in the blonde top

- D. The woman in the blue top

- A. The woman in black dress

- B. The woman in golden dress

- C. The man in black shirt

- D. The man in white shirt

- A. The man in brown jacket

- B. The man in black jacket

- C. The woman in black jacket

- D. The woman in green shirt

Visual Reference Audio Reference

Figure 14. Visualization of speech attribute tasks. Top: Speech Intensity. Middle: Speech Pitch. Bottom: Speech Rate.

Table 6. Task-specific goals and requirements for all annotation tasks.

Task Name Requirement Speaker Detection Goal: Test whether models can detect whether a person is currently speaking among multiple

visible people.

- • Select a 5–30s clip with at least two visible individuals, where only one person is clearly speaking during a short moment.
- • Choose a target person (e.g, the man in black suit), and a short utterance as an audio anchor or a short event as a visual anchor.
- • Formulate a question such as: “Does the [person] speak” or “Does the [person] speak before/after/when [utterance/event] occurs?”
- • Provide four answer options, with 2 No and 2 Yes. For each No option, put a visual reference (e.g, No, he just stares at the opposite person). For each Yes option, put an audio reference (e.g, Yes, he talks about his plan).
- • If there is not enough audio/visual reference, please create plausible distractors that fit the scene context.

Speaker Recognition Goal: Evaluate whether models can link an utterance to the correct visible speaker.

- • Select a 5–30s clip with multiple visible speakers.
- • Choose a short utterance as an audio anchor or a short event as a visual anchor.
- • Formulate a question using: “Who speaks right after/before/when [person] says [utterance]?”, “Who responds to [person] saying [utterance]?”, or “Who says [utterance]?”
- • Provide four answer options referring to different individuals (e.g., “the man in the black suit”, “the woman in the red dress”).
- • If fewer than three people appear, create plausible distractors that fit the scene.

Speaker Counting Goal: Assess whether models can count how many distinct people speak in a segment.

- • Select a 10–30s clip with multiple visible people and several distinct spoken turns.
- • Choose a short utterance as an audio anchor or a short event as a visual anchor.
- • Formulate a question such as: “From the start of the video until [utterance/event], how many people speak, out of the people in the video?” or “After [utterance/event] until the end of the video, how many people speak, out of the people in the video?”
- • Provide four numerical answer options in the format “n out of m” (e.g., “1 out of 2”, “2 out of 3”); the first number must be the true count of speakers in the specified interval, and the second number must be the number of visible people in that segment.
- • Ensure that speakers are visually distinguishable and separated by clear turn-taking; create plausible nearby counts as distractors.

Attribute Recognition Goal: Test whether models can connect a speaker’s appearance to their speech right before or after an anchor.

- • Select a 10–30s clip with multiple visible speakers.
- • Choose a short utterance as an audio anchor.
- • Formulate a question such as: “What is the appearance of the person who says [immediately before/after/when] the person says [utterance]?” or “What does the person who says [utterance] wear?”
- • Ask about attributes like clothes, clothing shape, clothing color, hair, hairstyle, or hair color.
- • Provide four answer options describing different plausible appearances; exactly one must match the correct speaker.
- • If fewer than four distinct appearances are visible, create plausible distractors consistent with the scene.

Activity Recognition Goal: Measure whether models can associate a speaker with actions that happen before/after/while they speak.

- • Select a 10–30s clip with multiple visible speakers and at least one clear action (e.g., standing up, waving, pointing).
- • Choose a short utterance as an audio anchor.
- • Formulate a question such as: “What does the person do [immediately before/after/when] the person says [utterance]?” or “When does the person say [utterance]?”
- • For the first question type, encode the activity (e.g., “stands up”, “raises a hand”) into the four answer choices.
- • For the second question type, encode [before/after/when] + [activity] inside the four answer choices.
- • If fewer than three distinct activities appear, create plausible distractors consistent with the scene.

Visual Counting Goal: Assess whether models can count visible entities conditioned on an audio anchor.

- • Select a 10–30s clip with multiple visible people.
- • Choose a short utterance as the reference time.
- • Formulate a question such as: “After/Before/When the [utterance] occurs, how many people are visible, even partly?”
- • Provide four numerical answer options (e.g., 1, 2, 3, 4) or phrases that clearly encode the count.
- • Ensure all counted entities are clearly visible in the frame; create plausible distractors using nearby counts.

Speech Recognition

- • Goal: Evaluate whether models can recognize the spoken content in the clip.
- • Select a 10–30s clip with multiple visible speakers.
- • Choose a short visual event as an anchor.
- • Formulate a question such as: “What does the person say [just before/after/when] the [event] occurs?”
- • Provide four answer options corresponding to different possible transcripts; exactly one must match the spoken phrase (allow light paraphrasing only for the correct option).
- • Ensure distractor phrases are grammatical and plausible in the context of the video.

Speech Duration Goal: Check whether models can reason about relative speaking time.

- • Select a 5–20s clip where at least two people each speak for clearly different durations.
- • Choose an event as a visual anchor.
- • Formulate a question such as: “Of those who speak, who speaks the most/least?”, “From the start of the video, until [event], who speaks the most/least?”, or “After [event], until the end of the video, who speaks the most/least?”
- • Provide four answer options referring to different individuals (e.g., “the man in the black suit”, “the woman in the red dress”).
- • Ensure the duration difference is perceptible (at least 1–2s); construct plausible but incorrect visual options as distractors.

Speech Pitch Goal: Test whether models can distinguish speakers by relative pitch.

- • Select a 5-10s clip where at least two speakers have clearly different pitch ranges.
- • Formulate a question such as: “Among those who speak, who has the highest/lowest pitch?”
- • Provide four answer options referring to different individuals in the scene (e.g., “the man in the black suit”, “the woman in the red dress”).
- • If fewer than three speakers are present, create plausible distractors based on other visible characters that could plausibly speak.

Speech Rate Goal: Measure whether models can compare how fast different speakers talk.

- • Select a 5–10s clip with at least two speakers whose speaking rates differ noticeably.
- • Formulate a question such as: “Among those who speak, who has the highest/lowest rate of speech?”
- • Provide four answer options referring to different speakers (e.g., by clothing or position).
- • Ensure the difference in speaking rate is clear (e.g., more syllables in similar time), and create plausible distractors when needed.

Speech Intensity Goal: Assess whether models can reason about speech loudness.

- • Select a 5–10s clip where at least two speakers talk with clearly different loudness levels (e.g., one shouting, one speaking softly).
- • Formulate a question such as: “Among those who speak, who has the highest/lowest voice?”
- • Provide four answer options referring to different individuals (e.g., “the man in the black suit”, “the woman in the red dress”).
- • Avoid clips where microphone distance alone explains loudness, unless this is visually clear; create plausible distractors as needed.

Speech Counting Goal: Evaluate whether models can count number of phrases or keywrods.

- • Select a 5–30s clip with multiple short utterances or turns of speech.
- • Choose an event as a visual anchor.
- • Formulate a question such as: “From the start of the video, until just before [event], how many times [phrase] is mentioned by [person/everyone]?” or “After [event], until the end of the video, how many times [phrase] is mentioned by [person/everyone]?”
- • Provide four numerical answer options; exactly one must match the true count.
- • Ensure utterances are separated enough to be countable (clear pauses or turn-taking), and use nearby counts (e.g., n − 1, n, n + 1, n + 2) as distractors.

