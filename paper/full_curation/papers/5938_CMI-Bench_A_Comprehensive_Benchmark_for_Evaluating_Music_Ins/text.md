# CMI-BENCH: A COMPREHENSIVE BENCHMARK FOR EVALUATING MUSIC INSTRUCTION FOLLOWING

Yinghao Ma1 Siyou Li1 Juntao Yu1 Emmanouil Benetos1 Akira Maezawa2 1 Queen Mary University of London, London, UK 2 Yamaha Corporation, Hamamatsu, Japan

emmanouil.benetos@qmul.ac.uk, akira.maezawa@music.yamaha.com

arXiv:2506.12285v2[eess.AS]27Jun2025

## ABSTRACT

Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multichoice evaluations that fail to reflect the complexity of realworld music analysis. We reinterpret a broad range of traditional MIR annotations as instruction-following formats and introduce CMI-Bench, a comprehensive music instruction following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. These include genre classification, emotion regression, emotion tagging, instrument classification, pitch estimation, key detection, lyrics transcription, melody extraction, vocal technique recognition, instrument performance technique detection, music tagging, music captioning, and (down)beat tracking — reflecting core challenges in MIR research. Unlike previous benchmarks, CMI-Bench adopts standardized evaluation metrics consistent with previous state-of-the-art MIR models, ensuring direct comparability with supervised approaches. We provide an evaluation toolkit supporting all open-source audio-textual LLMs, including LTU, Qwenaudio, SALMONN, MusiLingo, etc. Experiment results reveal significant performance gaps between LLMs and supervised models, along with their culture, chronological and gender bias, highlighting the potential and limitations of current models in addressing MIR tasks. CMI-Bench establishes a unified foundation for evaluating music instruction following, driving progress in music-aware LLMs.

1. INTRODUCTION The emergence of large language models (LLMs) has reshaped the landscape of natural language processing by enabling general-purpose models to solve a wide variety of tasks through instruction following. This paradigm—where models are trained not just on pre-text corpora but instruction-response pairs—has unlocked new possibilities in model generalization, few-shot learning, and cross-domain reasoning. Supervised fine-tuning (SfT), also

© Y. Ma, S. Li, J. Yu, E. Benetos, and A. Maezawa. Licensed under a Creative Commons Attribution 4.0 International License (CC BY

- 4.0). Attribution: Y. Ma, S. Li, J. Yu, E. Benetos, and A. Maezawa, “CMIBench: A Comprehensive Benchmark for Evaluating Music Instruction Following”, in Proc. of the 26th Int. Society for Music Information Retrieval Conf., Daejeon, South Korea, 2025.

known as instruction finetuning, and reinforcement learning from human feedback have further strengthened LLMs’ ability to align with human intent [1].

In the context of music, the instruction-following paradigm holds particular promise. Many music-related tasks are naturally multimodal and domain-specific and often lack large-scale annotated data. Instruction-tuned models can generalize to previously unseen problems such as chord generation under rhythmic constraints or personalized music recommendation based on context. Besides, by supporting in-context learning, LLMs offer a flexible path to interact with world music traditions, rare genres, and diverse user preferences—all without explicit retraining [2].

Recently, a growing number of audio LLMs [3–6], extended LLMs with audio encoders and instruction-following capabilities. However, these models have so far been evaluated on limited tasks, relying on caption similarity metrics on datasets like [3, 4], single-choice protocols [7–9], or multiple-choice question (MCQ) protocols [10] Despite these successes, such evaluations fail to capture the complexity of core music information retrieval (MIR) tasks and offer limited insight to real-world performance.

This work makes three key contributions: First, we reinterpret a broad range of core MIR annotations as instructionfollowing tasks as illustrated in Figure 5, enabling the use of a wide range of MIR datasets, including sequential tasks, not only for evaluation but for training and SFT audio-text LLMs. Second, we provide a standardized benchmarking framework that includes implementations of major opensource audio-text LLMs, along with evaluation metrics aligned with prior MIR literature. Unlike earlier MCQ protocols, CMI-bench adopts open-ended, task-specific metrics, allowing more rigorous comparisons. Last, we present an initial analysis of generalization outside training data, along with cultural and gender bias across models, uncovering potential limitations in their generalization and pointing future directions for culturally inclusive music AI. Together, these contributions lay groundwork for systematic progress in music instruction following and its intersection with traditional MIR. The code 1 and testset audio 2 are available.

This paper is organized as follows. Section 2 reviews related work. Section 3 introduces the CMI-Bench design and instruction construction procedure, Section 4 describes the

- 1 https://github.com/nicolaus625/CMI-bench/
- 2 https://huggingface.co/datasets/nicolaus625/

CMI-bench

Question: Estimate the arousal score of the given music on a scale from 1 to 9, where 1 represents the lowest arousal level (calm, relaxing) and 9 represents the highest arousal level (energetic, intense). Provide a numerical estimate based on your perception of the music.

- Example 1, when the music is slow and gentle, suggesting a low arousal level:\nEstimated score: 2.
- Example 2, when the music is energetic and driving, indicating a high arousal level:\nEstimated score: 8.
- Example 3, when the music has a balanced tempo and moderate intensity:\nEstimated score: 5.

Based on the description provided, I would estimate the music's arousal score as 7. The key of C minor typically carries a somber or melancholic connotation, which combined with a brisk tempo and a 4/4 time signature suggests an energetic but perhaps slightly melancholic feel. However, the addition of a jazzy element and complex chord progression might add some complexity and vibrancy to the overall mood, making it more dynamic and potentially uplifting.

#### Question: Please provide the instrument tag(s) of given audio.

The instrument tags for this audio are 'acoustic guitar' and 'lute'.

Figure 1. Instruction-Following Format Data in CMIBench and example response from Qwen2-audio.

experimental setup, including evaluation protocol and opensource models. Section 5 presents benchmarking results. Finally, section 6 draw the conclusion.

2. RELATED WORK

- 2.1 Instruction Following Dataset

Instruction following refers to the ability of LLMs to perform tasks based on natural language prompts that describe the task itself. This paradigm has become central to recent advances in NLP, where SFT models are trained on a wide range of instruction-response pairs. SuperNaturalInstructions [11] rewrote annotations from over 1,600 diverse NLP tasks into instruction-following formats, showing that models can generalize to unseen tasks given clear instructions. Self-Instruct [12] further advanced this approach by automatically generating diverse instruction–response pairs using the model’s own outputs, while Instructional-GPT [13] aligned models with human intent through SFT and reinforcement learning.

These techniques have recently been extended to music, enabling instruction-following models to engage with multimodal and domain-specific tasks. MusicQA [3] and MusicInstruct [4] repurpose descriptions and tags from MIR dataset to generate Q&A pairs. Such a dataset does not distinguish subtask on instrument, emotion, genre, and caption Q&A pairs, and the evaluating metrics are BERT-score, overestimate model’s music understanding capability without equally compared with traditional MIR algorithms. Finally, Audio-FLAN [14] presents a large-scale instructiontuning corpus across 80 tasks, unifying understanding and generation in audio, music, and speech. Yet, many tasks are paraphrased to an MCQ format, significantly smaller than the range of pre-defined classes or labels. Furthermore, these works do not provide a model performance benchmark on such tasks, and the evaluation metrics are not compatible with traditional MIR method.

- 2.2 Instruction-Following Benchmarks for Music

While instruction-following has shown great promise in natural language and vision tasks, its application in the music domain remains underexplored. ZIQI-Eval [15] is an

instruction following the benchmark on textual symbolic music. AIR-Bench [7] covers a broader range of audio types, including music, but emphasizes low-level tasks such as pitch and instrument recognition and relies primarily on MCQ formats. MMAU [10] includes music reasoning, yet only covers six MIR datasets, lacks alignment with MIRspecific evaluation metrics, and reports only average scores across tasks. MuCho Music [8] evaluates music understanding in multimodal models through 1,187 MCQs. AudioBench [16] and MusicBench [17] primarily target audio and text-to-music generation respectively, without addressing MIR tasks. MuChin [18], while valuable for colloquial descriptions, is tailored to Chinese pop song generation. Across these efforts, most benchmarks omit key MIR tasks popularized by MIREX , few support sequential tasks, and evaluation protocols often rely on multiple-choice questions rather than the task-specific metrics used in supervised MIR literature.

## 2.3 Audio-Textual Large Language Models

Current audio-textual LLMs typically consist of an encoder of speech, audio or music, an intermediate architecture and an LLM backbone. LTU [19] and LTU-AS [20] focus on general audio comprehension and reasoning, combining whisper speech encoder [21] w. MU-LLaMA [3], MusiLingo [4] and Llark [22] are tailored for music-related tasks, leveraging audio encoders and instructional datasets to support captioning and open-ended music question answering. Pengi [23] frames all audio tasks as text generation, unifying audio perception with LLM-based reasoning via a simple prefix-tuning strategy. GAMA [24] and GAMA-IT [24] integrate multi-layer audio features and instruction tuning (CompA-R) to support complex reasoning over general audio, including music. SALMONN-Audio [6] introduces a Q-former window architecture for sequential speech and sound understanding. Qwen-Audio [5] and Qwen2-Audio [25] scale instruction tuning across over 30 audio tasks with hierarchical or natural prompts. AudioFlamingo [26] and Audio-Flamingo2 [27] incorporate incontext learning and retrieval-based adaptation for audiotext interaction and dialogue. Beyond open-source models, proprietary systems Gemini-2.5 Pro and GPT-4o may represent the state-of-the-art (SOTA).

## 3. CMI-BENCHMARK

With CMI-Bench, we aim to address the following limitations in evaluating music understanding capabilities of audio-text LLMs. Previous benchmarks often cover only a narrow range of tasks, and no benchmark supports sequential tasks, overlooking many classic challenges which are central to MIR research. Moreover, evaluation protocols are typically inconsistent with standard MIR metrics, difficult to compare against traditional supervised models. To address these issues, we reformulate annotations from widely-used MIR datasets into instruction-following prompts and process model outputs into formats compatible with standard MIR Python library mir_eval [28].

## 3.1 Overview

|Tasks<br><br>|Dataset Metrics #Test Samples|
|---|---|
|Key detection|GS [29] Gmean score 2406<br><br>|
|Emotion Regression|EMO [30] R2 125<br><br>|
|Music tagging|MagnaTagATune [31] ROC-AUC, PR-AUC 5329<br><br>MTG-Top50 [32] ROC-AUC, PR-AUC 11356|
|Instrument Classification<br><br>|MTG-Instrument [32] ROC-AUC, PR-AUC 5115 Nsynth-Instrument [33] Accuracy 4096|
|Genre classification<br><br>|MTG-Genre [32] ROC-AUC, PR-AUC 11479 GTZAN [34] Accuracy 290|
|Emotion tagging|MTG-Emotion [32] ROC-AUC, PR-AUC 4231|
|Pitch Estimation|Nsynth-Pitch [33] Accuracy 4096<br><br>|
|Singing Techniques|VocalSet [35] Accuracy 1140<br><br>|
|Music Captioning<br><br>|SDD [36] BL., ME., RO., Bert-Score 1106 MusicCaps [37] BL., ME., RO., Bert-Score 2813|
|Lyrics Transcription<br><br>|DSing [38] WER, CER 482|
|Beat tracking<br><br>|GTZAN-Rhythm [34] F_measure 290 ballroom [39,40] F_measure 685|
|DownBeat tracking<br><br>|GTZAN-Rhythm F_measure 290 ballroom F_measure 685|
|Melody Extraction|MedleyDB v2 [41] Melody Accuracy 618<br><br>|
|Performance Technique<br><br>|GuZheng_99 [42] frame-level micro/macro-f1 94|

Table 1. Overview of tasks, datasets, evaluation metrics, and the number of test samples in the CMI-Bench.

The CMI-Benchmark encompasses 14 tasks spanning multi-class, multi-label, regression, captioning, and sequential prediction challenges, evaluated across 20 diverse datasets. This benchmark integrates traditional MIR tasks with emerging music-and-language objectives, providing a robust platform to assess computational music intelligence. The tasks and datasets used in the benchmark are shown in

- Table 1. By standardizing splits and metrics, CMI-bench ensures reproducibility and fair comparisons.

## 3.2 Self-Instruction of MIR Annotations

In this subsection, we introduce the self-instruction framework for CMI-Bench designed to unify diverse MIR tasks under a consistent NLP paradigm, outlining the design of instructions and input tailored to tasks such as key estimation, genre classification, emotion regression, instrument tagging, and temporal sequence annotations. Our approach leverages structured prompts with multi-class, regression, and sequence-based outputs, enriched with few-shot examples to guide annotation generation.

For multi-label tasks, we allow flexible outputs without providing pre-defined tags, reflecting real-world complexity. For clip-level multi-class tasks with a manageable number of categories, such as musical key estimation and genre and vocal techniques classification, instructions explicitly list all possible choices. For instance, key estimation requires selecting one of 24 major and minor keys , with few-shot examples like "Bb major" to clarify the format. In cases with larger class sets, such as pitch classification on short excerpts across MIDI numbers 9 to 119, we provide a definition of MIDI standard alongside examples (e.g., "A4: 69", "Middle C (C4) = 60") to anchor the task.

Regression tasks, such as arousal estimation, adopt a numerical scale (1 to 9) with descriptive anchors—1 for "calm, relaxing" and 9 for "energetic, intense." To better utilize LLM-s in-context learning capability, we include examples for few-shot learning on tie scores to musical characteristics (e.g., "slow and gentle: 2," "energetic and driving: 8"), enabling precise emotional annotation.

Temporal tasks, such as beat tracking and instrument performance technique detection, require structured outputs. Beat tracking outputs timestamps in a comma-separated format (e.g., "0.1s, 1.19s, 2.25s"), while Guzheng (traditional Chinese Kyoto) technique detection uses a Python-style list of tuples (e.g., "[(’70.8086’, ’71.4817’, ’Tremolo’)]"), covering techniques like Vibrato and Glissando. Default outputs "[(’0.0’, ’10.0’, ’No Tech’)]" handle cases with no detections. Melody extraction follows similar principles, balancing specificity and clarity. We forbid tuples to have time overlapping on melody and (down)beat tracking, but allow for playing technique detections.

Inputs are uniformly represented as audio placeholders (e.g., "<|SOA|><AUDIO><|EOA|>"), paired with metadata such as audio paths and time segments. This ensures compatibility with NLP models while preserving MIR task diversity, offering a scalable framework for future efforts.

## 4. EXPERIMENTS 4.1 Evaluation Protocol

To enable rigorous and fair comparison with traditional MIR systems, we design an evaluation pipeline that closely follows the original task definitions and metrics. All model outputs are automatically post-processed to conform to each task’s expected format, ensuring compatibility with MIR evaluation tools such as mir_eval. Below, we detail the evaluation strategies used for each task category.

- 4.1.1 Classification Tasks

Multi-Class Classification. Tasks include short-clip monophonic pitch estimation, instrument classification, singing technique classification, and genre classification. We evaluate using strict string matching: a model’s response is considered correct if it contains only the correct label (case, space-, and punctuation-insensitive) and no others. For pitch classification, we additionally require the model to follow the instruction format and return MIDI numbers. Accuracy is used as the metric.

Multi-Label Classification Tasks include music tagging, genre labelling, emotion tagging, and instrument recognition. As model responses may include synonyms or freeform text, we embed both the predicted and ground truth tag sets using the BGE encoder [43], a model optimized for retrieval and multi-label matching. Cosine similarity scores are then used to compute ROC and PR, providing a soft but semantically aligned evaluation quality.

- 4.1.2 Clip-level MIR Tasks

Key Detection. We adopt standard weighted score metric from mir_eval.key, which accounts for musically reasonable errors, such as relative minor or parallel key.

Regression. Model outputs are constrained to integers in the range [1, 9]; if a float is returned, we take the floor. Outputs are then z-score normalized to zero mean and unit variance. If a model fails to return a value, we assign the model’s mean value. The coefficient of determination (R2)

is computed between predictions and annotations on arousal and valance.

Music Captioning. We assess caption quality using four standard NLP metrics: BLEU [44,45], METEOR [46], ROUGE [47], and Bert-Score [48].

- 4.1.3 Sequential MIR Tasks

Lyrics Transcription. We extract lyrics from model outputs by removing typical prefixes (e.g., “lyrics is as follows:”). Word Error Rate (WER) and Character Error Rate (CER) are computed against ground-truth lyrics.

(Down)Beat Tracking. Model are expected to return a list of time points for (down)beat events. We filter nonnumeric outputs, sort the list by time, and apply F-measure metric from mir_eval.beat, with a 20ms tolerance.

Melody Extraction is treated as a sequential regression task on the fundamental frequency of notes calculated by mir_eval.melody.evaluate with 50 music cents tolerance. Models are instructed to return a list of (time, pitch) tuples. We discard invalid tuples (e.g., missing pitches, or improperly formatted entries, etc.). If multiple pitches are predicted for the same timestamp, we use only the first. Evaluation is based on frame-level accuracy.

Instrument Playing Technique Detection. For the GuZheng_99 dataset, we evaluate frame-level predictions using macro- and micro-F1 scores, allowing for overlapping techniques. Invalid predictions (e.g., incorrect tuple formats) are filtered out. Empty responses are interpreted as a "no technique" prediction covering the full time range.

## 4.2 Models

Model #Params Sound Music Speech Encoder Architecture Decoder

Pengi [23] 323M ✓ ✓ Audio-Flamingo [26] 2.2B ✓ ✓ LTU [19] 7B ✓ ✓ LTU-AS [20] 7B ✓ ✓ ✓ MusiLingo-long [4] 7B ✓ MuLLaMa [3] 7B ✓ GAMA [24] 7B ✓ ✓ GAMA-IT [24] 7B ✓ ✓ Qwen-Audio-Chat [5] 8.4B ✓ Qwen2-Audio-Instruct [25] 8.4B ✓ ✓ ✓ SALAMONN-Audio [6] 13B ✓ ✓ ✓

Table 2. Comparison of audio-textual LLMs by training domains.✓denotes coverage or presence; ✗absence.

To provide a broad and representative evaluation, we implement and benchmark 11 audio-text LLMs with publicly available weights demonstrated in table 2. Our selection covers a wide spectrum of model designs and training corpus, enabling a comprehensive comparison of instructionfollowing capabilities across various music-specific tasks.

5. RESULTS AND DISCUSSION

## 5.1 Benchmarking Results

Experiment results reveal several important observations about the current state of audio-text LLMs on MIR tasks.

- 5.1.1 LLMs Underperform Traditional MIR Baselines.

Despite LLMs have achieved excellent results on music captions and multi-choices QA, [4,8–10,37], all models in our study fall significantly short of the performance achieved by task-specific supervised systems when evaluated using standard MIR metrics besides music captioning. This is consistent across classification, regression, and sequential tasks. These findings suggest that instruction-following LLMs still lack the specialized precision and inductive bias of MIR models trained explicitly for each task.

- 5.1.2 Best Performance May Skew toward Training Set

Interestingly, the peak performance on each task is typically achieved by models whose datasets overlap significantly with their training corpus, revealing limited generalization. Qwen2-Audio performs best on MTG-Jamendorelated tasks such as MTG-top50, MTG-Emotion, and SDD captioning, while common on other tagging and caption datasets. This aligns with its use of MTG-Jamendo and FMA during model development via AIR-Bench, suggesting unsatisfying generalization capability. Besides, MusiLingo performs best on MusicCaps, the same dataset it was trained on for captioning and Q&A. Lastly, GAMA shows the best on MTT and NSynth-instrument and comparative on MusicCaps, while common on other datasets on same tasks, reflecting bias in its SFT corpus. These demonstrate that supervised instruction-tuned models can capture task-specific patterns well when training data is directly aligned, but their generalization to unseen or structurally different tasks remains limited.

- 5.1.3 All Models Perform Poorly on DSing Transcription

Despite the absence of instrumental accompaniment and use of English lyrics, none of the models reach usable performance levels on DSing for lyrics transcription, though it is relatively clean. This result is particularly striking for models like LTU and SALMONN, which include Whisper as their audio encoder and could theoretically benefit from ASR capabilities. LyricWhiz [64] utilizes GPT-4 to postprocess whisper ASR output on DSing dataset, providing results similar to SOTA without training.

- 5.1.4 Prompting Format May Impacts Performance.

Prompting without task-specific tokens used during training significantly degrades performance. Qwen-Audio performs far worse on Nsynth-Pitch than reported in its original paper. This is likely due to the absence of structured task tokens (e.g., “<|pitch|><|midi_pitch|>piano”) in our prompt. Instead, CMI-bench relies on general natural language instructions. This highlights a critical gap in current audio LLMs: without clearly defined prompting schemas, their ability to interpret instructions can be fragile and fail to generalize. While different prompts for MusiLingo do not provide a significant difference on MusicCaps.

- 5.1.5 Sequential Tasks Remain Challenging for All.

Tasks involving structured sequence-based outputs—such as melody extraction, instrument performance technique

Qw2. Qw. Salm. MusiL. LTU LTU-AS MU-L. auFla. Gama GamaI Pengi SOTA GS-K GES ↑ 8.28 6.51 7.70 9.50 7.61 1.42 7.56 8.21 7.69 7.70 0.00 74.3 [49] EMO

aR2 ↑ -0.75 -0.44 -0.51 -0.68 -1.14 -1.27 -0.03 -0.85 -1.08 -0.29 0.00 0.62 [50] vR2 ↑ -0.84 -0.78 0.0 -0.60 -1.13 -0.78 -0.12 -0.60 -1.30 -1.19 0.00 0.76 [51]

ROC ↑ 66.78 66.00 59.07 63.39 65.75 65.83 68.32 68.68 81.21 78.32 66.75 92.0 [52]

MTT

PR ↑ 19.15 16.99 15.08 12.25 17.78 15.72 18.65 20.16 34.26 27.53 17.82 41.4 [50] M-G

ROC ↑ 64.44 66.39 57.71 57.48 52.22 57.14 57.36 62.83 52.50 62.49 58.23 88.0 [51]

PR ↑ 9.23 8.07 5.62 4.99 3.62 4.98 4.97 6.85 3.90 6.01 5.47 20.5 [51] M-E

ROC ↑ 60.89 59.06 50.69 53.07 51.41 52.02 54.40 55.80 51.97 58.84 53.88 78.6 [53]

PR ↑ 7.85 6.09 3.65 3.95 3.98 3.72 4.35 4.60 4.07 5.27 3.93 16.1 [53] M-I

ROC ↑ 58.90 56.95 48.78 55.63 55.34 53.02 50.81 56.99 51.15 55.16 56.09 78.8 [54]

PR ↑ 12.41 11.35 7.44 9.24 10.98 8.90 8.24 10.71 9.01 10.69 9.36 22.0 [51] M-50

ROC ↑ 64.64 63.00 53.46 57.58 53.86 54.11 54.88 60.96 52.01 60.68 57.22 84.3 [53] PR ↑ 16.54 14.45 9.49 9.68 8.30 8.67 9.11 12.16 8.10 11.72 10.19 32.1 [53]

|GTZ.<br><br>|Acc. ↑<br><br>|72.07 71.38 32.76 7.24 2.76 16.90 8.97 50.34 21.38 42.41 6.21|83.9 [55]|
|---|---|---|---|
|VS-T<br><br>|Acc. ↑|14.91 15.18 15.61 1.23 7.11 0.53 4.56 11.32 7.72 7.89 0.00<br><br>|76.9 [56]|
|NI|Acc. ↑<br><br>|37.62 4.13 0.15 0.00 0.49 6.88 0.00 15.80 58.37 39.36 42.26<br><br>|78.2 [57]]|
|NP<br><br>|Acc. ↑|1.51 0.37 0.00 0.00 0.73 0.05 0.00 0.73 0.20 0.00 5.74<br><br>|89.2 [53]|

BL. ↑ 23.40 11.95 16.41 8.14 11.54 9.72 15.55 15.14 15.96 20.93 15.47 -

ME. ↑ 23.21 9.35 18.45 14.32 8.51 7.49 13.89 11.81 13.81 16.41 9.98 16.7 [58]

SDD

RO. ↑ 28.47 12.35 28.12 30.15 9.33 9.42 15.28 12.92 18.35 20.07 11.45 111.9 [58]

BS. ↑ 87.44 84.79 86.68 85.28 84.44 83.62 86.38 85.75 85.89 86.21 82.90 86.0 [58]

BL. ↑ 14.76 2.98 1.23 21.50 5.24 4.22 3.48 2.25 7.57 14.53 16.52 21.7 [4]

ME. ↑ 12.47 5.55 4.60 22.49 8.55 7.01 8.01 5.97 10.07 10.98 14.77 22.4 [58]

MC

RO. ↑ 12.35 6.68 6.26 30.29 9.39 7.51 8.58 6.94 11.38 12.46 12.64 30.8 [4]

BS. ↑ 84.38 82.37 82.98 85.75 83.84 83.59 83.00 83.43 84.30 84.57 83.22 87.8 [58] DS

WE. ↓ 793.0 115.7 816.1 2019 235.5 191.7 191.9 275.7 225.4 152.6 343.2 12.99 [59]

CE. ↓ 818.6 96.2 760.00 2311 210.8 185.5 168.3 262.6 201.3 165.2 368.0 -

G-B FM. ↑ 7.50 23.69 11.49 0.04 0.10 0.00 0.71 3.96 0.00 1.49 0.00 88.3 [56] G-D FM. ↑ 5.97 10.21 8.62 0.18 0.86 0.00 0.17 3.06 0.05 0.54 0.00 54.1 [60] BR-B FM. ↑ 7.12 21.96 14.97 0.01 0.15 0.00 0.22 4.69 0.02 1.02 0.00 96.8 [61] BR-D FM. ↑ 5.69 10.68 9.40 0.06 2.29 0.00 0.15 3.47 0.14 0.68 0.00 94.1 [61]

|MDB|Acc. ↑|5.06 0.08 0.00 0.00 0.00 0.00 0.00 0.01 0.66 0.00 0.00<br><br>|72.3 [62]|
|---|---|---|---|
|GZ<br><br>|maF1 ↑ miF1 ↑|3.18 1.66 0.03 0.00 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.89 0.44 0.01 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00<br><br>|90.0 [63] 80.4 [63]|

- Table 3. Performance of 11 open-source audio-text LLMs on CMI-Bench. Models: Qwen2-Audio (Qw2.), Qwen-Audio (Qw.), SALMONN-Audio (Salm.), MusiLingo (MusiL.), LTU, LTU-AS, MU-LLaMA (MU-L.), Audio-Flamingo (auFla.), GAMA, GAMA-IT (GamaI), Pengi. Tasks include key detection (GS-K), emotion regression (EMO), tagging (MTT, M-50), genre (M-G, GTZ.), emotion/instrument tagging (M-E, M-I), captioning (SDD, MC), lyrics transcription (DS), beat/downbeat tracking (G-B/G-D, BR-B/BR-D), melody (MDB), and Guzheng techniques (GZ). Metrics: GES, R², ROC-AUC, PR-AUC, Accuracy, BLEU (BL.), METEOR (ME.), ROUGE (RO.), BERTScore (BS.), WER/ CER, FM(F-Measure), Macro-F1 (maF1), Micro-F1 (miF1). Best scores are in bold.

detection, and (down)beat tracking—are poorly handled by all evaluated models. Even Qwen-Audio, which shows relatively strong performance in genre and beat tracking, falls far short of MIR baselines, sometimes copying the input examples. We hypothesize two key reasons: For one thing, the diversity and ambiguity in how sequence tasks are phrased (e.g., timestamps, tuple formats) reduces consistency in model outputs. For another, many models have only limited exposure to audio tasks with dense temporal supervision. If pretraining data includes timestamped output and matched decoding formats, performance may improve.

- 5.1.6 Emotion Regression Fails for All Models.

Despite clear instructions, carefully designed scales (1–9), and contextual music descriptions, and few-shot examples, all models fail to provide usable predictions for

arousal and valence. In fact, model outputs often cluster around meaningless values, sometimes performing worse than simply predicting the mean. Our post-processing rules convert empty or invalid outputs to dataset means, which often lead to better R² scores than the models themselves—highlighting the severe limitations in mapping continuous perceptual attributes from music using current audio-text LLMs.

These findings emphasize the gap between current SFT multimodal LLMs and traditional task-specific MIR systems. While open-source audio LLMs show promise in isolated tasks with aligned training data, substantial challenges remain in terms of generalization, structured output generation, and adaptation to real-world settings.

## 5.2 Culture and Gender Bias

We further analyze the performance of two top-performing models—Qwen2-Audio and Audio-Flamingo—on finegrained instrument, genre, and music tag categories. While both models show competitive results overall, our breakdown highlights notable performance disparities across instrument types, cultural genres, and voice-related tags.

90

80.76

80.45

79.79

78.43

80

75.15

72.2

69.95

68.68

70

65.89

63.81

62.83

63.64

60

56.99

56.5

56.02

54.83

54.24

52.82

51.57

50.22

49.87

49.63

49.54

50

47.85

45.95

Values

45.09

44.59

43.02

40

30

29.67

20

21.98

20.16

16.84

15.68

10

12.05

11.85

11.66

10.9

10.71

8.81

8.16

6.85

4.28

0

3.0

2.06

1.51

1.41

1.32

1.23

0.64

0.59

0.58

0.57

0.43

0.33

90

80

76.3

70.04

70

66.78

66.21

64.44

69.55

63.07

62.41

61.29

60.88

59.2

59.1

58.9

60

57.49

55.86

55.32

53.9

52.63

51.89

50.73

50.27

48.21

50

47.68

46.84

46.72

45.76

Values

41.43

40

28.69

30

20

19.15

10

13.35

12.93

12.41

11.56

9.23

8.88

7.45

7.24

6.55

5.75

5.51

5.22

4.89

4.01

3.75

3.64

0

3.2

2.04

1.3

1.2

1.02

0.92

0.85

0.3

0.17

All-instrumentaccordion bongoharmonica piano violinAll-Genremedieval 60s 70s 80s 90sBossanova Celticchanson ethno Latin world All-Tag female male woman manfemalevoicemalevoicefemalevocalmalevocal

Figure 2. Fine-grained evaluation of Qwen2-Audio and Audio-Flamingo on instrument (purple), genre (yellow), and vocal (red) tag classification. The upper extremity represents the ROC-AUC value, and the lower is PR-AUC.

- 5.2.1 Instrument Bias on MTG-Instrument

Both models achieve high scores on piano, reflecting the strong representation of piano in most training datasets. Western instruments such as violin and accordion perform close to the average, suggesting moderate robustness across common musical timbres. However, performance drops significantly on bongo and harmonica — commonly associated with world music. These results point to a persistent bias toward Western instruments and limited generalization to underrepresented timbres in current pre-training corpora.

- 5.2.2 Cultural Genre Imbalance on MTG-Genre

Genre classification results similarly reveal systematic disparities. Both models show relatively strong performance on mainstream Western pop genres (e.g., 80s, 90s), while genres associated with world music (e.g., Bossanova, Celtic, Chanson, Ethno, Latin) and music traditions (e.g., Medieval) consistently fall below average. For example, AudioFlamingo’s performance on Bossanova and Chanson drops severely. Qwen2-Audio performs slightly better on some long-tail genres, but still shows considerable degradation. These highlight a lack of cultural and historical diversity in the data used for instruction tuning and model pretraining.

- 5.2.3 Voice Tag Differences on MTT

A detailed comparison on vocal tags reveals an interesting divergence. Audio-Flamingo is consistently better at identifying *female* voices than male voices, indicating a possible gender-related acoustic or annotation bias. In contrast, Qwen2-Audio achieves higher ROC-AUC for *female* tags but lower PR-AUC, suggesting that while the

model ranks positive examples correctly, its absolute predictions remain sparse or overconfident. This mismatch implies that Qwen2-Audio is sensitive to class ranking but may lack calibration in estimating tag presence probabilities, an issue worth investigating for fairness and reliability in music model deployment.

## 5.3 Ablation Study on Different Prompts and Trials

[Figure 1]

Figure 3. Ablation Study on Prompt Sensitivity for Genre Classification and Arousal Regression

We conduct an ablation study on prompt design using GAMA and GAMA-IT models across two representative tasks: GTZAN genre classification and EMO arousal regression. Variant Prompts 1 and 2 are evaluated over three runs, and the bars report mean performance with standard variant as error bars. GTZAN results (bottom row) are relatively stable across prompts and have small variance for each prompt in multi-trials, indicating that most genre-related instructions are consistently followed. The low variance suggests robustness to prompt changes. In contrast, EMO-A results (top row) show relative sensitivity to prompt variation, particularly under the GAMA-IT model. This instability stems from a higher rate of invalid or non-responsible generations, which are scored as mean values during evaluation. Consequently, differences in prompt phrasing might lead to large deviations, especially when valid predictions diverge significantly from the mean score.

## 6. CONCLUSION

We introduce CMI-Bench, a comprehensive benchmark for evaluating audio-text LLMs across diverse MIR tasks. Our results highlight a significant performance gap between LLMs and supervised MIR systems, with best models like Qwen2-Audio and GAMA also struggling with generalization. Sequence-based tasks, such as melody extraction and beat tracking, pose particular challenges, likely due to limited timestamped pretraining and prompt sensitivity. Fine-grained analysis also reveals cultural and gender biases tied to training data imbalances. By offering a standardized evaluation framework and toolkit, CMI-Bench bridges NLP and MIR research, providing a foundation for future advancements. Progress will hinge on improved pretraining, sequential output handling, and bias mitigation, and we hope this work spurs collaboration toward more capable music-aware LLMs.

## A. ACKNOWLEDGMENTS

Yinghao Ma is a research student at the UKRI Centre for Doctoral Training in Artificial Intelligence and Music, supported by UK Research and Innovation [grant number EP/S022694/1].

Siyou Li is a research student at the Computational Linguistics Lab at Queen Mary University of London, funded by the QMUL-CSC PhD scholarships.

Yinghao Ma would also like to express heartfelt gratitude to the Student Philharmonic Chinese Orchestra at the Chinese Music Institute, Peking University (abbreviated as CMI, unrelated to the paper title). We warmly congratulate the orchestra on its 20th anniversary.

## B. ETHICS STATEMENT

CMI-Bench repurposes existing publicly available datasets in the MIR domain by reformatting their annotations into instruction-following formats. No new human annotations were collected, and no human participants were involved in the creation of this benchmark. All data used in the project are licensed under terms that permit non-commercial research use. In compliance with these terms, we license CMI-Bench under a Creative Commons AttributionNonCommercial-ShareAlike (CC BY-NC-SA) license. To promote long-term accessibility, we host the audio test set on Hugging Face with clear usage restrictions for noncommercial purposes.

The dataset primarily consists of Western, Englishlanguage popular music, with limited inclusion of instrumental tracks and non-English songs. Transcription tasks are restricted to English lyrics, and world music instruments besides Guzheng are underrepresented. We acknowledge this cultural and linguistic skew and encourage future extensions to improve global diversity and representation.

This work involves no safety, security, or environmental risks. The benchmark does not require high-compute model training or deployment of potentially harmful generative models. We release CMI-Bench and its evaluation toolkit to foster responsible and reproducible research in audiolanguage modeling.

## C. APPENDIX

Due to the limitation of the ISMIR proceeding, please refer to our arxiv version for more information on instruction examples, error case analysis and more discussion.

## D. REFERENCES

- [1] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., “A survey of large language models,” arXiv preprint arXiv:2303.18223, vol. 1, no. 2, 2023.
- [2] Y. Ma, A. Øland, A. Ragni, B. M. Del Sette, C. Saitis, C. Donahue, C. Lin, C. Plachouras, E. Benetos, E. Shatri et al., “Foundation models for music: A survey,” arXiv preprint arXiv:2408.14340, 2024.

- [3] S. Liu, A. S. Hussain, C. Sun, and Y. Shan, “Music understanding llama: Advancing text-to-music generation with question answering and captioning,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 286–290.
- [4] Z. Deng, Y. Ma, Y. Liu, R. Guo, G. Zhang, W. Chen, W. Huang, and E. Benetos, “Musilingo: Bridging music and text with pre-trained language models for music captioning and query response,” in NAACL-HLT (Findings), 2024.
- [5] Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and J. Zhou, “Qwen-audio: Advancing universal audio understanding via unified large-scale audiolanguage models,” arXiv preprint arXiv:2311.07919, 2023.
- [6] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, M. Zejun, and C. Zhang, “Salmonn: Towards generic hearing abilities for large language models,” in The Twelfth International Conference on Learning Representations.
- [7] Q. Yang, J. Xu, W. Liu, Y. Chu, Z. Jiang, X. Zhou, Y. Leng, Y. Lv, Z. Zhao, C. Zhou et al., “Air-bench: Benchmarking large audio-language models via generative comprehension,” in ACL (1), 2024.
- [8] B. Weck, I. Manco, E. Benetos, E. Quinton, G. Fazekas, and D. Bogdanov, “Muchomusic: Evaluating music understanding in multimodal audio-language models,” in Proceedings of the 25th International Society for Music Information Retrieval Conference, ISMIR 2024, San Francisco, California, USA and Online, November 10-14, 2024, B. Kaneshiro, G. J. Mysore, O. Nieto, C. Donahue, C. A. Huang, J. H. Lee, B. McFee, and M. C. McCallum, Eds., 2024, pp. 825–833. [Online]. Available: https://doi.org/10.5281/zenodo.14877459
- [9] Y. Li, G. Zhang, Y. Ma, R. Yuan, K. Zhu, H. Guo, Y. Liang, J. Liu, Z. Wang, J. Yang et al., “Omnibench: Towards the future of universal omni-language models,” arXiv preprint arXiv:2409.15272, 2024.
- [10] S. Sakshi, U. Tyagi, S. Kumar, A. Seth, R. Selvakumar, O. Nieto, R. Duraiswami, S. Ghosh, and D. Manocha, “Mmau: A massive multi-task audio understanding and reasoning benchmark,” in The Thirteenth International Conference on Learning Representations.
- [11] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap et al., “Supernaturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 5085–5109.
- [12] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning

- language models with self-generated instructions,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2023.
- [13] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.
- [14] L. Xue, Z. Zhou, J. Pan, Z. Li, S. Fan, Y. Ma, S. Cheng, D. Yang, H. Guo, Y. Xiao et al., “Audio-flan: A preliminary release,” arXiv preprint arXiv:2502.16584, 2025.
- [15] J. Li, L. Yang, M. Tang, C. Chenchong, Z. Li, P. Wang, and H. Zhao, “The music maestro or the musically challenged, a massive music evaluation benchmark for large language models,” in Findings of the Association for Computational Linguistics ACL 2024, 2024, pp. 3246– 3257.
- [16] B. Wang, X. Zou, G. Lin, S. Sun, Z. Liu, W. Zhang, Z. Liu, A. Aw, and N. F. Chen, “Audiobench: A universal benchmark for audio large language models,” arXiv preprint arXiv:2406.16020, 2024.
- [17] J. Melechovsky, Z. Guo, D. Ghosal, N. Majumder, D. Herremans, and S. Poria, “Mustango: Toward controllable text-to-music generation,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 8286–8309.
- [18] Z. Wang, S. Li, T. Zhang, Q. Wang, P. Yu, J. Luo, Y. Liu, M. Xi, and K. Zhang, “Muchin: a chinese colloquial description benchmark for evaluating language models in the field of music,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 2024, pp. 7771–7779.
- [19] Y. Gong, H. Luo, A. H. Liu, L. Karlinsky, and J. Glass, “Listen, think, and understand,” in International Conference on Learning Representations, 2024.
- [20] Y. Gong, A. H. Liu, H. Luo, L. Karlinsky, and J. Glass, “Joint audio and speech understanding,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.
- [21] A. Radford, J. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via largescale weak supervision (arxiv: 2212.04356). arxiv,” 2022.
- [22] J. P. Gardner, S. Durand, D. Stoller, and R. M. Bittner, “Llark: A multimodal instruction-following language model for music,” in International Conference on Machine Learning. PMLR, 2024, pp. 15037–15082.

- [23] S. Deshmukh, B. Elizalde, R. Singh, and H. Wang, “Pengi: An audio language model for audio tasks,” Advances in Neural Information Processing Systems, vol. 36, pp. 18090–18108, 2023.
- [24] S. Ghosh, S. Kumar, A. Seth, C. K. R. Evuru, U. Tyagi, S. Singh, O. Nieto, R. Duraiswami, and D. Manocha, “Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities,” CoRR, 2024.
- [25] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.
- [26] Z. Kong, A. Goel, R. Badlani, W. Ping, R. Valle, and B. Catanzaro, “Audio flamingo: a novel audio language model with few-shot learning and dialogue abilities,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 25125–25148.
- [27] S. Ghosh, Z. Kong, S. Kumar, S. Sakshi, J. Kim, W. Ping, R. Valle, D. Manocha, and B. Catanzaro, “Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities,” arXiv preprint arXiv:2503.03983, 2025.
- [28] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, D. P. Ellis, and C. C. Raffel, “Mir_eval: A transparent implementation of common mir metrics.” in ISMIR, 2014, pp. 367–372.
- [29] P. Knees, Á. Faraldo Pérez, H. Boyer, R. Vogl, S. Böck, F. Hörschläger, M. Le Goff et al., “Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections,” in Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR); 2015 Oct 26-30; Málaga, Spain.[Málaga]: International Society for Music Information Retrieval, 2015. p. 364-70. International Society for Music Information Retrieval (ISMIR), 2015.
- [30] M. Soleymani, M. N. Caro, E. M. Schmidt, C.-Y. Sha, and Y.-H. Yang, “1000 songs for emotional analysis of music,” in Proceedings of the 2nd ACM international workshop on Crowdsourcing for multimedia, 2013, pp. 1–6.
- [31] E. Law, K. West, M. I. Mandel, M. Bay, and J. S. Downie, “Evaluation of algorithms using games: The case of music tagging.” in ISMIR. Citeseer, 2009, pp. 387–392.
- [32] D. Bogdanov, M. Won, P. Tovstogan, A. Porter, and X. Serra, “The mtg-jamendo dataset for automatic music tagging,” in International Conference on Machine Learning. ICML, 2019.
- [33] J. Engel, C. Resnick, A. Roberts, S. Dieleman, M. Norouzi, D. Eck, and K. Simonyan, “Neural audio synthesis of musical notes with wavenet autoencoders,” in International Conference on Machine Learning. PMLR, 2017, pp. 1068–1077.

- [34] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Transactions on speech and audio processing, vol. 10, no. 5, pp. 293–302, 2002.
- [35] J. Wilkins, P. Seetharaman, A. Wahl, and B. Pardo, “Vocalset: A singing voice dataset.” in ISMIR, 2018, pp. 468–474.
- [36] I. Manco, B. Weck, S. Doh, M. Won, Y. Zhang, D. Bogdanov, Y. Wu, K. Chen, P. Tovstogan, E. Benetos et al., “The song describer dataset: a corpus of audio captions for music-and-language evaluation,” arXiv preprint arXiv:2311.10057, 2023.
- [37] A. Agostinelli, T. I. Denk, Z. Borsos, J. Engel, M. Verzetti, A. Caillon, Q. Huang, A. Jansen, A. Roberts, M. Tagliasacchi et al., “Musiclm: Generating music from text,” arXiv preprint arXiv:2301.11325, 2023.
- [38] G. Roa Dabike and J. Barker, “Automatic lyric transcription from karaoke vocal tracks: Resources and a baseline system,” in Proceedings of the 20th Annual Conference of the International Speech Communication Association (INTERSPEECH 2019), 2019.
- [39] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, “An experimental comparison of audio tempo induction algorithms,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832–1844, 2006.
- [40] F. Krebs, S. Böck, and G. Widmer, “Rhythmic pattern modeling for beat and downbeat tracking in musical audio.” in Ismir, 2013, pp. 227–232.
- [41] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. Bello, “Medleydb: A multitrack dataset for annotation-intensive mir research,” 10 2014.
- [42] D. Li, M. Che, W. Meng, Y. Wu, Y. Yu, F. Xia, and W. Li, “Frame-level multi-label playing technique detection using multi-scale network and self-attention mechanism,” in IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023. IEEE, 2023, pp. 1–5.
- [43] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, “C-pack: Packed resources for general chinese embeddings,” in Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 2024, pp. 641– 649.
- [44] K. Papineni, S. Roukos, T. Ward, and W. jing Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.
- [45] C.-Y. Lin and F. J. Och, “ORANGE: a method for evaluating automatic evaluation metrics for machine translation,” in COLING 2004: Proceedings of

- the 20th International Conference on Computational Linguistics. Geneva, Switzerland: COLING, aug 23–aug 27 2004, pp. 501–507. [Online]. Available: https://www.aclweb.org/anthology/C04-1072
- [46] S. Banerjee and A. Lavie, “Meteor: An automatic metric for mt evaluation with improved correlation with human judgments,” in Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005, pp. 65–72.
- [47] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available: https://www.aclweb.org/anthology/ W04-1013
- [48] T. Zhang*, V. Kishore*, F. Wu*, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with bert,” in International Conference on Learning Representations, 2020. [Online]. Available: https: //openreview.net/forum?id=SkeHuCVFDr
- [49] F. Korzeniowski and G. Widmer, “End-to-end musical key estimation using a convolutional neural network,” in 2017 25th European Signal Processing Conference (EUSIPCO). IEEE, 2017, pp. 966–970.
- [50] R. Castellon, C. Donahue, and P. Liang, “Codified audio language modeling learns useful representations for music information retrieval,” arXiv preprint arXiv:2107.05677, 2021.
- [51] R. Yuan, Y. Ma, Y. Li, G. Zhang, X. Chen, H. Yin, Y. Liu, J. Huang, Z. Tian, B. Deng et al., “Marble: Music audio representation benchmark for universal evaluation,” Advances in Neural Information Processing Systems, vol. 36, pp. 39626–39647, 2023.
- [52] Q. Huang, A. Jansen, J. Lee, R. Ganti, J. Y. Li, and D. P. Ellis, “Mulan: A joint embedding of music audio and natural language,” arXiv preprint arXiv:2208.12415, 2022.
- [53] M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon, and A. F. Ehmann, “Supervised and unsupervised learning of audio representations for music understanding,” Ismir 2022 Hybrid Conference, 2022.
- [54] P. Alonso-Jiménez, X. Serra, and D. Bogdanov, “Music representation learning based on editorial metadata from discogs,” 2022.
- [55] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “Masked modeling duo: Learning representations by encouraging both networks to model the input,” in ICASSP 2023-2023 IEEE International Conference On Acoustics, Speech And Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

- [56] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Lin, A. Ragni, E. Benetos, N. Gyenge, R. Dannenberg, R. Liu, W. Chen, G. Xia, Y. Shi, W. Huang, Y. Guo, and J. Fu, “Mert: Acoustic music understanding model with large-scale self-supervised training,” 2023.
- [57] L. Wang, P. Luc, Y. Wu, A. Recasens, L. Smaira, A. Brock, A. Jaegle, J.-B. Alayrac, S. Dieleman, J. Carreira et al., “Towards learning universal audio representations,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 4593–4597.
- [58] S. Doh, K. Choi, J. Lee, and J. Nam, “Lp-musiccaps: Llm-based pseudo music captioning,” in Ismir 2023 Hybrid Conference, 2023.
- [59] L. Ou, X. Gu, and Y. Wang, “Transfer learning of wav2vec 2.0 for automatic lyric transcription,” in Ismir 2022 Hybrid Conference, 2022.
- [60] M. Heydari, F. Cwitkowitz, and Z. Duan, “Beatnet: CRNN and particle filtering for online joint beat, downbeat and meter tracking,” in Proceedings of the 22nd International Society for Music Information Retrieval Conference, ISMIR 2021, Online, November 7-12, 2021, J. H. Lee, A. Lerch, Z. Duan, J. Nam, P. Rao, P. van Kranenburg, and A. Srinivasamurthy, Eds., 2021, pp. 270–277. [Online]. Available: https: //archives.ismir.net/ismir2021/paper/000033.pdf
- [61] J. Zhao, G. Xia, and Y. Wang, “Beat transformer: Demixed beat and downbeat tracking with dilated self-attention,” in Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, P. Rao, H. A. Murthy, A. Srinivasamurthy, R. M. Bittner, R. C. Repetto, M. Goto, X. Serra, and M. Miron, Eds., 2022, pp. 169–177. [Online]. Available: https://archives.ismir.net/ismir2022/paper/000019.pdf
- [62] S. Yu, X. He, K. Chen, and Y. Yu, “Hkdsme: Heterogeneous knowledge distillation for semi-supervised singing melody extraction using harmonic supervision,” in ACM Multimedia 2024.
- [63] D. Li, Y. Ma, W. Wei, Q. Kong, Y. Wu, M. Che, F. Xia, E. Benetos, and W. Li, “Mertech: Instrument playing technique detection using self-supervised pretrained model with multi-task finetuning,” in ICASSP, 2024.
- [64] L. Zhuo, R. Yuan, J. Pan, Y. Ma, Y. Li, G. Zhang, S. Liu, R. Dannenberg, J. Fu, C. Lin et al., “Lyricwhiz: Robust multilingual lyrics transcription by whispering to chatgpt.” International Society for Music Information Retrieval Conference (ISMIR), 2023.

E. SUPPLEMENTARY MATERIAL & APPENDIX E.1 Examples of Instructions

We change the few-shot examples on the output formats to text descriptions. The updates instructions are as follows.

{"instruction": "Please provide the description of

given song.", "input": "<|SOA|><AUDIO><|EOA|>", "output": "Uplifting English rock and roll song with a

positive feeling.", "uuid": "", "split": ["test"], "task_type": \begin{lstlisting} {"major": ["captioning

"], "minor": ["music_captioningn"]}, "domain": "music", "audio_path": ["data/SDD/audio/audio/71/1009671.2min.

mp3"], "audio_start": 0.0, "audio_end": 30.0, "source": "MTG-SDD", "other": \begin{lstlisting} {"tag": "null"}}

{"instruction": "You are given an audio clip of a music piece.

Estimate the musical key of the audio by selecting exactly one key from the following list:

C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major,

C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor.

Return only the name of the predicted key. No

explanation. ", "input": "<|SOA|><AUDIO><|EOA|>", "output": "E major", "uuid": "", "split": ["test"], "task_type": {"major": ["multi-class"], "minor": ["

key_detection"]},

"domain": "music", "audio_path": ["data/GS-key/ giantsteps_clips/wav/4693198-2.wav"], "audio_start ": 0.0,

"audio_end": 30.0, "source": "GiantSteps", "other": {"tag": "null"}}

{"instruction": "Arousal refers to the level of physiological or psychological activation a piece of music induces in a listener. It reflects how calming or energizing the music feels.

You are given a piece of music. Estimate its arousal level on a scale from 1 (very calm/soothing) to 9 (very intense/energetic).

Low arousal: Relaxing, calm, sleepy, or soothing music High arousal: Exciting, intense, energetic, or

stimulating music Return a single number between 1 and 9 representing the

perceived arousal level.", "input": "<|SOA|><AUDIO><|EOA|>", "output": "0.4645976536872577", "uuid": "", "split": ["train"], "task_type": {"major": ["regression"], "minor": ["emotion_regression"]}, "domain": "music", "audio_path": ["data/EMO/emomusic/wav/0020.wav"], "audio_start": 0.0, "audio_end": 45.0, "source": "EMO", "other": {"tag": "null"}}

{"instruction": "Valence refers to the emotional quality or pleasantness of the emotion experienced, whether the emotion is positive or negative.

You are given a piece of music. Estimate its valence level on a scale from 1 (very sad/melancholic) to 9 (very happy/cheerful).

Positive valence : Happy, joyful, content. Negative valence : Sad, angry, anxious. Return a single number between 1 and 9 representing the

perceived Valence level.",

"input": "<|SOA|><AUDIO><|EOA|>", "output": "2.268336892705501", "uuid": "", "split": ["train"], "task_type":{"major": ["regression"], "minor": ["emotion_regression"]}, "domain": "music", "audio_path": ["data/EMO/emomusic/wav/0025.wav"], "audio_start": 0.0, "audio_end": 45.0, "source": "EMO", "other": {"tag": "null"}}

Your response should contain only one MIDI pitch number

.\n\n

- Example 1, when the note is A1: 33\n

- Example 2, when the note is F5: 77\n

- Example 3, when the note is A4: 69", "input": "<|SOA|><AUDIO><|EOA|>", "output": 84, "uuid": "", "split": ["test"], "task_type

": \begin{lstlisting} {"major": ["multi-class"], "minor": ["pitch_classification"]}, "domain": "music", "audio_path": ["data/NSynth/nsynth-test/audio/

bass_synthetic_068-084-050.wav"], "audio_start": 0.0,

"audio_end": -1, "source": "Nsynth", "other": {"tag": "null"}}

{"instruction": "Please provide the tags of given music

.", "input": "<|SOA|><AUDIO><|EOA|>", "output": " classical, opera, singing, quiet, woman, voice, female voice, choral", "uuid": "", "split": [" train"],

{"instruction": "Please transcribe the lyrics of the

"task_type": {"major": ["multi_label"], "minor": ["

given song.", "input": "<|SOA|><AUDIO><|EOA|>", "output": "DOESN’T TEAR YOU APART ANYMORE", "uuid": "", "split": ["test"], "task_type": {"major": ["transcription"], "minor": ["lyrics_transcription"]}, "domain": "music", "audio_path": ["data/DSing/sing_300x30x2/GB/GBVocals /225741816_226995-475216816_227615877-GB-F

music_tagging"]}, "domain": "music", "audio_path": ["data/MTT/mp3/9/american_bach_soloists-

heinrich_schutz__musicalische_exequien-01musicalische_exequien_swv_279_teil_i_concert_in_form_einer_teutschen_begrabnismissa

-349-378.mp3"], "audio_start": 0.0, "audio_end": 30.0, "source": "The MagnaTagATune Dataset", "other": {"tag": "null"}}

-475215813.m4a"], "audio_start": 277.867, "audio_end": 283.778, "source": "DSing", "other":{"tag": "null"}}

{"instruction": "You are given an audio clip of a music

piece. Identify the genre by selecting exactly one option from the following list:

Genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock. Return only the name of the predicted genre. No

{"instruction": "Identify and list the timestamps of all downbeats in this audio track. Use the format of ‘0.0s,0.54s,1.0ss, ...‘",

explanation.", "input": "<|SOA|><AUDIO><|EOA|>", "output": "blues", "uuid": "", "split": ["train"], "task_type": {"major": ["multi_class"], "minor": ["genre_classification"]}, "domain": "music", "audio_path": ["data/GTZAN/Data/genres_original/blues/

"input": "<|SOA|><AUDIO><|EOA|>", "output": "1.61s,3.42s,5.25s,7.04s,8.85s,10.65s,12.47s

,14.27s,16.08s,17.88s,19.68s,21.5s,23.32s,25.12s ,26.94s,28.74s",

"uuid": "", "split": ["test"], "task_type": {"major": ["seq_multi-class"], "minor": ["downbeat_tracking"]}, "audio_path": ["data/ ballroom/BallroomData/Rumba-American/AlbumsAnaBelen_Veneo-13.wav"], "domain": "music",

blues.00034.wav"], "audio_start": 0.0, "audio_end": 30.0, "source": "GTZAN Dataset", "other": {"tag": "null"}}

"audio_start": 0.0, "audio_end": 30.0, "source": "ballroom", "other": {"tag": "null"}}\

{"instruction": "You are given an audio clip of a vocal performance.

Identify the singing technique used by selecting exactly one option from the list below:

{"instruction": "Detect the timestep occurrences of Guzheng (Chinese zither) playing techniques in the

Singing Techniques: belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry.

given audio. The possible techniques include: Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, and Tremolo.\n\n

Return only the name of the predicted technique. No

explanation", "input": "<|SOA|><AUDIO><|EOA|>", "output": "belt", "uuid": "", "split": ["train"], "task_type":{"major": ["multi-class"], "minor": ["singing_technique_classification"]}, "domain": "music", "audio_path": ["data/VocalSet/audio/belt/

The output format should be a Python string representation of a list containing tuples of ( start time second, end time second, technique). If

no technique is detected, return [(’start_time’, ’ end_time’, ’No Tech’)].\n\n

- Example 1:\n \"[(’5.5035’, ’6.0724’, ’Upward Portamento’), (’7.0708’,

’8.0809’, ’Upward Portamento’), (’9.6947’, ’10.0’, ’Upward Portamento’)]\"\n

- Example 2:\n \"[(’0.0’, ’10.0’, ’No Tech’)]\" ", "input": "<|SOA|><AUDIO><|EOA|>", "output": "[(’9.7591’, ’10.4321’, ’Upward Portamento’)

f1_scales_belt_o_01.wav"], "audio_start": 0.0, " audio_end": -1,

"source": "VocalSet", "other": {"tag": "null"}}

]", "uuid": "", "split": ["train"], "task_type": {"major": ["seq_multi-class"], "minor": ["technique_detection"]}, "domain": "music", "audio_path": ["data/Guzheng/Guzheng_Tech99/data/audio/

{"instruction": "Identify the pitch of the given audio.

You must choose exactly one pitch from the given range.\n\n

Pitches are represented using the MIDI standard, where each pitch is assigned an integer value between 0 (lowest) and 127 (highest). In this dataset, the lowest pitch is 9 (1 octave below A_0, 13.75Hz), and the highest pitch is 119 (B 8, 7902.13Hz). Be sure to convert a musical note to its MIDI number :\n

train/41.flac"], "audio_start": 0.0, "audio_end": 10.0, "source": "Guzheng Tech99", "other": {"tag": "null"}}

Middle C (C4) = 60\nA440 (A4) = 69\n One octave higher = +12 MIDI numbers\nOne octave lower

= -12 MIDI numbers\n\n

[Figure 2]

- Figure 4. Comparison of model rankings across four benchmarks: OmniBench-Music, MMAU-Music-test, MuCho Music, and our proposed CMI-Bench. While absolute scores vary across benchmarks, the relative ordering of models remains largely consistent, suggesting stable performance trends across different evaluation protocols.

## E.2 Comparison with Previous Benchmarks

For each task and each dataset in CMI-bench, we average all evaluation metrics (e.g., average 4 metrics for captioning on SDD and MusicCaps respectively) and rank all model performances from 0 to 10. For each model, we average the rank of each model among all tasks to get the universal rank on CMI-benchmark.

Our evaluation across four benchmarks—OmniBenchMusic, MMAU-Music-test, MuCho Music, and our proposed benchmark—reveals consistent performance trends among the tested LLMs for music-related tasks. As shown in the figure 4, models like Qwen-audio and Salmonn-audio exhibit robust rankings across all benchmarks, suggesting their adaptability to diverse music understanding challenges. The similarity in model ordering across benchmarks underscores a degree of stability in their relative capabilities, yet the nuanced differences suggest that task-specific design or training data may play a critical role. These findings invite further exploration into how LLMs balance generalizability and specialization in MIR tasks.

## E.3 Embedding-Based Evaluation for Multi-Label Tagging

In multi-label tagging tasks such as emotion and instrument classification, model outputs often include free-form text that may not exactly match the ground-truth tags. To address this, we compare several retrieval-based approaches for computing ROC-AUC and PR-AUC: (1) exact match (“Accurate”), (2) semantic similarity using BERTScore (roberta-large), (3) BGE encoder, and (4) GTE-Qwen embedding model. As shown in Table X, exact match yields the lowest scores, failing to capture near-synonymous predictions (e.g., "electric guitar" vs. "guitar"). While BERTScore offers modest improvements, it is not optimized for retrieval and underperforms in multi-label scenarios. BGE, a compact 330M model fine-tuned for text embedding and retrieval, consistently delivers strong results while remaining computationally efficient. GTE-Qwen, a larger 7B model with strong retrieval capabilities, achieves the best scores

overall but is significantly more expensive to run. Based on this analysis, we adopt BGE for all benchmark evaluations, striking a balance between retrieval quality and evaluation efficiency.

## E.4 Case Study of Qwen-audio Output

- E.4.1 Beat Tracking

{

"question": "Identify and list the timestamps of all downbeats in this audio track. Use the format of ‘0.0s,0.54s,1.0ss, ...‘",

"response": "The timestamps of all downbeats in

this audio track are ‘0.0s,0.54s,1.0ss, 1.62 s, 2.16s, 2.70s, 3.24s, 3.78s, 4.32s, 4.86s,

5.40s, 5.94s, 6.48s, 7.02s, 7.56s, 8.10s, 8.64s, 9.18s, 9.72s, 10.26s, 10.80s, 11.34s, 11.88s, 12.42s, 12.96s, 13.50s, 14.04s,

14.58s, 15.12s, 15.66s, 16.20s, 16.74s, 17.28s, 17.82s, 18.36s, 18.90s, 19.44s, 19.98s, 20.52s, 21.06s, 21.60s, 22.14s, 22.68s, 23.22s, 23.76s, 24.30s, 24.84s, 25.38s, 26.02s, 26.56s, 27.10s, 27.64s, 28.18s, 28.72s, 29.26s, 29.80s, 30.34s, 30.88s, 31.42s, 31.96s, 32.50s, 33.04s, 33.58s, 34.12s, 34.66s, 3",

"correct_answer": "1.61s,3.42s,5.25s,7.04s,8.85s ,10.65s,12.47s,14.27s,16.08s,17.88s,19.68s ,21.5s,23.32s,25.12s,26.94s,28.74s", "audioid": "../testdata/ballroom/BallroomData/

Rumba-American/Albums-AnaBelen_Veneo-13.wav", "other": ""

},

The downbeat tracking output of Qwen-audio on ballroom dataset, though it provides the best results among all models, are simply copying the input example always.

- E.4.2 Lyrics Transcription

{

"question": "Please transcribe the lyrics of the given song.", "response": "The lyrics of the given song are:\" You make me feel alive\".",

"correct_answer": "YOU MADE ME FEEL AS THOUGH", "audioid": "../testdata/DSing/sing_300x30x2/GB/

GBVocals/3770008_3770008-1036593237 _1553518113-GB-F-933206912.m4a",

"other": "" },

The results of Qwen-audio do not have hallucination on song description, music captions, lyrics context or emoji that are typically produced by other models, but we can still observe significant errors in the upper cases.

Besides, the model can generate the same word in the lyrics repeatedly without a stop when the singer sing a word twice or three times, as illustrated by the following case. If we exclude such cases in the testset, the WER performance of Qwen-audio model will decrease from 115.7(%) to 50(%), which is still significantly worse than the SOTA performance 13(%), but much more reasonable.

{High WER: 167.66666666666666, prediction: The lyrics of the given song are:"Go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go

MTG-Instrument

Qwen2-audio Qwen-audio Salmonn-audio MusiLingo LTU LTU-AS

MTG-Emotion

Qwen2-audio Qwen-audio Salmonn-audio MusiLingo LTU LTU-AS

- Figure 5. Comparison of multi-label evaluation methods on MTG-Emotion and MTG-Instrument tasks. "Accurate" denotes strict matching, while BERTScore, BGE, and GTE-Qwen represent embedding-based approaches..

go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go

go go go go go go go go go go go go go go go go go go go go,

reference: DON’T DON’T DON’T}, {WER: 62.0, prediction: The lyrics of the given song are:"Cause you

knew that I knew that I knew that I knew that I

knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I

knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I knew that I, reference: ’CAUSE YOU KNEW THAT I KNEW THAT I }, {WER: 2.25, prediction: The lyrics of the given song are:"Black

black black black black black Black black black black black black".

reference: BLACK BLACK OH BLACK}

### E.4.3 EMO

{

"question": "Arousal refers to the level of physiological or psychological activation a piece of music induces in a listener. It reflects how calming or energizing the music

feels.\nYou are given a piece of music. Estimate its arousal level on a scale from 1

(very calm/soothing) to 9 (very intense/ energetic).\nLow arousal: Relaxing, calm, sleepy, or soothing music\nHigh arousal: Exciting, intense, energetic, or stimulating

music\nReturn a single number between 1 and 9 representing the perceived arousal level

.", "response": "Based on the audio, I would estimate

a arousal level of 6.", "correct_answer": "1.9075890449018524", "audioid": "../testdata/EMO/emomusic/wav/0962.wav

", "other": ""

}, {

"question": "Arousal refers to the level of physiological or psychological activation a piece of music induces in a listener. It reflects how calming or energizing the music

feels.\nYou are given a piece of music. Estimate its arousal level on a scale from 1

(very calm/soothing) to 9 (very intense/ energetic).\nLow arousal: Relaxing, calm, sleepy, or soothing music\nHigh arousal: Exciting, intense, energetic, or stimulating

music\nReturn a single number between 1 and 9 representing the perceived arousal level

.",

"response": "Based on the audio, it sounds like the music is medium to high arousal.", "correct_answer": "1.330392488416014", "audioid": "../testdata/EMO/emomusic/wav/0970.wav

", "other": "" },

Qwen-audio response on EMO sometimes can give a reasonable result, but not always. Besides it may disobey the instruction commonly.

- E.4.4 GS-Key

{

"question": "You are given an audio clip of a music piece.\nEstimate the musical key of the audio by selecting exactly one key from the following list:\n\nC major, Db major, D major, Eb major, E major, F major, Gb major,

G major, Ab major, A major, Bb major, B major,\nC minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor.\n\nReturn

only the name of the predicted key. No explanation.\n",

"response": "The musical key of the audio is C

major.", "correct_answer": "D major", "audioid": "../testdata/GS-key/giantsteps_clips/

wav/0010089-3.wav", "other": ""

}, {

"question": "You are given an audio clip of a music piece.\nEstimate the musical key of the audio by selecting exactly one key from the following list:\n\nC major, Db major, D major, Eb major, E major, F major, Gb major,

G major, Ab major, A major, Bb major, B major,\nC minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor.\n\nReturn

only the name of the predicted key. No explanation.\n",

"response": "The musical key of the audio is A

major.", "correct_answer": "F minor", "audioid": "../testdata/GS-key/giantsteps_clips/

wav/0061415-0.wav", "other": ""

},

Qwen-audio can follow the instruction of Key detection, but its ZERO-SHOT performance does not surpass the random guess.

- E.4.5 Genre Classification

{

"question": "You are given an audio clip of a

music piece. Identify the genre by selecting exactly one option from the following list: \nGenres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.\

nReturn only the name of the predicted genre.

No explanation.\n", "response": "The predicted genre is blues.", "correct_answer": "blues", "audioid": "../testdata/GTZAN/Data/

genres_original/blues/blues.00027.wav", "other": ""

}, {

"question": "You are given an audio clip of a

music piece. Identify the genre by selecting exactly one option from the following list: \nGenres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.\

nReturn only the name of the predicted genre.

No explanation.\n", "response": "The predicted genre is blues.", "correct_answer": "blues", "audioid": "../testdata/GTZAN/Data/

genres_original/blues/blues.00028.wav", "other": ""

},

Qwen-audio can provide reasonable results on clip-level multi-class such as genre classification.

- E.4.6 Pitch Recognition

{

"question": "Identify the pitch of the given audio. You must choose exactly one pitch from the given range.\n\nPitches are represented using the MIDI standard, where each pitch is assigned an integer value between 0 (lowest) and 127 (highest). In this dataset, the lowest pitch is 9 (1 octave below A_0, 13.75Hz), and the highest pitch is 119 (B 8, 7902.13Hz). Be sure to convert a musical note to its MIDI number:\ nMiddle C (C4) = 60\nA440 (A4) = 69\nOne octave higher = +12 MIDI numbers\nOne octave

lower = -12 MIDI numbers\n\nYour response should contain only one MIDI pitch number.\n \nExample 1, when the note is A1: 33\ nExample 2, when the note is F5: 77\nExample

3, when the note is A4: 69", "response": "The given audio is the recording of

a voice message saying, \"This is the message\". The frequency of the voice message is around 135 Hz.",

"correct_answer": 49, "audioid": "../testdata/NSynth/nsynth-test/audio/

bass_synthetic_068-049-025.wav", "other": ""

},

135Hz is C#3 minus 45 cents, the MIDI number result is 49, but it does not follow the instruction format.

- E.4.7 Vocal Technique Classification

{ "question": "You are given an audio clip of a

vocal performance.\nIdentify the singing technique used by selecting exactly one option from the list below:\nSinging Techniques: belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry.\nReturn only the name of the predicted technique. No explanation.",

"response": "The singing technique used is \"belt

\".", "correct_answer": "belt", "audioid": "../testdata/VocalSet/audio/belt/

f2_arpeggios_belt_o_00.wav", "other": ""

}, {

"question": "You are given an audio clip of a vocal performance.\nIdentify the singing technique used by selecting exactly one option from the list below:\nSinging Techniques: belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry.\nReturn only the name of the predicted technique. No explanation.", "response": "The singing technique used is \"

vocal fry\".", "correct_answer": "belt", "audioid": "../testdata/VocalSet/audio/belt/

f2_arpeggios_belt_u_00.wav", "other": ""

}, {

"question": "You are given an audio clip of a vocal performance.\nIdentify the singing technique used by selecting exactly one option from the list below:\nSinging Techniques: belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry.\nReturn only the name of the predicted technique. No explanation.", "response": "The singing technique used is \"

vibrato\".", "correct_answer": "belt", "audioid": "../testdata/VocalSet/audio/belt/

f2_scales_belt_a_00.wav", "other": ""

},

On Vocal technique classification, Qwen-audio can always follow the instruction, but provides the wrong or biased answer.

E.4.8 Other tasks

{

"question": "Detect the timestep occurrences of Guzheng (Chinese zither) playing techniques in the given audio. The possible techniques include: Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, and Tremolo.\n\nThe output format

should be a Python string representation of a list containing tuples of (start time

second, end time second, technique). If no technique is detected, return [(’start_time’,

’end_time’, ’No Tech’)].\n\nExample 1:\n \"[(’5.5035’, ’6.0724’, ’Upward Portamento’),

(’7.0708’, ’8.0809’, ’Upward Portamento’), (’9.6947’, ’10.0’, ’Upward Portamento’)]\"\ nExample 2:\n\"[(’0.0’, ’10.0’, ’No Tech’) ]\" ",

"response": "The given audio contains a guzheng

playing a melody. Using a voice recognition model, the lyrics are \"\u5c71\u5916\u9752\ u5c71\u697c\u5916\u697c\". The song genre is

Chinese Traditional.",

"correct_answer": "[(’140.3988’, ’141.6056’, ’ Vibrato’), (’142.8589’, ’144.0425’, ’Vibrato ’), (’145.7715’, ’146.3285’, ’Vibrato’),

- (’148.1736’, ’148.7770’, ’Vibrato’),

- (’149.9955’, ’151.2023’, ’Vibrato’)]",

"audioid": "../testdata/Guzheng/Guzheng_Tech99/

data/audio/test/94.flac", "other": ""

}, {

"question": "Detect the timestep occurrences of Guzheng (Chinese zither) playing techniques in the given audio. The possible techniques include: Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, and Tremolo.\n\nThe output format

should be a Python string representation of a list containing tuples of (start time

second, end time second, technique). If no technique is detected, return [(’start_time’,

’end_time’, ’No Tech’)].\n\nExample 1:\n \"[(’5.5035’, ’6.0724’, ’Upward Portamento’),

(’7.0708’, ’8.0809’, ’Upward Portamento’), (’9.6947’, ’10.0’, ’Upward Portamento’)]\"\ nExample 2:\n\"[(’0.0’, ’10.0’, ’No Tech’) ]\" ",

"response": "The given audio contains a guzheng playing a melody. Using a music recognition model, the following steps can be taken to extract the techniques used in the melody:\n \n 1. Apply music recognition to identify

- the melody.\n 2. Extract the notes played in

- the melody.\n 3. Identify the techniques

used in the melody based on the notes played

.\n\nThe following is the Python code to extract the techniques used in the given audio:\n‘‘‘python\nimport music21\n\n# Load the audio file\naudio = music21.midi. readAudioFile(\"audio_file.mid\")\n\n# Create a stream from the audio file\nstream

= audio.createStream()\n\n# Get the note values and durations from the stream\nnotes

= stream.getNoteValues()\n\n# Identify the techniques used in the melody\ntechniques = []\nfor note in notes:\n if note.velocity < 40:\n techniques.append((’Vibrato’, note. start, note.end))\n elif note.velocity == 40:\n techniques.append((’Point Note’, note. start, note.end))\n elif note.velocity > 40:\n techniques.append((’Upward Portamento’,

note.start, note.end))\n techniques.append ((’Downward Portamento’, note.start, note. end))\n elif note.velocity < -40:\n techniques.append((’Plucks’, note.start, note.end))\n elif note.velocity == -40:\n techniques.append((’Glissando’, note.start, note.end))\n elif note.velocity > -40:\n techniques.append((’Tremolo’, note.start, note.end))\n\n# Print the techniques

detected in the melody\nprint(techniques)\n ‘‘‘\n\nThe output of this code will be a list of tuples containing the start time, end time, and technique used for each note in the melody. For example:\n‘‘‘python\n [(’0.0’, ’0.3437’, ’Vibrato’), (’0.3437’,

- ’0.6474’, ’Vibrato’), (’0.6474’, ’0.9511’, ’ Vibrato’), (’0.9511’, ’1.2548’, ’Vibrato’), (’1.2548’, ’1.5585’, ’Vibrato’), (’1.5585’,

- ’1.8622’, ’Vibrato’), (’1.8622’, ’2.165",

"correct_answer": "[(’154.1962’, ’154.7532’, ’ Vibrato’)]", "audioid": "../testdata/Guzheng/Guzheng_Tech99/

data/audio/test/94.flac", "other": ""

}, {

"question": "Detect the timestep occurrences of Guzheng (Chinese zither) playing techniques in the given audio. The possible techniques include: Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, and Tremolo.\n\nThe output format

should be a Python string representation of a list containing tuples of (start time

second, end time second, technique). If no technique is detected, return [(’start_time’,

’end_time’, ’No Tech’)].\n\nExample 1:\n \"[(’5.5035’, ’6.0724’, ’Upward Portamento’),

(’7.0708’, ’8.0809’, ’Upward Portamento’), (’9.6947’, ’10.0’, ’Upward Portamento’)]\"\ nExample 2:\n\"[(’0.0’, ’10.0’, ’No Tech’) ]\" ",

"response": "The Guzheng (Chinese zither) is playing techniques such as Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, and Tremolo in the given audio. Here is the list of detected

techniques with their corresponding start and end times in seconds: [(5.5035, 6.0724, ’ Upward Portamento’), (7.0708, 8.0809, ’ Upward Portamento’), (9.6947, 10.0, ’Upward Portamento’)]",

"correct_answer": "[(’0.0’, ’10.0’, ’No Tech’)]", "audioid": "../testdata/Guzheng/Guzheng_Tech99/

data/audio/test/94.flac", "other": ""

}

Qwen-audio cannot follow the instruction not only on task description but format requirement on instrument performance techniques detection. Not to mention the underrepresented instrument of world music.

{

"question": "Please provide the emotion tag(s) of given audio.", "response": "The emotion tag(s) of given audio

are: relax, calm, chill, peaceful, spa, easy listening, lounge, sleep, meditation, yoga, relax, ambient, downtempo, background,

instrumental, ambient, easy listening, lounge, sleep, meditation, yoga, relax, calm,

chill, peaceful, spa, easy listening, lounge, sleep, meditation, yoga, relax, ambient, downtempo, background, instrumental

.", "correct_answer": "meditative", "audioid": "../testdata/MTG/audio-low/83/5083.low.

mp3", "other": ""

}, {

"question": "Please provide the emotion tag(s) of

given audio.", "response": "The audio is of a jazz song.", "correct_answer": "cool", "audioid": "../testdata/MTG/audio-low/84/5084.low.

mp3",

"other": "" },

{

"question": "Please provide the genre tag(s) of given audio.",

"response": "Music genre: Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient House, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge,

New Age, Electronic, IDM, Ambient, Chillout, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill-out, Lounge, New Age, Electronic, IDM, Ambient, Chill", "correct_answer": "ambient, easylistening,

orchestral", "audioid": "../testdata/MTG/audio-low/50/1420850.

low.mp3", "other": ""

}, {

"question": "Please provide the genre tag(s) of

given audio.", "response": "Jazz Lounge 1", "correct_answer": "downtempo, electronic,

jazzfusion", "audioid": "../testdata/MTG/audio-low/99/1420899.

low.mp3", "other": ""

}, {

"question": "Please provide the genre tag(s) of given audio.",

"response": "The genre tag(s) of given audio are electronic, experimental, electroacoustic, instrumental.",

"correct_answer": "electronic, house, techno", "audioid": "../testdata/MTG/audio-low/14/1421614.

low.mp3", "other": ""

}

{

"question": "Please provide the instrument tag(s) of given audio.", "response": "The instrument tag(s) of given audio

are nylon string guitar.", "correct_answer": "guitar", "audioid": "../testdata/MTG/audio-low/88/4888.low.

mp3", "other": ""

}, {

"question": "Please provide the instrument tag(s) of given audio.", "response": "The instrument tag(s) of given audio

are: drum, piano, bass.", "correct_answer": "piano", "audioid": "../testdata/MTG/audio-low/90/4890.low.

mp3",

"other": "" },

Qwen-audio sometimes mistake the emotion instruction for the genre and instruments. While the results of genre and instrument labelling are relatively reasonable.

{

"question": "Please provide the description of given song.", "response": "This is a song whose genre is Electronic, and the lyrics are \"Butterflied wings you’ll find\".", "correct_answer": "Electronic music that has a constant melody throughout with accompanying

instruments used to supplement the melody which can be heard in possibly a casual setting",

"audioid": "../testdata/SDD/audio/audio

/34/1004034.2min.mp3", "other": ""

}, {

"question": "Please provide the description of

given song.", "response": "Solo electric guitar", "correct_answer": "acoustic guitar solo track with consistent rhythm and repeating progression, suitable for a relaxing afternoon tea",

"audioid": "../testdata/SDD/audio/audio

/74/1007274.2min.mp3", "other": ""

}, {

"question": "Please provide the description of given song.",

"response": "This is a song whose genre is Pop, and the lyrics are \"See the magic circle that surrounds my heart We need to touch it when he bites my hand There’s a mystical drawing somewhere in my mind Something about

love something higher than you do\".",

"correct_answer": "A rock song with a country vibe, it has male vocals, drums, and acoustic guitar. It is an upbeat song",

"audioid": "../testdata/SDD/audio/audio

/71/1009671.2min.mp3", "other": ""

},

Qwen-audio has reasonable results on caption generation. E.5 PCA Components of CMI-Benchmark

We regard the 34 metrics of CMI-benchmark as the 34 dimensions of LLM MIR capabilities, and we evaluate the PCA results of the 11 observation of models. The top 4 PCA components are as follows:

[Figure 3]

- (a) PCA component 1

[Figure 4]

- (b) PCA component 2

[Figure 5]

- (c) PCA component 3

[Figure 6]

- (d) PCA component 4

Figure 6. PCA components of CMI-Bench.

