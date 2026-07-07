arXiv:2506.01111v1[cs.SD]1Jun2025

# FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion

Shunian Chen1 ∗ Xinyuan Xie1,2 ∗ Zheshu Chen1 ∗ Liyan Zhao1 Owen Lee1 Zhan Su1 Qilin Sun1 Benyou Wang1 †

1The Chinese University of Hong Kong, Shenzhen 2South China University of Technology wangbenyou@cuhk.edu.cn

## Abstract

High-quality, large-scale audio captioning is crucial for advancing audio understanding, yet current automated methods often generate captions that lack finegrained detail and contextual accuracy, primarily due to their reliance on limited unimodal or superficial multimodal information. Drawing inspiration from human auditory perception, which adeptly integrates cross-modal cues and performs sophisticated auditory scene analysis, we introduce a novel two-stage automated pipeline. This pipeline first employs specialized pretrained models to extract diverse contextual cues (e.g., speech, music, general sounds, and visual information from associated video). A large language model (LLM) then synthesizes these rich, multimodal inputs to generate detailed and context-aware audio captions. Key contributions of this work include: (1) the proposed scalable method for fine-grained audio caption generation; (2) FusionAudio, a new large-scale dataset comprising 1.2 million such detailed captions, combined with 6 million QA pairs; and (3) enhanced audio models developed using FusionAudio, specifically a CLAP-based audio encoder with superior audio-text alignment and instruction following. This paper paves the way for more nuanced and accurate automated understanding of complex audio environments. Code and data can be found in https://github.com/satsuki2486441738/FusionAudio.

## 1 Introduction

The advancement of models like CLAP [1] for audio retrieval, and GAMA [2] or Qwen2-Audio [3] for broader audio understanding, heavily relies on large-scale, high-quality audio captioning datasets. Audio captioning has primarily followed two trajectories. Manual annotation [4, 5] offers high quality but lacks scalability due to high labor costs. In contrast, automated methods [6, 7] often use sparse metadata like text labels or tags to assist annotation, while others [8, 9, 10] leverage basic multimodal cues. These automated approaches, however, typically rely on limited textual or superficial information, failing to capture rich details (e.g., multimodal contextual details). This results in captions that lack fine-grained details and are prone to hallucinations [11], hindering nuanced audio interpretation.

Addressing this gap necessitates a paradigm shift. We turn to human auditory perception for inspiration (Figure 1). Human auditory understanding leverages sophisticated strategies at two complementary levels. Firstly, humans adeptly integrate cross-modal cues—visual information, for instance, aids speech intelligibility [12] and sound identification [13, 14]. Secondly, auditory scene analysis

∗Equal contribution. †Corresponding author

Preprint. Under review.

- Table 1: Comparison of generated captions for a sample audio clip with associated visual context. Hallucinations in prior work are highlighted in red. Improvements from our multimodal approach, FusionAudio, are highlighted in green, demonstrating enhanced accuracy and detail by leveraging visual and comprehensive auditory cues.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Method Audio Caption GAMA (Baseline) The audio is dominated by the sound of a motor vehicle engine and intermit-

tent male speech, with wind noise.

AudioSetCaps A male and female engage in conversation, their voices audible against a backdrop of ambient noise. The discussion is neutral in tone and does not involve any identifiable objects or language.

Auto-ACD A man speaks while a vehicle moves in the distance, possibly on a motor

scooter, in an engine room.

Sound-VECaps A man is speaking and a vintage motorcycle with a large headlamp, round fuel tank, and sidecar is parked on grass, with the sound of the engine and the man’s voice filling the air, while a vehicle passes by in the background.

FusionAudio-1.2M (Ours) Continuous motor vehicle engine noise is prominently featured, accompanied by intermittent male speech with a positive or confirming tone. Wind sounds suggest an outdoor environment, with the engine’s sustained roar maintaining a steady volume throughout the recording.

(ASA) [15] allows the auditory system to parse complex soundscapes into distinct streams like speech, music, and ambient sounds based on temporal-spectral regularities [16]. These sophisticated biological mechanisms offer a compelling blueprint for enhancing automated audio captioning. The impact of this multimodal integration is demonstrated in Table 1. Current systems, often processing audio in isolation, can misinterpret sounds (e.g., a stationary motorcycle as a moving scooter) or hallucinate details. In contrast, FusionAudio-1.2M leverages comprehensive audiovisual cues to produce more accurate and contextually rich descriptions.

Inspired by these principles, we introduce a two-stage pipeline for enhanced automated audio captioning. First, specialized pretrained models extract diverse contextual cues: an Automatic Speech Recognition (ASR) model [17] for speech, a music understanding model [18] for musical attributes, an audio understanding model [2] for general sounds, and a visual model [19] for video information. Second, a large language model (LLM) [20] acts as an integration engine, synthesizing these multimodal cues to generate fine-grained audio captions. This synthesis of rich, cross-modal context by an LLM aims to improve detail and accuracy, addressing prior limitations.

Sensory

Input

Music Org.

Speech Org.

[Figure 5]

Audio Org.

Visual Org.

[Figure 6]

Fusion Org.

Acoustic

Understanding

Figure 1: Human auditory perception integrates multisensory cues.

Our contributions are:

- • Automated fine-grained audio captioning: A pipeline using specialized unimodal models to extract diverse contextual cues, synthesized by an LLM to generate detailed, scalable captions.
- • FusionAudio-1.2M dataset: A large-scale dataset of 1.2M fine-grained audio captions to advance audio research.

Table 2: Comparison of open-source audio caption datasets.

Name Year # of Audio/QA Avg. Dur (s) Avg. Text Len Visual Music Speech Integration

AudioCaps [5] 2019 46k/46k 10.00 9.03 ✗ ✗ ✗ ✗ Clotho [4] 2019 5k/5k 22.50 11.00 ✗ ✗ ✗ ✗ LAION-Audio-630K [6] 2022 630k/630k 24.58 7.30 ✗ ✗ ✗ ✗ WavCaps [7] 2024 403k/403k 67.59 7.80 ✗ ✗ ✗ ✗ AudioSetCaps [8] 2024 1.9M/1.9M N/A 28.00 ✗ ✗ ✗ ✗ Auto-ACD [9] 2024 1.5M/1.5M 10.00 18.10 ✓ ✗ ✗ ✓ CompA-R [2] 2024 62k/200k 9.93 18.00 ✓ ✗ ✗ ✓ FusionAudio-1.2M (Ours) 2025 1.2M/6M 10.00 47.18 ✓ ✓ ✓ ✓

• Multimodal cue-enhanced audio models: A CLAP-based audio encoder with improved audio-text alignment, and an instruction-tuned MLLM with stronger audio comprehension and instruction-following.

## 2 Related Works

- 2.1 Audio Language Learning

The field of audio-language models has seen significant advancements in recent years, with researchers focusing on developing models that can effectively process, understand, and reason about sounds using natural language as a supervision signal. Early works like CLAP Learning Audio Concepts from Natural Language Supervision [21] laid the foundation for contrastive learning approaches in audio-language pre-training. Subsequent studies have explored various pre-training objectives, including generative and discriminative methods, to enhance audio representation learning and cross-modal alignment. For instance, CTAL [22] and FLAP [23] have investigated masked language modeling and cross-attention based masked acoustic modeling for joint representation learning of audio and language modalities. Multi-task learning approaches have also gained attention, with models like UniAudio [24] and SpeechX [25] demonstrating the potential of unifying diverse audio tasks under a single framework. Furthermore, the integration of large language models (LLMs) with audio processing has opened new avenues for creating more powerful and human-like audio understanding systems. Recent works such as Pengi [26], Qwen-audio [27], and Audio Flamingo [28] have shown impressive capabilities in handling complex audio tasks through instruction tuning and in-context learning. These advancements highlight the growing importance of audio-language models in bridging the gap between auditory information and language understanding, and their potential for real-world applications.

- 2.2 Audio Captioning

Early audio captioning research relied on manually annotated datasets like AudioCaps [5] and Clotho [4], which provided high-quality descriptions but were inherently limited in scale. To address this, the field increasingly adopted automated and weakly-supervised methods. These leverage large-scale web-sourced audio with associated sparse metadata (e.g., WavCaps [7], LAION-Audio630K [6]), employ existing textual tags to guide generation, or incorporate basic multimodal cues from loosely associated content [8, 9, 10]. While significantly improving scalability, these automated techniques typically yield captions lacking the fine-grained detail and rich contextual understanding characteristic of human annotations or, as our work posits, achievable through more sophisticated, deeply integrated multimodal information processing. As shown in Table 2, Our FusionAudio-1.2M dataset offers finer-grained captions than existing audio caption dataset, see the average text length.

- 3 Method: Fine-grained Audio Caption with Multimodal Contextual Fusion

- 3.1 Automated Captioning Pipeline

We introduce a two-stage pipeline, illustrated in Figure 2, designed to generate fine-grained audio captions: (1) Multimodal Contextual Cue Extraction using specialized expert models, and (2) LLMDriven Contextual Synthesis to integrate these diverse cues into a coherent caption. An initial pre-processing step is performed to enhance audio quality.

Multimodal Contextual Cue Extraction

Speech

Vocal

ASR Model

LLM-Driven Contextual

[Figure 7]

[Figure 8]

[Figure 9]

Synthesis

Audio Audio Track

Audio Music

Audio Caption

Audio Caption

Integrated

Model

Separation

Caption

Music Caption

Audio Event

Music Caption

Audio

Quality Filter

Detection

Model

Video

Video Caption

Video Caption

Model

Final Caption

- Figure 2: Overview of our proposed multimodal audio captioning pipeline. The process involves initial vocal separation, followed by a two-stage approach: multimodal contextual cue extraction and LLM-driven contextual synthesis.

Pre-processing: Audio Track Separation. To enhance the quality and specificity of downstream analyses, particularly for speech and distinct background sounds, we first apply a source separation technique. We employ the Demucs model [29] to isolate the vocal track from the non-vocal components (e.g., music, environmental sounds) within the audio stream.

- Stage 1: Multimodal Contextual Cue Extraction. This stage leverages a suite of specialized models to extract diverse, complementary information streams relevant to the auditory scene. The prompts used for these models can be found in Appendix B.

- • General Audio Events: To capture overall acoustic scene characteristics, we utilize GAMA [2] to generate descriptive captions focusing on sound events and environments.
- • Speech Content: The separated vocal stream is transcribed using the Whisper model [17].
- • Music Characteristics: For clips potentially containing music, we first employ YamNet [30] as a classifier to confirm the presence of music, mitigating hallucination risk on non-musical segments. If music is detected, OpenMu [18] is used to extract details regarding genre, instrumentation, tempo, and mood.
- • Visually-Grounded Context: We utilize the Qwen2.5-VL-72B vision-language model [19] to extract visual information from the video stream. This approach yields a detailed, timestamped visual record, providing visual context that aids in grounding physical events.

- Stage 2: LLM-Driven Contextual Synthesis. The extracted information streams serve as input to the synthesis model, QwQ-32B [20]. The LLM acts as a central integration engine. It is prompted to: (a) synthesize the multimodal inputs coherently, (b) resolve potential redundancies or minor inconsistencies across the different expert outputs, (c) infer relationships and context implied by the combined information, and (d) generate a final, fine-grained audio caption that reflects a comprehensive understanding of the auditory scene enriched by multimodal context.

### 3.2 Data Source

We utilize the AudioSet dataset [31] as the primary source material. AudioSet provides over 2 million 10-second YouTube video clips, each weakly annotated with audio event labels. We downloaded the corresponding audio and video streams for processing through our pipeline.

### 3.3 Data Quality Assurance

To ensure the quality and reliability of the automatically generated captions, we implement a multifaceted quality assurance protocol. This process involved both manual verification on a sample of the data and scalable automated filtering (described subsequently) to curate the final FusionAudio-1.2M dataset.

Manual Verification. To establish a benchmark for caption quality, we randomly sampled 300 generated captions for human evaluation. Trained annotators assessed each caption based on two criteria: (1) Detailness: Rated on a 3-point scale, a higher score means more details, evaluating the richness and specificity of the information conveyed. (2) Hallucination: Rated on a 5-point scale,

0.200

FusionAudio

FusionAudio

- 0

- 1

- 2

- 3

0.175

AverageCount

AudioSetCaps

AudioSetCaps

Percentage

0.150

Auto-ACD

Auto-ACD

0.125

Narration, monologue

Sound-VECaps

Sound-VECaps

0.100

Vehicle

Guitar

0.075

1.9%

0.050

4.1%

4.3%

0.025

0.000

0 10 20 30 40 50 60 70 80

Instrument Emotion Music Genre

Caption Length

Categories

(b) Caption Length Distribution

(c) Diversity of Object Types

Music 55.7% 33.9% Speech

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

UsagePercentage

0.020

0.8

Percentage

0.015

0.6

0.010

0.4

0.005

0.2

0.0

0.000

Audio Video Speech Music

0.1 0.2 0.3 0.4 0.5 0.6

Caption Type

Clap Score

(a) Top 5 Audio Labels Distribution

(d) CLAP Score Distribution

(e) Modal Usage Frequency

- Figure 3: Key statistics of FusionAudio-1.2M: (a) Proportion of top 5 audio labels from AudioSet; (b) Caption length comparison with existing datasets; (c) Diversity of semantic content types; (d) Proportion of captions utilizing different modalities; (e) Distribution of audio-text similarity measured by CLAP.

a higher score means less hallucination, assessing the factual accuracy of the caption against the audio-visual content. A score of ≤ 2 was considered indicative of notable hallucination. The detailed annotation guidelines and scoring rubrics are provided in Appendix A.

As shown in Table 3, the manually evaluated sample achieved a mean detailness score of 2.55 (out of

- 3). For hallucination, the average score given by human evaluator is 3.74, with 7% of the evaluated captions received a score of 2 or less, indicating a low prevalence of significant inaccuracies in this sample. The inter-annotator agreement, calculated using the exact match rate, was 0.67 for detailness and 0.79 for hallucination. These rates suggest moderate agreement between annotators, which is reasonable given the subjective nature of fine-grained caption quality assessment. The full distribution of scores for both metrics can also be found in Appendix A.

Table 3: Manual Verification Results. Detailness is rated 1-3 (higher is better). Hallucination is rated 1-5 (higher is better; ≤ 2 indicates notable hallucination). IAA is measured using the exact matching, before which hallucination score has been converted to 1 (score ≤ 2) or 0 (score > 2).

Caption Content Quality Inter-Annotator Agreement Detailness Hallucination Detailness Hallucination

2.55 3.74 0.67 0.79

Automatic Filtering To scale quality assessment to the entire dataset, we leveraged the CLAP model to automatically filter low-quality captions. We computed cosine similarity between CLAP-generated audio and caption embeddings as a quality indicator. Based on our human evaluation, we categorized hallucination scores ≤ 2 as the positive class (captions to discard) and scores > 2 as the negative class (captions to retain). We then evaluated various cosine similarity thresholds using the F1.05 score, which slightly emphasizes recall to prioritize removing hallucinated content (see Appendix A.3 for computation details).

A threshold of 0.08 achieved optimal alignment with human judgments, with the exact match rate being 88.3%. This threshold resulted in a 7.3% filter rate and balanced false positives and negatives. We applied this validated threshold to filter the entire caption set, yielding the final 1.2 million high-quality captions in the FusionAudio-1.2M dataset.

4 The Resulted Dataset: FusionAudio-1.2M

- 4.1 Quantitative Analysis

- Table 2 compares our proposed dataset with other publicly available datasets. FusionAudio-1.2M distinguishes itself through its large scale, longer caption length, and integration of multiple modalities.

Dataset Statistics We analyze FusionAudio-1.2M across several dimensions:

- • Audio Category Distribution: Figure 3a shows the top five audio categories from AudioSet [31], with "Music" being the most prevalent.
- • Caption Length: Figure 3b compares caption lengths (in tokens) with AudioCaps [5], Sound-VECaps [10] and Auto-ACD [9]. FusionAudio-1.2M captions are significantly longer, indicating greater descriptive richness.
- • Semantic Diversity: To showcase and compare the richness of semantic information across different datasets, we identified the presence of instruments, emotions, and music genres in each caption using GPT-4o-mini (prompts in Appendix B.3). Figure 3c shows FusionAudio-1.2M has higher coverage across most categories.
- • Audio-Text Alignment: Figure 3d shows the distribution of cosine similarity between audio and text embeddings calculated by CLAP [1]. Samples of different similarity scores can be found in Appendix C.2.
- • Modality Usage: To better understand the contribution of different modalities to the final caption content, we use GPT-4o-mini to automatically annotate each caption for explicit references to different modalities: audio events, speech, music, and visual context (see prompt in Appendix B.3). As shown in Figure 3e, over 50% of samples integrate information from two or more modalities, demonstrating effective multimodal fusion.

### 4.2 Qualitative Analysis

40

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

30

t-SNEComponent2

20

10

- 0

10

20

30

40

60 40 20 0 20 40 60

t-SNE Component 1

(a) FusionAudio

40

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

30

t-SNEComponent2

20

10

0

10

20

30

40

60 40 20 0 20 40 60

t-SNE Component 1

(b) AudioSetCaps

40

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

30

t-SNEComponent2

20

10

0

10

20

30

40

60 40 20 0 20 40 60

t-SNE Component 1

(c) Auto-ACD

40

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

30

t-SNEComponent2

20

10

0

10

20

30

40

60 40 20 0 20 40 60

t-SNE Component 1

(d) Sound-VECaps

Music Speech Vehicle

Figure 4: T-SNE Embedding of popular categories between different datasets

Case Study To further highlight the qualitative improvements enabled by FusionAudio, Table 1 compares captions generated for the same audio clip across different datasets. FusionAudio’s caption not only describes the primary sound event but also integrates visual cues, inferred context, and emotional tone, demonstrating a level of detail and reasoning absent from prior datasets. More samples can be found in Appendix C.

Embedding Projection for Visualizing Semantic Granularity Embedding projection techniques like t-SNE [32] visually reveal a dataset’s semantic structure, illustrating intra-class compactness and inter-class separability—key indicators of data quality for discriminative tasks. We applied this to FusionAudio-1.2M by projecting CLAP sentence embeddings of its captions and those from baseline datasets into two dimensions using t-SNE. Figure 4 demonstrates that FusionAudio’s captions form significantly more compact same-category clusters and exhibit greater separation between different categories compared to baselines. This visual evidence indicates FusionAudio’s superior semantic granularity and discriminative power, beneficial for nuanced audio understanding and cross-modal retrieval. Quantitative validation of inter- and intra-class distances is in Appendix D.1.

## 5 Applications of FusionAudio-1.2M

We use FusionAudio-1.2M for two popular downstream tasks: audio-text retrieval in Sec. 5.1 and audio understanding in Sec. 5.2. All experiments were conducted on a server equipped with 8 NVIDIA A800 80GB GPUs. Fine-tuning for the main experiments and ablation studies on the 25K data subset typically saved a checkpoint in under 30 minutes per run. The full evaluation process across all benchmark tasks required approximately 5 hours to complete per model evaluation. For the scaling study involving LLM fine-tuning, training sessions lasted between 10 to 12 hours.

### 5.1 Audio-text Retrieval

Table 4: Audio-text retrieval performance (R@k, %) on the AudioCaps test set.

Text - to - Audio Audio - to - Text

Dataset Model

Avg.

R@1 R@5 R@10 R@1 R@5 R@10 AC+CL HTSAT+BERT 36.1 71.8 83.9 46.8 82.9 90.7 68.7 WavCaps HTSAT+BERT 42.2 76.5 87.1 54.6 85.2 92.4 73.0 AudioSetCaps HTSAT+BERT 43.4 78.4 88.2 57.3 84.2 93.2 74.1 Auto-ACD HTSAT+RoBERTa 42.7 - 88.5 56.3 - 93.9 Sound-VECaps HTSAT+RoBERTa 39.2 74.1 85.0 54.0 82.5 93.2 71.3 FA(Ours) HTSAT + BERT 44.3 79.9 90.4 57.8 86.1 94.4 75.5

### 5.1.1 Experimental Setup

Tasks and Models We evaluate the quality of FusionAudio-1.2M by assessing its effectiveness as a pre-training corpus for the downstream task of cross-modal audio-text retrieval. This task requires retrieving the most relevant audio clip for a given textual query (text-to-audio retrieval) and, conversely, identifying the most pertinent text description for a given audio input (audio-to-text retrieval). For all experiments, we employ the HTSAT [33]-BERT [34] model architecture.

Two-Stage Training Our training methodology for all evaluated datasets, including FusionAudio-

- 1.2M and the baselines, follows a consistent two-stage protocol:

- • Pre-training: The HTSAT-BERT model is first pre-trained on the entirety of the respective source dataset (e.g., FusionAudio-1.2M, WavCaps, etc.). This stage utilizes a contrastive learning objective. For pre-training, the learning rate is set to 5e-5, the batch size is 196,training proceeds for 15 epochs.
- • Fine-tuning: Subsequently, the pre-trained model undergoes full-parameter fine-tuning on the official training split of the AudioCaps (AC) dataset [5]. For this fine-tuning stage, we use a learning rate of 1e-5, a batch size of 196, and train for 20 epochs.

Evaluation Setting The performance of all models, after the two-stage training protocol, is evaluated on the official test set of the AudioCaps dataset [5]. We report Recall@k (R@k) for k={1, 5, 10} for both text-to-audio and audio-to-text retrieval directions. R@k quantifies the percentage of queries for which the ground-truth item is successfully retrieved within the top-k ranked results. The comparative results are presented in Table 4.

### 5.1.2 Performance Analysis

The detailed comparison results of model evaluation are presented in Table 4. The models trained on our dataset have significant advantages in the recall metrics of audio and text,which achieves the highest score in each R@k among models trained by existing audio caption datasets.The excellent audio and text recall performance indicates that the audio captions of FusionAudio-1.2M can accurately capture the information in the audio, ensuring the model’s ability to distinguish fine-grained information. As a result, it can achieve high-accuracy matching even when dealing with similar audio.

### 5.2 Audio Understanding

To empirically validate the practical utility and superior quality of our proposed FusionAudio-1.2M dataset, we evaluated its impact on a comprehensive suite of audio understanding tasks. Specifically, we benchmarked the performance of the GAMA model [2] fine-tuned on FusionAudio-1.2M against instances of the same model fine-tuned on several established audio captioning datasets.

- Table 5: Performance comparison of the GAMA model fine-tuned on FusionAudio-1.2M against baseline datasets across a battery of audio understanding evaluation benchmarks. All models were fine-tuned on 25,000 QA pairs. M.J. denotes Model-Judge score (using GPT-4.1-mini). Best scores are in bold. The three main categories of evaluation tasks align with those in Table 9.

Adverse Acoustic Conditions High-Level Semantic Understanding Fine-grained Information

Dataset

AS

US8k

###### TAU

FSDns (mAP) Avg.

Genre

MAQA

Mood

Mchat

SAQA

Schat

ABSc (M.J) Avg.

Vocal

Instr

###### ESC

FSD (mAP) Avg.

(Acc.)

(mAP)

(mAP)

(Acc.)

(Acc.)

(Acc.)

(M.J)

(Acc.)

(M.J)

(Acc.)

(Acc.)

(Acc.)

GAMA(base) 48.0 56.6 23.5 81.9 52.5 42.8 44.1 28.3 45.4 50.1 58.9 56.0 46.5 63.5 68.7 68.9 45.8 61.7 AC+CL 50.3 65.3 21.3 81.9 54.7 49.4 50.7 28.4 47.0 52.3 55.4 61.3 49.2 68.2 68.9 65.7 39.9 60.8 WavCaps 55.4 64.5 25.0 77.6 55.6 53.4 51.6 33.2 27.7 45.1 29.7 52.7 41.9 55.4 69.7 58.8 32.4 54.1 ASC 45.4 51.3 22.3 77.8 49.2 56.0 57.6 31.6 51.3 49.9 59.5 58.5 52.1 51.8 70.5 57.7 30.5 52.6 CompA-R 56.5 63.3 22.7 83.7 56.6 60.1 54.7 33.9 47.0 56.1 58.3 60.1 52.9 63.5 68.6 62.3 38.4 58.2

FA(ours) 59.0 58.8 24.4 84.6 56.7 65.1 57.6 35.7 57.1 59.1 61.5 64.5 57.4 69.0 73.6 65.5 44.5 63.0 FA-high(ours) 59.7 64.0 25.1 88.2 59.3 64.2 60.0 38.3 57.9 58.4 62.3 64.0 57.9 71.0 73.9 71.3 47.4 65.9

### 5.2.1 Experimental Design

Tasks and Models We focused on general audio understanding beyond speech, employing the GAMA model architecture [2], a transformer-based audio-language model, as our foundation for fine-tuning. Fine-tuning utilized a learning rate of 5e-5, a batch size of 128, and 2 training epochs. Evaluation utilized t=0.1 for inference.

Training The GAMA model was fine-tuned independently on several datasets: our FusionAudio1.2M and its high-quality subset FusionAudio-high (top 25k QA pairs selected for quality and diversity), alongside established datasets. A critical aspect was normalizing training data to 25,000 QA pairs across all datasets, ensuring performance differences primarily reflect data quality, not quantity. Notably, while baseline datasets typically required 25,000 unique audio clips (one QA pair per clip) for this volume, FusionAudio-1.2M achieves this with only 9,000 unique audio clips, owing to its design of multiple rich QA pairs per audio instance.

Evaluation Fine-tuned models were evaluated on 15 diverse audio understanding tasks (Table 5), assessing capabilities across three key scenarios: (1) robustness to Adverse Acoustic Conditions, (2) proficiency in High-Level Semantic Understanding, and (3) acuity in discerning Fine-grained Information. For benchmarks requiring automated judgment (M.J. scores in Table 5), we employed GPT-4.1-mini.

- 5.2.2 Performance Analysis The results in Table 5 demonstrate the significant advantages of fine-tuning with FusionAudio.

Dominant Performance Driven by High-Quality and Efficient Data GAMA fine-tuned on FusionAudio, and especially its FusionAudio-high subset, consistently outperformed models trained on all benchmarked datasets across the majority of the 13 tasks, with FusionAudio-high achieving the highest average scores in all scenarios. This success is rooted in the superior intrinsic quality of FusionAudio, crafted by our method to maximize information richness per audio clip. As a direct result, models can learn more effectively from each sample, leading to the crucial observation that this dominant performance was achieved using substantially fewer unique audio clips than other datasets. This clearly demonstrates that FusionAudio-1.2M not only provides higher-caliber data overall but also facilitates more efficient learning and utilization of each individual audio piece.

6 Ablation Study

- 6.1 On the Effectiveness of Multimodal Cues

To rigorously evaluate the contribution of each component in our method for enhanced audio information, we conducted a comprehensive ablation study. This study aims to (1) ascertain the individual importance of each auxiliary modality (Speech, Music, Video) in augmenting the Sound modality, and (2) validate the effectiveness of our proposed automatic filtering module.

Experiment Setup All ablation experiments were performed on the same subset from AudioSet with a scale of 25k, using the same audio clips and training procedures. FusionAudio-1.2M incorporates all four modalities (Sound, Music, Speech, Video) and includes the multi-modal fusion quality threshold filtering module. We compared FusionAudio-1.2M against several ablated variants.

Ablation Results on Fusion As shown in Table 6, ablating auxiliary modalities (Music, Video, Speech) generally degraded performance. Removing video captions (w/o Video) caused the most significant decline, underscoring visual context’s critical role. Ablating music (w/o Music) and speech (w/o Speech) also reduced performance. An interesting exception was observed for Task 1, where removing speech (w/o Speech) led to a slight improvement. We attribute this to a combination of potentially poor ASR transcription quality in adverse acoustic conditions, which could introduce detrimental noise, and a possible task focus shift where non-speech acoustic analysis is prioritized, making speech content less critical and potentially diverting optimization from core modalities. Notably, the magnitude of these performance drops (-0.76 for Music, -1.18 for Video, and -0.93 for Speech on average) generally corresponds with the usage of these modalities in our dataset, as illustrated in Figure 3e. This suggests that modalities more frequently leveraged for information contribute more significantly to the overall performance.

Ablation Results on Filtering Finally, removing our quality filtering module led to a consistent, significant performance drop across all tasks, highlighting its effectiveness in mitigating issues from hallucinations introduced during the process.

### 6.2 On the Effectiveness of Data Scaling

To assess the impact of data volume, we conducted scaling experiments for the downstream tasks. This study evaluates performance gains as data size increases, providing insights into model scalability.

Experiment Setup We used nested subsets, starting from 1.25K audio clips. The Audio Understanding task scaled to 80k clips (355k QA pairs), while Retrieval utilized up to the full 1.2M clips. Model architectures and training hyperparameters remained consistent with previous experiments.

- Table 6: Ablation Study on FusionAudio-25K Dataset. Performance metrics are shown for Retrieval tasks (Text-to-Audio and Audio-to-Text) and Understanding tasks (Task I: Adverse Acoustic Conditions; Task II: High-level Semantic Understanding; Task III: Fine-grained Information).

Settings

Retrieval Task Understanding Task

Avg.

T-A A-T Task I: AAC Task II: HSU Task III: FI FusionAudio-1.2M 39.70 49.71 56.73 57.16 63.02 53.26

w/o Music 39.03 47.53 56.72 56.34 62.87 52.50(-0.76) w/o Video 38.53 48.79 55.90 56.12 61.08 52.08(-1.18) w/o Speech 38.09 47.87 57.38 56.06 62.27 52.33(-0.93) w/o Filter 39.45 49.14 55.30 55.25 61.35 52.10(-1.16)

Results As depicted in Figure 5, increasing data volume consistently improved performance for both tasks. For Audio Understanding, scaling from 1.25K to 80k clips enhanced average performance, likely due to increased exposure to diverse audio and associated QA pairs generated with a natural distribution. The Retrieval task exhibited a substantial and consistent rise in Recall@1 with data expansion, peaking with the full dataset. These findings underscore that greater data volume generally boosts model capabilities, highlighting the value of our dataset’s scale and richness.

- 7 Conclusion

This paper presents FusionAudio-1.2M, a large-scale dataset for fine-grained audio captioning created via a novel multimodal contextual fusion pipeline. Inspired by human auditory perception, the approach combines specialized expert models for speech, music, sound events, and visual context with LLM-based synthesis. Experiments show that models trained on FusionAudio-1.2M achieve strong performance using fewer unique audio samples due to richer per-clip annotations. Ablation

58.5

80.0k

FA (355k QA)

20.0k (88k QA)

58.0

5.0k (22k QA)

57.5

Average

57.0

56.5

1.25k (5.5k QA)

56.0

20 21 22 23 24 25 26 27

Dataset Size

(a) Audio Understanding

75

FA A T FA T A

70

WavC AACD VEC

65

| |
|---|

60

FA-1200k

ASC A T ASC T A

FA-800k

R@1Score

FA-400k

ASC-1912k

FA-200k

55

FA-100k

ASC-956k

FA-50k

ASC-478k

50

ASC-191k

FA-50k FA-100k FA-200k FA-400k FA-800k FA-1200k

ASC-96k

ASC-48k

45

ASC-1912k

FA-5k

ASC-478k ASC-956k

40

FA-1.25k

ASC-191k

ASC-96k

FA-1.25k FA-5k

35

ASC-48k

103 104 105 106 107

Dataset Size

(b) Audio-text Retrieval

Figure 5: Scaling result of understanding and retrieval tasks. Details of the legend in (b): A: Audio; T: Text; FA: FusionAudio-1.2M; WavC: WavCaps; AACD: Auto-ACD; VEC: SoundVECaps; ASC: AudioSetCaps.

studies confirm the significance of each modality, particularly visual context. This work could be further improved by polishing the caption generation method, diversifying the dataset with longer audio clips, probing more sophisticated multimodal fusion techniques, and performing a deeper societal impact analysis, which we leave as future work.

## Ackownledgement

This work was supported by the Shenzhen Science and Technology Program (JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608), Shenzhen Science and Technology Program (Shenzhen Key Laboratory Grant No. ZDSYS20230626091302006), and Shenzhen Stability Science Program 2023, Shenzhen Key Lab of Multi-Modal Cognitive Computing.

## Limitation

The study also acknowledges several limitations. First, the automated generation of audio captions may introduce hallucinations or errors, despite quality assurance measures such as human evaluation and automatic filtering. Second, the dataset primarily focuses on short audio clips (10 seconds), which may limit its applicability to longer or more complex audio contexts. Third, the multimodal fusion approach relies on the integration of speech, music, visual, and general audio information, but the interplay and weighting of different modalities are not thoroughly explored. Fourth, due to computational resource constraints, we were unable to conduct multiple experimental runs to establish robust error bars for all reported metrics, which could provide further statistical confidence. Lastly, while our pipeline incorporates a quality filtering module to mitigate LLM-induced inaccuracies, completely eliminating potential hallucinations in automated data generation remains an ongoing challenge, suggesting a need for continued refinement in robust AI-driven data creation. Future work could address these limitations by further refining the caption generation process, expanding the dataset to include longer audio clips, exploring more nuanced multimodal fusion strategies, and conducting a more comprehensive analysis of societal impacts.

## References

- [1] Yusong Wu*, Ke Chen*, Tianyu Zhang*, Yuchen Hui*, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keywordto-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2023.
- [2] Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. GAMA: A large audio-language model with advanced audio understanding and complex reasoning abilities. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6288–6313, Miami, Florida, USA, November

2024. Association for Computational Linguistics.

- [3] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.
- [4] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset, 2019.
- [5] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- [6] LAION-AI. Laion-audio-630k dataset, 2023. Accessed: 2024-04-16.
- [7] Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D. Plumbley, Yuexian Zou, and Wenwu Wang. WavCaps: A ChatGPT-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing, pages 1–15, 2024.
- [8] Jisheng Bai, Haohe Liu, Mou Wang, Dongyuan Shi, Wenwu Wang, Mark D. Plumbley, WoonSeng Gan, and Jianfeng Chen. Audiosetcaps: An enriched audio-caption dataset using automated generation pipeline with large audio and language models, 2024.
- [9] Luoyi Sun, Xuenan Xu, Mengyue Wu, and Weidi Xie. Auto-acd: A large-scale dataset for audio-language representation learning, 2024.
- [10] Yi Yuan, Dongya Jia, Xiaobin Zhuang, Yuanzhe Chen, Zhengxi Liu, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xubo Liu, Xiyuan Kang, Mark D. Plumbley, and Wenwu Wang. Sound-vecaps: Improving audio generation with visual enhanced captions, 2025.
- [11] Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. Air-bench: Benchmarking large audio-language models via generative comprehension, 2024.
- [12] W. H. Sumby and I. Pollack. Visual contribution to speech intelligibility in noise. Journal of the Acoustical Society of America, 26:212–215, 1954.
- [13] Christoph Kayser, Nikos K. Logothetis, and Stefano Panzeri. Visual enhancement of the information representation in auditory cortex. Current Biology, 20(1):19–24, 2010.
- [14] Marc O. Ernst and Heinrich H. Bülthoff. Merging the senses into a robust percept. Trends in Cognitive Sciences, 8(4):162–169, 2004.
- [15] Albert S. Bregman. Auditory scene analysis: The perceptual organization of sound. The MIT Press, 1990.
- [16] S. A. Shamma, M. Elhilali, and C. Micheyl. Temporal coherence and attention in auditory scene analysis. Trends in Neurosciences, 34(3):114–123, Mar 2011. Epub 2010 Dec 31.

- [17] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022.
- [18] Mengjie Zhao, Zhi Zhong, Zhuoyuan Mao, Shiqi Yang, Wei-Hsiang Liao, Shusuke Takahashi, Hiromi Wakaki, and Yuki Mitsufuji. Openmu: Your swiss army knife for music understanding. arXiv preprint arXiv:2410.15573, 2024.
- [19] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [20] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [21] B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang. Clap learning audio concepts from natural language supervision. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 1–5. IEEE, 2023.
- [22] H. Li, Y. Kang, T. Liu, W. Ding, and Z. Liu. Ctal: Pre-training crossmodal transformer for audio-and-language representations. arXiv preprint arXiv:2109.00181, 2021.
- [23] C.-F. Yeh, P.-Y. Huang, V. Sharma, S.-W. Li, and G. Gosh. Flap: Fast language-audio pretraining. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.
- [24] J. Tian, H. Dongchao Yang, et al. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704, 2023.
- [25] X. Wang, M. Thakker, Z. Chen, N. Kanda, S. Eskimez, M. Chen, S. Tang, J. Liu, T. Li, and T. Yoshioka. Speechx: Neural codec language model as a versatile speech transformer. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
- [26] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An audio language model for audio tasks. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 18090–18108. Curran Associates, Inc., 2023.
- [27] Y. Chu, J. Xu, X. Zhou, S. Yang, Z. Zhang, C. Yan, and J. Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.
- [28] Z. Kong, A. Goel, R. Badlani, W. Ping, R. Valle, and B. Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. arXiv preprint arXiv:2402.01831, 2024.
- [29] Simon Rouard, Francisco Massa, and Alexandre Défossez. Hybrid transformers for music source separation. In ICASSP 23, 2023.
- [30] TensorFlow. Yamnet: Audio event classification. https://github.com/tensorflow/ models/tree/master/research/audioset/yamnet, n.d. Accessed: 2025-04-19.
- [31] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017.
- [32] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008.
- [33] Ke Chen, Xingjian Du, Bilei Zhu, Zejun Ma, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Htsat: A hierarchical token-semantic audio transformer for sound classification and detection. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 646–650. IEEE, 2022.

- [34] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

## A Human Evaluation

- A.1 Evaluation Setup

We recruit five evaluators to assess the data. All evaluators are students with a bachelor’s degree or higher and have studied in an English-only teaching environment. The five evaluators are tasked with evaluating a total of 300 samples. Each evaluator is assigned 120 samples, ensuring that each sample is evaluated twice by different evaluators.

Evaluators are required to score the captions based on two dimensions: the level of detailness and the degree of hallucination.

- • Detailness: Evaluating the level of detail, specificity, and contextual information provided in the caption regarding the audio events and scene. Captions describing multiple relevant aspects accurately scored higher. Detailness is scored through 1-3.
- • Hallucination: Assessing the accuracy of the description against the source audio-visual content. This specifically penalizes hallucinated objects, events, or attributes not perceivable in the clip. Hallucination is scored through 1-5.

Specific scoring guidelines can be found in the Appendix A.2.

- A.2 Instruction for Human Evaluation The instruction used for human evaluation is shown in Figure 6.
- A.3 F-Score Computation

To balance precision and recall in our automatic filtering process, we used the F1.05 score, which slightly emphasizes recall over precision. This emphasis ensures that captions with high hallucination

rates are effectively discarded, even at the cost of filtering out some acceptable ones. The F1.05 score is calculated using the formula:

F1.05 =

(1 + 1.052) · Precision · Recall (1.052 · Precision) + Recall

Where precision and recall are computed from the confusion matrix as:

Precision =

TP TP + FP

and Recall =

TP TP + FN

- A.4 Human Rating Distribution

We statistically analyze the distribution of human ratings for detailness and hallucination,which are shown as Figure 7.

## B Prompt for models

- B.1 Video Caption Prompt

The prompt we used for Qwen2.5-VL-72B to extract video caption is shown in Figure 8. We try to let the model describe sound-related object only, but found that it would introduce additional hallucinations. Thus, we prompt the model to describe visual content only, and let the integration model tackle the modality issue.

- B.2 Audio Caption Prompt The prompt we used for GAMA to extract audio caption is shown in Figure 9.

Instruction for Human Evaluation

## Introduction

You are tasked with evaluating captions generated for audio clips. Please use the following guidelines to assess each caption based on two indicators: Detailing and Hallucinations

## 1. Detailing Key Things to Look For:

- • Whether the caption captures all major sounds and events in the audio (e.g., dog barking, doorbell ringing, etc.).
- • If the intensity or emotional context of the sound is conveyed (e.g., the dog barking intensely or the doorbell ringing in a rapid succession).
- • Whether the caption includes additional information when relevant (e.g., a dog barking repeatedly or distressed).

Scoring Guidelines:

Categorize captions into three detail levels (high, medium, low)based on their coverage of audio elements.

- • Low: Only generic descriptions without specific elements
- • Medium: Identifies main elements but lacks contextual details
- • High: Specifies sound sources, qualities, and relationships

## 2. Hallucinations

You will be given the highlighted words or phrases marked by DeepSeek-V3 that need to be verified in the original caption: 2 A [male voice] delivers a [scripted narration] [in Polish], likely from a [recorded radio or podcast segment], accompanied by [subtle studio ambiance] including [microphone hiss] and [paper rustling]. A [secondary listener] [wearing headphones] remains [audibly inactive], though [faint page-turning sounds] indicate [preparatory material review]. The spoken text references [program materials available at Lechia.net], suggesting a [structured broadcast format] with [editorial oversight]. Background contains [minimal environmental noise] consistent with a [sound-treated recording space].

Total flagged phrases: 17 Note: The total number of flagged phrases is provided for reference. If you believe other words or phrases are important in the context of the verification, please consider them in your calculation as well.

### Your Task

- • Listen to the audio and verify the highlighted elements.
- • Assign one of the following error values to each phrase:

|Label|Criteria|
|---|---|
|Correct (0)<br><br>|Directly verifiable from audio|
|Unverifiable (0.5)<br><br>|Neither confirmed nor disproven, or things you are not sure|
|Hallucination (1)|Contradicts audio or invents content|

### Scoring Calculation:

The final hallucination rate is calculated as follows:

(Error Values)

Hallucination Rate =

Total Content Units × 100% Based on the hallucination rate, assign a final score as follows: 0-10%: Score = 5 | 11-25%: Score = 4 | 25-40%: Score = 3 | 41-50%: Score = 2 | 51-100%: Score = 1

Figure 6: Instruction for Human Evaluation.

0.3

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Proportion

0.2

0.1

0.0

1.0 1.5 2.0 2.5 3.0

Detailness Rates Distribution

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.20

0.15

Proportion

0.10

0.05

0.00

1 2 3 4 5

Hallucination Rates Distribution

Figure 7: Detailness and Hallucination Rates Distribution of Human Rating

- B.3 Object Extraction Prompt

The prompt for asking GPT-4o mini to obtain instruments, emotions, and music styles from audio is shown in Figure 10.

- B.4 Modal Information Check Prompt The prompt we used to check if the modal information is used during the fusion is shown in Figure 11.

## C Dataset Samples

The code and dataset of this paper can be found in https://github.com/satsuki2486441738/ FusionAudio.

### C.1 Case study

- Table 7 presents representative FusionAudio captions, annotated with their information sources. These examples illustrate FusionAudio’s ability to synthesize and reason across modalities, generating descriptions that go beyond mere aggregation to provide holistic, context-rich interpretations.

C.2 Samples of different CLAP score

- Table 8 presents the hallucination situations of FusionAudio captions within different CLAP similarity intervals.

Table 7: Example of FusionAudio caption generation with annotated information sources.

Audio Caption The audio is characterized by the sound of brass instruments, possibly

trumpets, playing music. Speech Caption (empty)

Music Caption This is an instrumental marching band piece. The brass section plays the melody, while bagpipes provide a counter-melody. The mood is optimistic and cheerful. This piece may be used as background music for a national patriotic event.

Video Caption Chronological overall description:

- - 00:00: The video opens with close-ups of musicians, highlighting their focused expressions and the shiny brass instruments. The musicians are seated in a row, each holding a saxophone or trombone, preparing to play.
- - 00:01: As the conductor steps forward facing the musicians, a blue-andwhite striped shirt appears in the frame. The conductor’s gestures begin to guide the musicians as the performance starts.
- - 00:02 - 00:10: The camera remains relatively still, capturing the musicians playing their instruments. The conductor’s hand movements synchronize with the music, guiding the ensemble. The musicians’ fingers move rhythmically on the keys and valves, producing harmonious sounds. The conductor continues to lead, ensuring cohesion.
- - 00:07 - 00:10: The camera pans slightly right, revealing more trombone players at the far right. The additional musicians add depth to the ensemble, enhancing the overall sound and visual impact. The video ends with the musicians continuing to play under the conductor’s direction.

Reasoning Content The audio caption mentions "trumpets," but the music and video captions indicate a brass ensemble with saxophones, trombones, and bagpipes. The music caption is more reliable for instrument identification, so the final description should prioritize saxophones, trombones, and bagpipes.

There is no speech content, which matches the empty speech caption. The mood is cheerful and ceremonial, suitable for a patriotic event, as inferred from the music caption.

Potential ambiguities include confusion about the specific types of brass instruments and the presence of bagpipes, as well as uncertainty about whether the performance is indoors or outdoors.

... (Further details omitted for brevity)

Final Caption A lively instrumental piece performed by a brass ensemble, with saxophones and trombones carrying the main melody and bagpipes providing a contrasting counter-melody. The music maintains the rhythmic precision characteristic of a formally conducted ensemble, featuring the harmonious and dynamic qualities typical of patriotic or ceremonial performances. No prominent vocal content.

Table 8: The demonstration of the hallucination which is marked in red of audio captions within different clap similarity intervals

Clap Similarity Intervals

Audio ID Caption

- 0.0-0.1 -wyJ2cab4ic

A speech with strong tonal urgency is delivered, accompanied by persistent breathing sounds and faint intermittent background activity suggesting an indoor environment. The speaker’s vocal cadence appears strained, potentially reflecting either passionate delivery or underlying emotional tension.

- 0.1-0.2 -4t1LMiiHp4

A clear male speech is delivered with a strong vocal presence, accompanied by dynamic acoustic drums, a groovy bassline, and intermittent tambourine shakes in the background. Sporadic applause and crowd cheering weave throughout the speech, creating an energetic and engaged atmosphere. The musical elements maintain a steady rhythmic foundation while the vocal delivery appears deliberate and focused.

- 0.2-0.3 04Q_WeM7VIU

Continuous music with a groovy bass line, percussive drum patterns, keyboard harmonies, and synth brass melodies is heard in a lively setting. Intermittent male speech occurs in an upbeat tone, overlapping with the music’s rhythmic elements. The recording exhibits mono audio and background noise, suggesting a live performance environment with frequent equipment adjustments and energetic vocal exchanges.

- 0.3-0.4

CCsZneHL6s

A solo violin performs a slow, emotive melody with a smooth bowing technique, accompanied by steady rhythmic percussive sounds suggesting a handpan or similar instrument. The performance takes place in an indoor environment with subtle background reverberation, indicative of a studio or concert space. The audio quality is slightly degraded, but the interplay between the sustained violin tones and precise percussive elements creates a harmonious, intimate atmosphere.

- 0.4-0.5 -EKjvd8q_A0

The audio features a lively and energetic performance with rhythmic maracas, congas, and an accordion, accompanied by a saxophone adding depth. The upbeat tempo and festive soundscapes suggest a cultural celebration or live musical event.

- 0.5-0.6 00Twebqicmo

The audio is dominated by powerful car engine revving and acceleration sounds, accompanied by continuous background music. The combination of loud mechanical noises and energetic musical accompaniment creates a high-intensity atmosphere characteristic of an automotive event. Intermittent engine echoes suggest open-air acoustics typical of a racetrack or exhibition setting.

#### Prompt for video caption

Prompt: Please provide a comprehensive video description focusing exclusively on observable visual elements, including timestamps:

**1. Key Entities & Actions with Timestamps:**

- - List main objects/subjects and their visible actions with approximate timestamps (MM:SS format)
- - Describe:

- * Object/subject movements and interactions
- * Material properties (metal, wood, liquid)
- * Timing of significant visual events
- **2. Scene Description with Timeline:**

- - Overall scene dynamics and visual interactions
- - Notable visual events with timestamps:

- * Object collisions or impacts
- * Movement patterns
- * Material changes
- * Human/animal visible actions

- Environmental context (indoor/outdoor, spatial relationships)

- **3. Overall Description with Chronological Flow:**

- - Provide a comprehensive visual narrative of the video
- - Include timestamps for key moments and transitions (MM:SS format)
- - Focus on observable actions, and movements
- - Use specific, action-oriented language
- - Present events in chronological order with clear time markers Guidelines:
- - Describe only directly visible elements
- - Focus on observable actions and movements
- - Note material properties and physical interactions
- - Include **timestamps** for all significant events
- - Timestamp **should not** exceed the duration of the video
- - Use precise descriptive language for visual elements
- - Avoid assumptions about non-visible elements
- - Maintain strict focus on visual information

Example: Instead of "A car’s engine roars as it accelerates" Write "00:01 - A red sports car with chrome detailing accelerates down a paved road, tires creating visible spray on wet asphalt"

- "00:02 - The car’s rear suspension compresses during acceleration, exhaust emitting visible vapor"
- "00:03 - The car’s engine roars as it accelerates"

Figure 8: Prompt for video caption.

### C.3 Situations where multimodal contextual cues work

Our multimodal approach is designed to excel in challenging audio understanding scenarios (Table 9), such as interpreting audio in adverse conditions, achieving high-level semantic understanding (e.g., nuanced music interpretation), and enabling fine-grained acoustic entity recognition. Addressing these scenarios highlights the benefits of comprehensive multimodal integration.

### C.4 Samples of different sub-scenario

Table 10 shows the example dataset for each sub-scenario and corresponding example samples.

#### An example prompt for audio caption generation

Describe the audio in detail, but there is not need for association or speculation.

Figure 9: An example prompt for audio caption generation

An example prompt for extracting objects from audio

I will give you a sentence. Please extract some information I need in a JSON format. Sentence: ’caption’ My requirement:

- 1. Extract instruments and return as a list
- 2. Extract emotions and return as a list
- 3. Extract music genres and return as a list
- 4. Extract scenes and return as a list
- 5. All words must be found in the sentence.
- 6. Return a JSON format without any other words.
- 7. Words must be extracted from the corresponding caption.

The return format should only be like this:

{

"instrument": [], "emotion": [], "music genre": [], "scene": []

}

Figure 10: An example prompt for extracting objects from audio.

## D More on Dataset Statistics

### D.1 Embedding Space Quantitave Analysis

Table 11 presents a comprehensive comparison of inter- and intra-category embedding distances across different datasets. The analysis focuses on three key audio categories: Music (M), Vehicle (V), and Speech (S). Our proposed FusionAudio dataset demonstrates superior performance across all metrics. For inter-category distances, where higher values indicate better category separation, FusionAudio achieves significantly larger distances between different audio types (M-V: 0.7230, M-S: 0.5369, V-S: 0.5943) compared to competing datasets. This indicates that our dataset enables models to learn more discriminative representations that effectively distinguish between different audio categories. Simultaneously, FusionAudio exhibits smaller intra-category distances (Music: 0.8084, Vehicle: 0.7406, Speech: 0.8204), reflecting greater consistency within each category. The substantial improvement in both metrics—maximizing inter-category separation while minimizing intra-category variation—confirms that FusionAudio produces more cohesive and well-structured embedding spaces. This balance is crucial for downstream tasks such as audio classification, retrieval, and generation, as it facilitates more accurate identification and characterization of audio content while maintaining the nuanced variations within categories.

Shortened Prompt for Modal Integration Check

"You analyze descriptions from audio. ’final_cap’ is a comprehensive summary. Identify source captions (’audio_caption’, ’speech_caption’, ’music_caption’, ’video_caption’) essential for ’final_cap’ using provided JSON data: {cap_str}.

Requirements:

- 1. List contributing caption types.
- 2. Return as string keys list.
- 3. Format: [’type1’, ’type2’]"

Figure 11: Concise prompt for modal info check

Examples for audios with different clap scores.

Here we show the severity of hallucinations in audio captions under different clap similarity intervals. The red - marked parts are the hallucinatory parts of the audio captions.

Figure 12: Examples for audios with different clap scores.

Table 9: Key use-case scenarios where integrating multimodal contextual cues can significantly improve audio captioning. Challenges are listed per sub-scenario. Representative datasets and samples are detailed in Appendix C.4.

Scenario Sub-Scenario Key Challenges

Scene Recognition in Complex Soundscapes

High inherent acoustic complexity; Interwoven multi-source information; Background noise

Adverse Acoustic Conditions

Acoustically Degraded Conditions

Recording device limitations; Synthetic Artificial noise interference

Musical Genre Analysis; Emotional Expression; Artistic Intent; Aural Narratives

High-Level Semantic Understanding

Music Understanding

Sound Understanding Sound Implied Information; Attributes Inference Fine-grained Information Recognition

Acoustic Entity Recognition

Subtle acoustic cue discernment

An example prompt for multi-choice questions

Prompt: Please provide a comprehensive video description focusing exclusively on observable visual elements, including timestamps:

**1. Key Entities & Actions with Timestamps:**

- - List main objects/subjects and their visible actions with approximate timestamps (MM:SS format)
- - Describe:

- * Object/subject movements and interactions
- * Material properties (metal, wood, liquid)
- * Timing of significant visual events
- **2. Scene Description with Timeline:**

- - Overall scene dynamics and visual interactions
- - Notable visual events with timestamps:

- * Object collisions or impacts
- * Movement patterns
- * Material changes
- * Human/animal visible actions

- Environmental context (indoor/outdoor, spatial relationships)

- **3. Overall Description with Chronological Flow:**

- - Provide a comprehensive visual narrative of the video
- - Include timestamps for key moments and transitions (MM:SS format)
- - Focus on observable actions, and movements
- - Use specific, action-oriented language
- - Present events in chronological order with clear time markers Guidelines:
- - Describe only directly visible elements
- - Focus on observable actions and movements
- - Note material properties and physical interactions
- - Include **timestamps** for all significant events
- - Timestamp **should not** exceed the duration of the video
- - Use precise descriptive language for visual elements
- - Avoid assumptions about non-visible elements
- - Maintain strict focus on visual information

Example: Instead of "A car’s engine roars as it accelerates" Write "00:01 - A red sports car with chrome detailing accelerates down a paved road, tires creating visible spray on wet asphalt"

- "00:02 - The car’s rear suspension compresses during acceleration, exhaust emitting visible vapor"
- "00:03 - The car’s engine roars as it accelerates"

Figure 13: An example prompt for multi-choice questions.

#### Prompt for integration

Prompt: Rigorous Multimodal Information Integration and Purely Audio Description Expert

##### Core Task

You are an expert specializing in audio information processing. Your goal is to: integrate and analyze textual descriptions from multiple modalities as input, perform cross-referencing and correction while strictly controlling cross-modal information interference, and ultimately generate a description that is purely about the audio content, accurate, detailed, and fluently written in English, annotating potential ambiguities based solely on auditory perception. It is strictly prohibited to include any visual information, specific speech dialogue content, or ambiguity annotations based on audio-visual inconsistencies in the final output.

##### Input Information Sources (May contain errors, hallucinations, or be incomplete)

- • Audio Tags: A set of sound category tags annotated by humans, along with their corresponding quality estimations (confidence scores). Represents the most prominent human-perceived acoustic features in the audio. These tags are highly reliable, especially those with high percentages, but may not comprehensively cover all information in the audio. The format is TagName(Percentage%). e.g., Speech(100%). If empty, it indicates no human-annotated tag information is available.
- • Audio Description: A textual description of the audio content (may include sound events, ambient sounds, music, vocal characteristics, etc.). This is an important basis for describing audio facts and needs to be cross-validated with tags and music descriptions.
- • Speech Content: The textual result from Automatic Speech Recognition (ASR). This information is used only to confirm the presence of human voice, determine general vocal characteristics (e.g., speech vs. non-linguistic sounds, presence of distinct emotions [non-content related]), and assist in inferring possible scenarios or event backgrounds. Its specific textual content (including paraphrasing or summarization) must never appear in the final output. If empty, it indicates no distinct human voice, or other non-linguistic vocalizations (e.g., gasping, crying, background babble).
- • Music Description: A description of musical elements (features, instruments, rhythm, etc.) and other sound scenes. Music-related features herein are highly reliable. If empty, it indicates no distinct music. Other non-music descriptions (e.g., environment, human voice) have lower priority and primarily depend on "Audio Tags", "Audio Description", and "Speech Content" for judgment.
- • Video Description: A textual description of the video frames. Used only under specific conditions (see "Active Correction" in Processing Steps, step 2) to actively assist in identifying auditorily ambiguous sound sources, and to identify inconsistencies with auditory information (this inconsistency is only an internal decisionmaking flag for the model, not used to generate the output ambiguity list). Never speculate or describe the source, location, or on-screen actions of sounds based on video information itself. If empty, it indicates a lack of visual auxiliary information.

Processing Steps Please strictly follow the steps below:

##### 1. Multimodal Information Parsing:

- • Separately interpret each input description to extract core sound events, sound source characteristics, environmental ambiance, and musical elements.
- • Specifically parse "Audio Tags" to extract tag names and their confidence scores.
- • Special Note: From "Speech Content" (ASR results), primarily determine if human voice is present and its non-content features. In conjunction with its textual content (used only for auxiliary understanding), assist in inferring possible environments, emotional tones, or types of acoustic events, but never judge speaker gender, age, or other personal characteristics based on ASR content, and never quote, paraphrase, or summarize the specific textual content.

Figure 14: Prompt for integration.

##### 2. Auditory Fact Determination and Cross-modal Correction:

- • Initial Determination of Auditory Facts: First, based on "Audio Tags" (especially high-confidence tags, which have the highest priority for determining the types of sounds included in the tags), "Audio Description", "Music Description" (especially the music part), and "Speech Content" (presence of human voice and inferred characteristics), preliminarily determine auditorily perceived sound events, sound sources, ambient sounds, and music features. Identify and attempt to correct contradictions within these audio information sources (tags, audio description, music description, ASR inference), with the priority rule: High-confidence "Audio Tags" > Music part of "Music Description" ≈ "Audio Description" > "Speech Content" (presence of human voice) > Low-confidence "Audio Tags" > Non-music part of "Music Description".
- • Cross-modal Validation and (Conditional) Active Correction (for video information): After the initial determination of auditory facts, introduce "Video Description" for cross-validation. Its role is:

- – Active Correction (when audio information is ambiguous and video provides clear evidence): If the initially determined auditory fact (based on audio information sources) describes a general sound type that could have multiple auditory interpretations (e.g., a rumbling sound, a clicking sound, a rustling sound), and the "Video Description" clearly shows an object or event that is highly relevant to this general sound type and is a plausible sound source (e.g., the video clearly shows an airplane making a rumbling sound, or a person clicking a mouse making a clicking sound, or clothes/fabric in motion making a rustling sound), then the information provided by the video should be adopted to more precisely identify the general sound as a specific source or type (correcting rumbling to airplane sound, clicking to mouse click, rustling to fabric rustle). Note: If a high-confidence tag in "Audio Tags" already clearly indicates the specific sound type, then this sound is no longer considered a ’general sound type with multiple auditory interpretations,’ and this active correction step no longer applies to this sound. Under these limited and clear conditions, video information is used to enhance the understanding of audio facts, making the description more precise.
- – Identifying Inconsistencies or Lack of Corroboration (when video cannot clearly corroborate or conflicts):

- * If the sound event described by the initially determined auditory facts does not have a clearly corresponding visual sound source in the "Video Description", or if the visual information is inconsistent with or contradicts the perceived location or state of the sound source, then video information must never be used to negate or modify known auditory facts. In such cases, the model should internally flag the presence of an audio-visual inconsistency or lack of visual corroboration. This flag is only used in subsequent steps to adopt conservative wording when generating the final audio description and must never directly generate an ambiguity entry for output.
- * It is strictly prohibited to speculate, describe, or alter judgments about the sound event itself based on video information that cannot corroborate the audio (e.g., hearing a rumbling sound, the video shows the sky, but one cannot speculate it’s an airplane sound unless the video explicitly shows an airplane).

- • Determine Corrected Auditory Facts: Based on the results of the above multimodal cross-validation, determine the final auditory facts. The priority rule is listed above. When cross-modal information conflicts, audio information sources conflict internally, or there is high uncertainty (especially a lack of highconfidence tags or clear video corroboration for audio) making it difficult to determine auditory facts, the determined facts should reflect extreme conservatism, preferring to omit uncertain information rather than speculating based on non-auditory information. The model should internally retain a flag for the uncertain origin of audio information (e.g., whether it’s due to a lack of high-confidence tags, lack of support from audio description, or lack of video corroboration), to generate appropriately conservative descriptions in step 5.
- • Emotion Inference and Correction: If the emotion of a sound event (e.g., human voice, whose emotion can be inferred with ASR content assistance) conflicts with the emotion of background music, a comprehensive judgment must be made to provide the most likely primary emotional tone, but this is still based on auditory and ASR-assisted inference, without introducing visual information.

##### 3. Purely Auditory Ambiguity Reasoning and Annotation:

• Focus Solely on Pure Audition: Based on the determined auditory facts (which have considered tags and correction results), sound characteristics, common possibilities of auditory confusion, and potential auditory understanding biases in the original audio description, infer potential auditory understanding ambiguities that can be perceived or reasonably inferred solely through hearing.

Figure 15: Prompt for integration Cont.

- 4. • Sources of Ambiguity:

- – Auditory Similarity or Vagueness of the Sound Itself: Some sounds may be auditorily similar to others and easily confused (e.g., vehicle sound vs. airplane sound, typing sound vs. light tapping sound). The sound’s own quality, distance, or reverberation can also lead to vagueness or difficulty in determining the source.
- – Polysemy of Auditory Association: A sound event may reasonably correspond auditorily to multiple different sound sources or situations (e.g., a “bang” can have multiple causes, footsteps might come from multiple people).
- – Potential Purely Auditory Biases in the Original Audio Description: If, after multimodal correction, the original "Audio Description" is found to have incorrect or imprecise judgments about sound events or sources (and this error/imprecision is not caused by audio-visual inconsistency but by potential misinterpretations of audition itself), one should infer what common purely auditory misinterpretations the original description might have been based on.

• Strictly Exclude Non-Auditory Information as a Source of Ambiguity: Ambiguity annotation must only revolve around pure auditory perception and the associations arising therefrom. It is absolutely not allowed to use audio-visual synchronization, the way sound sources are presented on screen, or any visual content as the source or descriptive content of an ambiguity.

- 5. Information Reliability Assessment and Final Output Decision:

- • In this step, based on the analysis and correction results from steps 1-3, comprehensively assess the reliability and completeness of the determined auditory facts. In particular, consider whether high-confidence audio tags support key sound events.
- • If it is judged that the determined auditory facts are extremely scarce, various audio information sources (tags, audio description, music description, ASR inference) severely conflict and auditory facts cannot be reliably reconstructed, or even if tags exist but their confidence is generally very low and contradicts other information, the model directly outputs the unique specific string UNCERTAIN_AUDIO_INFORMATION_DETECTED.
- • Otherwise (if the determined auditory facts are sufficiently reliable and complete), proceed to the next step (generating JSON).

- 6. Generate Final Pure Audio Description (Audio Caption):

- • Execute this step only after passing the reliability assessment in step 4.
- • Pure Audio Focus: Generate a fluent, accurate, detailed, and concise English audio description. Describe only what can be heard and its purely auditory characteristics (e.g., sound source type [prioritizing those confirmed by high-confidence tags or clearly identified through active video correction], nature of sound events, type of ambient sound, music features, non-content features of human voice, spatial sense, loudness, timbre, duration, rhythm, etc.).
- • Integration and Augmentation: Integrate all valid auditory facts determined after multimodal correction (including those from audio tags, audio description, music description, ASR inference, and sound source types actively corrected via video). Supplement necessary auditory details of the scene (e.g., indoor/outdoor inferred from ambient sounds). If the model has internally flagged uncertainty in the audio information (e.g., lack of high-confidence audio tags supporting key sound events, original audio description being auditorily vague and lacking clear video corroboration, or internal conflicts within audio information sources), the final description must reflect this uncertainty, but through cautious wording to describe the perceived sound itself, rather than directly stating the uncertainty or vagueness. Use phrases like “sounds like,” “appears to be,” “potentially,” “suggests,” “a sound resembling X is heard” to express identification of less certain sound sources or events. Crucially, avoid sentences that explicitly state an inability to determine something or that something is ambiguous (e.g., do not say “the source cannot be determined” or “it is ambiguous whether X is present”). Instead, directly omit highly uncertain details or use cautious wording for what might be perceived.
- • Objective and Accurate: Base inferences on determined auditory facts, avoiding subjective speculation and over-extension. The description content must be supported by input text. Prohibit the introduction of irrelevant “new information,” unless it is reliable auditory inference based on multiple audio information sources (e.g., inferring the scene from ambient sounds). Ensure the description integrates facts confirmed by high-confidence tags, but never mention the confidence percentages themselves.

Figure 16: Prompt for integration Cont.

- • Cultural/Emotional Cues: If the sound contains clear cultural symbols or strong emotions, these can be briefly cued, but must be based on input audio evidence (e.g., emotion in human voice inferred from ASR, or emotion reflected by music features).
- • Final Check: Ensure this description absolutely contains no visual elements (objects, colors, actions, visual scenes, etc.). Even if the sound source type has been determined through high-confidence tags or active video correction, never describe the visual location, visual form, or specific on-screen behavior of that sound source. Absolutely prohibit the output of any specific speech text content (quotation, paraphrase, summary).

Output Format Requirements For most cases (i.e., when passing the reliability assessment in step 4), please strictly generate structured English output in the following JSON format (without any other explanations). However, in the special case of “scarce/unverifiable information” defined in step 4 of the processing flow, the model should directly output the predefined string UNCERTAIN_AUDIO_INFORMATION_DETECTED instead of JSON.

{

"Potential ambiguities": [ // List potential ambiguities based purely on auditory perception (English sentences). Does not include ambiguities requiring visual information to understand, nor ambiguities based on audio-visual inconsistencies.

- "Ambiguity description 1 based solely on auditory perception.",
- "Ambiguity description 2 based solely on auditory perception.",

...

], "Audio caption": "Final audio description focusing solely on audible elements and their auditory characteristics, detailed and fluent English. Use conservative language when audio facts are uncertain based on internal assessment." // Final pure audio description (concise and clear English sentence)

}

### Key Considerations:

- • Output Language: English.
- • Ignore Empty Inputs: If a modal description is empty, ignore that information source.
- • Strictly Prohibited: Including any visual information (objects, colors, actions, visual scenes, visual location/form/on-screen behavior of sound sources, audio-visual synchronization, etc.) in the final output (including Audio caption and Potential ambiguities). Even when dealing with audiovisual inconsistencies or unknown sound sources, never speculate, describe, or mention any visual content in the output.
- • Strictly Prohibited: Including any specific speech text content (quotation, paraphrase, summary, etc.) in the final output. Speech information is only used to infer the presence of human voice, non-content features, and to assist in understanding the scene ambiance.
- • Maintain Objectivity: Base inferences on determined auditory facts, avoiding subjective speculation and over-extension. Information not supported by the input or derived through reliable auditory inference must not appear in the output.
- • When the model internally flags audio information as uncertain (even if UNCERTAIN_AUDIO_INFORMATION_DETECTED is not triggered), the final Audio caption must strictly use cautious wording to describe the sound itself, focusing on auditory perception. It is strictly prohibited to directly state uncertainty or ambiguity; only provide confirmed acoustic facts and do not mention uncertain content in the output.
- • High-confidence audio tags are the highest priority source for determining sound type facts, but specific confidence values are not allowed in the output.

Figure 17: Prompt for integration Cont.

- Table 10: Dataset and examples corresponding to each sub-scenario, where cls is the classification task

Sub-Scenario Datasets(quantity) Examples Scene Recognition in Complex Soundscapes

AIR-Bench:

Acoustic scene cls(2,000) UrbanSound8K(8,732)

Identifying child playing scene Identifying kitchen scene

Acoustically Degraded Conditions

TAU Urban Sound-Mobile(5,265) FSDnoisy18K(947)

Identifying street pedestrian sound Identifying metro station scene

Music Understanding

AIR-Bench: Genre cls(2,000) MusicAQA(814) Mood detection(2,000) Chat-Music(500)

Identifying music genre Character portrayed by the tune Trumpet&accordion’s role in texture

Sound Understanding

AIR-Bench: SoundAQA(2,000) Chat-Sound(500)

AudioBench: Audio-Scene QA(9,131)

Location of dripping water Possible actions with the liquid Indications of a busy road

Acoustic Entity Recognition

AIR-bench: Vocal sound cls(1,000) Music instruments cls(2,000)

ESC-50(2,000) FSD50K(10,231)

Instrument recognition Acoustic event/ontology recognition Acoustic scene type recognition

- Table 11: Inter- (M-V, M-S, V-S) and Intra- (M, V, S) category embedding distances. Best interdistances (higher) and intra-distances (lower) are bolded.

Inter-category distance ↑ Intra-category distance ↓ M – V M – S V – S Music Vehicle Speech

Dataset/Method

FusionAudio 0.7230 0.5369 0.5943 0.8084 0.7406 0.8204 ASC 0.5685 0.4137 0.4523 0.8638 0.8216 0.8724 Auto-ACD 0.5685 0.4137 0.4523 0.8645 0.8402 0.8915 Sound-VECaps 0.5232 0.3770 0.4664 0.8578 0.7798 0.8920

