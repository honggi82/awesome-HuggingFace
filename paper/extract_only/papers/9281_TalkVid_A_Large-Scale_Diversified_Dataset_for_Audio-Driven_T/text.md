# arXiv:2508.13618v1[cs.CV]19Aug2025

## TalkVid: A Large-Scale Diversified Dataset for Audio-Driven Talking Head Synthesis

Shunian Chen1,∗, Hejin Huang1,2,∗, Yexin Liu3,*, Zihan Ye1, Pengcheng Chen1, Chenghao Zhu1, Michael Guan1, Rongsheng Wang1, Junying Chen1, Guanbin Li2, Ser-Nam Lim3,†, Harry Yang3,†, Benyou Wang1,† 1The Chinese University of Hong Kong, Shenzhen 2Sun Yat-sen University 3The Hong Kong University of Science and Technology wangbenyou@cuhk.edu.cn

### Abstract

Audio-driven talking head synthesis has achieved remarkable photorealism, yet state-of-the-art (SOTA) models exhibit a critical failure: they lack generalization to the full spectrum of human diversity in ethnicity, language, and age-groups. We argue this generalization gap is a direct symptom of limitations in existing training data, which lack the necessary scale, quality, and diversity. To address this challenge, we introduce TalkVid, a new large-scale, high-quality, and diverse dataset containing 1244 hours of video from 7729 unique speakers. TalkVid is curated through a principled, multi-stage automated pipeline that rigorously filters for motion stability, aesthetic quality, and facial detail, and is validated against human judgments to ensure its reliability. Furthermore, we construct and release TalkVid-Bench, a stratified evaluation set of 500 clips meticulously balanced across key demographic and linguistic axes. Our experiments demonstrate that a model trained on TalkVid outperforms counterparts trained on previous datasets, exhibiting superior cross-dataset generalization. Crucially, our analysis on TalkVid-Bench reveals performance disparities across subgroups that are obscured by traditional aggregate metrics, underscoring its necessity for future research. Code and data can be found in https:// github.com/FreedomIntelligence/TalkVid

### 1. Introduction

Audio-driven talking head synthesis has achieved remarkable photorealism, with recent models generating outputs that are often indistinguishable from real video under con-

*First three authors contributed to this work equally. Ser-Nam Lim, Harry Yang, and Benyou Wang are the corresponding authors.

trolled conditions [7, 22]. Yet, this success masks a critical fragility. SOTA models remain brittle; as our experiments will demonstrate, their performance degrades significantly when confronted with the full spectrum of human diversity. This generalization gap—the failure to handle varied ethnicities, unconstrained head poses, and diverse languages as shown in Figure 1—is not a minor flaw. It is the primary bottleneck preventing the widespread, reliable, and equitable application of this technology.

We argue this brittleness is a direct consequence of a foundational bottleneck: the data upon which these models are trained. Existing datasets force a difficult trade-off. On one hand, datasets like HDTF [26] offer high-resolution, front-facing videos but are narrowly curated, lacking the linguistic, demographic, and motion diversity needed for robust generalization. On the other hand, large-scale, “in-thewild” datasets like VoxCeleb2 [5] offer diversity but are rife with technical artifacts—motion blur, compression noise, and inconsistent framing—that compromise the training of high-fidelity generative models. Consequently, the field has lacked a resource that is simultaneously large-scale, diverse, and technically pristine. This data gap directly translates into models that are biased and unreliable.

This paper introduces a unified solution to this datacentric challenge. We present TalkVid, a new dataset designed from the ground up to eliminate the trade-offs of prior work. Its construction is guided by three core principles: 1) Scale and Diversity, sourcing over 6,000 hours of raw video to capture a broad cross-section of speakers, languages, and contexts; 2) High-Quality, enforced via a rigorous, multi-stage automated filtering pipeline that ensures technical excellence in motion, aesthetics, and facial detail; and 3) Reliability, validated through human verifications confirming our pipeline’s alignment with perceptual quality.

[Figure 1]

- Figure 1. Examples from our TalkVid dataset, showcasing the diversity in identity, ethnicity, and head pose that SOTA models must generalize to. Existing datasets lack this combined diversity and technical quality, leading to generalization failures.

Name Year Speaker Hours Resolution Language Age Body Included Caption Source

GRID [6] 2006 33 27.5 288p, 576p English 18–49 No No Lab Crema-D [2] 2014 91 11.1 720p English 20–74 No No Lab LRW [4] 2017 1k+ 173 120p English - No No Wild

- VoxCeleb1 [12] 2017 1.2k 352 256p - - No No Wild
- VoxCeleb2 [5] 2018 6.1k+ 2.4k 256p - - No No Wild MEAD [18] 2020 60 39 384p English 20–35 No No Lab HDTF [26] 2021 362 15.8 512p - - No No Wild Hallo3 [7] 2024 - 70 480p - - Yes No Wild MultiTalk [14] 2024 - 243.2 512p 20 lang - Yes No Wild TalkVid (Ours) 2025 7,729 1,244.33 1080p,2160p 15 lang 0-60+ Yes Yes Wild

Table 1. Comparison of open-source datasets for audio-driven talking-head generation.

Resolution abbreviations denote pixel dimensions (width×height): 120p (120×120), 256p (256×256), 288p (360×288), 384p (384×384), 480p (480×720), 512p (512×512), 576p (720×576), 720p (960×720), 1080p (1920×1080), and 2160p (3840×2160). For our TalkVid dataset, the listed resolutions comprise 87.59% of the data.

Crucially, better training requires better evaluation. We introduce TalkVid-Bench, a dedicated 500-clip evaluation benchmark stratified across key demographic (age, gender, ethnicity) and linguistic dimensions. Current evaluation practices, which rely on aggregate metrics, obscure critical failure modes and fairness issues. TalkVid-Bench enables a fine-grained analysis that reveals model-specific biases and quantifies true generalization. Our experiments, leveraging this benchmark, provide the definitive evidence that training on TalkVid yields superior performance and that TalkVid-Bench is essential for exposing robustness gaps invisible to previous evaluation protocols.

Our contributions are threefold: 1) we introduce TalkVid, a large-scale, high-quality dataset containing 1,244 hours of talking-head videos from 7,729 speakers, curated via a rigorous, human-validated pipeline; 2) we build TalkVid-Bench, a stratified evaluation benchmark balanced across demographic and linguistic dimensions to en-

able transparent assessment of model fairness and generalization; 3) we conduct comprehensive empirical validation demonstrating that models trained on TalkVid achieve state-of-the-art performance, superior cross-domain robustness, and reveal critical biases missed by existing methods, thereby providing the community with essential resources for developing equitable talking-head synthesis models.

### 2. Related Works

Audio-Driven Talking Head Synthesis. The synthesis of talking heads from audio has rapidly evolved, moving from GAN-based architectures to the now-dominant diffusion models. Early methods, often leveraging Generative Adversarial Networks (GANs) [10], achieved high-resolution, efficient synthesis. For instance, StyleHEAT [23] enabled one-shot generation, while others like SadTalker [25] and LipSync3D [13] focused on improving motion dynamics and lip-sync accuracy, respectively. Despite these ad-

Video Collection and Clip

###### Automated Multi-Stage Quality Filtering Human Validation

Segmentation

Clip

Video

Head-detail Filtering

Annotation Guidelines

Aesthetic

Annotator

Segmentation

Collection

Quality

Background

for Human

Movement Score

Re-encoding (H 264) Dover

Recording

and Training

Evaluators

Environment Requirements

Resolution Score

Annotator Background

Scene Split

General Rule

(PySceneDetect)

Speaker Diversity Requirements

Rotation Score Completeness Score Orientation Score

Motion Dynamics

Duration Filtering

Filter-Specific Instructions

Training Rule

Speaker Movement Requirements

Cotracker

Subtitle

Filtering

- Figure 2. The TalkVid construction pipeline. The process starts with (1) video collection and clip segmentation. Each candidate clip then undergoes (2) a multi-stage filtering cascade to enforce quality across aesthetics, motion, and facial detail. Finally, the pipeline’s effectiveness is (3) validated against human judgments.

vances, GAN-based approaches often suffer from temporal inconsistencies and limited expressiveness, particularly with large pose variations.

To address these limitations, recent work has overwhelmingly shifted towards Diffusion Models (DMs), which offer superior temporal stability and photorealism. Foundational models like VExpress [17] demonstrated the potential of DMs for this task. Subsequent works have focused on enhancing control and alignment; for example, AniPortrait [19] and Hallo [21] employ multistage pipelines and part-aware modules to improve audiovisual correspondence. The current SOTA, exemplified by VASA [22], Hallo3 [7], and EDTalk [15], introduces disentangled latent representations to generate highly expressive and controllable facial dynamics beyond simple lip movements. However, these methods often rely on complex pipelines and pre-trained priors, leaving a gap for a unified and efficiently controllable model.

Datasets for Talking Head Generation. Progress in talking head synthesis is intrinsically tied to the available training data. Early datasets, such as GRID [6] and CREMAD [2], were collected in controlled laboratory settings, providing clean audio-visual pairs but lacking diversity and scale. The advent of large-scale, ”in-the-wild” datasets like LRW [4] and VoxCeleb2 [5] was a major step forward, offering real-world variability. However, these datasets often suffer from inconsistent video quality and a lack of granular annotations necessary for modeling fine-grained expression. More recent high-quality datasets, including MEAD [18], HDTF [26], and MultiTalk [14], have improved upon resolution and speaker diversity. Nevertheless, a critical bottleneck persists: the absence of a truly largescale benchmark that pairs high-resolution video with rich, semantic annotations for detailed expressive and semantic

control. Our dataset is designed to fill this void, providing over 1,244 hours of high-fidelity video with dense captions to foster the next generation of controllable talking head models.

### 3. The TalkVid Dataset

This section details the construction methodology, human verification protocol, and presents the analysis of the dataset’s characteristics.

#### 3.1. Construction Methodology

The TalkVid construction pipeline is illustrated in Figure 2. We first describe each stage and then present a quantitative analysis validating the effectiveness of our filtering process against human assessments.

##### 3.1.1. Data Preprocess

- Stage 1: Video Collection. We begin by collecting over 30,000 videos from YouTube, totaling more than 6,000 hours of high-resolution (1080p or higher) content. To ensure a high-quality starting point, we target genres known for clear audio and stable video, such as educational lectures, technical reviews, and professional vlogs. For each source, we download the video, audio, and available autogenerated transcripts through yt-dlp [24]. Complete sourcing criteria can be found in Appendix A.
- Stage 2: Clip Segmentation. The collected videos are then processed through a segmentation pipeline. First, all videos are standardized by re-encoding them to the H.264 (MP4) format. We then use PySceneDetect [1] to detect shot boundaries. Segments shorter than 5 seconds are discarded, as they are typically too brief to contain a complete thought or gesture. Finally, using transcript timings, we remove segments without speech events.

[Figure 2]

- Figure 3. The data filtering cascade. This diagram quantifies the progressive refinement of the dataset, showing the hours of video retained and discarded at each preprocessing and content-based filtering stage.

20 40 60 80 100

Duration (s, clipped at 100)

0

5000

10000

15000

20000

25000

30000

35000

40000

Count

Duration Distribution

Mean: 17.93

8 9 10 11

Dover Score

0

2000

4000

6000

8000

Count

Dover Score Distribution

Mean: 8.55

0.850 0.875 0.900 0.925 0.950 0.975 1.000

Cotracker Ratio

0

1000

2000

3000

4000

5000

Count

Cotracker Ratio Distribution

Mean: 0.92

|Score Type<br><br>Movement| |
|---|---|
|Rotation Orientation| |
| | |
| | |
| | |

70 75 80 85 90 95 100

Score Value

0.0

0.1

0.2

0.3

0.4

Density

Head Detail Score Distributions

101 102 103

Total Duration (hours)

Personal Experience Vlogger/Creator Popular Science

Health Advice Global Culture

Interview Online Course/Lecture

Independent Media Motivational Speech

Language Learning

474.0h 357.6h

65.5h 45.5h

41.8h 27.8h

25.4h

23.2h 22.8h 22.4h

Video Category Distribution (Top 10)

100 101 102 103

Total Duration (hours)

English Chinese Spanish

Japanese

Hindi Korean

Russian Portuguese

French Other

867.1h 248.9h

41.6h 12.8h

9.5h 9.4h

3.6h 3.4h

1.8h 1.2h

Language Distribution (Top 10)

100 101 102 103

Total Duration (hours)

31-45

19-30

46-60

60+

0-19

814.8h

293.7h

105.6h

23.2h

2.4h

Age Group Distribution

0 200 400 600 800

Total Duration (hours)

Female

Male

Asian

White

African

Gender & Ethnicity Distribution

Group

| |
|---|

Gender Ethnicity

| |
|---|

- Figure 4. Statistical distributions of the TalkVid dataset. Top row: technical quality metrics for the final, filtered dataset. Bottom row: distributions of the high-level characteristics, including video categories, language, and speaker demographics.

##### 3.1.2. Content-Based Filtering

To ensure each clip meets the technical demands of modern generative modeling, we subject each candidate to a content-based filtering cascade. A clip is retained only if it satisfies all criteria across three key quality dimensions: aesthetic quality, motion dynamics, and head detail. The metrics and their corresponding thresholds are summarized in Table 2, while the rationale for each is elaborated below.

- Filter 1: Aesthetic Quality. To guarantee high visual fidelity, we employ DOVER [20], a no-reference video quality assessment model. By enforcing a minimum score, we filter out clips containing perceptible compression artifacts, noise, or excessive blur, retaining only those with a clean, high-quality appearance.
- Filter 2: Motion Dynamics. We select for clips with natural motion characteristics using the point tracking stability ratio from CoTracker [9]. The specified range serves a dual

Filtering Stage Metric Criterion / Threshold Aesthetic Quality DOVER Score [20] ≥ 7.0 Motion Stability CoTracker Ratio [9] ∈ [0.85, 0.999]

Movement Score Avg ≥ 80, Min ≥ 60 Rotation Score Avg ≥ 70, Min ≥ 60 Orientation Score Avg ≥ 70, Min ≥ 30 Resolution Score Avg ≥ 50, Min ≥ 40 Completeness Score = 100

Head Detail

Table 2. Filtering criteria for video clip selection, each candidate clip must satisfy all listed conditions.

purpose: the lower bound (≥ 0.85) removes clips with erratic motion or blur, which manifest as tracking failures, while the upper bound (≤ 0.999) crucially discards unnaturally static or ”frozen” shots. This ensures the retention of subtle, natural movements characteristic of a live speaker.

Filter 3: Head-Detail Filtering. Finally, we perform a fine-grained assessment of the subject’s head using a suite

Stage CoT. Dov. Comp. Move. Orient. Res. Rot. Avg. κ 0.74 0.90 0.80 0.66 0.80 0.72 0.90 0.79

- Table 3. Cohen’s Kappa, κ for quality filtering stages. Abbreviations are: CoTracker (CoT.), Dover (Dov.), Head-Completeness (Comp.), Head-Movement (Move.), Head-Orientation (Orient.), Head-Resolution (Res.), and Head-Rotation (Rot.).

of five metrics designed to ensure stability and clarity. These metrics collectively ensure the temporal stability of facial keypoints (Movement Score), smoothness of head orientation transitions (Rotation Score), a largely frontal view without extreme angles (Orientation Score), sufficient face resolution for detail (Resolution Score), and the consistent visibility of all facial parts (Completeness Score). A candidate must meet all five criteria, with further formulation details in Appendix B.

##### 3.1.3. Human Validation

Protocol. To confirm our automated filters serve as a reliable proxy for human quality judgments, we conduct a human evaluation study. For each of our seven filter criteria, we sample 100 borderline clips: 50 that marginally pass the filter and 50 that marginally fail. This focus on the decision boundary provides a stringent test of our thresholds. Two trained annotators, blind to the filter’s decision, assign a binary label (e.g., “Acceptable”/“Unacceptable”) to each clip based on a detailed rubric defining each quality attribute, more details can be found in Appendix C.2.

Results. The evaluation confirms the reliability of our pipeline. First, we measure high inter-annotator agreement (IAA), achieving an average Cohen’s Kappa (κ) of 0.79 across all criteria (Table 3). This indicates our quality standards are well-defined and consistently interpreted. Second, our automated filters demonstrate strong performance against the human-annotated ground truth, reaching an average accuracy of 95.1% and an F1-score of 95.3%. This result validates our automated pipeline as a reliable proxy for human quality assessment. Detailed per-criterion metrics are in the Appendix C.1.

#### 3.2. Quantitative Analysis

The final TalkVid dataset contains 1,244 hours of video from 7,729 unique speakers. Figure 3 provides a holistic overview of the data attrition throughout our pipeline, while Figure 4 details the statistical properties of the final dataset.1

##### 3.2.1. Composition

Clips average 17.93s in duration. The dataset is compositionally diverse, led by “Personal Experience” (474.0h) and “Vlogger/Creator” (357.6h) content. It spans over 15

1Data characteristics labeled as ”Unknown” are not taken into account.

[Figure 3]

Figure 5. Qualitative examples from TalkVid. The sequences illustrate demographic diversity and key technical challenges for synthesis: varied lighting, complex backgrounds, and occlusions.

languages, with English (867.1h) and Chinese (248.9h) being the most prominent. Speaker demographics are varied across age, gender, and ethnicity, with the 31-45 year group being the largest (814.8h).

##### 3.2.2. Technical Quality

Our filtering ensures high technical quality. A mean DOVER score of 8.55 confirms strong visual fidelity. The mean CoTracker ratio of 0.92 validates our selection for natural motion, successfully culling both overly static and erratic shots. Head detail scores (Movement, Rotation, Orientation) are sharply skewed towards their maxima, indicating stable, consistently trackable faces. This technical profile makes the dataset highly suitable for generative tasks.

##### 3.2.3. TalkVid-Core

We introduce TalkVid-Core, a high-purity and diverse subset comprising 160 hours of content. This subset is derived by applying a stringent set of thresholds to quality metrics. Importantly, the data is uniformly sampled across ethnicity, gender, and age categories to ensure a balanced representation. Following the selection of this high-quality video set, we generate annotations for each clip using Gemini 1.5 Pro2. Further details and qualitative examples of these annotations are provided in Appendix E.

#### 3.3. Qualitative Analysis

Figure 5 illustrates the dataset’s breadth. The examples confirm the demographic diversity (age, gender, ethnicity) and show difficult capture conditions. These include in-thewild lighting (Row 1), complex indoor backgrounds (Row

- 4), and accessories such as headwear and glasses (Rows 1,
- 5). The sequences also contain significant challenges for synthesis models, including occlusions from hands and microphones (Row 5) and a full range of motion dynamics,

2gemini-1.5-pro-002

Training Dataset

English Chinese Polish FID↓ FVD↓ Sync-C↑ Sync-D↓ FID↓ FVD↓ Sync-C↑ Sync-D↓ FID↓ FVD↓ Sync-C↑ Sync-D↓

HDTF 60.817 443.202 4.000 10.403 48.561 415.564 3.285 10.058 39.231 321.261 2.654 10.368

Language

Hallo3 59.842 387.507 4.753 9.536 52.514 342.062 4.005 9.450 38.458 343.553 3.424 9.690 TalkVid 59.562 357.603 4.567 9.867 47.509 306.131 4.041 9.521 39.271 288.178 3.695 9.663

White African Asian

HDTF 46.589 305.284 3.587 9.997 50.807 376.161 3.746 10.203 53.163 302.214 3.453 10.198

Ethnicity

Hallo3 40.927 267.492 4.218 9.292 48.218 350.025 4.296 9.636 52.492 303.478 4.076 9.517 TalkVid 40.740 274.226 4.176 9.587 44.373 326.840 4.352 9.724 48.511 303.997 4.056 9.744

Male Female

HDTF 46.525 306.947 3.540 9.965 46.173 297.840 3.489 10.223 Hallo3 41.549 299.984 3.935 9.496 42.659 258.583 4.034 9.704 TalkVid 39.398 294.709 3.984 9.639 41.967 241.920 4.051 9.788

Gender

19-30 31-45 60+

HDTF 45.591 283.927 3.679 10.078 58.843 295.236 3.592 10.448 53.192 350.580 3.630 10.018

Age

Hallo3 41.501 272.912 4.214 9.565 44.493 253.756 4.380 9.581 53.854 332.383 3.748 9.741 TalkVid 37.879 253.698 4.329 9.605 43.702 222.202 4.537 9.626 51.141 321.556 3.942 9.804

- Table 4. Comparison with other baseline training datasets, including HDTF [26] and Hallo3 [7] on TalkVid-bench across four dimensions, showing subgroup-level performance.

from large-amplitude expressions (Row 1) to subtle conversational movements (Row 2). The inclusion of diverse subjects under these conditions makes TalkVid a valuable resource for training and evaluating generative models.

### 4. Experiments

We conduct experiments to validate the benefits of TalkVid as a training corpus and to demonstrate the utility of TalkVid-Bench for revealing model-specific biases. We compare a SOTA model trained on TalkVid against the same model trained on prior datasets.

#### 4.1. Experimental Setup

##### 4.1.1. Model and Baselines

We train the open-source V-Express [17] model, a SOTA diffusion-based architecture for talking-head synthesis. We evaluate its performance when trained under three distinct dataset conditions: 1) HDTF [26]: A high-resolution talking-head dataset. 2) Hallo3 [7]: A curated dataset with clean motion conditions. 3) TalkVid-Core (Ours): A 160hour subset of our proposed TalkVid dataset.

##### 4.1.2. Implementation Details

For all conditions, we adhere strictly to the original threestage training protocol of V-Express (40k, 75k, and 50k steps) and its hyperparameters. We use the AdamW optimizer [11] with a learning rate of 1e-6 and global batch sizes of 8, 4, and 2 for each stage, respectively. Input video frames are preprocessed by cropping facial regions and re-

sizing to 512×512. Training for each condition requires 3 days on 4 NVIDIA A100 GPUs.

##### 4.1.3. Evaluation datasets

Evaluation is conducted on three test sets. First, we use a 100-clip subset of the HDTF test set. Second, we use a 167-clip subset from the Hallo3 test set to evaluate performance on cleaner motion and larger pose variations. Third, we introduce TalkVid-Bench, our primary benchmark of 500 five-second clips designed for robust and fair evaluation. Drawn from the TalkVid corpus but held-out from training, the benchmark is stratified and balanced across four dimensions: language, ethnicity, gender, and age. Whereas HDTF and Hallo3 only support aggregate scores, TalkVid-Bench enables granular analysis of model performance across subgroups, making it the definitive tool for the cross-domain and fairness experiments in this paper. Subgroup distributions are detailed in Appendix C.4.

##### 4.1.4. Evaluation metrics

We employ a set of metrics to evaluate performance: Visual Quality. We report Frechet Inception Distance (FID) [8] for per-frame realism and Frechet Video Distance (FVD) [16] for temporal coherence and video-level fidelity. Audio-Visual Synchronization. Following SyncNet [3], we measure audio-lip sync confidence (Sync-C) and the distance between audio-visual embeddings (Sync-D).

Language Ethnicity Training Dataset FID↓ FVD↓ Sync-C↑ Sync-D↓ FID↓ FVD↓ Sync-C↑ Sync-D↓ HDTF 31.385 205.990 3.109 10.567 36.381 214.488 3.605 10.135 Hallo3 29.721 184.465 3.849 9.887 34.650 194.847 4.204 9.488 TalkVid 28.686 178.396 3.842 10.064 32.588 187.368 4.205 9.686

Gender Age Training Dataset FID↓ FVD↓ Sync-C↑ Sync-D↓ FID↓ FVD↓ Sync-C↑ Sync-D↓ HDTF 38.950 225.611 3.513 10.100 34.257 166.276 3.542 10.168 Hallo3 35.260 208.815 3.987 9.605 32.934 158.799 4.052 9.596 TalkVid 34.347 199.003 4.019 9.717 30.790 151.580 4.112 9.731

- Table 5. Comparison with other baseline training datasets, including HDTF [26] and Hallo3 [7] on TalkVid-bench across four dimensions in general.

#### 4.2. Quantitative Results

##### 4.2.1. Fine-grained Results on TalkVid-Bench

The fine-grained evaluation on TalkVid-Bench, shown in Table 4, confirms that TalkVid produces models with superior generalization and reduced bias.

Cross-lingual Generalization While baselines perform well on English, our model achieves the best visual quality (FID/FVD) across English, Chinese, and Polish, significantly outperforming on non-English languages. This result validates that TalkVid’s linguistic breadth directly remedies the brittleness of models trained on narrower data.

Mitigating Ethnic Bias The Hallo3-trained model is competitive for White speakers but falters for African speakers, where our model is clearly superior. This performance delta, revealed by TalkVid-Bench, shows that TalkVid’s inclusive data fosters more equitable models.

Robustness Across Gender and Age Our model achieves the most consistent high performance for both male and female subjects and shows marked improvements for challenging age groups, particularly for speakers aged 60+. This underscores the value of TalkVid’s comprehensive demographic coverage.

##### 4.2.2. Overall Results on TalkVid-Bench

Table 5 shows the overall performance for each dimension of TalkVid-Bench. Three trends stand out.

Visual fidelity leads across the board TalkVid consistently records the lowest FID/FVD in all four dimensions, confirming that its higher-quality training data translates into universally sharper and more stable videos.

HDTF test set Training Dataset FID↓ FVD↓ Sync-C↑ Sync-D↓ HDTF 19.963 188.657 3.523 10.254 Hallo3 24.484 183.172 3.615 9.966 TalkVid 21.772 175.122 3.707 9.970

Hallo3 test set Training Dataset FID↓ FVD↓ Sync-C↑ Sync-D↓

HDTF 16.690 114.891 4.464 9.901 Hallo3 19.745 106.009 4.941 9.404 TalkVid 18.367 101.819 4.921 9.530

Table 6. Comparison with other baseline training datasets on HDTF and Hallo3 test sets.

Synchronisation remains competitive Hallo3 consistently achieves the lowest Sync-D, but the gaps are small. TalkVid matches or surpasses Hallo3 on Sync-C in three of the four dimensions and is only marginally lower on Language, showing that visual improvements do not come at a meaningful cost to lip-sync quality.

Balanced demographic performance From language and ethnicity to gender and age, TalkVid delivers the most uniform improvements, reinforcing the conclusion that a diverse training set yields a model that generalises well without introducing new biases.

##### 4.2.3. Comparison on Standard Benchmarks

We further assess generalization on the canonical HDTF and Hallo3 test sets (Table 6). The TalkVid-trained model demonstrates superior cross-domain robustness, achieving the best temporal coherence (FVD) on both benchmarks and strong lip-sync (Sync-C) scores. In contrast, models trained on HDTF and Hallo3 exhibit significant performance degradation when evaluated out-of-domain, high-

#### 4.3. Qualitative Results

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- 4.3.1. Diversity coverage and naturalness

Figure 6 showcases the performance of our model on diverse identities from TalkVid-Bench. The model accurately preserves identity and background across speakers. Crucially, it synthesizes natural, non-verbal behaviors often absent in prior work: subtle head motion (row 2) and realistic eye blinks (row 4) are generated in sync with speech. This confirms that the model generalizes well beyond nearfrontal, static poses.

- 4.3.2. Comparison with baseline datasets

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

Input Audio

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

The frame-by-frame comparison in Figure 7 reveals clear deficiencies in the baseline models. Both the HDTF and Hallo3-trained models produce static expressions with muted lip motion, failing to match the audio or replicate natural behaviors like eye blinks. In contrast, our model reproduces the ground-truth’s dynamic expression, including correctly timed eye blinks and larger, more articulate lip shapes. This qualitative evidence corroborates our quantitative findings, confirming that TalkVid’s rich motion diversity leads to more lifelike and accurate synthesis.

Reference Image

Generated Frames

- Figure 6. Qualitative examples from our TalkVid-trained model, evaluated on diverse samples from TalkVid-Bench spanning language, ethnicity, gender, and age.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

TalkVid

[Figure 30]

[Figure 31]

[Figure 32]

Hallo3

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

GT

HDTF

- Figure 7. Qualitative comparison on an unseen clip from TalkVid-Bench. From top: V-Express fine-tuned on HDTF, Hallo3, TalkVid-Core (ours), and Ground-Truth (GT).

### 5. Ethical Considerations

While generative models pose significant risks of misuse, we contend that an equally critical ethical failure is the status quo: creating biased technology from non-diverse data that systematically fails for underrepresented groups. Our work directly confronts this harm. TalkVid provides the demographically rich data to train fairer models, while TalkVid-Bench offers a standardized framework to audit and mitigate such algorithmic bias. To ensure responsible dissemination, we will distribute the dataset as source URLs and timestamps to verified researchers under a strict license. This protocol respects creator copyright and explicitly prohibits all malicious applications, including defamation and non-consensual content generation, thereby balancing research accessibility with accountability.

lighting their tendency to overfit. Notably, while the HDTF model secures the best FID on its own test set, it does so at the cost of the worst FVD. Our model makes a more effective trade-off, indicating that TalkVid’s diversity promotes a more balanced and generalizable synthesis that avoids overfitting to domain-specific visual artifacts.

Training on TalkVid yields a model that is robust across evaluation domains and metrics. The quantitative results, particularly the stratified analysis on TalkVid-Bench, prove that TalkVid provides a superior foundation for developing talking-head models that are both high-fidelity and equitable.

### 6. Conclusion

This paper addressed the critical brittleness of SOTA talking head models, a direct consequence of inadequate training data. We introduced TalkVid, a large-scale, diverse, and technically pristine dataset curated through a rigorous, human-validated pipeline. To enable fair assessment, we also presented TalkVid-Bench, a stratified benchmark that uncovers biases invisible to standard metrics. Our experiments show that training on TalkVid produces more robust and equitable models. By releasing this ecosystem, we hope it will spur further research into auditing and mitigating bias in generative video models.

### References

- [1] Breakthrough Apps. Pyscenedetect: Automated video scene detection. https://github.com/Breakthrough/ PySceneDetect, 2021. Version 0.6+. 3
- [2] Houwei Cao, David G Cooper, Michael K Keutmann, Ruben C Gur, Ani Nenkova, and Ragini Verma. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377–390, 2014. 2, 3
- [3] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Workshop on Multi-view Lip-reading, ACCV, 2016. 6
- [4] Joon Son Chung and Andrew Zisserman. Lip reading in the wild. In Computer Vision–ACCV 2016: 13th Asian Conference on Computer Vision, Taipei, Taiwan, November 2024, 2016, Revised Selected Papers, Part II 13, pages 87–103. Springer, 2017. 2, 3
- [5] Joon Son Chung, Arsha Nagrani, and Andrew Zisserman. Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622, 2018. 1, 2, 3
- [6] Martin Cooke, Jon Barker, Stuart Cunningham, and Xu Shao. An audio-visual corpus for speech perception and automatic speech recognition. The Journal of the Acoustical Society of America, 120(5):2421–2424, 2006. 2, 3
- [7] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks. arXiv preprint arXiv:2412.00733, 2024. 1, 2, 3, 6, 7
- [8] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, pages 6629–6640, 2017. 6
- [9] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudolabelling real videos. arXiv preprint arXiv:2410.11831,

2024. 4, 10

- [10] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [11] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6
- [12] Arsha Nagrani, Joon Son Chung, Weidi Xie, and Andrew Zisserman. Voxceleb: Large-scale speaker verification in the wild. Computer Speech & Language, 60:101027, 2020. 2
- [13] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, pages 484–492, 2020. 2
- [14] Kim Sung-Bin, Lee Chae-Yeon, Gihun Son, Oh Hyun-Bin, Janghoon Ju, Suekyeong Nam, and Tae-Hyun Oh. Multitalk: Enhancing 3d talking head generation across lan-

- guages with multilingual video dataset. arXiv preprint arXiv:2406.14272, 2024. 2, 3
- [15] Shuai Tan, Bin Ji, Mengxiao Bi, and Ye Pan. Edtalk: Efficient disentanglement for emotional talking head synthesis. In European Conference on Computer Vision, pages 398–

416. Springer, 2024. 3

- [16] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2019. 6
- [17] Cong Wang, Kuan Tian, Jun Zhang, Yonghang Guan, Feng Luo, Fei Shen, Zhiwei Jiang, Qing Gu, Xiao Han, and Wei Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024. 3, 6
- [18] Kaisiyuan Wang, Qianyi Wu, Linsen Song, Zhuoqian Yang, Wayne Wu, Chen Qian, Ran He, Yu Qiao, and Chen Change Loy. Mead: A large-scale audio-visual dataset for emotional talking-face generation. In European conference on computer vision, pages 700–717. Springer, 2020. 2, 3
- [19] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 3
- [20] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023. 4
- [21] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 3
- [22] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong, and Baining Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. Advances in Neural Information Processing Systems, 37:660–684, 2024. 1, 3
- [23] Fei Yin, Yong Zhang, Xiaodong Cun, Mingdeng Cao, Yanbo Fan, Xuan Wang, Qingyan Bai, Baoyuan Wu, Jue Wang, and Yujiu Yang. Styleheat: One-shot high-resolution editable talking face generation via pre-trained stylegan. In European conference on computer vision, pages 85–101. Springer, 2022. 2
- [24] yt-dlp contributors. yt-dlp. GitHub repository, 2025. Accessed: 2025-08-02, Version: 2025.01.01. 3
- [25] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8652–8661, 2023. 2
- [26] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3661–3670, 2021. 1, 2, 3, 6, 7

### A. Video Collection Criteria

To ensure consistency and quality in the collected data, we establish standardized guidelines across three key dimensions: recording environment, speaker behavior, and identity diversity.

Recording Environment Requirements. All videos are recorded indoors to avoid uncontrollable outdoor factors such as variable lighting or wind noise. Lighting conditions must be stable and evenly distributed, with strong side or backlighting strictly avoided. The background should be visually clean and preferably monochromatic to minimize distractions. Recording devices are required to support a minimum resolution of 1080p and a frame rate of at least 25 fps, and should be mounted stably—preferably on a tripod—to avoid camera shake or motion blur. Audio must be captured clearly, and free from background noise or interference (e.g., music, environmental sounds). The recorded audio should contain only a single speaker, with no overlapping dialogue or ambient speech.

Speaker Movement Requirements. During recording, speakers are instructed to face the camera directly, maintaining a natural and relaxed facial expression. Excessive head motion, exaggerated gestures, or sudden movements are discouraged to preserve alignment quality. The speaker’s face should remain unobstructed throughout the recording—masks, microphones, or large reflective glasses are not allowed, while standard eyeglasses are acceptable. The entire facial region, including the chin and forehead, must stay within the camera frame, with the face occupying approximately 30–40% of the frame area. Appropriate headroom and a consistent shooting distance (recommended 0.5–1 meter) should be maintained. Speech content must be delivered in clear, accent-neutral English at a moderate pace, with well-pronounced articulation.

Speaker Diversity Requirements. To promote fairness and generalizability in downstream applications, the dataset is curated to include a diverse range of speakers. We ensure balanced representation across genders, age groups, ethnic backgrounds, and speaking styles. Collected samples vary in facial expressions, emotional tone, speaking speed, and prosody. Each video features unique spoken content between 10 to 30 seconds in duration, avoiding repetition or overly scripted delivery. For quality assurance, collectors verify facial visibility and audiovisual synchronization and record basic demographic metadata (e.g., gender, age group, race).

### B. Video Filtering Details

#### B.1. Motion Filtering Details

We adopt the point-tracking stability ratio ρ provided by CoTracker [9] as a proxy for natural facial motion. For each 16-frame clip, CoTracker initializes K = 256 trajec-

tories on a uniform grid within the central 256 × 256 crop. A trajectory is deemed valid over the entire clip if: 1) its confidence remains ≥ 0.5 in every frame, 2) the per-frame displacement never exceeds 20 pixels (approx. 8% of the image diagonal), 3)it stays inside the frame boundaries. The stability ratio is then defined as:

#valid trajectories K ∈ [0,1]. (1)

ρ =

A high ρ indicates consistent tracking but not necessarily large motion; conversely, a low ρ usually corresponds to motion blur or tracking failure. We retain clips whose stability ratio satisfies

0.85 ≤ ρ ≤ 0.999. (2)

Lower bound (ρ ≥ 0.85). Clips with ρ < 0.85 exhibit > 10% trajectory loss, typically caused by severe motion blur or compression artifacts that would corrupt subsequent 3D landmark estimation.

Upper bound (ρ ≤ 0.999). Clips with ρ > 0.999 contain almost no detectable micro-motion (empirical mean displacement < 0.3 px), resulting in unnaturally static “frozen” faces that degrade the perceived liveliness of generated talking heads. All videos are resampled to 25 fps and down-scaled so that the shorter side is 512 px. We extract 16-frame clips with a sliding window stride of 8 frames.

#### B.2. Head-detail Filtering Details

To evaluate the facial quality in video clips, we define a scoring system based on five dimensions. All scores range from 0 to 100, with the exception of the Resolution Score which ranges from 0 to 3000. Higher values indicate better quality for all scores. A clip is retained only if all scores meet their respective thresholds.

Movement Score. This metric measures the temporal stability of facial keypoints. We compute the average displacement of keypoints between adjacent frames, normalized by the smaller dimension of the image (height or width). The score is defined as 100 − 100 × avg movement. Lower displacement leads to higher scores. Threshold: average ≥ 80, minimum ≥ 60.

Orientation Score. This score reflects how frontal the face is across frames. For each frame, we compute pitch, yaw, and roll scores as |θ|/180 × 100, and derive the final score as 100− pitch2 + yaw2 + roll2. A higher score indicates better alignment with the camera. Threshold: average ≥ 70, minimum ≥ 30.

Completeness Score. We assess whether key facial regions are fully visible within the frame. The score combines three regions: eyes (weight 0.3), nose (0.4), and mouth (0.3). For each region, the presence of all keypoints within image bounds contributes 1; otherwise, 0. The weighted sum gives the final score. Occlusions are tolerated as long as keypoints are within the visible area. Threshold: average = 100, minimum = 100.

Resolution Score. This score quantifies how large the face appears in the frame. It is calculated as 30 × (face area/image area) × 100. Larger face regions yield higher scores, which may exceed 100. Threshold: average ≥ 50, minimum ≥ 40.

Rotation Score. This metric evaluates the smoothness of head motion. It is defined as 100 − avg rotation amplitude, where the amplitude is computed by the 3D orientation change: ∆pitch2 + ∆yaw2 + ∆roll2 between adjacent frames. Smaller variations indicate higher stability. Threshold: average ≥ 70, minimum ≥ 60.

All metrics must satisfy their corresponding thresholds for a clip to pass the quality screening.

### C. Video Annotation

This appendix provides supplementary details for the human verification study discussed in the main paper.

#### C.1. Detailed Performance Metrics

A detailed breakdown of the evaluation results for each of the seven automated filtering stages is summarized in Table 7. For each category, the table presents the InterAnnotator Agreement (IAA) rate, which measures annotator consistency, alongside key classification metrics (Accuracy, Precision, Recall, and F1-score) for our automated filter when benchmarked against the Golden Standard. The consistently high scores reported in the table underscore the robustness and reliability of our data curation pipeline.

#### C.2. Annotator Background and Training

The human evaluation was conducted by a team of five annotators with strong technical and scientific backgrounds. The team comprised two Ph.D. students in Computer Science, one Ph.D. student in Applied Mathematics, one undergraduate student in Computer Science, and one undergraduate student in Statistics. All evaluators possess substantial experience with rigorous scientific research methodologies.

To ensure consistency and objectivity, a strict evaluation protocol was enforced. For each of the seven filtering categories, a pair of these annotators was assigned to evaluate the 100 sample clips independently. Prior to the main

annotation task, all participants underwent a dedicated calibration session. During this session, they were provided with detailed written guidelines and illustrative examples for each quality criterion (e.g., defining acceptable vs. unacceptable head movement). The goal of this phase was to establish a shared and consistent understanding of the task requirements, which directly contributed to the high interannotator agreement rates observed in our results.

Crucially, the entire evaluation process was blinded; the annotators had no knowledge of the decisions made by the automated filters, ensuring that their judgments remained completely unbiased.

#### C.3. Annotation Guidelines for Human Evaluators

To ensure all annotators applied the same criteria, we provided them with the following detailed instructions for each filtering category. These guidelines were also used during the calibration session to resolve ambiguities.

##### C.3.1. General Rule

A critical edge case applied to all categories: clips where a human face or head could not be reliably detected were generally to be labeled as negative, even if the clip otherwise met the quality criterion (e.g., high video quality or smooth motion).

##### C.3.2. Filter-Specific Instructions

Cotracker Filter The primary criterion is the spatiotemporal stability of the head. Clips were labeled as negative if they contained sudden, jerky movements resulting from large translations or rapid 3D rotations of the head.

Dover Filter This filter assesses overall visual and technical quality. Clips were labeled as negative if they suffered from low resolution, significant compression artifacts, poor lighting, or motion blur.

Head Detail Filters This is a composite evaluation. A clip must generally meet all five of the following subcriteria to be labeled as positive:

- • Movement Stability: The head must remain relatively stationary. Clips with large, abrupt translations or occlusions were rejected.
- • Frontal Orientation: The subject’s face must be predominantly front-facing. Clips containing significant head turns, downward gazes, or profiles were rejected.
- • Head Completeness: The entire facial region must be clearly visible and unobstructed. Clips showing only partial features or where the face was occluded were rejected.
- • Facial Resolution: The face must occupy a salient portion of the frame (heuristically, > 20%). Clips

##### Filtering Stage IAA Accuracy Precision Recall F1-score

Cotracker 86.87% 87.21% 88.64% 86.67% 87.64% Dover 94.00% 96.81% 95.83% 97.87% 96.84% Head Completeness 90.00% 96.67% 93.48% 100.00% 96.63% Head Movement 83.00% 96.39% 92.50% 100.00% 96.10% Head Orientation 90.00% 94.44% 97.78% 91.67% 94.62% Head Resolution 86.00% 97.67% 94.59% 100.00% 97.22% Head Rotation 95.00% 96.84% 95.83% 97.87% 96.84%

##### Average 89.3% 95.1% 95.5% 96.3% 95.3%

Table 7. Detailed Human Evaluation Results for Each Filtering Stage. We report Inter-Annotator Agreement (IAA) and the classification performance (Accuracy, Precision, Recall, F1-score) of our automated filter against the Golden Standard.

where the face was too small (e.g., < 10% of the frame area) were rejected.

• Rotational Stillness: This criterion is exceptionally strict. The head must maintain a fixed orientation with minimal rotation. Even a single, noticeable head turn, nod, or shake within the clip was sufficient for it to be labeled as negative. Annotators were instructed to watch the entire clip before making a final judgment.

##### C.3.3. Visual Examples

To further clarify the annotation criteria, this section provides visual examples. We first show an ideal positive case that passes all filters. Subsequently, we present seven negative cases, each illustrating a failure for a specific filtering criterion. Annotators were instructed to label the following types of clips as negative.

#### C.4. Benchmark Design

TalkVid-Bench comprises 500 carefully sampled and stratified video clips along four critical demographic and language dimensions: age, gender, ethnicity, and language. This stratified design enables granular analysis of model performance across diverse subgroups, mitigating biases hidden in traditional aggregate evaluations. Each dimension is divided into balanced categories:

- • Age: 0–19, 19–30, 31–45, 46–60, 60+, with a total of 105 samples.
- • Gender: Male, Female, with a total of 100 samples.
- • Ethnicity: Black, White, Asian, with a total of 100 samples.
- • Language: English, Chinese, Arabic, Polish, German, Russian, French, Korean, Portuguese, Japanese, Thai, Spanish, Italian, Hindi, and Other languages, with a total of 195 samples.

### D. Computational Efficiency

We quantify efficiency using the real-time factor (RTF), defined as the ratio between the input-video duration and the wall-clock processing time. An RTF greater than 1 indicates faster-than-real-time operation. Our pipeline comprises following sequential stages:

- 1. Rough segmentation + subtitle filtering (CPU-only, 96 cores) achieves an average RTF of 18.14.
- 2. Motion filtering (CoTracker) (96-core CPU with 8×NVIDIA A800 GPUs) reaches an average RTF of 64.21.
- 3. Quality filtering (DOVER) (96-core CPU with 8×NVIDIA A800 GPUs) reaches an average RTF of 87.36.
- 4. Head filtering (96-core CPU with 8×NVIDIA A800 GPUs) reaches an average RTF of 72.47.

[Figure 45]

- Figure 8. Distribution of TalkVid-bench across the language dimension (15 languages, 195 samples). The left panel shows the number of samples per language, while the right panel shows the distribution of these samples by language. Language abbreviations: ar (Arabic), pl (Polish), de (German), ru (Russian), fr (French), ko (Korean), pt (Portuguese), other (other languages), ja (Japanese), th (Thai), es (Spanish), it (Italian), hi (Hindi), en (English), zh (Chinese).

[Figure 46]

- Figure 9. Distribution of TalkVid-bench across three demographic dimensions. These statistics illustrate the diversity of TalkVid-bench in terms of participant demographics, providing a comprehensive benchmark for evaluating models under varied demographic conditions.

Examples of Sample Quality Based on Filter Cotracker Quality Level

Sample Example

[Figure 47]

Poor

[Figure 48]

Fair

[Figure 49]

Good

Table 8. Examples of Sample Quality Based on Filter Cotracker

Examples of Sample Quality Based on Filter Dover Quality Level

Sample Example

[Figure 50]

Poor

[Figure 51]

Fair

[Figure 52]

Good

Table 9. Examples of Sample Quality Based on Filter Dover

Examples of Sample Quality Based on Head Movement Quality Level

Sample Example

[Figure 53]

Poor

[Figure 54]

Fair

[Figure 55]

Good

Table 10. Examples of Sample Quality Based on Filter Head Movement

Examples of Sample Quality Based on Filter Head Orientation Quality Level

Sample Example

[Figure 56]

Poor

[Figure 57]

Fair

[Figure 58]

Good

Table 11. Examples of Sample Quality Based on Filter Head Orientation

Examples of Sample Quality Based on Filter Head Completeness Quality Level

Sample Example

[Figure 59]

Poor

[Figure 60]

Fair

[Figure 61]

Good

Table 12. Examples of Sample Quality Based on Filter Head Completeness

Examples of Sample Quality Based on Filter Head Resolution Quality Level

Sample Example

[Figure 62]

Poor

[Figure 63]

Fair

[Figure 64]

Good

Table 13. Examples of Sample Quality Based on Filter Head Resolution

Examples of Sample Quality Based on Filter Head Rotation Quality Level

Sample Example

[Figure 65]

Poor

[Figure 66]

Fair

[Figure 67]

Good

Table 14. Examples of Sample Quality Based on Filter Head Rotation

### E. Annotation Visualization

This section provides a visual overview of the generated annotations for the TalkVid-Core dataset.

- E.1. Annotation Details

We prompt the model to perform a detailed analysis focusing on anatomical movement patterns and behavioral dynamics, returning its findings as a structured JSON object. This process yields a rich set of 180,860 structured annotations. To quantify their richness, the descriptions have an average length of 84.6 tokens (σ = 28.4) and draw from a diverse vocabulary of over 18,000 unique tokens, underscoring the detail and variety of the generated text.

- E.2. Annotation Length Distribution

[Figure 68]

Figure 10. Distribution of caption lengths by token count across all 180k annotations in TalkVid-Core. The mean length is 84.6 tokens, with a standard deviation of 28.4, indicating a consistent descriptive depth.

- E.3. Qualitative Examples

### F. Details of Annotation Generation Process

This section provides the full prompt used for annotation generation and a representative example of the structured JSON output.

#### F.1. Full Generation Prompt

The following prompt was provided to a large multimodal model to instruct it on the analysis task and the required output format.

Analyze the video with precise focus on anatomical movement patterns and behavioral dynamics. Prioritize detailed descriptions of

body part trajectories and their temporal relationships.

IMPORTANT OUTPUT REQUIREMENTS:

- 1. Provide your analysis in a clean, minified JSON format without any line breaks or escape

characters

- 2. Do not include any explanatory text before or after the JSON

- 3. Ensure the JSON is valid and properly formatted

- 4. Use single-line format for the entire output

- 5. Do not include any comments or additional formatting

Expected JSON structure: {

"scene_context": { "background": ["Setting description", " Environmental elements", "Lighting details", "Camera angle and position"], "subject": ["Physical appearance", "Clothing description", "Notable features", " Demographic attributes"]

}, "movement_analysis": {

"head_movements": ["Sequential head rotations with angle measurements (in degrees)", "Tilt progression with directional markers", "

Orientation changes relative to initial position", "Facial expressions and changes"], "hand_actions": {

"left": ["Trajectory patterns with spatial coordinates", "Gesture type classification", "Interaction duration and objects", "Force/ intensity of movements"],

"right": ["Trajectory patterns with spatial coordinates", "Gesture type classification", "Interaction duration and objects", "Force/

intensity of movements"] }, "torso_movements": ["Rotation angles and direction", "Flexion/extension patterns", " Lateral movements", "Weight distribution shifts", "Postural stability assessment"], "body_posture": ["Postural transition timeline", "Spinal alignment changes", " Weight distribution shifts", "Balance and stability patterns"], "confidence_scores": {

"head_tracking": 0.0, "hand_tracking": 0.0, "torso_tracking": 0.0, "posture_analysis": 0.0, "overall_confidence": 0.0

}

}, "interactions": {

"environmental": ["Object interactions", " Space utilization", "Environmental adaptations"], "social": ["Interpersonal distances", "Social

gestures", "Interactive behaviors"], "object_handling": ["Object types", " Manipulation patterns", "Duration of interactions"]

}, "audio_behavioral_analysis": {

"speech": ["Voice characteristics", "Speech patterns", "Verbal expressions"], "non_verbal_sounds": ["Types of sounds", " Timing", "Context"]

}, "movement_metrics": {

"speed": {"unit": "meters/second", " measurements": []},

"acceleration": {"unit": "meters/secondˆ2", " measurements": []}, "angular_velocity": {"unit": "degrees/second ", "measurements": []}

}, "anomaly_detection": {

"unusual_movements": [], "irregular_patterns": [], "potential_concerns": []

}, "description_summary": "Comprehensive summary

describing the complete action sequence"

} RESPONSE FORMAT:

- - Output must be a single JSON object without any additional text

- - Do not include any markdown formatting

- - Do not include any explanatory text

- - Provide direct values without any placeholder text

- - The entire response should be valid JSON that can be parsed directly

#### F.2. Example of Generated JSON Output

This is a representative example of the structured JSON data produced by the model for a single video clip.

{

"scene_context": {

"background": [ "Indoor setting", "Plain, light-colored wall", "Even, bright lighting"

], "subject": [

"Middle-aged woman, light skin tone", "Dark pink/maroon long-sleeved top", "Shoulder-length blonde hair", "Presumably Caucasian"

]

}, "movement_analysis": {

"head_movements": [

"Slight head nodding throughout the video, particularly emphasizing certain words. Vertical rotation within approximately 5 degrees.",

"Minimal head tilt, maintaining a neutral position relative to the camera." ], "hand_actions": {

"left": [

"Initially rests at her side, out of frame.",

"At approximately 0:06 seconds, her left hand rises into the frame and executes a pointing gesture, moving upward in the sagittal plane and extending forward in the frontal plane.",

"The hand remains raised, with slight

movements emphasizing speech." ], "right": [

"Initially at her side and enters the frame slightly before the left hand, around

0:05 seconds.",

"Performs a similar pointing gesture as the left hand, rising in the sagittal plane and extending forward in the frontal plane, but with more pronounced movement.",

"The right hand is more active,

continuing to gesture with varied movements in all three planes (sagittal, frontal, and transverse) throughout the video to emphasize

points." ]

}, "body_posture": [

"The subject maintains a stationary, standing posture throughout the video.", "Upright posture with minimal spinal curvature changes.",

"Shoulders remain relaxed and relatively still.",

"Weight distribution appears evenly balanced." ]

}, "description_summary": "The video shows a woman

standing against a plain background, delivering a short explanation. She wears a dark pink top. From a medium close-up shot framing her from the chest up, she speaks directly to the camera. Throughout the video,

her head remains relatively still with subtle nodding for emphasis. Her hands, initially at her sides, rise into the frame to perform illustrative pointing gestures. The right hand exhibits more dynamic movement , emphasizing key points with a variety of gestures in multiple planes. Her overall body

posture remains static, standing upright and facing forward, with weight evenly

distributed. Her torso and shoulders show minimal movement."

}

[Figure 69]

The image collage shows a man sitting in a studio setting... He maintains a consistent seated posture... no significant body movement is observed.

[Figure 70]

The subject, a young woman, sits at a table... her primary movements are hand gestures

[Figure 71]

The subject, an adult female, is seated... She engages with the camera using a combination of hand gestures... She holds a small red box and gestures with it...

- Figure 11. Qualitative examples from the TalkVid-Core dataset. Each example displays sampled frames from a video clip, paired with its corresponding descriptive caption generated by Gemini 1.5 Pro. For brevity, captions are truncated.

[Figure 72]

- Figure 12. Positive Example (Passes All Filters). This visual guideline shows an ideal case that meets all quality criteria. The subject is consistently front-facing, stable, well-lit, and occupies a significant portion of the frame. This type of clip should be labeled as positive.

[Figure 73]

[Figure 74]

(a) Fails the Cotracker Filter due to it remains still. (b) Fails the Head Movement Filter due to abrupt scene changes.

[Figure 75]

[Figure 76]

(c) Fails the Head Rotation Filter due to distinct head turns. (d) Fails the Head Orientation Filter as the face is not front-facing.

[Figure 77]

[Figure 78]

(e) Fails the Head Completeness Filter due to facial occlusion. (f) Fails the Head Resolution Filter as the face is too small.

[Figure 79]

(g) Fails the Dover Filter due to motion blur and low quality.

- Figure 13. Visual guidelines illustrating negative examples for all seven filtering stages. Each sub-figure demonstrates a specific failure criterion, instructing evaluators to label such clips as negative.

