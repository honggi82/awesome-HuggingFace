## TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions

# arXiv:2602.08711v2[cs.CV]12Feb2026

#### Linli Yao12 Yuancheng Wei3 Yaojie Zhang4 Lei Li5 Xinlong Chen62 Feifan Song1 Ziyue Wang1 Kun Ouyang1 Yuanxin Liu1 Lingpeng Kong5 Qi Liu5 Pengfei Wan2 Kun Gai2 Yuanxing Zhang2 Xu Sun1

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

Audio-Visual Correlation

[Figure 12]

[Figure 13]

[Figure 14]

00:00 00:16 00:34 00:41 01:00

[Figure 15]

Omni Dense Captioning

###### Timestamp：00:34 – 00:41

Dialogue Content：Driver: 'Listen, brother-in-law, that's enough. I have to hurry back. It's my girlfriend's 60th birthday today, and she has no idea I took the car out.'

Detailed Events：From an overhead view, the white car continues to drive in circles. Inside, the curly-haired male driver is Xialuo's brotherin-law. He is a middle-aged East Asian man wearing a dark suit jacket and a dark blue shirt. He urges his brother-in-law, Xialuo, to stop showing off, explaining that he must leave immediately because it's his girlfriend's 60th birthday and he took the car without permission.

⚫ Timestamps

Camera State：The shot begins with a high-angle medium-long shot of a car. The camera then moves down and pans to the upper right, capturing a full shot of the car. Subsequently, the scene cuts to an exterior extreme close-up of the car, showing a shaky close-up of the driver through the windshield.

⚫ Detailed Events

⚫ Visual Background

⚫ Dialogue Content

⚫ Acoustics Content

Visual Background：The scene, from a bird's-eye view, shows the paved driveway of the manor and lush green trees.

⚫ Camera State

Shot Editing Style：The editing cuts between the high-angle shot and the driver's close-up, creating a contrast between the external scene of the car driving and the internal monologue that explains the driver's motivation.

⚫ Shot Editing Style

Acoustics Content：(1)Speaking tone：The brother-in-law's tone is impatient. (2)Background sound or music: Light, cheerful music continues to play.

Figure 1. Illustration of the OmniDenseCaptioning task. This paper introduces Omni Dense Captioning task, which generates fine-grained, temporally grounded descriptions for comprehensive audio-visual understanding. The term “dense” reflects two key properties: (1) temporally-dense: continuous scene segmentation with explicit timestamps, and (2) description-dense: structured captions spanning six dimensions: Audio-Visual Events, Visual Background, Camera State, Shot Editing, Dialogue, and Acoustic cues. These “script-like” descriptions allow readers to imagine the video scene-by-scene, as though reading a cinematic screenplay.

### Abstract

This paper proposes Omni Dense Captioning, a novel task designed to generate continuous, fine-grained, and structured audio-visual narratives with explicit timestamps. To ensure dense semantic coverage, we introduce a sixdimensional structural schema to create “scriptlike” captions, enabling readers to vividly imagine the video content scene by scene, akin to a cinematographic screenplay. To facilitate research, we construct OmniDCBench, a highquality, human-annotated benchmark, and pro-

1School of Computer Science, Peking University 2Kling Team, Kuaishou Technology 3South China University of Technology 4University of Electronic Science and Technology of China 5The University of Hong Kong 6Institute of Automation, Chinese Academy of Sciences. Correspondence to: Xu Sun <xusun@pku.edu.cn>.

Preprint. February 13, 2026.

pose SodaM, a unified metric that evaluates time-aware detailed descriptions while mitigating scene boundary ambiguity. Furthermore, we construct a training dataset, TimeChatCap-42K, and present TimeChat-Captioner-7B, a strong baseline trained via SFT and GRPO with task-specific rewards. Extensive experiments demonstrate that TimeChat-Captioner achieves state-of-the-art performance, surpassing Gemini-2.5-Pro, while its generated dense descriptions significantly boost downstream capabilities in audio-visual reasoning (DailyOmni and WorldSense) and temporal grounding (Charades-STA). All datasets, models, and code will be made publicly available at https://github.com/yaolinli/ TimeChat-Captioner.

### 1. Introduction

As video understanding (Xu et al., 2025b; Ye et al., 2025; Yao et al., 2024c; Sun et al., 2025; 2024; Ouyang et al., 2025; Yao et al., 2025b) and generation (Gan et al., 2025) enter the “sound” era, the alignment and interaction of omnimodal information (audio, visual, and text) have become pivotal research directions for Multimodal Large Language Models (MLLMs) (Arefeen et al., 2024; Bai et al., 2025a; Zhang et al., 2024; Yao et al., 2025a). Within this context, Omni-Video Captioning, which generates temporally grounded audio-visual-text triplets, emerges as a critical foundational task (Tang et al., 2025; Geng et al., 2025; Yuan et al., 2025a). Such high-quality audio-visual-text data provide comprehensive supervision signals that enable MLLMs to learn fine-grained cross-modal alignment during pre-training and post-training, while also benefiting downstream tasks such as Audio-Visual Reasoning (Zhou et al., 2025; Benchekroun et al., 2023) and Video-to-Audio Generation (Shi et al., 2025).

However, a performant omni-video captioning framework, accompanied by a dedicated benchmark and evaluation suite, remains a largely unexplored frontier in the open-source community. Existing audio-visual captioning (Tang et al., 2025; Wu et al., 2025) works primarily focus on generating global, paragraph-level descriptions without explicit timestamps. This lack of temporal granularity fails to provide the dense supervision signals necessary for MLLMs to master time-aware reasoning, such as temporal grounding (Wang et al., 2025b). On the other hand, traditional dense video captioning approaches (Ren et al., 2024; Yang et al., 2023) largely remain confined to the visual modality, neglecting the rich semantics embedded in audio. While recent advanced methods like LongVALE (Geng et al., 2025) have begun to incorporate audio cues, they predominantly focus on identifying salient events and generating concise summaries. This sparse and brief paradigm overlooks the continuous, fine-grained audio-visual nuances, thereby failing to capture the comprehensive semantics required for deep omni-modality alignment.

To bridge this gap, we propose a novel task Omni Dense Captioning with joint audio and visual semantics. Given a video with audio, the task goal is to semantically segment the input into continuous scenes and generate fine-grained audio-visual descriptions for each segment. Specifically, “dense” here entails two aspects: 1) dense timestamps, indicating continuous temporal segments that reveal the semantic scene changing and 2) dense captions, referring to finegrained descriptions covering the full audio-visual context (e.g., spatial attributes, actions, dialogue, and acoustic cues) along the temporal timeline. Unlike previous approaches that prioritize visual dominance, we explicitly enforce a sixdimension structural schema to ensure holistic audio-visual

coverage: (1) Overall Audio-Visual Events, (2) Background and Environment, (3) Camera State, (4) Multi-shot Editing Style, (5) Dialogue Content, and (6) Acoustic Cues. This structured design aims to produce “script-like” data where reading the captions allows one to reconstruct the video in imagination, scene-by-scene. These structural captions can serve as abundant supervision signals and provide downstream MLLMs with sufficient context for omni-video understanding or generation.

To facilitate research in this direction, we construct a high-quality benchmark named OmniDCBench, comprising 1,122 human-annotated samples. Evaluating this task presents unique challenges, particularly the ambiguity of continuous scene boundaries. To address this, we propose a novel unified metric SodaM, which jointly measures temporal timestamp accuracy and the semantic completeness of lengthy captions. SodaM incorporates a dynamic programming alignment process to mitigate the time boundary gap between model predictions and human references. Finally, we present a strong baseline TimeChat-Captioner-7B, trained on synthesized high-quality data via Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) stages. Extensive experiments demonstrate that TimeChat-Captioner not only achieves State-of-the-Art performance on OmniDCBench, surpassing Gemini-2.5Pro (Gemini Team, 2024), but also generates rich semantics that boost performance on downstream Audio-Visual Reasoning tasks like Daily-Omni (Zhou et al., 2025), and WorldSense (Benchekroun et al., 2023), and generalized to the temporal grounding task Charades-STA (Gao et al., 2017). We hope TimeChat-Captioner will deliver dense temporal and textual supervision that significantly enhances MLLMs’ omni-modal alignment capabilities.

### 2. Related Work

#### 2.1. Audio-Visual Captioning

Video captioning aims to generate textual descriptions of video content (Wang et al., 2024a; Yuan et al., 2025b; Wang et al., 2025a; 2024b; Yao et al., 2023; 2024b; 2022), with recent studies exploring fine-grained captioning that describes detailed temporal dynamics (Zhong et al., 2025; Li et al., 2024). The emergence of omni-modal models (Comanici et al., 2025; Xu et al., 2025a; AI et al., 2025) has shifted research from vision-centric to joint audio-visual understanding (Chen et al., 2025). Representative works include AVoCaDO (Chen et al., 2025) for audiovisual temporal coherence, video-SALMONN-2 (Tang et al., 2025), UGC-VideoCaptioner (Wu et al., 2025) for multimodal integration, and DiaDem (Chen et al., 2026) for dialogue-aware audiovisual captioning. However, these methods generate holistic captions without explicit temporal grounding. In contrast, TimeChat-Captioner outputs timestamped captions

with structured, fine-grained descriptions for each scene.

#### 2.2. Time-Aware Video Captioning

Dense video captioning (Krishna et al., 2017) localizes temporal segments and generates event-level descriptions, evolving from pipeline-based to end-to-end frameworks (Wang et al., 2021; Yang et al., 2023; Han et al., 2023). Recently, LongVALE (Geng et al., 2025) advances long-range temporal modeling for extended video durations, while ARCChapter (Pu et al., 2025) organizes videos into chapter-level units for structured descriptions. Despite these advances, existing methods typically generate sparse, event-centric captions with concise outputs or focus only on salient events. In contrast, OmniDenseCaptioning aims to capture comprehensive audiovisual semantics, producing multi-scene narratives with structured, fine-grained descriptions that cover all significant segments.

#### 2.3. Reinforcement Learning for Video Captioning

Reinforcement learning (RL) (Schulman et al., 2017; Guo et al., 2025; Zheng et al., 2025; Gao et al., 2025) has become an important paradigm in multimodal video understanding, particularly for aligning models with taskspecific objectives (Shao et al., 2025). CapRL (Xing et al., 2025) introduces verifiable rewards for caption generation, VideoCap-R1 (Meng et al., 2025) incorporates structured reasoning steps, and AVoCaDO (Chen et al., 2025) extends GRPO (Guo et al., 2025) with content coverage and length regularization rewards. Unlike these approaches targeting holistic quality, we propose SodaM, a reward that jointly optimizes temporal alignment and fine-grained coverage, applied within GRPO for temporally structured caption generation.

### 3. OmniDenseCaptioning Task and A New Benchmark

We first formally define the OmniDenseCaptioning task (Section 3.1). We then introduce OmniDCBench, a highquality benchmark with multi-dimensional scene-level annotations (Section 3.2). Finally, we propose SodaM, a unified metric that jointly evaluates temporal segmentation and caption quality (Section 3.3).

#### 3.1. Task Definition

Given an input video V with visual frames and audio signals, the goal of OmniDenseCaptioning is to generate detailed paragraph-level descriptions with explicit timestamps that segment the video into successive multi-scenes. The description for each scene should cover comprehensively audio-visual details to achieve the goal of that by reading it, a user can imagine the scene-by-scene visual plots with

synchronous audio information, as if they are watching the video.

What is a “Scene”? A scene is a semantically coherent video segment characterized by continuity in time, location, or narrative context. A single shot (Han et al., 2023) refers to one continuous camera take. In contrast, a scene may consist of multiple shots that together convey a unified event or situation. Scene boundaries are usually indicated by clear transitions in visual setting, audio context, or narrative progression.

Formally, let the video V be represented as a sequence of frames F = {f1,f2,...,fT} and audio signals A = {a1,a2,...,aT} over time steps T. The output script narrations S can be expressed as a sequence of scene-level finegrained captions C = {(t1,c1),(t2,c2),...,(tN,cN)}, where each scene description ci encompasses structural and multiple-dimension audio-visual captions, and ti denotes the timestamp MM:SS indicating the start and end time of each scene i in the video (e.g. “00:01-00:10”). The number of scenes N varies depending on the video’s specific content.

Specifically, we design each scene description to comprehensively cover six dimensions: (1) Overall Audiovisual Events (Events): detailed narration of audiovisual content and actions; (2) Background and Environment (Background): depiction of the setting, location, and atmosphere; (3) Camera State (Camera): description of camera movements, angles, and framing; (4) Multi-shot Editing Style (ShotEdit): description of post-production editing techniques and how multiple shots are organized, such as montage sequences; (5) Dialogue Content (Dialogue): transcription and summary of spoken words and conversations with corresponding speakers; (6) Acoustic Cues (Acoustic): portrayal of background sounds, music, and auditory ambiance.

These dimensions collectively cover holistic spatial and temporal context, fine-grained visual-audio cues, camera state, and shot editing techniques to produce high-quality detailed descriptions. We highlight the critical differences between the OmniDenseCaptioning task and existing dense video captioning (Yang et al., 2023; Geng et al., 2025) task:

- 1) Comprehensive Visual-Audio Coverage: Unlike dense video captioning that produces sparse event descriptions focusing only on salient moments, OmniDenseCaptioning aims to generate comprehensive and successive multi-scene narratives covering all significant scenes in a video, providing a holistic understanding of both visual and auditory content.
- 2) Structured and Fine-grained Output: Whereas existing methods typically provide brief descriptions spanning only a few sentences, OmniDenseCaptioning is designed to generate structured, comprehensive narratives across six

|| |
|---|
| |
|[Figure 16]<br><br>14.16|
| |
| |
| |
| |
| |
|[Figure 17]|
| |
|10.85|
| |
| |
| |
| |
| |
| |
| |
|[Figure 18]<br><br>7.46|
| |
| |
| |
| |
| |
|[Figure 19]|
| |
|4.15|
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
<br><br>0<br><br>2<br><br>4<br><br>6<br><br>8<br><br>10<br><br>12<br><br>14<br><br>16|
|---|

200

| |
|---|
| |
| |
| |
| |

|[Figure 20]<br><br>< 40s<br><br>[Figure 21]<br><br>3.03%<br><br>[Figure 22]<br><br>40-50s<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>12.57%<br><br>[Figure 26]<br><br>50-60s<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>35.29%<br><br>[Figure 30]<br><br>60-70s<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>45.10%<br><br>[Figure 34]<br><br>> 70s<br><br>[Figure 35]<br><br>4.01%<br><br>Frequency|
|---|

Avg. word num

180

Total 995

160

Events 343 Camera 298

140

###### Duration(s)

Frequency

Visual background 103

120

Acoustics content 89 ShotEdit style 88 Dialog content 75

100

80

60

40

20

0

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

223 563 903 1243 1923

1583

Human Captioner(ours) Gemini-2.5-Pro Qwen3-Omni

(a) Video Duration (b) Caption Length

(c)Avg. Scene Duration

- Figure 2. Statistics of human-annotated OmniDCBench. (a) Video duration distribution. (b) Caption length distribution with per-dimension. The benchmark features comprehensive annotations averaging 995 words per video. (c) Scene duration distribution (in seconds), compared against MLLM-generated outputs to highlight the granularity gap between human and model segmentations.

distinct dimensions. This approach enables the capture of subtle, nuanced visual and audio details, resulting in richer and more informative captions.

#### 3.2. Benchmark Dataset Curation

To support this novel and challenging task, we construct a high-quality benchmark OmniDCBench, through meticulous manual annotation.

Data Source. Ensuring video diversity and complexity is crucial for constructing representative multi-scene scripts. To this end, we curate a collection of high-resolution, clearsound movie clips from Movie101 (Yue et al., 2025), as well as diverse general YouTube videos from YT-Temporal1B (Zellers et al., 2022), thus providing a broad range of content for our benchmark.

Fully Manual Annotation Pipeline. To ensure the highest quality and reliability of our benchmark, all data is carefully annotated and verified entirely by human experts through a rigorous, systematic pipeline.

We structure the annotation process into three meticulous stages. First, crowd-sourced annotators review the pool of candidate videos, filtering out low-quality or unsuitable sources and assigning difficulty-level tags for annotation. Second, annotators watch each video in its entirety and segment it into multiple scenes, assigning continuous timestamps from a holistic perspective. Third, to annotate the six-dimensional scene descriptions, we assign different annotators to specific dimensions, as each requires distinct expertise. For instance, the Camera State and Shot Editing Style fields demand specialized knowledge of cinematography. To further ensure data integrity, both the timestamp and caption annotations are double-checked by independent annotators.

Data Statistics. Through this rigorous annotation process, OmniDCBench comprises 1,122 videos with comprehensive and detailed multi-scene descriptions. As illustrated in Figure 2, a key characteristic of the dataset is the depth and

richness of its annotations, with descriptions averaging 995 words per video.

#### 3.3. Evaluation Metric Design

An ideal evaluation framework for OmniDenseCaptioning should measure the alignment between predicted and ground-truth outputs in terms of both temporal boundaries and descriptive content. This presents three key challenges: (1) assessing the accuracy of timestamp predictions (Timestamp Accuracy), (2) measuring the quality of fine-grained, multi-dimensional paragraph descriptions (Caption Quality), and (3) proposing a unified metric that jointly considers temporal alignment and caption quality across variable-length scene sequences (Unified Metric).

Formally, let the predicted output be P = {(tˆ1,cˆ1),...,(tˆM,cˆM)} and the ground-truth be G = {(t1,c1),...,(tN,cN)}, where M and N denote the number of predicted and ground-truth scenes, respectively.

For clarity, we first outline the evaluation of timestamp accuracy and caption quality for a matched predicted and ground-truth scene pair < (tˆi,cˆi),(tj,cj) >.

Timestamp Accuracy. Given a predicted timestamp tˆ = [tˆs,tˆe] and a ground-truth timestamp t = [ts,te], we compute the Intersection over Union (IoU) (2018) as:

IoU(t,tˆ ) = |tˆ∩ t| |tˆ∪ t|

(1)

Caption Quality. Conventional metrics (BLEU, METEOR, CIDEr) rely on n-gram matching and are ill-suited for paragraph-length, multi-dimensional descriptions. Drawing inspiration from recent advances (Tang et al., 2025; Chen et al., 2025), we employ the CheckList Score for caption evaluation. Specifically, for each dimension d ∈ D, the ground-truth caption c is decomposed into a set of atomic elements Ed = {e1,e2,...,e|E

d|}. The predicted caption is

###### Stage 1: SFT

00:00-00:10

###### 00:11-00:20 time

Time token num ~ 0.7 % Cap token num ~ 99.3%

Visual: An ornate black iron gate, decorated with ... Background: The scene is the entrance to a luxurious … Camera: The camera starts with a low-angle medium shot … Cinematic: The scene combines shots from different angle … Dialogue: No dialogue. Acoustics: A light, cheerful piece of music begins to play ...

Visual: A white Maserati drives past a... Background: The main entrance of a … Camera: An eye-level medium shot …

Next Token Prediction

GT 00:00 – 00:17, Visual: A flower-covered black gate opens

###### …

Cinematic: This scene utilizes close-ups ... Dialogue: Driver: ‘Listen, brother-in-law, that’s … Acoustics: The brother-in-law's tone is impatient ...

00:00 – 00:10, Visual: An ornate black iron gate... Background: The scene is the entrance to a luxurious … Camera: The camera starts with a low-angle medium shot …

Pred

Qwen2.5-Omni Backbone

###### Stage 2: GRPO

{

00:00 – 00:10, Visual: An ornate black iron gate... Background: The scene is the entrance to a luxurious … Camera: The camera starts

Timestamp: [00:00, 00:10], Visual: An ornate black iron … Background: The scene is the … Camera: The camera starts …

###### Pred

… Interleaved Audio-Visual Tokens

[Figure 46]

|Text Prompt|
|---|

Parse

… }, …

with a low-angle medium shot …

###### JSON

Your task is to generate dense captions using joint audio and visual semantics . The goal is to segment the video into multiple meaningful intervals and provide six dimension fine-grained descriptions for each segment …

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Vision Encoder

Frames

[Figure 51]

[Figure 52]

Caption Reward GT Keypoints

Time Reward Pred [0, 10]

[Figure 53]

Format Check

###### Pred Caps

[Figure 54]

IoU

Flower-covered … Gate opens

An ornate black iron gate...

Audio Encoder

GT [0, 17]

[Figure 55]

[Figure 56]

[Figure 57]

Audio

[Figure 58]

[Figure 59]

Length Check

- Figure 3. Overview of TimeChat-Captioner Architecture. (Left) This model leverages Qwen2.5-Omni (Xu et al., 2025a) with interleaved audio-visual tokens to generate multi-scene timestamps and six-dimensional captions. (Right) Two-stage training: SFT for task format learning, followed by GRPO with rewards for format, length, timestamp accuracy, and time-aware fine-grained caption quality.

then assessed against each of these elements as follows:

merged prediction:

tˆmerged = [min(tˆk,s,...),max(tˆk,e,...)] (4) cˆmerged = Concat(ˆck,...,cˆk+l) (5)

|Ed|

1 d∈D |Ed| d∈D

Judge(ˆc,ei) (2)

CheckList(ˆc,c) =

i=1

where s and e denote the start time and end time of each predicted scene, respectively. This handles the common case where MLLMs generate finer-grained (shorter) segments than the ground-truth as shown in Figure 2 (c). We only merge predictions while keeping the ground-truth unchanged to ensure evaluation fairness.

Here, Judge(ˆc,ei) ∈ {0,1} indicates if cˆcovers element ei from a judge model Gemini-2.5-Flash. By averaging across all dimensions, we obtain the final CheckList score.

A Unified Metric SodaM. The key challenge is jointly assessing timestamp accuracy and caption quality without a natural one-to-one correspondence between M and N. Since “scene” is an inherently semantic concept with ambiguous boundaries, different models (and even humans) may produce varying numbers of segments for the same video, as illustrated in Figure 2 (c). This necessitates an alignment step before evaluation.

After alignment, we obtain K temporally matched pairs M = {(ˆp1,g1),...,(ˆpK,gK)} where K ≤ N. We then compute Timestamp Accuracy for each matched ⟨pred,gt⟩ pair and report the F1 score across thresholds {0.3,0.5,0.7,0.9}, as well as the mean IoU, to assess overall segmentation quality following (Liu & Yao, 2018). To evaluate temporally-aware caption quality, we calculate the CheckList Score for each matched pair and compute the F1 score for all pairs to obtain the final SodaM score following (Fujita et al., 2020).

The core idea of SodaM is a two-stage alignment strategy:

- 1. IoU-based Dynamic Programming Alignment: First, we find an optimal path through the (M,N) scene grid of ⟨pred,gt⟩ pairs, using only temporal IoU as the scoring cost.

S[i][j] = max

 



S[i − 1][j] S[i][j − 1] S[i − 1][j − 1] + IoU(ti,tˆj)

(3)

- 2. Merging of Many-to-One Predictions: Whenever multiple predicted scenes {pˆk,...,pˆk+l} are aligned to the same ground-truth scene gi, we concatenate their captions and expand their timestamp range to form a single

Summary. Compared to SODAc (Fujita et al., 2020), SodaM: (1) reduces judge-model cost from O(MN) to O(K) where K ≤ N by decoupling IoU matching from text evaluation, and (2) gracefully handles many-to-one alignments through merging, mitigating scene boundary ambiguity while ensuring holistic semantic coverage.

### 4. The TimeChat-Captioner Framework

We introduce TimeChat-Captioner, a specialized Video Large Language Model tailored for the OmniDenseCap-

Table 1. Quantitative comparison on the OmniDenseCaptioning task. Bold and underline highlight the best and second-best results among open-source models, respectively. † indicates expert models specialized in temporal-aware captioning. SodaM is the primary metric reflecting the quality of temporally-aligned, multi-dimensional captions.

Multi-Scene Seg. Time-aware Dense Captioning Quality F1 mIoU Camera Events Background Acoustics ShotEdit Dialogue

Model Modality

SodaM (Avg.) Proprietary Models

Gemini-2.5-Pro V + A 68.5 74.9 8.1 48.1 39.1 25.4 34.5 46.4 33.7 Gemini-2.5-Flash V + A 45.6 53.1 11.5 38.1 42.1 22.4 27.6 42.6 30.0

Open-source Models

|LongVALE†(7B) (2025) V + A Qwen2.5-Omni(7B) (2025a) V + A MiniCPM-o-2.6(8B) (2024c) V + A OmniVinci(9B) (2025) V + A Qwen3-Omni(30B-A3B) (2025b) V + A|45.2 55.6 0.8 0.6 1.3 0.5 3.1 5.0 1.8 37.2 43.4 1.6 3.9 12.3 3.4 3.3 15.7 4.6 49.9 60.2 1.2 5.3 16.5 1.5 7.4 11.3 5.4 29.0 39.7 1.6 8.2 15.7 1.3 6.7 14.3 6.9 54.8 64.2 3.1 20.2 21.6 5.1 14.1 25.4 14.3<br><br>|
|---|---|
|TimeChat-Captioner-7B-SFT (Ours) V + A TimeChat-Captioner-7B-GRPO (Ours) V + A<br><br>|62.4 70.8 8.9 30.5 36.3 30.6 33.9 44.8 32.6 61.2 69.6 12.4 39.6 49.2 38.2 43.5 54.3 35.0<br><br>|

tioning task. Built upon joint audio-visual understanding, TimeChat-Captioner achieves accurate multi-scene timestamp prediction while generating fine-grained, structured descriptions for each segment.

#### 4.1. Overall Architecture

As illustrated in Figure 3, we build TimeChat-Captioner upon the Qwen2.5-Omni (Xu et al., 2025a) backbone, leveraging its Thinker module for joint audio-visual perception with the Vision Encoder from Qwen2.5-VL (Bai et al., 2025b) and the Audio Encoder from Qwen2-Audio (Chu

- et al., 2024).

This backbone incorporates two key designs tailored to the requirements of OmniDenseCaptioning. First, it arranges audio and visual tokens in a temporally interleaved sequence, enabling synchronous cross-modal comprehension—unlike traditional methods that process each modality in isolation (Geng et al., 2025). Second, it employs Multimodal Rotary Position Embedding (M-RoPE) (Bai et al.,

- 2023) to encode absolute temporal positions, thereby facilitating precise scene boundary localization and continuous timestamp prediction.

#### 4.2. Training Data Collection

To construct high-quality training data for OmniDenseCaptioning, we develop a synthetic data pipeline powered by Gemini-2.5-Pro, as depicted in Figure 6. This pipeline proceeds through three stages: video source selection, a two-step caption generation process, and quality filtering.

Video Source Sampling. We curate videos from two complementary datasets: (1) MMTrail-2M (2024), which is dominant and features a diverse and carefully cleaned collection of trailer videos spanning a broad range of topics;

and (2) Movie101 (2025), which consists of movie commentary videos with abundant and rich audiovisual content. To balance annotation quality and information density, we segment the raw videos into 3-minute clips.

Two-Step Construction Pipeline. Recognizing that Gemini-2.5-Pro cannot reliably produce high-quality task data in a single pass, we adopt a coarse-to-fine approach:

- • Stage 1: Boundary Segmentation. Gemini-2.5-Pro analyzes each 3-minute clip to generate temporal segmentations accompanied by brief captions (e.g., “0:00-0:15: a boy singing...”).
- • Stage 2: Detailed Caption Generation. Using the Stage 1 segmentations as scaffolding, Gemini-2.5-Pro is prompted to produce fine-grained, multi-dimensional descriptions for each segment, comprehensively covering all six dimensions outlined in Section 3.1. Detailed prompts are provided in the appendix.

Data Quality Filtering. We ensure the fidelity of the training data through careful filtering: videos with fewer than two scene segments, lacking audio tracks, containing JSON formatting errors or missing caption fields, as well as segments below a minimum duration, are all excluded from the final dataset.

Summary. Upon completion of filtering, we obtain 42K high-quality time-aware video-caption pairs, which constitute the final training dataset. It is worth noting that our training data is entirely independent from the benchmark in terms of both video sources and annotation schema (synthetic annotations for training vs. manual annotations for evaluation), ensuring a fair assessment of generalization.

###### Table 2. Results on Omni-VideoQA benchmarks.

Model Size Daily-Omni World-Sense General Closed-source Models

Gemini-2.5-Pro – 60.2 33.8 Gemini-2.5-Flash – 55.3 31.0

General Open-source Models

|HumanOmniV2 (2025a) 7B ARC-Hunyuan-Video (2025) 7B MiniCPM-o-2.6 (2024c) 8B Qwen2.5-Omni (2025a) 7B UGC-VideoCaptioner (2025) 3B video-SALMONN-2 (2025) 7B Qwen3-Omni-Instruct (2025b) 30B-A3B Qwen3-Omni-Captioner(2025b) 30B-A3B<br><br>|8.2 6.6<br><br>8.6 8.7<br>9.8 7.2 13.4 8.6 17.0 11.2 29.9 18.2 17.5 12.7 27.2 14.1<br>|
|---|---|
|TimeChat-Captioner-7B-GRPO (Ours) 7B<br><br>|52.8 22.6<br><br>|

advantage for each response oi is computed relative to the group:

ri − mean({r1,...,rG}) std({r1,...,rG})

(6) The policy is then optimized using the following objective:

Ai =

G

πθ(oi|q) πθold(oi|q)

1 G

JGRPO(θ) = E

Ai,

min

i=1

πθ(oi|q) πθold(oi|q)

clip

, 1 − ϵ, 1 + ϵ Ai − β · DKL πθ∥πref

(7)

#### 4.3. Training Strategy

OmniDenseCaptioning is a challenging task that requires both accurate temporal segmentation and lengthy, structured textual output. To build a performant specialist model, we adopt Supervised Fine-Tuning (SFT) to teach the model the task format, followed by Group Relative Policy Optimization (GRPO) (Shao et al., 2025) strategy to jointly improve timestamp accuracy and caption quality.

- 4.3.1. SFT STAGE

We first fine-tune the Qwen2.5-Omni backbone on our training data using standard next-token prediction loss (Gui et al., 2024). The input consists of raw video frames and audio wavs, while the target output follows our structured format with timestamps and multi-dimensional captions. This stage enables the model to follow the basic output format and preliminarily learn this complex task.

- 4.3.2. GRPO STAGE

While SFT teaches the model to mimic the training distribution, it has inherent limitations for OmniDenseCaptioning:

where ϵ is the clipping threshold and β controls the KL divergence penalty from a reference policy πref. Reward Design. We design task-specific rewards to boost timestamp accuracy and caption quality:

- • Format Reward RF: A binary reward indicating whether the output can be parsed as a valid JSON list P = {(tˆ1,cˆ1),...,(tˆM,cˆM)}. If parsing succeeds, RF = 1; otherwise RF = 0.
- • Length Reward RL: To prevent the model from generating overly lengthy outputs prone to hallucination or repetitive content that fails to terminate, we apply a lengthregularized reward following (Chen et al., 2025)
- • Timestamp Reward RT: The average F1 score at IoU thresholds {0.3,0.5,0.7,0.9} between predicted and groundtruth timestamps along the optimal alignment path (as introduced in Section 3.3).
- • Time-aware Caption Reward RC: We adopt the unified SodaM metric as the reward to encourage comprehensive and temporally-aligned structural captions.

- • Token Imbalance: Timestamp-related tokens constitute only a small fraction of the output (0.7%), while caption tokens dominate. Standard cross-entropy loss treats all tokens equally, providing insufficient gradient signal for accurate temporal prediction.
- • Limited Generalization: SFT models tend to overfit to the scene count distribution in training data, struggling to generalize to videos with varying numbers of scenes.

To address these issues, we adopt Group Relative Policy Optimization (GRPO) (2025), a reinforcement learning algorithm that eliminates the need for a separate critic model

- as in PPO. For each training sample with input q, we sample G candidate outputs {o1,o2,...,oG} from the current pol-

icy πθ

and compute their rewards {r1,r2,...,rG}. The

old

The final reward R is a weighted sum of these components: R = αf · RF + αl · RL + αt · RT + αc · RC (8)

where each hyperparameter α controls the contribution of its respective reward.

### 5. Experiments

#### 5.1. Experimental Setup

Implementation Details. We adopt a two-stage training pipeline: SFT on 40K training samples for 2 epochs (lr=5e-5, batch size=128), followed by GRPO on 2K training samples for 1 epoch (lr=1e-5, batch size=64, rollout=8). Reward weights for format, length, temporal, and caption quality are set to 0.5, 0.5, 1.0, and 1.0, respectively. Videos are sampled

Table 3. Temporal grounding performance on Charades-STA. All models are fine-tuned on the Charades-STA training set. Models marked with † are expert models.

Charades-STA R1@0.3 R1@0.5 R1@0.7 mIoU

Method

TimeChat† (2024) – 46.7 23.7 – TimeSuite† (2024) 79.4 67.1 43.0 – TimeExpert† (2025b) – 64.1 43.3 –

Qwen2.5-Omni-7B 78.3 65.9 44.1 56.7 Ours 79.8 68.7 48.3 58.8

Table 4. Ablation study on training data scale and reward components. RC denotes the time-aware caption reward SodaM.

###### Model Variant OmniDCBench Daily-Omni

Qwen2.5-Omni 4.6 13.4 SFT (20K) 31.3 49.3 SFT (40K) 32.6 50.7 GRPO (w/o RC) 32.5 50.4 GRPO (w/ RC) 35.0 52.8

- at 2 FPS and the training maximum sequence length is 32K tokens. All experiments are conducted on 32×80G GPUs. Further details are provided in the Appendix.

Baselines. To comprehensively evaluate the performance of our proposed method, we compare it against three categories of models: (1) Closed-source MLLMs, including leading commercial systems such as Gemini-2.5-Pro and Gemini2.5-Flash; (2) Open-source MLLMs, including representative video-language models like Qwen2.5-Omni, MiniCPMo-2.6, and video-SALMONN-2; (3) Expert Models, which are specialized for temporal video understanding, such as LongVALE, TimeChat, and TimeSuite. Detailed results across multiple benchmarks are reported in the following sections.

#### 5.2. Main Results on OmniDCBench

As summarized in Table 1, TimeChat-Captioner achieves highly competitive results on the Omni-Video Scripting Benchmark. Regarding scene boundary localization, our model ranks second only to the industry-leading proprietary model, Gemini-2.5-Pro, while significantly outperforming all other open-source baselines. For time-aware captioning quality, evaluated via the SodaM metric (an aggregate average across six dimensions: camera, events, background, acoustics, shot editing, and dialogue), TimeChat-CaptionerGRPO achieves state-of-the-art (SOTA) performance with a score of 35.0. This result even surpasses the strongest closed-source model, Gemini-2.5-Pro, demonstrating that RL effectively improves accurate scene segmentation and fine-grained captioning. Qualitative cases are visualized in Figure 4.

#### 5.3. Results on Omni-VideoQA Benchmarks

Caption quality is assessed indirectly: each model produces audiovisual descriptions, which are used by Gemini-2.5Pro to answer QA questions. Higher QA accuracy reflects richer, more complete captions. As summarized in Table 2, although TimeChat-Captioner is primarily tailored for temporal-aware detailed video captioning, it demonstrates remarkable generalization on general VideoQA tasks. On

both Daily-Omni and World-Sense benchmarks, our model consistently outperforms all open-source baselines by a substantial margin, achieving scores of 52.8 and 22.6, respectively. This indicates that even under different data distributions, our task with dense audio-visual semantics has more comprehensive information to boost the downstream VideoQA task.

#### 5.4. Results on Temporal Grounding Benchmarks

To evaluate the generalization and transferability of our model in fine-grained temporal video understanding, we report the fine-tuning results on the Charades-STA (2017) benchmark in Table 3. TimeChat-Captioner-GRPO achieves superior performance across all evaluation metrics. Remarkably, our model consistently outperforms established expert models specifically designed for temporal video understanding tasks, such as TimeSuite and TimeExpert, as well as the Qwen2.5-Omni-7B baseline. These results validate that trained on OmniDenseCaptioning task with TimeChatCap-42K significantly enhances the model’s fundamental temporal understanding, thereby strengthening performance on downstream temporal grounding tasks.

#### 5.5. Ablation Studies

Impact of Data Scale. We first evaluate the effect of supervised SFT data quantity in Table 4. Increasing the training data from 20K to 40K samples leads to a consistent performance gain across all benchmarks, with the OmniDCBench score rising from 31.3 to 32.6. This demonstrates that more scripting data provides a stronger performance.

Effectiveness of SodaM Reward. Base rewards (format, length, and temporal alignment) ensure structural validity and basic temporal accuracy. Our ablation focuses on the unified SodaM reward (RC), which targets time-aware caption quality. As shown in Table 4, removing RC yields a stable but limited baseline, while incorporating RC significantly improves both temporal understanding and caption completeness, and even yields remarkable gains on the outof-domain DailyOmni benchmark. This validates that optimizing for time-aware dense captioning quality serves as an effective proxy task for enhancing general audiovisual com-

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

###### 00:00 00:16 00:34 00:41 01:00

Ground Truth

###### Timestamp：00:34 – 00:41

Dialogue Content：Driver: 'Listen, brother-in-law, that's enough. I have to hurry back. It's my girlfriend's 60th birthday today, and she has no idea I took the car out.'

Detailed Events：From an overhead view, the white car continues to drive in circles. Inside, the curly-haired male driver is Xialuo's brotherin-law. He is a middle-aged East Asian man wearing a dark suit jacket and a dark blue shirt. He urges his brother-in-law, Xialuo, to stop showing off, explaining that he must leave immediately because it's his girlfriend's 60th birthday and he took the car without permission.

Camera State：The shot begins with a high-angle medium-long shot of a car. The camera then moves down and pans to the upper right, capturing a full shot of the car. Subsequently, the scene cuts to an exterior extreme close-up of the car, showing a shaky close-up of the driver through the windshield.

Visual Background：The scene, from a bird's-eye view, shows the paved driveway of the manor and lush green trees.

Shot Editing Style：The editing cuts between the high-angle shot and the driver's close-up, creating a contrast between the external scene of the car driving and the internal monologue that explains the driver's motivation.

Acoustics Content：(1)Speaking tone：The brother-in-law's tone is impatient. (2)Background sound or music: Light, cheerful music continues to play.

###### Ours (7B Model)

Timestamp：00:32 – 00:42

Detailed Events：Luo continues to drive in circles, explaining that his girlfriend doesn't know he drove her car. In the back seat, a man with a distinctive, spiky haircut and a black suit, Xia Luo, leans forward to peer out the window, searching for something. The driver tells him to slow down and go for a couple more laps.

(2)Background sound or music: The hum of the car's engine. Dialogue Content：Luo: She doesn't know I drove her car here. Xia Luo: Alright, drive slowly. Go round a couple more times.

Camera State：The camera is positioned inside the car, looking at the driver and passenger. It then cuts to a close-up of the man in the back seat as he looks out the window.

Visual Background：The interior of the luxury Maserati is visible, with the opulent courtyard seen through the windows. Acoustics Content：(1)Speaking tone：Anxious, then commanding.

Shot Editing Style：The scene uses interior shots to focus on the characters' interactions and motivations

Fine-grained Alignment

[Figure 74]

###### Gemini-2.5-Pro

Timestamp：00:31 – 00:43 From inside the car, a woman driving tells her brother-in-law that she has to leave soon because it's her girlfriend's 60th birthday and she secretly took the car. The man in the passenger seat, Xia Luo, tells her to drive slowly and go around a couple more times.

Severe Hallucination

[Figure 75]

Qwen-3-Omini

Timestamp：00:35 – 00:46 A man dressed in a red ceremonial uniform with a tall black hat and white gloves stands by the entrance of the building, smiling and looking towards the camera. Timestamp：00:37 – 00:38 The man in the red uniform walks over to the white Maserati and opens the driver's side door, preparing to assist

the driver. Timestamp：00:39 – 00:40 An overhead shot shows the courtyard with the fountain and the white Maserati parked nearby. A man in a blue shirt walks across the courtyard.

Missed Main Event

[Figure 76]

- Figure 4. Qualitative case analysis. We compare TimeChat-Captioner with Gemini-2.5-Pro and Qwen-3-Omni on a sample from OmniDCBench. Our model achieves fine-grained alignment with the ground truth across all six annotation dimensions: detailed events, visual background, acoustics, dialogue, camera state, and shot editing style. In contrast, Gemini-2.5-Pro (Gemini Team, 2024) exhibits severe hallucination by misidentifying the male driver as a woman, fundamentally distorting the scene semantics. Qwen-3-Omni (Xu

- et al., 2025b) misses the main event entirely, describing irrelevant background elements (a doorman in red uniform) while ignoring the central conversation inside the car. These results demonstrate TimeChat-Captioner’s superior capability in accurate character recognition, faithful event grounding, and comprehensive multi-dimensional annotation.

### 6. Conclusions

prehension capabilities. Notably, the GRPO strategy with merely 2K training samples proves more effective than scaling up SFT training data from 20K to 40K, demonstrating the efficiency of our reward-guided optimization approach.

We introduce OmniDenseCaptioning, a novel task for generating temporally-aligned, multi-dimensional, and struc-

turally rich video captions. We present the high-quality, human-annotated OmniDCBench benchmark and propose tailored metrics such as SodaM to advance research in this area. Our specialized TimeChat-Captioner model, trained with synthetic audio-visual data and task-specific rewards, outperforms the proprietary Gemini-2.5-Pro and demonstrates strong generalization to related omni-video understanding tasks. We hope our approach will promote comprehensive omni-video understanding and support future multi-scene video generation by providing abundant, highquality, fine-grained data.

Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,

- Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
- Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

### Impact Statement

This paper advances omni-video understanding through dense, temporally-grounded audiovisual captioning. Positive impacts include improved accessibility for impaired users and enhanced video-based education. Potential risks involve inherited biases from pre-trained models and possible misuse for misinformation.

To ensure transparency and mitigate risks: (1) all videos in our training and evaluation datasets are sourced exclusively from publicly available academic datasets (MMTrail (Chi et al., 2024) and Movie101 (Yue et al., 2025)), with no private data collected; (2) we document all data sources and model limitations; and (3) we release resources under responsible use licenses. We believe the benefits outweigh the risks when appropriate safeguards are followed.

### References

AI, I., Gong, B., Zou, C., Zheng, C., Zhou, C., Yan, C., Jin, C., Shen, C., Zheng, D., Wang, F., et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025.

Arefeen, M. A., Debnath, B., Uddin, M. Y. S., and Chakradhar, S. Vita: An efficient video-to-text algorithm using vlm for rag-based video analysis system. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2266–2274, 2024.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 1(2):3, 2023.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z.,

Benchekroun, Y., Dervishi, M., Ibrahim, M., Gaya, J.-B., Martinet, X., Mialon, G., Scialom, T., Dupoux, E., Hupkes, D., and Vincent, P. Worldsense: A synthetic benchmark for grounded reasoning in large language models. 2023.

Chen, X., Ding, Y., Lin, W., Hua, J., Yao, L., Shi, Y., Li, B., Zhang, Y., Liu, Q., Wan, P., et al. Avocado: An audiovisual video captioner driven by temporal orchestration. arXiv preprint arXiv:2510.10395, 2025.

Chen, X., Lin, W., Hua, J., Yao, L., Ding, Y., Li, B., Zeng, B., Shi, Y., Liu, Q., Zhang, Y., et al. Diadem: Advancing dialogue descriptions in audiovisual video captioning for multimodal large language models. arXiv preprint arXiv:2601.19267, 2026.

Chi, X., Wang, Y., Cheng, A., Fang, P., Tian, Z., He, Y., Liu, Z., Qi, X., Pan, J., Zhang, R., et al. Mmtrail: A multimodal trailer video dataset with language and music descriptions. arXiv preprint arXiv:2407.20962, 2024.

Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou, C., and Zhou, J. Qwen2audio technical report. arXiv preprint arXiv:2407.10759, 2024.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Fujita, S., Hirao, T., Kamigaito, H., Okumura, M., and Nagata, M. Soda: Story oriented dense video captioning evaluation framework. In European Conference on Computer Vision, pp. 517–531. Springer, 2020.

Gan, Q., Yang, R., Zhu, J., Xue, S., and Hoi, S. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation, 2025. URL https: //arxiv.org/abs/2506.18866.

Gao, C., Zheng, C., Chen, X.-H., Dang, K., Liu, S., Yu,

- B., Yang, A., Bai, S., Zhou, J., and Lin, J. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347, 2025.

Gao, J., Sun, C., Yang, Z., and Nevatia, R. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pp. 5267–5275, 2017.

Ge, Y., Ge, Y., Li, C., Wang, T., Pu, J., Li, Y., Qiu, L., Ma, J., Duan, L., Zuo, X., et al. Arc-hunyuan-video7b: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939, 2025.

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv preprint, abs/2403.05530, 2024.

Geng, T., Zhang, J., Wang, Q., Wang, T., Duan, J., and Zheng, F. Longvale: Vision-audio-language-event benchmark towards time-aware omni-modal perception of long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18959–18969, 2025.

Gui, J., Chen, T., Zhang, J., Cao, Q., Sun, Z., Luo, H., and Tao, D. A survey on self-supervised learning: Algorithms, applications, and future trends, 2024. URL https:// arxiv.org/abs/2301.05712.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Han, M., Yang, L., Chang, X., and Wang, H. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. arXiv preprint arXiv:2312.10300, 2023.

Krishna, R., Hata, K., Ren, F., Fei-Fei, L., and Carlos Niebles, J. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pp. 706–715, 2017.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, L., Liu, Y., Yao, L., Zhang, P., An, C., Wang, L., Sun, X., Kong, L., and Liu, Q. Temporal reasoning transfer from text to video. arXiv preprint arXiv:2410.06166, 2024.

Liu, Y. and Yao, M. Best vision technologies submission to activitynet challenge 2018-task: Dense-captioning events in videos, 2018. URL https://arxiv.org/abs/ 1806.09278.

Meng, D., Huang, R., Dai, Z., Li, X., Xu, Y., Zhang, J., Huang, Z., Zhang, M., Zhang, L., Liu, Y., et al. Videocapr1: Enhancing mllms for video captioning via structured thinking. arXiv preprint arXiv:2506.01725, 2025.

Ouyang, K., Liu, Y., Yao, L., Cai, Y., Zhou, H., Zhou, J., Meng, F., and Sun, X. Conan: Progressive learning to reason like a detective over multi-scale visual evidence. arXiv preprint arXiv:2510.20470, 2025.

Pu, J., Wang, T., Ge, Y., Ge, Y., Li, C., and Shan, Y. Arc-chapter: Structuring hour-long videos into navigable chapters and hierarchical summaries. arXiv preprint arXiv:2511.14349, 2025.

Ren, S., Yao, L., Li, S., Sun, X., and Hou, L. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14313–14323, 2024.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Luo, Y., Lu, C., Ren, Z., Hu, J., Ye, T., Gou, Z., Ma, S., and Zhang, X. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025.

Shi, B., Tjandra, A., Hoffman, J., Wang, H., Wu, Y.-C., Gao, L., Richter, J., Le, M., Vyas, A., Chen, S., Feichtenhofer, C., Doll´ar, P., Hsu, W.-N., and Lee, A. Sam audio: Segment anything in audio. 2025. URL https://arxiv.org/abs/2512.18099.

Sun, G., Yu, W., Tang, C., Chen, X., Tan, T., Li, W., Lu, L., Ma, Z., Wang, Y., and Zhang, C. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024.

Sun, G., Yang, Y., Zhuang, J., Tang, C., Li, Y., Li, W., Ma, Z., and Zhang, C. video-salmonn-o1: Reasoningenhanced audio-visual large language model. arXiv preprint arXiv:2502.11775, 2025.

Tang, C., Li, Y., Yang, Y., Zhuang, J., Sun, G., Li, W., Ma, Z., and Zhang, C. video-SALMONN 2: CaptioningEnhanced Audio-Visual Large Language Models. arXiv preprint arXiv:2506.15220, 2025.

Wang, J., Yuan, L., Zhang, Y., and Sun, H. Tarsier: Recipes for training and evaluating large video description models, 2024a. URL https://arxiv.org/abs/2407. 00634.

Wang, T., Zhang, R., Lu, Z., Zheng, F., Cheng, R., and Luo, P. End-to-end dense video captioning with parallel

decoding. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6847–6857, 2021.

Wang, Y., Ren, S., Gao, R., Yao, L., Guo, Q., An, K., Bai, J., and Sun, X. Ladic: Are diffusion models really inferior to autoregressive counterparts for image-to-text generation? arXiv preprint arXiv:2404.10763, 2024b.

Wang, Y., Cai, Y., Ren, S., Yang, S., Yao, L., Liu, Y., Zhang, Y., Wan, P., and Sun, X. Rico: Improving accuracy and completeness in image recaptioning via visual reconstruction. arXiv preprint arXiv:2505.22613, 2025a.

Wang, Y., Wang, Z., Xu, B., Du, Y., Lin, K., Xiao, Z., Yue, Z., Ju, J., Zhang, L., Yang, D., Fang, X., He, Z., Luo, Z., Wang, W., Lin, J., Luan, J., and Jin, Q. Time-r1: Posttraining large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025b.

Wu, P., Liu, Y., Zhu, Z., Zhou, E., and Shen, S. Ugcvideocaptioner: An omni ugc video detail caption model and new benchmarks. arXiv preprint arXiv:2507.11336, 2025.

Xing, L., Dong, X., Zang, Y., Cao, Y., Liang, J., Huang, Q., Wang, J., Wu, F., and Lin, D. Caprl: Stimulating dense image caption capabilities via reinforcement learning. arXiv preprint arXiv:2509.22647, 2025.

Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., et al. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025a.

Xu, J., Guo, Z., Hu, H., Chu, Y., Wang, X., He, J., Wang,

- Y., Shi, X., He, T., Zhu, X., Lv, Y., Wang, Y., Guo, D., Wang, H., Ma, L., Zhang, P., Zhang, X., Hao, H., Guo,
- Z., Yang, B., Zhang, B., Ma, Z., Wei, X., Bai, S., Chen, K., Liu, X., Wang, P., Yang, M., Liu, D., Ren, X., Zheng,

- B., Men, R., Zhou, F., Yu, B., Yang, J., Yu, L., Zhou, J., and Lin, J. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025b.

Yang, A., Nagrani, A., Seo, P. H., Miech, A., Pont-Tuset, J., Laptev, I., Sivic, J., and Schmid, C. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10714–10726, 2023. URL https://api.semanticscholar.

org/CorpusID:257232853.

Yang, Q., Yao, S., Chen, W., Fu, S., Bai, D., Zhao, J., Sun, B., Yin, B., Wei, X., and Zhou, J. Humanomniv2: From understanding to omni-modal reasoning with context. arXiv preprint arXiv:2506.21277, 2025a.

Yang, Z., Yu, Y., Zhao, Y., Lu, S., and Bai, S. Timeexpert: An expert-guided video llm for video temporal grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24286–24296, 2025b.

Yao, L., Wang, W., and Jin, Q. Image difference captioning with pre-training and contrastive learning. 36(3):3108– 3116, 2022.

Yao, L., Chen, W., and Jin, Q. Capenrich: Enriching caption semantics for web images via cross-modal pre-trained knowledge. pp. 2392–2401, 2023.

Yao, L., Li, L., Ren, S., Wang, L., Liu, Y., Sun, X., and Hou, L. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985, 2024a.

Yao, L., Zhang, Y., Wang, Z., Hou, X., Ge, T., Jiang, Y., Sun, X., and Jin, Q. Edit as you wish: Video caption editing with multi-grained user control. pp. 1924–1933, 2024b.

Yao, L., Li, Y., Wei, Y., Li, L., Ren, S., Liu, Y., Ouyang, K., Wang, L., Li, S., Li, S., et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 10807–10816, 2025a.

Yao, L., Wu, H., Ouyang, K., Zhang, Y., Xiong, C., Chen, B., Sun, X., and Li, J. Generative frame sampler for long video understanding. arXiv preprint arXiv:2503.09146, 2025b.

Yao, L., Xing, L., Shi, Y., et al. Towards efficient multimodal large language models: A survey on token compression. TechRxiv, jan 2026. doi: 10.36227/techrxiv.176823010. 07236701/v1.

Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024c.

Ye, H., Yang, C.-H. H., Goel, A., Huang, W., Zhu, L., Su, Y., Lin, S., Cheng, A.-C., Wan, Z., Tian, J., et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. arXiv preprint arXiv:2510.15870, 2025.

Yuan, L., Wang, J., Sun, H., Zhang, Y., and Lin, Y. Tarsier2: Advancing large vision-language models from detailed video description to comprehensive video understanding. arXiv preprint arXiv:2501.07888, 2025a.

Yuan, L., Wang, J., Sun, H., Zhang, Y., and Lin, Y. Tarsier2: Advancing large vision-language models from detailed video description to comprehensive video understanding, 2025b. URL https://arxiv.org/abs/ 2501.07888.

Yue, Z., Zhang, Y., Wang, Z., and Jin, Q. Movie101v2: Improved movie narration benchmark. In Proceedings of

the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17081– 17095, 2025.

Zellers, R., Lu, J., Lu, X., Yu, Y., Zhao, Y., Salehi, M., Kusupati, A., Hessel, J., Farhadi, A., and Choi, Y. Merlot reserve: Neural script knowledge through vision and language and sound, 2022. URL https://arxiv.org/ abs/2201.02639.

Zeng, X., Li, K., Wang, C., Li, X., Jiang, T., Yan, Z., Li, S., Shi, Y., Yue, Z., Wang, Y., et al. Timesuite: Improving mllms for long video understanding via grounded tuning. arXiv preprint arXiv:2410.19702, 2024.

Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data, 2024. URL https://arxiv.org/abs/2410.02713.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zhong, C., Hou, Q., Zhou, Z., Hao, S., Lu, H., Zhang, Y., Tang, H., and Bai, X. Owlcap: Harmonizing motiondetail for video captioning via hmd-270k and caption set equivalence reward. arXiv preprint arXiv:2508.18634, 2025.

Zhou, Z., Wang, R., and Wu, Z. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862, 2025.

### A. Limitations and Future Work.

Our current work has several limitations that warrant future investigation. First, the 32K context-window constraint poses a significant challenge during training. Since the OmniDenseCaptioning task involves both lengthy inputs (video frames sampled at 2 FPS) and outputs (captions averaging 1K words), extending the context window is essential to accommodate more video frames and generate comprehensive captions. Second, our model demonstrates limited generalization to videos of varying durations, particularly hour-long content. To address this, we currently adopt a segment-then-caption strategy: dividing long videos into shorter clips (approximately one minute each) and applying TimeChat-Captioner sequentially to generate fine-grained omni-captions for each segment.

In future work, we plan to address these limitations through two directions: (1) collecting more diverse long-form videos to improve duration generalization and timestamp segmentation accuracy, and (2) incorporating efficient techniques such as token compression (Yao et al., 2026; 2024a) to reduce the sequence length of audio-video-text inputs, thereby lowering training costs, especially during the GRPO stage.

### B. Additional Experimental Results

Effect of Reward Weights. We investigate the sensitivity of our model to the reward weight coefficients (αf,αl,αt,αc) in Equation 8. As shown in Table 5, varying the weight of the coherence reward RC from 1.0 to 1.5 results in marginal performance differences across all metrics (less than 0.5% on F1, mIoU, and SodaM). This suggests that the four reward components provide complementary supervision signals without requiring extensive hyperparameter tuning.

Table 5. Ablation study on reward weight coefficients.

Table 6. Ablation study on SFT training epochs.

(αf,αl,αt,αc) F1 mIoU SodaM

(0.5, 0.5, 1.0, 1.0) 61.2 69.6 35.0 (0.5, 0.5, 1.0, 1.5) 61.0 69.4 34.6

SFT Training F1 mIoU SodaM

- Epoch 1 61.7 70.4 30.7
- Epoch 2 62.4 70.7 32.6

2500

5000

| | |
|---|---|
| |Avg. word num|
| | |
| |Total 877|
| | |

|[Figure 77]<br><br>< 50s<br><br>[Figure 78]<br><br>16.45%<br><br>[Figure 79]<br><br>50-55s<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>31.26%<br><br>[Figure 83]<br><br>55-60s<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>42.61%<br><br>[Figure 87]<br><br>> 60s<br><br>[Figure 88]<br><br>9.67%<br><br>Frequency|
|---|

Avg. duration Segment 10.04

4500

2000

4000

Events 306 Camera 142 Acoustics content 127 Visual background 120

3500

Frequency

Frequency

1500

3000

2500

Dialog content 93 ShotEdit style 88

1000

2000

1500

500

1000

500

0

0

82 832 1582 3082

2332

2 9 16 30

23

(c) Segment Duration

(a) Video Duration (b) Caption Length

Figure 5. Statistics of the training dataset TimeChatCap-42K. (a) Video duration distribution; most videos (73.9%) fall within 50-60 seconds. (b) Caption length distribution with per-dimension average word counts; annotations average 877 words per video across six dimensions. (c) Segment duration distribution; the average segment length is 10.04 seconds.

Effect of SFT Training Epochs. We examine the impact of supervised fine-tuning (SFT) duration on model performance. As shown in Table 6, extending SFT training from 1 to 2 epochs yields consistent improvements across all evaluation metrics. These results suggest that the OmniDenseCaptioning task is inherently complex, requiring sufficient SFT training for the model to adequately learn the structured output format and multi-dimensional annotation capabilities. Moreover, a well-trained SFT checkpoint serves as a stronger initialization for the subsequent GRPO stage, enabling more effective reward-guided optimization. We therefore adopt two-epoch SFT training as our default configuration.

### C. Details for Training Data and Benchmark Annotation

#### C.1. Training Data Construction

As illustrated in Figure 6, we design a three-stage pipeline to synthesize high-quality training samples for the OmniDenseCaptioning task. Detailed prompts are shown in Table 7 and Table 8.

###### Two-Step Detailed Caption Generation

###### Quality Check

###### Video Source Sampling

boundary segmentation

detailed caption generation

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

gemini

###### gemini

movie101

[{

(duration:3min)

Format Validation

Filter Short Duration

...

“timestamp”:0:00-0:15, “detailed_caption”:the boy which hold the .. “camera_state”: slightly low-angled, focusing on... “video_background”: a indoor scence...

0:00-0:15: a boy sing.. 0:16-0:22: he continue...

[Figure 104]

[Figure 105]

[Figure 106]

cut to 3min

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

###### ... ... 2:47-3:00:he speaked

[Figure 113]

[Figure 114]

[Figure 115]

###### “storyline”:.....

mmtrail

“speech_content”:(boy) ‘tell me how to get your love’ }.....]

that...

(duration:3min)

Audio Check

Filter One-Segment Videos

Figure 6. Overview of the synthetic training data construction pipeline for the training dataset TimeChatCap-42K.

Figure 5 presents the detailed statistics of TimeChatCap-42K: (a) Video duration distribution, where the majority of videos (73.9%) fall within the 50-60 second range. (b) Caption length distribution across dimensions, with annotations averaging 877 words per video spanning six dimensions. (c) Segment duration distribution, showing an average segment length of 10.04 seconds.

#### C.2. Human Annotation Details for OmniDCBench

The OmniDCBench is entirely annotated by human experts. We recruited approximately 15 annotators through crowdsourcing platforms, and the complete annotation process spanned approximately one month. As shown in Figure 7, the annotation interface is designed to be intuitive and user-friendly, featuring clear instructions and real-time feedback mechanisms to facilitate efficient task completion. To ensure annotation quality, each sample was reviewed by at least one additional annotator.

### D. Implementation Details

Our training procedure consists of two stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning via Group Relative Policy Optimization (GRPO). During the SFT stage, the model is fine-tuned for 2 epochs on 40K training samples, with a learning rate of 5e-5 and a global batch size of 128. In the subsequent GRPO phase, we utilize 2K training samples and employ a rollout size of 8 to compute group relative advantages. The learning rate, batch size, and number of epochs for RL alignment are set to 1e-5, 64, and 1, respectively. The KL penalty coefficient β is set to 0.04. The weights for format, length, temporal alignment, and caption quality rewards are configured as 0.5, 0.5, 1.0, and 1.0. To facilitate long-video understanding, we set the maximum sequence length to 32K tokens. All videos are uniformly sampled at 2 Frames Per Second (FPS). We limit the maximum pixels per frame to 297,920 and the total pixels per video to 20,070,400, ensuring a balance between visual fidelity and computational efficiency. All experiments are conducted on 32x80G GPUs using DeepSpeed ZeRO-2.

### E. Additional Qualitative Analysis

We present qualitative comparisons among TimeChat-Captioner, Gemini-2.5-Pro, and Qwen-3-Omni on a representative sample from OmniDCBench, as illustrated in Figure 4.

TimeChat-Captioner achieves fine-grained alignment with the ground truth across all six annotation dimensions, as shown in the following:

- • Detailed Events: Our model accurately identifies characters by their names (“Xia Luo”) and provides detailed appearance descriptions (e.g., “a man with a distinctive, spiky haircut and a black suit”). The phrase “continues to drive in circles” demonstrates temporal awareness and scene continuity from preceding segments. Fine-grained actions such as “leans forward to peer out the window, searching for something” are faithfully captured.
- • Visual Background: The model correctly recognizes the vehicle type (“luxury Maserati”) and simultaneously describes both interior and exterior environments (“the opulent courtyard seen through the windows”), maintaining spatial consistency.
- • Acoustics Content: The model captures nuanced tonal transitions in speech (“Anxious, then commanding”) and identifies

[Figure 116]

Figure 7. Interface page used for manual annotation during the construction of OmniDCBench.

ambient sounds (“the hum of the car’s engine”).

- • Dialogue Content: Speaker attribution is precise, with each utterance correctly assigned to the corresponding character (“Xia Luo: ...”), and the conversational content aligns with the visual narrative.
- • Camera State: Camera positioning (“inside the car, looking at the driver and passenger”) and shot transitions (“cuts to a close-up of the man in the back seat”) are accurately described.
- • Shot Editing Style: The model provides purposeful analysis of editing choices, noting that “interior shots focus on the characters’ interactions and motivations.”

In contrast, Gemini-2.5-Pro (Gemini Team, 2024) exhibits severe hallucination by misidentifying the male driver as “a woman,” fundamentally distorting the scene semantics. Qwen-3-Omni (Xu et al., 2025b) misses the main event entirelyinstead of describing the conversation inside the car, it focuses on irrelevant background elements such as “a man dressed in a red ceremonial uniform” standing outside the building. These comparisons highlight TimeChat-Captioner’s superior capability in accurate character recognition, consistent identity tracking across time, faithful event grounding, and comprehensive multi-dimensional annotation.

- F. Prompt Templates We provide the detailed prompt templates used in our training data construction pipeline and evaluation framework.

###### Stage-1 Prompt for Dense Timestamp Generation:

You are given a video clip of around 3 minutes. Your task is to generate dense captions for the video. The goal is to segment the video into multiple meaningful intervals and provide detailed descriptions for each segment.

###### 1. Segmentation Logic:

- • Split the video into natural segments according to scene changes, shot transitions, events, character actions, or core content shifts.
- • Each minute usually contains 4–5 segments, but prioritize the video’s logic over strict numbers. 2. Caption Requirements:
- • For each segment, provide a time range in the format: start time(0:00) - end time(0:05): caption

- • The caption should be concise but descriptive, summarizing what happens in that segment.
- • Include characters, actions, objects, emotions, and scene details where relevant.
- • Avoid redundancy, but ensure that important visual and narrative information is captured.

###### 3. Output Format:

- • A clean list of captions, each starting with the time range followed by the description.
- • Timestamp format: minutes:seconds (e.g., 0:00 - 0:11).
- • Segment boundaries should be clear and non-overlapping—the end time of one segment and start time of the next should be at least 1 second apart.
- • Example: if one segment ends at 0:11, the next one should begin at 0:12 or later. Output Example:

**0:00 - 0:07** A man walks into the dimly lit room and looks around cautiously.

**0:08 - 0:15** He notices a woman sitting by the window, staring outside in silence.

**0:16 - 0:25** The camera cuts to a close-up of his nervous expression.

- Table 7. The annotation prompt used in Stage-1 of training data construction. This prompt instructs Gemini-2.5-pro to segment videos into meaningful intervals with concise captions.

Stage-2 Prompt for Multi-Dimensional Structural Caption Generation: You are given:

- • A video (with both visual and audio content).
- • A set of ground-truth captions (GT captions) with timestamps. Your task is to use the GT captions as rough references for segmentation, but generate richer and more detailed descriptions directly from the video itself.

- 1. Segmentation:

- • Follow the timestamps provided in the GT captions. Each GT caption defines the rough boundaries of a segment.
- • Timestamp format: minutes:seconds (e.g., 0:00 - 0:11).
- • Segment boundaries should be clear and non-overlapping—at least 1 second apart between consecutive segments.

- 2. Generation Logic: Do not simply extract or paraphrase the GT caption. Use the video itself to expand with details:

- • Characters: actions, gestures, facial expressions, emotions.
- • Objects & Setting: relevant items, props, environment.
- • Camera: framing, movement, zoom, transitions.
- • Storyline: how the segment advances or changes the plot.
- • Speech: actual dialogue attributed to speakers.
- • Acoustics: speech tone, background music, sound effects.
- • Shooting Style: special techniques (montage, flashback, dissolve, long take, etc.).

- 3. Output Format (JSON Schema):

{

"timestamp": "start_time - end_time", "segment_detail_caption": "Detailed description of what happens

(gestures, expressions, setting details, etc.).", "camera_state": "Camera angle, framing, zoom, and movement.", "video_background": "Setting, environment, or background elements.", "storyline": "How this segment fits into the larger narrative.", "shooting_style": "Long take, montage, flashback, intercut,

or special transition effects.", "speech_content": "Full character dialogues with speaker attribution.", "acoustics_content": "1) Tone of speech. 2) Background sounds or music."

}

- Table 8. The annotation prompt used in Stage-2 of training data construction. Given Stage-1 captions as segmentation references, we prompt Gemini-2.5-pro to generate more enriched multi-dimensional annotations by directly perceiving the video content, covering detailed events, camera state, background, storyline, shooting style, speech, and acoustics.

###### Judge Prompt for SodaM Checklist Evaluation:

You are a strict evaluator for fine-grained audio-enhanced video captions. You will receive:

- 1. A list of ground-truth keypoints already organized in 6 dimensions.
- 2. One model-generated caption to evaluate. The ground-truth keypoints are already atomic and accurate. You only need to check whether each keypoint is explicitly mentioned or clearly implied in the model’s caption. Rules:

- • Mark a keypoint as correct if its meaning appears in the model’s caption with the same or equivalent semantics.
- • Ignore differences in phrasing, tense, or minor wording.
- • Do NOT infer or guess beyond the caption content.
- • Do NOT generate new keypoints or summaries.
- • Do NOT output any text other than the required JSON. Output Format (Strict JSON Only):

{

"by_dim": { "segment_detail_caption": {

"correct_keypoints": [<string>, ...], "correct_count": <int>}, "video_background": {

"correct_keypoints": [<string>, ...], "correct_count": <int>}, "acoustics_content": {

"correct_keypoints": [<string>, ...], "correct_count": <int>}, "shooting_style": {

"correct_keypoints": [<string>, ...], "correct_count": <int>}, "speech_content": {

"correct_keypoints": [<string>, ...], "correct_count": <int>}, "camera_state": {

"correct_keypoints": [<string>, ...], "correct_count": <int>} }

} Input Template: Ground-truth keypoints (by dimension):

- - segment_detail_caption: [<keypoint_1>, <keypoint_2>, ...]
- - video_background: [<keypoint_1>, <keypoint_2>, ...]
- - acoustics_content: [<keypoint_1>, <keypoint_2>, ...]
- - shooting_style: [<keypoint_1>, <keypoint_2>, ...]
- - speech_content: [<keypoint_1>, <keypoint_2>, ...]
- - camera_state: [<keypoint_1>, <keypoint_2>, ...]

Model-generated caption to evaluate: <model_caption>

- Table 9. The judge prompt used for SodaM using checklist score during evaluation. Given ground-truth keypoints decomposed into six dimensions and a model-generated caption, the judge model (Gemini-2-Flash) verifies whether each atomic keypoint is explicitly mentioned or semantically implied in the prediction, enabling fine-grained recall computation across all annotation dimensions.

