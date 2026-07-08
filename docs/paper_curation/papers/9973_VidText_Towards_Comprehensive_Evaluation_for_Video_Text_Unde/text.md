# arXiv:2505.22810v2[cs.CV]3Nov2025

## VidText: Towards Comprehensive Evaluation for Video Text Understanding

###### Zhoufaran Yang2,* Yan Shu1,* Jing Wang3 Zhifei Yang4,† Yan Zhang5,6 Yu Li2 Keyang Lu7 Gangyan Zeng8 Shaohui Liu2 Yu Zhou9 Nicu Sebe1 1UNITN 2HIT 3SEU 4PKU 5IIE, CAS 6UCAS 7BUAA 8NJUST 9NKU

∗Equal contribution. †Corresponding author.

https://github.com/shuyansy/VidText

#### Abstract

Visual texts embedded in videos carry rich semantic information, which is crucial for both holistic video understanding and fine-grained reasoning about local human actions. However, existing video understanding benchmarks largely overlook textual information, while OCR-specific benchmarks are constrained to static images, limiting their ability to capture the interaction between text and dynamic visual contexts. To address this gap, we propose VidText, a new benchmark designed for comprehensive and in-depth evaluation of video text understanding. VidText offers the following key features: 1) It covers a wide range of real-world scenarios and supports multilingual content, encompassing diverse settings where video text naturally appears. 2) It introduces a hierarchical evaluation framework with video-level, clip-level, and instance-level tasks, enabling assessment of both global summarization and local retrieval capabilities. 3) The benchmark also introduces a set of paired perception reasoning tasks, ranging from visual text perception to cross-modal reasoning between textual and visual information. Extensive experiments on 18 state-of-the-art Large Multimodal Models (LMMs) reveal that current models struggle across most tasks, with significant room for improvement. Further analysis highlights the impact of both model-intrinsic factors, such as input resolution and OCR capability, and external factors, including the use of auxiliary information and Chain-of-Thought reasoning strategies. We hope VidText will fill the current gap in video understanding benchmarks and serve as a foundation for future research on multimodal reasoning with video text in dynamic environments.

#### 1 Introduction

Large Multimodal Models (LMMs) [1–5] are rapidly emerging as general-purpose solutions for a wide range of vision-language tasks, demonstrating impressive perception and cognitive capabilities across various multimodal benchmarks. Building on this success, there is a growing interest in extending LMMs to video understanding [6–11], including video captioning, question-answering and retrieval [12]. To support this development, a number of video benchmarks [13–17] have recently been proposed to enable more comprehensive evaluations of LMMs in dynamic visual environments.

However, existing video understanding evaluations primarily focus on major events, character actions, and interpersonal relationships, while largely overlooking video text. As a self-descriptive visual component, text in videos plays a crucial role in visual understanding [18–20]. On one hand, it provides explicit perceptual cues, such as street signs, storefronts, or subtitles, that help identify key elements and clarify the scene. On the other hand, text also enables contextual reasoning, re-

Preprint. Under review.

- Table 1: Comparison of video understanding benchmarks. “Vid”, “Cli” and “Ins” denote videolevel, clip-level and instance-level tasks. “T”, “S” and “C" mean temporal, spatial and causal dimensions. “MC” and “OE” represent multiple-choice and open-ended questions. “TQ”: the percentage of questions containing visual text instances.

Perception Reasoning Paired Tasks

Multi Lingual

Multi Source

Multi Granularity

Benchmark Video QA TQ%

TaskType T S T S C

General Video Understanding Datasets NExT-QA [23] 5,440 52,044 – ✗ ✓ Vid+Cli ✗ ✗ ✓ ✗ ✓ ✗ MC+OE MVBench [15] 4,000 4,000 – ✗ ✓ Vid+Cli ✓ ✓ ✓ ✗ ✓ ✗ MC MovieChat-1K [9] 1,000 13,000 – ✗ ✓ Vid+Cli ✗ ✗ ✓ ✗ ✗ ✗ MC+OE Video-MME [13] 900 2,700 – ✓ ✓ Vid+Cli+Ins ✓ ✗ ✓ ✗ ✗ ✗ MC MLVU [14] 1,730 3,102 – ✓ ✓ Vid+Cli+Ins ✗ ✗ ✗ ✗ ✗ ✗ MC+OE Video Text Datasets BovText [24] 2,000 – – ✓ ✓ Ins ✓ ✓ ✗ ✗ ✗ ✗ OE RoadText1k [25] 1000 – – ✗ ✗ Ins ✓ ✓ ✗ ✗ ✗ ✗ OE M4ViteVQA [19] 680 2,103 40 ✗ ✓ Cli+Ins ✗ ✗ ✓ ✓ ✗ ✗ MC+OE RoadTextVQA [26] 329 1,052 60 ✗ ✗ Cli+Ins ✗ ✗ ✓ ✓ ✗ ✗ MC EgoTextVQA [27] 1,507 7,064 52 ✓ ✗ Cli+Ins ✗ ✗ ✓ ✓ ✗ ✗ OE Ours 939 2,857 65 ✓ ✓ Vid+Cli+Ins ✓ ✓ ✓ ✓ ✓ ✓ MC+OE

vealing underlying motivations or causal relationships. For example, a “SALE” sign in a shop may explain why people are gathering, which is not readily apparent from visual cues alone.

Compared to images, perceiving dynamic video text and modeling its interaction with evolving visual contexts in videos is significantly more challenging. It requires not only fine-grained localization at the instance level, but also temporal tracking and spotting at the clip level, as well as holistic understanding at the video level. Furthermore, video text appears in a wide range of scenarios and across multiple languages, which further increases the complexity of recognition and reasoning. Based on these insights, we propose VidText, a comprehensive benchmark for video text understanding, which introduces the following key advantages:

- • It encompasses a wide variety of video genres, including media, entertainment, ego-centric, sports, life record and knowledge, with 27 fine-grained categories covering diverse scenarios rich in visual text, such as scene text and subtitles. Moreover, it includes multilingual content, covering English, Chinese, Korean, Japanese, and German.
- • It supports multi-granularity evaluation, including video-level, clip-level, and instance-level tasks. Video-level tasks involve holistic OCR understanding and reasoning over global video content. Clip-level tasks are designed to require localized comprehension based on specific temporal segments. Instance-level tasks demand fine-grained temporal and spatial grounding of individual text instances to support precise question answering.
- • It spans from visual text perception to cross-modal reasoning with visual context. Building upon the meticulously annotated video text data, we produce video text-centric Chain-of-Thought (CoT) annotations, explicitly capturing the reasoning process between video descriptions and embedded texts, including spatial relationships with surrounding objects and temporal dependencies related to actions or events. In this way, we extend video text perception tasks into their corresponding reasoning counterparts, forming a comprehensive paired perception–reasoning framework that spans eight tasks covering multiple levels of understanding.

Tab. 1 shows that VidText enables a more comprehensive evaluation of video text understanding compared to both general video understanding benchmarks and video text-specific benchmarks. We conduct extensive evaluations on 18 popular LMMs, revealing several important insights. First, video text understanding remains a technically challenging task for existing models. Although Gemini 1.5 Pro [21] achieves the highest performance, it only reaches an average score of 46.8%, and all models perform poorly on multi-granularity tasks, which is far below estimated human-level performance. Second, several concurrent open-source models [6, 22] demonstrate competitive performance, narrowing the gap with proprietary systems. Third, our empirical findings highlight several influential factors in video text understanding, including the model’s image OCR capability, input resolution, the integration of auxiliary information, and the role of Chain-of-Thought reasoning strategies. Therefore, we believe VidText serves as a valuable complement to existing general video understanding benchmarks, while also providing new insights for the OCR and multimodal reasoning communities.

#### 2 Related Work

###### 2.1 Video Large Language Models

With the rapid advancement of large language models (LLMs), a series of video large language models (Video LLMs) have emerged [3, 28–30], leveraging LLMs as backbones to enhance video reasoning capabilities. Early Video LLMs primarily relied on sparsely sampled frames and temporal modeling mechanisms [5, 31], such as Q-Former and temporal pooling, to facilitate video captioning and question answering. Building upon these designs, subsequent models [1, 2, 6, 32– 37] have focused on addressing key challenges in video understanding, including fine-grained semantic alignment, temporal representation, and long-duration video comprehension. For instance, Qwen-VL 2.5 [1] introduces dynamic resolution processing and absolute temporal encoding to handle variable-resolution videos. Video-LLaMA3 [6] applies a frame compression strategy based on frame similarity to reduce the number of visual tokens, resulting in more compact and precise video representations. To handle extremely long videos, LongVA [34] extends the context length of the LLM backbone and transfers its long-context capability to the video domain. Video-XL [35] leverages the inherent key-value sparsification mechanism of LLMs to efficiently condense visual inputs. VideoChatFlash [32] proposes a hierarchical compression strategy, reducing token redundancy in both the video and language modules.

###### 2.2 Video Understanding Benchmarks

With the growing interest in video LLMs, the development of dedicated benchmarks has become increasingly emphasized. Existing benchmarks have been designed for a variety of video understanding tasks, including action reasoning, spatio-temporal inference, video captioning, and longvideo comprehension [13–16, 23, 38–40]. For example, NeXT-QA [23] evaluates temporal reasoning abilities by testing models on the relationships between human actions. VideoChatGPT-Bench [40] focuses on open-ended video conversation, constructing captioning and dialogue tasks to assess generative and interactive capabilities. TempCompass [39] introduces fine-grained temporal perturbations to assess whether models can answer questions based on temporal changes within the video. To support comprehensive video question answering, MVBench [15] proposes a large-scale benchmark covering 20 distinct subtasks, spanning multiple perception and reasoning dimensions. For long-video understanding, VideoMME [13], MLVU [14], LVBench [41] and LongVideoBench [16] curate diverse and extended-duration videos to evaluate multi-level abilities across extended temporal contexts.

As text carries rich and structured information in videos, several benchmarks have been proposed to evaluate video text understanding [19, 24, 25, 27, 42], including tasks such as text tracking, spotting, and reasoning. Specifically, RoadTextVQA [25] focuses on autonomous driving scenarios, while EgoTextVQA [27] targets egocentric perspectives in daily life settings. In addition, M4-ViteVQA [19] collects videos from nine diverse real-world scenarios, such as shopping, traveling, and movies, to evaluate the generalization capabilities of video-language models. However, these benchmarks exhibit two notable limitations. First, their task types are relatively simple, and therefore insufficient for comprehensively evaluating the diverse capabilities of modern video LLMs. Second, their video categories and language coverage remain limited, often constrained to specific application domains.

#### 3 Dataset Construction

In this section, we describe the dataset construction process for VidText. We begin by illustrating how the source videos are collected (Sec. 3.1), followed by a detailed explanation of the annotation pipeline (Sec. 3.2). Finally, we describe the task taxonomy of our benchmark (Sec. 3.3).

###### 3.1 Video Collection

In VidText, we aim to evaluate video text understanding across diverse scenarios, including both video category variety and language diversity. While several existing datasets [19, 24, 25, 42] provide detailed text annotations, they all suffer from several key limitations: (1) Limited scenario diversity: Most datasets focus on specific domains such as indoor scenes or egocentric videos, lacking coverage of richer, more interactive contexts such as sports events, livestreaming games, or daily

[  ，   ）

NumberofTextInstances

[   ，   ）

[   ，   ）

[   ，   ）

[Figure 1]

[   ，    ）

[    ，    ）

[    ，）

Video Number

Reasoning

Perception

HolisticOCR

HolisticReasoning

LocalOCR

LocalReasoning

TemporalCausal Reasoning

TextLocalization

SpatialReasoning

TextTracking

video-level clip-level instance-level

- Figure 1: Statistical overview of our VidText. (Left) Video genres included in VidText. (Top Right) Visual Text Instance Distribution. (Bottom Right) Hierarchical Task type settings.

vlogs. (2) Lack of language diversity: Nearly all existing datasets contain only English, failing to reflect the multilingual nature of real-world video text. (3) Short video duration: Many videos are only 10–15 seconds long, which limits their suitability for tasks involving cross-temporal reasoning or holistic understanding. Therefore, in addition to incorporating existing datasets, we further collect video data from comprehensive long-form video benchmarks [13, 14] and public platforms such as YouTube, in order to enhance the scenario diversity, temporal richness, and linguistic coverage of VidText.

For the manually collected videos, we leverage expert models to construct an effective selection pipeline. First, we ensure the presence of visual text in each video by using Gomatching [43], a video text detection tool, to assess text density. Second, we filter out low-quality videos containing blur, watermarks, or low resolution, using existing video quality assessment models [44, 45]. Third, we enforce a minimum duration threshold of 3 minutes to guarantee sufficient temporal content. As a result, we collect a total of 939 high-quality videos, each annotated with one of 27 predefined scene categories. Additionally, we record metadata for each video, including language type, resolution, frame rate, and text density. Fig. 1 presents basic statistics of VidText. More detailed statistics across multiple dimensions are provided in the Supplementary Materials.

###### 3.2 Annotations Generation

To support evaluation at both the perception and reasoning levels, VidText provides meticulously constructed annotations tailored to the requirements of each task.

Perception. For each qualified video, we adopt a bottom-up strategy to construct multi-granularity annotations, including instance-level, clip-level, and video-level information. First, annotators are instructed to track at least three clear visual text instances throughout the video. For each instance, we conduct frame-by-frame fine-grained annotation until it disappears, generating a sequence of annotations that include bounding boxes, transcriptions, and unique track IDs. Second, the video is segmented into multiple intervals based on its duration (i.e., longer videos are divided into more segments). For each segment, we check the presence of visual text using instance-level annotations and record clip-level labels, including the temporal span (start and end timestamps) and associated transcriptions. Third, a separate group of annotators performs video-level annotations, which involve recording all distinct transcriptions that appear across the entire video. Specifically, for Chinese, we use text lines as the basic annotation unit, while for other languages, annotations are performed at the word level.

Reasoning. Since the multi-granularity annotations constructed for perception address the question of “what texts appear in the video or clip”, we further investigate “what actions or events are linked to these texts”. To this end, we design a video text-centric Chain-of-Thought (CoT) annotation pipeline. First, for each video or clip (as defined by the time span annotations), we apply an adap-

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]|
|---|

[Figure 8]

EXITS I O E - W

L o n gI sE x p w y

EasternLongIs Manhnttan

RIGHTLANE

Timestamp:30.1

HolisticOCR HolisticReasoning

Q:Please provide all text in video. Q: Based on the video, which of the following

GT:[EXITS] [IOE-W][495] [Long] [is] [Expwy] statements accurately describe the video? [Eastern] [Long] [is] [Manhattan] [RIGHT] [LANE] [SPEED] [LIMIT] [40]

GT:1. The video was recorded on Interstate 495 (Long Island Expressway) in New York State.

- 2. Vehicles in the right can take Exit 10E.
- 3. The speed limit on this stretch of road is 40 mph.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

22s 52s 62s

Timestamp:2307.6

LocalOCR LocalReasoning

Q:What is the author’s training plan between 52

Q:What texts appear between 22s and 52s in the video?

s and 62 s in the video? GT:[HAPPY] [NEW] [YEAR!!!!] [HELLO] [2024]

GT:The training plan is divided into push, pull, legs, and full-body parts.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

###### Timestamp:598.7 457.3s 462.1s

|[Figure 19]|
|---|

|[Figure 20]|
|---|

TextLocalization TemporalCausalReasoning

Q:At what time in the video does the score “105:83” appear?

Q:When the score is 105:83, what move did

Lakers No.23 make? GT:[457.3s,462.1s]

GT:He stole the ball and dunked it.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

05:59 06:15

|[Figure 26]<br><br>BLG Bin|
|---|

|[Figure 27]<br><br>T1 Oner|
|---|

90.1s 116.3s

TextTracking SpatialReasoning

Q:What are the bounding box of Hero T1Oner at the start and at the end of the match interval from 5:59 to 6:15? GT:T1 Oner, positioned to the bottom-right of BLG

Q:At 5:59 in the match, in which direction is BLG Bin relative to T1Oner, and what happens next?

GT:[0.46,0.67...], [0.58, 0.69...]

BIN, and T1 Oner attempts to help his teammate gank BLG BIN.

- Figure 2: Examples from VidText. The benchmark includes eight tasks, featuring paired perception and reasoning components designed to evaluate the video-level, clip-level, and instance-level capabilities of LMMs. Given the video input and textual prompt, models are required to solve the tasks, with ground-truth answers highlighted in green.

tive sampling strategy to extract key frames. Then, we utilize the powerful vision-language model Aria [46] to generate high-quality frame-level captions, capturing both intra-frame and inter-frame contextual information. Based on the paired OCR transcripts and the multimodal descriptions, human annotators are instructed to design QA pairs that focus on the semantic or causal relationships between visual text and surrounding visual content. To ensure the quality of reasoning QA pairs, we enforce two post-validation principles: (1) Mask the visual texts and verify whether the question can be answered using only the visual content; (2) Mask the visual frames and check whether the question can be answered using only the textual information.

- 3.3 Task Taxonomy

Based on the detailed annotations encompassing perception and reasoning, we further define 8 hierarchical tasks which are demonstrated on Fig. 2.

Holistic OCR & Holistic Reasoning. Holistic OCR requires the model to recognize all visual texts appearing throughout the entire video. Redundant entries are removed, and the remaining text instances are sorted in chronological order. We evaluate this task using the F1-score, which is calculated based on instance-level precision and recall. Holistic Reasoning assesses the model’s ability to understand the overall topic of the video by integrating recognized textual information with global semantic context. The task is formulated as a multi-label selection problem, where the model is asked to choose three correct answers from seven candidate options. Performance is measured by top-3 accuracy.

Local OCR & Local Reasoning. In contrast to holistic tasks, Local OCR and Local Reasoning focus on the model’s ability to spot and interpret visual text within user-specified video segments. Local OCR requires recognizing all visual texts that appear within a given segment and is evaluated using the F1-score based on instance-level matching. Local Reasoning assesses the model’s ability to infer local semantic meaning or intent from the text. It is formulated as a multiple-choice question, and performance is measured by answer accuracy.

Text Localization & Temporal Causal Reasoning. Similar to temporal grounding tasks, Text Localization requires the model to accurately predict the temporal interval during which a specific text appears in the video. The task is evaluated using Mean Intersection-over-Union (mIoU) based on ground-truth temporal spans. The corresponding reasoning task, Temporal Causal Reasoning, extends beyond localization to assess whether the model can infer causal relationships between identified texts and subsequent multimodal events or actions. Standard evaluation is conducted in a multiple-choice format, with accuracy as the performance metric.

Text Tracking & Spatial Reasoning. Given a target text instance, Text Tracking requires the model to predict its spatial bounding box locations at its first and last appearance within the video. Spatial Reasoning extends this task by asking the model to infer spatial relationships between the textual instance and surrounding visual elements at a specified timestamp. To enable standardized evaluation with LMMs, both tasks are formatted as multiple-choice questions.

- 4 Experiments

- 4.1 Settings

We conduct a comprehensive evaluation of 18 large multimodal models (LMMs) using our VidText benchmark, encompassing both open-source and proprietary models. For proprietary models, we evaluate the Gemini series [21] and GPT series [47, 48], using their official multi-image evaluation APIs. For open-source models, we select current state-of-the-art video LMMs with diverse architectures and LLM sizes, enabling a broad assessment of video text understanding capabilities. All evaluations are conducted in a zero-shot manner. More details about the evaluation settings are provided in the Supplementary Materials.

- 4.2 Main Results

The overall evaluation results for all investigated LMMs in the VidText are shown in Tab. 2. Individual performances are reported for each task, while average performances are provided. From the results, we derive three primary conclusions:

- Table 2: The overall performance on VidText. HoliOCR: Holistic OCR; HoliRea.: Holistic Reasoning; LocalOCR: Local OCR; LocalRea.: Local Reasoning; TextLocal.: Text Localization; TempCauRea.: Temporal Causal Reasoning; TextTrac.: Text Tracking; SpaRea.: Spatial Reasoning; Avg.: the average performance of the eight tasks. The best Accuracy / Score results are highlighted.

Method Size Avg. HoliOCR HoliRea. LocalOCR LocalRea. TextLocal. TempCauRea. TextTrac. SpaRea. Human – 89.5 92.8 96.0 94.3 95.7 81.3 88.6 80.3 87.3 proprietary LMMs

GPT-4-Turbo [47] – 29.7 22.9 28.7 36.7 36.5 15.8 39.4 24.3 33.6 Gemini 1.5 Flash [21] – 34.7 26.3 34.0 40.2 42.4 28.9 40.0 30.7 35.4 GPT-4o [48] – 40.2 29.5 38.9 46.0 43.3 45.5 42.5 36.2 39.8 Gemini 1.5 Pro [21] – 45.3 34.8 43.6 50.2 50.1 48.7 47.0 40.3 47.9

###### Open-source LMMs

LongVU [49] 3B 17.0 5.8 20.4 15.4 17.0 15.6 15.9 15.4 30.5 Qwen2.5-VL [1] 3B 21.1 11.4 23.2 28.5 17.8 18.7 15.4 18.3 35.3 Video-XL-Pro [36] 3B 22.5 10.9 22.9 30.4 15.6 18.7 27.9 20.9 32.9 LongVA [34] 7B 19.2 4.8 5.6 3.2 46.9 4.5 28.3 29.6 30.5 MiniCPM-V2.6 [50] 7B 26.5 29.2 21.2 11.4 42.9 13.3 30.3 20.5 43.2 VideoChatFlash [32] 7B 29.2 13.6 13.3 1.0 50.1 45.1 42.4 23.3 44.3 Qwen2-VL[51] 7B 30.3 27.0 34.0 37.5 23.7 11.2 42.4 24.6 42.1 Qwen2.5-VL [1] 7B 31.9 35.9 36.0 37.0 26.5 26.5 35.4 22.4 35.2 VideoLLaMA3 [6] 7B 39.9 23.5 31.5 39.2 41.2 47.3 55.6 31.1 50.0 ShareGPT4Video [52] 8B 16.4 2.5 2.6 0.8 43.5 0.0 27.3 28.0 26.1 Oryx-1.5 [31] 32B 35.4 35.3 33.9 30.8 48.5 26.7 45.2 26.0 36.4 LLava-OV[53] 72B 36.1 20.1 28.1 41.3 49.4 9.9 54.6 31.8 53.4 Qwen2.5-VL [1] 72B 38.5 40.1 49.3 35.9 28.2 28.7 52.5 31.1 42.1 InternVL2.5 [22] 78B 39.8 40.2 37.4 29.0 50.4 30.5 48.5 29.9 52.3

- 1) Gemini 1.5 Pro [21] achieves the best performance on our benchmark. It significantly outperforms other models on video-text-based perception and reasoning tasks.
- 2) Proprietary models typically perform better than open-source models. However, some opensource models deliver surprisingly strong results on specific tasks. For example, VideoLLaMA3 [6] achieves the highest performance on both Temporal Causal Reasoning and Spatial Reasoning.
- 3) Video text understanding remains an overwhelming challenge for current video LMMs. First, even the best models still fall far short of human-level performance. Second, most LMMs show limited ability in fundamental video OCR tasks, where specialized video OCR models often perform better. Third, multimodal reasoning based on visual text cues in videos is significantly more difficult than in images: all video multiple-choice reasoning tasks yield accuracies below 60%, falling far behind the performance on similar image-based tasks. (ST-VQA [54] and Text-VQA [55])

Beyond the primary conclusions on overall performance, we further analyze model behaviors across individual tasks.

- 4) Among multi-granular tasks, video-level and instance-level tasks are more challenging than clip-level tasks, across both perception and reasoning settings. We hypothesize that this is due to the limited capabilities of current LMMs in two aspects: video-level tasks require global information aggregation, while instance-level tasks demand fine-grained retrieval and grounding, both of which remain weak points for existing models.
- 5) For video-level and instance-level tasks, the performance of perception and reasoning shows a strong correlation, while the two appear relatively independent in clip-level tasks. This may be because certain clip-level perception tasks, such as text localization, require accurate temporal grounding based on fine-grained visual cues. However, the corresponding reasoning tasks, such as temporal reasoning, can often be solved using local visual clues from sparsely sampled frames, allowing models to bypass the need for precise perception outputs.
- 6) Scaling up the size of LLMs leads to more significant performance gains on reasoning tasks compared to perception tasks. This suggests that video text perception cannot be effectively improved by model scale alone, and instead requires careful architectural design, specialized training data, and other task-specific considerations.

#### 5 Ablation Studies

This section begins with an investigation into the effectiveness of our hierarchical task design, including the multi-granularity structure and the extension from perception to reasoning. We also

###### Holistic Reasoning Local Reasoning Spatial Reasoning

52.6

Base Crop

Base Crop

Base Crop

50.0

50

50

50

43.8

41.2

38.4

40

40

40

36.034.9

35.2

31.529.4

30.2

26.5

30

30

30

20

20

20

10

10

10

0

0

0

VideoLLaMA3 Qwen2.5-VL

VideoLLaMA3 Qwen2.5-VL

VideoLLaMA3 Qwen2.5-VL

Figure 3: Ablation studies on the multi-granularity design of VidText.

###### Video content masking Video text masking

50

50

40

40

30

30

HR LR SR

HR LR SR

20

20

0 10 20 30

0 10 20 30

Masking Ratio Masking Ratio

[Figure 28]

[Figure 29]

Figure 4: Ablation studies on the joint reasoning of video texts and video contents. “HR”, “LR” and “SR” denote Holistic Reasoning, Local Reasoning and Spatial Reasoning, respectively. We visualize “Video content masking” and “Video Text masking” in the right part.

explore critical factors that affect model performance in video text understanding through a series of ablation studies.

###### 5.1 Investigating the effectiveness of VidText Design

Multi-granularity design. VidText includes multi-granular tasks spanning video-level, clip-level, and instance-level. To verify that tasks at different levels require correspondingly different levels of contextual information, we conduct ablation studies using VideoLLaMA3 [6] and Qwen2.5-VL [1]. Specifically, for holistic tasks, we randomly extract 50% of the video duration as a segment and evaluate performance on Holistic Reasoning. For clip-level and instance-level tasks, we select key clips based on their original task annotations. As shown in Fig. 3, clip-level and instance-level tasks benefit significantly from segment-based evaluation, as key frames provide concentrated visual text information. In contrast, Holistic Reasoning performance declines, as the task requires global information aggregation, which is lost when only partial segments are used.

Joint video text and multimodal contexts reasoning. VidText successfully extends perceptionlevel tasks into reasoning tasks, which require the joint modeling of video texts and their multimodal contextual information. To validate this, we perform an ablation study by selectively masking either the visual text regions or the surrounding video content at varying random ratios. As shown in Fig. 4, the performance on all reasoning tasks consistently drops as the masking ratio increases, confirming that both textual and visual cues are essential for reasoning under our task design.

###### 5.2 Exploring Crucial Factors of Video Text Understanding

Model-intrinsic Factors. As shown in Tab. 3, we conduct ablation studies on several influential factors. First, we examine the impact of input resolution using two representative models, Oryx-1.5 [31] and InternVL2.5 [2], both of which support adjustable input sizes. Increasing the resolution significantly improves video text understanding performance, especially in InternVL2.5 [2], where the input images are divided into sub-patches, where higher resolution allows better preservation of text details. Second, to assess the role of OCR capability, we refer to each model’s performance on standard OCR benchmarks such as OCRBench [56]. The results show that a model’s video text un-

|Impact of input resolution<br><br>|Impact of OCR ability|Impact of LLM|
|---|---|---|
|Model resolution Avg<br><br>|Model OCRBench Avg|Model LLM Avg|
|Oryx-1.5 4482 35.4<br><br>8962 38.6↑3.2<br><br>|LLaVA-OV 621 36.1 VideoLLaMA3 828 39.9↑3.8<br><br>|LLaVA-OV<br><br>Qwen2-7B 22.1 Qwen2-72B 36.1↑14.0|
|InternVL 4482 39.8<br><br>8962 44.8↑5.0|GPT4V 645 29.7<br><br>GPT-4o 822 40.2↑10.5<br><br>|LLaVA-Next<br><br>LLaMA3-8B 15.3 Qwen2-7B 20.8 ↑5.2|

- Table 3: Detailed analysis about the impact of input resolution, image OCR ability, and LLM Backbone. LLaVA-Next: LLaVA-Next-Video [33].

Method HR LR TR SR

Qwen2.5-VL 36.0 26.5 35.4 35.2 Qwen2.5-VL + Audio 36.3 26.6 35.2 35.4 Qwen2.5-VL + Text 37.2 28.3 37.9 38.1 Qwen2.5-VL + Audio + Text 37.6 29.5 38.0 39.5

Method HR LR TR SR

Qwen2.5-VL 36.0 26.5 35.4 35.2 Qwen2.5-VL + CoT 40.5 28.7 37.2 40.9

VideoLLaMA3 31.5 41.2 55.6 50.0 VideoLLaMA3 + CoT 33.8 44.6 56.2 53.8

- Table 4: Ablations about auxiliary information (Left) and CoT strategy (Right) to video text understanding.“HR”, “LR”. “TR” and “SR” mean Holistic Reasoning, Local Reasoning, Temporal Causal Reasoning and Spatial Reasoning.

derstanding performance generally aligns with its fundamental OCR accuracy. Finally, we compare different LLM backbones and find that certain architectures (e.g., Qwen2.5) exhibit stronger performance in multilingual scenarios, often outperforming LLaMA-based variants. These observations collectively indicate that video text understanding is influenced by a combination of input fidelity, OCR capacity, and language modeling strength.

External Factors. As demonstrated in Tab. 4, we first investigate whether external auxiliary information can enhance video text understanding, particularly for reasoning tasks. In this study, we consider audio transcripts and video text (e.g., subtitles or OCR outputs), both of which can be extracted using specialized tools. We convert these modalities into textual sequences and append them to the original query as contextual subtitles. As shown in our experiments, both sources contribute positively to performance. Video text provides stronger gains in global tasks that require long-range context, while audio transcripts are more beneficial for local tasks, possibly due to their alignment with short-term actions or events. Second, we propose a video text-centric Chain-ofThought (CoT) reasoning strategy, which decomposes complex reasoning processes into structured sub-steps. Specifically, the video is uniformly segmented into multiple clips. For each clip, the model is prompted to: (1) spot all visible texts, (2) generate a detailed description of the clip, and (3) infer whether any visual texts are semantically related to the description and answer the reasoning question accordingly. This CoT-based prompting strategy yields consistent improvements across all reasoning tasks, highlighting the potential of test-time reasoning augmentation for video-language models.

#### 6 Conclusion

This paper presents VidText, a novel benchmark for evaluating video text understanding in large multimodal models (LMMs). With three key innovations, including broad scenario and multilingual coverage, a multi-granular evaluation framework, and paired perception–reasoning tasks, VidText enables comprehensive and in-depth analysis of LMM performance on video text understanding. Empirical studies reveal that current LMMs still face significant challenges in both perceiving and reasoning over video texts. Future progress in this area will require the joint optimization of multiple complex factors, including model-intrinsic aspects (such as input resolution, OCR capability, and LLM backbone) and external strategies (such as auxiliary modality integration and Chain-ofThought prompting). We hope VidText will serve as a valuable resource for advancing research in the OCR and video understanding communities.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
- [5] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 23–29 Jul 2023.
- [6] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.
- [7] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [8] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023.
- [9] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024.
- [10] Han Fang, Zhifei Yang, Yuhan Wei, Xianghao Zang, Chao Ban, Zerun Feng, Zhongjiang He, Yongxiang Li, and Hao Sun. Alignment and generation adapter for efficient video-text understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2791–2797, 2023.
- [11] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. arXiv preprint arXiv:2404.03413, 2024.
- [12] Han Fang, Zhifei Yang, Xianghao Zang, Chao Ban, Zhongjiang He, Hao Sun, and Lanxiang Zhou. Mask to reconstruct: Cooperative semantics completion for video-text retrieval. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3847–3856, 2023.
- [13] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [14] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

- [15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [16] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024.
- [17] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Cristóbal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. Hourvideo: 1-hour videolanguage understanding. Advances in Neural Information Processing Systems, 37:53168– 53197, 2024.
- [18] Yan Zhang, Gangyan Zeng, Huawen Shen, Daiqing Wu, Yu Zhou, and Can Ma. Track the answer: Extending textvqa from image to video with spatio-temporal clues. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10275–10283, 2025.
- [19] Minyi Zhao, Bingjia Li, Jie Wang, Wanqing Li, Wenjing Zhou, Lan Zhang, Shijie Xuyang, Zhihang Yu, Xinkun Yu, Guangze Li, et al. Towards video text visual question answering: Benchmark and baseline. Advances in Neural Information Processing Systems, 35:35549– 35562, 2022.
- [20] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zeroshot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.
- [21] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [22] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [23] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021.
- [24] Weijia Wu, Yuanqiang Cai, Debing Zhang, Sibo Wang, Zhuang Li, Jiahong Li, Yejun Tang, and Hong Zhou. A bilingual, openworld video text dataset and end-to-end video text spotter with transformer. arXiv preprint arXiv:2112.04888, 2021.
- [25] Sangeeth Reddy, Minesh Mathew, Lluis Gomez, Marçal Rusinol, Dimosthenis Karatzas, and CV Jawahar. Roadtext-1k: Text detection & recognition dataset for driving videos. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 11074–11080. IEEE, 2020.
- [26] George Tom, Minesh Mathew, Sergi Garcia-Bordils, Dimosthenis Karatzas, and CV Jawahar. Reading between the lanes: Text videoqa on the road. In International Conference on Document Analysis and Recognition, pages 137–154. Springer, 2023.
- [27] Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng Wang, TatSeng Chua, and Angela Yao. Egotextvqa: Towards egocentric scene-text aware video question answering. arXiv preprint arXiv:2502.07411, 2025.
- [28] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

- [30] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.
- [31] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024.
- [32] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [34] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [35] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024.
- [36] Xiangrui Liu, Yan Shu, Zheng Liu, Ao Li, Yang Tian, and Bo Zhao. Video-xl-pro: Reconstructive token compression for extremely long video understanding. arXiv preprint arXiv:2503.18478, 2025.
- [37] Huaying Yuan, Zheng Liu, Minhao Qin, Hongjin Qian, Y Shu, Zhicheng Dou, and Ji-Rong Wen. Memory-enhanced retrieval augmentation for long video understanding. arXiv preprint arXiv:2503.09149, 2025.
- [38] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023.
- [39] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024.
- [40] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.
- [41] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024.
- [42] Weijia Wu, Yuzhong Zhao, Zhuang Li, Jiahong Li, Mike Zheng Shou, Umapada Pal, Dimosthenis Karatzas, and Xiang Bai. Icdar 2023 competition on video text reading for dense and small text. In International Conference on Document Analysis and Recognition, pages 405–

419. Springer, 2023.

- [43] Haibin He, Maoyuan Ye, Jing Zhang, Juhua Liu, Bo Du, and Dacheng Tao. Gomatching: A simple baseline for video text spotting via long and short term matching. arXiv preprint arXiv:2401.07080, 2024.
- [44] Wen Wen, Mu Li, Yabin Zhang, Yiting Liao, Junlin Li, Li Zhang, and Kede Ma. Modular blind video quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2763–2772, 2024.

- [45] Yachun Mi, Yan Shu, Yu Li, Chen Hui, Puchao Zhou, and Shaohui Liu. Clif-vqa: Enhancing video quality assessment by incorporating high-level semantic information related to human feelings. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 9989–9998, 2024.
- [46] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024.
- [47] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [48] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [49] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024.
- [50] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [51] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [52] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024.
- [53] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [54] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4291–4301, 2019.
- [55] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [56] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.
- [57] Yan Shu, Weichao Zeng, Fangmin Zhao, Zeyu Chen, Zhenhang Li, Xiaomeng Yang, Yu Zhou, Paolo Rota, Xiang Bai, Lianwen Jin, et al. Visual text processing: A comprehensive review and unified evaluation. arXiv preprint arXiv:2504.21682, 2025.
- [58] Zhenhang Li, Yan Shu, Weichao Zeng, Dongbao Yang, and Yu Zhou. First creating backgrounds then rendering texts: A new paradigm for visual text blending. arXiv preprint arXiv:2410.10168, 2024.
- [59] Weichao Zeng, Yan Shu, Zhenhang Li, Dongbao Yang, and Yu Zhou. Textctrl: Diffusion-based scene text editing with prior guidance control. Advances in Neural Information Processing Systems, 37:138569–138594, 2024.

- [60] Shangbang Long, Xin He, and Cong Yao. Scene text detection and recognition: The deep learning era. International Journal of Computer Vision, 129(1):161–184, 2021.
- [61] Yingying Zhu, Cong Yao, and Xiang Bai. Scene text detection and recognition: Recent advances and future trends. Frontiers of Computer Science, 10:19–36, 2016.
- [62] Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Yuan, Kai Ding, and Lianwen Jin. Swintextspotter: Scene text spotting via better synergy between text detection and text recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4593–4603, 2022.
- [63] Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang. East: an efficient and accurate scene text detector. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 5551–5560, 2017.
- [64] Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, and Xiang Bai. Real-time scene text detection with differentiable binarization. In Proceedings of the AAAI conference on artificial intelligence, pages 11474–11481, 2020.
- [65] Yan Shu, Wei Wang, Yu Zhou, Shaohui Liu, Aoting Zhang, Dongbao Yang, and Weipinng Wang. Perceiving ambiguity and semantics without recognition: an efficient and effective ambiguous scene text detector. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1851–1862, 2023.

#### A Overview of Appendix

- • B: Limitations.
- • C: Broader Impact.
- • D: Collecting Details of VidText.
- • E: Details of Annotation.
- • F: Detailed Experimental Results.
- • G: Model Prompt.
- • H: More Visualization Results.
- • Checklist

##### B Limitation We summarize the limitations of our work as follows:

- • Limited scenario coverage: Although VidText includes 27 fine-grained video categories, it still lacks representation of long-tail or high-risk domains such as medical emergencies, industrial workflows, or disaster scenarios.
- • Imbalanced language distribution: The majority of samples are in English and Chinese, with significantly fewer examples in other languages such as German, Korean, and Japanese. This imbalance prevents a thorough evaluation of multilingual OCR and reasoning capabilities.
- • Scarcity of challenging text instances: VidText contains relatively few examples involving difficult text conditions such as severe occlusion, low resolution, motion blur, unusual fonts, or multi-line arrangements. This limits the benchmark’s ability to fully assess model robustness under real-world noise and distortion.

B.1 Discussion

These dataset and model limitations are mutually reinforcing. Dataset gaps may conceal important weaknesses in current models, while existing models’ deficiencies highlight the need for broader and more diverse benchmarks. Future efforts should focus on expanding long-tail scene and language coverage in VidText, while also improving LMM architectures with better multilingual OCR, noise robustness, and cross-modal reasoning abilities. Furthermore, we also summarize three insights as follows:

- • Weak cross-domain transfer: Most LMMs are pretrained on image-based OCR tasks and struggle to generalize to unseen video scenes, such as sports broadcasts or livestream interfaces, where text appearance and context are highly dynamic.
- • Insufficient multilingual alignment: Current models show limited ability in detecting, transcribing, and semantically linking non-English texts to the visual context, resulting in degraded performance on multilingual content.
- • Low robustness to visual noise: Models often fail when confronted with noisy, blurry, or occluded text, particularly in tasks requiring instance-level grounding. This degrades downstream reasoning performance and reflects a need for stronger visual resilience.

#### C Broader Impact

The VidText benchmark is poised to make a significant contribution to both the OCR and video understanding communities by bridging the gap between low-level text perception [57–59] and highlevel semantic reasoning [60, 61] in video contexts.

For the OCR community, VidText offers a valuable opportunity to move beyond traditional imagebased text detection and recognition [62–65]. By shifting the focus to temporal and contextual dynamics in videos, it promotes the development of algorithms that can track, ground, and interpret visual texts over time.

For the video understanding community, VidText introduces the underexplored yet semantically rich modality of scene text into the landscape of video-language research. By incorporating fine-grained text perception tasks and their paired reasoning counterparts, VidText pushes video-language models to integrate visual texts with multimodal contextual cues, fostering more explainable, interpretable, and grounded video understanding.

#### D Collecting Details of VIDTEXT

This section outlines the procedures for sourcing, filtering, and analyzing the video content in VIDTEXT.

Sources. To ensure a broad coverage of video scenarios and textual styles, VIDTEXT integrates data from six public datasets:

- • BOVText [24] — Multi-scene videos suitable for holistic OCR tasks.
- • RoadText-1K [25] — Dense road-text detection in driving scenarios.
- • DSText [42] — Subtitles from indoor instructional videos.
- • M4-ViteVQA [19] — Clip- and instance-level multimodal QA videos.
- • Video-MME/MLVU [13, 14] — Long-form videos with strong temporal reasoning demands.

YouTube Supplementation. To supplement long-form data, we collect additional videos from YouTube, focusing on the following categories:

- • Sports highlights: NBA, FIFA World Cup, and related competitions.
- • Gaming commentary: live streams and post-game analysis.
- • TV shows and variety entertainment.

Retrieval and Filtering Criteria. Candidate videos were retrieved using targeted keyword queries such as "match subtitles", "game commentary", and "captioned recap". We applied the following filtering rules:

- • Minimum duration: ≥3 minutes for YouTube, >30 minutes for Video-MME.
- • Scene-text richness: We use the latest detector Gomatching [43] to calculate the proportion of frames containing text.
- • Density thresholds: Videos must meet a minimum ratio of text-bearing frames: 20% for YouTube videos and 10% for Video-MME.

Metadata Statistics. We also collect metadata such as video length, resolution, and frame rate to ensure coverage diversity across temporal and visual characteristics.

Scene and Language Distributions. Fig. 5 illustrates the distribution of visual text quantity across six video scene categories. The largest number of text instances appears in Entertainment and Sports-related content, while knowledge and Media are less dense in text content.

Video Duration Distribution. VIDTEXT exhibits a wide range of video durations, with an average length of 108.2 seconds. As shown in Fig. 6, this highlights the multi-duration characteristic of VIDTEXT, ensuring the temporal diversity needed to support both short-form and long-form video understanding tasks.

Semantic Content Word Cloud. To visualize the semantic richness and diversity of video-text interactions, we construct a word cloud using all questions and answers in VIDTEXT. As shown in Fig. 7, high-frequency words such as text, video, content, and EXIT reflect a strong alignment between text and semantic reasoning. The co-existence of spatial keywords (e.g., LEFT, RIGHT), functional terms (e.g., score, speed), and contextual references (e.g., player, talent) highlights the multi-granular reasoning needs of the dataset.

TextNumber

60,000

50,000

40,000

30,000

20,000

10,000

0

Entertainment Knowledge LifeRecord Ego-centric Media Sports

Figure 5: Text quantity distribution across six scene categories.

[Figure 30]

1-5 min,8.10% 5-30 min, 6.40%

>30 min, 1.80%

30-60 sec, 50.40%

###### 0-30 sec, 33.30%

Figure 6: Video duration distribution in VIDTEXT.

#### E Details of Annotation

###### D.1 Instance Annotation

Each video underwent a two-stage text annotation process. In the first stage, annotators drew tight bounding boxes around visible text lines and assigned each to a category: ClearText or Illegible. s created. A tracking tool automatically propagated bounding boxes across frames using consistent Track IDs.More Details are shown in Fig. 8.

###### D.2 Clip-Level Annotation

Videos shorter than 1 minute were split into 5-second clips; longer ones into 20-second clips. For each clip, annotators recorded all visible, legible text and its temporal span. Repeated instances within a clip were marked only once. Illegible or heavily blurred texts were ignored. More Details shown in Fig. 9.

###### D.3 Video-Level Text Collection

A separate annotation team reviewed the OCR predictions from our model. Annotators removed hallucinated content and added missing instances. Chinese was annotated by full lines; other lan-

[Figure 31]

Figure 7: Word cloud of all questions and answers in VIDTEXT.

###### Instance Annotation Guidelines

- Step 1: Text Detection (Bounding Box)

- 1.Draw a bounding box around 3-5 visible text instance in each video.
- 2.Annotate entire text lines, not individual words or characters.
- 3.If the same text appears across multiple frames, assign it the same Track ID using the tracking tool provided.

- Step 2: Text Classification Each bounding box must be assigned one of the following text categories: Clear Text:clearly visible text. Illegible: text that is unreadable due to blur, occlusion, or low resolution.
- Step 3: Text Transcription All Text instances require transcription. For tracked text across multiple frames, you only need to transcribe it once—the tool will propagate it across the track automatically Special Handling: Blur or Occlusion If a text instance becomes blurred or occluded: If the blur/occlusion lasts 3 frames or fewer, continue the original track. If it lasts more than 3 frames, end the current track and create a new one labeled as Illegible. If a text transitions from unreadable to readable (or vice versa), create a new track with the updated label

Figure 8: Instance-Level Annotation Guidelines.

guages (e.g., English, German) were annotated by words. Each unique string was listed once in the final inventory. More Details shown in Fig. 9.

###### D.4 Holistic Reasoning

Annotators watched the full video and consulted the video-level text inventory to write one multilabel question per video (see Fig. 10). Each question included seven options describing high-level semantics such as scene, role, topic, or sponsor.

###### D.5 Local Reasoning

For every clip (as defined in D.2), annotators created one four-option multiple-choice question requiring reasoning between localized text and visual context (e.g., subtitle character behavior). The

###### Clip & Video-Level Annotation Guidelines

###### Clip Level:

###### Video Segmentation

- 1.Divide the video into consecutive temporal segments ：If the video is shorter than 1 minute:divide it into clips of 5 seconds each.Else; divide it into clips of 20 seconds each.
- 2.Each segment should have a clear start_time and end_time. Text Identification

- 1.For each clip, annotate all readable text instances that appear within the clip’s time span.
- 2.Ignore illegible, blurred, or heavily occluded text.
- 3.If the same text appears multiple times within the clip, annotate it only once. Video Level: Global Text Collection

- 1.Watch through the full video and record all clearly visible and legible text content.
- 2.You will be provided with a preliminary list of detected texts (from an automatic text detection model). In this case, carefully review and correct the list by adding missing texts and removing false positives to ensure accuracy.
- 3.Each unique text instance should be annotated only once (no need to mark repetitions). Language-Based Annotation Rules

- 1.For Chinese text: annotate by complete text lines (e.g., subtitle or sign line).
- 2.For Non-Chinese languages (e.g., English, German): annotate by individual words.
- 3.For mixed-language cases, follow the dominant language rule and note exceptions when needed.

Figure 9: CLip&Video-Level Annotation Guidelines.

###### Annotation Guidelines for Holistic Reasoning

goal:: Given the overall textual and visual content throughout the video—including information across multiple time segments—annotate a global question that requires semantic reasoning across time and space.

###### Input

You will be given the full video and its OCR transcription. Your goal is to:Observe the entire video, noting important text and visual elements across different timepoints.Identify highlevel topics, roles, actions, or patterns that emerge over time. Create a multi-option question that tests understanding of the video’s overall narrative or semantic structure, including content distributed across time. Select 3 correct options from a set of 7 plausible answers.

###### CoT Expectation:

You should simulate how a model would connect multiple distributed cues, such as:“The subtitle shows the name + stage text shows show name + outfit = talent show” “Multiple timepoints include branding (e.g., sponsor, stage banner) → context clue” “Introduction + mid-performance + audience shot = global understanding of scene”

GOOD EXAMPLE: Question：Based on the video text and description, which of the following statements accurately describe the scene and content of the video?

- "A": "The young performer is identified as a 12-year-old talent from a rural background.",
- "B": "The show being referenced is \"中国达人秀\" (China's Got Talent).", "F": "The show features a challenge round sponsored by \"海飞丝\"

Figure 10: HolisticReasoning Annotation Guidelines.

question must require multimodal reasoning and not be solvable using text or image alone. More Details shown in Fig. 11.

###### D.6 Temporal Causal Reasoning (TCR)

Given a reference text (e.g., scoreboard or subtitle), annotators identified the timestamp of its appearance, observed the following 3–30 seconds, and formulated a causal reasoning question. The answer was a single factual sentence describing the resulting action. Each QA pair was anchored to the cue’s timestamp. More Details shown in Fig. 12.

###### Annotation Guidelines for Local Reasoning

goal:: Within a specific time segment of the video, reason over the text and visual context to answer a multimodal question grounded in localized semantics.

###### Input

You are given a specific video segment along with:

- •Detected OCR text within the segment
- •The corresponding video frames Your task is to: Understand the meaning and context of the visible text in the clip.Interpret surrounding visual content (e.g., characters, objects, layout) Construct a multiple-choice question that tests the model's semantic understanding and reasoning ability Provide 4 candidate options and select the correct answer CoT Expectation: Ask: what does the text cause / reflect / imply? Simulate the model making the connection: “If the subtitle says ‘stay still’, and the character hides behind a wall → he’s afraid / threatened”

GOOD EXAMPLE: Q: “In the clip, the text 'Final Round' is shown. What does it suggest about the competition?” A: “The winner will be decided in this match.” Q: “When the subtitle says 'Don't move', what is the person doing?” A: “They are hiding quietly behind the shelf.”

Figure 11: LocalReasoning Annotation Guidelines.

###### Annotation Guidelines for Temporal Causal Reasoning

goal:Track a specific text instance in the video, analyze the sequence of related events, and annotate a question–answer pair that reflects their causal relationships.

###### Input

1.Locate the reference text

- •Find the timestamp where the given text appears clearly (e.g., scoreboard, sign, subtitle).
- •Pause at that moment and record the text content and timestamp. 2.Observe what happens next
- •Watch the following 3–30 seconds of the video.
- •Identify any actions, changes, or reactions that may be caused by or related to the text. 3.Write the QA pair Question: Frame a question that highlights the relationship (e.g., “what happened after…” / “how did the player respond to…”). Answer: Describe the actual action concisely and factually. CoT Expectation: you should consider the temporal progression: what happened after the text appeared, and why it might be related. Example: a low score triggers a coach’s timeout; a red light prompts braking. GOOD EXAMPLE: "question": "At a score of 105:83, what move did James make to score?",

"answer": "He stole the ball and dunked it.“ "question": "At the beginning of the game when the score was 0:0, how did the Warriors player score while being defended by Player 1?",

“answer": "By scoring a three-point shot",

Figure 12: TemporalCausalReasoning Annotation Guidelines.

###### D.7 Spatial Reasoning (SR)

As shown in Fig. 13, at a given timestamp, annotators located a reference text or entity and constructed a question requiring reasoning over its spatial relation to nearby visual elements (e.g., direction, proximity, interaction).

Quality Control All annotations underwent double review. Each item was cross-validated by a second annotator, and disagreements were resolved by expert adjudication. On a random sample of 200 items, we achieved an average inter-annotator agreement of 0.81 (Cohen’s ), indicating high reliability.

###### Annotation Guidelines for Spatial Reasoning

goal:At a specific timestamp, infer the spatial relationship between a text instance (or person) and surrounding visual elements—such as direction, relative position, or interaction.

###### Input

- 1.Locate the reference text

- •Find the timestamp where the given text appears clearly (e.g., scoreboard, sign, subtitle).
- •Pause at that moment and record the text content and timestamp.

- 2.Observe what happens next

- •Watch the following 3–30 seconds of the video.
- •Analyze the scene: what object or person is near, behind, or interacting?

- 3.Write the QA pair Compose a multiple-choice or open-form reasoning question and answer CoT Expectation: you should consider Reason about spatial layout: who is positioned where, and what action is implied. Use directional and functional cues: “behind”, “to the right”, “blocking”, “following”. GOOD EXAMPLE:

"question": "When the score was 31:18 and 2:09 remained in the game, where was Player 8 located when attempting the three-point shot?",

"answer": "Bottom-middle of the image, right 45-degree three-point position" "question": "With the score 0:0, who is the player defending Timber-wolves' Player 5 (white jersey)?", “answer": "Player 31"

Figure 13: SpatialReasoning Annotation Guidelines.

#### F Details of Experimental Settings

###### E.1 Model Configuration

In this section, we outline the primary baselines evaluated on our VidText.To ensure fair comparison across both open- and closed-source models, we explicitly standardize frame sampling and spatial resolution for each baseline as summarized in Tab. 5.

For proprietary models such as GPT-4o, Gemini 1.5 (Pro and Flash), and GPT-4-Turbo, we follow their official or API-supported settings. GPT-4o models support up to approximately 500 images inputs, for which we adopt a uniform sampling rate of 0.5 fps with an input resolution of 512 × 512 to accommodate most of our videos. GPT-4-Turbo is restricted to 16 frames, uniformly sampled across the video, and resized to the same resolution.

For open-source models, we align each configuration with their original public implementations. VideoChat-Flash, Qwen2-VL (7B), and All Qwen2.5-VL variants (3B/7B/72B) operate under a 1 fps sampling strategy, with a maximum of 768 frames extracted per video. Models that support extended temporal contexts—such as VideoLLaMA 3, InternVL 2.5, and LLaVA-OV—are provided with 64 uniformly sampled frames, resized to 336×336. ShareGPT4Video also uses 64 frames, but with a reduced spatial resolution of 224 × 224. LongVU and LongVA, are evaluated with sparse and extended frame settings. LongVU uses 1 fps sampling, while LongVA accepts up to 128 uniformly distributed frames. MiniCPM-V2.6 applies a fixed 64-frame sliding window, following its official implementation.

###### E.2 Human Performance Study

To assess the upper-bound of performance on VIDTEXT, we conducted a controlled human evaluation across all tasks in our benchmark. Three annotators with experience in video analysis and text recognition were recruited to answer a representative subset of questions spanning all eight task types. Each participant was given access to the full video content and instructed to answer using their best judgment, without time constraints. The average human accuracy across all tasks reaches 89.5%, substantially outperforming all evaluated models. In particular, humans demonstrated nearperfect scores in holistic and local OCR, reasoning, and spatial understanding tasks, highlighting the gap between human-level comprehension and the capabilities of current multimodal large models. These results serve as a reference ceiling for future model development and underline the complexity

Table 5: Frame–sampling and input-resolution settings for baselines.

Model Size Sampling Resolution Proprietary MLLMs

GPT-4-Turbo – 16 frames 5122 Gemini 1.5 Flash – 1 fps 5122 GPT-4o – 0.5 fps 5122 Gemini 1.5 Pro – 1 fps 5122

Open-source MLLMs LongVU 3 B 1 fps 4482 Qwen2.5-VL 3 B 1 fps 4482 Video-XL-Pro 7 B 1 fps 4482 LongVA 7 B 128 frames – MiniCPM-V2.6 7 B 64 frames 4482 VideoChat-Flash 7 B 1 fps 4482 Qwen2-VL 7 B 1 fps 4482 Qwen2.5-VL 7 B 1 fps 4482 VideoLLaMA 3 7 B 64 frames 3362 ShareGPT4Video 8 B 64 frames 2242 Oryx-1.5 32 B 64 frames 3362 LLaVA-OV 72 B 64 frames 3362 Qwen2.5-VL 72 B 1 fps 4482 InternVL 2.5 78 B 64 frames 3362

and nuance of the video-text understanding challenges posed by VIDTEXT.More details are shown in Tab. 2.

###### E.3 Experiment Environment.

All experiments are conducted on a server equipped with 4×NVIDIA A100 GPUs (80GB each). Model inference and evaluation are implemented in PyTorch with mixed-precision support.

#### G Model Prompts

Fig. 14 shows the prompt template used to obtain detailed frame-level captions from the Aria model. The prompt includes instructions to describe the scene, detect visible text, summarize actions, and relate them spatially and semantically. Tab. 6 lists the standardized prompt templates used for each task in VidText.

###### Aria Caption Generation Prompt

You are given images sampled from a video. Please imagine yourself in the scene and describe in detail what you see from your viewpoint. Your description should focus on the following aspects:

- 1. What is the overall scene or environment?
- 2. What visible objects or people are present?
- 3. Are there any texts (e.g., signs, labels, instructions)? If yes, what do they say?
- 4. What activities or actions are happening in the scene?
- 5. Are there any meaningful relationships between the scene texts and the objects, people, or actions around them? Please write the description in a natural and informative way, as if explaining what you are currently seeing. Avoid mentioning “image” or “frame”, and do not speculate beyond what is visible. Output format:

- - Scene description: [...]
- - Visible texts: [...]
- - Human and object activities: [...]
- - Spatial or semantic relationships (if any): [...]

Figure 14: Prompt template used for Aria to generate frame-level captions.

Table 6: Prompt templates used for VidText tasks.

Task Prompt Template

"Recognize all visual texts in the video. If the text is not in English, do not provide an English translation. Do not include any descriptions, narrative, or context. Output only the extracted text lines, each on a new line."

Holistic OCR

"Watch the video carefully and select the correct three answers. Question: {question} Options: {options} Please output your answer in the format: Correct Answers: A, B, C"

Holistic Reasoning

"Watch the video and answer the following question based on its content. Question: {question} Please output only the texts that appear in the specified time interval as a JSON array of strings, with each element representing one piece of text. Do not include any additional description or translation."

Local OCR

"Watch the video and answer the following multiple-choice question based on its content. Question: {question} Options:

Local Reasoning

- Option A: {text}
- Option B: {text}

... Please select the correct option."

"Watch the video and answer the following question based on its content. Please provide the time interval (in seconds, precise to 0.1s) during which the text appears in the video. Output your answer in JSON format with keys ’start’ and ’end’. For example: {"start": 0.0, "end": 30.0}. Do not include any extra commentary."

Text Localization

"Watch the video and answer the following multiple-choice question based on its content. Question: {question} Options:

Temporal Causal Reasoning

- Option A: {text}
- Option B: {text}

... Please select the correct option."

Text Tracking (Same prompt as Spatial Reasoning)

"Watch the video and answer the following multiple-choice question based on its content. Question: {question} Options:

Spatial Reasoning

- Option A: {text}
- Option B: {text}

... Please select the correct option."

### HolisticOCR

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Q:Please provide all text in video Timestamp:30.1 GT: [12岁的天籁唱将12岁的天籁唱将], [克服紧张后惊艳全场] ,[（下）], [《中国达人秀》] ...

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Timestamp:110.2

Q:Please provide all text in video GT: [ When] ,[the], [time] ,[comes] ,[to],[pack] ,[and],[head] ,[back], [home] ...

### HolisticReasoning

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Q:Based on the video which are accurately depicted as part of the learning Timestamp:64.1 experience in the video? GT: 1.Completing a session on HTML Basics Level 3; 2. Completing a session on Calculus Level 1;...

.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Timestamp:27.6

Q:Based on the video, which of the following statements are true GT: 1.The character uses Google Classroom to check their math homework assignments ;2.The character initially forgets to write down their homework assignment.];...

### Local OCR

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Q:What text appears between 1295s and 1303s in the video? Timestamp.2814.1 GT: [IF],[You],[Like],[The],[Ride],[Tip],[The],[Guide!],[Tips],[Are],[Appreciated]...

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Timestamp:368.4

Q:What texts appear between 230s and 240s in the video? GT: [pendulum],[squat],[3x10],[reps].

- Figure 15: (Top) More examples of HolisticOCR. (Middle) More examples of HolisticReasoning. (Bottom) More examples of LocalOCR.

### Local Reasoning

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Q:"What is the special significance of the text '17 October 2021'? Timestamp:1819.1 GT: Marathon race day.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Timestamp:2452.6

Q:What does the text '1322' represent? GT: The author's race number.

### Textlocalization

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Q:When does the text 'MODE OF TRANSPORT E-SCOOTER 3/5' exist Timestamp.867.1 in the video? GT: [1215.1s,1220.2s]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Timestamp:550.3

Q: When does the text 'tAWE beach & bar grill' exist in the video? GT: [665.0s,667,8s]

### TemporalCausalReasoning

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Q:When the score was 10:7 and 8:19 remained in the first quarter, what Timestamp.811.7 offensive action did the Thunder's Player 2 (white jersey) choose? GT: Passed to a teammate for a three-point shot

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

7

[Figure 126]

Timestamp:530.3

Q: At 3:06 of the game, who did the red-black No.10 player pass the ball to, and what happened next? GT: Passed to No.7, who scored a goal.

- Figure 16: (Top) More examples of LocalReasoning. (Middle) More examples of TextLocalization. (Bottom) More examples of TemporalCausalReasoning.

### TextTracking

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

11

[Figure 135]

[Figure 136]

[Figure 137]

Q:From 19:14 remaining in the first quarter to 19:10, please provide the Timestamp.581.8 bounding box of the player wearing the white jersey number 10 at both the start and end time points. GT: [0.44, 0.83...],[0.67,0.83...]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Timestamp:550.3

Q: Please provide the start and end positions for the text \"CRAVE COFFEE\".? GT: [0.32, 0.77...],[0.54,0.55...]

### SpatialReasoning

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

11

[Figure 154]

[Figure 155]

Q:t 86:36 of the match, who is behind Argentina’s No.24 player,and Timestamp.182.8 what is he doing? GT: Green No.24, trying to stop the attack.

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

Timestamp:585.3

Q: When the score is 0:5, which player is defending Clippers number 24 (white) under the basket? GT: 15

- Figure 17: (Top) More examples of TextTracking. (Bottom) More examples of SpatialReasoning.

- H More Visualization Results We present additional visualizations of our VidText annotation examples in Fig 15, 16, and 17.

