# arXiv:2512.21334v2[cs.CV]10Apr2026

## Streaming Video Instruction Tuning

Jiaer Xia1∗ Peixian Chen2∗ Mengdan Zhang2 Xing Sun2 Kaiyang Zhou1

1 Hong Kong Baptist University 2 Tencent Youtu Lab https://jiaerxia.github.io/Streamo/

##### Abstract

We present Streamo, a real-time streaming video LLM that serves as a general-purpose interactive assistant. Unlike existing online video models that focus narrowly on question answering or captioning, Streamo performs a broad spectrum of streaming video tasks, including real-time narration, action understanding, event captioning, temporal event grounding, and time-sensitive question answering. To develop such versatility, we construct Streamo-Instruct465K, a large-scale instruction-following dataset tailored for streaming video understanding. The dataset covers diverse temporal contexts and multi-task supervision, enabling unified training across heterogeneous streaming tasks. After training end-to-end on the instruction-following dataset through a streamlined pipeline, Streamo exhibits strong temporal reasoning, responsive interaction, and broad generalization across a variety of streaming benchmarks. Extensive experiments show that Streamo bridges the gap between offline video perception models and realtime multimodal assistants, making a step toward unified, intelligent video understanding in continuous video streams.

##### 1. Introduction

Recent advances in video large language models (LLMs) [4, 24, 42, 51] have demonstrated remarkable capabilities in analyzing complete, pre-recorded videos, which establish strong baselines for offline video understanding. These models excel at holistic reasoning over long temporal sequences when given static, temporally bounded inputs [20, 55], enabling tasks such as video captioning, summarization, and question answering. However, the requirements of real-time interactive AI assistants are fundamentally different: they must process continuous, unbounded video streams and respond to dynamic instructions as events unfold, often under strict latency constraints.

∗Equal contribution Corresponding author

Existing offline models struggle to meet the demands of the streaming setting because they are designed to process entire clips before producing a single output [33, 39, 43]. In contrast, real-time applications require the model to continuously interpret an incoming video stream, detect when the visual context satisfies a task condition, and decide what information to output at that moment. This introduces two key challenges: 1) handling continuous, unbounded data flow without losing context, and 2) managing variable response timing and granularity across multiple tasks, which may require frame-level or longer-term temporal reasoning. A truly capable streaming video LLM must therefore integrate both task understanding and frame-level decision-making, enabling it to evaluate evolving visual contexts, determine appropriate moments to respond, and generate coherent outputs without delaying or missing critical information.

To address these challenges, recent studies [33, 39, 43] have attempted to extend offline video models for streaming by introducing a separate decision module that predicts response states before invoking the offline model to generate content. While this approach preserves the reasoning capacity of the base model, it creates a trade-off between accuracy and efficiency: lightweight decision modules often lack the capacity to fully understand complex instructions and temporal dependencies, while larger modules substantially increase computational cost and inference latency. Moreover, separating decision-making from response generation prevents tight coupling between perception and response, limiting the model’s ability to seamlessly adapt to rapidly changing streaming contexts.

In this work, we propose Streamo1, a real-time streaming video LLM that unifies decision-making and response generation in an end-to-end manner. Instead of relying on an external controller, we embed frame-level response state prediction directly into the model. Specifically, three decision heads—Silence, Standby, and Response—allow the model to continuously monitor the input stream and make fine-grained judgments about when to output. Once a re-

1The letter o in Streamo means ‘omni’, reflecting its multi-task and multi-modal capabilities.

[Figure 1]

The video features a bartender demonstrating how to make a pink cocktail using ingredients like Tahoe Blue Vodka, lemon, mint, and a red mixer.

[Figure 2]

[Figure 3]

Real-time Narration

Narrate the video in real time, updating the description frame-by-frame or moment-by-moment as events unfold.

[Figure 4]

Man picks up glass.

Man places glass down.

Man picked up a shaker.

Camera focuses on

Hand shovels ice cubes.

Pouring liquid into the shaker.

Lemon is cut in half. Lemon is squeezed.

bottle and shaker.

Action Caption

[Figure 5]

Locate and pinpoint a sequential series of specific actions or steps in the video.

[Figure 6]

<Silence>

Man shovels some ice cubes into shaker Man pours some Vodka into the shaker Man cuts lemon. Man squeezes lemon.

[Figure 7]

Event Caption

Detect and summarize each event sequence in the video.

[Figure 8]

The man begins to prepare the drink, while the woman watches.

He fills a blue shaker tin with ice cubes, followed by pouring Tahoe Blue Vodka into the shaker.

He then cuts lemons in half and squeezes fresh lemon juice into the shaker.

[Figure 9]

Event Grounding

Temporally localize the event: ‘Man added vodka to the shaker and squeezed in some lemon juice’. Respond once it has finished and summarize its time period.

[Figure 10]

<Silence>

Given event occurred between 34s to 50s.

[Figure 11]

Time-sensitive QA

What is the man currently holding in his hand.

[Figure 12]

shaker glass knife squeezer

<Silence> <Silence>

Figure 1. An example of multi-task annotation in Streamo-Instruct-465K. Each task is carefully labeled with the corresponding response time boundaries and content, following established annotation standards. The same video is annotated with multiple distinct tasks. The video shown in this example is sourced from ActivityNet [5].

sponse state is triggered, the model immediately produces the corresponding textual output, achieving one-pass inference that significantly improves both the accuracy of response timing and the efficiency of real-time generation.

Training Streamo requires high-quality, temporally consistent supervision, yet existing datasets often combine heterogeneous sources with inconsistent annotation standards [15, 17, 18]. These inconsistencies make it difficult for the model to learn precise temporal alignment or multi-task response behaviors. To overcome this problem, we construct Streamo-Instruct-465K, a large-scale, multitask instruction-following dataset designed specifically for streaming video understanding and interaction. The dataset standardizes three levels of response granualarity, provides unified temporal annotations for event boundaries, and covers diverse tasks including real-time narration, action and event captioning, temporal grounding, and time-sensitive question answering. Each video is annotated for multiple tasks, providing consistent guidance that strengthens both instruction-following and temporal reasoning. An example of the annotations is shown in Fig. 1.

Extensive experiments demonstrate that our end-to-end training paradigm effectively converts offline models into online streaming assistants. Streamo outperforms existing online approaches across both streaming and offline benchmarks, exhibiting strong temporal awareness,

accurate frame-level decision-making, and robust multitask instruction-following. To further support research in this domain, we also introduce a comprehensive streaming benchmark named Streamo-Bench, which evaluates instruction understanding across diverse interactive tasks.

Our contributions are threehold: 1) We propose a simple and effective end-to-end training framework that converts offline video models into real-time straeming assistants. 2) We introduce a multi-task instruction tuning dataset with unified temporal annotation and fine-grained response supervision. To our knowledge, this is the largest scale instruction tuning dataset for streaming video understanding and interaction. 3) We establish a comprehensive benchmark for streaming video instruction-following and provide strong baseline models for future research. All research resources including code, models, and datasets will be made publicly available.

##### 2. Related Work

Video Large Language Models The field of vision foundation models [9, 26, 29, 31] has made remarkable progress in recent years, extending capabilities from static image understanding to more general video comprehension. Building on this foundation, numerous advanced video LLMs have emerged. For example, InternVideo2.5 [44] can pro-

cess videos spanning several hours, while Keye-VL-1.5 [48] demonstrates sophisticated reasoning abilities, effectively performing complex thinking process based on video content. A critical limitation, however, is that these state-ofthe-art models operate in an offline fashion, requiring the entire video as input before producing any output. This single-pass approach prevents them from handling continuous video streams, as they lack mechanisms to identify the precise temporal moments for generating responses in ongoing streams.

Streaming Video Understanding To tackle real-time interaction, various methods have been proposed in the literature to turn offline video LLMs into online assistants that can identify the appropriate moment to respond in video streams. For instance, Dispider [33] and StreamBridge [39] employ an auxiliary model to segment a video stream into fixed-length clips before feeding them to an offline model. However, this strategy introduces significant computational overhead in both training and inference and often fails to maintain context during multi-turn interactions. On the other hand, VideoLLM-Online [6] and StreamingVLM [47] train the model in a supervised way to directly predict response timing using a special [EOS] token. However, this approach is limited to real-time narration and cannot balance between silence and response state. To overcome these problems, we propose an end-to-end training framework along with a multi-task instruction-following dataset specifically designed for streaming video understanding and interaction.

Streaming Video Benchmarks OVO-Bench [23] introduces 12 distinct tasks, incorporating tests for a model’s ability to proactively respond. Similarly, STREAMBENCH [46] and SVBENCH [49] concentrate on assessing multi-turn conversational abilities within continuous video contexts. A key limitation, however, is their predominant reliance on question-answer (QA) style setups—typically requiring the model to choose an answer from given options—which does not adequately assess broader instruction-following abilities such as event grounding and captioning. Motivated by the goal that streaming video models should evolve into real-time AI assistants, we introduce Streamo-Bench, a benchmark designed to probe a model’s perceptual and responsive capabilities across diverse instructions, moving beyond the constraints of traditional QA-based evaluation.

##### 3. Streamo: Architecture and Training

###### 3.1. Preliminaries

Traditional video understanding models [2, 7] follow an offline paradigm where the complete video V , question Q, and answer A are processed using a single-turn format. Formally, given a video V = {v1,v2,...,vT} of length T and a

question Q, the model directly generates an answer A. This approach assumes that the entire video is accessible before inference begins, which is impractical for real-time streaming scenarios where video frames arrive sequentially.

In contrast to offline settings, streaming video understanding processes video content as it arrives in a continuous stream. The model must make decisions based on partial observations V:t = v1,v2,...,vt, where t ≤ Tt meaning that the model does not have access to future frames. This temporal constraint requires fundamental changes to both the data structure and training paradigm.

###### 3.2. Data Structure

To simulate streaming scenarios during training, we reformulate the single-turn offline format into a multi-turn dialogue structure. Specifically, a complete video V is temporally segmented into N contiguous segments:

V = {V (1),V (2),...,V (N)} (1)

where V (i) denotes the i-th video segment. Each segment is explicitly annotated with temporal boundaries using special markers, e.g., <2s-3s>, to encode temporal information. The multi-turn dialogue is constructed as:

D = {(V (1),R(1)),(V (2),R(2)),...,(V (N),R(N))} (2)

where R(i) denotes the response at turn i. Questions and answers are strategically inserted at appropriate turns based on the dataset characteristics and task requirements.

To enable efficient parallel training while maintaining compatibility with standard supervised fine-tuning paradigms, we convert decision process into predictions for the following state tokens:

<Silence>: The model remains silent and continues processing incoming frames. <Standby>: The model detects relevant video input and waits for complete information. <Response>: The model receives enough information and will generate a response.

This design empowers the model with frame-level decision-making capabilities while maintaining the nexttoken prediction framework. As illustrated in Fig. 2, three discrete response states are directly integrated into the normal token prediction process: the model outputs <Standby> upon detecting relevant input and <Response> when it is ready to answer. A training example is shown in Tab. 1. With this multi-turn dialogue training format, we can simulate realistic streaming video interactions and pose questions at any point in time.

###### 3.3. Training

The multi-turn streaming format introduces severe class imbalance among the three response states. In typical streaming scenarios, <Silence> tokens dominate the distribution

Table 1. The format of a multi-turn dialogue.

###### <Silence> <Silence> …

<Standby> <Response> He is

LM Loss swimming in the ocean.

SYSTEM PROMPT USER <0s-1s><video> ASSISTANT <Silence>

#### Streamo

…

<1s-2s><video> Notify me when the light turns green.

USER

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

What is the

ASSISTANT <Silence>

man doing?

- USER <2s-3s><video> ASSISTANT <Silence>

- USER <3s-4s><video> ASSISTANT <Standby>

- USER <4s-5s><video> ASSISTANT <Response> The light just turned green.

t0 tn

Relevant frame

Figure 2. Streamo’s architecture. Streaming video data is organized into an interleaved, multi-turn dialogue structure that directly integrates a response-state token into the data sequence, enabling end-to-end parallel training.

class imbalance in streaming data. The LCE is the standard cross-entropy loss:

(often more than 80% of the time), while <Response> tokens are sparse. This imbalance biases the model toward remaining silent, making it difficult to learn response timing.

|V|

ez

, (6)

LCE(i,ti) = −log pt

− zi,t

= log

i,j

i

i

To mitigate this, we apply focal weighting [25] specifically to the three special state tokens. Let S = {ssilence,sstandby,sresponse} denote the special token for the three states. For each prediction, we compute a focal weight that emphasizes hard examples:

j=1

where zi,j is the logit for token j at position i and |V| is the vocabulary size. This computes the negative log-likelihood of the true token. The total loss averages over all valid (nonmasked) positions indicated by M:

)γ, (3)

wfocal(xi) = (1 − pc

1 |M| i∈M

i

Li. (7)

Ltotal =

where xi represents the input features at position i, and pc

is the predicted probability for the true class ci at position i. γ ≥ 0 is the focusing parameter that controls the rate at which easy examples are down-weighted. To further balance the rare classes, we introduce frequency-based alpha weights. For each special token k ∈ S with count nk in the current batch:

i

This ensures that the loss is not affected by sequence length variations across examples in the batch.

##### 4. Streamo-Instruct-465K

###### 4.1. Data Construction

To provide clear supervision for each round of response decisions, we re-annotated a large-scale training set with detailed temporal boundary labels based on the existing opensource video datasets. We predefined multiple tasks spanning different response granularities, assigning each video several types of task annotations. This approach offers several advantages. First, a unified annotation protocol is applied across datasets, avoiding the inconsistencies and biases that arise when naively mixing datasets with heterogeneous labeling standards. Additionally, each video carries multiple task types with clearly delineated response boundaries, enabling the model to better perceive and understand varying task requirements, develop robust instructionfollowing capabilities, and execute a range of real-time response tasks. Below, we detail the annotation protocol for each task.

nj nk

1 |S|

· j∈S

, (4)

αk =

where |S| = 3 is the number of special states. This assigns larger weights to less frequent special tokens.

The final loss combines the focal weighting and frequency balancing:

 

wfocal(i)LCE(i,ti), ti ∈ S LCE(i,ti), otherwise

αt

i

, (5)

Li =



The two weighting mechanisms are computed independently and multiplied into the cross-entropy loss. Together, they focus the model on both challenging and infrequent tokens, improving learning of response timing despite severe

###### Time-sensitive QA

###### Action Caption

###### 5.8%

Event Caption

6.7%

34.8%

12.7% Narration

13.8%

Offline QA

26.3%

Total Samples: 465.8K

Event Grounding

Total Videos: 135,875

68,273

70,000

60,000

NumberofVideos

50,000

40,000

###### 50.25%

30,000

21,834

20,529

19,153

20,000

16.07%

15.11% 6,086 4.48%

14.1%

10,000

0

0-30s 30-60s 60-120s 120-240s 240s+

Duration Range

Figure 3. Dataset distribution overview. Left: task distribution; Right: video duration distribution.

Real-time Narration This task performs real-time commentary over video, requiring second-by-second descriptions that capture fine-grained visual changes. The annotation protocol is: 1) segment each video at one-second intervals; 2) for every adjacent pair of one-second segments (i.e., a two-second window), use Qwen2.5-VL-72B [3] to describe the changes observed between them; 3) concatenate the per-second outputs and send the full narration to GLM-4.5 [50] for post-processing to remove repetitions and redundancies, smooth transitions, and ensure coherent, context-aware narration.

Event Caption This task is similar to standard video captioning but requires the model to detect event boundaries and provide the corresponding caption when an event ends. To construct supervision: 1) generate segment-level captions with the ARC-Hunyuan-Video-7B [16] model; 2) temporally ground each caption using the same model; 3) retain only those videos in which all segment captions have mutually consistent, overlapping time spans that align with the original output. This yields two benefits: it filters out erroneous, noisy data and produces samples with sharper, more explicit event boundaries, enabling clearer supervision.

Action Caption This task mirrors event captioning but narrows the focus from dense events to discrete actions or procedural steps. We reuse the event-caption pipeline and augment it with action-oriented prompts and targeted filtering. This produces cleaner, step-level supervision with sharper action delineation.

Event Grounding The grounding annotation is similar to the offline setup, where each sample pairs an event caption with its corresponding temporal span. The key difference in the online setting is that the caption is provided in advance, and the model must continuously monitor the subsequent video stream to detect the specified event and localize its

occurrence in time. We randomly sample captions from the event-caption annotations, rewrite them for grounding, and integrate existing datasets to broaden coverage and improve robustness.

Time-sensitive QA This task targets questions whose correct answers change over time in a dynamic video stream. To construct supervision: 1) process each video with GLM-4.5V [38] model to detect change points across multiple aspects—object attributes (e.g., color, size, state), spatial positions, actions and interactions, counts, and scene or context shifts; 2) generate question–answer pairs from these variations by posing a single, unified question and providing diverse, time-specific answers at the corresponding time points.

###### 4.2. Statistics

Using a unified annotation standard and protocol, we labeled and curated a total of 400K valid samples and additionally merged offline video QA data from the LLaVAVideo [53] dataset, culminating in Streamo-Instruct-465K, and the task distribution is shown on the left of Fig. 3. We integrated multiple open-source video datasets as sources, including Koala [41], LLaVA-Video [53], ActivityNet [5], QVHighlight [32], YouCook2 [57], HACS [54], EgoTimeQA [11], DiDeMo [1], and COIN [35], yielding 135,875 videos in total. The distribution of video durations is shown on the right of Fig. 3.

##### 5. Experiments 5.1. Models and Datasets

To assess the effectiveness of our training strategy, we adopt Qwen2.5-VL [3] as our base model, across both 3B and 7B model size. Meanwhile, we additionally conduct experiments based on several existing state-of-the-art offline

- Table 2. Comparison with state-of-the-art on OVO-Bench. ‘Streamo Framework’ denotes adapting offline models to the online setting using our training framework. ET-Instruct-3B is trained with ET-Instruct-164K and † indicates LLaVA-Video data is added as offline support. ∗ means the model is trained at 1 fps and evaluated at 2 fps.

Real-Time Visual Perception Backward Tracing Forward Active Responding Overall Avg. OCR ACR ATR STU FPD OJR Avg. EPM ASI HLD Avg. REC SSR CRR Avg. Overall Avg.

Model # Frames

###### Open-source Offline Models

Qwen2-VL-72B [40] 64 65.77 60.55 69.83 51.69 69.31 54.35 61.92 52.53 60.81 57.53 56.95 38.83 64.07 45 49.3 56.27 LLaVA-Video-7B [53] 64 69.13 58.72 68.83 49.44 74.26 59.78 63.52 56.23 57.43 7.53 40.4 34.1 69.95 60.42 54.82 52.91 LLaVA-OneVision-7B [21] 64 66.44 57.8 73.28 53.37 71.29 61.96 64.02 54.21 55.41 21.51 43.71 25.64 67.09 58.75 50.5 52.74 Qwen2-VL-7B [40] 64 60.4 50.46 56.03 47.19 66.34 55.43 55.98 47.81 35.48 56.08 46.46 31.66 65.82 48.75 48.74 50.39 InternVL-V2-8B [10] 64 67.11 60.55 63.79 46.07 68.32 56.52 60.39 48.15 57.43 24.73 43.44 26.5 59.14 54.14 46.6 50.15 LongVU-7B [34] 1fps 53.69 53.21 62.93 47.75 68.32 59.78 57.61 40.74 59.46 4.84 35.01 12.18 69.48 60.83 47.5 46.71

###### Open-source Online Models

Flash-VStream-7B [52] 1fps 24.16 29.36 28.45 33.71 25.74 28.8 28.37 39.06 37.16 5.91 27.38 8.02 67.25 60 45.09 33.61 VideoLLM-online-8B [6] 2fps 8.05 23.85 12.07 14.04 45.54 21.2 20.79 22.22 18.8 12.18 17.73 - - - - Dispider-7B [33] 1fps 57.72 49.54 62.07 44.94 61.39 51.63 54.55 48.48 55.41 4.3 36.06 18.05 37.36 48.75 34.72 41.78 ViSpeak-7B [14] 1fps 75.17 58.72 71.55 51.12 74.26 66.85 66.28 59.93 48.65 63.98 57.52 33.81 68.52 60.42 54.25 61.08

###### Streamo Framework

ET-Instruct-3B [28] 1fps 65.10 35.78 56.90 35.39 24.75 60.87 46.47 41.81 35.14 8.6 28.52 20.06 52.31 67.50 46.62 40.54 ET-Instruct-3B† [28] 1fps 71.14 50.46 67.24 37.08 60.40 60.33 57.78 48.82 48.56 11.29 36.22 13.68 48.62 60.00 40.77 44.92 Streamo-3B 1fps 78.52 52.29 67.24 44.38 55.45 71.20 61.51 51.18 57.43 16.67 41.76 27.94 50.72 82.5 53.72 52.33

- Streamo-7B 1fps 79.19 57.80 75.00 49.44 64.36 70.11 65.98 54.55 52.03 31.72 46.10 29.96 51.03 83.33 54.77 55.61

- Streamo-7B 2fps∗ 77.18 66.06 76.72 45.51 66.34 72.83 67.44 55.56 58.11 33.87 49.18 30.84 57.55 82.5 56.96 57.86

models, including Qwen3-VL [37], and InternVL-3 [58], to demonstrate the compatibility of our framework; these results are presented in the Supplementary material. In addition to training on our proposed Streamo-Instruct-465K dataset, we also compare against ET-Instruct-164K [28], a large-scale instruction-tuning dataset with rich temporal information that has been widely used in prior work to train online video models. To enable a fairer comparison with Streamo-Instruct-465K, we also report results on a mixed dataset comprising ET-Instruct-164K and LLaVA-Video.

###### 5.2. Benchmarks

We evaluated our model across three dimensions of benchmarks: Online, Offline, and Stream Instruction. For the online setting, we adopted OVO-Bench [23], which covers three temporal perception modes, including realtime, backward, and forward, and also spans a total of 12 subtasks. The offline evaluation used standard general video understanding benchmarks, including the shortvideo benchmarks MVBench [22] and TempCompass [27],

- as well as the long-video benchmarks VideoMME [12] and LongVideoBench [45], providing a comprehensive assessment of capabilities. In addition, to assess multi-instruction following in an online context, we constructed StreamoBench, which includes 300 videos and 3,000 instruction tasks. Each video is paired with tasks of varying temporal scopes and granularities to measure the model’s adherence

to instructions, providing an important metric for building a reliable real-time AI assistant. Detailed information for Streamo-Bench is given in the Supplementary material.

###### 5.3. Implementation Details

Across all models, we use a unified training setup. Full parameter tuning is applied with the vision encoder frozen, and only the connector and the LLM will be updated. Training runs for a single epoch with a batch size of 512 and a learning rate of 1e-5. For multi-turn dialogue construction, each video is split into turns of one second, and frames are sampled at 1 fps. The hyperparameter gamma in Eq. (3) is set to 2. In experiments that include LLaVA-Video, we restrict the training data to the same subset used by StreamoInstruct-465K to ensure a direct and fair comparison.

###### 5.4. Main Results

Comparison with SOTA on Online Video Benchmarks The main results are shown in Tab. 2. Using the Streamo framework, we train the models with ET-Instruct and Streamo-Instruct datasets and compare their performance to currently available open-source offline and online models. The key findings are as follows: 1) Streamo significantly outperforms SOTA. It is clear that our proposed Streamo-7B exceeds the previous SOTA, Dispider, by +13.83% on average performance. Moreover, we observe that the model trained at 1 fps can be directly evaluated

- Table 3. Results on offline video benchmarks. The table compares converted online models with their original offline base models and SOTA models. Numbers in parentheses denote performance differences from the corresponding offline models.

OVO Real-Time

OVO Backward

Model

MVBench TempCompass VideoMME LongVideoBench Avg

###### Proprietary Models

Gemini-1.5-pro [36] 69.3 62.5 60.5 67.1 75.0 64.0 66.4 GPT-4o [19] 64.5 60.8 64.6 70.9 71.9 66.7 66.6

###### Open-source Online Models

Flash-VStream-7B [52] 28.4 27.4 61.2 - 61.2 - VideoLLM-online-8B [6] 20.8 17.7 33.9 - 26.9 - Dispider-7B [33] 54.6 36.1 - - 57.2 - StreamingVLM-7B [47] 62.0 - 69.2 - 65.1 59.0 -

###### Streamo Framework

Qwen2.5-VL-3B [3] 54.6 37.8 67.0 64.4 61.5 54.2 56.6 ET-Instruct-3B [28] 46.5 (-8.1) 28.6 (-9.2) 65.8 (-1.2) 60.3 (-4.1) 56.6 (-4.9) 51.2 (-3.0) 51.5 (-5.1) ET-Instruct-3B† [28] 57.8 (+3.2) 36.2 (-1.6) 68.1 (+1.1) 63.7 (-0.7) 59.6 (-1.9) 54.9 (+0.7) 56.7 (+0.1) Streamo-3B 61.5 (+6.9) 41.8 (+4.0) 67.9 (+0.9) 66.2 (+1.8) 61.8 (+0.3) 56.2 (+2.0) 59.2 (+2.6)

Qwen2.5-VL-7B [3] 58.8 42.2 69.6 71.7 65.1 56.0 60.6 Streamo-7B 66.0 (+7.2) 46.1 (+3.9) 72.3 (+2.7) 71.8 (+0.1) 67.9 (+2.8) 59.2 (+3.2) 63.9 (+3.3)

- at 2 fps without retraining, achieving an additional +4.66% performance improvement, indicating robust generalization to higher test-time frame rates; 2) Streamo-Instruct-465K dataset surpasses existing dataset. Compared with the ET-Instruct-164K, our proposed Streamo-Instruct-465K delivers a comprehensive performance advantage, with +7.1% on forward task and +11.79% overall; 3) Offline supervision can hinder online learning. Augmenting ET-Instruct with the offline LLaVA-Video dataset boosts real-time perceptual accuracy but compromises streaming ability, revealing a trade-off inherent to offline-only supervision. This also demonstrates that Streamo-Instruct-465K transfers effectively to online, streaming scenarios while maintaining strong offline perceptual capability. Comparison with SOTA on Offline Video Benchmarks To evaluate the general video understanding capability of models after conversion to the online setting, we compare Streamo against the SOTA method and original offline base model on a suite of general offline video benchmarks, with results reported in Tab. 3. The findings show that, after conversion, Streamo retains strong perceptual performance on offline benchmarks across both short-form and long-form videos, surpassing the SOTA, StreamingVLM, in every benchmark. Meanwhile, models trained with our StreamoInstruct-465K exhibit consistent improvements over base models, with Streamo-7B achieves an average improvement of +3.4% based on Qwen2.5-VL-7B. Holding architecture and training setup constant, Streamo-Instruct-465K also provides a clear advantage over alternative data recipes, outperforming ET-Instruction and LLaVA-Video by +7.8%

Table 4. Ablation study of loss functions for online training on OVO-Bench Forward Active tasks.

Base Model Loss Type REC SSR CRR Qwen2.5-VL-3B CrossEntropy 6.45 20.99 41.67 Qwen2.5-VL-3B Loss Scale 18.62 41.02 49.17 Qwen2.5-VL-3B Focal Loss 27.94 50.72 82.5

InternVL3-2B CrossEntropy 9.46 20.50 40.42 InternVL3-2B Loss Scale 21.20 31.47 48.75 InternVL3-2B Focal Loss 29.23 47.38 80.42

and +2.5% on average, respectively. These results underscore that our training framework and data not only enable effective transformation of models for streaming video understanding but also preserve and enhance core perceptual capabilities on offline video tasks.

Streamo-Bench To evaluate the model’s ability to follow different instructions and perform varied tasks, we assign multiple instruction-driven tasks to a single video, including forward grounding, backward grounding, narration captions, dense captions, and time-sensitive question answering. Details, examples, and statistics for these tasks are presented in the Supplementary material.

As shown in Tab. 5, existing online models show deficiencies in comprehensive multi-task coverage. Our analysis indicates that these shortcomings stem largely from an inadequate ability to comprehend and follow com-

Table 5. Evaluation results on Streamo-Bench. Forward and backward grounding are determined by whether the query refers to a time point before or after the event period, and results are using the mIoU metric. Caption evaluation is conducted by calculating the win rate with Qwen2.5-VL-72B model. TSQA denotes Time-Sensitive QA, i.e., questions whose answers change over time.

Grounding Caption TSQA

Model

Average Forward Backward Narration Dence Caption Accuracy Recall

Flash-VStream-7B [52] 0 0 23.5 25.9 30.8 13.1 15.6 VideoLLM-online-8B [6] 0 0 42.0 6.6 19.6 7.6 12.6 Dispider-7B [33] 0 8.33 31.6 29.2 14.0 4.4 14.6 StreamingVLM-7B [47] 0 0 68.5 24.0 11.8 43.1 24.6 Streamo-3B 14.7 27.5 71.4 68.5 20.1 65.7 44.7 Streamo-7B 29.4 38.3 75.9 72.8 51.6 63.9 55.3

plex instructions. For instance, removing predefined options leads to widespread failure—as the grounding results show—highlighting a vulnerability to open-ended prompts. Furthermore, in standard QA scenarios, models frequently overlook instructions to update answers as conditions change, which severely degrades recall. We probe instruction comprehension and prompt sensitivity further with additional experiments in the Supplementary material. Collectively, these observations expose a critical gap in current capabilities. In contrast, Streamo demonstrates robust performance across tasks, clearly exhibiting strong instruction-following ability. This outcome validates both the diagnostic power of our benchmark and the effectiveness of our method in learning generalized instructionfollowing capabilities.

###### 5.5. Ablation

To evaluate the effectiveness of our focal loss for training the three decision states, <Silence>, <Standby>, and <Response>, we compare it to standard cross-entropy loss. As shown in Tab. 4, training without state-aware reweighting severely limits performance due to significant class imbalance. In the Streamo-Instruct-465K dataset, the empirical ratio of state labels is approximately <Silence>:<Standby>:<Response> = 12:3:2, which biases conventional training toward predicting Silence and suppresses actual Response predictions.

A straightforward remedy is to assign fixed class weights inversely proportional to label frequency. Specifically, we set the weights to 0.3, 1.3, and 2.0 for silence, standby, and response, respectively, to emphasize response timing. As illustrated in the line “Loss Scale” in Tab. 4), this adjustment effectively mitigates the degradation caused by imbalance. However, fixed weighting fails to capture token-level hardness and sequence-level heterogeneity in decision-state distributions—for instance, narration tasks may contain multiple responses, whereas a QA task might include only one.

Our proposed focal loss addresses this limitation by dynamically reweighting losses based on token-level hardness

and per-batch state frequency, thereby providing more adaptive supervision for response-timing decisions. Across both InternVL-3-2B and Qwen2.5-VL-3B backbones, training with the proposed focal loss consistently yields substantial improvements over both the vanilla cross-entropy and fixedweight baselines.

##### 6. Conclusion

Our work targets the advancement of streaming video by jointly addressing model training and data construction. We introduce an end-to-end training framework together with a large-scale instruction-tuning dataset, Streamo-Instruct465K, enabling the conversion of multiple state-of-the-art offline models into online version. The resulting model, Streamo, not only excels on streaming benchmarks but also rivals top-performing offline models. Furthermore, our proposed Streamo-Bench, which simulates complex multiinstruction scenarios, showcases Streamo’s robust multitasking capabilities. Collectively, these contributions mark a significant leap towards creating general-purpose, realtime, and interactive AI assistants.

##### 7. Limitations and Future Work

In terms of limitations, while our approach achieves strong accuracy, it is limited by the inherent challenges of streaming video’s unbounded temporal context. Our current pipeline lacks specialized long-sequence optimizations, leading to significant memory and latency costs that become prohibitive as sequence length grows.

By leveraging our framework’s compatibility with existing techniques, we can integrate KV-cache management and visual token pruning to reduce computational overhead, alongside exploring sliding-window attention and adaptive frame compression for refined context management. Collectively, these strategies are designed to enhance training and inference efficiency, extend the effective context length, and facilitate an unbounded, real-time data stream.

##### 8. Acknowledgement

This research is supported by Hong Kong Research Grants Council Early Career Scheme (No. 22200824).

##### References

- [1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 5
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5, 7, 2, 3, 8
- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021. 1
- [5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR,

2015. 2, 5

- [6] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024. 3, 6, 7, 8
- [7] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. NeurIPS, 2024. 3
- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv:2412.05271, 2024. 3
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 2
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 6, 7
- [11] Shangzhe Di and Weidi Xie. Grounded question-answering in long egocentric videos. In CVPR, 2024. 5
- [12] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen,

- Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv:2405.21075, 2024. 6
- [13] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. 3
- [14] Shenghao Fu, Qize Yang, Yuan-Ming Li, Yi-Xing Peng, Kun-Yu Lin, Xihan Wei, Jian-Fang Hu, Xiaohua Xie, and Wei-Shi Zheng. Vispeak: Visual instruction feedback in streaming videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21778–21788,

2025. 6, 1, 3

- [15] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, 2017. 2
- [16] Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, et al. Arc-hunyuan-video-7b: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939, 2025. 5, 2
- [17] Mingfei Han, Linjie Yang, Xiaojun Chang, and Heng Wang. Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos. arXiv:2312.10300, 2023. 2
- [18] Gabriel Huang, Bo Pang, Zhenhai Zhu, Clara Rivera, and Radu Soricut. Multimodal pretraining for dense video captioning. arXiv:2011.11760, 2020. 2
- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv:2410.21276, 2024. 7, 3, 8
- [20] Asif Ali Laghari, Sana Shahid, Rahul Yadav, Shahid Karim, Awais Khan, Hang Li, and Yin Shoulin. The state of art and review on video streaming. Journal of High Speed Networks, 29(3):211–236, 2023. 1
- [21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6, 7
- [22] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024. 6
- [23] Yifei Li, Junbo Niu, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? arXiv:2501.05510, 2025. 3, 6
- [24] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv:2311.10122, 2023. 1
- [25] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 4
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023. 2

- [27] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv:2403.00476, 2024. 6
- [28] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen. Et bench: Towards open-ended eventlevel video-language understanding. Advances in Neural Information Processing Systems, 37:32076–32110, 2024. 6, 7
- [29] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: Ondemand spatial-temporal understanding at arbitrary resolution. arXiv:2409.12961, 2024. 2
- [30] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model. arXiv preprint arXiv:2502.04328, 2025. 3
- [31] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv:2406.09418, 2024. 2
- [32] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23023–23033, 2023. 5
- [33] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. arXiv:2501.03218, 2025. 1, 3, 6, 7, 8
- [34] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 6, 7
- [35] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In CVPR, 2019. 5, 2
- [36] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024. 7, 3, 8
- [37] Qwen Team. Qwen3 technical report, 2025. 6, 1, 8
- [38] V Team, Wenyi Hong, Wenmeng Yu, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025. 5
- [39] Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your offline video large language model into a proactive streaming assistant. arXiv preprint arXiv:2505.05467, 2025. 1, 3
- [40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin

- Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6, 7
- [41] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025. 5
- [42] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 1
- [43] Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Jiansheng Wei, Huishuai Zhang, and Dongyan Zhao. Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format. arXiv:2411.17991, 2024. 1
- [44] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 2
- [45] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. NeurIPS, 2024. 6
- [46] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. Streaming video understanding and multi-round interaction with memoryenhanced knowledge. arXiv:2501.13468, 2025. 3
- [47] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 3, 7, 8, 2
- [48] Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025. 3
- [49] Zhenyu Yang, Yuhang Hu, Zemin Du, Dizhan Xue, Shengsheng Qian, Jiahong Wu, Fan Yang, Weiming Dong, and Changsheng Xu. Svbench: A benchmark with temporal multi-turn dialogues for streaming video understanding. arXiv:2502.10810, 2025. 3
- [50] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025. 5
- [51] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv:2306.02858, 2023. 1
- [52] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv:2406.08085, 2024. 6, 7, 8, 3
- [53] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with

- synthetic data. arXiv preprint arXiv:2410.02713, 2024. 5, 6, 7
- [54] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In ICCV,

2019. 5

- [55] Yucheng Zhao, Chong Luo, Chuanxin Tang, Dongdong Chen, Noel Codella, and Zheng-Jun Zha. Streaming video model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14602–14612,

2023. 1

- [56] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023. 2
- [57] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In AAAI, 2018. 5, 2
- [58] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 6, 1, 8

## Streaming Video Instruction Tuning Supplementary Material

##### A. Streamo

###### A.1. System Prompt

We design a dedicated system prompt for Streamo that enables the model to handle dynamic streaming video content, interpret three predefined response states, and make real-time decisions at the frame level. The full prompt is provided in Tab. 10. This deliberately crafted prompt helps the model quickly adapt to the streaming input pattern and perform the required behavior transformation.

###### A.2. Instruction Prompt

In Tab. 14, we present the prompt templates used for all tasks. These diverse task instructions help the model better understand different task requirements, thereby fostering more general multi-task instruction-following capabilities. This goes beyond prior setups where models were confined to standalone QA, and represents a step toward general realtime interactive AI.

###### A.3. More Experimental Results

Our training framework converts offline models into streaming-capable models with minimal intrusive modifications, enabling these base models to process streaming video data. This design yields strong compatibility and allows direct application to a wide range of offline models. In Tab.11 and12, we further report results using InternVL3[58] and Qwen3VL[37] as Streamo’s base models. These results show that our framework effectively leverages the capabilities of offline models and extends them to online streaming video processing. This is particularly advantageous given the rapid iteration of offline models, as our framework can readily harness their improvements for realtime interactive video understanding.

Meanwhile, we also evaluated Streamo on ViSpeakBench [14], shown in Tab. 8. The results show that our method achieves a clear advantage in response-time accuracy, demonstrating the effectiveness and soundness of our response architecture.

###### A.4. Visualization

In Fig.6 and7, we visualize the outputs of Streamo, which vividly illustrate its ability to interpret and appropriately respond even to instructions that were unseen during training. When confronted with task instructions that vary in both response granularity and content, the model consistently produces suitable outputs. These visualizations provide strong evidence that Streamo’s training framework successfully bridges the gap between offline model capabilities and the

- Table 6. Comparison of Existing Video Benchmarks. StreamoBench introduces the first mixed-task type specifically designed for streaming video.

Benchmark #Videos #Samples Streaming Task Type

MVBench 3,673 4,000 QA TempCompass 410 7,540 QA ET-Bench 7,002 7,289 Mix SVBench 1,353 49,979 QA StreamBench 306 1,800 QA OVOBench 644 2,814 QA Streamo-Bench 300 3000 Mix

requirements of online streaming interactions, enabling reliable real-time responses that go far beyond simple QA.

A.5. Further Analysis of the Three-State Design

To examine the rationale behind our training architecture more thoroughly, we compare the proposed Three-state Design with an alternative approach based on the [EOS] token. As shown in Tab. 7, the [EOS]-based model exhibits notable performance drops, particularly on proactive tasks (i.e., FAR) and grounding tasks. These results demonstrate that our three-state design consistently outperforms EOSonly training while introducing only negligible additional cost.

We attribute this gap to the fact that [EOS] maps both irrelevant and partially relevant segments to the same token. As a result, the model is encouraged to remain silent even when encountering relevant frames, causing it to miss the optimal timing for response. In contrast, the introduction of a [Standby] token alleviates this misalignment by explicitly marking relevant frames as soon as the event begins and preserving this state throughout the relevant interval. This leads to more accurate temporal alignment and more complete coverage, which is reflected in the higher grounding TIoU.

- Table 7. Comparison on the same training dataset, StreamoInstruct, where the only change is replacing the proposed threestate design with EOS-only training. Using only [EOS] degrades performance, especially on proactive prediction (FAR) and forward grounding, highlighting the benefit of the three-state design.

|Model<br><br>|OVOBench<br><br>|Streamo-Bench|
|---|---|---|
| |RTVP BT FAR AVG<br><br>|Forward Grounding|
|Streamo-3B Streamo-3B w/ EOS<br><br>|61.51 41.76 53.72 52.33 60.93 39.43 45.22 48.52|14.7 9.3<br><br>|

- A key advantage of [Standby] is that it explicitly mod-

els frames that are already relevant but not yet ready for a final response. As shown in Fig. 4, because the query is specifically about ASWIN, the model switches to [Standby] once ASWIN appears and the attempt becomes temporally relevant, even though the final outcome is still uncertain. This allows the model to preserve attention over the ongoing event instead of treating these frames as irrelevant. Meanwhile, for grounding, the continuous [Standby] state helps cover the full event span more completely, rather than activating only near the final decisive moment.

- B. Streamo-Instruct

B.1. Data Generation Prompt

We next elaborate on the prompts used in our data annotation pipeline. For event caption tasks, we leverage ARCHunyuan[16], which is specifically trained for video segmentation and grounding, and directly adopt its official prompt for initial data processing. We then use the prompt in Tab.13 to rewrite and clean the annotated caption sentences. For narration generation, which describes interframe temporal changes, the generation prompt is given in Tab.15, and the prompt for merging and cleaning the resulting descriptions is provided in Tab.16. For the TSQA task, the detailed prompt is presented in Tab. 17.

C. Streamo-Bench

In Tab. 6, we compare our proposed Streamo-Bench with existing video benchmarks. Streamo-Bench is, to the best of our knowledge, the first streaming video benchmark that integrates multiple task types. Existing streaming video benchmarks typically use QA as the sole evaluation task, which mainly measures perceptual understanding rather than the ability to perform diverse open-ended tasks. However, the ability to follow varied instructions and complete multiple tasks is a key requirement for streaming video models. By filling this gap, Streamo-Bench enables more comprehensive evaluation of a model’s instructionfollowing ability in open-ended streaming scenarios.

- C.1. Statistics

Our benchmark contains 300 videos sampled from COIN[35], YouCookv2[57], and ActivityNet [5]. Each video is annotated with multiple tasks, including Grounding, Narration, Caption, and Time-Sensitive QA, yielding a total of 3,000 task-specific instances. Each video in Streamo-Bench contains 2x grounding (forward + backward) tasks, 1x dense caption task, and 1x narration task, with the rest being TSQA. This comprehensive design enables a thorough examination of a model’s ability to process and respond to diverse instructions in streaming settings.

###### C.2. Metric

To comprehensively evaluate the performance of models on our Streamo-Bench, we detail the metrics used for each task type below.

Grounding Evaluation. For grounding tasks, we distinguish between forward (queries referring to time points before an event) and backward (queries referring to time points after an event) contexts. Performance is measured using mean Intersection over Union (mIoU), which quantifies the overlap between the model’s predicted temporal interval and the ground-truth interval.

Let the predicted and ground-truth temporal intervals, tpred and tgt, for sample i be:

tpredi = [spredi , epredi ], tgti = [sgti , egti ], (8)

where s and e represent the start and end timestamps, respectively. The IoU for sample i is defined as the ratio of intersection length to union length:

max 0, min(epredi ,egti ) − max(spredi ,sgti ) max(epredi ,egti ) − min(spredi ,sgti )

. (9)

IoUi =

The mean IoU (mIoU) over N samples is

1 N

mIoU =

N

IoUi. (10)

i=1

Narration and Caption Evaluation. Because narration and captioning are open-ended generation tasks, directly evaluating output quality is challenging. Following the evaluation protocol of Chatbot Arena [56] and StreamingVLM [47], we assess narration and caption quality via pairwise comparison against a strong baseline, Qwen2.5VL-72B [3]. The win rate is defined as the proportion of cases in which our model’s output is judged superior to the baseline’s output.

Time-Sensitive QA Evaluation. For Time-Sensitive QA, we require that a prediction be correct in both its content and its timestamp. Let Q be the set of TSQA questions. For each question q ∈ Q, the ground truth consists of mq time-stamped answers:

Gq = {(aqi,tqi)}mi=1q , (11)

where aqi is the answer content and tqi is its timestamp. The model produces nq predictions:

Pq = {(ˆaqj,tˆqj)}nj=1q , (12)

where aˆqj is the predicted content and tˆqj is the predicted timestamp.

A predicted pair (ˆaqj,tˆqj) may match a ground-truth pair (aqi,tqi) only if it is correct in both content and time. For the

Table 8. Performance of streamo compared to various MLLMs on ViSpeak-Bench.

|Method Params Frames Omni Streaming|Time Accuracy (%) AW VI HR VW VT GU All<br><br>|Text Score VR AW VI HR VW VT GU All<br><br>|Overall|
|---|---|---|---|
|Human (Avg) - - - Human (Max) - - - -<br><br>|70.00 100.00 90.00 92.00 96.00 98.80 91.13 70.00 100.00 100.00 100.00 100.00 100.00 95.00|4.80 2.45 4.58 3.06 5.00 5.00 2.85 3.96<br><br>5.00 2.71 5.00 3.62 5.00 5.00 3.19 4.22<br><br><br>|3.69 4.01|

Proprietary MLLMs

Gemini 1.5 pro [36] - - ✓ ✗ 46.00 60.00 85.00 84.00 48.00 97.00 70.00 3.03 2.34 2.93 1.36 4.66 4.68 2.07 3.01 2.19 GPT-4o [19] - - ✓ ✗ 48.50 82.00 96.00 99.00 100.00 99.50 87.50 3.18 2.27 3.53 1.71 5.00 4.98 2.22 3.27 2.99

###### Open-Source Video MLLMs

InternVL-2.5 [8] 8B 16 ✗ ✗ 41.50 55.50 46.00 96.00 72.00 99.50 68.42 2.93 2.16 3.67 0.74 3.05 4.81 1.26 2.66 1.98 Qwen2.5-VL [3] 7B 1 fps ✗ ✗ 42.50 78.00 31.00 95.00 85.00 98.50 71.67 2.34 2.31 2.31 1.32 5.00 3.91 1.02 2.60 2.25 Qwen2.5-VL [3] 72B 1 fps ✗ ✗ 44.50 81.00 77.00 91.00 91.00 93.00 79.58 3.15 2.64 3.36 1.00 5.00 5.00 1.50 3.09 2.62 VITA 1.5 [13] 7B 1 fps ✓ ✗ 18.00 46.00 40.00 88.00 49.00 97.50 56.42 2.40 2.08 0.57 0.85 4.57 4.49 1.18 2.31 1.54 Ola [30] 7B 1 fps ✓ ✗ 27.00 67.00 44.00 89.00 69.00 98.50 65.75 2.95 1.81 2.67 0.55 4.71 3.67 1.52 2.55 1.86 FlashVstream [52] 7B 1 fps ✗ ✓ 34.00 16.00 48.00 75.00 33.00 99.50 50.92 1.75 1.63 1.31 0.67 4.88 4.61 0.70 2.22 1.24 Dispider [33] 7B 16 ✗ ✓ 38.50 70.00 44.00 69.00 100.00 99.50 70.17 2.50 1.75 4.06 0.91 0.61 2.49 2.07 2.06 1.63 ViSpeak [14] 7B 1 fps ✓ ✓ 56.50 72.00 83.00 93.00 79.00 99.00 80.42 3.75 2.63 3.84 1.07 4.95 3.15 3.36 3.25 2.76 Streamo 7B 1 fps ✗ ✓ 59.00 79.00 82.00 97.00 86.00 100 83.83 2.73 2.31 3.62 1.33 4.96 3.62 2.97 3.08 2.71

content evaluation:

C(ˆaqj,aqi) =

1, if content matches, 0, otherwise.

(13)

For the timestamp, we define a non-negative tolerance parameter δt ≥ 0. Then we evaluate the correctness of the timestamp by:

T(tˆqj,tqi;δt) =

1, if |tˆqj − tqi| ≤ δt, 0, otherwise.

(14)

In our experimental setting, the δt is set to 3 seconds. For the i-th answer point of question q, we define an indicator Iiq that checks whether there exists at least one prediction satisfying both content and temporal constraints: :

Iiq =

1 if C(ˆaqj,aqi) = 1 ∧ T(tˆqj,tqi;δt) = 1 0 otherwise

(15)

The final accuracy and recall can be given as:

mq

1 q∈Q mq q∈Q

Iiq (16)

Accuracy =

i=1

mq

1 |Q| q∈Q

1 mq

Iiq (17)

Recall =

i=1

###### C.3. Sample Visualization

A sample instance from Streamo-Bench is illustrated in Fig. 5. Forward and backward grounding questions are randomly placed either before or after their corresponding target temporal intervals. The TSQA question is inserted before the first answer timestamp. Narration and event caption instructions are placed before the start of the video stream to capture the overall video content.

###### C.4. Further Analysis

We further analyze the performance of existing models on Streamo-Bench and observe that their primary failures stem from a lack of instruction–task comprehension: they struggle to distinguish different task types and to produce taskappropriate outputs. This limitation arises because these models are typically trained exclusively on captioning or QA data, which constrains them to generate outputs tailored to only those specific tasks.

Examples in Tab. 9 clearly illustrate this phenomenon: while the models can satisfy caption or narration requirements, they often fail to understand grounding instructions and instead fall back to generic video descriptions. For TSQA tasks, although models trained on QA data can answer content-related questions, they do not properly follow instructions that require real-time updates to answers over the video timeline, leading to task failure.

In summary, existing models generally lack robust multitask understanding, whereas Streamo-Bench is specifically designed to evaluate a model’s ability to interpret and respond to task-specific instructions in streaming scenarios.

Did ASWIN successfully make the jump attempt?

Temporally localize the event: ’Aswin from Kerala failed to clear the 4.40-meter bar in the pole vault competition’. Respond once it has finished and summarize its time period.

[Figure 21]

[Figure 22]

[Figure 23]

<Silence> <Silence> <Silence> <Silence>

<Silence> <Silence>

[Figure 24]

[Figure 25]

[Figure 26]

<Silence> <Silence> <Silence> <Silence> <Standby> <Standby>

[Figure 27]

[Figure 28]

[Figure 29]

<Standby> <Standby> <Standby> <Standby> <Standby> <Standby>

[Figure 30]

[Figure 31]

[Figure 32]

<Standby> <Standby>

<Response> No, he failed. <Response> Given event occurred between 20s to 43s.

<Standby> <Silence>

Figure 4. Visualization of the three-state decoding process. The model stays in [Silence] for irrelevant frames, switches to [Standby] once the query-relevant event involving ASWIN begins, and emits [Response] only after the outcome becomes clear. For the grounding task, the persistent [Standby] state helps preserve attention over the relevant interval and enables more complete temporal coverage of the event span.

###### Time-Sensitive QA Task

What action is being performed on the tree? A. Measuring B. Watering C. Mulching Please provide your answer by stating the letter followed by the full option. Answer again if it changes.

What is being done to the soil around the tree?

A. backfilled B. marked C. dug up Please provide your answer by stating the letter followed by the full option. Update when new info comes.

Question Time Question Time

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

A. Measuring

Localize this event: ‘Take the tree out of the pot

and measure its depth.’ in the following video.

Event Grounding Task(Forward)

Question Time

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

B. marked

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

What is the temporal window for the event:‘Take the sapling out of the pot and measure its depth.’

The event occurs during 48s – 66s.

Event Grounding Task(Backward)

Question Time

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

C. dug up B. Watering A. backfilled C. Mulching

Figure 5. Streamo-Bench example illustrating multi-task instruction-following evaluation.

Input Video

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Prompt Model Output

Real-Time Narration Task: Watch the following video and provide a real-time narration.

Here’s a typical mid-late 18th Century stoneware tea pot with the rounded spout and finial for pouring. After pouring when you were pouring out the last bit of liquid from the pot, you would be careful to stir away any bits of leaf which were floating around in the bottom.

Event Caption Task: List every event happening in the video.

We’ll go through all the steps to make tea the way the Victorians did step by step. When you think of what they would be using here in their kitchen spaces, you can see they’ve got the teapot, and she’s working on getting her leaves in. The method for both is basically exactly the same. With the fresh leaf version in this case, we’re just adding in equal amounts of leaves and water.

Event Grounding Task: Localize this event: ‘add some water to the tea’ in the following video.

In the Victorian era the family kitchen was an even more important part of home life than in previous as this was where most of their time was spent. It was where the family ate their meals but also where they would come together. Whether it was afternoons at tea or a simple evening meal, the Victorian kitchen was a place that families would gather together and enjoy good company and food. Here we have made our own little tea service here, and I will be showing you how Mrs Bronte would serve it in her kitchen.

Time-Sensitive QA Task: What is the woman doing? A. Preparing tea in kitchen B. Sitting at desk writing C. Drinking tea at desk. Please provide your answer by stating the letter followed by the full option. If the correct answer later changes, update your response.

A. Preparing tea in kitchen. You will learn to make a cup of tea the Victorian way.” To begin you’ll need something sweet to top with your tea and milk. Tea is made with sugar or honey and the Victorians actually used a lot of cream for their tea. I’m going to use some milk for my tea today. And it can take up to two minutes to boil, so if you’re making tea for more than one person.”, Start with just one cup at a time and then add more when ready.

Table 9. Example output from StreamingVLM illustrating a failure to follow diverse task instructions and generate the corresponding response.

|System Prompt: You are a helpful assistant specializing in streaming video analysis. You will receive input frame by frame, each labeled with absolute time intervals in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:<br><br>1. Use </Silence> when:<br><br>- No relevant event has started, OR<br>- The current input is irrelevant to the given question.<br><br><br>2. Use </Standby> when:<br><br>- An event is in progress but has not yet completed, OR<br>- The current input is relevant but the question cannot yet be answered.<br><br><br>3. Use </Response> only when:<br><br><br>- An event has fully concluded, OR<br>- The available information is sufficient to fully answer the question.<br><br><br>Provide a complete description at this point. Do not provide partial answers or speculate beyond the given information. Whenever you deliver an answer, begin with </Response>.|
|---|

Table 10. System prompt used in Streamo.

- Table 11. Additional online benchmark evaluation results of Streamo framework with different base models (InternVL3 and Qwen3VL). Our framework consistently enables strong real-time streaming performance across diverse offline backbones.

Real-Time Visual Perception Backward Tracing Forward Active Responding Overall Avg. OCR ACR ATR STU FPD OJR Avg. EPM ASI HLD Avg. REC SSR CRR Avg. Overall Avg.

Model # Frames

###### Open-source Offline Models

Qwen2-VL-72B [40] 64 65.77 60.55 69.83 51.69 69.31 54.35 61.92 52.53 60.81 57.53 56.95 38.83 64.07 45 49.3 56.27 LLaVA-Video-7B [53] 64 69.13 58.72 68.83 49.44 74.26 59.78 63.52 56.23 57.43 7.53 40.4 34.1 69.95 60.42 54.82 52.91 LLaVA-OneVision-7B [21] 64 66.44 57.8 73.28 53.37 71.29 61.96 64.02 54.21 55.41 21.51 43.71 25.64 67.09 58.75 50.5 52.74 Qwen2-VL-7B [40] 64 60.4 50.46 56.03 47.19 66.34 55.43 55.98 47.81 35.48 56.08 46.46 31.66 65.82 48.75 48.74 50.39 InternVL-V2-8B [10] 64 67.11 60.55 63.79 46.07 68.32 56.52 60.39 48.15 57.43 24.73 43.44 26.5 59.14 54.14 46.6 50.15 LongVU-7B [34] 1fps 53.69 53.21 62.93 47.75 68.32 59.78 57.61 40.74 59.46 4.84 35.01 12.18 69.48 60.83 47.5 46.71

###### Open-source Online Models

Flash-VStream-7B [52] 1fps 24.16 29.36 28.45 33.71 25.74 28.8 28.37 39.06 37.16 5.91 27.38 8.02 67.25 60 45.09 33.61 VideoLLM-online-8B [6] 2fps 8.05 23.85 12.07 14.04 45.54 21.2 20.79 22.22 18.8 12.18 17.73 - - - - Dispider-7B [33] 1fps 57.72 49.54 62.07 44.94 61.39 51.63 54.55 48.48 55.41 4.3 36.06 18.05 37.36 48.75 34.72 41.78

###### Streamo Framework

- Streamo-3B (Qwem2.5-VL) 1fps 78.52 52.29 67.24 44.38 55.45 71.20 61.51 51.18 57.43 16.67 41.76 27.94 50.72 82.5 53.72 52.33 Streamo-7B (Qwem2.5-VL) 1fps 79.19 57.80 75.00 49.44 64.36 70.11 65.98 54.55 52.03 31.72 46.10 29.96 51.03 83.33 54.77 55.61 Streamo-2B (InternVL3) 1fps 77.18 55.96 62.07 41.01 60.40 70.11 61.12 48.82 47.30 13.44 36.52 29.23 47.38 80.42 52.34 49.99

- Streamo-4B (Qwen3-VL) 1fps 82.55 69.72 74.14 52.25 73.27 81.52 72.24 58.19 52.70 17.20 42.70 31.38 53.90 84.17 56.48 55.10

- Table 12. Additional offline benchmarks results of Streamo framework with different base models (InternVL3 and Qwen3VL). The results show that our training framework preserves the underlying offline capability while extending it to streaming video processing.

OVO Real-Time

OVO Backward

Model

MVBench TempCompass VideoMME LongVideoBench Avg

###### Proprietary Models

Gemini-1.5-pro [36] 69.3 62.5 60.5 67.1 75.0 64.0 66.4 GPT-4o [19] 64.5 60.8 64.6 70.9 71.9 66.7 66.6

###### Open-source Online Models

Flash-VStream-7B [52] 28.4 27.4 61.2 - 61.2 - VideoLLM-online-8B [6] 20.8 17.7 33.9 - 26.9 - Dispider-7B [33] 54.6 36.1 - - 57.2 - StreamingVLM-7B [47] 62.0 - 69.2 - 65.1 59.0 -

###### Streamo Framework

Qwen2.5-VL-3B [3] 54.6 37.8 67.0 64.4 61.5 54.2 56.6

- Streamo-3B 61.5 (+6.9) 41.8 (+4.0) 67.9 (+0.9) 66.2 (+1.8) 61.8 (+0.3) 56.2 (+2.0) 59.2 (+2.6) Qwen2.5-VL-7B [3] 58.8 42.2 69.6 71.7 65.1 56.0 60.6 Streamo-7B 66.0 (+7.2) 46.1 (+3.9) 72.3 (+2.7) 71.8 (+0.1) 67.9 (+2.8) 59.2 (+3.2) 63.9 (+3.3) InternVL3-2B [58] 59.5 36.4 70.4 57.6 58.9 55.4 56.4 Streamo-2B 61.1 (+1.6) 36.5 (+0.1) 71.4 (+1.0) 57.8 (+0.2) 60.1 (+1.2) 56.5 (+1.1) 57.3 (+0.9) Qwen3-VL-4B [37] 66.5 42.8 68.9 65.8 69.3 53.2 61.1

- Streamo-4B 72.2 (+5.7) 42.7 (-0.1) 70.4 (+1.5) 66.3 (+0.5) 68.7 (-0.6) 56.1 (+2.9) 62.8 (+1.7)

|Event Rewriting Prompt: You are given a set of video captions, each describing a specific moment in a video. For each caption, perform the following tasks:<br><br>1. Remove any transition words, discourse markers, or sequence indicators (e.g., ”Finally ”Then ”Next ”Afterwards ”At the beginning ”At the end ”The video ends with ”The scene starts with etc.) at the beginning of the sentence or within the sentence, as these captions are now independent and do not need such connectors or structural descriptions.<br>2. Rewrite the caption to make it more concise and clear, without changing its meaning or omitting any important information.<br>3. Preserve all factual details and key actions described in the original caption.<br>4. Do not add any extra interpretation, information, or imagination not present in the original sentence. Only use the information given.<br>5. If the sentence includes a phrase describing the position of a shot or the sequence within the video (such as ”The video ends with ”At the start of the video ”In the next scene ”The video conclude with”), remove this part entirely. Focus only on describing the content of the shot.<br><br><br>Example: Original: ”Finally, the video cuts back to the man in the indoor setting, who concludes the presentation by holding the bow.” Optimized: ”The man in the indoor setting concludes the presentation by holding the bow.”<br><br>Process each caption in this way. Return the optimized sentence directly. Original:{sentences} Optimized:|
|---|

Table 13. Task prompt used for rewriting event caption.

|Real-time Narration Task:<br><br>- Provide a continuous, time-synchronized narration of the video, describing actions, objects, and scene changes as they occur.<br>- Narrate the video in real time, updating the description frame-by-frame or moment-by-moment as events unfold.<br>- Generate live commentary of the video, focusing on who is doing what, where, and when, and noting any transitions or new events immediately.<br>- Deliver an on-the-fly description of the video, highlighting salient actions, interactions, and changes in context as soon as they appear.<br>- Produce a running narration that captures ongoing activities, brief pauses, and resumptions, maintaining temporal alignment with the video timeline. Action Caption:<br>- Find, identify, and determine the temporal boundaries of a series of distinct actions or steps occurring throughout the video.<br>- Locate and describe a series of actions or steps in the video.<br>- Locate and pinpoint a sequential series of specific actions or steps in the video.<br>- Identify and mark the video segments corresponding to a series of actions or steps.<br>- Identify and localize a series of steps or actions occurring in the video.<br><br><br>Event Caption:<br><br>- Identify and describe all events in the following video.<br>- List every event happening in the following video with descriptions.<br>- Detect and summarize each event sequence in the following video.<br>- Extract and explain all notable events in the following video.<br>- Find all significant events in the following video and describe them. Event Grounding:<br>- Watch the following video and temporally localize the event. Respond once it has finished and summarize its time period. The given event is: ’{caption}’<br>- Monitor the following video, identify the event, then respond after it finishes with a summary of its time window. The given event is: ’{caption}’<br>- Analyze the following video, detect the event and report back upon its completion with its time period. The given event is: ’{caption}’<br>- Review the following video, localize the event in time, then notify me once it ends and summarize the interval it occupies. The given event is: ’{caption}’<br>- Identify and temporally segment the event in the following video. Report after it finishes with its time period and duration. The given event is: ’{caption}’ Time-sensitive QA:<br>- {question} If the answer changes over time, update your response accordingly.<br>- {question} Update your answer if it becomes different at a later time.<br>- {question} If it later differs, update your response promptly.<br>- {question} Refresh your answer upon any change.<br>- {question} If the correct answer later changes, update your response.<br>|
|---|

###### Table 14. Prompt template used for diverse streaming video tasks.

|Video Description Prompt: You are given two consecutive seconds in a video (2 frames per second). Please succinctly describe the most significant operation or change that occurred between these seconds, focusing on the following points:<br><br>1. Base your description solely on clearly observable information; avoid speculation or assumptions.<br>2. For each object or element that changed, briefly state what changed: position, movement, actions, shape, color, etc.<br>3. Only describe the main operation, event, or action that happened—avoid listing small movements or minor shifts.<br>4. Describe only the specific changed parts with clear and direct language; do not include unchanged content or summarize the overall scene.<br>5. Make your description short and focused, naming only the changes without referencing the sequence of frames or including explanations.<br><br><br>Example: ’A woman appears.’ ’You pick up a scissor.’ ’The cup moves to the left.’ ’A cat enters the frame.’ ’The red ball rolls closer.’ ’The lamp turns on.’ ’The book closes.’ ’A hand takes the remote.’ ’The door opens further.’<br><br>Only provide the most important description or a summary of multiple descriptions.|
|---|

Table 15. Task prompt used for frame-level video description generation.

|Narration Generation Prompt:<br><br>**Objective**: Clean the following second-by-second video descriptions to enhance coherence and eliminate redundancy. The original descriptions were generated with visibility of only the preceding and following 2 seconds, making them repetitive and disjointed.<br><br>**Task**: Transform the descriptions into a smooth, logical narrative by:<br><br>1. Removing Redundancy: Omit repeated descriptions of static or ongoing actions.<br>2. Filtering Insignificant Details: Exclude minor or fleeting actions that do not impact overall understanding.<br>3. Sentence Shortening: If a description significantly exceeds 5 words, rewrite it to approximately 5 words while preserving the main idea.<br>4. Merging Consecutive Events: Combine adjacent descriptions representing a continuous or complete action into a single, concise sentence (e.g., “002: Man touches socket” and “003: Socket disappears” → “003: Man removed socket”).<br><br><br>**Output Format and Rules**:<br><br>1. Use the format: SSS: one-sentence description.<br>2. When merging or omitting descriptions, skip the corresponding timestamps.<br>3. Do not add explanations, notes, or blank lines.<br>4. If the descriptions are repetitive, monotonous, lack meaningful variation, or are confusing, ambiguous, or insufficient, output only: Negative Sample.<br><br><br>Description: {Description}|
|---|

Table 16. Task prompt used for merging the frame description to generate real-time narration.

|TSQA Generation Prompt: You are a Time-Sensitive Video Question Generator. You need to identify all the elements in the video that change over time and formulate them into questions.<br><br>**CORE REQUIREMENT** Every question MUST have answers that CHANGE over time. If something doesn’t change during the video, DO NOT create a question about it.<br><br>**TASK**<br><br>1. Identify ONLY aspects that visibly CHANGE during the video. Ignore:<br><br>- Static elements that remain constant<br>- Transitions, previews, close-ups that don’t alter facts<br>- Opening/closing sequences<br><br><br>2. For each changing aspect, generate ONE question with MULTIPLE DIFFERENT answers:<br><br>- Each question MUST have at least 2 DISTINCT answer values<br>- Answers must represent actual changes observed at different times<br>- Never repeat the same answer value<br><br><br>3. Question types:<br><br>- **Descriptive**: What/Which/Who (e.g., ”What color is the ball?”)<br>- **Counting**: How many/How much (e.g., ”How many people are visible?”)<br>- **State**: What stage (e.g., ”What is the person doing?”)<br>- **Action** : What is being added/used (e.g., ”What ingredient is being added?”)<br>- **Binary**: Yes/No (e.g., ”Is the bacon cooked?”)<br><br><br>4. Answer format:<br><br><br>- List answers chronologically<br>- Include PRECISE time in seconds for each observed change<br>- If state returns to a previous value, include it as a new entry<br><br><br>**EXAMPLES** [{”question”: ”What color is the traffic light? ”answers”: [{”value”: ”red ”time”: 3.8}, {”value”: ”green ”time”: 8.7}, {”value”: ”yellow ”time”: 23.2}, {”value”: ”red ”time”: 26.4}]}, {”question”: ”How many people are in the frame? ”answers”: [{”value”: 1, ”time”: 0.0}, {”value”: 2, ”time”: 3.8}, {”value”: 3, ”time”: 17.1}, {”value”: 1, ”time”: 42.6}]}, {”question”: ”What is being poured into the glass? ”answers”: [{”value”: ”water ”time”: 2.3}, {”value”: ”milk ”time”: 19.7}, {”value”: ”orange juice ”time”: 31.4}]}, {”question”: ”Is the cake fully decorated? ”answers”: [{”value”: ”No ”time”: 13.7}, {”value”: ”Yes ”time”: 48.9}]}]<br><br>**OUTPUT**<br><br>- Minimum 2 distinct answers per question<br>- Precise timing for all changes. Times must reflect ACTUAL observed changes, not approxima-<br><br><br>tions<br><br>- If unsure of exact timing, watch that segment again Think step-by-step and ensure all requirements are met and all time are precise.|
|---|

###### Table 17. Task prompt used for Time-Sensitive QA generation.

### Instruction Video & Response

[Figure 62]

[Figure 63]

What is the person doing with the wallpaper?

A. smoothing B. applying adhesive C. measuring D. hanging on wall E. trimming Please provide your answer by stating the letter followed by the full option and update your answer when it changed.

<Response> Text changes to "How to hang wallpaper"

<Response> A title card reads 'How to hang wallpaper'

<Response> Tools and materials appear. <Response> A ladder appears on the right.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

<Response> The necessary tools and materials for wallpapering are laid out on a table, including a bucket, a paint roller, a ladder, and a roll of wallpaper.

- <Response> A man kneels down. <Response> The man stands up.

<Response> The ruler is replaced by a level. <Response> Powder is poured into the bucket.

<Response> The container is removed from the bucket.

<Response> The wallpaper is smoothed out.

List every event happening in the video.

Provide a continuous narration of the video.

<Response> C. measuring

- <Response> B. applying adhesive

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

<Response> A man in a yellow shirt kneels on the floor and marks the wallpaper.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

<Response> He uses a ruler and pencil to measure and mark the wall where the wallpaper will be hung.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

<Response> D. hanging on wall <Response> A. smoothing

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

<Response> He prepares the wallpaper by pouring powder from a container into a bucket and using a brush to apply the mixed water to the back of the wallpaper roll.

Figure 6. Visualization of Streamo’s outputs. We present the results of Streamo processing the same video under different task instructions and distinguish them using different colors, including TSQA, Narration, and Caption. Arrows indicate the frames corresponding to the response moment.

<Response> The man moves closer to the window.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

<Response> He climbs onto a wooden ladder and carefully places the wet wallpaper onto the wall, smoothing it down with his hands and a green sponge to ensure it adheres properly.

<Response> The man cuts the wallpaper. <Response> The man switches to a green tool.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

<Response> E. trimming

<Response> The man adjusts the wallpaper. <Response> A ruler is placed against the wallpaper.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

<Response> He measures the height of the window frame and cuts the excess wallpaper along the top edge with a utility knife.

<Response> Camera zoom out.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

<Response> He trims the bottom edge of the wallpaper with a utility knife to fit the window sill.

<Response> A desk and chair appear.

<Response> The scene changes to a yellow background with text.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

<Response> The logo for

<Response> A full view of the finished room shows the newly installed striped wallpaper on the wall next to the window, followed by a shot of the completed space with a desk, chair, and stool

"builders" appears on a

yellow background.

Figure 7. This is a continuation of the previous figure, showing the results for the same video.

