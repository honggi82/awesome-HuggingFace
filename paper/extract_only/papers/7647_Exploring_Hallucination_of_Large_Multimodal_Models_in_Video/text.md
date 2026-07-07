## Exploring Hallucination of Large Multimodal Models in Video Understanding: Benchmark, Analysis and Mitigation

# arXiv:2503.19622v1[cs.CV]25Mar2025

Hongcheng Gao∗1, Jiashu Qu∗2, Jingyi Tang∗1,3, Baolong Bi1, Yue Liu4, Hongyu Chen5 Li Liang†1,3, Li Su†1, Qingming Huang1,3 1University of Chinese Academy of Sciences 2University of Cincinnati 3Key Lab of Intell. Info. Process., Inst. of Comput. Tech., CAS 4National University of Singapore 5Beijing Jiaotong University

{gaohongcheng23,tjy23}@mails.ucas.ac.cn; quju@mail.uc.edu

### Abstract

The hallucination of large multimodal models (LMMs), providing responses that appear correct but are actually incorrect, limits their reliability and applicability. This paper aims to study the hallucination problem of LMMs in video modality, which is dynamic and more challenging compared to static modalities like image and text. From this motivation, we first present a comprehensive benchmark termed HAVEN for evaluating hallucinations of LMMs in video understanding tasks. It is built upon three dimensions, i.e., hallucination causes, hallucination aspects, and question formats, resulting in 6K questions. Then, we quantitatively study 7 influential factors on hallucinations, e.g., duration time of videos, model sizes, and model reasoning, via experiments of 16 LMMs on the presented benchmark. In addition, inspired by recent thinking models like OpenAI o1, we propose a video-thinking model to mitigate the hallucinations of LMMs via supervised reasoning fine-tuning (SRFT) and direct preference optimization (TDPO)—where SRFT enhances reasoning capabilities while TDPO reduces hallucinations in the thinking process. Extensive experiments and analyses demonstrate the effectiveness. Remarkably, it improves the baseline by 7.65% in accuracy on hallucination evaluation and reduces the bias score by 4.5%. The code and data are public at https://github.com/Hongcheng-Gao/HAVEN.

### 1. Introduction

In recent years, the success of Large Language Models (LLMs) [32, 33, 88, 89] has significantly advanced the development of Large Multimodal Models (LMMs)[1, 12, 57, 58, 67, 75, 106]. By integrating vision and audio modules with LLMs, LMMs are capable of generating accurate responses based on users’ textual prompts, visual inputs, and even au-

∗Equal contribution. †Correspondence to Li Su and Liang Li.

dio data [15, 20, 77, 78, 116]. However, LLMs are found to suffer from hallucinations—providing responses that appear correct but are actually incorrect—especially when faced with questions that conflict with their training data [43, 112]. Despite the advancements in LMMs, they also experience similar hallucination issues in both text and image understanding [13, 25, 30]. This phenomenon undermines user trust in LMMs and limits their applicability in high-stakes areas such as healthcare [55] and autonomous driving [87]. In response to these challenges, a series of studies have been proposed to evaluate and mitigate hallucinations in both LLMs and LMMs.

However, previous research on hallucinations of LMMs has primarily focused on image understanding [14, 26, 34, 37, 49, 54, 60, 62], as earlier LMMs could not process video inputs. These benchmarks are designed to evaluate hallucinations involving factors such as objects, relationships and attributes in a single image. With advancements in multimodal technologies, numerous LMMs now support video processing. Although many of these models did not incorporate audio inputs from videos, most can effectively process the visual content of video. Unlike image understanding, videos consist of sequences of multiple image frames over time, making video understanding more complex. It requires the analysis of continuous temporal dynamics, including sequential changes in human actions, object movements, and scene transitions. Hence, hallucinations in video understanding also differ from those in images.

To address the concern above, we first proposed a benchmark for HAllucination in Video UndErstaNding (HAVEN). HAVEN is meticulously designed to quantitatively evaluate the hallucination in video understanding for LMMs, which is constructed based on the following dimensions: (i) Three causes of hallucinations: conflict with prior knowledge, in-context conflict, and inherent capability deficiencies of LMMs. (ii) Three types of hallucination aspects in a video:

Hallucination Aspect

###### Hallucination Cause

Object Existance Relation Attributes ...

Scene Event

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Action Sequence Story ...

Location Season Time ...

Response Extract (optional)

Evaluated Model

###### Question

LLM Judge

[Figure 7]

[Figure 8]

[Figure 9]

Conflict with prior

2) Evaluation Process

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

What color is this cat?

Hallucination

[Figure 16]

1) Dataset

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

3) Metrics

In-context conflict

[Figure 26]

Calculate correct rate

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Consistancy

[Figure 33]

###### How many stars?

[Figure 34]

[Figure 35]

[Figure 36]

Is there a dog?

Binary-choice Multiple-choice Short-answer

[Figure 37]

[Figure 38]

[Figure 39]

What animal is it? A: Dog B: Cat

Rewrite to change answer and check for consistency

Capability deficiency

What animal is it?

Format

Figure 1. Construction protocol of HAVEN. The left section outlines the three dimensions of data construction and the associated categories within each, while the right section details the evaluation process and metrics.

object, scene, and event. (iii) Three formats of questions: binary-choice, multiple-choice, and short-answer.

Our evaluation of 16 existing LMMs reveals that ValleyEagle-7B and GPT4o-mini demonstrated the lowest hallucination rates among all models tested, while Qwen2.5-VL3B and Valley-Eagle-7B exhibited superior response consistency. The analysis further indicated that model performance is influenced by several factors: accuracy initially increases then decreases with longer video duration, decreases with more complex questions, improves with increased frame sampling, and generally shows reduced hallucinations and higher consistency with larger model sizes. Besides, chainof-thought reasoning can reduce hallucinations in LMMs.

In addition, we propose a thinking-based training strategy to reduce hallucinations by enhancing the LMM’s reasoning abilities. This training strategy is divided into two steps: supervised reasoning fine-tuning (SRFT) and thinking-based direct preference optimization (TDPO). In the SRFT stage, we perform supervised fine-tuning on the LMM using videos derived from images, incorporating long Chain-of-Thought answers distilled from image-thinking models like QVQ [85] and OpenAI o1 to equip the model with thinking capabilities. The TDPO stage directly optimizes the fine-grained thinking component at both the word and sentence levels, with fabricated reasoning receiving stronger feedback to ensure it remains factually grounded. Experimental results indicate that our training methodology yields substantial improvements in both reducing hallucinations and enhancing response consistency in LMM, as exemplified by LLaVANEXT-Video-DPO-7B.

### 2. Related Works

Large Multimodal Models (LMMs). Recent LMMs primarily focused on visual-language understanding, which can be categorized into image-language [1, 57, 58, 67] and videolanguage models [52, 63, 75, 110]. Early visual-language

models, designed for single-image inputs [31, 40, 46, 71], followed two main approaches: self-supervised learning with image-text pairs [31, 71], and adding image adapters to existing LLMs followed by alignment fine-tuning [40, 46]. Subsequent research extended to multi-image processing capabilities, naturally evolving into video-language models, which leverage frame-by-frame processing inherited from imagelanguage models to achieve video understanding through text-video alignment [6, 9, 51, 52, 63, 103, 110, 113]. VideoLLaVA [52], for example, incorporates a LanguageBind encoder to align visual features from both images and videos into a unified space for joint video-text encoding. Recent developments have further expanded to modalities including audio [9, 24] and 3D point clouds [104].

Hallucination Benchmarks. Hallucination is widespread in large models [32, 33, 88, 89], and the relevant benchmarks primarily target LLMs and VLMs. Hallucination benchmarks for LLMs[21, 43, 53, 65, 74, 96] mainly focus on identifying factual inaccuracies, deviations from original contexts and the self-awareness. For VLM, hallucination refers to the phenomenon that the generated text responses are inconsistent with the corresponding visual content [2]. Numerous benchmarks [4, 7, 11, 14, 26, 34, 35, 37, 49, 54, 60, 62, 81, 90, 91] have been developed to evaluate these hallucinations. Early benchmarks (e.g., CHAIR [76] and POPE [50]) primarily target object presence, while later ones like HallusionBench [16] and MMHal-Bench [81] also evaluate object relationships and scene understanding. Recent studies [11, 35, 60, 93] have begun addressing factual hallucinations in image-language models, yet most approaches focus on single images, neglecting continuous content. VideoHallucer [94] bridges this gap by introducing a video hallucination benchmark, though current methods still tend to consider only one hallucination dimension or treat multiple dimensions at the same level, leading to category overlaps.

0.14

0.0030

- 0.00
- 0.01
- 0.02
- 0.03
- 0.04
- 0.05
- 0.06

0.12

0.0025

0.10

0.0020

Frequency

Frequency

Frequency

0.08

0.0015

0.06

0.0010

0.04

0.0005

0.02

0.0000

0.00

0 10 20 30 40 50 60 70+

0 200 400 600 800 1000 1200+

0 10 20 30 40

Duration (Seconds)

Frame Count

Number of Tokens

(a) Duration Time

(b) Frame Count

(c) Question Length

Figure 2. Distribution of duration time, frame count, and question length.

and 260 manually collected video clips from YouTube1.

Answer: A

31.8% No Answer

#### 3.2. Construction Protocol

10.9%

Answer: B

32.3%

As shown in Fig. 1, we first constructed a dataset based on three dimensions—hallucination causes, hallucination aspects, and question formats—before implementing an automated evaluation process along with two metrics.

Short Answer

89.1%

47.8%

17.4%

MC

SA

5.9%

No Answer

25.7%

T/F

Answer: C

###### 3.2.1. Hallucination Causes

34.7%

45.7%

10.1%

Answer: No

We categorize hallucinations in LMMs into three types based on their underlying causes: conflicts with prior knowledge, in-context conflicts, and inherent capability deficiencies.

No Answer

48.4%

Answer: Yes

Figure 3. Question format distribution. Percentage share of each format-binary-choice (T/F), multiple-choice (MC), and shortanswer (SA)—and the proportion occupied by the detailed answer.

Conflict with Prior. When the content extracted from the video contradicts the model’s inherent prior knowledge acquired during training, the model may rely on its prior knowledge learned from established facts or widely accepted knowledge, leading to incorrect responses that don’t align with the provided information.

Hallucination Mitigation. Approaches to mitigating hallucinations can be broadly classified into two categories based on the stage at which they intervene. The first category deals with the training phase, which focuses on creating datasets specifically for hallucination-related tasks [17, 18, 56, 73, 79, 83, 107] and developing novel training objectives or methodologies aimed at mitigating hallucinations [5, 42, 80]. Although these methods have demonstrated effectiveness, they often require extensive training processes, making them both time-consuming and resource intensive. The second category focuses on the inference phase, usually involves CD-based [48] novel decoding strategies [8, 10, 38, 39, 61, 68, 69, 98, 115, 118]. Another approach is to detect potential hallucinations while generating and correct them [41, 66], among others [45, 97, 109, 111]. In addition, some researchers tackle hallucinations by manipulating attention weight assigned to the image [27, 47, 100, 102, 114, 117], and prompt-based methods [22, 70, 92, 94, 99, 105].

In-context Conflict. When task components conflict—such as discrepancies between the question and options or between the video and the questions/answers—valid answers cannot be derived from the given material. In such instances, the model should indicate uncertainty by responding with "I don’t know/no answers". However, models prone to hallucination often produce fabricated responses by focusing on limited contextual cues.

Capability Deficiencies. Most LMMs struggle with numerical tasks, often leading to miscounts or quantification errors in videos. For example, they might label a scene as having four cars when there are actually three, or miscount the number of people present. Such inaccuracies can undermine the reliability of video analysis by providing incorrect quantitative details.

### 3. HAVEN Benchmark

#### 3.1. Data Sources

Our video data comprises videos from three public video datasets (COIN [82], ActivityNet [3], and Sports1M [36])

###### 3.2.2. Hallucination Aspects

According to the components of a video, hallucination aspects can be categorized into three types: object, scene, and event hallucinations.

1https://www.youtube.com/

Object. Object hallucinations involve inaccuracies or fabrications related to the entities present in the video. This primarily includes the presence of objects, their inherent attributes such as size, color, and shape, as well as the emotions displayed by characters.

Scene. Scene hallucinations involve inaccuracies related to the overall setting and environment of the video. This includes errors concerning the background, location, season, time of day, lighting conditions, and the general context in which events take place. For example, a scene-level hallucination might mistakenly depict an indoor setting as outdoor, alter the time of day, or misrepresent environmental factors such as weather conditions.

Event. Event hallucinations involve distortions or false representations of the actions and occurrences within the video. This includes inaccuracies in the sequence of events, the nature of interactions between characters, and the specific actions taking place. For instance, an event-level hallucination might depict a character performing an action that does not actually occur in the video or misrepresent the timing and order of events.

###### 3.2.3. Question Types

We classify questions into three types: binary-choice(T/F), multiple-choice(MC), and short-answer(SA). Binary-choice questions are phrased as interrogative sentences (e.g., "Is there a dog?") and require a response of "yes" or "no." Multiple-choice questions present a set of candidate answers, while short-answer questions call for a concise response. If a question lacks a definitive answer, the model should respond with "no answer" or "I don’t know."

#### 3.3. Data Post-processing

To evaluate the inherent response biases of language models—namely their predisposition toward particular responses in binary-choice questions and multiple-choice questions—we implement a systematic question transformation protocol. For each binary question, we generate two complementary variants: one where "yes" is the correct answer and another where "no" is correct. Similarly, we transform each multiple-choice question into three parallel versions, with each version designating a different option (A, B, or C) as the correct answer. This approach aims to neutralize potential positional biases in model responses.

#### 3.4. Dataset Statistics

As shown in Table 1 and Fig. 3, our dataset comprises 6,497 questions. The three question formats are distributed approximately in a 5:2:3 ratio. Among the three types of hallucinations, the "prior conflict" category accounts for the largest share—around 75%—while the three aspects are proportioned roughly as 3:1:2.

Cause/Aspect Object Scene Event #Total Prior Conflict 2162 686 1763 4569

In-context Conflict 94 82 404 538

Capability 1107 121 78 1156 #Total 3363 889 2245 6497

Table 1. The Statistics of HAVEN

Figure 2 shows the distribution of our data in terms of duration, frame count, and question length. The duration spans from 0 to 70 seconds, with most instances falling within the 0–20 second interval. The frame count ranges from 0 to 1200 frames, with the majority concentrated between 0 and 400 frames. The question length is computed based on the token count obtained via GPT4 tokenizer, with a dominant range of 10–15 tokens.

- 3.5. Evaluation Metrics

Hallucination Evaluation. To evaluate hallucinations in LMMs, we measure the accuracy of their responses through LLM judging. Specifically, we employ GPT4o-mini [28] as the evaluation model because it is widely regarded as having human-like judgment capabilities. However, since some models may exhibit extended reasoning processes, an optional GPT4o-mini-based extraction step is performed to isolate the model’s answer before evaluation. Details can be found in Appendix A.

Consistency Evaluation. To quantify response consistency in LMMs, we introduce bias score to identify instances where a model provides inconsistent answers across different variants of the same base question. For binary-choice questions, the bias score is determined by the proportion of question pairs in which the model’s responses differ between the "yes"-correct and "no"-correct versions. Similarly, for multiple-choice questions, bias is measured by the frequency of inconsistent responses among the three variant forms of each base question. A lower bias score indicates more consistent reasoning across question variants.

- 4. Analysis of Video Hallucination

- 4.1. Settings

We evaluated 16 LMMs in total: one 3B model, ten 7B models, two 13B models, one 34B model, and one API model. To ensure a fair comparison, we maintained the original settings for all baselines, including frame counts and generation hyperparameters. Details of these LMMs can be found in Appendix B.

- 4.2. Hallucination Evaluation

Table 2 demonstrates a performance comparison of various LMMs in hallucination evaluation across different dimensions with accuracy. Higher scores represent lower levels

Prior In-context Capability

Model/Type

Total Object Scene Event Object Scene Event Object Scene Event

VideoChatGPT-7B[64] 34.78 43.88 38.00 17.02 13.41 17.57 32.52 47.11 20.51 34.69 Valley-Eagle-7B [101] 68.55 75.95 63.52 24.47 43.90 15.10 57.36 55.37 47.43 61.29 VideoLLaVA-7B [52] 47.55 57.29 50.59 14.89 12.19 10.15 35.68 33.88 34.61 43.73

VideoChat2-7B [44] 43.20 48.98 43.00 30.85 26.83 26.00 26.74 28.92 33.33 39.11 ShareGPT4Video [6] 51.48 62.24 48.27 17.02 19.51 8.91 43.00 38.84 32.05 46.28 LLaVA-v1.5-7B [59] 52.03 62.24 49.85 20.21 36.58 17.33 45.35 42.98 46.15 48.33 LLaMA-VID-7B [51] 48.75 56.71 50.09 20.21 21.95 21.78 37.76 42.98 29.49 45.31

LLaMA-VID-13B [51] 47.82 54.23 49.29 12.76 29.26 5.20 40.65 38.01 32.05 43.91 PLLaVA-7B [103] 44.26 60.93 41.41 20.21 37.80 9.90 43.63 15.70 32.05 40.17

PLLaVA-13B [103] 62.02 69.39 56.55 21.28 50.00 15.10 48.60 46.28 44.87 54.87 Qwen2.5-VL-3B-Instruct [86] 52.36 65.31 49.12 20.21 37.80 5.20 43.63 15.70 32.05 46.85 Qwen2.5-VL-7B-Instruct [86] 55.97 60.05 50.25 20.21 32.93 8.66 53.39 44.63 34.61 50.19

LLaVA-NeXT-Video-DPO-7B [113] 49.35 57.58 49.86 15.96 31.71 12.62 38.12 43.80 41.03 45.25 LLaVA-NeXT-Video-DPO-34B [113] 52.59 60.79 47.42 22.34 34.15 7.67 45.17 41.32 26.92 46.81

Video-LLaMA-2-13B [9] 28.95 40.96 30.80 11.70 9.76 12.62 16.71 21.49 26.92 26.97 GPT-4o-mini [28] 52.87 59.47 54.79 58.51 64.63 62.13 63.50 52.89 60.26 56.80

- Table 2. Hallucination Evaluation. Accuracy of LMMs on questions of different types of hallucination targets across three causes. The bold font indicates the best performance, and the underline mark indicates the second-best.

of hallucination. The evaluation results are categorized into three causes (prior conflict, in-context conflict, and capability deficiencies), with each category assessing three video aspects (object, scene, and event). Notably, Valley-Eagle-7B and GPT4o-mini emerged as the top performers, achieving total accuracy of 61.29% and 56.80% respectively. ValleyEagle-7B demonstrated superior performance across all subtypes of prior knowledge conflicts, while GPT4o-mini excelled in all in-context conflict scenarios. In contrast, VideoLLaMA-2-13B and VideoChatGPT-7B showed the weakest performance, particularly struggling with in-context conflict situations. We present some failure cases in Appendix C.1.

4.3. Consistency Evaluation

- Table 3 shows consistency evaluation with bias scores, where lower values indicate better performance. Qwen-VL-3B achieved the best overall score at 27.66%, with Valley-Eagle-

Model/Type Binary-choice Multiple-choice Total

VideoChatGPT-7B 57.58 39.19 46.97 Valley-Eagle-7B 42.41 17.99 28.31 VideoLLaVA-7B 59.08 43.62 50.16

VideoChat2-7B 52.43 49.75 50.88

ShareGPT4Video 60.11 20.90 37.48 LLaVA-v1.5-7B 55.90 35.38 44.06 LLaMA-VID-7B 52.15 40.20 45.25

LLaMA-VID-13B 59.83 30.75 43.05 PLLaVA-7B 61.14 42.51 50.38

PLLaVA-13B 52.62 24.02 36.11 Qwen2.5-VL-3B-Instruct 41.67 17.39 27.66 Qwen2.5-VL-7B-Instruct 48.50 19.90 31.99

LLaVA-NeXT-Video-DPO-7B 51.13 41.41 45.52 LLaVA-NeXT-Video-DPO-34B 49.06 18.90 31.65

Video-LLaMA-2-13B 66.48 24.32 42.14 GPT-4o-mini 34.08 29.95 31.69

Table 3. Consistence Evaluation

declines. This trend may be attributed to the fact that longer videos provide additional information; however, if the video is too long, the model may fail to sample the most relevant frames, leading to information loss. Conversely, as the length of the question text increases, the model’s performance consistently deteriorates, possibly because the surplus textual information diverts the model’s attention away from the video content.

- 7B coming in second at 28.31%. GPT-4o-mini demonstrated superior performance in binary-choice tasks with the lowest average bias score of 34.08%, while Qwen-VL-3B excelled in multiple-choice scenarios with a score of 17.39%. LLaVAv1.5-7B also exhibited noteworthy consistency while maintaining low hallucination rates. In contrast, VideoChat2-7B and PLLaVA-7B showed the weakest performance in consistency evaluations.

#### 4.5. On Sampling Number of Frames

As video information is frame-dependent, we examine the relationship between the number of sampled frames and model performance. We choose three models to demonstrate overall trends with mean values and variances. Fig. 6 shows how model hallucinations vary with increasing frame count. Performance improves with additional frames initially but deteriorates beyond a certain threshold, likely due to the growing disparity between training and inference frame counts.

#### 4.4. On Length of Videos and Questions

To analyze the relationship between LMM hallucination and various factors of input data, we conducted an analysis examining how accuracy varies with respect to video duration, frame count, and question length. As illustrated in Fig. 4, an increase in video duration or frame count initially enhances performance, but beyond a certain point, the performance

| |
|---|

1

- 0 5 10 15 20 25 30 Duration (Secounds)
- 1

| |
|---|

| |
|---|

1.0

0.8

0.8

0.8

Accuracy

0.6

Accuracy

0.6

Accuracy

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0

0

0.0

5 10 15 20 25 30 35 40

0 20 40 60 80 100 120 140

Number of Token

Frame Count

(c) Question Length

(a) Duration Time

(b) Frame Count

Figure 4. The impact of video duration, frame count, and question length on LLM hallucination.

|49|.1 58|.3 47|.9|
|---|---|---|---|
|19<br><br>39|.1 28<br><br>.5 36|.8 13<br><br>.8 34|.3<br><br>.2|
| | | | |

|48|.6 48|.2 22|.9|
|---|---|---|---|
|54<br><br>49|.0 58<br><br>.6 44|.7 31<br><br>.7 17|.0<br><br>.3|
| | | | |

|51|.2 57|.9 25|.4|
|---|---|---|---|
|52<br><br>45|.5 6.<br><br>.1 39|6 4.<br><br>.2 26|5<br><br>.3|
| | | | |

[Figure 40]

[Figure 41]

[Figure 42]

ObjectSceneEvent

PriorIn-contextCapability

PriorIn-contextCapability

50

50

50

40

40

Aspects

Causes

Causes

40

30

30

30

20

20

10

20

Object Scene Event

T/F MC SA

T/F MC SA

Aspects

Formats

Formats

(a) Causes-Aspects

(b) Formats-Aspects

(c) Formats-Causes

Figure 5. Accuracy heatmap along two dimensions.

[Figure 43]

Figure 6. Relationship between number of sampling frames and model performance.

#### 4.6. On Model Size

Fig. 7 illustrates how accuracy and bias scores change as the number of parameters increases across different model categories. We selected four distinct types of model across various sizes and presented the regression lines. Although a few LMMs don’t align perfectly with this trend, the overall evidence indicates that larger models tend to experience fewer hallucinations and exhibit lower levels of bias.

#### 4.7. On Co-impact of Each Two Dimensions

In Fig. 5, we visualize the co-impact of each pair of dimensions by calculating the average accuracy across all models for corresponding categories in each dimension pair. The results demonstrate that regardless of the classification method,

0.60

0.54

0.55

0.52

0.50

0.50

BiasScore

Accuracy

0.45

0.48

0.40

0.46

0.35

0.44

0.30

0.42

0.25

0.40

3B 7B 13B 34B

3B 7B 13B 34B

Model Size

Model Size

PLLaVA

LLaVA-NeXT-Video

Regression Std

PLLaVA

LLaVA-NeXT-Video

Regression Std

Qwen2.5-VL

LLaMA-VID

Qwen2.5-VL

LLaMA-VID

| |
|---|

| |
|---|

(a) Hallucination

(b) Consistency

Figure 7. Relationship between model size and performance.

hallucinations caused by in-context conflicts consistently show the most significant impact across all combinations. Additionally, short-answer questions consistently exhibit lower performance scores across all combinations.

#### 4.8. On Chain-of-Thought Reasoning

To explore whether enhanced reasoning capabilities could reduce model hallucinations, we conducted a comparative study of three LMMs, evaluating their performance before and after implementing Chain-of-Thought [95](CoT). As shown in Fig. 4, all LMMs demonstrated improved accuracy after incorporating CoT, indicating a reduction in hallucinations. Additionally, bias scores decreased for all LMMs except LLaMA-VID-7B, suggesting improved consistency.

[Figure 44]

Image Reasoning Problem

[Figure 45]

Who is picking up the phone?

[Figure 46]

- A: A dog
- B: A bird
- C: A person

[Figure 47]

Image Text

[Figure 48]

[Figure 49]

LMM-SRFT

[Figure 50]

Response of LMM-SRFT

Human Correction

[Figure 51]

Repeat

Segment-weightedDPO

[Figure 52]

The video depicts a cartoon dog sitting in front of a red telephone booth. The dog appears ...

The video depicts a cartoon dog sitting in front of a red house. The dog appears ...

[Figure 53]

[Figure 54]

[Figure 55]

...

o1-like VLM

- Step 1: Identify the main subject...
- Step 2: Determine the context...
- Step 3: Identify the action...After clear ring from the telephone...
- Step 4: Analyze the scene...The setting is outdoors, with grass and big trees in the background...
- Step 5: Consider the final answer... So the answer is: A. A dog.

- Step 1: Identify the main subject...
- Step 2: Determine the context...
- Step 3: Identify the action...After clear ring from the telephone...

- Step 4: Analyze the scene...The setting is outdoors, with grass and big trees in the background...

- Step 5: Consider the final answer... So the answer is: A. A dog.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Text Question

Long CoT Response

Video

Data Pair

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Modify/delete hallucination in thinking process.

SFT LMM LMM-SRFT

LMM-Thinking SRFT TDPO

Figure 8. Our thinking-based training strategy. Our training approach consists of two components: SRFT (for equipping the model with long reasoning capabilities) on the left, and TPDO (for preference learning to reduce hallucinations) on the right.

Model/Type Accuracy ↑ Bias Score ↓ PLLaVA-7B 40.17 50.38

+CoT 44.95 40.38 LLaMA-VID-7B 45.31 45.25

+CoT 46.05 50.20 LLaVA-NEXT-Video-7B 45.25 45.52

+CoT 49.56 45.15

- Table 4. Performance comparison of the model before and after incorporating Chain-of-Thought (CoT)
- 5. Mitigating Video Hallucination

The o1-like models [19, 29, 84] have demonstrated that long chain-of-thought (CoT) reasoning can significantly improve the capability and reliability of large models due to their thinking progress. Inspired by these o1-like models and the observation in Sec. 4.8, we apply supervised reasoning finetuning (SRFT) along with thinking-based direct preference optimization(TDPO) on our large video model, ultimately developing a video-thinking LMM that effectively mitigates video hallucination. The overview of our method is illustrated in Fig. 8.

#### 5.1. Supervised Reasoning Fine-tuning

To unlock the reasoning ability of LMMs, we proposed supervised reasoning fine-tuning (SRFT) on base model Mbase. This method consists two part: reasoning data synthesis and LoRA-based SFT.

Reasoning Data Synthesis. Due to the limitation that existing o1-like models either lack support for visual input or are restricted to processing a single image at a time, it is not feasible to directly employ such models for generating

video question-answer data. To address this limitation, we input both the question and image data into QVQ [85] to generate a reasoning response. Subsequently, by replicating the image to form a static video, we combine the question, the static video, and the generated reasoning response to create a training dataset. This dataset is designed to enhance the thinking capabilities of models with video understanding.

LoRA-based SFT. After constructing the reasoning training dataset D = {(vi,xi,yi)}Ni=1—where vi denotes the input video, xi corresponds to text inputs, and yi represents the target outputs—we perform supervised fine-tuning (SFT) with LoRA [23] on the base model Mbase that employs a weight matrix W ∈ Rd×k; since full-parameter fine-tuning may overwrite some prior knowledge and we only need to equip the model’s ability to generate long CoT style, we instead adopt a low-rank adaptation strategy by introducing the update ∆W = BA with B ∈ Rd×r and A ∈ Rr×k where r ≪ min(d,k), such that the adapted weight becomes W′ = W + αBA (with α as a scaling factor). It is formulated as follows:

LSFT(A, B) = −E(v,x,y)∼D log Pθ y v, x; W +αBA (1)

where Pθ denotes the model parameters. Through SRFT, we unlock the fundamental reasoning capabilities of the base model Mbase, resulting in LMM-SRFT, which acquires the ability to generate long CoT reasoning.

#### 5.2. Thinking-based DPO

Although SRFT endows the LMM with reasoning capability, the thought process still exhibits hallucinations. Inspired by RLHF-V [108], fine-grained preference learning is wellsuited for mitigating hallucinations in reasoning steps. This

Prior In-context Capability

Total Object Scene Event Object Scene Event Object Scene Event

Model/Type

PLLaVA 44.26 60.93 41.41 20.21 37.80 9.90 43.63 15.70 32.05 40.17

- +CoT 49.44 53.50 44.13 35.10 58.54 38.37 36.77 33.88 29.49 44.95

LLaMA-VID-7B 48.75 56.71 50.09 20.21 21.95 21.78 37.76 42.98 29.49 45.31

- +CoT 50.04 58.45 50.54 24.47 25.61 26.98 35.50 39.67 30.77 46.05

LLaVA-NeXT-Video-DPO-7B 51.48 62.24 48.27 17.02 19.51 8.91 43.00 38.84 32.05 45.25

+CoT 52.08 64.28 50.88 36.17 51.22 34.41 41.01 47.93 37.18 49.56 LLaVA-NeXT-Video-DPO-34B 52.59 60.79 47.42 22.34 34.15 7.67 45.17 41.32 26.92 46.81

GPT-4o-mini 52.87 59.47 54.79 58.51 64.63 62.13 63.50 52.89 60.26 56.80 LLaVA-NeXT-Video-7B-Thinking 58.28 69.97 53.37 39.36 56.10 40.84 39.38 37.19 34.61 52.90

Table 5. Hallucination evaluation of LLaVA-NeXT-Video-7B-Thinking comparing with other LMMs with/without CoT.

Model/Type Binary-choice Multiple-choice Total PLLaVA 61.14 42.51 50.38

+CoT 48.32 34.56 40.38 LLaMA-VID-7B 52.15 40.20 45.25

+CoT 65.70 38.86 50.20 LLaVA-NeXT-Video-DPO-7B 51.13 41.41 45.52

+CoT 56.84 36.58 45.15 LLaVA-NeXT-Video-DPO-34B 49.06 18.90 31.65

GPT-4o-mini 34.08 29.95 31.69 LLaVA-NeXT-Video-7B-Thinking 49.16 35.07 41.02

Table 6. Consistence evaluation of LLaVA-NeXT-Video-7BThinking comparing with other LMMs with/without CoT.

is because hallucinations in reasoning have two key characteristics: first, they manifest in the chain-of-thought as sporadic occurrences of specific words; and second, the thinking step naturally segments itself, which makes it straightforward to assign varying preference weights to different segments based on their degree of hallucination. Therefore, we propose TDPO, a weighted DPO method tailored for the thinking process. Specifically, TDPO is divided into two stages: constructing the fine-grained preference data and applying segment-weighted DPO.

Preference Data Construction. Given a video and a corresponding question, we initially employ the LMM to generate a response. Subsequently, a manual review is conducted to remove or modify any steps in the reasoning process that exhibit hallucinations. The original model-generated response is then treated as negative sample, while manually revised response serves as positive sample, thereby enabling the construction of the preference data.

Segment-weighted DPO. Refer to DDPO in RLHFV [108], we score the response as a weighted aggregation of the fine-grained thinking segments. Specifically, the likelihood of generating an output response y given an input question x and video v is defined as:

log π(y|x, v) = K

log p yi | x, v, y<i

yi∈yo

(2)

log p yi | x, v, y<i ,

+ γ

yi∈yh

where yo and yh respectively denote the token sets corresponding to the original and human-corrected segments. Unlike the standard DPO [72]—which uses the likelihood log π(y | x,v) = y

i∈y log p yi | x,v,y<i as action score—treats all segments equally, our segment-weighted DPO assigns greater weight to the corrected reasoning steps, thereby reinforces the human preference for responses that are factually grounded. Besides, the tradeoff hyperparameter γ can increase the emphasis on the corrected segments. As with DDPO, we also use a normalization constant, K = 1/(|yo| + γ|yh|), where |yo| and |yh| denote the lengths of the original and human corrected segments, respectively. This constant prevents longer responses from receiving disproportionately high scores. By applying segmentweighted likelihood within the DPO using our preference dataset, we can train LMM-SRFT to exhibit LMM-Thinking.

#### 5.3. Experiment Result

We conducted our thinking-based training on LLaVA-NeXTVideo-DPO-7B with about 5K synthetic data for SRFT and 3K preference data for TDPO. As shown in Table 5, the base model trained using our method—LLaVA-NeXT-Video-7BThinking—outperformed the original LLaVA-NeXT-Video7B by 7.65% and achieved a 3.34% improvement over the original model with CoT. Notably, it has even surpassed the performance of the 34B-parameter LLaVA-NeXT-VideoDPO. Table 6 shows the consistency evaluation of LLaVANeXT-Video-7B-Thinking alongside other LMMs. It demonstrates that LLaVA-NeXT-Video-7B-Thinking exhibits a higher consistency compared to the original LLaVA-NeXTVideo-DPO-7B—both with and without CoT—achieving bias score reductions of 4.5% and 4.13% respectively. We

present some cases of response from LLaVA-NeXT-Video7B-Thinking in Appendix C.2.

### 6. Conclusion

In this study, we presented HAVEN, a comprehensive benchmark specifically designed for evaluating hallucinations in video understanding for LMMs. The benchmark is built around three key dimensions and consists of 6,497 questions. Extensive evaluation across 16 models revealed insights into how factors such as video and question length, frame sampling and model size collectively impact model hallucinations. Furthermore, we proposed the SRFT and TDPO strategies for training video-thinking models. After applying to LLaVA-NeXT-Video-DPO-7B, our approach both improved its accuracy in hallucination evaluation and reduced its bias score in consistency evaluation. We will include more open-source and API LMMs in the future while also exploring automatic correction strategies to generate TDPO data instead of relying on human corrections.

### References

- [1] Anthropic. The claude 3 model family: Opus, sonnet, haiku.

2024. 1, 2

- [2] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 2
- [3] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 3
- [4] Sungguk Cha, Jusung Lee, Younghyun Lee, and Cheoljong Yang. Visually dehallucinative instruction generation: Know what you don’t know. arXiv preprint arXiv:2402.09717,

2024. 2

- [5] Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, and Heng Tao Shen. Alleviating hallucinations in large visionlanguage models through hallucination-induced optimization, 2024. 3
- [6] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 2, 5, 1
- [7] Xiang Chen, Chenxi Wang, Yida Xue, Ningyu Zhang, Xiaoyan Yang, Qiang Li, Yue Shen, Jinjie Gu, and Huajun Chen. Unified hallucination detection for multimodal large language models. arXiv preprint arXiv:2402.03190, 2024. 2
- [8] Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. Halc: Object hallucination reduction via adaptive focal-contrast decoding, 2024. 3
- [9] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal

- modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2, 5
- [10] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models,

2024. 3

- [11] Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges. arXiv preprint arXiv:2311.03287, 2023. 2
- [12] Wenliang Dai, Junnan Li, DONGXU LI, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Proceedings of the Advances in Neural Information Processing Systems, NeurIPS, pages 49250–49267, 2023. 1
- [13] Wenliang Dai, Zihan Liu, Ziwei Ji, Dan Su, and Pascale Fung. Plausible may not be faithful: Probing object hallucination in vision-language pre-training. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL, pages 2128–2140,

2023. 1

- [14] Laura Fieback, Jakob Spiegelberg, and Hanno Gottschalk. Metatoken: Detecting hallucination in image descriptions by meta classification. arXiv preprint arXiv:2405.19186, 2024. 1, 2
- [15] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 1
- [16] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566, 2023. 2
- [17] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2023. 3
- [18] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models,

2024. 3

- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 7
- [20] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 14953–14962,

2023. 1

- [21] Moonsu Han, Minki Kang, Hyunwoo Jung, and Sung Ju Hwang. Episodic memory reader: Learning what to remember for question answering from streaming data. arXiv preprint arXiv:1903.06164, 2019. 2
- [22] Zongbo Han, Zechen Bai, Haiyang Mei, Qianli Xu, Changqing Zhang, and Mike Zheng Shou. Skip: A simple method to reduce hallucination in large vision-language models, 2024. 3
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 7
- [24] Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946, 2025. 2
- [25] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023. 1
- [26] Wen Huang, Hongbin Liu, Minxin Guo, and Neil Zhenqiang Gong. Visual hallucinations of multi-modal large language models. arXiv preprint arXiv:2402.14683, 2024. 1, 2
- [27] Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. Self-introspective decoding: Alleviating hallucinations for large vision-language models, 2024. 3
- [28] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 4, 5, 1, 2
- [29] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 7
- [30] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, pages 1–38, 2023. 1
- [31] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 2

- [32] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1, 2
- [33] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. 1, 2

- [34] Chaoya Jiang, Wei Ye, Mengfan Dong, Hongrui Jia, Haiyang Xu, Ming Yan, Ji Zhang, and Shikun Zhang. Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models. arXiv preprint arXiv:2402.15721, 2024. 1, 2
- [35] Liqiang Jing, Ruosen Li, Yunmo Chen, Mengzhao Jia, and Xinya Du. Faithscore: Evaluating hallucinations in large vision-language models. arXiv preprint arXiv:2311.01477,

2023. 2

- [36] Soo Min Kang and Richard P Wildes. Review of action recognition and detection methods. arXiv preprint arXiv:1610.06906, 2016. 3
- [37] Prannay Kaul, Zhizhong Li, Hao Yang, Yonatan Dukler, Ashwin Swaminathan, CJ Taylor, and Stefano Soatto. Throne: An object-based hallucination benchmark for the free-form generations of large vision-language models. arXiv preprint arXiv:2405.05256, 2024. 1, 2
- [38] Junho Kim, Hyunjun Kim, Yeonju Kim, and Yong Man Ro. Code: Contrasting self-generated description to combat hallucination in large multi-modal models, 2024. 3
- [39] Sihyeon Kim, Boryeong Cho, Sangmin Bae, Sumyeong Ahn, and Se-Young Yun. Vacode: Visual augmented contrastive decoding, 2024. 3
- [40] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In International conference on machine learning, pages 5583–5594. PMLR, 2021. 2
- [41] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation, 2023. 3
- [42] Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and Bryan Catanzaro. Factuality enhanced language models for open-ended text generation,

2023. 3

- [43] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464, 2023. 1, 2
- [44] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 5, 1
- [45] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model, 2024. 3
- [46] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 2
- [47] Shawn Li, Jiashu Qu, Yuxiao Zhou, Yuehan Qin, Tiankai Yang, and Yue Zhao. Treble counterfactual vlms: A causal approach to hallucination, 2025. 3
- [48] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization, 2023. 3

- [49] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and JiRong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the Empirical Methods in Natural Language Processing, EMNLP, pages 292–305, 2023. 1, 2
- [50] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023. 2
- [51] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. 2024. 2, 5, 1
- [52] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 2, 5, 1
- [53] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021. 2
- [54] Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models. arXiv preprint arXiv:2310.14566, 2023. 1, 2
- [55] Fenglin Liu, Tingting Zhu, Xian Wu, Bang Yang, Chenyu You, Chenyang Wang, Lei Lu, Zhangdaihong Liu, Yefeng Zheng, Xu Sun, et al. A medical multimodal large language model for future pandemics. NPJ Digital Medicine, 6(1): 226, 2023. 1
- [56] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning, 2024. 3
- [57] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 2
- [58] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Proceedings of the Neural Information Processing Systems, NeurIPS, 2023. 1, 2
- [59] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 5, 1
- [60] Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. Phd: A prompted visual hallucination evaluation dataset. arXiv preprint arXiv:2403.11116, 2024. 1, 2
- [61] Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms, 2024. 3
- [62] Holy Lovenia, Wenliang Dai, Samuel Cahyawijaya, Ziwei Ji, and Pascale Fung. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. arXiv preprint arXiv:2310.05338, 2023. 1, 2
- [63] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 2

- [64] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 5, 1
- [65] Ian R McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, et al. Inverse scaling: When bigger isn’t better. arXiv preprint arXiv:2306.09479,

- 2023. 2

[66] Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. Kernel language entropy: Fine-grained uncertainty quantification for llms from semantic similarities,

- 2024. 3

- [67] OpenAI. Gpt-4v (ision) system card. 2023. 1, 2
- [68] Yeji Park, Deokyeong Lee, Junsuk Choe, and Buru Chang. Convis: Contrastive decoding with hallucination visualization for mitigating hallucinations in multimodal large language models, 2024. 3
- [69] Phuc Phan, Hieu Tran, and Long Phan. Distillation contrastive decoding: Improving llms reasoning with contrastive decoding and distillation, 2024. 3
- [70] Xiaoye Qu, Jiashuo Sun, Wei Wei, and Yu Cheng. Look, compare, decide: Alleviating hallucination in large visionlanguage models via multi-view multi-path reasoning, 2024. 3
- [71] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [72] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 8
- [73] Vikas Raunak, Arul Menezes, and Marcin JunczysDowmunt. The curious case of hallucinations in neural machine translation, 2021. 3
- [74] Abhilasha Ravichander, Shrusti Ghela, David Wadden, and Yejin Choi. Halogen: Fantastic llm hallucinations and where to find them. arXiv preprint arXiv:2501.08292, 2025. 2
- [75] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 2
- [76] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4035–4045, 2018. 2
- [77] Fawaz Sammani, Tanmoy Mukherjee, and Nikos Deligiannis. Nlx-gpt: A model for natural language explanations in vision and vision-language tasks. In proceedings of the

- IEEE/CVF conference on computer vision and pattern recognition, CVPR, pages 8322–8332, 2022. 1
- [78] Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. Prompting large language models with answer heuristics for knowledge-based visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 14974–14983, 2023. 1
- [79] Lei Shen, Haolan Zhan, Xin Shen, Hongshen Chen, Xiaofang Zhao, and Xiaodan Zhu. Identifying untrustworthy samples: Data filtering for open-domain dialogues with bayesian optimization. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management, page 1598–1608. ACM, 2021. 3
- [80] Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Gergely Szilvasy, Rich James, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, and Mike Lewis. In-context pretraining: Language modeling beyond document boundaries, 2024. 3
- [81] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023. 2
- [82] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207–1216,

2019. 3

- [83] Zilu Tang, Rajen Chatterjee, and Sarthak Garg. Mitigating hallucinated translations in large language models with hallucination-focused preference optimization. arXiv preprint arXiv:2501.17295, 2025. 3
- [84] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 7
- [85] Qwen Team. Qvq: To see the world with wisdom, 2024. 2, 7
- [86] Qwen Team. Qwen2.5-vl, 2025. 5, 1
- [87] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024. 1
- [88] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2
- [89] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2
- [90] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Ming Yan, Ji Zhang, and Jitao Sang.

- An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023. 2
- [91] Lei Wang, Jiabang He, Shenshen Li, Ning Liu, and EePeng Lim. Mitigating fine-grained hallucination by finetuning large vision-language models with caption rewrites. In International Conference on Multimedia Modeling, pages 32–45. Springer, 2024. 2
- [92] Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding, 2024. 3
- [93] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training videotext llms for grounding text in videos. arXiv preprint arXiv:2403.10228, 2024. 2
- [94] Yuxuan Wang, Yueqian Wang, Dongyan Zhao, Cihang Xie, and Zilong Zheng. Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models. arXiv preprint arXiv:2406.16338, 2024. 2, 3
- [95] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 6
- [96] Jiaheng Wei, Yuanshun Yao, Jean-Francois Ton, Hongyi Guo, Andrew Estornell, and Yang Liu. Measuring and reducing llm hallucination without gold-standard answers via expertise-weighting. arXiv preprint arXiv:2402.10412, 2024. 2
- [97] Sangmin Woo, Jaehyuk Jang, Donguk Kim, Yubin Choi, and Changick Kim. Ritual: Random image transformations as a universal anti-hallucination lever in large vision language models, 2024. 3
- [98] Sangmin Woo, Donguk Kim, Jaehyuk Jang, Yubin Choi, and Changick Kim. Don’t miss the forest for the trees: Attentional vision calibration for large vision language models,

2024. 3

- [99] Junfei Wu, Qiang Liu, Ding Wang, Jinghao Zhang, Shu Wu, Liang Wang, and Tieniu Tan. Logical closed loop: Uncovering object hallucinations in large vision-language models, 2024. 3
- [100] Kai Wu, Boyuan Jiang, Zhengkai Jiang, Qingdong He, Donghao Luo, Shengzhi Wang, Qingwen Liu, and Chengjie Wang. Noiseboost: Alleviating hallucination with noise perturbation for multimodal large language models, 2024. 3
- [101] Ziheng Wu, Zhenghao Chen, Ruipu Luo, Can Zhang, Yuan Gao, Zhentao He, Xian Wang, Haoran Lin, and Minghui Qiu. Valley2: Exploring multimodal models with scalable vision-language design. arXiv preprint arXiv:2501.05901,

2025. 5, 1

- [102] Xin Xiao, Bohong Wu, Jiacong Wang, Chunyuan Li, Xun Zhou, and Haoyuan Guo. Seeing the image: Prioritizing visual correlation by contrastive alignment, 2024. 3
- [103] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava : Parameter-free llava extension from images to videos for video dense captioning, 2024. 2, 5, 1

- [104] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In European Conference on Computer Vision, pages 131–147. Springer,

2025. 2

- [105] Xiaohan Xu, Chongyang Tao, Tao Shen, Can Xu, Hongbo Xu, Guodong Long, Jian guang Lou, and Shuai Ma. Rereading improves reasoning in large language models, 2024. 3
- [106] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023. 1
- [107] Qifan Yu, Juncheng Li, Longhui Wei, Liang Pang, Wentao Ye, Bosheng Qin, Siliang Tang, Qi Tian, and Yueting Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data, 2024. 3
- [108] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807– 13816, 2024. 7, 8
- [109] Zihao Yue, Liang Zhang, and Qin Jin. Less is more: Mitigating multimodal hallucination from an eos decision perspective, 2024. 3
- [110] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2
- [111] Shaolei Zhang, Tian Yu, and Yang Feng. Truthx: Alleviating hallucinations by editing large language models in truthful space, 2024. 3
- [112] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023. 1
- [113] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 2, 5, 1
- [114] Yi-Fan Zhang, Qingsong Wen, Xue Wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Onenet: Enhancing time series forecasting models under concept drift by online ensembling, 2023. 3
- [115] Zheng Zhao, Emilio Monti, Jens Lehmann, and Haytham Assem. Enhancing contextual understanding in large language models through contrastive decoding, 2024. 3
- [116] Gengze Zhou, Yicong Hong, and Qi Wu. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, pages 7641–7649, 2024. 1
- [117] Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. Ibd: Alleviating hallucinations in large visionlanguage models via image-biased decoding, 2024. 3
- [118] Wenhao Zhu, Sizhe Liu, Shujian Huang, Shuaijie She, Chris Wendler, and Jiajun Chen. Multilingual contrastive decoding via language-agnostic layers skipping, 2024. 3

## Exploring Hallucination of Large Multimodal Models in Video Understanding:

## Benchmark, Analysis and Mitigation Supplementary Material

### A. Evaluation Details

Our evaluation is conducted using GPT-4o-mini [28] by comparing human-generated answers with model responses to determine correctness.

For models without CoT, responses can be directly compared using GPT-4o-mini. Multiple-choice questions are evaluated using the prompt in Table 7, binary-choice questions using the prompt in Table 8, and short-answer questions using the prompt in Table 9.

For models using CoT, we found that long reasoning process might affect GPT-4o-mini’s judgment in multiplechoice and binary-choice questions. Therefore, we first use GPT-4o-mini to extract a concise answer from the model’s reasoning process based on the given options in the question. Then, we can conduct our evaluation based on the extract answer. For multiple-choice questions, we let GPT-4o-mini directly assess whether the model’s response aligns with the intended meaning of the correct answer based on its reasoning process. Multiple-choice questions are evaluated using the prompt in Table 11, binary-choice questions using the prompt in Table 10, and short-answer questions using the prompt in Table 12.

### B. Implementation Details of the LMMs

The models evaluated in our experiment are as follows:

Video-ChatGPT [64] combines the capabilities of LLMs with a pretrained visual encoder adapted for spatiotemporal video representation. The Video-ChatGPT model used in the experiment has a parameter size of 7B. For each video, we sample 100 frames and resize them to a resolution of 224×224.

Valley-Eagle [101] is a cutting-edge multimodal large model designed to handle a variety of tasks involving text, images, and video data, which is developed by ByteDance. The Valley-Eagle model used in the experiment has a parameter size of 7B. For each video, we sample 100 frames and resize them to a resolution of 384×384.

Video-LLaVA [52] unifies visual representation into the language feature space to advance the foundational LLM towards a unified LVLM. The Video-LLaVA model used in the experiment has a parameter size of 7B. For each video, 8 frames are uniformly sampled from the video and resize to 224*224 definition as visual input. The normalized mean and standard deviation of video frames are set to (0.48145466, 0.4578275, 0.40821073) and (0.26862954,

- 0.26130258, 0.27577711), respectively.

VideoChat2 [44] integrates video foundation models and large language models via a learnable neural interface. The VideoChat2 model used in the experiment has a parameter size of 7B. For each video, it uniformly samples 8 frames from the video as visual input. The normalized mean and standard deviation of video frames are set to (0.48145466, 0.4578275, 0.40821073) and (0.26862954, 0.26130258, 0.27577711), respectively.

ShareGPT4Video [6] facilitates the video understanding of large video-language models and the video generation of text-to-video models via dense and precise captions. The parameter size of the ShareGPT4Video model used in the experiment is 8B. For each video, it samples 16 frames as the visual input.

LLaVA [59] is an end-to-end trained large multimodal model that connects a vision encoder and LLM for generalpurpose visual and language. The parameter size of the ShareGPT4Video model used in the experiment is 7B. It randomly samples a frame and resizes it to 336×336 as the visual input.

LLaMA-VID [51] use a dual-token strategy to significantly reduce the overload of long videos while preserving critical information. In the experiments of this paper, two parameter sizes of the LLaMA-VID model were evaluated, including 7B and 13B. For each second of the video, the model samples one frame as visual input.

PLLaVA [103] proposes a simple but effective pooling strategy to smooth the feature distribution along the temporal dimension and thus reduce the dominant impacts from the extreme features. In the experiments of this paper, two parameter sizes of the PLLaVA model were evaluated, including 7B and 13B. For each video, it uniformly samples 16 frames from the video and resize them to a resolution of 336×336 as visual input. The target pooling shape is set to be 16 × 12 × 12 × d, where d corresponds to the input dimension of the LLMs.

Qwen2.5-VL-Instruct [86] introduces dynamic resolution processing and absolute time encoding, enabling it to process images of varying sizes and videos of extended durations with second-level event localization. In the experiments of this paper, two parameter sizes of the Qwen2.5VL-Instruct model were evaluated, including 3B and 7B. For each second of the video, the model samples two frames as visual input.

LLaVA-NEXT-Video-DPO [113] expands the applications to multi-image scenarios including multi-image, multiframe (video), multi-view (3D), and multi-patch (single-

Multiple-Choice: You are a professional homework grading tool. I will provide you with four rules for grading:

- 1. This is a multiple-choice question. Judge the correctness based on the selected letter and actual content of the provided answers.
- 2. Regardless of the question type, respond only with either 1 or 0, without any additional explanation.
- 3. 1 means the prediction is correct, and 0 means it is incorrect.
- 4. If the predicted answer matches the correct answer in meaning, even if it is phrased differently, consider it correct. For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1. Based on the question and its standard answer, is the prediction correct? If yes, return only 1; otherwise, return only 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {extract_answer}

Table 7. The evaluation prompts for multiple-choice questions employed in the experiment for models without CoT.

Binary-Choice: You are a professional homework grading tool. I will provide you with four rules for grading:

- 1. This is a yes/no question. The Standard Answer is only Yes/No, please directly compare the standard answer with ’yes’ or ’no’ in the predicted answer.
- 2. No matter what kind of questions, only response with one of 1 or 0. No more explanation.
- 3. 1 means correct (they are the same), 0 means wrong (they are different). For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1. Based on the question, is the prediction correct? If yes, only return 1, otherwise only return 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {extract_answer}

Table 8. The evaluation prompts for Binary-Choice questions employed in the experiment for models without CoT.

image) scenarios. In the experiments of this paper, two parameter sizes of the LLaVA-NEXT-Video-DPO model were evaluated, including 7B and 34B. For each video, it uniformly samples 32 frames from video and resize them to a resolution of 336×336 as visual input.

Video-LLaMA2 Video-LLaMA2 [9] incorporates a custom spatial temporal convolution (STC) connector, which effectively captures the intricate spatial and temporal dynamics of video data. For each video, it uniformly samples 8 frames from the video and resize them to a resolution of 336×336 as visual input.

GPT-4o-mini GPT-4o-mini [28] is a low-cost, lowlatency online model launched by OpenAI, capable of handling multimodal tasks in various complex scenarios. For each video, we sample 16 frames uniformly and resize to 768 pixels, keeping the original aspect ratio unchanged.

C. Case Demonstration

In addition to the experimental effects presented in Section

- 4 and Section 5.3, we also present some response cases for

different baseline LMMs and our improved model.

#### C.1. Evaluated baseline LMMs

Fig.9 and Fig.10 present examples of hallucinations observed in various LMMs. The responses from these models are concise and highly susceptible to hallucination. These examples highlight several factors contributing to the hallucinations analyzed above, including conflict with prior knowledge, conflict between contexts, and the capability deficiencies of those LMMs.

#### C.2. LLaVA-NeXT-Video-7B-Thinking

Fig.11 presents two cases of LLaVA-NeXT-Video-7BThinking. By leveraging the proposed thinking-based training strategy, LLaVA-NeXT-Video-7B-Thinking is capable of reasoning and thinking about both video content and problem statements, integrating the semantics of visual and textual modalities to derive the correct results.

Short-Answer: You are a professional homework grading tool. I will provide you with four rules for grading

- 1. The Standard Answer is a sentence. Compare the provided predicted answers based on their meaning rather than exact wording. The prediction is correct if it conveys the same intent.
- 2. If the question asks about identity or species, the predicted answer is correct as long as the core identity or species is correct, even if descriptive adjectives differ.
- 3. If the question asks about a scene, the predicted answer is correct if the described scene exists in the standard answer or is similar.
- 4. If the answer indicates that the asked character, object or event is not visible or does not exist, it should be considered as "No answer."
- 5. Regardless of the question type, respond only with either 1 or 0, without any additional explanation.
- 6. 1 means correct, and 0 means incorrect. For example, if the prediction conveys the same meaning as the standard answer, even if phrased differently, you should respond with 1. Based on the question, is the prediction correct? If yes, return only 1; otherwise, return only 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {res[idx]}

Table 9. The evaluation prompts for Short Answer questions employed in the experiment for models without CoT.

###### Multiple-Choice:

- (1) Extracting Answer: This is a multiple-choice question. Based on the given question and reasoning process, extract the corresponding answer of the reasoning process. Question: {question[idx]} Reasoning Process: {res[idx]} Instructions:

- 1. Identify the correct answer based on the reasoning process.
- 2. If the reasoning process directly mentions one of the given choices (A, B, or C), return the corresponding letter along with the full text of that option (e.g., "A. Option text").
- 3. If the reasoning process provides an answer that does not explicitly mention A, B, or C, compare its meaning to the given choices and return the best-matching option in the format "Letter. Option text".
- 4. If the reasoning process concludes that the correct answer is "no answer" or "I don’t know", return "no answer".
- 5. Return only the final answer, without explanation or additional text.
- 6. Foucs more on the final summary sentence.

- (2) Judging: You are a professional homework grading tool. I will provide you with four rules for grading:

- 1. This is a multiple-choice question. Judge the correctness based on selected letter and actual content of the provided answers.
- 2. Regardless of the question type, respond only with either 1 or 0, without any additional explanation.
- 3. 1 means the prediction is correct, and 0 means it is incorrect.
- 4. If the predicted answer matches the correct answer in meaning, even if it is phrased differently, consider it correct. For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1. Based on the question and its standard answer, is the prediction correct? If yes, return only 1; otherwise, return only 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {extract_answer}

Table 10. The evaluation prompts for multiple-choice questions employed in the experiment for models with CoT.

###### Binary-Choice:

- (1) Extracting Answer: This is a multiple-choice question. Based on the given question and reasoning process, extract the corresponding answer of the reasoning process. Question: {question[idx]} Reasoning Process: {res[idx]} Instructions:

- 1. Identify the correct answer based on the reasoning process.
- 2. If the reasoning process explicitly states "yes" or "no", return direct "yes" or "no"
- 3. If the reasoning process concludes that the correct answer is "no answer" or "I don’t know", return "no answer".

5. Return only the final "yes" or "no", without explanation or additional text.

- (2) Judging: You are a professional homework grading tool. I will provide you with four rules for grading

- 1. This is a yes/no question. The Standard Answer is only Yes/No, please directly compare the standard answer with ’yes’ or ’no’ in the predicted answer.
- 2. No matter what kind of questions, only response with one of 1 or 0. No more explaination.
- 3. 1 means correct (they are the same), 0 means wrong (they are different). For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1. Based on the question, is the prediction correct? If yes, only return 1, otherwise only return 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {extract_answer}

Table 11. The evaluation prompts for Binary-Choice questions employed in the experiment for models with CoT.

Short-Answer: You are a professional homework grading tool. I will provide you with four rules for grading

- 1. The Standard Answer is a sentence. Compare the provided predicted answers based on their meaning rather than exact wording. The prediction is correct if it conveys the same intent.
- 2. If the question asks about identity or species, the predicted answer is correct as long as the core identity or species is correct, even if descriptive adjectives differ.
- 3. If the question asks about a scene, the predicted answer is correct if the described scene exists in the standard answer or is similar.
- 4. If the answer indicates that the asked character, object or event is not visible or does not exist, it should be considered as "No answer."
- 5. Regardless of the question type, respond only with either 1 or 0, without any additional explanation.
- 6. 1 means correct, and 0 means incorrect. For example, if the prediction conveys the same meaning as the standard answer, even if phrased differently, you should respond with 1. Based on the question, is the prediction correct? If yes, return only 1; otherwise, return only 0. Question: {question[idx]} Standard Answer: {answer[idx]} The Predicted Answer: {res[idx]}

Table 12. The evaluation prompts for Short Answer questions employed in the experiment for models with CoT.

Video-ChatGPT

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

##### Question

##### Answer Response

Extracted

C. Mouse. C.Mouse

The cat is holding a mouse in the video.

What is the cat holding in the video? A. Sword. B. Fish. C. Mouse.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Question

##### Answer Response

Extracted

Black

I don't know/No answer

What is the color of the man's car? The color of the man's car is black.

VideoLLaMA-7B

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Question

##### Answer Response

Extracted

Alarm clock Water

The man immersed a glass of water in the water.

What item did the man immerse in the water?

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Question

##### Answer Response

Extracted

No Yes

Yes, the object in the video is a brown balloon.

Is the object in the video a brown balloon?

LLaMA-VID-7B

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Question

##### Answer Response

##### Extracted

B.Blue cat A. Mouse

A. Mouse

Who is holding a bouquet of flowers? A. Mouse. B. Blue cat. C. Boy.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Question

##### Answer Response

Extracted

No Yes

Yes, the poker tower on the table in the video collapsed.

Did the poker tower on the table in the video collapse?

PLLaVA

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

##### Question

##### Answer Response

Extracted

C. Lamp tube. A. Bookshelf.

A

What is placed on the wooden table? A. Bookshelf. B. Cup. C. Lamp tube.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Question

##### Answer Response

Extracted

A. The person. B. The airplane.

B

Which is larger in the video? A. The person. B. The airplane.

Video-LLaVA

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Question

##### Answer Response

Extracted

No Yes

Yes, there is an individual in a wheelchair present in the video.

Is there an individual in a wheelchair?

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Question

##### Answer Response

Extracted

B. Suitcase A. Clothes.

A. Clothes.

What did the man place in the suitcase? A. Clothes. B. Suitcase. C. Shoes.

VideoChat2

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Question

##### Answer Response

##### Extracted

B. Socks. C. Gloves.

The man in the video wore gloves on his hands.

What did the man wear on his hands? A. Bag. B. Socks. C. Gloves.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Question

##### Answer Response

Extracted

A pot. A hat.

The thing that is placed on the boy's head is a hat.

What is the item positioned on the boy's head?

LLaVA-NeXT-Video-7B-Thinking

[Figure 177]

[Figure 178]

##### Question

[Figure 179]

Answer

Who is utilizing the mousetrap in the video? A. A person. B. A cat

B. A cat.

[Figure 180]

[Figure 181]

[Figure 182]

Response

Extracted

B. A cat.

In the video, a cat is utilizing the mousetrap.

The cat is seen holding a mousetrap with a pink ball attached to it, which is a common design for a mousetrap. The cat is also holding a piece of cheese or a treat, which is likely bait for the mousetrap. The cat is positioned in front of the mousetrap, and it appears to be either setting up the mousetrap or interacting with it in some way. The cat's actions suggest that it is either setting up the mousetrap to catch a mouse or has already caught a mouse and is about to eat it.

[Figure 183]

[Figure 184]

The cat's behavior is consistent with the typical actions of a cat when it encounters a mousetrap. Cats are known to be curious and playful creatures, and they often interact with mousetraps as a form of entertainment or to satisfy their hunting instincts. The cat's focus on the mousetrap and its interest in the pink ball indicate that it is engaged with the device.

[Figure 185]

Therefore, the answer is B. A cat.

[Figure 186]

[Figure 187]

Question

[Figure 188]

Answer

What did the boy utilize to achieve flight? A. A big bird. B. A broom. C. An airplane.

B. A broom.

[Figure 189]

[Figure 190]

[Figure 191]

Response

##### Extracted

B. A broom.

The image depicts a scene from a Harry Potter movie, where a group of young witches and wizards are practicing flying on broomsticks. The boy in the foreground is holding a broomstick and appears to be in the process of flying. The question asks what he utilized to achieve flight.

[Figure 192]

Let's analyze the options:

- A. A big bird: This is not a viable option in the context of the image.
- B. A broom: The boy is holding a broomstick, which is a common prop used in the Harry Potter series to achieve flight.
- C. An airplane: This is not a viable option in the context of the image, as the boy is not using an airplane. Given the context of the image, the most likely option is B. A broom. Therefore, the final answer is B. A broom.

[Figure 193]

[Figure 194]

Figure 11. Case of response from LLaVA-NeXT-Video-7B-Thinking.

