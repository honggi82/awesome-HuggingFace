## Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding

Yuanhan Zhang*1 Yunice Chew*2 Yuhao Dong1 Aria Leo2 Bo Hu2 Ziwei Liu1

1S-Lab, Nanyang Technological University 2Independent Researcher

{yuanhan002, ziwei.liu}@ntu.edu.sg yunicechew1119@gmail.com

|Human-level|
|---|
|Intelligence<br><br>[Figure 1]|
|[Figure 2]<br><br>💁<br><br>|84 %|
|---|
<br><br>[Figure 3]<br><br>*Primary Q Correctness Track<br><br>[Figure 4]<br><br>|27%|
|---|
<br><br>|37 %|
|---|
<br><br>|36 %|
|---|
<br><br>*Robustness Track<br><br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>💁<br><br>|64 %|
|---|
<br><br>|22%|
|---|
|

|Correctness -Complex Visual Narratives| | |Robustness - Natural Adversarial Questions|
|---|---|---|---|
|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>|Primary Q|
|---|
|Is the tissue box real or cake?|
<br><br>|Correctness (1/1) 👍<br><br>[Figure 14]<br><br>[Figure 15]|
|---|
<br><br>[Figure 16]<br><br>🤖<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>|[Figure 33]<br><br>✅<br><br>The tissue box is a cake, as shown when the person bites it, and reveals the insides to be made of cake.|
|---|
<br><br>Narrative Complexity<br><br>—World Knowledge<br><br>Visual Complexity — Illusion<br><br>“Hyper-realistic cakes VS real items identification challenge” Video Uniform sampled w/ 80 frames<br><br>[Figure 34]<br><br>🤔<br><br>[Figure 35]<br><br>| |
|---|
<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]| | ||Multi-Choice Q| |
|---|---|
| | |
|What is the tissue box truly?<br><br>A. Cake, B. Real box, C. Plastic, D. Paper.| |
<br><br>| | |
|---|---|
|[Figure 40]<br><br>🤖|[Figure 41]<br><br>✅|
| | |
|It is a real box.|[Figure 42]<br><br>❌|
| | |
|Yes, a real box.|[Figure 43]<br><br>❌|
| | |
|Yes, a cake.|[Figure 44]<br><br>✅|
| | |
|B. Real box.|[Figure 45]<br><br>❌|
| | |
<br><br>|Rephrased Q| |
|---|---|
| | |
|What is the tissue box truly?| |
<br><br>|Robustness (2/5) 👎<br><br>[Figure 46]<br><br>[Figure 47]|
|---|
<br><br>|Wrongly-led Q| |
|---|---|
| | |
|Is the tissue box real?| |
<br><br>|Correctly-led Q| |
|---|---|
| | |
|Is the tissue box a cake?| |
<br><br>|Primary Q| |
|---|---|
| | |
| | |
<br><br>[Figure 48]<br><br>😈 😈 😈 😈<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>🤔|

# arXiv:2507.15028v1[cs.CV]20Jul2025

Figure 1. Overview of the Video Thinking Test. Video-TT introduces two challenges: (1) ensuring correctness in understanding complex visual stories; (2) maintaining robustness against natural adversarial conditions.

##### 1. Introduction

##### Abstract

Human intelligence requires correctness and robustness, with the former being foundational for the latter. In video understanding, correctness ensures the accurate interpretation of visual content, and robustness maintains consistent performance in challenging conditions. Despite advances in video large language models (video LLMs), existing benchmarks inadequately reflect the gap between these models and human intelligence in maintaining correctness and robustness in video interpretation. We introduce the Video Thinking Test (Video-TT), to assess if video LLMs can interpret real-world videos as effectively as humans. Video-TT reflects genuine gaps in understanding complex visual narratives, and evaluates robustness against natural adversarial questions. Video-TT comprises 1,000 YouTube Shorts videos, each with one open-ended question and four adversarial questions that probe visual and narrative complexity. Our evaluation shows a significant gap between video LLMs and human performance.

Human intelligence fundamentally depends on two key aspects: correctness and robustness, with correctness being a necessary condition for robustness [13, 47]. Correctness ensures that a system’s outputs align with the expected standards or truths. In video understanding, this translates to making accurate judgments about the visual content. Building on this foundation, robustness is essential for ensuring that these interpretations remain reliable and consistent under various conditions, including ambiguity and conflicting information. These attributes are vital for providing dependable insights and managing complex situations. The development of video large language models (video LLMs) [4, 18, 20, 34, 46, 50, 51] have brought their capabilities closer to human intelligence. Developing benchmarks that accurately highlight current shortcomings is crucial for further improving video LLMs’ performance.

However, current benchmarks fail to accurately reflect the differences between video LLMs and human intelligence. Regarding correctness, existing benchmarks [2, 6, 7, 11, 15, 37, 52] do not clearly distinguish between errors caused by insufficient frame sampling and errors caused by

*Equal Contribution. Project page: https://zhangyuanhan- ai.github.io/

video-tt/

[Figure 53]

- Figure 2. Dataset Comparisons. Left: We present Video thinking Test (Video-TT) for the following features: ensure the questions are complex. addresses the issue of selecting frames from the video. provided rationale for each answer. Middle: In Video-TT, the top-performing model reaches only half of human performance. Right: The lower performance of GPT-4o in the VideoMME-Long track may be due to the selection of sparse frames rather than a genuine gap in understanding between humans and models.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Dataset #Clips/#QA NExT-QA [38] 1,000/8,564 Social-IQ [45] 1,250/7,500 MVBench [15] 3,641/4,000 EgoSchema [23] 5,063/5,063 TempCompass [17] 504/13,157 Video-MMMU [11] 300/900 WorldQA [52] 303/1,007 VideoMME [7] 900/2,700 TemperalBench [2] 410/7,540 Video-TT 1,000/5,000

Human performance interval

VideoMME-Short

65.3

VideoMME-Long

384

Max sampling number of GPT-4o

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

ral adversarial questions: We assess model performance against natural adversarial questions crafted to view a query from different angles. For instance, if the query is “Which player’s head did the man tap?” and the correct answer is “Number 8,” the model should also handle a misleading version like “Did the man tap the head of the player wearing number 9?” These questions simulate real-world adversarial conditions.

failures in actual video understanding. As a result, the large performance gap between models and humans might reflect the limitations of frame sampling rather than a true understanding gap (Fig.2 right). In cases where models can sample enough frames—especially for shorter videosadvanced models can perform at levels comparable to humans (Fig.2 middle). This can lead to the mistaken impression that current models have reached human-level video understanding. Therefore, it is crucial to develop benchmarks that challenge video LLMs in areas where they underperform, clearly separating issues with frame sampling from genuine limitations in understanding. Regarding robustness, recent studies [26] investigate how video LLMs respond to adversarial changes, such as visual pixel alterations or distorted words in instructions. However, these scenarios are often artificial and do not reflect the complexities of real-world conditions, making the true impact of natural adversarial conditions [10] less clear.

In Video-TT, we selected 1,000 YouTube Shorts videos and created one primary open-ended question and four related adversarial questions for each, based on eight visual or narrative complexity factors. We evaluated both top opensource video LLMs and proprietary models. Our comparison of these models with human revealed significant insights for enhancing video understanding. Our key findings are summarized as follows:

- • We introduce the Video Thinking Test, a crucial benchmark for assessing the correctness and robustness of large video language models in understanding videos. This benchmark is crafted to ensure that any mistakes in the model’s responses are due to its lack of understanding rather than errors in selecting key frames. Our results reveal a significant gap in performance between humans and the top-performing video model. Humans achieve an accuracy of 84.3% and robustness of 64.3%, while the model only reaches 36.6% accuracy and 36.0% robustness, indicating major areas for improvement.
- • This study is the first to demonstrate that current open-source models significantly lag behind GPT-4o in terms of natural adversarial robustness. While they show comparable performance in the correctness aspect of the Video Thinking Test, in the robustness track, the top open-source model—Qwen2.5-VL-

To address these problems, we introduce the Video Thinking Test (Video-TT), a new benchmark highlight current shortcomings in video LLM. This test focuses on: (i) Correctness toward complex visual narratives: We measure this by evaluating the accuracy of model responses to complex questions, highlighting differences in video understanding between models and humans. We define “visual complexity” and “narrative complexity” as guidelines for creating complex questions. Each question is created after examining a manageable set of video frames. Therefore, the questions are complex yet answerable within a reasonable number of frames. (ii) Robustness toward natu-

Current video LLMs typically follow a two-step process: first, they sample a limited number of frames, and then they understand the content within these frames.

For example, changing the query from “Which player’s head did the man tap?” to “Which player’s heed did the man tap?”

72B—scores 13.8 points lower than GPT-4o.

• Our error analysis of all errors made by GPT-4o shows that for recognizing content, GPT-4o struggles with unclear or unusual content, often guessing the most likely scenario rather than accurately representing the video. It also faces challenges in distinguishing different scenes, which impacts its ability to track actions and identify participants in multiple scenes. For cognitive ability, it lacks the integration of world knowledge needed to think about likely intentions, goals, and social dynamics in videos, and it struggles to use correctly recognized cues to deduce hidden information.

##### 2. Related Works

Our work lies within in the fild of evaluating the video understanding of video large langauge model through visual question-answering (QA) [1]. VQA [3, 8, 11, 12, 14, 36, 38–44, 52] is a key task in video-language research in diverse visual domains.

Correctness in Video Understanding Recently, several benchmarks [7, 15] have been proposed to evaluate video large language models (Video LLM) correctness in opendomain video understanding. MVBench [15] integrates 11 public video benchmarks using a static-to-dynamic method. However, this design has issues because these academic datasets are already well known and widely used in the research community. This means that many models may already be trained or fine-tuned on these videos. To address this, VideoMME [7] collects new videos by sourcing them from YouTube. This benchmark largely advances the development of Video LLM. In VideoMME, As illustrated in Fig. 2 (right), the maximum number of frames able to be sampled by GPT-4o is 384. As video duration increases, it becomes more challenging to sample key frames in the VideoMME-Long track, which is a major hurdle in improving performance. This issue also occurs in other long video understanding datasets [6, 37, 53]. While handling long videos is a crucial aspect of video research, our work focuses on the “understanding” capability of Video LLMs. We meticulously ensure each question is answerable with manageable video frames. On the other hand, in the datasets, such as VideoMME-short track, where most video frames can be sampled, the model’s performance has reached a limit. Thus, which scenarios in short videos still challenge current Video LLM is an open question. This motivates the creation of Video-TT. Meanwhile, several benchmarks also try to find scenarios that current Video LLM cannot handle. For example, FunQA [39] tests video reasoning limits with counter-intuitive and humorous content. TemporalBench [2, 17] examines the model’s grasp of fine-grained temporal dynamics. Unlike these benchmarks, Video-TT covers diverse scenarios without being limited to a specific video domain or type of question. We aim to build

a complex and comprehensive video Q&A benchmark.

Robustness in Video Understanding Recent studies [26] evaluate the robustness of multimodal models by testing their performance under artificial distortions of instructions or video pixels. In this work, we focus on the significance of assessing natural adversarial robustness. This is crucial to determine whether models genuinely comprehend video content.

##### 3. Dataset

In the Video Thinking Test, we aim to present a challenge that underlines the differences in accuracy and robustness of video understanding between models and humans. In Sec.3.1, we explore methods to pose complex questions that test the models’ ability to accurately interpret video content. In Sec.3.2, we investigate how to ask natural adversarial questions to ensure that these interpretations remain reliable.

###### 3.1. How to Ask a Complex Question?

One question that guides this benchmark is: What factors make a question complex? We propose that the complexity of a question does not solely depend on its type (e.g. “object color” vs. “plot understanding”), but also on the context, reasons, or scenarios under which the question is asked. For example, the question “What is the color of the second car in the video?” might appear simple, but it becomes difficult if the car is moving fast, partially obscured, or viewed from an unusual angle. To explore how complex questions are formed, we start by identifying components within a video that could be questioned. We analyze the video content hierarchy based on [54], which categorizes it from bottom to top as: element, event, plot . Each level can be the focus of a question. We then consider which factors make these contents hard for viewers to grasp, leading to complex questions.

First, from the perspective of video content, following [9, 24, 29, 31], we introduce visual complexity. This idea from cognitive science shows how complex visual content is. It is defined by the number of elements, the range of shapes, the variety of colors, the amount of texture, and the way items are arranged. We identify the following factors that affect visual complexity: (1) Unclear & Unusual Content: Does the content differ from what we normally see? Does the video have noise, blur, occlusion, or other issues that hide its content? (2) Movement Speed: Is any part of the video or the camera moving too fast, making it hard to identify or track objects? (3) Spatial-temporal Arrangement: How are objects arranged and interacting within the scene? Is there an abundance of spatial or temporal information that increases the cognitive load required to identify specific elements? (4) Illusions: Are there any techniques

Adversarial Q Annotation

Ensuring Complexity Primary Q Annotation

Answer Annotation Sampling Check

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Reformats complexity

Key Conclusion

Video retained if complexity exists

[Figure 67]

Original

[Figure 68]

[Figure 69]

[Figure 70]

Q: …?

Human Logical Rationale

80 Frames Removed Audio

Q

Expand

|AlignmentCheck<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>👎<br><br>[Figure 77]<br><br>[Figure 78]<br><br>👍 👍 👎<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>Retained<br><br>Rejected<br><br>(Done in every stage)<br><br>||<br><br>&&<br><br>[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

👍

Model Wrongness Clarification

Q: …?

Refine

A: …

A: …

A: …

[Figure 90]

[Figure 91]

[Figure 92]

❌

✅

❌

[Figure 93]

A: …

[Figure 94]

✅

[Figure 95]

Video-TT

Complete answer

Retained

Retained if wrong >1

- Figure 3. Benchmark Curation Pipeline. Our annotation pipeline ensures that each question: (1) is complex enough to differentiate between human and model video understanding capabilities; (2) can be understood with a limited number of sampled frames; (3) also assesses the models’ robustness against natural adversarial conditions.

Q-5: Why did the woman bump into the door instead of opening it herself at the end of the video?

Q-6: Why in the photo the person didn't have wings had a reflection of them with wings in the water?

Q-7: What narrative purpose does the scene with the person drinking from a cup serve in the video?

Q-8: What scenario is the woman in the video imitating with her actions?

Complex Plot Technical Editing Narrative Editing World Knowledge

Unclear Movement Speed Spatial-temporal Illusion

Q-2: How does

the speed of the orange watering can change?

Q-1: Where are the last few targets come from？

Q-3: Which player throws the ball first in the indoor stadium background?

Q-4: What's in the middle of the road?

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

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

- Figure 4. Eight Complex Factors in Our Datasets. Video links of each case: Q-1, Q-2, Q-3, Q-4, Q-5, Q-6, Q-7, Q-8

Primary Question

|[Figure 120]|
|---|

|Open-endedQuestion: Which player‘s head did the man tap? Answer: Number 8|
|---|

NaturalAdversarial Questions

|Rephrased open-endedQuestion: Whose head did the man tap?<br><br>|
|---|

|Correctly-led open-endedQuestion: Did the man tap the head of the player wearing the number 8 jersey?<br><br>|
|---|

|Wrongly-led open-endedQuestion: Did the man tap the head of the player wearing the number 9 jersey?|
|---|

|Multi-choice Question: Which player‘s head did the man tap? A. Number 7,B. Number 8,C. Number 9 , D. Number 10|
|---|

Figure 5. VQA Question Prototypes. We present our five question prototypes. Man highlights the man framed by a bounding box.

###### 3.2. How to Ask a Natural Adversarial Question?

To reach human-level understanding of videos, it is not enough to just answer questions correctly; we must also explore how changing the wording of a question affects model performance. These natural adversarial questions broaden our study and help users gauge the reliability of the model. Consider the primary open-ended question: Which player’s head did the man in the gray coat next to a red pole tap?, based on which we derived four natural adversarial questions, as shown in Fig. 5. Specifically, these questions include: (1) Rephrased Open-ended Question, which rewords the primary question with minor semantic alterations. (2) Correctly-led Open-ended Question, which provides accurate cues about key points, helping guide the model toward the correct understanding. (3) Wrongly-led Openended Question, which gives misleading cues about key points, directing the model towards an incorrect understanding, and (4) Multiple-choice Question, where the a combination of correct/wrong-led options are designed to test the model’s comprehension of the video.

that create illusions and make it hard to recognize the content?

Second, from the perspective of the video producer, referring to [28, 32], we discuss narrative complexity, which includes special design choices that go beyond linear storytelling and require more active engagement from viewers. We define four elements of narrative complexity: (1) Complex Plot: Does the plot include twists or an unexpected conclusion? (2) Narrative Editing: Are there convoluted combined shots, such as montage methods, to present a story? (3) Technical Editing: Are there special filming techniques or post-production manipulations that are seamlessly integrated and hard to detect? (4) World Knowledge: Does the video require world knowledge or assumptions for full understanding?

These complexities at various levels require viewers to engage more deeply with the video content.

###### 3.3. Data Curation Process

Primary Question Annotation Based on the understanding of visual complexity and narrative complexity, we asked the annotators to select videos and annotate them with question-answer pairs. The selected video and the all questions should meet the following standard: (1) Ensuring Complexity for Human: Each question must involve at least one complex factor as previously discussed. In Fig. 5, identifying a man in a gray coat next to a red pole in a video (an example of visual complexity—unclear element) lead to a question like: “Which player did the man in the gray coat next to the red pole tap on the head?” (2) Ensuring Complexity for Model: Questions tested against GPT-4o [25], LLaVA-Video-7B [51], and Qwen2.5-VL-7B [34]. If any of these models fail to provide a correct answer in at least one out of three attempts, the question is considered sufficiently complex and kept for further use.

Answer and Rationale Annotation Besides providing an answer, annotators must explain their reasoning process in answering the primary open-ended question. This includes a detailed explanation of how they arrived at the correct answer and a discussion of the flaws in an incorrect answer provided by prior models. Please see the example in Fig. 7.

Sampling Check Annotators are instructed to formulate questions answerable by viewing only 80 uniformly sampled frames. This criterion ensures the frame sampling does not hinder video understanding, addressing a common issue in recent video understanding benchmarks [7, 37]. Additionally, it establishes that our dataset emphasizes visual rather than auditory information.

Adversarial Question Expansion The same individual who developed the initial primary open-ended questions also crafted four adversarial variants. Specifically, the annotator constructed misleading open-ended and multiplechoice questions by referring to incorrect responses from GPT-4o, LLaVA-Video-7B, and Qwen2.5-VL-7B. Annotators should adjust the answer and rationale to the primary open-ended question minimally to as the answer and rationale to any related adversarial questions.

Alignment Check As illustrated in Fig. 3, during the stages of Ensuring Complexity, Primary Question Annotation, Answer Annotation, Sampling Check and Adversarial Question Annotation, we involve two additional annotators to maintain consistency among three annotators. Any question displaying inconsistent annotations is excluded. Specifically, during the Answer and Rational Annotation stage, questions addressing the cause of an event with several potential explanations are omitted unless there is unanimous agreement among the annotators. Videos and their associated question-answer pairs that do not meet these standards are eliminated by the verifier.

Distribution of Question Categories and Subcategories Distribution ofVideo Durations

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

Figure 6. Benchmark Statistics. Left: We show our 18 question by category alongside the number of questions. Right: The histogram of the video length.

###### 3.4. Dataset Statistics

Overall, we collected 5,000 questions and answers for 1,000 videos. Initially, questions were classified into one of three levels based on video content, i.e., element, event, and plot. Additionally, questions were organized into categories reflecting the nature of the inquiry. For instance, “Attributes” typically involve “what/who” questions, while “Localization” questions often concern ”after/before/when.” Furthermore, within a specific question category, such as “Element Attributes,” a new sub-category may emerge if a particular complex factor becomes prominent. For example, if over 50 questions are driven by the same factor, this significance leads to the creation of a sub-category like Element Attributes-Illusion. In total, Fig. 6 shows that there are 18 types of questions in Video-TT. The top three content categories in our dataset include comedy, sports, and daily life. To ensure quality and safety, we filtered out videos containing violent or explicit material, as well as suspected AIgenerated videos. All videos are shorter than 65 seconds. We also report the human hours for the whole curation process in Appendix 1.

##### 4. Experiments 4.1. Experimental Setup

Benchmark Models. We thoroughly evaluate eight video LLM from various model families, covering different sizes and training methods. For proprietary models, we include Gemini1.5 [33] and GPT-4o [25]. For open-source models, we assess Ola [20], Oryx [18], InternVL2.5 [4], Qwen2.5VL [34], and LLaVA-Video [51], using the lmms-eval codebase [48]. Unless specified, we default to 7B and 70(+)B model in each family. However, since the InternVL-2.578B model could not be implemented on eight H100 GPUs, we utilized its 38B version instead. All evaluations are conducted under zero-shot settings and using each model’s default prompts. The number of input frames is 80.

###### Blind Baseline and Human Level Performance. Be-

Table 1. Correctness score (accuracy) are reported for each question type and their average across types for each model. The Robustness (RB) score is derived from further statistical analysis of these accuracies.

Primary RephrasedCorrectly-LedWrongly-LedMulti-Choice

Model

Avg RB

Blind - Language Only Gemini Pro 9.1 8.3 22.4 5.4 5.3 9.1 2.9 GPT-4o 8.5 9.3 58.9 14.7 15.3 21.3 12.9 Video-Language Models Open-source models

Qwen2.5-VL-7B 20.9 22.5 45.3 39.3 39.9 33.6 14.4 LLaVA-Video-7B 21.4 22.5 49.2 37.2 41.8 34.4 13.7 Ola-7B 21.2 22.7 57.5 29.1 45.5 35.2 17.0 InternVL-2.5-8B 20.6 22.7 65.7 24.5 44.7 35.6 10.9 Oryx-1.5-7B 23.0 23.6 67.9 26.0 44.8 37.1 14.8 InternVL-2.5-38B 24.6 27.5 53.5 22.6 47.1 35.1 11.1 Qwen2.5-VL-72B 26.6 25.7 31.1 49.8 45.6 35.8 22.2 LLaVA-Video-72B 24.4 25.7 57.7 32.6 47.5 37.6 19.7

Proprietary models

Gemini Pro 28.8 29.7 50.2 29.2 42.3 38.2 20.5 GPT-4o 36.6 35.4 67.5 39.8 46.6 45.2 36.0

Human Baseline 84.3 83.9 83.9 76.2 87.5 83.2 64.4

yond video LMM model, we also introduce two baselines. First, we introduce the “blind” baseline based on the GPT4o and Gemini-Pro. Specifically, such baseline indicates we prompt models with video question only without using video frames as input. Second, we also ask human evaluators independently answer each question.

Metric. For assessing correctness score (accuracy), we use the Qwen2.5-72B model to score open-ended responses. Answers are scored on a scale from 0 to 5, detailed in the Appendix 3. An answer scoring above three is considered correct. For multiple-choice questions, we compare the selected option from the model’s response to the correct answer. A match confirms the response as correct. Correctness is essential for robustness. In videos where the model accurately answers the primary open question, we aim to assess how the model handles naturally adversarial scenario questions. We define the robustness score as the ratio of videos where all five questions are answered correctly to those where only the primary open-ended question is correctly answered. This measure helps identify and address any inconsistencies in responses to different versions of the same question.

- 4.2. Main Results: Accuracy Across Question Types

We present the evaluation results at Table. 1. Among the Video-Language Models, open-source systems vary greatly in performance. For instance, InternVL-2.5-8B scores high on Correctly-Led questions (65.7%), outperforming both LLaVA-Video-7B and Qwen2.5-VL-7B. However, its accuracy drops to 24.5% on Wrongly-Led questions, suggesting it struggles with misleading prompts. The LLaVA-Video72B stands out as the best performing open-source model.

Examining proprietary Models, Gemini Pro and GPT-4o achieve higher overall accuracies than many open-source al-

The mathematical definition of the robustness score is provided in the Appendix-Sec 2.

ternatives but still fall short of human performance. GPT4o shows strong results for Correctly-Led (67.5%) and Wrongly-Led (39.8%) questions, indicating better resistance to misleading prompts compared to most open-source models. However, even the best proprietary models reach only about half the accuracy of human annotators on openended tasks, emphasizing the ongoing challenges in complex video reasoning.

In addtion, although the best open-source model, LLaVA-Video-72B, performs similarly to GPT-4o in the multiple-choice setting (47.5 vs. 46.6), it lags significantly behind in primary open-ended questions. Primary openended questions better reflect realistic user interactions, where questions are often naturally phrased and less constrained than pre-defined options. This gap highlights an important improvement area for open-source models. Moreover, this observation also reveals a limitation of current video-language benchmarks, which tend to focus heavily on multiple-choice questions. Such benchmarks may overestimate model performance, failing to capture the true challenges presented by open-ended reasoning in real-world scenarios.

###### 4.3. Natural Adversarial Robustness

Table 1 shows a clear ranking in robustness performance among various models. Human annotators achieve the highest score at 64.4%, followed by GPT-4o at 36.0%. This significant difference between human performance and the top-performing model highlights the ongoing challenge of reaching human-level robustness in complex tasks.

For the group of open-source models including InternVL-2.5, LLaVA-Video, and Qwen2.5-VL, the performance of InternVL-2.5-8B is notably lower at 10.9%, while its 38B-Instruct version shows almost no improvement at 11.1%. However, LLaVA-Video, and Qwen2.5-VL underscore the potential improvements from larger model configurations.

##### 5. How far is Video LMM from Humans?

In this section, we aim to understand the differences between humans and models in processing video. We examine the errors made by GPT-4o in primary open-ended question. As we directed annotators to create questions reflecting visual and narrative complexity, our findings suggest a strong correlation between these complexities and the observed errors. Notably, among the 18 types of questions, five categories—Plot Attributes (technique editing), objective causality (narrative editing), elements attributes (illusion), element duration & speed, and professional knowledgeare directly linked to specific complex factors: technique editing, narrative editing, illusions, movement speeds, and world knowledge, respectively. In this section, we examined the errors across the other 13 question types, illustrat-

|Spatial-Temporal Arrangement Same Elements in Multiple Frames<br><br>Elements Counting|
|---|
|Q-1: How many picture frames are showing?<br><br>A: The video displays 10 frames. As the camera pans from left to right and then returns left, the frames at the end of the video are the same to those at the beginning.<br><br>GPT-4o:The video shows 12 frames.<br><br>|[Figure 121]|
|---|
<br><br>|

Spatial-temporalArrangement — EventSequence

Event Localization

Character Reaction

World Knowledge Psychological Activity

[Figure 122]

[Figure 123]

###### 👎 ImperfectCognition

Q-2:What are the characteristics of the second person who successfully did a flip in the video?

[Figure 124]

[Figure 125]

Q-3:What is the reaction of the person who came in second place in the video? A: She appears to be of disappointment. GPT-4o: The person who came in second place appears to be calm and relaxed.

A: The second person to attempt a flip is the one wearing a black hoodie. The shirtless man in the first scene tries to flip twice, but fails the first time.

[Figure 126]

RequiredWorld Knowledge

Silver medalists are the least happy, as they narrowly miss gold, while bronze medalists feel relieved to make the podium.

GPT-4o:The man who is shirtless.

[Figure 127]

[Figure 128]

[Figure 129]

- 👍 Correct Description to Scene 1

Q-4:Combining the different scenes, what is the video trying to imply narratively?

[Figure 130]

[Figure 131]

- 👍 Correct Description to Scene 2 A: The video illustrates the financial impact of damaging a rented house cause by the person playing baseball outside.

[Figure 132]

###### 👎 Fails to link different scenes to create a logical sequence

Complex Plot – In-context Reasoning

Plot Attributes

GPT4o: A person is seen at a baseball field, taking swings with various wood bats of different price points. The background includes a large building with advertisements

| | | | | |
|---|---|---|---|---|
| |[Figure 133]<br><br>[Figure 134]|[Figure 135]<br><br>[Figure 136]|[Figure 137]|[Figure 138]|
| | | | | |
| | | | | |

[Figure 139]

[Figure 140]

GPT4o: A person is sitting in a kitchen setting, holding a large white mug. The text on the screen reads "My Deposit $2500" initially, and then changes to "-$2000" and finally "-$1500." .The person is seen sipping from the mug and at one point.

GPT-4o: The video shows the amount of money this person spent on baseball games.

###### Scene 1 Scene 2

- Figure 7. Error cases in typical question types. We mark rationale answers with a grey background. Video links of each case : Q-1, Q-2, Q-3, Q-4.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 8. Human-conducted analysis of errors by question type.

appears, disappears, and then reappears in different frames. This error highlights the model’s difficulty in keeping track of elements consistently across multiple frames. Moreover, the model has trouble following sequences of actions and pinpointing who is doing what, particularly when it involves terms that specify order, like “second”, “third”,etc., as seen in Q2.

World Knowledge Deficiency in Character Reaction and Motivation: For the Character Reaction and Motivation category, 44% of errors arise from a lack of world knowledge. This indicates that the model frequently misinterprets why characters respond as they do. Many of these errors happen because the model lacks the commonsense or cultural knowledge needed to understand character actions. To better its performance, the model requires a stronger foundation in social norms, emotions, and contextual expectations. For example, as shown in Fig.7-Q3, a person might look calm or relaxed. Yet, recognizing this expression as disappointment depends on understanding the context provided by world knowledge.

ing how these complexities lead to the models’ errors, as depicted in Fig. 8. We highlight three main errors in this section, with additional analysis provided in the Appendix 4.

Spatial-Temporal Confusion in Physical and Counting Tasks: In tasks that involve understanding temporal and spatial relationship, such as Element/Event Localization and Event Counting, the most frequent errors (79% and 88%, respectively) arise from confusion in how events and objects are arranged over time and space. This confusion indicates that the model struggles to maintain a clear and consistent understanding of where and when events occur, which leads to mistakes in recognizing the sequence and location of these events and objects. For example, as shown in Fig. 7-Q1, the model can correctly count the number of photo frames in a single frame, but it fails when a frame

Complex Plot Confusion in Plot Attributes and Objective Causality: In the Plot Attributes and Objective Causality category, a significant 55% of errors stem from a misunderstanding of complex plots. This shows that the model struggles to keep a coherent cause-and-effect relationship across multiple events. When the storyline requires linking different elements to create a logical sequence, the model often fails. Enhancing the model’s capability to track extended causal relationships is crucial for improving its performance in this reasoning type. An example is shown in

- Figure 9. (a-b) Comparison of human and average model performance based on correctness and robustness across question types. (c) Comparison of human performance and that of two models across different numbers of frames. (d) Relative performance change (%) when adding Chain-of-Thought (CoT) reasoning and audio transcript information.

Fig.7-Q4.

##### 6. Further Analysis

Human-Model Behavior Correlation We compare the performance of humans and models in terms of accuracy and robustness across different types of questions. Figure 9 (a) reveals a moderate positive correlation (r = 0.49), suggesting that question types where humans excel are generally easier for models as well. Despite this, models consistently fall short of human performance, even on simpler tasks. Figure 9 (b) displays a negative correlation (r = -0.50) between the model and human scores. The model’s performance decreases significantly in question types involving Element Counting or Displacement. These types are often linked to visual complexity, suggesting the model’s reduced effectiveness in complex visual situations. In contrast, humans show robust performance in these scenarios. This negative correlation merits further investigation.

Impact of Frame Numbers Figure 9 (c) shows how performance changes with more input frames. Human performance improves steadily with an increase in frames, reaching nearly perfect accuracy at 64 frames. On the other hand, model performance saturates after about 8 frames. This pattern differs from other datasets [6, 7, 49], where more frames significantly boost performance. This observation supports the core design of our benchmark: Annotators are asked to create questions that can be answered with just 80 uniformly sampled frames.

Impact of Chain-of-Thought Prompting Chain-ofThought Prompting (CoT) techniques improve the reasoning skills of large models in various tasks [5, 19, 27, 30, 35]. Given these successes, we investigate CoT could also enhance performance in Video-TT. We assess the CoT, which adds “Let’s think step by step” to the prompts. We present results at Figure 9 (d). Performance on Wrongly-led shows a relative increase of about 6.8%, which indicates that struc-

tured thinking helps the model spot and bypass misleading hints more effectively. For Multiple-choice Questions, however, CoT shows no noticeable advantage, with similar performance between models. This result suggests that CoT aids tasks requiring unstructured thinking (like open-ended questions), whereas tasks with structured formats such as multiple-choice benefit less.

Impact of Audio Transcript Audio transcripts shows impact on model performance in recent multi-choice video LLM benchmarks [21, 49], leading us to examine their influence. In Multiple-choice Questions, transcripts do not improve performance over the GPT-4o baseline. This contrasts with other benchmarks where audio has enhanced performance. Nevertheless, our results align with our dataset’s focus on visual content. Furthermore, transcripts significantly boost Robustness Performance—an almost 15% relative gain. This improvement underscores the value of including spoken input to increase robustness.

##### 7. Conclusion

We introduced the Video Thinking Test (Video-TT), a new benchmark designed to assess the correctness and robustness of video large language models (video LLMs) in understanding complex real-world videos. Video-TT separates errors due to not enough frame sampling from those due to genuine comprehension issues, offering a more reliable way to test these models. In terms of correctness, opensource models perform well on the multi-choice track but fall short in the open-ended track compared to GPT-4o. The open-ended models also show less resilience against naturally tricky questions compared to GPT-4o. However, both GPT-4o and open-source models are still far behind human performance. Error analysis shows that video LLMs have difficulties with understanding space and time together, integrating world knowledge, and linking different elements in video to create a logical response. These results high-

light the urgent need to improve reasoning, resilience, and real-world comprehension in video LLMs, providing a clear direction for future research in video intelligence.

##### Acknowledgement

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP202210012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

##### References

- [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 3
- [2] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Towards fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 1, 2, 3
- [3] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pages 190–200, 2011. 3
- [4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 1, 5
- [5] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024. 8
- [6] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 1, 3, 8
- [7] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 1, 2, 3, 5, 8
- [8] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11287–11297, 2021. 3
- [9] Christopher Heaps and Stephen Handel. Similarity and features of natural textures. Journal of Experimental Psychology: Human Perception and Performance, 25(2):299, 1999. 3
- [10] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021. 2
- [11] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. 1, 2, 3
- [12] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE con-

- ference on computer vision and pattern recognition, pages 2758–2766, 2017. 3
- [13] Michael D Lee, Amy H Criss, Berna Devezer, Christopher Donkin, Alexander Etz, F´abio P Leite, Dora Matzke, Jeffrey N Rouder, Jennifer S Trueblood, Corey N White, et al. Robust modeling in cognitive science. Computational Brain & Behavior, 2:141–153, 2019. 1
- [14] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. arXiv preprint arXiv:1809.01696, 2018. 3
- [15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark, 2023. 1, 2, 3
- [16] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022. 3
- [17] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 2, 3
- [18] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 1, 5
- [19] Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, and Jiwen Lu. Chain-of-spot: Interactive reasoning improves large vision-language models. arXiv preprint arXiv:2403.12966,

2024. 8

- [20] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment. arXiv preprint arXiv:2502.04328, 2025. 1, 5
- [21] LMMs-Lab. Video detail caption, 2024. 8
- [22] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 1
- [23] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36, 2024. 2
- [24] Aude Olivia, Michael L Mack, Mochan Shrestha, and Angela Peeper. Identifying the perceptual dimensions of visual complexity of scenes. In Proceedings of the annual meeting of the cognitive science society, 2004. 3
- [25] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 5
- [26] Madeline Schiappa, Shruti Vyas, Hamid Palangi, Yogesh Rawat, and Vibhav Vineet. Robustness analysis of videolanguage models against visual and language perturbations. Advances in Neural Information Processing Systems, 35: 34405–34420, 2022. 2, 3

- [27] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024. 8
- [28] Jan Simons. Complex narratives. In Hollywood puzzle films, pages 17–34. Routledge, 2014. 4
- [29] Joan G Snodgrass and Mary Vanderwart. A standardized set of 260 pictures: norms for name agreement, image agreement, familiarity, and visual complexity. Journal of experimental psychology: Human learning and memory, 6(2):174,

1980. 3

- [30] Guangyan Sun, Mingyu Jin, Zhenting Wang, Cheng-Long Wang, Siqi Ma, Qifan Wang, Tong Geng, Ying Nian Wu, Yongfeng Zhang, and Dongfang Liu. Visual agents as fast and slow thinkers. arXiv preprint arXiv:2408.08862, 2024. 8
- [31] Zekun Sun and Chaz Firestone. Curious objects: How visual complexity guides attention and engagement. Cognitive Science, 45(4):e12933, 2021. 3
- [32] John Sweller. Cognitive load during problem solving: Effects on learning. Cognitive science, 12(2):257–285, 1988. 4
- [33] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 5
- [34] Qwen Team. Qwen2.5-vl, 2025. 1, 5
- [35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 8
- [36] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. 3
- [37] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. 1, 3, 5
- [38] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 2, 3
- [39] Binzhu Xie, Sicheng Zhang, Zitang Zhou, Bo Li, Yuanhan Zhang, Jack Hessel, Jingkang Yang, and Ziwei Liu. Funqa: Towards surprising video comprehension. arXiv preprint arXiv:2306.14899, 2023. 3
- [40] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international confer-

ence on Multimedia, pages 1645–1653, 2017.

- [41] Li Xu, He Huang, and Jun Liu. Sutd-trafficqa: A question answering benchmark and an efficient network for video reasoning over traffic events. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9878–9888, 2021.
- [42] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.
- [43] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019.
- [44] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 3
- [45] Amir Zadeh, Michael Chan, Paul Pu Liang, Edmund Tong, and Louis-Philippe Morency. Social-iq: A question answering benchmark for artificial social intelligence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8807–8817, 2019. 2
- [46] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1
- [47] Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, and Kaipeng Zhang. Bavibench: Towards evaluating the robustness of large visionlanguage model on black-box adversarial visual-instructions,

2024. 1

- [48] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 5
- [49] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 8
- [50] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 1
- [51] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. 1, 5
- [52] Yuanhan Zhang, Kaichen Zhang, Bo Li, Fanyi Pu, Christopher Arif Setiadharma, Jingkang Yang, and Ziwei Liu. Worldqa: Multimodal world knowledge in videos through long-chain reasoning, 2024. 1, 2, 3
- [53] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264,

2024. 3

[54] Xingquan Zhu, Jianping Fan, Ahmed K Elmagarmid, and Xindong Wu. Hierarchical video content description and summarization using unified semantic and visual similarity. Multimedia Systems, 9:31–53, 2003. 3

## Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding

### Supplementary Material

##### 1. Annotation Detail

We present the number of human hours at each stage in the Data Curation Process as follows. In total, the annotation process cost 8227.32 human hours.

Table 2. Time Estimation for Dataset Curation Process. Notes: *2 indicates that two people are required for this stage; *4 refers to four natural adversarial questions per video; *5 covers all questions for human baseline annotation. Q stands for Question; A stands for Answer; R stands for Rationale.

Dataset Curation Stage #Data Hour/Data Total (hour)

Trial Data Annotation 226 0.5 113 Trial Data Alignment 226 0.25 56.5 Complex Video Collection 2,977 0.16 496.17 Complex Video Alignment 2,977 0.05*2 297.7 Primary Q&A&R Annotation 2,338 0.5 1,169 Primary Q&A&R Alignment 2,338 0.3*2 1,402.8 Sampling Check 1,344 0.25*2 672 Adversarial Question Annotation 1,300*4 0.16 832 Adversarial Question Alignment 1,300*4 0.08*2 832 Human Baseline Annotation 1,300*5 0.16 1,040

Total 8227.32

##### 2. Mathematical Definition of the Robustness Score

- • Aprimary correct be the set of videos where the primary openended question is answered correctly.

- • Aparaphrased correct be the set of videos where the paraphrased open-ended question is answered correctly.

- • Acorrectly led correct be the set of videos where the correctlyled open-ended question is answered correctly.

- • Awrongly led correct be the set of videos where the wronglyled open-ended question is answered correctly.

- • Amultiple choice correct be the set of videos where the multiple-choice question is answered correctly.

The set of videos where all five questions are answered correctly, denoted as Afull correct, is the intersection of all these sets:

R = |Afull correct| |Aprimary correct|

Where |A| denotes the cardinality (size) of the set A, representing the number of videos in that set.

##### 3. Prompt for Evaluating Open-ended Answer

Table. 3 shows the prompt for evaluating open-ended answers. A score of 3 or higher is considered correct, while scores below 3 are deemed incorrect. We refer to the prompt introduced in VideoChatGPT [22].

System Message

You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task:

—INSTRUCTIONS:

- - Focus on the meaningful match between the predicted answer and the correct answer.
- - Consider synonyms or paraphrases as valid matches.
- - Evaluate the correctness of the prediction compared to the answer. Please evaluate the following video-based question-answer pair: Question: question Correct Answer: answer Predicted Answer: pred Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. ” Please generate the response in the form of a Python dictionary string with keys ’pred’ and ’score’, where value of ’pred’ is a string of ’yes’ or ’no’ and value of ’score’ is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: ’pred’: ’yes’, ’score’: 4.

Table 3. System message for evaluating the open-ended answer.

Afull correct =Aprimary correct ∩ Aparaphrased correct ∩ Acorrectly led correct ∩ Awrongly led correct ∩ Amultiple choice correct

Thus, the Robustness Score (RB) becomes:

##### 4. Error Analysis

In this section, we give more analysis about the errors made by GPT-4o.

Rejected Videos [639]

Rejected Primary Q&A&R [994]

Complex Video Collection [2977]

Primary Q&A&R Annotation [2338]

Sampling Check [1344]

Video Turing Test [1000]

Reserved for Other Use Cases [300]

Rejected in Sampling Check [44]

|Loading [MathJax]/extensions/MathMenu.js|
|---|

Figure 10. The data annoation flow of Video Turing Test. Q stands for Question; A stands for Answer; R stands for Rationale.

World Knowledge ElementsAttributes Unusual PlotAttributes Metaphor

EventAttributes Unclear

Q-1: Which part of the person on the ladder precisely catches the hammer handed by the individual on the ground?

|[Figure 141]|
|---|

|[Figure 142]|
|---|

Q-3: After solving the Rubik's Cube, which color is facing up?

[Figure 143]

A: The color of the magic side facing up is white, which can be determined by looking at the top face of the cube in the image. The orange color is the face facing the kid.

A: The worker on the ladder clamps the

hammer with his buttocks.The worker below adeptly tosses the hammer upwards with one hand, to the worker high on the ladder.

*Battle in a trench

GPT-4o: The hammer is received by the worker catching it mid-air with right hand.

GPT-4o: The color facing up on the Rubik's Cube is yellow.

Complex Plot –

SpatialArrangement -

SpatialArrangement – Elements Localization ActionSender/Receiver

EventCounting Theboundaryofanevent

PlotAttributes In-context Reasoning

[Figure 144]

###### 👍 Prefect Recognition

Q-2: What are the characteristics of the person who puts the bamboo into the machine? A: The man placing the bamboo into the machine is wearing blue-black leather gloves, and black-and-white checked sleeves. The person in blue apron is the one moving baboo away from the machine GPT-4o: The person putting the bamboo into the machine is wearing a blue apron over a camouflage-patterned shirt and red gloves.

Q-4: How many times did this person fold the lotus root stem in total??

[Figure 145]

[Figure 146]

[Figure 147]

- Q-6: What does the two boy near the table do in the video? GPT-4o: They take turns flipping a partially filled water bottle, attempting to make it land upright on the table.

GPT-4o: The man is imitating riding a motorcycle.

Q-5:What does this man want to imitate?

[Figure 148]

👎 ImperfectCognition

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

RequiredWorld Knowledge

[Figure 152]

- Q-7: Who is the winner of the game? GPT-4o: The winner of the game is the man in blue as he reached 60 successful bottle flips first . (A: The boy in gray shirt.）

A: The person folds the lotus root stem a total of 15 times. Every time he folds the lotus stem, he bends it downward and switches between his left and right hands.

[Figure 153]

👎ImperfectCognition

GPT-4o: The person folded the lotus root stem a total of 10 times

Figure 11. Error cases in typical question types. We mark rationale answers with a grey background. Video links of each case : Q-1 Q-2, Q-3, Q-4, Q-5, Q-6 & Q-7.

- 4.1. Recognition: Detecting objects and their attributes

ity issues, as shown in Fig. 11-Q3, the model struggles to accurately identify the color of a small Rubik’s Cube in the video frames.

In this subsection, we analyze errors in six question types focused on the “element” and “event” categories. These errors typically stem from visual complexity, challenging the recognition capabilities of the model.

Event Counting. In this category, one specific errors arise from the model’s difficulty in accurately identifying the start and end points of repeated events, despite correctly classifying the event type (Fig. 11-Q4).

Element Attributes and Event Attributes. In this category, 80% of errors involve unclear or unusual subjects in the questions, which relate to elements or events. These errors are linked to issues of unclear and the presence of unusual content in visual complexity. For instance, as depicted in Fig. 11-Q1, when confronted with unusual content, the model often defaults to the most common outcome rather than what is actually depicted in the video. For clar-

Element Localization and Event Localization. Errors in this category, which make up 79%, are related to spatiotemporal challenges. In spatial terms, a common error occurs when multiple individuals are present in a scene, and the model incorrectly assigns actions to the wrong person. This issue is particularly prevalent in interactions involving

two people, leading to confusion over who is performing and who is receiving the action (Fig. 11-Q2).

Positional Relationship. Understanding the relative positions of elements is a fundamental human skill. Yet, we observed that models struggle with this task. For instance, when asked whether element A is on the left or right side of B, the model typically responds “left” if A visually appears on the left side of the video frame. This response disregards their actual spatial relationship within the context of the video. Such findings indicate a significant limitation in the model’s ability to accurately interpret positional relationships.

Displacement. For a frame-based model, these questions challenge the model’s ability to track the development of the event across consecutive frames. For instance, considering the displacement of an object from the previous frame to the current one poses a significant challenge if the model’s vision encoder struggles with fine-grained spatial localization grounding [16].

- 4.2. Cognition: Reasoning the likely intents, goals, and social dynamics of people

In this subsection, we analyze errors in question types associated with the “plot.” These errors are typically due to narrative complexity. When prompted, the model demonstrates recognition-level perception abilities; however, the narrative complexity challenges the model in addressing “cognition” level questions.

Character Reaction and Character Motivation. As discussed in Sec.3.1, world knowledge significantly contributes to narrative complexity. Fully understanding characters’ reactions and motivations requires applying this knowledge. Commonly, this involves grasping psychological activities, which are subjective by nature. To answer relevant questions effectively, the model must do more than just describe the video; it needs to link these descriptions to broader world knowledge.

Plot Attributes and Objective Causality. The typical errors in “plot attributes and objective causality” stem from a lack of world knowledge and in-context reasoning ability. An interesting aspect of necessary world knowledge is its multi-modal nature, essential for correct responses in this category. For example, as shown in Fig. 11-Q5, while the model can accurately describe a man’s actions in the video, understanding what these actions imply—such as imitating a battle scene in a trench—requires linking the video content with relevant world scenes. Moreover, the model’s limited in-context reasoning is evident as it struggles to integrate diverse perceptual inputs into a cohesive understanding of social dynamics, despite accurately answering recognition-level questions about actions observed in the video.

